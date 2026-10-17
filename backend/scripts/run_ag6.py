"""AG6综合评测：冻结场景、隔离执行和证据导出，不替业务Agent完成研究。

本脚本只编排测试用户输入。真实研究仍经FastAPI、MySQL和TaskExecutor；
程序检查与原文语义核对分开，未核对的输出不能自动标记质量通过。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlparse
from unittest.mock import patch

from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.providers.deepseek import DeepSeekProvider, RequestNotSent, validate_messages
from research_loop.providers.pdf import PDFProvider
from research_loop.providers.dpr_acquisition import PaperAcquisitionError
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor


ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {'completed', 'partial', 'failed', 'cancelled', 'needs_input'}


def write_json(path: Path, value) -> None:
    """原子保存评测记录；历史批次目录不复用，当前账本可持续追加。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    os.replace(temporary, path)


def digest(value: bytes) -> str:
    """返回原始字节SHA-256，用于绑定源码、输入和成果。"""
    return hashlib.sha256(value).hexdigest()


def validate_manifest(manifest: dict) -> None:
    """检查完整分母、权限、额度及消息步骤；不评价答案内容。

    Args:
        manifest: 包含12场景和统一批次预算的JSON对象。

    Raises:
        ValueError: 场景重复、预算非法或测试步骤缺少输入。
    """
    cases = manifest.get('cases', [])
    if [c.get('id') for c in cases] != [f'E{i:02d}' for i in range(1, 13)]:
        raise ValueError('必须包含按顺序固定的E01–E12，不能删除失败分母')
    if manifest['physical_cap'] < sum(c['request_cap'] for c in cases):
        raise ValueError('批次额度小于场景额度之和')
    if manifest['token_cap'] <= 0 or manifest['wall_seconds'] <= 0:
        raise ValueError('token/时间预算必须为正')
    for case in cases:
        if case['save_policy'] not in {'allow', 'deny'} or case['request_cap'] < 1:
            raise ValueError('非法保存权限或场景预算')
        if not case.get('checks') or not case.get('steps'):
            raise ValueError('场景缺少检查点或步骤')
        for step in case['steps']:
            if step['kind'] in {'task', 'resume', 'cancel_continue'} and not step.get('prompt'):
                raise ValueError('消息步骤缺少输入')


def validate_isolation(url: str, output: Path, root: Path = ROOT) -> str:
    """拒绝生产库、远端数据库及artifact外路径；返回安全数据库名。"""
    parsed = urlparse(url)
    name = parsed.path.lstrip('/')
    if parsed.scheme != 'mysql' or parsed.hostname not in {'127.0.0.1', 'localhost'}:
        raise ValueError('评测必须使用本机隔离MySQL')
    if not re.fullmatch(r'researchloop_test_ag6_[a-zA-Z0-9_]+', name):
        raise ValueError('评测数据库必须以researchloop_test_ag6_开头')
    output.resolve().relative_to((root / 'artifacts').resolve())
    return name


def summarize_usage(receipts: list[dict]) -> dict:
    """按收据汇总已知用量，未知/缺字段单列，绝不补零冒充已知。"""
    totals = dict.fromkeys(['prompt_tokens', 'completion_tokens', 'total_tokens'], 0)
    missing, not_sent = [], []
    for receipt in {r['call_id']: r for r in receipts}.values():
        if receipt.get('sent_state') == 'not_sent':
            not_sent.append(receipt['call_id'])
            continue
        usage = json.loads(receipt.get('usage_json') or '{}')
        if any(not isinstance(usage.get(k), int) for k in totals):
            missing.append(receipt['call_id'])
        for key in totals:
            if isinstance(usage.get(key), int):
                totals[key] += usage[key]
    return {'known_tokens': totals, 'missing_usage_call_ids': missing, 'not_sent_call_ids': not_sent,
            'unknown': sum(r['status'] in {'unknown', 'started'} and r.get('sent_state') != 'not_sent' for r in {r['call_id']: r for r in receipts}.values()),
            'provider_failures': sum(r['status'] in {'response_error','rejected'} for r in {r['call_id']: r for r in receipts}.values())}


class EvaluationLedger:
    """跨并行Researcher的物理请求账本；额度检查在网络发送前执行。"""

    def __init__(self, path: Path, manifest: dict):
        """创建新批次账本；不接受覆盖先前发送记录。"""
        if path.exists():
            raise ValueError('账本已存在，禁止自动重跑')
        self.path, self.manifest = path, manifest
        self.lock = threading.Lock()
        self.records = []
        self.case = None
        self.started = time.monotonic()

    def reserve(self, kind: str, messages: list, max_tokens: int | None, tools=None) -> int:
        """预留本次调用及保守输入/输出估计；未知记录会阻断下一次发送。"""
        validate_messages(messages)
        estimate = len(json.dumps({'messages': messages, 'tools': tools}, ensure_ascii=False).encode()) + (max_tokens or 32768)
        with self.lock:
            case = next(c for c in self.manifest['cases'] if c['id'] == self.case)
            used = sum(r['case'] == self.case for r in self.records)
            committed = sum((r.get('usage') or {}).get('total_tokens', r['estimated_tokens']) for r in self.records)
            if any(r['status'] in {'unknown','response_error'} for r in self.records):
                raise RequestNotSent('AG6_UNKNOWN_STOP')
            if len(self.records) >= self.manifest['physical_cap'] or used >= case['request_cap']:
                raise RequestNotSent('AG6_PHYSICAL_CAP_BEFORE_SEND')
            if committed + estimate > self.manifest['token_cap']:
                raise RequestNotSent('AG6_TOKEN_CAP_BEFORE_SEND')
            if time.monotonic() - self.started > self.manifest['wall_seconds']:
                raise RequestNotSent('AG6_TIME_CAP_BEFORE_SEND')
            n = len(self.records)
            self.records.append({'number': n + 1, 'case': self.case, 'kind': kind,
                                 'status': 'started', 'estimated_tokens': estimate,
                                 'started_at': datetime.now(timezone.utc).isoformat()})
            write_json(self.path, self.records)
            print(f'{self.case} physical_request={n + 1} {kind}', flush=True)
            return n

    def settle(self, n: int, raw: dict | None, error: Exception | None = None) -> None:
        """记录真实响应与usage；发生异常时保留原因而不自动重发。"""
        with self.lock:
            row = self.records[n]
            row['finished_at'] = datetime.now(timezone.utc).isoformat()
            if error:
                row.update(status='unknown' if not hasattr(error, 'status_code') else 'response_error',
                           error_type=type(error).__name__, error=str(error)[:1000])
            else:
                row.update(status='succeeded', usage=raw.get('usage'), request_id=raw.get('id'))
            write_json(self.path, self.records)


class EvaluationRunner:
    """把冻结的测试用户步骤送入真实产品入口，并保存独立核对所需证据。"""

    def __init__(self, settings: Settings, out: Path, manifest: dict):
        """初始化隔离存储；业务类与正常使用相同，供应商仅增加计数。"""
        self.settings, self.out, self.manifest = settings, out, manifest
        self.db = Database(settings.database_url)
        self.db.ensure_schema()
        if self.db.fetchone('SELECT COUNT(*) AS n FROM tasks')['n']:
            raise ValueError('本批次要求独立空库，不删除或恢复历史任务')
        self.ledger = EvaluationLedger(out / 'physical.json', manifest)
        self.cancel_received, self.cancel_release = threading.Event(), threading.Event()
        self.cancel_hook = False
        self.results = []

    def provider(self, settings):
        """包装当前DeepSeek，记录所有角色和流式/非流式的真实发送。"""
        owner = self

        class Counted(DeepSeekProvider):
            """不改提示和响应；取消用同步点让用户停止落在真实运行窗口。"""
            def _run(self, kind, method, args, kwargs):
                """先记物理发送、后结算用量；异常不触发重试。"""
                messages = args[0] if args else kwargs['messages']
                tools = args[1] if len(args) > 1 else kwargs.get('tools')
                n = owner.ledger.reserve(kind, messages, kwargs.get('max_tokens'), tools)
                try:
                    raw = method(*args, **kwargs)
                except Exception as exc:
                    owner.ledger.settle(n, None, exc)
                    raise
                owner.ledger.settle(n, raw)
                if owner.cancel_hook:
                    owner.cancel_hook = False
                    owner.cancel_received.set()
                    if not owner.cancel_release.wait(20):
                        raise RuntimeError('cancel test controller did not release')
                return raw

            def send(self, *args, **kwargs):
                """记录一次普通模型调用。"""
                return self._run('json', super().send, args, kwargs)

            def send_stream(self, *args, **kwargs):
                """记录一次真实SSE调用，不把片段计为额外请求。"""
                return self._run('stream', super().send_stream, args, kwargs)

        return Counted(settings.deepseek_api_key, settings.deepseek_model,
                       settings.deepseek_base_url, settings.request_timeout_seconds)

    def request(self, client, method, path, **kwargs):
        """检查真实HTTP接口返回；错误调用与正文保留在当前场景证据。"""
        response = getattr(client, method)(path, **kwargs)
        if response.is_error:
            raise ValueError(f'{method} {path}: {response.status_code} {response.text[:1000]}')
        return response.json()

    def collect(self, client, task_id: str) -> dict:
        """读回父子任务、原文、草稿、收据和公开产物，不靠终态猜事实正确。"""
        family = self.db.fetchall('SELECT task_id,role,status FROM tasks WHERE root_task_id=?', (task_id,))
        return {'view': self.request(client, 'get', '/api/tasks/' + task_id),
                'family': family,
                'actions': [a for task in family for a in self.db.list_actions(task['task_id'])],
                'events': self.db.list_events(task_id),
                'receipts': self.db.fetchall('SELECT m.* FROM model_calls m JOIN tasks t ON t.task_id=m.task_id WHERE t.root_task_id=? ORDER BY m.created_at', (task_id,)),
                'drafts': self.db.fetchall('SELECT d.* FROM drafts d JOIN tasks t ON t.task_id=d.task_id WHERE t.root_task_id=?', (task_id,)),
                'reviews': self.db.fetchall('SELECT * FROM review_results WHERE task_id=?', (task_id,)),
                'evidence': self.db.fetchall('SELECT DISTINCT e.* FROM page_evidence e JOIN task_evidence te ON te.evidence_id=e.evidence_id JOIN tasks t ON t.task_id=te.task_id WHERE t.root_task_id=?', (task_id,)),
                'context': self.db.fetchall('SELECT * FROM context_archives WHERE task_id=?', (task_id,)),
                'artifacts': self.request(client, 'get', f'/api/tasks/{task_id}/artifacts')}

    def execute(self, task_id, case, overrides=None):
        """以场景剩余额度执行正常Worker；并行子任务共享原业务预算。"""
        spent = sum(r['case'] == case['id'] for r in self.ledger.records)
        remaining = case['request_cap'] - spent
        if remaining <= 0:
            raise RuntimeError('scenario budget exhausted before task')
        existing = self.db.fetchone('SELECT model_used FROM task_budgets WHERE root_task_id=?', (task_id,))
        consumed = existing['model_used'] if existing else 0
        self.db.execute('UPDATE task_budgets SET model_limit=?,reserve_tail=? WHERE root_task_id=?',
                        (consumed + remaining, min(4, remaining // 3), task_id))
        current = self.settings.model_copy(update={
            'max_model_requests': consumed + remaining,
            'reserve_model_requests': min(4, remaining // 3), **(overrides or {})})
        pdf_factory = None
        if case.get('controlled_pdf_failure'):
            class Unavailable(PDFProvider):
                """仅当前E09受控获取故障，不能记为自然上游失效。"""
                def ingest(self, source, paper_id):
                    raise PaperAcquisitionError('CONTROLLED_PDF_UNAVAILABLE', 'AG6 controlled download failure; no text obtained')
                ingest_pages = ingest
            pdf_factory = lambda s, f: Unavailable(f)
        TaskExecutor(current, self.db, self.provider, pdf_factory=pdf_factory).execute(task_id)

    def scenario(self, client, case):
        """按清单进行一次场景；仅预先给定的测试用户答案可以恢复澄清。"""
        self.ledger.case = case['id']
        folder = self.out / case['id']
        folder.mkdir()
        record = {'id': case['id'], 'input': case, 'tasks': [], 'steps': [],
                  'semantic_verdict': 'not_reviewed', 'user_usability': 'not_reviewed'}
        cid = self.request(client, 'post', '/api/conversations', json={
            'title': 'AG6 ' + case['title'], 'idempotency_key': case['id']})['conversation_id']
        record['conversation_id'] = cid
        task_id = None
        try:
            for number, step in enumerate(case['steps']):
                key = f"{case['id']}-{number}"
                if step['kind'] == 'new_conversation':
                    cid = self.request(client, 'post', '/api/conversations', json={
                        'title': 'AG6 ' + key, 'idempotency_key': key})['conversation_id']
                    record['steps'].append({'kind': step['kind'], 'conversation_id': cid})
                    continue
                if step['kind'] == 'forget':
                    profile = self.request(client, 'get', '/api/profile')
                    entries = self.db.fetchall("SELECT * FROM memory_entries WHERE status='active' AND kind='explicit'")
                    selected = [r for r in entries if step['contains'] in r['content']]
                    if not selected:
                        raise ValueError('expected explicit preference was not persisted')
                    responses = [self.request(client, 'post', '/api/profile/forget', json={
                        'memory_id': r['memory_id'], 'expected_version': r['version']}) for r in selected]
                    record['steps'].append({'kind': 'forget', 'before': profile, 'responses': responses})
                    continue
                if step['kind'] == 'seed_context':
                    from research_loop.services.conversations import ConversationService
                    conv = ConversationService(self.db)
                    with conv.store.transaction() as tx:
                        for i, content in enumerate(step['messages']):
                            conv.add_message(tx, cid, 'synthetic-'+key, key+str(i),
                                             'user' if i % 2 == 0 else 'assistant', content)
                    record['steps'].append({'kind': 'seed_context', 'synthetic': True, 'count': len(step['messages'])})
                    continue
                if step['kind'] == 'resume':
                    before = self.collect(client, task_id)
                    record['steps'].append({'kind': 'before_resume', 'view': before['view'], 'actions': before['actions']})
                    question = before['view'].get('clarification')
                    if not question:
                        raise ValueError('expected clarification did not occur')
                    self.request(client, 'post', f'/api/tasks/{task_id}/resume', json={
                        'clarification_id': question['clarification_id'], 'answer': step['prompt'], 'idempotency_key': key})
                else:
                    attachments = []
                    for material in step.get('materials', []):
                        item = self.manifest['materials'][material]
                        path = ROOT / item['path']
                        upload = self.request(client, 'post', '/api/uploads', params={'filename': path.name}, content=path.read_bytes())
                        attachments.append({'input_ref': upload['input_ref'], 'filename': path.name})
                    attachments += step.get('attachments', [])
                    payload = {'conversation_id': cid, 'prompt': step['prompt'], 'attachments': attachments,
                               'save_policy': case['save_policy'], 'idempotency_key': key}
                    task = self.request(client, 'post', '/api/tasks', json=payload)
                    task_id = task['task_id']
                    if step['kind'] == 'cancel_continue':
                        self.cancel_received.clear(); self.cancel_release.clear(); self.cancel_hook = True
                        errors = []
                        def run_cancelled():
                            """后台执行实际Worker；控制线程只提交用户取消/接续输入。"""
                            try: self.execute(task_id, case)
                            except Exception as exc: errors.append(str(exc))
                        thread = threading.Thread(target=run_cancelled)
                        thread.start()
                        try:
                            if not self.cancel_received.wait(180):
                                raise RuntimeError('no model response reached cancellation synchronization point')
                            continuation = self.request(client, 'post', f'/api/tasks/{task_id}/continue', json={
                                'conversation_id': cid, 'prompt': step['continuation_prompt'],
                                'save_policy': 'deny', 'idempotency_key': key+'-continued'})
                        finally:
                            self.cancel_release.set(); thread.join(180)
                        if thread.is_alive() or errors:
                            raise RuntimeError('cancellation worker failed: '+str(errors))
                        cancelled = self.collect(client, task_id)
                        record['tasks'].append(cancelled)
                        from research_loop.services.continuations import ContinuationService
                        ContinuationService(self.db).release_ready()
                        row = self.db.fetchone('SELECT * FROM task_continuations WHERE continuation_id=?', (continuation['continuation_id'],))
                        record['steps'].append({'kind': 'cancel_continue', 'continuation': row, 'controlled_response_barrier': True})
                        task_id = row['next_task_id']
                        if not task_id: raise ValueError('continuation did not become queued')
                self.execute(task_id, case, step.get('settings'))
                value = self.collect(client, task_id)
                record['tasks'].append(value)
                write_json(folder / 'result.json', record)
                print(json.dumps({'case': case['id'], 'step': number, 'status': value['view']['status'],
                                  'requests': value['view']['usage']['requests']}, ensure_ascii=False), flush=True)
                if value['view']['usage']['unknown']:
                    raise RuntimeError('unknown receipt; stop batch')
                if any(r['status'] in {'unknown','response_error'} for r in self.ledger.records):
                    raise RuntimeError('provider failure; stop remaining steps and batch')
        except Exception as exc:
            record['execution_error'] = f'{type(exc).__name__}: {exc}'
            if task_id:
                record['last_task_readback'] = self.collect(client, task_id)
        finally:
            record['profile_after'] = self.request(client, 'get', '/api/profile')
            record['usage'] = summarize_usage([r for t in record['tasks'] for r in t['receipts']])
            record['physical_requests'] = sum(r['case'] == case['id'] for r in self.ledger.records)
            record['elapsed_seconds'] = round(time.monotonic() - self.case_started, 3)
            record['structural_checks'] = structural_checks(record)
            write_json(folder / 'result.json', record)
            self.results.append(record)

    def run(self):
        """单次串行运行场景；每场景内部仍允许业务Researcher并行。"""
        from research_loop.main import app
        with ExitStack() as stack:
            for module in ['dependencies']:
                stack.enter_context(patch(f'research_loop.api.{module}.get_settings', return_value=self.settings))
            client = stack.enter_context(TestClient(app))
            for case in self.manifest['cases']:
                self.case_started = time.monotonic()
                self.scenario(client, case)
                self.summary()
                if any(r['status'] in {'unknown', 'response_error'} for r in self.ledger.records):
                    break
                if time.monotonic() - self.ledger.started > self.manifest['wall_seconds']:
                    break
        return self.summary()

    def summary(self):
        """始终保留12场景分母；未执行和未审阅不能自动计入通过。"""
        receipts = self.db.fetchall('SELECT * FROM model_calls ORDER BY created_at')
        value = {'stage': 'AG6', 'stage_complete': False, 'semantic_review': 'pending',
                 'planned_scenarios': 12, 'executed_scenarios': len(self.results),
                 'unexecuted': [c['id'] for c in self.manifest['cases'] if c['id'] not in {r['id'] for r in self.results}],
                 'physical_requests': len(self.ledger.records), 'usage': summarize_usage(receipts),
                 'money_cost': None, 'money_cost_reason': '未核实当前供应商计费与缓存价格',
                 'results': [{k: r.get(k) for k in ['id', 'physical_requests', 'elapsed_seconds', 'execution_error', 'structural_checks', 'semantic_verdict']} for r in self.results]}
        write_json(self.out / 'summary.json', value)
        return value


def structural_checks(record: dict) -> dict:
    """只核对可程序证明的边界；语义通过必须另行原文核验。"""
    tasks = record['tasks']
    calls = {r['call_id']: r for t in tasks for r in t['receipts']}
    actions = {a['action_id']: a for t in tasks for a in t['actions']}
    saves = [a for a in actions.values() if a['tool_name'] == 'save_note' and a['status'] == 'succeeded']
    deny = record['input']['save_policy'] == 'deny'
    return {'all_tasks_settled': bool(tasks) and all(t['view']['status'] in TERMINAL for t in tasks),
            'no_unknown': summarize_usage(list(calls.values()))['unknown'] == 0,
            'save_permission_respected': not (deny and saves),
            'has_read_evidence': any(t['evidence'] for t in tasks),
            'saved_action_count': len(saves),
            'semantic_support': 'not_evaluated_by_code'}


def main():
    """冻结输入和源码后执行；缺显式--real时只做清单校验。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=str(ROOT / 'backend/tests/acceptance/ag6_cases.json'))
    parser.add_argument('--output')
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    validate_manifest(manifest)
    if not args.real:
        print('AG6 manifest valid; no model calls'); return
    out = Path(args.output).resolve()
    url = os.environ.get('AG6_TEST_DATABASE_URL', '')
    name = validate_isolation(url, out)
    if out.exists(): raise ValueError('批次目录已存在，不能覆盖/重跑')
    for material in manifest['materials'].values():
        if digest((ROOT / material['path']).read_bytes()) != material['sha256']:
            raise ValueError('material hash changed')
    settings = Settings.from_env(ROOT).model_copy(update={
        'database_url': url, 'data_dir': out / 'data', 'output_dir': out / 'notes'})
    if not settings.deepseek_api_key: raise ValueError('模型密钥未配置')
    # 只创建命名隔离数据库，绝不DROP或清空已有数据库。
    import pymysql
    parsed = urlparse(url)
    with pymysql.connect(host=parsed.hostname, port=parsed.port or 3306,
                         user=parsed.username, password=parsed.password) as conn:
        with conn.cursor() as cur: cur.execute('CREATE DATABASE IF NOT EXISTS `'+name+'` CHARACTER SET utf8mb4')
    out.mkdir(parents=True)
    freeze = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sorted((ROOT / 'backend').rglob('*.py'))}
    write_json(out / 'freeze.json', {'manifest': manifest, 'code': freeze,
                                    'model': settings.deepseek_model, 'database_name': name,
                                    'frozen_at': datetime.now(timezone.utc).isoformat()})
    runner = EvaluationRunner(settings, out, manifest)
    try:
        runner.run()
    finally:
        result = runner.summary()
        result['frozen_code_unchanged'] = all(digest((ROOT / p).read_bytes()) == h for p, h in freeze.items())
        write_json(out / 'summary.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
