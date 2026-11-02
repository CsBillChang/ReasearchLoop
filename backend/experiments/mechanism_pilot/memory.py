"""固定Memory接续入口：只安装合成历史，由当前原生Agent自主选择任务路线。"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx

from experiments.mechanism_pilot.reflection import deny_network, _strict_actual_usage
from experiments.mechanism_pilot.resources import validate_runtime
from experiments.mechanism_pilot.transport import QualificationTransport
from research_loop.domain.tasks import TaskCreate, TaskResult, TaskStatus
from research_loop.providers.deepseek import RequestNotSent
from research_loop.services.conversations import ConversationService
from research_loop.services.coverage import CoverageService
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor


def load_memory_case(path, expected_sha256):
    """读取独立问题/固定历史，不读取标签，不把期望路线预先填进任务。"""
    path = Path(path).resolve()
    if 'labels' in path.parts:
        raise PermissionError('MEMORY_LABEL_FILE_FORBIDDEN')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('MEMORY_CASE_DRIFT')
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get('case_id') not in {'M1', 'M2', 'M3', 'M4'}
            or not isinstance(value.get('prompt'), str) or not value['prompt'].strip()
            or set(value) - {'case_id', 'prompt', 'history', 'history_records'}):
        raise ValueError('MEMORY_NEUTRAL_INPUT_REQUIRED')
    if not isinstance(value.get('history', []), list) or any(not isinstance(s, str) or not s.strip() for s in value.get('history', [])):
        raise ValueError('MEMORY_HISTORY_REQUIRED')
    records = value.get('history_records', [])
    if (not isinstance(records, list) or any(not isinstance(row, dict)
            or set(row) != {'record_key', 'text'}
            or any(not isinstance(row[k], str) or not row[k].strip() for k in ('record_key', 'text')) for row in records)):
        raise ValueError('MEMORY_HISTORY_RECORD_REQUIRED')
    if len({row['record_key'] for row in records}) != len(records):
        raise ValueError('MEMORY_DUPLICATE_HISTORY_RECORD')
    return value


def install_history(db, settings, case):
    """通过正式会话/任务完成接口安装明确标记的合成前史，无模型收据。"""
    conversations = ConversationService(db, settings)
    tasks = TaskService(db, settings)
    cid = conversations.create('合成Memory固定历史', uuid4().hex)['conversation_id']
    seeds = []

    def seed(conversation, key, text):
        payload = TaskCreate(prompt='合成历史夹具记录 ' + key + '；仅供接续回归，不代表真实模型运行。',
            conversation_id=conversation, save_policy='deny', idempotency_key=uuid4().hex)
        identity = tasks.create(payload).task_id
        body = '[合成固定历史；非真实模型输出]\n' + text
        db.append_event(identity, 'mechanism_synthetic_history_seed', {
            'record_key': key, 'model_generated': False, 'model_receipt_created': False,
            'frozen_text_sha256': hashlib.sha256(text.encode()).hexdigest()})
        tasks.complete(identity, TaskStatus.completed, TaskResult(answer_markdown=body, answer_body_markdown=body))
        conversations.reconcile(identity)
        if db.list_model_calls(identity):
            raise ValueError('MEMORY_SEED_MUST_HAVE_NO_MODEL_RECEIPTS')
        seeds.append({'record_key': key, 'task_id': identity, 'conversation_id': conversation,
                      'model_generated': False, 'frozen_text_sha256': hashlib.sha256(text.encode()).hexdigest()})

    for index, text in enumerate(case.get('history', [])):
        seed(cid, f'{case["case_id"]}-history-{index + 1}', text)
    records = case.get('history_records', [])
    if records:
        source_cid = conversations.create('合成跨会话研究记录', uuid4().hex)['conversation_id']
        for row in records:
            seed(source_cid, row['record_key'], row['text'])
    return cid, seeds


def execute_memory_case(case_path, runtime_dir, *, expected_input_sha256, settings,
                        provider_factory, stop_state=lambda: None, execution_kind='substitute',
                        authorize=None, freeze_sha256=None, on_task_created=None):
    """通过TaskService和TaskExecutor执行本格，真实资格在建目录/任务前核对。

    A/B需分别由固定源码的独立Python进程调用，禁止同进程更换已导入业务模块。
    真实模式仍须新授权和共享QualificationTransport，本轮只运行替身资格。
    """
    from experiments.mechanism_pilot.contract import authorize_execution
    authorize_execution(settings, execution_kind, authorize, freeze_sha256)
    case = load_memory_case(case_path, expected_input_sha256)
    root = validate_runtime(runtime_dir, settings)
    stopped = stop_state()
    if stopped:
        return {'created': False, 'case_id': case['case_id'], 'global_stop': stopped}
    root.mkdir(parents=True, exist_ok=False)
    db, task_id, seeds, error = None, None, [], None
    gate = {'stop': None, 'first_failed_action': None}

    def failed_action_stop():
        if db is not None and task_id and gate['first_failed_action'] is None:
            rows = db.fetchall('SELECT a.* FROM actions a JOIN tasks t ON t.task_id=a.task_id '
                "WHERE (t.task_id=? OR t.root_task_id=?) AND a.status='failed' "
                'ORDER BY a.created_at,a.sequence_no,a.action_id LIMIT 1', (task_id, task_id))
            if rows:
                row = dict(rows[0])
                row['arguments'] = json.loads(row.pop('arguments_json'))
                row['observation'] = json.loads(row.pop('observation_json'))
                gate['first_failed_action'] = row
                gate['stop'] = gate['stop'] or 'failed_action:' + row['tool_name']
        return gate['stop']

    class GuardedProvider:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(self.original, name)

        def _invoke(self, method, *args, **kwargs):
            reason = failed_action_stop() or stop_state()
            if reason:
                raise RequestNotSent(str(reason))
            if execution_kind == 'real':
                authorize_execution(settings, execution_kind, authorize, freeze_sha256)
            try:
                raw = method(*args, **kwargs)
            except BaseException as exc:
                gate['stop'] = gate['stop'] or 'provider_error:' + type(exc).__name__
                raise
            if _strict_actual_usage(raw.get('usage')) is None:
                gate['stop'] = gate['stop'] or 'successful_call_usage_unknown'
            return raw

        def send(self, *args, **kwargs):
            return self._invoke(self.original.send, *args, **kwargs)

        def send_stream(self, *args, **kwargs):
            return self._invoke(self.original.send_stream, *args, **kwargs)

        def complete(self, *args, **kwargs):
            return self.original.parse_response(self.send(*args, **kwargs))

        def close(self):
            self.original.close()
            self.original.client.close()

    def factory(current):
        provider = provider_factory(current)
        transport = getattr(getattr(provider, 'client', None), '_transport', None)
        qualified = isinstance(transport, QualificationTransport) and transport.mode == execution_kind
        if not qualified and not (execution_kind == 'substitute' and isinstance(transport, httpx.MockTransport)):
            provider.close()
            raise PermissionError('MEMORY_QUALIFICATION_TRANSPORT_REQUIRED')
        return GuardedProvider(provider)

    guard = deny_network() if execution_kind == 'substitute' else nullcontext()
    with guard, patch('research_loop.config.get_settings', return_value=settings), \
            patch('research_loop.config.Settings.from_env', side_effect=PermissionError('MEMORY_DEFAULT_ENV_DENIED')):
        try:
            db = Database(settings.database_url)
            db.ensure_schema()
            cid, seeds = install_history(db, settings, case)
            task_id = TaskService(db, settings).create(TaskCreate(prompt=case['prompt'],
                conversation_id=cid, save_policy='deny', idempotency_key=uuid4().hex)).task_id
            if on_task_created is not None:
                on_task_created(task_id, db)
            db.append_event(task_id, 'mechanism_memory_case_input', {'case_id': case['case_id'],
                'input_sha256': expected_input_sha256, 'route_preselected': False,
                'synthetic_history': True, 'seed_task_ids': [s['task_id'] for s in seeds]})
            TaskExecutor(settings, db, provider_factory=factory).execute(task_id, 'memory-mechanism-pilot')
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        failed_action_stop()
        view = TaskService(db, settings).view(db.get_task(task_id)).model_dump(mode='json') if task_id else None
        actions = db.list_actions(task_id) if task_id else []
        rows = db.fetchall('SELECT m.* FROM model_calls m JOIN tasks t ON t.task_id=m.task_id '
            'WHERE t.task_id=? OR t.root_task_id=? ORDER BY m.created_at,m.call_id', (task_id, task_id)) if task_id else []
        plan = CoverageService(db).plan(task_id) if task_id else None
    usage_audit = []
    for row in rows:
        amount = _strict_actual_usage(json.loads(row.get('response_json') or '{}').get('usage'))
        usage_audit.append({'call_id': row['call_id'], 'task_id': row['task_id'],
            'usage_status': 'not_sent' if row['sent_state'] == 'not_sent' else 'known' if amount is not None else 'unknown',
            'actual_tokens': amount})
    unknown = sum(r['usage_status'] == 'unknown' for r in usage_audit)
    known = sum(r['actual_tokens'] for r in usage_audit if r['usage_status'] == 'known')
    history_actions = [a for a in actions if a['status'] == 'succeeded'
                       and a['tool_name'] in {'search_memory', 'read_memory'}]
    from research_loop.agents import memory as current_memory
    from research_loop.tools import delegation as current_delegation
    import research_loop
    source_imports = {}
    for name, module in [('research_loop', research_loop), ('context', current_memory), ('delegation', current_delegation)]:
        path = Path(module.__file__).resolve()
        source_imports[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    report = {'source_imports': source_imports, 'case_id': case['case_id'], 'created': bool(task_id), 'task_id': task_id,
        'execution_kind': execution_kind, 'real_model_calls': 0 if execution_kind == 'substitute' else None,
        'error': error, 'view': view, 'actions': actions, 'physical_receipts': rows,
        'strict_usage_audit': usage_audit, 'strict_usage_summary': {
            'known_tokens_subtotal': known, 'total_tokens': None if unknown else known,
            'unknown_receipts': unknown, 'raw_view_usage_is_not_qualified_for_billing': True},
        'synthetic_history_seeds': seeds, 'route_preselected': False, 'actual_plan': plan,
        'history_read_actions': history_actions, 'semantic_quality': 'not_automatically_evaluated',
        'first_failed_action': gate['first_failed_action'],
        'global_stop': gate['stop'] or stop_state() or ('case_exception:' + error['type'] if error else None)}
    (root / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
