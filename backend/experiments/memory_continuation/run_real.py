"""逐轮执行独立接续pilot；只将冻结提示交给真实模型，来源与用量完整留证。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'backend'), str(Path(__file__).parent)]
TOOLS = {'set_research_plan', 'finish_task', 'get_profile', 'search_memory', 'read_memory', 'report_task_gap'}
LIMITS = {'max_requests': 12, 'token_limit': 160000, 'seconds': 600, 'output_tokens': 8192}


def write_json(path: Path, value) -> None:
    """只在本轮新工件内写审计对象，不输出凭据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def sha(path: Path) -> str:
    """计算原字节身份，不导入配置或业务服务。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict:
    """冻结候选源码及新入口；本机docs/artifacts和环境不加入公开源码清单。"""
    names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-co', '--exclude-standard', '-z']).decode().split('\0')
    return {name: sha(ROOT / name) for name in sorted(set(names)) if name and (ROOT / name).is_file()}


def credential_values(path: Path) -> dict:
    """仅在内存提取当前模型的三项配置，绝不复制私有配置文件。"""
    wanted = {'DEEPSEEK_API_KEY', 'DEEPSEEK_MODEL', 'DEEPSEEK_BASE_URL'}
    values = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.strip().partition('=')
        if separator and key in wanted:
            values[key] = value.strip().strip('\"\'')
    if not values.get('DEEPSEEK_API_KEY'):
        raise ValueError('未配置模型凭据；没有发送请求')
    values.setdefault('DEEPSEEK_MODEL', 'deepseek-chat')
    values.setdefault('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    from urllib.parse import urlparse
    endpoint = urlparse(values['DEEPSEEK_BASE_URL'])
    if (endpoint.scheme != 'https' or endpoint.hostname != 'api.deepseek.com'
            or endpoint.port not in {None, 443} or endpoint.path not in {'', '/', '/v1', '/v1/'}
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError('只允许已配置的官方模型端点；没有发送请求')
    return values


def public_model_ips(rows: list) -> set[str]:
    """只接受官方DNS结果中的公网单播地址，拒绝回环与私网旁路。

    Args:
        rows: getaddrinfo 返回的地址记录。

    Returns:
        已检查的IPv4或IPv6地址；不会调用DNS或建立连接。

    Raises:
        PermissionError: 结果为空、地址无效或包含非公网地址。
    """
    addresses = set()
    for row in rows:
        try:
            address = row[4][0]
            if not isinstance(address, str):
                raise ValueError('address must be text')
            parsed = ipaddress.ip_address(address)
        except (IndexError, TypeError, ValueError) as exc:
            raise PermissionError('拒绝无效的模型DNS结果') from exc
        if '%' in address or not parsed.is_global or parsed.is_multicast:
            raise PermissionError('拒绝模型DNS中的非公网地址')
        addresses.add(address)
    if not addresses:
        raise PermissionError('模型DNS没有可用公网地址')
    return addresses


def semantic_review_binding(output: Path, turn_id: str, task_id: str) -> dict:
    """生成逐轮审阅应绑定的实际工件身份，不产生通过结论。"""
    directory = output / turn_id
    return {'turn_id': turn_id, 'task_id': task_id,
            'artifact_sha256': {name: sha(directory / name)
                for name in ('answer.md', 'memory.json', 'trace.json')}}


def validate_semantic_review(output: Path, turn_id: str, task_id: str) -> dict:
    """核对离线语义审阅身份；关系的语义正确性由审阅者明确逐项判断。

    Args:
        output: 当前批次隔离输出目录。
        turn_id: 已执行且即将作为前置条件的轮次。
        task_id: 该轮真实创建的根任务身份。

    Returns:
        身份、工件和逐项判据都通过的审阅记录。

    Raises:
        ValueError: 审阅缺失、错绑、工件变化或必要语义项没有明确通过。
    """
    directory = output / turn_id
    try:
        review = json.loads((directory / 'semantic-review.json').read_text())
        binding = semantic_review_binding(output, turn_id, task_id)
        memory = json.loads((directory / 'memory.json').read_text())
        trace = json.loads((directory / 'trace.json').read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError('上一轮语义审阅或实际工件缺失，停止后续真实调用') from exc
    if (not isinstance(review, dict) or review.get('passed') is not True
            or any(review.get(key) != value for key, value in binding.items())):
        raise ValueError('上一轮语义审阅未明确通过或未绑定实际任务和工件')
    if (not isinstance(memory, dict) or memory.get('task_id') != task_id
            or not isinstance(trace, dict) or not isinstance(trace.get('root'), dict)
            or trace['root'].get('task_id') != task_id):
        raise ValueError('上一轮实际工件不属于该根任务')
    checks = review.get('checks')
    if (not isinstance(checks, dict)
            or any(checks.get(key) is not True for key in ('answer_semantics', 'summary_semantics'))):
        raise ValueError('上一轮回答及摘要语义尚未逐项审阅通过')
    note = memory.get('continuation_memory') if isinstance(memory, dict) else None
    if not summary_has_conclusions(note):
        raise ValueError('上一轮没有可审阅的非空接续结论')
    if turn_id == 'T2':
        relations = note.get('corrections')
        assessments = review.get('correction_target_reviews')
        if (not isinstance(relations, list) or not relations
                or not isinstance(assessments, list) or len(assessments) != len(relations)):
            raise ValueError('T2需要逐条审阅更新关系的具体目标')
        fields = ('previous_task_id', 'previous_key', 'previous_content_hash', 'replacement_key', 'relation')
        expected = [tuple(item.get(key) for key in fields) for item in relations if isinstance(item, dict)]
        assessed = []
        for item in assessments:
            if (not isinstance(item, dict) or item.get('passed') is not True
                    or not isinstance(item.get('reason'), str) or not item['reason'].strip()):
                raise ValueError('T2关系目标的人工语义审阅未明确通过')
            assessed.append(tuple(item.get(key) for key in fields))
        if (any(not isinstance(value, str) or not value for row in [*expected, *assessed] for value in row)
                or len(expected) != len(relations) or len(set(expected)) != len(expected)
                or sorted(expected) != sorted(assessed)):
            raise ValueError('T2语义审阅未绑定实际更新关系')
    return review


def summary_has_conclusions(note: dict) -> bool:
    """只承认已保存且含非空结论的摘要，状态applied本身不等于覆盖完成。"""
    if not isinstance(note, dict):
        return False
    conclusions = note.get('conclusions')
    return (note.get('status') == 'applied' and isinstance(conclusions, list) and bool(conclusions)
            and all(isinstance(item, dict) and isinstance(item.get('text'), str)
                    and bool(item['text'].strip()) for item in conclusions))


def sanitized_environment() -> None:
    """只修改本测试进程的环境，移除服务、追踪及代理继承配置。"""
    for key in list(os.environ):
        if (key.startswith(('RESEARCHLOOP_', 'DEEPSEEK_', 'SUPABASE_', 'OPENALEX_', 'LANGCHAIN_', 'LANGSMITH_'))
                or key == 'DATABASE_URL' or key.endswith('_TEST_DATABASE_URL')):
            os.environ.pop(key)
    os.environ['LANGSMITH_TRACING'] = 'false'


def isolated_settings(output: Path, values: dict):
    """重新构造隔离设置，生产的数据库、目录、工具和预算均不继承。"""
    from research_loop.config import Settings
    runtime = output / 'runtime'
    return Settings(project_root=runtime, data_dir=runtime / 'data', output_dir=runtime / 'notes',
        database_url='sqlite:///' + str(runtime / 'pilot.sqlite'),
        deepseek_api_key=values['DEEPSEEK_API_KEY'], deepseek_model=values['DEEPSEEK_MODEL'],
        deepseek_base_url=values['DEEPSEEK_BASE_URL'], request_timeout_seconds=90,
        retrieval_enabled=False, discovery_backends={}, web_search_enabled=False, web_fetch_enabled=False,
        execution_budget_mode='enforce', max_model_requests=5, max_tool_actions=12, reserve_model_requests=1,
        task_token_limit=LIMITS['token_limit'], task_token_tail=0, researcher_concurrency=1,
        context_output_reserve=LIMITS['output_tokens'], max_reflection_rounds=0)


def prepare(output: Path, credentials: Path) -> None:
    """冻结本次用户授权、源文件、三轮提示及停止条件，不调用模型。"""
    if output.exists():
        raise ValueError('本轮输出目录已经存在，拒绝覆盖')
    values = credential_values(credentials)
    sanitized_environment()
    output.mkdir(parents=True)
    (output / 'runtime').mkdir()
    settings = isolated_settings(output, values)
    scenarios = json.loads((Path(__file__).parent / 'real_scenarios.json').read_text())
    freeze = {'schema_version': 1, 'business_base_commit': subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip(), 'source_manifest': source_manifest(),
        'scenario_sha256': sha(Path(__file__).parent / 'real_scenarios.json'),
        'credential_file_sha256': sha(credentials), 'model': settings.deepseek_model,
        'base_url': settings.deepseek_base_url, 'limits': LIMITS, 'tools': sorted(TOOLS),
        'authorization': '用户2026-10-26授权继续隔离修复零论文任务规划类型，先用已有失败记录离线回归，通过后受控真实Memory；保留GitHub回滚点和失败，控制模型预算，不影响部署。',
        'scope': '一个合成场景、三轮真实调用；不是自然压缩或Memory因果收益实验',
        'stop_conditions': ['任何HTTP拒绝/unknown/缺usage', '任一工具失败', '摘要或更新关系未通过',
                            '语义审阅未通过', '预算/时限/隔离不符'],
        'output': str(output), 'settings': {'database_url': settings.database_url,
            'project_root': str(settings.project_root), 'data_dir': str(settings.data_dir),
            'output_dir': str(settings.output_dir), 'save_policy': 'deny', 'output_tokens': settings.context_output_reserve}}
    write_json(output / 'freeze.json', freeze)
    write_json(output / 'scenario-review.json', scenarios)
    write_json(output / 'state.json', {'turns': {}})
    print(json.dumps({'prepared': True, 'model': settings.deepseek_model, 'limits': LIMITS,
                      'new_model_calls': 0, 'output': str(output)}, ensure_ascii=False))


def execute_turn(output: Path, credentials: Path, turn_id: str) -> None:
    """只执行一个尚未创建的根；下一轮必须先完成上一轮真实输出的离线审阅。"""
    freeze = json.loads((output / 'freeze.json').read_text())
    if freeze['source_manifest'] != source_manifest() or freeze['credential_file_sha256'] != sha(credentials):
        raise ValueError('来源或凭据文件身份变化，未发送请求')
    scenarios = json.loads((Path(__file__).parent / 'real_scenarios.json').read_text())
    turns = scenarios['turns']
    turn = next(item for item in turns if item['id'] == turn_id)
    position = turns.index(turn)
    state = json.loads((output / 'state.json').read_text())
    if turn_id in state['turns']:
        raise ValueError('该轮已经创建；禁止重跑或恢复发送')
    if position:
        previous = turns[position - 1]['id']
        previous_task = state['turns'].get(previous, {}).get('task_id')
        if not isinstance(previous_task, str) or not previous_task:
            raise ValueError('上一轮根任务身份缺失，停止后续真实调用')
        validate_semantic_review(output, previous, previous_task)
        if not state['turns'].get(previous, {}).get('mechanism_ready'):
            raise ValueError('上一轮机制未通过，停止后续真实调用')
    values = credential_values(credentials)
    sanitized_environment()
    settings = isolated_settings(output, values)
    if settings.deepseek_model != freeze['model'] or settings.deepseek_base_url != freeze['base_url']:
        raise ValueError('模型配置变化，未发送请求')
    import httpx
    from real_gate import GuardedTransport, SharedGate
    from research_loop.providers.deepseek import DeepSeekProvider, ProviderResponseError, RequestNotSent
    from research_loop.storage.mysql import Database
    from research_loop.storage.files import FileStore
    from research_loop.services.tasks import TaskService
    from research_loop.services.conversations import ConversationService
    from research_loop.domain.tasks import TaskCreate
    from research_loop.workers.executor import TaskExecutor
    from research_loop.storage.token_budget import actual_usage
    gate = SharedGate(output / 'http', max_requests=LIMITS['max_requests'], token_limit=LIMITS['token_limit'], seconds=LIMITS['seconds'])
    admission = gate.summary()
    if (admission.get('stop_reason') or admission['reserved_tokens']
            or admission['remaining_seconds'] <= 0 or admission['admitted_requests'] >= LIMITS['max_requests']):
        raise ValueError('全局发送门已关闭，未创建新根或发送请求')
    original_database = Database.__init__
    original_files = FileStore.__init__
    original_connect = socket.socket.connect
    original_getaddrinfo = socket.getaddrinfo
    allowed_ips = public_model_ips(original_getaddrinfo('api.deepseek.com', 443, type=socket.SOCK_STREAM))

    def database_guard(database, url, *args, **kwargs):
        """所有业务连接只允许本轮唯一SQLite文件。"""
        if url != settings.database_url:
            raise PermissionError('拒绝非本轮数据库')
        return original_database(database, url, *args, **kwargs)

    def files_guard(files, data_dir, output_dir, project_root):
        """文件服务只能绑定本轮runtime目录。"""
        if (Path(data_dir).resolve(), Path(output_dir).resolve(), Path(project_root).resolve()) != (
                settings.data_dir.resolve(), settings.output_dir.resolve(), settings.project_root.resolve()):
            raise PermissionError('拒绝非本轮文件服务')
        return original_files(files, data_dir, output_dir, project_root)

    def connect_guard(sock, address):
        """拒绝本机生产服务及官方模型以外的连接。"""
        if not isinstance(address, tuple) or address[0] not in allowed_ips or address[1] != 443:
            raise PermissionError('拒绝非模型端点网络连接')
        return original_connect(sock, address)

    def dns_guard(host, port, *args, **kwargs):
        """不解析其他外部服务或生产地址。"""
        if host not in {'api.deepseek.com', b'api.deepseek.com'} or port != 443:
            raise PermissionError('拒绝非模型端点DNS')
        rows = original_getaddrinfo(host, port, *args, **kwargs)
        allowed_ips.update(public_model_ips(rows))
        return rows

    class PilotProvider:
        """保留原生适配器与SSE，只在实际发送和结算边界增加全局门。"""

        def __init__(self, inner, db, task_id):
            """注入原生适配器及本轮隔离根身份。"""
            self.inner, self.db, self.task_id = inner, db, task_id
            self.model = inner.model

        def close(self):
            """关闭本根客户端，不启动后台任务。"""
            self.inner.client.close()

        def _invoke(self, method, *args, **kwargs):
            """首个工具错误后不追加模型调用；完整响应先交现有图持久化。"""
            if any(item['status'] == 'failed' for item in self.db.list_actions(self.task_id)):
                gate.stop('tool_error')
                raise RequestNotSent('首个工具错误后停止，不自动重试')
            if kwargs.get('max_tokens') != LIMITS['output_tokens']:
                raise RequestNotSent('隔离输出额度与冻结不符')
            supplied = args[1] if len(args) > 1 else kwargs.get('tools', [])
            if any(item['function']['name'] not in TOOLS for item in supplied or []):
                raise RequestNotSent('工具范围变化')
            try:
                raw = method(*args, **kwargs)
            except RequestNotSent:
                raise
            except ProviderResponseError as exc:
                gate.stop('provider_http_' + str(exc.status_code))
                raise
            except Exception:
                gate.stop('unknown_model_result')
                raise
            gate.settle(gate.active_index, raw.get('usage') or {})
            return raw

        def send(self, *args, **kwargs):
            """沿用原生非流式请求，不修补真实响应。"""
            return self._invoke(self.inner.send, *args, **kwargs)

        def send_stream(self, *args, **kwargs):
            """沿用原生流式请求和公开正文回调。"""
            return self._invoke(self.inner.send_stream, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(Database, '__init__', database_guard))
        stack.enter_context(patch.object(FileStore, '__init__', files_guard))
        stack.enter_context(patch('socket.socket.connect', connect_guard))
        stack.enter_context(patch('socket.socket.connect_ex', side_effect=PermissionError('拒绝旁路连接')))
        stack.enter_context(patch('socket.getaddrinfo', dns_guard))
        db = Database(settings.database_url)
        db.ensure_schema()
        conversations = ConversationService(db, settings)
        cid = state.get('conversation_id')
        if not cid or turn['conversation'] == 'new':
            cid = conversations.create('真实接续pilot RISK-SMALL-01', 'pilot-' + turn_id)['conversation_id']
        state['conversation_id'] = cid
        task_id = TaskService(db, settings).create(TaskCreate(prompt=turn['prompt'], save_policy='deny',
            conversation_id=cid, idempotency_key='real-pilot-' + turn_id)).task_id
        state['turns'][turn_id] = {'task_id': task_id, 'created': True}
        write_json(output / 'state.json', state)

        def provider_factory(current):
            """每根创建唯一原生客户端，发送次数由跨根gate统一记录。"""
            transport = GuardedTransport(gate)
            client = httpx.Client(transport=transport, timeout=90, trust_env=False, follow_redirects=False)
            inner = DeepSeekProvider(values['DEEPSEEK_API_KEY'], current.deepseek_model,
                current.deepseek_base_url, client=client)
            return PilotProvider(inner, db, task_id)

        started = time.monotonic()
        result = TaskExecutor(settings, db, provider_factory=provider_factory,
                              supervisor_tools=TOOLS, continuation_memory=True).execute(task_id)
        elapsed = time.monotonic() - started
        task = db.get_task(task_id)
        calls = db.list_model_calls(task_id)
        actions = db.list_actions(task_id)
        memory = db.fetchone('SELECT record_json FROM research_memories WHERE task_id=?', (task_id,))
        memory = json.loads(memory['record_json']) if memory else {}
        note = memory.get('continuation_memory') or {}
        known = [actual_usage(json.loads(call.get('usage_json') or '{}')) for call in calls]
        ready = (task['status'] == 'completed' and summary_has_conclusions(note) and bool(calls)
                 and all(call['status'] == 'succeeded' and count is not None for call, count in zip(calls, known))
                 and not any(item['status'] == 'failed' for item in actions)
                 and not gate.summary().get('stop_reason'))
        if turn_id == 'T2':
            ready = ready and bool(note.get('corrections')) and all(
                item['relation'] == 'updates'
                and item['previous_task_id'] == state['turns']['T1']['task_id']
                for item in note.get('corrections', []))
        directory = output / turn_id
        write_json(directory / 'result.json', result.model_dump(mode='json'))
        write_json(directory / 'trace.json', {'root': task, 'model_calls': calls, 'actions': actions})
        write_json(directory / 'memory.json', memory)
        (directory / 'answer.md').write_text(result.answer_markdown or '')
        audit = {'turn_id': turn_id, 'task_id': task_id, 'status': task['status'],
            'worker_elapsed_seconds': elapsed, 'model_receipts': len(calls),
            'known_total_tokens': sum(value for value in known if value is not None),
            'usage_missing_receipts': sum(value is None for value in known),
            'summary_status': note.get('status'), 'corrections': note.get('corrections', []),
            'mechanism_ready': ready, 'gate': gate.summary(), 'semantic_review': 'pending',
            'semantic_review_binding': semantic_review_binding(output, turn_id, task_id),
            'required_semantic_checks': ['answer_semantics', 'summary_semantics'],
            'correction_target_review_required': turn_id == 'T2'}
        write_json(directory / 'audit.json', audit)
        state['turns'][turn_id].update(mechanism_ready=ready, status=task['status'])
        write_json(output / 'state.json', state)
        if not ready:
            gate.stop('turn_mechanism_failed')
        print(json.dumps(audit, ensure_ascii=False))


def main() -> None:
    """显式准备与逐轮发送分开，默认没有真实请求动作。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'execute'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--turn', choices=['T1', 'T2', 'T3'])
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'artifacts'):
        parser.error('输出必须在当前worktree artifacts下')
    if args.mode == 'prepare':
        prepare(output, args.credentials.resolve())
    elif args.turn:
        execute_turn(output, args.credentials.resolve(), args.turn)
    else:
        parser.error('执行须指定一个turn，不自动跑下一轮')


if __name__ == '__main__':
    main()
