"""以隔离 HTTP、原生执行器和合成响应核对 H1 合同，不测真实模型收益。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import traceback
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]


class IsolationError(RuntimeError):
    """实验尝试越过独立存储、配置或网络边界。"""


class Boundary:
    """限制本进程所有线程的 Python 可审计写入和网络操作。"""

    def __init__(self, output: Path):
        """只允许向本次新建的输出目录写入，不授予业务资源访问权。"""
        self.output = output
        self.enabled = True
        self.phase = 'qualification'
        self.denials = []
        self.sqlite_paths = set()

    def reject(self, operation: str):
        """记录阻断类别，避免输出可能含凭据的调用参数。"""
        origin = [{'file': frame.filename, 'line': frame.lineno, 'function': frame.name}
                  for frame in traceback.extract_stack(limit=8)[:-1]]
        self.denials.append({'phase': self.phase, 'operation': operation, 'origin': origin})
        raise IsolationError('isolated qualification rejected: ' + operation)

    @staticmethod
    def fd_root(fd: int) -> Path:
        """还原相对 dir_fd 的实际目录，支持 Linux 与 macOS。"""
        if sys.platform == 'darwin':
            import fcntl
            raw = fcntl.fcntl(fd, 50, bytes(1024))  # macOS F_GETPATH。
            return Path(os.fsdecode(raw.split(b'\0', 1)[0])).resolve()
        return Path('/proc/self/fd/' + str(fd)).resolve()

    def allowed_path(self, value, fd: int = -1) -> bool:
        """路径先解析符号链接，防止用相对目录或链接写入日常实例。"""
        if isinstance(value, int):
            return True
        path = Path(os.fsdecode(value))
        if not path.is_absolute() and fd not in {-1, None}:
            path = self.fd_root(fd) / path
        return path.resolve().is_relative_to(self.output)

    def write_path(self, value, fd: int = -1):
        """拒绝输出根以外的文件与目录变更。"""
        if not self.allowed_path(value, fd):
            self.reject('write_outside_output')

    def audit(self, event: str, args):
        """覆盖文件变更、SQLite 连接、进程启动和套接字连接边界。"""
        if not self.enabled:
            return
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.bind', 'subprocess.Popen'}:
            self.reject(event)
        if event == 'open':
            path, mode, flags = args
            if not isinstance(path, int) and Path(os.fsdecode(path)).name == '.env':
                self.reject('dotenv_read')
            writes = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if writes:
                self.write_path(path)
        elif event == 'sqlite3.connect':
            path = args[0]
            if path != ':memory:' and not self.allowed_path(path):
                self.reject('sqlite_outside_output')
            self.sqlite_paths.add(str(path))
        elif event in {'os.remove', 'os.rmdir'}:
            self.write_path(args[0], args[1])
        elif event in {'os.mkdir', 'os.chmod'}:
            self.write_path(args[0], args[2])
        elif event in {'os.rename', 'os.link'}:
            self.write_path(args[0], args[2])
            self.write_path(args[1], args[3])
        elif event in {'os.truncate', 'os.chown', 'os.chflags'}:
            self.write_path(args[0])
        elif event == 'os.utime':
            self.write_path(args[0], args[3])
        elif event == 'os.symlink':
            self.reject('symlink_creation')


def write_json(path: Path, value):
    """写本轮可复核 JSON，不输出模型凭据或请求头。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def settings_patches(stack: ExitStack, boundary: Boundary, scratch: Path):
    """对现有测试的显式与隐式配置接点注入统一隔离约束。"""
    from research_loop.config import Settings
    from research_loop import config
    original_init = Settings.__init__

    def isolated_init(self, **values):
        """保留测试自有参数，服务默认值一律指向无网络的隔离边界。"""
        values.setdefault('retrieval_enabled', False)
        values.setdefault('qdrant_url', 'http://127.0.0.1:1')
        values.setdefault('discovery_backends', {})
        values.setdefault('web_search_enabled', False)
        values.setdefault('web_fetch_enabled', False)
        original_init(self, **values)
        for name in ('project_root', 'data_dir', 'output_dir'):
            if not getattr(self, name).resolve().is_relative_to(boundary.output):
                boundary.reject('settings_path_outside_output')
        if not self.database_url.startswith('sqlite:///'):
            boundary.reject('non_sqlite_database')
        if not boundary.allowed_path(self.database_url.removeprefix('sqlite:///')):
            boundary.reject('settings_database_outside_output')
        if self.deepseek_api_key or self.supabase_key or self.openalex_api_key:
            boundary.reject('unexpected_provider_credentials')

    stack.enter_context(patch.object(Settings, '__init__', isolated_init))
    fallback = Settings(project_root=scratch / 'default', data_dir=scratch / 'default/data',
                        output_dir=scratch / 'default/notes',
                        database_url='sqlite:///' + str(scratch / 'default/default.db'))
    fallback.project_root.mkdir(parents=True)
    stack.enter_context(patch.object(config, 'get_settings', return_value=fallback))
    stack.enter_context(patch.object(Settings, 'from_env', side_effect=lambda *a, **k: boundary.reject('Settings.from_env')))
    stack.enter_context(patch.object(config, '_load_dotenv', side_effect=lambda *a, **k: boundary.reject('_load_dotenv')))
    from research_loop import main
    from research_loop.api import dependencies, health
    for module in (main, dependencies, health):
        stack.enter_context(patch.object(module, 'get_settings', return_value=fallback))
    return Settings


def guard_checks(boundary: Boundary):
    """用拒绝例证明守卫先于实际文件访问和网络连接生效。"""
    from research_loop.config import Settings
    boundary.phase = 'guard_selfcheck'
    results = []
    probes = [('from_env', lambda: Settings.from_env()),
              ('dotenv', lambda: (ROOT / '.env').read_text()),
              ('foreign_write', lambda: (ROOT / 'harness-forbidden.tmp').write_text('blocked')),
              ('network', lambda: socket.create_connection(('127.0.0.1', 9)))]
    for name, call in probes:
        try:
            call()
        except IsolationError:
            results.append({'case': name, 'blocked_before_operation': True})
        else:
            raise AssertionError('guard selfcheck did not block: ' + name)
    boundary.phase = 'qualification'
    return results


def run_existing(output: Path):
    """复用阶段进展和原生交接用例，真实子进程场景明确留待独立资格。"""
    from unit.test_research_progress import ResearchProgressTests
    from integration.test_research_progress_handoff import ProgressHandoffTests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([loader.loadTestsFromTestCase(ResearchProgressTests)])
    excluded = []
    for name in loader.getTestCaseNames(ProgressHandoffTests):
        if name == 'test_worker_sigkill_after_progress_commit':
            excluded.append({'test': 'ProgressHandoffTests.' + name,
                             'reason': 'process-local patches do not propagate to the SIGKILL subprocess'})
        else:
            suite.addTest(ProgressHandoffTests(name))
    with (output / 'unittest.log').open('w') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    return {'tests_run': result.testsRun, 'failed': len(result.failures),
            'errors': len(result.errors), 'skipped': len(result.skipped),
            'successful': result.wasSuccessful(), 'excluded_not_run': excluded,
            'failure_ids': [case.id() for case, _ in result.failures + result.errors]}


def run_probe(output: Path, Settings):
    """HTTP 受理后运行原生角色图，捕获原生供应商适配器的最终请求 schema。"""
    import httpx
    from fastapi.testclient import TestClient
    from research_loop.api.dependencies import get_runtime_settings, get_database
    from research_loop.main import app
    from research_loop.providers.deepseek import DeepSeekProvider
    from research_loop.storage.mysql import Database
    from research_loop.workers.executor import TaskExecutor
    from integration.test_research_progress_handoff import ProgressScenario, FixturePDF

    root = output / 'runtime/probe'
    root.mkdir(parents=True)
    settings = Settings(project_root=root, data_dir=root / 'data', output_dir=root / 'notes',
                        database_url='sqlite:///' + str(root / 'probe.db'))
    db = Database(settings.database_url)
    db.ensure_schema()
    scenario = ProgressScenario()
    requests = []
    clients = []

    def handle(request):
        """以完整合成 delta 经过原生流解析器，不修改 Agent 或工具行为。"""
        body = json.loads(request.content)
        tools = body.get('tools', [])
        names = {tool['function']['name'] for tool in tools}
        role = 'researcher' if 'finish_research' in names else 'supervisor'
        requests.append({'sequence': len(requests) + 1, 'role': role,
                         'model': body['model'], 'stream': body.get('stream', False), 'tools': tools})
        raw = scenario.send(body['messages'], tools)
        if not body.get('stream'):
            return httpx.Response(200, json=raw)
        choice = raw['choices'][0]
        message = choice['message']
        delta = {'role': 'assistant', 'content': message.get('content', '')}
        if message.get('tool_calls'):
            delta['tool_calls'] = [dict(call, index=index) for index, call in enumerate(message['tool_calls'])]
        chunk = {'id': raw['id'], 'model': body['model'], 'usage': raw['usage'],
                 'choices': [{'index': 0, 'delta': delta, 'finish_reason': choice['finish_reason']}]}
        wire = 'data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n'
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=wire.encode())

    def factory(_):
        """只使用内存 MockTransport；假凭据不具备外部服务权限。"""
        client = httpx.Client(transport=httpx.MockTransport(handle))
        clients.append(client)
        return DeepSeekProvider('synthetic-fixture', client=client)

    old_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_runtime_settings] = lambda: settings
    app.dependency_overrides[get_database] = lambda: db
    try:
        with TestClient(app) as client:
            attachments = []
            for name in ('A', 'B'):
                path = root / (name + '.pdf')
                path.write_bytes(b'%PDF-synthetic')
                attachments.append({'input_ref': 'input-' + name, 'paper_id': 'paper-' + name,
                                    'local_path': str(path)})
            created = client.post('/api/tasks', json={
                'prompt': '比较两份附件语言机制，不保存', 'attachments': attachments,
                'save_policy': 'deny', 'idempotency_key': 'harness-h1-synthetic-probe'})
            assert created.status_code == 202, created.text
            tid = created.json()['task_id']
            assert created.json()['status'] == 'queued'
            TaskExecutor(settings, db, provider_factory=factory,
                         pdf_factory=lambda *_: FixturePDF()).execute(tid, 'harness-h1-fixture')
            paths = {'task': '/api/tasks/' + tid,
                     'events': '/api/tasks/' + tid + '/events',
                     'subtasks': '/api/tasks/' + tid + '/subtasks',
                     'stream': '/api/tasks/' + tid + '/stream'}
            responses = {key: client.get(path) for key, path in paths.items()}
            assert all(response.status_code == 200 for response in responses.values())
            view = responses['task'].json()
            assert view['status'] == 'completed', view
            assert 'input-A' in view['result']['answer_markdown']
            assert 'input-B' in view['result']['answer_markdown']
            events = responses['events'].json()['events']
            assert {'task_created', 'worker_started', 'task_finished'} <= {e['event_type'] for e in events}
            children = responses['subtasks'].json()['subtasks']
            assert len(children) == 2
            assert 'event: settled' in responses['stream'].text
            assert '"status":"completed"' in responses['stream'].text
            assert not list(root.rglob('*.md'))
            assert view['usage']['unknown'] == 0
            assert view['usage']['requests'] == len(requests)
            for role in ('supervisor', 'researcher'):
                rows = [row for row in requests if row['role'] == role]
                assert rows, role
                assert any('record_research_progress' in {t['function']['name'] for t in row['tools']} for row in rows)
                write_json(output / 'schemas' / (role + '.json'), {'role': role, 'requests': rows})
            write_json(output / 'probe-task-view.json', view)
            write_json(output / 'probe-events.json', events)
            write_json(output / 'probe-subtasks.json', children)
            (output / 'probe-terminal-sse.txt').write_text(responses['stream'].text)
            return {'status': 'passed', 'http': {'create': 202, **{key: value.status_code for key, value in responses.items()}},
                    'root_tasks': 1, 'researcher_subtasks': 2,
                    'mock_provider_requests': len(requests),
                    'roles': {role: sum(row['role'] == role for row in requests) for role in ('supervisor', 'researcher')},
                    'synthetic_usage': view['usage'],
                    'usage_kind': 'synthetic fixture integers; not real model tokens',
                    'provider_boundary': 'native DeepSeekProvider with httpx.MockTransport; supervisor SSE and researcher JSON',
                    'terminal_sse_checked': True}
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(old_overrides)
        for client in clients:
            client.close()


def main() -> int:
    """接受全新输出目录；保留失败与原始日志，不覆盖既有实验结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error('qualification requires assertions; do not use -O or PYTHONOPTIMIZE')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / 'runtime/tmp'
    scratch.mkdir(parents=True)
    previous_tempdir, previous_cwd = tempfile.tempdir, Path.cwd()
    tempfile.tempdir = str(scratch)
    os.chdir(scratch)
    sys.path[:0] = [str(ROOT / 'backend'), str(ROOT / 'backend/tests')]
    boundary = Boundary(output)
    sys.addaudithook(boundary.audit)
    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'real_model_calls': 0, 'real_model_tokens': None,
              'quality_and_efficiency': 'not evaluated; zero-model qualification only'}
    try:
        with ExitStack() as stack:
            # urllib3 导入时会绑定 ::1 探测 IPv6；资格过程禁用该探测，网络守卫仍全部拒绝。
            stack.enter_context(patch.object(socket, 'has_ipv6', False))
            stack.enter_context(patch.dict(os.environ, {'PYTHONDONTWRITEBYTECODE': '1',
                'RESEARCHLOOP_READONLY': '0'}, clear=True))
            Settings = settings_patches(stack, boundary, output / 'runtime')
            report['guard_selfchecks'] = guard_checks(boundary)
            report['tests'] = run_existing(output)
            report['probe'] = run_probe(output, Settings)
        unexpected = [d for d in boundary.denials if d['phase'] != 'guard_selfcheck']
        report['status'] = 'passed' if report['tests']['successful'] and not unexpected else 'failed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['exception_type'] = type(exc).__name__
        (output / 'exception.log').write_text(traceback.format_exc())
    finally:
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        report['isolation'] = {'denials': boundary.denials,
                               'sqlite_databases': sorted(boundary.sqlite_paths),
                               'writes_restricted_to_output': True,
                               'Settings_from_env_and_dotenv_blocked': True,
                               'network_and_subprocess_blocked': True,
                               'environment_replaced_for_process_only': True,
                               'urllib3_import_ipv6_bind_probe_disabled': True}
        report['source_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path(__file__), ROOT / 'backend/research_loop/domain/research_progress.py',
                                               ROOT / 'backend/research_loop/agents/researcher.py',
                                               ROOT / 'backend/research_loop/services/research_progress.py',
                                               ROOT / 'backend/tests/unit/test_research_progress.py',
                                               ROOT / 'backend/tests/integration/test_research_progress_handoff.py']}
        write_json(output / 'result.json', report)
        boundary.enabled = False
        tempfile.tempdir = previous_tempdir
        os.chdir(previous_cwd)
    print(json.dumps({'status': report['status'], 'tests': report.get('tests'),
                      'probe': report.get('probe', {}).get('status'), 'output': str(output)}, ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
