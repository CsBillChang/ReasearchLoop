"""本机单用户启停、状态、诊断；只操作带确切身份记录的本项目进程。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from uuid import uuid4

from research_loop.config import Settings
from research_loop.services.runtime import (application_lock, atomic_json, heartbeat, identity, runtime_root,
    runtime_fingerprint, local_component_readiness, configure_runtime)
from research_loop.storage.mysql import Database

ROOT = Path(__file__).resolve().parents[2]


def process_signature(pid: int) -> str:
    """同时核验启动时间和完整命令，PID 重用不视为原进程。"""
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'command='], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ''


def owned(row: dict) -> bool:
    """记录、命令中的随机 token 与当前系统进程全部匹配才可发信号。"""
    signature = process_signature(row['pid'])
    return bool(signature and signature == row['signature'] and row['token'] in signature
                and str(Path(__file__).resolve()) in signature)


def available(port: int) -> bool:
    """以实际 bind 检查本机端口，不通过杀进程解决冲突。"""
    with socket.socket() as sock:
        try:
            # 正常停服留下 TIME_WAIT 不代表仍有监听者；与服务端一致允许地址复用。
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def probe(url: str) -> dict:
    """只读 HTTP 探测；错误只报告类型，避免包含口令。"""
    try:
        with urlopen(url, timeout=3) as response:
            body = response.read()
            return {'ok': response.status == 200, 'json': json.loads(body) if body.startswith(b'{') else None}
    except Exception as exc:
        return {'ok': False, 'error_type': type(exc).__name__}


def diagnose(settings: Settings) -> dict:
    """分开返回配置、数据库迁移、索引服务和恢复风险；始终零模型调用。"""
    result = {'model_key_present': bool(settings.deepseek_api_key), 'model': settings.deepseek_model,
              'database_ready': False, 'configuration_id': identity(settings),
              'qdrant': probe(settings.qdrant_url + '/healthz'), 'model_requests': 0,
              'runtime_fingerprint': runtime_fingerprint(settings),
              'local_components': local_component_readiness(settings)}
    try:
        db = Database(settings.database_url)
        versions = {r['version'] for r in db.fetchall('SELECT version FROM schema_version')}
        expected = {p.stem for p in (ROOT/'backend/migrations').glob('*.sql')}
        result.update(database_ready=expected <= versions, missing_migrations=sorted(expected - versions),
            tasks=db.fetchall('SELECT status,COUNT(*) AS n FROM tasks GROUP BY status'),
            unsettled_receipts=db.fetchall("SELECT status,COUNT(*) AS n FROM model_calls WHERE status IN ('started','unknown') GROUP BY status"))
    except Exception as exc:
        result['database_error_type'] = type(exc).__name__
    return result


class Manager:
    """管理一组本机进程；配置变更必须先停止，不默默接管其他实例。"""

    def __init__(self, settings: Settings, profile: str = 'daily'):
        if not re.fullmatch(r'[a-z0-9_-]{1,40}', profile):
            raise ValueError('INVALID_PROFILE')
        self.settings = settings
        self.profile = profile
        self.directory = runtime_root(ROOT)/profile
        self.directory.mkdir(exist_ok=True, mode=0o700)
        self.state_path = self.directory/'processes.json'

    @contextmanager
    def locked(self):
        """串行启停，防止两条本机命令同时接管同一 profile。"""
        with (self.directory/'manager.lock').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def read(self) -> dict:
        """读已登记进程；缺失表示尚未启动。"""
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else {'services': {}}

    def status(self) -> dict:
        """进程活着、HTTP 可响应、Worker 就绪分别展示，不混作模型可用。"""
        state = self.read()
        expected = runtime_fingerprint(self.settings)['fingerprint_id']
        state['expected_runtime_fingerprint_id'] = expected
        registered = (state.get('runtime_fingerprint') or {}).get('fingerprint_id')
        for name, row in state['services'].items():
            row['owned_alive'] = owned(row)
            beat = self.directory/(name+'.ready.json')
            report = json.loads(beat.read_text()) if beat.exists() else {}
            row['runtime'] = report if (report.get('token') == row['token'] and
                report.get('pid') == row['pid'] and row['owned_alive']) else None
            running = (row['runtime'] or {}).get('runtime_fingerprint_id')
            row['runtime_consistency'] = ('not_running' if not row['owned_alive'] else
                'unknown_legacy' if not registered or not running else
                'match' if expected == registered == running else 'mismatch')
        if state.get('api_port'):
            state['api'] = probe(f"http://127.0.0.1:{state['api_port']}/api/health")
        if state.get('frontend_port'):
            state['frontend'] = probe(f"http://127.0.0.1:{state['frontend_port']}/")
        return state

    def start(self, api_port: int = 18082, frontend_port: int = 4173,
              *, readonly: bool = False, scheduler: bool = False, worker: bool = True) -> dict:
        """预检通过后启动 API、前端及可选 Worker/简报；失败回收本次已启动进程。"""
        with self.locked(), application_lock(ROOT):
            previous = self.read()
            if any(owned(row) for row in previous['services'].values()):
                expected = (identity(self.settings), api_port, frontend_port, readonly, scheduler, worker and not readonly)
                actual = tuple(previous.get(k) for k in ('configuration_id','api_port','frontend_port','readonly','scheduler','worker'))
                if actual != expected:
                    raise RuntimeError('ALREADY_RUNNING_DIFFERENT_CONFIG')
                if not all(owned(row) for row in previous['services'].values()):
                    raise RuntimeError('PARTIAL_RUNTIME: 先停止后重启')
                status = self.status()
                comparisons = [row['runtime_consistency'] for row in status['services'].values()]
                if 'unknown_legacy' in comparisons:
                    raise RuntimeError('RUNTIME_FINGERPRINT_UNKNOWN: 旧进程缺少可核对指纹；请显式停止后启动')
                if any(value != 'match' for value in comparisons):
                    raise RuntimeError('RUNTIME_FINGERPRINT_MISMATCH: 活动进程与当前预期不同；未自动重启')
                return status
            if api_port == frontend_port or not all(1024 <= p <= 65535 and available(p) for p in (api_port, frontend_port)):
                raise RuntimeError('PORT_IN_USE_OR_INVALID')
            diag = diagnose(self.settings)
            if not diag['database_ready']:
                raise RuntimeError('DATABASE_NOT_READY: 请先 make infra 和 make migrate')
            if worker and not readonly and not self.settings.deepseek_api_key:
                raise RuntimeError('MODEL_KEY_MISSING: 配置 .env 或使用 --readonly')
            if not readonly and (self.settings.data_dir/'restore-hold.json').exists():
                raise RuntimeError('RESTORE_EXECUTION_HELD: 恢复实例先以 --readonly 检查，不自动执行旧任务')
            if not shutil.which('node') or not (ROOT/'frontend/node_modules/marked').exists():
                raise RuntimeError('FRONTEND_DEPENDENCIES_MISSING')
            state = {'configuration_id': identity(self.settings),
                'runtime_fingerprint': diag['runtime_fingerprint'], 'api_port': api_port,
                'frontend_port': frontend_port, 'readonly': readonly, 'scheduler': scheduler,
                'worker': worker and not readonly, 'services': {}}
            names = ['api', 'frontend'] + (['worker'] if worker and not readonly else []) + (['scheduler'] if scheduler and not readonly else [])
            try:
                for name in names:
                    token = uuid4().hex
                    env = {**os.environ, 'PYTHONPATH': str(ROOT/'backend'), 'PYTHONDONTWRITEBYTECODE': '1',
                        'DATABASE_URL': self.settings.database_url, 'RESEARCHLOOP_DATA_DIR': str(self.settings.data_dir),
                        'RESEARCHLOOP_OUTPUT_DIR': str(self.settings.output_dir),
                        'RESEARCHLOOP_RETRIEVAL_NAMESPACE': self.settings.retrieval_namespace,
                        'RESEARCHLOOP_READONLY': '1' if readonly else '0',
                        'RESEARCHLOOP_EXPECTED_RUNTIME_FINGERPRINT': state['runtime_fingerprint']['fingerprint_id'],
                        'RESEARCHLOOP_PROCESS_TOKEN': token, 'RESEARCHLOOP_HEARTBEAT': str(self.directory/(name+'.ready.json')),
                        'PORT': str(frontend_port), 'RESEARCHLOOP_API_URL': f'http://127.0.0.1:{api_port}'}
                    command = [sys.executable, str(Path(__file__).resolve()), 'serve', '--service', name,
                               '--token', token, '--api-port', str(api_port)]
                    log_path = self.directory/(name+'.log')
                    with log_path.open('ab') as log:
                        os.chmod(log_path, 0o600)
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
                    time.sleep(.15)
                    row = {'pid': process.pid, 'signature': process_signature(process.pid), 'token': token, 'log': str(log_path)}
                    state['services'][name] = row
                    atomic_json(self.state_path, state)
                    for _ in range(100):
                        if process.poll() is not None:
                            raise RuntimeError('SERVICE_EXITED: '+name)
                        report = self.status()['services'][name]['runtime']
                        if report and (name not in {'api','frontend'} or probe(f'http://127.0.0.1:{api_port}/api/health' if name == 'api' else f'http://127.0.0.1:{frontend_port}/')['ok']):
                            break
                        time.sleep(.1)
                    else:
                        raise RuntimeError('SERVICE_START_TIMEOUT: '+name)
            except Exception:
                self._stop(state, 15)
                raise
            return self.status()

    def _stop(self, state: dict, timeout: float) -> dict:
        """先停止新 HTTP 输入，再请求 Worker 在当前行动结束后退出；从不 SIGKILL。"""
        mismatched = []
        for name in ('api', 'frontend', 'scheduler', 'worker'):
            row = state['services'].get(name)
            if not row:
                continue
            if owned(row):
                os.kill(row['pid'], signal.SIGTERM)
            elif process_signature(row['pid']):
                mismatched.append(name)
        deadline = time.monotonic() + timeout
        while any(owned(row) for row in state['services'].values()) and time.monotonic() < deadline:
            time.sleep(.1)
        result = self.status()
        result['identity_mismatches_not_signalled'] = mismatched
        result['stopped'] = not any(row['owned_alive'] for row in result['services'].values()) and not mismatched
        return result

    def stop(self, timeout: float = 15) -> dict:
        """幂等停止本 profile；慢调用继续结算，报告 draining 而非强杀。"""
        with self.locked():
            return self._stop(self.read(), timeout)


def serve(service: str, api_port: int) -> None:
    """受管进程主体；共享锁持续到服务退出，保证冷备份期间不能启动。"""
    with application_lock(ROOT):
        configure_runtime(Settings.from_env(ROOT))
        if service == 'api':
            import uvicorn
            uvicorn.run('research_loop.main:app', host='127.0.0.1', port=api_port)
            return
        if service == 'frontend':
            child = subprocess.Popen(['node', str(ROOT/'frontend/dev-server.mjs')])
            def stop_child(signum, frame):
                """将退出请求交给唯一前端子进程，不影响同名的其他 Node 进程。"""
                child.terminate()
            signal.signal(signal.SIGTERM, stop_child)
            heartbeat('ready')
            raise SystemExit(child.wait())
        # Worker/Scheduler 自身实现收尾和共享锁，手工入口也受到保护。
        sys.argv = [sys.argv[0]]
        if service == 'worker':
            from scripts.run_worker import main
        else:
            from scripts.run_scheduler import main
        main()


def main() -> None:
    """CLI 输出可诊断 JSON；配置/启动错误非零退出，日志不包含密钥。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start','stop','status','diagnose','serve'])
    parser.add_argument('--profile', default='daily')
    parser.add_argument('--api-port', type=int, default=18082)
    parser.add_argument('--frontend-port', type=int, default=4173)
    parser.add_argument('--readonly', action='store_true')
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--no-worker', action='store_true')
    parser.add_argument('--service', choices=['api','frontend','worker','scheduler'])
    parser.add_argument('--token')
    args = parser.parse_args()
    if args.command == 'serve':
        serve(args.service, args.api_port)
        return
    settings = Settings.from_env(ROOT)
    manager = Manager(settings, args.profile)
    try:
        if args.command == 'start':
            result = manager.start(args.api_port, args.frontend_port, readonly=args.readonly, scheduler=args.scheduler, worker=not args.no_worker)
        elif args.command == 'diagnose':
            result = {'configuration': diagnose(settings), 'runtime': manager.status()}
        else:
            result = getattr(manager, args.command)()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == 'stop' and not result['stopped']:
            raise SystemExit(2)
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc


if __name__ == '__main__':
    main()
