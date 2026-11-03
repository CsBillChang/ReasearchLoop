"""冻结小样本的共同后端接线；凭据只经标准输入，默认不构造真实传输。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import httpx

from experiments.mechanism_pilot.budget import pilot_ledger
from experiments.mechanism_pilot.contract import RunContract, cancel_case, finalize_reservations
from experiments.mechanism_pilot.lifecycle import run_process
from experiments.mechanism_pilot.transport import QualificationTransport


def source_identity():
    """记录实际导入业务源码，A/B不能只靠调用方自报版本。"""
    import research_loop
    from research_loop.agents import memory
    from research_loop.tools import delegation
    return {name: {'path': str(Path(module.__file__).resolve()),
                   'sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
            for name, module in [('research_loop', research_loop), ('context', memory), ('delegation', delegation)]}


def execute_prepared_case(payload, *, inner_factory=None):
    """通过原生Worker、冻结资源与共享HTTP预算运行一格；无批准先拒绝。

    inner_factory仅供零模型资格注入MockTransport；真实池在完整授权检查后
    才由QualificationTransport调用工厂构造。本函数不读取部署.env。
    """
    from research_loop.config import Settings
    from research_loop.providers.deepseek import DeepSeekProvider
    from experiments.mechanism_pilot.contract import authorize_execution
    from experiments.mechanism_pilot.memory import execute_memory_case
    from experiments.mechanism_pilot.reflection import execute_case
    from experiments.mechanism_pilot.resources import FrozenResourcesFactory
    manifest = json.loads(Path(payload['manifest_path']).read_text())
    contract = RunContract(manifest, payload['authorization_path'])
    if contract.freeze_sha256 != payload['freeze_sha256']:
        raise PermissionError('MECHANISM_DRIVER_FREEZE_MISMATCH')
    phase, variant = payload['phase'], payload['variant']
    frozen_case = manifest.get('cases', {}).get(payload['case_id'])
    if (not frozen_case or [payload['case_id'], variant] not in manifest['cells']
            or phase != frozen_case['phase']
            or Path(payload['input_path']).resolve() != Path(frozen_case['input_path']).resolve()
            or payload['input_sha256'] != frozen_case['input_sha256']):
        raise PermissionError('MECHANISM_CASE_BINDING_MISMATCH')
    if phase == 'R' and (payload.get('resources_sha256') != manifest['resources']['sha256']
            or Path(payload['resources_manifest']).resolve() != Path(manifest['resources']['path']).resolve()):
        raise PermissionError('MECHANISM_RESOURCE_BINDING_MISMATCH')
    expected_backend = Path(manifest['business_roots'][variant]).resolve() / 'backend'
    imported = source_identity()
    if any(not Path(item['path']).is_relative_to(expected_backend) for item in imported.values()):
        raise PermissionError('MECHANISM_IMPORTED_SOURCE_MISMATCH')
    settings = Settings(**payload['settings'])
    mode = payload.get('execution_kind', 'substitute')
    if mode == 'real' and 'substitute_responses' in payload:
        raise PermissionError('MECHANISM_SUBSTITUTE_SCRIPT_IN_REAL_MODE')
    root = Path(settings.project_root).resolve()
    authorize = contract.authorizer(settings, root, phase=phase, real=mode == 'real')
    grant = authorize_execution(settings, mode, authorize, contract.freeze_sha256)
    if mode == 'real' and payload['started_at'] != grant['started_at']:
        raise PermissionError('MECHANISM_APPROVAL_CLOCK_MISMATCH')
    if mode == 'substitute' and inner_factory is None:
        responses = payload.get('substitute_responses')
        if not isinstance(responses, list) or not responses:
            raise PermissionError('MECHANISM_EXPLICIT_MOCK_REQUIRED')
        pending = iter(responses)

        def scripted(request):
            try:
                raw = next(pending)
                if json.loads(request.content).get('stream'):
                    message = json.loads(json.dumps(raw['choices'][0]['message']))
                    for index, call in enumerate(message.get('tool_calls', [])):
                        call['index'] = index
                    event = {'id': raw['id'], 'choices': [{'index': 0, 'delta': message,
                        'finish_reason': raw['choices'][0]['finish_reason']}], 'usage': raw['usage']}
                    return httpx.Response(200, content='data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n',
                        headers={'content-type': 'text/event-stream'})
                return httpx.Response(200, json=raw)
            except StopIteration as exc:
                raise RuntimeError('MECHANISM_SUBSTITUTE_SCRIPT_EXHAUSTED') from exc

        inner_factory = lambda: httpx.MockTransport(scripted)
    runtime_root = Path(manifest['runtime_root']).resolve()
    ledger_path = Path(payload['ledger_path']).resolve()
    if ledger_path.parent != runtime_root or ledger_path.name != 'budget.sqlite3':
        raise PermissionError('MECHANISM_LEDGER_OUTSIDE_BATCH')
    ledger = pilot_ledger(ledger_path, allowed_cells=manifest['cells'], start_time=payload['started_at'],
                          limits=manifest['limits'])
    if ledger.snapshot()['started_at'] != payload['started_at']:
        raise PermissionError('MECHANISM_LEDGER_CLOCK_MISMATCH')
    box = {}

    def created(task_id, db):
        ledger.begin_root(task_id, payload['case_id'], variant)
        box['root_id'] = task_id
        (root/'owned-task.json').write_text(json.dumps({'task_id': task_id}))

    def cancelled():
        from research_loop.storage.mysql import Database
        if 'root_id' not in box:
            return False
        task = Database(settings.database_url).get_task(box['root_id'])
        return bool(task.get('cancel_requested'))

    def provider_factory(current):
        limits = manifest['phases'][phase]
        transport = QualificationTransport(inner_factory if inner_factory is not None else
            lambda: httpx.HTTPTransport(retries=0), ledger, box['root_id'],
            input_token_limit=limits['input_token_limit'], max_output_tokens=limits['max_output_tokens'],
            directory=root/'http', mode=mode, authorize=authorize,
            freeze_sha256=contract.freeze_sha256, cancelled=cancelled, expected_model=current.deepseek_model)
        return DeepSeekProvider('synthetic-only' if mode == 'substitute' else current.deepseek_api_key,
            current.deepseek_model, current.deepseek_base_url, current.request_timeout_seconds,
            client=httpx.Client(transport=transport, timeout=current.request_timeout_seconds))

    common = dict(expected_input_sha256=payload['input_sha256'], settings=settings,
        provider_factory=provider_factory, stop_state=lambda: ledger.snapshot()['halt_reason'],
        execution_kind=mode, authorize=authorize, freeze_sha256=contract.freeze_sha256,
        on_task_created=created, recovery_policy=manifest.get('recovery_policy'))
    if phase == 'R':
        factory = FrozenResourcesFactory(payload['resources_manifest'], payload['resources_sha256'], prepare_lexical=True)
        result = execute_case(payload['input_path'], root, resource_factory=factory, **common)
    elif phase == 'M':
        result = execute_memory_case(payload['input_path'], root, **common)
    else:
        raise ValueError('MECHANISM_PHASE_NOT_FROZEN')
    if result.get('global_stop'):
        ledger.halt(result['global_stop'], operation='case_complete', root_id=box.get('root_id'))
    result['source_imports'] = imported
    result['budget'] = ledger.snapshot()
    (root/'qualified-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def launch_case(payload, runtime_root, *, deadline, interpreter=sys.executable):
    """一个案例一个进程组；同一批沿用固定时钟和账本，组退出后收尾。"""
    from research_loop.config import Settings
    from research_loop.storage.mysql import Database
    manifest = json.loads(Path(payload['manifest_path']).read_text())
    source = Path(manifest['business_roots'][payload['variant']]).resolve()
    adapters = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items() if not any(word in k.upper()
           for word in ('API_KEY', 'DATABASE_URL', 'SECRET', 'ACCESS_TOKEN'))}
    env.update(PYTHONPATH=os.pathsep.join([str(source/'backend'), str(adapters)]),
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    settings = Settings(**payload['settings'])
    contract = RunContract(manifest, payload['authorization_path'])
    if contract.freeze_sha256 != payload['freeze_sha256']:
        raise PermissionError('MECHANISM_DRIVER_FREEZE_MISMATCH')
    grant = contract.validate(settings, settings.project_root, phase=payload['phase'],
                              real=payload.get('execution_kind', 'substitute') == 'real')
    if grant.get('allow_real_model') and grant['started_at'] != deadline.started_wall:
        raise PermissionError('MECHANISM_APPROVAL_CLOCK_MISMATCH')
    unbounded = manifest['limits'].get('budget_mode') == 'unbounded'
    if (deadline.enforce_global_deadline == unbounded
            or deadline.admission_seconds != manifest['limits']['admission_seconds']
            or deadline.total_seconds != manifest['limits']['total_seconds']):
        raise PermissionError('MECHANISM_SUPERVISOR_DEADLINE_MISMATCH')
    batch_root = Path(manifest['runtime_root']).resolve()
    ledger_path = Path(payload['ledger_path']).resolve()
    if (ledger_path.parent != batch_root or ledger_path.name != 'budget.sqlite3'
            or Path(runtime_root).resolve().parent != batch_root
            or Path(settings.project_root).resolve().parent != Path(runtime_root).resolve()):
        raise PermissionError('MECHANISM_SUPERVISOR_PATH_MISMATCH')
    ledger = pilot_ledger(ledger_path, allowed_cells=manifest['cells'], start_time=payload['started_at'],
                          limits=manifest['limits'])
    if ledger.snapshot()['started_at'] != payload['started_at'] or payload['started_at'] != deadline.started_wall:
        raise PermissionError('MECHANISM_FIXED_START_MISMATCH')

    def stop_admission(reason):
        # 正常一格退出仍可继续固定矩阵；异常停止才关闭整批发送门。
        if reason != 'process_exited':
            ledger.halt(reason, operation='process_supervisor')

    def soft_cancel(reason):
        task_file = Path(settings.project_root)/'owned-task.json'
        if task_file.is_file():
            cancel_case(settings, Database(settings.database_url), json.loads(task_file.read_text())['task_id'])

    def finalize(record):
        task_file = Path(settings.project_root)/'owned-task.json'
        owner = json.loads(task_file.read_text())['task_id'] if task_file.is_file() else None
        finalize_reservations(ledger, {**record, 'owned_root_id': owner})

    report = run_process([interpreter, '-m', 'experiments.mechanism_pilot.runner'], runtime_root,
        deadline=deadline, env=env, payload=json.dumps(payload), soft_cancel=soft_cancel,
        stop_requested=lambda: ledger.snapshot()['halt_reason'], stop_admission=stop_admission,
        finalize_ledger=finalize)
    if report.get('returncode') not in (None, 0) or not report.get('cleanup_complete'):
        ledger.halt('case_process_or_cleanup_failed', operation='process_result')
    return report


if __name__ == '__main__':
    try:
        request = json.load(sys.stdin)
        result = execute_prepared_case(request)
        print(json.dumps({'case_id': result['case_id'], 'global_stop': result.get('global_stop')}))
        sys.exit(1 if result.get('global_stop') else 0)
    except Exception as exc:
        print(json.dumps({'error_type': type(exc).__name__, 'message': str(exc)}))
        sys.exit(2)
