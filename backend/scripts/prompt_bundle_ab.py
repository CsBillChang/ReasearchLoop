"""联合提示词的来源冻结与离线冷启动入口；不在准备阶段调用模型或RAG。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
from types import CodeType
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
KIND = 'prompt_bundle_v1'
VERSION = 'joint-prompt-source-v1'
BUSINESS_FILES = frozenset({
    'backend/research_loop/agents/graph.py',
    'backend/research_loop/agents/memory.py',
    'backend/research_loop/agents/prompt_parts.py',
    'backend/research_loop/agents/researcher.py',
    'backend/research_loop/agents/reviewer.py',
    'backend/research_loop/agents/supervisor.py',
    'backend/research_loop/services/tasks.py',
    'backend/research_loop/tools/delegation.py',
    'backend/research_loop/tools/memory.py',
    'backend/research_loop/tools/notes.py',
    'backend/research_loop/tools/papers.py',
})
NEW_PARTS = 'backend/research_loop/agents/prompt_parts.py'


def digest(value):
    """绑定原始字节，或稳定JSON编码的结构值。"""
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(value).hexdigest()


def inventory(root):
    """完整绑定参与后端运行的源码、共同脚本、迁移、配置和依赖锁。"""
    from harness_ab_experiment import source_inventory
    root = Path(root).resolve(strict=True)
    result = source_inventory(root)
    for name in ('backend/pyproject.toml', 'backend/uv.lock'):
        path = root / name
        if path.is_symlink():
            raise ValueError('dependency symlink forbidden')
        if path.is_file():
            result[name] = digest(path.read_bytes())
    return result


def build_contract(roots):
    """允许已审的联合提示词文件差异，其余运行源码必须逐字一致。"""
    if not isinstance(roots, dict) or set(roots) != {'A', 'B'}:
        raise ValueError('two independent source roots required')
    roots = {v: Path(p).resolve(strict=True) for v, p in roots.items()}
    a, b = roots['A'], roots['B']
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise ValueError('source roots must be independent')
    files = {v: inventory(p) for v, p in roots.items()}
    if set(files['B']) - set(files['A']) != {NEW_PARTS} or set(files['A']) - set(files['B']):
        raise ValueError('only prompt_parts may be added, no source removal')
    changed = {p for p in set(files['A']) | set(files['B']) if files['A'].get(p) != files['B'].get(p)}
    if changed != BUSINESS_FILES:
        raise ValueError('joint prompt file scope differs: ' + ','.join(sorted(changed ^ BUSINESS_FILES)))
    for variant in ('A', 'B'):
        for name in BUSINESS_FILES & set(files[variant]):
            ast.parse((roots[variant] / name).read_text())
    return {'version': VERSION, 'business_files': sorted(BUSINESS_FILES), 'variants': {
        v: {'source_tree_sha256': digest(files[v]),
            'business_file_sha256': {p: files[v].get(p) for p in sorted(BUSINESS_FILES)},
            'source_files': files[v]} for v in ('A', 'B')}}


def validate_sources(manifest):
    """每次启动及生成前重算冻结源码，不能借新标签调用旧业务候选。"""
    if manifest.get('experiment_kind') != KIND:
        raise PermissionError('PROMPT_BUNDLE_KIND_REQUIRED')
    if any(key in manifest for key in ('handoff_source_contract', 'supervisor_prompt_contract')):
        raise PermissionError('PROMPT_BUNDLE_LEGACY_CONTRACT_FORBIDDEN')
    try:
        expected = build_contract(manifest['source_roots'])
        if manifest.get('prompt_bundle_source_contract') != expected:
            raise ValueError('frozen bundle changed')
        for value in manifest.get('settings', {}):
            if value == 'context_deduplicate_read_paper':
                raise ValueError('dedup is not part of deployment baseline')
        if manifest.get('variant_settings') != {'A': {}, 'B': {}}:
            raise ValueError('no per-variant runtime switch allowed')
    except (OSError, KeyError, TypeError, ValueError, SyntaxError) as exc:
        raise PermissionError('PROMPT_BUNDLE_SOURCE_CONTRACT_INVALID') from exc
    return expected


def _literal(value):
    """只摘要可稳定序列化的运行常量，不输出凭据或对象repr。"""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (list, tuple)):
        return [_literal(x) for x in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_literal(x) for x in value)
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _literal(v) for k, v in value.items()}
    raise TypeError('not a literal')


def normalize_code(value):
    """以带类型的确定性结构绑定代码，不依赖marshal的对象引用编码。

    Args:
        value: 代码对象或其中的不可变常量。

    Returns:
        保留类型、字节和嵌套代码字段的JSON可编码结构。

    Raises:
        TypeError: 常量类型不受支持；不能静默省略未知内容。
    """
    if isinstance(value, CodeType):
        fields = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount',
                  'co_nlocals', 'co_stacksize', 'co_flags', 'co_code', 'co_consts',
                  'co_names', 'co_varnames', 'co_filename', 'co_name', 'co_qualname',
                  'co_firstlineno', 'co_linetable', 'co_exceptiontable',
                  'co_freevars', 'co_cellvars')
        return ['code', {key: normalize_code(getattr(value, key)) for key in fields}]
    if value is None:
        return ['none']
    if value is Ellipsis:
        return ['ellipsis']
    if type(value) is bool:
        return ['bool', value]
    if type(value) is int:
        return ['int', str(value)]
    if type(value) is float:
        # 二进制编码保留负零及NaN位，避免JSON数字合并不同常量。
        return ['float', struct.pack('!d', value).hex()]
    if type(value) is complex:
        return ['complex', normalize_code(value.real), normalize_code(value.imag)]
    if type(value) is str:
        return ['str', value.encode('utf-8', errors='surrogatepass').hex()]
    if type(value) is bytes:
        return ['bytes', value.hex()]
    if type(value) is slice:
        return ['slice', normalize_code(value.start), normalize_code(value.stop),
                normalize_code(value.step)]
    if type(value) is tuple:
        return ['tuple', [normalize_code(item) for item in value]]
    if type(value) is frozenset:
        items = [normalize_code(item) for item in value]
        return ['frozenset', sorted(items, key=lambda item: json.dumps(
            item, sort_keys=True, ensure_ascii=True, separators=(',', ':')))]
    raise TypeError('unsupported code constant type: ' + type(value).__name__)


def code_digest(code):
    """所有函数、类方法和工具handler共用版本化结构指纹。"""
    if not isinstance(code, CodeType):
        raise TypeError('code object required')
    return digest({'format': 'structured-code-v1', 'code': normalize_code(code)})


def loaded_identity(source):
    """检查实际模块位置、执行代码及六种角色装配，不能只核对磁盘字符串。"""
    source = Path(source).resolve(strict=True)
    backend = source / 'backend'
    modules = {}
    for name in sorted(BUSINESS_FILES):
        path = source / name
        if not path.exists() and name == NEW_PARTS:
            continue
        module_name = name.removeprefix('backend/').removesuffix('.py').replace('/', '.')
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve() != path:
            raise PermissionError('PROMPT_BUNDLE_LOADED_SOURCE_MISMATCH')
        values = {}
        for key, value in vars(module).items():
            if key.startswith('__'):
                continue
            if inspect.isfunction(value) and value.__module__ == module_name:
                values[key] = {'code_sha256': code_digest(value.__code__)}
            elif inspect.isclass(value) and value.__module__ == module_name:
                methods = {}
                for method_name, method in vars(value).items():
                    if isinstance(method, (staticmethod, classmethod)):
                        method = method.__func__
                    if isinstance(method, property):
                        for slot in ('fget', 'fset', 'fdel'):
                            function = getattr(method, slot)
                            if function is not None:
                                methods[method_name + '.' + slot] = code_digest(function.__code__)
                    elif inspect.isfunction(method):
                        methods[method_name] = code_digest(method.__code__)
                values[key] = {'methods': methods}
            else:
                try:
                    values[key] = {'literal_sha256': digest(_literal(value))}
                except (TypeError, ValueError):
                    pass
        modules[name] = digest(values)
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if name.startswith('research_loop.') and filename and not Path(filename).resolve().is_relative_to(backend):
            raise PermissionError('PROMPT_BUNDLE_MIXED_BUSINESS_MODULES')
    from research_loop.agents import supervisor, researcher, reviewer
    from research_loop.tools.registry import ToolRegistry
    from research_loop.tools.papers import specs as paper_specs
    from research_loop.tools.delegation import specs as delegation_specs
    from research_loop.tools.notes import specs as note_specs
    from research_loop.tools.memory import specs as memory_specs
    samples = {'supervisor': supervisor.SYSTEM_PROMPT}
    for mode in ('screen', 'investigate', 'deep_read'):
        samples['researcher_' + mode] = researcher.researcher_state(
            'synthetic-child', 'synthetic objective', attachments={'input-synthetic': {}},
            brief={'mode': mode, 'input_refs': ['input-synthetic']})['messages']
    samples['reviewer_note'] = reviewer.reviewer_state(
        'synthetic-review', 'synthetic-draft', '0' * 64, 'synthetic text', [], [], ['synthetic question'])['messages']
    samples['reviewer_target'] = reviewer.research_reviewer_state(
        'synthetic-review', {'target_id': 'synthetic-target', 'content_hash': '0' * 64}, ['synthetic question'], [])['messages']
    registries = {}
    for web_search in (False, True):
        for web_fetch in (False, True):
            for memory in (False, True):
                name = f'supervisor_websearch{int(web_search)}_webfetch{int(web_fetch)}_memory{int(memory)}'
                specs = paper_specs(web_search_enabled=web_search, web_fetch_enabled=web_fetch) + note_specs() + delegation_specs()
                registries[name] = ToolRegistry(specs + (memory_specs() if memory else []))
    # 捕获原角色工厂实际传给图的注册表；不构造图、服务、数据库或Provider。
    with patch.object(researcher, 'build_graph', side_effect=lambda provider, registry, *args, **kwargs: registry):
        registries['researcher'] = researcher.build_researcher_graph(None, {})
    with patch.object(reviewer, 'build_graph', side_effect=lambda provider, registry, *args, **kwargs: registry):
        for name, services in (('note', {'review_basis_required': True}),
                               ('legacy_note', {'review_basis_required': False}),
                               ('target', {'review_target': {'target_id': 'synthetic'}})):
            registries['reviewer_' + name] = reviewer.build_reviewer_graph(None, services)
    registry_identities = {}
    for name, registry in registries.items():
        identities = []
        for tool_name in registry.names():
            spec = registry.get(tool_name)
            identities.append({'schema': spec.schema(), 'roles': spec.roles,
                'handler': spec.handler.__module__ + '.' + spec.handler.__qualname__,
                'handler_code': code_digest(spec.handler.__code__)})
        registry_identities[name] = digest(identities)
    return {'code_fingerprint_format': 'structured-code-v1',
            'python_runtime': {'implementation': sys.implementation.name,
                               'version': list(sys.version_info[:3])},
            'modules': modules, 'assembled_prompt_sha256': {k: digest(v) for k, v in samples.items()},
            'registry_sha256': registry_identities}


def verify_loaded(manifest, variant):
    """正式格与每次传输都核对冷启动已冻结的实际装配身份。"""
    expected = manifest.get('loaded_prompt_contract', {}).get(variant)
    if not isinstance(expected, dict):
        raise PermissionError('PROMPT_BUNDLE_COLD_IDENTITY_REQUIRED')
    actual = loaded_identity(manifest['source_roots'][variant])
    if actual != expected:
        raise PermissionError('PROMPT_BUNDLE_LOADED_IDENTITY_MISMATCH')
    return actual


def validate_activation(manifest, mode, *, clock=time.time):
    """准备文件不能仅因借用旧approved记录而进行实时资格或真实生成。"""
    if mode not in {'qualify', 'execute'}:
        raise PermissionError('PROMPT_BUNDLE_LIVE_MODE_INVALID')
    from harness_ab_pair_gate import manifest_identity
    key = 'live_qualification_released' if mode == 'qualify' else 'real_execution_released'
    if manifest.get(key) is not True:
        raise PermissionError('PROMPT_BUNDLE_NEW_AUTHORIZATION_REQUIRED')
    path = Path(manifest['authorization_path'])
    if path.is_symlink():
        raise PermissionError('PROMPT_BUNDLE_AUTHORIZATION_SYMLINK')
    resolved = path.resolve(strict=True)
    protected = [Path(manifest['runtime_root']).resolve(),
                 *(Path(value).resolve() for value in manifest['source_roots'].values())]
    if any(resolved.is_relative_to(root) for root in protected):
        raise PermissionError('PROMPT_BUNDLE_AUTHORIZATION_OUTSIDE_MODEL_AND_SOURCE_REQUIRED')
    auth = json.loads(path.read_text())
    if (auth.get('approved') is not True or auth.get('experiment_kind') != KIND
            or auth.get('batch_id') != manifest.get('batch_id')
            or not isinstance(auth.get('batch_id'), str) or not auth['batch_id']
            or auth.get('manifest_sha256') != manifest_identity(manifest)
            or auth.get('ledger_limits') != manifest.get('ledger_limits')
            or auth.get('execution_order') != manifest.get('execution_order')
            or mode not in auth.get('allowed_modes', [])):
        raise PermissionError('PROMPT_BUNDLE_AUTHORIZATION_MISMATCH')
    if 'completion_contract' not in manifest and (
            type(auth.get('expires_at')) not in (int, float)
            or not math.isfinite(auth['expires_at']) or clock() >= auth['expires_at']):
        raise PermissionError('PROMPT_BUNDLE_AUTHORIZATION_MISMATCH')
    validate_round_clock(manifest, auth, clock=clock)
    return auth


def validate_round_clock(manifest, auth, *, clock=time.time):
    """资格开始、评分等待及执行共用授权起点，payload不能重置总时钟。"""
    started, deadline = auth.get('round_started_at'), auth.get('round_absolute_deadline')
    if 'completion_contract' in manifest:
        from agent_final_continuation import validate_continuation
        context = validate_continuation(manifest)
        contract = context['clock']
        if (contract.get('policy') != 'financial_only' or auth.get('policy') != 'financial_only'
                or auth.get('human_approved') is not True
                or 'expires_at' not in auth or auth['expires_at'] is not None
                or 'round_absolute_deadline' not in auth or deadline is not None
                or contract.get('model_absolute_deadline') is not None
                or contract.get('round_absolute_deadline') is not None
                or manifest.get('ledger_limits', {}).get('enforce_batch_seconds') is not False
                or type(started) not in (int, float) or not math.isfinite(started)
                or not 0 < started <= clock() or started != contract['round_started_at']):
            raise PermissionError('PROMPT_BUNDLE_COMPLETION_CLOCK_MISMATCH')
        return started, None
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (started, deadline)):
        raise PermissionError('PROMPT_BUNDLE_ROUND_CLOCK_REQUIRED')
    limits = manifest.get('ledger_limits', {})
    total = limits.get('batch_seconds')
    carried = limits.get('carried_usage', {}).get('seconds', 0)
    now = clock()
    if (type(total) is not int or type(carried) is not int or not 0 <= carried < total
            or not 0 < started <= now < deadline or deadline != started + total - carried):
        raise PermissionError('PROMPT_BUNDLE_ROUND_CLOCK_MISMATCH')
    if 'continuation_contract' in manifest:
        contract = manifest['continuation_contract']
        clock_contract = contract.get('clock') if isinstance(contract, dict) else None
        if (not isinstance(clock_contract, dict)
                or started != clock_contract.get('round_started_at')
                or deadline != clock_contract.get('round_absolute_deadline')
                or auth.get('expires_at') != clock_contract.get('model_absolute_deadline')):
            raise PermissionError('PROMPT_BUNDLE_CONTINUATION_CLOCK_MISMATCH')
    return started, deadline


def validate_outer_start(manifest, started, *, clock=time.time):
    """批次/launcher必须采用已在资格之前开始的同一个授权时钟。"""
    auth = validate_activation(manifest, 'execute', clock=clock)
    if type(started) not in (int, float) or not math.isfinite(started) or started != auth['round_started_at']:
        raise PermissionError('PROMPT_BUNDLE_OUTER_START_MISMATCH')
    return auth['round_absolute_deadline']


def _probe(source, output):
    """独立冷进程禁止网络、dotenv、神经库与输出根以外写入。"""
    import importlib.abc
    from harness_ab_boundary import Boundary, IsolationError
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve(strict=True)
    sys.path[:0] = [str(source / 'backend'), str(source / 'backend/tests')]
    neural = {'torch', 'onnxruntime', 'transformers', 'tokenizers', 'sentence_transformers'}

    class NoNeural(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in neural:
                raise PermissionError('PROMPT_BUNDLE_NEURAL_IMPORT_FORBIDDEN')
            return None

    guard = Boundary(output)
    guard.phase = 'cold_import'
    sys.addaudithook(guard.audit)
    sys.meta_path.insert(0, NoNeural())
    (output / 'home').mkdir()
    (output / 'cache').mkdir()
    with patch.object(socket, 'has_ipv6', False), patch.dict(os.environ, {
            'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'TMPDIR': str(output), 'HOME': str(output / 'home'), 'XDG_CACHE_HOME': str(output / 'cache')}, clear=True):
        identity = loaded_identity(source)
        from research_loop.config import Settings
        if 'context_deduplicate_read_paper' in Settings.model_fields:
            raise AssertionError('unexpected old dedup business switch')
        denied = []
        guard.phase = 'selfcheck'
        for label, operation in (
                ('dotenv', lambda: (source / '.env').read_text()),
                ('outside_write', lambda: (source / 'forbidden.tmp').write_text('denied')),
                ('network', lambda: socket.create_connection(('127.0.0.1', 6339)))):
            try:
                operation()
            except IsolationError:
                denied.append(label)
            else:
                raise AssertionError('cold selfcheck not rejected: ' + label)
        value = {'status': 'passed', 'identity': identity, 'generation_calls': 0, 'rag_runs': 0,
                 'denied_selfchecks': denied, 'sqlite_connections': list(guard.sqlite_paths),
                 'import_denials': [d for d in guard.denials if d['phase'] == 'cold_import'],
                 'neural_modules_loaded': sorted(neural & set(sys.modules))}
        (output / 'probe.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return value


def prepare(roots, output):
    """只创建新的离线冻结结果；不生成授权或启动服务。"""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    contract = build_contract(roots)
    probes = {}
    for variant in ('A', 'B'):
        target = output / variant
        target.mkdir()
        result = subprocess.run([sys.executable, '-I', '-B', str(Path(__file__).resolve()),
            'probe', '--source', roots[variant], '--output', str(target)],
            capture_output=True, text=True, timeout=60)
        (target / 'cold.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError('cold import failed for ' + variant)
        probes[variant] = json.loads((target / 'probe.json').read_text())
    value = {'experiment_kind': KIND, 'real_execution_released': False, 'live_qualification_released': False,
        'variant_settings': {'A': {}, 'B': {}}, 'source_roots': roots,
        'prompt_bundle_source_contract': contract,
        'loaded_prompt_contract': {v: probes[v]['identity'] for v in probes},
        'real_model_calls': 0, 'live_rag_qualification': 'not_run'}
    (output / 'contract.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return value


def main():
    """prepare和probe均为零模型；正式执行沿用带新授权门的共同后端入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    item = sub.add_parser('prepare')
    item.add_argument('--baseline', required=True)
    item.add_argument('--candidate', required=True)
    item.add_argument('--output', required=True)
    item = sub.add_parser('probe')
    item.add_argument('--source', required=True)
    item.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare({'A': args.baseline, 'B': args.candidate}, args.output)
    else:
        _probe(args.source, args.output)
    print(json.dumps({'status': 'passed', 'command': args.command, 'real_model_calls': 0}))


if __name__ == '__main__':
    main()
