"""明确的三题补测合同；保留旧96格，只为新目录派生12格与财务子额度。"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path


VERSION = 'agent_supplement12_v1'
ANCHOR_SEAL = 'e46a98e49c87b8ec87529f313f0dd9dd6f432bd9723199ff9c2788021546362d'
CASE_IDS = ('AFE-REC-01', 'AFE-REC-02', 'AFE-QA-01')
ORDER = [[f'{case}-R{turn}', variant] for turn in (1, 2) for case in CASE_IDS
         for variant in (('A', 'B') if turn == 1 else ('B', 'A'))]
CAPS = {'tokens': 12_000_000, 'requests': 288, 'nano_yuan': 32_000_000_000}


def _require(value, reason):
    if not value:
        raise PermissionError('SUPPLEMENT12_' + reason)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _hash(path):
    from agent_final_completion import _verified_hash
    return _verified_hash(path)


def _read(path):
    path = Path(path)
    _require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(), 'READ_PATH_REDIRECTED')
    return json.loads(path.read_text())


def _anchor(root):
    """只读核对既有封存及关闭收据，不实例化历史账本或恢复历史任务。"""
    root = Path(root)
    _require(root.is_absolute() and root.resolve(strict=True) == root, 'ANCHOR_CANONICAL_REQUIRED')
    _require(_hash(root/'SEAL.json') == ANCHOR_SEAL, 'APPROVED_ANCHOR_SEAL_REQUIRED')
    seal = _read(root/'SEAL.json')['files']
    def load(name):
        value = seal[name]
        expected = value['sha256'] if isinstance(value, dict) else value
        _require(_hash(root/name) == expected, 'ANCHOR_FILE_CHANGED')
        return _read(root/name)
    manifest = load('manifest.formal.json')
    ledger = load('runtime/ledger-final.json')
    closed = load('formal-preflight/authorization-closed.json')
    _require(closed.get('status') == 'closed' and closed.get('no_resume_or_resend') is True,
             'ANCHOR_MUST_BE_CLOSED')
    charged = ledger['charged_or_reserved']
    _require(all(type(charged[k]) is int and charged[k] >= 0 for k in ('tokens', 'requests', 'nano')),
             'ANCHOR_FINANCE_INVALID')
    carry = {'tokens': charged['tokens'], 'requests': charged['requests'], 'nano_yuan': charged['nano'],
             'seconds': ledger['carried_usage']['seconds']}
    return {'root': str(root), 'manifest': manifest, 'ledger': ledger, 'seal': seal,
            'cases': load('inputs/cases.json'), 'materials': load('inputs/material-snapshot.json'),
            'carried_usage': carry}


def build_contract(anchor_root, scope_receipt_path):
    """派生待审合同；调用本函数不会创建授权或发送请求。"""
    base = _anchor(anchor_root)
    scope_path = Path(scope_receipt_path)
    scope = _read(scope_path)
    _require(scope.get('authorized_by_user') is True and scope.get('cancel_global_time_limit') is True
             and isinstance(scope.get('user_request'), str) and scope['user_request'].strip()
             and _canonical(scope.get('generation_limits')) == _canonical(CAPS), 'EXPLICIT_SCOPE_REQUIRED')
    clock = deepcopy(base['manifest']['completion_contract']['clock'])
    _require(clock.get('policy') == 'financial_only' and clock.get('round_absolute_deadline') is None
             and clock.get('model_absolute_deadline') is None, 'ANCHOR_CLOCK_INVALID')
    return {'version': VERSION, 'anchor_root': base['root'], 'anchor_seal_sha256': ANCHOR_SEAL,
            'scope_receipt_path': str(scope_path), 'scope_receipt_sha256': _hash(scope_path),
            'new_execution_order': deepcopy(ORDER), 'full_execution_order': deepcopy(ORDER),
            'retained_rows': [], 'retained_count': 0, 'retry_positions': [],
            'carried_usage': deepcopy(base['carried_usage']), 'clock': clock}


def expected_limits(base):
    """仅替换本轮总额度与矩阵，沿用原模型输出及单根执行合同。"""
    carry = base['carried_usage']
    limits = deepcopy(base['manifest']['ledger_limits'])
    limits.update(allowed_cells=deepcopy(ORDER), carried_usage=deepcopy(carry), enforce_batch_seconds=False,
                  batch_tokens=carry['tokens'] + CAPS['tokens'], batch_requests=carry['requests'] + CAPS['requests'],
                  batch_yuan=str(Decimal(carry['nano_yuan'] + CAPS['nano_yuan']) / Decimal(1_000_000_000)))
    return limits


def _case_semantics(value):
    """允许材料复制路径变化，问题、材料身份和其他前置不得变化。"""
    result = deepcopy(value)
    for material in result.get('materials', []):
        material.pop('pdf_path', None)
    return result


def _business_sources(manifest, base):
    """业务源码须与原对应变体一致；共同测量脚本另由原来源合同绑定。"""
    for variant in ('A', 'B'):
        old = Path(base['manifest']['source_roots'][variant])
        new = Path(manifest['source_roots'][variant])
        prefix = str(old.relative_to(Path(base['root']))) + '/backend/research_loop/'
        expected = {name[len(prefix):]: (item['sha256'] if isinstance(item, dict) else item)
                    for name, item in base['seal'].items() if name.startswith(prefix)}
        actual = {str(path.relative_to(new/'backend/research_loop')): _hash(path)
                  for path in (new/'backend/research_loop').rglob('*')
                  if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
        _require(actual == expected, 'BUSINESS_SOURCE_CHANGED')


def validate(manifest):
    """严核新补测身份、财务、输入和业务字节；旧96合同不经过此分支。"""
    supplied = manifest['completion_contract']
    _require(isinstance(supplied, dict) and supplied.get('version') == VERSION, 'VERSION_REQUIRED')
    expected = build_contract(supplied['anchor_root'], supplied['scope_receipt_path'])
    _require(_canonical(supplied) == _canonical(expected), 'CONTRACT_CHANGED')
    base = _anchor(supplied['anchor_root'])
    _require(manifest.get('experiment_kind') == 'prompt_bundle_v1'
             and _canonical(manifest.get('execution_order')) == _canonical(ORDER), 'EXACT_12_ORDER_REQUIRED')
    _require(manifest.get('qualification_cells') == ORDER, 'EXACT_12_QUALIFICATION_REQUIRED')
    _require(manifest.get('settings') == base['manifest']['settings']
             and manifest.get('variant_settings') == {'A': {}, 'B': {}}, 'SETTINGS_CHANGED')
    _require(_canonical(manifest.get('ledger_limits')) == _canonical(expected_limits(base)), 'FINANCE_OR_ROOT_LIMIT_CHANGED')
    from agent_final_completion import EVALUATION_POLICY
    from agent_final_observation_recovery import OBSERVATION_POLICY
    from harness_ab_pair_gate import FINAL_SUITE_POLICY
    _require(manifest.get('evaluation_policy') == EVALUATION_POLICY
             and manifest.get('observation_policy') == OBSERVATION_POLICY
             and manifest.get('final_suite_policy') == FINAL_SUITE_POLICY, 'POLICY_CHANGED')
    root = Path(manifest['experiment_root'])
    anchor = Path(base['root'])
    _require(root.is_absolute() and root.resolve(strict=True) == root
             and not root.is_relative_to(anchor) and not anchor.is_relative_to(root), 'NEW_OWNED_ROOT_REQUIRED')
    for key, relative in {'runtime_root': 'runtime', 'semantic_gate_root': 'semantic-gates',
                          'prepared_input_root': 'inputs', 'cases_path': 'inputs/cases.json',
                          'model_manifest_path': 'inputs/model-manifest.json'}.items():
        path = Path(manifest[key])
        _require(path == root/relative and path.resolve() == path, 'OWNED_GEOMETRY_REQUIRED')
    for variant in ('A', 'B'):
        path = Path(manifest['source_roots'][variant])
        _require(path == root/'snapshots'/variant and path.resolve(strict=True) == path, 'OWNED_SOURCE_REQUIRED')
    auth = Path(manifest['authorization_path'])
    scope = Path(supplied['scope_receipt_path'])
    for path in (auth, scope):
        _require(path.is_relative_to(root) and path.resolve() == path
                 and not any(path.is_relative_to(root/name) for name in ('runtime', 'inputs', 'snapshots')),
                 'OWNED_CONTROL_REQUIRED')
    inputs = Path(manifest['prepared_input_root'])
    for relative, key in [('cases.json', 'final_cases_sha256'), ('execution-plan.json', 'final_plan_sha256'),
                          ('material-snapshot.json', 'material_snapshot_sha256')]:
        _require(_hash(inputs/relative) == manifest[key], 'NEW_INPUT_HASH_CHANGED')
    cases = _read(inputs/'cases.json')['cases']
    old_cases = {case['case_id']: case for case in base['cases']['cases']}
    wanted = {cell[0] for cell in ORDER}
    _require(len(cases) == 6 and {case['case_id'] for case in cases} == wanted
             and all(_case_semantics(case) == _case_semantics(old_cases[case['case_id']]) for case in cases),
             'FROZEN_6_CASES_CHANGED')
    _require(_read(inputs/'execution-plan.json').get('execution_order') == ORDER, 'PLAN_ORDER_CHANGED')
    snapshot = _read(inputs/'material-snapshot.json')
    old_materials = {item['pdf_sha256']: item for item in base['materials']['materials']}
    materials = snapshot['materials']
    def identity(item):
        return {key: value for key, value in item.items() if key not in ('pdf_path', 'parsed_path')}
    _require(len(materials) == 54 and {item['pdf_sha256'] for item in materials} == set(old_materials)
             and all(identity(item) == identity(old_materials[item['pdf_sha256']]) for item in materials),
             'MATERIAL_IDENTITIES_CHANGED')
    _business_sources(manifest, base)
    _require(manifest.get('carried_budget_source') == {'version': VERSION, 'anchor_root': base['root'],
             'seal_sha256': ANCHOR_SEAL, 'ledger_path': str(anchor/'runtime/ledger-final.json'),
             'ledger_sha256': _hash(anchor/'runtime/ledger-final.json')}, 'CARRY_SOURCE_CHANGED')
    return {**deepcopy(expected), 'anchor_manifest': deepcopy(base['manifest']),
            'latest_anchor_manifest': deepcopy(base['manifest']), 'closed_ledger_roots': [base['root']],
            'technical_attempts': [], 'pending_reservations': [], 'deferred_review_cases': []}
