"""固定任务集的显式续跑合同；只读取旧封存，不恢复旧账本或授权。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


VERSION = 'agent_final_continuation_v1'
_ROLES = {'supervisor', 'researcher_screen', 'researcher_investigate',
          'researcher_deep_read', 'reviewer_note', 'reviewer_target'}
_CELL_FILES = ('result.json', 'task-view.json', 'model-calls.json', 'actions.json')
_EXPERIMENT_SCRIPTS = frozenset('backend/scripts/' + name for name in (
    'harness_ab_transport.py', 'run_read_paper_ab.py', 'run_read_paper_batch.py',
    'harness_ab_pair_gate.py', 'prompt_bundle_ab.py',
    'agent_final_continuation.py', 'agent_final_analysis.py'))
_HASH_CACHE = {}


def _require(value, reason):
    if not value:
        raise PermissionError('FINAL_CONTINUATION_' + reason)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _equal(a, b):
    # JSON 类型也属于合同；Python 的 True == 1 不能放宽授权。
    return _canonical(a) == _canonical(b)


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _hash(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink() and path.resolve() == path,
             'REGULAR_FILE_REQUIRED')
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _HASH_CACHE.get(str(path))
    if cached and cached[0] == identity:
        return cached[1]
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    _require(identity == (after.st_dev, after.st_ino, after.st_size,
                          after.st_mtime_ns, after.st_ctime_ns), 'FILE_CHANGED_DURING_READ')
    result = digest.hexdigest()
    _HASH_CACHE[str(path)] = (identity, result)
    return result


def _inside(root, relative):
    relative = Path(relative)
    _require(not relative.is_absolute() and '..' not in relative.parts,
             'ANCHOR_RELATIVE_PATH_REQUIRED')
    path = root / relative
    _require(path.resolve() == path and path.is_relative_to(root), 'ANCHOR_PATH_REDIRECTED')
    return path


def _integer(value):
    return type(value) is int and value >= 0


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _anchor(anchor_root):
    root = Path(anchor_root)
    _require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'ANCHOR_ROOT_REQUIRED')
    seal_path = root / 'SEAL.json'
    seal = _read(seal_path)
    _require(isinstance(seal.get('files'), dict), 'SEAL_REQUIRED')
    anchors = {'SEAL.json': _hash(seal_path)}

    def load(relative, parse=True):
        relative = str(relative)
        path = _inside(root, relative)
        declared = seal['files'].get(relative)
        digest = _hash(path)
        _require(isinstance(declared, dict) and declared.get('sha256') == digest,
                 'ANCHOR_SEAL_ENTRY_CHANGED')
        anchors[relative] = digest
        return _read(path) if parse else None

    manifest = load('manifest.formal.json')
    _require(manifest.get('experiment_kind') == 'prompt_bundle_v1'
             and manifest.get('final_suite_policy') == {
                 'version': 'agent_final_v1', 'semantic_failures': 'record_and_continue'},
             'ANCHOR_FIXED_SUITE_REQUIRED')
    _require(Path(manifest['experiment_root']) == root
             and Path(manifest['runtime_root']) == root / 'runtime'
             and Path(manifest['semantic_gate_root']) == root / 'semantic-gates',
             'ANCHOR_DIRECTORY_MISMATCH')
    order = manifest['execution_order']
    _require(isinstance(order, list) and len(order) == 96
             and all(isinstance(c, list) and len(c) == 2 and isinstance(c[0], str)
                     and c[1] in ('A', 'B') for c in order)
             and len({tuple(c) for c in order}) == 96
             and all(order[i][0] == order[i + 1][0]
                     and {order[i][1], order[i + 1][1]} == {'A', 'B'}
                     for i in range(0, 96, 2)), 'ANCHOR_96_ORDER_REQUIRED')
    cases = load('inputs/cases.json')
    plan = load('inputs/execution-plan.json')
    materials = load('inputs/material-snapshot.json')
    load('inputs/model-manifest.json')
    for variant in ('A', 'B'):
        for relative, digest in manifest['prompt_bundle_source_contract']['variants'][variant]['source_files'].items():
            old_path = Path(manifest['source_roots'][variant]) / relative
            rel = str(old_path.relative_to(root))
            load(rel, parse=False)
            _require(anchors[rel] == digest, 'OLD_RUNTIME_SOURCE_CHANGED')
    _require(plan['execution_order'] == order and len(cases['cases']) == 48
             and {c['case_id'] for c in cases['cases']} == {c[0] for c in order},
             'ANCHOR_INPUT_MATRIX_MISMATCH')
    for key, rel in [('final_cases_sha256', 'inputs/cases.json'),
                     ('final_plan_sha256', 'inputs/execution-plan.json'),
                     ('material_snapshot_sha256', 'inputs/material-snapshot.json')]:
        _require(manifest[key] == anchors[rel], 'ANCHOR_INPUT_HASH_MISMATCH')
    batch = load('runtime/batch-results.json')
    ledger = load('runtime/ledger-final.json')
    auth = load('formal-execution-auth.json')
    closure = load('formal-preflight/authorization-closed.json')
    controller = load('formal-control/controller-result.json')
    _require(auth.get('approved') is False and auth.get('allowed_modes') == []
             and auth.get('status') == 'closed_no_resume'
             and closure.get('status') == 'closed' and closure.get('no_resume_or_resend') is True,
             'OLD_AUTHORIZATION_NOT_CLOSED')
    identity = hashlib.sha256(_canonical({k: v for k, v in manifest.items()
                                         if k != 'cell_absolute_deadline'}).encode()).hexdigest()
    _require(auth.get('manifest_sha256') == identity, 'OLD_AUTHORIZATION_IDENTITY_MISMATCH')
    _require(controller.get('cleanup_all_passed') is True
             and controller.get('batch_group_cleaned') is True
             and controller.get('wrapper_finished_before_original_total_deadline') is True,
             'OLD_CONTROLLER_NOT_CLOSED')
    limits = manifest['ledger_limits']
    prior = ledger['carried_usage']
    _require(set(prior) == {'tokens', 'requests', 'nano_yuan', 'seconds'}
             and all(_integer(v) for v in prior.values()), 'OLD_CARRY_INVALID')
    clock = {'round_started_at': auth['round_started_at'],
             'round_absolute_deadline': auth['round_absolute_deadline'],
             'model_absolute_deadline': controller['model_deadline'],
             'batch_seconds': limits['batch_seconds'],
             'pre_round_carried_seconds': prior['seconds']}
    _require(all(_finite(clock[k]) for k in ('round_started_at', 'round_absolute_deadline',
                                            'model_absolute_deadline'))
             and _integer(clock['batch_seconds']) and clock['batch_seconds'] > prior['seconds'],
             'OLD_CLOCK_INVALID')
    total, started = clock['round_absolute_deadline'], clock['round_started_at']
    _require(total == started + clock['batch_seconds'] - prior['seconds']
             and ledger['started_at'] == batch['started_at'] == closure['original_started_at'] == started
             and ledger['batch_absolute_deadline'] == closure['absolute_deadline'] == total
             and controller['original_absolute_total_deadline'] == total
             and auth['expires_at'] == clock['model_absolute_deadline']
             and total - clock['model_absolute_deadline'] == 300,
             'OLD_CLOCK_MISMATCH')
    _require(_equal(limits, batch['limits']) and _equal(prior, limits['carried_usage']),
             'OLD_LIMITS_MISMATCH')
    reservations = ledger['reservations']
    _require(all(r['status'] in ('settled', 'not_sent') for r in reservations),
             'UNRESOLVED_PHYSICAL_REQUEST')
    settled = [r for r in reservations if r['status'] == 'settled']
    _require(all(_integer(r[k]) for r in settled for k in ('actual_tokens', 'actual_nano')),
             'OLD_SETTLEMENT_INVALID')
    actual = {'tokens': sum(r['actual_tokens'] for r in settled), 'requests': len(settled),
              'nano_yuan': sum(r['actual_nano'] for r in settled)}
    for short, lk, bk in [('tokens', 'actual_known_tokens', 'actual_known_tokens'),
                           ('requests', 'physical_sent_requests', 'physical_requests'),
                           ('nano_yuan', 'actual_known_estimated_nano_yuan',
                            'actual_known_estimated_nano_yuan')]:
        _require(_integer(ledger[lk]) and _integer(batch[bk])
                 and actual[short] == ledger[lk] == batch[bk], 'OLD_FINANCE_MISMATCH')
    carry = {k: actual[k] + prior[k] for k in actual}
    _require(_equal(ledger['charged_or_reserved'], {'tokens': carry['tokens'],
             'requests': carry['requests'], 'nano': carry['nano_yuan']}), 'OLD_TOTALS_MISMATCH')
    carry['seconds'] = prior['seconds']
    rows = batch['rows']
    _require(batch['status'] == 'stopped' and len(rows) == 5
             and [[r['case_id'], r['variant']] for r in rows] == order[:5]
             and batch['not_created'] == order[5:]
             and all(r['status'] == 'completed' and r.get('issues') == [] for r in rows[:4])
             and rows[4]['status'] in ('partial', 'failed', 'interrupted', 'unknown')
             and rows[4].get('issues') and ledger.get('halt_reason')
             and (ledger.get('first_error') or {}).get('root_id') == rows[4]['root_id'],
             'ONLY_FIFTH_TECHNICAL_RETRY_ALLOWED')
    _require(len({r['root_id'] for r in rows}) == 5, 'OLD_ROOT_ID_DUPLICATE')
    static = load('scoring/STATIC-SEAL.json')
    for name, item in static['files'].items():
        load('scoring/' + name, parse=False)
        _require(anchors['scoring/' + name] == item['sha256'], 'STATIC_SCORING_CHANGED')
    retained = []
    failed = None
    for position, row in enumerate(rows):
        cell = root / 'runtime/cells' / (row['case_id'] + '-' + row['variant'])
        contents = {name: load(str((cell / name).relative_to(root)),
                               parse=name in ('result.json', 'task-view.json'))
                    for name in _CELL_FILES}
        result, view = contents['result.json'], contents['task-view.json']
        _require(result['root_id'] == view['task_id'] == row['root_id']
                 and result['case_id'] == row['case_id'] and result['variant'] == row['variant'],
                 'OLD_CELL_IDENTITY_MISMATCH')
        provenance = {'anchor_root': str(root), 'cell_root': str(cell),
                      'files': {name: anchors[str((cell / name).relative_to(root))]
                                for name in _CELL_FILES}}
        enriched = {**deepcopy(row), 'original_position': position,
                    'retained': position < 4, 'provenance': provenance}
        if position < 4:
            case = row['case_id']
            for filename in ('challenge-private.json', 'review-packet.json', 'review.json', 'decision.json'):
                load('semantic-gates/' + case + '/' + filename)
            review = load('semantic-gates/' + case + '/review.json')
            decision = load('semantic-gates/' + case + '/decision.json')
            proofs = review.get('independent_reviews', [])
            _require(len(proofs) == 2 and len({p['identity'] for p in proofs}) == 2,
                     'INDEPENDENT_REVIEW_PROOFS_REQUIRED')
            for proof in [*proofs, review['adjudication_source']]:
                relative = str(Path(proof['path']).relative_to(root))
                original = load(relative)
                _require(anchors[relative] == proof['sha256']
                         and original.get('challenge_sha256') == review['challenge_sha256']
                         and original.get('paid_model_calls') == 0, 'ORIGINAL_REVIEW_CHANGED')
            _require(decision['status'] == 'passed' and decision['stop_reasons'] == []
                     and decision['quality'][row['variant']] in ('full', 'partial_usable', 'unusable', 'failed')
                     and result['status'] == 'executed' and view['status'] == 'completed',
                     'RETAINED_RESULT_OR_GRADE_INVALID')
            enriched['quality'] = decision['quality'][row['variant']]
            provenance.update(grade_decision_path=str(root / 'semantic-gates' / case / 'decision.json'),
                              grade_decision_sha256=anchors['semantic-gates/' + case + '/decision.json'])
            retained.append(enriched)
        else:
            failed = enriched
    contract = {'version': VERSION, 'anchor_root': str(root), 'anchors': anchors,
                'retained_rows': retained, 'failed_attempt': failed,
                'new_execution_order': deepcopy(order[4:]), 'carried_usage': carry, 'clock': clock,
                'retry_policy': {'original_position': 4, 'max_new_attempts': 1,
                                 'preserve_failed_attempt': True}}
    return contract, manifest, cases, materials


def build_continuation_contract(anchor_root):
    """从已关闭批次只读生成待审合同，不发放执行授权。"""
    try:
        return _anchor(anchor_root)[0]
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise PermissionError('FINAL_CONTINUATION_ANCHOR_INVALID') from exc


def _material_semantics(materials):
    return [{k: deepcopy(v) for k, v in item.items() if k not in {'pdf_path', 'parsed_path'}}
            for item in materials]


def _case_semantics(data):
    result = deepcopy(data)
    for case in result['cases']:
        case['materials'] = _material_semantics(case.get('materials', []))
        if case.get('material') is not None:
            case['material'] = _material_semantics([case['material']])[0]
    return result


def _check_inputs(manifest, anchor, old_cases, old_materials):
    cases = _read(manifest['cases_path'])
    _require(_equal(_case_semantics(cases), _case_semantics(old_cases)),
             'CASE_OR_PRECONDITIONS_CHANGED')
    current = _read(Path(manifest['prepared_input_root']) / 'material-snapshot.json')
    _require(_equal(_material_semantics(current['materials']),
                    _material_semantics(old_materials['materials'])), 'MATERIAL_SEMANTICS_CHANGED')
    old_copies = sorted((i['sha256'], i['bytes']) for i in old_materials['copies'])
    _require(sorted((i['sha256'], i['bytes']) for i in current['copies']) == old_copies,
             'MATERIAL_COPY_SET_CHANGED')
    hashes = {}
    material_root = Path(manifest['prepared_input_root']).resolve()
    for item in current['copies']:
        path = Path(item['copy'])
        _require(path.is_relative_to(material_root / 'materials')
                 and path.stat().st_nlink == 1 and _hash(path) == item['sha256']
                 and path.stat().st_size == item['bytes'], 'MATERIAL_COPY_CHANGED')
        hashes[str(path)] = item['sha256']
    for item in current['materials']:
        _require(hashes.get(item['pdf_path']) == item['pdf_sha256']
                 and hashes.get(item['parsed_path']) == item['parsed_cache_sha256'],
                 'MATERIAL_PATH_HASH_MISMATCH')
    for case in cases['cases']:
        for item in case.get('materials', []):
            _require(hashes.get(item['pdf_path']) == item['pdf_sha256'], 'CASE_MATERIAL_CHANGED')
    plan = _read(Path(manifest['prepared_input_root']) / 'execution-plan.json')
    _require(plan['execution_order'] == anchor['execution_order'], 'PLAN_ORDER_CHANGED')


def _runtime_files(root):
    """采用既有运行清单的范围，避免引入只适用于完整业务包的Supervisor前置。"""
    paths = []
    for relative in ('backend/research_loop', 'backend/scripts', 'backend/migrations', 'config'):
        directory = root / relative
        _require(not directory.is_symlink(), 'RUNTIME_DIRECTORY_SYMLINK')
        if not directory.exists():
            continue
        for path in directory.rglob('*'):
            if '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            _require(not path.is_symlink(), 'RUNTIME_FILE_SYMLINK')
            if path.is_file():
                paths.append(path)
    for relative in ('backend/pyproject.toml', 'backend/uv.lock'):
        path = root / relative
        _require(not path.is_symlink(), 'DEPENDENCY_FILE_SYMLINK')
        if path.is_file():
            paths.append(path)
    return {str(path.relative_to(root)): path for path in paths}


def _check_sources(manifest, anchor):
    for key in ('settings', 'variant_settings', 'pair_review_policy', 'final_suite_policy'):
        _require(_equal(manifest.get(key), anchor.get(key)), 'EXPERIMENT_CONDITIONS_CHANGED')
    _require(_equal(_read(manifest['model_manifest_path']), _read(anchor['model_manifest_path'])),
             'MODEL_RESOURCE_IDENTITY_CHANGED')
    common_scripts = []
    for variant in ('A', 'B'):
        expected = anchor['prompt_bundle_source_contract']['variants'][variant]['source_files']
        current_root = Path(manifest['source_roots'][variant])
        actual = _runtime_files(current_root)
        _require(set(expected).issubset(actual)
                 and set(actual) - set(expected) <= _EXPERIMENT_SCRIPTS,
                 'RUNTIME_SOURCE_SET_CHANGED')
        for relative, digest in expected.items():
            if relative not in _EXPERIMENT_SCRIPTS:
                _require(_hash(actual[relative]) == digest, 'RUNTIME_SOURCE_CHANGED')
        common_scripts.append({relative: _hash(path) for relative, path in actual.items()
                               if relative in _EXPERIMENT_SCRIPTS})
        old = anchor['loaded_prompt_contract'][variant]
        new = manifest['loaded_prompt_contract'][variant]
        _require(set(old['assembled_prompt_sha256']) == _ROLES
                 and _equal(new['assembled_prompt_sha256'], old['assembled_prompt_sha256'])
                 and _equal(new['python_runtime'], old['python_runtime'])
                 and new['code_fingerprint_format'] == old['code_fingerprint_format']
                 and set(new['registry_sha256']) == set(old['registry_sha256']),
                 'ROLE_OR_TOOL_CONTRACT_CHANGED')
        # handler 指纹包含 co_filename；跨路径不比较其值，真实新路径身份由冷启动门绑定。
    _require(_equal(common_scripts[0], common_scripts[1]), 'COMMON_EXPERIMENT_SCRIPT_DRIFT')


def validate_continuation(manifest):
    """缺少续跑键时保持旧入口；键一旦存在便严格校验全部锚与派生值。

    Returns:
        无该键时为 None；否则为只读旧批次身份和92格执行计划的副本。
    """
    if 'continuation_contract' not in manifest:
        return None
    try:
        supplied = manifest['continuation_contract']
        _require(isinstance(supplied, dict) and supplied.get('version') == VERSION,
                 'EXPLICIT_VERSION_REQUIRED')
        expected, anchor, cases, materials = _anchor(supplied['anchor_root'])
        _require(_equal(supplied, expected), 'CONTRACT_OR_ANCHOR_CHANGED')
        _require(manifest.get('experiment_kind') == 'prompt_bundle_v1'
                 and _equal(manifest['execution_order'], anchor['execution_order'])
                 and _equal(manifest['ledger_limits']['allowed_cells'], anchor['ledger_limits']['allowed_cells']),
                 'FULL_96_ORDER_REQUIRED')
        limits = manifest['ledger_limits']
        _require(_equal({k: v for k, v in limits.items() if k != 'carried_usage'},
                        {k: v for k, v in anchor['ledger_limits'].items() if k != 'carried_usage'})
                 and _equal(limits.get('carried_usage'), expected['carried_usage']),
                 'NEW_FINANCE_OR_CLOCK_RESET')
        root = Path(expected['anchor_root'])
        source = {'batch_results_path': str(root / 'runtime/batch-results.json'),
                  'batch_results_sha256': expected['anchors']['runtime/batch-results.json'],
                  'ledger_final_path': str(root / 'runtime/ledger-final.json'),
                  'ledger_final_sha256': expected['anchors']['runtime/ledger-final.json']}
        _require(_equal(manifest.get('carried_budget_source'), source), 'LATEST_FINANCIAL_SOURCE_REQUIRED')
        owned = Path(manifest['experiment_root'])
        _require(owned.is_absolute() and owned.resolve() == owned
                 and not owned.is_relative_to(root) and not root.is_relative_to(owned),
                 'OLD_RUNTIME_REUSE_FORBIDDEN')
        paths = {'runtime_root': owned / 'runtime', 'semantic_gate_root': owned / 'semantic-gates',
                 'prepared_input_root': owned / 'inputs', 'cases_path': owned / 'inputs/cases.json',
                 'model_manifest_path': owned / 'inputs/model-manifest.json'}
        for key, required in paths.items():
            target = Path(manifest[key])
            _require(target.resolve() == target == required, 'OWNED_DIRECTORY_GEOMETRY_REQUIRED')
        for variant in ('A', 'B'):
            target = Path(manifest['source_roots'][variant])
            _require(target.resolve() == target == owned / 'snapshots' / variant,
                     'OWNED_SOURCE_GEOMETRY_REQUIRED')
        authorization = Path(manifest['authorization_path'])
        _require(authorization.resolve() == authorization and authorization.is_relative_to(owned)
                 and not any(authorization.is_relative_to(p) for p in
                             [owned / 'runtime', owned / 'inputs', owned / 'snapshots'])
                 and authorization != Path(anchor['authorization_path']),
                 'NEW_AUTHORIZATION_PATH_REQUIRED')
        _check_inputs(manifest, anchor, cases, materials)
        _check_sources(manifest, anchor)
        return {**deepcopy(expected), 'anchor_manifest': anchor,
                'full_execution_order': deepcopy(anchor['execution_order']), 'retained_count': 4}
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise PermissionError('FINAL_CONTINUATION_INVALID') from exc


def planned_order(manifest):
    """返回显式授权的新92格顺序；旧合同仍返回完整原顺序。"""
    context = validate_continuation(manifest)
    return deepcopy(context['new_execution_order'] if context else manifest['execution_order'])


def retained_rows(manifest):
    """返回带旧路径、原评分和哈希出处的四条只读保留行。"""
    context = validate_continuation(manifest)
    return deepcopy(context['retained_rows']) if context else []


def validate_shared_clock_carry(manifest):
    """财务结转到最新批次，时间仅保留原T0之前的历史耗时。"""
    context = validate_continuation(manifest)
    _require(context is not None, 'SHARED_CLOCK_CONTRACT_REQUIRED')
    return deepcopy(context['carried_usage'])
