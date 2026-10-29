"""固定任务集的第二代补测合同；未知旧请求只结转预留，不重开或重发。"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3


VERSION = 'agent_final_completion_v2'
EVALUATION_POLICY = {'version': 'agent_completion_descriptive_v1',
                     'minimum_B_full': None, 'retain_quality_comparison': True}
_ANCHOR_CACHE = {}


def _require(value, reason):
    if not value:
        raise PermissionError('FINAL_COMPLETION_' + reason)


def _helpers():
    # v1 dispatcher会调用本模块；仅在函数执行时取得其只读原语。
    import agent_final_continuation as original
    return original


def _digest(value):
    return hashlib.sha256(_helpers()._canonical(value).encode()).hexdigest()


def _equal(a, b):
    return _helpers()._equal(a, b)


def _sealed(root):
    """核对完整密封清单；相同inode/stat的重复检查复用既有字节hash缓存。"""
    h = _helpers()
    root = Path(root)
    _require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'ANCHOR_ROOT_REQUIRED')
    seal_path = root / 'SEAL.json'
    seal = h._read(seal_path)
    _require(isinstance(seal.get('files'), dict) and seal['files'], 'SEALED_FILES_REQUIRED')
    digest = h._hash(seal_path)
    _require((root / 'SEAL.sha256').read_text().split()[0] == digest, 'SEAL_DIGEST_CHANGED')
    readback = h._read(root / 'SEAL-READBACK.json')
    _require(readback.get('status') == 'passed' and readback.get('seal_sha256') == digest,
             'SEAL_READBACK_CHANGED')
    excluded = {'SEAL.json', 'SEAL.sha256', 'SEAL-READBACK.json'}
    actual = set()
    for path in root.rglob('*'):
        _require(not path.is_symlink(), 'ANCHOR_SYMLINK')
        if path.is_file() and str(path.relative_to(root)) not in excluded:
            actual.add(str(path.relative_to(root)))
    _require(actual == set(seal['files']), 'SEALED_FILE_SET_CHANGED')
    for relative, item in seal['files'].items():
        path = h._inside(root, relative)
        _require(isinstance(item, dict) and path.stat().st_size == item.get('bytes')
                 and h._hash(path) == item.get('sha256'), 'SEALED_FILE_BYTES_CHANGED')
    return root, seal, digest


def _loader(root, seal, anchors):
    h = _helpers()

    def load(relative, parse=True):
        path = h._inside(root, relative)
        digest = h._hash(path)
        _require(seal['files'].get(relative, {}).get('sha256') == digest, 'ANCHOR_ENTRY_CHANGED')
        anchors[relative] = digest
        return h._read(path) if parse else None
    return load


def _ledger(root, load):
    """只读打开已封存SQLite，拒绝忽略未合并WAL或热回滚日志。"""
    path = root / 'runtime/budget.sqlite3'
    load('runtime/budget.sqlite3', False)
    for suffix in ('-wal', '-journal'):
        side = path.with_name(path.name + suffix)
        _require(not side.exists() or side.stat().st_size == 0, 'LEDGER_UNCHECKPOINTED')
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        db.row_factory = sqlite3.Row
        meta = {r['key']: json.loads(r['value']) for r in db.execute('SELECT * FROM metadata')}
        roots = [dict(r) for r in db.execute('SELECT * FROM roots')]
        reservations = [dict(r) for r in db.execute('SELECT * FROM reservations ORDER BY created_at,reservation_id')]
    return {'metadata': meta, 'roots': roots, 'reservations': reservations}


def _usage(raw):
    value = json.loads(raw) if isinstance(raw, str) else raw
    _require(isinstance(value, dict), 'USAGE_REQUIRED')
    fields = tuple(value[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    _require(all(type(v) is int and v >= 0 for v in fields) and fields[0] + fields[1] == fields[2],
             'USAGE_INVALID')
    return value


def _receipts(root_id, native, physical):
    """保留格必须原生与物理收据一一对账，未知预留只能属于旧中断格。"""
    _require(native and physical and len(native) == len(physical), 'RETAINED_RECEIPT_COUNT')
    by_id = {}
    for call in native:
        _require(call.get('status') == 'succeeded' and call.get('sent_state', 'sent') == 'sent',
                 'RETAINED_NATIVE_NOT_SUCCESS')
        response = json.loads(call['response_json'])
        identity = response.get('id')
        _require(isinstance(identity, str) and identity and identity not in by_id
                 and call.get('request_id') in (None, identity), 'RETAINED_NATIVE_IDENTITY')
        by_id[identity] = _usage(call['usage_json'])
    seen = set()
    for row in physical:
        _require(row['root_id'] == root_id and row['status'] == 'settled', 'RETAINED_PHYSICAL_NOT_SETTLED')
        response = json.loads(row['response_json'])
        identity = response.get('response_id')
        usage = _usage(row['usage_json'])
        _require(identity in by_id and identity not in seen and _equal(by_id[identity], usage)
                 and [row[k] for k in ('actual_input', 'actual_output', 'actual_tokens')]
                 == [usage[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')],
                 'RETAINED_RECEIPT_MISMATCH')
        seen.add(identity)
    return {'physical_receipts_verified': True,
            'native_status_counts': dict(Counter(r['status'] for r in native)),
            'physical_status_counts': dict(Counter(r['status'] for r in physical))}


def _old_grade(root, manifest, case, load):
    """保留原挑战、原分与裁定；不借新manifest重新批准旧答案。"""
    files = ('challenge-private.json', 'review-packet.json', 'review.json', 'decision.json')
    values = {name: load('semantic-gates/' + case + '/' + name) for name in files}
    review, decision = values['review.json'], values['decision.json']
    _require(decision.get('status') == 'passed' and decision.get('stop_reasons') == [], 'OLD_REVIEW_NOT_PASSED')
    _require(review.get('challenge_sha256') == _digest(values['challenge-private.json'])
             and decision.get('challenge_sha256') == review['challenge_sha256']
             and decision.get('review_sha256') == _digest(review), 'OLD_REVIEW_BINDING_CHANGED')
    proofs = review.get('independent_reviews', [])
    _require(len(proofs) == 2 and len({x['identity'] for x in proofs}) == 2, 'TWO_OLD_REVIEWS_REQUIRED')
    for proof in [*proofs, review['adjudication_source']]:
        path = Path(proof['path'])
        _require(path.is_relative_to(root), 'OLD_REVIEW_PATH_OUTSIDE_ANCHOR')
        value = load(str(path.relative_to(root)))
        _require(_helpers()._hash(path) == proof['sha256'] and value.get('paid_model_calls') == 0
                 and value.get('challenge_sha256') == review['challenge_sha256'], 'OLD_REVIEW_PROOF_CHANGED')
    # 同一冻结评价器重新核对，保持旧语义门的原始字段及质量标签。
    from harness_ab_pair_gate import evaluate_pair_review, manifest_identity
    challenge = values['challenge-private.json']
    pair_rows = [{'case_id': case, 'variant': v, 'root_id': x['root_id'], 'issues': []}
                 for v, x in challenge['identity']['answers'].items()]
    expected = evaluate_pair_review(manifest, case, pair_rows, manifest_identity(manifest), challenge, review,
                                    clock=lambda: review['reviewed_at'])
    _require(_equal(expected, decision), 'OLD_DECISION_CHANGED')
    return decision


def _row(root, manifest, position, ledger, load, *, graded):
    case, variant = manifest['execution_order'][position]
    relative = 'runtime/cells/' + case + '-' + variant
    cell = root / relative
    result = load(relative + '/result.json')
    view = load(relative + '/task-view.json')
    native = load(relative + '/model-calls.json')
    load(relative + '/actions.json', False)
    _require(result.get('status') == 'executed' and result.get('case_id') == case
             and result.get('variant') == variant and result.get('root_id') == view.get('task_id')
             and view.get('status') in {'completed', 'partial'}, 'RETAINED_TERMINAL_REQUIRED')
    _require(any(r['root_id'] == result['root_id'] and r['case_id'] == case and r['variant'] == variant
                 for r in ledger['roots']), 'RETAINED_LEDGER_ROOT_IDENTITY')
    physical = [x for x in ledger['reservations'] if x['root_id'] == result['root_id']]
    proofs = _receipts(result['root_id'], native, physical)
    row = {'case_id': case, 'variant': variant, 'root_id': result['root_id'], 'status': view['status'],
           'issues': [], 'returncode': 0, 'original_position': position, 'retained': True,
           'source_manifest_path': str(root / 'manifest.formal.json'),
           'source_manifest_sha256': _helpers()._hash(root / 'manifest.formal.json'), **proofs,
           'ledger_root': str(root), 'ledger_path': str(root / 'runtime/budget.sqlite3'),
           'provenance': {'anchor_root': str(root), 'cell_root': str(cell),
               'files': {name: _helpers()._hash(cell / name) for name in _helpers()._CELL_FILES}}}
    if graded:
        decision = _old_grade(root, manifest, case, load)
        row['quality'] = decision['quality'][variant]
        row['provenance'].update(grade_decision_path=str(root / 'semantic-gates' / case / 'decision.json'),
                                grade_decision_sha256=_helpers()._hash(root / 'semantic-gates' / case / 'decision.json'))
    else:
        row['quality'] = None
        _require(not (root / 'semantic-gates' / case / 'decision.json').exists(), 'UNEXPECTED_OLD_BRIDGE_GRADE')
    return row


def _safety(root, load, controller):
    """仅接受已密封的特定旧自动门失败及独立最终安全证据，不改写原false。"""
    supplement = load('audit/post-run-safety-supplement.json')
    report = load('audit/post-run-protection.json')
    group = load('audit/final-process-group-readback.json')
    vector = load('formal-preflight/vector-close.json')
    close = load('formal-preflight/authorization-closed.json')
    watcher = load('scoring/metadata-watch-closed.json')
    _require(controller.get('batch_group_cleaned') is False and controller.get('cleanup_all_passed') is True
             and controller.get('status') == 'model_hard_deadline'
             and controller.get('wrapper_finished_before_original_total_deadline') is True,
             'ORIGINAL_CONTROLLER_STATE_CHANGED')
    _require(report.get('status') == 'failed'
             and report.get('errors') == [{'check': 'controller cleanup complete', 'details': None}]
             and report['checks_passed'] + 1 == report['checks_total']
             and report.get('actual_remaining_owned_group_members') == []
             and report.get('actual_remaining_scoped_processes') == []
             and report.get('audit_finished_before_original_total_deadline') is True,
             'OLD_PROTECTION_FAILURE_NOT_ISOLATED')
    _require(supplement.get('status') == 'passed_final_safety_state_with_original_automatic_gate_failure'
             and supplement.get('original_automatic_batch_group_gate_passed') is False
             and all(supplement.get(k) is True for k in (
                 'original_controller_record_unchanged', 'original_failed_protection_report_unchanged',
                 'automatic_end_to_end_closure_gate_not_rewritten', 'new6461_verified_stopped',
                 'formal_authorization_verified_closed', 'original5_containers_unchanged',
                 'old6441_6451_verified_stopped', 'protected_hashes_and_seals_frozen_sources_passed',
                 'pending_not_reclassified_or_released', 'before_original_total_deadline'))
             and supplement.get('final_live_owned_process_group_members') == []
             and supplement.get('final_live_scoped_controller_model_metadata_processes') == []
             and all(supplement.get(k) == 0 for k in ('additional_model_calls', 'additional_kills',
                                                    'service_operations', 'Ledger_instantiations', 'ledger_mutations')),
             'FINAL_SAFETY_SUPPLEMENT_REQUIRED')
    _require(group.get('status') == 'passed' and group.get('initial_controller_result_preserved') is True
             and group.get('controller_initial_batch_group_cleaned') is False
             and group.get('alive_group_members_after') == []
             and group.get('before_original_total_deadline') is True
             and vector.get('status') == 'passed' and vector['identity']['running'] is False
             and close.get('status') == 'closed' and close.get('no_resume_or_resend') is True
             and watcher.get('status') == 'closed' and watcher.get('model_calls') == 0,
             'FINAL_OWNED_RESOURCES_NOT_CLOSED')
    _require(set(supplement['sources']) == {
        'audit/post-run-protection.json', 'audit/final-process-group-readback.json',
        'audit/readonly-post-run.json', 'formal-control/controller-result.json',
        'formal-preflight/vector-close.json', 'formal-preflight/authorization-closed.json',
        'scoring/metadata-watch-closed.json'}, 'COMPLETE_SAFETY_SOURCES_REQUIRED')
    for relative, digest in supplement['sources'].items():
        load(relative, False)
        _require(_helpers()._hash(root / relative) == digest, 'SAFETY_SOURCE_CHANGED')
    return supplement


def _anchor(anchor_root):
    h = _helpers()
    root, seal, seal_sha = _sealed(anchor_root)
    manifest = h._read(root / 'manifest.formal.json')
    previous = manifest['continuation_contract']
    old_root, old_seal, old_sha = _sealed(previous['anchor_root'])
    _require(old_sha == previous['anchors']['SEAL.json'], 'ORIGINAL_N_SEAL_CHANGED')
    key = (str(root), seal_sha, str(old_root), old_sha)
    if key in _ANCHOR_CACHE:
        return deepcopy(_ANCHOR_CACHE[key])
    n_contract, original, cases, materials = h._anchor(old_root)
    _require(_equal(previous, n_contract), 'PREVIOUS_CONTINUATION_CHANGED')
    anchors = {'SEAL.json': seal_sha}
    load = _loader(root, seal, anchors)
    manifest = load('manifest.formal.json')
    _require(manifest['execution_order'] == original['execution_order']
             and Path(manifest['experiment_root']) == root
             and Path(manifest['runtime_root']) == root / 'runtime', 'R2_IDENTITY_CHANGED')
    auth = load('formal-execution-auth.json')
    _require(auth.get('approved') is False and auth.get('allowed_modes') == []
             and auth.get('status') == 'closed_no_resume'
             and auth.get('manifest_sha256') == _digest({k: v for k, v in manifest.items() if k != 'cell_absolute_deadline'}),
             'OLD_AUTH_NOT_CLOSED')
    controller = load('formal-control/controller-result.json')
    supplement = _safety(root, load, controller)
    _require(controller['original_absolute_total_deadline'] == n_contract['clock']['round_absolute_deadline']
             and controller['model_deadline'] == n_contract['clock']['model_absolute_deadline']
             and supplement['original_total_deadline'] == n_contract['clock']['round_absolute_deadline'],
             'ORIGINAL_CUTOFF_HISTORY_CHANGED')
    ledger = _ledger(root, load)
    meta, config = ledger['metadata'], ledger['metadata']['config']
    _require(meta['started_at'] == n_contract['clock']['round_started_at']
             and meta.get('halt_reason') is None and meta.get('first_error') is None
             and _equal(config.get('carried_usage'), n_contract['carried_usage'])
             and _equal(manifest['ledger_limits']['carried_usage'], n_contract['carried_usage']),
             'OLD_LEDGER_CARRY_OR_START_CHANGED')
    for k in ('allowed_cells', 'batch_tokens', 'batch_requests', 'batch_seconds', 'root_tokens',
              'root_requests', 'root_seconds', 'max_output_tokens'):
        expected = manifest['ledger_limits'][k]
        _require(_equal(config[k], sorted(expected) if k == 'allowed_cells' else expected), 'OLD_LEDGER_LIMITS_CHANGED')
    _require(config['batch_nano_yuan'] == int(__import__('decimal').Decimal(str(manifest['ledger_limits']['batch_yuan'])) * 10**9),
             'OLD_LEDGER_MONEY_CHANGED')
    order = original['execution_order']
    roots = ledger['roots']
    _require(len(roots) == 8 and len({x['root_id'] for x in roots}) == 8
             and sorted([x['case_id'], x['variant']] for x in roots) == sorted(order[4:12]), 'EXACT_EIGHT_OLD_ROOTS_REQUIRED')
    _require(all(x['root_id'] in {r['root_id'] for r in roots} for x in ledger['reservations']),
             'OLD_PHYSICAL_ROOT_OUTSIDE_MATRIX')
    last = next(x for x in roots if [x['case_id'], x['variant']] == order[11])
    pending = [x for x in ledger['reservations'] if x['status'] in {'reserved', 'sent', 'unknown'}]
    _require(len(pending) == 2 and all(x['status'] == 'sent' and x['root_id'] == last['root_id']
             and x['actual_tokens'] is None and x['actual_nano'] is None for x in pending),
             'EXACT_OLD_UNKNOWN_RESERVATIONS_REQUIRED')
    _require(all(x['status'] in {'settled', 'sent', 'not_sent'} for x in ledger['reservations']), 'OLD_STATUS_UNCLASSIFIED')
    charge = {'tokens': 0, 'requests': 0, 'nano_yuan': 0}
    for row in ledger['reservations']:
        if row['status'] == 'not_sent':
            continue
        _require(all(h._integer(row[k]) for k in ('actual_tokens', 'actual_nano') if row['status'] == 'settled')
                 and all(h._integer(row[k]) for k in ('reserved_tokens', 'reserved_nano')), 'OLD_COST_INVALID')
        charge['tokens'] += row['actual_tokens'] if row['status'] == 'settled' else row['reserved_tokens']
        charge['nano_yuan'] += row['actual_nano'] if row['status'] == 'settled' else row['reserved_nano']
        charge['requests'] += 1
    carry = {k: charge[k] + n_contract['carried_usage'][k] for k in charge}
    _require(_equal(charge, supplement['new_charged_or_reserved'])
             and _equal(carry, supplement['cumulative_charged_or_reserved']), 'CONSERVATIVE_CARRY_CHANGED')
    _require({x['reservation_id'] for x in pending} == {x['reservation_id'] for x in supplement['true_unknown_or_pending']},
             'UNKNOWN_RESERVATION_IDENTITY_CHANGED')
    for row in pending:
        proof = next(x for x in supplement['true_unknown_or_pending'] if x['reservation_id'] == row['reservation_id'])
        _require(all(row[k] == proof[k] for k in proof), 'UNKNOWN_RESERVATION_CHANGED')
    carry['seconds'] = n_contract['clock']['pre_round_carried_seconds']
    old_load = _loader(old_root, old_seal, {})
    old_ledger = _ledger(old_root, old_load)
    retained = [_row(old_root, original, i, old_ledger, old_load, graded=True) for i in range(4)]
    retained.extend(_row(root, manifest, i, ledger, load, graded=i < 10) for i in range(4, 11))
    grants = root / 'semantic-gates/grants'
    _require({p.name for p in grants.iterdir()} == {str(i) + '.json' for i in range(4, 12)}, 'OLD_GRANT_SET_CHANGED')
    for i in range(4, 12):
        grant = load('semantic-gates/grants/' + str(i) + '.json')
        _require(grant['position'] == i and [grant['case_id'], grant['variant']] == order[i]
                 and grant['manifest_sha256'] == _digest({k: v for k, v in manifest.items() if k != 'cell_absolute_deadline'})
                 and re.fullmatch('[0-9a-f]{64}', grant['attempt_id'])
                 and grant['attempt_kind'] == ('technical_retry' if i == 4 else 'first_attempt')
                 and grant['continuation_sha256'] == _digest(previous), 'OLD_GRANT_IDENTITY_CHANGED')
        if i < 11:
            result = h._read(root / 'runtime/cells' / (order[i][0] + '-' + order[i][1]) / 'result.json')
            _require(result['experiment_attempt'] == {k: grant[k] for k in
                     ('attempt_id', 'attempt_kind', 'position', 'continuation_sha256')}, 'OLD_ATTEMPT_CHANGED')
    cell = root / 'runtime/cells' / (order[11][0] + '-' + order[11][1])
    db_path = cell / 'task.sqlite3'
    load(str(db_path.relative_to(root)), False)
    with closing(sqlite3.connect(db_path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        stored = db.execute('SELECT status FROM tasks WHERE task_id=?', (last['root_id'],)).fetchone()
    _require(stored == ('running',) and not (cell / 'result.json').exists(), 'OLD_INTERRUPTED_ROOT_CHANGED')
    failed = deepcopy(n_contract['failed_attempt'])
    failed.update(source_manifest_path=str(old_root / 'manifest.formal.json'),
                  source_manifest_sha256=h._hash(old_root / 'manifest.formal.json'))
    interrupted = {'case_id': order[11][0], 'variant': order[11][1], 'root_id': last['root_id'],
        'status': stored[0], 'stored_status': stored[0], 'attempt_outcome': 'interrupted_at_original_model_cutoff',
        'original_position': 11, 'retained': False, 'issues': ['original_model_hard_deadline', 'unresolved_physical_receipts'],
        'source_manifest_path': str(root / 'manifest.formal.json'), 'source_manifest_sha256': h._hash(root / 'manifest.formal.json'),
        'provenance': {'anchor_root': str(root), 'cell_root': str(cell),
                       'files': {'task.sqlite3': h._hash(db_path)}}}
    result = {'anchor_root': str(root), 'anchor_seals': {str(old_root): old_sha, str(root): seal_sha},
        'anchors': anchors, 'retained_rows': retained, 'technical_attempts': [failed, interrupted],
        'retained_count': len(retained), 'new_execution_order': deepcopy(order[11:]),
        'full_execution_order': deepcopy(order), 'carried_usage': carry,
        'closed_ledger_roots': [str(old_root), str(root)], 'pending_reservations': pending,
        'original_clock': deepcopy(n_contract['clock']), 'anchor_manifest': original,
        'latest_anchor_manifest': manifest, 'old_cases': cases, 'old_materials': materials,
        'final_safety_proof': {'original_automatic_gate_passed': False,
                             'supplement_path': str(root / 'audit/post-run-safety-supplement.json'),
                             'supplement_sha256': anchors['audit/post-run-safety-supplement.json']}}
    _ANCHOR_CACHE[key] = deepcopy(result)
    return result


def _timing(value, base):
    if value is None:
        return None, {'policy': 'pending', 'round_started_at': base['original_clock']['round_started_at'],
            'round_absolute_deadline': None, 'model_absolute_deadline': None,
            'pre_round_carried_seconds': base['carried_usage']['seconds'],
            'batch_seconds': base['original_clock']['batch_seconds'], 'global_deadline': None, 'model_deadline': None}
    _require(isinstance(value, dict)
             and set(value) == {'policy', 'global_deadline', 'model_deadline', 'authorized_by_user',
                                'scope_receipt_path', 'scope_receipt_sha256'}
             and value.get('policy') == 'financial_only'
             and value.get('global_deadline', 1) is None and value.get('model_deadline', 1) is None
             and value.get('authorized_by_user') is True, 'FINANCIAL_ONLY_USER_AUTH_REQUIRED')
    path = Path(value['scope_receipt_path'])
    _require(path.is_absolute() and path.resolve() == path
             and _helpers()._hash(path) == value['scope_receipt_sha256'], 'TIMING_SCOPE_HASH_CHANGED')
    scope = _helpers()._read(path)
    _require(scope.get('authorized_by_user') is True and scope.get('cancel_global_time_limit') is True
             and isinstance(scope.get('user_request'), str) and scope['user_request'].strip(), 'TIME_CANCELLATION_NOT_EXPLICIT')
    _, clock = _timing(None, base)
    clock['policy'] = 'financial_only'
    return deepcopy(value), clock


def build_completion_contract(anchor_root, timing_authorization):
    """从两份旧密封派生85格补测计划；本函数不产生执行授权或账本。"""
    try:
        base = _anchor(anchor_root)
        timing, clock = _timing(timing_authorization, base)
        keys = ('anchor_root', 'anchor_seals', 'anchors', 'retained_rows', 'technical_attempts',
                'retained_count', 'new_execution_order', 'full_execution_order', 'carried_usage',
                'closed_ledger_roots', 'pending_reservations', 'final_safety_proof')
        return {**{k: deepcopy(base[k]) for k in keys}, 'version': VERSION,
                'clock': clock, 'original_clock': deepcopy(base['original_clock']),
                'timing_authorization': timing, 'evaluation_policy': deepcopy(EVALUATION_POLICY),
                'retry_policy': {'original_position': 11, 'max_new_attempts': 1,
                                 'preserve_both_old_technical_attempts': True,
                                 'never_resend_old_unknown_requests': True}}
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError, sqlite3.Error) as exc:
        raise PermissionError('FINAL_COMPLETION_ANCHOR_INVALID') from exc


def _sources(manifest, original):
    h = _helpers()
    allowed = h._EXPERIMENT_SCRIPTS | {'backend/scripts/agent_final_completion.py',
        'backend/scripts/harness_ab_budget.py', 'backend/scripts/harness_financial_process.py'}
    for key in ('settings', 'variant_settings', 'pair_review_policy', 'final_suite_policy'):
        _require(_equal(manifest.get(key), original.get(key)), 'EXPERIMENT_CONDITIONS_CHANGED')
    _require(_equal(h._read(manifest['model_manifest_path']), h._read(original['model_manifest_path'])),
             'MODEL_RESOURCE_CHANGED')
    common = []
    for variant in ('A', 'B'):
        expected = original['prompt_bundle_source_contract']['variants'][variant]['source_files']
        actual = h._runtime_files(Path(manifest['source_roots'][variant]))
        _require(set(expected) <= set(actual) and set(actual) - set(expected) <= allowed, 'RUNTIME_FILE_SET_CHANGED')
        hashes = {rel: h._hash(path) for rel, path in actual.items()}
        _require(all(hashes[rel] == digest for rel, digest in expected.items() if rel not in allowed),
                 'BUSINESS_OR_DEPENDENCY_BYTES_CHANGED')
        _require(_equal(manifest['prompt_bundle_source_contract']['variants'][variant]['source_files'], hashes),
                 'DECLARED_RUNTIME_INVENTORY_CHANGED')
        old, new = original['loaded_prompt_contract'][variant], manifest['loaded_prompt_contract'][variant]
        _require(set(new['assembled_prompt_sha256']) == h._ROLES
                 and _equal(new['assembled_prompt_sha256'], old['assembled_prompt_sha256'])
                 and _equal(new['python_runtime'], old['python_runtime'])
                 and new['code_fingerprint_format'] == old['code_fingerprint_format']
                 and set(new['registry_sha256']) == set(old['registry_sha256']), 'BUSINESS_PROMPTS_OR_RUNTIME_CHANGED')
        common.append({name: digest for name, digest in hashes.items() if name in allowed})
    _require(_equal(common[0], common[1]), 'COMMON_EXPERIMENT_OVERLAY_CHANGED')


def validate_completion(manifest):
    """只识别显式v2键；双合同、缺授权或任何源数据漂移均拒绝放行。"""
    if 'completion_contract' not in manifest:
        return None
    try:
        _require('continuation_contract' not in manifest, 'TWO_CONTRACT_KEYS_FORBIDDEN')
        supplied = manifest['completion_contract']
        _require(isinstance(supplied, dict) and supplied.get('version') == VERSION, 'EXPLICIT_VERSION_REQUIRED')
        expected = build_completion_contract(supplied['anchor_root'], supplied['timing_authorization'])
        _require(_equal(supplied, expected), 'CONTRACT_OR_ANCHOR_CHANGED')
        base = _anchor(supplied['anchor_root'])
        original = base['anchor_manifest']
        _require(manifest.get('experiment_kind') == 'prompt_bundle_v1', 'FIXED_SUITE_KIND_REQUIRED')
        _require(_equal(manifest.get('evaluation_policy'), EVALUATION_POLICY), 'DESCRIPTIVE_EVALUATION_POLICY_REQUIRED')
        _require(_equal(manifest['execution_order'], expected['full_execution_order']), 'ORIGINAL_96_ORDER_REQUIRED')
        old_limits = original['ledger_limits']
        limits = manifest['ledger_limits']
        desired = {**old_limits, 'carried_usage': expected['carried_usage'], 'enforce_batch_seconds': False}
        _require(_equal(limits, desired), 'FINANCE_OR_ROOT_LIMITS_CHANGED')
        _require(manifest.get('carried_budget_source') == {
            'version': VERSION, 'anchor_root': expected['anchor_root'],
            'ledger_path': str(Path(expected['anchor_root']) / 'runtime/budget.sqlite3'),
            'ledger_sha256': expected['anchors']['runtime/budget.sqlite3'],
            'seal_sha256': expected['anchor_seals'][expected['anchor_root']]}, 'LATEST_CONSERVATIVE_CARRY_SOURCE_REQUIRED')
        owned = Path(manifest['experiment_root'])
        _require(owned.is_absolute() and owned.resolve() == owned, 'OWNED_ROOT_REQUIRED')
        for old in expected['closed_ledger_roots']:
            path = Path(old)
            _require(not owned.is_relative_to(path) and not path.is_relative_to(owned), 'OLD_RUNTIME_REUSE_FORBIDDEN')
        for key, relative in {'runtime_root': 'runtime', 'semantic_gate_root': 'semantic-gates',
                              'prepared_input_root': 'inputs', 'cases_path': 'inputs/cases.json',
                              'model_manifest_path': 'inputs/model-manifest.json'}.items():
            path = Path(manifest[key])
            _require(path.resolve() == path == owned / relative, 'OWNED_GEOMETRY_REQUIRED')
        for variant in ('A', 'B'):
            path = Path(manifest['source_roots'][variant])
            _require(path.resolve() == path == owned / 'snapshots' / variant, 'OWNED_SOURCE_GEOMETRY_REQUIRED')
        authorization = Path(manifest['authorization_path'])
        _require(authorization.resolve() == authorization and authorization.is_relative_to(owned)
                 and not any(authorization.is_relative_to(owned / p) for p in ('runtime', 'inputs', 'snapshots')),
                 'NEW_OWNED_AUTHORIZATION_REQUIRED')
        if expected['clock']['policy'] == 'pending':
            _require(manifest.get('real_execution_released') is False
                     and manifest.get('live_qualification_released') is False, 'PENDING_TIMING_CANNOT_RELEASE')
        _helpers()._check_inputs(manifest, original, base['old_cases'], base['old_materials'])
        _sources(manifest, original)
        return {**deepcopy(expected), 'anchor_manifest': deepcopy(original),
                'latest_anchor_manifest': deepcopy(base['latest_anchor_manifest'])}
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError, sqlite3.Error) as exc:
        raise PermissionError('FINAL_COMPLETION_INVALID') from exc


def retained_cell(manifest, case_id, variant):
    """返回严格核验的旧格来源，允许新配对桥接旧B与新A。"""
    context = validate_completion(manifest)
    if context is None:
        return None
    for row in context['retained_rows']:
        if [row['case_id'], row['variant']] == [case_id, variant]:
            return {**deepcopy(row), 'cell_root': row['provenance']['cell_root'],
                    'source_manifest': _helpers()._read(row['source_manifest_path']),
                    'grade_decision_path': row['provenance'].get('grade_decision_path')}
    return None


def prior_pair_manifest(manifest, offset):
    """仅完整的旧五对使用其旧manifest；桥接对及后续对属于新合同。"""
    context = validate_completion(manifest)
    if context is None:
        return None
    _require(type(offset) is int and offset >= 0 and offset % 2 == 0, 'PAIR_OFFSET_INVALID')
    if offset >= context['retained_count'] - 1:
        return None
    return deepcopy(context['anchor_manifest'] if offset < 4 else context['latest_anchor_manifest'])


def technical_attempts(context):
    """给汇总器提供原始旧失败与中断；不将重试成功覆盖旧尝试。"""
    return deepcopy(context.get('technical_attempts', []))
