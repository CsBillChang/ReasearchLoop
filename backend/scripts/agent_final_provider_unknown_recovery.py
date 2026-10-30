"""V6已发送Provider未知收据的只读续接；失败仍属原主观测。"""
from __future__ import annotations
from collections import Counter
import ast
import hashlib
from datetime import datetime
import math
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import re
import sqlite3

VERSION = 'agent_final_completion_v6'
PROOF_VERSION = 'provider_unknown_stop_proof_v1'
STOP_REASON = 'transport_or_receipt_RemoteProtocolError'
BATCH_STOP = 'cell_receipt_or_isolation_gate'
_PARSED_ANCHORS = {}
_ERRORS = ['ledger_halted', 'native_response_unknown', 'unresolved_physical_receipt']
_ACTIONS = {'real_model_calls', 'network_calls', 'Ledger_instances', 'Settings_instances', 'old_file_mutations', 'sqlite_mutations'}


def _c():
    import agent_final_completion as c
    return c


def _chain():
    import agent_final_completion_chain as chain
    return chain


def _require(value, reason):
    if not value:
        raise PermissionError('FINAL_PROVIDER_UNKNOWN_' + reason)


def _integer(v):
    return type(v) is int and v >= 0


def _proof_ref(ref):
    c = _c()
    _require(isinstance(ref, dict) and set(ref) == {'version', 'path', 'sha256'} and ref['version'] == PROOF_VERSION
             and isinstance(ref['path'], str) and isinstance(ref['sha256'], str)
             and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'EXACT_PROOF_REFERENCE_REQUIRED')
    p = Path(ref['path'])
    _require(p.is_absolute() and p.resolve() == p and not p.is_symlink() and p.is_file()
             and c._verified_hash(p) == ref['sha256'], 'PROOF_BYTES_CHANGED')
    return c._helpers()._read(p)


def _normal_safety(root, load):
    """新锚必须直接通过完整原审计；历史特殊修正仅由父V5自行验证。"""
    report = load('audit/post-run-protection.json')
    _require(report.get('status') == 'passed' and report.get('safety_closed') is True
        and report.get('measurement_complete') is False and report.get('errors') == []
        and report.get('data_integrity_errors') == [], 'NORMAL_POST_AUDIT_REQUIRED')
    scopes = {'closure','containers','dataset','git_refs','old_failure','physical_budget_readonly',
        'processes','protected','seals','selected_production_materials','sources'}
    actions = report.get('actions') or {}; counts = report.get('checks_by_scope') or {}
    _require(set(report.get('sections',{})) == scopes and set(counts) == {'data','safety'}
        and all(_integer(v) and v > 0 for v in counts.values())
        and report.get('checks_total') == report.get('checks_passed') == sum(counts.values())
        and actions.get('only_new_report_written') is True
        and {'Ledger_instantiations','business_Settings_instantiations','model_calls','business_SQLite_writes',
            'pending_releases','service_mutations'} <= set(actions)
        and all(type(v) is int and v == 0 for k,v in actions.items() if k != 'only_new_report_written'),
        'COMPLETE_NORMAL_POST_AUDIT_REQUIRED')
    return report


def _totals(rows, at=None):
    selected = [r for r in rows if r['status'] != 'not_sent' and (at is None or r['created_at'] <= at)]
    def settled(r):
        return r['status'] == 'settled' and (at is None or r['finished_at'] <= at)
    return {'requests': len(selected),
        'tokens': sum(r['actual_tokens'] if settled(r) else r['reserved_tokens'] for r in selected),
        'nano': sum(r['actual_nano'] if settled(r) else r['reserved_nano'] for r in selected)}


def _finite_time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _frozen_rate(row, config):
    c = _c(); usage = c._usage(row['usage_json']); cached = usage.get('prompt_cache_hit_tokens', 0)
    _require(_integer(cached) and cached <= usage['prompt_tokens']
        and [row.get(k) for k in ('actual_input', 'actual_output', 'actual_tokens')] ==
            [usage[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
        and row['actual_nano'] == cached * config['cached_input_nano_per_token']
            + (usage['prompt_tokens'] - cached) * config['input_nano_per_token']
            + usage['completion_tokens'] * config['output_nano_per_token'], 'PHYSICAL_FROZEN_RATE_CHANGED')


def _source_contract(root, manifest, proof, load):
    """保留原预算准入证明，再核发送后异常到unknown的原始路径。"""
    from agent_final_root_budget_recovery import _source_contract as budget_source
    budget_source(root, manifest, proof, load)
    tree = ast.parse((root / 'snapshots/B/backend/scripts/harness_ab_transport.py').read_text())
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_handle_request']
    _require(len(handlers) == 1, 'UNIQUE_TRANSPORT_REQUIRED')
    statements = handlers[0].body
    sent = [i for i, stmt in enumerate(statements) if any(isinstance(n, ast.Call)
        and ast.unparse(n.func) == 'self.ledger.mark_sent' for n in ast.walk(stmt))]
    received = [i for i, stmt in enumerate(statements) if any(isinstance(n, ast.Call)
        and ast.unparse(n.func) == 'self.inner.handle_request' for n in ast.walk(stmt))]
    _require(len(sent) == len(received) == 1 and sent[0] < received[0], 'SEND_BEFORE_PROVIDER_REQUIRED')
    exception = statements[received[0]]
    _require(isinstance(exception, ast.Try) and len(exception.handlers) == 1
        and ast.unparse(exception.handlers[0].type) == 'BaseException'
        and len(exception.handlers[0].body) == 2
        and ast.unparse(exception.handlers[0].body[0]) == "self._unknown(identity, 'transport_or_receipt_' + type(exc).__name__)"
        and isinstance(exception.handlers[0].body[1], ast.Raise), 'SOURCE_UNKNOWN_EXCEPTION_PATH_CHANGED')


def _provider_evidence(ledger, root_id):
    """从真实只读账本重算唯一已发送未知请求，不接受证明文件自填占用。"""
    c = _c(); meta = ledger['metadata']; error = meta.get('first_error') or {}; config = meta['config']
    rows = ledger['reservations']; roots = ledger['roots']; root_map = {r['root_id']: r for r in roots}
    pending = [r for r in rows if r.get('status') != 'settled']
    _require(len(root_map) == len(roots) and len(pending) == 1 and pending[0].get('status') == 'unknown', 'ONE_NEW_UNKNOWN_REQUIRED')
    unknown = pending[0]; at = error.get('detected_at')
    _require(meta.get('halt_reason') == error.get('reason') == unknown.get('reason') == STOP_REASON
        and error.get('operation') == 'receipt_unknown' and error.get('root_id') == root_id == unknown.get('root_id')
        and error.get('reservation_id') == unknown.get('reservation_id') and _finite_time(at)
        and c._equal(error.get('limits'), config), 'EXACT_PROVIDER_FIRST_ERROR_REQUIRED')
    _require(len({r['reservation_id'] for r in rows}) == len(rows) and root_id in root_map
        and all(_finite_time(r.get('started_at')) and r['started_at'] <= at for r in roots), 'ROOT_TIME_OR_ID_INVALID')
    for row in rows:
        _require(row['root_id'] in root_map and all(_finite_time(row.get(k)) for k in ('created_at','sent_at','finished_at'))
            and root_map[row['root_id']]['started_at'] <= row['created_at'] <= row['sent_at'] <= row['finished_at']
            and row['created_at'] <= at and row['sent_at'] <= at
            and row['sent_at'] <= root_map[row['root_id']]['started_at'] + config['root_seconds'], 'POST_HALT_CREATE_OR_SEND_OR_TIME_INVALID')
        _require(all(_integer(row.get(k)) and row[k] > 0 for k in ('reserved_input','reserved_output','reserved_tokens','reserved_nano'))
            and row['reserved_input'] == 2 ** 20 and row['reserved_output'] <= config['max_output_tokens']
            and row['reserved_tokens'] == row['reserved_input'] + row['reserved_output']
            and row['reserved_nano'] == row['reserved_input'] * config['input_nano_per_token']
                + row['reserved_output'] * config['output_nano_per_token'], 'PHYSICAL_RESERVATION_BOUNDS_CHANGED')
        if row['status'] == 'settled':
            _frozen_rate(row, config)
        else:
            _require(all(row.get(k) is None for k in ('actual_input','actual_output','actual_tokens','actual_nano'))
                and row['finished_at'] <= at and row.get('usage_json') in (None, 'null')
                and row.get('response_json') in (None, 'null'), 'UNKNOWN_MUST_KEEP_NULL_ACTUALS')
    root = _totals([r for r in rows if r['root_id'] == root_id], at); batch = _totals(rows, at)
    for key, src in (('tokens','tokens'),('requests','requests'),('nano','nano_yuan')):
        batch[key] += config['carried_usage'][src]
    _require(c._equal(root, error.get('root_occupancy')) and c._equal(batch, error.get('batch_occupancy')), 'FIRST_ERROR_OCCUPANCY_CHANGED')
    _require(0 <= at - root_map[root_id]['started_at'] < config['root_seconds']
        and root['tokens'] <= config['root_tokens'] and root['requests'] <= config['root_requests']
        and batch['tokens'] <= config['batch_tokens'] and batch['requests'] <= config['batch_requests']
        and batch['nano'] <= config['batch_nano_yuan'], 'ORIGINAL_ROOT_OR_GLOBAL_CAP_EXCEEDED')
    return {'first_error_sha256': c._digest(error), 'unknown_reservation_id': unknown['reservation_id'],
        'unknown_reservation_sha256': c._digest(unknown), 'root_occupancy': root, 'batch_occupancy': batch,
        'final_root': _totals([r for r in rows if r['root_id'] == root_id])}


def _receipts(root_id, native, physical, config=None):
    """完整结算部分逐一核验；唯一已发送unknown保持用量未知。"""
    c = _c(); success = [r for r in native if r.get('status') == 'succeeded']
    pending = [r for r in native if r.get('status') != 'succeeded']
    unknown = [r for r in physical if r.get('status') != 'settled']
    _require(native and len({r.get('call_id') for r in native}) == len(native)
        and all(isinstance(r.get('call_id'), str) and r['call_id'] for r in native)
        and len(pending) == len(unknown) == 1, 'ONE_NATIVE_PHYSICAL_UNKNOWN_REQUIRED')
    call, row = pending[0], unknown[0]
    response = json.loads(call.get('response_json') or 'null'); usage = json.loads(call.get('usage_json') or 'null')
    _require(call.get('status') == 'unknown' and call.get('sent_state') == 'sent'
        and type(call.get('attempt_no')) is int and call['attempt_no'] == 1
        and call.get('request_id') is None and usage == {} and isinstance(response, dict)
        and set(response) == {'error'} and isinstance(response['error'], str) and bool(response['error'])
        and row.get('root_id') == root_id and row.get('status') == 'unknown' and row.get('reason') == STOP_REASON
        and all(row.get(k) is None for k in ('actual_input','actual_output','actual_tokens','actual_nano'))
        and row.get('usage_json') in (None, 'null') and row.get('response_json') in (None, 'null'), 'SENT_UNKNOWN_WITHOUT_RECEIPT_REQUIRED')
    settled = [r for r in physical if r.get('status') == 'settled']
    c._receipts(root_id, success, settled)
    if config is not None:
        for item in settled: _frozen_rate(item, config)
    return {'physical_receipts_verified': False, 'receipt_state_verified': True, 'actual_usage_known': False,
        'provider_unknown_stop_verified': True, 'unconfirmed_request_count': 1,
        'native_status_counts': dict(Counter(r['status'] for r in native)),
        'physical_status_counts': dict(Counter(r['status'] for r in physical))}


def _binding(root, proof, load, relative, *, parse=True):
    c = _c(); value = load(relative, parse)
    _require(proof.get('bindings', {}).get(relative) == c._verified_hash(root / relative), 'COMPLETE_PROOF_BINDING_REQUIRED')
    return value


def _unknown_row(root, manifest, position, ledger, proof, load, raw):
    """保留原failed主观测及未知收据，不改成额外技术attempt。"""
    c = _c(); case, variant = manifest['execution_order'][position]; rel = 'runtime/cells/' + case + '-' + variant; cell = root / rel
    def get(name, parse=True):
        return _binding(root, proof, load, rel + '/' + name, parse=parse)
    result = get('result.json'); view = get('task-view.json'); native = get('model-calls.json'); actions = get('actions.json')
    events = get('events.json'); stored = get('root-row.json'); snapshot = get('ledger-snapshot.json'); get('task.sqlite3', False)
    rid = raw['root_id']
    _require(result.get('status') == 'executed' and result.get('ledger_halt') == STOP_REASON and result.get('case_id') == case
        and result.get('variant') == variant and result.get('root_id') == view.get('task_id') == rid
        and result.get('task_status') == view.get('status') == raw.get('status') == 'failed'
        and c._equal(result.get('loaded_supervisor'), manifest['loaded_prompt_contract'][variant]), 'ORIGINAL_FAILED_PROVIDER_IDENTITY_REQUIRED')
    prompt = next(x['prompt'] for x in load('inputs/cases.json')['cases'] if x['case_id'] == case)
    _require(view.get('prompt') == stored.get('prompt') == prompt and stored.get('task_id') == rid
        and stored.get('status') == 'failed' and stored.get('parent_task_id') is None
        and c._equal(json.loads(stored['result_json']), view.get('result')), 'ORIGINAL_ROOT_EXPORT_CHANGED')
    guard = result.get('live_isolation_guard') or {}; isolation = result.get('isolation') or {}
    _require(isinstance(isolation.get('denials'), list) and all(x.get('phase') == 'guard_selfcheck' for x in isolation['denials'])
        and not isolation.get('denied_connections') and guard.get('first_error') is None and guard.get('closed') is False
        and guard.get('ledger_bound') is True and guard.get('persistence_errors', []) == [], 'RUNTIME_ISOLATION_ERROR_NOT_RECOVERABLE')
    from agent_final_observation_recovery import exported_stop, _archives
    observation = exported_stop(view, events)
    for suffix in ('-wal', '-journal'):
        side = cell / ('task.sqlite3' + suffix)
        _require(not side.exists() or side.stat().st_size == 0, 'TASK_UNCHECKPOINTED')
    with closing(sqlite3.connect((cell / 'task.sqlite3').as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        db.execute('PRAGMA query_only=ON'); db.row_factory = sqlite3.Row
        sql_root = [dict(x) for x in db.execute('SELECT * FROM tasks WHERE task_id=?', (rid,))]
        sql_native = [dict(x) for x in db.execute('SELECT * FROM model_calls')]
        sql_events = [dict(x) for x in db.execute('SELECT * FROM task_events')]
        sql_tasks = [dict(x) for x in db.execute('SELECT * FROM tasks')]
        family = {x[0] for x in db.execute('SELECT task_id FROM tasks WHERE task_id=? OR root_task_id=?', (rid, rid))}
    _require(len(sql_root) == 1 and all(sql_root[0].get(k) == stored.get(k) for k in ('task_id', 'status', 'prompt', 'result_json', 'parent_task_id'))
        and all(x.get('task_id') in family for x in native)
        and c._equal(sorted(sql_native, key=lambda x: x['call_id']), sorted(native, key=lambda x: x['call_id'])), 'NATIVE_SQL_EXPORT_CHANGED')
    _native_stop(sql_root[0], sql_tasks, sql_events, view, events)
    physical = [r for r in ledger['reservations'] if r['root_id'] == rid]
    _require(any(r['root_id'] == rid and r['case_id'] == case and r['variant'] == variant for r in ledger['roots'])
        and c._equal([r for r in snapshot['reservations'] if r['root_id'] == rid], physical), 'CELL_LEDGER_SNAPSHOT_CHANGED')
    receipts = _receipts(rid, native, physical, ledger['metadata']['config'])
    _archives(root, case, variant, [r for r in physical if r['status'] == 'settled'], lambda relative, parse=True: _binding(root, proof, load, relative, parse=parse))
    request_names = {r['reservation_id'] + '.request.json' for r in physical}
    _require({p.name for p in (cell / 'requests').glob('*.request.json')} == request_names, 'UNMATCHED_HTTP_REQUEST_ARCHIVE')
    for p in (cell / 'requests').iterdir():
        _require(any(p.name == row['reservation_id'] + suffix for row in physical for suffix in
            ('.request.json', '.response.json', '.response.sse', '.response-meta.json')), 'UNMATCHED_HTTP_ARCHIVE')
        _binding(root, proof, load, str(p.relative_to(root)), parse=False)
    for group in actions.values() if isinstance(actions, dict) else [actions]:
        for action in group:
            data = (action.get('observation') or {}).get('data') or {}
            if action.get('tool_name') == 'retrieve_evidence' and action.get('status') == 'succeeded':
                _require(data.get('status') not in {'degraded', 'unavailable'} and not data.get('degradations')
                    and (data.get('strategy') or {}).get('name') == 'located_fp16' and data.get('effective_path') == 'located_bm25_rerank', 'ACTUAL_RETRIEVAL_CHANGED')
    _require(c._equal(receipts['native_status_counts'], raw.get('native_status_counts'))
        and c._equal(receipts['physical_status_counts'], raw.get('physical_status_counts')), 'RAW_RECEIPT_COUNTS_CHANGED')
    _unknown_request(root, manifest, native, physical, proof, load)
    return {**deepcopy(raw), 'retained': True, 'quality': None, **receipts,
        'source_manifest_path': str(root / 'manifest.formal.json'), 'source_manifest_sha256': c._verified_hash(root / 'manifest.formal.json'),
        'ledger_root': str(root), 'ledger_path': str(root / 'runtime/budget.sqlite3'), 'observed_status': 'failed', 'observation_stop': observation,
        'source_batch_record': deepcopy(raw), 'original_reconciliation_issues': deepcopy(_ERRORS), 'original_reconciliation_passed': False,
        'actual_task_family_tokens': None, 'actual_task_family_nano_yuan': None,
        'verified_settled_tokens_lower_bound': sum(r['actual_tokens'] for r in physical if r['status'] == 'settled'),
        'verified_settled_nano_yuan_lower_bound': sum(r['actual_nano'] for r in physical if r['status'] == 'settled'),
        'provider_unknown_stop_verified': True, 'artifact_owner': 'original_sealed_manifest', 'score_owner': 'current_manifest', 'deferred_review': True,
        'provenance': {'anchor_root': str(root), 'cell_root': str(cell), 'files': {name: c._verified_hash(cell / name)
            for name in (*c._helpers()._CELL_FILES, 'events.json', 'task.sqlite3', 'root-row.json', 'ledger-snapshot.json')}}}


def _anchor(anchor_root, proof_ref, seen):
    c = _c()
    return c._cached_anchor(VERSION + ':' + c._digest(proof_ref), anchor_root, seen,
        lambda: _anchor_uncached(anchor_root, proof_ref, seen))


def _anchor_uncached(anchor_root, proof_ref, seen):
    c = _c(); chain = _chain(); root = Path(anchor_root)
    _require((root / 'SEAL.json').is_file() and str(root) not in seen, 'SEALED_NONCYCLIC_ANCHOR_REQUIRED')
    root, seal, seal_sha = c._sealed(root); anchors = {'SEAL.json': seal_sha}; load = c._loader(root, seal, anchors)
    manifest = load('manifest.formal.json')
    _require(manifest.get('completion_contract', {}).get('version') in {'agent_final_completion_v5', VERSION}, 'V5_OR_V6_PARENT_REQUIRED')
    parent = chain._parent(manifest, {*seen, str(root)})
    _require(root == Path(manifest['experiment_root']) and Path(manifest['runtime_root']) == root / 'runtime'
        and str(root) not in parent['closed_ledger_roots'], 'NEW_CLOSED_ROOT_REQUIRED')
    _require(not any(Path(proof_ref['path']).is_relative_to(Path(old)) for old in [*parent['closed_ledger_roots'], str(root)]), 'PROOF_MUST_BE_OUTSIDE_OLD_ROOTS')
    proof = _proof_ref(proof_ref)
    _require(proof.get('version') == PROOF_VERSION and proof.get('status') == 'passed_exact_provider_unknown_observation_stop'
        and proof.get('anchor_root') == str(root) and proof.get('anchor_seal_sha256') == seal_sha
        and proof.get('parent_contract_sha256') == c._digest(manifest['completion_contract'])
        and proof.get('original_records_unchanged') is True and proof.get('original_reconciliation_passed') is False
        and proof.get('no_retry_or_reclassification') is True, 'EXACT_PROVIDER_UNKNOWN_PROOF_REQUIRED')
    _require(isinstance(proof.get('actions'), dict) and set(proof['actions']) == _ACTIONS
        and all(type(v) is int and v == 0 for v in proof['actions'].values()), 'ZERO_MUTATION_PROOF_REQUIRED')
    _require(isinstance(proof.get('bindings'), dict) and proof['bindings'], 'PROOF_BINDINGS_REQUIRED')
    _require(c._equal(proof['bindings'], {name: item['sha256'] for name, item in seal['files'].items()}),
        'EXACT_COMPLETE_SEALED_BINDINGS_REQUIRED')
    required = {'manifest.formal.json', 'inputs/cases.json', 'runtime/budget.sqlite3', 'runtime/ledger-final.json',
        'audit/post-run-protection.json', 'audit/post-run-github-readback.json', 'private-protection/post_run_protection.py',
        'formal-execution-auth.json', 'formal-preflight/authorization-closed.json', 'formal-preflight/vector-close.json',
        'formal-control/controller-result.json', 'audit/readonly-post-run.json'}
    for variant in ('A', 'B'):
        required.update('snapshots/' + variant + '/backend/scripts/' + name for name in
            ('harness_ab_budget.py', 'harness_ab_transport.py', 'run_read_paper_batch.py'))
    _require(required <= set(proof['bindings']), 'COMPLETE_ANCHOR_BINDINGS_REQUIRED')
    for relative, digest in proof['bindings'].items():
        _require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'PROOF_BINDING_SHAPE')
        _binding(root, proof, load, relative, parse=False)
    _normal_safety(root, load)
    chain._closure(root, manifest, load)
    _source_contract(root, manifest, proof, load)
    key = (str(root), seal_sha, c._digest(parent), c._digest(proof_ref), c._digest(proof))
    if key in _PARSED_ANCHORS:
        return deepcopy(_PARSED_ANCHORS[key])
    ledger = c._ledger(root, load); chain._ledger_limits(manifest, ledger, parent)
    batch = _binding(root, proof, load, 'runtime/batch-results.json'); first = parent['retained_count']; order = parent['full_execution_order']
    rows = batch['rows'][first:]; last = len(batch['rows']) - 1
    _require(len(order) == 96 and len(parent['technical_attempts']) == 3 and first < last < len(order)
        and last % 2 == 1 and order[last][0] == order[last - 1][0] and c._equal(batch['rows'][:first], parent['retained_rows'])
        and len(rows) == len(ledger['roots']) and batch.get('retained_observations') == first
        and batch.get('new_attempted_count') == len(rows) and batch.get('new_roots_planned') == len(order) - first
        and c._equal(batch.get('limits'), manifest['ledger_limits']) and batch.get('not_created') == order[last + 1:]
        and batch.get('status') == 'stopped' and batch.get('stop_reason') == BATCH_STOP, 'EXACT_CLOSED_PAIR_PREFIX_REQUIRED')
    raw = rows[-1]; rid = raw['root_id']; case, variant = order[last]
    _require(proof.get('case_id') == case and proof.get('variant') == variant
        and proof.get('position') == last and proof.get('root_id') == rid
        and proof.get('original_batch_stop_reason') == BATCH_STOP and proof.get('original_ledger_halt_reason') == STOP_REASON
        and proof.get('original_reconciliation_issues') == _ERRORS and raw.get('issues') == _ERRORS
        and raw.get('returncode') == 0, 'ORIGINAL_STOP_FIELDS_CHANGED')
    _require(c._equal(_provider_evidence(ledger, rid), proof.get('provider_receipt_evidence')), 'PROVIDER_RECEIPT_EVIDENCE_CHANGED')
    report = _binding(root, proof, load, 'runtime/' + case + '-' + variant + '-reconciliation.json')
    _require(report.get('issues') == _ERRORS and report.get('ledger_halt_reason') == STOP_REASON
        and c._equal(report.get('first_error'), ledger['metadata']['first_error'])
        and report.get('unmatched_native_response_ids') == report.get('unmatched_physical_response_ids') == []
        and report.get('native_not_sent') == [], 'ORIGINAL_RECONCILIATION_CHANGED')
    retained = deepcopy(parent['retained_rows']); chain._bind_deferred_reviews(root, manifest, parent, retained, load)
    identities = {r['root_id'] for r in retained + parent['technical_attempts']}; attempts = set()
    session = _binding(root, proof, load, 'semantic-gates/session.json')
    _require(session.get('manifest_sha256') == c._digest(manifest), 'SCOPE_SESSION_CHANGED')
    for offset, row in enumerate(rows):
        position = first + offset
        _require(row.get('original_position') == position and [row['case_id'], row['variant']] == order[position]
            and row['root_id'] not in identities, 'PRIMARY_ORDER_OR_ROOT_REUSED'); identities.add(row['root_id'])
        relative = 'runtime/cells/' + row['case_id'] + '-' + row['variant']
        result = _binding(root, proof, load, relative + '/result.json')
        grant = _binding(root, proof, load, 'semantic-gates/grants/' + str(position) + '.json')
        _require(grant.get('position') == position and [grant.get('case_id'), grant.get('variant')] == order[position]
            and grant.get('manifest_sha256') == c._digest(manifest) and grant.get('continuation_sha256') == c._digest(manifest['completion_contract'])
            and grant.get('session_nonce') == session.get('nonce') and grant.get('attempt_kind') == 'first_attempt'
            and isinstance(grant.get('attempt_id'), str) and re.fullmatch('[0-9a-f]{64}', grant['attempt_id']) and grant['attempt_id'] not in attempts
            and c._equal(result.get('experiment_attempt'), {k: grant[k] for k in ('attempt_id', 'attempt_kind', 'position', 'continuation_sha256')}), 'ORIGINAL_GRANT_CHANGED')
        attempts.add(grant['attempt_id'])
        if position == last:
            enriched = _unknown_row(root, manifest, position, ledger, proof, load, row)
        else:
            _require(row.get('issues') == [] and row.get('returncode') == 0 and result.get('ledger_halt') is None, 'OTHER_CELL_RECONCILIATION_NOT_CLEAN')
            enriched = c._row(root, manifest, position, ledger, load, graded=position < last - 1, allow_observation_stops=True)
            from agent_final_observation_recovery import _archives
            _archives(root, row['case_id'], row['variant'], [r for r in ledger['reservations'] if r['root_id'] == row['root_id']],
                lambda relative, parse=True: _binding(root, proof, load, relative, parse=parse))
            if position == last - 1:
                enriched.update(artifact_owner='original_sealed_manifest', score_owner='current_manifest', deferred_review=True)
        retained.append(enriched)
    _require(not (root / 'semantic-gates' / case).exists(), 'DEFERRED_PAIR_MUST_REMAIN_ABSENT')
    charge, pending = chain._charged(ledger)
    _require(len(pending) == 1 and pending[0]['root_id'] == rid and pending[0]['status'] == 'unknown'
        and not {r['reservation_id'] for r in ledger['reservations']} & {
        r['reservation_id'] for r in parent['pending_reservations']}, 'NEW_PENDING_OR_OLD_REQUEST_REUSED')
    carry = {k: parent['carried_usage'][k] + charge[k] for k in charge}; carry['seconds'] = parent['carried_usage']['seconds']
    final = _binding(root, proof, load, 'runtime/ledger-final.json')
    _require(c._equal(final.get('limits'), ledger['metadata']['config'])
        and c._equal(final.get('carried_usage'), parent['carried_usage']) and final.get('actual_known_tokens') == sum(r['actual_tokens'] for r in ledger['reservations'] if r['status'] == 'settled')
        and final.get('physical_sent_requests') == charge['requests'] and final.get('actual_known_estimated_nano_yuan') == sum(r['actual_nano'] for r in ledger['reservations'] if r['status'] == 'settled')
        and final.get('halt_reason') == STOP_REASON and c._equal(final.get('first_error'), ledger['metadata']['first_error'])
        and c._equal(final.get('charged_or_reserved'), {'tokens': carry['tokens'], 'requests': carry['requests'], 'nano': carry['nano_yuan']}), 'ORIGINAL_LEDGER_EXPORT_CHANGED')
    lineage = {'anchor_root': str(root), 'proof_ref': deepcopy(proof_ref), 'root_id': rid, 'position': last}; original = parent['anchor_manifest']
    value = {**deepcopy(parent), 'anchor_root': str(root), 'anchors': anchors,
        'anchor_seals': {**parent['anchor_seals'], str(root): seal_sha}, 'retained_rows': retained, 'retained_count': len(retained),
        'new_execution_order': deepcopy(order[len(retained):]), 'technical_attempts': deepcopy(parent['technical_attempts']), 'carried_usage': carry,
        'closed_ledger_roots': [*parent['closed_ledger_roots'], str(root)], 'pending_reservations': [*deepcopy(parent['pending_reservations']), *[dict(r, anchor_root=str(root)) for r in pending]],
        'latest_anchor_manifest': manifest, 'parent_contract_sha256': c._digest(manifest['completion_contract']), 'leaf_charged_or_reserved': charge,
        'minimum_complete_attempts': len(order) + len(parent['technical_attempts']), 'deferred_review_cases': [case], 'retry_positions': [],
        'provider_unknown_recovery_proof': deepcopy(proof_ref),
        'provider_unknown_recovery_lineage': [*deepcopy(parent.get('provider_unknown_recovery_lineage', [])), lineage],
        'old_cases': c._helpers()._read(original['cases_path']),
        'old_materials': c._helpers()._read(Path(original['prepared_input_root']) / 'material-snapshot.json')}
    _PARSED_ANCHORS[key] = deepcopy(value)
    return value


def build_provider_unknown_recovery_contract(anchor_root, timing_authorization, *, proof_ref, _seen=frozenset()):
    """只派生失败主观测、未知全额结转及未启动后缀，不生成授权。"""
    try:
        base = _anchor(anchor_root, proof_ref, _seen); timing, clock = _c()._timing(timing_authorization, base)
        excluded = {'anchor_manifest', 'latest_anchor_manifest', 'old_cases', 'old_materials', 'clock', 'timing_authorization', 'version', 'retry_policy'}
        return {**{k: deepcopy(v) for k, v in base.items() if k not in excluded}, 'version': VERSION, 'clock': clock, 'timing_authorization': timing,
            'retry_policy': {'retry_positions': [], 'preserve_all_old_technical_attempts': True,
                'never_resend_old_unknown_requests': True, 'never_retry_retained_primary_observations': True}}
    except PermissionError:
        raise
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError, sqlite3.Error) as exc:
        raise PermissionError('FINAL_PROVIDER_UNKNOWN_ANCHOR_INVALID') from exc


def validate_provider_unknown_recovery(manifest, *, _seen=frozenset()):
    """每次核当前manifest，缓存只承接不变的旧封存字节。"""
    c = _c()
    try:
        supplied = manifest['completion_contract']
        _require('continuation_contract' not in manifest and supplied.get('version') == VERSION, 'EXPLICIT_V6_REQUIRED')
        expected = build_provider_unknown_recovery_contract(supplied['anchor_root'], supplied['timing_authorization'], proof_ref=supplied['provider_unknown_recovery_proof'], _seen=_seen)
        _require(c._equal(supplied, expected), 'CONTRACT_OR_PROOF_CHANGED')
        base = _anchor(supplied['anchor_root'], supplied['provider_unknown_recovery_proof'], _seen); original = base['anchor_manifest']
        from agent_final_observation_recovery import validate_observation_policy
        from agent_final_pdf_acquisition import SOURCE_ACQUISITION_CONTRACT, validate_source_acquisition_contract
        _require(validate_observation_policy(manifest) and base.get('observation_recovery_lineage'), 'INHERITED_OBSERVATION_POLICY_REQUIRED')
        _require(manifest.get('experiment_kind') == 'prompt_bundle_v1' and c._equal(manifest.get('evaluation_policy'), c.EVALUATION_POLICY)
            and c._equal(manifest['execution_order'], expected['full_execution_order']) and validate_source_acquisition_contract(manifest)
            and c._equal(manifest.get('source_acquisition_contract'), SOURCE_ACQUISITION_CONTRACT), 'FIXED_SUITE_OR_ACQUISITION_CHANGED')
        _require(c._equal(manifest['ledger_limits'], {**original['ledger_limits'], 'carried_usage': expected['carried_usage'], 'enforce_batch_seconds': False}), 'ORIGINAL_FINANCIAL_AND_ROOT_CAPS_REQUIRED')
        _require(c._equal(manifest.get('carried_budget_source'), {'version': VERSION, 'anchor_root': expected['anchor_root'],
            'ledger_path': str(Path(expected['anchor_root']) / 'runtime/budget.sqlite3'), 'ledger_sha256': expected['anchors']['runtime/budget.sqlite3'],
            'seal_sha256': expected['anchor_seals'][expected['anchor_root']]}), 'EXACT_CARRY_SOURCE_REQUIRED')
        owned = Path(manifest['experiment_root'])
        _require(owned.is_absolute() and owned.resolve() == owned and all(not owned.is_relative_to(Path(p))
            and not Path(p).is_relative_to(owned) for p in expected['closed_ledger_roots']), 'NEW_OWNED_ROOT_REQUIRED')
        for key, rel in {'runtime_root': 'runtime', 'semantic_gate_root': 'semantic-gates', 'prepared_input_root': 'inputs',
                'cases_path': 'inputs/cases.json', 'model_manifest_path': 'inputs/model-manifest.json'}.items():
            path = Path(manifest[key]); _require(path.resolve() == path == owned / rel, 'OWNED_GEOMETRY_REQUIRED')
        for variant in ('A', 'B'):
            path = Path(manifest['source_roots'][variant]); _require(path.resolve() == path == owned / 'snapshots' / variant, 'OWNED_SOURCE_REQUIRED')
        auth = Path(manifest['authorization_path'])
        _require(auth.resolve() == auth and auth.is_relative_to(owned) and not any(auth.is_relative_to(owned / p)
            for p in ('runtime', 'inputs', 'snapshots')), 'NEW_AUTHORIZATION_REQUIRED')
        if expected['clock']['policy'] == 'pending':
            _require(manifest.get('real_execution_released') is False and manifest.get('live_qualification_released') is False, 'PENDING_CANNOT_RELEASE')
        c._helpers()._check_inputs(manifest, original, base['old_cases'], base['old_materials']); c._sources(manifest, original)
        for variant in ('A', 'B'):
            current = manifest['prompt_bundle_source_contract']['variants'][variant]['source_files']
            previous = base['latest_anchor_manifest']['prompt_bundle_source_contract']['variants'][variant]['source_files']
            _require(all(current.get(name) == previous.get(name) and current.get(name) is not None for name in
                ('backend/scripts/harness_ab_budget.py', 'backend/scripts/run_read_paper_batch.py')), 'LEDGER_OR_BATCH_DRIVER_CHANGED')
        return {**deepcopy(expected), 'anchor_manifest': deepcopy(original), 'latest_anchor_manifest': deepcopy(base['latest_anchor_manifest'])}
    except PermissionError:
        raise
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError, sqlite3.Error) as exc:
        raise PermissionError('FINAL_PROVIDER_UNKNOWN_MANIFEST_INVALID') from exc


def retained_provider_unknown_receipts(manifest, cell, result, view, retained, *, context=None):
    """仅当前V6权威合同可解释旧未知或继承预算停点，当前新格不得自授例外。"""
    c = _c(); supplied = manifest.get('completion_contract') or {}
    if supplied.get('version') != VERSION: return None
    validated = c.validate_completion(manifest)
    if context is not None: _require(c._equal(context, validated), 'UNVERIFIED_CURRENT_CONTEXT')
    context = validated
    _require(context.get('version') == VERSION and c._equal({k: v for k, v in context.items()
        if k not in {'anchor_manifest','latest_anchor_manifest'}}, supplied), 'CURRENT_CONTEXT_CHANGED')
    matches = [r for r in context['retained_rows'] if r.get('root_id') == result.get('root_id')
        and (r.get('provider_unknown_stop_verified') is True or r.get('root_budget_stop_verified') is True)]
    if not matches: return None
    _require(len(matches) == 1 and isinstance(retained, dict), 'EXACT_RETAINED_ROW_REQUIRED'); expected = matches[0]
    _require(all(c._equal(retained.get(k), v) for k, v in expected.items())
        and Path(cell) == Path(expected['provenance']['cell_root']) and view.get('task_id') == expected['root_id']
        and result.get('status') == 'executed' and result.get('task_status') == view.get('status') == expected['status'], 'RETAINED_IDENTITY_CHANGED')
    _proof_ref(supplied['provider_unknown_recovery_proof'])
    for name, digest in expected['provenance']['files'].items():
        _require(c._verified_hash(Path(cell) / name) == digest, 'RETAINED_DEPENDENCY_CHANGED')
    for name, value in (('result.json',result),('task-view.json',view)):
        _require(c._equal(c._helpers()._read(Path(cell) / name), value), 'RETAINED_BYTES_CHANGED')
    if expected.get('root_budget_stop_verified') is True:
        from agent_final_root_budget_recovery import STOP_REASON as prior_reason, _ERRORS as prior_errors
        _require(expected['status'] == 'partial' and result.get('ledger_halt') == prior_reason
            and expected.get('issues') == prior_errors and expected.get('original_reconciliation_passed') is False,
            'INHERITED_BUDGET_STOP_CHANGED')
        return {k: deepcopy(expected[k]) for k in ('physical_receipts_verified','native_status_counts','physical_status_counts','root_budget_stop_verified')}
    _require(expected['status'] == 'failed' and result.get('ledger_halt') == STOP_REASON
        and expected.get('issues') == _ERRORS and expected.get('original_reconciliation_passed') is False
        and expected.get('physical_receipts_verified') is False and expected.get('actual_usage_known') is False
        and expected.get('receipt_state_verified') is True, 'RETAINED_UNKNOWN_CHANGED')
    return {k: deepcopy(expected[k]) for k in ('physical_receipts_verified','receipt_state_verified','actual_usage_known',
        'provider_unknown_stop_verified','unconfirmed_request_count','native_status_counts','physical_status_counts')}


def _native_stop(stored, tasks, events, view, exported):
    """实际SQLite停止事件与导出一致，不能用控制器结束时间补造任务交付。"""
    rid = stored['task_id']; own = [r for r in events if r['task_id'] == rid]
    starts = [r for r in own if r['event_type'] == 'task_created']; stops = [r for r in own if r['event_type'] == 'task_finished']
    _require(len(starts) == len(stops) == 1 and stored['status'] == 'failed'
        and view.get('result', {}).get('answer_kind') == 'failure_report', 'SQL_UNIQUE_FAILED_STOP_REQUIRED')
    ids = {r['task_id'] for r in tasks}
    _require(len(ids) == len(tasks) and rid in ids
        and all(r.get('root_task_id') == rid and ((r['task_id'] == rid and r.get('parent_task_id') is None)
            or (r['task_id'] != rid and r.get('parent_task_id') in ids)) for r in tasks), 'TASK_FAMILY_OWNERSHIP_CHANGED')
    parents = {r['task_id']: r.get('parent_task_id') for r in tasks}
    for identity in ids:
        visited = set(); current = identity
        while current != rid:
            _require(current not in visited and current in parents, 'TASK_FAMILY_CYCLE_OR_DISCONNECTED')
            visited.add(current); current = parents[current]
    payload = json.loads(stops[0]['payload_json'])
    _require(payload.get('status') == 'failed' and _c()._equal(payload.get('result'), view.get('result')), 'SQL_STOP_EXPORT_CHANGED')
    at = lambda value: datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
    begin, end = at(starts[0]['created_at']), at(stops[0]['created_at'])
    exported = exported.get('events') if isinstance(exported, dict) else exported
    _require(_finite_time(begin) and _finite_time(end) and end >= begin
        and abs(at(stored['created_at']) - begin) < .01
        and all(r['status'] in {'completed','partial','failed','cancelled'} and at(r['updated_at']) <= end + .01 for r in tasks)
        and all(len([x for x in exported if x.get('task_id') == rid and x.get('event_type') == r['event_type']]) == 1
            and any(x.get('task_id') == rid and x.get('event_type') == r['event_type']
                and x.get('created_at') == r['created_at']
                and _c()._equal(x.get('payload'), json.loads(r['payload_json']))
                for x in exported) for r in [starts[0],stops[0]]), 'TASK_FAMILY_NOT_CLOSED_AT_STOP')


def _unknown_request(root, manifest, native, physical, proof, load):
    """分别核原生请求hash、HTTP紧凑body hash与归档文件hash；三者不能混用。"""
    c = _c(); call = next(r for r in native if r['status'] == 'unknown')
    row = next(r for r in physical if r['status'] == 'unknown'); rid = row['reservation_id']
    owner = next(r for r in load('runtime/batch-results.json')['rows'] if r['root_id'] == row['root_id'])
    relative = 'runtime/cells/' + owner['case_id'] + '-' + owner['variant'] + '/requests/' + rid
    wire = _binding(root, proof, load, relative + '.request.json')
    for suffix in ('.response.json','.response.sse','.response-meta.json','.error-body'):
        _require(not (root / (relative + suffix)).exists(), 'UNKNOWN_MUST_NOT_HAVE_COMPLETED_HTTP_RECEIPT')
    request = json.loads(call['request_json']); metadata = json.loads(row['request_json'])
    _require(set(request) == {'messages','tools','max_tokens','force_text','terminal_only'}
        and request['force_text'] is False and request['terminal_only'] is False
        and set(wire) == {'messages','tools','max_tokens','model','temperature','tool_choice'}
        and all(c._equal(request[k],wire[k]) for k in ('messages','tools','max_tokens'))
        and wire['model'] == manifest['settings']['deepseek_model'] == metadata['model']
        and wire['temperature'] == .1 and wire['tool_choice'] == 'auto'
        and type(wire['max_tokens']) is int and wire['max_tokens'] == row['reserved_output']
        and isinstance(wire['messages'],list) and isinstance(wire['tools'],list) and bool(wire['tools'])
        and metadata.get('stream') is False and metadata.get('request_kind') == 'text' and metadata.get('image_count') == 0,
        'NATIVE_WIRE_REQUEST_LOGICAL_BINDING_CHANGED')
    _require(call.get('request_hash') == hashlib.sha256(call['request_json'].encode()).hexdigest()
        and metadata.get('body_sha256') == hashlib.sha256(json.dumps(wire,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        'PRODUCER_SPECIFIC_REQUEST_HASH_CHANGED')
