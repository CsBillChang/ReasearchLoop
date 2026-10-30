"""V5根Token预留拒绝的只读续接；保留原始失败与全部历史费用。"""
from __future__ import annotations
from collections import Counter
import ast
import math
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import re
import sqlite3

VERSION = 'agent_final_completion_v5'
PROOF_VERSION = 'root_budget_stop_proof_v1'
STOP_REASON = 'token reservation budget exhausted'
BATCH_STOP = 'cell_receipt_or_isolation_gate'
SAFETY_PATH = 'audit/post-run-safety-supplement.json'
SAFETY_STATUS = 'passed_final_safety_with_original_missing_post_readback'
_PARSED_ANCHORS = {}
_ERRORS = ['ledger_halted', 'unexpected_native_not_sent']
_ACTIONS = {'real_model_calls', 'network_calls', 'Ledger_instances', 'Settings_instances', 'old_file_mutations', 'sqlite_mutations'}


def _c():
    import agent_final_completion as c
    return c


def _chain():
    import agent_final_completion_chain as chain
    return chain


def _require(value, reason):
    if not value:
        raise PermissionError('FINAL_ROOT_BUDGET_' + reason)


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


def _safety(root, load):
    """仅核漏建GitHub回读的精确旧误报，不接受任意补充报告。"""
    c = _c(); root = Path(root)
    paths = {'original_failed_report': 'audit/post-run-protection.json',
        'missing_receipt_diagnosis': 'audit/failures/post-audit-missing-github-readback.json',
        'corrected_full_audit': 'audit/post-run-protection-corrected.json',
        'post_run_github_readback': 'audit/post-run-github-readback.json',
        'audit_producer': 'private-protection/post_run_protection.py',
        'readonly_sql_report': 'audit/readonly-post-run.json', 'controller': 'formal-control/controller-result.json'}
    p = load(SAFETY_PATH)
    fixed = {'schema': 'root_budget_stop_safety_supplement_v1', 'status': SAFETY_STATUS, 'safety_closed': True,
        'measurement_complete': False, 'original_automatic_gate_passed': False, 'original_failed_report_unchanged': True,
        'old_failures_or_scores_rewritten': False, 'pending_not_reclassified_or_released': True,
        'checks_skipped_or_overridden': 0, 'model_calls': 0, 'service_mutations': 0, 'git_ref_mutations': 0}
    _require(all(c._equal(p.get(k), v) for k, v in fixed.items()), 'EXACT_SAFETY_SUPPLEMENT_REQUIRED')
    bindings = p.get('evidence_bindings'); values = {}
    _require(isinstance(bindings, dict) and set(bindings) == set(paths), 'SAFETY_BINDING_SET_CHANGED')
    for key, rel in paths.items():
        b = bindings[key]
        _require(isinstance(b, dict) and set(b) == {'path', 'sha256'} and b['path'] == str(root / rel)
                 and c._verified_hash(root / rel) == b['sha256'], 'SAFETY_BINDING_CHANGED')
        values[key] = load(rel, parse=key != 'audit_producer')
    failed = values['original_failed_report']; corrected = values['corrected_full_audit']; d = values['missing_receipt_diagnosis']
    _require(failed.get('status') == 'failed' and failed.get('safety_closed') is False
        and failed.get('measurement_complete') is False and failed.get('data_integrity_errors') == []
        and failed.get('errors') == [{'check': 'git_refs verified without exception', 'details': {'error_type': 'FileNotFoundError'}}]
        and _integer(failed.get('checks_total')) and failed['checks_total'] > 0
        and failed.get('checks_passed') == failed['checks_total'] - 1, 'ORIGINAL_FAILURE_NOT_EXACT')
    _require(d.get('status') == 'original_audit_failure_precise_path_verified'
        and d.get('original_audit_sha256') == bindings['original_failed_report']['sha256']
        and d.get('producer_sha256') == bindings['audit_producer']['sha256']
        and d.get('section') == 'git_refs' and d.get('exception_type') == 'FileNotFoundError'
        and d.get('missing_path') == str(root / paths['post_run_github_readback'])
        and d.get('all_preceding_local_git_checks_passed') is True and d.get('source_files_modified') is False
        and type(d.get('model_calls')) is int and d['model_calls'] == 0
        and type(d.get('network_calls')) is int and d['network_calls'] == 0, 'MISSING_RECEIPT_DIAGNOSIS_CHANGED')
    prov = corrected.get('correction_provenance') or {}
    expected = {bindings[k]['path']: bindings[k]['sha256'] for k in
        ('original_failed_report', 'missing_receipt_diagnosis', 'post_run_github_readback', 'audit_producer')}
    _require(corrected.get('status') == 'passed' and corrected.get('safety_closed') is True
        and corrected.get('measurement_complete') is False and corrected.get('errors') == corrected.get('data_integrity_errors') == []
        and _integer(corrected.get('checks_total')) and corrected['checks_total'] == failed['checks_total'] + 2
        and corrected.get('checks_passed') == corrected['checks_total']
        and prov.get('kind') == 'fresh_full_original_audit_after_missing_remote_receipt'
        and prov.get('original_audit_source_unchanged') is True and prov.get('original_failed_report_preserved') is True
        and c._equal(prov.get('input_sha256'), expected)
        and all(type(prov.get(k)) is int and prov[k] == 0 for k in
            ('checks_skipped_or_overridden', 'harness_business_or_prompt_changes', 'model_calls')), 'CORRECTED_FULL_ORIGINAL_AUDIT_REQUIRED')
    manifest = load('manifest.formal.json'); batch = load('runtime/batch-results.json'); remote = values['post_run_github_readback']
    scopes = {'closure', 'containers', 'dataset', 'git_refs', 'old_failure', 'physical_budget_readonly',
              'processes', 'protected', 'seals', 'selected_production_materials', 'sources'}
    _require(set(failed.get('sections', {})) == set(corrected.get('sections', {})) == scopes
        and all(report.get('checks_by_scope', {}).keys() == {'data', 'safety'}
            and all(_integer(v) for v in report['checks_by_scope'].values())
            and sum(report['checks_by_scope'].values()) == report['checks_total'] for report in (failed, corrected))
        and corrected['checks_by_scope']['data'] == failed['checks_by_scope']['data']
        and corrected['checks_by_scope']['safety'] == failed['checks_by_scope']['safety'] + 2
        and corrected['sections']['git_refs'].get('remote_readback_sha256') == bindings['post_run_github_readback']['sha256']
        and corrected['sections']['git_refs'].get('git_network_commands_executed_by_audit') == 0,
        'FULL_ORIGINAL_AUDIT_SCOPE_CHANGED')
    _require(remote.get('status') == 'passed' and isinstance(manifest.get('github_commit'), str)
        and re.fullmatch('[0-9a-f]{40}', manifest['github_commit']) and remote.get('commit') == remote.get('branch_sha') == manifest.get('github_commit')
        and isinstance(remote.get('tree'), str) and re.fullmatch('[0-9a-f]{40}', remote['tree'])
        and isinstance(remote.get('main_sha'), str) and re.fullmatch('[0-9a-f]{40}', remote['main_sha'])
        and remote.get('main_sha') == remote.get('tag_sha') and remote.get('main_merged') is False
        and remote.get('private_artifacts_uploaded') is False
        and type(remote.get('git_ref_mutations')) is int and remote['git_ref_mutations'] == 0, 'FRESH_GITHUB_READBACK_CHANGED')
    _require(values['readonly_sql_report'].get('status') == 'passed_selected_production_materials_unchanged'
        and values['controller'].get('cleanup_all_passed') is True and p.get('primary_recorded') == len(batch['rows'])
        and p.get('unstarted') == len(batch['not_created']), 'SAFETY_COVERAGE_OR_CLOSURE_CHANGED')
    for report in (failed, corrected):
        actions = report.get('actions') or {}
        _require(actions.get('only_new_report_written') is True
            and all(type(v) is int and v == 0 for k, v in actions.items() if k != 'only_new_report_written')
            and {'Ledger_instantiations', 'business_Settings_instantiations', 'model_calls',
                 'business_SQLite_writes', 'pending_releases', 'service_mutations'} <= set(actions), 'SAFETY_AUDIT_ACTIONS_CHANGED')
    return p


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
    """只解析原码，核冻结hash及拒绝发生在HTTP之前的关键路径。"""
    c = _c(); copies = []
    names = ('harness_ab_budget.py', 'harness_ab_transport.py', 'run_read_paper_batch.py')
    for variant in ('A', 'B'):
        expected = manifest['prompt_bundle_source_contract']['variants'][variant]['source_files']; hashes = {}
        for name in names:
            relative = 'snapshots/' + variant + '/backend/scripts/' + name
            _binding(root, proof, load, relative, parse=False)
            hashes[name] = c._verified_hash(root / relative)
            _require(expected.get('backend/scripts/' + name) == hashes[name], 'FROZEN_SOURCE_HASH_CHANGED')
        copies.append(hashes)
    _require(c._equal(copies[0], copies[1]), 'COMMON_GUARD_SOURCE_CHANGED')
    def function(tree, name):
        nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
        _require(len(nodes) == 1, 'UNIQUE_GUARD_FUNCTION_REQUIRED')
        return nodes[0]
    def called(node, target):
        return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == target]
    transport = ast.parse((root / 'snapshots/B/backend/scripts/harness_ab_transport.py').read_text())
    handler = function(transport, '_handle_request'); locations = {}
    for key, target in (('reserve', 'self.ledger.reserve'), ('mark', 'self.ledger.mark_sent'), ('http', 'self.inner.handle_request')):
        found = [(i, statement) for i, statement in enumerate(handler.body) if called(statement, target)]
        _require(len(found) == 1 and len(called(handler, target)) == 1 and isinstance(found[0][1], ast.Try), 'TRANSPORT_ADMISSION_SEQUENCE_CHANGED')
        locations[key] = found[0]
    _require(locations['reserve'][0] < locations['mark'][0] < locations['http'][0], 'HTTP_MUST_FOLLOW_RESERVATION_AND_SENT_MARK')
    reserve_try = locations['reserve'][1]; reserve_call = called(reserve_try, 'self.ledger.reserve')[0]
    _require(len(reserve_try.body) == 1 and len(reserve_try.handlers) == 1
        and ast.unparse(reserve_try.handlers[0].type) == 'Exception' and len(reserve_try.handlers[0].body) == 1
        and isinstance(reserve_try.handlers[0].body[0], ast.Raise)
        and isinstance(reserve_try.handlers[0].body[0].exc, ast.Call)
        and ast.unparse(reserve_try.handlers[0].body[0].exc.func) == 'RequestNotSent'
        and any(isinstance(n, ast.Constant) and n.value == 'H1 budget denied before transport: '
            for n in ast.walk(reserve_try.handlers[0]))
        and any(k.arg == 'input_upper_bound' and isinstance(k.value, ast.Constant) and k.value.value == 2 ** 20
            for k in reserve_call.keywords), 'PRE_TRANSPORT_BUDGET_REFUSAL_CHANGED')
    budget = ast.parse((root / 'snapshots/B/backend/scripts/harness_ab_budget.py').read_text()); reserve = function(budget, 'reserve')
    tests = {
        "total['requests'] + 1 > self.config['batch_requests'] or family['requests'] + 1 > self.config['root_requests']": 'request budget exhausted',
        "total['tokens'] + tokens > self.config['batch_tokens'] or family['tokens'] + tokens > self.config['root_tokens']": STOP_REASON,
        "total['nano'] + nano > self.config['batch_nano_yuan']": 'money reservation budget exhausted'}
    for expression, reason in tests.items():
        expected = ast.dump(ast.parse(expression, mode='eval').body)
        matches = [n for n in ast.walk(reserve) if isinstance(n, ast.If) and ast.dump(n.test) == expected]
        _require(len(matches) == 1 and len(matches[0].body) == 1 and isinstance(matches[0].body[0], ast.Raise)
            and isinstance(matches[0].body[0].exc, ast.Call) and ast.unparse(matches[0].body[0].exc.func) == 'BudgetHalt'
            and len(matches[0].body[0].exc.args) == 1 and isinstance(matches[0].body[0].exc.args[0], ast.Constant)
            and matches[0].body[0].exc.args[0].value == reason, 'ORIGINAL_LEDGER_RESERVATION_GUARD_CHANGED')
    constructor = function(budget, '__init__')
    assignments = [n for n in ast.walk(constructor) if isinstance(n, ast.Assign)
        and any(ast.unparse(target) == 'self.config' for target in n.targets) and isinstance(n.value, ast.Dict)]
    _require(len(assignments) == 1, 'FROZEN_RATE_SOURCE_REQUIRED')
    rate_names = {'input_nano_per_token', 'cached_input_nano_per_token', 'output_nano_per_token'}
    rates = {key.value: ast.literal_eval(value) for key, value in zip(assignments[0].value.keys, assignments[0].value.values)
             if isinstance(key, ast.Constant) and key.value in rate_names}
    final_config = _binding(root, proof, load, 'runtime/ledger-final.json').get('limits') or {}
    _require(set(rates) == rate_names and all(_integer(v) and v > 0 for v in rates.values())
        and c._equal(rates, {k: final_config.get(k) for k in rate_names}), 'LEDGER_SOURCE_RATE_CONFIG_CHANGED')
    admission = function(budget, '_admission'); check_open = function(budget, '_check_open')
    _require(called(admission, 'self._halt') and any(isinstance(n, ast.ExceptHandler)
        and isinstance(n.type, ast.Name) and n.type.id == 'BudgetHalt' for n in ast.walk(admission))
        and any(isinstance(n, ast.Constant) and n.value == 'batch halted: ' for n in ast.walk(check_open)), 'PERSISTENT_FIRST_ERROR_GUARD_CHANGED')


def _budget_evidence(ledger, root_id):
    """按首错时刻重放在途预留，不以最终actual替代当时占用。"""
    c = _c(); meta = ledger['metadata']; error = meta.get('first_error') or {}; config = meta['config']
    _require(meta.get('halt_reason') == error.get('reason') == STOP_REASON and error.get('operation') == 'reserve'
        and error.get('root_id') == root_id and error.get('reservation_id') is None
        and c._equal(error.get('limits'), config), 'EXACT_ROOT_RESERVE_HALT_REQUIRED')
    candidate = (error.get('details') or {}).get('candidate')
    _require(isinstance(candidate, dict) and set(candidate) == {'reserved_input', 'reserved_output', 'reserved_tokens', 'reserved_nano_yuan'}
        and all(_integer(v) and v > 0 for v in candidate.values()) and candidate['reserved_input'] == 2 ** 20
        and candidate['reserved_output'] <= config['max_output_tokens']
        and candidate['reserved_tokens'] == candidate['reserved_input'] + candidate['reserved_output']
        and candidate['reserved_nano_yuan'] == candidate['reserved_input'] * config['input_nano_per_token']
            + candidate['reserved_output'] * config['output_nano_per_token'], 'FROZEN_RESERVATION_CANDIDATE_REQUIRED')
    rows = ledger['reservations']; at = error.get('detected_at')
    root_map = {r['root_id']: r for r in ledger['roots']}
    _require(len(root_map) == len(ledger['roots']) and _finite_time(at)
        and all(_finite_time(r.get('started_at')) and r['started_at'] <= at for r in ledger['roots']), 'ROOT_TIME_INVALID')
    _require(_finite_time(at) and all(r['status'] == 'settled' and r.get('sent_at') is not None
        and all(_integer(r.get(k)) for k in ('actual_input', 'actual_output', 'actual_tokens', 'actual_nano', 'reserved_tokens', 'reserved_nano'))
        and r['actual_tokens'] == r['actual_input'] + r['actual_output']
        and all(_finite_time(r.get(k)) for k in ('created_at', 'sent_at', 'finished_at'))
        and r['root_id'] in root_map and root_map[r['root_id']]['started_at'] <= r['created_at'] <= r['sent_at'] <= r['finished_at'] for r in rows), 'NEW_PENDING_OR_INVALID_SETTLEMENT')
    _require(len({r['reservation_id'] for r in rows}) == len(rows) and all(r['created_at'] <= at and r['sent_at'] <= at for r in rows), 'POST_HALT_RESERVATION_OR_SEND_OR_DUPLICATE')
    for row in rows:
        _require(all(_integer(row.get(k)) and row[k] > 0 for k in ('reserved_input', 'reserved_output', 'reserved_tokens', 'reserved_nano'))
            and row['reserved_input'] == 2 ** 20 and row['reserved_output'] <= config['max_output_tokens']
            and row['reserved_tokens'] == row['reserved_input'] + row['reserved_output']
            and row['reserved_nano'] == row['reserved_input'] * config['input_nano_per_token']
                + row['reserved_output'] * config['output_nano_per_token'], 'PHYSICAL_RESERVATION_BOUNDS_CHANGED')
        _frozen_rate(row, config)
    roots = [r for r in ledger['roots'] if r['root_id'] == root_id]
    _require(len(roots) == 1 and 0 <= at - roots[0]['started_at'] < config['root_seconds'], 'ROOT_TIME_NOT_AVAILABLE')
    root = _totals([r for r in rows if r['root_id'] == root_id], at); batch = _totals(rows, at)
    for k, src in (('tokens', 'tokens'), ('requests', 'requests'), ('nano', 'nano_yuan')):
        batch[k] += config['carried_usage'][src]
    _require(c._equal(root, error.get('root_occupancy')) and c._equal(batch, error.get('batch_occupancy')), 'FIRST_ERROR_OCCUPANCY_REPLAY_CHANGED')
    _require(root['tokens'] + candidate['reserved_tokens'] > config['root_tokens']
        and batch['tokens'] + candidate['reserved_tokens'] <= config['batch_tokens']
        and batch['nano'] + candidate['reserved_nano_yuan'] <= config['batch_nano_yuan']
        and batch['requests'] + 1 <= config['batch_requests'] and root['requests'] + 1 <= config['root_requests'], 'ROOT_TOKEN_ONLY_REFUSAL_REQUIRED')
    return {'first_error_sha256': c._digest(error), 'root_occupancy': root, 'batch_occupancy': batch,
            'candidate': candidate, 'final_root': _totals([r for r in rows if r['root_id'] == root_id])}


def _receipts(root_id, native, physical, config=None):
    """仅分离确定的发送前预算拒绝；成功收据仍严格一一核对。"""
    c = _c(); success = []; denied = []; ids = set()
    _require(isinstance(native, list) and native, 'NATIVE_RECORDS_REQUIRED')
    for call in native:
        identity = call.get('call_id')
        _require(isinstance(identity, str) and identity and identity not in ids, 'UNIQUE_NATIVE_CALL_ID_REQUIRED'); ids.add(identity)
        if call.get('status') == 'succeeded':
            success.append(call); continue
        response = json.loads(call.get('response_json', 'null')); usage = json.loads(call.get('usage_json', 'null'))
        _require(call.get('status') == call.get('sent_state') == 'not_sent' and call.get('request_id') is None and usage == {}
            and isinstance(response, dict) and set(response) == {'error', 'error_type'} and response['error_type'] == 'RequestNotSent'
            and response['error'] in {'H1 budget denied before transport: BudgetHalt: ' + STOP_REASON,
                'H1 budget denied before transport: BudgetHalt: batch halted: ' + STOP_REASON}, 'CONFIRMED_BUDGET_NOT_SENT_REQUIRED')
        denied.append(call)
    _require(denied and any(json.loads(x['response_json'])['error'] == 'H1 budget denied before transport: BudgetHalt: ' + STOP_REASON
                            for x in denied), 'FIRST_UNSENT_REFUSAL_REQUIRED')
    proofs = c._receipts(root_id, success, physical)
    if config is not None:
        for row in physical:
            _frozen_rate(row, config)
    return {**proofs, 'native_status_counts': dict(Counter(r['status'] for r in native))}


def _binding(root, proof, load, relative, *, parse=True):
    c = _c(); value = load(relative, parse)
    _require(proof.get('bindings', {}).get(relative) == c._verified_hash(root / relative), 'COMPLETE_PROOF_BINDING_REQUIRED')
    return value


def _budget_row(root, manifest, position, ledger, proof, load, raw):
    """保留原partial及非clean问题，不把预算停点作为额外技术attempt。"""
    c = _c(); case, variant = manifest['execution_order'][position]; rel = 'runtime/cells/' + case + '-' + variant; cell = root / rel
    def get(name, parse=True):
        return _binding(root, proof, load, rel + '/' + name, parse=parse)
    result = get('result.json'); view = get('task-view.json'); native = get('model-calls.json'); actions = get('actions.json')
    events = get('events.json'); stored = get('root-row.json'); snapshot = get('ledger-snapshot.json'); get('task.sqlite3', False)
    rid = raw['root_id']
    _require(result.get('status') == 'executed' and result.get('ledger_halt') == STOP_REASON and result.get('case_id') == case
        and result.get('variant') == variant and result.get('root_id') == view.get('task_id') == rid
        and result.get('task_status') == view.get('status') == raw.get('status') == 'partial'
        and c._equal(result.get('loaded_supervisor'), manifest['loaded_prompt_contract'][variant]), 'ORIGINAL_PARTIAL_IDENTITY_REQUIRED')
    prompt = next(x['prompt'] for x in load('inputs/cases.json')['cases'] if x['case_id'] == case)
    _require(view.get('prompt') == stored.get('prompt') == prompt and stored.get('task_id') == rid
        and stored.get('status') == 'partial' and stored.get('parent_task_id') is None
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
        family = {x[0] for x in db.execute('SELECT task_id FROM tasks WHERE task_id=? OR root_task_id=?', (rid, rid))}
    _require(len(sql_root) == 1 and all(sql_root[0].get(k) == stored.get(k) for k in ('task_id', 'status', 'prompt', 'result_json', 'parent_task_id'))
        and all(x.get('task_id') in family for x in native)
        and c._equal(sorted(sql_native, key=lambda x: x['call_id']), sorted(native, key=lambda x: x['call_id'])), 'NATIVE_SQL_EXPORT_CHANGED')
    physical = [r for r in ledger['reservations'] if r['root_id'] == rid]
    _require(any(r['root_id'] == rid and r['case_id'] == case and r['variant'] == variant for r in ledger['roots'])
        and c._equal([r for r in snapshot['reservations'] if r['root_id'] == rid], physical), 'CELL_LEDGER_SNAPSHOT_CHANGED')
    receipts = _receipts(rid, native, physical, ledger['metadata']['config'])
    _archives(root, case, variant, physical, lambda relative, parse=True: _binding(root, proof, load, relative, parse=parse))
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
    return {**deepcopy(raw), 'retained': True, 'quality': None, **receipts,
        'source_manifest_path': str(root / 'manifest.formal.json'), 'source_manifest_sha256': c._verified_hash(root / 'manifest.formal.json'),
        'ledger_root': str(root), 'ledger_path': str(root / 'runtime/budget.sqlite3'), 'observed_status': 'partial', 'observation_stop': observation,
        'source_batch_record': deepcopy(raw), 'original_reconciliation_issues': deepcopy(_ERRORS), 'original_reconciliation_passed': False,
        'root_budget_stop_verified': True, 'artifact_owner': 'original_sealed_manifest', 'score_owner': 'current_manifest', 'deferred_review': True,
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
    manifest = load('manifest.formal.json'); parent = chain._parent(manifest, {*seen, str(root)})
    _require(root == Path(manifest['experiment_root']) and Path(manifest['runtime_root']) == root / 'runtime'
        and str(root) not in parent['closed_ledger_roots'], 'NEW_CLOSED_ROOT_REQUIRED')
    _require(not any(Path(proof_ref['path']).is_relative_to(Path(old)) for old in [*parent['closed_ledger_roots'], str(root)]), 'PROOF_MUST_BE_OUTSIDE_OLD_ROOTS')
    proof = _proof_ref(proof_ref)
    _require(proof.get('version') == PROOF_VERSION and proof.get('status') == 'passed_exact_root_token_reservation_stop'
        and proof.get('anchor_root') == str(root) and proof.get('anchor_seal_sha256') == seal_sha
        and proof.get('parent_contract_sha256') == c._digest(manifest['completion_contract'])
        and proof.get('original_records_unchanged') is True and proof.get('original_reconciliation_passed') is False
        and proof.get('no_retry_or_reclassification') is True, 'EXACT_ROOT_PROOF_REQUIRED')
    _require(isinstance(proof.get('actions'), dict) and set(proof['actions']) == _ACTIONS
        and all(type(v) is int and v == 0 for v in proof['actions'].values()), 'ZERO_MUTATION_PROOF_REQUIRED')
    _require(isinstance(proof.get('bindings'), dict) and proof['bindings'], 'PROOF_BINDINGS_REQUIRED')
    required = {'manifest.formal.json', 'inputs/cases.json', 'runtime/budget.sqlite3', 'runtime/ledger-final.json',
        SAFETY_PATH, 'audit/post-run-protection.json', 'audit/post-run-protection-corrected.json',
        'audit/failures/post-audit-missing-github-readback.json', 'audit/post-run-github-readback.json',
        'private-protection/post_run_protection.py', 'formal-execution-auth.json',
        'formal-preflight/authorization-closed.json', 'formal-preflight/vector-close.json',
        'formal-control/controller-result.json', 'audit/readonly-post-run.json'}
    for variant in ('A', 'B'):
        required.update('snapshots/' + variant + '/backend/scripts/' + name for name in
            ('harness_ab_budget.py', 'harness_ab_transport.py', 'run_read_paper_batch.py'))
    _require(required <= set(proof['bindings']), 'COMPLETE_ANCHOR_BINDINGS_REQUIRED')
    for relative, digest in proof['bindings'].items():
        _require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'PROOF_BINDING_SHAPE')
        _binding(root, proof, load, relative, parse=False)
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
    _require(proof.get('case_id') == case and proof.get('position') == last and proof.get('root_id') == rid
        and proof.get('original_batch_stop_reason') == BATCH_STOP and proof.get('original_ledger_halt_reason') == STOP_REASON
        and proof.get('original_reconciliation_issues') == _ERRORS and raw.get('issues') == _ERRORS
        and raw.get('returncode') == 0, 'ORIGINAL_STOP_FIELDS_CHANGED')
    _require(c._equal(_budget_evidence(ledger, rid), proof.get('root_budget_evidence')), 'ROOT_BUDGET_EVIDENCE_CHANGED')
    report = _binding(root, proof, load, 'runtime/' + case + '-' + variant + '-reconciliation.json')
    _require(report.get('issues') == _ERRORS and report.get('ledger_halt_reason') == STOP_REASON
        and c._equal(report.get('first_error'), ledger['metadata']['first_error'])
        and report.get('unmatched_native_response_ids') == report.get('unmatched_physical_response_ids') == []
        and all(x.get('classification') == 'confirmed_not_sent' for x in report.get('native_not_sent', [])), 'ORIGINAL_RECONCILIATION_CHANGED')
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
            enriched = _budget_row(root, manifest, position, ledger, proof, load, row)
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
    _require(not pending and not {r['reservation_id'] for r in ledger['reservations']} & {
        r['reservation_id'] for r in parent['pending_reservations']}, 'NEW_PENDING_OR_OLD_REQUEST_REUSED')
    carry = {k: parent['carried_usage'][k] + charge[k] for k in charge}; carry['seconds'] = parent['carried_usage']['seconds']
    final = _binding(root, proof, load, 'runtime/ledger-final.json')
    _require(c._equal(final.get('limits'), ledger['metadata']['config'])
        and c._equal(final.get('carried_usage'), parent['carried_usage']) and final.get('actual_known_tokens') == charge['tokens']
        and final.get('physical_sent_requests') == charge['requests'] and final.get('actual_known_estimated_nano_yuan') == charge['nano_yuan']
        and final.get('halt_reason') == STOP_REASON and c._equal(final.get('first_error'), ledger['metadata']['first_error'])
        and c._equal(final.get('charged_or_reserved'), {'tokens': carry['tokens'], 'requests': carry['requests'], 'nano': carry['nano_yuan']}), 'ORIGINAL_LEDGER_EXPORT_CHANGED')
    lineage = {'anchor_root': str(root), 'proof_ref': deepcopy(proof_ref), 'root_id': rid, 'position': last}; original = parent['anchor_manifest']
    value = {**deepcopy(parent), 'anchor_root': str(root), 'anchors': anchors,
        'anchor_seals': {**parent['anchor_seals'], str(root): seal_sha}, 'retained_rows': retained, 'retained_count': len(retained),
        'new_execution_order': deepcopy(order[len(retained):]), 'technical_attempts': deepcopy(parent['technical_attempts']), 'carried_usage': carry,
        'closed_ledger_roots': [*parent['closed_ledger_roots'], str(root)], 'pending_reservations': deepcopy(parent['pending_reservations']),
        'latest_anchor_manifest': manifest, 'parent_contract_sha256': c._digest(manifest['completion_contract']), 'leaf_charged_or_reserved': charge,
        'minimum_complete_attempts': len(order) + len(parent['technical_attempts']), 'deferred_review_cases': [case], 'retry_positions': [],
        'root_budget_recovery_proof': deepcopy(proof_ref),
        'root_budget_recovery_lineage': [*deepcopy(parent.get('root_budget_recovery_lineage', [])), lineage],
        'old_cases': c._helpers()._read(original['cases_path']),
        'old_materials': c._helpers()._read(Path(original['prepared_input_root']) / 'material-snapshot.json')}
    _PARSED_ANCHORS[key] = deepcopy(value)
    return value


def build_root_budget_recovery_contract(anchor_root, timing_authorization, *, proof_ref, _seen=frozenset()):
    """只派生保留格、待评计划与累计财务，不生成新授权。"""
    try:
        base = _anchor(anchor_root, proof_ref, _seen); timing, clock = _c()._timing(timing_authorization, base)
        excluded = {'anchor_manifest', 'latest_anchor_manifest', 'old_cases', 'old_materials', 'clock', 'timing_authorization', 'version', 'retry_policy'}
        return {**{k: deepcopy(v) for k, v in base.items() if k not in excluded}, 'version': VERSION, 'clock': clock, 'timing_authorization': timing,
            'retry_policy': {'retry_positions': [], 'preserve_all_old_technical_attempts': True,
                'never_resend_old_unknown_requests': True, 'never_retry_retained_primary_observations': True}}
    except PermissionError:
        raise
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError, sqlite3.Error) as exc:
        raise PermissionError('FINAL_ROOT_BUDGET_ANCHOR_INVALID') from exc


def validate_root_budget_recovery(manifest, *, _seen=frozenset()):
    """每次核当前manifest，缓存只承接不变的旧封存字节。"""
    c = _c()
    try:
        supplied = manifest['completion_contract']
        _require('continuation_contract' not in manifest and supplied.get('version') == VERSION, 'EXPLICIT_V5_REQUIRED')
        expected = build_root_budget_recovery_contract(supplied['anchor_root'], supplied['timing_authorization'], proof_ref=supplied['root_budget_recovery_proof'], _seen=_seen)
        _require(c._equal(supplied, expected), 'CONTRACT_OR_PROOF_CHANGED')
        base = _anchor(supplied['anchor_root'], supplied['root_budget_recovery_proof'], _seen); original = base['anchor_manifest']
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
        raise PermissionError('FINAL_ROOT_BUDGET_MANIFEST_INVALID') from exc


def retained_budget_stop_receipts(manifest, cell, result, view, retained, *, context=None):
    """仅当前已验证V5预算保留格可承接旧非clean收据。"""
    c = _c(); supplied = manifest.get('completion_contract') or {}
    if supplied.get('version') != VERSION:
        return None
    validated = c.validate_completion(manifest)
    if context is not None:
        _require(c._equal(context, validated), 'UNVERIFIED_CURRENT_CONTEXT')
    context = validated
    _require(context.get('version') == VERSION and c._equal({k: v for k, v in context.items()
        if k not in {'anchor_manifest', 'latest_anchor_manifest'}}, supplied), 'CURRENT_CONTEXT_CHANGED')
    candidates = [r for r in context['retained_rows'] if r.get('root_budget_stop_verified') is True and r['root_id'] == result.get('root_id')]
    if not candidates:
        return None
    _require(len(candidates) == 1 and isinstance(retained, dict), 'EXACT_RETAINED_BUDGET_ROW_REQUIRED'); expected = candidates[0]
    _require(all(c._equal(retained.get(k), v) for k, v in expected.items()) and Path(cell) == Path(expected['provenance']['cell_root'])
        and result.get('status') == 'executed' and result.get('task_status') == view.get('status') == 'partial'
        and view.get('task_id') == expected['root_id'] and result.get('ledger_halt') == STOP_REASON
        and expected.get('issues') == _ERRORS and expected.get('original_reconciliation_passed') is False, 'RETAINED_BUDGET_IDENTITY_CHANGED')
    _proof_ref(supplied['root_budget_recovery_proof'])
    for name, digest in expected['provenance']['files'].items():
        _require(c._verified_hash(Path(cell) / name) == digest, 'RETAINED_BUDGET_DEPENDENCY_CHANGED')
    for name, value in (('result.json', result), ('task-view.json', view)):
        path = Path(cell) / name
        _require(c._verified_hash(path) == expected['provenance']['files'][name] and c._equal(c._helpers()._read(path), value), 'RETAINED_BUDGET_BYTES_CHANGED')
    return {k: deepcopy(expected[k]) for k in ('physical_receipts_verified', 'native_status_counts', 'physical_status_counts', 'root_budget_stop_verified')}
