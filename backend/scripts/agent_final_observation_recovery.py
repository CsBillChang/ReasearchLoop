"""精确测量门暂停的V4恢复；保留停点原状，在新批次先补独立源审。"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
import sqlite3

VERSION = 'agent_final_completion_v4'
PROOF_PATH = 'audit/measurement-gate-stop-proof.json'
STOP_REASON = 'semantic_pair_gate_invalid:PermissionError'
OBSERVATION_POLICY = {'version': 'agent_observation_stop_v1',
    'statuses': ['completed', 'partial', 'failed', 'cancelled', 'needs_input'],
    'non_delivery_statuses': ['failed', 'cancelled', 'needs_input'],
    'require_root_task_finished_event': True, 'non_delivery_execution_complete': False,
    'allow_absent_answer_only_for_non_delivery': True}
_PARSED_ANCHORS = {}


def _c():
    import agent_final_completion as completion
    return completion


def _chain():
    import agent_final_completion_chain as chain
    return chain


def _require(value, reason):
    if not value:
        raise PermissionError('FINAL_OBSERVATION_RECOVERY_' + reason)


def validate_observation_policy(manifest):
    """核固定停点协议；继承权威另由chain的封存派生合同等值校验。"""
    contract = manifest.get('completion_contract')
    v4 = isinstance(contract, dict) and contract.get('version') == VERSION
    v5 = isinstance(contract, dict) and contract.get('version') == 'agent_final_completion_v5'
    v6 = isinstance(contract, dict) and contract.get('version') == 'agent_final_completion_v6'
    lineage_present = isinstance(contract, dict) and 'observation_recovery_lineage' in contract
    if 'observation_policy' not in manifest and not v4 and not v5 and not v6 and not lineage_present:
        return False
    inherited = lineage_present and contract.get('version') == 'agent_final_completion_v3'
    if lineage_present:
        lineage = contract['observation_recovery_lineage']
        _require(isinstance(lineage, list) and lineage, 'RECOVERY_LINEAGE_SHAPE_REQUIRED')
        roots = set()
        for item in lineage:
            _require(isinstance(item, dict) and set(item) == {
                'anchor_root', 'proof_path', 'proof_sha256', 'case_id'}
                and all(isinstance(v, str) and v for v in item.values()), 'RECOVERY_LINEAGE_SHAPE_REQUIRED')
            root = Path(item['anchor_root'])
            _require(root.is_absolute() and str(root) == item['anchor_root'] and '..' not in root.parts
                     and item['proof_path'] == str(root / PROOF_PATH)
                     and re.fullmatch('[0-9a-f]{64}', item['proof_sha256'])
                     and item['anchor_root'] not in roots, 'RECOVERY_LINEAGE_IDENTITY_REQUIRED')
            roots.add(item['anchor_root'])
    if v5 or v6:
        proof = contract.get('root_budget_recovery_proof')
        _require(lineage_present and isinstance(contract.get('root_budget_recovery_lineage'), list)
                 and contract['root_budget_recovery_lineage'] and isinstance(proof, dict)
                 and set(proof) == {'version', 'path', 'sha256'}
                 and proof['version'] == 'root_budget_stop_proof_v1'
                 and isinstance(proof['path'], str) and Path(proof['path']).is_absolute()
                 and isinstance(proof['sha256'], str) and re.fullmatch('[0-9a-f]{64}', proof['sha256']),
                 'EXPLICIT_V5_ROOT_BUDGET_PROOF_REQUIRED')
    if v6:
        proof = contract.get('provider_unknown_recovery_proof')
        lineage = contract.get('provider_unknown_recovery_lineage')
        _require(isinstance(proof, dict) and set(proof) == {'version','path','sha256'}
            and proof['version'] == 'provider_unknown_stop_proof_v1'
            and isinstance(proof['path'], str) and Path(proof['path']).is_absolute()
            and isinstance(proof['sha256'], str) and re.fullmatch('[0-9a-f]{64}', proof['sha256'])
            and isinstance(lineage,list) and bool(lineage), 'EXPLICIT_V6_PROVIDER_UNKNOWN_PROOF_REQUIRED')
        roots = set()
        for item in lineage:
            _require(isinstance(item,dict) and set(item) == {'anchor_root','proof_ref','root_id','position'}
                and isinstance(item['anchor_root'],str) and Path(item['anchor_root']).is_absolute()
                and item['anchor_root'] not in roots and isinstance(item['root_id'],str) and bool(item['root_id'])
                and type(item['position']) is int and 0 <= item['position'] < 96
                and isinstance(item['proof_ref'],dict) and set(item['proof_ref']) == {'version','path','sha256'}
                and item['proof_ref']['version'] == 'provider_unknown_stop_proof_v1'
                and isinstance(item['proof_ref']['path'],str) and Path(item['proof_ref']['path']).is_absolute()
                and isinstance(item['proof_ref']['sha256'],str) and re.fullmatch('[0-9a-f]{64}',item['proof_ref']['sha256']),
                'PROVIDER_UNKNOWN_LINEAGE_CHANGED')
            roots.add(item['anchor_root'])
    _require((v4 or inherited or v5 or v6) and _c()._equal(manifest.get('observation_policy'), OBSERVATION_POLICY),
             'EXACT_V4_OBSERVATION_POLICY_REQUIRED')
    return True


def exported_stop(view, events):
    """核对唯一原生停止事件，不把needs_input改写为后台完成状态。"""
    c = _c(); root_id = view.get('task_id'); status = view.get('status')
    rows = events.get('events') if isinstance(events, dict) else events
    _require(isinstance(root_id, str) and root_id and status in OBSERVATION_POLICY['statuses']
             and isinstance(rows, list), 'EXPORTED_STOP_SHAPE')
    if isinstance(events, dict):
        _require(events.get('task_id') == root_id, 'EVENT_ROOT_CHANGED')
    starts = [x for x in rows if x.get('task_id') == root_id and x.get('event_type') == 'task_created']
    stops = [x for x in rows if x.get('task_id') == root_id and x.get('event_type') == 'task_finished']
    _require(len(starts) == len(stops) == 1, 'UNIQUE_ROOT_STOP_REQUIRED')
    try:
        times = [datetime.fromisoformat(x['created_at'].replace('Z', '+00:00')).timestamp()
                 for x in (starts[0], stops[0])]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PermissionError('FINAL_OBSERVATION_RECOVERY_STOP_TIME_INVALID') from exc
    _require(all(c._helpers()._finite(x) for x in times) and times[0] <= times[1], 'STOP_TIME_INVALID')
    payload = stops[0].get('payload')
    _require(isinstance(payload, dict) and payload.get('status') == status
             and c._equal(payload.get('result'), view.get('result')), 'STOP_VIEW_CHANGED')
    return {'version': 'agent_observation_stop_v1', 'observed_status': status,
            'task_created_event_sha256': c._digest(starts[0]),
            'task_finished_event_sha256': c._digest(stops[0]),
            'task_finished_at': stops[0]['created_at'],
            'delivery_complete': status == 'completed', 'backend_status_preserved': True}


def _proof(root, manifest, parent, load):
    """证明仅修复精确的旧状态门误拒，不泛化接受任意PermissionError。"""
    c = _c(); first = parent['retained_count']; order = parent['full_execution_order']
    case = order[first][0]
    _require(first == 12 and len(parent['technical_attempts']) == 3 and len(order) == 96
             and order[first:first + 2] == [[case, 'A'], [case, 'B']], 'EXACT_PARENT_GEOMETRY_REQUIRED')
    proof = load(PROOF_PATH)
    fixed = {'version': 'measurement_gate_stop_v1', 'status': 'passed_exact_legacy_pair_identity_replay',
        'original_stop_reason': STOP_REASON, 'exception_type': 'PermissionError',
        'exception_message': 'PAIR_ENDED_CELL_IDENTITY_INVALID', 'function': 'pair_identity',
        'case_id': case, 'positions': [first, first + 1],
        'backend_statuses': {'A': 'partial', 'B': 'needs_input'},
        'original_records_unchanged': True, 'new_challenge_created': False,
        'new_score_created': False, 'original_gate_passed': False}
    _require(all(c._equal(proof.get(k), v) for k, v in fixed.items()), 'EXACT_REPLAY_PROOF_REQUIRED')
    actions = proof.get('actions')
    _require(isinstance(actions, dict) and set(actions) == {
        'real_model_calls', 'network_calls', 'Ledger_instances', 'sqlite_mutations', 'old_file_mutations'}
        and all(type(v) is int and v == 0 for v in actions.values()), 'REPLAY_MUST_BE_ZERO_MUTATION')
    required = {'manifest.formal.json', 'inputs/cases.json', 'runtime/batch-results.json',
        'runtime/ledger-final.json', 'runtime/budget.sqlite3',
        'snapshots/B/backend/scripts/harness_ab_pair_gate.py', 'semantic-gates/session.json',
        *(f'semantic-gates/grants/{n}.json' for n in (first, first + 1))}
    for variant in ('A', 'B'):
        required.update('runtime/cells/' + case + '-' + variant + '/' + name for name in
            (*c._helpers()._CELL_FILES, 'events.json', 'task.sqlite3', 'root-row.json', 'ledger-snapshot.json'))
    bindings = proof.get('bindings')
    _require(isinstance(bindings, dict) and required <= set(bindings), 'COMPLETE_REPLAY_BINDINGS_REQUIRED')
    for relative, digest in bindings.items():
        _require(isinstance(relative, str) and isinstance(digest, str)
                 and re.fullmatch('[0-9a-f]{64}', digest), 'REPLAY_BINDING_INVALID')
        load(relative, False)
        _require(c._verified_hash(c._helpers()._inside(root, relative)) == digest, 'REPLAY_BINDING_CHANGED')
    _require(not (root / 'semantic-gates' / case).exists(), 'ORIGINAL_PACKET_OR_GRADE_MUST_REMAIN_ABSENT')
    return proof, case


def _archives(root, case, variant, physical, load):
    """实际响应归档必须包含同一远端ID和usage，SSE必须有DONE。"""
    c = _c()
    for row in physical:
        base = 'runtime/cells/' + case + '-' + variant + '/requests/' + row['reservation_id']
        if (root / (base + '.response.json')).is_file():
            response = load(base + '.response.json'); identity = response.get('id'); usage = response.get('usage')
        else:
            relative = base + '.response.sse'; load(relative, False)
            identity = usage = None; done = False
            for line in (root / relative).read_text().splitlines():
                if not line.startswith('data:'):
                    continue
                text = line[5:].strip()
                if text == '[DONE]':
                    done = True; continue
                chunk = json.loads(text)
                if chunk.get('id'):
                    _require(identity in (None, chunk['id']), 'ARCHIVE_RESPONSE_ID_CHANGED')
                    identity = chunk['id']
                if chunk.get('usage') is not None:
                    usage = chunk['usage']
            _require(done, 'COMPLETE_SSE_ARCHIVE_REQUIRED')
        expected = json.loads(row['response_json'])
        _require(identity == expected.get('response_id') and c._equal(c._usage(usage), c._usage(row['usage_json'])),
                 'ARCHIVE_NATIVE_PHYSICAL_USAGE_CHANGED')


def _anchor(anchor_root, seen):
    return _c()._cached_anchor(VERSION, anchor_root, seen, lambda: _anchor_uncached(anchor_root, seen))


def _anchor_uncached(anchor_root, seen):
    c = _c(); chain = _chain(); root = Path(anchor_root)
    _require((root / 'SEAL.json').is_file(), 'ANCHOR_PENDING_SEAL')
    _require(str(root) not in seen, 'ANCHOR_CYCLE')
    seen = {*seen, str(root)}
    root, seal, seal_sha = c._sealed(root); anchors = {'SEAL.json': seal_sha}
    load = c._loader(root, seal, anchors); manifest = load('manifest.formal.json')
    _require(Path(manifest['experiment_root']) == root and Path(manifest['runtime_root']) == root / 'runtime',
             'EXACT_CLOSED_ROOT_REQUIRED')
    parent = chain._parent(manifest, seen)
    _require(str(root) not in parent['closed_ledger_roots'], 'DUPLICATE_CLOSED_ROOT')
    chain._closure(root, manifest, load)
    proof, case = _proof(root, manifest, parent, load)
    # 热校验仍核整棵seal、父合同、闭合和原错证明；只复用不可变SQL解析。
    key = (str(root), seal_sha, c._digest(parent), c._digest(proof))
    if key in _PARSED_ANCHORS:
        return deepcopy(_PARSED_ANCHORS[key])
    ledger = c._ledger(root, load); chain._ledger_limits(manifest, ledger, parent)
    batch = load('runtime/batch-results.json'); first = parent['retained_count']; order = parent['full_execution_order']
    rows = batch.get('rows', [])[first:]
    _require(c._equal(batch.get('rows', [])[:first], parent['retained_rows']) and len(rows) == len(ledger['roots']) == 2
             and batch.get('retained_observations') == first and batch.get('new_roots_planned') == len(order) - first
             and batch.get('new_attempted_count') == 2 and c._equal(batch.get('limits'), manifest['ledger_limits'])
             and batch.get('not_created') == order[first + 2:] and batch.get('status') == 'stopped'
             and batch.get('stop_reason') == STOP_REASON, 'EXACT_TWO_CELL_PREFIX_REQUIRED')
    session = load('semantic-gates/session.json')
    _require(session.get('manifest_sha256') == c._digest(manifest), 'ORIGINAL_SCOPE_SESSION_CHANGED')
    retained = deepcopy(parent['retained_rows']); ids = {x['root_id'] for x in retained + parent['technical_attempts']}
    attempts = set(); new_ids = {}
    cases = load('inputs/cases.json')['cases']; prompt = next(x['prompt'] for x in cases if x['case_id'] == case)
    for offset, row in enumerate(rows):
        position = first + offset; variant = order[position][1]
        _require(row.get('original_position') == position and [row.get('case_id'), row.get('variant')] == order[position]
                 and row.get('root_id') not in ids and row.get('issues') == [] and row.get('returncode') == 0,
                 'ORIGINAL_ROW_IDENTITY_CHANGED')
        ids.add(row['root_id']); new_ids[variant] = row['root_id']
        relative = 'runtime/cells/' + case + '-' + variant
        result = load(relative + '/result.json'); view = load(relative + '/task-view.json')
        root_row = load(relative + '/root-row.json'); grant = load('semantic-gates/grants/' + str(position) + '.json')
        expected_kind = 'technical_retry' if offset == 0 else 'first_attempt'
        _require(grant.get('position') == position and [grant.get('case_id'), grant.get('variant')] == order[position]
                 and grant.get('manifest_sha256') == c._digest(manifest)
                 and grant.get('continuation_sha256') == c._digest(manifest['completion_contract'])
                 and grant.get('session_nonce') == session.get('nonce') and grant.get('attempt_kind') == expected_kind
                 and isinstance(grant.get('attempt_id'), str) and re.fullmatch('[0-9a-f]{64}', grant['attempt_id'])
                 and grant['attempt_id'] not in attempts
                 and c._equal(result.get('experiment_attempt'), {k: grant[k] for k in
                     ('attempt_id', 'attempt_kind', 'position', 'continuation_sha256')}), 'ORIGINAL_GRANT_CHANGED')
        attempts.add(grant['attempt_id'])
        _require(c._equal(result.get('loaded_supervisor'), manifest['loaded_prompt_contract'][variant])
                 and result.get('status') == 'executed' and result.get('task_status') == row['status'] == view.get('status')
                 and row['status'] == proof['backend_statuses'][variant] and view.get('prompt') == prompt
                 and root_row.get('task_id') == row['root_id'] and root_row.get('status') == row['status']
                 and root_row.get('prompt') == prompt and c._equal(json.loads(root_row['result_json']), view.get('result')),
                 'ORIGINAL_STOP_OR_PROMPT_CHANGED')
        guard = result.get('live_isolation_guard') or {}
        _require(isinstance(result['isolation'].get('denials'), list)
                 and all(x.get('phase') == 'guard_selfcheck' for x in result['isolation']['denials'])
                 and not result['isolation'].get('denied_connections') and result.get('ledger_halt') is None
                 and guard.get('closed') is False and guard.get('first_error') is None
                 and guard.get('ledger_bound') is True and guard.get('persistence_errors', []) == [],
                 'RUNTIME_ISOLATION_ERROR_NOT_RECOVERABLE')
        answer = (view.get('result') or {}).get('answer_markdown')
        _require(isinstance(answer, str) and answer.strip(), 'ORIGINAL_NONEMPTY_REPORT_REQUIRED')
        if variant == 'B':
            _require(view['result'].get('answer_kind') == 'failure_report'
                     and isinstance(view.get('clarification'), dict) and view['clarification'].get('status') == 'pending'
                     and view['clarification'].get('task_id') == row['root_id'], 'ORIGINAL_NEEDS_INPUT_REQUIRED')
        enriched = c._row(root, manifest, position, ledger, load, graded=False, allow_observation_stops=True)
        _require(c._equal(enriched['native_status_counts'], row.get('native_status_counts'))
                 and c._equal(enriched['physical_status_counts'], row.get('physical_status_counts')), 'RECEIPT_COUNTS_CHANGED')
        physical = [x for x in ledger['reservations'] if x['root_id'] == row['root_id']]
        _archives(root, case, variant, physical, load)
        enriched.update(quality=None, artifact_owner='original_sealed_manifest', score_owner='current_manifest',
                        deferred_review=True)
        retained.append(enriched)
    _require(c._equal(proof.get('root_ids'), new_ids), 'REPLAY_ROOT_ID_CHANGED')
    meta = ledger['metadata']; error = meta.get('first_error') or {}
    _require(meta.get('halt_reason') == STOP_REASON and error.get('reason') == STOP_REASON
             and error.get('operation') == 'continuation_controller' and error.get('root_id') == new_ids['B']
             and error.get('details', {}).get('new_attempted_cells') == 2, 'EXACT_ORIGINAL_HALT_REQUIRED')
    charge, pending = chain._charged(ledger)
    _require(not pending, 'NEW_UNKNOWN_CANNOT_BE_RECOVERED')
    _require(not {x['reservation_id'] for x in ledger['reservations']} & {
        x['reservation_id'] for x in parent['pending_reservations']}, 'OLD_PENDING_REQUEST_ID_REUSED')
    carry = {k: parent['carried_usage'][k] + charge[k] for k in charge}
    carry['seconds'] = parent['carried_usage']['seconds']
    final = load('runtime/ledger-final.json')
    _require(c._equal(final.get('carried_usage'), parent['carried_usage'])
             and final.get('actual_known_tokens') == charge['tokens'] and final.get('physical_sent_requests') == charge['requests']
             and final.get('actual_known_estimated_nano_yuan') == charge['nano_yuan']
             and c._equal(final.get('charged_or_reserved'), {'tokens': carry['tokens'], 'requests': carry['requests'], 'nano': carry['nano_yuan']})
             and final.get('halt_reason') == STOP_REASON, 'ORIGINAL_LEDGER_EXPORT_CHANGED')
    original = parent['anchor_manifest']
    lineage = {'anchor_root': str(root), 'proof_path': str(root / PROOF_PATH),
               'proof_sha256': c._verified_hash(root / PROOF_PATH), 'case_id': case}
    value = {**deepcopy(parent), 'anchor_root': str(root), 'anchors': anchors,
        'anchor_seals': {**parent['anchor_seals'], str(root): seal_sha}, 'retained_rows': retained,
        'retained_count': len(retained), 'new_execution_order': deepcopy(order[len(retained):]),
        'technical_attempts': deepcopy(parent['technical_attempts']), 'carried_usage': carry,
        'closed_ledger_roots': [*parent['closed_ledger_roots'], str(root)],
        'pending_reservations': deepcopy(parent['pending_reservations']), 'latest_anchor_manifest': manifest,
        'parent_contract_sha256': c._digest(manifest['completion_contract']), 'leaf_charged_or_reserved': charge,
        'minimum_complete_attempts': len(order) + len(parent['technical_attempts']),
        'deferred_review_cases': [case], 'retry_positions': [], 'observation_recovery_lineage': [lineage],
        'old_cases': c._helpers()._read(original['cases_path']),
        'old_materials': c._helpers()._read(Path(original['prepared_input_root']) / 'material-snapshot.json')}
    _PARSED_ANCHORS[key] = deepcopy(value)
    return value


def build_observation_recovery_contract(anchor_root, timing_authorization, *, _seen=frozenset()):
    """只派生原观察与待审计划，不创建审核结论、授权或新账本。"""
    try:
        base = _anchor(anchor_root, _seen); timing, clock = _c()._timing(timing_authorization, base)
        excluded = {'anchor_manifest', 'latest_anchor_manifest', 'old_cases', 'old_materials',
                    'clock', 'timing_authorization', 'version', 'retry_policy'}
        return {**{k: deepcopy(v) for k, v in base.items() if k not in excluded}, 'version': VERSION,
            'clock': clock, 'timing_authorization': timing,
            'retry_policy': {'retry_positions': [], 'preserve_all_old_technical_attempts': True,
                             'never_resend_old_unknown_requests': True}}
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError, sqlite3.Error) as exc:
        raise PermissionError('FINAL_OBSERVATION_RECOVERY_ANCHOR_INVALID') from exc


def validate_observation_recovery(manifest, *, _seen=frozenset()):
    """显式V4仍核原96格、源码、材料和财务上限，不缓存当前授权。"""
    c = _c()
    try:
        _require('continuation_contract' not in manifest and validate_observation_policy(manifest), 'EXPLICIT_V4_REQUIRED')
        supplied = manifest['completion_contract']
        expected = build_observation_recovery_contract(supplied['anchor_root'], supplied['timing_authorization'], _seen=_seen)
        _require(c._equal(supplied, expected), 'CONTRACT_OR_ANCHOR_CHANGED')
        base = _anchor(supplied['anchor_root'], _seen); original = base['anchor_manifest']
        _require(manifest.get('experiment_kind') == 'prompt_bundle_v1'
                 and c._equal(manifest.get('evaluation_policy'), c.EVALUATION_POLICY)
                 and c._equal(manifest['execution_order'], expected['full_execution_order']), 'FIXED_SUITE_OR_EVALUATION_CHANGED')
        from agent_final_pdf_acquisition import SOURCE_ACQUISITION_CONTRACT, validate_source_acquisition_contract
        _require(validate_source_acquisition_contract(manifest)
                 and c._equal(manifest.get('source_acquisition_contract'), SOURCE_ACQUISITION_CONTRACT), 'SOURCE_ACQUISITION_CONTRACT_REQUIRED')
        _require(c._equal(manifest['ledger_limits'], {**original['ledger_limits'], 'carried_usage': expected['carried_usage'],
                 'enforce_batch_seconds': False}), 'ORIGINAL_FINANCIAL_AND_ROOT_CAPS_REQUIRED')
        _require(c._equal(manifest.get('carried_budget_source'), {'version': VERSION, 'anchor_root': expected['anchor_root'],
            'ledger_path': str(Path(expected['anchor_root']) / 'runtime/budget.sqlite3'),
            'ledger_sha256': expected['anchors']['runtime/budget.sqlite3'],
            'seal_sha256': expected['anchor_seals'][expected['anchor_root']]}), 'EXACT_CARRY_SOURCE_REQUIRED')
        owned = Path(manifest['experiment_root'])
        _require(owned.is_absolute() and owned.resolve() == owned and all(not owned.is_relative_to(Path(p))
                 and not Path(p).is_relative_to(owned) for p in expected['closed_ledger_roots']), 'NEW_OWNED_ROOT_REQUIRED')
        for key, relative in {'runtime_root': 'runtime', 'semantic_gate_root': 'semantic-gates', 'prepared_input_root': 'inputs',
                             'cases_path': 'inputs/cases.json', 'model_manifest_path': 'inputs/model-manifest.json'}.items():
            path = Path(manifest[key]); _require(path.resolve() == path == owned / relative, 'OWNED_GEOMETRY_REQUIRED')
        for variant in ('A', 'B'):
            path = Path(manifest['source_roots'][variant]); _require(path.resolve() == path == owned / 'snapshots' / variant, 'OWNED_SOURCE_REQUIRED')
        auth = Path(manifest['authorization_path'])
        _require(auth.resolve() == auth and auth.is_relative_to(owned) and not any(auth.is_relative_to(owned / p)
                 for p in ('runtime', 'inputs', 'snapshots')), 'NEW_AUTHORIZATION_REQUIRED')
        if expected['clock']['policy'] == 'pending':
            _require(manifest.get('real_execution_released') is False and manifest.get('live_qualification_released') is False,
                     'PENDING_CANNOT_RELEASE')
        c._helpers()._check_inputs(manifest, original, base['old_cases'], base['old_materials']); c._sources(manifest, original)
        return {**deepcopy(expected), 'anchor_manifest': deepcopy(original), 'latest_anchor_manifest': deepcopy(base['latest_anchor_manifest'])}
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError, sqlite3.Error) as exc:
        raise PermissionError('FINAL_OBSERVATION_RECOVERY_INVALID') from exc
