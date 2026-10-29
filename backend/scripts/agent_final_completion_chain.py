"""已封存补测叶的通用追加；不重开旧账本，不改写旧主观测或评分。"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import re
import sqlite3

VERSION = 'agent_final_completion_v3'
_PARSED_ANCHORS = {}


def _c():
    import agent_final_completion as completion
    return completion


def _require(value, reason):
    if not value:
        raise PermissionError('FINAL_COMPLETION_CHAIN_' + reason)


def _parent(manifest, seen):
    value = manifest.get('completion_contract')
    _require(isinstance(value, dict), 'EXPLICIT_PARENT_REQUIRED')
    if value.get('version') == VERSION:
        return validate_completion_chain(manifest, _seen=seen)
    _require(value.get('version') == _c().VERSION, 'PARENT_VERSION_UNSUPPORTED')
    return _c().validate_completion(manifest)


def _closure(root, manifest, load):
    """核实原闭权、停止和有限收尾；保留原测量失败。"""
    c = _c()
    auth = load('formal-execution-auth.json')
    closed = load('formal-preflight/authorization-closed.json')
    vector = load('formal-preflight/vector-close.json')
    controller = load('formal-control/controller-result.json')
    _require(auth.get('approved') is False and auth.get('allowed_modes') == []
             and auth.get('status') == 'closed_no_resume'
             and auth.get('manifest_sha256') == c._digest({k:v for k,v in manifest.items() if k != 'cell_absolute_deadline'})
             and Path(manifest['authorization_path']) == root / 'formal-execution-auth.json', 'OLD_AUTH_NOT_CLOSED')
    clock = manifest['completion_contract']['clock']
    _require(closed.get('status') == 'closed' and closed.get('no_resume_or_resend') is True
             and closed.get('round_started_at') == clock['round_started_at']
             and closed.get('round_absolute_deadline', 1) is None and closed.get('policy') == 'financial_only',
             'OLD_CLOSURE_CLOCK_CHANGED')
    _require(vector.get('status') == 'passed' and vector['identity']['running'] is False,
             'OLD_VECTOR_NOT_STOPPED')
    _require(controller.get('status') == 'exited' and controller.get('returncode') == 0
             and controller.get('batch_group_cleaned') is True and controller.get('cleanup_all_passed') is True
             and controller.get('policy') == 'financial_only' and controller.get('global_deadline', 1) is None
             and controller.get('original_T0') == clock['round_started_at']
             and controller.get('original_clock_reset') is False and not controller.get('wrapper_failure')
             and controller.get('wrapper_finished_before_cleanup_deadline') is True,
             'OLD_CONTROLLER_NOT_SAFELY_CLOSED')
    start, stop, finish, final = (controller[k] for k in ('cleanup_started_at','model_processes_stopped_at',
                                                       'finished_at','wrapper_finalized_at'))
    _require(all(c._helpers()._finite(x) for x in (start,stop,finish,final))
             and controller['cleanup_deadline_after_stop'] == start + 300
             and controller['cleanup_steps_deadline_after_stop'] == start + 290
             and start <= stop <= finish <= final < start + 300 and stop <= start + 290,
             'OLD_CLEANUP_WINDOW_CHANGED')
    _require(controller.get('final_process_groups') and all(x['alive_group_members'] == []
             for x in controller['final_process_groups']), 'OLD_OWNED_PROCESS_GROUP_REMAINS')
    _protection(root, load)
    return controller


def _protection(root, load):
    """只接受完整原审计通过，或明确绑定的跨路径旧误报修正证明。"""
    c = _c()
    original = load('audit/post-run-protection.json')
    if original.get('status') == 'passed':
        _require(original.get('errors') == [] and original.get('data_integrity_errors',[]) == []
                 and original.get('safety_closed') is True, 'ORIGINAL_PROTECTION_NOT_PASSED')
        return
    corrected = load('audit/post-run-protection-corrected.json')
    proof = load('audit/post-run-safety-supplement.json')
    _require(original.get('status') == 'failed' and original.get('safety_closed') is False
             and original.get('errors') == [{'check':'full cold loaded identity unchanged','details':None}] * 2,
             'ORIGINAL_FAILURE_NOT_EXACT_CROSS_ROOT_GATE')
    _require(corrected.get('status') == 'passed' and corrected.get('safety_closed') is True
             and corrected.get('errors') == [] and corrected.get('data_integrity_errors') == [],
             'CORRECTED_FULL_PROTECTION_REQUIRED')
    _require(proof.get('status') == 'passed_final_safety_with_original_cross_root_gate_failure'
             and proof.get('safety_closed') is True and proof.get('measurement_complete') is False
             and proof.get('original_automatic_protection_audit_passed') is False
             and proof.get('original_cross_root_identity_gate_passed') is False
             and proof.get('original_failed_protection_report_unchanged') is True
             and proof.get('original_failed_report_safety_closed_remains_false') is True
             and proof.get('original_failed_report_error_count_remains') == 2
             and proof.get('original_audit_code_unchanged') is True
             and proof.get('actual_runtime_or_business_or_prompt_change') is False
             and proof.get('opaque_fingerprints_transformed') is False
             and proof.get('pending_not_reclassified_or_released') is True
             and proof.get('old_failures_or_scores_rewritten') is False, 'EXACT_SAFETY_SUPPLEMENT_REQUIRED')
    bindings = proof['evidence_bindings']
    _require(set(bindings) == {'original_failed_report','original_audit_code','corrected_audit_code','offline_seal',
        'R3_formal_manifest','R3_cold_contract','E3_cold_contract'}, 'COMPLETE_SAFETY_BINDINGS_REQUIRED')
    corrected_binding = proof['corrected_full_audit']
    _require(corrected_binding.get('status') == 'passed' and corrected_binding.get('safety_closed') is True
             and bindings['original_failed_report'].get('status') == 'failed'
             and bindings['original_failed_report'].get('safety_closed') is False
             and bindings['original_failed_report'].get('error_count') == 2
             and all(type(value) is int and value == 0 for value in proof['actions'].values())
             and proof['replacement_observations'] == [
                 {'original_cross_root_condition':False,'original_label':'full cold loaded identity unchanged',
                  'replacement_verified':True,'variant':variant} for variant in ('A','B')],
             'SAFETY_CORRECTION_SCOPE_CHANGED')
    for item in [*bindings.values(),corrected_binding]:
        path = Path(item['path'])
        _require(path.is_absolute() and path.resolve() == path and c._helpers()._hash(path) == item['sha256'],
                 'SAFETY_BINDING_CHANGED')
        if path.is_relative_to(root):
            load(str(path.relative_to(root)), False)
    _require(Path(bindings['original_failed_report']['path']) == root / 'audit/post-run-protection.json'
             and Path(corrected_binding['path']) == root / 'audit/post-run-protection-corrected.json'
             and Path(bindings['R3_formal_manifest']['path']) == root / 'manifest.formal.json'
             and Path(bindings['R3_cold_contract']['path']) == root / 'qualification/cold/contract.json',
             'SAFETY_PROOF_OWNER_CHANGED')


def _ledger_limits(manifest, ledger, parent):
    c = _c()
    meta, limits = ledger['metadata'], manifest['ledger_limits']
    config = meta['config']
    _require(meta['started_at'] == parent['clock']['round_started_at']
             and c._equal(config['carried_usage'], parent['carried_usage'])
             and c._equal(limits['carried_usage'], parent['carried_usage'])
             and config.get('enforce_batch_seconds') is False, 'LEAF_START_CARRY_CHANGED')
    for key in ('allowed_cells','batch_tokens','batch_requests','batch_seconds','root_tokens',
                'root_requests','root_seconds','max_output_tokens'):
        _require(c._equal(config[key], sorted(limits[key]) if key == 'allowed_cells' else limits[key]), 'LEAF_LIMITS_CHANGED')
    _require(config['batch_nano_yuan'] == int(Decimal(str(limits['batch_yuan'])) * 1000000000), 'LEAF_MONEY_LIMIT_CHANGED')


def _charged(ledger):
    c = _c(); charge = dict(tokens=0,requests=0,nano_yuan=0); pending = []
    seen = set(); roots = {x['root_id'] for x in ledger['roots']}
    for row in ledger['reservations']:
        _require(row['reservation_id'] not in seen and row['root_id'] in roots, 'LEAF_PHYSICAL_IDENTITY')
        seen.add(row['reservation_id'])
        _require(row['status'] in {'settled','reserved','sent','unknown','not_sent'}, 'LEAF_PHYSICAL_STATUS')
        if row['status'] == 'not_sent':continue
        charge['requests'] += 1
        if row['status'] == 'settled':
            _require(all(c._helpers()._integer(row[k]) for k in ('actual_tokens','actual_nano')), 'LEAF_SETTLEMENT_INVALID')
            charge['tokens'] += row['actual_tokens']; charge['nano_yuan'] += row['actual_nano']
        else:
            _require(row['actual_tokens'] is None and row['actual_nano'] is None
                     and all(c._helpers()._integer(row[k]) for k in ('reserved_tokens','reserved_nano')), 'LEAF_PENDING_INVALID')
            charge['tokens'] += row['reserved_tokens']; charge['nano_yuan'] += row['reserved_nano']; pending.append(row)
    return charge,pending


def _technical(root, manifest, row, ledger, load):
    c = _c(); case,variant = row['case_id'],row['variant']; cell = root / 'runtime/cells' / (case+'-'+variant)
    relative = str(cell.relative_to(root)); result = load(relative+'/result.json'); view = load(relative+'/task-view.json')
    native = load(relative+'/model-calls.json');load(relative+'/actions.json', False);load(relative+'/task.sqlite3', False)
    _require(result['case_id'] == case and result['variant'] == variant and result['root_id'] == row['root_id'] == view['task_id']
             and result['status'] == 'failed' and row.get('issues') and view['status'] in {'completed','partial','failed','cancelled','needs_input'},
             'LEAF_FAILED_CELL_IDENTITY')
    physical = [x for x in ledger['reservations'] if x['root_id'] == row['root_id']]
    proofs = c._receipts(row['root_id'],native,physical)
    return {**deepcopy(row),'retained':False,'status':view['status'],'stored_status':view['status'],
        'attempt_outcome':'cell_receipt_or_isolation_gate','result_status':result['status'],**proofs,
        'source_manifest_path':str(root/'manifest.formal.json'),'source_manifest_sha256':c._helpers()._hash(root/'manifest.formal.json'),
        'provenance':{'anchor_root':str(root),'cell_root':str(cell),
            'files':{name:c._helpers()._hash(cell/name) for name in (*c._helpers()._CELL_FILES,'task.sqlite3')}}}


def _anchor(anchor_root, seen):
    c = _c(); root = Path(anchor_root)
    _require((root/'SEAL.json').is_file(), 'ANCHOR_PENDING_SEAL')
    _require(str(root) not in seen, 'ANCHOR_CYCLE')
    seen = {*seen,str(root)}
    root,seal,seal_sha = c._sealed(root); anchors = {'SEAL.json':seal_sha}; load = c._loader(root,seal,anchors)
    manifest = load('manifest.formal.json')
    _require(Path(manifest['experiment_root']) == root and Path(manifest['runtime_root']) == root/'runtime', 'LEAF_GEOMETRY')
    parent = _parent(manifest, seen)
    _require(str(root) not in parent['closed_ledger_roots'], 'DUPLICATE_LEAF_ROOT')
    # 热身份门仍重验整棵seal、父合同与闭合证明，仅复用已验不可变SQL解析。
    _closure(root,manifest,load)
    cache_key = (str(root),seal_sha,c._digest(parent))
    if cache_key in _PARSED_ANCHORS:
        return deepcopy(_PARSED_ANCHORS[cache_key])
    ledger = c._ledger(root,load);_ledger_limits(manifest,ledger,parent)
    batch = load('runtime/batch-results.json');order = parent['full_execution_order'];first = parent['retained_count']
    old_rows,rows = batch['rows'][:first],batch['rows'][first:]
    _require(c._equal(old_rows,parent['retained_rows']) and rows and len(rows) == len(ledger['roots'])
             and batch['retained_observations'] == first and batch['new_roots_planned'] == len(order)-first
             and batch['new_attempted_count'] == len(rows) and c._equal(batch['limits'],manifest['ledger_limits'])
             and batch['not_created'] == order[first+len(rows):], 'LEAF_BATCH_PREFIX_CHANGED')
    retained = deepcopy(parent['retained_rows']);technical = deepcopy(parent['technical_attempts'])
    root_ids = {x['root_id'] for x in retained+technical}; new_good = [];failed = None;attempt_ids = set()
    for offset,row in enumerate(rows):
        position = first+offset
        _require(row.get('original_position') == position and [row['case_id'],row['variant']] == order[position]
                 and row['root_id'] not in root_ids and failed is None, 'LEAF_ORDER_OR_ROOT_CHANGED')
        root_ids.add(row['root_id'])
        cell = root/'runtime/cells'/(row['case_id']+'-'+row['variant']); result = load(str(cell.relative_to(root))+'/result.json')
        grant = load('semantic-gates/grants/'+str(position)+'.json')
        _require(grant['position'] == position and grant['case_id'] == row['case_id'] and grant['variant'] == row['variant']
                 and grant['manifest_sha256'] == c._digest({k:v for k,v in manifest.items() if k != 'cell_absolute_deadline'})
                 and grant['continuation_sha256'] == c._digest(manifest['completion_contract'])
                 and c._equal(result.get('experiment_attempt'),{k:grant[k] for k in ('attempt_id','attempt_kind','position','continuation_sha256')}),
                 'LEAF_GRANT_CHANGED')
        _require(re.fullmatch('[0-9a-f]{64}',grant['attempt_id']) and grant['attempt_id'] not in attempt_ids
                 and grant['attempt_kind'] == ('technical_retry' if position == first else 'first_attempt'),
                 'LEAF_ATTEMPT_KIND_OR_ID_CHANGED')
        attempt_ids.add(grant['attempt_id'])
        if result['status'] == 'executed' and row.get('issues') == []:
            _require(all(x['phase'] == 'guard_selfcheck' for x in result['isolation']['denials']), 'RETAINED_RUNTIME_DENIAL')
            grade = (root/'semantic-gates'/row['case_id']/'decision.json').exists()
            enriched = c._row(root,manifest,position,ledger,load,graded=grade)
            retained.append(enriched);new_good.append(enriched)
        else:
            failed = _technical(root,manifest,row,ledger,load);technical.append(failed)
    meta = ledger['metadata']
    _require(failed is not None and batch['status'] == 'stopped' and batch['stop_reason'] == 'cell_receipt_or_isolation_gate'
             and meta.get('halt_reason') == batch['stop_reason']
             and (meta.get('first_error') or {}).get('root_id') == failed['root_id'], 'EXACT_LEAF_FIRST_ERROR_REQUIRED')
    for position in range(0,len(retained)-1,2):
        pair = retained[position:position+2]
        if not any(x in new_good for x in pair):continue
        decision = c._old_grade(root,manifest,pair[0]['case_id'],load)
        for item in pair:
            _require(item.get('quality') in (None,decision['quality'][item['variant']]), 'OLD_GRADE_REWRITE_FORBIDDEN')
            item.update(quality=decision['quality'][item['variant']],score_manifest_path=str(root/'manifest.formal.json'),
                        score_manifest_sha256=c._helpers()._hash(root/'manifest.formal.json'))
            path = root/'semantic-gates'/item['case_id']/'decision.json'
            item['provenance'].update(grade_decision_path=str(path),grade_decision_sha256=c._helpers()._hash(path))
    charge,pending = _charged(ledger)
    _require(not pending, 'LEAF_UNRESOLVED_RECEIPTS_REQUIRE_EXPLICIT_RECOVERY_CONTRACT')
    old_pending = deepcopy(parent['pending_reservations']);all_pending = old_pending+pending
    _require(len({x['reservation_id'] for x in all_pending}) == len(all_pending), 'PENDING_IDENTITY_REUSED')
    carry = {k:parent['carried_usage'][k]+charge[k] for k in charge};carry['seconds'] = parent['carried_usage']['seconds']
    original = parent['anchor_manifest']
    result = {**deepcopy(parent),'anchor_root':str(root),'anchors':anchors,
        'anchor_seals':{**parent['anchor_seals'],str(root):seal_sha},'retained_rows':retained,
        'retained_count':len(retained),'technical_attempts':technical,'new_execution_order':deepcopy(order[len(retained):]),
        'carried_usage':carry,'closed_ledger_roots':parent['closed_ledger_roots']+[str(root)],
        'pending_reservations':all_pending,'latest_anchor_manifest':manifest,
        'parent_contract_sha256':c._digest(manifest['completion_contract']),
        'leaf_charged_or_reserved':charge,'minimum_complete_attempts':len(order)+len(technical),
        'old_cases':c._helpers()._read(original['cases_path']),
        'old_materials':c._helpers()._read(Path(original['prepared_input_root'])/'material-snapshot.json')}
    _PARSED_ANCHORS[cache_key] = deepcopy(result)
    return result


def build_completion_chain_contract(anchor_root, timing_authorization, *, _seen=frozenset()):
    """只从真实密封叶及其父合同派生；无SEAL明确拒绝，不生成授权。"""
    try:
        base = _anchor(anchor_root,_seen);timing,clock = _c()._timing(timing_authorization,base)
        excluded = {'anchor_manifest','latest_anchor_manifest','old_cases','old_materials','clock','timing_authorization','version','retry_policy'}
        value = {k:deepcopy(v) for k,v in base.items() if k not in excluded}
        return {**value,'version':VERSION,'clock':clock,'timing_authorization':timing,
            'retry_policy':{'original_position':base['retained_count'],'max_new_attempts':1,
                'preserve_all_old_technical_attempts':True,'never_resend_old_unknown_requests':True}}
    except (OSError,KeyError,TypeError,ValueError,ArithmeticError,sqlite3.Error) as exc:
        if isinstance(exc,PermissionError) and str(exc) == 'FINAL_COMPLETION_CHAIN_ANCHOR_PENDING_SEAL':
            raise
        raise PermissionError('FINAL_COMPLETION_CHAIN_ANCHOR_INVALID') from exc


def validate_completion_chain(manifest, *, _seen=frozenset()):
    """校验新路径与完整固定矩阵；旧版本只由原验证器解释。"""
    c = _c()
    try:
        _require('continuation_contract' not in manifest, 'TWO_CONTRACT_KEYS_FORBIDDEN')
        supplied = manifest['completion_contract']
        _require(isinstance(supplied,dict) and supplied.get('version') == VERSION, 'EXPLICIT_VERSION_REQUIRED')
        expected = build_completion_chain_contract(supplied['anchor_root'],supplied['timing_authorization'],_seen=_seen)
        _require(c._equal(expected,supplied), 'CONTRACT_OR_ANCHOR_CHANGED')
        base = _anchor(supplied['anchor_root'],_seen);original = base['anchor_manifest']
        _require(manifest.get('experiment_kind') == 'prompt_bundle_v1' and c._equal(manifest.get('evaluation_policy'),c.EVALUATION_POLICY)
                 and c._equal(manifest['execution_order'],expected['full_execution_order']), 'FIXED_SUITE_OR_EVALUATION_CHANGED')
        from agent_final_pdf_acquisition import SOURCE_ACQUISITION_CONTRACT, validate_source_acquisition_contract
        _require(validate_source_acquisition_contract(manifest) and c._equal(
            manifest.get('source_acquisition_contract'),SOURCE_ACQUISITION_CONTRACT), 'SOURCE_ACQUISITION_CONTRACT_REQUIRED')
        _require(c._equal(manifest['ledger_limits'],{**original['ledger_limits'],'carried_usage':expected['carried_usage'],
                 'enforce_batch_seconds':False}), 'ORIGINAL_FINANCIAL_AND_ROOT_CAPS_REQUIRED')
        _require(manifest.get('carried_budget_source') == {'version':VERSION,'anchor_root':expected['anchor_root'],
            'ledger_path':str(Path(expected['anchor_root'])/'runtime/budget.sqlite3'),
            'ledger_sha256':expected['anchors']['runtime/budget.sqlite3'],
            'seal_sha256':expected['anchor_seals'][expected['anchor_root']]}, 'CONSERVATIVE_LEAF_CARRY_SOURCE_REQUIRED')
        owned = Path(manifest['experiment_root'])
        _require(owned.is_absolute() and owned.resolve() == owned and all(not owned.is_relative_to(Path(p))
                 and not Path(p).is_relative_to(owned) for p in expected['closed_ledger_roots']), 'NEW_OWNED_ROOT_REQUIRED')
        for key,relative in {'runtime_root':'runtime','semantic_gate_root':'semantic-gates','prepared_input_root':'inputs',
                             'cases_path':'inputs/cases.json','model_manifest_path':'inputs/model-manifest.json'}.items():
            path = Path(manifest[key]);_require(path.resolve() == path == owned/relative, 'OWNED_GEOMETRY_REQUIRED')
        for variant in ('A','B'):
            path = Path(manifest['source_roots'][variant]);_require(path.resolve() == path == owned/'snapshots'/variant, 'OWNED_SOURCE_GEOMETRY_REQUIRED')
        auth = Path(manifest['authorization_path'])
        _require(auth.resolve() == auth and auth.is_relative_to(owned) and not any(auth.is_relative_to(owned/p)
                 for p in ('runtime','inputs','snapshots')), 'NEW_AUTHORIZATION_REQUIRED')
        if expected['clock']['policy'] == 'pending':
            _require(manifest.get('real_execution_released') is False and manifest.get('live_qualification_released') is False,
                     'PENDING_CANNOT_RELEASE')
        c._helpers()._check_inputs(manifest,original,base['old_cases'],base['old_materials']);c._sources(manifest,original)
        return {**deepcopy(expected),'anchor_manifest':deepcopy(original),
                'latest_anchor_manifest':deepcopy(base['latest_anchor_manifest'])}
    except (OSError,KeyError,TypeError,ValueError,ArithmeticError,sqlite3.Error) as exc:
        raise PermissionError('FINAL_COMPLETION_CHAIN_INVALID') from exc
