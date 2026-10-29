"""V4停点恢复的通用合成回归；无真实材料、网络或模型调用。"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'scripts'), str(Path(__file__).resolve().parent)]
import agent_final_completion as completion
import agent_final_completion_chain as chain
import agent_final_observation_recovery as recovery
from harness_ab_budget import Ledger
from test_agent_final_completion_chain import make_leaf, sealed
from test_agent_final_continuation import write_json, digest

ORIGINAL_PARENT = chain._parent


def make_recovery_fixture(directory):
    """父context是既有合成V3叶，恢复叶使用真实临时SQLite和完整普通文件。"""
    f, decision = make_leaf(directory / 'parent')
    with patch.object(chain, '_parent', return_value=f['base']), patch.object(completion, '_old_grade', return_value=decision):
        parent = chain._anchor(f['new_root'], frozenset())
        contract = chain.build_completion_chain_contract(f['new_root'], f['timing'])
    root = directory / 'closed-observation'; root.mkdir()
    m = deepcopy(f['manifest'])
    m.update(experiment_root=str(root), runtime_root=str(root / 'runtime'), semantic_gate_root=str(root / 'semantic-gates'),
             prepared_input_root=str(root / 'inputs'), cases_path=str(root / 'inputs/cases.json'),
             authorization_path=str(root / 'formal-execution-auth.json'), completion_contract=contract)
    shutil.copytree(Path(f['manifest']['cases_path']).parent, root / 'inputs')
    m['ledger_limits']['carried_usage'] = deepcopy(parent['carried_usage'])
    write_json(root / 'manifest.formal.json', m)
    code = root / 'snapshots/B/backend/scripts/harness_ab_pair_gate.py'; code.parent.mkdir(parents=True)
    code.write_text('# Generic immutable legacy gate fixture.\n')
    ledger = Ledger(root / 'runtime/budget.sqlite3', clock=lambda: 2000,
                    start_time=parent['clock']['round_started_at'], **m['ledger_limits'])
    first = parent['retained_count']; case = m['execution_order'][first][0]
    rows = []; root_ids = {}; nonce = 'synthetic-parent-session'
    write_json(root / 'semantic-gates/session.json', {'manifest_sha256': completion._digest(m), 'nonce': nonce})
    cases = json.loads((root / 'inputs/cases.json').read_text())['cases']
    prompt = next(x['prompt'] for x in cases if x['case_id'] == case)
    for offset in range(2):
        position = first + offset; variant = m['execution_order'][position][1]
        identity = 'observation-' + variant; root_ids[variant] = identity
        status = 'partial' if variant == 'A' else 'needs_input'
        ledger.begin_root(identity, case, variant)
        reservation = ledger.reserve(identity, max_output_tokens=10, input_upper_bound=10)
        ledger.mark_sent(reservation['reservation_id'])
        usage = {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5, 'prompt_cache_hit_tokens': 0}
        response_id = 'synthetic-response-' + variant
        ledger.settle(reservation['reservation_id'], usage, response_metadata={'response_id': response_id})
        grant = {'position': position, 'case_id': case, 'variant': variant,
            'manifest_sha256': completion._digest(m), 'continuation_sha256': completion._digest(contract),
            'attempt_kind': 'technical_retry' if offset == 0 else 'first_attempt',
            'attempt_id': ('a' if offset == 0 else 'b') * 64, 'session_nonce': nonce}
        write_json(root / 'semantic-gates/grants' / f'{position}.json', grant)
        cell = root / 'runtime/cells' / (case + '-' + variant)
        answer = {'answer_markdown': 'Synthetic incomplete observation; no quality claim.',
                  'answer_kind': 'answer' if variant == 'A' else 'failure_report'}
        view = {'task_id': identity, 'status': status, 'prompt': prompt, 'result': answer}
        if variant == 'B': view['clarification'] = {'task_id': identity, 'status': 'pending', 'question': 'Synthetic question?'}
        result = {'case_id': case, 'variant': variant, 'root_id': identity, 'status': 'executed', 'task_status': status,
            'loaded_supervisor': m['loaded_prompt_contract'][variant], 'isolation': {'denials': []},
            'live_isolation_guard': {'closed': False, 'first_error': None, 'ledger_bound': True, 'persistence_errors': []},
            'experiment_attempt': {k: grant[k] for k in ('attempt_id', 'attempt_kind', 'position', 'continuation_sha256')}}
        root_row = {'task_id': identity, 'status': status, 'prompt': prompt, 'parent_task_id': None,
                    'result_json': json.dumps(answer)}
        write_json(cell / 'result.json', result); write_json(cell / 'task-view.json', view)
        write_json(cell / 'root-row.json', root_row); write_json(cell / 'actions.json', {})
        write_json(cell / 'model-calls.json', [{'status': 'succeeded', 'sent_state': 'sent', 'request_id': response_id,
            'response_json': json.dumps({'id': response_id}), 'usage_json': json.dumps(usage)}])
        times = [datetime.fromtimestamp(2000 + offset * 20 + d, timezone.utc).isoformat() for d in (0, 10)]
        events = {'task_id': identity, 'events': [
            {'task_id': identity, 'event_type': 'task_created', 'created_at': times[0], 'payload': {'status': 'queued'}},
            {'task_id': identity, 'event_type': 'task_finished', 'created_at': times[1], 'payload': {'status': status, 'result': answer}}]}
        write_json(cell / 'events.json', events); write_json(cell / 'ledger-snapshot.json', ledger.snapshot())
        write_json(cell / 'requests' / (reservation['reservation_id'] + '.response.json'), {'id': response_id, 'usage': usage})
        db = sqlite3.connect(cell / 'task.sqlite3')
        db.execute('CREATE TABLE tasks(task_id TEXT,status TEXT,prompt TEXT,result_json TEXT,parent_task_id TEXT)')
        db.execute('INSERT INTO tasks VALUES (?,?,?,?,?)', (identity, status, prompt, json.dumps(answer), None)); db.commit(); db.close()
        rows.append({'case_id': case, 'variant': variant, 'root_id': identity, 'status': status, 'issues': [],
            'returncode': 0, 'original_position': position, 'native_status_counts': {'succeeded': 1},
            'physical_status_counts': {'settled': 1}})
    ledger.halt(recovery.STOP_REASON, operation='continuation_controller', root_id=root_ids['B'], details={'new_attempted_cells': 2})
    write_json(root / 'runtime/ledger-final.json', ledger.snapshot())
    write_json(root / 'runtime/batch-results.json', {'status': 'stopped', 'stop_reason': recovery.STOP_REASON,
        'rows': parent['retained_rows'] + rows, 'retained_observations': first, 'new_roots_planned': 96 - first,
        'new_attempted_count': 2, 'not_created': m['execution_order'][first + 2:], 'limits': m['ledger_limits']})
    write_json(root / 'formal-execution-auth.json', {'approved': False, 'allowed_modes': [], 'status': 'closed_no_resume',
        'manifest_sha256': completion._digest(m)})
    write_json(root / 'formal-preflight/authorization-closed.json', {'status': 'closed', 'no_resume_or_resend': True,
        'round_started_at': parent['clock']['round_started_at'], 'round_absolute_deadline': None, 'policy': 'financial_only'})
    write_json(root / 'formal-preflight/vector-close.json', {'status': 'passed', 'identity': {'running': False}})
    write_json(root / 'formal-control/controller-result.json', {'status': 'exited', 'returncode': 0,
        'batch_group_cleaned': True, 'cleanup_all_passed': True, 'policy': 'financial_only', 'global_deadline': None,
        'original_T0': parent['clock']['round_started_at'], 'original_clock_reset': False,
        'wrapper_finished_before_cleanup_deadline': True, 'cleanup_started_at': 2100, 'model_processes_stopped_at': 2101,
        'finished_at': 2102, 'wrapper_finalized_at': 2103, 'cleanup_deadline_after_stop': 2400,
        'cleanup_steps_deadline_after_stop': 2390, 'final_process_groups': [{'alive_group_members': []}]})
    write_json(root / 'audit/post-run-protection.json', {'status': 'passed', 'safety_closed': True, 'errors': [], 'data_integrity_errors': []})
    proof = {'version': 'measurement_gate_stop_v1', 'status': 'passed_exact_legacy_pair_identity_replay',
        'original_stop_reason': recovery.STOP_REASON, 'exception_type': 'PermissionError',
        'exception_message': 'PAIR_ENDED_CELL_IDENTITY_INVALID', 'function': 'pair_identity', 'case_id': case,
        'positions': [first, first + 1], 'root_ids': root_ids, 'backend_statuses': {'A': 'partial', 'B': 'needs_input'},
        'original_records_unchanged': True, 'new_challenge_created': False, 'new_score_created': False, 'original_gate_passed': False,
        'actions': dict.fromkeys(['real_model_calls', 'network_calls', 'Ledger_instances', 'sqlite_mutations', 'old_file_mutations'], 0)}
    write_json(root / recovery.PROOF_PATH, proof)
    fixture = {'parent': parent, 'root': root, 'manifest': m, 'case': case, 'timing': f['timing']}
    rebind_proof(fixture)
    return fixture


def rebind_proof(f):
    root = f['root']; path = root / recovery.PROOF_PATH; proof = json.loads(path.read_text())
    # 只用于合成攻击夹具重封；真实旧锚永不由测试改写。
    proof['bindings'] = {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()
        and p.name not in {'SEAL.json', 'SEAL.sha256', 'SEAL-READBACK.json'} and p != path}
    write_json(path, proof); sealed(root)


class ObservationRecoveryTests(unittest.TestCase):
    def setUp(self):
        recovery._PARSED_ANCHORS.clear(); chain._PARSED_ANCHORS.clear()
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.f = make_recovery_fixture(Path(self.temp.name).resolve())
        self.parent = patch.object(chain, '_parent', return_value=self.f['parent']); self.parent.start(); self.addCleanup(self.parent.stop)
        self.addCleanup(recovery._PARSED_ANCHORS.clear)

    def build(self):
        return recovery.build_observation_recovery_contract(self.f['root'], self.f['timing'])

    def mutate(self, relative, function):
        path = self.f['root'] / relative; value = json.loads(path.read_text()); function(value); write_json(path, value); rebind_proof(self.f)

    def test_exact_pair_keeps14_raw_stops_unknowns_and_deferred_current_owner(self):
        x = self.build()
        self.assertEqual([x['retained_count'], len(x['new_execution_order']), len(x['technical_attempts']), x['minimum_complete_attempts']], [14, 82, 3, 99])
        self.assertEqual(x['retained_rows'][:12], self.f['parent']['retained_rows'])
        self.assertEqual([r['status'] for r in x['retained_rows'][-2:]], ['partial', 'needs_input'])
        self.assertTrue(all(r['quality'] is None and r['score_owner'] == 'current_manifest' for r in x['retained_rows'][-2:]))
        self.assertEqual(x['retry_positions'], []); self.assertEqual(x['deferred_review_cases'], [self.f['case']])
        self.assertEqual(x['pending_reservations'], self.f['parent']['pending_reservations'])
        self.assertEqual(x['carried_usage']['tokens'], self.f['parent']['carried_usage']['tokens'] + 10)
        self.assertEqual(x['carried_usage']['seconds'], self.f['parent']['carried_usage']['seconds'])
        self.assertTrue(all(r['source_manifest_path'] == str(self.f['root'] / 'manifest.formal.json') for r in x['retained_rows'][-2:]))

    def test_legacy_row_still_rejects_needs_input_without_explicit_flag(self):
        c = completion; root, seal, _ = c._sealed(self.f['root']); load = c._loader(root, seal, {})
        ledger = c._ledger(root, load)
        with self.assertRaisesRegex(PermissionError, 'RETAINED_TERMINAL_REQUIRED'):
            c._row(root, self.f['manifest'], 13, ledger, load, graded=False)

    def test_exact_error_proof_cannot_cover_other_permission_error_or_claim_success(self):
        for key, value in [('exception_message', 'another_permission_error'), ('original_gate_passed', True),
                           ('new_score_created', True), ('original_records_unchanged', False)]:
            before = (self.f['root'] / recovery.PROOF_PATH).read_bytes()
            try:
                self.mutate(recovery.PROOF_PATH, lambda x: x.update({key: value}))
                with self.subTest(key=key), self.assertRaises(PermissionError): self.build()
            finally:
                (self.f['root'] / recovery.PROOF_PATH).write_bytes(before); rebind_proof(self.f)

    def test_status_event_prompt_loaded_identity_grant_and_runtime_denial_are_strict(self):
        case = self.f['case']; targets = [
            (f'runtime/cells/{case}-B/events.json', lambda x: x['events'].pop()),
            (f'runtime/cells/{case}-B/task-view.json', lambda x: x.update(status='completed')),
            (f'runtime/cells/{case}-A/task-view.json', lambda x: x.update(prompt='changed')),
            (f'runtime/cells/{case}-A/result.json', lambda x: x['loaded_supervisor'].update(changed=True)),
            ('semantic-gates/grants/12.json', lambda x: x.update(attempt_kind='first_attempt')),
            (f'runtime/cells/{case}-A/result.json', lambda x: x['isolation']['denials'].append({'phase': 'execution'}))]
        for path, change in targets:
            original = (self.f['root'] / path).read_bytes()
            try:
                self.mutate(path, change)
                with self.subTest(path=path), self.assertRaises(PermissionError): self.build()
            finally:
                (self.f['root'] / path).write_bytes(original); rebind_proof(self.f)

    def test_native_id_usage_raw_archive_and_extra_unknown_rejected(self):
        root = self.f['root']; case = self.f['case']
        native = f'runtime/cells/{case}-B/model-calls.json'
        self.mutate(native, lambda x: x[0].update(request_id='foreign'))
        with self.assertRaises(PermissionError): self.build()

    def test_original_halt_closure_seal_and_score_cannot_be_rewritten(self):
        self.mutate('formal-execution-auth.json', lambda x: x.update(approved=True))
        with self.assertRaises(PermissionError): self.build()

    def test_typed_policy_only_v4_and_real_event_status(self):
        self.assertFalse(recovery.validate_observation_policy({}))
        m = {'completion_contract': {'version': recovery.VERSION}, 'observation_policy': deepcopy(recovery.OBSERVATION_POLICY)}
        self.assertTrue(recovery.validate_observation_policy(m))
        for change in ('missing', 'legacy', 'bool'):
            value = deepcopy(m)
            if change == 'missing': del value['observation_policy']
            elif change == 'legacy': value['completion_contract']['version'] = chain.VERSION
            else: value['observation_policy']['require_root_task_finished_event'] = 1
            with self.subTest(change=change), self.assertRaises(PermissionError): recovery.validate_observation_policy(value)

    def test_inherited_v3_policy_requires_nonempty_typed_lineage_and_exact_policy(self):
        lineage = {'anchor_root': str(Path(self.temp.name).resolve() / 'sealed-observation'),
            'proof_sha256': 'a' * 64, 'case_id': 'generic-observation-pair'}
        lineage['proof_path'] = str(Path(lineage['anchor_root']) / recovery.PROOF_PATH)
        m = {'completion_contract': {'version': chain.VERSION, 'observation_recovery_lineage': [lineage]},
             'observation_policy': deepcopy(recovery.OBSERVATION_POLICY)}
        self.assertTrue(recovery.validate_observation_policy(m))
        self.assertFalse(recovery.validate_observation_policy({'completion_contract': {'version': chain.VERSION}}))
        for change in ('missing-policy', 'legacy-policy', 'empty', 'boolean', 'bad-item', 'extra-field',
                       'bad-sha', 'foreign-path', 'relative-root', 'duplicate', 'unknown-version', 'typed-policy'):
            value = deepcopy(m); contract = value['completion_contract']
            if change == 'missing-policy': del value['observation_policy']
            elif change == 'legacy-policy': del contract['observation_recovery_lineage']
            elif change == 'empty': contract['observation_recovery_lineage'] = []
            elif change == 'boolean': contract['observation_recovery_lineage'] = True
            elif change == 'bad-item': contract['observation_recovery_lineage'] = [True]
            elif change == 'extra-field': contract['observation_recovery_lineage'][0]['trusted'] = True
            elif change == 'bad-sha': contract['observation_recovery_lineage'][0]['proof_sha256'] = 'changed'
            elif change == 'foreign-path': contract['observation_recovery_lineage'][0]['proof_path'] = '/foreign/proof.json'
            elif change == 'relative-root': contract['observation_recovery_lineage'][0]['anchor_root'] = 'relative'
            elif change == 'duplicate': contract['observation_recovery_lineage'].append(deepcopy(lineage))
            elif change == 'unknown-version': contract['version'] = 'unknown'
            else: value['observation_policy']['require_root_task_finished_event'] = 1
            with self.subTest(change=change), self.assertRaises(PermissionError):
                recovery.validate_observation_policy(value)
        missing = deepcopy(m); missing.pop('observation_policy'); missing['completion_contract']['observation_recovery_lineage'] = []
        with self.assertRaises(PermissionError): recovery.validate_observation_policy(missing)

    def test_later_v3_candidate_cannot_remove_policy_or_forge_frozen_lineage(self):
        candidate = self.current_manifest(); base = self.build()
        base.update(anchor_manifest=self.f['parent']['anchor_manifest'], latest_anchor_manifest=self.f['manifest'],
                    old_cases={}, old_materials={})
        with patch.object(chain, '_anchor', return_value=base), patch.object(completion, '_sources'), \
             patch.object(completion._helpers(), '_check_inputs'):
            contract = chain.build_completion_chain_contract(self.f['root'], self.f['timing'])
            candidate['completion_contract'] = contract; candidate['carried_budget_source']['version'] = chain.VERSION
            result = chain.validate_completion_chain(candidate)
            self.assertEqual(result['observation_recovery_lineage'], base['observation_recovery_lineage'])
            self.assertEqual(result['retained_rows'][-1]['source_manifest_path'],
                             str(self.f['root'] / 'manifest.formal.json'))
            for change in ('missing-policy', 'typed-policy', 'empty-lineage', 'drop-lineage', 'forged-proof', 'case-change'):
                value = deepcopy(candidate)
                if change == 'missing-policy': value.pop('observation_policy')
                elif change == 'typed-policy': value['observation_policy']['non_delivery_execution_complete'] = 0
                elif change == 'empty-lineage': value['completion_contract']['observation_recovery_lineage'] = []
                elif change == 'drop-lineage': del value['completion_contract']['observation_recovery_lineage']
                elif change == 'forged-proof': value['completion_contract']['observation_recovery_lineage'][0]['proof_sha256'] = 'b' * 64
                else: value['completion_contract']['observation_recovery_lineage'][0]['case_id'] = 'other-pair'
                with self.subTest(change=change), self.assertRaises(PermissionError):
                    chain.validate_completion_chain(value)

    def test_hot_cache_revalidates_without_sql_and_cold_miss_keeps_cell_boundary(self):
        expected = self.build()
        from agent_final_runtime import CellBoundary
        runtime = Path(self.temp.name).resolve() / 'runtime'; cell = runtime / 'current'; cell.mkdir(parents=True)
        boundary = CellBoundary(cell, runtime); sys.addaudithook(boundary.audit)
        try:
            with patch.object(sqlite3, 'connect', wraps=sqlite3.connect) as connect:
                self.assertEqual(self.build(), expected); self.assertEqual(connect.call_count, 0)
            recovery._PARSED_ANCHORS.clear()
            from harness_ab_boundary import IsolationError
            with self.assertRaises(IsolationError): self.build()
            self.assertTrue(boundary.denials)
        finally: boundary.enabled = False

    def test_dispatch_and_deferred_prior_owner_do_not_adopt_new_artifact_identity(self):
        x = self.build()
        with patch.object(completion, 'validate_completion', return_value=x):
            self.assertIsNone(completion.prior_pair_manifest({}, 12))
            cell = completion.retained_cell({}, self.f['case'], 'B')
            self.assertEqual(cell['cell_root'], str(self.f['root'] / 'runtime/cells' / (self.f['case'] + '-B')))
        with patch.object(recovery, 'validate_observation_recovery', return_value={'sentinel': True}) as validate:
            self.assertEqual(ORIGINAL_PARENT({'completion_contract': {'version': recovery.VERSION}}, {'seen'}), {'sentinel': True})
            validate.assert_called_once()

    def test_later_chain_binds_deferred_grade_to_actual_new_score_owner_only(self):
        x = self.build(); parent = {**x}; retained = deepcopy(x['retained_rows'])
        root = Path(self.temp.name).resolve() / 'later-score'; root.mkdir(); write_json(root / 'manifest.formal.json', {'generic': True})
        write_json(root / 'semantic-gates' / self.f['case'] / 'decision.json', {'generic': True})
        decision = {'quality': {'A': 'partial_usable', 'B': 'failed'}}
        with patch.object(completion, '_old_grade', return_value=decision):
            chain._bind_deferred_reviews(root, {}, parent, retained, lambda *args: None)
        self.assertEqual(retained[:12], x['retained_rows'][:12])
        for row in retained[-2:]:
            self.assertEqual(row['source_manifest_path'], str(self.f['root'] / 'manifest.formal.json'))
            self.assertEqual(row['score_manifest_path'], str(root / 'manifest.formal.json'))
            self.assertFalse(row['deferred_review'])
        self.assertEqual(retained[-1]['quality'], 'failed')


    def current_manifest(self):
        from agent_final_pdf_acquisition import SOURCE_ACQUISITION_CONTRACT
        context = self.build(); owned = Path(self.temp.name).resolve() / 'current-candidate'; owned.mkdir(exist_ok=True)
        m = deepcopy(self.f['manifest'])
        m.update(completion_contract=context, observation_policy=deepcopy(recovery.OBSERVATION_POLICY),
            evaluation_policy=deepcopy(completion.EVALUATION_POLICY), source_acquisition_contract=deepcopy(SOURCE_ACQUISITION_CONTRACT),
            experiment_root=str(owned), runtime_root=str(owned / 'runtime'), semantic_gate_root=str(owned / 'semantic-gates'),
            prepared_input_root=str(owned / 'inputs'), cases_path=str(owned / 'inputs/cases.json'),
            model_manifest_path=str(owned / 'inputs/model-manifest.json'), authorization_path=str(owned / 'auth.json'),
            source_roots={v: str(owned / 'snapshots' / v) for v in ('A', 'B')},
            real_execution_released=False, live_qualification_released=False)
        m['ledger_limits']['carried_usage'] = deepcopy(context['carried_usage'])
        m['carried_budget_source'] = {'version': recovery.VERSION, 'anchor_root': str(self.f['root']),
            'ledger_path': str(self.f['root'] / 'runtime/budget.sqlite3'),
            'ledger_sha256': context['anchors']['runtime/budget.sqlite3'],
            'seal_sha256': context['anchor_seals'][str(self.f['root'])]}
        return m

    def test_current_contract_caps_pending_retry_policy_and_owner_geometry_are_bound(self):
        m = self.current_manifest()
        with patch.object(completion, '_sources'), patch.object(completion._helpers(), '_check_inputs'):
            self.assertEqual(completion.validate_completion(m)['retained_count'], 14)
            for change in ('cap', 'retry', 'deferred', 'pending', 'carry', 'policy', 'dual', 'source'):
                value = deepcopy(m)
                if change == 'cap': value['ledger_limits']['batch_tokens'] += 1
                elif change == 'retry': value['completion_contract']['retry_positions'] = [14]
                elif change == 'deferred': value['completion_contract']['deferred_review_cases'] = []
                elif change == 'pending': value['completion_contract']['pending_reservations'][0]['status'] = 'not_sent'
                elif change == 'carry': value['ledger_limits']['carried_usage']['requests'] += 1
                elif change == 'policy': value['observation_policy']['non_delivery_execution_complete'] = 0
                elif change == 'dual': value['continuation_contract'] = {}
                else: value['source_roots']['A'] = str(self.f['root'] / 'snapshots/A')
                with self.subTest(change=change), self.assertRaises(PermissionError): completion.validate_completion(value)

    def test_raw_response_archive_and_true_unknown_cannot_be_approved_by_new_proof(self):
        root = self.f['root']; case = self.f['case']
        archive = next((root / 'runtime/cells' / (case + '-B') / 'requests').glob('*.response.json'))
        original = archive.read_bytes()
        try:
            value = json.loads(original); value['usage']['total_tokens'] += 1; write_json(archive, value); rebind_proof(self.f)
            with self.assertRaises(PermissionError): self.build()
        finally: archive.write_bytes(original); rebind_proof(self.f)
        db = sqlite3.connect(root / 'runtime/budget.sqlite3')
        db.execute("UPDATE reservations SET status='unknown' WHERE root_id=?", ('observation-B',)); db.commit(); db.close()
        rebind_proof(self.f)
        with self.assertRaises(PermissionError): self.build()

    def test_native_sql_status_cannot_be_replaced_by_view_and_stop_export(self):
        case = self.f['case']; path = self.f['root'] / 'runtime/cells' / (case + '-B') / 'task.sqlite3'
        db = sqlite3.connect(path); db.execute("UPDATE tasks SET status='completed'"); db.commit(); db.close()
        rebind_proof(self.f)
        with self.assertRaises(PermissionError) as error: self.build()
        self.assertIn('RETAINED_OBSERVATION_SQL_STATE_CHANGED', str(error.exception.__cause__))

    def test_cache_is_detached_and_closure_rechecked_before_parsed_rows(self):
        x = self.build(); x['retained_rows'][-1]['quality'] = 'full'
        self.assertIsNone(self.build()['retained_rows'][-1]['quality'])
        self.mutate('formal-control/controller-result.json', lambda v: v.update(cleanup_all_passed=False))
        with self.assertRaises(PermissionError): self.build()

    def test_non_delivery_exported_status_remains_explicit_and_requires_stop_event(self):
        case = self.f['case']; cell = self.f['root'] / 'runtime/cells' / (case + '-B')
        view = json.loads((cell / 'task-view.json').read_text()); events = json.loads((cell / 'events.json').read_text())
        for status in ('failed', 'cancelled'):
            v, e = deepcopy(view), deepcopy(events); v.update(status=status, result=None)
            e['events'][-1]['payload'] = {'status': status, 'result': None}
            observation = recovery.exported_stop(v, e)
            self.assertEqual(observation['observed_status'], status); self.assertFalse(observation['delivery_complete'])
        events['events'].append(deepcopy(events['events'][-1]))
        with self.assertRaises(PermissionError): recovery.exported_stop(view, events)


    def test_v4_runtime_cannot_omit_the_new_bound_helper(self):
        value = deepcopy(self.f['manifest'])
        value['completion_contract'] = {'version': recovery.VERSION}
        with self.assertRaisesRegex(PermissionError, 'RECOVERY_RUNTIME_HELPER_REQUIRED'):
            completion._sources(value, self.f['parent']['anchor_manifest'])


    def test_uncheckpointed_task_sidecar_is_not_ignored_by_immutable_read(self):
        cell = self.f['root'] / 'runtime/cells' / (self.f['case'] + '-B')
        (cell / 'task.sqlite3-wal').write_bytes(b'generic uncheckpointed fixture')
        rebind_proof(self.f)
        with self.assertRaises(PermissionError) as error: self.build()
        self.assertIn('RETAINED_OBSERVATION_UNCHECKPOINTED', str(error.exception.__cause__))


if __name__ == '__main__': unittest.main()
