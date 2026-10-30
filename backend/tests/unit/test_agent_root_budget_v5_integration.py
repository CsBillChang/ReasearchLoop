"""V5边界集成回归；所有工件为临时合成记录，证明权威由专用模块单测覆盖。"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import agent_final_analysis as analysis
import agent_final_completion as completion
import agent_final_completion_chain as chain
import agent_final_observation_recovery as observation
import harness_ab_pair_gate as gate

V5 = 'agent_final_completion_v5'
SAFETY = 'passed_final_safety_with_original_missing_post_readback'
ISSUES = ['ledger_halted', 'unexpected_native_not_sent']


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


class RootBudgetV5IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.proof = {'version': 'root_budget_stop_proof_v1', 'path': str(self.root / 'proof.json'), 'sha256': 'a' * 64}
        self.manifest = {'completion_contract': {'version': V5,
            'observation_recovery_lineage': [{'anchor_root': str(self.root / 'old'),
                'proof_path': str(self.root / 'old' / observation.PROOF_PATH),
                'proof_sha256': 'b' * 64, 'case_id': 'SYNTH-REC-01-R1'}],
            'root_budget_recovery_proof': deepcopy(self.proof),
            'root_budget_recovery_lineage': [{'anchor_root': str(self.root / 'closed'),
                'proof_ref': deepcopy(self.proof), 'root_id': 'budget-root', 'position': 7}]},
            'experiment_kind': 'prompt_bundle_v1', 'observation_policy': deepcopy(observation.OBSERVATION_POLICY),
            'final_suite_policy': deepcopy(gate.FINAL_SUITE_POLICY)}
        self.module = ModuleType('agent_final_root_budget_recovery')
        self.module._PARSED_ANCHORS = {}
        self.module.validate_root_budget_recovery = Mock(return_value={'version': V5, 'verified': True})
        self.module.retained_budget_stop_receipts = Mock(return_value=None)
        self.module._safety = Mock()
        self.module_patch = patch.dict(sys.modules, {'agent_final_root_budget_recovery': self.module})
        self.module_patch.start(); self.addCleanup(self.module_patch.stop)
        completion.clear_validation_caches(); self.addCleanup(completion.clear_validation_caches)
        self.case = 'SYNTH-REC-01-R1'
        self.proofs = {'physical_receipts_verified': True,
            'native_status_counts': {'succeeded': 1, 'not_sent': 1}, 'physical_status_counts': {'settled': 1},
            'root_budget_stop_verified': True}
        self.retained = {'case_id': self.case, 'variant': 'A', 'root_id': 'budget-root', 'issues': list(ISSUES),
                         **self.proofs}
        self.cell = self.root / 'closed/runtime/cells' / (self.case + '-A')
        self.result = {'case_id': self.case, 'variant': 'A', 'root_id': 'budget-root', 'status': 'executed',
            'task_status': 'partial', 'ledger_halt': 'token reservation budget exhausted',
            'isolation': {'denials': [], 'denied_connections': []},
            'live_isolation_guard': {'first_error': None, 'closed': False, 'ledger_bound': True, 'persistence_errors': []}}
        self.view = {'task_id': 'budget-root', 'status': 'partial', 'prompt': 'Synthetic question',
                     'result': {'answer_markdown': 'Synthetic partial report', 'reason': 'actual budget stop'}}
        self.make_cell(self.cell, self.result, self.view)

    def make_cell(self, cell, result, view):
        root = result['root_id']
        at = lambda d: datetime.fromtimestamp(1000 + d, timezone.utc).isoformat()
        for name, value in {'result.json': result, 'task-view.json': view, 'actions.json': {},
            'events.json': [{'task_id': root, 'event_type': 'task_created', 'created_at': at(0), 'payload': {}},
                {'task_id': root, 'event_type': 'task_finished', 'created_at': at(1),
                 'payload': {'status': view['status'], 'result': view['result']}}],
            'model-calls.json': [{'status': 'succeeded'}, {'status': 'not_sent'}],
            'ledger-snapshot.json': {'roots': [{'root_id': root, 'case_id': result['case_id'], 'variant': result['variant']}],
                'reservations': [{'root_id': root, 'status': 'settled'}]}}.items():
            write(cell / name, value)

    def test_v5_dispatch_and_parent_preserve_seen_cycle_context(self):
        expected = self.module.validate_root_budget_recovery.return_value
        self.assertEqual(completion.validate_completion(self.manifest), expected)
        self.assertEqual(chain._parent(self.manifest, {'ancestor'}), expected)
        self.module.validate_root_budget_recovery.assert_called_with(self.manifest, _seen={'ancestor'})
        with patch.object(observation, 'validate_observation_recovery', return_value='legacy-v4') as old:
            m = deepcopy(self.manifest); m['completion_contract']['version'] = observation.VERSION
            self.assertEqual(completion.validate_completion(m), 'legacy-v4'); old.assert_called_once_with(m)
        m = deepcopy(self.manifest); m['continuation_contract'] = {}
        with self.assertRaises(PermissionError): completion.validate_completion(m)

    def test_v5_observation_policy_requires_explicit_proof_and_both_lineages(self):
        self.assertTrue(observation.validate_observation_policy(self.manifest))
        for field in ('root_budget_recovery_proof', 'root_budget_recovery_lineage', 'observation_recovery_lineage'):
            with self.subTest(field=field):
                m = deepcopy(self.manifest); del m['completion_contract'][field]
                with self.assertRaises(PermissionError): observation.validate_observation_policy(m)
        m = deepcopy(self.manifest); m['observation_policy']['non_delivery_execution_complete'] = True
        with self.assertRaises(PermissionError): observation.validate_observation_policy(m)

    def test_only_proved_retained_v5_budget_stop_bypasses_halt_and_not_sent(self):
        self.module.retained_budget_stop_receipts.return_value = self.proofs
        stop = gate.observation_stop(self.manifest, self.cell, self.result, self.view, self.retained)
        self.assertEqual(stop['observed_status'], 'partial')
        self.assertEqual(stop['observation_stop']['measurement_kind'], 'non_complete_observation')
        self.assertEqual(self.result['ledger_halt'], 'token reservation budget exhausted')
        self.assertEqual(self.retained['issues'], ISSUES)
        for version, retained, proofs in [(V5, None, self.proofs), (V5, self.retained, None),
                (observation.VERSION, self.retained, self.proofs), ('agent_final_completion_v3', self.retained, self.proofs)]:
            with self.subTest(version=version, retained=retained is not None, proofs=proofs is not None):
                m = deepcopy(self.manifest); m['completion_contract']['version'] = version
                self.module.retained_budget_stop_receipts.return_value = proofs
                with self.assertRaises(PermissionError): gate.observation_stop(m, self.cell, self.result, self.view, retained)

    def test_authority_denial_and_receipt_count_drift_fail_closed(self):
        for error in ('unknown native receipt', 'bad not_sent evidence', 'changed root proof'):
            self.module.retained_budget_stop_receipts.side_effect = PermissionError(error)
            with self.subTest(error=error), self.assertRaises(PermissionError):
                gate.observation_stop(self.manifest, self.cell, self.result, self.view, self.retained)
        self.module.retained_budget_stop_receipts.side_effect = None
        self.module.retained_budget_stop_receipts.return_value = {**self.proofs, 'native_status_counts': {'succeeded': 2}}
        with self.assertRaisesRegex(PermissionError, 'RETAINED_RECEIPTS_CHANGED'):
            gate.observation_stop(self.manifest, self.cell, self.result, self.view, self.retained)

    def test_pair_identity_preserves_exact_old_issues_and_rejects_any_rewrite(self):
        source = deepcopy(self.manifest); source['loaded_prompt_contract'] = {'A': {}, 'B': {}}
        self.manifest['cases_path'] = str(self.root / 'cases.json')
        write(self.manifest['cases_path'], {'cases': [{'case_id': self.case, 'prompt': self.view['prompt']}]})
        self.result['loaded_supervisor'] = {}; write(self.cell / 'result.json', self.result)
        other = self.cell.with_name(self.case + '-B')
        rb = {**deepcopy(self.result), 'variant': 'B', 'root_id': 'other-root', 'ledger_halt': None}
        vb = {**deepcopy(self.view), 'task_id': 'other-root'}
        self.make_cell(other, rb, vb)
        clean_proofs = {'physical_receipts_verified': True, 'native_status_counts': {'succeeded': 1},
                        'physical_status_counts': {'settled': 1}}
        write(other / 'model-calls.json', [{'status': 'succeeded', 'sent_state': 'sent', 'request_id': 'id',
            'response_json': json.dumps({'id': 'id'}), 'usage_json': json.dumps({'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2})}])
        snapshot = json.loads((other / 'ledger-snapshot.json').read_text())
        snapshot['reservations'][0].update(response_json=json.dumps({'response_id': 'id'}),
            usage_json=json.dumps({'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}),
            actual_input=1, actual_output=1, actual_tokens=2)
        write(other / 'ledger-snapshot.json', snapshot)
        origins = {'A': (self.cell, source, self.retained), 'B': (other, source, clean_proofs)}
        self.module.retained_budget_stop_receipts.side_effect = lambda m,c,r,v,retained,**kw: self.proofs if r['variant']=='A' else None
        rows = [{'case_id': self.case, 'variant': v, 'root_id': r, 'issues': list(ISSUES) if v=='A' else []}
                for v,r in [('A','budget-root'),('B','other-root')]]
        with patch.object(gate, '_cell_origin', side_effect=lambda m,c,v: origins[v]), \
             patch.object(gate, 'paired_source_contract', return_value={}), \
             patch('prompt_bundle_cases.case_materials', return_value=[]), \
             patch('prompt_bundle_cases.validate_preconditions', return_value={}):
            identity, _, _ = gate.pair_identity(self.manifest, self.case, rows)
            self.assertEqual(identity['answers']['A']['backend_status'], 'partial')
            for change in ('empty', 'missing', 'replacement', 'additional'):
                attacked = deepcopy(rows)
                if change == 'empty': attacked[0]['issues'] = []
                elif change == 'missing': del attacked[0]['issues']
                elif change == 'replacement': attacked[0]['issues'] = ['different_issue']
                else: attacked[0]['issues'].append('unrelated_integrity_failure')
                with self.subTest(change=change), self.assertRaisesRegex(PermissionError, 'ENDED_CELL_IDENTITY_INVALID'):
                    gate.pair_identity(self.manifest, self.case, attacked)

    def test_trusted_review_reconstruction_preserves_issues_in_prior_and_old_grade_readback(self):
        retained = deepcopy(self.retained)
        retained['provenance'] = {'cell_root': str(self.cell)}
        context = {'version': V5, 'retained_rows': [retained], 'retained_count': 2,
                   'deferred_review_cases': [self.case]}
        self.module.validate_root_budget_recovery.return_value = context
        self.module.retained_budget_stop_receipts.return_value = self.proofs
        answers = {'A': {'root_id': 'budget-root'}, 'B': {'root_id': 'other-root'}}
        reconstructed = gate._review_identity_rows(self.manifest, self.case, answers)
        self.assertEqual(reconstructed[0]['issues'], ISSUES); self.assertEqual(reconstructed[1]['issues'], [])
        self.module.retained_budget_stop_receipts.return_value = None
        with self.assertRaises(PermissionError): gate._review_identity_rows(self.manifest, self.case, answers)
        self.module.retained_budget_stop_receipts.return_value = self.proofs
        owner = self.root/'score-owner'; directory = owner/'semantic-gates'/self.case
        self.manifest.update(execution_order=[[self.case, 'A'], [self.case, 'B']],
                             semantic_gate_root=str(owner/'semantic-gates'))
        challenge = {'identity': {'answers': answers}}
        decision = {'status': 'passed', 'stop_reasons': []}
        for name, value in [('challenge-private.json', challenge), ('review.json', {}), ('decision.json', decision)]:
            write(directory/name, value)
        with patch.object(gate, 'evaluate_pair_review', return_value=decision) as evaluate, \
             gate._gate_context_scope(self.manifest, context):
            gate._prior_reviews(self.manifest, 2, context, lambda: 100)
            self.assertEqual(evaluate.call_args.args[2], reconstructed)
        challenge_sha = completion._digest(challenge); proofs = []
        for name in ('one', 'two', 'adjudication'):
            path = directory/'source-proofs'/(name+'.json')
            write(path, {'paid_model_calls': 0, 'challenge_sha256': challenge_sha})
            proofs.append({'path': str(path), 'sha256': gate.sha256(path.read_bytes()), 'identity': name})
        review = {'challenge_sha256': challenge_sha, 'reviewed_at': 100,
                  'independent_reviews': proofs[:2], 'adjudication_source': proofs[2]}
        decision.update(challenge_sha256=challenge_sha, review_sha256=completion._digest(review))
        for name, value in [('review-packet.json', {}), ('review.json', review), ('decision.json', decision)]:
            write(directory/name, value)
        with patch.object(gate, 'evaluate_pair_review', return_value=decision) as evaluate:
            result = completion._old_grade(owner, self.manifest, self.case, lambda rel: json.loads((owner/rel).read_text()))
            self.assertEqual(result, decision); self.assertEqual(evaluate.call_args.args[2], reconstructed)
        legacy = deepcopy(self.manifest); legacy['completion_contract']['version'] = observation.VERSION
        self.assertEqual(gate._review_identity_rows(legacy, self.case, answers)[0]['issues'], [])

    def test_narrow_closure_dispatch_never_swallows_proof_failure(self):
        files = {'audit/post-run-protection.json': {'status': 'failed'},
                 'audit/post-run-protection-corrected.json': {'status': 'passed'},
                 'audit/post-run-safety-supplement.json': {'status': SAFETY}}
        load = lambda relative: files[relative]
        chain._protection(self.root, load); self.module._safety.assert_called_once_with(self.root, load)
        self.module._safety.side_effect = PermissionError('changed missing-readback proof')
        with self.assertRaisesRegex(PermissionError, 'changed missing-readback proof'): chain._protection(self.root, load)
        files['audit/post-run-safety-supplement.json']['status'] = 'generic-passed'
        self.module._safety.reset_mock()
        with self.assertRaises(PermissionError): chain._protection(self.root, load)
        self.module._safety.assert_not_called()

    def test_cache_registers_root_module_and_external_proof_dependency(self):
        proof = self.root / 'external-proof.json'; write(proof, {'status': 'passed'})
        expected = completion._helpers()._hash(proof); calls = []
        self.module._PARSED_ANCHORS['parsed'] = {'verified': True}
        def validate():
            calls.append(1); completion._capture_seal(self.root, 'c' * 64)
            completion._require(completion._verified_hash(proof) == expected, 'EXTERNAL_PROOF_CHANGED')
            return {'anchor_seals': {str(self.root): 'c' * 64}}
        build = lambda: completion._cached_anchor(V5, self.root, frozenset(), validate)
        build(); build(); self.assertEqual(len(calls), 1)
        self.module._PARSED_ANCHORS.clear(); build(); self.assertEqual(len(calls), 2)
        write(proof, {'status': 'failed'})
        with self.assertRaises(PermissionError): build()
        completion.clear_validation_caches(); self.assertEqual(self.module._PARSED_ANCHORS, {})

    def test_analysis_raw_receipts_keep_partial_usage_known_but_unknown_and_new_stop_null(self):
        usage = {'prompt_tokens': 9, 'completion_tokens': 3, 'total_tokens': 12}
        native = [{'call_id': 'sent-call', 'task_id': 'budget-root', 'status': 'succeeded',
            'request_id': 'remote-response', 'response_json': json.dumps({'id': 'remote-response'}),
            'usage_json': json.dumps(usage)}, {'call_id': 'not-sent-call', 'task_id': 'budget-root',
            'status': 'not_sent', 'sent_state': 'not_sent', 'response_json': json.dumps({'error_type': 'RequestNotSent'}),
            'usage_json': '{}'}]
        physical = {'root_id': 'budget-root', 'reservation_id': 'reservation', 'status': 'settled', 'sent_at': 1000,
            'response_json': json.dumps({'response_id': 'remote-response'}), 'usage_json': json.dumps(usage),
            'actual_input': 9, 'actual_output': 3, 'actual_tokens': 12, 'actual_nano': 39, 'request_json': '{}'}
        ledger = {'metadata': {'config': {'input_nano_per_token': 3, 'cached_input_nano_per_token': 1,
            'output_nano_per_token': 4}}, 'roots': [{'root_id': 'budget-root', 'case_id': self.case, 'variant': 'A'}],
            'reservations': [physical]}
        task = {'task_id': 'budget-root', 'root_task_id': 'budget-root', 'parent_task_id': None,
            'status': 'partial', 'created_at': 1000, 'updated_at': 1001, 'result_json': json.dumps(self.view['result'])}
        write(self.cell/'root-row.json', task); write(self.cell/'model-calls.json', native)
        write(self.cell/'requests/reservation.response.json', {'id': 'remote-response', 'usage': usage})
        owner = {'runtime_root': str(self.cell.parents[1]), 'completion_contract': {'version': chain.VERSION}}
        reader = analysis.Reader()
        def table(path, name):
            return {'tasks': [task], 'model_calls': native, 'actions': [],
                    'task_events': json.loads((self.cell/'events.json').read_text())}[name]
        def analyze():
            with patch.object(reader, 'table', side_effect=table), \
                 patch.object(analysis, 'semantic_score', return_value={'whole_answer_quality': None, 'issues': []}):
                return analysis.analyze_cell(reader, self.root, owner, self.case, 'A', ledger, {}, artifact_cell=self.cell)
        row = analyze()
        self.assertEqual(row['actual_task_family_tokens'], 12); self.assertTrue(row['usage_reconciled'])
        self.assertEqual(row['status'], 'partial'); self.assertIsNone(row['end_to_end']['successful_delivery_seconds'])
        native[1]['status'] = 'unknown'; write(self.cell/'model-calls.json', native)
        row = analyze(); self.assertIsNone(row['actual_task_family_tokens']); self.assertFalse(row['usage_reconciled'])
        native[1]['status'] = 'not_sent'; write(self.cell/'model-calls.json', native)
        owner['completion_contract']['version'] = V5
        row = analyze(); self.assertIsNone(row['actual_task_family_tokens']); self.assertFalse(row['usage_reconciled'])
        self.assertIn('ledger_halted', row['issues'])
        self.assertIn('native_not_sent_requires_retained_root_budget_proof', row['issues'])

    def test_analysis_closure_requires_passed_audit_or_exact_authority(self):
        root = self.root/'closure'; auth_path = root/'authorization.json'
        manifest = deepcopy(self.manifest); manifest.update(authorization_path=str(auth_path),
            formal_controller_contract={'policy': 'financial_only', 'cleanup_reserve_seconds': 300})
        manifest['completion_contract']['clock'] = {'round_started_at': 50}
        write(root/'formal-control/controller-result.json', {'policy': 'financial_only', 'status': 'financial_budget_halt',
            'started_at': 100, 'cleanup_started_at': 1000, 'model_processes_stopped_at': 1005,
            'cleanup_deadline_after_stop': 1300, 'finished_at': 1010, 'wrapper_finalized_at': 1012,
            'batch_group_cleaned': True, 'cleanup_all_passed': True})
        write(auth_path, {'approved': False, 'allowed_modes': [], 'round_started_at': 50})
        write(root/'formal-preflight/authorization-closed.json', {'status': 'closed', 'policy': 'financial_only',
            'round_started_at': 50, 'no_resume_or_resend': True, 'at': 1013})
        write(root/'formal-preflight/vector-close.json', {'identity': {'running': False}, 'at': 1014})
        write(root/'audit/post-run-protection.json', {'status': 'failed', 'safety_closed': False, 'errors': ['arbitrary failure']})
        self.assertFalse(analysis.closure_status(analysis.Reader(), root, manifest)['closed'])
        self.module._safety.assert_not_called()
        write(root/'audit/post-run-safety-supplement.json', {'status': SAFETY})
        self.assertTrue(analysis.closure_status(analysis.Reader(), root, manifest)['closed'])
        self.module._safety.side_effect = PermissionError('incorrect missing receipt diagnosis')
        with self.assertRaises(PermissionError): analysis.closure_status(analysis.Reader(), root, manifest)

    def test_summary_and_report_disclose_original_budget_gate_failure_separately(self):
        import test_agent_completion_analysis as fixture
        f = fixture.CompletionAnalysisTests(methodName='runTest'); f.setUp(); self.addCleanup(f.doCleanups)
        f.order = lambda: [[f'SYNTH-QA-{i:02d}-R{r}',v] for r in (1,2) for i in range(1,25) for v in ('A','B')]
        case, variant = f.order()[0]; identity = case+'-'+variant
        def alter(manifest, context, ledgers, closure):
            manifest['completion_contract']['version'] = V5; context['version'] = V5
            context['retained_rows'][0]['quality'] = 'partial_usable'
            context['root_budget_recovery_lineage'] = [{'root_id': identity, 'proof_ref': deepcopy(self.proof)}]
        def row_alter(row, technical):
            if not technical and row['expanded_case_id'] == case and row['variant'] == variant:
                row.update(status='partial', root_budget_stop_verified=True, original_reconciliation_passed=False,
                    original_reconciliation_issues=list(ISSUES), budget_stop_outcome='partial_root_token_reservation_budget_stop')
                row['score']['whole_answer_quality'] = 'partial_usable'
                row['end_to_end'].update(delivery_ready_observed=False, successful_delivery_seconds=None)
        result, _ = f.result(alter=alter, row_alter=row_alter)
        self.assertEqual(result['issues'], [])
        self.assertEqual(result['preserved_execution_gate_failure_count'], 1)
        self.assertEqual(result['expected_budget_stop_observation_count'], 1)
        entry = result['preserved_execution_gate_failures'][0]
        self.assertEqual(entry['original_reconciliation_issues'], ISSUES)
        self.assertFalse(entry['original_reconciliation_passed']); self.assertEqual(entry['proof_ref'], self.proof)
        self.assertTrue(entry['actual_usage_known']); self.assertEqual(entry['actual_task_family_tokens'], 3)
        self.assertTrue(entry['retained_primary_observation']); self.assertFalse(entry['technical_retry'])
        self.assertEqual(entry['status'], 'partial'); self.assertEqual(result['continuation']['primary_denominator'], 96)
        self.assertEqual(len(result['technical_attempts']), 2)
        self.assertIsNotNone(entry['source']['anchor_root']); self.assertIsNotNone(entry['source']['manifest_path'])
        report = analysis.report_text(result)
        for text in ('原执行门失败保留 1 个', '预算停点未改判clean或成功', '保留为原主观测',
                     '未归入technical_attempts或重跑', '原reconciliation_passed=False', 'ledger_halted',
                     'unexpected_native_not_sent', self.proof['sha256'], '实际用量可知 True'):
            self.assertIn(text, report)
        budget = next(row for row in result['rows'] if row['root_id'] == identity)
        self.assertIsNone(budget['end_to_end']['successful_delivery_seconds'])
        with self.assertRaises(ValueError): analysis._execution_gate_failures([budget], {})
        unknown = deepcopy(budget); unknown.update(usage_reconciled=False, actual_task_family_tokens=None)
        disclosure = analysis._execution_gate_failures([unknown], {'root_budget_recovery_lineage': [
            {'root_id': identity, 'proof_ref': deepcopy(self.proof)}]})['expected_budget_stop_observations'][0]
        self.assertFalse(disclosure['actual_usage_known']); self.assertIsNone(disclosure['actual_task_family_tokens'])

    def test_analysis_budget_usage_known_with_original_failure_disclosed(self):
        owner_root = self.root / 'closed'; owner = {'experiment_root': str(owner_root), 'runtime_root': str(owner_root/'runtime')}
        path = owner_root/'manifest.formal.json'; write(path, owner)
        item = {**self.retained, 'quality': None, 'source_manifest_path': str(path),
            'source_manifest_sha256': analysis.sha(path.read_bytes()),
            'source_batch_record': {'issues': list(ISSUES), 'reconciliation_passed': False},
            'provenance': {'anchor_root': str(owner_root), 'cell_root': str(self.cell)}}
        context = {'version': V5, 'closed_ledger_roots': [str(owner_root)], 'retained_rows': [item], 'technical_attempts': []}
        cell_row = {'status': 'partial', 'score': {'whole_answer_quality': None}, 'issues': [],
                    'usage_reconciled': True, 'actual_task_family_tokens': 123456}
        ledger = {'metadata': {}, 'roots': [], 'reservations': []}
        self.module.retained_budget_stop_receipts.return_value = self.proofs
        with patch.object(analysis, 'read_ledger', return_value=ledger), \
             patch.object(analysis, 'closure_status', return_value={'controller': {}}), \
             patch.object(analysis, 'analyze_cell', return_value=deepcopy(cell_row)):
            retained, technical, _, _ = analysis.completion_artifacts(analysis.Reader(), self.manifest, context)
            row = retained[(self.case, 'A')]
            self.assertEqual(row['actual_task_family_tokens'], 123456); self.assertTrue(row['usage_reconciled'])
            self.assertEqual(row['issues'], []); self.assertEqual(row['original_reconciliation_issues'], ISSUES)
            self.assertFalse(row['original_reconciliation_passed']); self.assertTrue(row['root_budget_stop_verified'])
            self.assertEqual(row['original_batch_record'], item); self.assertEqual(technical, [])
            self.assertEqual(row['status'], 'partial')
            self.module.retained_budget_stop_receipts.return_value = {**self.proofs, 'root_budget_stop_verified': False}
            with self.assertRaises(PermissionError):
                analysis.completion_artifacts(analysis.Reader(), self.manifest, context)
            for reason in ('unknown native receipt', 'unproved not_sent'):
                self.module.retained_budget_stop_receipts.side_effect = PermissionError(reason)
                with self.subTest(reason=reason), self.assertRaises(PermissionError):
                    analysis.completion_artifacts(analysis.Reader(), self.manifest, context)


if __name__ == '__main__': unittest.main(verbosity=2)
