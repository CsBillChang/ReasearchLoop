"""显式V4停止观测、原配对补评和首次新格的零网络控制回归。"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_agent_completion_runtime as fixture
import harness_ab_pair_gate as gate
import run_read_paper_batch as batch


class ObservationRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.f = f = fixture.CompletionRuntimeTests(methodName='runTest')
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.addCleanup(f.tearDown)
        from agent_final_observation_recovery import OBSERVATION_POLICY
        self.policy = deepcopy(OBSERVATION_POLICY)
        owner = deepcopy(f.owners['R2'])
        owner.update(runtime_root=str(f.root/'R5/runtime'), semantic_gate_root=str(f.root/'R5/control'),
                     cases_path=str(f.root/'R5/cases.json'), test_source_contract={'owner': 'R5'},
                     completion_contract={'version': 'synthetic-parent-v3'},
                     loaded_prompt_contract={v: {'owner': 'R5', 'variant': v} for v in ('A', 'B')})
        fixture.write(owner['cases_path'], {'cases': f.cases})
        f.owners['R5'] = owner
        original_write = f.write_cell

        def write_cell(cell, source, case, variant, root, status, answer, native, attempt=None):
            native = deepcopy(native)
            for i, row in enumerate(native):
                identity = row.get('request_id') or 'synthetic-response-'+root+'-'+str(i)
                row.update(request_id=identity, sent_state='sent', task_id=root,
                    response_json=json.dumps({'id': identity}), usage_json=json.dumps(fixture.USAGE))
            original_write(cell, source, case, variant, root, status, answer, native, attempt)
            result = fixture.read(cell/'result.json')
            result.update(ledger_halt=None, isolation={'denials': [], 'denied_connections': []},
                live_isolation_guard={'first_error': None, 'closed': False, 'ledger_bound': True,
                                      'persistence_errors': []})
            fixture.write(cell/'result.json', result)
            view = fixture.read(cell/'task-view.json')
            view['result'].update(answer_kind='failure_report' if status in self.policy['non_delivery_statuses']
                else 'research_result', reason='synthetic actual stop')
            fixture.write(cell/'task-view.json', view)
            at = lambda delta: datetime.fromtimestamp(f.now[0]+delta, timezone.utc).isoformat()
            events = [{'task_id': root, 'event_type': 'task_created', 'created_at': at(-2), 'payload': {}},
                {'task_id': root, 'event_type': 'task_finished', 'created_at': at(-1),
                 'payload': {'status': status, 'result': view['result']}}]
            if status == 'needs_input':
                events.insert(1, {'task_id': root, 'event_type': 'clarification_requested', 'created_at': at(-1.5),
                    'payload': {'question': 'Synthetic clarification?', 'status': 'pending', 'task_id': root}})
            fixture.write(cell/'events.json', events)
            physical = [{'root_id': root, 'status': 'settled', 'response_json': json.dumps({'response_id': row['request_id']}),
                'usage_json': row['usage_json'], 'actual_input': 10, 'actual_output': 5, 'actual_tokens': 15}
                for row in native]
            fixture.write(cell/'ledger-snapshot.json', {'roots': [{'root_id': root, 'case_id': case, 'variant': variant}],
                                                       'reservations': physical})
        f.write_cell = write_cell
        for position in (11, 12, 13):
            case, variant = f.order[position]
            status = 'completed' if position == 11 else 'partial' if position == 12 else 'needs_input'
            root = 'synthetic-retained-'+str(position)
            cell = Path(owner['runtime_root'])/'cells'/(case+'-'+variant)
            write_cell(cell, owner, case, variant, root, status, 'Synthetic actual '+status,
                       [{'status': 'succeeded', 'request_id': 'old-response-'+str(position)}])
            row = {'case_id': case, 'variant': variant, 'root_id': root, 'status': status, 'issues': [],
                   'original_position': position, 'retained': True, 'quality': None}
            f.retained.append(row)
            f.old_cells[(case, variant)] = {'cell_root': str(cell), 'source_manifest': owner,
                'original_position': position, 'quality': None, 'physical_receipts_verified': True,
                'native_status_counts': {'succeeded': 1}, 'physical_status_counts': {'settled': 1}}
        previous_prior = f.prior_manifest
        f.prior_manifest = lambda m, offset, context: owner if offset == 10 else previous_prior(m, offset, context)
        f.manifest.update(completion_contract={'version': 'agent_final_completion_v4'},
                          observation_policy=deepcopy(self.policy), experiment_root=str(f.root/'current'))
        self.deferred = f.order[12][0]
        f.context.update(retained_count=14, retained_rows=f.retained, new_execution_order=f.order[14:],
                         deferred_review_cases=[self.deferred], retry_positions=[])
        f.context['technical_attempts'].append({'root_id': 'third-preserved-technical'})
        original_patches = f.patches
        def patches():
            stack = original_patches()
            stack.enter_context(patch.object(batch, 'planned_order', return_value=deepcopy(f.order[14:])))
            return stack
        f.patches = patches
        with f.patches():
            rows = f.retained[10:12];case = f.order[10][0]
            challenge = gate.prepare_pair_gate(owner, case, rows, gate.manifest_identity(owner), clock=lambda: f.now[0]-10)
            review = f.review_for(challenge, labels=('partial_usable', 'partial_usable'), at=f.now[0]-5)
            directory = Path(owner['semantic_gate_root'])/case
            fixture.write(directory/'review.json', review)
            fixture.write(directory/'decision.json', gate.evaluate_pair_review(owner, case, rows,
                gate.manifest_identity(owner), challenge, review, clock=lambda: f.now[0]))
        self.review_mode = 'ordinary'
        f.review_pair = self.review_pair
        fixture.write(f.path, f.manifest)
        self.old_hashes = {name: fixture.file_hashes(f.root/name) for name in ('N', 'R2', 'R5')}

    def review(self, challenge):
        review = self.f.review_for(challenge, labels=('partial_usable', 'failed'))
        for variant, label in challenge['anonymous_ids'].items():
            row = next(x for x in review['reviews'] if x['anonymous_id'] == label)
            if challenge['identity']['answers'][variant]['backend_status'] in self.policy['non_delivery_statuses']:
                row.update(whole_answer_quality='failed', core_complete=False,
                           requested_scope_complete=False, execution_complete=False)
            else:
                row['whole_answer_quality'] = 'partial_usable'
        return self.bind_proofs(challenge, review)

    def bind_proofs(self, challenge, review):
        directory = self.f.control/challenge['identity']['case_id']/'source-proofs'
        checks = [{'source_identity': 'synthetic-protocol-only', 'scope': 'fixture',
                   'evidence': 'This does not claim a real semantic review.'}]
        header = {'challenge_sha256': gate.sha256(gate.canonical(challenge)), 'blinded': True,
                  'full_authorized_source_checked': True, 'paid_model_calls': 0,
                  'source_checks': checks, 'reviews': deepcopy(review['reviews'])}
        review.update(source_checks=deepcopy(checks), adjudication_notes='Synthetic source adjudication only.')
        originals = []
        for name in ('one', 'two'):
            path = directory/('reviewer-'+name+'.json');identity = 'synthetic-independent-'+name
            fixture.write(path, {**deepcopy(header), 'reviewer_identity': identity})
            originals.append({'path': str(path), 'sha256': gate.sha256(path.read_bytes()), 'identity': identity})
        path = directory/'adjudication.json'
        fixture.write(path, {**deepcopy(header), 'reviewer_identity': 'synthetic-adjudicator',
                             'adjudication_notes': review['adjudication_notes']})
        review.update(independent_reviews=originals,
                      adjudication_source={'path': str(path), 'sha256': gate.sha256(path.read_bytes())})
        return review

    def review_pair(self, manifest, case, rows, identity, deadline):
        f = self.f
        f.review_calls.append(case)
        if case == self.deferred:
            self.assertEqual(f.child_calls, [], 'no new cell may start before retained review')
        directory = f.control/case
        def deliver(_delay):
            if self.review_mode == 'cancel':
                raise KeyboardInterrupt('synthetic review cancellation')
            challenge = fixture.read(directory/'challenge-private.json')
            review = self.review(challenge)
            if self.review_mode == 'unknown': review['reviews'][0]['whole_answer_quality'] = 'unknown'
            if self.review_mode == 'severe': review['reviews'][0]['forged_success'] = True
            self.bind_proofs(challenge, review)
            if self.review_mode == 'missing_proofs': review.pop('independent_reviews')
            fixture.write(directory/'review.json', review)
        return gate.wait_pair_review(manifest, case, rows, identity, deadline,
                                     clock=lambda: f.now[0], sleep=deliver)

    def pair(self):
        return gate.pair_identity(self.f.manifest, self.deferred, self.f.retained)

    def test_deferred_real_gate_before_all82_first_attempts_and_14_retained(self):
        f = self.f;result = f.run_driver()
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(f.child_calls, list(range(14, 96)))
        self.assertEqual(f.review_calls[0], self.deferred)
        self.assertEqual(len(f.review_calls), 42)
        self.assertEqual(result['rows'][:14], f.retained)
        self.assertEqual(result['total_root_attempts'], 99)
        self.assertEqual(result['new_roots_planned'], 82)
        self.assertTrue(all(fixture.read(f.control/'grants'/f'{i}.json')['attempt_kind'] == 'first_attempt'
                            for i in range(14, 96)))
        self.assertEqual(self.old_hashes, {name: fixture.file_hashes(f.root/name) for name in self.old_hashes})

    def test_empty_natural_failure_and_cancel_use_actual_empty_answer_not_synthetic_terminal(self):
        f = self.f;cell = Path(f.old_cells[tuple(f.order[13])]['cell_root'])
        for status in ('failed', 'cancelled'):
            f.write_cell(cell, f.owners['R5'], *f.order[13], f.retained[13]['root_id'], status, None,
                         [{'status': 'succeeded', 'request_id': 'old-response-13'}])
            with f.patches(): identity, answers, _ = self.pair()
            self.assertEqual(answers['B'], '')
            self.assertTrue(identity['answers']['B']['answer_absent'])
            self.assertEqual(identity['answers']['B']['answer_sha256'], gate.sha256(''))
            self.assertEqual(identity['answers']['B']['observed_status'], status)
        with f.patches():
            challenge = gate.prepare_pair_gate(f.manifest, self.deferred, f.retained,
                gate.manifest_identity(f.manifest), clock=lambda: f.now[0])
            review = self.review(challenge)
            self.assertEqual(gate.evaluate_pair_review(f.manifest, self.deferred, f.retained,
                gate.manifest_identity(f.manifest), challenge, review, clock=lambda: f.now[0])['status'], 'passed')
        packet = fixture.read(f.control/self.deferred/'review-packet.json')
        absent = next(x for x in packet['answers'] if x['answer_absent'])
        self.assertEqual(absent['final_answer'], '')
        self.assertNotIn('task_created_at', absent['observation_stop'])
        self.assertNotIn('task_finished_at', absent['observation_stop'])
        self.assertTrue(all('created_at' not in x for x in absent['observation_stop']['actual_stop_metadata']))

    def test_legacy_rejects_needs_input_and_v4_non_delivery_cannot_claim_full_or_complete(self):
        f = self.f
        with f.patches():
            old = deepcopy(f.manifest);old.pop('observation_policy');old['completion_contract']['version'] = 'legacy-v3'
            with self.assertRaisesRegex(PermissionError, 'ENDED_CELL_IDENTITY'):
                gate.pair_identity(old, self.deferred, f.retained)
            challenge = gate.prepare_pair_gate(f.manifest, self.deferred, f.retained,
                gate.manifest_identity(f.manifest), clock=lambda: f.now[0])
            for field, value in [('whole_answer_quality', 'full'), ('execution_complete', True)]:
                review = self.review(challenge)
                row = next(x for x in review['reviews'] if x['anonymous_id'] == challenge['anonymous_ids']['B'])
                row[field] = value
                with self.subTest(field=field), self.assertRaisesRegex(PermissionError, 'NON_DELIVERY'):
                    gate.evaluate_pair_review(f.manifest, self.deferred, f.retained, gate.manifest_identity(f.manifest),
                                              challenge, review, clock=lambda: f.now[0])

    def test_observation_policy_cannot_enable_non_fixed_suite_or_unreconciled_rows(self):
        f = self.f
        with f.patches():
            m = deepcopy(f.manifest);m.pop('final_suite_policy')
            with self.assertRaisesRegex(PermissionError, 'OBSERVATION_FIXED_SUITE'):
                gate.pair_identity(m, self.deferred, f.retained)
            rows = deepcopy(f.retained);rows[-1].pop('issues')
            with self.assertRaisesRegex(PermissionError, 'OBSERVATION_RECONCILIATION'):
                gate.pair_identity(f.manifest, self.deferred, rows)

    def test_lineage_marker_requires_explicit_policy_and_cannot_fall_back_to_legacy(self):
        f = self.f;m = deepcopy(f.manifest);root = f.root/'R5'
        m['completion_contract'] = {'version': 'agent_final_completion_v3',
            'observation_recovery_lineage': [{'anchor_root': str(root),
                'proof_path': str(root/'audit/measurement-gate-stop-proof.json'),
                'proof_sha256': '1'*64, 'case_id': self.deferred}]}
        with f.patches():
            identity, _, _ = gate.pair_identity(m, self.deferred, f.retained)
            self.assertEqual(identity['answers']['B']['observed_status'], 'needs_input')
            missing = deepcopy(m);missing.pop('observation_policy')
            empty = deepcopy(m);empty['completion_contract']['observation_recovery_lineage'] = []
            for value in (missing, empty):
                with self.assertRaises(PermissionError): gate.pair_identity(value, self.deferred, f.retained)

    def test_zero_receipts_only_belong_to_non_delivery_observation_stops(self):
        f = self.f;cell = Path(f.old_cells[tuple(f.order[13])]['cell_root'])
        for status in self.policy['statuses']:
            f.write_cell(cell, f.owners['R5'], *f.order[13], f.retained[13]['root_id'], status, None
                         if status in self.policy['non_delivery_statuses'] else 'Synthetic output.', [])
            result, view = fixture.read(cell/'result.json'), fixture.read(cell/'task-view.json')
            if status in self.policy['non_delivery_statuses']:
                self.assertEqual(gate.observation_stop(f.manifest, cell, result, view)['observed_status'], status)
            else:
                with self.subTest(status=status), self.assertRaisesRegex(PermissionError, 'RECEIPTS_REQUIRED'):
                    gate.observation_stop(f.manifest, cell, result, view)

    def test_independent_originals_and_adjudication_proofs_cannot_be_omitted_or_substituted(self):
        f = self.f
        with f.patches():
            challenge = gate.prepare_pair_gate(f.manifest, self.deferred, f.retained,
                                               gate.manifest_identity(f.manifest), clock=lambda: f.now[0])
            for change in ('missing', 'duplicate-path', 'duplicate-identity', 'changed-bytes', 'paid-bool',
                           'paid-float', 'challenge', 'answer', 'sources', 'empty-checks', 'adjudication-grades',
                           'adjudication-checks', 'adjudication-notes', 'input-path', 'outside-path', 'symlink'):
                review = self.review(challenge)
                ref = review['independent_reviews'][0];path = Path(ref['path']);raw = fixture.read(path)
                if change == 'missing': review.pop('independent_reviews')
                elif change == 'duplicate-path': review['independent_reviews'][1] = deepcopy(ref)
                elif change == 'duplicate-identity':
                    other = review['independent_reviews'][1];value = fixture.read(other['path'])
                    value['reviewer_identity'] = ref['identity'];fixture.write(other['path'], value)
                    other.update(identity=ref['identity'], sha256=gate.sha256(Path(other['path']).read_bytes()))
                elif change == 'changed-bytes': path.write_bytes(path.read_bytes()+b' ')
                elif change.startswith('adjudication-'):
                    ref = review['adjudication_source'];path = Path(ref['path']);raw = fixture.read(path)
                    if change == 'adjudication-grades': raw['reviews'][0]['whole_answer_quality'] = 'full'
                    elif change == 'adjudication-checks': raw['source_checks'][0]['scope'] = 'changed'
                    else: raw['adjudication_notes'] = 'changed'
                    fixture.write(path, raw);ref['sha256'] = gate.sha256(path.read_bytes())
                elif change in {'input-path', 'outside-path', 'symlink'}:
                    target = (f.root/'current/inputs/proof.json' if change == 'input-path'
                              else f.root/'external-proof.json' if change == 'outside-path'
                              else f.root/'current/source-proof-link.json')
                    if change == 'symlink': target.symlink_to(path)
                    else: fixture.write(target, raw)
                    ref['path'] = str(target)
                else:
                    if change == 'paid-bool': raw['paid_model_calls'] = False
                    elif change == 'paid-float': raw['paid_model_calls'] = 0.0
                    elif change == 'challenge': raw['challenge_sha256'] = 'wrong'
                    elif change == 'answer': raw['reviews'][0]['answer_sha256'] = 'wrong'
                    elif change == 'sources': raw['reviews'][0]['source_sha256s'] = ['wrong']
                    elif change == 'empty-checks': raw['source_checks'] = []
                    fixture.write(path, raw);ref['sha256'] = gate.sha256(path.read_bytes())
                with self.subTest(change=change), self.assertRaises(PermissionError):
                    gate.evaluate_pair_review(f.manifest, self.deferred, f.retained, gate.manifest_identity(f.manifest),
                                              challenge, review, clock=lambda: f.now[0])

    def test_missing_independent_deferred_proofs_prevent_first_grant_and_model(self):
        self.review_mode = 'missing_proofs';result = self.f.run_driver()
        self.assertEqual(self.f.child_calls, [])
        self.assertEqual(result['stop_reason'], 'deferred_semantic_pair_gate_invalid:PermissionError')
        self.assertFalse((self.f.control/'grants').exists())

    def test_missing_or_changed_event_identity_code_receipt_denial_rejected(self):
        f = self.f;cell = Path(f.old_cells[tuple(f.order[13])]['cell_root'])
        original = {name: (cell/name).read_bytes() for name in
                    ('result.json', 'task-view.json', 'events.json', 'model-calls.json', 'ledger-snapshot.json')}
        for change in ('root', 'prompt', 'loaded', 'event-status', 'missing-event', 'duplicate-event',
                       'native-unknown', 'physical-unknown', 'runtime-denial', 'queued', 'technical-result'):
            try:
                name = ('events.json' if 'event' in change else 'task-view.json' if change in {'prompt', 'queued'}
                        else 'model-calls.json' if change == 'native-unknown'
                        else 'ledger-snapshot.json' if change == 'physical-unknown' else 'result.json')
                value = fixture.read(cell/name)
                if change == 'root': value['root_id'] = 'wrong-root'
                elif change == 'prompt': value['prompt'] += 'changed'
                elif change == 'loaded': value['loaded_supervisor'] = {'wrong': 'code'}
                elif change == 'event-status': value[-1]['payload']['status'] = 'completed'
                elif change == 'missing-event': value.pop()
                elif change == 'duplicate-event': value.append(deepcopy(value[-1]))
                elif change == 'native-unknown': value[0]['status'] = 'unknown'
                elif change == 'physical-unknown': value['reservations'][0]['status'] = 'sent'
                elif change == 'runtime-denial': value['isolation']['denials'] = [{'phase': 'execution'}]
                elif change == 'queued': value['status'] = 'queued'
                elif change == 'technical-result': value['status'] = 'failed'
                fixture.write(cell/name, value)
                with f.patches(), self.subTest(change=change), self.assertRaises(PermissionError): self.pair()
            finally:
                for name, data in original.items(): (cell/name).write_bytes(data)

    def test_unknown_deferred_grade_stops_before_any_new_cell(self):
        self.review_mode = 'unknown';result = self.f.run_driver()
        self.assertEqual(self.f.child_calls, [])
        self.assertIn('review_unknown', result['stop_reason'])

    def test_severe_deferred_grade_stops_before_any_new_cell(self):
        self.review_mode = 'severe';result = self.f.run_driver()
        self.assertEqual(self.f.child_calls, [])
        self.assertIn('forged_success', result['stop_reason'])

    def test_review_cancel_closes_batch_without_any_new_cell(self):
        self.review_mode = 'cancel';result = self.f.run_driver()
        self.assertEqual(self.f.child_calls, [])
        self.assertEqual(result['stop_reason'], 'deferred_semantic_pair_gate_invalid:KeyboardInterrupt')

    def test_blind_stop_metadata_does_not_forward_nested_receipts_ids_cost_or_time(self):
        with self.f.patches(): identity, _, _ = self.pair()
        answer = identity['answers']['B']
        answer['observation_stop']['actual_stop_metadata'].append({'event_type': 'worker_failed',
            'created_at': '2026-01-01T00:00:00Z', 'payload': {'status': 'needs_input',
                'reason': answer['root_id'], 'question': '30 tokens in 2 yuan',
                'fields': ['synthetic field'], 'result': {'usage': 19, 'task_id': answer['root_id']},
                'error': {'duration': 3}, 'usage': {'total_tokens': 19}}})
        answer['observation_stop']['actual_stop_metadata'].append({'event_type': 'clarification_requested',
            'payload': {'reason': 'Token总量120，响应时间3.5秒，¥0.01', 'question': '本次Token用量120',
                        'fields': ['运行3.5秒'], 'suggested_values': ['支付¥0.01']}})
        projected = gate.review_observation(answer)
        payload = projected['observation_stop']['actual_stop_metadata'][-2]['payload']
        self.assertEqual(payload, {'status': 'needs_input', 'fields': ['synthetic field']})
        self.assertEqual(projected['observation_stop']['actual_stop_metadata'][-1]['payload'], {})
        self.assertNotIn(answer['root_id'], json.dumps(projected))
        self.assertNotIn('created_at', json.dumps(projected))

    def test_first_grant_requires_deferred_current_review_and_rejects_wrong_retry_identity(self):
        f = self.f
        with f.patches():
            f.start_control()
            case, variant = f.order[14];deadline = f.now[0]+900
            with self.assertRaises((FileNotFoundError, PermissionError)):
                gate.issue_cell_grant(f.manifest, case, variant, deadline)
            with patch.object(batch, 'wait_pair_review', side_effect=self.review_pair):
                self.assertIsNone(batch.review_deferred_pairs(f.manifest, f.context, f.retained,
                                  gate.manifest_identity(f.manifest), None))
            grant = gate.issue_cell_grant(f.manifest, case, variant, deadline)
            self.assertEqual(grant['attempt_kind'], 'first_attempt')
            grant['attempt_kind'] = 'technical_retry'
            fixture.write(f.control/'grants/14.json', grant)
            with self.assertRaises(PermissionError):
                gate.validate_prior_pair_gates({**f.manifest, 'cell_absolute_deadline': deadline}, case, variant,
                                               clock=lambda: f.now[0])

    def test_next_variant_accepts_actual_needs_input_stop_but_rejects_preceding_code_drift(self):
        f = self.f
        with f.patches():
            f.start_control()
            with patch.object(batch, 'wait_pair_review', side_effect=self.review_pair):
                batch.review_deferred_pairs(f.manifest, f.context, f.retained, gate.manifest_identity(f.manifest), None)
            case, variant = f.order[14];deadline = f.now[0]+900
            gate.issue_cell_grant(f.manifest, case, variant, deadline)
            cell = f.runtime/'cells'/(case+'-'+variant)
            f.write_cell(cell, f.manifest, case, variant, 'synthetic-paused-new', 'needs_input',
                         'Actual synthetic clarification stop.', [{'status': 'succeeded', 'request_id': 'paused-response'}])
            result = fixture.read(cell/'result.json');bad = deepcopy(result);bad['loaded_supervisor'] = {'wrong': 'code'}
            fixture.write(cell/'result.json', bad)
            with self.assertRaisesRegex(PermissionError, 'PRECEDING_OBSERVATION_SOURCE_IDENTITY'):
                gate.issue_cell_grant(f.manifest, *f.order[15], deadline)
            fixture.write(cell/'result.json', result)
            grant = gate.issue_cell_grant(f.manifest, *f.order[15], deadline)
            self.assertEqual(grant['attempt_kind'], 'first_attempt')


if __name__ == '__main__': unittest.main()
