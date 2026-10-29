"""11格保留、混合配对和85格补测的零网络控制链回归。"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND / 'scripts'), str(BACKEND), str(BACKEND / 'tests')]
import agent_final_continuation as continuation_module
import harness_ab_pair_gate as gate
import prompt_bundle_ab
import run_read_paper_batch as batch
from harness_ab_budget import Ledger
from harness_ab_transport import BudgetTransport, bind_deep_reading_client, deep_reading_request_contract, visual_request_contract
from research_loop.agents.deep_reading import _ReceiptClient, _RunInterrupted
from research_loop.config import Settings
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.storage.mysql import Database

USAGE = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
ATTEMPT_KEYS = ('attempt_id', 'attempt_kind', 'position', 'continuation_sha256')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def file_hashes(root):
    return {str(p.relative_to(root)): gate.sha256(p.read_bytes()) for p in root.rglob('*') if p.is_file()}


class CompletionRuntimeTests(unittest.TestCase):
    """实际driver/grant/收据/预算/配对门运行，仅模拟锚资格和最底层交付。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='completion-runtime-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.now = [time.time()]
        self.start = self.now[0] - 100000
        self.sockets = []
        for method in ('connect', 'connect_ex'):
            guard = patch.object(socket.socket, method, side_effect=AssertionError('real socket forbidden'))
            self.sockets.append(guard.start())
            self.addCleanup(guard.stop)
        self.cases = [{'case_id': f'AFE-MEM-{i:02d}-R{r}', 'suite_kind': 'agent_final_v1',
            'prompt': f'Synthetic Memory task {i} round {r}', 'materials': [],
            'preconditions': {'catalog': [], 'save_policy': 'deny',
                'history_messages': [{'role': 'user', 'content': 'Synthetic history'}]}}
            for r in (1, 2) for i in range(1, 25)]
        self.order = [[case['case_id'], variant] for i, case in enumerate(self.cases)
                      for variant in (('A', 'B') if i % 2 == 0 else ('B', 'A'))]
        self.owners = {}
        for name in ('N', 'R2', 'current'):
            root = self.root / name
            write(root / 'cases.json', {'cases': self.cases})
            self.owners[name] = {'experiment_kind': 'prompt_bundle_v1',
                'cases_path': str(root / 'cases.json'), 'runtime_root': str(root / 'runtime'),
                'semantic_gate_root': str(root / 'control'), 'execution_order': self.order,
                'final_suite_policy': deepcopy(gate.FINAL_SUITE_POLICY),
                'pair_review_policy': {'poll_seconds': .01},
                'test_source_contract': {'owner': name},
                'loaded_prompt_contract': {v: {'owner': name, 'variant': v} for v in ('A', 'B')}}
        self.manifest = self.owners['current']
        self.runtime = Path(self.manifest['runtime_root'])
        self.runtime.mkdir()
        self.control = Path(self.manifest['semantic_gate_root'])
        self.limits = {'allowed_cells': self.order, 'batch_tokens': 10_000_000,
            'batch_requests': 200, 'batch_yuan': '20', 'batch_seconds': 20,
            'root_tokens': 4_000_000, 'root_requests': 32, 'root_seconds': 900,
            'max_output_tokens': 65536, 'enforce_batch_seconds': False,
            'carried_usage': {'tokens': 137, 'requests': 17, 'nano_yuan': 97000, 'seconds': 10}}
        self.manifest.update(completion_contract={'version': 'synthetic-completion-v2'},
            ledger_limits=self.limits, batch_id='synthetic-completion',
            authorization_path=str(self.root / 'current/auth.json'),
            frozen_files=[{'path': self.manifest['cases_path'],
                           'sha256': gate.sha256(Path(self.manifest['cases_path']).read_bytes())}])
        self.retained = []
        self.old_cells = {}
        for position, (case, variant) in enumerate(self.order[:11]):
            owner = self.owners['N' if position < 4 else 'R2']
            root_id = f'original-root-{position}'
            cell = Path(owner['runtime_root']) / 'cells' / (case + '-' + variant)
            self.write_cell(cell, owner, case, variant, root_id, 'completed', 'Synthetic retained answer.（完）',
                [{'status': 'succeeded', 'request_id': f'old-response-{position}'}])
            quality = None if position == 10 else ('full' if position % 3 == 0 else 'partial_usable')
            row = {'case_id': case, 'variant': variant, 'root_id': root_id, 'status': 'completed',
                'issues': [], 'original_position': position, 'retained': True, 'quality': quality}
            self.retained.append(row)
            self.old_cells[(case, variant)] = {'cell_root': str(cell), 'source_manifest': owner,
                'original_position': position, 'quality': quality, 'physical_receipts_verified': True,
                'native_status_counts': {'succeeded': 1}, 'physical_status_counts': {'settled': 1}}
        self.context = {'retained_count': 11, 'retained_rows': self.retained,
            'full_execution_order': self.order, 'new_execution_order': self.order[11:],
            'carried_usage': self.limits['carried_usage'],
            'clock': {'policy': 'financial_only', 'round_started_at': self.start,
                      'model_absolute_deadline': None, 'round_absolute_deadline': None},
            'technical_attempts': [{'root_id': 'old-technical-failure'},
                                   {'root_id': 'old-interrupted-root', 'pending_requests': 2}]}
        self.path = self.root / 'current/manifest.json'
        write(self.manifest['authorization_path'], {'approved': True, 'ledger_limits': self.limits})
        write(self.path, self.manifest)
        self.child_calls = []
        self.http_calls = []
        self.review_calls = []
        self.mode = 'success'
        self.review_mode = 'approve'
        with self.patches():
            for offset in range(0, 10, 2):
                owner = self.owners['N' if offset < 4 else 'R2']
                case = self.order[offset][0]
                old_rows = self.retained[offset:offset + 2]
                challenge = gate.prepare_pair_gate(owner, case, old_rows,
                    gate.manifest_identity(owner), clock=lambda: self.now[0] - 10)
                review = self.review_for(challenge, labels=('partial_usable', 'full'), at=self.now[0] - 5)
                directory = Path(owner['semantic_gate_root']) / case
                write(directory / 'review.json', review)
                decision = gate.evaluate_pair_review(owner, case, old_rows, gate.manifest_identity(owner),
                    challenge, review, clock=lambda: self.now[0])
                write(directory / 'decision.json', decision)
        self.old_hashes = {name: file_hashes(self.root / name) for name in ('N', 'R2')}

    def tearDown(self):
        for guard in self.sockets:
            self.assertEqual(guard.call_count, 0)

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(continuation_module, 'validate_continuation', return_value=self.context))
        stack.enter_context(patch.object(batch, 'validate_continuation', return_value=self.context))
        stack.enter_context(patch.object(batch, 'planned_order', side_effect=lambda _: deepcopy(self.order[11:])))
        stack.enter_context(patch.object(batch, 'retained_rows', side_effect=lambda _: deepcopy(self.retained)))
        stack.enter_context(patch.object(gate, 'paired_source_contract', side_effect=lambda m: m['test_source_contract']))
        stack.enter_context(patch.object(gate, '_retained_cell', side_effect=self.retained_cell))
        stack.enter_context(patch.object(gate, '_prior_pair_manifest', side_effect=self.prior_manifest))
        stack.enter_context(patch.object(prompt_bundle_ab, 'validate_outer_start',
            side_effect=lambda _m, started: self.assertEqual(started, self.start)))
        stack.enter_context(patch.object(gate.time, 'time', side_effect=lambda: self.now[0]))
        return stack

    def retained_cell(self, manifest, case, variant):
        return self.old_cells.get((case, variant)) if 'completion_contract' in manifest else None

    def prior_manifest(self, _manifest, offset, _context):
        self.assertIn(offset, range(0, 10, 2), 'mixed/new pair cannot reuse an old approval')
        return self.owners['N' if offset < 4 else 'R2']

    def write_cell(self, cell, owner, case, variant, root, status, answer, native, attempt=None):
        prompt = next(c['prompt'] for c in self.cases if c['case_id'] == case)
        result = {'status': 'executed', 'root_id': root, 'case_id': case, 'variant': variant,
            'task_status': status, 'loaded_supervisor': owner['loaded_prompt_contract'][variant]}
        if attempt:
            result['experiment_attempt'] = attempt
        for name, value in {'result.json': result, 'task-view.json': {'task_id': root, 'status': status,
            'prompt': prompt, 'result': {'answer_markdown': answer}},
            'model-calls.json': native, 'actions.json': {}}.items():
            write(cell / name, value)

    def ledger(self):
        return Ledger(self.runtime / 'budget.sqlite3', clock=lambda: self.now[0], **self.limits)

    def review_for(self, challenge, *, labels=('full', 'full'), at=None):
        return {'version': 1, 'challenge_nonce': challenge['nonce'],
            'challenge_sha256': gate.sha256(gate.canonical(challenge)),
            'reviewed_at': self.now[0] if at is None else at,
            'reviewer': {'kind': 'agent_assisted', 'identity': 'synthetic-contract-review',
                'blinded': True, 'full_authorized_source_checked': True, 'paid_model_calls': 0},
            'reviews': [{'anonymous_id': challenge['anonymous_ids'][v],
                'answer_sha256': challenge['identity']['answers'][v]['answer_sha256'],
                'source_sha256s': [], 'whole_answer_quality': label, 'core_complete': True,
                'critical_core_error': False, 'requested_scope_complete': True,
                'execution_complete': True, 'unauthorized_delivery_write': False, 'forged_success': False,
                'evidence_notes': 'Synthetic only; no real semantic grade.'}
                for v, label in zip(('A', 'B'), labels)]}

    def child(self, argv, **options):
        case = argv[argv.index('--case') + 1]
        variant = argv[argv.index('--variant') + 1]
        deadline = float(argv[argv.index('--deadline') + 1])
        position = self.order.index([case, variant])
        self.child_calls.append(position)
        self.assertEqual(argv[1:3], ['-I', '-B'])
        self.assertGreater(deadline, self.now[0])
        self.assertLessEqual(deadline, self.now[0] + 900)
        session = gate.validate_prior_pair_gates({**self.manifest, 'cell_absolute_deadline': deadline},
            case, variant, clock=lambda: self.now[0])
        grant = session['active_grant']
        cell = self.runtime / 'cells' / (case + '-' + variant)
        cell.mkdir(parents=True, exist_ok=False)
        root_id = 'new-' + uuid4().hex
        ledger = self.ledger()
        ledger.begin_root(root_id, case, variant)
        db = Database('sqlite:///' + str(cell / 'task.sqlite3'))
        db.ensure_schema()
        settings = Settings(project_root=cell, data_dir=cell / 'data', output_dir=cell / 'output',
            database_url=db.url, deepseek_model='deepseek-flash', execution_budget_mode='observe',
            model_max_output_tokens=65536, context_output_reserve=8192)
        prompt = next(c['prompt'] for c in self.cases if c['case_id'] == case)
        db.create_task(root_id, None, 'synthetic', prompt, 'deny', [], token_settings=settings)

        def respond(request):
            self.http_calls.append(position)
            response = {'id': 'response-' + uuid4().hex, 'model': 'deepseek-flash',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'Synthetic answer.（完）'}}]}
            if self.mode != 'missing_usage':
                response['usage'] = USAGE
            return httpx.Response(200, json=response)

        transport = BudgetTransport(ledger, root_id, cell / 'requests',
            inner=httpx.MockTransport(respond), deadline=deadline,
            visual_contract=visual_request_contract(), deep_reading_contract=deep_reading_request_contract())
        state = {'task_id': root_id, 'root_task_id': root_id, 'max_model_requests': 48,
            'execution_policy': db.execution_policy(root_id)}
        answer, status = 'Synthetic interrupted response.', 'partial'
        with httpx.Client(transport=transport, trust_env=False) as http:
            provider = DeepSeekProvider('synthetic-key-no-network', model='deepseek-flash', client=http)
            native = _ReceiptClient(state, {'base_sequence': 0}, provider, db,
                lambda update: state.update(update))
            native.kwargs.update(max_tokens=4096, temperature=.3)
            try:
                with bind_deep_reading_client(_ReceiptClient):
                    response = native.chat([{'role': 'user', 'content': prompt}])
                answer, status = response['content'], 'completed'
            except _RunInterrupted:
                if self.mode != 'missing_usage':
                    raise
        self.write_cell(cell, self.manifest, case, variant, root_id, status, answer,
            db.list_model_calls(root_id), {k: grant[k] for k in ATTEMPT_KEYS})
        db.update_task(root_id, status=status, result_json=json.dumps({'answer_markdown': answer}))
        self.now[0] += 1
        return subprocess.CompletedProcess(argv, 0, stdout='synthetic only', stderr='')

    def review_pair(self, manifest, case, rows, identity, deadline):
        self.review_calls.append(case)
        self.assertIsNone(deadline)
        directory = self.control / case

        def deliver(_delay):
            if self.review_mode == 'cancel':
                raise KeyboardInterrupt('synthetic reviewer cancellation')
            challenge = read(directory / 'challenge-private.json')
            review = self.review_for(challenge, labels=('partial_usable', 'partial_usable'))
            if self.review_mode == 'severe':
                review['reviews'][0]['unauthorized_delivery_write'] = True
            write(directory / 'review.json', review)

        decision = gate.wait_pair_review(manifest, case, rows, identity, deadline,
            clock=lambda: self.now[0], sleep=deliver)
        if self.review_mode == 'tamper':
            altered = deepcopy(decision);altered['review_sha256'] = '0' * 64
            write(directory / 'decision.json', altered)
        return decision

    def run_driver(self):
        with self.patches(), ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['driver', '--manifest', str(self.path)]))
            stack.enter_context(patch.object(sys, 'stdin', io.StringIO(json.dumps({
                'api_key': 'synthetic-key-no-network', 'started_at': self.start}))))
            stack.enter_context(patch.object(batch, 'validate_manifest'))
            stack.enter_context(patch.object(batch.subprocess, 'run', side_effect=self.child))
            stack.enter_context(patch.object(batch, 'wait_pair_review', side_effect=self.review_pair))
            stack.enter_context(redirect_stdout(io.StringIO()))
            batch.main()
        return read(self.runtime / 'batch-results.json')

    def start_control(self):
        Ledger(self.runtime / 'budget.sqlite3', clock=lambda: self.now[0], start_time=self.start, **self.limits)
        gate.initialize_scope_session(self.manifest, self.start, clock=lambda: self.now[0])

    def assert_old_unchanged(self):
        self.assertEqual({n: file_hashes(self.root / n) for n in ('N', 'R2')}, self.old_hashes)

    def test_all85_new_native_receipts_and_43_pairs_keep_11_old_observations(self):
        result = self.run_driver()
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(self.child_calls, list(range(11, 96)))
        self.assertEqual(self.http_calls, list(range(11, 96)))
        self.assertEqual(len(self.review_calls), 43)
        self.assertEqual(result['rows'][:11], self.retained)
        self.assertEqual(result['new_roots_planned'], 85)
        self.assertEqual(result['total_root_attempts'], 98)
        self.assertEqual(result['primary_measurement_cells'], 96)
        self.assertEqual(len(result['excluded_technical_attempts']), 2)
        self.assertEqual(result['not_created'], [])
        grants = [read(self.control / 'grants' / f'{i}.json') for i in range(11, 96)]
        self.assertEqual({p.name for p in (self.control / 'grants').iterdir()}, {f'{i}.json' for i in range(11, 96)})
        self.assertEqual(grants[0]['attempt_kind'], 'technical_retry')
        self.assertTrue(all(g['attempt_kind'] == 'first_attempt' for g in grants[1:]))
        self.assertEqual(len({g['attempt_id'] for g in grants}), 85)
        self.assertEqual(grants[0]['continuation_sha256'], gate.sha256(gate.canonical(self.manifest['completion_contract'])))
        first = read(self.control / self.order[11][0] / 'challenge-private.json')
        old_b = first['identity']['answers']['B']['source_provenance']
        new_a = first['identity']['answers']['A']['source_provenance']
        self.assertTrue(old_b['retained']);self.assertFalse(new_a['retained'])
        self.assertEqual(old_b['cell_root'], self.old_cells[tuple(self.order[10])]['cell_root'])
        self.assertEqual(old_b['manifest_sha256'], gate.manifest_identity(self.owners['R2']))
        self.assertEqual(new_a['manifest_sha256'], gate.manifest_identity(self.manifest))
        self.assertNotEqual(old_b['source_contract_sha256'], new_a['source_contract_sha256'])
        snapshot = read(self.runtime / 'ledger-final.json')
        self.assertEqual(len(snapshot['roots']), 85)
        self.assertEqual(snapshot['actual_known_tokens'], 85 * 15)
        self.assertEqual(snapshot['charged_or_reserved']['tokens'], 137 + 85 * 15)
        self.assertEqual(snapshot['charged_or_reserved']['requests'], 17 + 85)
        self.assertEqual(snapshot['started_at'], self.start)
        self.assertIsNone(snapshot['batch_absolute_deadline'])
        self.assertIsNone(snapshot['halt_reason'])
        self.assert_old_unchanged()

    def test_retained_receipt_failure_blocks_first_retry_before_model(self):
        self.old_cells[tuple(self.order[10])]['physical_receipts_verified'] = False
        result = self.run_driver()
        self.assertEqual(self.http_calls, [])
        self.assertEqual(result['stop_reason'], 'cell_scheduler_grant_invalid')
        self.assertEqual(len(result['rows']), 11)
        self.assertTrue(read(self.runtime / 'ledger-final.json')['halt_reason'])
        self.assert_old_unchanged()

    def test_old_review_tamper_blocks_retry_without_resigning_old_scores(self):
        path = Path(self.owners['R2']['semantic_gate_root']) / self.order[8][0] / 'decision.json'
        value = read(path);value['review_sha256'] = '0' * 64;write(path, value)
        result = self.run_driver()
        self.assertEqual(self.http_calls, [])
        self.assertEqual(result['stop_reason'], 'cell_scheduler_grant_invalid')

    def test_new_unknown_usage_preserves_reservation_and_stops_after_one_root(self):
        self.mode = 'missing_usage'
        result = self.run_driver()
        self.assertEqual(self.child_calls, [11])
        self.assertEqual(self.http_calls, [11])
        self.assertEqual(result['new_attempted_count'], 1)
        snapshot = read(self.runtime / 'ledger-final.json')
        self.assertEqual(snapshot['reservations'][0]['status'], 'unknown')
        self.assertGreater(snapshot['charged_or_reserved']['tokens'], 137)
        self.assertIsNotNone(snapshot['first_error'])
        self.assert_old_unchanged()

    def test_retained_answer_cannot_adopt_new_source_or_prompt_identity(self):
        cell = Path(self.old_cells[tuple(self.order[10])]['cell_root'])
        result = read(cell / 'result.json')
        result['loaded_supervisor'] = self.manifest['loaded_prompt_contract']['B']
        write(cell / 'result.json', result)
        with self.patches(), self.assertRaisesRegex(PermissionError, 'LOADED_SOURCE_IDENTITY'):
            gate.pair_identity(self.manifest, self.order[10][0], [
                self.retained[10], self.synthetic_new_a()])

    def synthetic_new_a(self):
        case, variant = self.order[11]
        root_id = 'synthetic-mixed-new-a'
        cell = self.runtime / 'cells' / (case + '-' + variant)
        self.write_cell(cell, self.manifest, case, variant, root_id, 'completed',
            'Synthetic new answer.', [{'status': 'succeeded', 'request_id': 'new-response'}])
        return {'case_id': case, 'variant': variant, 'root_id': root_id, 'issues': []}

    def test_old_answer_copy_cannot_replace_retained_artifact_origin(self):
        current_cell = self.runtime / 'cells' / (self.order[10][0] + '-B')
        self.write_cell(current_cell, self.manifest, self.order[10][0], 'B',
            self.retained[10]['root_id'], 'completed', 'Copied replacement answer.',
            [{'status': 'succeeded', 'request_id': 'copied-response'}])
        with self.patches():
            identity, answers, _case = gate.pair_identity(self.manifest, self.order[10][0], [
                self.retained[10], self.synthetic_new_a()])
        self.assertEqual(answers['B'], 'Synthetic retained answer.（完）')
        self.assertEqual(identity['answers']['B']['source_provenance']['cell_root'],
            self.old_cells[tuple(self.order[10])]['cell_root'])
        self.assert_old_unchanged()

    def test_severe_review_stops_but_ordinary_partial_did_not_stop_full_chain(self):
        self.review_mode = 'severe'
        result = self.run_driver()
        self.assertEqual(self.child_calls, [11])
        self.assertIn('unauthorized_delivery_write', result['stop_reason'])
        self.assertIsNotNone(read(self.runtime / 'ledger-final.json')['first_error'])

    def test_mixed_new_review_tamper_prevents_next_pair(self):
        self.review_mode = 'tamper'
        result = self.run_driver()
        self.assertEqual(self.child_calls, [11])
        self.assertEqual(result['stop_reason'], 'cell_scheduler_grant_invalid')

    def test_unbounded_review_wait_can_be_cancelled_without_next_model(self):
        self.review_mode = 'cancel'
        result = self.run_driver()
        self.assertEqual(self.child_calls, [11])
        self.assertEqual(result['stop_reason'], 'semantic_pair_gate_invalid:KeyboardInterrupt')
        self.assertIsNotNone(read(self.runtime / 'ledger-final.json')['halt_reason'])

    def test_financial_only_grants_are_finite_root_bounded_and_not_replayable(self):
        with self.patches():
            self.start_control()
            case, variant = self.order[11]
            for invalid in (None, float('inf'), True, self.now[0] + 901):
                with self.subTest(invalid=invalid), self.assertRaises(PermissionError):
                    gate.issue_cell_grant(self.manifest, case, variant, invalid)
            grant = gate.issue_cell_grant(self.manifest, case, variant, self.now[0] + 900)
            self.assertEqual(grant['position'], 11)
            with self.assertRaises(PermissionError):
                gate.issue_cell_grant(self.manifest, case, variant, self.now[0] + 900)
            with self.assertRaises(PermissionError):
                gate.issue_cell_grant(self.manifest, *self.order[13], self.now[0] + 900)

    def test_no_time_cap_cannot_be_inferred_for_legacy_contract(self):
        manifest = deepcopy(self.manifest);manifest.pop('completion_contract')
        with patch.object(continuation_module, 'validate_continuation', return_value=None):
            with self.assertRaisesRegex(PermissionError, 'FINANCIAL_ONLY'):
                gate.wait_pair_review(manifest, 'unused', [], 'unused', None)


if __name__ == '__main__':
    unittest.main()
