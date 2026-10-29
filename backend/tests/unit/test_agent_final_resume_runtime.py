"""92格续跑调度、配对门、原生收据与预算的零网络集成回归。"""
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
import harness_ab_pair_gate as gate
import run_read_paper_batch as batch
from agent_final_continuation import validate_continuation
from harness_ab_budget import BudgetHalt, Ledger
from harness_ab_transport import (BudgetTransport, bind_deep_reading_client,
    deep_reading_request_contract, visual_request_contract)
from research_loop.agents.deep_reading import _ReceiptClient, _RunInterrupted
from research_loop.config import Settings
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.storage.mysql import Database
from unit.test_agent_final_continuation import digest, make_fixture, write_json


USAGE = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
ATTEMPT_KEYS = ('attempt_id', 'attempt_kind', 'position', 'continuation_sha256')


class ContinuationRuntimeTests(unittest.TestCase):
    """真实控制器运行；只替换外部资格验证与子进程/最底层HTTP。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='final-resume-runtime-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.now = [time.time()]
        self.socket_guards = []
        for operation in ('connect', 'connect_ex'):
            guard = patch.object(socket.socket, operation,
                side_effect=AssertionError('real socket forbidden'))
            self.socket_guards.append(guard.start())
            self.addCleanup(guard.stop)
        self.fixture = make_fixture(self.root, started_at=self.now[0] - 40)
        self.manifest = self.fixture['manifest']
        self.runtime = Path(self.manifest['runtime_root'])
        self.runtime.mkdir()
        self.control = Path(self.manifest['semantic_gate_root'])
        self.order = self.fixture['order']
        self.clock = self.fixture['clock']
        self.manifest.update(batch_id='synthetic-continuation-runtime',
            real_execution_released=True, authorization_path=str(self.fixture['new_root'] / 'auth.json'),
            frozen_files=[{'path': self.manifest['cases_path'], 'sha256': digest(self.manifest['cases_path'])}])
        self.manifest_path = self.fixture['new_root'] / 'manifest.integration.json'
        self.refresh_authorization()
        self.anchor_hashes = {str(p.relative_to(self.fixture['anchor_root'])): digest(p)
            for p in self.fixture['anchor_root'].rglob('*') if p.is_file()}
        self.child_calls = []
        self.http_calls = []
        self.review_calls = []
        self.validation_calls = []
        self.child_mode = 'success'
        self.review_mode = 'approve'
        self.after_child = None

    def tearDown(self):
        for guard in self.socket_guards:
            self.assertEqual(guard.call_count, 0, 'test attempted a real socket connection')

    def refresh_authorization(self):
        """合成新授权仍通过真实activation/共享时钟校验，不借用旧approved。"""
        auth = {'approved': True, 'experiment_kind': 'prompt_bundle_v1',
            'batch_id': self.manifest['batch_id'], 'manifest_sha256': gate.manifest_identity(self.manifest),
            'ledger_limits': self.manifest['ledger_limits'], 'execution_order': self.manifest['execution_order'],
            'allowed_modes': ['execute'], 'expires_at': self.clock['model_absolute_deadline'],
            'round_started_at': self.clock['round_started_at'],
            'round_absolute_deadline': self.clock['round_absolute_deadline']}
        write_json(self.manifest['authorization_path'], auth)
        write_json(self.manifest_path, self.manifest)

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text())

    def ledger(self):
        return Ledger(self.runtime / 'budget.sqlite3', clock=lambda: self.now[0],
            **self.manifest['ledger_limits'])

    def assert_anchor_unchanged(self):
        actual = {str(p.relative_to(self.fixture['anchor_root'])): digest(p)
            for p in self.fixture['anchor_root'].rglob('*') if p.is_file()}
        self.assertEqual(actual, self.anchor_hashes)

    def run_driver(self, *, started_at=None):
        """运行真实main；合成资格替身不替换续跑合同、grant、对账或pair门。"""
        payload = {'api_key': 'synthetic-key-no-network',
            'started_at': self.clock['round_started_at'] if started_at is None else started_at}
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['run_read_paper_batch.py',
                '--manifest', str(self.manifest_path)]))
            stack.enter_context(patch.object(sys, 'stdin', io.StringIO(json.dumps(payload))))
            stack.enter_context(patch.object(batch.time, 'time', side_effect=lambda: self.now[0]))
            stack.enter_context(patch.object(batch, 'validate_manifest',
                side_effect=lambda m, c, v, mode: self.validation_calls.append((c, v, mode))))
            stack.enter_context(patch.object(batch.subprocess, 'run', side_effect=self.child))
            stack.enter_context(patch.object(batch, 'wait_pair_review', side_effect=self.review_pair))
            stack.enter_context(redirect_stdout(io.StringIO()))
            batch.main()
        return self.read(self.runtime / 'batch-results.json')

    def child(self, argv, **options):
        """合成格进程仍走真实原生收据、Provider、预算传输与内存HTTP。

        候选业务包是合同夹具的极小合成文件，不能冒充完整Worker资格。
        本层只模拟子进程交付，序号授权与控制器之后的对账均为原函数。
        """
        self.assertEqual(argv[1:3], ['-I', '-B'])
        self.assertEqual(json.loads(options['input'])['api_key'], 'synthetic-key-no-network')
        case = argv[argv.index('--case') + 1]
        variant = argv[argv.index('--variant') + 1]
        deadline = float(argv[argv.index('--deadline') + 1])
        position = self.order.index([case, variant])
        self.child_calls.append({'case_id': case, 'variant': variant, 'position': position,
            'deadline': deadline, 'timeout': options['timeout']})
        session = gate.validate_prior_pair_gates({**self.manifest, 'cell_absolute_deadline': deadline},
            case, variant, clock=lambda: self.now[0])
        grant = session['active_grant']
        self.assertLessEqual(deadline, self.clock['model_absolute_deadline'])
        self.assertGreater(options['timeout'], 0)
        self.assertLessEqual(options['timeout'], deadline - self.now[0])
        cell = self.runtime / 'cells' / (case + '-' + variant)
        cell.mkdir(parents=True, exist_ok=False)
        root_id = 'new-root-' + uuid4().hex
        ledger = self.ledger()
        ledger.begin_root(root_id, case, variant)
        db = Database('sqlite:///' + str(cell / 'task.sqlite3'))
        db.ensure_schema()
        settings = Settings(project_root=cell, data_dir=cell / 'data', output_dir=cell / 'output',
            database_url=db.url, deepseek_model='deepseek-flash', execution_budget_mode='observe',
            model_max_output_tokens=65536, context_output_reserve=8192)
        prompt = next(c['prompt'] for c in self.read(self.manifest['cases_path'])['cases'] if c['case_id'] == case)
        db.create_task(root_id, None, 'synthetic', prompt, 'deny', [], token_settings=settings)

        def respond(request):
            self.http_calls.append({'position': position, 'payload': json.loads(request.content)})
            if self.child_mode == 'timeout':
                raise httpx.ReadTimeout('synthetic in-flight timeout', request=request)
            response = {'id': 'synthetic-response-' + uuid4().hex, 'model': 'deepseek-flash',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'Synthetic answer; no quality claim.（完）'}}]}
            if self.child_mode != 'missing_usage':
                response['usage'] = USAGE
            return httpx.Response(200, json=response)

        transport = BudgetTransport(ledger, root_id, cell / 'requests',
            inner=httpx.MockTransport(respond), deadline=deadline,
            visual_contract=visual_request_contract(), deep_reading_contract=deep_reading_request_contract())
        state = {'task_id': root_id, 'root_task_id': root_id, 'max_model_requests': 48,
                 'execution_policy': db.execution_policy(root_id)}
        answer, status = 'Synthetic incomplete response.', 'partial'
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
                if self.child_mode not in {'missing_usage', 'timeout'}:
                    raise
        result = {'case_id': case, 'variant': variant, 'root_id': root_id,
            'status': 'executed', 'task_status': status,
            'experiment_attempt': {key: grant[key] for key in ATTEMPT_KEYS},
            'synthetic_cell_process_only': True}
        if self.child_mode == 'wrong_attempt':
            result['experiment_attempt']['attempt_id'] = '0' * 64
        write_json(cell / 'result.json', result)
        write_json(cell / 'task-view.json', {'task_id': root_id, 'status': status, 'prompt': prompt,
            'result': {'answer_markdown': answer}})
        write_json(cell / 'model-calls.json', db.list_model_calls(root_id))
        write_json(cell / 'actions.json', {})
        db.update_task(root_id, status=status, result_json=json.dumps({'answer_markdown': answer}))
        if self.after_child:
            self.after_child(position)
        if self.child_mode == 'timeout':
            raise subprocess.TimeoutExpired(argv, options['timeout'])
        return subprocess.CompletedProcess(argv, 0, stdout='synthetic child only\n', stderr='')

    def review_pair(self, manifest, case, rows, identity, deadline):
        """真实wait创建挑战，合成独立评语仅替代人工等待，不跳过evaluate。"""
        self.review_calls.append({'case_id': case, 'deadline': deadline})
        self.assertLessEqual(deadline, self.clock['model_absolute_deadline'])
        directory = self.control / case

        def deliver(_seconds):
            if self.review_mode == 'missing':
                self.now[0] = deadline
                return
            challenge = self.read(directory / 'challenge-private.json')
            sources = [m['pdf_sha256'] for m in challenge['identity']['materials']]
            review = {'version': 1, 'challenge_nonce': challenge['nonce'],
                'challenge_sha256': gate.sha256(gate.canonical(challenge)), 'reviewed_at': self.now[0],
                'reviewer': {'kind': 'agent_assisted', 'identity': 'synthetic runtime reviewer',
                    'blinded': True, 'full_authorized_source_checked': True, 'paid_model_calls': 0},
                'reviews': [{'anonymous_id': challenge['anonymous_ids'][variant],
                    'answer_sha256': challenge['identity']['answers'][variant]['answer_sha256'],
                    'source_sha256s': sources, 'whole_answer_quality': 'full',
                    'core_complete': True, 'critical_core_error': False, 'requested_scope_complete': True,
                    'execution_complete': True, 'unauthorized_delivery_write': False, 'forged_success': False,
                    'evidence_notes': 'Synthetic contract response; no actual answer quality grade.'}
                    for variant in ('A', 'B')]}
            gate.write_new(directory / 'review.json', review)

        decision = gate.wait_pair_review(manifest, case, rows, identity, deadline,
            clock=lambda: self.now[0], sleep=deliver)
        if self.review_mode == 'tamper':
            altered = deepcopy(decision)
            altered['review_sha256'] = '0' * 64
            write_json(directory / 'decision.json', altered)
        return decision

    def start_control(self):
        """只为直接grant回归建立同一新账本与控制会话。"""
        Ledger(self.runtime / 'budget.sqlite3', clock=lambda: self.now[0],
            start_time=self.clock['round_started_at'], **self.manifest['ledger_limits'])
        gate.initialize_scope_session(self.manifest, self.clock['round_started_at'], clock=lambda: self.now[0])

    def run_granted_cell(self, position, deadline):
        case, variant = self.order[position]
        return self.child([sys.executable, '-I', '-B', 'synthetic-cell.py', '--case', case,
            '--variant', variant, '--deadline', str(deadline)],
            input=json.dumps({'api_key': 'synthetic-key-no-network'}), timeout=deadline - self.now[0])

    def test_full_driver_executes_only_suffix_and_preserves_four_original_grades(self):
        result = self.run_driver()
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual([c['position'] for c in self.child_calls], list(range(4, 96)))
        self.assertEqual([list(c[:2]) for c in self.validation_calls], self.order[4:])
        self.assertEqual(len(self.http_calls), 92)
        grants = [self.read(self.control / 'grants' / f'{i}.json') for i in range(4, 96)]
        self.assertEqual({p.name for p in (self.control / 'grants').iterdir()}, {f'{i}.json' for i in range(4, 96)})
        self.assertEqual(len({g['attempt_id'] for g in grants}), 92)
        self.assertEqual(grants[0]['attempt_kind'], 'technical_retry')
        self.assertTrue(all(g['attempt_kind'] == 'first_attempt' for g in grants[1:]))
        retained = validate_continuation(self.manifest)['retained_rows']
        self.assertEqual(result['rows'][:4], retained)
        self.assertEqual([r['quality'] for r in result['rows'][:4]], ['full', 'partial_usable', 'partial_usable', 'full'])
        self.assertEqual(result['new_created_count'], 92)
        self.assertEqual(result['primary_measurement_cells'], 96)
        self.assertEqual(result['total_root_attempts'], 97)
        self.assertEqual(result['not_created'], [])
        self.assertNotEqual(result['rows'][4]['root_id'], result['excluded_technical_attempts'][0]['root_id'])
        self.assertEqual(result['excluded_technical_attempts'][0]['root_id'], 'old-root-4')
        snapshot = self.read(self.runtime / 'ledger-final.json')
        self.assertEqual(len(snapshot['roots']), 92)
        self.assertEqual(snapshot['physical_sent_requests'], 92)
        self.assertEqual(snapshot['actual_known_tokens'], 92 * 15)
        self.assertEqual(snapshot['charged_or_reserved']['tokens'], 115 + 92 * 15)
        self.assertEqual(snapshot['charged_or_reserved']['requests'], 15 + 92)
        self.assertEqual(snapshot['carried_usage']['seconds'], 10)
        self.assertEqual(snapshot['started_at'], self.clock['round_started_at'])
        self.assertEqual(snapshot['batch_absolute_deadline'], self.clock['round_absolute_deadline'])
        self.assertIsNone(snapshot['first_error'])
        self.assert_anchor_unchanged()

    def test_missing_first_new_pair_review_stops_before_any_next_pair_grant(self):
        self.review_mode = 'missing'
        result = self.run_driver()
        self.assertEqual([c['position'] for c in self.child_calls], [4, 5])
        self.assertEqual(result['new_created_count'], 2)
        self.assertEqual(result['retained_observations'], 4)
        self.assertIn('semantic_review_deadline', result['stop_reason'])
        self.assertFalse((self.control / 'grants/6.json').exists())
        self.assertEqual([c['case_id'] for c in self.review_calls], [self.order[4][0]])
        self.assertFalse((self.control / self.order[0][0]).exists())
        self.assert_anchor_unchanged()

    def test_grants_cannot_repeat_skip_retained_or_replace_preceding_new_cell(self):
        self.start_control()
        deadline = self.now[0] + 30
        with patch.object(gate.time, 'time', side_effect=lambda: self.now[0]):
            for position in (0, 3, 5, 6):
                with self.subTest(position=position), self.assertRaises(PermissionError):
                    gate.issue_cell_grant(self.manifest, *self.order[position], deadline)
            first = gate.issue_cell_grant(self.manifest, *self.order[4], deadline)
            with self.assertRaises(PermissionError):
                gate.issue_cell_grant(self.manifest, *self.order[4], deadline)
            with self.assertRaises((PermissionError, OSError)):
                gate.issue_cell_grant(self.manifest, *self.order[5], deadline)
            self.run_granted_cell(4, deadline)
            gate.issue_cell_grant(self.manifest, *self.order[5], deadline)
            self.run_granted_cell(5, deadline)
            with self.assertRaises((PermissionError, OSError)):
                gate.issue_cell_grant(self.manifest, *self.order[6], deadline)
        self.assertEqual(first['position'], 4)
        self.assertEqual(first['attempt_kind'], 'technical_retry')
        self.assertEqual(len(self.http_calls), 2)
        self.assertFalse((self.control / 'grants/6.json').exists())

    def test_old_authorization_or_reset_outer_clock_cannot_start_new_ledger(self):
        path = Path(self.manifest['authorization_path'])
        original = path.read_bytes()
        old_auth = self.read(self.fixture['anchor_root'] / 'formal-execution-auth.json')
        changed_hash = self.read(path); changed_hash['manifest_sha256'] = old_auth['manifest_sha256']
        for auth in (old_auth, changed_hash):
            with self.subTest(auth=auth.get('status', 'wrong-old-hash')):
                write_json(path, auth)
                with self.assertRaises(PermissionError):
                    self.run_driver()
                self.assertFalse((self.runtime / 'budget.sqlite3').exists())
        path.write_bytes(original)
        with self.assertRaises(PermissionError):
            self.run_driver(started_at=self.clock['round_started_at'] + 1)
        self.assertEqual(self.http_calls, [])
        self.assertFalse((self.runtime / 'started.json').exists())

    def test_old_score_or_hash_drift_stops_before_process_or_grant(self):
        for relative in ('scoring/S00-R1.reviewer-1.json',
                'semantic-gates/S01-R1/decision.json', 'runtime/cells/S02-R1-A/model-calls.json'):
            path = self.fixture['anchor_root'] / relative
            original = path.read_bytes()
            try:
                path.write_bytes(original + b' ')
                with self.subTest(relative=relative), self.assertRaises(PermissionError):
                    self.run_driver()
                self.assertFalse(self.control.exists())
                self.assertFalse((self.runtime / 'budget.sqlite3').exists())
            finally:
                path.write_bytes(original)
        self.assertEqual(self.child_calls, [])
        self.assertEqual(self.http_calls, [])

    def test_fresh_authorization_cannot_shift_shared_clock_or_extend_expiry(self):
        """新授权须绑定旧T0/total/modelcut，内部算式成立也不能移后窗口。"""
        path = Path(self.manifest['authorization_path'])
        original = path.read_bytes()
        auth = self.read(path)
        for change in ('shift-start-and-total', 'extend-expiry'):
            altered = deepcopy(auth)
            started = self.clock['round_started_at']
            if change == 'shift-start-and-total':
                altered['round_started_at'] += 1
                altered['round_absolute_deadline'] += 1
                started += 1
            else:
                altered['expires_at'] += 1
            try:
                write_json(path, altered)
                with self.subTest(change=change), self.assertRaisesRegex(
                        PermissionError, 'CONTINUATION_CLOCK_MISMATCH'):
                    self.run_driver(started_at=started)
                self.assertFalse((self.runtime / 'budget.sqlite3').exists())
                self.assertFalse((self.runtime / 'started.json').exists())
                self.assertFalse(self.control.exists())
            finally:
                path.write_bytes(original)
        self.assertEqual(self.child_calls, [])
        self.assertEqual(self.http_calls, [])

    def test_unknown_usage_first_error_preserves_reservation_and_never_retries(self):
        self.child_mode = 'missing_usage'
        result = self.run_driver()
        self.assertEqual([c['position'] for c in self.child_calls], [4])
        self.assertEqual(result['stop_reason'], 'cell_receipt_or_isolation_gate')
        snapshot = self.read(self.runtime / 'ledger-final.json')
        self.assertEqual(snapshot['reservations'][0]['status'], 'unknown')
        self.assertEqual(snapshot['actual_known_tokens'], 0)
        self.assertEqual(snapshot['first_error']['root_id'], result['rows'][4]['root_id'])
        self.assertEqual(snapshot['current_charged_or_reserved']['tokens'], 1048576 + 4096)
        self.assertFalse((self.control / 'grants/5.json').exists())
        with self.assertRaises(PermissionError):
            self.run_driver()
        self.assertEqual(len(self.http_calls), 1)
        self.assert_anchor_unchanged()

    def test_model_cutoff_finalizes_without_restarting_total_clock_or_sending_b(self):
        self.after_child = lambda _: self.now.__setitem__(0, self.clock['model_absolute_deadline'])
        result = self.run_driver()
        self.assertEqual(result['stop_reason'], 'original_model_deadline')
        self.assertEqual([c['position'] for c in self.child_calls], [4])
        self.assertEqual(result['started_at'], self.clock['round_started_at'])
        self.assertEqual(result['execution_seconds'], self.clock['model_absolute_deadline'] - self.clock['round_started_at'])
        snapshot = self.read(self.runtime / 'ledger-final.json')
        self.assertEqual(snapshot['batch_absolute_deadline'], self.clock['round_absolute_deadline'])
        self.assertEqual(snapshot['carried_usage']['seconds'], self.clock['pre_round_carried_seconds'])
        self.assertEqual(snapshot['actual_known_tokens'], 15)
        self.assertEqual(snapshot['halt_reason'], 'original_model_deadline')
        self.assertEqual(snapshot['first_error']['reason'], 'original_model_deadline')
        self.assertEqual(snapshot['first_error']['operation'], 'continuation_controller')
        self.assertEqual(snapshot['first_error']['root_id'], result['rows'][4]['root_id'])
        ledger = self.ledger()
        with self.assertRaisesRegex(BudgetHalt, 'batch halted: original_model_deadline'):
            ledger.reserve(result['rows'][4]['root_id'], input_upper_bound=1, max_output_tokens=1)
        after = ledger.snapshot()
        self.assertEqual(after['first_error'], snapshot['first_error'])
        self.assertEqual(after['reservations'], snapshot['reservations'])
        self.assertEqual(len(self.http_calls), 1)
        self.assertFalse((self.control / 'grants/5.json').exists())
        self.assertTrue((self.runtime / 'progress.json').exists())
        self.assert_anchor_unchanged()

    def test_expired_original_model_cutoff_cannot_create_marker_or_ledger(self):
        self.now[0] = self.clock['model_absolute_deadline']
        with self.assertRaisesRegex(PermissionError, 'original model deadline'):
            self.run_driver()
        self.assertFalse((self.runtime / 'started.json').exists())
        self.assertFalse((self.runtime / 'budget.sqlite3').exists())
        self.assertEqual(self.http_calls, [])

    def test_completed_physical_receipt_does_not_hide_attempt_identity_mismatch(self):
        self.child_mode = 'wrong_attempt'
        result = self.run_driver()
        self.assertEqual(result['stop_reason'], 'cell_receipt_or_isolation_gate')
        self.assertIn('new_attempt_identity_mismatch', result['rows'][4]['issues'])
        self.assertEqual(len(self.http_calls), 1)
        self.assertFalse((self.control / 'grants/5.json').exists())
        before = digest(self.runtime / 'budget.sqlite3')
        with self.assertRaises(PermissionError):
            self.run_driver()
        self.assertEqual(digest(self.runtime / 'budget.sqlite3'), before)
        self.assertEqual(len(self.http_calls), 1)

    def test_parent_timeout_keeps_unknown_raw_root_and_single_retry_attempt(self):
        self.child_mode = 'timeout'
        result = self.run_driver()
        self.assertEqual(result['stop_reason'], 'absolute_cell_or_batch_deadline')
        row = result['rows'][4]
        self.assertEqual(row['status'], 'interrupted')
        self.assertEqual(row['original_position'], 4)
        self.assertEqual(row['attempt_kind'], 'technical_retry')
        self.assertEqual(len(row['attempt_id']), 64)
        self.assertIsNone(row['root_id'])
        snapshot = self.read(self.runtime / 'ledger-final.json')
        self.assertEqual(len(snapshot['roots']), 1)
        self.assertEqual(snapshot['reservations'][0]['status'], 'unknown')
        self.assertEqual(snapshot['first_error']['root_id'], snapshot['roots'][0]['root_id'])
        self.assertEqual(snapshot['started_at'], self.clock['round_started_at'])
        self.assertEqual(snapshot['batch_absolute_deadline'], self.clock['round_absolute_deadline'])
        self.assertEqual(len(self.http_calls), 1)
        self.assertFalse((self.control / 'grants/5.json').exists())
        self.assert_anchor_unchanged()

    def test_tampered_new_pair_score_blocks_next_pair_even_with_passed_status(self):
        self.review_mode = 'tamper'
        result = self.run_driver()
        self.assertEqual(result['stop_reason'], 'cell_scheduler_grant_invalid')
        self.assertEqual([c['position'] for c in self.child_calls], [4, 5])
        decision = self.read(self.control / self.order[4][0] / 'decision.json')
        self.assertEqual(decision['status'], 'passed')
        self.assertEqual(decision['review_sha256'], '0' * 64)
        self.assertFalse((self.control / 'grants/6.json').exists())
        self.assertEqual(len(self.http_calls), 2)
        self.assert_anchor_unchanged()


if __name__ == '__main__':
    unittest.main()
