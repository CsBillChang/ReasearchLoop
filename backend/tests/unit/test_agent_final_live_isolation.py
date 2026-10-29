"""隔离首拒绝的持久化、并发准入与在途收据测试；不连接真实服务。"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND / 'scripts'), str(BACKEND)]
from harness_ab_boundary import Boundary, IsolationError
from harness_ab_budget import BudgetHalt, Ledger
from harness_ab_transport import BudgetTransport
from research_loop.providers.deepseek import RequestNotSent
from run_read_paper_ab import LiveIsolationLatch

USAGE = {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7}
PAYLOAD = {'model': 'deepseek-flash', 'temperature': .1, 'max_tokens': 100,
           'messages': [{'role': 'user', 'content': 'synthetic request only'}]}


class IsolationLatchTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack();self.addCleanup(self.stack.close)
        self.temp = tempfile.TemporaryDirectory(prefix='isolation-latch-');self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.network = [self.stack.enter_context(patch.object(socket.socket, method,
            side_effect=AssertionError('real socket forbidden'))) for method in ('connect', 'connect_ex')]
        self.now = [2000.0]
        self.events = []
        self.latch = LiveIsolationLatch(lambda value: self.events.append(value), clock=lambda: self.now[0])
        self.ledger = Ledger(self.root / 'budget.sqlite3', allowed_cells=[['CASE', 'A']],
            clock=lambda: self.now[0], start_time=1., batch_seconds=10, enforce_batch_seconds=False,
            carried_usage={'tokens': 50, 'requests': 1, 'nano_yuan': 1000, 'seconds': 0})
        self.guarded = self.latch.bind(self.ledger, 'root')
        self.guarded.begin_root('root', 'CASE', 'A')

    def tearDown(self):
        for mock in self.network:
            self.assertEqual(mock.call_count, 0)

    def reserve(self):
        return self.guarded.reserve('root', max_output_tokens=100, input_upper_bound=100)

    def test_first_runtime_denial_is_persisted_once_and_blocks_new_reserve_and_send(self):
        pending = self.reserve()
        self.latch.record('execution', 'network_name_outside_experiment')
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot['first_error']['operation'], 'live_isolation')
        self.assertEqual(len(self.events), 1)
        with self.assertRaises(BudgetHalt):
            self.reserve()
        with self.assertRaises(BudgetHalt):
            self.guarded.mark_sent(pending['reservation_id'])
        self.latch.record('execution', 'second_denial')
        self.assertEqual(len(self.events), 1)
        self.assertIn('network_name_outside_experiment', self.ledger.snapshot()['halt_reason'])
        self.assertEqual(len(self.ledger.snapshot()['reservations']), 1)

    def test_guard_selfchecks_do_not_close_live_admission(self):
        for operation in ('dotenv_read', 'write_outside_output', 'network_name_outside_experiment'):
            self.latch.record('guard_selfcheck', operation)
        value = self.reserve()
        self.guarded.mark_sent(value['reservation_id'])
        self.guarded.settle(value['reservation_id'], USAGE)
        self.assertEqual(self.events, [])
        self.assertFalse(self.latch.snapshot()['closed'])
        self.assertIsNone(self.ledger.snapshot()['first_error'])

    def test_preledger_denial_has_artifact_and_halts_binding_before_root_creation(self):
        path = self.root / 'startup-first-error.json'
        def persist(value):
            with path.open('x') as handle:
                json.dump(value, handle)
        early = LiveIsolationLatch(persist, clock=lambda: self.now[0])
        early.record('execution', 'startup_configuration_read')
        self.assertTrue(path.is_file())
        ledger = Ledger(self.root / 'startup-budget.sqlite3', allowed_cells=[['CASE', 'A']])
        with self.assertRaises(BudgetHalt):
            early.bind(ledger, 'unstarted')
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot['roots'], [])
        self.assertEqual(snapshot['reservations'], [])
        self.assertEqual(snapshot['first_error']['operation'], 'live_isolation')
        self.assertTrue(early.snapshot()['ledger_persisted'])

    def test_inflight_success_and_unknown_remain_accounted_after_closure(self):
        first, second = self.reserve(), self.reserve()
        self.guarded.mark_sent(first['reservation_id']);self.guarded.mark_sent(second['reservation_id'])
        self.latch.record('execution', 'controlled_pdf_hash_changed')
        self.guarded.settle(first['reservation_id'], USAGE)
        self.guarded.unknown(second['reservation_id'], 'inflight outcome unconfirmed')
        with self.assertRaises(BudgetHalt):
            self.guarded.not_sent(second['reservation_id'], 'must not release sent')
        snapshot = self.ledger.snapshot()
        self.assertCountEqual([r['status'] for r in snapshot['reservations']], ['settled', 'unknown'])
        self.assertEqual(snapshot['charged_or_reserved']['tokens'], 50 + 7 + 200)
        self.assertEqual(snapshot['first_error']['operation'], 'live_isolation')

    def test_denial_after_request_journal_prevents_HTTP_and_releases_only_known_unsent(self):
        calls = []
        latch = self.latch
        class DenyAfterJournal(BudgetTransport):
            def _write(self, path, value):
                super()._write(path, value)
                if path.name.endswith('.request.json'):
                    latch.record('execution', 'write_outside_current_cell')
        transport = DenyAfterJournal(self.guarded, 'root', self.root / 'requests',
            inner=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(500)))
        with httpx.Client(transport=transport, trust_env=False) as client, self.assertRaises(RequestNotSent):
            client.post('https://api.deepseek.com/chat/completions', json=PAYLOAD)
        snapshot = self.ledger.snapshot()
        self.assertEqual(calls, [])
        self.assertEqual(snapshot['reservations'][0]['status'], 'not_sent')
        self.assertEqual(snapshot['charged_or_reserved']['tokens'], 50)
        self.assertEqual(snapshot['first_error']['operation'], 'live_isolation')

    def test_concurrent_inflight_response_settles_and_next_HTTP_is_not_admitted(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def respond(request):
            calls.append(request)
            entered.set()
            if not release.wait(3):
                raise AssertionError('inflight test was not released')
            return httpx.Response(200, json={'id': 'fake-response', 'choices': [], 'usage': USAGE})
        transport = BudgetTransport(self.guarded, 'root', self.root / 'requests', inner=httpx.MockTransport(respond))
        with httpx.Client(transport=transport, trust_env=False) as client, ThreadPoolExecutor(max_workers=2) as pool:
            running = pool.submit(client.post, 'https://api.deepseek.com/chat/completions', json=PAYLOAD)
            self.assertTrue(entered.wait(3))
            self.latch.record('execution', 'network_destination_outside_experiment')
            with self.assertRaises(RequestNotSent):
                client.post('https://api.deepseek.com/chat/completions', json=PAYLOAD)
            release.set()
            self.assertEqual(running.result(timeout=3).status_code, 200)
        self.assertEqual(len(calls), 1)
        self.assertEqual([r['status'] for r in self.ledger.snapshot()['reservations']], ['settled'])

    def test_closed_flag_wins_against_queued_concurrent_reservations(self):
        entered, release = threading.Event(), threading.Event()
        reserve = self.ledger.reserve
        def slow_reserve(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise AssertionError('admitted reservation not released')
            return reserve(*args, **kwargs)
        with patch.object(self.ledger, 'reserve', side_effect=slow_reserve), ThreadPoolExecutor(max_workers=3) as pool:
            admitted = pool.submit(self.reserve)
            self.assertTrue(entered.wait(3))
            denial = pool.submit(self.latch.record, 'execution', 'first_isolation_denial')
            self.assertTrue(self.latch.closed.wait(3))
            queued = pool.submit(self.reserve)
            release.set()
            old = admitted.result(timeout=3)
            denial.result(timeout=3)
            with self.assertRaises(BudgetHalt):
                queued.result(timeout=3)
        with self.assertRaises(BudgetHalt):
            self.guarded.mark_sent(old['reservation_id'])
        self.assertEqual(len(self.ledger.snapshot()['reservations']), 1)
        self.assertEqual(self.ledger.snapshot()['physical_sent_requests'], 0)

    def test_failed_artifact_writer_still_halts_ledger_and_new_admission(self):
        latch = LiveIsolationLatch(lambda _event: (_ for _ in ()).throw(OSError('synthetic full disk')))
        guarded = latch.bind(self.ledger, 'root')
        latch.record('execution', 'isolation_denied')
        with self.assertRaises(BudgetHalt):
            guarded.reserve('root', 100, 100)
        self.assertTrue(latch.snapshot()['closed'])
        self.assertFalse(latch.snapshot()['artifact_persisted'])
        self.assertTrue(latch.snapshot()['ledger_persisted'])
        self.assertTrue(latch.snapshot()['persistence_errors'])

    def test_recursive_audit_failure_during_persistence_does_not_deadlock(self):
        state = {}
        def persist(_event):
            state['latch'].record('execution', 'nested_logging_denial')
            raise IsolationError('synthetic nested audit')
        latch = LiveIsolationLatch(persist)
        state['latch'] = latch
        latch.bind(self.ledger, 'root')
        latch.record('execution', 'first_denial')
        self.assertEqual(latch.snapshot()['first_error']['operation'], 'first_denial')
        self.assertTrue(latch.snapshot()['ledger_persisted'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
