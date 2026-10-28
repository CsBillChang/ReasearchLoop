"""首错与发送放行共用持久门；所有响应均为内存夹具。"""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import multiprocessing
import sqlite3
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

from unit.test_harness_ab_transport import BUDGET, TRANSPORT, USAGE
from research_loop.providers.deepseek import RequestNotSent


def denied_in_child(ledger, identity, connection):
    """fork子进程沿用账本对象；不以重新构造Ledger轮询活动账本。"""
    denied = []
    for name, operation in [
        ('reserve', lambda: ledger.reserve('root', input_upper_bound=1, max_output_tokens=1)),
        ('mark_sent', lambda: ledger.mark_sent(identity)),
        ('begin_root', lambda: ledger.begin_root('other', 'H1-Q2', 'A'))]:
        try:
            operation()
        except BUDGET.BudgetHalt:
            denied.append(name)
    connection.send({'denied': denied, 'first_error': ledger.snapshot()['first_error']})
    connection.close()


class HarnessAbFirstErrorTests(unittest.TestCase):
    """真实SQLite锁、线程和进程配合MockTransport验证停止边界。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='harness-first-error-')
        self.root = Path(self.temp.name)
        self.now = 100.0
        self.ledger = BUDGET.Ledger(self.root / 'budget.db', clock=lambda: self.now)
        self.ledger.begin_root('root', 'H1-Q1', 'A')
        self.clients = []
        self.inner_calls = []

    def tearDown(self):
        for client in self.clients:
            client.close()
        self.temp.cleanup()

    def client(self, handler=None):
        def inner(request):
            self.inner_calls.append(request)
            if handler:
                return handler(request)
            return httpx.Response(200, json={'id': 'synthetic', 'model': 'deepseek-flash', 'usage': USAGE})
        transport = TRANSPORT.BudgetTransport(self.ledger, 'root', self.root / 'requests',
            inner=httpx.MockTransport(inner))
        client = httpx.Client(transport=transport)
        self.clients.append(client)
        return client, transport

    @staticmethod
    def send(client, **changes):
        payload = {'model': 'deepseek-flash', 'temperature': 0.1, 'max_tokens': 64,
            'messages': [{'role': 'user', 'content': 'synthetic'}], **changes}
        return client.post('https://api.deepseek.com/chat/completions', json=payload)

    def test_denial_is_committed_and_cannot_reopen_after_inflight_settles(self):
        ledger = BUDGET.Ledger(self.root / 'small.db', batch_tokens=210, clock=lambda: self.now)
        ledger.begin_root('root', 'H1-Q1', 'A')
        request = ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)
        ledger.mark_sent(request['reservation_id'])
        with self.assertRaisesRegex(BUDGET.BudgetHalt, 'token reservation'):
            ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)
        first = copy.deepcopy(ledger.snapshot()['first_error'])
        self.assertEqual(first['reason'], 'token reservation budget exhausted')
        self.assertEqual(first['batch_occupancy']['tokens'], 110)
        self.assertEqual(first['details']['candidate']['reserved_tokens'], 110)
        ledger.settle(request['reservation_id'], {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2})
        with self.assertRaisesRegex(BUDGET.BudgetHalt, 'halted'):
            ledger.reserve('root', input_upper_bound=1, max_output_tokens=1)
        reopened = BUDGET.Ledger(ledger.path, batch_tokens=210, clock=lambda: self.now)
        self.assertEqual(reopened.snapshot()['first_error'], first)
        self.assertEqual(reopened.snapshot()['actual_known_tokens'], 2)

    def test_preflight_denial_blocks_new_sends_but_preserves_existing_inflight(self):
        entered, release = threading.Event(), threading.Event()
        def handler(_):
            entered.set()
            if not release.wait(5):
                raise AssertionError('synthetic response not released')
            return httpx.Response(200, json={'id': 'inflight', 'model': 'deepseek-flash', 'usage': USAGE})
        first_client, _ = self.client(handler)
        next_client, _ = self.client()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.send, first_client)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(RequestNotSent):
                    self.send(next_client, model='outside-freeze')
                first_error = copy.deepcopy(self.ledger.snapshot()['first_error'])
                with self.assertRaisesRegex(RequestNotSent, 'halted'):
                    self.send(next_client)
                self.assertEqual(len(self.inner_calls), 1)
            finally:
                release.set()
            self.assertEqual(future.result(timeout=5).status_code, 200)
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot['first_error'], first_error)
        self.assertEqual(snapshot['actual_known_tokens'], 120)

    def test_reserved_but_unadmitted_request_is_released_after_other_role_halts(self):
        client, _ = self.client()
        reserve = self.ledger.reserve
        def reserve_then_halt(*args, **kwargs):
            value = reserve(*args, **kwargs)
            self.ledger.halt('other role rejected', operation='source_identity', root_id='root')
            return value
        with patch.object(self.ledger, 'reserve', side_effect=reserve_then_halt):
            with self.assertRaisesRegex(RequestNotSent, 'other role rejected'):
                self.send(client)
        snapshot = self.ledger.snapshot()
        self.assertFalse(self.inner_calls)
        self.assertEqual(snapshot['reservations'][0]['status'], 'not_sent')
        self.assertEqual(snapshot['current_charged_or_reserved']['requests'], 0)
        self.assertEqual(snapshot['first_error']['reason'], 'other role rejected')

    def test_dispatch_granted_before_error_remains_inflight_even_if_inner_starts_later(self):
        client, _ = self.client()
        mark = self.ledger.mark_sent
        def mark_then_halt(identity):
            mark(identity)
            self.ledger.halt('parallel rejection after grant', operation='request_rejected')
        with patch.object(self.ledger, 'mark_sent', side_effect=mark_then_halt):
            self.assertEqual(self.send(client).status_code, 200)
        self.assertEqual(len(self.inner_calls), 1)
        self.assertEqual(self.ledger.snapshot()['actual_known_tokens'], 120)
        with self.assertRaises(RequestNotSent):
            self.send(client)
        self.assertEqual(len(self.inner_calls), 1)

    def test_first_error_crosses_processes_and_unknown_does_not_overwrite_it(self):
        reserved = self.ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)
        sent = self.ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)
        self.ledger.mark_sent(sent['reservation_id'])
        self.ledger.halt('first preflight failure', operation='source_identity', root_id='root')
        first = copy.deepcopy(self.ledger.snapshot()['first_error'])
        context = multiprocessing.get_context('fork')
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=denied_in_child,
            args=(self.ledger, reserved['reservation_id'], send))
        process.start()
        send.close()
        try:
            self.assertTrue(receive.poll(5))
            result = receive.recv()
        finally:
            receive.close()
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(result['denied'], ['reserve', 'mark_sent', 'begin_root'])
        self.assertEqual(result['first_error'], first)
        self.ledger.unknown(sent['reservation_id'], 'later incomplete receipt')
        self.ledger.not_sent(reserved['reservation_id'], 'never admitted')
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot['first_error'], first)
        self.assertEqual(snapshot['current_charged_or_reserved']['tokens'], 110)

    def test_multiple_threads_cannot_replace_first_error(self):
        def reject(index):
            return self.ledger.halt('failure-' + str(index), operation='request_rejected')
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(reject, range(24)))
        self.assertTrue(all(value == results[0] for value in results))
        self.assertEqual(self.ledger.snapshot()['first_error'], results[0])

    def test_identity_errors_outside_transport_freeze_the_same_batch(self):
        client, _ = self.client()
        def invalid_identity():
            raise PermissionError('synthetic frozen source drift')
        with self.assertRaisesRegex(RequestNotSent, 'synthetic frozen source drift'):
            TRANSPORT.require_generation_identity(self.ledger, 'root', invalid_identity,
                operation='source_identity')
        self.assertEqual(self.ledger.snapshot()['first_error']['operation'], 'source_identity')
        with self.assertRaises(RequestNotSent):
            self.send(client)
        self.assertFalse(self.inner_calls)

    def test_contract_rejection_retains_specific_cause_and_no_reservation(self):
        client, _ = self.client()
        with self.assertRaisesRegex(RequestNotSent, 'parameters differ'):
            self.send(client, temperature=0.9)
        first = self.ledger.snapshot()['first_error']
        self.assertIn('parameters differ', first['reason'])
        self.assertEqual(first['batch_occupancy']['requests'], 0)
        self.assertFalse(self.ledger.snapshot()['reservations'])
        self.assertFalse(self.inner_calls)

    def test_legacy_policy_is_not_migrated_or_resumed(self):
        path = self.root / 'legacy-contract.db'
        legacy = BUDGET.Ledger(path, clock=lambda: self.now)
        with sqlite3.connect(path) as db:
            config = legacy.config.copy()
            config.pop('first_error_policy_version')
            db.execute('UPDATE metadata SET value=? WHERE key=?', (json.dumps(config), 'config'))
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            BUDGET.Ledger(path, clock=lambda: self.now)
        self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
