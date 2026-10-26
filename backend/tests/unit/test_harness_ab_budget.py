"""真实调用前检查持久预算原子性与未知结果的保守处理。"""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[2] / 'scripts/harness_ab_budget.py'
SPEC = importlib.util.spec_from_file_location('harness_ab_budget', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Ledger, BudgetHalt = MODULE.Ledger, MODULE.BudgetHalt


class HarnessAbBudgetTests(unittest.TestCase):
    """使用合成用量，不导入业务服务或访问网络。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = 100.0
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def ledger(self, **limits):
        self.counter += 1
        return Ledger(self.root / f'{self.counter}.db', clock=lambda: self.now, **limits)

    def started(self, **limits):
        ledger = self.ledger(**limits)
        ledger.begin_root('root', 'H1-Q1', 'A')
        return ledger

    @staticmethod
    def sent(ledger, *, prompt_bound=100, output_bound=10):
        request = ledger.reserve('root', input_upper_bound=prompt_bound, max_output_tokens=output_bound)
        ledger.mark_sent(request['reservation_id'])
        return request['reservation_id']

    def test_known_usage_counts_each_token_once_and_cached_input_cost(self):
        ledger = self.started()
        identity = self.sent(ledger)
        usage = {'prompt_tokens': 80, 'completion_tokens': 8, 'total_tokens': 88,
            'prompt_cache_hit_tokens': 20, 'prompt_cache_miss_tokens': 60,
            'prompt_tokens_details': {'cached_tokens': 20},
            'completion_tokens_details': {'reasoning_tokens': 5}}
        ledger.settle(identity, usage)
        ledger.settle(identity, usage)
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot['actual_known_tokens'], 88)
        self.assertEqual(snapshot['physical_sent_requests'], 1)
        self.assertEqual(snapshot['charged_or_reserved']['tokens'], 88)
        self.assertEqual(snapshot['actual_known_estimated_nano_yuan'], 60 * 2000 + 20 * 40 + 8 * 8000)

    def test_parallel_reservations_cannot_overspend_last_request(self):
        ledger = self.started(batch_requests=1)
        def reserve(_):
            try:
                return ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)
            except BudgetHalt:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(reserve, range(20)))
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(ledger.snapshot()['charged_or_reserved']['requests'], 1)

    def test_parallel_roots_share_batch_token_budget(self):
        ledger = self.started(batch_tokens=210, root_tokens=210)
        ledger.begin_root('child-family', 'H1-Q2', 'B')
        ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)
        with self.assertRaisesRegex(BudgetHalt, 'token'):
            ledger.reserve('child-family', input_upper_bound=100, max_output_tokens=10)

    def test_all_root_and_batch_count_money_token_limits_are_enforced(self):
        dimensions = [({'batch_requests': 1}, 'request'), ({'root_requests': 1}, 'request'),
            ({'batch_tokens': 150}, 'token'), ({'root_tokens': 150}, 'token'),
            ({'batch_yuan': 0.0004}, 'money')]
        for limits, reason in dimensions:
            with self.subTest(limits=limits):
                ledger = self.started(**limits)
                ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)
                with self.assertRaisesRegex(BudgetHalt, reason):
                    ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)

    def test_batch_and_root_deadlines_and_clock_backwards_block_before_send(self):
        for limits, offset, reason in [({'batch_seconds': 5}, 5, 'batch time'),
                ({'root_seconds': 5}, 5, 'root time'), ({}, -1, 'clock moved')]:
            with self.subTest(reason=reason):
                self.now = 100
                ledger = self.started(**limits)
                self.now += offset
                with self.assertRaisesRegex(BudgetHalt, reason):
                    ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)

    def test_deadline_is_rechecked_after_reserve_before_mark_sent(self):
        ledger = self.started(root_seconds=1)
        request = ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)
        self.now += 1
        with self.assertRaisesRegex(BudgetHalt, 'root time'):
            ledger.mark_sent(request['reservation_id'])
        ledger.not_sent(request['reservation_id'], 'transport was not entered')
        self.assertEqual(ledger.snapshot()['charged_or_reserved']['requests'], 0)

    def test_incomplete_stream_holds_reservation_and_blocks_batch_after_reopen(self):
        ledger = self.started()
        identity = self.sent(ledger)
        ledger.unknown(identity, 'stream ended before DONE')
        ledger.unknown(identity, 'same unknown notification')
        self.assertEqual(ledger.snapshot()['charged_or_reserved']['tokens'], 110)
        with self.assertRaisesRegex(BudgetHalt, 'halted'):
            ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)
        reopened = Ledger(ledger.path, clock=lambda: self.now)
        self.assertEqual(reopened.snapshot()['reservations'][0]['status'], 'unknown')
        with self.assertRaisesRegex(BudgetHalt, 'halted'):
            reopened.begin_root('next', 'H1-Q1', 'B')

    def test_reopening_sent_or_reserved_ledger_never_blindly_resumes(self):
        for mark_sent in (False, True):
            with self.subTest(mark_sent=mark_sent):
                ledger = self.started()
                request = ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)
                if mark_sent:
                    ledger.mark_sent(request['reservation_id'])
                reopened = Ledger(ledger.path, clock=lambda: self.now)
                self.assertEqual(reopened.snapshot()['charged_or_reserved']['tokens'], 110)
                with self.assertRaisesRegex(BudgetHalt, 'unresolved'):
                    reopened.reserve('root', input_upper_bound=100, max_output_tokens=10)

    def test_missing_inconsistent_or_over_bound_usage_becomes_unknown(self):
        invalid = [None, {}, {'prompt_tokens': 1, 'completion_tokens': 2},
            {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 4},
            {'prompt_tokens': 1, 'completion_tokens': 11, 'total_tokens': 12},
            {'prompt_tokens': 101, 'completion_tokens': 1, 'total_tokens': 102},
            {'prompt_tokens': True, 'completion_tokens': 1, 'total_tokens': 2},
            {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2,
             'prompt_cache_hit_tokens': 2},
            {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2,
             'completion_tokens_details': {'reasoning_tokens': 2}}]
        for usage in invalid:
            with self.subTest(usage=usage):
                ledger = self.started()
                identity = self.sent(ledger)
                with self.assertRaisesRegex(BudgetHalt, 'cannot be settled'):
                    ledger.settle(identity, usage)
                snapshot = ledger.snapshot()
                self.assertEqual(snapshot['charged_or_reserved']['tokens'], 110)
                self.assertEqual(snapshot['reservations'][0]['status'], 'unknown')
                self.assertTrue(snapshot['halt_reason'])

    def test_not_sent_cannot_release_any_sent_or_settled_charge(self):
        ledger = self.started()
        identity = self.sent(ledger)
        with self.assertRaisesRegex(BudgetHalt, 'cannot release'):
            ledger.not_sent(identity, 'not allowed')
        ledger.settle(identity, {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2})
        with self.assertRaisesRegex(BudgetHalt, 'cannot release'):
            ledger.not_sent(identity, 'not allowed')
        with self.assertRaisesRegex(BudgetHalt, 'no resend'):
            ledger.mark_sent(identity)

    def test_confirmed_not_sent_releases_reserved_allowance_but_keeps_record(self):
        ledger = self.started(batch_requests=1)
        request = ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)
        ledger.not_sent(request['reservation_id'], 'adapter rejected before transport')
        ledger.not_sent(request['reservation_id'], 'idempotent')
        ledger.reserve('root', input_upper_bound=100, max_output_tokens=10)
        snapshot = ledger.snapshot()
        self.assertEqual(len(snapshot['reservations']), 2)
        self.assertEqual(snapshot['charged_or_reserved']['requests'], 1)
        self.assertEqual(snapshot['physical_sent_requests'], 0)

    def test_only_frozen_eight_cells_and_one_root_per_cell(self):
        ledger = self.started()
        with self.assertRaisesRegex(ValueError, 'frozen'):
            ledger.begin_root('new', 'H1-Q5', 'A')
        with self.assertRaisesRegex(BudgetHalt, 'no repeat'):
            ledger.begin_root('new', 'H1-Q1', 'A')
        with self.assertRaisesRegex(BudgetHalt, 'no repeat'):
            ledger.begin_root('root', 'H1-Q2', 'A')

    def test_limit_changes_and_oversized_output_are_rejected(self):
        ledger = self.started()
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            Ledger(ledger.path, batch_tokens=13_000_000, clock=lambda: self.now)
        with self.assertRaisesRegex(BudgetHalt, 'output allowance'):
            ledger.reserve('root', input_upper_bound=100, max_output_tokens=65_537)


if __name__ == '__main__':
    unittest.main()
