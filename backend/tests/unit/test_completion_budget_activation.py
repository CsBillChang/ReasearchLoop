"""取消全局时限不能取消财务、单根时限、时钟倒退及一次性授权。"""
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'scripts'),
                str(Path(__file__).resolve().parents[2])]
from harness_ab_budget import BudgetHalt, Ledger
from harness_financial_process import supervise_financial_batch
from prompt_bundle_ab import validate_round_clock
from run_read_paper_ab import bounded_deadline


class FinancialBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'budget.sqlite3'
        self.now = [2000.0]

    def ledger(self, **extra):
        options = dict(allowed_cells=[['C', 'A']], start_time=1.0,
                       batch_seconds=100, carried_usage={'tokens': 100, 'requests': 1,
                       'nano_yuan': 1000, 'seconds': 5}, clock=lambda: self.now[0],
                       enforce_batch_seconds=False)
        options.update(extra)
        return Ledger(self.path, **options)

    def test_expired_historical_batch_clock_does_not_block_but_root_clock_still_does(self):
        ledger = self.ledger()
        ledger.begin_root('r', 'C', 'A')
        self.assertIsNone(ledger.snapshot()['batch_absolute_deadline'])
        self.assertEqual(ledger.snapshot()['started_at'], 1.0)
        self.now[0] += 900
        with self.assertRaises(BudgetHalt):
            ledger.reserve('r', max_output_tokens=1, input_upper_bound=1)
        self.assertIn('root time', ledger.snapshot()['halt_reason'])

    def test_carry_remains_charged_and_financial_limit_still_halts(self):
        ledger = self.ledger(batch_tokens=110)
        ledger.begin_root('r', 'C', 'A')
        with self.assertRaises(BudgetHalt):
            ledger.reserve('r', max_output_tokens=6, input_upper_bound=6)
        self.assertEqual(ledger.snapshot()['charged_or_reserved']['tokens'], 100)

    def test_time_policy_cannot_be_changed_when_reopening(self):
        self.ledger()
        with self.assertRaises(ValueError):
            self.ledger(enforce_batch_seconds=True)

    def test_pending_reservation_still_halts_reopen(self):
        ledger = self.ledger()
        ledger.begin_root('r', 'C', 'A')
        reserved = ledger.reserve('r', max_output_tokens=1, input_upper_bound=1)
        ledger.mark_sent(reserved['reservation_id'])
        reopened = self.ledger()
        self.assertIsNotNone(reopened.snapshot()['halt_reason'])
        self.assertEqual(reopened.snapshot()['current_charged_or_reserved']['tokens'], 2)

    def test_legacy_deadline_is_still_enforced_by_default(self):
        ledger = self.ledger(enforce_batch_seconds=True)
        with self.assertRaises(BudgetHalt):
            ledger.begin_root('r', 'C', 'A')

    def test_clock_backwards_is_not_waived(self):
        ledger = self.ledger()
        self.now[0] -= 1
        with self.assertRaises(BudgetHalt):
            ledger.begin_root('r', 'C', 'A')


class FinancialActivationTests(unittest.TestCase):
    def test_optional_global_deadline_never_removes_finite_root_limit(self):
        self.assertEqual(bounded_deadline(2900, None, 2800), 2800)
        self.assertEqual(bounded_deadline(2900, None), 2900)
        self.assertEqual(bounded_deadline(2900, 2400, 2800), 2400)
        for value in [None, True, float('inf'), float('nan'), -1]:
            with self.subTest(root=value), self.assertRaises(ValueError):
                bounded_deadline(value, None)
        for value in [True, float('inf'), float('nan'), -1]:
            with self.subTest(optional=value), self.assertRaises(ValueError):
                bounded_deadline(2900, value)

    def manifest(self):
        return {'completion_contract': {}, 'ledger_limits': {'enforce_batch_seconds': False}}

    def context(self):
        return {'clock': {'policy': 'financial_only', 'round_started_at': 1,
                          'round_absolute_deadline': None, 'model_absolute_deadline': None}}

    def auth(self):
        return {'policy': 'financial_only', 'human_approved': True,
                'round_started_at': 1, 'round_absolute_deadline': None, 'expires_at': None}

    def test_none_expiry_only_with_explicit_validated_financial_contract(self):
        with patch('agent_final_continuation.validate_continuation', return_value=self.context()):
            self.assertEqual(validate_round_clock(self.manifest(), self.auth(), clock=lambda: 3000), (1, None))
            for key in ['policy', 'human_approved', 'expires_at', 'round_absolute_deadline']:
                auth = self.auth()
                auth.pop(key)
                with self.subTest(key=key), self.assertRaises(PermissionError):
                    validate_round_clock(self.manifest(), auth, clock=lambda: 3000)

    def test_none_expiry_cannot_cancel_a_legacy_clock(self):
        with self.assertRaises(PermissionError):
            validate_round_clock({'ledger_limits': {'batch_seconds': 100}}, self.auth(), clock=lambda: 3000)


class FinancialProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_natural_exit_reaps_descendant_and_runs_only_owned_cleanup(self):
        marker = self.root / 'cleaned.json'
        spawn = 'import subprocess,sys;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])'
        cleanup = [sys.executable, '-c', 'from pathlib import Path;import sys;Path(sys.argv[1]).write_text("closed")', str(marker)]
        with (self.root / 'driver.log').open('w') as log:
            result = supervise_financial_batch([sys.executable, '-c', spawn], 'private-stdin-test', log,
                cleanup_steps=[{'name': 'owned-cleanup', 'command': cleanup, 'timeout': 3}],
                event_path=self.root / 'events.jsonl')
        self.assertEqual(result['status'], 'exited')
        self.assertTrue(result['batch_group_cleaned'])
        self.assertTrue(result['cleanup_all_passed'])
        self.assertIsNone(result['global_deadline'])
        self.assertEqual(marker.read_text(), 'closed')
        self.assertEqual(result['cleanup_deadline_after_stop']-result['cleanup_started_at'], 300)
        self.assertEqual(result['cleanup_steps_deadline_after_stop']-result['cleanup_started_at'], 290)
        self.assertTrue(all(row['finished_at'] <= result['cleanup_steps_deadline_after_stop']
                            for row in result['cleanup']))
        self.assertNotIn('private-stdin-test', json.dumps(result)+(self.root / 'events.jsonl').read_text())

    def test_cancellation_stops_owned_group_without_global_deadline(self):
        scriptdir = str(Path(__file__).resolve().parents[2] / 'scripts')
        code = ('import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
                'from harness_financial_process import supervise_financial_batch;'
                'root=Path(sys.argv[2]);log=(root/"cancel.log").open("w");'
                'result=supervise_financial_batch([sys.executable,"-c","import time;time.sleep(60)"],"",log,event_path=root/"cancel.events");'
                '(root/"cancel-result.json").write_text(json.dumps(result))')
        process = subprocess.Popen([sys.executable, '-I', '-B', '-c', code, scriptdir, str(self.root)], start_new_session=True)
        child_group = None
        try:
            deadline = time.monotonic()+8
            while time.monotonic() < deadline:
                path = self.root / 'cancel.events'
                if path.exists() and path.stat().st_size:
                    child_group = json.loads(path.read_text().splitlines()[0])['pid']
                    break
                time.sleep(.02)
            self.assertIsNotNone(child_group)
            os.kill(process.pid, signal.SIGTERM)
            process.wait(timeout=8)
            result = json.loads((self.root / 'cancel-result.json').read_text())
            self.assertEqual(result['status'], 'cancelled')
            self.assertTrue(result['batch_group_cleaned'])
            self.assertTrue(result['finished_before_cleanup_deadline'])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            if child_group is not None:
                try:
                    os.killpg(child_group, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == '__main__':
    unittest.main()
