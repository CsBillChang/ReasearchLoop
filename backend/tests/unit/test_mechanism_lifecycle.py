"""有界所属进程回收的本地合成验证；不访问网络、模型或部署数据库。"""
from pathlib import Path
import sys
import tempfile
import time
import unittest
import math

from experiments.mechanism_pilot.lifecycle import Deadline, run_process


class LifecycleTests(unittest.TestCase):
    """短截止覆盖迟到响应、取消与孙进程，不等待真实分钟预算。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def deadline(admission=.3, total=.9):
        return Deadline(time.time(), time.monotonic(), admission, total)

    def run_child(self, script, *, name='run', **kwargs):
        return run_process([sys.executable, '-I', '-B', '-c', script], self.root / name,
            deadline=kwargs.pop('deadline', self.deadline()),
            poll_seconds=.02, term_grace_seconds=.08, callback_seconds=.08,
            finalize_reserve_seconds=.04, **kwargs)

    def test_normal_exit_finalizes_only_after_group_reaped(self):
        seen = []
        result = self.run_child('print("synthetic completion")',
            finalize_ledger=lambda record: seen.append((record['leader_reaped'], record['alive_group_members'])))
        self.assertEqual(result['stop_reason'], 'process_exited')
        self.assertEqual(seen, [(True, [])])
        self.assertTrue(result['cleanup_complete'])
        self.assertTrue(result['finished_before_total_deadline'])
        self.assertEqual((self.root / 'run/process.log').read_text().strip(), 'synthetic completion')

    def test_cancel_reaps_owned_grandchild_and_keeps_unrelated_process(self):
        # 孙进程继承本组；另一个组仅用于证明不会被宽泛进程匹配误杀。
        import subprocess
        other = subprocess.Popen([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(3)'],
                                 start_new_session=True)
        def stop_other():
            if other.poll() is None:
                other.kill()
            other.wait(timeout=2)
        self.addCleanup(stop_other)
        stopped = []
        result = self.run_child('import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",'
            '"import time; time.sleep(3)"]); time.sleep(3)',
            stop_admission=lambda reason: stopped.append(reason),
            soft_cancel=lambda reason: stopped.append('soft:' + reason))
        self.assertEqual(result['stop_reason'], 'admission_deadline')
        self.assertTrue(result['leader_reaped'])
        self.assertEqual(result['alive_group_members'], [])
        self.assertTrue(result['cleanup_complete'])
        self.assertIsNone(other.poll())
        self.assertEqual(len(stopped), 2)

    def test_cancelled_flag_does_not_claim_stopped_before_kill(self):
        began = time.monotonic()
        result = self.run_child('import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(3)',
            stop_requested=lambda: 'synthetic_user' if time.monotonic() - began > .2 else False)
        self.assertEqual(result['stop_reason'], 'cancelled:synthetic_user')
        self.assertIn('SIGKILL', result['signals_sent'])
        self.assertEqual(result['alive_group_members'], [])
        self.assertTrue(result['cleanup_complete'])

    def test_fixed_origin_and_clock_rollback_cannot_extend_deadline(self):
        wall, monotonic = [100.], [20.]
        deadline = Deadline(100, 20, 40, 45, lambda: wall[0], lambda: monotonic[0])
        wall[0], monotonic[0] = 95, 64
        self.assertEqual(deadline.remaining(), 1)
        self.assertEqual(deadline.admission_remaining(), 0)
        wall[0], monotonic[0] = 145, 21
        self.assertEqual(deadline.remaining(), 0)
        expired = Deadline(time.time() - 2, time.monotonic() - 2, .1, .5)
        result = self.run_child('raise AssertionError("must not start")', deadline=expired)
        self.assertFalse(result['created'])
        self.assertFalse((self.root / 'run').exists())

    def test_cleanup_is_capped_by_original_hard_deadline(self):
        deadline = self.deadline(.2, .6)
        result = self.run_child('import time; time.sleep(3)', deadline=deadline)
        self.assertLessEqual(result['cleanup_deadline_wall'], deadline.started_wall + .6)
        self.assertLessEqual(result['cleanup_deadline_monotonic'], deadline.started_monotonic + .6)
        self.assertTrue(result['finished_before_total_deadline'])

    def test_future_start_cannot_expand_authorized_window(self):
        with self.assertRaisesRegex(ValueError, 'FUTURE'):
            Deadline(time.time() + 30, time.monotonic(), .2, .8)
        with self.assertRaisesRegex(ValueError, 'FUTURE'):
            Deadline(time.time(), time.monotonic() + 30, .2, .8)

    def test_hanging_callback_is_reported_and_cannot_claim_cleanup(self):
        result = self.run_child('print("done")',
            soft_cancel=lambda reason: None,
            stop_admission=lambda reason: time.sleep(1))
        self.assertFalse(result['cleanup_complete'])
        self.assertFalse(result['ledger_finalized'])
        self.assertEqual(result['callback_threads_alive'], ['mechanism-stop_admission'])
        self.assertTrue(result['finished_before_total_deadline'])

    def test_inflight_usage_is_preserved_by_ledger_finalizer(self):
        # 监督器只在组停后通知账本，不凭 cancelled 状态释放任何 sent 预留。
        ledger = {'unsent': 'reserved', 'inflight': 'sent', 'late': 'settled'}
        def finalize(record):
            self.assertTrue(record['process_group_stopped'])
            ledger['unsent'] = 'not_sent'
            ledger['inflight'] = 'unknown'
        result = self.run_child('print("received late usage already settled")', finalize_ledger=finalize)
        self.assertTrue(result['ledger_finalized'])
        self.assertEqual(ledger, {'unsent': 'not_sent', 'inflight': 'unknown', 'late': 'settled'})

    def test_runtime_cannot_reuse_directory_or_enter_checkout(self):
        (self.root / 'run').mkdir()
        with self.assertRaisesRegex(ValueError, 'NEW_RUNTIME'):
            self.run_child('print("never")')
        checkout = self.root / 'checkout'
        checkout.mkdir(); (checkout / '.git').write_text('synthetic worktree')
        with self.assertRaisesRegex(ValueError, 'OUTSIDE_CHECKOUT'):
            run_process([sys.executable, '-c', 'print("never")'], checkout / 'run', deadline=self.deadline())

    def test_explicit_unbounded_mode_keeps_original_start_and_no_fake_deadline(self):
        started_wall, started_monotonic = time.time() - 3000, time.monotonic() - 3000
        deadline = Deadline.without_global_limit(started_wall, started_monotonic)
        self.assertTrue(math.isinf(deadline.remaining()))
        self.assertTrue(math.isinf(deadline.admission_remaining()))
        result = self.run_child('print("no global deadline")', deadline=deadline)
        self.assertEqual(result['original_started_wall'], started_wall)
        self.assertEqual(result['original_started_monotonic'], started_monotonic)
        self.assertIsNone(result['admission_deadline_wall'])
        self.assertIsNone(result['total_deadline_wall'])
        self.assertIsNone(result['finished_before_total_deadline'])
        self.assertFalse(result['enforce_global_deadline'])
        self.assertTrue(result['cleanup_complete'])
        self.assertTrue(result['finished_before_cleanup_deadline'])

    def test_unbounded_cancel_still_reaps_group_with_bounded_cleanup(self):
        begin = time.monotonic()
        deadline = Deadline.without_global_limit(time.time() - 3000, begin - 3000)
        result = self.run_child('import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(3)',
            deadline=deadline, stop_requested=lambda: 'synthetic_stop' if time.monotonic() - begin > .2 else False)
        self.assertEqual(result['stop_reason'], 'cancelled:synthetic_stop')
        self.assertIn('SIGKILL', result['signals_sent'])
        self.assertTrue(result['process_group_stopped'])
        self.assertTrue(result['cleanup_complete'])
        self.assertLessEqual(result['cleanup_deadline_wall'] - result['cleanup_started_wall'], 300)
        self.assertLess(time.monotonic() - begin, 2)

    def test_unbounded_mode_requires_explicit_none_limits(self):
        with self.assertRaisesRegex(ValueError, 'FIXED_DEADLINE'):
            Deadline(time.time(), time.monotonic(), None, None)
        with self.assertRaisesRegex(ValueError, 'MUST_BE_NONE'):
            Deadline(time.time(), time.monotonic(), .2, .8, enforce_global_deadline=False)


if __name__ == '__main__':
    unittest.main()
