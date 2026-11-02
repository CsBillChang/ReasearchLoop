"""所属实验进程的固定截止与有界收尾；不操作部署或改写模型收据。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Callable


@dataclass(frozen=True)
class Deadline:
    """共享固定始点；显式无全局截止模式仍保留停止后的有界收尾。"""

    started_wall: float
    started_monotonic: float
    admission_seconds: float | None = 2400
    total_seconds: float | None = 2700
    wall_clock: Callable[[], float] = field(default=time.time, repr=False, compare=False)
    monotonic_clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)
    enforce_global_deadline: bool = True

    def __post_init__(self):
        values = (self.started_wall, self.started_monotonic)
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in values) or type(self.enforce_global_deadline) is not bool):
            raise ValueError('MECHANISM_FIXED_DEADLINE_REQUIRED')
        if self.enforce_global_deadline:
            if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in (self.admission_seconds, self.total_seconds))
                    or not 0 < self.admission_seconds < self.total_seconds):
                raise ValueError('MECHANISM_FIXED_DEADLINE_REQUIRED')
        elif self.admission_seconds is not None or self.total_seconds is not None:
            raise ValueError('MECHANISM_UNBOUNDED_DEADLINES_MUST_BE_NONE')
        if self.started_wall > self.wall_clock() or self.started_monotonic > self.monotonic_clock():
            raise ValueError('MECHANISM_DEADLINE_START_IS_IN_FUTURE')

    def remaining(self) -> float:
        """返回原总截止剩余秒数；重建监督器不会重置已消耗时间。"""
        if not self.enforce_global_deadline:
            return math.inf
        return max(0., min(self.started_wall + self.total_seconds - self.wall_clock(),
                           self.started_monotonic + self.total_seconds - self.monotonic_clock()))

    def admission_remaining(self) -> float:
        """返回允许新增物理请求的剩余时间，不包括预留的收尾窗口。"""
        if not self.enforce_global_deadline:
            return math.inf
        return max(0., min(self.started_wall + self.admission_seconds - self.wall_clock(),
                           self.started_monotonic + self.admission_seconds - self.monotonic_clock()))

    @classmethod
    def without_global_limit(cls, started_wall, started_monotonic, *,
                             wall_clock=time.time, monotonic_clock=time.monotonic):
        """用户明确撤销全局时限后使用；不伪造任意远期截止。"""
        return cls(started_wall, started_monotonic, None, None,
                   wall_clock, monotonic_clock, enforce_global_deadline=False)


@contextmanager
def _cancellation_signals(state):
    """先接管信号再启动子进程，避免 Popen 返回前丢失取消请求。"""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        def capture(signum, _frame):
            state['signal'] = state.get('signal') or signal.Signals(signum).name
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, capture)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _members(pgid: int, timeout: float) -> list[int]:
    """只返回直属新会话组内仍可执行的成员；僵尸不再发送请求。"""
    text = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,stat='], text=True,
                                   timeout=max(.001, timeout))
    return [int(parts[0]) for line in text.splitlines()
            if len(parts := line.split()) == 3 and int(parts[1]) == pgid
            and not parts[2].startswith('Z')]


def run_process(command, runtime_root, *, deadline: Deadline, env=None, payload=None,
                soft_cancel=lambda reason: None, stop_requested=lambda: False,
                stop_admission=lambda reason: None, finalize_ledger=lambda report: None,
                poll_seconds=.05, term_grace_seconds=.25, callback_seconds=.25,
                finalize_reserve_seconds=10) -> dict:
    """监督自己的进程组；取消标记不等于线程、远端请求或账本已经结束。

    Args:
        command: 已准备的 argv；不通过 shell 执行。
        runtime_root: 尚不存在且位于源码工作树之外的本轮目录。
        deadline: 整批固定时钟；有界默认40/45分钟，显式无全局模式不设总截止。
        env: 已准备的子进程环境；内容不写日志。
        payload: 可选标准输入内容；内容不写监督报告。
        soft_cancel: 调用方对本格 TaskService.cancel 的有界回调。
        stop_requested: 用户/预算停止信号；返回真或具体原因。
        stop_admission: 先关闭共享发送门，不依据任务终态释放在途预留。
        finalize_ledger: 所属组确认已停后核对账本；迟到已结算保持，sent保持未知。

    Returns:
        实际 PID/PGID、终止与回收观察及回调状态；未确认退出不报告收尾成功。
    """
    if (not isinstance(deadline, Deadline) or not isinstance(command, (list, tuple))
            or not command or any(not isinstance(v, str) or not v for v in command)
            or (payload is not None and not isinstance(payload, str))
            or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                   or v <= 0 for v in (poll_seconds, term_grace_seconds, callback_seconds,
                                      finalize_reserve_seconds))):
        raise ValueError('MECHANISM_PROCESS_CONTRACT_INVALID')
    root = Path(runtime_root).resolve()
    if Path(runtime_root).is_symlink() or any((p / '.git').exists() for p in (root, *root.parents)):
        raise ValueError('MECHANISM_RUNTIME_OUTSIDE_CHECKOUT_REQUIRED')
    if root.exists() or not root.parent.is_dir():
        raise ValueError('MECHANISM_NEW_RUNTIME_ROOT_REQUIRED')
    result = {'created': False, 'started_wall': deadline.wall_clock(), 'pid': None, 'pgid': None,
        'original_started_wall': deadline.started_wall,
        'original_started_monotonic': deadline.started_monotonic,
        'enforce_global_deadline': deadline.enforce_global_deadline,
        'admission_deadline_wall': (deadline.started_wall + deadline.admission_seconds
                                    if deadline.enforce_global_deadline else None),
        'total_deadline_wall': (deadline.started_wall + deadline.total_seconds
                               if deadline.enforce_global_deadline else None),
        'stop_reason': None, 'signals_sent': [], 'observations': [], 'callbacks': [],
        'alive_group_members': None, 'leader_reaped': False, 'ledger_finalized': False}
    if deadline.admission_remaining() <= 0:
        result.update(stop_reason='admission_deadline_before_start', cleanup_complete=True,
                      finished_wall=deadline.wall_clock(), created=False)
        return result
    root.mkdir()
    process = None
    signals = {'signal': None}
    pending_callbacks = []
    cleanup_wall = cleanup_mono = None
    reserve = (min(float(finalize_reserve_seconds), deadline.total_seconds * .1)
               if deadline.enforce_global_deadline else float(finalize_reserve_seconds))

    def remaining():
        return max(0., min(deadline.remaining(), cleanup_wall - deadline.wall_clock(),
                           cleanup_mono - deadline.monotonic_clock()))

    def callback(name, function, arg):
        """阻塞回调只占自身有界窗口，未完成者明确阻断成功结算声明。"""
        event = threading.Event()
        record = {'name': name, 'status': 'pending'}
        result['callbacks'].append(record)
        def invoke():
            try:
                function(arg)
                record['status'] = 'completed'
            except BaseException as exc:
                record.update(status='failed', error_type=type(exc).__name__)
            finally:
                event.set()
        thread = threading.Thread(target=invoke, name='mechanism-' + name, daemon=True)
        thread.start()
        event.wait(min(float(callback_seconds), max(0., remaining() - reserve)))
        if not event.is_set():
            pending_callbacks.append(thread)

    def observe():
        if process is None:
            return []
        try:
            alive = _members(process.pid, min(.2, max(.001, remaining() - reserve)))
            result['observations'].append({'at': deadline.wall_clock(), 'alive': alive})
            return alive
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            result['observation_error'] = type(exc).__name__
            return None

    with (root / 'process.log').open('x') as log, _cancellation_signals(signals):
        try:
            process = subprocess.Popen(list(command), stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, cwd=root, env=env, text=True,
                start_new_session=True)
            result.update(created=True, pid=process.pid, pgid=process.pid)
            unsent = payload
            while True:
                requested = signals.get('signal') or stop_requested()
                if requested:
                    result['stop_reason'] = 'cancelled:' + str(requested)
                    break
                if deadline.admission_remaining() <= 0:
                    result['stop_reason'] = 'admission_deadline'
                    break
                try:
                    process.communicate(input=unsent, timeout=max(.001, min(float(poll_seconds),
                        deadline.admission_remaining(), deadline.remaining())))
                    result['stop_reason'] = 'process_exited'
                    break
                except subprocess.TimeoutExpired:
                    unsent = None
        except BaseException as exc:
            result.update(stop_reason='controller_error', error_type=type(exc).__name__)
        finally:
            # 无全局模式仅取消整体时限；第一次停止后仍固定300秒清理窗口。
            cleanup_wall = deadline.wall_clock() + 300
            cleanup_mono = deadline.monotonic_clock() + 300
            if deadline.enforce_global_deadline:
                cleanup_wall = min(cleanup_wall, deadline.started_wall + deadline.total_seconds)
                cleanup_mono = min(cleanup_mono, deadline.started_monotonic + deadline.total_seconds)
            result.update(cleanup_started_wall=deadline.wall_clock(),
                          cleanup_deadline_wall=cleanup_wall, cleanup_deadline_monotonic=cleanup_mono)
            callback('stop_admission', stop_admission, result['stop_reason'])
            if result['stop_reason'] != 'process_exited':
                callback('soft_cancel', soft_cancel, result['stop_reason'])
            alive = observe()
            if process is not None and alive != []:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    result['signals_sent'].append('SIGTERM')
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    result['term_error'] = type(exc).__name__
                grace = min(float(term_grace_seconds), max(0., remaining() - reserve))
                until = deadline.monotonic_clock() + grace
                while alive != [] and deadline.monotonic_clock() < until:
                    if process.poll() is not None:
                        result['leader_reaped'] = True
                    time.sleep(min(.02, max(0., until - deadline.monotonic_clock())))
                    alive = observe()
                if alive != []:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                        result['signals_sent'].append('SIGKILL')
                    except ProcessLookupError:
                        pass
                    except OSError as exc:
                        result['kill_error'] = type(exc).__name__
            if process is not None:
                try:
                    process.wait(timeout=max(.001, min(.5, remaining() - reserve)))
                    result.update(leader_reaped=True, returncode=process.returncode)
                except subprocess.TimeoutExpired:
                    result['leader_reap_timeout'] = True
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                alive = observe()
                while alive and remaining() > reserve + .02:
                    time.sleep(.01)
                    alive = observe()
            result['alive_group_members'] = alive
            result['callback_threads_alive'] = [thread.name for thread in pending_callbacks if thread.is_alive()]
            group_stopped = process is None or (result['leader_reaped'] and alive == [])
            result['process_group_stopped'] = group_stopped
            if group_stopped and not result['callback_threads_alive']:
                callback('finalize_ledger', finalize_ledger, result)
                result['ledger_finalized'] = result['callbacks'][-1]['status'] == 'completed'
            result['cleanup_complete'] = bool(group_stopped and result['ledger_finalized']
                and all(c['status'] == 'completed' for c in result['callbacks']))
    result.update(received_signal=signals['signal'], finished_wall=deadline.wall_clock(),
                  finished_before_total_deadline=(deadline.remaining() > 0
                      if deadline.enforce_global_deadline else None),
                  finished_before_cleanup_deadline=remaining() > 0)
    with (root / 'controller-result.json').open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    return result
