"""无全局时限实验的所属进程监督；结束或取消后仍有明确的收尾边界。"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


@contextmanager
def cancellation_scope(state):
    """只记录取消请求，避免Popen尚未返回handle时丢失所属进程。"""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('supervision requires main-thread signal ownership')
    previous = {}

    def capture(signum, _frame):
        state['signals'].append(signum)
        if not state['cleaning'] and state.get('cancel_signal') is None:
            state['cancel_signal'] = signum

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, capture)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _members(pgid, timeout):
    """僵尸不再执行或发请求；只查询本控制器创建的进程组。"""
    listing = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,stat='],
                                     text=True, timeout=timeout)
    return [int(row[0]) for line in listing.splitlines()
            if len(row := line.split()) >= 3 and int(row[1]) == pgid
            and not row[2].startswith('Z')]


def _stop_group(process, deadline, monotonic_deadline):
    """重复观察退出状态，保留首次观察，不把瞬时退出中的成员当最终结果。"""
    record = {'pid': process.pid, 'pgid': process.pid, 'kill_sent': False,
              'leader_reaped': False, 'termination_started_at': time.time()}
    remaining = lambda: max(0, min(deadline-time.time(), monotonic_deadline-time.monotonic()))
    try:
        os.killpg(process.pid, signal.SIGKILL)
        record['kill_sent'] = True
    except ProcessLookupError:
        record['group_already_absent'] = True
    except OSError as exc:
        record['kill_error_type'] = type(exc).__name__
    try:
        process.wait(timeout=min(1, remaining()))
        record.update(leader_reaped=True, returncode=process.returncode)
    except subprocess.TimeoutExpired:
        record['leader_reap_timed_out'] = True
    if process.stdin is not None and not process.stdin.closed:
        try:
            process.stdin.close()
        except OSError as exc:
            record['stdin_close_error_type'] = type(exc).__name__
    observations = []
    probe_stop = time.monotonic() + min(3, remaining())
    while remaining() > 0:
        try:
            alive = _members(process.pid, max(.001, min(.5, remaining())))
            observations.append({'at': time.time(), 'alive_group_members': alive})
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            record['group_observation_error'] = type(exc).__name__
            break
        if not alive or time.monotonic() >= probe_stop:
            break
        time.sleep(min(.05, remaining()))
    record['observations'] = observations
    record['alive_group_members'] = observations[-1]['alive_group_members'] if observations else None
    return record


def supervise_financial_batch(command, payload, log, *, cleanup_steps=(), event_path=None,
                             cleanup_reserve_seconds=300, wrapper_finalize_reserve_seconds=10):
    """等待完成、财务门停止或用户取消；不设置隐含的整体运行截止。

    Args:
        command: 直接拥有的批次argv，不得嵌套另一个会话launcher。
        payload: 仅经标准输入传入的内容，不写入事件或结果。
        log: 本次新建的输出文件句柄。
        cleanup_steps: 只操作本实验资源的无模型收尾步骤。
        event_path: 可选的新事件日志路径，拒绝覆盖。
        cleanup_reserve_seconds: 结束或取消后包含组回收的收尾上限。
        wrapper_finalize_reserve_seconds: 总收尾窗口内留给调用方写最终收据的时间。
    """
    if (not command or not isinstance(payload, str) or cleanup_reserve_seconds != 300
            or wrapper_finalize_reserve_seconds != 10):
        raise ValueError('explicit command, stdin payload and 300-second cleanup required')
    if any(Path(str(arg)).name == 'launch_read_paper_ab.py' for arg in command):
        raise PermissionError('direct batch ownership required')
    for step in cleanup_steps:
        if not step.get('name') or not step.get('command') or not 0 < step.get('timeout', 0) <= 300:
            raise ValueError('invalid bounded cleanup step')
    event = Path(event_path).open('x') if event_path is not None else None
    state = {'cleaning': False, 'signals': []}
    result = {'policy': 'financial_only', 'started_at': time.time(), 'global_deadline': None,
              'cleanup_reserve_seconds': 300, 'wrapper_finalize_reserve_seconds': 10,
              'status': 'not_started', 'cleanup': []}
    process = None

    def emit(event_name, **fields):
        if event is not None:
            try:
                event.write(json.dumps({'event': event_name, 'at': time.time(), **fields})+'\n')
                event.flush()
            except OSError:
                result['event_write_failed'] = True

    try:
        with cancellation_scope(state):
            try:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log,
                    stderr=subprocess.STDOUT, text=True, start_new_session=True)
                result['pid'] = process.pid
                emit('batch_started', pid=process.pid, pgid=process.pid)
                unsent = payload
                while state.get('cancel_signal') is None:
                    try:
                        process.communicate(input=unsent, timeout=.25)
                        result.update(status='exited', returncode=process.returncode)
                        break
                    except subprocess.TimeoutExpired:
                        unsent = None
                if state.get('cancel_signal') is not None:
                    result.update(status='cancelled', cancel_signal=signal.Signals(state['cancel_signal']).name)
            except KeyboardInterrupt:
                result['status'] = 'cancelled'
            except BaseException as exc:
                result.update(status='controller_error', error_type=type(exc).__name__)
            finally:
                state['cleaning'] = True
                started = time.time()
                deadline = started + 290
                monotonic_deadline = time.monotonic() + 290
                result.update(cleanup_started_at=started, cleanup_deadline_after_stop=started+300,
                              cleanup_steps_deadline_after_stop=deadline)
                emit('cleanup_started', deadline=started+300, cleanup_steps_deadline=deadline)
                if process is not None:
                    result['batch'] = _stop_group(process, deadline, monotonic_deadline)
                result['model_processes_stopped_at'] = time.time()
                remaining = lambda: max(0, min(deadline-time.time(), monotonic_deadline-time.monotonic()))
                for step in cleanup_steps:
                    record = {'name': step['name'], 'started_at': time.time()}
                    result['cleanup'].append(record)
                    child = None
                    try:
                        if remaining() <= .5:
                            record['status'] = 'not_started_cleanup_deadline_exhausted'
                            continue
                        child = subprocess.Popen(step['command'], stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        child.wait(timeout=min(step['timeout'], max(.001, remaining()-.5)))
                        record.update(status='exited', returncode=child.returncode)
                    except subprocess.TimeoutExpired:
                        record['status'] = 'step_timeout'
                    except BaseException as exc:
                        record.update(status='step_error', error_type=type(exc).__name__)
                    finally:
                        if child is not None:
                            record['group'] = _stop_group(child, deadline, monotonic_deadline)
                        record['finished_at'] = time.time()
                        emit('cleanup_step_finished', **record)
                # 最后统一复查，旧的首次/逐步观察均保留，不覆写为从未发生。
                final_groups = []
                for group in [result.get('batch'), *(x.get('group') for x in result['cleanup'])]:
                    if group is None:
                        continue
                    try:
                        alive = _members(group['pgid'], max(.001, min(.5, remaining()))) if remaining() else None
                    except (subprocess.SubprocessError, OSError, ValueError):
                        alive = None
                    final_groups.append({'pgid': group['pgid'], 'alive_group_members': alive})
                result['final_process_groups'] = final_groups
                final = {x['pgid']: x['alive_group_members'] for x in final_groups}
                result['batch_group_cleaned'] = process is None or bool(
                    result['batch']['leader_reaped'] and final.get(process.pid) == [])
                result['cleanup_all_passed'] = all(
                    x.get('status') == 'exited' and x.get('returncode') == 0
                    and x.get('group', {}).get('leader_reaped')
                    and final.get(x['group']['pgid']) == [] for x in result['cleanup'])
    finally:
        result.update(signals=state['signals'], finished_at=time.time())
        result['finished_before_cleanup_deadline'] = result['finished_at'] <= result.get('cleanup_deadline_after_stop', 0)
        emit('supervision_finished', status=result['status'], batch_group_cleaned=result.get('batch_group_cleaned'))
        if event is not None:
            event.close()
    return result
