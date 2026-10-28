"""为整个正式批次设置独立进程组绝对截止，覆盖父驱动收尾时间。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def execution_deadline(manifest, started):
    """整个进程组的截止也扣除历史时间，不能只在模型预算检查时扣减。"""
    limits = manifest['ledger_limits']
    carried = limits.get('carried_usage', {}).get('seconds', 0)
    if (isinstance(carried, bool) or not isinstance(carried, int)
            or not 0 <= carried < limits['batch_seconds']):
        raise PermissionError('invalid carried execution seconds')
    return started + limits['batch_seconds'] - carried


def run_guarded(command, payload, deadline, log):
    """到期终止本次独立组；格进程继承组，不波及部署或其他实验。"""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    try:
        process.communicate(input=payload, timeout=max(0.001, deadline - time.time()))
        return {'status': 'exited', 'returncode': process.returncode, 'pid': process.pid}
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return {'status': 'hard_deadline', 'returncode': process.returncode, 'pid': process.pid,
            'reason': 'entire_owned_batch_process_group_terminated'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    payload = json.loads(sys.stdin.read())
    started = payload['started_at']
    if not isinstance(started, (int, float)) or not 0 < started <= time.time():
        raise PermissionError('invalid outer start')
    deadline = execution_deadline(manifest, started)
    if manifest.get('experiment_kind') == 'prompt_bundle_v1':
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from prompt_bundle_ab import validate_outer_start
        if deadline != validate_outer_start(manifest, started):
            raise PermissionError('PROMPT_BUNDLE_LAUNCHER_DEADLINE_MISMATCH')
    runtime = Path(manifest['runtime_root'])
    with (runtime / 'driver.log').open('x') as log:
        result = run_guarded([sys.executable, '-I', '-B',
            str(Path(__file__).with_name('run_read_paper_batch.py')), '--manifest', str(args.manifest)],
            json.dumps(payload), deadline, log)
    result.update(started_at=started, finished_at=time.time(), deadline=deadline)
    (runtime / 'launcher-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    sys.exit(0 if result['status'] == 'exited' and result['returncode'] == 0 else 1)


if __name__ == '__main__':
    main()
