"""一次性运行冻结的去重对照矩阵；计时、请求、Token和金额共享硬上限。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness_ab_budget import Ledger
from run_read_paper_ab import validate_manifest


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def reconcile_cell(cell, snapshot):
    """同时核对原生与HTTP收据，不把外层已结算误当作原生协议完整。

    Args:
        cell: 已结束的本格目录。
        snapshot: 共同账本的事后快照；不得以构造新Ledger来轮询在途请求。

    Returns:
        停止原因列表。完整合法的工具参数拒绝不在此冒充远端传输失败。
    """
    result = read(cell / 'result.json')
    root = result.get('root_id')
    native = read(cell / 'model-calls.json')
    physical = [r for r in snapshot['reservations'] if r['root_id'] == root]
    issues = []
    if result.get('status') != 'executed':
        issues.append('cell_execution_not_complete')
    if any(r['status'] != 'settled' for r in physical):
        issues.append('unresolved_physical_receipt')
    if any(r['status'] != 'succeeded' for r in native):
        issues.append('native_response_not_successful')
    if len(native) != len(physical):
        issues.append('native_physical_count_mismatch')
    by_id = {}
    for receipt in native:
        try:
            response = json.loads(receipt['response_json'])
            usage = json.loads(receipt['usage_json'])
            values = [usage[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
            if (any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values)
                    or values[0] + values[1] != values[2] or response['id'] in by_id):
                raise ValueError('invalid usage or repeated identity')
            by_id[response['id']] = values
        except (ValueError, TypeError, KeyError):
            issues.append('native_usage_missing_or_invalid')
    for receipt in physical:
        try:
            meta = json.loads(receipt['response_json'])
            if by_id[meta['response_id']] != [receipt['actual_input'], receipt['actual_output'], receipt['actual_tokens']]:
                issues.append('native_physical_usage_mismatch')
        except (KeyError, TypeError, ValueError):
            issues.append('physical_response_identity_missing')
    for actions in read(cell / 'actions.json').values():
        for action in actions:
            data = (action.get('observation') or {}).get('data') or {}
            if action['tool_name'] == 'retrieve_evidence' and action['status'] == 'succeeded':
                if (data.get('status') in {'degraded', 'unavailable'} or data.get('degradations')
                        or data.get('strategy', {}).get('name') != 'located_fp16'
                        or data.get('effective_path') != 'located_bm25_rerank'):
                    issues.append('actual_retrieval_route_or_readiness_changed')
    return sorted(set(issues))


def main():
    """凭据仅经内存标准输入传递；一次性标记阻止重复付费执行。"""
    started = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.manifest)
    runtime = Path(manifest['runtime_root'])
    order = manifest['execution_order']
    limits = manifest['ledger_limits']
    if len(order) != len({tuple(c) for c in order}) or sorted(order) != sorted(limits['allowed_cells']):
        raise PermissionError('matrix order does not match frozen allowed cells')
    authorization = read(manifest['authorization_path'])
    if not authorization.get('approved') or authorization.get('ledger_limits') != limits:
        raise PermissionError('real execution not authorized')
    for case, variant in order:
        validate_manifest({**manifest, 'cell_absolute_deadline': started + limits['root_seconds']}, case, variant, 'execute')
    for entry in manifest['frozen_files']:
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise PermissionError('frozen source or resource drift')
    input_payload = json.loads(sys.stdin.read())
    secret = input_payload['api_key']
    outer_started = input_payload.get('started_at', started)
    if not isinstance(outer_started, (int, float)) or not 0 < outer_started <= started:
        raise PermissionError('invalid outer execution start')
    started = outer_started
    if not isinstance(secret, str) or not re.fullmatch(r'[A-Za-z0-9_-]{16,200}', secret):
        raise PermissionError('invalid credential shape; value suppressed')
    if (runtime / 'budget.sqlite3').exists():
        raise PermissionError('existing ledger cannot be restarted')
    marker = runtime / 'started.json'
    with marker.open('x') as handle:
        json.dump({'started_at': started, 'pid': os.getpid(),
            'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()}, handle)
    ledger = Ledger(runtime / 'budget.sqlite3', start_time=started, **limits)
    deadline = started + limits['batch_seconds']
    # 留出终止、收据读取及父驱动落盘时间，不把最后一请求送到总截止之后。
    finish_reserve = 5.0
    rows = []
    stop = None
    script = Path(__file__).with_name('run_read_paper_ab.py')
    for case, variant in order:
        snapshot = ledger.snapshot()
        usage = snapshot['charged_or_reserved']
        reserve = 1_048_576 + limits['max_output_tokens']
        reserve_nano = 1_048_576 * 2000 + limits['max_output_tokens'] * 8000
        if snapshot['halt_reason']:
            stop = 'ledger_halted'
        elif time.time() >= deadline - finish_reserve:
            stop = 'batch_deadline'
        elif (usage['requests'] + 1 > limits['batch_requests'] or usage['tokens'] + reserve > limits['batch_tokens']
                or usage['nano'] + reserve_nano > ledger.config['batch_nano_yuan']):
            stop = 'insufficient_next_request_reservation'
        if stop:
            break
        cell_deadline = min(time.time() + limits['root_seconds'], deadline - finish_reserve)
        print(json.dumps({'event': 'cell_start', 'case': case, 'variant': variant, 'known_tokens': snapshot['actual_known_tokens']}, ensure_ascii=False), flush=True)
        cell = runtime / 'cells' / (case + '-' + variant)
        try:
            run = subprocess.run([sys.executable, '-I', '-B', str(script), '--manifest', str(args.manifest),
                '--case', case, '--variant', variant, '--mode', 'execute', '--deadline', str(cell_deadline)],
                input=json.dumps({'api_key': secret}), text=True, capture_output=True,
                timeout=max(0.01, cell_deadline - time.time()))
            (runtime / (case + '-' + variant + '.log')).write_text(run.stdout + run.stderr)
            if run.returncode or not (cell / 'result.json').exists():
                stop = 'cell_process_failed'
            snapshot = ledger.snapshot()
            issues = reconcile_cell(cell, snapshot) if (cell / 'result.json').exists() else ['cell_result_missing']
            if issues:
                stop = 'cell_receipt_or_isolation_gate'
            result = read(cell / 'result.json') if (cell / 'result.json').exists() else {}
            row = {'case_id': case, 'variant': variant, 'status': result.get('task_status', 'failed'),
                'root_id': result.get('root_id'), 'issues': issues, 'returncode': run.returncode}
            rows.append(row)
            print(json.dumps({'event': 'cell_end', **row, 'known_tokens': snapshot['actual_known_tokens'],
                'requests': snapshot['physical_sent_requests'], 'estimated_yuan': snapshot['actual_known_estimated_nano_yuan'] / 1e9}, ensure_ascii=False), flush=True)
        except subprocess.TimeoutExpired:
            stop = 'absolute_cell_or_batch_deadline'
            rows.append({'case_id': case, 'variant': variant, 'status': 'interrupted', 'root_id': None,
                'issues': ['inflight_receipts_preserved_no_resend']})
        except BaseException as exc:
            stop = 'parent_postcell_error:' + type(exc).__name__
            rows.append({'case_id': case, 'variant': variant, 'status': 'unknown', 'root_id': None,
                'issues': ['raw_cell_and_ledger_preserved']})
        write(runtime / 'progress.json', {'rows': rows, 'stop_reason': stop})
        if stop:
            break
    snapshot = ledger.snapshot()
    write(runtime / 'ledger-final.json', snapshot)
    uncreated = [c for c in order if tuple(c) not in {(r['case_id'], r['variant']) for r in rows}]
    write(runtime / 'batch-results.json', {'status': 'complete' if not stop and len(rows) == len(order) else 'stopped',
        'rows': rows, 'not_created': uncreated, 'stop_reason': stop, 'started_at': started,
        'finished_at': time.time(), 'execution_seconds': time.time() - started,
        'limits': limits, 'actual_known_tokens': snapshot['actual_known_tokens'],
        'physical_requests': snapshot['physical_sent_requests'], 'actual_known_estimated_nano_yuan': snapshot['actual_known_estimated_nano_yuan']})
    print(json.dumps({'event': 'batch_end', 'created': len(rows), 'not_created': len(uncreated), 'stop_reason': stop,
        'known_tokens': snapshot['actual_known_tokens']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
