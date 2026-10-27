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
from harness_ab_experiment import SUPERVISOR_KINDS
from harness_ab_pair_gate import (assert_fresh_gate_root, initialize_scope_session,
    issue_cell_grant, manifest_identity, wait_pair_review)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def reconcile_cell(cell, snapshot):
    """返回停止原因；确定未发送也不得被静默过滤为成功。"""
    return reconcile_cell_report(cell, snapshot)['issues']


def reconcile_cell_report(cell, snapshot):
    """分别核对远端成功、确定未发送及不完整收据，保留原始缺失值。

    Args:
        cell: 已结束的本格目录。
        snapshot: 共同账本的事后快照；不得以构造新Ledger来轮询在途请求。

    Returns:
        逐类核对报告。原生 not_sent 不要求远端 usage，也不补零；只有所有
        实际物理收据均与成功原生记录匹配，才确认它没有隐藏的远端请求。
    """
    result = read(cell / 'result.json')
    root = result.get('root_id')
    native = read(cell / 'model-calls.json')
    physical = [r for r in snapshot['reservations'] if r['root_id'] == root]
    issues = []
    if result.get('status') != 'executed' or not root:
        issues.append('cell_execution_not_complete')
    if any(r['status'] in {'reserved', 'sent', 'unknown'} for r in physical):
        issues.append('unresolved_physical_receipt')
    if any(r['status'] == 'not_sent' for r in physical):
        issues.append('unexpected_physical_not_sent')
    if any(r['status'] not in {'settled', 'reserved', 'sent', 'unknown', 'not_sent'} for r in physical):
        issues.append('invalid_physical_receipt_status')
    by_id = {}
    native_not_sent = []
    native_status_counts = {}
    physical_status_counts = {}
    for receipt in native:
        status = receipt['status']
        native_status_counts[status] = native_status_counts.get(status, 0) + 1
        if status == 'not_sent':
            try:
                response = json.loads(receipt['response_json'])
                usage = json.loads(receipt['usage_json'])
                shape_valid = (receipt.get('sent_state') == 'not_sent'
                    and receipt.get('request_id') is None and not response.get('id')
                    and usage == {})
            except (ValueError, TypeError, AttributeError):
                response, usage, shape_valid = None, None, False
            native_not_sent.append({'call_id': receipt.get('call_id'),
                'status': status, 'sent_state': receipt.get('sent_state'),
                'request_id': receipt.get('request_id'), 'response': response,
                'usage': usage, 'local_record_consistent': shape_valid})
            # 即使能证明未发送，当前冻结批次也不允许跳过失败分支后继续。
            issues.append('unexpected_native_not_sent')
            continue
        if status != 'succeeded':
            issues.append('native_response_' + status)
            continue
        try:
            response = json.loads(receipt['response_json'])
            identity = response['id']
            if not isinstance(identity, str) or not identity:
                raise ValueError('invalid response identity')
        except (ValueError, TypeError, KeyError):
            issues.append('native_response_identity_missing_or_invalid')
            continue
        if identity in by_id:
            issues.append('duplicate_native_response_identity')
        try:
            usage = json.loads(receipt['usage_json'])
            values = [usage[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
            if (any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values)
                    or values[0] + values[1] != values[2]):
                raise ValueError('invalid usage')
        except (ValueError, TypeError, KeyError):
            issues.append('native_usage_missing_or_invalid')
            values = None
        if receipt.get('request_id') not in {None, identity}:
            issues.append('native_request_response_identity_mismatch')
        if receipt.get('sent_state', 'sent') != 'sent':
            issues.append('native_success_sent_state_invalid')
        by_id[identity] = values
    physical_by_id = {}
    for receipt in physical:
        status = receipt['status']
        physical_status_counts[status] = physical_status_counts.get(status, 0) + 1
        if status != 'settled':
            continue
        try:
            meta = json.loads(receipt['response_json'])
            identity = meta['response_id']
            if not isinstance(identity, str) or not identity:
                raise ValueError('invalid physical response identity')
        except (KeyError, TypeError, ValueError):
            issues.append('physical_response_identity_missing_or_invalid')
            continue
        if identity in physical_by_id:
            issues.append('duplicate_physical_response_identity')
        values = [receipt.get('actual_input'), receipt.get('actual_output'), receipt.get('actual_tokens')]
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values)
                or values[0] + values[1] != values[2]):
            issues.append('physical_usage_missing_or_invalid')
            values = None
        physical_by_id[identity] = values
    native_ids, physical_ids = set(by_id), set(physical_by_id)
    unmatched_native = sorted(native_ids - physical_ids)
    unmatched_physical = sorted(physical_ids - native_ids)
    if unmatched_native:
        issues.append('native_success_without_physical_receipt')
    if unmatched_physical:
        issues.append('physical_receipt_without_native_success')
    matched = []
    for identity in sorted(native_ids & physical_ids):
        if by_id[identity] is None or physical_by_id[identity] is None:
            continue
        if by_id[identity] != physical_by_id[identity]:
            issues.append('native_physical_usage_mismatch')
        else:
            matched.append({'response_id': identity, 'usage': by_id[identity]})
    complete_remote_match = (len(matched) == len(physical)
        and len(matched) == native_status_counts.get('succeeded', 0)
        and set(native_status_counts) <= {'succeeded', 'not_sent'}
        and not any(issue.startswith(('duplicate_', 'native_request_response_',
            'native_success_sent_state_')) for issue in issues))
    for receipt in native_not_sent:
        receipt['classification'] = ('confirmed_not_sent' if receipt['local_record_consistent']
            and complete_remote_match else 'not_sent_confirmation_ambiguous')
        if receipt['classification'] != 'confirmed_not_sent':
            issues.append('native_not_sent_not_confirmed')
    for actions in read(cell / 'actions.json').values():
        for action in actions:
            data = (action.get('observation') or {}).get('data') or {}
            if action['tool_name'] == 'retrieve_evidence' and action['status'] == 'succeeded':
                if (data.get('status') in {'degraded', 'unavailable'} or data.get('degradations')
                        or data.get('strategy', {}).get('name') != 'located_fp16'
                        or data.get('effective_path') != 'located_bm25_rerank'):
                    issues.append('actual_retrieval_route_or_readiness_changed')
    return {'issues': sorted(set(issues)), 'native_records': len(native),
        'native_status_counts': native_status_counts,
        'physical_reservations': len(physical), 'physical_status_counts': physical_status_counts,
        'matched_successful_remote_receipts': matched,
        'native_not_sent': native_not_sent,
        'unmatched_native_response_ids': unmatched_native,
        'unmatched_physical_response_ids': unmatched_physical}


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
    if manifest.get('experiment_kind') in SUPERVISOR_KINDS:
        assert_fresh_gate_root(manifest)
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
    if manifest.get('experiment_kind') in SUPERVISOR_KINDS:
        initialize_scope_session(manifest, started)
    deadline = ledger.snapshot()['batch_absolute_deadline']
    # 留出终止、收据读取及父驱动落盘时间，不把最后一请求送到总截止之后。
    finish_reserve = 5.0
    rows = []
    stop = None
    script = Path(__file__).with_name('run_read_paper_ab.py')
    manifest_sha256 = manifest_identity(manifest)
    for index, (case, variant) in enumerate(order):
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
        if manifest.get('experiment_kind') in SUPERVISOR_KINDS:
            try:
                issue_cell_grant(manifest, case, variant, cell_deadline)
            except (OSError, ValueError, KeyError, TypeError, PermissionError):
                stop = 'cell_scheduler_grant_invalid'
                break
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
            reconciliation = (reconcile_cell_report(cell, snapshot) if (cell / 'result.json').exists()
                else {'issues': ['cell_result_missing']})
            issues = reconciliation['issues']
            write(runtime / (case + '-' + variant + '-reconciliation.json'), reconciliation)
            if issues:
                stop = 'cell_receipt_or_isolation_gate'
            result = read(cell / 'result.json') if (cell / 'result.json').exists() else {}
            row = {'case_id': case, 'variant': variant, 'status': result.get('task_status', 'failed'),
                'root_id': result.get('root_id'), 'issues': issues, 'returncode': run.returncode,
                'native_status_counts': reconciliation.get('native_status_counts'),
                'physical_status_counts': reconciliation.get('physical_status_counts')}
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
        if not stop and manifest.get('experiment_kind') in SUPERVISOR_KINDS and index % 2 == 1:
            print(json.dumps({'event': 'pair_review_required', 'case_id': case,
                'control_root': manifest['semantic_gate_root']}, ensure_ascii=False), flush=True)
            try:
                decision = wait_pair_review(manifest, case, rows, manifest_sha256, deadline - finish_reserve)
                if decision['status'] != 'passed':
                    stop = 'semantic_pair_gate:' + ','.join(decision['stop_reasons'])
            except BaseException as exc:
                stop = 'semantic_pair_gate_invalid:' + type(exc).__name__
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
