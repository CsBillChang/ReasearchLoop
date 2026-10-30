"""V5根预算恢复的合成回归；不构造Ledger/Settings，不访问网络或真实实验材料。"""
from copy import deepcopy
import ast
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import agent_final_completion as c
import agent_final_completion_chain as chain
import agent_final_root_budget_recovery as recovery


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root):
    def read(relative, parse=True):
        return json.loads((root / relative).read_text()) if parse else None
    return read


def seal(root):
    excluded = {'SEAL.json', 'SEAL.sha256', 'SEAL-READBACK.json'}
    files = {str(p.relative_to(root)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in root.rglob('*') if p.is_file() and str(p.relative_to(root)) not in excluded}
    write(root / 'SEAL.json', {'files': files})
    digest = sha(root / 'SEAL.json'); (root / 'SEAL.sha256').write_text(digest + '  SEAL.json\n')
    write(root / 'SEAL-READBACK.json', {'status': 'passed', 'seal_sha256': digest})
    return digest


def synthetic_ledger():
    size = 2 ** 20; maximum = 64
    config = {'carried_usage': {'tokens': 123456, 'requests': 7, 'nano_yuan': 123456, 'seconds': 9},
        'root_tokens': 2 * size, 'root_requests': 8, 'root_seconds': 100, 'max_output_tokens': maximum,
        'batch_tokens': 16 * size, 'batch_requests': 100, 'batch_nano_yuan': 100 * size,
        'input_nano_per_token': 2, 'output_nano_per_token': 8, 'cached_input_nano_per_token': 1,
        'enforce_batch_seconds': False, 'batch_seconds': 1000}
    usage = {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5, 'prompt_cache_hit_tokens': 0}
    rows = []
    for variant, start, finish in [('B', 1010.0, 1020.0), ('A', 1040.0, 1060.0)]:
        rows.append({'reservation_id': 'physical-' + variant, 'root_id': 'root-' + variant, 'status': 'settled',
            'created_at': start, 'sent_at': start + 1, 'finished_at': finish, 'reserved_input': size, 'reserved_output': maximum, 'reserved_tokens': size + maximum,
            'reserved_nano': size * 2 + maximum * 8, 'actual_input': 3, 'actual_output': 2, 'actual_tokens': 5,
            'actual_nano': 22, 'usage_json': json.dumps(usage), 'response_json': json.dumps({'response_id': 'response-' + variant})})
    roots = [{'root_id': 'root-B', 'case_id': 'SYNTH-PAIR-1', 'variant': 'B', 'started_at': 1000.0},
             {'root_id': 'root-A', 'case_id': 'SYNTH-PAIR-1', 'variant': 'A', 'started_at': 1030.0}]
    at = 1050.0; root = recovery._totals([rows[1]], at); batch = recovery._totals(rows, at)
    for key, src in [('tokens', 'tokens'), ('requests', 'requests'), ('nano', 'nano_yuan')]:
        batch[key] += config['carried_usage'][src]
    candidate = {'reserved_input': size, 'reserved_output': maximum, 'reserved_tokens': size + maximum,
                 'reserved_nano_yuan': size * 2 + maximum * 8}
    error = {'reason': recovery.STOP_REASON, 'operation': 'reserve', 'root_id': 'root-A', 'reservation_id': None,
        'detected_at': at, 'limits': deepcopy(config), 'root_occupancy': root, 'batch_occupancy': batch, 'details': {'candidate': candidate}}
    return {'metadata': {'config': config, 'first_error': error, 'halt_reason': recovery.STOP_REASON, 'started_at': 1000.0},
            'roots': roots, 'reservations': rows}


def native_records():
    usage = {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5, 'prompt_cache_hit_tokens': 0}
    return [{'call_id': 'sent-call', 'task_id': 'root-A', 'status': 'succeeded', 'sent_state': 'sent', 'request_id': 'response-A',
        'response_json': json.dumps({'id': 'response-A'}), 'usage_json': json.dumps(usage)},
        {'call_id': 'denied-call', 'task_id': 'root-A', 'status': 'not_sent', 'sent_state': 'not_sent', 'request_id': None,
        'response_json': json.dumps({'error_type': 'RequestNotSent', 'error': 'H1 budget denied before transport: BudgetHalt: ' + recovery.STOP_REASON}), 'usage_json': '{}'}]


def write_sql(path, tables):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db:
        for table, rows in tables.items():
            columns = list(rows[0])
            db.execute('CREATE TABLE ' + table + '(' + ','.join('"' + k + '"' for k in columns) + ')')
            for row in rows:
                db.execute('INSERT INTO ' + table + ' VALUES (' + ','.join('?' for _ in columns) + ')', [row[k] for k in columns])
        db.commit()


def anchor_fixture(base):
    root = base / 'closed'; root.mkdir(); original = base / 'original'; ledger = synthetic_ledger()
    order = [[f'SYNTH-PAIR-{i}', variant] for i in range(48) for variant in ('B', 'A')]
    cases = {'cases': [{'case_id': f'SYNTH-PAIR-{i}', 'prompt': 'synthetic prompt'} for i in range(48)]}
    write(root / 'inputs/cases.json', cases); write(original / 'inputs/cases.json', cases)
    write(original / 'inputs/material-snapshot.json', {'materials': []})
    scripts = Path(__file__).resolve().parents[2] / 'scripts'
    source_bytes = {name: (scripts / name).read_bytes() for name in
        ('harness_ab_budget.py', 'harness_ab_transport.py', 'run_read_paper_batch.py')}
    tree = ast.parse(source_bytes['harness_ab_budget.py'])
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(ast.unparse(target) == 'self.config' for target in node.targets) and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and key.value in ledger['metadata']['config'] and key.value.endswith('_nano_per_token'):
                    value.value = ledger['metadata']['config'][key.value]
    source_bytes['harness_ab_budget.py'] = ast.unparse(tree).encode()
    frozen = {'backend/scripts/' + name: hashlib.sha256(content).hexdigest() for name, content in source_bytes.items()}
    manifest = {'experiment_root': str(root), 'runtime_root': str(root / 'runtime'), 'execution_order': order,
        'completion_contract': {'version': 'agent_final_completion_v4'}, 'loaded_prompt_contract': {'A': {}, 'B': {}},
        'ledger_limits': {'carried_usage': ledger['metadata']['config']['carried_usage']}, 'github_commit': 'a' * 40,
        'prompt_bundle_source_contract': {'variants': {v: {'source_files': frozen} for v in ('A', 'B')}}}
    write(root / 'manifest.formal.json', manifest)
    old = [{'root_id': 'old-' + str(i), 'case_id': order[i][0], 'variant': order[i][1], 'quality': 'partial_usable'} for i in range(2)]
    parent = {'retained_count': 2, 'retained_rows': old, 'technical_attempts': [{'root_id': 'technical-' + str(i)} for i in range(3)],
        'full_execution_order': order, 'closed_ledger_roots': [str(original)], 'pending_reservations': [{'reservation_id': 'old-unknown'}],
        'carried_usage': deepcopy(ledger['metadata']['config']['carried_usage']), 'anchor_seals': {}, 'deferred_review_cases': [],
        'anchor_manifest': {'cases_path': str(original / 'inputs/cases.json'), 'prepared_input_root': str(original / 'inputs')},
        'original_clock': {'round_started_at': 1000.0, 'batch_seconds': 1000}, 'observation_recovery_lineage': [{'synthetic': True}]}
    nonce = 'synthetic-session'; write(root / 'semantic-gates/session.json', {'manifest_sha256': c._digest(manifest), 'nonce': nonce})
    rows = []
    for position, variant in [(2, 'B'), (3, 'A')]:
        raw = {'case_id': order[position][0], 'variant': variant, 'root_id': 'root-' + variant, 'status': 'partial',
            'issues': deepcopy(recovery._ERRORS) if variant == 'A' else [], 'returncode': 0, 'original_position': position,
            'native_status_counts': {'succeeded': 1, 'not_sent': 1} if variant == 'A' else {'succeeded': 1},
            'physical_status_counts': {'settled': 1}}
        rows.append(raw)
        grant = {'position': position, 'case_id': raw['case_id'], 'variant': variant, 'manifest_sha256': c._digest(manifest),
            'continuation_sha256': c._digest(manifest['completion_contract']), 'session_nonce': nonce, 'attempt_kind': 'first_attempt', 'attempt_id': str(position) * 64}
        write(root / f'semantic-gates/grants/{position}.json', grant)
        cell = root / 'runtime/cells' / (raw['case_id'] + '-' + variant)
        result = {'status': 'executed', 'case_id': raw['case_id'], 'variant': variant, 'root_id': raw['root_id'], 'task_status': 'partial',
            'ledger_halt': recovery.STOP_REASON if variant == 'A' else None, 'loaded_supervisor': {}, 'isolation': {'denials': []},
            'live_isolation_guard': {'first_error': None, 'closed': False, 'ledger_bound': True, 'persistence_errors': []},
            'experiment_attempt': {k: grant[k] for k in ('attempt_id', 'attempt_kind', 'position', 'continuation_sha256')}}
        write(cell / 'result.json', result)
        if variant == 'B':
            physical = ledger['reservations'][0]
            write(cell / 'requests/physical-B.request.json', {'synthetic': True})
            write(cell / 'requests/physical-B.response.json', {'id': 'response-B', 'usage': json.loads(physical['usage_json'])})
            continue
        body = {'answer_markdown': 'Synthetic incomplete result.'}; view = {'task_id': raw['root_id'], 'status': 'partial', 'prompt': 'synthetic prompt', 'result': body}
        stored = {'task_id': raw['root_id'], 'status': 'partial', 'prompt': 'synthetic prompt', 'result_json': json.dumps(body), 'parent_task_id': None, 'root_task_id': raw['root_id']}
        native = native_records()
        write(cell / 'task-view.json', view); write(cell / 'root-row.json', stored); write(cell / 'model-calls.json', native); write(cell / 'actions.json', {})
        write(cell / 'events.json', {'task_id': raw['root_id'], 'events': [{'task_id': raw['root_id'], 'event_type': event,
            'created_at': datetime.fromtimestamp(t, timezone.utc).isoformat(), 'payload': {'status': 'partial', 'result': body}}
            for event, t in [('task_created', 1030), ('task_finished', 1061)]]})
        write(cell / 'ledger-snapshot.json', ledger)
        write_sql(cell / 'task.sqlite3', {'tasks': [stored], 'model_calls': native})
        physical = ledger['reservations'][1]
        write(cell / 'requests/physical-A.request.json', {'synthetic': True})
        write(cell / 'requests/physical-A.response.json', {'id': 'response-A', 'usage': json.loads(physical['usage_json'])})
    charge, _ = chain._charged(ledger); carry = parent['carried_usage']
    final = {'limits': deepcopy(ledger['metadata']['config']), 'carried_usage': carry, 'actual_known_tokens': charge['tokens'], 'physical_sent_requests': charge['requests'],
        'actual_known_estimated_nano_yuan': charge['nano_yuan'], 'halt_reason': recovery.STOP_REASON, 'first_error': ledger['metadata']['first_error'],
        'charged_or_reserved': {'tokens': carry['tokens'] + charge['tokens'], 'requests': carry['requests'] + charge['requests'], 'nano': carry['nano_yuan'] + charge['nano_yuan']}}
    write(root / 'runtime/ledger-final.json', final)
    write(root / 'runtime/batch-results.json', {'rows': old + rows, 'retained_observations': 2, 'new_attempted_count': 2,
        'new_roots_planned': 94, 'limits': manifest['ledger_limits'], 'not_created': order[4:], 'status': 'stopped', 'stop_reason': recovery.BATCH_STOP})
    write(root / 'runtime/SYNTH-PAIR-1-A-reconciliation.json', {'issues': recovery._ERRORS, 'ledger_halt_reason': recovery.STOP_REASON,
        'first_error': ledger['metadata']['first_error'], 'unmatched_native_response_ids': [], 'unmatched_physical_response_ids': [],
        'native_not_sent': [{'classification': 'confirmed_not_sent'}]})
    write_sql(root / 'runtime/budget.sqlite3', {'metadata': [{'key': k, 'value': json.dumps(v)} for k, v in ledger['metadata'].items()],
        'roots': ledger['roots'], 'reservations': ledger['reservations']})
    # 只为验证proof绑定形状建立普通合成文件；闭合校验另有专门测试。
    required = [recovery.SAFETY_PATH, 'audit/post-run-protection.json', 'audit/post-run-protection-corrected.json',
        'audit/failures/post-audit-missing-github-readback.json', 'audit/post-run-github-readback.json', 'private-protection/post_run_protection.py',
        'formal-execution-auth.json', 'formal-preflight/authorization-closed.json', 'formal-preflight/vector-close.json',
        'formal-control/controller-result.json', 'audit/readonly-post-run.json']
    required += ['snapshots/' + v + '/backend/scripts/' + name for v in ('A', 'B')
        for name in ('harness_ab_budget.py', 'harness_ab_transport.py', 'run_read_paper_batch.py')]
    for name in required:
        if name.startswith('snapshots/'):
            destination = root / name; destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source_bytes[destination.name])
        else:
            write(root / name, {'synthetic': True})
    proof = {'version': recovery.PROOF_VERSION, 'status': 'passed_exact_root_token_reservation_stop', 'anchor_root': str(root),
        'parent_contract_sha256': c._digest(manifest['completion_contract']), 'case_id': 'SYNTH-PAIR-1', 'position': 3, 'root_id': 'root-A',
        'original_batch_stop_reason': recovery.BATCH_STOP, 'original_ledger_halt_reason': recovery.STOP_REASON,
        'original_reconciliation_issues': recovery._ERRORS, 'original_records_unchanged': True, 'original_reconciliation_passed': False,
        'no_retry_or_reclassification': True, 'actions': dict.fromkeys(recovery._ACTIONS, 0), 'root_budget_evidence': recovery._budget_evidence(ledger, 'root-A')}
    proof['anchor_seal_sha256'] = seal(root)
    proof['bindings'] = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file() and not p.name.startswith('SEAL')}
    path = base / 'new-proof/proof.json'; write(path, proof)
    return {'root': root, 'parent': parent, 'ledger': ledger, 'manifest': manifest, 'proof': proof,
            'ref': {'version': recovery.PROOF_VERSION, 'path': str(path), 'sha256': sha(path)}}


class RootBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.f = anchor_fixture(Path(self.temp.name).resolve())
        c.clear_validation_caches(); self.addCleanup(c.clear_validation_caches)

    def build(self):
        def clean_row(root, manifest, position, ledger, load, **kwargs):
            row = deepcopy(load('runtime/batch-results.json')['rows'][position]); row.update(quality=None, retained=True)
            return row
        with patch.object(chain, '_parent', return_value=self.f['parent']), patch.object(chain, '_closure'), \
                patch.object(chain, '_ledger_limits'), patch.object(c, '_row', side_effect=clean_row):
            return recovery.build_root_budget_recovery_contract(self.f['root'], None, proof_ref=self.f['ref'])

    def test_real_sqlite_partial_stays_primary_and_carry_added_once(self):
        value = self.build()
        self.assertEqual((value['retained_count'], len(value['new_execution_order']), len(value['technical_attempts'])), (4, 92, 3))
        self.assertEqual(value['minimum_complete_attempts'], 99)
        self.assertEqual(value['retained_rows'][-1]['issues'], recovery._ERRORS)
        self.assertFalse(value['retained_rows'][-1]['original_reconciliation_passed'])
        self.assertEqual(value['retained_rows'][-1]['status'], 'partial')
        self.assertIsNone(value['retained_rows'][-1]['quality'])
        self.assertEqual(value['retry_positions'], [])
        self.assertEqual(value['carried_usage']['tokens'], self.f['parent']['carried_usage']['tokens'] + 10)
        self.assertEqual(value['carried_usage']['requests'], self.f['parent']['carried_usage']['requests'] + 2)
        self.assertEqual(value['pending_reservations'], self.f['parent']['pending_reservations'])

    def test_first_error_replay_retains_inflight_reservation(self):
        evidence = recovery._budget_evidence(self.f['ledger'], 'root-A')
        self.assertGreater(evidence['root_occupancy']['tokens'], evidence['final_root']['tokens'])
        self.assertEqual(evidence['final_root']['tokens'], 5)

    def test_global_exhaustion_other_guard_or_unknown_never_recovered(self):
        for mutation in ['batch_tokens', 'batch_money', 'batch_requests', 'root_requests', 'root_time', 'unknown', 'occupancy', 'candidate', 'reason', 'post_halt', 'sent_after_halt', 'other_root_sent_after_halt', 'time_bool', 'time_nan', 'reserved_input', 'reserved_output', 'reserved_tokens', 'reserved_nano']:
            ledger = deepcopy(self.f['ledger']); config = ledger['metadata']['config']; error = ledger['metadata']['first_error']
            if mutation == 'batch_tokens': config['batch_tokens'] = 1
            elif mutation == 'batch_money': config['batch_nano_yuan'] = 1
            elif mutation == 'batch_requests': config['batch_requests'] = 1
            elif mutation == 'root_requests': config['root_requests'] = 1
            elif mutation == 'root_time': config['root_seconds'] = 1
            elif mutation == 'unknown': ledger['reservations'][1]['status'] = 'unknown'
            elif mutation == 'occupancy': error['root_occupancy']['tokens'] += 1
            elif mutation == 'candidate': error['details']['candidate']['reserved_input'] = 123456
            elif mutation == 'reason': error['reason'] = 'another halt'
            elif mutation == 'post_halt': ledger['reservations'][1]['created_at'] = 1051.0
            elif mutation == 'sent_after_halt': ledger['reservations'][1]['sent_at'] = 1051.0
            elif mutation == 'other_root_sent_after_halt':
                ledger['reservations'][0].update(sent_at=1051.0, finished_at=1060.0)
            elif mutation == 'time_bool': ledger['reservations'][1]['sent_at'] = True
            elif mutation == 'time_nan': ledger['reservations'][1]['finished_at'] = float('nan')
            elif mutation in {'reserved_input', 'reserved_output', 'reserved_tokens', 'reserved_nano'}:
                ledger['reservations'][1][mutation] += 1
            error['limits'] = deepcopy(config)
            with self.subTest(mutation=mutation), self.assertRaises(PermissionError): recovery._budget_evidence(ledger, 'root-A')


    def test_inflight_settlement_after_halt_remains_allowed(self):
        ledger = deepcopy(self.f['ledger'])
        self.assertGreater(ledger['reservations'][1]['finished_at'], ledger['metadata']['first_error']['detected_at'])
        self.assertEqual(recovery._budget_evidence(ledger, 'root-A')['final_root']['tokens'], 5)

    def test_other_root_settlement_rate_is_checked(self):
        ledger = deepcopy(self.f['ledger']); ledger['reservations'][0]['actual_nano'] += 1
        with self.assertRaisesRegex(PermissionError, 'PHYSICAL_FROZEN_RATE_CHANGED'):
            recovery._budget_evidence(ledger, 'root-A')

    def test_source_hash_and_pre_transport_admission_are_verified(self):
        root = self.f['root']; proof = deepcopy(self.f['proof']); manifest = deepcopy(self.f['manifest'])
        recovery._source_contract(root, manifest, proof, load(root))
        for variant in ('A', 'B'):
            relative = 'snapshots/' + variant + '/backend/scripts/harness_ab_transport.py'; path = root / relative
            path.write_text(path.read_text().replace('input_upper_bound=1048576', 'input_upper_bound=123456'))
            proof['bindings'][relative] = sha(path)
            manifest['prompt_bundle_source_contract']['variants'][variant]['source_files']['backend/scripts/harness_ab_transport.py'] = sha(path)
        with self.assertRaisesRegex(PermissionError, 'PRE_TRANSPORT_BUDGET_REFUSAL_CHANGED'):
            recovery._source_contract(root, manifest, proof, load(root))

    def test_not_sent_requires_exact_before_transport_receipt(self):
        physical = [self.f['ledger']['reservations'][1]]; records = native_records()
        self.assertEqual(recovery._receipts('root-A', records, physical)['native_status_counts'], {'succeeded': 1, 'not_sent': 1})
        for field, value in [('sent_state', 'sent'), ('request_id', 'remote-id'), ('usage_json', '{"total_tokens":0}'), ('status', 'unknown'), ('response_json', '{"error":"other"}')]:
            bad = deepcopy(records); bad[1][field] = value
            with self.subTest(field=field), self.assertRaises(PermissionError): recovery._receipts('root-A', bad, physical)
        with self.assertRaises(PermissionError): recovery._receipts('root-A', records, [])
        with self.assertRaises(PermissionError): recovery._receipts('root-A', records + [records[-1]], physical)

    def test_changed_physical_rate_is_rejected(self):
        physical = deepcopy([self.f['ledger']['reservations'][1]]); physical[0]['actual_nano'] += 1
        with self.assertRaisesRegex(PermissionError, 'PHYSICAL_FROZEN_RATE_CHANGED'):
            recovery._receipts('root-A', native_records(), physical, self.f['ledger']['metadata']['config'])

    def test_external_proof_hash_change_invalidates_hot_context(self):
        self.build(); path = Path(self.f['ref']['path']); proof = deepcopy(self.f['proof']); proof['root_id'] = 'wrong-root'; write(path, proof)
        with self.assertRaises(PermissionError): self.build()

    def test_new_proof_reference_has_distinct_cached_identity(self):
        self.build(); path = Path(self.f['ref']['path']).with_name('other.json'); proof = deepcopy(self.f['proof']); proof['no_retry_or_reclassification'] = False
        write(path, proof); self.f['ref'] = {'version': recovery.PROOF_VERSION, 'path': str(path), 'sha256': sha(path)}
        with self.assertRaises(PermissionError): self.build()

    def test_sql_rate_must_equal_frozen_constructor_literal(self):
        root = self.f['root']; proof = deepcopy(self.f['proof']); path = root / 'runtime/ledger-final.json'
        final = json.loads(path.read_text()); final['limits']['input_nano_per_token'] += 1; write(path, final)
        proof['bindings']['runtime/ledger-final.json'] = sha(path)
        with self.assertRaisesRegex(PermissionError, 'LEDGER_SOURCE_RATE_CONFIG_CHANGED'):
            recovery._source_contract(root, self.f['manifest'], proof, load(root))

    def test_missing_binding_or_forged_gate_success_is_rejected(self):
        for kind in ['binding', 'gate', 'action']:
            proof = deepcopy(self.f['proof'])
            if kind == 'binding': proof['bindings'].pop('runtime/budget.sqlite3')
            elif kind == 'gate': proof['original_reconciliation_passed'] = True
            else: proof['actions']['Ledger_instances'] = 1
            write(Path(self.f['ref']['path']), proof); self.f['ref']['sha256'] = sha(Path(self.f['ref']['path']))
            with self.subTest(kind=kind), self.assertRaises(PermissionError): self.build()

    def test_retained_helper_matches_current_contract_and_original_bytes(self):
        context = self.build(); manifest = {'completion_contract': context}; retained = context['retained_rows'][-1]
        cell = Path(retained['provenance']['cell_root']); result = json.loads((cell / 'result.json').read_text()); view = json.loads((cell / 'task-view.json').read_text())
        with patch.object(c, 'validate_completion', return_value=context):
            proof = recovery.retained_budget_stop_receipts(manifest, cell, result, view, retained, context=context)
        self.assertTrue(proof['root_budget_stop_verified'])
        bad = deepcopy(retained); bad['issues'] = []
        with patch.object(c, 'validate_completion', return_value=context), self.assertRaises(PermissionError):
            recovery.retained_budget_stop_receipts(manifest, cell, result, view, bad, context=context)
        result['root_id'] = 'unrelated'
        with patch.object(c, 'validate_completion', return_value=context):
            self.assertIsNone(recovery.retained_budget_stop_receipts(manifest, cell, result, view, retained, context=context))

    def test_warm_budget_dependency_and_untrusted_context_are_rejected(self):
        context = self.build(); manifest = {'completion_contract': context}; retained = context['retained_rows'][-1]
        cell = Path(retained['provenance']['cell_root']); result = json.loads((cell / 'result.json').read_text()); view = json.loads((cell / 'task-view.json').read_text())
        write(cell / 'model-calls.json', [])
        with patch.object(c, 'validate_completion', return_value=context), self.assertRaisesRegex(PermissionError, 'DEPENDENCY_CHANGED'):
            recovery.retained_budget_stop_receipts(manifest, cell, result, view, retained, context=context)
        untrusted = deepcopy(context); untrusted['retained_count'] += 1
        with patch.object(c, 'validate_completion', return_value=context), self.assertRaisesRegex(PermissionError, 'UNVERIFIED_CURRENT_CONTEXT'):
            recovery.retained_budget_stop_receipts(manifest, cell, result, view, retained, context=untrusted)


    def test_current_manifest_cannot_expand_caps_or_change_ledger_driver(self):
        import agent_final_observation_recovery as observation
        import agent_final_pdf_acquisition as pdf
        contract = self.build(); root = Path(self.temp.name).resolve() / 'next'
        original = deepcopy(self.f['parent']['anchor_manifest']); original['ledger_limits'] = {'root_tokens': 123456}
        base = {**contract, 'anchor_manifest': original, 'latest_anchor_manifest': self.f['manifest'], 'old_cases': {}, 'old_materials': {}}
        manifest = {'completion_contract': contract, 'experiment_kind': 'prompt_bundle_v1', 'evaluation_policy': c.EVALUATION_POLICY,
            'execution_order': contract['full_execution_order'], 'source_acquisition_contract': pdf.SOURCE_ACQUISITION_CONTRACT,
            'ledger_limits': {'root_tokens': 123456, 'carried_usage': contract['carried_usage'], 'enforce_batch_seconds': False},
            'experiment_root': str(root), 'authorization_path': str(root / 'new-auth.json'),
            'real_execution_released': False, 'live_qualification_released': False,
            'source_roots': {v: str(root / 'snapshots' / v) for v in ('A', 'B')},
            'prompt_bundle_source_contract': deepcopy(self.f['manifest']['prompt_bundle_source_contract']),
            'carried_budget_source': {'version': recovery.VERSION, 'anchor_root': contract['anchor_root'],
                'ledger_path': str(Path(contract['anchor_root']) / 'runtime/budget.sqlite3'),
                'ledger_sha256': contract['anchors']['runtime/budget.sqlite3'], 'seal_sha256': contract['anchor_seals'][contract['anchor_root']]}}
        for key, rel in {'runtime_root': 'runtime', 'semantic_gate_root': 'semantic-gates', 'prepared_input_root': 'inputs',
                         'cases_path': 'inputs/cases.json', 'model_manifest_path': 'inputs/model-manifest.json'}.items():
            manifest[key] = str(root / rel)
        with patch.object(recovery, 'build_root_budget_recovery_contract', return_value=contract), \
                patch.object(recovery, '_anchor', return_value=base), patch.object(observation, 'validate_observation_policy', return_value=True), \
                patch.object(pdf, 'validate_source_acquisition_contract', return_value=True), \
                patch.object(c._helpers(), '_check_inputs'), patch.object(c, '_sources'):
            self.assertEqual(recovery.validate_root_budget_recovery(manifest)['version'], recovery.VERSION)
            bad = deepcopy(manifest); bad['ledger_limits']['root_tokens'] += 1
            with self.assertRaisesRegex(PermissionError, 'ORIGINAL_FINANCIAL_AND_ROOT_CAPS_REQUIRED'):
                recovery.validate_root_budget_recovery(bad)
            bad = deepcopy(manifest); bad['prompt_bundle_source_contract']['variants']['A']['source_files']['backend/scripts/harness_ab_budget.py'] = 'changed'
            with self.assertRaisesRegex(PermissionError, 'LEDGER_OR_BATCH_DRIVER_CHANGED'):
                recovery.validate_root_budget_recovery(bad)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name).resolve()
        r = self.root; actions = dict.fromkeys(['Ledger_instantiations', 'business_Settings_instantiations', 'model_calls', 'business_SQLite_writes', 'pending_releases', 'service_mutations'], 0)
        actions['only_new_report_written'] = True
        write(r / 'manifest.formal.json', {'github_commit': 'a' * 40}); write(r / 'runtime/batch-results.json', {'rows': [1, 2], 'not_created': [3, 4]})
        write(r / 'private-protection/post_run_protection.py', {'synthetic': 'original producer'})
        write(r / 'audit/post-run-protection.json', {'status': 'failed', 'safety_closed': False, 'measurement_complete': False,
            'data_integrity_errors': [], 'errors': [{'check': 'git_refs verified without exception', 'details': {'error_type': 'FileNotFoundError'}}],
            'checks_total': 10, 'checks_passed': 9, 'actions': actions})
        write(r / 'audit/post-run-github-readback.json', {'status': 'passed', 'commit': 'a' * 40, 'branch_sha': 'a' * 40,
            'tree': 'b' * 40, 'main_sha': 'c' * 40, 'tag_sha': 'c' * 40, 'main_merged': False, 'private_artifacts_uploaded': False, 'git_ref_mutations': 0})
        paths = {'original_failed_report': 'audit/post-run-protection.json', 'missing_receipt_diagnosis': 'audit/failures/post-audit-missing-github-readback.json',
            'corrected_full_audit': 'audit/post-run-protection-corrected.json', 'post_run_github_readback': 'audit/post-run-github-readback.json',
            'audit_producer': 'private-protection/post_run_protection.py', 'readonly_sql_report': 'audit/readonly-post-run.json', 'controller': 'formal-control/controller-result.json'}
        write(r / paths['missing_receipt_diagnosis'], {'status': 'original_audit_failure_precise_path_verified',
            'original_audit_sha256': sha(r / paths['original_failed_report']), 'producer_sha256': sha(r / paths['audit_producer']),
            'section': 'git_refs', 'exception_type': 'FileNotFoundError', 'missing_path': str(r / paths['post_run_github_readback']),
            'all_preceding_local_git_checks_passed': True, 'source_files_modified': False, 'model_calls': 0, 'network_calls': 0})
        expected = {str(r / paths[k]): sha(r / paths[k]) for k in ['original_failed_report', 'missing_receipt_diagnosis', 'post_run_github_readback', 'audit_producer']}
        write(r / paths['corrected_full_audit'], {'status': 'passed', 'safety_closed': True, 'measurement_complete': False,
            'errors': [], 'data_integrity_errors': [], 'checks_total': 12, 'checks_passed': 12, 'actions': actions,
            'correction_provenance': {'kind': 'fresh_full_original_audit_after_missing_remote_receipt', 'original_audit_source_unchanged': True,
                'original_failed_report_preserved': True, 'checks_skipped_or_overridden': 0, 'harness_business_or_prompt_changes': 0, 'model_calls': 0, 'input_sha256': expected}})
        write(r / paths['readonly_sql_report'], {'status': 'passed_selected_production_materials_unchanged'})
        write(r / paths['controller'], {'cleanup_all_passed': True})
        self.proof = {'schema': 'root_budget_stop_safety_supplement_v1', 'status': recovery.SAFETY_STATUS, 'safety_closed': True,
            'measurement_complete': False, 'original_automatic_gate_passed': False, 'original_failed_report_unchanged': True,
            'old_failures_or_scores_rewritten': False, 'pending_not_reclassified_or_released': True, 'checks_skipped_or_overridden': 0,
            'model_calls': 0, 'service_mutations': 0, 'git_ref_mutations': 0, 'primary_recorded': 2, 'unstarted': 2,
            'evidence_bindings': {k: {'path': str(r / rel), 'sha256': sha(r / rel)} for k, rel in paths.items()}}
        self.paths = paths
        scopes = {'closure', 'containers', 'dataset', 'git_refs', 'old_failure', 'physical_budget_readonly',
                  'processes', 'protected', 'seals', 'selected_production_materials', 'sources'}
        for key in ('original_failed_report', 'corrected_full_audit'):
            path = r / paths[key]; report = json.loads(path.read_text()); report['sections'] = {name: {} for name in scopes}
            report['checks_by_scope'] = {'data': 2, 'safety': report['checks_total'] - 2}
            if key == 'corrected_full_audit':
                report['sections']['git_refs'] = {'remote_readback_sha256': sha(r / paths['post_run_github_readback']), 'git_network_commands_executed_by_audit': 0}
            write(path, report)
        diagnosis_path = r / paths['missing_receipt_diagnosis']; diagnosis = json.loads(diagnosis_path.read_text())
        diagnosis['original_audit_sha256'] = sha(r / paths['original_failed_report']); write(diagnosis_path, diagnosis)
        corrected_path = r / paths['corrected_full_audit']; report = json.loads(corrected_path.read_text())
        report['correction_provenance']['input_sha256'] = {str(r / paths[k]): sha(r / paths[k]) for k in
            ('original_failed_report', 'missing_receipt_diagnosis', 'post_run_github_readback', 'audit_producer')}
        write(corrected_path, report)
        self.proof['evidence_bindings'] = {k: {'path': str(r / rel), 'sha256': sha(r / rel)} for k, rel in paths.items()}
        write(r / recovery.SAFETY_PATH, self.proof)

    def update(self, key, change):
        path = self.root / self.paths[key]; value = json.loads(path.read_text()); change(value); write(path, value)
        self.proof['evidence_bindings'][key]['sha256'] = sha(path); write(self.root / recovery.SAFETY_PATH, self.proof)

    def test_exact_missing_receipt_correction_preserves_failed_report(self):
        before = sha(self.root / self.paths['original_failed_report'])
        self.assertEqual(recovery._safety(self.root, load(self.root))['status'], recovery.SAFETY_STATUS)
        self.assertEqual(sha(self.root / self.paths['original_failed_report']), before)

    def test_other_failure_cannot_be_covered_by_supplement(self):
        self.update('original_failed_report', lambda x: x['errors'].append({'check': 'other'}))
        with self.assertRaisesRegex(PermissionError, 'ORIGINAL_FAILURE_NOT_EXACT'): recovery._safety(self.root, load(self.root))

    def test_missing_path_cannot_be_any_other_file(self):
        self.update('missing_receipt_diagnosis', lambda x: x.update(missing_path=str(self.root / 'other.json')))
        with self.assertRaisesRegex(PermissionError, 'MISSING_RECEIPT_DIAGNOSIS_CHANGED'): recovery._safety(self.root, load(self.root))

    def test_skipped_checks_or_different_producer_are_rejected(self):
        self.update('corrected_full_audit', lambda x: x['correction_provenance'].update(checks_skipped_or_overridden=1))
        with self.assertRaisesRegex(PermissionError, 'CORRECTED_FULL_ORIGINAL_AUDIT_REQUIRED'): recovery._safety(self.root, load(self.root))

    def test_corrected_count_must_match_full_original_audit(self):
        self.update('corrected_full_audit', lambda x: x.update(checks_total=11, checks_passed=11))
        with self.assertRaisesRegex(PermissionError, 'CORRECTED_FULL_ORIGINAL_AUDIT_REQUIRED'): recovery._safety(self.root, load(self.root))


if __name__ == '__main__':
    def guard(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.bind', 'subprocess.Popen', 'os.system'}:
            raise PermissionError('SYNTHETIC_RECOVERY_TEST_NO_NETWORK_OR_PROCESS')
    sys.addaudithook(guard)
    unittest.main()
