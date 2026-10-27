"""新矩阵范围和双层收据停止门，不访问模型、容器或部署实例。"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import subprocess
import signal

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from run_read_paper_ab import validate_manifest, qualification_configuration, validate_carried_budget_source
from run_read_paper_batch import reconcile_cell, reconcile_cell_report
from launch_read_paper_ab import run_guarded


def fixture():
    from harness_ab_transport import visual_request_contract
    return {'experiment_kind': 'read_paper_exact_duplicate',
        'source_roots': {'A': '/same/source', 'B': '/same/source'},
        'variant_settings': {'A': {'context_deduplicate_read_paper': False},
                             'B': {'context_deduplicate_read_paper': True}},
        'settings': {'retrieval_route': 'located_fp16', 'retrieval_reranker_backend': 'mps_fp16',
            'deepseek_model': 'deepseek-flash', 'deepseek_base_url': 'https://api.deepseek.com',
            'request_timeout_seconds': 120},
        'request_contracts': {'visual': visual_request_contract()},
        'ledger_limits': {'allowed_cells': [[f'DP-R{r}-Q{q}', v] for r in [1, 2] for q in range(1, 5) for v in ['A', 'B']]}}


def successful_native(identity='id'):
    return {'call_id': 'call-' + identity, 'status': 'succeeded', 'sent_state': 'sent',
        'request_id': identity, 'response_json': json.dumps({'id': identity}),
        'usage_json': json.dumps({'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12})}


def settled_physical(identity='id'):
    return {'root_id': 'root', 'status': 'settled',
        'response_json': json.dumps({'response_id': identity}),
        'actual_input': 10, 'actual_output': 2, 'actual_tokens': 12}


def local_not_sent():
    return {'call_id': 'local-call', 'status': 'not_sent', 'sent_state': 'not_sent',
        'request_id': None, 'response_json': json.dumps({'error': 'fixture local refusal'}),
        'usage_json': '{}'}


def write_cell(root, native, actions=None):
    for name, value in {'result.json': {'root_id': 'root', 'status': 'executed'},
            'model-calls.json': native, 'actions.json': actions or {}}.items():
        (root / name).write_text(json.dumps(value))


def attach_carried_source(manifest, root):
    """仅创建合成历史JSON，不接触真实旧账本或Ledger。"""
    batch = {'actual_known_tokens': 540984, 'physical_requests': 9,
        'actual_known_estimated_nano_yuan': 765684960, 'execution_seconds': 157.39793920516968}
    ledger = {'reservations': [{'status': 'settled',
            'actual_tokens': 540984 if index == 0 else 0,
            'actual_nano': 765684960 if index == 0 else 0} for index in range(9)],
        'actual_known_tokens': 540984, 'physical_sent_requests': 9,
        'actual_known_estimated_nano_yuan': 765684960,
        'charged_or_reserved': {'tokens': 540984, 'requests': 9, 'nano': 765684960}}
    source = {}
    for name, value in [('batch_results', batch), ('ledger_final', ledger)]:
        path = root / (name + '.json')
        path.write_text(json.dumps(value))
        source[name + '_path'] = str(path)
        source[name + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['carried_budget_source'] = source
    manifest['ledger_limits']['carried_usage'] = {
        'tokens': 540984, 'requests': 9, 'nano_yuan': 765684960, 'seconds': 158}


class DedupAbRunnerTests(unittest.TestCase):
    def test_outer_deadline_kills_only_owned_batch_group(self):
        process = Mock(pid=12345, returncode=-9)
        process.communicate.side_effect = [subprocess.TimeoutExpired('fixture', 0.1), (None, None)]
        with patch('launch_read_paper_ab.subprocess.Popen', return_value=process) as spawn, \
                patch('launch_read_paper_ab.os.killpg') as kill:
            result = run_guarded(['fixture'], 'secret-in-memory', 1.0, None)
        self.assertEqual(result['status'], 'hard_deadline')
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        kill.assert_called_once_with(12345, signal.SIGKILL)

    def test_both_rounds_are_distinct_and_unapproved_cell_rejected(self):
        manifest = fixture()
        for case, variant in manifest['ledger_limits']['allowed_cells']:
            validate_manifest(manifest, case, variant, 'qualify')
        with self.assertRaises(PermissionError):validate_manifest(manifest, 'DP-R3-Q1', 'A', 'qualify')

    def test_second_variable_or_old_cpu_route_is_rejected(self):
        for key in ['retrieval_route', 'retrieval_reranker_backend']:
            manifest = fixture();manifest['settings'][key] = 'old_route'
            with self.assertRaises(PermissionError):validate_manifest(manifest, 'DP-R1-Q1', 'A', 'qualify')
        manifest = fixture();manifest['variant_settings']['B']['temperature'] = 0.2
        with self.assertRaises(PermissionError):validate_manifest(manifest, 'DP-R1-Q1', 'B', 'qualify')
        manifest = fixture();manifest['source_roots']['B'] = '/different/source'
        with self.assertRaises(PermissionError):validate_manifest(manifest, 'DP-R1-Q1', 'B', 'qualify')

    def test_visual_component_model_endpoint_and_timeout_cannot_drift(self):
        for field, value in [('deepseek_model', 'different-model'),
                ('deepseek_base_url', 'https://different.example'), ('request_timeout_seconds', 60)]:
            with self.subTest(field=field):
                manifest = fixture()
                manifest['settings'][field] = value
                with self.assertRaises(PermissionError):
                    validate_manifest(manifest, 'DP-R1-Q1', 'A', 'qualify')

    def test_execute_requires_freeze_qualification_and_deadline(self):
        manifest = fixture()
        with self.assertRaises(PermissionError):validate_manifest(manifest, 'DP-R1-Q1', 'A', 'execute')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'qualification.json'
            attach_carried_source(manifest, Path(temp))
            manifest.update(real_execution_released=True, frozen_files=[{'path':'fixture'}], github_commit='fixed',
                qualification_receipt=str(path), cell_absolute_deadline=1000.0,
                qualification_cells=[['DP-R1-Q1', 'A']])
            receipt = {'status':'passed', 'real_model_calls':0,
                'configuration':qualification_configuration(manifest),
                'rows':[{'case_id':'DP-R1-Q1','variant':'A','status':'passed','real_model_calls':0}]}
            path.write_text(json.dumps(receipt))
            validate_manifest(manifest, 'DP-R1-Q1', 'A', 'execute')
            receipt['configuration']['github_commit'] = 'another-code-version'
            path.write_text(json.dumps(receipt))
            with self.assertRaises(PermissionError):validate_manifest(manifest, 'DP-R1-Q1', 'A', 'execute')

    def test_visual_contract_is_bound_to_the_qualification_receipt(self):
        manifest = fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'qualification.json'
            attach_carried_source(manifest, Path(temp))
            manifest.update(real_execution_released=True, frozen_files=[{'path': 'fixture'}],
                github_commit='fixed', qualification_receipt=str(path), cell_absolute_deadline=1000.0,
                qualification_cells=[['DP-R1-Q1', 'A']])
            receipt = {'status': 'passed', 'real_model_calls': 0,
                'configuration': qualification_configuration(manifest),
                'rows': [{'case_id': 'DP-R1-Q1', 'variant': 'A', 'status': 'passed', 'real_model_calls': 0}]}
            self.assertEqual(receipt['configuration']['request_contracts'], manifest['request_contracts'])
            path.write_text(json.dumps(receipt))
            validate_manifest(manifest, 'DP-R1-Q1', 'A', 'execute')
            old_receipt = copy.deepcopy(receipt)
            old_receipt['configuration']['request_contracts'] = None
            path.write_text(json.dumps(old_receipt))
            with self.assertRaises(PermissionError):
                validate_manifest(manifest, 'DP-R1-Q1', 'A', 'execute')
            path.write_text(json.dumps(receipt))
            absent_contract = copy.deepcopy(manifest)
            absent_contract.pop('request_contracts')
            with self.assertRaises(PermissionError):
                validate_manifest(absent_contract, 'DP-R1-Q1', 'A', 'execute')
            expanded_contract = copy.deepcopy(manifest)
            expanded_contract['request_contracts']['visual']['max_images'] += 1
            with self.assertRaises(PermissionError):
                validate_manifest(expanded_contract, 'DP-R1-Q1', 'A', 'qualify')

    def test_execute_cannot_omit_or_zero_carried_usage(self):
        manifest = fixture()
        validate_manifest(manifest, 'DP-R1-Q1', 'A', 'qualify')
        with self.assertRaisesRegex(PermissionError, 'DEDUP_CARRIED_BUDGET_SOURCE_REQUIRED'):
            validate_manifest(manifest, 'DP-R1-Q1', 'A', 'execute')
        with tempfile.TemporaryDirectory() as temp:
            attach_carried_source(manifest, Path(temp))
            self.assertEqual(validate_carried_budget_source(manifest),
                {'tokens': 540984, 'requests': 9, 'nano_yuan': 765684960, 'seconds': 158})
            for mutation in [None, {}, {'tokens': 0, 'requests': 0, 'nano_yuan': 0, 'seconds': 0}]:
                with self.subTest(mutation=mutation):
                    altered = copy.deepcopy(manifest)
                    if mutation is None:
                        altered['ledger_limits'].pop('carried_usage')
                    else:
                        altered['ledger_limits']['carried_usage'] = mutation
                    with self.assertRaisesRegex(PermissionError, 'DEDUP_CARRIED_BUDGET_SOURCE_INVALID'):
                        validate_manifest(altered, 'DP-R1-Q1', 'A', 'qualify')

    def test_carried_source_hash_drift_and_receipt_disagreement_are_rejected(self):
        for receipt in ['batch_results', 'ledger_final']:
            with self.subTest(receipt=receipt), tempfile.TemporaryDirectory() as temp:
                manifest = fixture()
                attach_carried_source(manifest, Path(temp))
                path = Path(manifest['carried_budget_source'][receipt + '_path'])
                path.write_text(path.read_text() + '\n')
                with self.assertRaisesRegex(PermissionError, 'DEDUP_CARRIED_BUDGET_SOURCE_INVALID'):
                    validate_carried_budget_source(manifest)
        with tempfile.TemporaryDirectory() as temp:
            manifest = fixture()
            attach_carried_source(manifest, Path(temp))
            path = Path(manifest['carried_budget_source']['batch_results_path'])
            batch = json.loads(path.read_text())
            batch['actual_known_tokens'] -= 1
            path.write_text(json.dumps(batch))
            manifest['carried_budget_source']['batch_results_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(PermissionError, 'DEDUP_CARRIED_BUDGET_SOURCE_INVALID'):
                validate_carried_budget_source(manifest)
        with tempfile.TemporaryDirectory() as temp:
            manifest = fixture()
            attach_carried_source(manifest, Path(temp))
            path = Path(manifest['carried_budget_source']['ledger_final_path'])
            ledger = json.loads(path.read_text())
            ledger['charged_or_reserved'] = {'tokens': 0, 'requests': 0, 'nano': 0}
            path.write_text(json.dumps(ledger))
            manifest['carried_budget_source']['ledger_final_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest['ledger_limits']['carried_usage'] = {
                'tokens': 0, 'requests': 0, 'nano_yuan': 0, 'seconds': 158}
            with self.assertRaisesRegex(PermissionError, 'DEDUP_CARRIED_BUDGET_SOURCE_INVALID'):
                validate_carried_budget_source(manifest)

    def test_carried_source_unresolved_history_cannot_release_new_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = fixture()
            attach_carried_source(manifest, Path(temp))
            path = Path(manifest['carried_budget_source']['ledger_final_path'])
            ledger = json.loads(path.read_text())
            ledger['reservations'][0]['status'] = 'unknown'
            path.write_text(json.dumps(ledger))
            manifest['carried_budget_source']['ledger_final_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(PermissionError, 'DEDUP_CARRIED_BUDGET_SOURCE_INVALID'):
                validate_carried_budget_source(manifest)

    def test_carried_source_detail_totals_and_boolean_values_are_rejected(self):
        for mutation in ['token_sum', 'nano_sum', 'request_count', 'boolean_tokens', 'boolean_nano']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                manifest = fixture()
                attach_carried_source(manifest, Path(temp))
                source = manifest['carried_budget_source']
                ledger_path = Path(source['ledger_final_path'])
                batch_path = Path(source['batch_results_path'])
                ledger = json.loads(ledger_path.read_text())
                batch = json.loads(batch_path.read_text())
                if mutation == 'token_sum':
                    ledger['reservations'][0]['actual_tokens'] -= 1
                elif mutation == 'nano_sum':
                    ledger['reservations'][0]['actual_nano'] -= 1
                elif mutation == 'request_count':
                    ledger['reservations'].pop()
                elif mutation == 'boolean_tokens':
                    ledger['reservations'][0]['actual_tokens'] = True
                    ledger['actual_known_tokens'] = batch['actual_known_tokens'] = 1
                    ledger['charged_or_reserved']['tokens'] = 1
                    manifest['ledger_limits']['carried_usage']['tokens'] = 1
                else:
                    ledger['reservations'][0]['actual_nano'] = True
                    ledger['actual_known_estimated_nano_yuan'] = batch['actual_known_estimated_nano_yuan'] = 1
                    ledger['charged_or_reserved']['nano'] = 1
                    manifest['ledger_limits']['carried_usage']['nano_yuan'] = 1
                for name, path, value in [('ledger_final', ledger_path, ledger), ('batch_results', batch_path, batch)]:
                    path.write_text(json.dumps(value))
                    source[name + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(PermissionError, 'DEDUP_CARRIED_BUDGET_SOURCE_INVALID'):
                    validate_carried_budget_source(manifest)

    def test_native_unknown_stops_even_if_transport_already_settled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_cell(root, [{'status': 'unknown', 'response_json': '{}', 'usage_json': '{}'}])
            snapshot = {'reservations': [settled_physical()]}
            issues = reconcile_cell(root, snapshot)
            self.assertIn('native_response_unknown', issues)
            self.assertIn('physical_receipt_without_native_success', issues)
            self.assertNotIn('native_usage_missing_or_invalid', issues)

    def test_matching_native_and_http_receipts_pass_but_usage_drift_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_cell(root, [successful_native()])
            snapshot = {'reservations': [settled_physical()]}
            self.assertEqual(reconcile_cell(root, snapshot), [])
            snapshot['reservations'][0].update(actual_input=11, actual_tokens=13)
            self.assertIn('native_physical_usage_mismatch', reconcile_cell(root, snapshot))

    def test_confirmed_not_sent_is_separate_and_still_stops_without_fake_missing_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_cell(root, [successful_native(), local_not_sent()])
            report = reconcile_cell_report(root, {'reservations': [settled_physical()]})
            self.assertEqual(report['issues'], ['unexpected_native_not_sent'])
            self.assertEqual(report['native_status_counts'], {'succeeded': 1, 'not_sent': 1})
            self.assertEqual(report['physical_status_counts'], {'settled': 1})
            self.assertEqual(len(report['matched_successful_remote_receipts']), 1)
            self.assertEqual(report['native_not_sent'][0]['classification'], 'confirmed_not_sent')
            self.assertEqual(report['native_not_sent'][0]['usage'], {})
            self.assertEqual(report['physical_reservations'], 1)

    def test_not_sent_with_unmatched_remote_receipt_is_not_confirmed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_cell(root, [successful_native(), local_not_sent()])
            report = reconcile_cell_report(root, {'reservations': [settled_physical(), settled_physical('other')]})
            self.assertIn('physical_receipt_without_native_success', report['issues'])
            self.assertIn('native_not_sent_not_confirmed', report['issues'])
            self.assertEqual(report['native_not_sent'][0]['classification'], 'not_sent_confirmation_ambiguous')

    def test_not_sent_with_remote_identity_or_usage_is_not_confirmed(self):
        for mutation in [{'request_id': 'remote-id'}, {'sent_state': 'sent'},
                {'response_json': '{"id":"remote-id"}'}, {'usage_json': '{"total_tokens":0}'}]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                receipt = {**local_not_sent(), **mutation}
                write_cell(root, [receipt])
                report = reconcile_cell_report(root, {'reservations': []})
                self.assertIn('native_not_sent_not_confirmed', report['issues'])

    def test_unknown_or_unsent_physical_receipts_cannot_be_filtered_away(self):
        for status in ['reserved', 'sent', 'unknown', 'not_sent']:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_cell(root, [local_not_sent()])
                report = reconcile_cell_report(root, {'reservations': [{'root_id': 'root', 'status': status}]})
                self.assertIn('native_not_sent_not_confirmed', report['issues'])
                self.assertIn('unexpected_physical_not_sent' if status == 'not_sent'
                    else 'unresolved_physical_receipt', report['issues'])

    def test_succeeded_native_missing_or_invalid_usage_stops(self):
        for usage in ['{}', '{"prompt_tokens":10,"completion_tokens":2,"total_tokens":13}',
                '{"prompt_tokens":true,"completion_tokens":2,"total_tokens":3}', 'null']:
            with self.subTest(usage=usage), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_cell(root, [{**successful_native(), 'usage_json': usage}])
                self.assertIn('native_usage_missing_or_invalid', reconcile_cell(root,
                    {'reservations': [settled_physical()]}))

    def test_physical_invalid_usage_and_identity_still_stop(self):
        for mutation, expected in [({'actual_tokens': 13}, 'physical_usage_missing_or_invalid'),
                ({'actual_input': True}, 'physical_usage_missing_or_invalid'),
                ({'response_json': '{}'}, 'physical_response_identity_missing_or_invalid')]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_cell(root, [successful_native()])
                self.assertIn(expected, reconcile_cell(root,
                    {'reservations': [{**settled_physical(), **mutation}]}))

    def test_duplicate_native_and_physical_identities_stop(self):
        for duplicate_native in [True, False]:
            with self.subTest(duplicate_native=duplicate_native), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_cell(root, [successful_native()] * (2 if duplicate_native else 1))
                physical = [settled_physical()] * (1 if duplicate_native else 2)
                self.assertIn('duplicate_native_response_identity' if duplicate_native
                    else 'duplicate_physical_response_identity', reconcile_cell(root, {'reservations': physical}))

    def test_success_without_physical_or_changed_request_identity_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_cell(root, [successful_native()])
            self.assertIn('native_success_without_physical_receipt', reconcile_cell(root, {'reservations': []}))
            write_cell(root, [{**successful_native(), 'request_id': 'different-id'}])
            self.assertIn('native_request_response_identity_mismatch', reconcile_cell(root,
                {'reservations': [settled_physical()]}))

    def test_rejected_native_response_stops_even_with_settled_physical(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_cell(root, [{**successful_native(), 'status': 'rejected'}])
            self.assertIn('native_response_rejected', reconcile_cell(root,
                {'reservations': [settled_physical()]}))

    def test_retrieval_degradation_remains_a_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            actions = {'root': [{'tool_name': 'retrieve_evidence', 'status': 'succeeded',
                'observation': {'data': {'status': 'ready', 'degradations': [],
                    'strategy': {'name': 'located_fp16'}, 'effective_path': 'located_bm25_rerank'}}}]}
            write_cell(root, [successful_native()], actions)
            self.assertEqual(reconcile_cell(root, {'reservations': [settled_physical()]}), [])
            actions['root'][0]['observation']['data']['degradations'] = ['fixture degraded']
            write_cell(root, [successful_native()], actions)
            self.assertIn('actual_retrieval_route_or_readiness_changed', reconcile_cell(root,
                {'reservations': [settled_physical()]}))
