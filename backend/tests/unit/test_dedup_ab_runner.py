"""新矩阵范围和双层收据停止门，不访问模型、容器或部署实例。"""
import copy
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
from run_read_paper_ab import validate_manifest, qualification_configuration
from run_read_paper_batch import reconcile_cell
from launch_read_paper_ab import run_guarded


def fixture():
    return {'experiment_kind': 'read_paper_exact_duplicate',
        'source_roots': {'A': '/same/source', 'B': '/same/source'},
        'variant_settings': {'A': {'context_deduplicate_read_paper': False},
                             'B': {'context_deduplicate_read_paper': True}},
        'settings': {'retrieval_route': 'located_fp16', 'retrieval_reranker_backend': 'mps_fp16'},
        'ledger_limits': {'allowed_cells': [[f'DP-R{r}-Q{q}', v] for r in [1, 2] for q in range(1, 5) for v in ['A', 'B']]}}


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

    def test_execute_requires_freeze_qualification_and_deadline(self):
        manifest = fixture()
        with self.assertRaises(PermissionError):validate_manifest(manifest, 'DP-R1-Q1', 'A', 'execute')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'qualification.json'
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

    def test_native_unknown_stops_even_if_transport_already_settled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def write(name, value):(root / name).write_text(json.dumps(value))
            write('result.json', {'root_id':'root','status':'executed'})
            write('model-calls.json', [{'status':'unknown','response_json':'{}','usage_json':'{}'}])
            write('actions.json', {})
            snapshot = {'reservations':[{'root_id':'root','status':'settled','response_json':json.dumps({'response_id':'id'}),
                'actual_input':10,'actual_output':2,'actual_tokens':12}]}
            issues = reconcile_cell(root, snapshot)
            self.assertIn('native_response_not_successful', issues)
            self.assertIn('native_usage_missing_or_invalid', issues)

    def test_matching_native_and_http_receipts_pass_but_usage_drift_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def write(name, value):(root / name).write_text(json.dumps(value))
            write('result.json', {'root_id':'root','status':'executed'})
            write('model-calls.json', [{'status':'succeeded','response_json':json.dumps({'id':'id'}),
                'usage_json':json.dumps({'prompt_tokens':10,'completion_tokens':2,'total_tokens':12})}])
            write('actions.json', {})
            snapshot = {'reservations':[{'root_id':'root','status':'settled','response_json':json.dumps({'response_id':'id'}),
                'actual_input':10,'actual_output':2,'actual_tokens':12}]}
            self.assertEqual(reconcile_cell(root, snapshot), [])
            snapshot['reservations'][0]['actual_tokens'] = 13
            self.assertIn('native_physical_usage_mismatch', reconcile_cell(root, snapshot))
