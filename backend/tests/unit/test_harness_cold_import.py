"""新进程覆盖共同入口冷导入，提前停止于数据库与模型初始化之前。"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
SOURCE = Path(__file__).resolve().parents[3]
NEURAL_ROOTS = {'torch', 'onnxruntime', 'transformers', 'tokenizers', 'sentence_transformers'}


class ColdImportStop(RuntimeError):
    """身份检查已抵达；此回归不继续构造数据库或模型。"""


def _prompt_digest(source):
    tree = ast.parse((source / 'backend/research_loop/agents/supervisor.py').read_text())
    prompt = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'SYSTEM_PROMPT' for target in node.targets))
    return hashlib.sha256(prompt.encode()).hexdigest()


def _child(mode, root, runner_path, source):
    """从实际入口或角色冷导入开始审计，拒绝网络和神经库加载。"""
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(SCRIPTS), str(source / 'backend')]
    attempts = []

    class NoNeuralImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.', 1)[0] in NEURAL_ROOTS:
                attempts.append(fullname)
                raise RuntimeError('COLD_IMPORT_NEURAL_LIBRARY_FORBIDDEN')
            return None

    assert not any(name in sys.modules for name in NEURAL_ROOTS)
    sys.meta_path.insert(0, NoNeuralImports())
    from harness_ab_boundary import Boundary, IsolationError
    from harness_ab_experiment import HANDOFF_KIND
    spec = importlib.util.spec_from_file_location('cold_import_runner', runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    manifest = {'experiment_kind': HANDOFF_KIND,
        'source_roots': {'A': str(source), 'B': str(source)},
        'handoff_source_contract': {'variants': {'A': {'system_prompt_sha256': _prompt_digest(source)}}}}
    runtime = root / 'runtime'
    runtime.mkdir()
    cases = root / 'cases.json'
    cases.write_text(json.dumps({'cases': [{'case_id': 'COLD-TEST'}]}))
    frozen = root / 'frozen-source.txt'
    frozen.write_text('synthetic frozen source identity')
    manifest.update(runtime_root=str(runtime), cases_path=str(cases), vector_port=6396,
        official_ip_addresses=[], frozen_files=[{'path': str(frozen),
        'sha256': hashlib.sha256(frozen.read_bytes()).hexdigest()}])
    if mode == 'frozen_drift':
        manifest['frozen_files'][0]['sha256'] = '0' * 64
    if mode == 'prompt_drift':
        manifest['handoff_source_contract']['variants']['A']['system_prompt_sha256'] = '0' * 64
    boundary = Boundary(root)
    boundary.phase = 'cold_import'
    sys.addaudithook(boundary.audit)
    verification, error, loaded, report = [], None, None, None
    try:
        if mode in {'entry', 'frozen_drift'}:
            def stop_at_verification(*args):
                verification.append({'ipv6_disabled': socket.has_ipv6 is False})
                raise ColdImportStop('ZERO_MODEL_STOP_BEFORE_SETTINGS_DB_RAG')

            # 合成配置无需通过材料合同；实际入口、边界及环境补丁保持原实现。
            with patch.object(runner, 'select_variant_source', return_value=source), \
                    patch.object(runner, 'validate_manifest'), \
                    patch.object(runner, 'verify_loaded_supervisor', side_effect=stop_at_verification):
                report = runner.run_cell(manifest, 'COLD-TEST', 'A', 'qualify')
        else:
            assert 'urllib3' not in sys.modules, 'requires cold urllib3 import'
            with patch.object(socket, 'has_ipv6', mode == 'unpatched'):
                loaded = runner.verify_loaded_supervisor(manifest, 'A')
                if mode == 'business_drift':
                    from research_loop.agents import researcher
                    with patch.object(researcher, '__file__', str(root / 'other-source.py')):
                        runner.verify_loaded_supervisor(manifest, 'A')
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    denials = list(boundary.denials)
    boundary.phase = 'manual_bind_selfcheck'
    bind_blocked = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(('127.0.0.1', 0))
        except IsolationError:
            bind_blocked = True
    value = {'mode': mode, 'verification': verification, 'error': error, 'report': report,
        'loaded_source': loaded, 'import_denials': denials, 'manual_bind_blocked': bind_blocked,
        'manual_bind_denials': boundary.denials[len(denials):], 'sqlite_connections': len(boundary.sqlite_paths),
        'neural_import_attempts': attempts,
        'neural_modules_loaded': sorted(name for name in sys.modules if name.split('.', 1)[0] in NEURAL_ROOTS),
        'generation_calls': 0, 'local_neural_calls': 0, 'rag_runs': 0}
    (root / 'runtime' / 'probe-result.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'result': str(root / 'runtime' / 'probe-result.json')}))
    return 0


class HarnessColdImportTests(unittest.TestCase):
    """每格使用新解释器，禁止由其他用例预热urllib3来掩盖副作用。"""

    def probe(self, mode):
        with tempfile.TemporaryDirectory(prefix='harness-cold-import-') as temporary:
            root = Path(temporary).resolve()
            runner = Path(os.environ.get('HARNESS_COLD_RUNNER', str(SCRIPTS / 'run_read_paper_ab.py')))
            source = Path(os.environ.get('HARNESS_COLD_SOURCE', str(SOURCE)))
            result = subprocess.run([sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                '--probe', mode, '--root', str(root), '--runner', str(runner), '--source', str(source)],
                capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            value = json.loads((root / 'runtime' / 'probe-result.json').read_text())
            self.assertTrue(value['manual_bind_blocked'])
            self.assertEqual([row['operation'] for row in value['manual_bind_denials']], ['socket.bind'])
            self.assertEqual(value['sqlite_connections'], 0)
            self.assertEqual(value['neural_import_attempts'], [])
            self.assertEqual(value['neural_modules_loaded'], [])
            return value

    def test_entry_patches_ipv6_before_verification_and_stops_before_resources(self):
        value = self.probe('entry')
        self.assertEqual(value['verification'], [{'ipv6_disabled': True}])
        self.assertIsNone(value['error'])
        self.assertEqual(value['report']['exception_type'], 'ColdImportStop')
        self.assertEqual(value['report']['real_model_calls'], 0)
        self.assertEqual(value['import_denials'], [])

    def test_frozen_source_drift_stops_before_verification(self):
        value = self.probe('frozen_drift')
        self.assertEqual(value['verification'], [])
        self.assertEqual(value['error']['type'], 'ValueError')
        self.assertIn('frozen source or material changed', value['error']['message'])
        self.assertEqual(value['import_denials'], [])

    def test_real_role_import_with_patch_has_no_bind_probe(self):
        value = self.probe('patched')
        self.assertIsNone(value['error'])
        self.assertIsNotNone(value['loaded_source'])
        self.assertEqual(value['import_denials'], [])

    @unittest.skipUnless(socket.has_ipv6, 'system does not advertise IPv6 probe capability')
    def test_original_cold_import_reproduces_denied_ipv6_probe(self):
        value = self.probe('unpatched')
        self.assertIsNone(value['error'])
        self.assertEqual([row['operation'] for row in value['import_denials']], ['socket.bind'])
        self.assertTrue(any(frame['function'] == '_has_ipv6' for frame in value['import_denials'][0]['origin']))

    def test_loaded_prompt_identity_drift_is_rejected(self):
        value = self.probe('prompt_drift')
        self.assertEqual(value['error']['type'], 'PermissionError')
        self.assertIn('SUPERVISOR_LOADED_SOURCE_OR_PROMPT_MISMATCH', value['error']['message'])
        self.assertEqual(value['import_denials'], [])

    def test_loaded_business_module_identity_drift_is_rejected(self):
        value = self.probe('business_drift')
        self.assertEqual(value['error']['type'], 'PermissionError')
        self.assertIn('HANDOFF_LOADED_BUSINESS_SOURCE_MISMATCH', value['error']['message'])
        self.assertEqual(value['import_denials'], [])


if __name__ == '__main__':
    if '--probe' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--probe', required=True)
        parser.add_argument('--root', type=Path, required=True)
        parser.add_argument('--runner', type=Path, required=True)
        parser.add_argument('--source', type=Path, required=True)
        args = parser.parse_args()
        raise SystemExit(_child(args.probe, args.root, args.runner, args.source))
    unittest.main()
