"""候选准备入口的合成文件、独占输出与零推理合同。"""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_retrieval_candidates as preparation


class PreparationTests(unittest.TestCase):
    """不依赖私有artifact的公开零模型测试。"""

    def setUp(self):
        """建立独立临时根与最小未执行清单。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / preparation.CANDIDATES / 'preparation/candidate.json'
        self.value = {'stage': 'prepared_not_executed', 'counts': dict(preparation.COUNTS),
                      'execution': {'real_model_initialized': False, 'score_calls': 0}}

    def test_reference_hash_size_and_redirect_are_checked(self):
        """文件内容、长度和符号链接均不能悄悄改变输入身份。"""
        path = self.root / 'input.json'
        path.write_text('{}')
        row = preparation.reference(path)
        self.assertEqual(preparation.verify_reference(row, self.root), path)
        with self.assertRaisesRegex(ValueError, 'SIZE_MISMATCH'):
            preparation.verify_reference({**row, 'bytes': 1}, self.root)
        path.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'HASH_MISMATCH'):
            preparation.verify_reference(row, self.root)
        link = self.root / 'redirect.json'
        link.symlink_to(path)
        with self.assertRaisesRegex(ValueError, 'REDIRECTED'):
            preparation.verify_reference({**row, 'path': str(link)}, self.root)

    def test_prepare_is_exclusive_and_does_not_revalidate_existing_output(self):
        """成功只写新文件，重复调用在预检前拒绝覆盖。"""
        with patch.object(preparation, 'build_preparation', return_value=self.value) as build:
            self.assertEqual(preparation.prepare(self.root, self.output), self.value)
            original = self.output.read_bytes()
            with self.assertRaises(FileExistsError):
                preparation.prepare(self.root, self.output)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(self.output.read_bytes(), original)

    def test_validation_failure_leaves_no_completed_manifest_or_directory(self):
        """材料验证失败不产生完成清单。"""
        with patch.object(preparation, 'build_preparation', side_effect=ValueError('fixture hash drift')):
            with self.assertRaisesRegex(ValueError, 'hash drift'):
                preparation.prepare(self.root, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.parent.exists())

    def test_outputs_cannot_target_old_artifacts_outside_or_symlinks(self):
        """360/361/362、相对路径、越界和链接目的地全部拒绝。"""
        forbidden = [self.root / preparation.SOURCE / 'new.json', self.root / preparation.RUNTIME / 'new.json',
                     self.root / preparation.RESULTS / 'new.json', self.root / 'other.json', Path('relative.json')]
        allowed = self.output.parent
        allowed.mkdir(parents=True)
        outside = self.root / 'outside'
        outside.mkdir()
        (allowed / 'redirect').symlink_to(outside, target_is_directory=True)
        forbidden.append(allowed / 'redirect/new.json')
        with patch.object(preparation, 'build_preparation', side_effect=AssertionError('must reject before read')):
            for output in forbidden:
                with self.subTest(output=str(output)), self.assertRaisesRegex(ValueError, 'OUTPUT_OUTSIDE'):
                    preparation.prepare(self.root, output)

    def test_all_three_profiles_are_complete_and_independent(self):
        """未来准备完整记录三个固定策略，只有单独变量改变。"""
        profiles = [preparation.get_retrieval_strategy(name).as_dict() for name in preparation.PROFILES]
        self.assertEqual([p['name'] for p in profiles], list(preparation.PROFILES))
        for profile in profiles:
            self.assertEqual(profile['candidate_limit'], 24)
            self.assertEqual(profile['rrf_k'], 60)
            self.assertEqual(profile['reranker_kwargs']['threads'], 2)
            self.assertEqual(profile['reranker_kwargs']['batch_size'], 4)
        self.assertEqual([p['reranker_kwargs']['sort_by_length'] for p in profiles], [False, False, True])
        self.assertEqual(profiles[0]['fusion_weights'], profiles[2]['fusion_weights'])
        self.assertEqual(profiles[1]['fusion_weights'], {'bm25': 2.0, 'vector': 1.0})

    def test_help_and_validate_never_construct_model_or_read_settings(self):
        """help只解析参数；validate只有只读预检入口，拒绝run命令。"""
        with patch.object(preparation, 'build_preparation', return_value=self.value) as build, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as help_exit:
                preparation.main(['--help'])
            self.assertEqual(help_exit.exception.code, 0)
            build.assert_not_called()
            self.assertEqual(preparation.main(['--root', str(self.root), 'validate']), 0)
            build.assert_called_once_with(self.root)
            with self.assertRaises(SystemExit) as run_exit, patch('sys.stderr', new=io.StringIO()):
                preparation.main(['run'])
            self.assertEqual(run_exit.exception.code, 2)

    def test_validate_existing_manifest_rejects_drift(self):
        """旧准备清单不能冒称当前实现身份通过。"""
        manifest = self.root / 'old.json'
        manifest.write_text(json.dumps({**self.value, 'stage': 'wrong'}))
        with patch.object(preparation, 'build_preparation', return_value=self.value):
            with self.assertRaisesRegex(ValueError, 'PREPARATION_MANIFEST_CHANGED'):
                preparation.main(['--root', str(self.root), 'validate', '--manifest', str(manifest)])

    def test_import_has_no_neural_or_runtime_side_effect(self):
        """重新导入仅允许标准库和纯策略合同，无神经/网络/Settings装配。"""
        original = __import__

        def guarded(name, *args, **kwargs):
            """拒绝准备模块尝试读取运行时、模型或网络库。"""
            if name.startswith(('onnxruntime', 'tokenizers', 'research_loop.config',
                                'research_loop.providers', 'scripts.component_eval_adapters', 'httpx', 'requests')):
                raise AssertionError('forbidden import: ' + name)
            return original(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=guarded):
            importlib.reload(preparation)
