"""Reflection准备器的身份、标签隔离与非授权边界测试。"""
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_reflection_baseline import freeze_package, validate_package


class ReflectionPreparationTests(unittest.TestCase):
    """所有输入是合成材料，验证清单不能变成真实发送授权。"""

    def setUp(self):
        """构造一个待人工审核、原文可验证的最小包。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write('input.json', {'case_id': 'sample', 'model_input': {
            'question': '核验当前稿。', 'draft': 'Expected results are reported.',
            'documents': [{'allowed_pages': [1], 'initial_pages': [{
                'physical_page': 1, 'text': 'Source text.',
                'page_text_sha256': hashlib.sha256(b'Source text.').hexdigest()}]}]}})
        self.write('label.json', {'case_id': 'sample', 'approval_status': 'pending_user_material_review'})
        self.manifest = self.root / 'manifest.json'
        self.refresh()

    def write(self, name, value):
        """以UTF-8保存合成输入。"""
        (self.root / name).write_text(json.dumps(value, ensure_ascii=False))

    def refresh(self):
        """冻结当前文件身份，模拟实际材料准备阶段。"""
        self.write('manifest.json', {'schema_version': 1,
            'files': [{'path': name, 'sha256': hashlib.sha256((self.root / name).read_bytes()).hexdigest()}
                      for name in ['input.json', 'label.json']],
            'cases': [{'case_id': 'sample', 'batch': 'pilot', 'input': 'input.json', 'label': 'label.json'}]})

    def test_preparation_never_authorizes_network_or_real_execution(self):
        """即使包结构通过，人评和真实装配仍必须单独满足。"""
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('network prohibited')):
            result = validate_package(self.manifest)
        self.assertTrue(result['structural_passed'])
        self.assertFalse(result['real_execution_allowed'])
        self.assertEqual(result['semantic_labels_pending'], 1)

    def test_label_fields_in_model_payload_are_rejected(self):
        """只拒绝结构性泄漏，不误判正文中的普通英文expected。"""
        value = json.loads((self.root / 'input.json').read_text())
        value['model_input']['documents'][0]['expected_text'] = 'gold leakage'
        self.write('input.json', value)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'LABEL_IN_MODEL_INPUT'):
            validate_package(self.manifest)

    def test_file_drift_is_rejected_before_parsing(self):
        """改完稿后不能沿用旧来源和标签冻结。"""
        self.write('input.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'FILE_DRIFT'):
            validate_package(self.manifest)

    def test_scope_and_source_hash_are_checked(self):
        """即使重新冻结文件，也不能声明不一致的页范围。"""
        value = json.loads((self.root / 'input.json').read_text())
        value['model_input']['documents'][0]['allowed_pages'] = [2]
        self.write('input.json', value)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'PAGE_OUTSIDE_SCOPE'):
            validate_package(self.manifest)

    def test_package_path_escape_and_overwrite_are_rejected(self):
        """包外引用不能伪装成已冻结输入，旧快照不被覆盖。"""
        output = self.root / 'frozen.json'
        freeze_package(self.manifest, output)
        with self.assertRaises(FileExistsError):
            freeze_package(self.manifest, output)
        value = json.loads(self.manifest.read_text())
        value['files'][0]['path'] = '../outside.json'
        self.write('manifest.json', value)
        with self.assertRaisesRegex(ValueError, 'PATH_ESCAPE'):
            validate_package(self.manifest)

    def test_natural_page_hash_and_provenance_are_not_skipped(self):
        """自然回放字段别名不能绕过原文身份和历史稿件校验。"""
        value = {'case_id': 'sample', 'model_input': {
            'original_question': '原问题', 'draft_markdown': '原稿',
            'historical_materials': [{'pages': [{'page_number': 1,
                'text': 'Source text.', 'text_sha256': '0' * 64}]}]},
            'provenance': {'draft_sha256': hashlib.sha256('原稿'.encode()).hexdigest()}}
        self.write('input.json', value)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'PAGE_TEXT_DRIFT'):
            validate_package(self.manifest)
        value['model_input']['historical_materials'][0]['pages'][0]['text_sha256'] = hashlib.sha256(b'Source text.').hexdigest()
        value['provenance']['draft_sha256'] = '0' * 64
        self.write('input.json', value)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'PROVENANCE_MISMATCH'):
            validate_package(self.manifest)

    def test_conflicting_aliases_are_rejected(self):
        """原始和规范字段并存时必须逐字一致。"""
        value = json.loads((self.root / 'input.json').read_text())
        value['model_input']['draft_markdown'] = '另一份稿'
        self.write('input.json', value)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'CONFLICTING_INPUT_ALIASES'):
            validate_package(self.manifest)


if __name__ == '__main__':
    unittest.main()
