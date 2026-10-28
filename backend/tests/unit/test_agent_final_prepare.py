"""固定任务编译只处理合成材料，验证身份与隔离边界，不启动业务或模型。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import agent_final_prepare as prepare_module


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class FinalPrepareTests(unittest.TestCase):
    """源材料不变、双轮身份完整、标签不进输入，异常在输出创建前拒绝。"""

    def setUp(self):
        # /tmp在macOS是系统链接；测试使用它的解析路径作为明确许可根。
        self.temp = tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve()))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.design = self.root / 'design'; self.design.mkdir()
        self.sources = self.root / 'original'; self.sources.mkdir()
        self.output_root = self.root / 'experiments'; self.output_root.mkdir()
        self.output = self.output_root / 'new'
        self.corpus = []; self.catalog = []
        for i in range(1, 55):
            pdf = self.sources / f'P{i:02d}.pdf'
            parsed = self.sources / f'P{i:02d}.json'
            pdf.write_bytes(f'%PDF-synthetic-{i}'.encode())
            _write(parsed, {'pages': [{'page_number': 1, 'text': f'synthetic source {i}'}]})
            row = {'material_id': f'P{i:02d}', 'title': f'Synthetic source {i}',
                   'pdf_path': str(pdf), 'parsed_path': str(parsed),
                   'pdf_sha256': _sha(pdf.read_bytes()), 'parsed_cache_sha256': _sha(parsed.read_bytes()),
                   'physical_pages': 1, 'authorized_physical_pages': [1]}
            self.corpus.append(row)
            text = f'synthetic abstract {i}'
            self.catalog.append({'material_id': row['material_id'], 'title': row['title'],
                'pdf_sha256': row['pdf_sha256'], 'excerpt': text, 'excerpt_sha256': _sha(text.encode()),
                'source_url': None, 'authors': None, 'source': None})
        self.cases = []
        for kind, count in prepare_module.EXPECTED_CATEGORIES.items():
            for i in range(1, count + 1):
                row = self.corpus[i - 1]
                materials = [] if kind in {'REC', 'MEM'} else [{k: row[k] for k in
                    ['material_id', 'title', 'pdf_path', 'pdf_sha256', 'physical_pages', 'authorized_physical_pages']}]
                self.cases.append({'case_id': f'AFE-{kind}-{i:02d}', 'category': kind, 'title': 'Synthetic task',
                    'prompt': f'Synthetic {kind} request {i}', 'materials': materials,
                    'preconditions': {'save_policy': 'allow' if kind == 'DEEP' else 'deny',
                        'history_messages': [{'role': 'user', 'content': 'Synthetic history'}] if kind == 'MEM' else [],
                        'catalog_ref': 'materials/catalog-54.json' if kind == 'REC' else None}})
        self.design_value = {'schema_version': 'agent_final_eval_design_v1', 'execute_authorized': False,
                             'cases': self.cases}
        self.order = []
        for round_no in [1, 2]:
            for c in self.cases:
                for variant in ['A', 'B']:
                    self.order.append({'ordinal': len(self.order) + 1, 'case_id': c['case_id'],
                        'round': round_no, 'variant': variant,
                        'cell_id': f"{c['case_id']}-R{round_no}-{variant}",
                        'status': 'not_created', 'model_calls': None, 'actual_tokens': None,
                        'end_to_end_seconds': None})
        self._save_design()

    def _save_design(self):
        _write(self.design / 'cases/inputs.json', self.design_value)
        _write(self.design / 'materials/corpus-54.json', self.corpus)
        _write(self.design / 'materials/catalog-54.json', self.catalog)
        _write(self.design / 'ORDER.json', {'order': self.order})
        self._seal()

    def _seal(self):
        files = {str(p.relative_to(self.design)): {'sha256': _sha(p.read_bytes()), 'bytes': p.stat().st_size}
                 for p in sorted(self.design.rglob('*')) if p.is_file() and p.name != 'SEAL.json'}
        manifest = _sha(json.dumps(files, sort_keys=True, separators=(',', ':')).encode())
        _write(self.design / 'SEAL.json', {'file_count': len(files), 'files': files,
                                         'manifest_sha256': manifest})

    def _prepare(self, **kwargs):
        return prepare_module.prepare(self.design, self.output, allowed_output_root=self.output_root, **kwargs)

    def _input_hashes(self):
        return {str(p): _sha(p.read_bytes()) for root in [self.design, self.sources]
                for p in root.rglob('*') if p.is_file()}

    def test_fixed_matrix_uses_real_independent_copies_and_keeps_sources(self):
        before = self._input_hashes()
        receipt = self._prepare(protected_roots=[self.sources])
        self.assertEqual((receipt['pdf_count'], receipt['parsed_count'], receipt['root_count']), (54, 54, 96))
        self.assertEqual(before, self._input_hashes())
        snapshot = json.loads((self.output / 'material-snapshot.json').read_text())
        self.assertEqual(len(snapshot['copies']), 108)
        for row in snapshot['copies']:
            original, copied = Path(row['source']), Path(row['copy'])
            self.assertEqual(original.read_bytes(), copied.read_bytes())
            self.assertFalse(copied.is_symlink())
            self.assertEqual(copied.stat().st_nlink, 1)
            self.assertNotEqual((original.stat().st_dev, original.stat().st_ino),
                                (copied.stat().st_dev, copied.stat().st_ino))
        self.assertFalse(receipt['real_execution_released'])
        self.assertEqual(receipt['real_model_calls'], 0)

    def test_two_round_ids_preserve_order_without_reusing_cells(self):
        self._prepare()
        runtime = json.loads((self.output / 'cases.json').read_text())
        plan = json.loads((self.output / 'execution-plan.json').read_text())
        self.assertEqual(len({c['case_id'] for c in runtime['cases']}), 48)
        self.assertEqual(len({tuple(c) for c in plan['execution_order']}), 96)
        self.assertEqual(plan['execution_order'][0], ['AFE-QA-01-R1', 'A'])
        self.assertEqual(plan['execution_order'][48], ['AFE-QA-01-R2', 'A'])
        self.assertEqual(len(plan['lineage']), 48)
        self.assertTrue(all(c['suite_kind'] == 'agent_final_v1' for c in runtime['cases']))

    def test_catalog_pool_is_full_and_not_required_attachments_in_other_tasks(self):
        self._prepare()
        cases = json.loads((self.output / 'cases.json').read_text())['cases']
        for case in cases:
            pre = case['preconditions']
            if case['case_id'].startswith('AFE-REC-'):
                self.assertEqual(len(case['materials']), 54)
                self.assertEqual([r['material_index'] for r in pre['catalog']], list(range(54)))
                self.assertTrue(all(r['source_url'] == '' and r['authors'] == [] for r in pre['catalog']))
            elif case['case_id'].startswith('AFE-MEM-'):
                self.assertEqual(case['materials'], [])
                self.assertEqual(pre['catalog'], [])
                self.assertEqual(pre['history_messages'][0]['role'], 'user')
            else:
                self.assertEqual(len(case['materials']), 1)
                self.assertEqual(pre['catalog'], [])

    def test_gold_fields_are_not_copied_into_runtime_inputs(self):
        self.design_value['cases'][0]['rubric'] = {'core': 'SECRET_GOLD'}
        self.catalog[0]['score'] = 'SECRET_GOLD'
        self._save_design()
        self._prepare()
        raw = (self.output / 'cases.json').read_text()
        self.assertNotIn('SECRET_GOLD', raw)
        self.assertNotIn('rubric', raw)
        self.assertNotIn('score', raw)

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / 'keep'; sentinel.write_text('immutable')
        with self.assertRaises(FileExistsError): self._prepare()
        self.assertEqual(sentinel.read_text(), 'immutable')

    def test_output_cannot_target_source_or_unapproved_root(self):
        original = self.output
        for destination in [self.design / 'new', self.sources / 'new', self.root / 'outside']:
            with self.subTest(destination=destination):
                self.output = destination
                with self.assertRaises(PermissionError): self._prepare(protected_roots=[self.sources])
                self.assertFalse(destination.exists())
        self.output = original

    def test_symlink_output_parent_is_rejected(self):
        link = self.output_root / 'alias'; link.symlink_to(self.output_root, target_is_directory=True)
        self.output = link / 'new'
        with self.assertRaises(PermissionError): self._prepare()
        self.assertFalse((self.output_root / 'new').exists())

    def test_symlink_source_is_rejected_before_any_copy(self):
        pdf = Path(self.corpus[0]['pdf_path'])
        original = pdf.with_suffix('.original'); pdf.rename(original); pdf.symlink_to(original)
        with self.assertRaises(PermissionError): self._prepare()
        self.assertFalse(self.output.exists())

    def test_original_material_drift_is_rejected_before_output_creation(self):
        Path(self.corpus[0]['pdf_path']).write_bytes(b'changed')
        with self.assertRaises(PermissionError): self._prepare()
        self.assertFalse(self.output.exists())

    def test_sealed_file_drift_or_unsealed_extra_file_is_rejected(self):
        path = self.design / 'cases/inputs.json'
        original = path.read_bytes(); path.write_bytes(original + b' ')
        with self.assertRaises(PermissionError): self._prepare()
        path.write_bytes(original)
        (self.design / 'extra.json').write_text('{}')
        with self.assertRaises(PermissionError): self._prepare()
        self.assertFalse(self.output.exists())

    def test_duplicate_material_or_matrix_slot_is_rejected(self):
        old_id = self.corpus[1]['material_id']; self.corpus[1]['material_id'] = self.corpus[0]['material_id']
        self._save_design()
        with self.assertRaises(PermissionError): self._prepare()
        self.corpus[1]['material_id'] = old_id
        self.order[1] = deepcopy(self.order[0]); self.order[1]['ordinal'] = 2
        self._save_design()
        with self.assertRaises(PermissionError): self._prepare()
        self.assertFalse(self.output.exists())

    def test_case_cannot_replace_frozen_material_or_add_memory_pdf(self):
        old = deepcopy(self.cases[0]['materials'])
        self.cases[0]['materials'][0]['pdf_sha256'] = '0' * 64
        self._save_design()
        with self.assertRaises(PermissionError): self._prepare()
        self.cases[0]['materials'] = old
        memory = next(c for c in self.cases if c['case_id'].startswith('AFE-MEM-'))
        memory['materials'] = deepcopy(old)
        self._save_design()
        with self.assertRaises(PermissionError): self._prepare()
        self.assertFalse(self.output.exists())

    def test_forged_history_and_expanded_save_permissions_are_rejected(self):
        memory = next(c for c in self.cases if c['case_id'].startswith('AFE-MEM-'))
        memory['preconditions']['history_messages'][0]['role'] = 'tool'
        self._save_design()
        with self.assertRaises(PermissionError): self._prepare()
        memory['preconditions']['history_messages'][0]['role'] = 'user'
        self.cases[0]['preconditions']['save_policy'] = 'allow'
        self._save_design()
        with self.assertRaises(PermissionError): self._prepare()
        self.assertFalse(self.output.exists())

    def test_case_path_injection_and_boolean_ordinal_are_rejected(self):
        original_id = self.cases[0]['case_id']
        self.cases[0]['case_id'] = original_id + '/../../outside'
        self._save_design()
        with self.assertRaises(PermissionError): self._prepare()
        self.cases[0]['case_id'] = original_id
        self.order[0]['ordinal'] = True
        self._save_design()
        with self.assertRaises(PermissionError): self._prepare()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
