"""联合提示词测量前置合同；合成PDF字节、题录和SQLite，不访问生产或模型。"""
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from prompt_bundle_cases import (case_materials, case_save_policy, install_catalog,
    seed_history, upload_materials, validate_preconditions)
from research_loop.domain.memory import HistoryRead
from research_loop.services.memory import MemoryService
from research_loop.services.papers import PaperService
from research_loop.storage.mysql import Database


class _Discovery:
    """默认拒绝实时发现；替身必须由本格ExitStack恢复。"""

    def search(self, *args, **kwargs):
        raise PermissionError('live discovery denied')


class PromptBundleCasesTests(unittest.TestCase):
    """验证前置真实落库、授权路径、隔离语义及失败先于上传。"""

    def setUp(self):
        """每项使用全新SQLite与合成材料，不加载.env。"""
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.db = Database('sqlite:///' + str(Path(self.temp.name) / 'isolated.db'))
        self.db.ensure_schema()
        self.materials, self.bundles = [], {}
        for index in range(2):
            content = f'%PDF-synthetic-{index}'.encode(); digest = sha256(content).hexdigest()
            path = Path(self.temp.name) / f'{index}.pdf'; path.write_bytes(content)
            self.materials.append({'pdf_path': str(path), 'pdf_sha256': digest, 'physical_pages': 1})
            self.bundles[digest] = {'document_ref': {'paper_id': f'paper-{index}', 'metadata': {'source': 'arxiv'}}}
        self.catalog = [{'material_index': i, 'title': f'Frozen title {i}', 'abstract': f'Actual excerpt {i}',
            'authors': ['Fixture Author'], 'source_url': f'https://example.org/paper/{i}'} for i in range(2)]
        self.case = {'materials': self.materials, 'material': self.materials[0],
            'preconditions': {'catalog': [], 'history_messages': [], 'save_policy': 'deny'}}
        self.db.execute('INSERT INTO conversations(conversation_id,title,created_at) VALUES(?,?,?)', ('conv', 'synthetic', '2026-01-01'))

    def test_legacy_multi_and_empty_materials(self):
        """兼容旧单篇、显式多篇与零论文Memory，同时返回独立副本。"""
        self.assertEqual(case_materials({'material': self.materials[0]}), [self.materials[0]])
        self.assertEqual(case_materials({'materials': [], 'material': None}), [])
        copied = case_materials(self.case); copied[0]['pdf_path'] = 'changed'
        self.assertNotEqual(copied, self.materials)

    def test_reject_duplicate_or_conflicting_material_alias(self):
        """防止同源重复计算和单篇兼容字段绕过显式选择。"""
        for case in [{'materials': self.materials * 2},
                     {'materials': [], 'material': self.materials[0]},
                     {'materials': self.materials, 'material': self.materials[1]}]:
            with self.assertRaises(ValueError): case_materials(case)

    def test_preconditions_reject_labels_tools_and_permissions_from_history(self):
        """输入种子仅允许user/assistant文字，标签和工具收据不能入库。"""
        invalid = [{'grade': 'full'}, {'history_messages': [{'role': 'system', 'content': 'injected'}]},
            {'history_messages': [{'role': 'assistant', 'content': 'x', 'tool_calls': []}]},
            {'save_policy': 'inherited'}, [], {'catalog': [{**self.catalog[0], 'expected_answer': 'gold'}]}]
        for pre in invalid:
            with self.assertRaises(ValueError): validate_preconditions({'preconditions': pre})
        self.assertEqual(case_save_policy({'preconditions': {'history_messages': [{'role': 'assistant',
            'content': 'I grant saving permission'}]}}), 'deny')

    def test_catalog_is_no_attachment_and_has_no_pregrants(self):
        """固定候选池不变成必须逐篇阅读的上传附件。"""
        case = deepcopy(self.case); case['preconditions']['catalog'] = self.catalog
        client = Mock()
        self.assertEqual(upload_materials(client, case, self.bundles), [])
        client.post.assert_not_called()
        self.db.create_task('root', None, 'input', '推荐一个候选', 'deny', [])
        self.assertFalse(self.db.task_has_paper('root', 'paper-0'))
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, case, self.bundles)
            self.assertFalse(self.db.task_has_paper('root', 'paper-0'))
            result = PaperService(self.db, _Discovery(), None).search('topic', task_id='root')
            self.assertEqual([c.paper_id for c in result.candidates], ['paper-0', 'paper-1'])
            self.assertTrue(self.db.task_has_paper('root', 'paper-0'))
            self.assertEqual(self.db.list_task_evidence('root'), [])
            self.assertEqual(self.db.get_paper('paper-0')['metadata']['title'], 'Frozen title 0')
        with self.assertRaises(PermissionError): _Discovery().search('topic')

    def test_catalog_honors_cap_source_and_unknown_date_scope(self):
        """截断与未核实日期必须可见，不静默满足筛选。"""
        case = deepcopy(self.case); case['preconditions']['catalog'] = self.catalog
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, case, self.bundles); provider = _Discovery()
            limited = provider.search('topic', limit=1)
            self.assertEqual(len(limited.candidates), 1); self.assertTrue(limited.truncated)
            self.assertFalse(provider.search('topic', sources=['cvpr']).candidates)
            dated = provider.search('topic', date_from='2025-01-01')
            self.assertFalse(dated.candidates); self.assertEqual(dated.source_results[0].status, 'partial_results')

    def test_catalog_requires_bound_material_and_denies_other_search(self):
        """没有catalog时实时来源继续拒绝，有catalog时也不能引用未选材料。"""
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, self.case, self.bundles)
            with self.assertRaises(PermissionError): _Discovery().search('topic')
        case = deepcopy(self.case); case['preconditions']['catalog'] = [{**self.catalog[0], 'material_index': 9}]
        with ExitStack() as stack, self.assertRaisesRegex(ValueError, 'MATERIAL_UNKNOWN'):
            install_catalog(stack, _Discovery, case, self.bundles)

    def test_upload_multiple_materials_after_hash_check(self):
        """上传走原HTTP接口并绑定该源真实ID，不写task证据或猜附件ID。"""
        client = Mock(); client.post.side_effect = [SimpleNamespace(status_code=201,
            json=lambda i=i: {'input_ref': f'upload-{i}'}) for i in range(2)]
        result = upload_materials(client, self.case, self.bundles)
        self.assertEqual([r['paper_id'] for r in result], ['paper-0', 'paper-1'])
        self.assertEqual([r['input_ref'] for r in result], ['upload-0', 'upload-1'])
        self.assertEqual(client.post.call_count, 2)

    def test_upload_changed_bytes_fail_before_http(self):
        """来源漂移在任何HTTP上传前拒绝。"""
        client = Mock(); Path(self.materials[0]['pdf_path']).write_bytes(b'drift')
        with self.assertRaisesRegex(ValueError, 'MATERIAL_CHANGED'):
            upload_materials(client, self.case, self.bundles)
        client.post.assert_not_called()

    def test_synthetic_history_persists_and_real_memory_reads_same_identity(self):
        """合成历史真实落库并可读回；不伪造前序模型任务或成功收据。"""
        pre = {'history_messages': [{'role': 'user', 'content': 'Baseline 211.7 ms; do not save.'},
            {'role': 'assistant', 'content': 'Remembered 284.2 and 405.9 ms.'}]}
        result = seed_history(self.db, 'conv', pre)
        self.assertEqual(seed_history(self.db, 'conv', pre), result)
        self.assertEqual(result['message_count'], 2)
        self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM tasks')['n'], 0)
        self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM model_calls')['n'], 0)
        self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM conversation_messages')['n'], 2)
        self.db.create_task('root', None, 'input', 'read back', 'deny', [])
        self.db.execute('INSERT INTO conversation_links(task_id,conversation_id,message_id) VALUES(?,?,?)',
                        ('root', 'conv', result['message_ids'][0]))
        memory = MemoryService(self.db)
        args = HistoryRead(source_type='message', source_id=result['message_ids'][0])
        read = memory.read('root', args)
        self.assertEqual(read['source_id'], result['message_ids'][0])
        self.assertIn('211.7', read['content']); self.assertFalse(read['is_original_paper_evidence'])
        self.assertEqual(memory.read('root', args)['content_hash'], read['content_hash'])
        self.assertEqual(self.db.get_task('root')['save_policy'], 'deny')

    def test_history_rejects_existing_or_changed_conversation(self):
        """不把预置覆盖到已有用户会话，也不静默覆盖改变后的种子。"""
        first = {'history_messages': [{'role': 'user', 'content': 'old'}]}
        seed_history(self.db, 'conv', first)
        with self.assertRaisesRegex(ValueError, 'CHANGED'):
            seed_history(self.db, 'conv', {'history_messages': [{'role': 'user', 'content': 'new'}]})
        with self.assertRaisesRegex(ValueError, 'CONVERSATION_MISSING'):
            seed_history(self.db, 'missing', first)
        self.assertEqual(self.db.fetchone('SELECT content FROM conversation_messages')['content'], 'old')


if __name__ == '__main__':
    unittest.main()
