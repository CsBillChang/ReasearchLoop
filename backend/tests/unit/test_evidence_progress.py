"""取证增量判断：重复结果、新问题与参数失败不能混作事实审校。"""
import unittest
from research_loop.services.research_progress import evidence_progress


class EvidenceProgressTests(unittest.TestCase):
    """比较正文及定位，不使用模型评分或论文特判。"""

    def setUp(self):
        """构造相同正文、不同chunk身份的检索观察。"""
        self.obs = {'ok': True, 'data': {'hits': [{'document_id': 'doc', 'evidence_id': 'one', 'page_number': 2, 'text': 'known sentence'}]}}
        self.actions = [{'status': 'succeeded', 'tool_name': 'retrieve_evidence', 'arguments': {'query': 'old'}, 'observation': self.obs}]

    def test_changed_query_or_chunk_id_does_not_invent_evidence(self):
        """查询和ID变化但内容相同，准确报告无新增材料。"""
        other = {'ok': True, 'data': {'hits': [{**self.obs['data']['hits'][0], 'evidence_id': 'two'}]}}
        report = evidence_progress(self.actions, 'retrieve_evidence', {'query': 'new'}, other)
        self.assertEqual(report['kind'], 'existing_evidence')
        self.assertEqual(report['new_source_count'], 0)
        self.assertFalse(report['visibility_known'])

    def test_readback_is_not_a_failed_scientific_attempt(self):
        """恢复被压缩的同页文字保留可用性，不触发事实反证。"""
        self.assertEqual(evidence_progress(self.actions, 'read_paper', {'target_ref': 'x'}, self.obs, [])['kind'], 'context_readback')

    def test_new_position_or_text_is_increment_but_not_verdict(self):
        """新文本和位置是材料增量，不等于结论已经正确。"""
        changed = {'ok': True, 'data': {'pages': [{'document_id': 'doc', 'page_number': 3, 'text': 'opposing evidence'}]}}
        report = evidence_progress(self.actions, 'read_paper', {}, changed)
        self.assertEqual(report['kind'], 'new_evidence')
        self.assertFalse(report['semantic_verdict'])

    def test_parameter_error_is_not_contradiction(self):
        """参数错误没有返回证据，不能被描述为论文内容错误。"""
        self.assertEqual(evidence_progress(self.actions, 'read_paper', {}, {'ok': False})['kind'], 'action_failed')

    def test_changed_source_version_is_new_even_with_identical_text(self):
        """正文相同但来源版本变化时不能按旧版本缓存身份处理。"""
        changed = {'ok': True, 'data': {'content_sha256': 'new-version', 'hits': self.obs['data']['hits']}}
        self.assertEqual(evidence_progress(self.actions, 'retrieve_evidence', {}, changed)['kind'], 'new_evidence')
