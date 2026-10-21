"""来源位置和研究增量的纯函数合同；不使用网络、生成模型或私人材料。"""
import copy
import hashlib
import unittest

from research_loop.domain.evidence import source_reference, valid_source_reference, validate_source_window
from research_loop.services.research_progress import continuation_fingerprint, readback_handoff


class EvidenceAccessContractTests(unittest.TestCase):
    """严格版本、字符位置和实际分析修订分开检验。"""

    def setUp(self):
        """固定合成页，所有反例只在独立副本上修改。"""
        self.page = {'evidence_id': 'page:A', 'document_id': 'doc:A', 'page_number': 2,
                     'text': 'A reported metric does not establish universal reliability.'}
        self.document = {'document_id': 'doc:A', 'paper_id': 'paper:A', 'content_sha256': 'a' * 64}
        self.reference = source_reference(self.page, self.document)

    def test_reference_requires_complete_shape_and_valid_hashes(self):
        """不补齐模型遗漏字段，不接受额外入口或非法物理页。"""
        self.assertTrue(valid_source_reference(self.reference))
        for key in self.reference:
            with self.subTest(missing=key):
                self.assertFalse(valid_source_reference({k: v for k, v in self.reference.items() if k != key}))
        for update in ({'page_number': True}, {'page_number': 0}, {'source_sha256': 'A' * 64},
                       {'target_ref': 'ungranted-paper'}):
            self.assertFalse(valid_source_reference({**self.reference, **update}))

    def test_page_and_hit_share_precise_version_window_contract(self):
        """整页与片段都保留真实版本，命中不能借Python负下标或截断越界。"""
        validate_source_window(self.page, {'content_sha256': 'a' * 64}, self.page, self.document)
        hit = {**self.page, 'char_start': 2, 'char_end': 10, 'text': self.page['text'][2:10],
               'source_sha256': 'a' * 64}
        hit['text_sha256'] = hashlib.sha256(hit['text'].encode()).hexdigest()
        validate_source_window(hit, {}, self.page, self.document, kind='hit')
        for update in ({'char_start': -1}, {'char_start': True}, {'char_end': 999},
                       {'page_number': 2.0}, {'source_sha256': 'b' * 64}, {'text_sha256': 'f' * 64}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_source_window({**hit, **update}, {}, self.page, self.document, kind='hit')
        for key in ('document_id', 'evidence_id', 'page_number', 'text', 'source_sha256'):
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_source_window({k: v for k, v in hit.items() if k != key}, {}, self.page, self.document, kind='hit')

    def test_plan_edits_do_not_release_handoff_but_analysis_revision_does(self):
        """旧证据可支持新分析；仅缺口、下一问题或自增revision不冒充研究增量。"""
        finding = {'finding_id': 'f:A', 'text': 'A bounded interpretation.', 'sources': [self.reference]}
        progress = {'findings': [finding], 'gaps': ['not yet checked'], 'next_question': 'initial', 'revision': 1}
        changed_plan = {**progress, 'gaps': ['reworded'], 'next_question': 'reworded', 'revision': 9}
        self.assertEqual(continuation_fingerprint(progress), continuation_fingerprint(changed_plan))
        changed_finding = copy.deepcopy(progress)
        changed_finding['findings'][0]['text'] = 'The same evidence supports a narrower interpretation.'
        self.assertNotEqual(continuation_fingerprint(progress), continuation_fingerprint(changed_finding))
        def stage(value):
            """模拟持久成功进展而不引入数据库写入。"""
            return {'tool_name': 'record_research_progress', 'status': 'succeeded',
                    'observation': {'data': {'research_progress': {**value, 'base_revision': 0}}}}
        def read(number):
            """模拟重复原页读回，不制造新证据。"""
            return {'tool_name': 'read_paper', 'status': 'succeeded', 'action_id': str(number),
                    'arguments': {'target_ref': 'paper:A'},
                    'observation': {'data': {'research_increment': {'kind': 'existing_evidence'}}}}
        self.assertIsNotNone(readback_handoff([stage(progress), read(1), stage(changed_plan), read(2)]))
        self.assertIsNone(readback_handoff([stage(progress), read(1), stage(changed_finding), read(2)]))
