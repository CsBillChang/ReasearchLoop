"""角色交接与混合工具封装的无模型回归，不依赖论文名称。"""
from copy import deepcopy
import unittest

from research_loop.agents.graph import reflection_handoffs
from research_loop.domain.context import factor_source_envelope
from research_loop.storage.memory import dump


def source_batch():
    """构造多个可精确重建的命中，覆盖中文、转义、页码和版本身份。"""
    hits = []
    for n in range(8):
        hits.append({'document_id': 'document-' + 'd' * 64, 'paper_id': 'paper-' + 'p' * 64,
            'source_sha256': 'a' * 64, 'section_origin': 'parsed_section',
            'context_capacity_gap': '摘录仅覆盖给定区间；全文保留在原件。' * 8,
            'context_excerpted': True, 'evidence_id': 'page:' + str(n), 'page_number': n + 1,
            'text': '条件\\甲\n"引用"' + str(n), 'text_sha256': str(n) * 64,
            'char_start': n * 100, 'char_end': n * 100 + 20,
            'context_spans': [{'start': 0, 'end': 5, 'text': '条件\\甲\n'}],
            'context_readback': {'target_ref': 'attachment-A', 'start_page': n + 1, 'page_count': 1}})
    return {'ok': True, 'data': {'hits': hits, 'coverage': {'ready': True}}}


def action(name, data, status='succeeded', ok=True):
    """仅生成持久行动的最小结构；无需模拟模型决策。"""
    return {'tool_name': name, 'status': status, 'observation': {'ok': ok, 'data': data}}


class ReflectionRoleContextTests(unittest.TestCase):
    """来源无损去重、版本分离和核验身份进展的边界。"""

    def test_common_fields_exactly_reconstruct_every_hit(self):
        """逐条还原全部键值，且原文位置、工具读回和原件没有变化。"""
        raw = source_batch()
        original = deepcopy(raw)
        out = factor_source_envelope(raw)
        self.assertLess(len(dump(out).encode()), len(dump(raw).encode()))
        defaults = out['data']['hits_defaults']
        for before, after in zip(raw['data']['hits'], out['data']['hits']):
            self.assertEqual({**defaults, **after}, before)
            for key in ('evidence_id', 'page_number', 'text', 'text_sha256', 'char_start', 'char_end', 'context_spans', 'context_readback'):
                self.assertEqual(after[key], before[key])
        self.assertEqual(raw, original)
        self.assertEqual(out, factor_source_envelope(out))

    def test_distinct_versions_and_absent_fields_are_never_inherited(self):
        """不同版本及缺失字段不被公共封装抹平，类型也不能被等值比较混淆。"""
        raw = source_batch()
        raw['data']['hits'][1].update(source_sha256='b' * 64, document_id='another', context_excerpted=1)
        raw['data']['hits'][2].pop('paper_id')
        out = factor_source_envelope(raw)
        defaults = out['data']['hits_defaults']
        self.assertFalse({'source_sha256', 'document_id', 'paper_id', 'context_excerpted'} & defaults.keys())
        self.assertEqual([{**defaults, **hit} for hit in out['data']['hits']], raw['data']['hits'])

    def test_small_failed_and_non_source_tools_stay_unchanged(self):
        """没有收益或不是成功来源观察时，不变更工具载荷。"""
        samples = [{'ok': True, 'data': {'hits': [{'document_id': 'a'}, {'document_id': 'a'}]}},
                   {'ok': False, 'data': source_batch()['data']},
                   {'ok': True, 'data': {'content': '档案原文', 'offset': 0, 'next_offset': 4}}]
        for sample in samples:
            self.assertEqual(factor_source_envelope(sample), sample)

    def test_only_new_successful_reflection_identity_advances_boundary(self):
        """反证也是有效核验交接；失败、同ID重放、非核验数据不算新进展。"""
        first = action('review_research', {'review_id': 'r1', 'content_hash': 'h1', 'status': 'issues'})
        actions = [first, action('read_paper', {}), deepcopy(first),
                   action('review_research', {'review_id': 'r2'}, status='failed'),
                   action('review_research', {'review_id': 'r3'}, ok=False),
                   action('read_paper', {'review_id': 'not-a-review'})]
        ids, index = reflection_handoffs(actions)
        self.assertEqual(ids, {('review', 'r1', 'h1')})
        self.assertEqual(index, 0)
        actions.append(action('revise_research', {'revision': {'key': 'patch1', 'after_hash': 'h2'}}))
        self.assertEqual(reflection_handoffs(actions)[1], 6)
        actions.append(action('close_research_issue', {'research_disposition': {'key': 'end1'}}))
        self.assertEqual(reflection_handoffs(actions)[1], 7)
        self.assertEqual(len(reflection_handoffs(actions)[0]), 3)
