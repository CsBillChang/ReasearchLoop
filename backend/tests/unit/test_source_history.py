"""累计来源投影的去重和省略边界，不使用私有论文或模型。"""
from copy import deepcopy
import json
import unittest

from research_loop.domain.context import source_history


def row(index, doc='doc-a', version='v1', text='language mechanism', page=1):
    """构造单页观察；重复ID必须连同版本、字符区间和原句一起比较。"""
    item = {'document_id': doc, 'evidence_id': f'{doc}:{page}', 'page_number': page, 'text': text}
    return (index, {'tool_call_id': str(index)},
            {'ok': True, 'data': {'content_sha256': version, 'pages': [item]}},
            {'name': 'read_paper', 'arguments': json.dumps({'target_ref': doc})})


class SourceHistoryTests(unittest.TestCase):
    """不同版本/正文不能被重复项吞掉，容量省略不能伪装为没有证据。"""

    def test_exact_duplicate_keeps_latest_and_original_unchanged(self):
        """100次重复观察只呈现一次原文，完整输入仍可用于恢复。"""
        rows = [row(i) for i in range(100)]
        original = deepcopy(rows)
        result = source_history(rows, 'language')
        self.assertEqual(sum(len(r[2]['data']['pages']) for r in result), 1)
        self.assertEqual(result[-1][2]['data']['pages'][0]['text'], 'language mechanism')
        self.assertEqual(sum(r[2]['data']['context_duplicate_count'] for r in result), 99)
        self.assertEqual(rows, original)

    def test_versions_ranges_and_text_remain_distinct(self):
        """身份或文字不同的项保留到外层真实性校验，不以去重静默掩盖差异。"""
        rows = [row(0), row(1, version='v2'), row(2, text='changed'), row(3, doc='doc-b')]
        result = source_history(rows, 'language')
        self.assertEqual(sum(len(r[2]['data']['pages']) for r in result), 4)

    def test_capacity_keeps_each_document_and_marks_omission(self):
        """按材料轮转保留窗口；没有把长论文或后读论文当作唯一依据。"""
        rows = [row(i, doc=doc, page=i + 1) for doc in ('doc-a', 'doc-b') for i in range(20)]
        result = source_history(rows, 'language', max_items=2)
        kept = [p for r in result for p in r[2]['data']['pages']]
        self.assertEqual({p['document_id'] for p in kept}, {'doc-a', 'doc-b'})
        self.assertEqual(sum(r[2]['data']['context_omitted_count'] for r in result), 38)

    def test_latest_source_not_displaced_by_repeated_old_read(self):
        """容量受限时每份材料仍保留新观察，不被旧定向补读占满。"""
        old, new = row(0, page=1), row(1, page=2)
        old[3]['arguments'] = json.dumps({'target_ref': 'doc-a', 'page_count': 1})
        result = source_history([old, new], 'language', max_items=1)
        kept = [p for r in result for p in r[2]['data']['pages']]
        self.assertEqual([p['page_number'] for p in kept], [2])
