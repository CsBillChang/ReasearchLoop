"""矩阵真实批次暴露的页范围入口缺口；只扩展明确语法，不猜歧义范围。"""
import unittest
from research_loop.domain.reading import user_page_range


class MatrixPageScopeTests(unittest.TestCase):
    """复现双附件连续页限制及仍需澄清的边界。"""

    def test_explicit_shared_page_scope(self):
        """限定两篇/每篇/各篇的物理范围有同一确定含义。"""
        for text in ['仅使用附件 A 与 B，限定两篇的物理第1至2页。不要搜索。',
                     '仅限每篇物理第1–2页，整理比较矩阵。',
                     '只看各篇的物理第1到2页。','限定3篇论文的第1至2页。']:
            with self.subTest(text=text):
                scope=user_page_range([text]);self.assertEqual((scope.start,scope.end),(1,2))

    def test_ambiguous_and_conflicting_scopes_still_rejected(self):
        """不将不同论文的页范围合并或扩大为全文权限。"""
        for text in ['A仅看第1至2页，B仅看第4至5页。','限定前面部分页。','只读第1页和第3页。']:
            with self.subTest(text=text),self.assertRaises(ValueError):user_page_range([text])

    def test_unrestricted_default_and_later_answer(self):
        """普通任务仍读全文，后续有效回答可缩小范围。"""
        self.assertIsNone(user_page_range(['比较两篇论文，不限制页数。']))
        self.assertIsNone(user_page_range(['不限定第1至2页，请阅读全文。']))
        scope=user_page_range(['只读第1至8页','限定每篇物理第1至2页'])
        self.assertEqual((scope.start,scope.end),(1,2))
