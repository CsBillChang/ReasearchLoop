"""分材料页范围纯合同；仅真实用户原话参与解析，不调用模型。"""
import unittest

from research_loop.domain.reading import material_page_scopes, user_page_range


class MaterialPageScopeTests(unittest.TestCase):
    """保护局部更新、待澄清、全局兼容及名称歧义。"""

    def scopes(self, *instructions, materials=None):
        """为每例提供确定的服务端材料身份。"""
        return material_page_scopes(list(instructions), materials or {'A': [], 'B': []})

    def pair(self, scope):
        """断言范围前先通过待澄清保护。"""
        value = scope.require_range()
        return (value.start, value.end) if value else None

    def test_independent_ranges(self):
        values = self.scopes('仅使用附件A的物理第2至3页；仅使用附件B的物理第1至4页。')
        self.assertEqual(self.pair(values['A']), (2, 3))
        self.assertEqual(self.pair(values['B']), (1, 4))

    def test_joint_named_range_and_casefold_name_collision(self):
        values = self.scopes('仅使用附件 Alpha 与 Beta 的物理第1至2页。', materials={'first': ['Alpha'], 'second': ['Beta']})
        self.assertEqual([self.pair(v) for v in values.values()], [(1, 2), (1, 2)])
        values = self.scopes('只读附件 Alpha 的物理第1页。', materials={'first': ['Alpha'], 'second': ['alpha']})
        self.assertTrue(all(v.status == 'pending' for v in values.values()))

    def test_pending_does_not_block_other_material(self):
        values = self.scopes('仅使用附件A的物理第1至2页；附件B只允许前面几页。')
        self.assertEqual(self.pair(values['A']), (1, 2))
        with self.assertRaises(ValueError):
            values['B'].require_range()

    def test_real_r2_verbatim_contract(self):
        initial = ('比较本对话附件A（D09 LoRASculpt）和附件B（D10 LaVer）的核心方法，只使用用户许可的物理页。'
            '仅使用附件A的物理第1至2页；附件B只允许前面几页。B的具体页范围尚未确定，请先向我澄清B的物理页范围，'
            '在收到回答前不要读取B或猜测其范围。保留A的明确硬范围。不要重新检索，不做八点精读，不保存文件。')
        answer = ('B的范围补充为：仅使用附件B的物理第1至3页。同时调整A原来的硬范围：仅使用附件A的物理第2至3页。'
            '两份材料按各自范围比较；不要把两篇统一成同一页范围，不读取A第1页或B第4页及以后。不重新检索，不做八点精读，不保存文件。')
        values = self.scopes(initial, answer)
        self.assertEqual(self.pair(values['A']), (2, 3))
        self.assertEqual(self.pair(values['B']), (1, 3))

    def test_local_update_keeps_other_pending(self):
        values = self.scopes('仅读附件A的物理第1至2页；附件B只读前面几页。', '仅读附件A的物理第2至3页。')
        self.assertEqual(self.pair(values['A']), (2, 3))
        self.assertEqual(values['B'].status, 'pending')

    def test_global_update_changes_all(self):
        values = self.scopes('附件A只读前两页；附件B只读前面几页。', '所有材料只读物理第3至4页。')
        self.assertEqual([self.pair(v) for v in values.values()], [(3, 4), (3, 4)])

    def test_local_release_keeps_other_range(self):
        values = self.scopes('附件A只读前两页；附件B只读前3页。', '解除附件A的页数限制。')
        self.assertIsNone(self.pair(values['A']))
        self.assertEqual(self.pair(values['B']), (1, 3))

    def test_reference_pages_are_not_read_permission(self):
        values = self.scopes('只依据这两份附件并注明原文页码，不保存。')
        self.assertTrue(all(v.status == 'full' for v in values.values()))

    def test_same_name_requires_clarification(self):
        values = self.scopes('仅使用附件报告的物理第1至2页。', materials={'A': ['报告'], 'B': ['报告']})
        self.assertTrue(all(v.status == 'pending' for v in values.values()))

    def test_unknown_name_does_not_authorize_any_attachment(self):
        values = self.scopes('仅使用附件其他材料的物理第1至2页。')
        self.assertTrue(all(v.status == 'pending' for v in values.values()))

    def test_unclear_cross_comma_is_not_fulltext(self):
        for text in ('只读，前面几页。', '附件B只读，前面几页。', '附件B只看，标注页码对应的内容。'):
            with self.subTest(text=text):
                values = self.scopes(text)
                self.assertEqual(values['B'].status, 'pending')
        self.assertEqual(self.scopes('附件B只读，前面几页。')['A'].status, 'full')

    def test_invalid_discrete_and_conflicting_ranges_remain_denied(self):
        for text in ('附件A只读第0页。', '附件A只读第4至2页。', '附件A只读第1页和第3页。',
                     '附件A只读第1页；附件A只读第3页。'):
            with self.subTest(text=text):
                self.assertEqual(self.scopes(text)['A'].status, 'pending')

    def test_legacy_global_helper_keeps_its_explicit_boundary(self):
        self.assertEqual(user_page_range(['只读前两页']).end, 2)
        self.assertIsNone(user_page_range(['不限制页数。']))
        with self.assertRaises(ValueError):
            user_page_range(['附件A只读第1页；附件B只读第3页。'])
