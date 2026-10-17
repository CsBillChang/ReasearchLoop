"""出处页码输出要求与实际阅读权限分离，不依赖论文名称或生成模型。"""
import unittest
from research_loop.domain.reading import user_page_range


class PageReferenceScopeTests(unittest.TestCase):
    """覆盖出处要求、真正限制及二者混合时的保守权限边界。"""

    def test_reference_requests_do_not_restrict_fulltext(self):
        """独立短句或并列的页码标注是输出要求，不是阅读范围。"""
        for text in [
            '只依据这两份附件，注明原文页码。',
            '只依据附件并注明原文页码。',
            '仅使用论文；请标注对应的原文页码。',
            '只根据论文内容，给出出处页码。',
            '只标注引用页码。',
            '仅需注明原文页码。',
            '只依据附件，同时给出原文的具体页码。',
            '只依据附件，请注明物理页码而非印刷页码。',
        ]:
            with self.subTest(text=text):
                self.assertIsNone(user_page_range([text]))

    def test_actual_range_is_not_removed_by_reference_request(self):
        """连续范围仍从原话解析，标注要求不能扩大权限。"""
        for text, expected in [
            ('只读前两页，并注明原文页码。', (1, 2)),
            ('注明原文页码；仅使用附件 Alpha 与 Beta 的物理第3至4页。', (3, 4)),
            ('只依据附件前两页生成比较，注明出处页码。', (1, 2)),
            ('only read physical pages 9-25, cite page numbers.', (9, 25)),
        ]:
            with self.subTest(text=text):
                scope=user_page_range([text])
                self.assertEqual((scope.start,scope.end),expected)
                self.assertIn(scope.instruction,text)

    def test_unclear_reading_restrictions_remain_rejected(self):
        """不因出处提示或逗号就删除真正但不明确的限制。"""
        for text in [
            '只看某几页，并注明原文页码。',
            '仅使用附件 Alpha 与 Beta 的物理前面几页；注明原文页码。',
            '限定前面部分页，注明原文页码。',
            '只读，前面几页，并注明原文页码。',
            '只看，标注页码对应的内容。',
            '只读指定页码的内容，并注明原文页码。',
        ]:
            with self.subTest(text=text),self.assertRaises(ValueError):
                user_page_range([text])

    def test_conflicting_noncontiguous_and_invalid_ranges_remain_rejected(self):
        """冲突、离散、倒序与零页仍拒绝，不以提示位置掩盖冲突。"""
        for text in [
            '只读第1页和第3页，并注明原文页码。',
            'A仅看第1至2页，B仅看第4至5页，并注明原文页码。',
            '只读第3至1页，注明出处页码。',
            '只读第0至2页，注明原文页码。',
        ]:
            with self.subTest(text=text),self.assertRaises(ValueError):
                user_page_range([text])

    def test_reference_only_clarification_does_not_erase_prior_range(self):
        """标注要求不是解除限制；显式新范围与解除指令仍按原优先级处理。"""
        scope=user_page_range(['只读前两页。','请注明原文页码。'])
        self.assertEqual((scope.start,scope.end),(1,2))
        scope=user_page_range(['只读前两页。','仅限物理第3至4页，并注明原文页码。'])
        self.assertEqual((scope.start,scope.end),(3,4))
        self.assertIsNone(user_page_range(['只读前两页。','解除页数限制，阅读全文并注明原文页码。']))

    def test_user_instructions_are_not_rewritten(self):
        """仅用于歧义检查的副本可变，持久用户文本和返回原话不变。"""
        text='仅使用附件的物理第1至2页，并注明原文页码。'
        instructions=[text];scope=user_page_range(instructions)
        self.assertEqual(instructions,[text]);self.assertIn(scope.instruction,text)


if __name__=='__main__':unittest.main()
