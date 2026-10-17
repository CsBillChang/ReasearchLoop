"""明确附件名称不应使物理页限制失效；未知或冲突范围仍需澄清。"""
import unittest

from research_loop.domain.reading import user_page_range


# 逐字保留真实失败输入，避免用更容易解析的简化指令掩盖回归。
ORIGINAL_TEXT_PROMPT = (
    '仅根据附件 LaVer 物理第4页的正文，说明视觉头采用什么结构，用一句中文回答并注明物理页码。'
    '不要分析图表、外部检索、生成精读笔记或保存文件。'
)


class NamedAttachmentScopeTests(unittest.TestCase):
    """覆盖固定复验暴露的通用句式，不依赖具体论文或模型。"""

    def test_named_attachments_preserve_explicit_range(self):
        """名称、空格和连接词变化不改变明确的连续物理页范围。"""
        for text in [
            '仅使用附件 Alpha 与 Beta 的物理第1至2页，比较语言信息。',
            '只读取附件 语言模型报告 和 检测实验 的物理第3到4页。',
            '限定附件 methods-v2.pdf的物理前两页。',
            '仅使用附件的物理第1至2页。',
        ]:
            with self.subTest(text=text):
                scope = user_page_range([text])
                expected = (3, 4) if '第3到4' in text else (1, 2)
                self.assertEqual((scope.start, scope.end), expected)
                self.assertIn(scope.instruction, text)

    def test_conflicting_or_ambiguous_named_ranges_rejected(self):
        """解析不了时拒绝，不回退为全文权限或吞掉另一项页码限制。"""
        for text in [
            '仅使用附件 Alpha 的物理第1至2页；仅使用附件 Beta 的物理第3至4页。',
            '仅使用附件 Alpha 与 Beta 的物理前面几页。',
            '仅使用附件 Alpha 与 Beta 的物理第1页和第3页。',
            '仅使用附件 Alpha 与 Beta 的物理第3至1页。',
            '仅使用附件 Alpha 与 Beta 的物理第0至2页。',
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                user_page_range([text])

    def test_negation_and_user_clarification_keep_priority(self):
        """否定不是范围授权，用户后续澄清继续具有最高优先级。"""
        self.assertIsNone(user_page_range(['不要只读取附件 Alpha 与 Beta 的物理第1至2页，请读全文。']))
        self.assertIsNone(user_page_range(['附件 Alpha 的物理第1页介绍模型，请阅读全文。']))
        scope = user_page_range(['仅使用附件 Alpha 与 Beta 的物理第1至2页。', '仅限每篇物理第2至3页。'])
        self.assertEqual((scope.start, scope.end), (2, 3))
        self.assertIsNone(user_page_range(['仅使用附件 Alpha 与 Beta 的物理第1至2页。', '解除页数限制，请读全文。']))

    def test_original_text_and_visual_instructions_keep_page_four(self):
        """原失败全文必须成功解析，已通过的视觉指令保持相同单页身份。"""
        original_visual = (
            '只依据附件 LoRASculpt 的物理第4页，转写式(2)与式(4)，说明式(2)运算符的含义及式(4)指数的位置。'
            '若文字提取的符号损坏，请核对这一页原图，无法确认的地方明确保留。'
            '不要外部检索、重做整篇精读或保存文件。'
        )
        for text in (ORIGINAL_TEXT_PROMPT, original_visual):
            with self.subTest(text=text):
                instructions = [text]
                scope = user_page_range(instructions)
                self.assertEqual((scope.start, scope.end), (4, 4))
                self.assertIn(scope.instruction, text)
                self.assertEqual(instructions, [text])

    def test_named_connectors_and_optional_particle_do_not_change_range(self):
        """根据/依据、有无“的”及不同标题均解析明确范围，不按论文名分支。"""
        for verb in ('根据', '依据'):
            for particle in ('', '的', ' 的 '):
                for name in ('Alpha-v2.pdf', 'A Multimodal Study', '方法比较报告'):
                    for pages, expected in (('第4页', (4, 4)), ('第3至4页', (3, 4)), ('前两页', (1, 2))):
                        text = f'仅{verb}附件 {name} {particle}物理{pages}，回答问题并注明原文页码。'
                        with self.subTest(text=text):
                            scope = user_page_range([text])
                            self.assertEqual((scope.start, scope.end), expected)

    def test_new_wording_does_not_silence_invalid_or_conflicting_ranges(self):
        """新增句式仍拒绝离散、倒序、零页和跨附件冲突，不回退全文。"""
        for verb in ('根据', '依据'):
            for particle in ('', '的'):
                prefix = f'仅{verb}附件 Alpha {particle}物理'
                for suffix in ('前面几页', '第4页和第6页', '第4至2页', '第0页'):
                    with self.subTest(prefix=prefix, suffix=suffix), self.assertRaises(ValueError):
                        user_page_range([prefix + suffix + '，注明页码。'])
                with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                    user_page_range([prefix + '第4页；只依据附件 Beta 物理第5页。'])

    def test_new_wording_keeps_negation_reference_and_clarification_semantics(self):
        """描述位置、出处要求和否定不新增限制，后续用户指令仍最高优先。"""
        for text in (
            '附件 Alpha 物理第4页讨论结构，请阅读全文。',
            '不要仅根据附件 Alpha 物理第4页回答，请读全文。',
            '只依据附件 Alpha，并注明原文页码。',
        ):
            with self.subTest(text=text):
                self.assertIsNone(user_page_range([text]))
        scope = user_page_range([ORIGINAL_TEXT_PROMPT, '只依据附件 Alpha 物理第5至6页。'])
        self.assertEqual((scope.start, scope.end), (5, 6))
        scope = user_page_range([ORIGINAL_TEXT_PROMPT, '请注明原文页码。'])
        self.assertEqual((scope.start, scope.end), (4, 4))
        self.assertIsNone(user_page_range([ORIGINAL_TEXT_PROMPT, '解除页数限制，阅读全文。']))

    def test_named_prefix_cannot_bridge_sentence_boundaries(self):
        """附件名匹配不得跨逗号/句界拼出另一短句的阅读授权。"""
        for separator in ('，', '；', '。', '\n'):
            text = f'仅根据附件 Alpha{separator}物理第4页内容待核实。'
            with self.subTest(separator=separator):
                try:
                    scope = user_page_range([text])
                except ValueError:
                    continue
                # 独立后句不是阅读授权；维持原有拒绝或无范围语义，不拼成第4页。
                self.assertIsNone(scope)
