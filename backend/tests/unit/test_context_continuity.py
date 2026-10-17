"""上下文接续纯函数测试：区分真实往返、普通文本与当前工具范围。"""
import unittest
from research_loop.domain.context import archive_summary, contains_source_exchange, source_spans, textual_delivery
from research_loop.tools.registry import ToolRegistry, schema_names
from research_loop.tools.memory import specs


class ContextContinuityUnitTests(unittest.TestCase):
    """源观察不可在无读回能力时当作普通可丢摘要。"""

    def test_schema_names_respect_role_and_frozen_scope(self):
        """可见名字来自实际schema，不从全局注册表误推权限。"""
        registry = ToolRegistry(specs(), allowed_names={'read_memory'})
        self.assertEqual(schema_names(registry.schemas('supervisor')), {'read_memory'})
        self.assertEqual(schema_names(registry.schemas('researcher')), set())
        with self.assertRaises(KeyError):
            registry.get('remember_preference')

    def test_only_typed_archive_metadata_is_recognized(self):
        """普通正文、工具回复与空JSON均不能冒充档案指令。"""
        for message in ({'role':'assistant','content':'read_memory something'},
                        {'role':'assistant','content':'[]'},
                        {'role':'tool','content':'{"archived_tool_exchange":true}'}):
            self.assertIsNone(archive_summary(message))
        self.assertTrue(archive_summary({'role':'assistant','content':'{"archived_tool_exchange":true}'}))

    def test_source_and_side_effects_have_different_retention(self):
        """原文/历史/委派结果保留；草稿写入仍可按身份摘要且不能重做。"""
        for name, expected in [('read_paper',True),('retrieve_evidence',True),
                               ('read_memory',True),('delegate_research',True),('create_note',False)]:
            with self.subTest(name=name):
                group=[{'role':'assistant','tool_calls':[{'function':{'name':name}}]}]
                self.assertEqual(contains_source_exchange(group),expected)

    def test_source_windows_are_exact_and_cover_relevant_tail(self):
        """字符窗口不改写原句，非ASCII和文末内容仍能精确定位。"""
        text = '首段：输入条件。' + 'ordinary text. '*400 + 'TAIL language embedding 条件。'
        spans=source_spans(text,'language embedding',200)
        self.assertLessEqual(sum(s['end']-s['start'] for s in spans),200)
        self.assertTrue(any('TAIL' in s['text'] for s in spans))
        for span in spans:self.assertEqual(span['text'],text[span['start']:span['end']])

    def test_textual_finish_is_candidate_not_executable(self):
        """JSON或围栏内的交付参数不执行；普通JSON数据和Markdown不误伤。"""
        for value in ('{"goals":[],"answer_markdown":"候选"}', '```json\n{"goals":[],"answer_markdown":"候选"}\n```'):
            self.assertEqual(textual_delivery(value),'候选')
        self.assertEqual(textual_delivery('{"tool_calls":[{"name":"save_note"}]}'),'')
        self.assertIsNone(textual_delivery('{"accuracy":0.8}'))
        self.assertIsNone(textual_delivery('# 正常回答'))
