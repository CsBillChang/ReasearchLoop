"""DPR 行为对照：以本地上游函数为预期，比较实际请求序列、返回文本和发布边界。"""
import ast
import copy
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from integration import test_dpr_consolidation as consolidation
from research_loop.providers import dpr_summary
from research_loop.domain.notes import CreateNoteInput, ReviewResult
from research_loop.domain.tasks import SavePolicy, TaskCreate
from research_loop.services.notes import NoteService
from research_loop.services.tasks import TaskService
from research_loop.workers.executor import TaskExecutor

UPSTREAM = Path(os.environ.get("RESEARCHLOOP_DPR_REFERENCE", str(Path.home() / "Downloads/daily-paper-reader-main/src/6.generate_docs.py")))
FUNCTIONS = ('call_llm_text', 'strip_auto_sections', 'yaml_escape_value',
             'build_markdown_content', 'generate_deep_summary')


def executable_tree(node):
    """忽略中文文档说明和源码位置，只对照函数的实际语法树。"""
    node = copy.deepcopy(node)
    if (isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body.pop(0)
    return ast.dump(node, include_attributes=False)


@unittest.skipUnless(UPSTREAM.exists(), 'requires the explicitly downloaded DPR source for differential comparison')
class DPRSourceComparisonTests(unittest.TestCase):
    """不执行上游模块全局代码，只抽取五个被复用的纯函数做独立对照。"""

    def setUp(self):
        self.source = ast.parse(UPSTREAM.read_text())
        self.functions = [n for n in self.source.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
        self.waits = []
        self.oracle = {'os': os, 'json': json, 'Any': Any, 'Dict': dict, 'List': list,
                       'DeepSeekClient': Any, 'LLM_CLIENT': None, 'log': lambda value: None,
                       'time': types.SimpleNamespace(sleep=self.waits.append)}
        exec(compile(ast.Module(body=self.functions, type_ignores=[]), str(UPSTREAM), 'exec'), self.oracle)

    def test_five_functions_match_upstream_executable_ast(self):
        """生成、文本调用、元数据、转义与去旧稿五个函数可执行 AST 必须与上游相同。"""
        current = ast.parse(Path(dpr_summary.__file__).read_text())
        actual = {n.name: n for n in current.body if isinstance(n, ast.FunctionDef)}
        for node in self.functions:
            with self.subTest(function=node.name):
                self.assertEqual(executable_tree(node), executable_tree(actual[node.name]))

    def test_requests_outputs_retries_and_fallback_match_upstream(self):
        """覆盖首轮完成、续写、三轮失败、空响应、普通异常与非末尾结束标记。"""
        cases = [
            ['完整（完）'], ['前半', '后半（完）'],
            ['a', 'b', 'c（完）'], ['a', 'b', 'c', 'd', 'e', 'f（完）'],
            ['a', 'b', 'c', 'd', 'e', 'f'], ['', '', ''],
            ['', 'ready（完）'], [ValueError('known error'), 'ready（完）'],
            ['a', ValueError('known continuation error'), 'b（完）'],
            ['含（完）但结尾还有文字'],
        ]
        class Client:
            def __init__(self, responses):
                self.kwargs = {'thinking': {'type': 'disabled'}}
                self.responses = list(responses)
                self.calls = []
            def chat(self, messages, response_format=None):
                self.calls.append({'messages': copy.deepcopy(messages), **self.kwargs})
                value = self.responses.pop(0)
                if isinstance(value, Exception): raise value
                return {'content': value, 'finish_reason': 'length'}
        with tempfile.TemporaryDirectory() as directory:
            md, txt = Path(directory)/'paper.md', Path(directory)/'paper.txt'
            md.write_text('# Paper\nAbstract\n\n---\n\n## 论文详细总结（自动生成）\nOLD SUMMARY')
            txt.write_text('Full original text')
            for case in cases:
                with self.subTest(case=[str(x) for x in case]):
                    expected, actual = Client(case), Client(case)
                    self.waits.clear()
                    value = self.oracle['generate_deep_summary'](str(md), str(txt), client=expected)
                    with patch.object(dpr_summary, 'time', types.SimpleNamespace(sleep=lambda seconds: None)):
                        current = dpr_summary.generate_deep_summary(str(md), str(txt), client=actual)
                    self.assertEqual(current, value)
                    self.assertEqual(actual.calls, expected.calls)
                    self.assertLessEqual(len(actual.calls), 6)
                    self.assertNotIn('OLD SUMMARY', json.dumps(actual.calls))
            md.unlink()
            self.assertIsNone(dpr_summary.generate_deep_summary(str(md), str(txt), client=Client([])))


class DPRPublicationTests(unittest.TestCase):
    """复用已有真实服务夹具，验证移除额外发布门后的实际 Worker 行为。"""

    setUp = consolidation.DPRConsolidationTests.setUp
    summary_fixture = consolidation.DPRConsolidationTests.summary_fixture
    raw = staticmethod(consolidation.DPRConsolidationTests.raw)

    def test_three_rounds_replay_receipts_and_return_last_initial(self):
        calls = []
        def handler(req):
            calls.append(json.loads(req.content))
            return httpx.Response(200, json=self.raw('output-' + str(len(calls)), 'length'))
        tid, state, graph = self.summary_fixture(handler)
        result = graph.invoke(state)
        self.assertEqual([c['max_tokens'] for c in calls], [4096,2048,4096,2048,4096,2048])
        self.assertEqual(calls[0]['messages'], calls[2]['messages'])
        self.assertEqual(calls[2]['messages'], calls[4]['messages'])
        self.assertEqual(result['final_answer'], 'output-5')
        self.assertEqual(result['deep_reading']['status'], 'partial')
        self.assertNotIn('evidence:one', json.dumps(calls))
        graph.invoke(self.db.get_checkpoint(tid))
        self.assertEqual(len(calls), 6)

    def test_marker_not_finish_reason_or_tail_controls_completeness(self):
        tid, state, graph = self.summary_fixture(lambda req: httpx.Response(
            200, json=self.raw('正文（完）附言', 'length')))
        result = graph.invoke(state)
        self.assertEqual(result['deep_reading']['status'], 'complete')
        self.assertEqual(len(self.db.list_model_calls(tid)), 1)

    def test_optional_reviewer_does_not_rewrite_or_gate_saved_markdown(self):
        task = TaskService(self.db).create(TaskCreate(prompt='保存笔记', save_policy='allow'))
        notes = NoteService(self.db, self.files)
        text = '# 原始 Markdown\n\n原文未提供 GPU 驱动版本。'
        draft = notes.create(task.task_id, CreateNoteInput(title='test', markdown=text))
        notes.save_review(ReviewResult(review_id='optional', task_id=task.task_id,
                                      draft_id=draft.draft_id, content_hash=draft.content_hash,
                                      status='insufficient', gaps=['figure not interpreted']))
        saved = notes.save(task.task_id, SavePolicy.allow, draft.draft_id, draft.content_hash)
        self.assertEqual(saved.status, 'saved')
        self.assertEqual(Path(saved.path).read_text(), text)
        self.assertEqual(saved.review_status, 'insufficient')
        self.assertNotIn('Sources', draft.markdown)
        repeated = notes.save(task.task_id, SavePolicy.allow, draft.draft_id, draft.content_hash)
        self.assertEqual(repeated.status, 'already_saved')
        Path(saved.path).write_text(text+'用户编辑')
        with self.assertRaises(ValueError):
            notes.save(task.task_id, SavePolicy.allow, draft.draft_id, draft.content_hash)

    def test_strict_publication_rejects_exhausted_summary(self):
        """三轮仍未完整时按 DPR 严格入口拒绝发布，不要求 Reviewer 改变结论。"""
        task = TaskService(self.db).create(TaskCreate(prompt='精读并保存', save_policy='allow'))
        child = self.db.create_subtask(task.task_id, task.task_id, 0, 'read', {}, 'read')
        self.db.save_subtask_result(child['subtask_id'], 'partial', {'deep_reading_status':'partial'})
        notes = NoteService(self.db, self.files)
        draft = notes.create(task.task_id, CreateNoteInput(title='test', markdown='没有完整生成'))
        with self.assertRaisesRegex(ValueError, '未完整生成'):
            notes.save(task.task_id, SavePolicy.allow, draft.draft_id, draft.content_hash)
        self.assertFalse(self.db.get_receipt(draft.draft_id, draft.content_hash))

    def test_legacy_search_response_completes_without_new_plan_contract(self):
        """旧checkpoint按冻结合同恢复，不强迫历史只读回答执行新目标工具。"""
        legacy_prompt = '你是ResearchLoop的Supervisor。仅查找论文，不强制全文精读。'
        class Provider:
            def send(inner, messages, tools, **kwargs):
                self.assertEqual(legacy_prompt, messages[0]['content'])
                return self.raw('找到符合条件的论文：https://paper.example/title')
            def close(inner): pass
        task = TaskService(self.db).create(TaskCreate(prompt='找一篇今年CVPR论文'))
        from research_loop.agents.graph import initial_state
        legacy_state = initial_state(task.task_id, '找一篇今年CVPR论文', 'deny')
        # 历史恢复测试必须冻结旧规则，不能临时从最新版提示词猜测旧措辞。
        legacy_state['messages'][0]['content'] = legacy_prompt
        self.db.save_checkpoint(task.task_id, legacy_state)
        executor = TaskExecutor(self.settings, self.db, lambda settings: Provider())
        result = executor.execute(task.task_id)
        self.assertEqual(self.db.get_task(task.task_id)['status'], 'completed')
        self.assertIsNone(result.draft_id)
        self.assertFalse(self.db.list_subtasks(task.task_id))
        self.assertEqual(len(self.db.list_model_calls(task.task_id)), 1)

    def test_warnings_not_blocking_but_explicit_missing_requirement_is_partial(self):
        from research_loop.domain.tasks import TaskResult
        task = TaskService(self.db).create(TaskCreate(prompt='查找论文'))
        executor = TaskExecutor(self.settings, self.db)
        result = TaskResult(answer_markdown='论文链接', gaps=['全文已覆盖', 'figure not interpreted'])
        self.assertEqual(executor._delivery_status(task.task_id, {'final_reason':'model finished'}, result).value, 'completed')
        self.db.save_action({'action_id':'gap','task_id':task.task_id,'sequence':1,'tool_name':'report_task_gap',
                             'arguments':{'requirement':'high citations'},'status':'succeeded','observation':{}})
        self.assertEqual(executor._delivery_status(task.task_id, {'final_reason':'model finished'}, result).value, 'partial')
