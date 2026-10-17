# 行动循环测试：用可控模型验证决策、工具观察、取消和有界结束。
import unittest
from typing import Any

from pydantic import BaseModel

from research_loop.agents.graph import build_graph, initial_state
from research_loop.domain.tasks import Observation
from research_loop.providers.deepseek import ModelResponse, ToolCall
from research_loop.tools.base import ToolSpec
from research_loop.tools.registry import ToolRegistry


class EmptyArgs(BaseModel):
    pass


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools=None, force_text=False):
        self.calls.append({"tools": tools, "force_text": force_text, "messages": messages})
        return self.responses.pop(0)


def response(*calls, content=None):
    return ModelResponse(content=content, tool_calls=[ToolCall(str(index), name, args) for index, (name, args) in enumerate(calls)])


class AgentLoopTests(unittest.TestCase):
    def _registry(self, handlers):
        return ToolRegistry([ToolSpec(name, name, EmptyArgs, handler) for name, handler in handlers.items()])

    def test_successful_tool_then_finish(self):
        provider = FakeProvider([response(("ok", {})), response(content="完成")])
        registry = self._registry({"ok": lambda args, ctx: Observation(ok=True, kind="success", data={"value": 1})})
        result = build_graph(provider, registry, {}).invoke(initial_state("t", "task", "deny"))
        self.assertTrue(result["finished"])
        self.assertEqual(result["final_answer"], "完成")
        self.assertEqual(result["tool_actions"], 1)

    def test_failure_observation_allows_changed_action(self):
        provider = FakeProvider([response(("bad", {})), response(("good", {})), response(content="改路后完成")])
        registry = self._registry({
            "bad": lambda args, ctx: Observation(ok=False, kind="execution_error", message="controlled failure", retryable=True),
            "good": lambda args, ctx: Observation(ok=True, kind="success", data={"done": True}),
        })
        result = build_graph(provider, registry, {}).invoke(initial_state("t", "task", "deny"))
        self.assertEqual(result["final_answer"], "改路后完成")
        self.assertEqual([a["tool_name"] for a in result["actions"]], ["bad", "good"])

    def test_cancel_and_budget_finish_without_hidden_retry(self):
        provider = FakeProvider([response(("ok", {})), response(content="收尾")])
        registry = self._registry({"ok": lambda args, ctx: Observation(ok=True, kind="success")})
        cancelled = initial_state("t", "task", "deny")
        cancelled["cancelled"] = True
        result = build_graph(provider, registry, {}).invoke(cancelled)
        self.assertEqual(len(provider.calls), 0)
        self.assertEqual(result["final_reason"], "cancelled")

        result = build_graph(provider, registry, {}, max_model_requests=2).invoke(initial_state("b", "budget", "deny", max_model_requests=2))
        self.assertEqual(result["model_requests"], 2)
        self.assertTrue(provider.calls[-1]["force_text"])

    def test_last_request_keeps_existing_structured_finish(self):
        """最后请求保留交付schema，可结束任务而不把工具调用输出成正文。"""
        provider = FakeProvider([response(("finish_task", {}))])
        registry = self._registry({"finish_task": lambda args, ctx: Observation(
            ok=True, kind="success", data={"delivery": {"answer_markdown": "已有成果；缺证已披露", "draft_id": None, "saved_path": None, "gaps": ["仍缺原文"]}})})
        result = build_graph(provider, registry, {}).invoke(initial_state("last", "回顾", "deny", max_model_requests=1))
        self.assertFalse(provider.calls[0]["force_text"])
        self.assertEqual([t["function"]["name"] for t in provider.calls[0]["tools"]], ["finish_task"])
        self.assertTrue(result["finished"])
        self.assertEqual(result["delivery"]["answer_markdown"], "已有成果；缺证已披露")

    def test_last_request_cannot_execute_hidden_nonterminal_tool(self):
        """模型即使提出未提供的动作也不能在收尾时发生新的副作用。"""
        effects = []
        provider = FakeProvider([response(("save", {}))])
        registry = self._registry({
            "finish_task": lambda args, ctx: Observation(ok=True, kind="success"),
            "save": lambda args, ctx: effects.append("saved"),
        })
        result = build_graph(provider, registry, {}).invoke(initial_state("last", "回顾", "allow", max_model_requests=1))
        self.assertEqual(effects, [])
        self.assertEqual(result["actions"][0]["observation"]["data"]["error_code"], "TOOL_NOT_AVAILABLE")
        self.assertIn("budget exhausted", result["final_reason"])

    def test_textual_tool_invocation_is_not_a_public_answer_or_action(self):
        """供应商返回伪调用文本时停止且保留错误，不解析执行或发布为研究答案。"""
        for content in ['<finish_task><answer_markdown>未核对</answer_markdown></finish_task>',
                        '<｜｜DSML｜｜ calls><invoke name="save_note">text</invoke>']:
            with self.subTest(content=content):
                provider=FakeProvider([response(content=content)])
                result=build_graph(provider,self._registry({}),{}).invoke(initial_state('bad-output','研究','deny'))
                self.assertIn('textual tool invocation',result['final_reason'])
                self.assertFalse(result.get('final_answer'))
                self.assertFalse(result.get('actions'))
