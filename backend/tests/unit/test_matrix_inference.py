"""矩阵请求模式边界、实际 HTTP 参数和旧收据恢复；全部使用本地替身。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx

from research_loop.agents.graph import build_graph, matrix_thinking
from research_loop.domain.tasks import ResearchBrief
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.storage.mysql import Database
from research_loop.tools.registry import ToolRegistry


def response():
    """返回可完整持久化的合成响应，不代表真实模型质量。"""
    return {'id': 'fixture', 'usage': {'total_tokens': 9}, 'choices': [
        {'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'fixture answer'}}]}


class MatrixInferenceTests(unittest.TestCase):
    """非矩阵与未取证保持原设置，矩阵参数穿透两种供应商入口。"""

    def test_only_planned_supervisor_with_evidence(self):
        """缺省、子角色和无证据不会切模式；不匹配论文正文关键词。"""
        state = {'role': 'supervisor', 'research_plan': {'evidence_matrix_requested': True}, 'evidence_ids': ['page']}
        self.assertEqual(matrix_thinking(state), {'type': 'disabled'})
        for change in ({'role': 'researcher'}, {'role': 'reviewer'}, {'evidence_ids': []},
                       {'research_plan': {}}, {'research_plan': {'evidence_matrix_requested': False}}):
            with self.subTest(change=change):
                self.assertIsNone(matrix_thinking({**state, **change}))
        self.assertFalse(ResearchBrief(goal='ordinary question').evidence_matrix_requested)

    def test_provider_wire_payload_stream_and_regular(self):
        """实际构造 POST JSON；未显式传参时两条路径均省略 thinking。"""
        captured = []

        def handle(request):
            captured.append(json.loads(request.content))
            if captured[-1].get('stream'):
                data = {'id': 'fixture', 'usage': {'total_tokens': 9}, 'choices': [
                    {'index': 0, 'delta': {'content': 'fixture answer'}, 'finish_reason': 'stop'}]}
                return httpx.Response(200, text='data: ' + json.dumps(data) + '\n\ndata: [DONE]\n\n')
            return httpx.Response(200, json=response())

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            provider = DeepSeekProvider('fixture', client=client)
            for stream in (False, True):
                for explicit in (False, True):
                    kwargs = {'thinking': {'type': 'disabled'}} if explicit else {}
                    if stream:
                        kwargs['on_public'] = lambda _: None
                    call = provider.send_stream if stream else provider.send
                    raw = call([{'role': 'user', 'content': 'hello'}], max_tokens=8192, **kwargs)
                    self.assertEqual(raw['usage']['total_tokens'], 9)
                    self.assertEqual(captured[-1].get('thinking'), {'type': 'disabled'} if explicit else None)
                    self.assertEqual(captured[-1]['max_tokens'], 8192)

    def test_graph_persists_and_replays_frozen_parameters(self):
        """流式/非流式传递一致；历史成功请求无参数时不补写，也不重发。"""
        with tempfile.TemporaryDirectory() as folder:
            db = Database('sqlite:///' + str(Path(folder) / 'test.db'))
            db.ensure_schema()
            for streaming in (False, True):
                for replay in (False, True):
                    with self.subTest(streaming=streaming, replay=replay):
                        tid = f'task-{streaming}-{replay}'
                        db.create_task(tid, None, 'h', 'matrix', 'deny', [])
                        messages = [{'role': 'user', 'content': 'matrix'}]
                        frozen = {'messages': messages, 'tools': [], 'force_text': False, 'terminal_only': False}
                        if replay:
                            db.save_model_call(tid, 1, 'succeeded', frozen, response=response(),
                                               usage={'total_tokens': 9}, request_id='old', finish_reason='stop')
                        provider = Mock(spec=['send', 'send_stream']) if streaming else Mock(spec=['send'])
                        provider.send.return_value = response()
                        if streaming:
                            provider.send_stream.return_value = response()
                        context = Mock()
                        context.prepare.return_value = (messages, {})
                        context.settings.context_output_reserve = 8192
                        state = {'task_id': tid, 'prompt': 'matrix', 'messages': messages, 'role': 'supervisor',
                                 'research_plan': {'evidence_matrix_requested': True}, 'evidence_ids': ['page']}
                        result = build_graph(provider, ToolRegistry(), {'context': context}, db=db).invoke(state)
                        self.assertEqual(result['final_answer'], 'fixture answer')
                        saved = json.loads(db.get_model_call(tid, 1)['request_json'])
                        if replay:
                            self.assertEqual(saved, frozen)
                            provider.send.assert_not_called()
                            if streaming:
                                provider.send_stream.assert_not_called()
                        else:
                            call = provider.send_stream if streaming else provider.send
                            self.assertEqual(call.call_args.kwargs['thinking'], {'type': 'disabled'})
                            self.assertEqual(saved['thinking'], {'type': 'disabled'})
