"""逐调用准入与交接错误反馈的节点测试；完整调度由独立集成用例验证。"""
import json
import unittest

from pydantic import BaseModel
from research_loop.agents.graph import build_graph, initial_state
from research_loop.domain.tasks import Observation
from research_loop.providers.deepseek import validate_messages
from research_loop.tools.base import ToolSpec
from research_loop.tools.registry import ToolRegistry
from research_loop.services.research_progress import progress_fingerprint


class NoArgs(BaseModel):
    """本测试合成工具的空参数，不用于生产schema。"""


class RecoverableToolCallTests(unittest.TestCase):
    """校验范围拒绝无副作用、配对观察完整、依赖操作延后。"""

    def setUp(self):
        """构造不发送请求的图，只调用execute_tools节点。"""
        self.effects = []

        def read(args, ctx):
            """返回授权档案的导航，不能冒充科研进展。"""
            self.effects.append('read')
            return Observation(ok=True, kind='success', data={'source_type': 'context_archive',
                'source_id': 'archive-A', 'content_hash': 'version-A', 'offset': 0,
                'content': '原文片段', 'total_chars': 100})

        def finish(args, ctx):
            """模拟精确交付，不执行文件写入。"""
            self.effects.append('finish')
            return Observation(ok=True, kind='success', data={'delivery': {'answer_markdown': '保留正文',
                'draft_id': None, 'saved_path': None, 'gaps': []}})

        registry = ToolRegistry([ToolSpec('read_memory', 'read', NoArgs, read),
            ToolSpec('finish_task', 'finish', NoArgs, finish),
            ToolSpec('retrieve_evidence', 'hidden', NoArgs, lambda *_: self.fail('hidden tool executed'))])
        self.node = build_graph(None, registry, {}).nodes['execute_tools']

    def state(self, names):
        """精确声明本批响应对应的工具范围，避免从注册表扩大权限。"""
        state = initial_state('synthetic', '交接', 'deny')
        calls = [{'call_id': str(i), 'name': name,
                  'arguments': {'source_type': 'context_archive'} if name == 'read_memory' else {}}
                 for i, name in enumerate(names)]
        state.update(pending_tool_calls=calls, pending_tool_scope=['read_memory', 'finish_task'],
            research_handoff={'phase': 'requested', 'trigger_action_id': 'boundary',
                              'progress_fingerprint': progress_fingerprint({})})
        state['messages'].append({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': c['call_id'], 'type': 'function', 'function': {
                'name': c['name'], 'arguments': json.dumps(c['arguments'])}} for c in calls]})
        return state

    def test_mixed_batch_keeps_independent_read_in_both_orders(self):
        """非法调用在前或后，合法只读都保留且整批tool消息配对。"""
        for names in [('read_memory', 'retrieve_evidence'), ('retrieve_evidence', 'read_memory')]:
            with self.subTest(names=names):
                self.effects.clear()
                result = self.node.invoke(self.state(names))
                self.assertEqual(self.effects, ['read'])
                self.assertEqual(result['tool_actions'], 1)
                self.assertEqual(result['action_sequence'], 2)
                self.assertFalse(result.get('finished'))
                self.assertEqual(result['research_handoff']['phase'], 'readback')
                rejected = next(a for a in result['actions'] if a['tool_name'] == 'retrieve_evidence')
                self.assertFalse(rejected['observation']['data']['tool_executed'])
                self.assertFalse(rejected['observation']['data']['budget_counted'])
                validate_messages(result['messages'])

    def test_failed_predecessor_defers_state_change(self):
        """不猜同批调用的依赖关系；状态变更留待模型看到错误后选择。"""
        result = self.node.invoke(self.state(['retrieve_evidence', 'finish_task']))
        self.assertEqual(self.effects, [])
        self.assertEqual(result['tool_actions'], 0)
        self.assertEqual(result['actions'][1]['observation']['data']['error_code'], 'TOOL_DEPENDENCY_UNRESOLVED')
        self.assertTrue(result['research_handoff'].get('tool_error'))
        self.assertFalse(result.get('finished'))

    def test_same_rejected_action_stops_without_new_progress(self):
        """错误反馈可用于纠正，但同一非法动作不能无限刷新交接。"""
        state = self.state(['retrieve_evidence'])
        first = self.node.invoke(state)
        again = self.state(['retrieve_evidence'])
        again.update({k: v for k, v in first.items() if k not in {'messages', 'pending_tool_calls', 'pending_tool_scope'}})
        again['messages'] = first['messages'] + again['messages'][-1:]
        result = self.node.invoke(again)
        self.assertTrue(result['finished'])
        self.assertIn('unavailable', result['final_reason'])
        self.assertEqual(result['tool_actions'], 0)
        self.assertEqual(self.effects, [])

    def test_recovery_keeps_supplied_scope_instead_of_registry(self):
        """恢复的空/缩小范围不能因日常注册表有同名工具而放行。"""
        state = self.state(['read_memory'])
        state['pending_tool_scope'] = []
        result = self.node.invoke(state)
        self.assertEqual(self.effects, [])
        self.assertEqual(result['actions'][0]['observation']['data']['available_tools'], [])

