"""当前Memory结束门的合成边界复现；仅执行原生工具节点，无模型调用。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from research_loop.agents.graph import build_graph, initial_state
from research_loop.config import Settings
from research_loop.domain.execution_policy import inherit_execution_policy
from research_loop.domain.memory import RememberInput
from research_loop.domain.tasks import TaskCreate
from research_loop.services.conversations import ConversationService
from research_loop.services.coverage import CoverageService
from research_loop.services.memory import MemoryService
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.tools.delegation import specs as delegation_specs
from research_loop.tools.memory import specs as memory_specs
from research_loop.tools.registry import ToolRegistry


class MemoryCompletionBoundaryTests(unittest.TestCase):
    """刻画现有门禁，禁止把无关档案读取当成写入或撤销的事实。"""

    observations = []

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = Settings(project_root=root, data_dir=root/'data', output_dir=root/'output',
            database_url='sqlite:///'+str(root/'runtime.sqlite3'), reserve_model_requests=0)
        self.db = Database(self.settings.database_url)
        self.db.ensure_schema()
        self.tasks = TaskService(self.db, self.settings)
        self.memory = MemoryService(self.db)
        names = {'set_research_plan', 'finish_task', 'get_profile', 'remember_preference', 'forget_preference'}
        registry = ToolRegistry([*delegation_specs(), *memory_specs()], allowed_names=names)
        self.graph = build_graph(None, registry,
            {'coverage': CoverageService(self.db), 'memory': self.memory}, self.db,
            reserve_model_requests=0)

    def create(self, prompt):
        conversation = ConversationService(self.db, self.settings).create('合成边界', uuid4().hex)
        identity = self.tasks.create(TaskCreate(prompt=prompt, conversation_id=conversation['conversation_id'],
            save_policy='deny', idempotency_key=uuid4().hex)).task_id
        state = initial_state(identity, prompt, 'deny', reserve_model_requests=0)
        inherit_execution_policy(state, self.db.execution_policy(identity), self.db.budget_summary(identity))
        return identity, state

    def execute(self, state, name, arguments):
        """程序来源动作走真实工具校验和持久化，在模型决策之前暂停。"""
        identity = 'programmatic-boundary-'+uuid4().hex
        state['pending_tool_calls'] = [{'call_id': identity, 'name': name, 'arguments': arguments}]
        state['pending_tool_scope'] = [name]
        state['messages'].append({'role': 'assistant', 'content': '合成边界程序动作，非模型输出。',
            'reasoning_content': '', 'tool_calls': [{'id': identity, 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}}]})
        with patch('research_loop.config.get_settings', return_value=self.settings):
            state = self.graph.invoke(state, interrupt_after=['execute_tools'])
        self.assertEqual(self.db.list_model_calls(state['task_id']), [])
        return state, state['actions'][-1]['observation']

    def plan(self, state, description):
        state, result = self.execute(state, 'set_research_plan', {'task_kind': 'memory',
            'goal': description, 'goals': [{'goal_id': 'answer', 'origin': 'user',
                'source_quote': state['prompt'], 'description': description, 'min_papers': 0}]})
        self.assertTrue(result['ok'], result)
        return state

    @staticmethod
    def finish(answer):
        return {'answer_markdown': answer, 'goals': [{'goal_id': 'answer',
            'status': 'satisfied', 'reason': '合成程序提交的待校验完成声明'}]}

    def test_current_input_requires_unrelated_memory_operation(self):
        identity, state = self.create('仅依据当前输入计算：8条中保留6条，保留比例是多少？')
        state = self.plan(state, '计算6/8')
        state, blocked = self.execute(state, 'finish_task', self.finish('6/8=75%。'))
        self.assertFalse(blocked['ok'])
        self.assertIn('memory任务必须有真实记忆操作或读回', blocked['message'])
        state, profile = self.execute(state, 'get_profile', {})
        self.assertTrue(profile['ok'])
        state, accepted = self.execute(state, 'finish_task', self.finish('6/8=75%。'))
        self.assertTrue(accepted['ok'], accepted)
        self.assertEqual(self.memory.active(), [])
        actions = self.db.list_actions(identity)
        self.assertFalse(any(a['tool_name'] in {'read_memory', 'remember_preference', 'forget_preference'}
                             for a in actions))
        self.observations.append({'case': 'current_input_only', 'before_profile': 'rejected',
            'after_profile': 'accepted', 'memory_entries': 0, 'real_model_calls': 0})

    def test_profile_read_does_not_prove_claimed_preference_write(self):
        identity, state = self.create('以后请默认用中文回答，请记住这个长期偏好。')
        state = self.plan(state, '记录长期语言偏好')
        state, blocked = self.execute(state, 'finish_task', self.finish('已经记住默认中文。'))
        self.assertFalse(blocked['ok'])
        state, profile = self.execute(state, 'get_profile', {})
        self.assertTrue(profile['ok'])
        state, accepted = self.execute(state, 'finish_task', self.finish('已经记住默认中文。'))
        self.assertTrue(accepted['ok'], accepted)
        self.assertEqual(self.memory.active(), [])
        self.assertFalse(any(a['tool_name'] == 'remember_preference' for a in self.db.list_actions(identity)))
        self.observations.append({'case': 'claim_write_after_profile_only', 'finish_accepted': True,
            'actual_write': False, 'semantic_success': False, 'real_model_calls': 0})

    def test_profile_read_does_not_prove_claimed_revocation(self):
        seed_id, _ = self.create('以后默认使用中文。')
        entry = self.memory.remember(seed_id, RememberInput(slot='language', kind='explicit',
            content='中文', source_quote='以后默认使用中文。'))
        identity, state = self.create('请删除之前默认使用中文的长期偏好。')
        state = self.plan(state, '撤销旧语言偏好')
        state, profile = self.execute(state, 'get_profile', {})
        self.assertTrue(profile['ok'])
        state, accepted = self.execute(state, 'finish_task', self.finish('已删除旧语言偏好。'))
        self.assertTrue(accepted['ok'], accepted)
        self.assertEqual(self.memory.active()[0]['memory_id'], entry['memory_id'])
        self.assertFalse(any(a['tool_name'] == 'forget_preference' for a in self.db.list_actions(identity)))
        self.observations.append({'case': 'claim_revocation_after_profile_only', 'finish_accepted': True,
            'entry_still_active': True, 'semantic_success': False, 'real_model_calls': 0})

    def test_actual_write_and_revocation_have_operation_receipts(self):
        identity, state = self.create('以后默认使用中文。')
        state = self.plan(state, '记录长期语言偏好')
        state, written = self.execute(state, 'remember_preference', {'slot': 'language',
            'kind': 'explicit', 'content': '中文', 'source_quote': '以后默认使用中文。'})
        self.assertTrue(written['ok'], written)
        entry = written['data']['memory']
        self.assertEqual(self.memory.active()[0]['memory_id'], entry['memory_id'])
        state, revoked = self.execute(state, 'forget_preference', {'memory_id': entry['memory_id'],
            'expected_version': entry['version']})
        self.assertTrue(revoked['ok'], revoked)
        self.assertEqual(self.memory.active(), [])
        operations = self.db.fetchall('SELECT * FROM memory_operations')
        self.assertEqual(len(operations), 2)
        self.observations.append({'case': 'actual_operations', 'write_and_revoke_succeeded': True,
            'persistent_operation_receipts': len(operations), 'real_model_calls': 0})
