"""Memory 路由说明的请求接线与既有硬合同；全程合成数据并阻断网络。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from research_loop.agents.memory import PREFIX
from research_loop.config import Settings
from research_loop.domain.tasks import ActionRecord, ResearchBrief, SavePolicy, TaskCreate
from research_loop.services.conversations import ConversationService
from research_loop.services.coverage import CoverageService
from research_loop.services.memory import MemoryService
from research_loop.services.tasks import TaskService
from research_loop.storage.memory import dump
from research_loop.storage.mysql import Database
from research_loop.tools.base import ToolContext
from research_loop.tools.delegation import specs as delegation_specs
from research_loop.tools.memory import specs as memory_specs
from research_loop.tools.registry import ToolRegistry
from research_loop.workers.executor import TaskExecutor


class CaptureClarificationProvider:
    """捕获正式 Worker 请求，再用澄清结束本轮，不模拟语义路由成功。"""

    def __init__(self):
        self.requests = []

    def close(self):
        """替身没有外部资源。"""

    def send(self, messages, tools, **kwargs):
        self.requests.append(deepcopy({'messages': messages, 'tools': tools}))
        return {'id': 'synthetic-clarification',
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                'choices': [{'finish_reason': 'tool_calls', 'message': {
                    'role': 'assistant', 'content': '', 'tool_calls': [{
                        'id': 'synthetic-call', 'type': 'function', 'function': {
                            'name': 'request_clarification',
                            'arguments': dump({'question': '请明确本轮交付范围。',
                                               'fields': ['scope'], 'suggested_values': []})}}]}}]}


class MemoryRouteDescriptionTests(unittest.TestCase):
    """只核验工具说明可见性及程序边界，不把替身当真实模型效果。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = Settings(project_root=root, data_dir=root / 'data',
            output_dir=root / 'notes', database_url='sqlite:///' + str(root / 'test.db'),
            deepseek_api_key=None, discovery_backends={}, retrieval_enabled=False,
            max_model_requests=5, reserve_model_requests=0)
        for name in ('socket.socket.connect', 'socket.socket.connect_ex',
                     'httpx.HTTPTransport.handle_request',
                     'httpx.AsyncHTTPTransport.handle_async_request'):
            blocker = patch(name, side_effect=AssertionError('零模型回归禁止网络'))
            blocker.start()
            self.addCleanup(blocker.stop)
        self.db = Database(self.settings.database_url)
        self.db.ensure_schema()
        self.tasks = TaskService(self.db, self.settings)
        self.coverage = CoverageService(self.db)
        self.memory = MemoryService(self.db)
        self.registry = ToolRegistry(delegation_specs() + memory_specs())
        self.prompt = '合成数据：8条断言中7条正确、1条错误；删除2条正确断言。只用本条数据计算，不找论文、不精读、不保存。'

    def task(self, *, conversation=True):
        cid = (ConversationService(self.db, self.settings).create('合成讨论', uuid4().hex)
               ['conversation_id']) if conversation else None
        return self.tasks.create(TaskCreate(prompt=self.prompt, conversation_id=cid,
            save_policy='deny', idempotency_key=uuid4().hex)).task_id

    @staticmethod
    def plan(kind='memory'):
        return {'goal': '按合成数据解释当前状态', 'task_kind': kind, 'goals': [{
            'goal_id': 'derive', 'description': '按当前数据计算', 'origin': 'user',
            'source_quote': '只用本条数据计算', 'min_papers': 0}]}

    @staticmethod
    def finish():
        return {'answer_markdown': '当前保留6条，其中5条正确、1条错误。',
                'goals': [{'goal_id': 'derive', 'status': 'satisfied',
                           'reason': '按本轮给定数据完成推导'}]}

    def execute_tool(self, tid, name, arguments):
        """执行正式工具并保存实际观察，供结束合同读取成功行动。"""
        result = self.registry.execute(name, arguments, ToolContext(task_id=tid,
            save_policy=SavePolicy.deny, services={'coverage': self.coverage, 'memory': self.memory}))
        record = ActionRecord(task_id=tid, sequence=len(self.db.list_actions(tid)) + 1,
            tool_name=name, arguments=arguments,
            status='succeeded' if result.ok else 'failed', observation=result.model_dump(mode='json'))
        self.db.save_action(record.model_dump(mode='json'))
        return result

    def test_description_only_change_keeps_frozen_parameters_and_bindings(self):
        """参数指纹取自 V1.6.0；说明变更不能顺带调整参数或调用权限。"""
        frozen = {
            'set_research_plan': ('4e24e09a89077c8218ac2853fd474a43260230b50ba6e60ea5563f07723c527c', '_plan'),
            'finish_task': ('a97ae685d4694f46ee0bb129f1e78a4627aa78cb603e9a9828334c35a3d782aa', '_finish_task'),
        }
        for name, (expected_hash, handler) in frozen.items():
            spec = self.registry.get(name)
            parameters = json.dumps(spec.schema()['function']['parameters'], ensure_ascii=False,
                                    sort_keys=True, separators=(',', ':')).encode()
            self.assertEqual(hashlib.sha256(parameters).hexdigest(), expected_hash)
            self.assertEqual(spec.handler.__module__, 'research_loop.tools.delegation')
            self.assertEqual(spec.handler.__name__, handler)
            self.assertEqual(spec.roles, ('supervisor',))

    def test_worker_preserves_tool_descriptions_on_short_nonconversation_fast_path(self):
        """动态 Memory 头被快路径跳过，工具说明仍实际传至供应商边界。"""
        tid = self.task(conversation=False)
        provider = CaptureClarificationProvider()
        TaskExecutor(self.settings, self.db, provider_factory=lambda _: provider,
            supervisor_tools={'set_research_plan', 'finish_task', 'request_clarification'}).execute(tid)
        self.assertEqual(self.db.get_task(tid)['status'], 'needs_input')
        self.assertEqual(len(provider.requests), 1)
        request = provider.requests[0]
        self.assertFalse(any(str(m.get('content', '')).startswith(PREFIX)
                             for m in request['messages']))
        sent = {t['function']['name']: t for t in request['tools']}
        for name in ('set_research_plan', 'finish_task'):
            self.assertEqual(sent[name], self.registry.get(name).schema())
        receipt = self.db.list_model_calls(tid)[0]
        self.assertEqual(json.loads(receipt['request_json'])['tools'], request['tools'])

    def test_conversation_request_has_memory_context_and_same_tool_contracts(self):
        """有会话的短请求仍装配 Memory，不依赖容量压力才提供上下文。"""
        tid = self.task()
        provider = CaptureClarificationProvider()
        TaskExecutor(self.settings, self.db, provider_factory=lambda _: provider,
            supervisor_tools={'set_research_plan', 'finish_task', 'request_clarification',
                              'get_profile', 'search_memory', 'read_memory'}).execute(tid)
        self.assertEqual(self.db.get_task(tid)['status'], 'needs_input')
        self.assertEqual(len(provider.requests), 1)
        request = provider.requests[0]
        headers = [m['content'] for m in request['messages']
                   if str(m.get('content', '')).startswith(PREFIX + '\n')]
        self.assertEqual(len(headers), 1)
        context = json.loads(headers[0].rsplit('\n', 1)[-1])
        self.assertIn('get_profile', context['available_tools'])
        self.assertTrue(context['conversation_id'])
        sent = {t['function']['name']: t for t in request['tools']}
        self.assertEqual(sent['set_research_plan'], self.registry.get('set_research_plan').schema())

    def test_memory_finish_requires_successful_read_without_creating_preference(self):
        tid = self.task()
        self.assertTrue(self.execute_tool(tid, 'set_research_plan', self.plan()).ok)
        self.assertFalse(self.execute_tool(tid, 'finish_task', self.finish()).ok)
        self.assertTrue(self.execute_tool(tid, 'get_profile', {}).ok)
        delivery = self.execute_tool(tid, 'finish_task', self.finish())
        self.assertTrue(delivery.ok, delivery.message)
        self.assertEqual(delivery.data['delivery']['status'], 'completed')
        for table in ('memory_entries', 'papers', 'documents', 'page_evidence', 'save_receipts'):
            self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM ' + table)['n'], 0)

    def test_research_requires_user_material_goal_even_with_strategy_papers(self):
        tid = self.task()
        for kind in ('search', 'compare', 'survey'):
            for extra in ([], [{'goal_id': 'extra', 'description': '自拟取材',
                                'origin': 'strategy', 'min_papers': 2}]):
                value = self.plan(kind)
                value['goals'].extend(extra)
                with self.subTest(kind=kind, strategy=bool(extra)), self.assertRaisesRegex(ValueError, 'min_papers>0'):
                    self.coverage.prepare_plan(tid, ResearchBrief.model_validate(value))

    def test_memory_cannot_bypass_conversation_or_paper_deliverables(self):
        tid = self.task()
        for fields in ({'requested_count': 2}, {'deep_reading_requested': True}, {'save_requested': True}):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, '不使用memory计划'):
                self.coverage.prepare_plan(tid, ResearchBrief.model_validate({**self.plan(), **fields}))
        with self.assertRaisesRegex(ValueError, 'requires a conversation'):
            self.coverage.prepare_plan(self.task(conversation=False), ResearchBrief.model_validate(self.plan()))

    def test_zero_paper_plan_preserves_quote_and_actual_save_permission(self):
        tid = self.task()
        bad = self.plan()
        bad['goals'][0]['source_quote'] = '只用本条数据计算……不保存'
        with self.assertRaisesRegex(ValueError, 'verbatim'):
            self.coverage.prepare_plan(tid, ResearchBrief.model_validate(bad))
        good = self.coverage.prepare_plan(tid, ResearchBrief.model_validate({**self.plan(), 'save_policy': 'allow'}))
        self.assertEqual(good['save_policy'], 'deny')
        self.assertEqual(self.db.get_task(tid)['save_policy'], 'deny')


if __name__ == '__main__':
    unittest.main()
