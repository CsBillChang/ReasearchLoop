"""零论文规划类型的隔离合同；合成数据与模型替身不证明真实模型路由质量。"""
from __future__ import annotations

import copy
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


class ZeroPaperProvider:
    """执行正式画像读取、零论文计划和结束工具，不构造论文事实或来源。"""

    def __init__(self, plan: dict, answer: str):
        """注入本例计划与回答，只替换供应商生成动作。"""
        self.plan = copy.deepcopy(plan)
        self.answer = answer
        self.calls = []

    def close(self):
        """替身没有外部资源。"""

    def send(self, messages, tools, **kwargs):
        """保留正式任务的三个模型收据及对应工具观察。"""
        self.calls.append(copy.deepcopy({'messages': messages, 'tools': tools}))
        done = [call['function']['name'] for message in messages
                for call in message.get('tool_calls', [])]
        if 'get_profile' not in done:
            name, args = 'get_profile', {}
        elif 'set_research_plan' not in done:
            name, args = 'set_research_plan', self.plan
        else:
            name, args = 'finish_task', {'answer_markdown': self.answer,
                'goals': [{'goal_id': goal['goal_id'], 'status': 'satisfied',
                           'reason': '仅使用本轮自足数据及实际可见的会话记录'}
                          for goal in self.plan['goals']]}
        sequence = len(self.calls)
        return {'id': 'zero-paper-fixture-' + str(sequence),
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
            'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant',
                'content': '', 'tool_calls': [{'id': 'zero-paper-call-' + str(sequence),
                    'type': 'function', 'function': {'name': name, 'arguments': dump(args)}}]}}]}


class ZeroPaperPlanRoutingTests(unittest.TestCase):
    """允许自足会话推导，保留研究材料门、来源引句和保存权限边界。"""

    def setUp(self):
        """每例创建临时SQLite，阻断socket和真实HTTP传输。"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.settings = Settings(project_root=self.root, data_dir=self.root / 'data',
            output_dir=self.root / 'notes', database_url='sqlite:///' + str(self.root / 'test.db'),
            max_model_requests=5, reserve_model_requests=0, retrieval_enabled=False,
            discovery_backends={}, deepseek_api_key=None)
        for target in ('socket.socket.connect', 'socket.socket.connect_ex',
                       'httpx.HTTPTransport.handle_request',
                       'httpx.AsyncHTTPTransport.handle_async_request'):
            blocker = patch(target, side_effect=AssertionError('零论文回归禁止网络与部署服务'))
            blocker.start()
            self.addCleanup(blocker.stop)
        self.db = Database(self.settings.database_url)
        self.db.ensure_schema()
        self.tasks = TaskService(self.db, self.settings)
        self.conversations = ConversationService(self.db, self.settings)
        self.coverage = CoverageService(self.db)
        self.memory = MemoryService(self.db)
        self.registry = ToolRegistry(delegation_specs() + memory_specs())
        self.prompt = ('合成算例：8条断言中7条正确、1条错误；删除2条正确断言。'
                       '固定基数错误负担以初始8条为分母，覆盖率也以8条为分母。'
                       '只用本条数据计算，不找论文、不精读、不保存。')

    def conversation(self) -> str:
        """创建仅属于当前临时数据库的会话。"""
        return self.conversations.create('零论文合成规划', uuid4().hex)['conversation_id']

    def task(self, prompt=None, *, cid=None, conversation=True) -> str:
        """通过正式服务新建根任务，默认每次独立会话且禁止保存。"""
        return self.tasks.create(TaskCreate(prompt=prompt or self.prompt,
            conversation_id=(cid or self.conversation()) if conversation else None,
            save_policy='deny', idempotency_key=uuid4().hex)).task_id

    @staticmethod
    def plan(kind='memory', *, quote='只用本条数据计算') -> dict:
        """给出来自当前用户原句的零论文目标。"""
        return {'goal': '按合成数据计算并保留评价条件', 'task_kind': kind,
                'goals': [{'goal_id': 'derive', 'description': '按本轮给定数据推导指标',
                    'origin': 'user', 'source_quote': quote, 'min_papers': 0}]}

    def execute_tool(self, task_id: str, name: str, arguments: dict):
        """执行原ToolSpec并保存实际观察，不能直接伪造成功记忆行动。"""
        context = ToolContext(task_id=task_id, save_policy=SavePolicy.deny,
            services={'coverage': self.coverage, 'memory': self.memory})
        result = self.registry.execute(name, arguments, context)
        sequence = len(self.db.list_actions(task_id)) + 1
        action = ActionRecord(task_id=task_id, sequence=sequence, tool_name=name,
            arguments=arguments, status='succeeded' if result.ok else 'failed',
            observation=result.model_dump(mode='json'))
        self.db.save_action(action.model_dump(mode='json'))
        return result

    @staticmethod
    def finish(answer='保留6条，其中1条错误；固定基数错误负担12.5%，覆盖率75%。') -> dict:
        """使用本套件合成数据，不复用私有pilot的答案或结论身份。"""
        return {'answer_markdown': answer,
                'goals': [{'goal_id': 'derive', 'status': 'satisfied', 'reason': '按用户给定数据完成推导'}]}

    def worker(self, task_id: str, plan: dict, answer: str) -> ZeroPaperProvider:
        """运行正式Worker链，只有供应商生成内容使用可控替身。"""
        provider = ZeroPaperProvider(plan, answer)
        executor = TaskExecutor(self.settings, self.db, provider_factory=lambda settings: provider,
            supervisor_tools={'get_profile', 'set_research_plan', 'finish_task'})
        result = executor.execute(task_id)
        self.assertEqual(self.db.get_task(task_id)['status'], 'completed', result)
        self.assertEqual(result.answer_markdown, answer)
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(len(self.db.list_model_calls(task_id)), 3)
        self.assertEqual([item['tool_name'] for item in self.db.list_actions(task_id)],
                         ['get_profile', 'set_research_plan', 'finish_task'])
        return provider

    def test_memory_routes_self_contained_zero_paper_derivation(self):
        """memory允许自足数据推导，实际记忆行动后零论文正常交付。"""
        task_id = self.task()
        answer = '保留6条，其中5条正确、1条错误；固定基数错误负担1/8=12.5%，覆盖率6/8=75%。'
        self.worker(task_id, self.plan(), answer)
        plan = self.coverage.plan(task_id)
        self.assertEqual(plan['task_kind'], 'memory')
        self.assertEqual(plan['goals'][0]['min_papers'], 0)
        for table in ('papers', 'documents', 'page_evidence', 'task_evidence', 'save_receipts'):
            self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM ' + table)['n'], 0)

    def test_same_conversation_update_reuses_history_without_new_papers(self):
        """后续同会话口径更新沿用实际历史，不以min_papers强求新论文。"""
        cid = self.conversation()
        first = self.task(cid=cid)
        first_answer = '当前保留6条、错误1条；固定基数错误负担为12.5%，覆盖率75%。'
        self.worker(first, self.plan(), first_answer)
        prompt = ('更新合成算例：错误指标改用当前保留集合为分母；覆盖率定义沿用上一轮。'
                  '不新增删除操作。只解释口径变化，不找论文、不精读、不保存。')
        second = self.task(prompt, cid=cid)
        second_plan = self.plan(quote='只解释口径变化')
        second_answer = '仍保留6条、错误1条；新错误率1/6约16.67%，覆盖率仍为75%。'
        provider = self.worker(second, second_plan, second_answer)
        headers = [message['content'] for message in provider.calls[0]['messages']
                   if message.get('role') == 'system'
                   and str(message.get('content', '')).startswith(PREFIX + '\n')]
        self.assertEqual(len(headers), 1)
        context = json.loads(headers[0].rsplit('\n', 1)[-1])
        self.assertTrue(any(item.get('task_id') == first and item.get('content') == first_answer
                            for item in context['messages']))
        self.assertEqual(self.db.get_task(second)['save_policy'], 'deny')
        self.assertEqual(self.db.fetchall('SELECT * FROM task_evidence'), [])
        self.assertEqual(self.db.fetchall('SELECT * FROM save_receipts'), [])

    def test_compare_search_survey_keep_required_material_goal(self):
        """三个研究类型的全零目标继续拒绝，不能静默降格成memory。"""
        task_id = self.task()
        for kind in ('compare', 'search', 'survey'):
            with self.subTest(task_kind=kind):
                with self.assertRaisesRegex(ValueError, 'min_papers>0'):
                    self.coverage.prepare_plan(task_id, ResearchBrief.model_validate(self.plan(kind)))
        self.assertIsNone(self.coverage.plan(task_id))

    def test_strategy_material_goal_cannot_replace_user_material_gate(self):
        """模型自拟的取材步骤不能替用户零材料目标承担研究材料门。"""
        task_id = self.task()
        for kind in ('compare', 'search', 'survey'):
            value = self.plan(kind)
            value['goals'].append({'goal_id': 'strategy_papers', 'description': '模型自拟找材料',
                                   'origin': 'strategy', 'min_papers': 2})
            with self.subTest(task_kind=kind), self.assertRaisesRegex(ValueError, 'min_papers>0'):
                self.coverage.prepare_plan(task_id, ResearchBrief.model_validate(value))

    def test_research_user_material_requirement_cannot_be_claimed_without_papers(self):
        """有真实user材料目标的计划可建立，但无材料不能凭strategy汇总交付完成。"""
        task_id = self.task('比较两篇论文，并给出解释，不保存。')
        value = self.plan('compare', quote='比较两篇论文')
        value['goals'][0]['min_papers'] = 2
        value['goals'].append({'goal_id': 'summary', 'description': '模型自拟汇总',
                              'origin': 'strategy', 'min_papers': 0})
        self.assertTrue(self.execute_tool(task_id, 'set_research_plan', value).ok)
        self.assertTrue(self.execute_tool(task_id, 'get_profile', {}).ok)
        finish = self.finish()
        finish['goals'].append({'goal_id': 'summary', 'status': 'satisfied', 'reason': '已整理'})
        result = self.execute_tool(task_id, 'finish_task', finish)
        self.assertFalse(result.ok)
        self.assertEqual(result.data['coverage_blocker']['goal_id'], 'derive')

    def test_memory_cannot_request_new_count_deep_reading_or_save(self):
        """memory不能绕过新检索数量、精读和生成保存的原研究入口。"""
        task_id = self.task()
        for constraint in ({'requested_count': 2}, {'deep_reading_requested': True}, {'save_requested': True}):
            value = {**self.plan(), **constraint}
            with self.subTest(constraint=constraint), self.assertRaisesRegex(ValueError, '不使用memory计划'):
                self.coverage.prepare_plan(task_id, ResearchBrief.model_validate(value))

    def test_memory_without_conversation_is_rejected(self):
        """没有会话身份的根不能使用memory计划。"""
        task_id = self.task(conversation=False)
        with self.assertRaisesRegex(ValueError, 'requires a conversation'):
            self.coverage.prepare_plan(task_id, ResearchBrief.model_validate(self.plan()))

    def test_zero_paper_type_does_not_weaken_verbatim_source_quote(self):
        """零论文不豁免用户来源校验，拼接两个原句的引句仍拒绝。"""
        task_id = self.task()
        value = self.plan(quote='只用本条数据计算……不保存')
        with self.assertRaisesRegex(ValueError, 'verbatim'):
            self.coverage.prepare_plan(task_id, ResearchBrief.model_validate(value))

    def test_model_save_policy_cannot_replace_actual_deny(self):
        """模型即使在计划中传allow，实际任务权限仍是deny。"""
        task_id = self.task()
        value = {**self.plan(), 'save_policy': 'allow'}
        result = self.execute_tool(task_id, 'set_research_plan', value)
        self.assertTrue(result.ok)
        self.assertEqual(result.data['research_plan']['save_policy'], 'deny')
        self.assertEqual(self.db.get_task(task_id)['save_policy'], 'deny')
        self.assertEqual(self.db.fetchall('SELECT * FROM save_receipts'), [])

    def test_zero_paper_plan_alone_cannot_claim_memory_completion(self):
        """只有规划而无成功记忆行动时，结束工具继续按既有合同拒绝。"""
        task_id = self.task()
        self.assertTrue(self.execute_tool(task_id, 'set_research_plan', self.plan()).ok)
        result = self.execute_tool(task_id, 'finish_task', self.finish())
        self.assertFalse(result.ok)
        self.assertIn('必须有真实记忆操作或读回', result.message)
        self.assertEqual([row['status'] for row in self.db.list_actions(task_id)], ['succeeded', 'failed'])

    def test_successful_profile_read_allows_zero_paper_finish(self):
        """真实执行get_profile并落行动后，同一结束合同允许零论文交付。"""
        task_id = self.task()
        self.assertTrue(self.execute_tool(task_id, 'set_research_plan', self.plan()).ok)
        profile = self.execute_tool(task_id, 'get_profile', {})
        self.assertTrue(profile.ok)
        self.assertIn('statistics', profile.data)
        result = self.execute_tool(task_id, 'finish_task', self.finish())
        self.assertTrue(result.ok, result.message)
        delivery = result.data['delivery']
        self.assertEqual(delivery['status'], 'completed')
        self.assertEqual(delivery['coverage']['paper_ids'], [])
        self.assertIsNone(delivery['saved_path'])
        self.assertEqual(self.db.fetchall('SELECT * FROM save_receipts'), [])


if __name__ == '__main__':
    unittest.main()
