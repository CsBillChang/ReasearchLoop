"""动态委派与冻结策略隔离验证：供应商和图可显式替身，数据库独立。"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from pydantic import BaseModel

from research_loop.agents.graph import build_graph, initial_state
from research_loop.agents.researcher import researcher_state
from research_loop.config import Settings
from research_loop.domain.execution_policy import graph_run_config, inherit_execution_policy, handoff_reservations
from research_loop.domain.tasks import Observation, TaskCreate
from research_loop.providers.deepseek import ModelResponse, ToolCall
from research_loop.services.tasks import TaskService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.mysql import Database
from research_loop.storage.token_budget import TokenBudgetStore
from research_loop.tools.base import ToolSpec, ToolContext
from research_loop.tools.delegation import DelegateResearchArgs, specs
from research_loop.tools.registry import ToolRegistry
from research_loop.workers.executor import TaskExecutor
from research_loop.domain.tasks import SavePolicy


class EmptyArgs(BaseModel):
    pass


class ExecutionPolicyQueueTests(unittest.TestCase):
    """默认SQLite；可通过显式专用测试URL复验真实MySQL条件更新。"""

    def setUp(self):
        """独立目录与身份，不读取生产环境文件或连接生产数据库。"""
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        url = os.getenv('EXECUTION_POLICY_TEST_DATABASE_URL') or 'sqlite:///' + str(root/'queue.db')
        if not url.startswith('sqlite:') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_'):
            raise ValueError('explicit dedicated test database required')
        self.settings = Settings(project_root=root, data_dir=root/'data', output_dir=root/'notes', database_url=url,
                                 retrieval_enabled=False)
        self.db = Database(url); self.db.ensure_schema()
        self.tasks = TaskService(self.db, self.settings)
        self.tid = self.tasks.create(TaskCreate(prompt='合成五个独立研究目标', idempotency_key=uuid4().hex)).task_id
        self.specs = [{'objective': '合成独立研究目标' + str(i), 'mode': 'investigate'} for i in range(5)]

    def test_five_subtasks_share_schema_and_handler_contract(self):
        """schema与执行入口都不把规划清单截成并发槽位数量。"""
        args = DelegateResearchArgs(subtasks=self.specs)
        self.assertEqual(len(args.subtasks), 5)
        schema = DelegateResearchArgs.model_json_schema()['properties']['subtasks']
        self.assertNotIn('maxItems', schema)
        delegation = Mock(return_value=[{'subtask_id': str(i)} for i in range(5)])
        result = ToolRegistry(specs()).execute('delegate_research', {'subtasks': self.specs},
            ToolContext(self.tid, SavePolicy.deny, role='supervisor', services={'delegation': delegation}))
        self.assertTrue(result.ok)
        self.assertEqual(len(delegation.call_args.args[1]), 5)

    def test_queue_runs_all_five_with_two_slots_and_inherits_root(self):
        """五项全部持久化/回传，实际并发为二，子任务共享根冻结策略。"""
        claimed = self.db.claim_task('fixture', task_id=self.tid)
        lock = threading.Lock(); active = 0; peak = 0; seen = []

        class Graph:
            """只替换图计算，沿用真实队列、租约、权限及结果持久化。"""
            def invoke(self, state, config):
                """延迟形成可观测重叠，返回合成部分结果且保留每项缺口。"""
                nonlocal active, peak
                with lock:
                    active += 1; peak = max(peak, active); seen.append(deepcopy(state))
                time.sleep(0.035)
                with lock:
                    active -= 1
                return {**state, 'final_answer': state['prompt'], 'final_reason': 'fixture partial',
                        'structured_result': {'status': 'partial', 'findings': [state['prompt']], 'gaps': ['合成范围缺口']},
                        'gaps': ['合成范围缺口']}

        executor = TaskExecutor(self.settings, self.db, lambda _: Mock())
        with patch('research_loop.workers.executor.build_researcher_graph', return_value=Graph()):
            result = executor._run_research_batch(self.tid, self.specs, 'fixture', claimed['lease_token'], self.settings, Mock())
        self.assertEqual(peak, 2)
        self.assertEqual(len(result), 5)
        self.assertEqual(len(self.db.list_subtasks(self.tid)), 5)
        self.assertEqual([r['objective'] for r in result], [s['objective'] for s in self.specs])
        self.assertTrue(all(r['findings'] and r['gaps'] and r['status'] == 'partial' for r in result))
        for state in seen:
            self.assertEqual(state['execution_policy'], self.db.execution_policy(self.tid))
            self.assertEqual((state['max_model_requests'], state['max_tool_actions']), (48, 80))
        with patch('research_loop.workers.executor.build_researcher_graph', side_effect=AssertionError('completed child replayed')):
            replay = executor._run_research_batch(self.tid, self.specs, 'fixture', claimed['lease_token'], self.settings, Mock())
        self.assertEqual(replay, result)

    def test_observe_counts_past_quotas_and_unknown_still_blocks(self):
        """监测模式越过累计额度仍记账，unknown不得借此重发。"""
        settings = self.settings.model_copy(update={'execution_budget_mode': 'observe', 'max_model_requests': 1,
            'max_tool_actions': 1, 'reserve_model_requests': 0, 'task_token_limit': 1, 'task_token_tail': 0})
        tid = TaskService(self.db, settings).create(TaskCreate(prompt='合成监测模式', idempotency_key=uuid4().hex)).task_id
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertTrue(all(pool.map(lambda _: self.db.reserve_model_slot(tid, 'researcher'), range(8))))
            self.assertTrue(all(pool.map(lambda _: self.db.reserve_tool_slot(tid, 50), range(8))))
        tokens = TokenBudgetStore(self.db)
        request = {'messages': [{'role': 'user', 'content': '合成材料'}], 'tools': [], 'max_tokens': 8192}
        tokens.reserve(tid, tid, 1, request, 'supervisor')
        self.assertGreater(tokens.summary(tid)['reserved_tokens'], 1)
        self.db.save_model_call(tid, 1, 'unknown', request, sent_state='sent')
        with self.assertRaisesRegex(ValueError, 'UNKNOWN'):
            tokens.reserve(tid, tid, 2, request, 'supervisor')
        self.assertEqual(self.db.budget_summary(tid)['model_used'], 8)
        self.assertEqual(self.db.budget_summary(tid)['tool_used'], 8)

    def test_observe_graph_surpasses_previous_recursion_and_local_caps(self):
        """超过旧264次图步仍可正常结束；取消继续优先。"""
        state = initial_state('pure-observe', '合成循环', 'deny', max_model_requests=1, max_tool_actions=1)
        state['execution_policy'] = {'mode': 'observe'}
        class Provider:
            def __init__(self):
                self.calls = 0
            def complete(self, messages, tools, force_text=False):
                self.calls += 1
                if self.calls > 135:
                    return ModelResponse(content='合成结束', tool_calls=[])
                return ModelResponse(content=None, tool_calls=[ToolCall(str(self.calls), 'tick', {})])
        provider = Provider()
        registry = ToolRegistry([ToolSpec('tick', '合成进展', EmptyArgs,
            lambda args, ctx: Observation(ok=True, kind='success', data={'tick': True}))])
        result = build_graph(provider, registry, {}).invoke(state, config=graph_run_config(state))
        self.assertEqual(provider.calls, 136)
        self.assertEqual(result['tool_actions'], 135)
        state['cancelled'] = True
        build_graph(provider, registry, {}).invoke(state, config=graph_run_config(state))
        self.assertEqual(provider.calls, 136)

    def test_existing_checkpoint_and_root_policy_remain_frozen(self):
        """变更新配置不改旧快照、请求JSON及根持久策略。"""
        old = researcher_state('legacy', '旧子任务')
        old.update(max_model_requests=12, max_tool_actions=20)
        CheckpointStore(self.db).save('legacy', old)
        changed = self.settings.model_copy(update={'execution_budget_mode': 'observe', 'max_model_requests': 99})
        existing = self.db.execution_policy(self.tid)
        TaskService(self.db, changed).create(TaskCreate(prompt='新任务', idempotency_key=uuid4().hex))
        self.assertEqual(CheckpointStore(self.db).load('legacy'), old)
        self.assertNotIn('execution_policy', CheckpointStore(self.db).load('legacy'))
        self.assertEqual(self.db.execution_policy(self.tid), existing)
        new = inherit_execution_policy(researcher_state('new', '新子任务'), existing)
        self.assertEqual(new['max_model_requests'], 48)
        self.assertEqual(handoff_reservations('researcher', False, 5), 6)
        self.assertEqual(handoff_reservations('researcher', True, 5), 5)

    def test_reviewer_uses_role_context_and_frozen_output(self):
        """Reviewer同样先准备上下文，输出预留来自冻结准备结果。"""
        state = inherit_execution_policy(initial_state(self.tid, '合成核验', 'deny'), self.db.execution_policy(self.tid))
        state['role'] = 'reviewer'
        context = Mock()
        context.settings = self.settings
        context.prepare_role.side_effect = lambda state, messages, tools: (messages, {'output_reserve_tokens': 111})
        provider = Mock(spec=['complete'])
        provider.complete.return_value = ModelResponse(content='合成核验范围说明', tool_calls=[], usage={'total_tokens': 2})
        result = build_graph(provider, ToolRegistry([]), {'context': context}, self.db).invoke(state, config=graph_run_config(state))
        context.prepare_role.assert_called_once()
        context.prepare.assert_not_called()
        request = json.loads(self.db.get_model_call(self.tid, 1)['request_json'])
        self.assertEqual(request['max_tokens'], 111)
        self.assertTrue(result['finished'])

    def test_frozen_unsent_receipt_rejects_model_change_without_rewrite(self):
        """恢复尚未发送请求也核对根模型，历史参数不改写、供应商不发送。"""
        state = inherit_execution_policy(initial_state(self.tid, '合成恢复', 'deny'), self.db.execution_policy(self.tid))
        request = {'messages': state['messages'], 'tools': [], 'force_text': False, 'max_tokens': 111}
        self.db.begin_model_call(self.tid, 1, request)
        before = self.db.get_model_call(self.tid, 1)['request_json']
        context = Mock()
        context.settings = self.settings.model_copy(update={'deepseek_model': 'different-model'})
        provider = Mock(spec=['complete'])
        result = build_graph(provider, ToolRegistry([]), {'context': context}, self.db).invoke(state, config=graph_run_config(state))
        self.assertIn('TOKEN_POLICY_MODEL_CHANGED', result['final_reason'])
        provider.complete.assert_not_called()
        self.assertEqual(self.db.get_model_call(self.tid, 1)['request_json'], before)

    def test_success_receipt_replays_even_after_model_configuration_change(self):
        """成功收据只读回放，不重新计量发送，也不改历史请求JSON。"""
        state = inherit_execution_policy(initial_state(self.tid, '合成已完成响应', 'deny'), self.db.execution_policy(self.tid))
        request = {'messages': state['messages'], 'tools': [], 'force_text': False, 'max_tokens': 111}
        raw = {'choices': [{'message': {'role': 'assistant', 'content': '历史合成成果'}, 'finish_reason': 'stop'}]}
        self.db.save_model_call(self.tid, 1, 'succeeded', request, response=raw, usage={'total_tokens': 2}, sent_state='sent')
        before = self.db.get_model_call(self.tid, 1)['request_json']
        context = Mock()
        context.settings = self.settings.model_copy(update={'deepseek_model': 'different-model'})
        provider = Mock(spec=['complete'])
        result = build_graph(provider, ToolRegistry([]), {'context': context}, self.db).invoke(state, config=graph_run_config(state))
        self.assertIn('历史合成成果', result['final_answer'])
        provider.complete.assert_not_called()
        self.assertEqual(self.db.get_model_call(self.tid, 1)['request_json'], before)

    def test_invalid_fifth_scope_leaves_no_partial_batch(self):
        """先验证整批授权，第五项越权不能留下前四项半成品。"""
        claimed = self.db.claim_task('scope-fixture', task_id=self.tid)
        executor = TaskExecutor(self.settings, self.db, lambda _: self.fail('invalid batch constructed provider'))
        invalid = deepcopy(self.specs)
        invalid[-1]['paper_ids'] = ['unauthorized-paper']
        with self.assertRaises(PermissionError):
            executor._run_research_batch(self.tid, invalid, 'scope-fixture', claimed['lease_token'], self.settings, Mock())
        self.assertEqual(self.db.list_subtasks(self.tid), [])

    def test_stale_cancelled_batch_cannot_create_or_commit_children(self):
        """根租约转移后，旧Worker不能借取消分支创建或结算子任务。"""
        claimed = self.db.claim_task('stale', task_id=self.tid)
        self.db.update_task(self.tid, claimed_by='replacement', lease_token='replacement-token', cancel_requested=1)
        executor = TaskExecutor(self.settings, self.db, lambda _: self.fail('stale queue created provider'))
        with self.assertRaisesRegex(PermissionError, 'lease lost'):
            executor._run_research_batch(self.tid, self.specs, 'stale', claimed['lease_token'], self.settings, Mock())
        self.assertEqual(self.db.list_subtasks(self.tid), [])

    def test_paid_request_callback_fences_lease_transfer_before_send(self):
        """网页收据已占位但租约转移时，真实图传入的回调阻止付费发送。"""
        from research_loop.storage.web_requests import WebRequestStore
        claimed = self.db.claim_task('lease-fixture', task_id=self.tid)
        state = inherit_execution_policy(initial_state(self.tid, '合成网页工具', 'deny'), self.db.execution_policy(self.tid))
        state.update(worker_id='lease-fixture', lease_token=claimed['lease_token'],
            pending_tool_calls=[{'call_id': 'paid', 'name': 'paid', 'arguments': {}}], pending_tool_scope=['paid'])
        state['messages'].append({'role': 'assistant', 'tool_calls': [{'id': 'paid', 'type': 'function',
            'function': {'name': 'paid', 'arguments': '{}'}}]})
        sent = []
        def handler(args, context):
            """仅模拟发送窗口，不访问真实HTTP端点。"""
            checks = 0
            def active():
                """第二次检查前转移租约，模拟排队后失权。"""
                nonlocal checks
                checks += 1
                if checks == 2:
                    self.db.update_task(self.tid, claimed_by='replacement', lease_token='replacement-token')
                return not context.cancelled()
            WebRequestStore(self.db).run(self.tid, 'search', {'query': 'fixture'}, lambda: sent.append(True), active)
            return Observation(ok=True, kind='success')
        registry = ToolRegistry([ToolSpec('paid', '合成独立付费工具', EmptyArgs, handler)])
        with self.assertRaisesRegex(RuntimeError, 'lease lost'):
            build_graph(None, registry, {}, self.db).nodes['execute_tools'].invoke(state)
        self.assertEqual(sent, [])
        self.assertEqual(self.db.fetchone('SELECT status FROM web_request_receipts WHERE root_task_id=?', (self.tid,))['status'], 'not_sent')

    def test_paid_response_receipt_survives_lease_loss_after_send(self):
        """已发送成功响应仍持久为事实，租约失效只拒绝图结果写入。"""
        from research_loop.storage.web_requests import WebRequestStore
        claimed = self.db.claim_task('lease-response', task_id=self.tid)
        state = inherit_execution_policy(initial_state(self.tid, '合成网页返回', 'deny'), self.db.execution_policy(self.tid))
        state.update(worker_id='lease-response', lease_token=claimed['lease_token'],
            pending_tool_calls=[{'call_id': 'paid', 'name': 'paid', 'arguments': {}}], pending_tool_scope=['paid'])
        state['messages'].append({'role': 'assistant', 'tool_calls': [{'id': 'paid', 'type': 'function',
            'function': {'name': 'paid', 'arguments': '{}'}}]})
        sent = []
        def handler(args, context):
            """模拟供应商响应抵达时租约已转移。"""
            def send():
                """本函数仅返回合成JSON。"""
                sent.append(True)
                self.db.update_task(self.tid, claimed_by='replacement', lease_token='replacement-token')
                return {'results': []}
            WebRequestStore(self.db).run(self.tid, 'search', {'query': 'fixture-response'}, send, lambda: not context.cancelled())
            return Observation(ok=True, kind='success')
        registry = ToolRegistry([ToolSpec('paid', '合成独立付费工具', EmptyArgs, handler)])
        with self.assertRaisesRegex(RuntimeError, 'lease lost'):
            build_graph(None, registry, {}, self.db).nodes['execute_tools'].invoke(state)
        self.assertEqual(sent, [True])
        self.assertEqual(self.db.fetchone('SELECT status FROM web_request_receipts WHERE root_task_id=?', (self.tid,))['status'], 'completed')

    def test_observe_repeated_invalid_arguments_stop_without_numeric_quota(self):
        """相同非法参数或重试失败没有新进展时停止，更正仍可成功。"""
        from pydantic import Field
        class RequiredArgs(BaseModel):
            value: int = Field(ge=1)
        for arguments in [{'_invalid_json': '{'}, {}, {'value': 1}]:
            with self.subTest(arguments=arguments):
                state = initial_state('invalid-observe', '合成错误恢复', 'deny')
                state['execution_policy'] = {'mode': 'observe'}
                provider = Mock(spec=['complete'])
                provider.complete.return_value = ModelResponse(content=None, tool_calls=[ToolCall('bad', 'tick', arguments)])
                registry = ToolRegistry([ToolSpec('tick', '合成重试失败', RequiredArgs,
                    lambda args, ctx: Observation(ok=False, kind='execution_error', retryable=True, message='合成暂时错误'))])
                result = build_graph(provider, registry, {}).invoke(state, config=graph_run_config(state))
                self.assertEqual(provider.complete.call_count, 2)
                self.assertTrue(result['finished'])
                self.assertIn('repeated', result['final_reason'])
        state = initial_state('corrected-observe', '合成参数更正', 'deny')
        state['execution_policy'] = {'mode': 'observe'}
        provider = Mock(spec=['complete'])
        provider.complete.side_effect = [ModelResponse(content=None, tool_calls=[ToolCall('bad', 'tick', {})]),
            ModelResponse(content=None, tool_calls=[ToolCall('good', 'tick', {'value': 1})]),
            ModelResponse(content='参数已更正', tool_calls=[])]
        registry = ToolRegistry([ToolSpec('tick', '合成更正', RequiredArgs,
            lambda args, ctx: Observation(ok=True, kind='success'))])
        result = build_graph(provider, registry, {}).invoke(state, config=graph_run_config(state))
        self.assertEqual(result['final_answer'], '参数已更正')

    def test_cancelled_queue_never_constructs_provider(self):
        """已取消队列不会被线程重新启动，全部子结果仍可回传。"""
        claimed = self.db.claim_task('cancel-fixture', task_id=self.tid)
        self.tasks.cancel(self.tid)
        executor = TaskExecutor(self.settings, self.db, lambda _: self.fail('cancelled queue created provider'))
        results = executor._run_research_batch(self.tid, self.specs, 'cancel-fixture', claimed['lease_token'], self.settings, Mock())
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r['status'] == 'cancelled' for r in results))
