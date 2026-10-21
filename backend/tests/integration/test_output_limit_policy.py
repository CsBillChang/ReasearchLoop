"""普通角色输出策略的隔离回归；只用临时SQLite和合成HTTP，不调用真实模型。"""
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from pydantic import BaseModel

from research_loop.agents.graph import build_graph, initial_state
from research_loop.agents.memory import ContextService
from research_loop.config import Settings
from research_loop.domain.execution_policy import graph_run_config, inherit_execution_policy
from research_loop.domain.tasks import ActionRecord, Observation, TaskStatus
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.services.tasks import TaskService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.mysql import Database
from research_loop.storage.token_budget import TokenBudgetStore
from research_loop.tools.base import ToolSpec
from research_loop.tools.registry import ToolRegistry
from research_loop.workers.executor import TaskExecutor


class EmptyArgs(BaseModel):
    pass


class OutputLimitPolicyTests(unittest.TestCase):
    """检查真实图、收据和Worker交付函数；不启动服务或连接生产数据库。"""

    def setUp(self):
        """每例创建独立临时目录，网络发送只能进入MockTransport。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = Settings(project_root=self.root, data_dir=self.root / 'data',
                                 output_dir=self.root / 'notes', deepseek_model='deepseek-flash',
                                 retrieval_enabled=False)
        self.db = Database('sqlite:///' + str(self.root / 'output.db'))
        self.db.ensure_schema()
        blocker = patch.object(socket, 'create_connection', side_effect=AssertionError('network prohibited'))
        blocker.start()
        self.addCleanup(blocker.stop)
        self.sent = []
        self.executed = []
        self.registry = ToolRegistry([ToolSpec('save_note', '合成保存探针', EmptyArgs, self.tool)])

    def tool(self, args, context):
        """只记录是否进入工具；不允许写入任何笔记文件。"""
        self.executed.append(context.task_id)
        return Observation(ok=True, kind='success', data={'synthetic': True})

    def task(self, settings=None, role='supervisor'):
        """为每个合成情形冻结根策略，避免子例互相改写账本。"""
        tid = uuid4().hex
        self.db.create_task(tid, None, 'synthetic', '合成只读研究', 'deny', [],
                            role=role, token_settings=settings or self.settings)
        state = inherit_execution_policy(initial_state(tid, '合成只读研究', 'deny'), self.db.execution_policy(tid))
        state['role'] = role
        return state

    def raw(self, content='', *, finish='length', completion=65536, reasoning=65536, calls=None):
        """构造独立合成响应形状，不复制或公开原失败的内部推理。"""
        message = {'role': 'assistant', 'content': content, 'reasoning_content': 'PRIVATE_SYNTHETIC_REASONING'}
        if calls:
            message['tool_calls'] = calls
        return {'id': 'synthetic', 'model': 'deepseek-flash',
                'choices': [{'index': 0, 'finish_reason': finish, 'message': message}],
                'usage': {'prompt_tokens': 31, 'completion_tokens': completion,
                          'completion_tokens_details': {'reasoning_tokens': reasoning},
                          'total_tokens': 31 + completion}}

    def provider(self, raw):
        """通过真实适配器装配合成流或JSON，并记录实际wire参数。"""
        def handle(request):
            """只响应内存HTTP，不访问供应商域名。"""
            payload = json.loads(request.content)
            self.sent.append(payload)
            if not payload.get('stream'):
                return httpx.Response(200, json=raw)
            choice = raw['choices'][0]
            delta = deepcopy(choice['message'])
            for index, call in enumerate(delta.get('tool_calls') or []):
                call['index'] = index
            chunk = {k: raw[k] for k in ('id', 'model', 'usage') if k in raw}
            chunk['choices'] = [{'index': 0, 'delta': delta, 'finish_reason': choice['finish_reason']}]
            content = 'data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n'
            return httpx.Response(200, text=content, headers={'content-type': 'text/event-stream'})
        client = httpx.Client(transport=httpx.MockTransport(handle))
        self.addCleanup(client.close)
        return DeepSeekProvider('synthetic-key', 'deepseek-flash', client=client)

    def run_graph(self, state, raw, *, context=True, settings=None, streaming=True):
        """执行实际图与收据持久化；上下文可关闭以验证备用路径。"""
        services = {'stream_public': streaming}
        if context:
            services['context'] = ContextService(self.db, None, settings or self.settings)
        return build_graph(self.provider(raw), self.registry, services, self.db,
                           checkpoint_store=CheckpointStore(self.db)).invoke(state, config=graph_run_config(state))

    def delivery(self, state):
        """使用Worker原交付与终态函数写回合成任务，不启动Worker循环。"""
        executor = TaskExecutor(self.settings, self.db)
        result = executor._deliverable_result(state['task_id'], state)
        status = executor._delivery_status(state['task_id'], state, result)
        TaskService(self.db, self.settings).complete(state['task_id'], status, result)
        return result, status

    def assert_private(self, state, result):
        """内部推理可以保留收据，但不得进入公开结果或SSE事件。"""
        public = result.model_dump_json() + json.dumps(self.db.list_events(state['task_id']))
        self.assertNotIn('PRIVATE_SYNTHETIC_REASONING', public)
        self.assertNotIn('PRIVATE_SYNTHETIC_REASONING', state.get('final_answer') or '')

    def test_new_roles_use_same_context_reservation_and_wire_cap(self):
        """三角色均采用根64K，流式/非流式请求与输入容量、账本一致。"""
        for role in ('supervisor', 'researcher', 'reviewer'):
            for streaming in (True, False):
                with self.subTest(role=role, streaming=streaming):
                    state = self.task(role=role)
                    result = self.run_graph(state, self.raw('合成完整正文', finish='stop', completion=12, reasoning=2),
                                            streaming=streaming)
                    self.assertEqual(self.sent[-1]['max_tokens'], 65536)
                    self.assertEqual(result['context_metrics']['output_reserve_tokens'], 65536)
                    self.assertEqual(result['context_metrics']['input_capacity'], 1000000 - 65536 - 4096)
                    reservation = self.db.fetchone('SELECT * FROM token_reservations WHERE task_id=?', (state['task_id'],))
                    self.assertEqual(reservation['reserved_tokens'], result['context_metrics']['estimated_input_tokens'] + 65536)
                    self.assertEqual(TokenBudgetStore(self.db).summary(state['task_id'])['used_tokens'], 43)

    def test_no_context_uses_frozen_cap_and_matrix_keeps_shared_reserve(self):
        """备用路径读取根值；matrix disabled例外继续共享同一64K预留。"""
        for reserve in (65536, 8192, 1024):
            for matrix in (False, True):
                with self.subTest(reserve=reserve, matrix=matrix):
                    state = self.task(self.settings.model_copy(update={'context_output_reserve': reserve}))
                    if matrix:
                        state.update(research_plan={'evidence_matrix_requested': True}, evidence_ids=['synthetic-page'])
                    self.run_graph(state, self.raw('合成正文', finish='stop', completion=7, reasoning=1), context=False)
                    self.assertEqual(self.sent[-1]['max_tokens'], reserve)
                    self.assertEqual(self.sent[-1].get('thinking'), {'type': 'disabled'} if matrix else None)

    def test_old_frozen_and_legacy_roots_keep_previous_default(self):
        """冻结8192和无计量策略老根均不被新默认改为64K，显式当前值仍优先。"""
        for legacy in (False, True):
            for explicit in (False, True):
                with self.subTest(legacy=legacy, explicit=explicit):
                    state = self.task(self.settings.model_copy(update={'context_output_reserve': 8192}))
                    if legacy:
                        self.db.execute('DELETE FROM token_counting_policies WHERE root_task_id=?', (state['task_id'],))
                    current = self.settings.model_copy(update={'context_output_reserve': 2048}) if explicit else self.settings
                    result = self.run_graph(state, self.raw('合成正文', finish='stop', completion=7, reasoning=1), settings=current)
                    expected = 2048 if legacy and explicit else 8192
                    self.assertEqual(self.sent[-1]['max_tokens'], expected)
                    self.assertEqual(result['context_metrics']['output_reserve_tokens'], expected)
                    self.assertEqual(self.settings.context_output_reserve, 65536)
                    self.assertNotIn('context_output_reserve', self.settings.model_fields_set)

    def test_reasoning_only_limit_is_failed_without_new_body(self):
        """原失败形状在两发送路径明确归因，完整usage保持且不发布推理。"""
        for streaming in (True, False):
            with self.subTest(streaming=streaming):
                state = self.task(self.settings.model_copy(update={'context_output_reserve': 8192}))
                raw = self.raw(completion=8192, reasoning=8192)
                before = len(self.sent)
                final = self.run_graph(state, raw, streaming=streaming)
                result, status = self.delivery(final)
                self.assertEqual(len(self.sent) - before, 1)
                self.assertTrue(final['final_reason'].startswith('OUTPUT_LIMIT_REASONING_ONLY:'))
                self.assertEqual(status, TaskStatus.failed)
                self.assertEqual(result.answer_kind, 'failure_report')
                receipt = self.db.get_model_call(state['task_id'], 1)
                self.assertEqual(receipt['status'], 'succeeded')
                self.assertEqual(json.loads(receipt['usage_json']), raw['usage'])
                self.assertEqual(TokenBudgetStore(self.db).summary(state['task_id'])['used_tokens'], 8223)
                self.assert_private(final, result)

    def test_partial_body_is_retained_and_marked_partial(self):
        """部分正文逐字保留，不能用length报正常完成。"""
        for streaming in (True, False):
            with self.subTest(streaming=streaming):
                state = self.task()
                final = self.run_graph(state, self.raw('已生成但未完成的合成正文', reasoning=65000), streaming=streaming)
                result, status = self.delivery(final)
                self.assertEqual(result.answer_body_markdown, '已生成但未完成的合成正文')
                self.assertTrue(final['final_reason'].startswith('OUTPUT_LIMIT_PARTIAL_CONTENT:'))
                self.assertEqual(status, TaskStatus.partial)
                self.assert_private(final, result)

    def test_truncated_tools_and_same_batch_valid_tools_never_execute(self):
        """截断工具参数与同批完整动作全部阻断，正文保留而保存没有副作用。"""
        calls = [{'id': 'bad', 'type': 'function', 'function': {'name': 'save_note', 'arguments': '{"markdown":"unfinished'}},
                 {'id': 'complete', 'type': 'function', 'function': {'name': 'save_note', 'arguments': '{}'}}]
        for streaming in (True, False):
            with self.subTest(streaming=streaming):
                state = self.task()
                before = len(self.sent)
                final = self.run_graph(state, self.raw('已有独立合成正文', calls=calls), streaming=streaming)
                result, status = self.delivery(final)
                self.assertEqual(len(self.sent) - before, 1)
                self.assertEqual(self.executed, [])
                self.assertEqual(self.db.list_actions(state['task_id']), [])
                self.assertEqual(final.get('pending_tool_calls'), [])
                self.assertTrue(final['final_reason'].startswith('OUTPUT_LIMIT_TOOL_CALL:'))
                self.assertEqual(result.answer_body_markdown, '已有独立合成正文')
                self.assertEqual(status, TaskStatus.partial)
                self.assertIsNone(result.saved_path)
                self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM drafts')['n'], 0)
                self.assertFalse(self.settings.output_dir.exists())
                self.assert_private(final, result)

    def test_existing_body_and_stage_findings_survive_reasoning_limit(self):
        """空正文截断不吞先前正文，阶段成果仍由Worker原权威记录读回。"""
        for stage in (False, True):
            with self.subTest(stage=stage):
                state = self.task()
                if stage:
                    progress = {'revision': 1, 'findings': [{'finding_id': 'synthetic', 'text': '既有阶段发现',
                                'sources': [], 'verification_status': 'unreviewed'}], 'gaps': ['原有材料缺口']}
                    action = ActionRecord(task_id=state['task_id'], sequence=1,
                                          tool_name='record_research_progress', arguments={}).model_dump()
                    action.update(status='succeeded', observation={'ok': True, 'data': {'research_progress': progress}})
                    self.db.save_action(action)
                else:
                    state['final_answer'] = '既有正文'
                final = self.run_graph(state, self.raw())
                result, status = self.delivery(final)
                self.assertIn('既有阶段发现' if stage else '既有正文', result.answer_body_markdown)
                self.assertEqual(status, TaskStatus.partial)
                self.assert_private(final, result)

    def test_old_success_receipt_replays_length_without_send_or_rewrite(self):
        """旧8192成功收据重放不补64K或重发；已保存usage及原响应保持。"""
        state = self.task()
        request = {'messages': state['messages'], 'tools': [], 'force_text': False, 'max_tokens': 8192}
        raw = self.raw(completion=8192, reasoning=8192)
        self.db.save_model_call(state['task_id'], 1, 'succeeded', request, response=raw,
                                usage=raw['usage'], finish_reason='length', sent_state='sent')
        before = deepcopy(self.db.get_model_call(state['task_id'], 1))
        final = self.run_graph(state, self.raw('不得发送'), settings=self.settings.model_copy(update={'deepseek_model': 'other-model'}))
        self.assertEqual(self.sent, [])
        self.assertTrue(final['final_reason'].startswith('OUTPUT_LIMIT_REASONING_ONLY:'))
        self.assertEqual(self.db.get_model_call(state['task_id'], 1), before)

    def test_old_pending_checkpoint_with_length_receipt_cannot_execute_or_resend(self):
        """旧图已挂起的截断工具恢复时仍读原收据终止，不执行同批完整动作。"""
        for already_executed in (False, True):
            with self.subTest(already_executed=already_executed):
                state = self.task()
                request = {'messages': state['messages'], 'tools': self.registry.schemas('supervisor'),
                           'force_text': False, 'max_tokens': 8192}
                calls = [{'id': 'half', 'type': 'function', 'function': {'name': 'save_note', 'arguments': '{'}},
                         {'id': 'complete', 'type': 'function', 'function': {'name': 'save_note', 'arguments': '{}'}}]
                raw = self.raw('旧截断正文', completion=8192, reasoning=8000, calls=calls)
                self.db.save_model_call(state['task_id'], 1, 'succeeded', request, response=raw,
                                        usage=raw['usage'], finish_reason='length', sent_state='sent')
                before = deepcopy(self.db.get_model_call(state['task_id'], 1))
                state.update(model_requests=1, pending_tool_scope=['save_note'], pending_tool_calls=[
                    {'call_id': 'half', 'name': 'save_note', 'arguments': {'_invalid_json': '{'}},
                    {'call_id': 'complete', 'name': 'save_note', 'arguments': {}}])
                if already_executed:
                    old_action = ActionRecord(task_id=state['task_id'], sequence=1,
                                              tool_name='save_note', arguments={}).model_dump()
                    old_action.update(status='succeeded', observation={'ok': True, 'data': {'prior_artifact': 'keep'}})
                    self.db.save_action(old_action)
                    state.update(tool_actions=1, action_sequence=1)
                old_actions = deepcopy(self.db.list_actions(state['task_id']))
                state['messages'].append({'role': 'assistant', 'content': '旧截断正文', 'tool_calls': calls})
                final = self.run_graph(state, self.raw('不应发送'))
                result, status = self.delivery(final)
                self.assertTrue(final['final_reason'].startswith('OUTPUT_LIMIT_TOOL_CALL:'))
                self.assertEqual(result.answer_body_markdown, '旧截断正文')
                self.assertEqual(status, TaskStatus.partial)
                self.assertEqual(final['model_requests'], 1)
                self.assertEqual(final['pending_tool_calls'], [])
                self.assertEqual(self.sent, [])
                self.assertEqual(self.executed, [])
                self.assertEqual(self.db.list_actions(state['task_id']), old_actions)
                self.assertNotIn('本次工具均未执行', final['final_reason'])
                self.assertIn('待执行工具已阻止', final['final_reason'])
                self.assertEqual(self.db.get_model_call(state['task_id'], 1), before)
                self.assert_private(final, result)

    def test_frozen_unsent_request_keeps_explicit_cap(self):
        """已持久但未发请求仍按当时输出上限发送。"""
        state = self.task()
        request = {'messages': state['messages'], 'tools': [], 'force_text': False, 'max_tokens': 1024}
        self.db.begin_model_call(state['task_id'], 1, request)
        self.run_graph(state, self.raw('合成正文', finish='stop', completion=7, reasoning=1))
        self.assertEqual(self.sent[-1]['max_tokens'], 1024)
        self.assertEqual(json.loads(self.db.get_model_call(state['task_id'], 1)['request_json']), request)

    def test_unknown_still_blocks_without_inventing_usage(self):
        """旧unknown不重发、不补usage为零，原请求保持。"""
        state = self.task()
        request = {'messages': state['messages'], 'tools': [], 'max_tokens': 8192}
        self.db.save_model_call(state['task_id'], 1, 'unknown', request, sent_state='sent')
        before = deepcopy(self.db.get_model_call(state['task_id'], 1))
        final = self.run_graph(state, self.raw())
        self.assertIn('unknown', final['final_reason'])
        self.assertEqual(self.sent, [])
        self.assertEqual(self.db.get_model_call(state['task_id'], 1), before)

    def test_length_without_exhausted_cap_does_not_claim_output_exhaustion(self):
        """length来源不全时仅报告长度截断；bool或缺usage不伪造输出耗尽。"""
        cases = [(12, 12), (1, True), (True, True), (0, 0)]
        for completion, reasoning in cases:
            with self.subTest(completion=completion, reasoning=reasoning):
                state = self.task()
                raw = self.raw(completion=completion, reasoning=reasoning)
                final = self.run_graph(state, raw, streaming=False)
                self.assertTrue(final['final_reason'].startswith('MODEL_LENGTH_'))
                if type(reasoning) is not int or type(completion) is not int or completion == 0:
                    self.assertTrue(final['final_reason'].startswith('MODEL_LENGTH_NO_CONTENT:'))
                self.assertNotIn('模型达到单请求输出上限', final['final_reason'])
        state = self.task()
        raw = self.raw()
        raw.pop('usage')
        final = self.run_graph(state, raw, streaming=False)
        self.assertTrue(final['final_reason'].startswith('MODEL_LENGTH_NO_CONTENT:'))

    def test_normal_stop_and_complete_tool_call_remain_valid(self):
        """正常stop可完成，正常工具调用仍执行；length保护不吞合法动作。"""
        state = self.task()
        final = self.run_graph(state, self.raw('完整合成正文', finish='stop', completion=8, reasoning=1))
        result, status = self.delivery(final)
        self.assertEqual(final['final_reason'], 'model finished')
        self.assertEqual(status, TaskStatus.completed)
        self.assert_private(final, result)
        state = self.task()
        calls = [{'id': 'ok', 'type': 'function', 'function': {'name': 'save_note', 'arguments': '{}'}}]
        raw = self.raw(finish='tool_calls', calls=calls, completion=8, reasoning=1)
        graph = build_graph(self.provider(raw), self.registry, {}, self.db)
        decided = graph.nodes['decide'].invoke(state)
        graph.nodes['execute_tools'].invoke({**state, **decided})
        self.assertEqual(self.executed, [state['task_id']])
