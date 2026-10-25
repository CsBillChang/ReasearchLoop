"""接续记忆的隔离 TaskService/Worker 链；只使用合成对话与模型替身。"""
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
from research_loop.domain.memory import RememberInput
from research_loop.domain.tasks import TaskCreate
from research_loop.services.continuation_memory import ContinuationMemoryService, PROJECTION
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.services.tasks import TaskService
from research_loop.storage.memory import dump
from research_loop.storage.mysql import Database
from research_loop.tools.delegation import specs as delegation_specs
from research_loop.workers.executor import TaskExecutor


OMITTED = object()


def memory_context(messages: list[dict]) -> dict:
    """读取实际模型请求中的 Memory 头，不从数据库替模型补上下文。"""
    for message in messages:
        content = message.get('content')
        if (message.get('role') == 'system' and isinstance(content, str)
                and content.startswith(PREFIX + '\n')):
            return json.loads(content.rsplit('\n', 1)[-1])
    return {}


class ContinuationProvider:
    """替身通过正式工具响应提交摘要，记录实际 schema 与待发上下文。"""

    def __init__(self, answer: str, candidate=OMITTED, *, read_request=None, after_read=None):
        """接收固定回答与候选，候选回调可以检查模型实际可见历史。"""
        self.answer = answer
        self.candidate = candidate
        self.calls = []
        self.read_request = read_request
        self.after_read = after_read
        self.read_callback_done = False
        self.expected_calls = 4 if read_request else 3

    def close(self):
        """替身不持有网络资源。"""

    def send(self, messages, tools, **kwargs):
        """先实际读取画像，再建立合成记忆计划并经原结束工具交付。"""
        self.calls.append(copy.deepcopy({'messages': messages, 'tools': tools}))
        names = [call['function']['name'] for message in messages
                 for call in message.get('tool_calls', [])]
        if 'get_profile' not in names:
            name, args = 'get_profile', {}
        elif 'set_research_plan' not in names:
            name = 'set_research_plan'
            args = {'goal': '延续合成讨论', 'task_kind': 'memory',
                    'goals': [{'goal_id': 'memory', 'description': '延续合成讨论', 'min_papers': 0}]}
        elif self.read_request and 'read_memory' not in names:
            name, args = 'read_memory', self.read_request
        else:
            if self.after_read and not self.read_callback_done:
                self.after_read()
                self.read_callback_done = True
            name = 'finish_task'
            args = {'answer_markdown': self.answer,
                    'goals': [{'goal_id': 'memory', 'status': 'satisfied', 'reason': '已交付合成讨论'}]}
            if self.candidate is not OMITTED:
                args['continuation_memory'] = (self.candidate(memory_context(messages))
                                               if callable(self.candidate) else self.candidate)
        sequence = len(self.calls)
        return {'id': 'continuation-fixture-' + str(sequence),
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                'choices': [{'finish_reason': 'tool_calls', 'message': {
                    'role': 'assistant', 'content': '', 'tool_calls': [{
                        'id': 'continuation-call-' + str(sequence), 'type': 'function',
                        'function': {'name': name, 'arguments': dump(args)}}]}}]}


class BatchReadFinishProvider(ContinuationProvider):
    """在同一响应中提出读历史和结束，模拟尚未观察读回内容的引用。"""

    def send(self, messages, tools, **kwargs):
        """前两步沿用画像与计划，第三步同时返回 read_memory 和 finish_task。"""
        names = [call['function']['name'] for message in messages
                 for call in message.get('tool_calls', [])]
        if 'set_research_plan' not in names:
            return super().send(messages, tools, **kwargs)
        self.calls.append(copy.deepcopy({'messages': messages, 'tools': tools}))
        args = {'answer_markdown': self.answer,
                'goals': [{'goal_id': 'memory', 'status': 'satisfied', 'reason': '当前合成回答已交付'}],
                'continuation_memory': self.candidate}
        sequence = len(self.calls)
        return {'id': 'continuation-batch-fixture-' + str(sequence),
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                'choices': [{'finish_reason': 'tool_calls', 'message': {
                    'role': 'assistant', 'content': '', 'tool_calls': [
                        {'id': 'batch-read-' + str(sequence), 'type': 'function',
                         'function': {'name': 'read_memory', 'arguments': dump(self.read_request)}},
                        {'id': 'batch-finish-' + str(sequence), 'type': 'function',
                         'function': {'name': 'finish_task', 'arguments': dump(args)}}]}}]}


class ContinuationMemoryTests(unittest.TestCase):
    """验证实验开启、跨轮接续、关系链、旁路拒绝与无模型恢复。"""

    def setUp(self):
        """每例独立临时 SQLite 与目录，并拒绝网络和真实供应商。"""
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
            blocker = patch(target, side_effect=AssertionError('接续记忆测试禁止外网和部署服务'))
            blocker.start()
            self.addCleanup(blocker.stop)
        self.db = Database(self.settings.database_url)
        self.db.ensure_schema()
        self.tasks = TaskService(self.db, self.settings)
        self.conversations = ConversationService(self.db, self.settings)
        self.cid = self.conversation()
        self.continuation = ContinuationMemoryService(self.db)

    def conversation(self) -> str:
        """创建合成会话，不访问部署记录。"""
        return self.conversations.create('合成接续测试', uuid4().hex)['conversation_id']

    def submit(self, prompt: str, cid: str | None = None, save_policy='deny') -> str:
        """通过原任务服务提交新根任务，权限仅取当前输入。"""
        return self.tasks.create(TaskCreate(prompt=prompt, conversation_id=cid or self.cid,
            idempotency_key=uuid4().hex, save_policy=save_policy)).task_id

    def execute(self, task_id: str, provider: ContinuationProvider, *, enabled=True):
        """只替换模型供应商，保留正式 Worker、角色图、工具与收据。"""
        worker = TaskExecutor(self.settings, self.db, provider_factory=lambda settings: provider,
            continuation_memory=enabled,
            supervisor_tools={'get_profile', 'set_research_plan', 'finish_task', 'read_memory'})
        result = worker.execute(task_id)
        self.assertEqual(self.db.get_task(task_id)['status'], 'completed', result)
        self.assertEqual(result.answer_markdown, provider.answer)
        self.assertEqual(len(provider.calls), provider.expected_calls)
        self.assertEqual(len(self.db.list_model_calls(task_id)), provider.expected_calls)
        return worker, result

    def record(self, task_id: str) -> dict:
        """读取已经持久化的研究历史，不重新生成模型答案。"""
        row = self.db.fetchone('SELECT record_json FROM research_memories WHERE task_id=?', (task_id,))
        self.assertIsNotNone(row)
        return json.loads(row['record_json'])

    @staticmethod
    def candidate(answer: str, *, key='conclusion', conditions=()) -> dict:
        """合成有当前回答逐字来源的候选，不宣称科学核验通过。"""
        return {'research_goal': '保留合成讨论的接续状态',
                'conclusions': [{'key': key, 'text': answer, 'conditions': list(conditions),
                                 'sources': [{'source_type': 'current_answer', 'quote': answer}]}],
                'unresolved': ['具体论文事实仍需当前授权原文核验'],
                'next_steps': ['需要时按原始历史身份读回'], 'corrections': []}

    def correction_candidate(self, answer: str, old_task: str, *, relation='corrects'):
        """从真实待发目录选择旧结论身份，再提出本轮更正关系。"""
        def create(context):
            notes = context['continuation_memory']['notes']
            old = next(note for note in notes if note['task_id'] == old_task)
            conclusion = old['conclusions'][0]
            candidate = self.candidate(answer)
            candidate['corrections'] = [{'previous_task_id': old_task,
                'previous_key': conclusion['key'], 'previous_content_hash': conclusion['content_hash'],
                'replacement_key': 'conclusion', 'relation': relation, 'reason': '当前合成问题明确要求更正'}]
            return candidate
        return create

    def test_disabled_schema_and_context_keep_existing_contract(self):
        """默认关闭时不暴露新字段，不创建策略或向下一轮注入实验记忆。"""
        default = {item.name: item.schema() for item in delegation_specs()}
        disabled = {item.name: item.schema() for item in delegation_specs(continuation_memory=False)}
        self.assertEqual(default, disabled)
        self.assertNotIn('continuation_memory', default['finish_task']['function']['parameters']['properties'])
        task_id = self.submit('合成讨论：当前回答只解释局部条件，不保存。')
        provider = ContinuationProvider('这是合成回答。')
        self.execute(task_id, provider, enabled=False)
        self.assertFalse(self.continuation.enabled(task_id))
        self.assertNotIn('continuation_memory', self.record(task_id))
        for call in provider.calls:
            self.assertNotIn('continuation_memory', memory_context(call['messages']))

    def test_model_candidate_persists_and_is_visible_next_turn(self):
        """摘要从模型响应经正式行动写入，下一轮可见且不继承旧保存权限。"""
        answer = '只在同一数据集和相同分辨率下比较。'
        first = self.submit('比较需要保留实验条件，本轮允许保存。', save_policy='allow')
        provider = ContinuationProvider(answer, self.candidate(answer, conditions=['同一数据集', '相同分辨率']))
        self.execute(first, provider)
        schema = next(tool for tool in provider.calls[0]['tools'] if tool['function']['name'] == 'finish_task')
        self.assertIn('continuation_memory', schema['function']['parameters']['properties'])
        note = self.record(first)['continuation_memory']
        self.assertEqual(note['status'], 'applied')
        self.assertEqual(note['conclusions'][0]['verification_status'], 'historical_unverified')
        self.assertEqual(note['conclusions'][0]['conditions'], ['同一数据集', '相同分辨率'])
        second = self.submit('接着讨论，但本轮不保存。')
        later = ContinuationProvider('当前要求优先，本轮不保存。')
        self.execute(second, later)
        context = memory_context(later.calls[0]['messages'])
        notes = context['continuation_memory']['notes']
        self.assertEqual(next(item for item in notes if item['task_id'] == first)['conclusions'][0]['text'], answer)
        self.assertEqual(self.db.get_task(second)['save_policy'], 'deny')
        self.assertEqual(self.db.fetchall('SELECT * FROM save_receipts'), [])
        self.assertEqual(self.db.fetchall('SELECT * FROM task_evidence'), [])

    def test_correction_chain_keeps_original_history_and_connected_notes(self):
        """A→B→C 保留原回答和显式关系，从旧节点也能找回后续更正。"""
        first = self.submit('合成旧说法：删去正确项后比例不变。')
        first_answer = '合成旧说法：比例不变。'
        self.execute(first, ContinuationProvider(first_answer, self.candidate(first_answer)))
        original = self.record(first)
        second = self.submit('更正：分母也会变化，请更新之前的解释。')
        second_answer = '更正为按保留集合计算分母。'
        self.execute(second, ContinuationProvider(second_answer, self.correction_candidate(second_answer, first)))
        third = self.submit('再补充：比较时必须同时记录覆盖率。')
        third_answer = '补充条件：同时记录覆盖率和错误比例。'
        self.execute(third, ContinuationProvider(third_answer,
            self.correction_candidate(third_answer, second, relation='qualifies')))
        self.assertEqual(self.record(first), original)
        self.assertEqual(self.record(second)['continuation_memory']['corrections'][0]['previous_task_id'], first)
        self.assertEqual(self.record(third)['continuation_memory']['corrections'][0]['previous_task_id'], second)
        fourth = self.submit('沿之前的讨论继续，本轮仍不保存。')
        self.assertTrue(self.continuation.enable_new_task(fourth))
        context = self.continuation.context(fourth, [first])
        self.assertEqual({note['task_id'] for note in context['notes']}, {first, second, third})
        self.assertTrue(all(note['readback']['source_type'] == 'research_history' for note in context['notes']))

    def test_malformed_candidate_does_not_block_delivery(self):
        """格式错误的旁路候选明确拒绝，正式回答仍然 completed。"""
        task_id = self.submit('本轮只解释合成问题，不保存。')
        self.execute(task_id, ContinuationProvider('正式合成回答保持。', {'conclusions': '无效格式'}))
        note = self.record(task_id)['continuation_memory']
        self.assertEqual(note['status'], 'rejected')
        self.assertIn('reason', note)

    def test_omitted_candidate_is_recorded_without_inventing_summary(self):
        """模型未提出摘要时单列 not_proposed，不伪造接续成功。"""
        task_id = self.submit('只交付当前回答，不保存。')
        self.execute(task_id, ContinuationProvider('仅当前回答。'))
        note = self.record(task_id)['continuation_memory']
        self.assertEqual(note['status'], 'not_proposed')
        self.assertNotIn('conclusions', note)

    def test_quote_mismatch_rejects_memory_but_keeps_answer(self):
        """不存在的逐字来源不能写入摘要，也不影响正文交付。"""
        task_id = self.submit('合成问题只涉及实验条件。')
        candidate = self.candidate('合法合成回答。')
        candidate['conclusions'][0]['sources'][0]['quote'] = '实际输入输出均没有的引句'
        self.execute(task_id, ContinuationProvider('合法合成回答。', candidate))
        self.assertEqual(self.record(task_id)['continuation_memory']['reason'], 'CONTINUATION_QUOTE_MISMATCH')

    def test_stale_correction_hash_is_rejected(self):
        """旧结论身份匹配而 hash 过期时拒绝关系，不改旧记录。"""
        first = self.submit('合成旧讨论。')
        old_answer = '原结论必须保留。'
        self.execute(first, ContinuationProvider(old_answer, self.candidate(old_answer)))
        original = self.record(first)
        second = self.submit('本轮提出一个更正。')
        builder = self.correction_candidate('本轮新解释。', first)
        def stale(context):
            candidate = builder(context)
            candidate['corrections'][0]['previous_content_hash'] = '0' * 64
            return candidate
        self.execute(second, ContinuationProvider('本轮新解释。', stale))
        self.assertEqual(self.record(second)['continuation_memory']['status'], 'rejected')
        self.assertEqual(self.record(first), original)

    def test_self_correction_target_is_rejected(self):
        """本任务不能作为自己的旧结论，防止自指和循环关系。"""
        task_id = self.submit('合成更正问题。')
        candidate = self.candidate('合成更正回答。')
        candidate['corrections'] = [{'previous_task_id': task_id, 'previous_key': 'conclusion',
            'previous_content_hash': '0' * 64, 'replacement_key': 'conclusion',
            'relation': 'corrects', 'reason': '伪造自指'}]
        self.execute(task_id, ContinuationProvider('合成更正回答。', candidate))
        self.assertEqual(self.record(task_id)['continuation_memory']['status'], 'rejected')

    def test_unseen_history_reference_is_rejected(self):
        """仅在数据库存在的无关历史不能被模型猜身份后用于摘要。"""
        first = self.submit('zebra cormorant', cid=self.conversation())
        old_answer = 'zebra cormorant forest'
        self.execute(first, ContinuationProvider(old_answer, self.candidate(old_answer)))
        old_record = self.record(first)
        task_id = self.submit('helium orbit telescope')
        candidate = self.candidate('本轮回答仍交付。')
        candidate['conclusions'][0]['sources'] = [{'source_type': 'research_history',
            'task_id': first, 'quote': old_record['answer']}]
        provider = ContinuationProvider('本轮回答仍交付。', candidate)
        self.execute(task_id, provider)
        for call in provider.calls:
            visible = memory_context(call['messages'])['continuation_memory']['visible_history']
            self.assertNotIn(first, {item['task_id'] for item in visible})
        self.assertEqual(self.record(task_id)['continuation_memory']['reason'],
                         'CONTINUATION_HISTORY_OUTSIDE_VISIBLE_SCOPE')

    def test_projection_recovery_is_idempotent_and_does_not_send_model(self):
        """终态后投影中断只重试数据库，重复恢复不增加模型收据或关系。"""
        task_id = self.submit('合成恢复问题，本轮不保存。')
        provider = ContinuationProvider('恢复后仍是原回答。', self.candidate('恢复后仍是原回答。'))
        with patch.object(ContinuationMemoryService, 'apply', side_effect=RuntimeError('合成投影中断')):
            with self.assertLogs(level='ERROR'):
                self.execute(task_id, provider)
        before_calls = copy.deepcopy(self.db.list_model_calls(task_id))
        first = self.continuation.apply(task_id)
        self.assertEqual(first['status'], 'applied')
        frozen = copy.deepcopy(self.record(task_id))
        self.assertEqual(self.continuation.apply(task_id), first)
        self.continuation.recover_pending()
        self.conversations.reconcile(task_id)
        self.assertEqual(self.record(task_id), frozen)
        self.assertEqual(self.db.list_model_calls(task_id), before_calls)
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM memory_operations WHERE request_key=?',
                                        (PROJECTION + task_id,))['n'], 1)

    def test_default_worker_recovers_frozen_projection_without_enabling_new_tasks(self):
        """重启后的默认 Worker 补齐已冻结的旁路投影，不构造供应商或重发模型。"""
        task_id = self.submit('合成恢复窗口，本轮不保存。')
        answer = '已交付回答只需补齐记忆投影。'
        provider = ContinuationProvider(answer, self.candidate(answer))
        with patch.object(ContinuationMemoryService, 'apply', side_effect=RuntimeError('合成投影中断')):
            with self.assertLogs(level='ERROR'):
                self.execute(task_id, provider)
        before_calls = copy.deepcopy(self.db.list_model_calls(task_id))
        default = TaskExecutor(self.settings, self.db,
            provider_factory=lambda settings: self.fail('恢复旁路不应构造模型供应商'))
        self.assertFalse(default.continuation_memory)
        self.assertIsNone(default.run_once('default-recovery-worker'))
        self.assertEqual(self.record(task_id)['continuation_memory']['status'], 'applied')
        frozen = copy.deepcopy(self.record(task_id))
        self.assertIsNone(default.run_once('default-recovery-worker'))
        self.assertEqual(self.record(task_id), frozen)
        self.assertEqual(self.db.list_model_calls(task_id), before_calls)
        self.assertEqual(len(provider.calls), 3)

    def test_existing_receipts_cannot_be_retroactively_enabled(self):
        """已经执行的旧任务不能因为新 Worker 开关而补开实验。"""
        task_id = self.submit('合成旧流程。')
        self.execute(task_id, ContinuationProvider('按旧流程交付。'), enabled=False)
        self.assertFalse(self.continuation.enable_new_task(task_id))
        self.assertFalse(self.continuation.enabled(task_id))

    def test_run_once_enables_fresh_task_after_atomic_claim(self):
        """正式队列领取路径在首次模型请求前冻结新任务的实验策略。"""
        task_id = self.submit('队列任务只整理当前合成讨论，不保存。')
        answer = '通过正式队列交付的合成回答。'
        provider = ContinuationProvider(answer, self.candidate(answer))
        worker = TaskExecutor(self.settings, self.db, provider_factory=lambda settings: provider,
            continuation_memory=True,
            supervisor_tools={'get_profile', 'set_research_plan', 'finish_task', 'read_memory'})
        self.assertEqual(worker.run_once('continuation-queue-worker'), task_id)
        self.assertEqual(self.db.get_task(task_id)['status'], 'completed')
        self.assertTrue(self.continuation.enabled(task_id))
        self.assertEqual(self.record(task_id)['continuation_memory']['status'], 'applied')
        self.assertEqual(len(provider.calls), 3)
        schema = next(tool for tool in provider.calls[0]['tools'] if tool['function']['name'] == 'finish_task')
        self.assertIn('continuation_memory', schema['function']['parameters']['properties'])

    def test_revoked_preference_origin_summary_is_not_reinjected(self):
        """撤销偏好后不重新注入所属历史的模型摘要，原件仍留库。"""
        phrase = '以后使用中文回答'
        first = self.submit(phrase + '，本轮不保存。')
        memory = MemoryService(self.db)
        preference = memory.remember(first, RememberInput(slot='language', kind='explicit',
            content=phrase, source_quote=phrase))
        answer = phrase + '；科研细节仍按原文核对。'
        self.execute(first, ContinuationProvider(answer, self.candidate(answer)))
        original = copy.deepcopy(self.record(first))
        memory.forget(preference['memory_id'], preference['version'])
        second = self.submit('本轮使用英文回答，继续之前讨论，不保存。')
        provider = ContinuationProvider('Current instructions take precedence.')
        self.execute(second, provider)
        context = memory_context(provider.calls[0]['messages'])
        notes = context['continuation_memory']['notes']
        self.assertNotIn(first, {item['task_id'] for item in notes})
        self.assertEqual(self.record(first), original)
        self.assertEqual(self.db.get_task(second)['save_policy'], 'deny')

    def test_context_limit_externalizes_whole_note_without_cutting_conditions(self):
        """目录额度不足时整条摘要卸载，结论条件和原始摘要不被裁断。"""
        first = self.submit('整理长的合成接续状态，本轮不保存。')
        answer = 'Synthetic source anchor.'
        candidate = self.candidate(answer)
        candidate['conclusions'] = [
            {'key': 'item' + str(index), 'text': ('synthetic research state ' * 45) + str(index),
             'conditions': ['ONLY_WITH_IDENTICAL_DATASET_AND_RESOLUTION_' + str(index)],
             'sources': [{'source_type': 'current_answer', 'quote': answer}]}
            for index in range(8)]
        self.execute(first, ContinuationProvider(answer, candidate))
        original = copy.deepcopy(self.record(first))
        second = self.submit('按之前的合成状态继续，不保存。')
        self.assertTrue(self.continuation.enable_new_task(second))
        complete = self.continuation.context(second, [first], max_tokens=100000)
        self.assertEqual(len(complete['notes']), 1)
        limited = self.continuation.context(second, [first], max_tokens=1024)
        self.assertEqual(limited['notes'], [])
        self.assertEqual(limited['omitted_notes'], 1)
        self.assertEqual(self.record(first), original)
        self.assertEqual(complete['notes'][0]['conclusions'][0]['conditions'],
                         ['ONLY_WITH_IDENTICAL_DATASET_AND_RESOLUTION_0'])

    def test_history_quote_outside_actual_read_window_is_rejected(self):
        """成功读回历史的一小段，不授权引用同一记录中尚未呈现的其他句子。"""
        first = self.submit('zebra cormorant forest', cid=self.conversation())
        hidden = 'UNSEEN_OLD_FACT_OUTSIDE_READ_WINDOW'
        old_answer = 'VISIBLE_OLD_FACT ' + ('x' * 300) + hidden
        self.execute(first, ContinuationProvider(old_answer, self.candidate(old_answer)))
        task_id = self.submit('helium orbit telescope')
        candidate = self.candidate('本轮合成回答。')
        candidate['conclusions'][0]['sources'] = [{'source_type': 'research_history',
            'task_id': first, 'quote': hidden}]
        provider = ContinuationProvider('本轮合成回答。', candidate,
            read_request={'source_type': 'research_history', 'source_id': first, 'offset': 0, 'limit': 100})
        self.execute(task_id, provider)
        read = next(action for action in self.db.list_actions(task_id) if action['tool_name'] == 'read_memory')
        self.assertEqual(read['status'], 'succeeded')
        self.assertNotIn(hidden, read['observation']['data']['content'])
        self.assertEqual(self.record(task_id)['continuation_memory']['status'], 'rejected')

    def test_history_changed_after_read_rejects_old_source_snapshot(self):
        """来源在成功读回后变化时拒绝旧窗口引用，即使该引句仍存在。"""
        first = self.submit('zebra cormorant forest', cid=self.conversation())
        old_answer = 'VISIBLE_OLD_FACT remains in the source.'
        self.execute(first, ContinuationProvider(old_answer, self.candidate(old_answer)))
        original = copy.deepcopy(self.record(first))
        task_id = self.submit('helium orbit telescope')
        candidate = self.candidate('本轮合成回答。')
        candidate['conclusions'][0]['sources'] = [{'source_type': 'research_history',
            'task_id': first, 'quote': 'VISIBLE_OLD_FACT'}]
        def change_after_read():
            changed = copy.deepcopy(original)
            changed['prompt'] += ' changed after successful read'
            self.db.execute('UPDATE research_memories SET record_json=? WHERE task_id=?',
                            (dump(changed), first))
        provider = ContinuationProvider('本轮合成回答。', candidate, after_read=change_after_read,
            read_request={'source_type': 'research_history', 'source_id': first, 'offset': 0, 'limit': 100})
        self.execute(task_id, provider)
        read = next(action for action in self.db.list_actions(task_id) if action['tool_name'] == 'read_memory')
        self.assertEqual(read['status'], 'succeeded')
        self.assertIn('VISIBLE_OLD_FACT', read['observation']['data']['content'])
        self.assertNotEqual(self.record(first)['prompt'], original['prompt'])
        self.assertEqual(self.record(task_id)['continuation_memory']['status'], 'rejected')

    def test_same_response_read_and_finish_cannot_cite_unobserved_result(self):
        """同批读回虽已执行，未进入后继模型请求的内容仍不能充当已见依据。"""
        first = self.submit('zebra cormorant forest', cid=self.conversation())
        old_answer = 'BATCH_ONLY_OLD_FACT remains in this unrelated record.'
        self.execute(first, ContinuationProvider(old_answer, self.candidate(old_answer)))
        task_id = self.submit('helium orbit telescope')
        candidate = self.candidate('同批结束时仍交付当前合成回答。')
        candidate['conclusions'][0]['sources'] = [{'source_type': 'research_history',
            'task_id': first, 'quote': 'BATCH_ONLY_OLD_FACT'}]
        provider = BatchReadFinishProvider('同批结束时仍交付当前合成回答。', candidate,
            read_request={'source_type': 'research_history', 'source_id': first, 'offset': 0, 'limit': 100})
        provider.expected_calls = 3
        self.execute(task_id, provider)
        actions = self.db.list_actions(task_id)
        read = next(action for action in actions if action['tool_name'] == 'read_memory')
        finish = next(action for action in actions if action['tool_name'] == 'finish_task')
        self.assertEqual(read['status'], 'succeeded')
        self.assertEqual(finish['status'], 'succeeded')
        self.assertIn('BATCH_ONLY_OLD_FACT', read['observation']['data']['content'])
        for call in provider.calls:
            self.assertNotIn(first, {item['task_id'] for item in
                memory_context(call['messages'])['continuation_memory']['visible_history']})
            tool_names = {item['id']: item['function']['name'] for message in call['messages']
                          for item in message.get('tool_calls', [])}
            self.assertFalse(any(message.get('role') == 'tool'
                and tool_names.get(message.get('tool_call_id')) == 'read_memory'
                for message in call['messages']))
        self.assertEqual(self.record(task_id)['continuation_memory']['status'], 'rejected')

    def test_revocation_follows_observed_history_even_with_current_answer_sources(self):
        """B 实际看过 A 后仅引用本轮回答，撤销 A 仍排除 A 与 B 的派生摘要。"""
        phrase = '以后使用中文回答'
        first = self.submit(phrase + '，本轮不保存。')
        memory = MemoryService(self.db)
        preference = memory.remember(first, RememberInput(slot='language', kind='explicit',
            content=phrase, source_quote=phrase))
        first_answer = phrase + '，这是需要保持来源身份的合成讨论。'
        self.execute(first, ContinuationProvider(first_answer, self.candidate(first_answer)))
        second = self.submit('继续之前讨论，整理接续状态，不保存。')
        second_answer = '本轮根据前面的讨论整理了一份合成回答。'
        provider = ContinuationProvider(second_answer, self.candidate(second_answer))
        self.execute(second, provider)
        observed = memory_context(provider.calls[0]['messages'])['continuation_memory']['notes']
        self.assertIn(first, {note['task_id'] for note in observed})
        second_note = self.record(second)['continuation_memory']
        self.assertEqual(second_note['status'], 'applied')
        self.assertIn(first, second_note['history_dependencies'])
        self.assertTrue(all(source['source_type'] == 'current_answer'
            for conclusion in second_note['conclusions'] for source in conclusion['sources']))
        original_first, original_second = copy.deepcopy(self.record(first)), copy.deepcopy(self.record(second))
        memory.forget(preference['memory_id'], preference['version'])
        third = self.submit('本轮使用英文回答，不保存。')
        later = ContinuationProvider('Use the current instruction for this turn.')
        self.execute(third, later)
        notes = memory_context(later.calls[0]['messages'])['continuation_memory']['notes']
        self.assertTrue({first, second}.isdisjoint({note['task_id'] for note in notes}))
        self.assertEqual(self.record(first), original_first)
        self.assertEqual(self.record(second), original_second)


if __name__ == '__main__':
    unittest.main()
