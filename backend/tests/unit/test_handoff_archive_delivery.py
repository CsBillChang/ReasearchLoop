"""交接档案导航和明确回答身份的零模型边界，不使用真实论文。"""
from copy import deepcopy
import hashlib
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from research_loop.agents.graph import handoff_navigation, handoff_schemas
from research_loop.config import Settings
from research_loop.domain.memory import HistoryRead
from research_loop.domain.tasks import ActionRecord, TaskCreate
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor


def navigation_action(offset=0, text='原文', source='archive-A', version='hash-A', kind='context_archive'):
    """生成实际read_memory观察的必要字段，不将其伪装成原文证据。"""
    return {'tool_name': 'read_memory', 'status': 'succeeded', 'observation': {'ok': True, 'data': {
        'source_type': kind, 'source_id': source, 'content_hash': version, 'offset': offset,
        'total_chars': 1000, 'content': text, 'is_original_paper_evidence': False}}}


class HandoffNavigationTests(unittest.TestCase):
    """新范围允许导航，重复/过期/不可用来源不刷新研究进展。"""

    def test_schema_keeps_only_existing_role_authority(self):
        """Supervisor保留已授权read_memory，其他角色及缺失注册工具不增加权限。"""
        schemas = [{'function': {'name': name}} for name in ['read_memory', 'read_paper', 'finish_task', 'finish_research']]
        self.assertEqual([t['function']['name'] for t in handoff_schemas(schemas, 'supervisor')], ['read_memory', 'finish_task'])
        self.assertEqual([t['function']['name'] for t in handoff_schemas(schemas, 'researcher')], ['finish_research'])
        self.assertEqual(handoff_schemas(schemas[1:2], 'supervisor'), [])

    def test_new_range_can_continue_but_not_repeat_or_roll_archive_identity(self):
        """分页连续/重叠可推进；全部已覆盖和换档案版本不能无限续期。"""
        state = {'trigger_action_id': 'boundary'}; original = deepcopy(state)
        navigation, changed = handoff_navigation(state, [navigation_action()])
        self.assertTrue(changed); self.assertEqual(state, original)
        state['navigation'] = navigation
        self.assertFalse(handoff_navigation(state, [navigation_action()])[1])
        self.assertFalse(handoff_navigation(state, [navigation_action(source='new-archive')])[1])
        self.assertFalse(handoff_navigation(state, [navigation_action(version='new-version')])[1])
        navigation, changed = handoff_navigation(state, [navigation_action(1, '另一段')])
        self.assertTrue(changed)
        self.assertEqual(navigation['context_archive']['ranges'], [[0, 4]])

    def test_failure_empty_and_non_navigation_never_advance(self):
        """失败、空范围和历史检索不是交接档案推进，也不算新事实。"""
        failed = navigation_action(); failed['status'] = 'failed'
        for action in [failed, navigation_action(text=''), navigation_action(kind='research_history'),
                       navigation_action(offset=1000)]:
            self.assertFalse(handoff_navigation({}, [action])[1])


class ReferencedAnswerTests(unittest.TestCase):
    """回答身份绑定当前会话/消息序号/内容版本，不默认取最近一条。"""

    def setUp(self):
        """临时SQLite与deny配置，无外部模型或文件写入。"""
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.settings = Settings(project_root=root, database_url='sqlite:///' + str(root/'test.db'),
            data_dir=root/'data', output_dir=root/'notes', retrieval_enabled=False)
        self.db = Database(self.settings.database_url); self.db.ensure_schema()
        self.conversations = ConversationService(self.db, self.settings)
        self.body = '原有回答。保留符号 $x$ 与未确认条件。'

    def seed(self, prompt=None, role='assistant'):
        """先写旧消息再创建真实会话任务，源任务可为导入历史身份。"""
        cid = self.conversations.create('synthetic', uuid4().hex)['conversation_id']
        with self.conversations.store.transaction() as tx:
            mid = self.conversations.add_message(tx, cid, 'historical-only', uuid4().hex, role, self.body)
        prompt = prompt if prompt is not None else '核对原回答 hash=' + hashlib.sha256(self.body.encode()).hexdigest()
        tid = TaskService(self.db, self.settings).create(TaskCreate(prompt=prompt, conversation_id=cid, save_policy='deny')).task_id
        return cid, mid, tid

    def test_explicit_hash_preserves_answer_and_failure_separately(self):
        """容量异常未建核验对象时，保留明确引用的原回答及准确原因。"""
        _, mid, tid = self.seed()
        worker = TaskExecutor(self.settings, self.db)
        state = {'coverage_required': True, 'final_reason': 'CONTEXT_CAPACITY: synthetic'}
        result = worker._deliverable_result(tid, state)
        self.assertEqual(result.answer_body_markdown, self.body)
        self.assertIn('CONTEXT_CAPACITY', result.processing_summary)
        self.assertIn('未完成本轮独立核验', result.processing_summary)
        self.assertEqual(result.retained_answer_source['source_id'], mid)
        self.assertFalse(result.retained_answer_source['is_original_paper_evidence'])
        self.assertFalse(result.retained_answer_source['save_permission_inherited'])
        self.assertEqual(result.review_status, 'unreviewed')
        self.assertEqual(worker._delivery_status(tid, state, result).value, 'partial')
        self.assertIsNone(result.saved_path)

    def test_no_reference_does_not_pick_latest_or_foreign_answer(self):
        """没有明确来源的普通任务不能把任意旧回复当成新任务成果。"""
        _, mid, tid = self.seed(prompt='找一篇新论文')
        self.assertIsNone(self.conversations.referenced_answer(tid))
        cid = self.conversations.create('other', uuid4().hex)['conversation_id']
        other = TaskService(self.db, self.settings).create(TaskCreate(prompt='hash='+hashlib.sha256(self.body.encode()).hexdigest(),
            conversation_id=cid, save_policy='deny')).task_id
        self.assertIsNone(self.conversations.answer_message(other, mid))
        self.assertIsNone(self.conversations.referenced_answer(other))

    def test_changed_user_or_future_message_not_accepted(self):
        """角色、版本和当前输入顺序是硬边界。"""
        cid, mid, tid = self.seed(role='user')
        self.assertIsNone(self.conversations.referenced_answer(tid))
        with self.conversations.store.transaction() as tx:
            late = self.conversations.add_message(tx, cid, 'later', uuid4().hex, 'assistant', self.body)
        self.assertIsNone(self.conversations.answer_message(tid, late))
        self.db.execute("UPDATE conversation_messages SET role='assistant',content=? WHERE message_id=?", ('changed', mid))
        self.assertIsNone(self.conversations.referenced_answer(tid))

    def test_successful_message_read_can_select_natural_language_followup(self):
        """普通自然语言追问可以沿用现有message读回身份，无需用户填写hash。"""
        _, mid, tid = self.seed(prompt='根据上次回答继续核验')
        data = MemoryService(self.db).read(tid, HistoryRead(source_type='message', source_id=mid))
        record = ActionRecord(task_id=tid, sequence=1, tool_name='read_memory',
            arguments={'source_type': 'message', 'source_id': mid}, status='succeeded',
            observation={'ok': True, 'kind': 'success', 'data': data})
        self.db.save_action(record.model_dump())
        self.assertEqual(self.conversations.referenced_answer(tid)['content'], self.body)
        self.db.execute('UPDATE conversation_messages SET content=? WHERE message_id=?', ('changed', mid))
        self.assertIsNone(self.conversations.referenced_answer(tid))

    def test_new_current_answer_has_priority(self):
        """已形成的新回答不会被旧历史覆盖，旧保存授权也不会被继承。"""
        _, _, tid = self.seed()
        result = TaskExecutor(self.settings, self.db)._deliverable_result(tid,
            {'coverage_required': True, 'final_reason': 'stopped', 'final_answer': '本轮已有新回答'})
        self.assertEqual(result.answer_body_markdown, '本轮已有新回答')
        self.assertIsNone(result.retained_answer_source)
