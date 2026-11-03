"""跨会话研究记录保留真实澄清；仅本地合成数据，不调用模型。"""
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from research_loop.agents.graph import initial_state
from research_loop.config import Settings
from research_loop.domain.memory import HistoryRead, RememberInput
from research_loop.domain.tasks import ResumeInput, TaskCreate, TaskResult, TaskStatus
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.services.tasks import TaskService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.mysql import Database
from research_loop.tools.base import ToolContext, execute_spec
from research_loop.tools.memory import specs as memory_specs


class ClarificationHistoryTests(unittest.TestCase):
    """真实会话/澄清接口→终态投影→新会话历史检索和读回。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = Settings(project_root=root, data_dir=root/'data',
            output_dir=root/'output', database_url='sqlite:///'+str(root/'runtime.sqlite3'),
            deepseek_api_key=None)
        self.addCleanup(patch.stopall)
        patch('research_loop.config.get_settings', return_value=self.settings).start()
        patch('research_loop.config.Settings.from_env',
            side_effect=AssertionError('ISOLATED_TEST_NO_ENV')).start()
        patch.object(socket.socket, 'connect',
            side_effect=AssertionError('ZERO_MODEL_NETWORK_FORBIDDEN')).start()
        self.db = Database(self.settings.database_url)
        self.db.ensure_schema()
        self.tasks = TaskService(self.db, self.settings)
        self.conversations = ConversationService(self.db, self.settings)
        self.memory = MemoryService(self.db)
        self.cid = self.new_conversation()

    def new_conversation(self):
        return self.conversations.create('合成澄清回归', uuid4().hex)['conversation_id']

    def new_task(self, cid=None, prompt='请先确认本次比较口径，再给出结论。', policy='deny'):
        return self.tasks.create(TaskCreate(prompt=prompt, conversation_id=cid or self.cid,
            save_policy=policy, idempotency_key=uuid4().hex)).task_id

    def answer(self, task_id, question, answer):
        """经正式暂停与resume接口提交回答；不直接拼接记忆数据。"""
        clarification = self.tasks.request_clarification(task_id, question,
            ['criterion'], ['合成建议值，不代表用户决定'])
        state = CheckpointStore(self.db).load(task_id) or initial_state(
            task_id, self.db.get_task(task_id)['prompt'], self.db.get_task(task_id)['save_policy'])
        state.update(final_reason='clarification requested', finished=True)
        CheckpointStore(self.db).save(task_id, state)
        self.db.update_task(task_id, status='needs_input')
        payload = ResumeInput(clarification_id=clarification['clarification_id'],
            answer=answer, idempotency_key=uuid4().hex)
        self.tasks.resume(payload, task_id)
        return clarification, payload

    def finish(self, task_id, status=TaskStatus.completed):
        """合成任务结束，故意不在最终答案中复述用户澄清参数。"""
        self.tasks.complete(task_id, status,
            TaskResult(answer_markdown='已按确认的口径完成本次说明。'))

    def read_in_new_conversation(self, source_task_id):
        target = self.new_task(self.new_conversation(), '继续此前已确认的口径，本次禁止保存。')
        result = self.memory.read(target,
            HistoryRead(source_type='research_history', source_id=source_task_id, limit=16000))
        return target, result, json.loads(result['content'])

    def test_answered_fact_is_recoverable_with_exact_provenance(self):
        task_id = self.new_task()
        fact = '错误率分母固定为六条，合成口径码 ALPHA-SIX。'
        clarification, payload = self.answer(task_id, '错误率分母采用哪一组？', fact)
        # 同一回答的重放不能生成第二次澄清事实。
        self.tasks.resume(payload, task_id)
        self.finish(task_id)
        target, result, record = self.read_in_new_conversation(task_id)
        self.assertIn(fact, result['content'])
        items = record['answered_clarifications']
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item['clarification_id'], clarification['clarification_id'])
        self.assertEqual(item['task_id'], task_id)
        self.assertEqual(item['conversation_id'], self.cid)
        self.assertEqual(item['question'], '错误率分母采用哪一组？')
        self.assertEqual(item['answer'], fact)
        for field, role, text in [('question_message_id', 'assistant', item['question']),
                                  ('answer_message_id', 'user', fact)]:
            row = self.db.fetchone('SELECT * FROM conversation_messages WHERE message_id=?',
                (item[field],))
            self.assertEqual((row['conversation_id'], row['task_id'], row['role'], row['content']),
                (self.cid, task_id, role, text))
        self.assertFalse(result['is_original_paper_evidence'])
        self.assertEqual(self.db.get_task(target)['save_policy'], 'deny')
        self.assertEqual(self.db.list_model_calls(task_id), [])
        self.assertEqual(self.db.list_model_calls(target), [])

    def test_fact_only_in_clarification_is_searchable(self):
        task_id = self.new_task()
        self.answer(task_id, '唯一回查标记是什么？', 'SYNTHETIC-ORCHID-783 条件适用于当前实验。')
        self.finish(task_id)
        hits = self.memory.search('ORCHID-783', 'research_history', 5)
        self.assertIn(task_id, [hit['source_id'] for hit in hits])
        target, _, record = self.read_in_new_conversation(task_id)
        self.assertEqual(record['answered_clarifications'][0]['answer'],
            'SYNTHETIC-ORCHID-783 条件适用于当前实验。')
        self.assertEqual(self.db.list_task_evidence(target), [])

    def test_answer_order_and_reconcile_are_stable(self):
        task_id = self.new_task()
        first, _ = self.answer(task_id, '先确认分母？', '六条。')
        second, _ = self.answer(task_id, '再确认覆盖率分母？', '八条。')
        self.finish(task_id)
        row = self.db.fetchone('SELECT * FROM research_memories WHERE task_id=?', (task_id,))
        record = json.loads(row['record_json'])
        self.assertEqual([i['clarification_id'] for i in record['answered_clarifications']],
            [first['clarification_id'], second['clarification_id']])
        before = hashlib.sha256(row['record_json'].encode()).hexdigest()
        for _ in range(3):
            self.conversations.reconcile(task_id)
        after = self.db.fetchone('SELECT * FROM research_memories WHERE task_id=?', (task_id,))
        self.assertEqual(before, hashlib.sha256(after['record_json'].encode()).hexdigest())
        self.assertEqual(row['search_text'], after['search_text'])
        self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM research_memories WHERE task_id=?',
            (task_id,))['n'], 1)

    def test_unanswered_question_and_suggestion_are_not_user_facts(self):
        task_id = self.new_task()
        self.tasks.request_clarification(task_id, 'PENDING-SYNTHETIC-QUESTION 是否适用？',
            ['criterion'], ['UNACCEPTED-SYNTHETIC-SUGGESTION'])
        self.finish(task_id, TaskStatus.partial)
        _, result, record = self.read_in_new_conversation(task_id)
        self.assertNotIn('PENDING-SYNTHETIC-QUESTION', result['content'])
        self.assertNotIn('UNACCEPTED-SYNTHETIC-SUGGESTION', result['content'])
        self.assertEqual(record.get('answered_clarifications', []), [])
        self.assertEqual(record['status'], 'partial')

    def test_new_record_does_not_expand_message_or_save_permissions(self):
        task_id = self.new_task(policy='allow')
        self.answer(task_id, '本次保存要求是什么？', '旧任务本次允许保存；不代表后续授权。')
        self.finish(task_id)
        target, _, record = self.read_in_new_conversation(task_id)
        self.assertEqual(self.db.get_task(target)['save_policy'], 'deny')
        self.assertEqual(self.db.list_task_evidence(target), [])
        with self.assertRaisesRegex(KeyError, 'outside task scope'):
            self.memory.read(target, HistoryRead(source_type='conversation', source_id=self.cid))
        message_id = record['answered_clarifications'][0]['answer_message_id']
        with self.assertRaisesRegex(KeyError, 'outside task scope or unavailable'):
            self.memory.read(target, HistoryRead(source_type='message', source_id=message_id))
        with self.assertRaisesRegex(ValueError, 'MEMORY_SOURCE'):
            self.memory.remember(target, RememberInput(slot='synthetic-save-policy',
                kind='explicit', content='允许保存',
                source_quote='旧任务本次允许保存；不代表后续授权。'))

    def test_legacy_history_is_readable_without_rewrite_or_invented_answers(self):
        task_id = self.new_task()
        self.answer(task_id, '合成旧记录中未保存的问题？', '合成旧回答。')
        self.finish(task_id)
        legacy = self.db.fetchone('SELECT * FROM research_memories WHERE task_id=?', (task_id,))
        record = json.loads(legacy['record_json'])
        record.pop('answered_clarifications', None)
        original = json.dumps(record, ensure_ascii=False)
        self.db.execute('UPDATE research_memories SET record_json=? WHERE task_id=?', (original, task_id))
        self.conversations.reconcile(task_id)
        _, result, recovered = self.read_in_new_conversation(task_id)
        self.assertEqual(result['content'], original)
        self.assertNotIn('answered_clarifications', recovered)
        self.assertEqual(recovered['task_id'], task_id)

    def test_scope_denied_feedback_only_repeats_successful_legal_readbacks(self):
        """正常工具拒绝跨会话读取；只复述本任务成功观察，不泄露新ID或内容。"""
        old = self.new_task()
        self.answer(old, '合成口径是什么？', '合成事实 SECRET-SIX。')
        self.finish(old)
        target = self.new_task(self.new_conversation(), '继续历史口径，不保存。')
        context = ToolContext(task_id=target, save_policy='deny', services={'memory':self.memory})
        spec = next(s for s in memory_specs() if s.name == 'read_memory')
        args = {'source_type':'research_history', 'source_id':old, 'limit':16000}
        allowed = execute_spec(spec, args, context)
        self.assertTrue(allowed.ok)
        self.db.save_action({'action_id':uuid4().hex, 'task_id':target, 'sequence':1,
            'tool_name':'read_memory', 'arguments':args, 'status':'succeeded',
            'observation':allowed.model_dump(mode='json')})
        # 失败读回和其他工具的ID不能冒充已成功合法入口。
        for sequence, tool, status in [(2, 'read_memory', 'failed'), (3, 'search_memory', 'succeeded')]:
            self.db.save_action({'action_id':uuid4().hex, 'task_id':target, 'sequence':sequence,
                'tool_name':tool, 'arguments':{'source_type':'research_history','source_id':'UNSEEN'},
                'status':status, 'observation':{'ok':status=='succeeded', 'data':{
                    'source_type':'research_history','source_id':'UNSEEN'}}})
        tables = ['conversation_messages','research_memories','memory_entries','memory_operations','task_inputs']
        before = {table:self.db.fetchall('SELECT * FROM '+table) for table in tables}
        denied = execute_spec(spec, {'source_type':'conversation', 'source_id':self.cid}, context)
        self.assertFalse(denied.ok)
        self.assertFalse(denied.retryable)
        self.assertEqual(denied.kind, 'execution_error')
        self.assertEqual(denied.data, {'error_code':'MEMORY_SCOPE_DENIED','scope_denied':True,
            'data_disclosed':False,'state_changed':False,'legal_readbacks':[
                {'source_type':'research_history','source_id':old,'next_offset':None}]})
        self.assertNotIn('SECRET-SIX', denied.model_dump_json())
        self.assertNotIn('UNSEEN', denied.model_dump_json())
        self.assertEqual(before, {table:self.db.fetchall('SELECT * FROM '+table) for table in tables})
        self.assertEqual(self.db.get_task(target)['save_policy'], 'deny')
        # 同一个工具仍能按原历史入口取得事实，不能直接读取旧消息。
        self.assertIn('SECRET-SIX', execute_spec(spec,args,context).data['content'])
        message = self.db.fetchone("SELECT message_id FROM conversation_messages WHERE task_id=? AND role='user' ORDER BY sequence_no DESC", (old,))
        missing = execute_spec(spec, {'source_type':'message','source_id':message['message_id']},context)
        self.assertFalse(missing.ok)
        self.assertEqual(missing.model_dump(), denied.model_dump())
        self.assertEqual(before, {table:self.db.fetchall('SELECT * FROM '+table) for table in tables})

    def test_message_scope_denial_hides_existence_and_preserves_state(self):
        """旧消息和不存在的消息同样拒绝，不探查其他会话或改变权限。"""
        old = self.new_task(prompt='合成旧消息 SECRET-MESSAGE。')
        self.finish(old)
        old_message = self.db.fetchone("SELECT message_id FROM conversation_messages WHERE task_id=? AND role='user'", (old,))['message_id']
        target = self.new_task(self.new_conversation(), '只读取当前范围，不保存。')
        context = ToolContext(task_id=target, save_policy='deny', services={'memory':self.memory})
        spec = next(s for s in memory_specs() if s.name == 'read_memory')
        tables = ['conversation_links','conversation_messages','research_memories',
                  'memory_entries','memory_operations','task_inputs']
        before = {table:self.db.fetchall('SELECT * FROM '+table) for table in tables}
        denied = [execute_spec(spec, {'source_type':'message','source_id':sid},context)
                  for sid in [old_message, uuid4().hex]]
        self.assertEqual(denied[0].model_dump(), denied[1].model_dump())
        self.assertFalse(denied[0].ok)
        self.assertFalse(denied[0].retryable)
        self.assertEqual(denied[0].data, {'error_code':'MEMORY_SCOPE_DENIED',
            'scope_denied':True,'data_disclosed':False,'state_changed':False,'legal_readbacks':[]})
        self.assertNotIn('SECRET-MESSAGE', denied[0].model_dump_json())
        self.assertNotIn(old_message, denied[0].model_dump_json())
        # 没有会话绑定的任务也不触发全库消息查询。
        queries = []
        fetchone = self.db.fetchone
        def tracked(sql, params=()):
            queries.append(sql)
            return fetchone(sql,params)
        with patch.object(self.db,'fetchone',side_effect=tracked):
            with self.assertRaisesRegex(KeyError,'outside task scope'):
                self.memory.read(uuid4().hex,HistoryRead(source_type='message',source_id=old_message))
        self.assertFalse(any('FROM conversation_messages' in sql for sql in queries))
        self.assertEqual(before, {table:self.db.fetchall('SELECT * FROM '+table) for table in tables})
        self.assertEqual(self.db.get_task(target)['save_policy'],'deny')

    def test_current_conversation_message_read_preserves_pagination(self):
        """当前会话的合法消息仍按原始内容和字符偏移分页。"""
        text = '合法当前消息：甲乙丙丁，不保存。' * 12
        target = self.new_task(prompt=text)
        message = self.db.fetchone("SELECT message_id FROM conversation_messages WHERE task_id=? AND role='user'", (target,))['message_id']
        context = ToolContext(task_id=target, save_policy='deny', services={'memory':self.memory})
        spec = next(s for s in memory_specs() if s.name == 'read_memory')
        observed = execute_spec(spec, {'source_type':'message','source_id':message,'offset':2,'limit':100},context)
        self.assertTrue(observed.ok)
        self.assertEqual(observed.data['source_id'],message)
        self.assertEqual(observed.data['content'],text[2:102])
        self.assertEqual(observed.data['next_offset'],102)
        self.assertFalse(observed.data['is_original_paper_evidence'])

    def test_scope_denied_without_prior_success_has_no_suggested_ids(self):
        """无先前成功观察时只反馈拒绝，不扫描或透露历史记录身份。"""
        old = self.new_task()
        self.finish(old)
        target = self.new_task(self.new_conversation())
        spec = next(s for s in memory_specs() if s.name == 'read_memory')
        context = ToolContext(task_id=target, save_policy='deny', services={'memory':self.memory})
        denied = execute_spec(spec, {'source_type':'conversation','source_id':self.cid},context)
        self.assertEqual(denied.data['legal_readbacks'], [])
        self.assertNotIn(old, denied.model_dump_json())
