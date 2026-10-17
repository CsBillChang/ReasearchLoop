"""交接模式档案读回→Reviewer/交付及异常保留的隔离API链路。"""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from integration import test_research_reflection as base
from research_loop.agents.memory import ContextService
from research_loop.services.conversations import ConversationService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.memory import dump
from research_loop.storage.mysql import _now
from research_loop.workers.executor import TaskExecutor
from research_loop.providers.deepseek import DeepSeekProvider


class HandoffArchiveScenario(base.ResearchReflectionScenario):
    """合成模型依实际观察选择档案页、核验和修订，不替服务写结果。"""

    def __init__(self, mode):
        """选择正常分页、同范围重复、外来档案或消息读回。"""
        super().__init__('answer')
        self.navigation_mode = mode
        self.archive_id = uuid4().hex
        self.message_id = None
        self.handoff_requests = []

    def factory(self, settings):
        """同会话根任务走真实SSE组装，Reviewer继续走普通JSON适配。"""
        def transport(request):
            """将同一合成决定编码为供应商实际请求要求的传输形状。"""
            response = self.handler(request)
            if not json.loads(request.content).get('stream'):
                return response
            raw = response.json(); choice = raw['choices'][0]
            calls = [{**call, 'index': index} for index, call in enumerate(choice['message'].get('tool_calls', []))]
            chunks = [{'id': raw['id'], 'choices': [{'index': 0, 'finish_reason': None,
                       'delta': {'role': 'assistant', 'content': '', 'tool_calls': calls}}]},
                      {'id': raw['id'], 'choices': [{'index': 0, 'finish_reason': choice['finish_reason'], 'delta': {}}],
                       'usage': raw['usage']}]
            text = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks) + 'data: [DONE]\n\n'
            return httpx.Response(200, text=text, headers={'content-type': 'text/event-stream'})
        return DeepSeekProvider('fixture', client=httpx.Client(transport=httpx.MockTransport(transport)))

    def handler(self, request):
        """重复原文读回触发真实图路由，再使用实际可见的记忆工具。"""
        body = json.loads(request.content)
        names = {t['function']['name'] for t in body.get('tools', [])}
        observations = [json.loads(m['content']) for m in body['messages'] if m['role'] == 'tool']
        calls = [c['function']['name'] for m in body['messages'] for c in m.get('tool_calls', [])]
        if not base.is_reviewer_request(body):
            if calls.count('read_paper') == 1:
                self.requests.append(body)
                return self.response('read_paper', {'target_ref': 'input-A', 'start_page': 1, 'page_count': 2})
            if 'read_memory' in names and 'read_paper' not in names:
                self.handoff_requests.append(deepcopy(body))
                reads = [o['data'] for o in observations if o.get('data', {}).get('source_type') in {'context_archive', 'message'}]
                required = 1 if self.navigation_mode == 'message' else 2
                if len(reads) < required:
                    self.requests.append(body)
                    kind = 'message' if self.navigation_mode == 'message' else 'context_archive'
                    offset = 0 if not reads or self.navigation_mode == 'repeat' else reads[-1]['next_offset']
                    return self.response('read_memory', {'source_type': kind,
                        'source_id': self.message_id if kind == 'message' else self.archive_id,
                        'offset': offset, 'limit': 100})
        return super().handler(request)


class HandoffArchiveDeliveryTests(unittest.TestCase):
    """默认SQLite，显式REFLECTION_TEST_DATABASE_URL才进入隔离MySQL。"""

    setUp = base.ResearchReflectionTests.setUp

    def run_case(self, mode='archive', recovery=False):
        """新任务走正式API/Worker；缓存页、供应商响应和故障均为显式替身。"""
        from research_loop.api.dependencies import get_runtime_settings
        from research_loop.main import app
        from research_loop.services.tasks import TaskService
        scenario = HandoffArchiveScenario(mode)
        conversation = ConversationService(self.db, self.settings)
        cid = conversation.create('synthetic source reply', uuid4().hex)['conversation_id']
        with conversation.store.transaction() as tx:
            scenario.message_id = conversation.add_message(tx, cid, 'imported-fixture', uuid4().hex,
                                                          'assistant', scenario.before)
        app.dependency_overrides[get_runtime_settings] = lambda: self.settings
        self.addCleanup(app.dependency_overrides.pop, get_runtime_settings, None)
        with TestClient(app) as client:
            created = client.post('/api/tasks', json={'conversation_id': cid,
                'prompt': '定向核验已有回答，禁止保存。原回答SHA-256=' + hashlib.sha256(scenario.before.encode()).hexdigest(),
                'save_policy': 'deny', 'attachments': [{'input_ref': 'input-A', 'paper_id': 'paper-A',
                                                       'local_path': str(self.root/'A.pdf')}]})
            self.assertEqual(created.status_code, 202, created.text)
            tid = created.json()['task_id']
            content = dump({'answer': scenario.before, 'padding': '档案只读内容'*500})
            from research_loop.domain.tasks import TaskCreate
            owner = TaskService(self.db, self.settings).create(TaskCreate(prompt='外部任务', save_policy='deny')).task_id if mode == 'foreign' else tid
            self.db.execute('INSERT INTO context_archives(snapshot_id,task_id,sequence_no,original_json,projected_json,metrics_json,created_at) VALUES(?,?,?,?,?,?,?)',
                (scenario.archive_id, owner, 0, content, '[]', '{}', _now()))
            executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda s, f: base.ScopedPDF())
            prepare = ContextService.prepare
            interrupted = []

            def fault(context, state, messages, tools):
                """只在真实交接工具已收缩时注入容量故障，检验保留路径。"""
                if mode == 'capacity' and any('当前进入研究进展交接' in m.get('content', '') for m in messages):
                    raise ValueError('CONTEXT_CAPACITY: controlled handoff failure')
                return prepare(context, state, messages, tools)

            def stop(store, task_id, state):
                """在成功导航响应后、行动执行前退出；恢复必须复用原收据。"""
                if (not interrupted and state.get('research_handoff') and
                        any(c['name'] == 'read_memory' for c in state.get('pending_tool_calls', []))):
                    interrupted.append(task_id)
                    raise SystemExit('controlled archive receipt boundary')
                return save(store, task_id, state)

            saved_receipts = {}
            if recovery:
                save = CheckpointStore.save
                with patch.object(CheckpointStore, 'save', stop), self.assertRaises(SystemExit):
                    executor.execute(tid)
                saved_receipts = {r['call_id']: r for r in self.db.list_model_calls(tid) if r['status'] == 'succeeded'}
                self.db.execute('UPDATE tasks SET lease_until=? WHERE root_task_id=?', ('2000-01-01T00:00:00Z', tid))
            with patch.object(ContextService, 'prepare', fault):
                executor.execute(tid)
            view = client.get('/api/tasks/' + tid).json()
            self.assertEqual(view['status'], 'partial' if mode in {'repeat', 'foreign', 'capacity'} else 'completed', view)
            result = view['result']
            self.assertEqual(self.db.fetchone('SELECT content FROM conversation_messages WHERE message_id=?',
                                             (scenario.message_id,))['content'], scenario.before)
            if mode in {'repeat', 'foreign', 'capacity'}:
                self.assertEqual(result['answer_body_markdown'], scenario.before)
                self.assertEqual(result['retained_answer_source']['source_id'], scenario.message_id)
                self.assertEqual(result['review_status'], 'unreviewed')
                self.assertTrue(result['processing_summary'])
                self.assertFalse(self.db.list_subtasks(tid))
            else:
                self.assertEqual(result['answer_body_markdown'], scenario.after)
                self.assertEqual(sum(base.is_reviewer_request(r) for r in scenario.requests), 2)
                self.assertTrue(scenario.handoff_requests)
            if mode == 'capacity':
                self.assertIn('CONTEXT_CAPACITY', result['processing_summary'])
            reads = [a for a in self.db.list_actions(tid) if a['tool_name'] == 'read_memory']
            if mode == 'archive':
                self.assertEqual([a['arguments']['offset'] for a in reads], [0, 100])
                self.assertTrue(all(a['status'] == 'succeeded' for a in reads))
            elif mode == 'repeat':
                self.assertEqual([a['arguments']['offset'] for a in reads], [0, 0])
                self.assertIn('no progress', result['reason'])
            elif mode == 'foreign':
                self.assertEqual(reads[0]['status'], 'failed')
            for r in self.db.list_model_calls(tid):
                if r['call_id'] in saved_receipts:
                    self.assertEqual(r, saved_receipts[r['call_id']])
            before = len(scenario.requests)
            self.db.update_task(tid, status='queued', result_json=None)
            executor.execute(tid, 'replayed')
            self.assertEqual(len(scenario.requests), before)
            self.assertEqual(TaskService(self.db).view(self.db.get_task(tid)).result.model_dump(mode='json'), result)
            self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
            self.assertFalse(self.db.fetchall('SELECT * FROM drafts WHERE task_id=?', (tid,)))

    def test_archive_pages_continue_to_review_revision_and_delivery(self):
        """交接模式下两页档案读回后仍能进入独立核验和局部修订。"""
        self.run_case()

    def test_message_readback_can_continue_to_review(self):
        """同会话此前回答通过现有message工具读回，不被当作新原文。"""
        self.run_case('message')

    def test_repeated_archive_range_stops_with_retained_answer(self):
        """重复区间不刷新交接，原回答和停止原因仍交付。"""
        self.run_case('repeat')

    def test_foreign_archive_cannot_extend_handoff(self):
        """跨任务档案拒绝且停止；明确引用的本会话原回答仍可保留。"""
        self.run_case('foreign')

    def test_capacity_failure_delivers_exact_existing_answer(self):
        """未创建核验对象即失败时，也能从明确引用保留原回答。"""
        self.run_case('capacity')

    def test_navigation_receipt_recovery_does_not_resend(self):
        """档案导航前中断后恢复，成功请求及使用量完全复用。"""
        self.run_case(recovery=True)
