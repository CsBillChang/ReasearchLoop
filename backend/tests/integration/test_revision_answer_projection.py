"""同版修订正文的API/Worker/档案隔离验证；供应商仅使用HTTP替身。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from integration import test_research_reflection as base
from integration.test_context_continuity import memory_payload
from integration.test_handoff_archive_delivery import HandoffArchiveScenario
from research_loop.agents.memory import estimate
from research_loop.agents.reflection import ReflectionService
from research_loop.api.dependencies import get_runtime_settings
from research_loop.domain.memory import HistoryRead
from research_loop.main import app
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.workers.executor import TaskExecutor


class RevisionProjectionScenario(base.ResearchReflectionScenario):
    """依据实际模型可见状态检验去重，替身不直接写核验或交付状态。"""

    def __init__(self, case):
        """记录修订后实际请求，保留生产HTTP/SSE编码路径。"""
        super().__init__('answer')
        self.case, self.linked_requests = case, []

    def factory(self, settings):
        """为同会话根任务使用SSE替身，Reviewer仍为普通JSON响应。"""
        return HandoffArchiveScenario.factory(self, settings)

    def handler(self, request):
        """核对真实待发送消息中的引用，不从数据库替模型补充隐藏正文。"""
        body = json.loads(request.content)
        if not base.is_reviewer_request(body):
            calls = {call['id']: call['function']['name'] for message in body['messages']
                     for call in message.get('tool_calls', [])}
            for message in body['messages']:
                if message.get('role') != 'tool' or calls.get(message.get('tool_call_id')) != 'revise_research':
                    continue
                value = json.loads(message['content'])
                if not value.get('ok'): continue
                context = memory_payload(body['messages'])
                target = value['data']['target']; item = target['items'][0]
                answer = context['current_task']['current_review_answer']
                self.case.assertNotIn('text', item)
                self.case.assertEqual(item['text_in'], 'current_task.current_review_answer.text')
                self.case.assertEqual(answer['text'], self.after)
                self.case.assertEqual((target['target_id'], target['content_hash']),
                                      (answer['target_id'], answer['content_hash']))
                self.case.assertEqual(context['current_task']['save_policy'], 'deny')
                self.case.assertEqual(value['data']['quality_status'], 'unreviewed')
                self.case.assertLessEqual(estimate({'messages': body['messages'], 'tools': body['tools']}), 115712)
                self.linked_requests.append(deepcopy(body))
        return super().handler(request)


class RevisionAnswerProjectionTests(unittest.TestCase):
    """默认临时SQLite；显式环境变量只允许researchloop_test_* MySQL。"""

    setUp = base.ResearchReflectionTests.setUp

    def run_case(self, recovery=False):
        """API→Worker→局部修订→后续请求→交付，并检查完整档案和成功收据。

        Args:
            recovery: 在修订成功落库后、下一次请求前模拟进程退出。
        """
        scenario = RevisionProjectionScenario(self)
        cid = ConversationService(self.db, self.settings).create('合成局部修订', uuid4().hex)['conversation_id']
        app.dependency_overrides[get_runtime_settings] = lambda: self.settings
        self.addCleanup(app.dependency_overrides.pop, get_runtime_settings, None)
        executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda *_: base.ScopedPDF())
        with TestClient(app) as client, patch.object(httpx.HTTPTransport, 'handle_request',
                side_effect=AssertionError('live HTTP forbidden')):
            created = client.post('/api/tasks', json={'conversation_id': cid, 'prompt': '核对当前回答，禁止保存。',
                'save_policy': 'deny', 'attachments': [{'input_ref': 'input-A', 'paper_id': 'paper-A',
                                                       'local_path': str(self.root / 'A.pdf')}]})
            self.assertEqual(created.status_code, 202, created.text)
            tid = created.json()['task_id']
            saved = {}
            if recovery:
                original = CheckpointStore.save; interrupted = []

                def stop_after_revision(store, task_id, state):
                    """只在已成功的修订行动写入后中断，不伪造工具或供应商结果。"""
                    if task_id == tid and not interrupted and any(
                            action['tool_name'] == 'revise_research' and action['status'] == 'succeeded'
                            for action in self.db.list_actions(tid)):
                        interrupted.append(task_id)
                        raise SystemExit('AFTER_REVISION_ACTION')
                    return original(store, task_id, state)

                with patch.object(CheckpointStore, 'save', stop_after_revision), self.assertRaises(SystemExit):
                    executor.execute(tid)
                saved = {row['call_id']: row for row in self.db.list_model_calls(tid)}
                self.db.execute('UPDATE tasks SET lease_until=? WHERE task_id=?', ('2000-01-01T00:00:00Z', tid))
            executor.execute(tid, 'restored' if recovery else 'direct')
            view = client.get('/api/tasks/' + tid).json()
            self.assertEqual(view['status'], 'completed', view)
            self.assertEqual(view['result']['answer_body_markdown'], scenario.after)
            self.assertTrue(scenario.linked_requests)
            self.assertEqual(sum(a['tool_name'] == 'revise_research' for a in self.db.list_actions(tid)), 1)
            self.assertTrue(all(a['status'] == 'succeeded' for a in self.db.list_actions(tid)))
            for row in self.db.list_model_calls(tid):
                if row['call_id'] in saved: self.assertEqual(row, saved[row['call_id']])
            calls_before = len(scenario.requests)
            self.db.update_task(tid, status='queued', result_json=None)
            executor.execute(tid, 'replayed')
            self.assertEqual(len(scenario.requests), calls_before)
            self.assertEqual(client.get('/api/tasks/' + tid).json()['result'], view['result'])

        snapshot = ReflectionService(self.db).snapshot(tid)
        target = next(iter(snapshot['research_targets'].values()))
        self.assertEqual([v['items'][0]['text'] for v in target['versions']], [scenario.before, scenario.after])
        self.assertEqual(target['attempts'][0]['result']['findings'][0]['status'], 'contradicted')
        self.assertEqual(target['attempts'][1]['result']['findings'][0]['status'], 'supported')
        archive_rows = self.db.fetchall('SELECT * FROM context_archives WHERE task_id=? ORDER BY sequence_no', (tid,))
        linked = [row for row in archive_rows if any(
            message.get('role') == 'tool'
            and (json.loads(message['content']).get('data') or {}).get('target', {}).get('items', [{}])[0].get('text_in')
                == 'current_task.current_review_answer.text'
            for message in json.loads(row['projected_json']))]
        self.assertTrue(linked)
        for row in linked:
            # 经正式分页服务读回完整档案，既有判断、旧工具对象和原文都不能随投影消失。
            memory = MemoryService(self.db); offset = 0; chunks = []
            while True:
                page = memory.read(tid, HistoryRead(source_type='context_archive', source_id=row['snapshot_id'],
                                                   offset=offset, limit=16000))
                chunks.append(page['content'])
                if page['next_offset'] is None: break
                offset = page['next_offset']
            self.assertEqual(''.join(chunks), row['original_json'])
            archive = json.loads(''.join(chunks))
            current_revision = [v for v in archive['tool_results'].values()
                                if v.get('data', {}).get('revision')]
            self.assertTrue(current_revision)
            self.assertEqual(current_revision[-1]['data']['target']['items'][0]['text'], scenario.after)
            self.assertIn(scenario.before, row['original_json'])
            self.assertIn('The method uses language prefix embeddings.', row['original_json'])
            self.assertIn('contradicted', row['original_json'])
            with self.assertRaises(KeyError):
                memory.read('foreign-task', HistoryRead(source_type='context_archive', source_id=row['snapshot_id']))
        self.assertEqual(ReflectionService(self.db).snapshot(tid), snapshot)
        self.assertEqual(self.db.get_task(tid)['save_policy'], 'deny')
        self.assertFalse(self.db.fetchall('SELECT draft_id FROM drafts WHERE task_id=?', (tid,)))
        self.assertFalse(self.db.fetchall('SELECT * FROM save_receipts'))
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))

    def test_revision_projection_continues_and_preserves_archive(self):
        """当前正文与工具结果去重后仍可定向复审并完成公开交付。"""
        self.run_case()

    def test_revision_action_restart_does_not_repeat_successful_requests(self):
        """修订后中断恢复保留同版引用，原成功收据不重发，行动不重复。"""
        self.run_case(recovery=True)
