"""当前核验对象及程序边界的API/数据库/上下文验证；模型只用HTTP替身。"""
from copy import deepcopy
import hashlib
import json
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient
from integration import test_research_reflection as base
from research_loop.agents.graph import initial_state
from research_loop.agents.memory import ContextService
from research_loop.agents.reflection import ReflectionService
from research_loop.api.dependencies import get_runtime_settings
from research_loop.domain.memory import HistoryRead
from research_loop.main import app
from research_loop.providers.deepseek import validate_messages
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.storage.memory import dump
from research_loop.tools.memory import specs as memory_specs


class CurrentReviewAnswerTests(unittest.TestCase):
    """默认SQLite；REFLECTION_TEST_DATABASE_URL只接受researchloop_test_专用库。"""

    setUp = base.ResearchReflectionTests.setUp

    def referenced_task(self, explicit=True):
        """从真实会话API建任务，完整正文以明确hash绑定，不依赖论文名称。"""
        conversations = ConversationService(self.db, self.settings)
        cid = conversations.create('合成当前核验对象', uuid4().hex)['conversation_id']
        body = ('方法条件与限制。'*300)[:1929]
        with conversations.store.transaction() as tx:
            mid = conversations.add_message(tx, cid, 'fixture-history', uuid4().hex, 'assistant', body)
        prompt = '核对已有回答，禁止保存。' + ('原回答SHA256=' + hashlib.sha256(body.encode()).hexdigest() if explicit else '')
        app.dependency_overrides[get_runtime_settings] = lambda: self.settings
        self.addCleanup(app.dependency_overrides.pop, get_runtime_settings, None)
        with TestClient(app) as client:
            response = client.post('/api/tasks', json={'prompt': prompt, 'conversation_id': cid, 'save_policy': 'deny'})
            self.assertEqual(response.status_code, 202, response.text)
            tid = response.json()['task_id']
        return tid, mid, body

    def test_bound_answer_survives_capacity_pressure_without_paging_again(self):
        """旧大型观察先外置，短核验对象完整保留；原件和实际容量都可读回验证。"""
        tid, mid, body = self.referenced_task()
        state = initial_state(tid, self.db.get_task(tid)['prompt'], 'deny')
        args = HistoryRead(source_type='message', source_id=mid, offset=0, limit=6000)
        page = MemoryService(self.db).read(tid, args)
        state['messages'] += [
            {'role': 'assistant', 'content': '', 'reasoning_content': '旧推理保持', 'tool_calls': [
                {'id': 'old', 'type': 'function', 'function': {'name': 'get_profile', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'old', 'content': dump({'ok': True, 'data': {'old': '无关旧资料'*40000}})},
            {'role': 'assistant', 'content': '', 'reasoning_content': '当前推理保持', 'tool_calls': [
                {'id': 'current', 'type': 'function', 'function': {'name': 'read_memory', 'arguments': dump(args.model_dump())}}]},
            {'role': 'tool', 'tool_call_id': 'current', 'content': dump({'ok': True, 'data': page})}]
        original = deepcopy(state['messages'])
        tools = [s.schema() for s in memory_specs() if s.name in {'read_memory', 'get_profile'}]
        context = ContextService(self.db, MemoryService(self.db), self.settings)
        projected, metrics = context.prepare(state, state['messages'], tools)
        header = next(m for m in projected if m.get('role') == 'system' and m.get('content', '').startswith('[ResearchLoop Memory v1]'))
        current = json.loads(header['content'][header['content'].index('{'):])['current_task']
        self.assertEqual(current['current_review_answer']['text'], body)
        self.assertEqual(current['current_review_answer']['source_id'], mid)
        self.assertFalse(current['current_review_answer']['is_original_paper_evidence'])
        self.assertFalse(current['current_review_answer']['save_permission_inherited'])
        data = json.loads(next(m['content'] for m in projected if m.get('tool_call_id') == 'current'))['data']
        self.assertEqual(data['content_in'], 'current_task.current_review_answer.text')
        self.assertIsNone(data['next_offset'])
        self.assertTrue(metrics['current_review_answer_retained'])
        self.assertLessEqual(metrics['after'], metrics['input_capacity'])
        self.assertEqual(state['messages'], original)
        self.assertEqual(projected[-2], original[-2]); validate_messages(projected)
        archived = self.db.fetchone('SELECT original_json FROM context_archives WHERE snapshot_id=?', (metrics['snapshot_id'],))
        self.assertEqual(json.loads(archived['original_json'])['messages'], original)
        self.assertEqual(MemoryService(self.db).read(tid, args), page)
        self.assertEqual(self.db.list_model_calls(tid), [])
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))

    def test_unreferenced_and_foreign_history_are_not_selected(self):
        """没有明确hash或实际授权读回，不把最近回答默认当核验对象。"""
        tid, mid, _ = self.referenced_task(False)
        ctx = ContextService(self.db, MemoryService(self.db), self.settings)
        self.assertIsNone(ctx._current_review_answer(tid, {}))
        other, other_mid, _ = self.referenced_task()
        self.assertIsNone(ConversationService(self.db, self.settings).answer_message(other, mid))
        self.assertNotEqual(mid, other_mid)

    def test_local_revision_retained_and_reviewer_receives_program_scope(self):
        """API→Worker→Reviewer实际消息中含程序检查；当前局部版本优先于旧回答。"""
        tid, _, scenario = base.ResearchReflectionTests.run_case(self, 'answer')
        reviewer = [r for r in scenario.requests if base.is_reviewer_request(r)]
        self.assertTrue(reviewer)
        for request in reviewer:
            value = json.loads(request['messages'][1]['content'].split('\n', 1)[1])
            self.assertEqual(value['execution_checks']['save_policy'], 'deny')
            self.assertEqual(value['execution_checks']['successful_save_actions'], [])
            self.assertEqual(value['execution_checks']['unrelated_text_preservation'], 'not_checked')
        state = initial_state(tid, self.db.get_task(tid)['prompt'], 'deny')
        ctx = ContextService(self.db, MemoryService(self.db), self.settings)
        from research_loop.services.coverage import CoverageService
        current = CoverageService(self.db).current_context(tid, state)
        answer = ctx._current_review_answer(tid, current)
        stored = next(iter(ReflectionService(self.db).snapshot(tid)['research_targets'].values()))['current']
        self.assertEqual(answer['text'], stored['items'][0]['text'])
        self.assertEqual(answer['content_hash'], stored['content_hash'])
        self.assertEqual(answer['text'], scenario.after)
        # 决策时使用的投影与持久完整历史不同；恢复后仍以原receipt重算可见性。
        from research_loop.services.research_progress import evidence_progress
        for family in [tid] + [c['subtask_id'] for c in self.db.list_subtasks(tid)]:
            previous = []
            for action in self.db.list_actions(family):
                increment = (action.get('observation') or {}).get('data', {}).get('research_increment')
                if increment:
                    receipt = self.db.get_model_call(family, increment['request_sequence'])
                    expected = evidence_progress(previous, action['tool_name'], action['arguments'],
                        action['observation'], json.loads(receipt['request_json'])['messages'])
                    self.assertEqual(increment['already_visible_count'], expected['already_visible_count'])
                    self.assertTrue(increment['visibility_known'])
                previous.append(action)
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
