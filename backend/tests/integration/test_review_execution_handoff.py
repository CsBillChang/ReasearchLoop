"""Reviewer组件与Supervisor调度分层验证；供应商HTTP和PDF均为显式替身。"""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from integration import test_research_reflection as base
from integration.test_handoff_archive_delivery import HandoffArchiveScenario
from research_loop.agents.reflection import ReflectionService
from research_loop.api.dependencies import get_runtime_settings
from research_loop.domain.reflection import ResearchReviewInput
from research_loop.domain.tasks import TaskCreate
from research_loop.main import app
from research_loop.services.conversations import ConversationService
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.workers.executor import TaskExecutor


class ReferenceScenario(HandoffArchiveScenario):
    """按真实观察选择核验与局部修改，正文参数始终由服务器解析。"""

    def __init__(self, flow='direct'):
        """指定合成故障，不替执行器写入行动或核验结论。"""
        super().__init__('message')
        self.flow, self.injected = flow, False
        if flow == 'partial':
            self.mode = 'missing'

    def handler(self, request):
        """返回正常或混合工具批次，之后根据实际失败观察修正动作。"""
        body = json.loads(request.content)
        names = {t['function']['name'] for t in body.get('tools', [])}
        calls = [c['function']['name'] for m in body['messages'] for c in m.get('tool_calls', [])]
        handoff = 'read_memory' in names and 'read_paper' not in names
        if handoff and 'record_research_progress' not in names:
            hints = [m['content'] for m in body['messages'] if m.get('role') == 'system'
                     and '当前进入研究进展交接' in m.get('content', '')]
            assert all('先用record_research_progress' not in hint for hint in hints)
        if not base.is_reviewer_request(body) and self.flow not in {'direct', 'partial'}:
            if calls.count('read_paper') == 1:
                self.requests.append(body)
                return self.response('read_paper', {'target_ref': 'input-A', 'start_page': 1, 'page_count': 2})
            if handoff and (not self.injected or self.flow == 'repeat'):
                self.injected = True
                self.requests.append(deepcopy(body))
                if self.flow == 'parameters':
                    return self.response('review_research', {'target_type': 'answer',
                        'verification_questions': ['方法是否使用语言前缀？']})
                unknown = {'id': uuid4().hex, 'type': 'function', 'function': {'name': 'retrieve_evidence',
                           'arguments': json.dumps({'query': 'unavailable search'})}}
                raw = self.response('read_memory', {'source_type': 'message', 'source_id': self.message_id,
                                                   'offset': 0, 'limit': 100}).json()
                if self.flow == 'mixed':
                    raw['choices'][0]['message']['tool_calls'].append(unknown)
                elif self.flow == 'reverse':
                    raw['choices'][0]['message']['tool_calls'].insert(0, unknown)
                elif self.flow == 'defer':
                    dependent = {'id': uuid4().hex, 'type': 'function', 'function': {'name': 'review_research',
                        'arguments': json.dumps({'target_type': 'answer', 'reason': '核对方法',
                            'verification_questions': ['方法是否使用语言前缀？']})}}
                    raw['choices'][0]['message']['tool_calls'] = [unknown, dependent]
                else:
                    raw['choices'][0]['message']['tool_calls'] = [unknown]
                return __import__('httpx').Response(200, json=raw)
        response = base.ResearchReflectionScenario.handler(self, request)
        raw = response.json()
        for call in raw['choices'][0]['message'].get('tool_calls', []):
            if call['function']['name'] == 'review_research':
                args = json.loads(call['function']['arguments'])
                args.pop('answer_markdown', None)
                call['function']['arguments'] = json.dumps(args)
        return __import__('httpx').Response(200, json=raw)


def create_referenced_task(case, scenario):
    """通过真实服务建立会话与明确引用，不把历史当作原文证据。"""
    conversations = ConversationService(case.db, case.settings)
    cid = conversations.create('reference fixture', uuid4().hex)['conversation_id']
    with conversations.store.transaction() as tx:
        scenario.message_id = conversations.add_message(tx, cid, 'fixture-history', uuid4().hex,
                                                       'assistant', scenario.before)
    return {'conversation_id': cid, 'prompt': '核对当前回答，禁止保存。hash=' +
            hashlib.sha256(scenario.before.encode()).hexdigest(), 'save_policy': 'deny',
            'attachments': [{'input_ref': 'input-A', 'paper_id': 'paper-A', 'local_path': str(case.root/'A.pdf')}]}


class ReviewerComponentTests(unittest.TestCase):
    """直接启动独立Reviewer组件，完全跳过Supervisor规划。"""

    setUp = base.ResearchReflectionTests.setUp

    def run_component(self, insufficient=False):
        """仍使用正式子任务、收据、finish_review及核验服务，不伪造通过结果。"""
        scenario = ReferenceScenario('partial' if insufficient else 'direct')
        payload = create_referenced_task(self, scenario)
        tid = TaskService(self.db, self.settings).create(TaskCreate(**payload)).task_id
        owner = self.db.claim_task('component-worker', task_id=tid)
        papers = PaperService(self.db, None, base.ScopedPDF())
        pages = papers.read(tid, 'input-A', attachments={'input-A': payload['attachments'][0]})['pages']
        executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda *_: base.ScopedPDF())
        args = ResearchReviewInput(target_type='answer', reason='核对当前明确引用回答',
            verification_questions=['方法是否使用语言前缀？'], evidence_ids=[p['evidence_id'] for p in pages])
        service = ReflectionService(self.db, owner=('component-worker', owner['lease_token']))
        execute = lambda attempt, target, refs, questions: executor._review_research_once(
            tid, attempt, target, refs, questions, 'component-worker', owner['lease_token'], self.settings, papers)
        result = service.review_research(tid, args, execute)
        before = len(scenario.requests)
        repeated = service.review_research(tid, args, execute)
        self.assertEqual(repeated['review_id'], result['review_id'])
        self.assertEqual(len(scenario.requests), before)
        self.assertTrue(all(base.is_reviewer_request(r) for r in scenario.requests))
        target = json.loads(scenario.requests[0]['messages'][1]['content'].split('\n', 1)[1])['target']
        self.assertEqual(target['items'][0]['text'], scenario.before)
        self.assertEqual(target['origin']['answer_source']['source_id'], scenario.message_id)
        self.assertEqual(result['findings'][0]['status'], 'insufficient' if insufficient else 'contradicted')
        self.assertEqual(result['status'], 'issues')
        self.assertEqual(len(self.db.list_subtasks(tid)), 1)
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))

    def test_direct_reviewer_with_counterevidence(self):
        """反证是组件的合法输出，不要求其批准当前回答。"""
        self.run_component()

    def test_direct_reviewer_with_insufficient_evidence(self):
        """证据不足也能结构化返回，不能伪装执行失败或反证。"""
        self.run_component(True)


class SupervisorHandoffTests(unittest.TestCase):
    """另验API→数据库→Worker→Supervisor→Reviewer及正常/部分交付。"""

    setUp = base.ResearchReflectionTests.setUp

    def run_flow(self, flow='mixed', recovery=False, restrict_progress=False):
        """恢复成功响应时保留原工具范围，实际模型传输仅为MockTransport。"""
        scenario = ReferenceScenario(flow)
        payload = create_referenced_task(self, scenario)
        allowed = {'set_research_plan', 'read_paper', 'retrieve_evidence', 'read_memory', 'review_research',
                   'revise_research', 'close_research_issue', 'report_task_gap', 'finish_task'} if restrict_progress else None
        executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda *_: base.ScopedPDF(),
                                supervisor_tools=allowed)
        app.dependency_overrides[get_runtime_settings] = lambda: self.settings
        self.addCleanup(app.dependency_overrides.pop, get_runtime_settings, None)
        with TestClient(app) as client:
            response = client.post('/api/tasks', json=payload)
            self.assertEqual(response.status_code, 202, response.text)
            tid = response.json()['task_id']
            saved, interrupted = {}, []
            original = CheckpointStore.save

            def stop(store, task_id, state):
                """在混合响应成功落收据、行动尚未执行时中断。"""
                if (not interrupted and state.get('pending_tool_calls') and
                        any(c['name'] == 'retrieve_evidence' for c in state['pending_tool_calls'])):
                    interrupted.append(True)
                    raise SystemExit('controlled mixed receipt boundary')
                return original(store, task_id, state)

            if recovery:
                with patch.object(CheckpointStore, 'save', stop), self.assertRaises(SystemExit):
                    executor.execute(tid)
                saved = {r['call_id']: r for r in self.db.list_model_calls(tid) if r['status'] == 'succeeded'}
                self.db.execute('UPDATE tasks SET lease_until=? WHERE root_task_id=?', ('2000-01-01T00:00:00Z', tid))
            executor.execute(tid)
            view = client.get('/api/tasks/' + tid).json()
            self.assertEqual(view['status'], 'partial' if flow in {'repeat', 'partial'} else 'completed', view)
            result = view['result']
            self.assertEqual(result['answer_body_markdown'], scenario.before if flow in {'repeat', 'partial'} else scenario.after)
            self.assertIsNone(result['saved_path'])
            self.assertEqual(view['save_policy'], 'deny')
            self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
            self.assertFalse(self.db.fetchall('SELECT * FROM drafts WHERE task_id=?', (tid,)))
            actions = self.db.list_actions(tid)
            unavailable = [a for a in actions if a['tool_name'] == 'retrieve_evidence']
            for action in unavailable:
                self.assertFalse(action['observation']['data']['tool_executed'])
                self.assertFalse(action['observation']['data']['budget_counted'])
            if flow in {'mixed', 'reverse'}:
                reads = [a for a in actions if a['tool_name'] == 'read_memory']
                self.assertEqual(len(reads), 1)
                self.assertEqual(reads[0]['status'], 'succeeded')
            if flow == 'defer':
                self.assertTrue(any(a['observation'].get('data', {}).get('error_code') == 'TOOL_DEPENDENCY_UNRESOLVED' for a in actions))
            reviews = [a for a in actions if a['tool_name'] == 'review_research' and a['status'] == 'succeeded']
            self.assertEqual(len(reviews), 0 if flow == 'repeat' else (1 if flow == 'partial' else 2))
            self.assertTrue(all('answer_markdown' not in a['arguments'] for a in reviews))
            if flow == 'repeat':
                self.assertFalse(self.db.list_subtasks(tid))
                self.assertIn('unavailable', result['processing_summary'])
                self.assertEqual(result['review_status'], 'unreviewed')
            elif flow == 'partial':
                self.assertEqual(result['research_reviews'][0]['quality_status'], 'issues')
            last = self.db.fetchone('SELECT content FROM conversation_messages WHERE task_id=? AND role=? ORDER BY sequence_no DESC LIMIT 1', (tid, 'assistant'))
            self.assertEqual(last['content'], result['answer_markdown'])
            for row in self.db.list_model_calls(tid):
                if row['call_id'] in saved:
                    self.assertEqual(row, saved[row['call_id']])
            count = len(scenario.requests)
            self.db.update_task(tid, status='queued', result_json=None)
            executor.execute(tid, 'restored')
            self.assertEqual(len(scenario.requests), count)
            self.assertEqual(client.get('/api/tasks/' + tid).json()['result'], result)

    def test_direct_reference_does_not_require_copying_answer(self):
        """Supervisor可按选择直接交接已有回答，之后完成局部修订。"""
        self.run_flow('direct')

    def test_mixed_tools_keep_readback_and_review(self):
        """原失败形状：合法读回与不可用检索混合，后续仍能核验。"""
        self.run_flow('mixed')

    def test_unavailable_predecessor_keeps_independent_read(self):
        """非法工具在前也不吞掉独立只读导航。"""
        self.run_flow('reverse')

    def test_all_unavailable_can_choose_review_next(self):
        """整批非法的观察交回模型，下一轮改为合法核验。"""
        self.run_flow('recover')

    def test_repeated_unavailable_ends_with_existing_answer(self):
        """拒绝反馈不算研究进展；重复非法操作准确partial结束。"""
        self.run_flow('repeat')

    def test_dependency_is_deferred_until_next_decision(self):
        """前序范围失败后，核验留待下一轮明确重新选择。"""
        self.run_flow('defer')

    def test_parameter_error_can_be_corrected(self):
        """缺少reason的具体参数错误可反馈，不直接吞掉交接。"""
        self.run_flow('parameters')

    def test_uncertainty_is_valid_partial_delivery(self):
        """保留不足与原正文即可交付，不要求所有Reviewer结论通过。"""
        self.run_flow('partial')

    def test_mixed_success_receipt_recovery_never_resends(self):
        """成功响应后退出，恢复使用原收据、范围和request_id。"""
        self.run_flow('mixed', recovery=True)

    def test_restricted_handoff_prompt_matches_actual_tools(self):
        """验收收缩掉进展工具时，提示也不再要求调用它；日常权限不变。"""
        self.run_flow('mixed', restrict_progress=True)

