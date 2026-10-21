"""交接→八意见空判断→正常/异常收尾的真实API与隔离数据库链路。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from integration import test_research_reflection as base
from integration.test_handoff_archive_delivery import HandoffArchiveScenario
from research_loop.agents.memory import ContextService
from research_loop.agents.reflection import ReflectionService
from research_loop.services.conversations import ConversationService
from research_loop.services.tasks import TaskService
from research_loop.workers.executor import TaskExecutor
from research_loop.api.dependencies import get_runtime_settings
from research_loop.main import app
from fastapi.testclient import TestClient


class NullVerdictScenario(HandoffArchiveScenario):
    """供应商替身提交第六条空判断；Supervisor只能按实际失败观察收尾。"""

    def __init__(self, mode='finish'):
        """选择正常收尾、收尾参数错误、容量异常或中断恢复。"""
        super().__init__('archive'); self.case_mode = mode; self.review_calls = 0
        self.closeout_requests = []; self.summary = '核验执行失败，保留当前回答；未确认事项不自动批准。'

    def handler(self, request):
        """重用真实工具观察和SSE传输，不直接构造数据库通过结果。"""
        body = json.loads(request.content)
        observations = [json.loads(m['content']) for m in body['messages'] if m['role'] == 'tool']
        if base.is_reviewer_request(body):
            self.requests.append(body); self.review_calls += 1
            if self.review_calls < 4:
                start, count = [(1, 1), (2, 1), (1, 2)][self.review_calls-1]
                return self.response('read_paper', {'target_ref': 'input-A', 'start_page': start, 'page_count': count})
            rows = [{'item_id': 'answer', 'status': 'supported', 'basis': 'scope_disclosure',
                'reason': f'合成范围说明{n}', 'evidence_ids': ['page:paper-A:1'], 'suggested_action': 'keep'} for n in range(8)]
            rows[5]['status'] = None
            return self.response('finish_review', {'findings': rows})
        failed = next((o['data'] for o in reversed(observations) if o.get('data', {}).get('status') == 'execution_failed'), None)
        if failed:
            self.requests.append(body); self.closeout_requests.append(deepcopy(body))
            target = failed['target']
            return self.response('finish_task', {'reflection_target_id': target['target_id'],
                'reflection_content_hash': '0'*64 if self.case_mode == 'invalid_finish' else target['content_hash'],
                'processing_summary': self.summary,
                'goals': [{'goal_id': 'method', 'status': 'unresolved', 'reason': '核验执行错误，未获得语义判断', 'subtask_ids': []}]})
        # 本夹具无需实际档案内容；旧回答和原文已经可见，只复现交接边界。
        if 'read_memory' in {t['function']['name'] for t in body.get('tools', [])} and self.handoff_requests:
            return base.ResearchReflectionScenario.handler(self, request)
        calls = [c['function']['name'] for m in body['messages'] for c in m.get('tool_calls', [])]
        if calls.count('read_paper') > 1:
            self.handoff_requests.append(deepcopy(body))
            return base.ResearchReflectionScenario.handler(self, request)
        return super().handler(request)


class ReviewFailureCloseoutTests(unittest.TestCase):
    """默认SQLite，显式环境变量时走独立MySQL；外部模型全部替身。"""

    setUp = base.ResearchReflectionTests.setUp

    def run_case(self, mode='finish', recovery=False):
        """完整HTTP→Worker链，不复用或改写旧真实任务。"""
        scenario = NullVerdictScenario(mode)
        conversations = ConversationService(self.db, self.settings)
        cid = conversations.create('synthetic review failure', __import__('uuid').uuid4().hex)['conversation_id']
        app.dependency_overrides[get_runtime_settings] = lambda: self.settings
        self.addCleanup(app.dependency_overrides.pop, get_runtime_settings, None)
        with TestClient(app) as client:
            created = client.post('/api/tasks', json={'prompt': '核对当前回答，禁止保存。', 'save_policy': 'deny',
                'conversation_id': cid, 'attachments': [{'input_ref': 'input-A', 'paper_id': 'paper-A', 'local_path': str(self.root/'A.pdf')}]})
            self.assertEqual(created.status_code, 202, created.text); tid = created.json()['task_id']
            executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda s, f: base.ScopedPDF())
            prepare = ContextService.prepare; interrupted = []; saved = self.db.save_action

            def capacity(service, state, messages, tools):
                """只在失败观察后的父请求注入故障，验证已有target兜底。"""
                if mode == 'capacity' and any('失败观察：' in m.get('content', '') for m in messages):
                    raise ValueError('CONTEXT_CAPACITY: synthetic after review failure')
                return prepare(service, state, messages, tools)

            def stop_after_failed_observation(action):
                """失败工具观察已提交而checkpoint未写时中断，恢复不得再审。"""
                saved(action)
                if not interrupted and action['tool_name'] == 'review_research' and action['status'] == 'failed':
                    interrupted.append(True); raise SystemExit('controlled failed review observation boundary')

            receipts = {}
            if recovery:
                with patch.object(self.db, 'save_action', stop_after_failed_observation), self.assertRaises(SystemExit):
                    executor.execute(tid)
                receipts = {r['call_id']: r for child in self.db.list_subtasks(tid)
                            for r in self.db.list_model_calls(child['subtask_id'])}
                self.db.execute('UPDATE tasks SET lease_until=? WHERE root_task_id=?', ('2000-01-01T00:00:00Z', tid))
            with patch.object(ContextService, 'prepare', capacity): executor.execute(tid)
            view = client.get('/api/tasks/' + tid).json(); result = view['result']
            self.assertEqual(view['status'], 'partial', view)
            self.assertEqual(result['answer_body_markdown'], scenario.before)
            self.assertTrue(result['processing_summary'])
            self.assertIn('findings[5]', result['answer_markdown'])
            self.assertIn('不能据此认定原文缺证', result['answer_markdown'])
            self.assertEqual(result['answer_markdown'].count(scenario.before), 1)
            # 三次有范围的补读后给一次参数更正机会；重复同一坏判断即无进展停止。
            self.assertEqual(scenario.review_calls, 5)
            self.assertEqual(len(self.db.list_subtasks(tid)), 1)
            record = next(iter(ReflectionService(self.db).snapshot(tid)['research_targets'].values()))
            self.assertEqual(len(record['attempts']), 1); self.assertEqual(record['attempts'][0]['result']['findings'], [])
            self.assertEqual(record['current']['items'][0]['text'], scenario.before)
            reviewer_id = self.db.list_subtasks(tid)[0]['subtask_id']
            submitted = next(a for a in self.db.list_actions(reviewer_id) if a['tool_name'] == 'finish_review')
            self.assertEqual(len(submitted['arguments']['findings']), 8)
            self.assertIsNone(submitted['arguments']['findings'][5]['status'])
            if mode == 'capacity':
                self.assertFalse(scenario.closeout_requests); self.assertIn('CONTEXT_CAPACITY', result['processing_summary'])
            else:
                self.assertEqual(len(scenario.closeout_requests), 1)
                names = {t['function']['name'] for t in scenario.closeout_requests[0]['tools']}
                self.assertEqual(names, {'finish_task', 'read_memory'})
                self.assertIn('findings[5]', json.dumps(scenario.closeout_requests[0], ensure_ascii=False))
            if mode == 'invalid_finish':
                self.assertIn('missing or stale', result['processing_summary'])
                self.assertEqual(result['reason'], 'review failure closeout incomplete')
            if mode != 'finish':
                self.assertIn('findings[5]', result['processing_summary'])
                self.assertEqual(result['retained_answer_source']['targets'][0]['content_hash'], record['current']['content_hash'])
            last = self.db.fetchone('SELECT content FROM conversation_messages WHERE task_id=? AND role=? ORDER BY sequence_no DESC LIMIT 1', (tid, 'assistant'))
            self.assertEqual(last['content'], result['answer_markdown'])
            for child in self.db.list_subtasks(tid):
                for receipt in self.db.list_model_calls(child['subtask_id']):
                    if receipt['call_id'] in receipts: self.assertEqual(receipt, receipts[receipt['call_id']])
            count = len(scenario.requests)
            self.db.update_task(tid, status='queued', result_json=None); executor.execute(tid, 'restored')
            self.assertEqual(len(scenario.requests), count)
            self.assertEqual(client.get('/api/tasks/' + tid).json()['result'], result)
            self.assertFalse(self.db.fetchall('SELECT * FROM drafts WHERE task_id=?', (tid,)))
            self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
        return tid, result

    def test_failed_review_can_deliver_bound_body_and_summary(self):
        """真实图中的失败核验后仍有一次父角色收尾，不重开核验。"""
        self.run_case()

    def test_observe_invalid_review_stops_and_preserves_parent_delivery(self):
        """没有累计额度终止时，同一非法判断仍有界交回父任务保留正文。"""
        self.settings = self.settings.model_copy(update={'execution_budget_mode': 'observe',
            'max_model_requests': 1, 'max_tool_actions': 1, 'reserve_model_requests': 0,
            'task_token_limit': 1, 'task_token_tail': 0})
        self.run_case()

    def test_capacity_after_review_preserves_target_and_specific_error(self):
        """收尾请求发送前容量错误仍交付已有target和两层诊断。"""
        self.run_case('capacity')

    def test_invalid_finish_retains_answer_without_approval(self):
        """收尾过期身份保持拒绝，不放宽hash合同；已有成果保留。"""
        self.run_case('invalid_finish')

    def test_failed_observation_recovery_never_resends_review(self):
        """观察落库后中断，再次执行复用成功收据和原失败身份。"""
        self.run_case(recovery=True)

    def test_current_local_revision_takes_priority_in_exception_delivery(self):
        """已有局部新版本优先于历史/旧原句，故障不覆盖已提交修改。"""
        tid, _, _ = base.ResearchReflectionTests.run_case(self, 'answer')
        result = TaskExecutor(self.settings, self.db)._deliverable_result(tid,
            {'coverage_required': True, 'final_reason': 'synthetic after local revision'})
        current = next(iter(ReflectionService(self.db).snapshot(tid)['research_targets'].values()))['current']
        self.assertEqual(result.answer_body_markdown, current['items'][0]['text'])
        self.assertNotIn('方法不使用', result.answer_body_markdown)
        self.assertEqual(result.retained_answer_source['targets'][0]['content_hash'], current['content_hash'])
        self.assertIsNone(result.saved_path)
