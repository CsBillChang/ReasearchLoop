"""多意见与分离交付的API/Worker验证；所有供应商响应来自MockTransport。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from integration import test_research_reflection as existing
from research_loop.agents.reflection import ReflectionService
from research_loop.domain.tasks import FinishTaskInput
from research_loop.services.coverage import CoverageService
from research_loop.workers.executor import TaskExecutor


class MultipleFindingsScenario(existing.ResearchReflectionScenario):
    """覆盖多意见修订、带缺证交付和最后一次结果拒绝三个路径。"""

    def __init__(self, mode):
        """复用现有取证/计划替身，不伪造服务核验结果。"""
        super().__init__('answer' if mode == 'revision' else 'missing')
        self.case_mode = mode
        self.review_calls = 0
        self.summary = '已处理本轮核验；未确认的范围继续保留，不宣称全部正确。'

    def handler(self, request):
        """经真实工具观察返回合成决策，仍由正式参数和证据校验执行。"""
        body = json.loads(request.content)
        if existing.is_reviewer_request(body) and self.case_mode == 'fault':
            self.review_calls += 1
            if self.review_calls <= 3:
                self.requests.append(body)
                start, count = [(1, 1), (2, 1), (1, 2)][self.review_calls - 1]
                return self.response('read_paper', {'target_ref': 'input-A', 'start_page': start, 'page_count': count})
        response = super().handler(request)
        raw = response.json()
        call = raw['choices'][0]['message']['tool_calls'][0]['function']
        args = json.loads(call['arguments'])
        if call['name'] == 'finish_review':
            original = args['findings'][0]
            rows = [deepcopy(original)]
            for index in range(6):
                rows.append({**deepcopy(original), 'status': 'supported', 'basis': 'direct_statement',
                    'reason': f'合成支持意见{index}，并不替代首条意见',
                    'source_quote': ('Experimental context only.' if self.mode == 'missing'
                                     else 'The method uses language prefix embeddings.'),
                    'suggested_action': 'keep'})
            if self.case_mode == 'fault':
                rows[1]['source_quote'] = 'Experimental ... only.'
            args['findings'] = rows
        elif self.case_mode == 'fault' and call['name'] == 'close_research_issue':
            args['reason'] = '核验结果参数未通过校验后触及局部停止设置，保留正文，不把执行故障说成缺证。'
        elif call['name'] == 'finish_task':
            args['processing_summary'] = self.summary
        return self.response(call['name'], args)


class ReflectionDeliveryTests(unittest.TestCase):
    """沿用临时库或显式隔离MySQL入口，全部任务禁止文件保存。"""

    setUp = existing.ResearchReflectionTests.setUp

    def run_scenario(self, mode):
        """从API入队到DB/Worker交付，随后重放已结算任务检查不重发。"""
        from research_loop.main import app
        scenario = MultipleFindingsScenario(mode)
        executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda s, f: existing.ScopedPDF())
        with patch('research_loop.api.dependencies.get_settings', return_value=self.settings), TestClient(app) as client:
            created = client.post('/api/tasks', json={'prompt': '核验当前回答，保留不确定性，禁止保存。',
                'save_policy': 'deny', 'attachments': [{'input_ref': 'input-A', 'paper_id': 'paper-A',
                                                       'local_path': str(self.root/'A.pdf')}]})
            self.assertEqual(created.status_code, 202, created.text)
            tid = created.json()['task_id']; executor.execute(tid)
            view = client.get('/api/tasks/' + tid).json()
            self.assertEqual(view['status'], 'completed' if mode == 'revision' else 'partial', view)
            calls = len(scenario.requests)
            self.db.update_task(tid, status='queued', result_json=None)
            executor.execute(tid, 'restored')
            self.assertEqual(len(scenario.requests), calls)
            self.assertEqual(client.get('/api/tasks/' + tid).json()['result'], view['result'])
        result = view['result']
        self.assertEqual(result['answer_body_markdown'], scenario.after if mode == 'revision' else scenario.before)
        self.assertEqual(result['processing_summary'], scenario.summary)
        self.assertEqual(result['answer_markdown'].count(scenario.summary), 1)
        self.assertIsNone(result['saved_path'])
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
        self.assertFalse(self.db.fetchall('SELECT * FROM drafts WHERE task_id=?', (tid,)))
        return tid, result, scenario

    def test_seven_findings_patch_and_restore(self):
        """七条意见都保留，局部修改后正常结束；恢复不产生第二份结果。"""
        tid, result, _ = self.run_scenario('revision')
        reviews = ReflectionService(self.db).research_context(tid)
        self.assertEqual(len(reviews[0]['findings']), 7)
        self.assertEqual(len({i['issue_id'] for i in reviews[0]['issues']}), 7)
        snapshot = ReflectionService(self.db).snapshot(tid)
        record = next(iter(snapshot['research_targets'].values()))
        self.assertEqual(len(record['patches']), 1)
        self.assertEqual([a['result']['status'] for a in record['attempts']], ['issues', 'supported'])
        self.assertEqual(result['review_notices'], [])

    def test_uncertain_delivery_does_not_require_all_supported(self):
        """存在缺证也能交付，原正文与处理说明分离，旧版本与额外改文仍拒绝。"""
        tid, result, scenario = self.run_scenario('uncertain')
        review = result['research_reviews'][0]
        self.assertEqual(review['quality_status'], 'issues')
        self.assertEqual(len(review['findings']), 7)
        self.assertEqual(len(review['issues']), 7)
        args = {'reflection_target_id': review['target_id'], 'reflection_content_hash': review['content_hash'],
                'goals': result['coverage']['goals'], 'processing_summary': scenario.summary}
        for changes, message in [({'answer_markdown': scenario.before + '\n额外处理说明'}, 'processing_summary'),
                                 ({'reflection_content_hash': '0'*64}, 'stale')]:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                CoverageService(self.db).finish(tid, FinishTaskInput(**{**args, **changes}))

    def test_result_error_survives_local_stop_and_reaches_delivery(self):
        """第四次返回坏引句后停止：首因、末因都进父任务，仍有可读交付。"""
        tid, result, scenario = self.run_scenario('fault')
        self.assertEqual(scenario.review_calls, 4)
        review = result['research_reviews'][0]
        self.assertEqual(review['quality_status'], 'execution_failed')
        error = review['execution_error']
        self.assertEqual(error['first_error']['stage'], 'result_validation')
        self.assertIn('findings[1]', error['first_error']['message'])
        self.assertIn('model request budget exhausted', error['stop_reason'])
        self.assertEqual(review['findings'], [])
        self.assertIn(error['first_error']['message'], result['answer_markdown'])
        self.assertIn('不能据此认定原文缺证', result['answer_markdown'])
        parent_observation = next(a['observation'] for a in self.db.list_actions(tid)
                                  if a['tool_name'] == 'review_research')
        self.assertIn('findings[1]', parent_observation['message'])
        self.assertEqual(parent_observation['data']['execution_error'], error)
        self.assertEqual(len(self.db.list_subtasks(tid)), 1)
