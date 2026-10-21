"""计划归属与候选筛选的API→数据库→Worker替身链；不访问真实模型或论文网络。"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from research_loop.api.dependencies import get_runtime_settings
from research_loop.config import Settings
from research_loop.domain.evidence import Document, PageEvidence, PaperCandidate, SearchResult, SourceResult
from research_loop.domain.tasks import ActionRecord, ResearchBrief
from research_loop.main import app
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.services.coverage import CoverageService
from research_loop.storage.memory import dump
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor


class NonStreamingFixtureProvider(DeepSeekProvider):
    """供应商替身以普通响应交付，仍使用正式HTTP解析与收据。"""

    def send_stream(self, messages, tools=None, force_text=False, *, on_public, **kwargs):
        """本套件只验证交付合同，不把Mock JSON冒称SSE验收。"""
        return self.send(messages, tools, force_text=force_text, **kwargs)


class SyntheticPDF:
    """只有明确全文比较场景才允许调用的隔离PDF边界。"""

    def ingest(self, source, paper_id):
        """为合成论文提供稳定文档身份。"""
        return Document(document_id='plan-doc-' + paper_id, paper_id=paper_id,
            content_sha256=hashlib.sha256(paper_id.encode()).hexdigest(), parser_name='fixture',
            parser_version='1', config_sha256='a' * 64, total_pages=1, cache_path=source)

    def pages(self, document, start, count):
        """提供唯一物理页，内容只用于程序合同断言。"""
        return [PageEvidence(evidence_id='plan-page-' + document.paper_id, document_id=document.document_id,
            page_number=1, text='Synthetic language embeddings align detected regions.',
            parsed_markdown='Synthetic language embeddings align detected regions.')]

    def close(self):
        """替身没有网络资源。"""


class PlanScenario:
    """固定供应商决策，实际参数校验、工具、权限与行动持久化由业务执行。"""

    def __init__(self, root_steps, *, mode='screen', assessments=False, claims=False, paper_count=2):
        """root_steps可含依赖实际子任务观察的函数，不手工注入业务结果。"""
        self.root_steps = root_steps
        self.mode = mode
        self.assessments = assessments
        self.claims = claims
        self.paper_count = paper_count
        self.prefix = uuid4().hex
        self.requests = []
        self.searches = []
        self.root_index = 0

    def search(self, query, date_from, date_to, sources, limit):
        """只替换论文网络，候选授权与保存仍由真实服务执行。"""
        self.searches.append((query, date_from, date_to, sources, limit))
        return SearchResult(candidates=[PaperCandidate(paper_id=f'{self.prefix}-{n}',
            title=f'Synthetic language localization {n}', source='cvpr', published='2025-06-01',
            abstract='Language embeddings align detected regions.',
            paper_url=f'https://fixture.invalid/paper/{n}', pdf_url=f'https://fixture.invalid/paper/{n}.pdf')
            for n in range(self.paper_count)], source_results=[SourceResult(source='cvpr', status='queried_success')])

    @staticmethod
    def response(name=None, args=None):
        """返回真实供应商形状，不调用模型。"""
        message = {'role': 'assistant', 'content': '' if name else '本轮已有结果保留。'}
        if name:
            message['tool_calls'] = [{'id': uuid4().hex, 'type': 'function',
                'function': {'name': name, 'arguments': dump(args)}}]
        return httpx.Response(200, json={'id': uuid4().hex, 'choices': [{
            'finish_reason': 'tool_calls' if name else 'stop', 'message': message}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})

    def handler(self, request):
        """按角色响应；筛选分支绝不调用read_paper。"""
        body = json.loads(request.content)
        self.requests.append(body)
        messages = body['messages']
        validate_messages(messages)
        tool_names = {t['function']['name'] for t in body.get('tools', [])}
        observations = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
        done = [c['function']['name'] for m in messages for c in m.get('tool_calls', [])]
        if 'finish_research' in tool_names:
            if 'search_papers' not in done:
                return self.response('search_papers', {'query': 'synthetic language localization', 'limit': self.paper_count})
            candidates = next(o['data']['candidates'] for o in observations if 'candidates' in o.get('data', {}))
            if self.mode == 'investigate' and 'read_paper' not in done:
                return self.response('read_paper', {'target_ref': candidates[0]['paper_id']})
            pages = [page for o in observations for page in o.get('data', {}).get('pages', [])]
            args = {'status': 'completed', 'findings': ['按标题摘要推荐：语言表征参与区域匹配，供用户选择细读。'],
                'paper_ids': [p['paper_id'] for p in candidates],
                'evidence_ids': [p['evidence_id'] for p in pages], 'gaps': []}
            if self.assessments:
                args['assessments'] = [{'claim': 'language helps localization', 'status': 'supported',
                    'basis': 'direct_statement', 'source_quote': 'Language embeddings align detected regions.',
                    'reason': 'title and abstract', 'evidence_ids': []}]
            return self.response('finish_research', args)
        index = self.root_index
        self.root_index += 1
        if index >= len(self.root_steps):
            return self.response()
        step = self.root_steps[index]
        name, args = step(observations) if callable(step) else step
        return self.response(name, args)

    def factory(self, settings):
        """保留生产供应商解析与收据，替换HTTP传输。"""
        return NonStreamingFixtureProvider('fixture', client=httpx.Client(transport=httpx.MockTransport(self.handler)))


class PlanOwnershipScreeningTests(unittest.TestCase):
    """同组业务样例可在SQLite和显式隔离MySQL运行，不把引擎数算样本数。"""

    def setUp(self):
        """拒绝非专属外库，临时目录隔离附件与输出。"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        url = os.getenv('PLAN_OWNERSHIP_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root / 'test.db')
        if not url.startswith('sqlite:') and not urlparse(url).path.lstrip('/').startswith('researchloop_test_plan_'):
            raise ValueError('explicit researchloop_test_plan_* database required')
        self.settings = Settings(project_root=self.root, data_dir=self.root / 'data', output_dir=self.root / 'notes',
            database_url=url, deepseek_api_key='fixture', retrieval_enabled=False, discovery_backends={},
            max_model_requests=16, reserve_model_requests=0)
        self.db = Database(url)
        self.db.ensure_schema()
        for target in ('httpx.HTTPTransport.handle_request', 'httpx.AsyncHTTPTransport.handle_async_request'):
            blocker = patch(target, side_effect=AssertionError('real HTTP forbidden in plan screening tests'))
            blocker.start()
            self.addCleanup(blocker.stop)
        p = patch('research_loop.main.get_settings', return_value=self.settings)
        p.start()
        self.addCleanup(p.stop)
        old = app.dependency_overrides.get(get_runtime_settings)
        app.dependency_overrides[get_runtime_settings] = lambda: self.settings
        self.addCleanup(self.restore_override, old)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        response = self.client.post('/api/conversations', json={'title': '合成计划筛选', 'idempotency_key': uuid4().hex})
        self.assertEqual(response.status_code, 201, response.text)
        self.cid = response.json()['conversation_id']
        self.prompt = '找近三年两篇语言辅助目标定位论文，给原文链接和推荐理由，我再选一篇细看。不要保存文件。'

    @staticmethod
    def restore_override(old):
        """测试结束恢复共享应用依赖。"""
        if old is None:
            app.dependency_overrides.pop(get_runtime_settings, None)
        else:
            app.dependency_overrides[get_runtime_settings] = old

    def submit(self, prompt=None, attachments=()):
        """正式API提交自然请求，无显式兴趣声明。"""
        response = self.client.post('/api/tasks', json={'conversation_id': self.cid, 'prompt': prompt or self.prompt,
            'save_policy': 'deny', 'attachments': list(attachments), 'idempotency_key': uuid4().hex})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['task_id']

    def plan(self, *, expanded=False, legacy=False):
        """合成用户结果目标和可撤回步骤；新计划顶层来源日期只作检索偏好。"""
        goal = {'goal_id': 'recommend', 'description': '推荐两篇论文供后续选择', 'min_papers': 2,
                'requires_fulltext': False, 'origin': 'user', 'source_quote': '两篇'}
        goals = [goal]
        if expanded:
            goals.extend([{'goal_id': 'collect_extra', 'description': '模型自行扩大候选池', 'min_papers': 4,
                           'origin': 'strategy'}, {'goal_id': 'read_extra', 'description': '模型自行追加全文阅读',
                           'min_papers': 2, 'requires_fulltext': True, 'origin': 'strategy'}])
        if legacy:
            for g in goals:
                g.pop('origin', None)
                g.pop('source_quote', None)
        return {'goal': '先推荐再由用户选读', 'task_kind': 'search', 'requested_count': 2, 'count_confirmed': True,
            'date_from': '2023-10-19', 'date_to': '2026-10-19', 'sources': ['cvpr'],
            'save_requested': False, 'deep_reading_requested': False, 'goals': goals}

    @staticmethod
    def retract(plan):
        """撤回模型自拟步骤而保留每个旧ID及修改原因。"""
        value = copy.deepcopy(plan)
        for goal in value['goals']:
            if goal.get('origin') == 'strategy':
                goal.update(status='withdrawn', change_reason='标题摘要已足以满足本次推荐，不再追加全文取证')
        return value

    @staticmethod
    def delegate(*, mode='screen', claims=False, goal_ids=None):
        """以当前真实目标ID委派，claims只用于兼容历史过度核验回归。"""
        return ('delegate_research', {'subtasks': [{'objective': '按标题摘要筛选两篇候选', 'mode': mode,
            'goal_ids': goal_ids or ['recommend'], 'sources': ['cvpr'], 'date_from': '2023-10-19',
            'date_to': '2026-10-19', 'verification_claims': ['language helps localization'] if claims else []}]})

    def finish(self, *, interest=False, goal_id='recommend', status='satisfied'):
        """绑定工具真实返回的子身份，再从同次finish附带兴趣候选。"""
        def step(observations):
            children = [s for o in observations for s in o.get('data', {}).get('subtasks', [])]
            value = {'answer_markdown': '两篇合成论文均提供原文链接；理由来自标题摘要，留待选择后细读。',
                'goals': [{'goal_id': goal_id, 'status': status, 'reason': '实际候选已按来源日期筛选',
                           'subtask_ids': [s['subtask_id'] for s in children]}]}
            if interest:
                value['interest_candidates'] = [{'topic_key': 'plan-' + self.cid, 'content': '语言辅助目标定位',
                    'reason': '本轮自然提问选择了该研究主题', 'source_quote': self.prompt}]
            return 'finish_task', value
        return step

    def execute(self, tid, scenario):
        """正式Worker执行，全家族生成请求仅进入MockTransport。"""
        executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda s, f: SyntheticPDF())
        with patch('research_loop.providers.discovery.DiscoveryProvider.search', side_effect=scenario.search):
            executor.execute(tid)
        result = self.client.get('/api/tasks/' + tid).json()
        self.assertEqual(result['usage']['unknown'], 0, result)
        self.assertEqual(result['usage']['requests'], len(scenario.requests), result)
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
        return result, executor

    def test_retract_strategy_screen_two_and_same_finish_interest(self):
        """自拟全文步骤撤回后两篇摘要筛选闭环，并由结束响应自动形成兴趣。"""
        tid = self.submit()
        first = self.plan(expanded=True)
        revised = self.retract(first)
        scenario = PlanScenario([('set_research_plan', first), ('set_research_plan', revised),
                                 self.delegate(), self.finish(interest=True)])
        view, executor = self.execute(tid, scenario)
        self.assertEqual(view['status'], 'completed', view)
        self.assertEqual(len(view['result']['coverage']['paper_ids']), 2)
        plans = [a for a in self.db.list_actions(tid) if a['tool_name'] == 'set_research_plan']
        self.assertEqual([a['status'] for a in plans], ['succeeded', 'succeeded'])
        self.assertEqual(plans[0]['arguments'], first)
        self.assertEqual(plans[1]['arguments'], revised)
        self.assertEqual(plans[0]['observation']['data']['research_plan']['goals'][1]['status'], 'active')
        current = CoverageService(self.db).plan(tid)
        self.assertEqual([g['status'] for g in current['goals']], ['active', 'withdrawn', 'withdrawn'])
        child, = view['subtasks']
        self.assertEqual([a['tool_name'] for a in self.db.list_actions(child['subtask_id'])], ['search_papers', 'finish_research'])
        self.assertEqual(child['evidence_ids'], [])
        self.assertFalse(child['deep_reading_status'])
        profile = self.client.get('/api/profile').json()
        entry, = [i for i in profile['inferred_interests'] if i['source'].get('topic_key') == 'plan-' + self.cid]
        self.assertEqual(entry['kind'], 'inferred')
        self.assertEqual(entry['source']['distinct_task_count'], 1)
        from research_loop.services.interests import InterestService
        source, = InterestService(self.db).sources(entry['memory_id'])
        self.assertEqual(source['quote'], self.prompt)
        self.assertEqual(source['content_hash'], hashlib.sha256(self.prompt.encode()).hexdigest())
        self.assertEqual(len(scenario.requests), 6)
        self.assertEqual(self.db.get_task(tid)['save_policy'], 'deny')
        self.assertNotIn('remember_preference', [a['tool_name'] for a in self.db.list_actions(tid)])
        requests_before = len(scenario.requests)
        self.db.update_task(tid, status='queued', result_json=None)
        executor.execute(tid, 'receipt-replay')
        self.assertEqual(len(scenario.requests), requests_before)
        self.assertEqual(self.client.get('/api/profile').json()['inferred_interests'], profile['inferred_interests'])

    def test_screen_legacy_claims_and_abstract_assessments_do_not_require_fulltext(self):
        """旧委派核验字段不能把筛选升级为全文取证，原参数仍可审计。"""
        tid = self.submit()
        scenario = PlanScenario([('set_research_plan', self.plan()), self.delegate(claims=True), self.finish()],
                                claims=True, assessments=True)
        view, _ = self.execute(tid, scenario)
        self.assertEqual(view['status'], 'completed', view)
        child, = view['subtasks']
        self.assertEqual(child['status'], 'completed', child)
        actions = self.db.list_actions(child['subtask_id'])
        self.assertEqual([a['tool_name'] for a in actions], ['search_papers', 'finish_research'])
        finish = actions[-1]
        self.assertEqual(finish['arguments']['assessments'][0]['status'], 'supported')
        self.assertEqual(finish['observation']['data']['assessments'], [])
        self.assertEqual(finish['observation']['data']['pending_verification_claims'], ['language helps localization'])
        self.assertEqual(child['gaps'], [])
        self.assertIn('核验', finish['observation']['message'])

    def test_explicit_comparison_keeps_investigate_fulltext_route(self):
        """明确全文问题仍读取原文，不因筛选修复被降为只读摘要。"""
        tid = self.submit('比较一篇附件方向论文的原文方法，不保存文件。')
        plan = self.plan()
        plan.update(task_kind='compare', requested_count=1)
        plan['goals'] = [{'goal_id': 'recommend', 'description': '读取原文比较方法', 'min_papers': 1,
                          'requires_fulltext': True, 'origin': 'user', 'source_quote': '原文方法'}]
        scenario = PlanScenario([('set_research_plan', plan), self.delegate(mode='investigate'), self.finish()],
                                mode='investigate', paper_count=1)
        view, _ = self.execute(tid, scenario)
        self.assertEqual(view['status'], 'completed', view)
        child, = view['subtasks']
        self.assertEqual(child['mode'], 'investigate')
        self.assertEqual([a['tool_name'] for a in self.db.list_actions(child['subtask_id'])],
                         ['search_papers', 'read_paper', 'finish_research'])
        self.assertTrue(child['evidence_ids'])
        self.assertFalse(child['deep_reading_status'])

    def test_root_count_dates_sources_and_save_cannot_be_relaxed(self):
        """根数量、确认及保存交付约束逐项经真实工具拒绝，数据库禁存保持。"""
        mutations = [('requested_count', 1), ('count_confirmed', False), ('save_requested', True),
                     ('deep_reading_requested', True)]
        # 一个任务按次尝试每种非法变更，再以合法原计划结束；不是多个独立用户样本。
        tid = self.submit()
        first = self.plan()
        steps = [('set_research_plan', first)]
        for key, value in mutations:
            bad = copy.deepcopy(first)
            bad[key] = value
            steps.append(('set_research_plan', bad))
        scenario = PlanScenario(steps + [self.delegate(), self.finish()])
        view, _ = self.execute(tid, scenario)
        self.assertEqual(view['status'], 'completed', view)
        actions = [a for a in self.db.list_actions(tid) if a['tool_name'] == 'set_research_plan']
        self.assertEqual([a['status'] for a in actions], ['succeeded'] + ['failed'] * len(mutations))
        current = CoverageService(self.db).plan(tid)
        self.assertEqual(current['requested_count'], 2)
        self.assertEqual(current['sources'], ['cvpr'])
        self.assertEqual(current['save_policy'], 'deny')

    def test_new_retrieval_preferences_change_but_user_goal_scope_cannot_weaken(self):
        """新计划顶层偏好可改，用户目标硬来源及日期仍逐项保护。"""
        tid = self.submit()
        first = self.plan()
        first['goals'][0].update(sources=['cvpr'], date_from='2023-10-19', date_to='2026-10-19')
        revised = copy.deepcopy(first)
        revised.update(sources=['arxiv'], date_from=None, date_to=None)
        changes = []
        for key, value in [('sources', []), ('date_from', None), ('date_to', None)]:
            bad = copy.deepcopy(revised)
            bad['goals'][0][key] = value
            changes.append(('set_research_plan', bad))
        scenario = PlanScenario([('set_research_plan', first), ('set_research_plan', revised)] + changes +
                                [self.delegate(), self.finish()])
        view, _ = self.execute(tid, scenario)
        self.assertEqual(view['status'], 'completed', view)
        actions = [a for a in self.db.list_actions(tid) if a['tool_name'] == 'set_research_plan']
        self.assertEqual([a['status'] for a in actions], ['succeeded', 'succeeded', 'failed', 'failed', 'failed'])
        current = CoverageService(self.db).plan(tid)
        self.assertEqual(current['_scope_semantics'], 'goal_owned_constraints')
        self.assertEqual(current['sources'], ['arxiv'])
        self.assertIsNone(current['date_from'])
        self.assertIsNone(current['date_to'])
        self.assertEqual(current['goals'][0]['sources'], ['cvpr'])
        self.assertEqual(current['goals'][0]['date_from'], '2023-10-19')
        self.assertEqual(current['goals'][0]['date_to'], '2026-10-19')
        self.assertEqual(current['save_policy'], 'deny')
        self.assertEqual(actions[0]['observation']['data']['research_plan']['sources'], ['cvpr'])

    def test_unmarked_persisted_plan_retains_root_source_date_constraints(self):
        """无标记历史观察不被prepare静默迁移，根来源日期继续按硬范围保护。"""
        tid = self.submit()
        old = self.plan(legacy=True)
        action = ActionRecord(task_id=tid, sequence=1, tool_name='set_research_plan', arguments=old,
            status='succeeded', observation={'ok': True, 'kind': 'success', 'data': {'research_plan': old}})
        self.db.save_action(action.model_dump(mode='json'))
        original = self.db.get_action(tid, 1)
        coverage = CoverageService(self.db)
        prepared = coverage.prepare_plan(tid, ResearchBrief.model_validate(old))
        self.assertNotIn('_scope_semantics', prepared)
        for key, value in [('sources', []), ('date_from', None), ('date_to', None)]:
            bad = copy.deepcopy(old)
            bad[key] = value
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'task constraint'):
                coverage.prepare_plan(tid, ResearchBrief.model_validate(bad))
        self.assertEqual(self.db.get_action(tid, 1), original)
        self.assertEqual(coverage.plan(tid), old)
        self.assertEqual(self.db.get_task(tid)['save_policy'], 'deny')

    def test_user_goal_cannot_be_deleted_relabelled_or_withdrawn(self):
        """用户目标不能通过删ID、转strategy或withdrawn伪装满足。"""
        tid = self.submit()
        first = self.plan(expanded=True)
        changes = []
        for variant in ('delete', 'origin', 'withdraw', 'min', 'quote'):
            value = copy.deepcopy(first)
            goal = value['goals'][0]
            if variant == 'delete':
                value['goals'].pop(0)
            elif variant == 'origin':
                goal.update(origin='strategy', change_reason='模型改计划')
            elif variant == 'withdraw':
                goal.update(status='withdrawn', change_reason='模型不再做')
            elif variant == 'min':
                goal['min_papers'] = 1
            else:
                goal['source_quote'] = '此句只来自模型，用户未说'
            changes.append(('set_research_plan', value))
        scenario = PlanScenario([('set_research_plan', first)] + changes)
        self.execute(tid, scenario)
        actions = [a for a in self.db.list_actions(tid) if a['tool_name'] == 'set_research_plan']
        self.assertEqual([a['status'] for a in actions], ['succeeded'] + ['failed'] * len(changes))
        self.assertEqual(CoverageService(self.db).plan(tid)['goals'][0]['origin'], 'user')

    def test_strategy_revision_requires_reason_and_retains_old_ids(self):
        """撤回有原因且旧ID保留；无理由、删除和改归属均拒绝。"""
        tid = self.submit()
        first = self.plan(expanded=True)
        changes = []
        for variant in ('reason', 'delete', 'origin'):
            bad = copy.deepcopy(first)
            if variant == 'reason':
                bad['goals'][1]['status'] = 'withdrawn'
            elif variant == 'delete':
                bad['goals'].pop(1)
            else:
                bad['goals'][1].update(origin='user', source_quote='两篇')
            changes.append(('set_research_plan', bad))
        scenario = PlanScenario([('set_research_plan', first)] + changes +
                                [('set_research_plan', self.retract(first)), self.delegate(), self.finish()])
        view, _ = self.execute(tid, scenario)
        self.assertEqual(view['status'], 'completed', view)
        actions = [a for a in self.db.list_actions(tid) if a['tool_name'] == 'set_research_plan']
        self.assertEqual([a['status'] for a in actions], ['succeeded', 'failed', 'failed', 'failed', 'succeeded'])

    def test_legacy_goal_missing_ownership_keeps_frozen_contract(self):
        """历史无归属字段默认legacy，不自动当作可撤回模型步骤。"""
        tid = self.submit()
        first = self.plan(legacy=True)
        bad = copy.deepcopy(first)
        bad['goals'][0].update(origin='strategy', status='withdrawn', change_reason='试图解释为旧策略')
        scenario = PlanScenario([('set_research_plan', first), ('set_research_plan', bad),
                                 self.delegate(), self.finish()])
        view, _ = self.execute(tid, scenario)
        self.assertEqual(view['status'], 'completed', view)
        actions = [a for a in self.db.list_actions(tid) if a['tool_name'] == 'set_research_plan']
        self.assertEqual([a['status'] for a in actions], ['succeeded', 'failed'])
        self.assertEqual(CoverageService(self.db).plan(tid)['goals'][0]['origin'], 'legacy')

    def test_real_resume_answer_precedes_old_plan_then_allows_new_count(self):
        """真实resume未更新计划时拒绝委派，更新后采用用户新数量而保留禁存。"""
        tid = self.submit()
        old = self.plan()
        new = copy.deepcopy(old)
        new['requested_count'] = 1
        new['goals'][0].update(min_papers=1, source_quote='改成1篇')
        scenario = PlanScenario([('set_research_plan', old),
            ('request_clarification', {'question': '本轮数量要调整吗？', 'fields': ['count'], 'suggested_values': ['1篇']}),
            self.delegate(), ('set_research_plan', new), self.delegate(), self.finish()], paper_count=1)
        paused, _ = self.execute(tid, scenario)
        self.assertEqual(paused['status'], 'needs_input', paused)
        clarification = self.db.get_pending_clarification(tid)
        response = self.client.post('/api/tasks/' + tid + '/resume', json={
            'clarification_id': clarification['clarification_id'], 'answer': '改成1篇，仍不保存文件',
            'idempotency_key': uuid4().hex})
        self.assertEqual(response.status_code, 200, response.text)
        view, _ = self.execute(tid, scenario)
        self.assertEqual(view['status'], 'completed', view)
        delegates = [a for a in self.db.list_actions(tid) if a['tool_name'] == 'delegate_research']
        self.assertEqual([a['status'] for a in delegates], ['failed', 'succeeded'])
        self.assertIn('clarification', delegates[0]['observation']['message'])
        current = CoverageService(self.db).plan(tid)
        self.assertEqual(current['requested_count'], 1)
        self.assertEqual(current['clarification_id'], clarification['clarification_id'])
        self.assertEqual(self.db.get_task(tid)['save_policy'], 'deny')

    def test_bound_material_scope_cannot_be_removed_or_replaced(self):
        """模型修订不能移除根材料范围或把用户指定材料换成另一授权附件。"""
        local = self.root / 'synthetic.pdf'
        local.write_bytes(b'%PDF-synthetic')
        tid = self.submit(attachments=[{'input_ref': ref, 'local_path': str(local)} for ref in ('input-A', 'input-B')])
        first = self.plan()
        first['input_refs'] = ['input-A']
        first['goals'][0]['input_refs'] = ['input-A']
        variants = []
        for target, value in [('root', []), ('goal', ['input-B']), ('foreign', ['foreign-input'])]:
            bad = copy.deepcopy(first)
            if target == 'root':
                bad['input_refs'] = value
            else:
                bad['goals'][0]['input_refs'] = value
            variants.append(('set_research_plan', bad))
        scenario = PlanScenario([('set_research_plan', first)] + variants)
        self.execute(tid, scenario)
        actions = [a for a in self.db.list_actions(tid) if a['tool_name'] == 'set_research_plan']
        self.assertEqual([a['status'] for a in actions], ['succeeded', 'failed', 'failed', 'failed'])
        current = CoverageService(self.db).plan(tid)
        self.assertEqual(current['input_refs'], ['input-A'])
        self.assertEqual(current['goals'][0]['input_refs'], ['input-A'])

    def test_withdrawn_strategy_cannot_receive_new_delegation(self):
        """撤回步骤保留审计身份，同时禁止按旧步骤继续派发。"""
        tid = self.submit()
        first = self.plan(expanded=True)
        scenario = PlanScenario([('set_research_plan', first), ('set_research_plan', self.retract(first)),
            self.delegate(goal_ids=['read_extra']), self.delegate(), self.finish()])
        view, _ = self.execute(tid, scenario)
        self.assertEqual(view['status'], 'completed', view)
        delegates = [a for a in self.db.list_actions(tid) if a['tool_name'] == 'delegate_research']
        self.assertEqual([a['status'] for a in delegates], ['failed', 'succeeded'])
        self.assertEqual(len(view['subtasks']), 1)

    def test_active_strategy_gap_remains_visible_without_blocking_user_delivery(self):
        """未撤回策略仍如实列未完成，但不能把自拟全文步骤变为用户交付前置。"""
        tid = self.submit()
        plan = self.plan(expanded=True)
        def finish(observations):
            name, args = self.finish()(observations)
            args['goals'].extend([{'goal_id': goal, 'status': 'unresolved', 'subtask_ids': [],
                                  'reason': '用户先选候选；当前不继续自拟补读'}
                                 for goal in ('collect_extra', 'read_extra')])
            return name, args
        scenario = PlanScenario([('set_research_plan', plan), self.delegate(), finish])
        view, _ = self.execute(tid, scenario)
        self.assertEqual(view['status'], 'completed', view)
        self.assertEqual(view['result']['gaps'], [])
        strategy = [g for g in view['result']['coverage']['goals'] if g['origin'] == 'strategy']
        self.assertEqual(len(strategy), 2)
        self.assertTrue(all(g['status'] == 'unresolved' and g['reason'] for g in strategy))

    def test_strategy_material_goal_cannot_substitute_user_material_requirement(self):
        """材料门不能只由可选策略满足；真实材料目标与零额外汇总仍可组合交付。"""
        tid = self.submit()
        invalid = self.plan()
        invalid.update(requested_count=None, count_confirmed=False)
        invalid['goals'][0]['min_papers'] = 0
        invalid['goals'].append({'goal_id': 'retrieve_optional', 'description': '模型自拟额外取材',
                                 'min_papers': 1, 'origin': 'strategy'})
        valid = copy.deepcopy(invalid)
        valid['goals'][0]['min_papers'] = 2
        valid['goals'].append({'goal_id': 'synthesize', 'description': '仅汇总同一批材料，不额外取材',
                              'min_papers': 0, 'origin': 'user', 'source_quote': '推荐理由'})
        def finish(observations):
            name, args = self.finish()(observations)
            args['goals'].extend([
                {'goal_id': 'retrieve_optional', 'status': 'unresolved', 'reason': '无需追加模型自拟材料'},
                {'goal_id': 'synthesize', 'status': 'satisfied', 'reason': '仅汇总已获候选，不新增材料'}])
            return name, args
        scenario = PlanScenario([('set_research_plan', invalid), ('set_research_plan', valid),
                                 self.delegate(), finish])
        view, _ = self.execute(tid, scenario)
        self.assertEqual(view['status'], 'completed', view)
        plans = [a for a in self.db.list_actions(tid) if a['tool_name'] == 'set_research_plan']
        self.assertEqual([a['status'] for a in plans], ['failed', 'succeeded'])
        self.assertIn('材料', plans[0]['observation']['message'])
        self.assertEqual(view['result']['gaps'], [])
        self.assertEqual(len(view['result']['coverage']['paper_ids']), 2)
