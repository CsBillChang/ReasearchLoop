"""通用Reflection的API→数据库→Worker测试；供应商、搜索和PDF均为显式替身。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from research_loop.agents.reflection import ReflectionService
from research_loop.config import Settings
from research_loop.domain.evidence import PageEvidence
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_adaptive_research import AdaptiveScenario
from integration.test_collaboration_contract import FixturePDF


def is_reviewer_request(body):
    """按角色工具权限识别核验请求，避免提示词措辞变化破坏故障注入。"""
    return any(t['function']['name'] == 'finish_review' for t in body.get('tools', []))


class ScopedPDF(FixturePDF):
    """两页合成材料：第一页不含方法结论，第二页含明确语言前缀。"""

    def pages(self, document, start, count):
        """返回受请求页范围约束的合成文字，不读取真实论文。"""
        return [PageEvidence(evidence_id=f'page:{document.paper_id}:{n}', document_id=document.document_id,
            page_number=n, text='Experimental context only.' if n == 1 else 'The method uses language prefix embeddings.')
            for n in range(start, min(start + count, 3))]


class ResearchReflectionScenario(AdaptiveScenario):
    """通过实际工具观察决策，不在Worker预置通过状态或修改行动日志。"""

    def __init__(self, kind='answer'):
        """选择局部修订、补证、缺证、准确披露或候选适用性场景。"""
        super().__init__(); self.mode = kind
        self.before = '保留开头。方法不使用语言前缀。保留结尾。'
        self.after = '保留开头。方法使用语言前缀。保留结尾。'

    def handler(self, request):
        """只通过生产DeepSeek适配器的HTTP替身返回工具调用。"""
        body = json.loads(request.content); self.requests.append(body)
        messages = body['messages']; validate_messages(messages)
        names = {t['function']['name'] for t in body.get('tools', [])}
        observations = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
        done = [c['function']['name'] for m in messages for c in m.get('tool_calls', [])]
        reply = self.response
        if 'finish_review' in names:
            context = json.loads(messages[1]['content'].split('\n', 1)[1])
            target = context['target']; bundle = json.loads(messages[2]['content'].split('\n', 1)[1])
            assert 'fixture_old_criticism' not in json.dumps(messages, ensure_ascii=False)
            assessments = []
            for item in target['items']:
                if target['target_type'] == 'candidates':
                    assessments.append({'item_id': item['item_id'], 'suitability': 'unsuitable', 'basis': 'metadata',
                        'metadata_fields': ['abstract'], 'reason': '摘要讨论分割而非当前要求的检测', 'suggested_action': 'exclude'})
                    continue
                text = item['text']
                if self.mode == 'disclosure':
                    status, basis, quote = 'supported', 'scope_disclosure', None
                elif self.mode == 'missing' or not any(p['page_number'] == 2 for p in bundle):
                    status, basis, quote = 'insufficient', 'not_found', None
                else:
                    status = 'contradicted' if '不使用' in text else 'supported'
                    basis, quote = 'direct_statement', bundle[-1]['text']
                assessments.append({'item_id': item['item_id'], 'status': status, 'basis': basis,
                    'reason': 'fixture_old_criticism' if status == 'contradicted' else '按当前范围核验',
                    'evidence_ids': [p['evidence_id'] for p in bundle], 'source_quote': quote,
                    'suggested_action': 'revise' if status == 'contradicted' else ('supplement' if status == 'insufficient' else 'keep')})
            return reply('finish_review', {'findings': assessments, 'gaps': []})
        if 'finish_research' in names:
            if self.mode == 'candidates':
                if 'search_papers' not in done: return reply('search_papers', {'query': 'detection', 'limit': 1})
                candidate = observations[0]['data']['candidates'][0]
                return reply('finish_research', {'status': 'completed', 'findings': ['候选尚需确认主题'],
                    'paper_ids': [candidate['paper_id']], 'evidence_ids': [], 'gaps': []})
            if 'read_paper' not in done: return reply('read_paper', {'target_ref': 'input-A'})
            pages = observations[-1]['data']['pages']
            return reply('finish_research', {'status': 'completed', 'findings': [self.before],
                'paper_ids': ['paper-A'], 'evidence_ids': [p['evidence_id'] for p in pages], 'gaps': []})
        if 'set_research_plan' not in done:
            return reply('set_research_plan', {'goal': '核对方法', 'task_kind': 'search' if self.mode == 'candidates' else 'compare',
                'goals': [{'goal_id': 'method', 'description': '核对方法语言信息', 'min_papers': 1,
                           'requires_fulltext': self.mode != 'candidates'}]})
        if self.mode in {'findings', 'candidates'}:
            if 'delegate_research' not in done:
                return reply('delegate_research', {'subtasks': [{'objective': '取得待核验材料',
                    'mode': 'screen' if self.mode == 'candidates' else 'investigate', 'goal_ids': ['method'],
                    'input_refs': [] if self.mode == 'candidates' else ['input-A']}]})
            children = next(o['data']['subtasks'] for o in observations if 'subtasks' in o.get('data', {}))
            refs = children[0]['evidence_ids']
        else:
            if 'read_paper' not in done:
                return reply('read_paper', {'target_ref': 'input-A', 'start_page': 1,
                    'page_count': 1 if self.mode in {'supplement', 'missing', 'disclosure'} else 2})
            children = []
            refs = [p['evidence_id'] for o in observations for p in o.get('data', {}).get('pages', [])]
        if not refs and any(m.get('content', '').startswith('[ResearchLoop Memory v1]') for m in messages):
            # 原文tool观察可变为档案预览；替身从实际可见的授权来源目录取身份。
            from integration.test_context_continuity import memory_payload
            refs = [p['evidence_id'] for p in memory_payload(messages)['current_task']['authorized_evidence']]
        reviews = [o['data'] for o in observations if o.get('data', {}).get('review_id')]
        if not reviews:
            args = {'target_type': self.mode if self.mode in {'findings', 'candidates'} else 'answer',
                'reason': '用户要求核对当前具体问题', 'verification_questions': ['方法是否使用语言前缀？'],
                'evidence_ids': refs, 'goal_ids': ['method']}
            if children: args['source_task_id'] = children[0]['subtask_id']
            else: args['answer_markdown'] = ('在已读取第一页范围无法确认方法设置。' if self.mode == 'disclosure'
                else ('方法使用语言前缀。' if self.mode == 'supplement' else self.before))
            return reply('review_research', args)
        review = reviews[-1]
        current = next((o['data']['target'] for o in reversed(observations) if o.get('data', {}).get('target')), review['target'])
        if self.mode in {'answer', 'findings'} and 'revise_research' not in done:
            return reply('revise_research', {'target_id': current['target_id'], 'content_hash': current['content_hash'],
                'review_id': review['review_id'], 'patches': [{'item_id': current['items'][0]['item_id'],
                    'old_text': '方法不使用语言前缀。', 'new_text': '方法使用语言前缀。', 'reason': '原文明确报告语言前缀', 'evidence_ids': refs}]})
        if self.mode == 'supplement' and done.count('read_paper') == 1:
            return reply('read_paper', {'target_ref': 'input-A', 'start_page': 2, 'page_count': 1})
        # 用当前版本是否已核验来决策，不能把失败的工具调用计成成功核验。
        if ((self.mode in {'answer', 'findings'} and current['content_hash'] != review['content_hash']) or
                (self.mode == 'supplement' and len(reviews) == 1)):
            return reply('review_research', {'target_type': current['target_type'], 'target_id': current['target_id'],
                'content_hash': current['content_hash'], 'reason': '定向更新后核验当前内容',
                'verification_questions': ['方法是否使用语言前缀？'], 'evidence_ids': refs})
        if self.mode in {'missing', 'candidates'} and 'close_research_issue' not in done:
            return reply('close_research_issue', {'target_id': current['target_id'], 'content_hash': current['content_hash'],
                'review_id': review['review_id'], 'action': 'exclude' if self.mode == 'candidates' else 'end',
                'reason': '候选主题不适用，保留检索结果。' if self.mode == 'candidates' else '当前材料不足，无法确认，不作全篇否定。'})
        goals = [{'goal_id': 'method', 'status': 'unresolved' if self.mode in {'missing', 'candidates'} else 'satisfied',
                  'reason': '范围不足保留成果' if self.mode in {'missing', 'candidates'} else '当前内容与实际材料一致',
                  'subtask_ids': [c['subtask_id'] for c in children]}]
        if current['target_type'] == 'answer':
            return reply('finish_task', {'reflection_target_id': current['target_id'],
                'reflection_content_hash': current['content_hash'], 'goals': goals})
        return reply('finish_task', {'answer_markdown': current['items'][0].get('text', '候选摘要与检测任务不匹配，未进一步读取。'), 'goals': goals})

    def search(self, query, date_from, date_to, sources, limit):
        """实际搜索服务接到的合成候选，与审校对象一一对应。"""
        result = super().search(query, date_from, date_to, sources, 1)
        for candidate in result.candidates: candidate.abstract = 'Semantic segmentation of images.'
        return result


class RepeatedReadScenario(ResearchReflectionScenario):
    """先重复读已有页，随后由实际可见工具转交核验或直接结束。"""

    def __init__(self, case, direct=False):
        """记录交接窗口；不修改数据库行动或模型预算。"""
        super().__init__('answer')
        self.case, self.direct = case, direct
        self.injected = False
        self.handoff_seen = False

    def handler(self, request):
        """只在根任务插入一次重复取证，并检查后续工具权限。"""
        body = json.loads(request.content)
        names = {t['function']['name'] for t in body.get('tools', [])}
        done = [c['function']['name'] for m in body['messages'] for c in m.get('tool_calls', [])]
        if 'finish_task' in names and not is_reviewer_request(body):
            if 'read_paper' in done and not self.injected:
                self.injected = True
                self.requests.append(body)
                return self.response('read_paper', {'target_ref': 'input-A', 'start_page': 1, 'page_count': 2})
            if self.injected and 'review_research' not in done:
                self.handoff_seen = True
                self.case.assertNotIn('read_paper', names)
                self.case.assertIn('review_research', names)
                self.case.assertIn('revise_research', names)
                if self.direct:
                    self.requests.append(body)
                    return self.response('finish_task', {'answer_markdown': self.after,
                        'goals': [{'goal_id': 'method', 'status': 'satisfied', 'reason': '依已有原文交付'}]})
            if 'review_research' in done:
                # 核验已经结算，即使结论是反证，也不能仍被旧重复读回永久限制。
                self.case.assertIn('read_paper', names)
        return super().handler(request)


class ResearchReflectionTests(unittest.TestCase):
    """每例独立根任务；真实MySQL仅使用明确命名的测试库。"""

    def setUp(self):
        """构造临时PDF标记文件，所有外部数据入口均替换。"""
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        url = os.getenv('REFLECTION_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root/'test.db')
        if url.startswith('mysql') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_'):
            raise ValueError('isolated database required')
        self.settings = Settings(project_root=self.root, data_dir=self.root/'data', output_dir=self.root/'notes', database_url=url)
        self.db = Database(url); self.db.ensure_schema(); (self.root/'A.pdf').write_bytes(b'%PDF-fixture')

    def run_case(self, kind, scenario=None):
        """运行真实路由和Worker，最后重新执行根任务验证成功收据复用。"""
        from research_loop.main import app
        from research_loop.api.dependencies import get_runtime_settings
        scenario = scenario or ResearchReflectionScenario(kind)
        executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda s, f: ScopedPDF())
        app.dependency_overrides[get_runtime_settings] = lambda: self.settings
        self.addCleanup(app.dependency_overrides.pop, get_runtime_settings, None)
        with TestClient(app) as client, patch('research_loop.providers.discovery.DiscoveryProvider.search', scenario.search):
            response = client.post('/api/tasks', json={'prompt': '按需核验当前问题，禁止保存。', 'save_policy': 'deny',
                'attachments': [{'input_ref': 'input-A', 'paper_id': 'paper-A', 'local_path': str(self.root/'A.pdf')}]})
            self.assertEqual(response.status_code, 202, response.text)
            tid = response.json()['task_id']; executor.execute(tid)
            view = client.get('/api/tasks/' + tid).json()
            self.assertEqual(view['status'], 'partial' if kind in {'missing', 'candidates'} else 'completed', json.dumps(view, ensure_ascii=False))
            before = len(scenario.requests)
            self.db.update_task(tid, status='queued', result_json=None)
            executor.execute(tid, 'restored')
            self.assertEqual(len(scenario.requests), before)
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
        self.assertFalse(self.db.fetchall('SELECT * FROM drafts WHERE task_id=?', (tid,)))
        actions = self.db.list_actions(tid)
        failures = [a for a in actions if a['status'] != 'succeeded']
        self.assertEqual(failures, [], json.dumps(failures, ensure_ascii=False))
        receipts = self.db.fetchall('SELECT m.* FROM model_calls m JOIN tasks t ON t.task_id=m.task_id WHERE t.task_id=? OR t.root_task_id=?', (tid, tid))
        self.assertTrue(receipts)
        self.assertTrue(all(r['status'] == 'succeeded' for r in receipts))
        # 子任务通过task.root_task_id绑定；所有供应商响应来自MockTransport。
        self.assertTrue(scenario.requests)
        return tid, view, scenario

    def test_answer_local_revision_and_current_snapshot_delivery(self):
        """反证→精确修改→独立复核→直接回答，不创建笔记。"""
        tid, view, scenario = self.run_case('answer')
        self.assertEqual(view['result']['answer_markdown'], scenario.after)
        current = ReflectionService(self.db).research_context(tid)[0]
        self.assertEqual(current['quality_status'], 'supported')
        self.assertTrue(current['reflection']['progress']['resolved_item_ids'])

    def test_repeated_read_can_handoff_review_and_revise(self):
        """重复取证→独立核验→局部修订→再核验→交付，恢复不重发。"""
        scenario = RepeatedReadScenario(self)
        tid, view, _ = self.run_case('answer', scenario)
        self.assertTrue(scenario.handoff_seen)
        self.assertEqual(view['result']['answer_markdown'], scenario.after)
        self.assertEqual(sum(is_reviewer_request(r) for r in scenario.requests), 2)
        self.assertEqual(sum(a['tool_name'] == 'read_paper' for a in self.db.list_actions(tid)), 2)

    def test_repeated_read_does_not_force_review(self):
        """允许交付已有可靠回答，不把Reflection升级成所有任务必经门。"""
        scenario = RepeatedReadScenario(self, direct=True)
        tid, view, _ = self.run_case('answer', scenario)
        self.assertTrue(scenario.handoff_seen)
        self.assertEqual(view['result']['answer_markdown'], scenario.after)
        self.assertFalse(any(is_reviewer_request(r) for r in scenario.requests))
        self.assertFalse(self.db.list_subtasks(tid))

    def test_findings_local_overlay_keeps_child_immutable(self):
        """子任务原始错误发现保留，根任务局部新版本进入最终交付。"""
        tid, view, scenario = self.run_case('findings')
        child = next(c for c in self.db.list_subtasks(tid) if self.db.get_task(c['subtask_id'])['role'] == 'researcher')
        self.assertEqual(child['result']['result']['findings'], [scenario.before])
        self.assertEqual(view['result']['answer_markdown'], scenario.after)

    def test_directed_supplement_enters_independent_review(self):
        """原问题缺证后补读第二页，保持回答原文且新证据实际进入Reviewer输入。"""
        tid, view, scenario = self.run_case('supplement')
        checks = [r for r in scenario.requests if is_reviewer_request(r)]
        self.assertEqual(len(checks), 2)
        self.assertNotIn('language prefix embeddings', checks[0]['messages'][2]['content'])
        self.assertIn('language prefix embeddings', checks[1]['messages'][2]['content'])
        self.assertTrue(ReflectionService(self.db).research_context(tid)[0]['reflection']['progress']['evidence_changed'])

    def test_missing_evidence_ends_with_readable_partial(self):
        """无可行补证路径时明确结束，原语义状态不伪装通过。"""
        tid, view, _ = self.run_case('missing')
        self.assertIn('无法确认', view['result']['answer_markdown'])
        current = ReflectionService(self.db).research_context(tid)[0]
        self.assertEqual(current['quality_status'], 'issues')
        self.assertEqual(current['decisions'][0]['action'], 'end')

    def test_accurate_scope_disclosure_can_be_supported(self):
        """程序读取范围不要求在论文中出现，准确披露无需改成确定事实。"""
        _, view, scenario = self.run_case('disclosure')
        self.assertIn('无法确认', view['result']['answer_markdown'])
        self.assertEqual(sum(is_reviewer_request(r) for r in scenario.requests), 1)

    def test_candidate_metadata_review_without_fulltext(self):
        """适用性使用实际摘要，不为了候选筛选强制读取全文。"""
        tid, view, _ = self.run_case('candidates')
        root_actions = self.db.list_actions(tid)
        self.assertFalse(any(a['tool_name'] == 'read_paper' for a in root_actions))
        for child in self.db.list_subtasks(tid):
            self.assertFalse(any(a['tool_name'] == 'read_paper' for a in self.db.list_actions(child['subtask_id'])))
        self.assertIn('不匹配', view['result']['answer_markdown'])

    @unittest.skipUnless(os.getenv('REFLECTION_TEST_DATABASE_URL'), 'explicit isolated MySQL required for SIGKILL')
    def test_sigkill_review_receipt_and_patch_recover_without_resend(self):
        """两处真实进程终止：成功审校和局部版本只提交一次。"""
        import subprocess
        import sys
        from research_loop.domain.tasks import TaskCreate
        from research_loop.services.tasks import TaskService
        for point in ['review', 'patch']:
            with self.subTest(point=point):
                folder = self.root/point; folder.mkdir()
                pdf = folder/'A.pdf'; pdf.write_bytes(b'%PDF-fixture')
                tid = TaskService(self.db, self.settings).create(TaskCreate(prompt='核验语言条件，禁止保存', save_policy='deny',
                    attachments=[{'input_ref': 'input-A', 'paper_id': 'paper-A', 'local_path': str(pdf)}])).task_id
                env = {**os.environ, 'REFLECTION_ROOT': str(folder), 'REFLECTION_TASK': tid}
                command = [sys.executable, str(Path(__file__).with_name('research_reflection_recovery_process.py'))]
                killed = subprocess.run(command + [point], env=env, capture_output=True, timeout=40)
                self.assertEqual(killed.returncode, -9, killed.stderr.decode())
                self.assertTrue((folder/'killed.ready').exists())
                receipts = {r['call_id']: (r['request_id'], r['usage_json']) for task in [tid] + [c['subtask_id'] for c in self.db.list_subtasks(tid)]
                            for r in self.db.list_model_calls(task) if r['status'] == 'succeeded'}
                self.db.update_task(tid, lease_until='2000-01-01T00:00:00+00:00')
                resumed = subprocess.run(command + ['resume'], env=env, capture_output=True, timeout=40)
                self.assertEqual(resumed.returncode, 0, resumed.stderr.decode())
                view = TaskService(self.db, self.settings).view(self.db.get_task(tid))
                self.assertEqual(view.status.value, 'completed', view)
                requests = [json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
                self.assertEqual(sum(r['reviewer'] for r in requests), 2)
                state = next(iter(ReflectionService(self.db).snapshot(tid)['research_targets'].values()))
                self.assertEqual(len(state['patches']), 1)
                self.assertEqual(len(state['attempts']), 2)
                for task in [tid] + [c['subtask_id'] for c in self.db.list_subtasks(tid)]:
                    for receipt in self.db.list_model_calls(task):
                        if receipt['call_id'] in receipts:
                            self.assertEqual((receipt['request_id'], receipt['usage_json']), receipts[receipt['call_id']])
                self.assertFalse(self.db.has_unknown_family_call(tid))
                self.assertFalse(list(folder.rglob('*.md')))

    def test_unknown_reviewer_receipt_blocks_retry_and_retains_answer(self):
        """模拟供应商读超时：unknown保持未知，恢复不重发，当前候选不丢。"""
        from research_loop.domain.tasks import TaskCreate
        from research_loop.services.tasks import TaskService
        scenario = ResearchReflectionScenario('answer')
        handle = scenario.handler
        def timeout(request):
            """仅Reviewer模拟已发送读超时，其余仍用确定替身。"""
            body = json.loads(request.content)
            if is_reviewer_request(body):
                scenario.requests.append(body)
                raise httpx.ReadTimeout('controlled unknown receipt', request=request)
            return handle(request)
        scenario.handler = timeout
        executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda s, f: ScopedPDF())
        tid = TaskService(self.db, self.settings).create(TaskCreate(prompt='核验当前方法，禁止保存', save_policy='deny',
            attachments=[{'input_ref': 'input-A', 'paper_id': 'paper-A', 'local_path': str(self.root/'A.pdf')}])).task_id
        executor.execute(tid)
        view = TaskService(self.db, self.settings).view(self.db.get_task(tid))
        self.assertTrue(self.db.has_unknown_family_call(tid))
        self.assertIn(scenario.before, view.result.answer_markdown)
        reviews = ReflectionService(self.db).research_context(tid)
        self.assertEqual(reviews[0]['quality_status'], 'execution_failed')
        count = len(scenario.requests)
        self.db.update_task(tid, status='queued', result_json=None)
        executor.execute(tid, 'restored')
        self.assertEqual(len(scenario.requests), count)
        self.assertTrue(self.db.has_unknown_family_call(tid))
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
