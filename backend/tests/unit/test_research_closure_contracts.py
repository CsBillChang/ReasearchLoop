"""研究覆盖与异常交付的合成合同，临时库且阻断真实网络。"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from research_loop.agents.reflection import ReflectionService
from research_loop.config import Settings
from research_loop.domain.reflection import ResearchReviewInput
from research_loop.domain.research_progress import ResearchProgressInput, ProgressFinding
from research_loop.domain.tasks import ActionRecord, FinishTaskInput, ResearchBrief, SubtaskSpec, TaskCreate
from research_loop.services.coverage import CoverageService
from research_loop.services.papers import PaperService
from research_loop.services.research_progress import ResearchProgressService, continuation_fingerprint
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_collaboration_contract import FixturePDF


class ClosureContracts(unittest.TestCase):
    """只验证执行事实、身份和交付结构，不将替身判断称为语义质量。"""

    def setUp(self):
        """建立独立数据库与论文替身，任何外网连接都令测试失败。"""
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        url = os.getenv('FIVE_ROOT_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root/'closure.db')
        if not url.startswith('sqlite:') and not url.rsplit('/', 1)[-1].split('?', 1)[0].startswith('researchloop_test_'):
            raise ValueError('only isolated researchloop_test_ database is allowed')
        if url.startswith('sqlite:'):
            blocker = patch.object(socket.socket, 'connect', side_effect=AssertionError('real network forbidden'))
            blocker.start(); self.addCleanup(blocker.stop)
        self.settings = Settings(project_root=self.root, data_dir=self.root/'data', output_dir=self.root/'notes',
            database_url=url, retrieval_enabled=False)
        self.db = Database(url); self.db.ensure_schema()
        self.tid = TaskService(self.db, self.settings).create(TaskCreate(prompt='解释合成材料机制，不保存',
            save_policy='deny', idempotency_key=uuid4().hex)).task_id
        self.coverage = CoverageService(self.db)
        self.pid = 'fixture-' + uuid4().hex
        self.db.save_paper(self.pid, {'paper_id': self.pid, 'title': 'Synthetic language mechanism',
            'source': 'cvpr', 'published': '2025-06-01', 'pdf_url': 'https://fixture.invalid/p.pdf'})
        self.db.grant_paper(self.tid, self.pid, self.tid)
        self.data = PaperService(self.db, None, FixturePDF()).read(self.tid, self.pid, 1, 1)
        self.plan = ResearchBrief(goal='解释方法', task_kind='search', requested_count=1, goals=[{
            'goal_id': 'mechanism', 'description': '解释方法', 'min_papers': 1, 'requires_fulltext': True}])
        self.record(self.tid, 'set_research_plan', {'research_plan': self.coverage.prepare_plan(self.tid, self.plan)})

    def record(self, tid, tool, data):
        """写入真实行动形状，调用次数及ID只在临时库生成。"""
        seq = max([a['sequence_no'] for a in self.db.list_actions(tid)] or [0]) + 1
        action = ActionRecord(task_id=tid, sequence=seq, tool_name=tool, arguments={}, status='succeeded',
            observation={'ok': True, 'kind': 'success', 'data': data})
        self.db.save_action(action.model_dump(mode='json'))
        return action

    def finish(self, **changes):
        """Supervisor显式选择当前目标；不自动推测科学支持。"""
        return self.coverage.finish(self.tid, FinishTaskInput(answer_markdown='合成机制结论。', goals=[{
            'goal_id': 'mechanism', 'status': 'satisfied', 'reason': '基于返回的授权原文', **changes}]))

    def hit(self):
        """产生精确原文区间，不把整页已存储误作本次整页可见。"""
        page = self.data['pages'][0]
        return {**page, 'source_sha256': self.data['content_sha256'], 'paper_id': self.pid,
            'text': page['text'][:12], 'char_start': 0, 'char_end': 12,
            'text_sha256': hashlib.sha256(page['text'][:12].encode()).hexdigest(), 'chunk_id': 'chunk-synthetic'}

    def test_read_acquire_and_retrieve_each_support_direct_delivery(self):
        """三种返回入口均可直接满足材料身份，无需再建Researcher。"""
        for tool in ('read_paper', 'acquire_paper_pdf', 'retrieve_evidence'):
            with self.subTest(tool=tool):
                self.db.execute('DELETE FROM actions WHERE task_id=? AND tool_name<>?', (self.tid, 'set_research_plan'))
                data = {'hits': [self.hit()]} if tool == 'retrieve_evidence' else self.data
                self.record(self.tid, tool, data)
                result = self.finish()
                self.assertEqual(result['coverage']['paper_ids'], [self.pid])
                actual = result['coverage']['goals'][0]['material_coverage']
                self.assertEqual(len(actual), 1)
                self.assertFalse(actual[0]['full_document_returned'])
                self.assertEqual(actual[0]['material_scope'], 'retrieved_chunk' if tool == 'retrieve_evidence' else 'returned_page')
                self.assertEqual(self.db.list_subtasks(self.tid), [])

    def test_ready_document_id_empty_and_invalid_windows_are_not_original_content(self):
        """即使工具成功，索引ready、空命中、错正文或错版本都不得凑覆盖。"""
        invalid = [({'status': 'ready', 'document_id': self.data['document_id'], 'hits': []}, 'retrieve_evidence'),
                   ({'paper_id': self.pid, 'document_id': self.data['document_id'], 'pages': []}, 'read_paper')]
        wrong = deepcopy(self.data); wrong['pages'][0]['text'] = 'not the returned page'
        invalid.append((wrong, 'acquire_paper_pdf'))
        wrong = deepcopy(self.data); wrong['content_sha256'] = '0'*64
        invalid.append((wrong, 'read_paper'))
        hit = self.hit(); hit['char_end'] += 1
        invalid.append(({'hits': [hit]}, 'retrieve_evidence'))
        for data, tool in invalid:
            self.record(self.tid, tool, data)
        self.assertEqual(self.coverage.source_coverage(self.tid), [])
        with self.assertRaisesRegex(ValueError, 'lacks qualified papers'):
            self.finish()

    def test_scope_version_and_revoked_page_authority_remain_required(self):
        """真实权限/版本撤销后旧成功观察不能继续证明当前材料覆盖。"""
        self.record(self.tid, 'acquire_paper_pdf', self.data)
        self.assertTrue(self.coverage.source_coverage(self.tid))
        self.db.execute('DELETE FROM task_evidence WHERE task_id=?', (self.tid,))
        self.assertEqual(self.coverage.source_coverage(self.tid), [])
        self.db.mark_task_evidence(self.tid, self.data['pages'][0]['evidence_id'])
        self.db.execute('UPDATE page_evidence SET text=? WHERE evidence_id=?',
            ('changed source', self.data['pages'][0]['evidence_id']))
        self.assertEqual(self.coverage.source_coverage(self.tid), [])

    def test_same_document_across_tools_cannot_inflate_paper_count(self):
        """同页多次获取和检索只是一篇材料，fulltext字段不声称整篇理解。"""
        self.record(self.tid, 'read_paper', self.data)
        self.record(self.tid, 'acquire_paper_pdf', self.data)
        self.record(self.tid, 'retrieve_evidence', {'hits': [self.hit()]})
        result = self.finish()
        self.assertEqual(result['coverage']['paper_ids'], [self.pid])
        # 明确同版题录别名变化仍归一篇，且不由同文档显示身份授予题录权限。
        alias = 'alias-' + uuid4().hex
        for pid in (self.pid, alias):
            self.db.save_paper(pid, {'paper_id': pid, 'title': 'Synthetic language mechanism',
                'source': 'cvpr', 'published': '2025-06-01',
                'pdf_url': 'https://arxiv.org/pdf/2601.12345v1',
                'paper_url': 'https://arxiv.org/abs/2601.12345v1'})
        self.db.execute('UPDATE documents SET paper_id=? WHERE document_id=?', (alias, self.data['document_id']))
        self.assertEqual(self.finish()['coverage']['paper_ids'], [alias])
        self.assertFalse(self.db.task_has_paper(self.tid, alias))
        raw = deepcopy(self.plan.model_dump(mode='json')); raw['goals'][0]['min_papers'] = 2
        self.record(self.tid, 'set_research_plan', {'research_plan': raw})
        with self.assertRaisesRegex(ValueError, 'lacks qualified papers'):
            self.finish()

    def test_child_acquisition_and_chunk_only_partial_findings_are_reusable(self):
        """partial子任务的有效发现与实际片段可以支持父级，来源/范围保持。"""
        spec = SubtaskSpec(objective='解释机制', mode='investigate', goal_ids=['mechanism'], paper_ids=[self.pid]).model_dump(mode='json')
        child = self.db.create_subtask(self.tid, self.tid, 0, spec['objective'], spec, uuid4().hex)
        cid = child['subtask_id']
        self.db.grant_paper(cid, self.pid, self.tid)
        self.db.grant_evidence(cid, self.data['pages'][0]['evidence_id'], self.tid)
        self.record(cid, 'retrieve_evidence', {'hits': [self.hit()]})
        self.db.save_subtask_result(cid, 'partial', {'findings': ['机制已解释；实验未核对。'], 'paper_ids': [],
            'gaps': ['实验未核对'], 'stop_reason': 'partial findings retained'})
        result = self.finish(subtask_ids=[cid])
        self.assertEqual(result['coverage']['paper_ids'], [self.pid])
        # 多目标目录只引用一份子成果/来源正文，不能按goal数复制整份材料。
        other_spec = {**spec, 'objective': '独立实验问题'}
        other = self.db.create_subtask(self.tid, self.tid, 1, other_spec['objective'], other_spec, uuid4().hex)
        self.db.grant_evidence(other['subtask_id'], self.data['pages'][0]['evidence_id'], self.tid)
        self.record(other['subtask_id'], 'retrieve_evidence', {'hits': [self.hit()]})
        plan = deepcopy(self.plan.model_dump(mode='json'))
        plan['goals'].append({**plan['goals'][0], 'goal_id': 'experiments', 'description': '实验条件'})
        self.record(self.tid, 'set_research_plan', {'research_plan': plan})
        context = self.coverage.current_context(self.tid, {})
        self.assertEqual(len(context['goal_achievements']), 2)
        self.assertEqual(len(context['subtasks']), 2)
        self.assertEqual(sum(len(item['material_coverage']) for item in context['subtasks']), 2)
        for goal in context['goal_achievements']:
            for entry in goal['reuse_candidates']:
                self.assertNotIn('material_coverage', entry)
                self.assertIn(entry['subtask_id'], entry['material_coverage_in'])
                self.assertTrue(all('sources' not in item for item in entry['finding_refs']))
        candidate = context['goal_achievements'][0]['reuse_candidates'][0]
        self.assertEqual(candidate['status'], 'partial')
        self.assertIn('实验未核对', context['subtasks'][0]['gaps'])
        self.assertEqual(candidate['material_summary']['scope_kinds'], ['retrieved_chunk'])
        self.assertEqual(context['goal_achievements'][0]['unaddressed_scope']['semantic_completion'], 'not_assessed')

    def test_plan_wording_is_not_continuation_but_finding_revision_is(self):
        """无需强制新增原文，但不能仅用next_question/缺口改写解除交接。"""
        original = {'findings': [{'finding_id': 'f', 'revision': 1, 'text': '机制A', 'sources': []}],
            'gaps': ['条件未知'], 'next_question': '下一步'}
        planned = deepcopy(original); planned.update(gaps=['改写缺口'], next_question='换句话说')
        self.assertEqual(continuation_fingerprint(original), continuation_fingerprint(planned))
        revised = deepcopy(original); revised['findings'][0]['text'] = '机制A与条件关系'
        self.assertNotEqual(continuation_fingerprint(original), continuation_fingerprint(revised))

    def test_independent_answers_are_grouped_and_bound_answer_is_preserved_once(self):
        """多个独立回答不按最后时间拼接；精确绑定正文与阶段材料分开。"""
        service = ReflectionService(self.db)
        def execute(attempt, target, refs, questions):
            """固定核验失败形状，不假造科学判断。"""
            return {'target_id': target['target_id'], 'content_hash': target['content_hash'],
                'status': 'execution_failed', 'findings': [], 'gaps': ['synthetic reviewer failed']}
        first = service.review_research(self.tid, ResearchReviewInput(target_type='answer',
            answer_markdown='第一独立正文。', reason='核对', verification_questions=['问题A']), execute)
        other = self.db.create_subtask(self.tid, self.tid, 0, '独立问题B',
            SubtaskSpec(objective='独立问题B').model_dump(mode='json'), uuid4().hex)
        service.review_research(self.tid, ResearchReviewInput(target_type='answer', source_task_id=other['subtask_id'],
            answer_markdown='第二独立正文。', reason='核对', verification_questions=['问题B']), execute)
        worker = TaskExecutor(self.settings, self.db)
        result = worker._deliverable_result(self.tid, {'coverage_required': True}, 'synthetic failure')
        self.assertIsNone(result.retained_answer_source['primary_target'])
        self.assertIn('阶段来源', result.answer_body_markdown)
        bound = worker._deliverable_result(self.tid, {'coverage_required': True,
            'reflection_target_id': first['target_id'], 'reflection_content_hash': first['content_hash']}, 'synthetic failure')
        self.assertEqual(bound.answer_body_markdown, '第一独立正文。')
        self.assertEqual(bound.answer_markdown.count('第一独立正文。'), 1)
        self.assertIn('第二独立正文。', bound.answer_markdown)
        self.assertIn('synthetic failure', bound.processing_summary)
        self.assertEqual(bound.review_status, 'unreviewed')
        self.assertIsNone(bound.saved_path)
