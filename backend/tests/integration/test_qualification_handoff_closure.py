"""资格失败后的图→Worker闭环；默认临时SQLite禁socket；显式专用MySQL由runner放行数据库。"""
from copy import deepcopy
import json
import os
from contextlib import nullcontext
from urllib.parse import urlparse
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from research_loop.config import Settings
from research_loop.domain.tasks import TaskCreate
from research_loop.domain.evidence import PageEvidence
from research_loop.providers.deepseek import validate_messages
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_collaboration_contract import FixturePDF
from integration.test_research_closure_pipeline import reply


class QualificationScript:
    """固定工具选择替身，只测试状态及身份接线。"""

    def __init__(self, pid, path):
        """保存材料及测试路径；调用数是本地实际替身请求数。"""
        self.pid, self.path, self.calls, self.step = pid, path, [], 0

    def plan(self):
        """来源硬条件和自拟步骤分别使用既有目标归属。"""
        goals = [{'goal_id': 'mechanism', 'description': '解释方法', 'origin': 'user',
            'source_quote': '解释方法', 'min_papers': 1, 'requires_fulltext': True,
            'sources': [] if self.path.startswith('withdraw') else ['iclr']}]
        if self.path.startswith('withdraw'):
            goals.append({'goal_id': 'strategy', 'description': '先检查arxiv来源', 'origin': 'strategy',
                'min_papers': 1, 'requires_fulltext': True, 'sources': ['arxiv']})
        return {'goal': '解释方法', 'task_kind': 'search', 'requested_count': 1, 'count_confirmed': True, 'goals': goals}

    def send(self, messages, tools, **kwargs):
        """脚本绝不发外部请求，按真实工具观察选择精确身份。"""
        validate_messages(messages)
        self.calls.append({'messages': deepcopy(messages), 'tools': deepcopy(tools)})
        observations = [json.loads(m['content']) for m in messages if m.get('role') == 'tool']
        names = {t['function']['name'] for t in tools}
        if 'finish_research' in names:
            data = next((o['data'] for o in observations if o.get('data', {}).get('pages')), None)
            if not data:
                return reply('read_paper', {'target_ref': self.pid, 'start_page': 1, 'page_count': 1})
            return reply('finish_research', {'findings': ['合成方法发现。'], 'paper_ids': [self.pid],
                'evidence_ids': [data['pages'][0]['evidence_id']]})
        step = self.step; self.step += 1
        if step == 0:
            return reply('set_research_plan', self.plan())
        if step == 1:
            return reply('read_paper', {'target_ref': self.pid, 'start_page': 1, 'page_count': 1})
        if step == 2:
            return reply('finish_task', {'answer_markdown': '明确候选正文，仅说明材料方法。', 'processing_summary': '候选处理说明。',
                'goals': [{'goal_id': goal['goal_id'], 'status': 'satisfied', 'reason': '基于合成原文'} for goal in self.plan()['goals']]})
        if self.path.startswith('withdraw'):
            if step == 3:
                plan = self.plan(); plan['goals'][1].update(status='withdrawn', change_reason='自拟来源步骤与本次材料无关')
                return reply('set_research_plan', plan)
            if self.path in {'withdraw_fail', 'withdraw_cancel'}:
                if step == 4:
                    return reply('report_task_gap', {'requirement': '未核对实验', 'reason': '本轮仅方法'})
                if self.path == 'withdraw_cancel':
                    self.cancel()
                raise RuntimeError('synthetic post-correction failure')
            return reply('finish_task', {'answer_markdown': '撤回自拟步骤后交付方法。',
                'goals': [{'goal_id': 'mechanism', 'status': 'satisfied', 'reason': '已读取授权材料'}]})
        if self.path == 'inactive_goals':
            plan = self.plan()
            plan['goals'].extend({'goal_id': 'withdrawn-'+str(number), 'description': '已撤回自拟步骤',
                'origin': 'strategy', 'status': 'withdrawn', 'change_reason': '该步骤不需要执行',
                'min_papers': 0} for number in range(3, step+1))
            return reply('set_research_plan', plan)
        if self.path.startswith('new_goal'):
            if step == 3:
                plan = self.plan(); plan['goals'].append({'goal_id': 'experiments', 'description': '比较实验问题',
                    'origin': 'strategy', 'min_papers': 1, 'requires_fulltext': True})
                return reply('set_research_plan', plan)
            if step == 4 or self.path in {'new_goal_repeat', 'new_goal_rewrite'}:
                return reply('delegate_research', {'subtasks': [{'objective': '同材料另一实验问题'+(str(step) if self.path == 'new_goal_rewrite' else ''), 'mode': 'investigate',
                    'paper_ids': [self.pid], 'goal_ids': ['experiments'], 'start_page': 1, 'page_count': 1}]})
            children = [c['subtask_id'] for o in observations for c in o.get('data', {}).get('subtasks', [])]
            return reply('finish_task', {'answer_markdown': '方法来源资格未满足，另一个实验问题已有成果。',
                'goals': [{'goal_id': 'mechanism', 'status': 'unresolved', 'reason': '缺硬来源'},
                    {'goal_id': 'experiments', 'status': 'satisfied', 'reason': '子成果', 'subtask_ids': children}]})
        if self.path == 'multi_new_pages':
            if step in (3, 4, 5):
                return reply('read_paper', {'target_ref': self.pid, 'start_page': step-1, 'page_count': 1})
            return reply('finish_task', {'answer_markdown': '连续新页已读取，原来源资格未满足。',
                'goals': [{'goal_id': 'mechanism', 'status': 'unresolved', 'reason': '实际来源仍非ICLR'}]})
        if self.path == 'new_page':
            if step == 3:
                return reply('read_paper', {'target_ref': self.pid, 'start_page': 2, 'page_count': 1})
            return reply('finish_task', {'answer_markdown': '新页范围已读，来源资格仍明确披露。',
                'goals': [{'goal_id': 'mechanism', 'status': 'unresolved', 'reason': '增加页面不改变会议来源'}]})
        if self.path == 'unresolved':
            return reply('finish_task', {'answer_markdown': '明确候选正文，仅说明材料方法。',
                'processing_summary': 'ICLR资格信息未满足，明确披露。',
                'goals': [{'goal_id': 'mechanism', 'status': 'unresolved', 'reason': '当前材料不满足ICLR硬来源，未新增证据'}]})
        if self.path == 'repeat_read':
            return reply('read_paper', {'target_ref': self.pid, 'start_page': 1, 'page_count': 1})
        if self.path == 'rewrite_finish':
            return reply('finish_task', {'answer_markdown': '改写第'+str(step)+'版候选。',
                'goals': [{'goal_id': 'mechanism', 'status': 'satisfied', 'reason': '改写原因'+str(step)}]})
        if self.path == 'rewrite_delegate':
            return reply('delegate_research', {'subtasks': [{'objective': '改写方法问题'+str(step), 'mode': 'investigate',
                'paper_ids': [self.pid], 'goal_ids': ['mechanism'], 'start_page': 1, 'page_count': 1}]})
        raise AssertionError('unexpected script path')

    def close(self):
        """替身没有资源待关闭。"""


class QualificationPipelineTests(unittest.TestCase):
    """运行真实Worker状态机，所有业务模型/外站/生产DB调用为0。"""

    def run_path(self, path):
        """临时建任务并完成图运行，返回可检查的实际库行与替身轨迹。"""
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        url = os.getenv('FIVE_ROOT_TEST_DATABASE_URL') or 'sqlite:///'+str(root/'qualification.db')
        if not url.startswith('sqlite:'):
            parsed = urlparse(url)
            if not parsed.scheme.startswith('mysql') or not parsed.path.strip('/').startswith('researchloop_test_'):
                raise ValueError('only explicit mysql researchloop_test_ database is allowed')
        settings = Settings(project_root=root, data_dir=root/'data', output_dir=root/'notes',
            database_url=url, retrieval_enabled=False)
        db = Database(settings.database_url); db.ensure_schema()
        tid = TaskService(db, settings).create(TaskCreate(prompt='解释方法，不保存' if path.startswith('withdraw') else '解释方法；只用ICLR来源，不保存',
            save_policy='deny', idempotency_key=uuid4().hex)).task_id
        pid = 'qualification-'+uuid4().hex
        db.save_paper(pid, {'paper_id': pid, 'title': 'Synthetic mechanism', 'source': 'cvpr',
            'published': '2025-06-01', 'pdf_url': 'https://fixture.invalid/p.pdf'})
        db.grant_paper(tid, pid, tid)
        script = QualificationScript(pid, path)
        script.cancel = lambda: db.update_task(tid, cancel_requested=1)
        class ExtendedPDF(FixturePDF):
            """只为连续新页分支提供固定五页合成原文。"""
            def ingest(self, source, paper_id):
                """保持原身份/hash，仅合成fixture页数变为五页。"""
                document = super().ingest(source, paper_id)
                return document.model_copy(update={'total_pages': 5})
            def pages(self, document, start, count):
                """生成真实页序和稳定内容，不调用网络或模型。"""
                return [PageEvidence(evidence_id=f'evidence:{document.paper_id}:{number}',
                    document_id=document.document_id, page_number=number,
                    text=f'{document.paper_id}: synthetic physical page {number}',
                    parsed_markdown=f'page {number}') for number in range(start, min(start+count, 6))]
        worker = TaskExecutor(settings, db, lambda _: script,
                              lambda *_: ExtendedPDF() if path == 'multi_new_pages' else FixturePDF())
        # SQLite禁止全部socket；显式专用MySQL由调用runner只放行数据库端口。
        blocker = patch.object(socket.socket, 'connect', side_effect=AssertionError('all sockets forbidden')) if url.startswith('sqlite:') else nullcontext()
        with blocker:
            result = worker.execute(tid)
        return db, tid, result, script

    def test_explicit_unresolved_finish_delivers_and_stops_model(self):
        """硬资格不满足时可明确unresolved正式交付，不再追加模型请求。"""
        db, tid, result, script = self.run_path('unresolved')
        self.assertEqual(db.get_task(tid)['status'], 'partial')
        self.assertEqual(len(script.calls), 4)
        self.assertEqual(result.answer_body_markdown, '明确候选正文，仅说明材料方法。')
        self.assertIn('ICLR资格', result.processing_summary)
        self.assertEqual(db.list_actions(tid)[-1]['status'], 'succeeded')
        failure = db.list_actions(tid)[2]
        candidate = failure['observation']['data']['finish_candidate']
        self.assertEqual(candidate['action_id'], failure['action_id'])
        self.assertEqual(candidate['task_id'], tid)
        self.assertNotIn('answer_body_markdown', candidate)
        self.assertNotIn('processing_summary', candidate)

    def test_strategy_can_be_withdrawn_and_original_user_goal_delivered(self):
        """修正自拟资格步骤释放交接，用户问题仍正常满足。"""
        db, tid, result, script = self.run_path('withdraw')
        self.assertEqual(db.get_task(tid)['status'], 'completed', result.model_dump())
        self.assertEqual(len(script.calls), 5)
        self.assertEqual(result.answer_body_markdown, '撤回自拟步骤后交付方法。')

    def test_same_reads_do_not_clear_qualification_and_keep_candidate(self):
        """重复读取不制造资格进展，原候选正文优先于分组阶段材料。"""
        db, tid, result, script = self.run_path('repeat_read')
        self.assertEqual(db.get_task(tid)['status'], 'partial')
        self.assertEqual(result.reason, 'qualification handoff unresolved')
        self.assertEqual(result.answer_body_markdown, '明确候选正文，仅说明材料方法。')
        self.assertLessEqual(len(script.calls), 5)
        self.assertEqual(len([a for a in db.list_actions(tid) if a['tool_name'] == 'finish_task']), 1)

    def test_rewritten_finish_cannot_reset_qualification_boundary(self):
        """改写交付参数仍有同一资格问题，不能反复重开纠错。"""
        db, tid, result, script = self.run_path('rewrite_finish')
        self.assertEqual(db.get_task(tid)['status'], 'partial')
        self.assertEqual(result.reason, 'qualification handoff unresolved')
        self.assertLessEqual(len(script.calls), 5)
        self.assertEqual(result.retained_answer_source['action_id'], db.list_actions(tid)[-1]['action_id'])

    def test_rewritten_delegation_cannot_clear_same_qualification(self):
        """完整spec仍创建独立子任务，但新objective不解除未变化的资格条件。"""
        db, tid, result, script = self.run_path('rewrite_delegate')
        self.assertEqual(db.get_task(tid)['status'], 'partial')
        self.assertEqual(result.reason, 'qualification handoff unresolved')
        self.assertEqual(result.answer_body_markdown, '明确候选正文，仅说明材料方法。')
        children = db.list_subtasks(tid)
        self.assertEqual(len(children), 2)
        self.assertNotEqual(children[0]['objective'], children[1]['objective'])

    def test_other_goal_on_same_material_remains_available(self):
        """旧目标仍阻断时，新目标可处理相同材料的另一个问题并交付。"""
        db, tid, result, script = self.run_path('new_goal')
        self.assertEqual(db.get_task(tid)['status'], 'partial', result.model_dump())
        self.assertEqual(result.reason, 'task delivered')
        self.assertEqual(len(db.list_subtasks(tid)), 1)
        self.assertIn('另一个实验问题已有成果', result.answer_body_markdown)

    def test_unread_page_scope_is_allowed_after_blocker(self):
        """未处理页范围允许读取，取得新范围后仍可明确unresolved结束。"""
        db, tid, result, script = self.run_path('new_page')
        self.assertEqual(result.reason, 'task delivered', result.model_dump())
        reads = [a for a in db.list_actions(tid) if a['tool_name'] == 'read_paper']
        self.assertEqual([a['arguments']['start_page'] for a in reads], [1, 2])
        self.assertTrue(all(a['status'] == 'succeeded' for a in reads))

    def test_multiple_unprocessed_ranges_continue_without_clearing_qualification(self):
        """连续三批新页作为范围导航继续，旧硬来源阻断仍须unresolved披露。"""
        db, tid, result, script = self.run_path('multi_new_pages')
        self.assertEqual(result.reason, 'task delivered', result.model_dump())
        reads = [a for a in db.list_actions(tid) if a['tool_name'] == 'read_paper']
        self.assertEqual([a['arguments']['start_page'] for a in reads], [1, 2, 3, 4])
        self.assertEqual(db.get_task(tid)['status'], 'partial')

    def test_candidate_survives_resolved_qualification_and_later_failure(self):
        """资格已撤回且其他工具执行后异常，原合法候选引用仍独立保留。"""
        db, tid, result, script = self.run_path('withdraw_fail')
        self.assertEqual(result.answer_body_markdown, '明确候选正文，仅说明材料方法。', result.model_dump())
        self.assertEqual(db.get_task(tid)['status'], 'partial')
        self.assertIn('synthetic post-correction failure', result.reason)
        self.assertEqual(db.list_actions(tid)[-1]['tool_name'], 'report_task_gap')

    def test_candidate_survives_resolved_qualification_and_later_cancellation(self):
        """资格已撤回后的取消保留当前候选，终态仍为cancelled。"""
        db, tid, result, script = self.run_path('withdraw_cancel')
        self.assertEqual(result.answer_body_markdown, '明确候选正文，仅说明材料方法。', result.model_dump())
        self.assertEqual(db.get_task(tid)['status'], 'cancelled')

    def test_exact_other_goal_reuse_cannot_extend_blocker_forever(self):
        """另一目标精确spec复用同child允许幂等，但不能每轮豁免旧资格收尾。"""
        db, tid, result, script = self.run_path('new_goal_repeat')
        self.assertEqual(result.reason, 'qualification handoff unresolved', result.model_dump())
        self.assertEqual(db.get_task(tid)['status'], 'partial')
        self.assertEqual(len(db.list_subtasks(tid)), 1)
        delegated = [a['observation']['data']['subtasks'][0]['subtask_id'] for a in db.list_actions(tid)
            if a['tool_name'] == 'delegate_research' and a['observation'].get('ok')]
        self.assertGreaterEqual(len(delegated), 2)
        self.assertEqual(len(set(delegated)), 1)
        self.assertLessEqual(len(script.calls), 10)

    def test_new_withdrawn_goal_cannot_extend_blocker(self):
        """新增已撤回策略身份不是新active目标，不能反复刷新资格收尾。"""
        db, tid, result, script = self.run_path('inactive_goals')
        self.assertEqual(result.reason, 'qualification handoff unresolved', result.model_dump())
        self.assertEqual(db.get_task(tid)['status'], 'partial')
        plans = [a for a in db.list_actions(tid) if a['tool_name'] == 'set_research_plan']
        self.assertEqual(len(plans), 3)
        self.assertTrue(all(a['status'] == 'succeeded' for a in plans))
        additions = plans[-1]['observation']['data']['research_plan']['goals'][1:]
        self.assertTrue(additions)
        self.assertTrue(all(goal['status'] == 'withdrawn' for goal in additions))
        self.assertEqual(len(script.calls), 5)

    def test_other_goal_objective_rewrite_without_contribution_cannot_extend_blocker(self):
        """另一目标只改objective生成新child，同范围/同发现仍不能无限续期。"""
        db, tid, result, script = self.run_path('new_goal_rewrite')
        self.assertEqual(result.reason, 'qualification handoff unresolved', result.model_dump())
        self.assertEqual(db.get_task(tid)['status'], 'partial')
        children = db.list_subtasks(tid)
        self.assertEqual(len(children), 3)
        self.assertEqual(len({child['objective'] for child in children}), 3)
        findings = [tuple(child['result']['result']['findings']) for child in children]
        self.assertEqual(len(set(findings)), 1)
        self.assertLessEqual(len(script.calls), 14)
