"""临时SQLite验证无题录上传、历史计划及完成资格诊断的真实服务链。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from research_loop.domain.tasks import ActionRecord, FinishTaskInput, ResearchBrief, SavePolicy
from research_loop.services.coverage import CoverageEligibilityError, CoverageService
from research_loop.services.papers import PaperService
from research_loop.storage.mysql import Database
from research_loop.tools.base import ToolContext, execute_spec
from research_loop.tools.delegation import specs
from integration.test_collaboration_contract import FixturePDF


class ScopePipelineTests(unittest.TestCase):
    """只用本地合成PDF替身，实际数据库/页授权/覆盖入口不替换。"""

    def setUp(self):
        """建立全新SQLite，记录模型及外站请求均为零。"""
        blocker = patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden'))
        blocker.start(); self.addCleanup(blocker.stop)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'synthetic.pdf'; self.source.write_bytes(b'synthetic provider input only')
        self.db = Database('sqlite:///' + str(self.root/'scope.db')); self.db.ensure_schema()
        self.db.create_task('root', None, 'synthetic', '解释上传材料。仅限ICLR 2025年的论文。不要保存。',
            'deny', [{'input_ref': 'upload-a', 'local_path': str(self.source)}])
        self.coverage = CoverageService(self.db)
        self.user = {'goal_id': 'mechanism', 'description': '解释上传材料', 'origin': 'user',
                     'source_quote': '解释上传材料', 'min_papers': 1, 'requires_fulltext': True}
        self.brief = ResearchBrief(goal='解释上传材料', requested_count=1,
            sources=['arxiv'], date_from='2026-01-01', input_refs=['upload-a'], goals=[self.user])
        self.data = PaperService(self.db, None, FixturePDF()).read('root', 'upload-a', 1, 1,
            {'upload-a': {'local_path': str(self.source)}})
        self.record('read_paper', self.data)
        self.plan = self.coverage.prepare_plan('root', self.brief)
        self.record('set_research_plan', {'research_plan': self.plan})
        self.payload = FinishTaskInput(answer_markdown='合成材料机制解释。', processing_summary='题录来源和日期未知。',
            goals=[{'goal_id': 'mechanism', 'status': 'satisfied', 'reason': '实际授权原文已返回'}])
        self.action = {'task_id': 'root', 'action_id': 'finish-a', 'input_hash': hashlib.sha256(
            json.dumps(self.payload.model_dump(mode='json'), sort_keys=True, ensure_ascii=False).encode()).hexdigest()}

    def record(self, tool, data, task_id="root"):
        """持久化成功观察，动作身份不参与重复材料判断。"""
        action = ActionRecord(task_id=task_id, sequence=len(self.db.list_actions(task_id))+1,
            tool_name=tool, arguments={}, status='succeeded',
            observation={'ok': True, 'kind': 'success', 'data': data})
        self.db.save_action(action.model_dump(mode='json'))
        return action

    def hard_goal(self, **changes):
        """建立新用户硬约束计划；测试不通过改变原计划来绕过保护。"""
        self.db.execute('DELETE FROM actions WHERE task_id=? AND tool_name=?', ('root', 'set_research_plan'))
        brief = ResearchBrief.model_validate(self.brief.model_dump(mode='json') | {'goals': [self.user | changes]})
        self.plan = self.coverage.prepare_plan('root', brief)
        self.record('set_research_plan', {'research_plan': self.plan})

    def test_upload_without_catalog_satisfies_new_plan_and_never_fabricates_metadata(self):
        """真实页授权满足材料门，模型自定arxiv/日期不剔除上传。"""
        self.assertIsNone(self.db.get_paper('upload-a'))
        result = self.coverage.finish('root', self.payload)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['coverage']['paper_ids'], ['upload-a'])
        self.assertIsNone(self.db.get_paper('upload-a'))
        self.assertEqual(self.db.get_task('root')['save_policy'], 'deny')
        self.assertEqual(self.db.fetchall('SELECT * FROM save_receipts'), [])

    def test_unknown_hard_source_and_date_expose_specific_ownership_and_legal_candidate(self):
        """相同上传不能满足明确来源/日期；诊断和候选正文分开绑定。"""
        self.hard_goal(sources=['iclr'], date_from='2025-01-01', date_to='2025-12-31',
                       source_quote='仅限ICLR 2025年的论文')
        with self.assertRaises(CoverageEligibilityError) as caught:
            self.coverage.finish('root', self.payload, action_identity=self.action)
        error = caught.exception
        self.assertEqual(error.diagnostic['code'], 'GOAL_QUALIFICATION_BLOCKED')
        self.assertEqual({c['fact'] for c in error.diagnostic['conditions']}, {'source_unknown', 'published_unknown'})
        self.assertTrue(all(c['origin'] == 'user' and c['scope'] == 'goal' for c in error.diagnostic['conditions']))
        self.assertNotIn('retrieve more', str(error))
        self.assertIn('provide_verified_metadata_or_matching_authorized_material', error.diagnostic['suggested_actions'])
        self.assertEqual(error.candidate['action_id'], 'finish-a')
        self.assertNotIn('answer_body_markdown', error.candidate)
        self.assertNotIn('processing_summary', error.candidate)
        self.assertEqual(error.candidate['answer_body_chars'], len(self.payload.answer_markdown))
        self.assertEqual(error.candidate['answer_content_hash'], hashlib.sha256(self.payload.answer_markdown.encode()).hexdigest())
        self.assertEqual(error.diagnostic['state_fingerprint'], self.coverage.qualification_fingerprint('root', 'mechanism'))
        unresolved = self.payload.model_copy(update={'goals': [self.payload.goals[0].model_copy(update={
            'status': 'unresolved', 'reason': '上传题录来源和日期未知，不能证明满足用户要求。'})]})
        self.assertEqual(self.coverage.finish('root', unresolved)['status'], 'partial')

    def test_old_persisted_plan_keeps_original_hard_scope_and_original_bytes(self):
        """同形旧计划不能被恢复或prepare静默改为新语义。"""
        self.db.execute('DELETE FROM actions WHERE task_id=? AND tool_name=?', ('root', 'set_research_plan'))
        old = self.brief.model_dump(mode='json')
        recorded = self.record('set_research_plan', {'research_plan': old})
        before = self.db.get_action('root', recorded.sequence)
        prepared = self.coverage.prepare_plan('root', self.brief)
        self.assertNotIn('_scope_semantics', prepared)
        self.assertFalse(self.coverage.in_scope({}, old['goals'][0], old, {'upload-a': 'upload-a'}, 'upload-a'))
        with self.assertRaises(CoverageEligibilityError) as caught:
            self.coverage.finish('root', self.payload, action_identity=self.action)
        self.assertTrue(all(c['scope'] == 'plan' and c['origin'] == 'legacy' for c in caught.exception.diagnostic['conditions']))
        self.assertEqual(self.db.get_action('root', recorded.sequence), before)
        self.record('set_research_plan', {'research_plan': prepared})
        self.assertNotIn('_scope_semantics', self.coverage.plan('root'))

    def test_strategy_condition_is_retractable_without_weakening_user_goal(self):
        """只撤回策略条件及其目标，上传用户目标仍保持原材料门。"""
        strategy = {'goal_id': 'source_check', 'description': '可选会议核对', 'origin': 'strategy', 'sources': ['iclr']}
        revised = self.coverage.prepare_plan('root', ResearchBrief.model_validate(self.plan | {
            'goals': [self.user, strategy]}))
        self.record('set_research_plan', {'research_plan': revised})
        payload = self.payload.model_copy(update={'goals': [*self.payload.goals,
            type(self.payload.goals[0])(goal_id='source_check', status='satisfied', reason='测试错误声明') ]})
        with self.assertRaises(CoverageEligibilityError) as caught:
            self.coverage.finish('root', payload, action_identity=self.action)
        self.assertEqual(caught.exception.diagnostic['goal_id'], 'source_check')
        self.assertIn('revise_or_withdraw_strategy_with_change_reason', caught.exception.diagnostic['suggested_actions'])
        withdrawn = strategy | {'status': 'withdrawn', 'change_reason': '用户未要求会议筛选，已有上传可解释问题。'}
        revised = self.coverage.prepare_plan('root', ResearchBrief.model_validate(revised | {'sources': [], 'date_from': None,
            'goals': [self.user, withdrawn]}))
        self.record('set_research_plan', {'research_plan': revised})
        self.assertEqual(self.coverage.finish('root', self.payload)['status'], 'completed')

    def test_wrong_or_missing_action_identity_does_not_bind_candidate(self):
        """服务直调无服务器行动身份，仍有诊断但不能构造可交付候选。"""
        self.hard_goal(sources=['iclr'])
        for identity in (None, {'task_id': 'other', 'action_id': 'a', 'input_hash': 'b'*64}):
            with self.subTest(identity=identity), self.assertRaises(CoverageEligibilityError) as caught:
                self.coverage.finish('root', self.payload, action_identity=identity)
            self.assertIsNone(caught.exception.candidate)

    def test_foreign_draft_reflection_and_subtask_rejected_before_eligibility(self):
        """外来对象、过期核验hash及空正文不进入合法候选路径。"""
        self.hard_goal(sources=['iclr'])
        self.db.create_task('other', None, 'other', '其他任务', 'deny', [])
        self.db.save_draft({'draft_id': 'foreign-draft', 'task_id': 'other', 'title': '合成外来稿', 'markdown': '外来正文',
                            'content_hash': 'a'*64, 'evidence_ids': []})
        for changes in ({'draft_id': 'foreign-draft'}, {'reflection_target_id': 'foreign-reflection',
                'reflection_content_hash': 'a'*64}, {'answer_markdown': ' '},
                {'goals': [self.payload.goals[0].model_copy(update={'subtask_ids': ['other']})]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError) as caught:
                self.coverage.finish('root', self.payload.model_copy(update=changes), action_identity=self.action)
            self.assertNotIsInstance(caught.exception, CoverageEligibilityError)
            self.assertFalse(hasattr(caught.exception, 'candidate'))

    def test_fingerprint_duplicate_pages_do_not_change_qualification_but_real_version_does(self):
        """完成资格按材料计数，重复页/新增页不改计数，新正文版本会改变。"""
        before = self.coverage.qualification_fingerprint('root', 'mechanism')
        self.record('read_paper', self.data)
        self.assertEqual(self.coverage.qualification_fingerprint('root', 'mechanism'), before)
        data = PaperService(self.db, None, FixturePDF()).read('root', 'upload-a', 2, 1,
            {'upload-a': {'local_path': str(self.source)}})
        self.record('read_paper', data)
        self.assertEqual(self.coverage.qualification_fingerprint('root', 'mechanism'), before)

        class ChangedPDF(FixturePDF):
            """只在合成提供方产生另一真实版本hash。"""
            def ingest(self, source, paper_id):
                """不改变文件或生产记录，提供新的正文版本。"""
                return super().ingest(source, paper_id).model_copy(update={'content_sha256': 'c'*64})

        data = PaperService(self.db, None, ChangedPDF()).read('root', 'upload-a', 1, 1,
            {'upload-a': {'local_path': str(self.source)}})
        self.record('read_paper', data)
        self.assertNotEqual(self.coverage.qualification_fingerprint('root', 'mechanism'), before)
        self.assertIsNone(self.coverage.qualification_fingerprint('root', 'unrelated'))

    def test_fingerprint_ignores_more_wrong_source_pages_until_hard_scope_is_met(self):
        """来源硬阻断只由匹配实际事实解除，无关根页和同材料新页保持阻断。"""
        self.hard_goal(sources=['iclr'])
        before = self.coverage.qualification_fingerprint('root', 'mechanism')
        data = PaperService(self.db, None, FixturePDF()).read('root', 'upload-a', 2, 1,
            {'upload-a': {'local_path': str(self.source)}})
        self.record('read_paper', data)
        self.db.save_paper('unrelated', {'paper_id': 'unrelated', 'source': 'cvpr', 'title': 'Synthetic unrelated',
            'pdf_url': 'https://fixture.invalid/other.pdf'})
        self.db.grant_paper('root', 'unrelated', 'root')
        self.record('read_paper', PaperService(self.db, None, FixturePDF()).read('root', 'unrelated', 1, 1))
        self.assertEqual(self.coverage.qualification_fingerprint('root', 'mechanism'), before)
        self.db.save_paper('upload-a', {'paper_id': 'upload-a', 'source': 'iclr', 'published': '2025'})
        self.assertNotEqual(self.coverage.qualification_fingerprint('root', 'mechanism'), before)

    def test_revocation_still_excludes_upload_from_new_scope(self):
        """新语义不放宽真实页权限，撤权后材料数归零。"""
        self.db.execute('DELETE FROM task_evidence WHERE task_id=?', ('root',))
        self.assertEqual(self.coverage.source_coverage('root'), [])
        with self.assertRaises(CoverageEligibilityError) as caught:
            self.coverage.finish('root', self.payload, action_identity=self.action)
        self.assertEqual(caught.exception.diagnostic['missing_facts']['qualified_papers'], 0)

    def test_real_resume_record_takes_priority_and_cannot_be_replayed_to_weaken_again(self):
        """经实际resume记录重定合同，旧计划先阻断，更新后只生效一次。"""
        self.hard_goal(sources=['iclr'])
        self.db.save_clarification({'task_id': 'root', 'clarification_id': 'clarify-a', 'question': '按什么范围？'})
        answer = '取消会议和日期要求，只解释上传材料，不保存。'
        self.db.answer_clarification('clarify-a', answer)
        self.db.save_resume_request('root', 'clarify-a', 'resume-a', answer)
        with self.assertRaisesRegex(ValueError, 'new user clarification'):
            self.coverage.finish('root', self.payload, action_identity=self.action)
        updated = self.coverage.prepare_plan('root', ResearchBrief.model_validate(self.plan | {
            'goals': [self.user | {'source_quote': answer}], 'save_policy': 'allow'}))
        self.assertEqual(updated['clarification_id'], 'clarify-a')
        self.assertEqual(updated['save_policy'], 'deny')
        self.record('set_research_plan', {'research_plan': updated})
        self.assertEqual(self.coverage.finish('root', self.payload)['status'], 'completed')
        with self.assertRaisesRegex(ValueError, 'existing goals'):
            self.coverage.prepare_plan('root', ResearchBrief.model_validate(updated | {
                'goals': [self.user | {'min_papers': 2, 'source_quote': answer}]}))

    def test_fingerprint_ignores_rewritten_children_and_unrelated_goal_material(self):
        """同材料同版改写objective或无关目标新页都不能解除当前资格阻断。"""
        before = self.coverage.qualification_fingerprint('root', 'mechanism')
        for index, (goal, page) in enumerate([('mechanism', 1), ('mechanism', 1), ('unrelated', 2)]):
            child = self.db.create_subtask('root', 'root', index+1, f'改写研究问题{index}',
                {'goal_ids': [goal], 'input_refs': ['upload-a'], 'mode': 'investigate'}, f'child-{index}')
            child_id = child['subtask_id']
            self.db.update_task(child_id, attachments_json=self.db.get_task('root')['attachments_json'])
            self.db.grant_input(child_id, 'upload-a', 'root')
            data = PaperService(self.db, None, FixturePDF()).read(child_id, 'upload-a', page, 1,
                {'upload-a': {'local_path': str(self.source)}})
            self.record('read_paper', data, child_id)
            self.db.save_subtask_result(child_id, 'partial', {'findings': ['合成发现'], 'paper_ids': ['upload-a']})
            self.assertEqual(self.coverage.qualification_fingerprint('root', 'mechanism'), before)
        child = self.db.create_subtask('root', 'root', 4, '读取尚未处理第二页',
            {'goal_ids': ['mechanism'], 'input_refs': ['upload-a'], 'mode': 'investigate'}, 'new-page')
        child_id = child['subtask_id']
        self.db.update_task(child_id, attachments_json=self.db.get_task('root')['attachments_json'])
        self.db.grant_input(child_id, 'upload-a', 'root')
        data = PaperService(self.db, None, FixturePDF()).read(child_id, 'upload-a', 2, 1,
            {'upload-a': {'local_path': str(self.source)}})
        self.record('read_paper', data, child_id)
        self.assertEqual(self.coverage.qualification_fingerprint('root', 'mechanism'), before)

    def test_fingerprint_tracks_first_ready_child_even_when_its_goal_id_differs(self):
        """finish允许选同根其他目标成果；指纹同样跟踪就绪后的真实材料贡献。"""
        self.db.execute('DELETE FROM actions WHERE task_id=? AND tool_name=?', ('root', 'read_paper'))
        child = self.db.create_subtask('root', 'root', 1, '同材料另一问题',
            {'goal_ids': ['different'], 'input_refs': ['upload-a'], 'mode': 'investigate'}, 'ready-child')
        child_id = child['subtask_id']
        self.db.update_task(child_id, attachments_json=self.db.get_task('root')['attachments_json'])
        self.db.grant_input(child_id, 'upload-a', 'root')
        data = PaperService(self.db, None, FixturePDF()).read(child_id, 'upload-a', 1, 1,
            {'upload-a': {'local_path': str(self.source)}})
        self.record('read_paper', data, child_id)
        before = self.coverage.qualification_fingerprint('root', 'mechanism')
        self.db.save_subtask_result(child_id, 'partial', {'findings': [], 'paper_ids': ['upload-a']})
        self.assertEqual(self.coverage.qualification_fingerprint('root', 'mechanism'), before)
        self.db.save_subtask_result(child_id, 'partial', {'findings': ['首次合成成果'], 'paper_ids': ['upload-a']})
        self.assertNotEqual(self.coverage.qualification_fingerprint('root', 'mechanism'), before)
        payload = self.payload.model_copy(update={'goals': [self.payload.goals[0].model_copy(update={'subtask_ids': [child_id]})]})
        self.assertEqual(self.coverage.finish('root', payload)['status'], 'completed')

    def test_metadata_only_child_cannot_change_already_read_material_qualification(self):
        """已有实际原文时，另一子任务声称同题录成果不制造材料增量。"""
        self.hard_goal(requires_fulltext=False)
        before = self.coverage.qualification_fingerprint('root', 'mechanism')
        child = self.db.create_subtask('root', 'root', 1, '重复同材料题录',
            {'goal_ids': ['mechanism'], 'mode': 'screen'}, 'metadata-duplicate')
        self.db.grant_paper(child['subtask_id'], 'upload-a', 'root')
        self.db.save_subtask_result(child['subtask_id'], 'completed', {
            'findings': ['来自题录的合成发现'], 'paper_ids': ['upload-a']})
        self.assertEqual(self.coverage.qualification_fingerprint('root', 'mechanism'), before)

    def test_candidate_resolver_uses_exact_failed_action_and_rejects_changed_identity(self):
        """正文仅存原arguments；恢复读回须校验行动、参数hash及对象和正文hash。"""
        self.hard_goal(sources=['iclr'])
        with self.assertRaises(CoverageEligibilityError) as caught:
            self.coverage.finish('root', self.payload, action_identity=self.action)
        candidate = caught.exception.candidate
        action = {'task_id': 'root', 'action_id': 'finish-a', 'tool_name': 'finish_task', 'status': 'failed',
                  'arguments': self.payload.model_dump(mode='json')}
        resolved = self.coverage.resolve_finish_candidate('root', action, candidate)
        self.assertEqual(resolved['answer_body_markdown'], self.payload.answer_markdown)
        self.assertEqual(resolved['processing_summary'], self.payload.processing_summary)
        for changes in ({'input_hash': 'b'*64}, {'answer_content_hash': 'c'*64}, {'draft_id': 'foreign'},
                        {'reflection_target_id': 'foreign'}, {'action_id': 'another'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.coverage.resolve_finish_candidate('root', action, candidate | changes)
        changed_action = deepcopy(action); changed_action['arguments']['answer_markdown'] = '改写另一份正文'
        with self.assertRaises(ValueError):
            self.coverage.resolve_finish_candidate('root', changed_action, candidate)

    def test_tool_observation_contains_verified_candidate_and_structured_blocker(self):
        """真实工具参数解析→Coverage→Observation保持精确身份。"""
        self.hard_goal(sources=['iclr'])
        spec = next(item for item in specs() if item.name == 'finish_task')
        context = ToolContext('root', SavePolicy.deny, services={'coverage': self.coverage,
            'finish_action_identity': self.action})
        observation = execute_spec(spec, self.payload.model_dump(mode='json'), context)
        self.assertFalse(observation.ok)
        self.assertFalse(observation.retryable)
        self.assertEqual(observation.data['finish_candidate']['input_hash'], self.action['input_hash'])
        self.assertEqual(observation.data['coverage_blocker']['goal_id'], 'mechanism')
