"""通用核验对象的零模型合同；所有语义结论均为明确测试替身。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from research_loop.agents.reflection import ReflectionService
from research_loop.domain.reflection import (ResearchReviewInput, ResearchRevisionInput,
    CloseResearchIssueInput, TargetAssessment, review_progress)
from research_loop.domain.tasks import TaskCreate
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database


class ResearchTargetTests(unittest.TestCase):
    """临时SQLite或显式隔离MySQL，不读取生产模型配置。"""

    def setUp(self):
        """建立独立根任务，默认禁止保存。"""
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        url = os.getenv('REFLECTION_TEST_DATABASE_URL') or 'sqlite:///' + str(Path(self.temp.name)/'test.db')
        if url.startswith('mysql') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_'):
            raise ValueError('isolated test database required')
        self.db = Database(url); self.db.ensure_schema()
        self.task = TaskService(self.db).create(TaskCreate(prompt='按需核验，禁止保存', save_policy='deny')).task_id
        self.service = ReflectionService(self.db)
        self.calls = []

    def args(self, result=None, **changes):
        """构造首轮或当前版本核验请求。"""
        value = {'target_type': 'answer', 'answer_markdown': '保留开头。错误句。保留结尾。',
                 'reason': '核验具体疑点', 'verification_questions': ['当前方法是否使用语言信息？']}
        if result:
            value.pop('answer_markdown')
            target = result.get('target') or result
            value.update(target_id=target['target_id'], content_hash=target['content_hash'])
        return ResearchReviewInput(**{**value, **changes})

    def execute(self, status='contradicted', **extras):
        """提供显式结构化判断替身，并记录回调次数。"""
        def callback(attempt, target, refs, questions):
            self.calls.append(attempt)
            return {'review_id': uuid4().hex, 'target_id': target['target_id'], 'content_hash': target['content_hash'],
                'status': 'supported' if status == 'supported' else ('execution_failed' if status == 'execution_failed' else 'issues'),
                'findings': [{'item_id': i['item_id'], 'status': status, 'reason': 'fixture',
                              'suggested_action': 'keep' if status == 'supported' else 'revise'} for i in target['items']],
                'gaps': [], **extras}
        return callback

    def review(self, result=None, status='contradicted', **changes):
        """调用正式服务，独立语义判断来自固定回调。"""
        return self.service.review_research(self.task, self.args(result, **changes), self.execute(status))

    def revise(self, result, old='错误句。', new='修正句。'):
        """只改一个唯一原句，使用服务返回的当前身份。"""
        return self.service.revise_research(self.task, ResearchRevisionInput(target_id=result['target_id'],
            content_hash=result['target']['content_hash'], review_id=result['review_id'], patches=[{
                'item_id': 'answer', 'old_text': old, 'new_text': new, 'reason': '按已取得反证修改'}]))

    def test_exact_replay_patch_and_independent_current_version(self):
        """恢复复用判断，局部修改保留其他字节，旧hash拒绝。"""
        first = self.review(); self.review()
        self.assertEqual(len(self.calls), 1)
        updated = self.revise(first)
        self.assertEqual(updated['target']['items'][0]['text'], '保留开头。修正句。保留结尾。')
        self.assertEqual(self.revise(first)['target']['content_hash'], updated['target']['content_hash'])
        with self.assertRaises(ValueError): self.review(first)
        second = self.review(updated, status='supported')
        self.assertEqual(second['quality_status'], 'supported')
        self.assertEqual(second['reflection']['progress']['resolved_item_ids'], ['answer'])
        self.assertFalse(self.db.fetchall('SELECT * FROM drafts WHERE task_id=?', (self.task,)))

    def test_no_progress_is_per_problem_and_stops_without_approval(self):
        """同一问题重复反证不会因改了hash而视为已修复。"""
        first = self.review(); updated = self.revise(first)
        second = self.review(updated)
        self.assertEqual(second['reflection']['stop_reason'], 'no_progress')
        third = self.review(updated)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(third['status'], 'issues')
        with self.assertRaises(ValueError): self.revise(second, '修正句。', '再改句。')

    def test_other_question_cannot_clear_old_unresolved_issue(self):
        """新问题不继承旧判词，也不能隐藏前一个未解决的问题。"""
        first = self.review()
        second = self.review(first, status='supported', verification_questions=['另一独立问题是否有据？'])
        self.assertEqual(second['status'], 'supported')
        self.assertEqual(second['quality_status'], 'issues')
        self.assertEqual(len(second['pending_checks']), 1)

    def test_wrong_parameters_do_not_consume_attempts(self):
        """参数错误不算语义无进展，版本检查先于独立核验。"""
        first = self.review()
        with self.assertRaises(ValueError): self.review(first, target_type='findings')
        with self.assertRaises(ValueError): self.review(first, content_hash='0'*64)
        with self.assertRaises(ValueError): self.revise(first, old='不存在')
        self.assertEqual(len(self.calls), 1)

    def test_cancel_and_lost_lease_block_new_review(self):
        """权限拒绝在回调前发生，不能伪造核验判断。"""
        self.service = ReflectionService(self.db, owner=('missing', 'owner'))
        with self.assertRaises(PermissionError): self.review()
        self.service = ReflectionService(self.db)
        self.db.update_task(self.task, cancel_requested=1)
        with self.assertRaises(PermissionError): self.review()
        self.assertEqual(self.calls, [])

    def test_different_target_not_locked_by_supported_answer(self):
        """回答的结束不影响另一个实际候选对象。"""
        self.review(status='supported')
        # 通过真实成功action结构提供候选，不靠全局papers表冒充检索结果。
        from research_loop.domain.tasks import ActionRecord
        pid = 'candidate-' + uuid4().hex
        candidate = {'paper_id': pid, 'title': 'Segmentation', 'abstract': 'Semantic segmentation', 'source': 'cvpr'}
        self.db.save_paper(pid, candidate); self.db.grant_paper(self.task, pid, self.task)
        action = ActionRecord(action_id=uuid4().hex, task_id=self.task, sequence=1, tool_name='search_papers',
            arguments={'query': 'detection'}, status='succeeded', observation={'ok': True, 'kind': 'success', 'data': {'candidates': [candidate]}})
        self.db.save_action(action.model_dump())
        args = ResearchReviewInput(target_type='candidates', reason='判断主题是否匹配', verification_questions=['是否属于检测？'])
        result = self.service.review_research(self.task, args, self.execute('insufficient'))
        self.assertEqual(len(self.calls), 2)
        self.assertNotEqual(result['target_id'], self.review(status='supported')['target_id'])

    def test_unknown_execution_does_not_become_semantic_error(self):
        """执行失败保留为独立状态，同输入不自动重发。"""
        first = self.review(status='execution_failed'); self.review()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(first['reflection']['stop_reason'], 'review_execution_failed')

    def test_progress_uses_same_items_not_fewer_messages(self):
        """移除意见不等于解决原问题，缺证转反证才是具体问题被查清。"""
        old = {'findings': [{'item_id': 'a', 'status': 'contradicted'}, {'item_id': 'b', 'status': 'insufficient'}]}
        self.assertFalse(review_progress(old, {'findings': [{'item_id': 'a', 'status': 'contradicted'}]})['effective_change'])
        changed = review_progress(old, {'findings': [{'item_id': 'b', 'status': 'contradicted'}]})
        self.assertEqual(changed['clarified_item_ids'], ['b'])

    def test_cross_task_target_and_source_rejected(self):
        """模型不能自行指定外部任务/对象读取隐私材料。"""
        other = TaskService(self.db).create(TaskCreate(prompt='other')).task_id
        with self.assertRaises(PermissionError): self.review(target_type='findings', source_task_id=other, answer_markdown=None)
        with self.assertRaises(ValueError): self.service.research_target(other, self.review()['target_id'])

    def test_explicit_disclosure_is_idempotent_not_approval(self):
        """结束选择保留原意见，不能变成supported或保存授权。"""
        result = self.review(status='insufficient')
        payload = CloseResearchIssueInput(target_id=result['target_id'], content_hash=result['content_hash'],
            review_id=result['review_id'], action='disclose', reason='在已读取范围无法确认，不作全篇否定。')
        first = self.service.close_research(self.task, payload)
        self.assertEqual(first, self.service.close_research(self.task, payload))
        self.assertEqual(first['review_status'], 'issues')
        self.review(result, verification_questions=['重问能否重置？'])
        self.assertEqual(len(self.calls), 1)

    def test_incomplete_basis_cannot_be_contradiction(self):
        """不存在原文反证时，参数合同不能把缺证升级成错误。"""
        with self.assertRaises(ValueError):
            TargetAssessment(item_id='a', status='contradicted', basis='not_found', reason='未找到', suggested_action='revise')

    def test_cyclic_patch_cannot_reset_progress_with_new_hash(self):
        """A→B→A原文循环不能借添加证据或新动作包装通过。"""
        first = self.review(status='insufficient'); updated = self.revise(first)
        second = self.review(updated, verification_questions=['另一个实际问题？'])
        with self.assertRaisesRegex(ValueError, 'existing version'):
            self.revise(second, old='修正句。', new='错误句。')

    def test_resume_reserved_attempt_keeps_identity(self):
        """会话结算前退出后沿用相同attempt，不新建审校轮次。"""
        reserved = []
        def killed(attempt, target, refs, questions):
            reserved.append(attempt); raise SystemExit('simulated exit after receipt')
        with self.assertRaises(SystemExit): self.service.review_research(self.task, self.args(), killed)
        self.review()
        self.assertEqual(reserved, self.calls)

    def test_old_problem_can_be_revised_after_different_question(self):
        """无关问题的后续核验不使尚未修改的原问题失去可修订身份。"""
        first = self.review()
        self.review(first, status='supported', verification_questions=['另一问题？'])
        revised = self.revise(first)
        self.assertIn('修正句', revised['target']['items'][0]['text'])

    def test_audit_wrappers_do_not_change_target_content_identity(self):
        """来源版本相同而行动序号不同不产生新研究内容身份。"""
        from research_loop.domain.reflection import target_fingerprint
        before = {'target_type': 'candidates', 'source_task_id': 'task', 'items': [
            {'item_id': 'p', 'metadata': {'title': 'title'}, 'source_action_id': 'a'}], 'origin': {}}
        after = json.loads(json.dumps(before)); after['items'][0]['source_action_id'] = 'b'
        self.assertEqual(target_fingerprint(before), target_fingerprint(after))

    def test_scope_disclosure_cannot_be_used_for_contradiction(self):
        """范围披露是独立类别，不能作为原文反证捷径。"""
        with self.assertRaises(ValueError):
            TargetAssessment(item_id='a', status='contradicted', basis='scope_disclosure',
                             reason='没有找到', evidence_ids=['e'], suggested_action='revise')

    def source(self, text='The method uses language information.'):
        """创建真实存储形状的合成页与授权，用于位置/版本验证。"""
        from research_loop.domain.evidence import Document, PageEvidence
        doc = Document(document_id=uuid4().hex, paper_id=uuid4().hex, content_sha256='a'*64,
                       parser_name='fixture', parser_version='1', config_sha256='b'*64, total_pages=1, cache_path='unused')
        self.db.save_document(doc.model_dump())
        ref = uuid4().hex
        self.db.save_page(PageEvidence(evidence_id=ref, document_id=doc.document_id, page_number=1, text=text).model_dump())
        self.db.mark_task_evidence(self.task, ref)
        return ref

    def test_duplicate_source_id_is_not_new_evidence(self):
        """同一原文位置/文字的新evidence_id不能产生额外核验调用。"""
        ref = self.source()
        first = self.review(evidence_ids=[ref])
        clone = dict(self.db.get_page(ref)); clone['evidence_id'] = uuid4().hex
        self.db.save_page(clone); self.db.mark_task_evidence(self.task, clone['evidence_id'])
        self.review(first, evidence_ids=[clone['evidence_id']])
        self.assertEqual(len(self.calls), 1)

    def test_changed_source_invalidates_delivery_identity(self):
        """已核验页被改写后，旧判断不能批准当前交付。"""
        ref = self.source(); first = self.review(status='supported', evidence_ids=[ref])
        self.db.execute('UPDATE page_evidence SET text=? WHERE evidence_id=?', ('Changed source.', ref))
        with self.assertRaisesRegex(ValueError, 'source version changed'):
            self.service.research_target(self.task, first['target_id'], first['content_hash'])

    def test_review_validator_checks_selection_metadata_and_quotes(self):
        """引句与元数据字段只接受实际存在的内容，不能漏项或错引另一页。"""
        from research_loop.services.verification import validate_target_assessments
        from research_loop.services.papers import PaperService
        papers = PaperService(self.db, None, None)
        ref = self.source()
        target = {'target_type': 'findings', 'items': [{'item_id': 'a', 'text': 'Uses language.'}]}
        valid = TargetAssessment(item_id='a', status='supported', basis='direct_statement',
            evidence_ids=[ref], source_quote='The method uses language information.', reason='原文直接说明', suggested_action='keep')
        validate_target_assessments(self.task, target, [valid], papers)
        with self.assertRaises(ValueError): validate_target_assessments(self.task, target, [], papers)
        with self.assertRaisesRegex(ValueError, 'source_quote'):
            validate_target_assessments(self.task, target, [valid.model_copy(update={'source_quote': 'Never uses language.'})], papers)
        candidate = {'target_type': 'candidates', 'items': [{'item_id': 'p', 'metadata': {'title': 'Segmentation'}}]}
        finding = TargetAssessment(item_id='p', suitability='unsuitable', basis='metadata', metadata_fields=['abstract'],
            reason='主题不适用', suggested_action='exclude')
        with self.assertRaisesRegex(ValueError, 'metadata_fields'):
            validate_target_assessments(self.task, candidate, [finding], papers)

    def test_existing_round_configuration_cannot_be_reset_by_questions(self):
        """使用原有max_rounds配置，新问题身份不能重置对象已有尝试数。"""
        self.service = ReflectionService(self.db, max_rounds=0)
        first = self.review()
        second = self.review(first, verification_questions=['换个问题措辞？'])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(second['reflection']['stop_reason'], 'round_limit')

    def test_cancel_after_callback_rejects_result_commit(self):
        """核验期间取消后不把结果写成已通过；成功调用身份仍可审计。"""
        original = self.execute('supported')
        def cancelled(attempt, target, refs, questions):
            result = original(attempt, target, refs, questions)
            self.db.update_task(self.task, cancel_requested=1)
            return result
        with self.assertRaises(PermissionError):
            self.service.review_research(self.task, self.args(), cancelled)
        record = next(iter(self.service.snapshot(self.task)['research_targets'].values()))
        self.assertFalse(record['attempts'][0].get('result'))

    def test_disclosure_after_local_edit_does_not_require_more_calls(self):
        """无法继续时可结束未复审新版本，不继承原版审校通过。"""
        first = self.review(status='insufficient')
        updated = self.revise(first, new='已读范围无法确认该方法设置。')
        payload = CloseResearchIssueInput(target_id=first['target_id'], content_hash=updated['target']['content_hash'],
            review_id=first['review_id'], action='disclose', reason='无更多可用依据，交付未复审的范围说明。')
        result = self.service.close_research(self.task, payload)
        self.assertEqual(result['review_status'], 'unreviewed')
        self.assertEqual(len(self.calls), 1)

    def test_reviewers_own_supplement_is_not_repeated_as_new_evidence(self):
        """独立角色已补读的页再次入包也复用原判断。"""
        ref = self.source()
        initial = self.execute('insufficient')
        def supplemented(attempt, target, refs, questions):
            result = initial(attempt, target, refs, questions)
            result['findings'][0]['evidence_ids'] = [ref]
            return result
        result = self.service.review_research(self.task, self.args(), supplemented)
        self.review(result, evidence_ids=[ref])
        self.assertEqual(len(self.calls), 1)

    def test_refresh_authoritative_finding_and_independent_selection(self):
        """新进展显式刷新不重置尝试；同一子目标中另一发现可独立核验。"""
        from research_loop.domain.tasks import ActionRecord
        from research_loop.services.research_progress import ResearchProgressService
        ref = self.source(); source = ResearchProgressService(self.db).source(self.task, ref)
        def stage(number, first_text):
            """用成功行动形状记录两个稳定发现，模拟Researcher提交。"""
            progress = {'revision': number, 'findings': [
                {'finding_id': 'f1', 'text': first_text, 'sources': [source], 'verification_status': 'unreviewed'},
                {'finding_id': 'f2', 'text': '另一独立发现。', 'sources': [source], 'verification_status': 'unreviewed'}], 'gaps': []}
            action = ActionRecord(action_id=uuid4().hex, task_id=self.task, sequence=number,
                tool_name='record_research_progress', arguments={}, status='succeeded',
                observation={'ok': True, 'kind': 'success', 'data': {'research_progress': progress}})
            self.db.save_action(action.model_dump())
        stage(1, '原待确认发现。')
        args = ResearchReviewInput(target_type='findings', source_task_id=self.task, item_ids=['f1'],
            reason='核对发现', verification_questions=['该发现是否有据？'])
        first = self.service.review_research(self.task, args, self.execute('insufficient'))
        stage(2, '新进展已经修正的发现。')
        refresh = ResearchReviewInput(target_type='findings', target_id=first['target_id'],
            content_hash=first['content_hash'], refresh_source=True, reason='依据实际新进展更新快照',
            verification_questions=['该发现是否有据？'])
        updated = self.service.review_research(self.task, refresh, self.execute('supported'))
        self.assertNotEqual(updated['content_hash'], first['content_hash'])
        self.assertEqual(updated['reflection']['round'], 1)
        other = self.service.review_research(self.task, args.model_copy(update={'item_ids': ['f2']}), self.execute('supported'))
        self.assertNotEqual(other['target_id'], first['target_id'])
        self.assertEqual(other['reflection']['round'], 0)
