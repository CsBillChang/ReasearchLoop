"""核验轮次、原子局部修改及恢复边界；这里的结论是测试替身，不是模型语义成绩。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from research_loop.agents.reflection import ReflectionService
from research_loop.domain.notes import CreateNoteInput, RevisionInput
from research_loop.domain.tasks import TaskCreate
from research_loop.services.notes import NoteService
from research_loop.services.tasks import TaskService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database


class ReflectionTests(unittest.TestCase):
    """所有持久化分支同时支持临时 SQLite 与显式隔离 MySQL。"""

    def setUp(self):
        """创建独立根任务，不删除生产数据或其他测试任务。"""
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        url = os.getenv('AG3_TEST_DATABASE_URL') or 'sqlite:///' + str(root / 'db.sqlite')
        if url.startswith('mysql') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_'):
            raise ValueError('isolated test database required')
        self.db = Database(url); self.db.ensure_schema()
        self.task = TaskService(self.db).create(TaskCreate(prompt='核对后仅局部修改', save_policy='allow')).task_id
        self.notes = NoteService(self.db, FileStore(root / 'data', root / 'notes', root))
        self.draft = self.notes.create(self.task, CreateNoteInput(title='fixture', markdown='保留开头。\n错误句。\n保留结尾。'))
        self.service = ReflectionService(self.db)
        self.calls = []

    def review(self, draft=None, count=1, status='issues', refs=None, questions=None):
        """让独立替身返回已知问题数，记录是否错误重发回调。"""
        draft = draft or self.draft.model_dump()
        def execute(n, evidence, focus):
            self.calls.append(n)
            return {'review_id': str(uuid4()), 'task_id': self.task, 'draft_id': draft['draft_id'],
                    'content_hash': draft['content_hash'], 'status': status,
                    'findings': [{'claim': '错误句。', 'status': 'contradicted', 'reason': 'fixture source', 'evidence_ids': []} for _ in range(count)], 'gaps': []}
        return self.service.review(self.task, draft['draft_id'], draft['content_hash'], '核验具体陈述', refs or [], questions or [], execute)

    def patch(self, review, old='错误句。', new='修正句。', draft=None):
        """绑定精确版本与问题索引构造原子修改。"""
        draft = draft or self.draft.model_dump()
        payload = RevisionInput(draft_id=draft['draft_id'], content_hash=draft['content_hash'], review_id=review['review_id'],
            patches=[{'finding_index': 0, 'old_text': old, 'new_text': new, 'reason': '依据已读取原文修正'}])
        return self.service.revise(self.task, payload), payload

    def test_local_patch_replay_and_independent_new_hash(self):
        """提交后重放不产生第二份版本，非目标文字逐字保留。"""
        review = self.review(); revised, payload = self.patch(review)
        self.assertEqual(revised['markdown'], '保留开头。\n修正句。\n保留结尾。')
        replay = self.service.revise(self.task, payload)
        self.assertEqual(replay['draft_id'], revised['draft_id'])
        checks = self.service.revision_checks(self.task, revised['draft_id'])
        self.assertTrue(checks['non_target_text_preserved'])
        self.assertEqual(checks['exact_patch_count'], 1)
        self.assertEqual(self.db.get_draft(self.draft.draft_id)['markdown'], self.draft.markdown)
        passed = self.review(revised, count=0, status='supported')
        self.assertEqual(passed['reflection']['stop_reason'], 'supported')
        self.assertEqual(self.calls, [0, 1])

    def test_same_inputs_cached_and_no_progress_stops(self):
        """相同输入复用，无语义进展时不消耗第二轮修复。"""
        review = self.review(); self.review()
        self.assertEqual(self.calls, [0])
        revised, _ = self.patch(review)
        second = self.review(revised)
        self.assertEqual(second['reflection']['stop_reason'], 'no_progress')
        with self.assertRaises(ValueError):
            self.patch(second, old='修正句。', new='再改。', draft=revised)

    def test_two_round_limit_cannot_reset_with_new_draft(self):
        """确有进展也只能初审加两次复核；新建全文稿不能重置轮次。"""
        first = self.review(count=3); revised, _ = self.patch(first)
        second = self.review(revised, count=2)
        third_draft, _ = self.patch(second, old='修正句。', new='再次修正。', draft=revised)
        third = self.review(third_draft, count=1)
        self.assertEqual(third['reflection']['stop_reason'], 'round_limit')
        self.assertEqual(self.calls, [0, 1, 2])
        another = self.notes.create(self.task, CreateNoteInput(title='new', markdown='另一个稿')).model_dump()
        self.review(another)
        self.assertEqual(self.calls, [0, 1, 2])

    def test_invalid_patch_does_not_consume_round(self):
        """无效参数、过期 hash 和越权证据不算取证无进展或已修复。"""
        review = self.review()
        for old, new in [('不存在', '修改'), ('错误句。', '错误句。')]:
            with self.assertRaises(ValueError): self.patch(review, old, new)
        payload = RevisionInput(draft_id=self.draft.draft_id, content_hash='0'*64, review_id=review['review_id'], patches=[{
            'finding_index':0, 'old_text':'错误句。', 'new_text':'修正。', 'reason':'test'}])
        with self.assertRaises(ValueError): self.service.revise(self.task, payload)
        self.assertEqual(len(self.service.snapshot(self.task)['attempts']), 1)
        self.assertFalse(self.service.snapshot(self.task)['patches'])

    def test_new_evidence_recheck_without_whole_draft_rewrite(self):
        """补证复核可保持正文不变；新页和具体核验问题必须同时存在。"""
        self.review(); ref = 'fixture-' + uuid4().hex
        self.db.grant_evidence(self.task, ref, self.task)
        with self.assertRaises(ValueError): self.review(refs=[ref])
        result = self.review(refs=[ref], questions=['原文是否支持该句？'], count=0, status='supported')
        self.assertEqual(result['reflection']['round'], 1)

    def test_unknown_or_execution_error_ends_without_approval(self):
        """Reviewer 执行错误独立于语义错误，不批准或继续修复。"""
        result = self.review(count=0, status='execution_failed')
        self.assertEqual(result['reflection']['stop_reason'], 'review_execution_failed')

    def test_cancel_and_lost_lease_reject_writes(self):
        """租约保护不依赖模型行为。"""
        self.service = ReflectionService(self.db, owner=('missing', 'lease'))
        with self.assertRaises(PermissionError): self.review()
        self.service = ReflectionService(self.db)
        self.db.update_task(self.task, cancel_requested=1)
        with self.assertRaises(PermissionError): self.review()

    def test_review_response_recovery_uses_same_attempt(self):
        """模拟收据后进程退出：轮次身份保持，真实请求复用由子图收据层负责。"""
        def killed(n, refs, q):
            self.assertEqual(n, 0)
            raise SystemExit('SIGKILL analogue before session commit')
        with self.assertRaises(SystemExit):
            self.service.review(self.task, self.draft.draft_id, self.draft.content_hash, 'test', [], [], killed)
        self.review()
        self.assertEqual(len(self.service.snapshot(self.task)['attempts']), 1)
