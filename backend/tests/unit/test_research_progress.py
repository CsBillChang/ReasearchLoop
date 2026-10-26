"""阶段进展的版本、权限与中断恢复；不调用生成模型。"""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from research_loop.domain.tasks import ActionRecord, TaskCreate
from research_loop.domain.research_progress import ResearchProgressInput, ProgressFinding
from research_loop.services.research_progress import ResearchProgressService
from research_loop.services.tasks import TaskService
from research_loop.services.papers import PaperService
from research_loop.storage.mysql import Database
from integration.test_collaboration_contract import FixturePDF


class ResearchProgressTests(unittest.TestCase):
    """只使用独立SQLite或显式隔离MySQL，测试日志本身是持久真值。"""

    def setUp(self):
        """构造授权论文与真实服务读取的合成页。"""
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        url = os.getenv('CORE_LOOP_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root / 'isolated.db')
        if not url.startswith('sqlite:') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_core_loop_'):
            raise ValueError('only isolated core-loop database is allowed')
        self.db = Database(url); self.db.ensure_schema()
        self.tid = TaskService(self.db).create(TaskCreate(prompt='读取并比较，不保存', idempotency_key=uuid4().hex)).task_id
        self.db.grant_paper(self.tid, 'fixture-' + uuid4().hex, self.tid)
        paper = self.db.list_task_papers(self.tid)[0]
        self.db.save_paper(paper, {'paper_id': paper, 'source': 'arxiv', 'title': 'synthetic', 'pdf_url': 'https://example.invalid/test.pdf'})
        data = PaperService(self.db, None, FixturePDF()).read(self.tid, paper)
        self.eid = data['pages'][0]['evidence_id']
        self.service = ResearchProgressService(self.db)

    def tearDown(self):
        """清理测试临时目录，不删除任何生产表。"""
        self.temp.cleanup()

    def commit(self, text='语言特征参与训练', **kwargs):
        """模拟图的intent→校验→事务提交；中间不落Checkpoint。"""
        payload = ResearchProgressInput(findings=[ProgressFinding(text=text, evidence_ids=[self.eid])], **kwargs)
        value = self.service.prepare(self.tid, payload)
        seq = max([a['sequence_no'] for a in self.db.list_actions(self.tid)] or [0]) + 1
        action = ActionRecord(task_id=self.tid, sequence=seq, tool_name='record_research_progress', arguments=payload.model_dump())
        self.db.save_action(action.model_dump(mode='json'))
        action.status = 'succeeded'; action.observation = {'ok': True, 'kind': 'success', 'data': {'research_progress': value}}
        self.service.commit_action(action.model_dump(mode='json'), None)
        return action.model_dump(mode='json')

    def test_action_survives_missing_checkpoint_and_replay_is_idempotent(self):
        """行动成功后快照尚未写入，恢复只取得一份候选发现。"""
        action = self.commit(gaps=['推理阶段尚未核实'])
        self.assertIsNone(self.db.get_checkpoint(self.tid))
        self.service.commit_action(action, None)
        state = ResearchProgressService(self.db).snapshot(self.tid)
        self.assertEqual(state['revision'], 1)
        self.assertEqual(len(state['findings']), 1)
        self.assertEqual(state['findings'][0]['verification_status'], 'unreviewed')
        self.assertEqual(state['gaps'], ['推理阶段尚未核实'])

    def test_stale_revision_and_foreign_evidence_are_rejected(self):
        """旧进展不能覆盖新版本，知道其他页ID也不构成授权。"""
        self.commit()
        with self.assertRaisesRegex(ValueError, 'revision stale'):
            self.service.prepare(self.tid, ResearchProgressInput(base_revision=0))
        other = TaskService(self.db).create(TaskCreate(prompt='other', idempotency_key=uuid4().hex)).task_id
        with self.assertRaises(PermissionError):
            self.service.prepare(other, ResearchProgressInput(findings=[ProgressFinding(text='x', evidence_ids=[self.eid])]))

    def test_same_finding_deduplicates_and_gaps_require_explicit_clear(self):
        """重复记录不增长版本，省略gaps不会清空未解决问题。"""
        self.commit(gaps=['未读实验'])
        self.commit()
        state = self.service.snapshot(self.tid)
        self.assertEqual(state['revision'], 1)
        self.assertEqual(state['gaps'], ['未读实验'])
        self.commit(gaps=[])
        self.assertEqual(self.service.snapshot(self.tid)['gaps'], [])

    def test_existing_finding_id_updates_without_losing_sources(self):
        """复制本任务真实ID更新发现；省略证据不会删除已取得的来源。"""
        self.commit()
        original = self.service.snapshot(self.tid)['findings'][0]
        candidate = self.service.prepare(self.tid, ResearchProgressInput(
            findings=[ProgressFinding(finding_id=original['finding_id'], text='补充语言特征的适用条件')]))
        self.assertEqual(len(candidate['findings']), 1)
        updated = candidate['findings'][0]
        self.assertEqual(updated['finding_id'], original['finding_id'])
        self.assertEqual(updated['text'], '补充语言特征的适用条件')
        self.assertEqual(updated['sources'], original['sources'])
        self.assertEqual(updated['revision'], original['revision'] + 1)
        self.assertEqual(self.service.snapshot(self.tid)['findings'][0], original)

    def test_invented_and_foreign_finding_ids_are_rejected(self):
        """自拟编号和其他任务真实ID均不得被悄悄当作新增发现。"""
        self.commit()
        original = self.service.snapshot(self.tid)['findings'][0]
        for invented in ('f1', 'F1', 'invented-topic'):
            with self.subTest(finding_id=invented):
                with self.assertRaisesRegex(ValueError, 'unknown finding_id'):
                    self.service.prepare(self.tid, ResearchProgressInput(
                        findings=[ProgressFinding(finding_id=invented, text='不能据此新增')]))
        other = TaskService(self.db).create(TaskCreate(
            prompt='独立任务，不保存', idempotency_key=uuid4().hex)).task_id
        with self.assertRaisesRegex(ValueError, 'unknown finding_id'):
            self.service.prepare(other, ResearchProgressInput(findings=[ProgressFinding(
                finding_id=original['finding_id'], text='不能复用其他任务编号')]))
        self.assertEqual(self.service.snapshot(self.tid)['findings'], [original])
        self.assertEqual(self.service.snapshot(other)['findings'], [])

    def test_committed_action_cannot_be_replaced(self):
        """同一成功行动的修改请求不能静默覆盖历史。"""
        action = self.commit()
        action['observation']['data']['research_progress']['gaps'] = ['tampered']
        with self.assertRaisesRegex(ValueError, 'immutable'):
            self.service.commit_action(action, None)

    def test_source_change_is_visible_without_erasing_original(self):
        """原文变更后保留原发现，标记来源过期而非继续当作已核验依据。"""
        self.commit()
        self.db.execute('UPDATE page_evidence SET text=? WHERE evidence_id=?', ('changed', self.eid))
        state = self.service.retained(self.tid)
        self.assertEqual(state['findings'][0]['text'], '语言特征参与训练')
        self.assertEqual(state['findings'][0]['source_integrity'], 'stale')

    def test_cancel_and_lost_owner_block_commit(self):
        """租约失权和取消均不得写入新的阶段成果。"""
        action = self.commit()
        with self.assertRaises(PermissionError):
            self.service.commit_action(action, (self.tid, 'wrong-worker', 'wrong-token'))
        TaskService(self.db).cancel(self.tid)
        with self.assertRaises(PermissionError):
            self.service.commit_action(action, None)

    def test_concurrent_prepared_revision_accepts_only_one(self):
        """同一前版的两个候选不能后写覆盖先写，失败候选不成为进展。"""
        values = [self.service.prepare(self.tid, ResearchProgressInput(findings=[ProgressFinding(text=t, evidence_ids=[self.eid])])) for t in ('candidate A','candidate B')]
        actions = []
        for seq,value in enumerate(values,1):
            a=ActionRecord(task_id=self.tid,sequence=seq,tool_name='record_research_progress',arguments={'candidate':seq})
            self.db.save_action(a.model_dump(mode='json'))
            a.status='succeeded';a.observation={'ok':True,'kind':'success','data':{'research_progress':value}}
            actions.append(a.model_dump(mode='json'))
        self.service.commit_action(actions[0],None)
        with self.assertRaisesRegex(ValueError,'changed before commit'):
            self.service.commit_action(actions[1],None)
        self.assertEqual(self.service.snapshot(self.tid)['findings'][0]['text'],'candidate A')

    def test_local_pdf_read_cache_checks_bytes_and_current_authorization(self):
        """同版本本地文件可复用；改文件或撤授权后必须重新走读取服务。"""
        import pymupdf
        from research_loop.providers.pdf import PDFProvider
        from research_loop.storage.files import FileStore

        with pymupdf.open() as document:
            document.new_page().insert_text(
                (40, 40), 'Synthetic local cache evidence: language features.\n' * 40)
            content = document.tobytes()
        pdf = self.root / 'local.pdf'
        pdf.write_bytes(content)
        self.db.grant_input(self.tid, 'input', self.tid)
        attachments = {'input': {'local_path': str(pdf)}}
        provider = PDFProvider(FileStore(self.root / 'data', self.root / 'notes', self.root))
        self.addCleanup(provider.close)
        # 正式读取同时建立内容版本和材料授权映射，避免拼接互不相符的旧页。
        data = PaperService(self.db, None, provider).read(
            self.tid, 'input', attachments=attachments)
        args = {'target_ref': 'input'}
        action = ActionRecord(
            task_id=self.tid, sequence=1, tool_name='read_paper', arguments=args,
            status='succeeded', observation={'ok': True, 'kind': 'success', 'data': data})
        self.db.save_action(action.model_dump(mode='json'))
        self.assertIsNotNone(self.service.cached_read(
            self.tid, args, attachments, provider=provider))
        pdf.write_bytes(b'%PDF-updated')
        self.assertIsNone(self.service.cached_read(
            self.tid, args, attachments, provider=provider))
        pdf.write_bytes(content)
        self.db.execute('DELETE FROM task_evidence WHERE task_id=?', (self.tid,))
        self.assertIsNone(self.service.cached_read(
            self.tid, args, attachments, provider=provider))

    def test_unknown_page_number_is_not_reported_as_physical_zero(self):
        """Jina类来源没有物理页码时保留未知，不生成第0页定位。"""
        self.db.execute('UPDATE page_evidence SET page_number=0 WHERE evidence_id=?',(self.eid,))
        self.db.execute('UPDATE documents SET total_pages=0 WHERE document_id=(SELECT document_id FROM page_evidence WHERE evidence_id=?)', (self.eid,))
        self.assertIsNone(self.service.source(self.tid,self.eid)['page_number'])
