"""V1.5五项材料身份合同的合成正反例；不改写原五项失败测试。"""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pymupdf

from research_loop.domain.research_progress import ResearchProgressInput
from research_loop.domain.tasks import ActionRecord
from research_loop.providers.paper_acquisition import PaperAcquisitionError
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.services.research_progress import ResearchProgressService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.agents.reflection import ReflectionService
from integration import test_alias_reflection_versions as historical


class HistoricalAliasIdentityTests(unittest.TestCase):
    """原历史兼容入口单列；新绑定成功不能替代这里的目标。"""

    def setUp(self):
        """复用原合成前置，保留原角色只有旧paper授权的分支。"""
        self.fixture = historical.AliasReflectionVersionTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def test_historical_alias_pending_progress_keeps_original_source(self):
        """同版别名应允许旧待提交成果，且不扩大child的新身份授权。"""
        fixture = self.fixture
        action = fixture.prepared_action()
        expected = deepcopy(action['observation'])
        fixture.rebind()
        self.assertFalse(fixture.db.task_has_paper(fixture.child, fixture.new_id))
        fixture.progress.commit_action(action, None)
        self.assertEqual(fixture.db.list_actions(fixture.child)[0]['status'], 'succeeded')
        self.assertEqual(fixture.progress.snapshot(fixture.child)['findings'],
                         expected['data']['research_progress']['findings'])
        self.assertFalse(fixture.db.task_has_paper(fixture.child, fixture.new_id))

    def test_historical_alias_retained_progress_remains_intact(self):
        """已成功旧成果不得只因同版显示别名变stale或改写来源。"""
        fixture = self.fixture
        fixture.progress.commit_action(fixture.prepared_action(), None)
        before = fixture.db.list_actions(fixture.child)
        fixture.rebind()
        retained = fixture.progress.retained(fixture.child)
        self.assertEqual(retained['findings'][0]['source_integrity'], 'intact')
        self.assertEqual(fixture.db.list_actions(fixture.child), before)
        self.assertFalse(fixture.db.task_has_paper(fixture.child, fixture.new_id))

    def test_historical_alias_reflection_keeps_original_session(self):
        """独立核验应继续认可明确同版旧来源，历史session逐值不变。"""
        fixture = self.fixture
        fixture.rebind()
        target = ReflectionService(fixture.db).research_target(fixture.root, fixture.target_id, 'd' * 64)
        self.assertEqual(target['current']['items'][0]['sources'][0], fixture.stored)
        self.assertEqual(fixture.db.fetchone('SELECT state_json FROM reflection_sessions WHERE task_id=?',
                                            (fixture.root,))['state_json'], fixture.state_json)
        self.assertFalse(fixture.db.task_has_paper(fixture.child, fixture.new_id))

    def test_real_content_change_safely_preserves_uncommitted_action(self):
        """安全断言单列：真实换版提前拒绝时仍不得提交或改写旧成果。"""
        fixture = self.fixture
        action = fixture.prepared_action()
        before = fixture.db.list_actions(fixture.child)
        old_page = fixture.db.get_page(fixture.eid)
        old_session = fixture.db.fetchone('SELECT state_json FROM reflection_sessions WHERE task_id=?',
                                         (fixture.root,))
        fixture.db.save_document({**fixture.document, 'paper_id': fixture.new_id, 'content_sha256': 'e' * 64})
        # 这是保护性旁证；原方法仍严格要求ValueError及原错误文本，不在此替换它。
        with self.assertRaisesRegex(PermissionError, 'historical material outside current role references'):
            fixture.progress.commit_action(action, None)
        self.assertEqual(fixture.db.list_actions(fixture.child), before)
        self.assertEqual(fixture.progress.snapshot(fixture.child)['revision'], 0)
        self.assertEqual(fixture.db.get_page(fixture.eid), old_page)
        self.assertEqual(fixture.db.fetchone('SELECT state_json FROM reflection_sessions WHERE task_id=?',
                                            (fixture.root,)), old_session)
        self.assertFalse(fixture.db.task_has_paper(fixture.child, fixture.new_id))


class MaterialIdentityContractsTests(unittest.TestCase):
    """真实PaperService/PDFProvider与合成PDF验证正式授权及新引用路径。"""

    def setUp(self):
        """每例独立材料和任务；只允许显式本机专用MySQL或临时SQLite。"""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix='researchloop-identity5-')))
        supplied = os.getenv('V15_IDENTITY_TEST_DATABASE_URL')
        url = supplied or 'sqlite:///' + str(self.directory / 'identity.db')
        parsed = urlsplit(url)
        allowed = None
        if supplied:
            if (parsed.scheme != 'mysql' or not parsed.path.lstrip('/').startswith('researchloop_test_')
                    or parsed.hostname not in {'localhost', '127.0.0.1'}):
                raise AssertionError('only explicit local researchloop_test_ MySQL schema is allowed')
            allowed = (parsed.hostname, parsed.port or 3306)
        original = socket.socket.connect

        def guarded(sock, address):
            """只放行显式测试数据库端点，其余真实网络一律拒绝。"""
            if allowed and isinstance(address, tuple) and address[:2] == allowed:
                return original(sock, address)
            raise AssertionError('IDENTITY5_NETWORK_FORBIDDEN')

        self.stack.enter_context(patch.object(socket.socket, 'connect', guarded))
        self.db = Database(url)
        self.db.ensure_schema()
        suffix = uuid4().hex
        self.root, self.foreign = 'identity-root-' + suffix, 'identity-foreign-' + suffix
        self.old_id, self.alias_id = 'user-url:' + suffix, '2603.00595v1-' + suffix
        self.path = self.directory / 'A.pdf'
        self.write_pdf(self.path, 'Synthetic original material')
        self.attachment = {'input_ref': 'input-A', 'filename': 'A.pdf', 'paper_id': self.old_id,
                           'local_path': str(self.path)}
        self.new_path = self.directory / 'new-version.pdf'
        self.new_path.write_bytes(self.path.read_bytes())
        self.new_attachment = {**self.attachment, 'input_ref': 'input-new', 'filename': 'New.pdf',
                               'local_path': str(self.new_path)}
        self.attachments = {'input-A': self.attachment}
        self.db.create_task(self.root, None, uuid4().hex, '读取附件A，仅限第1页，不保存。',
                            'deny', [self.attachment, self.new_attachment])
        # 附件元信息经正式创建登记；新ref直到对应分支显式grant前不可读取。
        self.db.execute('DELETE FROM task_inputs WHERE task_id=? AND input_ref=?', (self.root, 'input-new'))
        self.db.create_task(self.foreign, None, uuid4().hex, '无材料授权，不保存。', 'deny', [])
        child = self.db.create_subtask(self.root, self.root, 1, '仅处理附件A第一页',
                                       {'input_refs': ['input-A'], 'mode': 'investigate'}, uuid4().hex)
        self.child = child['subtask_id']
        self.db.grant_input(self.child, 'input-A', self.root)
        for paper_id in (self.old_id, self.alias_id):
            self.db.save_paper(paper_id, {'paper_id': paper_id, 'title': 'Synthetic exact version',
                'source': 'arxiv', 'paper_url': 'https://arxiv.org/pdf/2603.00595v1',
                'pdf_url': str(self.path), 'version': 'v1'})
        client = httpx.Client(transport=httpx.MockTransport(lambda request: self.fail('no HTTP authorized')))
        self.addCleanup(client.close)
        self.provider = PDFProvider(FileStore(self.directory / 'synthetic-cache', self.directory / 'out', self.directory), client=client)
        self.service = PaperService(self.db, None, self.provider)
        self.progress = ResearchProgressService(self.db)
        self.first = self.service.read(self.root, 'input-A', 1, 1, self.attachments)
        self.service.read(self.child, 'input-A', 1, 1, self.attachments)
        self.eid = self.first['pages'][0]['evidence_id']
        self.old_source = self.progress.source(self.child, self.eid)
        self.frozen = self.freeze_old()

    @staticmethod
    def write_pdf(path, label):
        """生成两页可实际解析的合成原文，无用户数据。"""
        document = pymupdf.open()
        for number in (1, 2):
            page = document.new_page()
            page.insert_text((40, 40), (f'{label}, page {number}.\n' +
                              'Controlled synthetic methods and results.\n' * 30), fontsize=8)
        document.save(path)
        document.close()

    def bindings(self):
        """读回根任务权威映射，保持其他JSON字段完整。"""
        return json.loads(self.db.get_task(self.root)['brief_json'])['material_read_bindings']

    def freeze_old(self):
        """冻结旧document、缓存、页和绑定，用于验证新版本不回写。"""
        document = self.db.fetchone('SELECT * FROM documents WHERE document_id=?', (self.first['document_id'],))
        return {'document': document, 'cache': Path(document['cache_path']).read_bytes(),
                'page': self.db.get_page(self.eid), 'binding': deepcopy(self.bindings()['input-A'])}

    def assert_old_unchanged(self):
        """新授权和版本处理不能改写旧版本持久证据。"""
        current = self.db.fetchone('SELECT * FROM documents WHERE document_id=?', (self.first['document_id'],))
        # 重复保存会推进存储时间；版本、解析配置、缓存路径及其他列须逐值相等。
        self.assertEqual({key: value for key, value in current.items() if key != 'created_at'},
                         {key: value for key, value in self.frozen['document'].items() if key != 'created_at'})
        self.assertEqual(Path(self.frozen['document']['cache_path']).read_bytes(), self.frozen['cache'])
        self.assertEqual(self.db.get_page(self.eid), self.frozen['page'])
        self.assertEqual(self.bindings()['input-A'], self.frozen['binding'])

    def pending(self, task, eid, text='Synthetic bounded finding.'):
        """按正式prepare与行动日志建立待提交成果，不伪造版本hash。"""
        args = ResearchProgressInput(findings=[{'text': text, 'evidence_ids': [eid]}])
        value = self.progress.prepare(task, args)
        action = ActionRecord(task_id=task, sequence=len(self.db.list_actions(task)) + 1,
            tool_name='record_research_progress', arguments=args.model_dump(), status='started',
            observation={'ok': True, 'data': {'research_progress': value}}).model_dump(mode='json')
        self.db.save_action(action)
        return action

    def new_input(self, parser=False):
        """新ref经正式grant/read形成绑定；原ref和文件保留。"""
        if parser:
            self.provider.parser_version = 'identity5-next-parser'
        else:
            self.new_path.unlink()
            self.write_pdf(self.new_path, 'Synthetic changed content version')
        self.db.grant_input(self.root, 'input-new', self.root)
        self.attachments['input-new'] = self.new_attachment
        return self.service.read(self.root, 'input-new', 1, 1, self.attachments)

    def test_bound_alias_keeps_child_progress_without_alias_grant(self):
        """已正式读取的旧绑定支持同版显示别名，不增加child的新身份权限。"""
        action = self.pending(self.child, self.eid)
        old = deepcopy(action['observation']['data']['research_progress']['findings'])
        row = self.frozen['document']
        self.db.save_document({**row, 'paper_id': self.alias_id})
        self.assertFalse(self.db.task_has_paper(self.child, self.alias_id))
        self.progress.commit_action(action, None)
        self.assertEqual(self.progress.snapshot(self.child)['findings'], old)
        self.assertEqual(self.progress.retained(self.child)['findings'][0]['source_integrity'], 'intact')
        self.assertEqual(self.bindings()['input-A'], self.frozen['binding'])
        self.assertFalse(self.db.task_has_paper(self.child, self.alias_id))
        with self.assertRaises(PermissionError):
            self.service.authorized_source(self.foreign, self.eid)
        with self.assertRaises(PermissionError):
            self.service.read(self.child, self.alias_id, 1, 1, {})

    def test_parser_change_rejects_old_ref_without_rebinding(self):
        """旧ref不接受重解析的新document，拒绝后原绑定和来源保持。"""
        self.provider.parser_version = 'identity5-next-parser'
        with self.assertRaisesRegex(ValueError, 'material document/version changed; rebind with a new input_ref'):
            self.service.read(self.root, 'input-A', 1, 1, self.attachments)
        self.assert_old_unchanged()
        self.assertEqual(self.progress.source(self.child, self.eid), self.old_source)

    def test_parser_change_new_ref_forms_separate_usable_version(self):
        """新ref经grant和真实重解析可用，旧缓存/页/成果来源不变。"""
        action = self.pending(self.child, self.eid)
        self.progress.commit_action(action, None)
        before = self.progress.snapshot(self.child)
        second = self.new_input(parser=True)
        self.assertNotEqual(second['document_id'], self.first['document_id'])
        self.assertEqual(second['content_sha256'], self.first['content_sha256'])
        self.assertEqual(self.bindings()['input-new']['document_id'], second['document_id'])
        readback = self.service.authorized_source(self.root, second['pages'][0]['evidence_id'])['readback']
        self.assertTrue(readback['available'], readback)
        self.assertEqual(self.progress.snapshot(self.child), before)
        self.assert_old_unchanged()

    def test_content_change_rejects_old_ref_without_rebinding(self):
        """相同旧ref换成不同真实字节时拒绝，不能用同paper身份覆盖。"""
        path = self.directory / 'replacement.pdf'
        self.write_pdf(path, 'Synthetic changed bytes')
        changed = {'input-A': {**self.attachment, 'local_path': str(path)}}
        with self.assertRaisesRegex(ValueError, 'material document/version changed; rebind with a new input_ref'):
            self.service.read(self.root, 'input-A', 1, 1, changed)
        self.assert_old_unchanged()
        self.assertEqual(self.progress.source(self.child, self.eid), self.old_source)

    def test_content_change_new_ref_requires_fresh_read_and_progress(self):
        """新内容先读取再显式提交新依据；旧发现不自动获得新来源。"""
        old_action = self.pending(self.root, self.eid)
        self.progress.commit_action(old_action, None)
        old_findings = self.progress.snapshot(self.root)['findings']
        second = self.new_input()
        self.assertNotEqual(second['document_id'], self.first['document_id'])
        self.assertNotEqual(second['content_sha256'], self.first['content_sha256'])
        self.assertEqual(self.progress.snapshot(self.root)['findings'], old_findings)
        new_eid = second['pages'][0]['evidence_id']
        action = self.pending(self.root, new_eid, 'Synthetic new-version finding.')
        self.progress.commit_action(action, None)
        findings = self.progress.snapshot(self.root)['findings']
        self.assertEqual(findings[0], old_findings[0])
        self.assertEqual(findings[1]['sources'][0]['source_sha256'], second['content_sha256'])
        self.assertEqual(findings[1]['sources'][0]['document_id'], second['document_id'])
        self.assert_old_unchanged()

    def test_repeated_same_page_does_not_create_progress_revision(self):
        """同版同页重复读与同发现prepare不制造新的研究进展版本。"""
        action = self.pending(self.child, self.eid)
        self.progress.commit_action(action, None)
        before = self.progress.snapshot(self.child)
        repeated = self.service.read(self.child, 'input-A', 1, 1, self.attachments)
        self.assertEqual(repeated['pages'][0]['evidence_id'], self.eid)
        prepared = self.progress.prepare(self.child, ResearchProgressInput(findings=[{
            'text': 'Synthetic bounded finding.', 'evidence_ids': [self.eid]}]))
        self.assertEqual(prepared['revision'], before['revision'])
        self.assertEqual(prepared['progress_change']['kind'], 'unchanged')
        self.assertEqual(self.progress.snapshot(self.child), before)
        self.assert_old_unchanged()

    def revoke(self):
        """仅撤销根附件入口，历史页和旧绑定原样保留。"""
        self.db.execute('DELETE FROM task_inputs WHERE task_id=? AND input_ref=?', (self.root, 'input-A'))

    def test_revoked_attachment_keeps_page_but_no_navigation_or_read(self):
        """保留旧页/同hash不恢复被撤销的附件入口。"""
        self.revoke()
        source = self.service.authorized_source(self.root, self.eid)
        self.assertFalse(source['readback']['available'])
        self.assertTrue(self.db.task_has_evidence(self.root, self.eid))
        with self.assertRaises(PaperAcquisitionError):
            self.service.read(self.root, 'input-A', 1, 1, self.attachments)
        self.assert_old_unchanged()

    def test_paper_grant_alone_cannot_restore_revoked_attachment_mapping(self):
        """仅给文档paper grant仍不能沿旧attachment绑定恢复导航。"""
        self.revoke()
        self.db.grant_paper(self.root, self.old_id, self.root)
        with self.assertRaisesRegex(PermissionError, 'source material outside current role or actual read mapping'):
            self.service.authorized_source(self.root, self.eid)
        self.assertNotIn(self.old_id, self.bindings())
        self.assert_old_unchanged()

    def test_fresh_authorized_paper_read_restores_explicit_navigation(self):
        """撤附件后正式读取已grant题录，建立新映射后才可导航使用。"""
        self.revoke()
        self.db.grant_paper(self.root, self.old_id, self.root)
        second = self.service.read(self.root, self.old_id, 1, 1, {})
        self.assertEqual(second['pages'][0]['evidence_id'], self.eid)
        readback = self.service.authorized_source(self.root, self.eid)['readback']
        self.assertTrue(readback['available'])
        self.assertEqual(readback['target_ref'], self.old_id)
        self.assertEqual(self.bindings()[self.old_id]['document_id'], self.first['document_id'])
        replay = self.service.read(self.root, readback['target_ref'], readback['start_page'], readback['page_count'], {})
        self.assertEqual(replay['pages'][0]['evidence_id'], self.eid)
        self.assertEqual(self.progress.source(self.root, self.eid), self.old_source)
        self.assert_old_unchanged()

    def test_new_binding_does_not_grant_foreign_or_child_or_out_of_range(self):
        """新引用不扩大其他角色或旧附件的页范围。"""
        second = self.new_input()
        eid = second['pages'][0]['evidence_id']
        with self.assertRaises(PermissionError):
            self.service.authorized_source(self.foreign, eid)
        with self.assertRaises(PermissionError):
            self.service.authorized_source(self.child, eid)
        with self.assertRaises(PaperAcquisitionError):
            self.service.read(self.child, 'input-new', 1, 1, self.attachments)
        with self.assertRaisesRegex(PermissionError, 'user-specified physical page range'):
            self.service.read(self.child, 'input-A', 2, 1, self.attachments)
        self.assertFalse(self.db.task_has_input(self.child, 'input-new'))
        self.assertFalse(self.db.task_has_paper(self.child, self.alias_id))
        self.assert_old_unchanged()
