"""补读导航→真实工具执行闭环；临时SQLite、本地合成载体，所有socket禁止。"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from research_loop.domain.evidence import PageEvidence, PaperCandidate
from research_loop.domain.tasks import SavePolicy
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs
from research_loop.tools.registry import ToolRegistry


class CarrierReadbackClosureTests(unittest.TestCase):
    """导航只表明当前入口可尝试，读取时重新核对文件和权限。"""

    def setUp(self):
        """生成两页合成解析结果与SQL授权，不读取本机用户论文。"""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('NETWORK_FORBIDDEN')))
        self.stack.enter_context(patch.object(socket, 'create_connection', side_effect=AssertionError('NETWORK_FORBIDDEN')))
        self.db = Database('sqlite:///' + str(self.root / 'test.db'))
        self.db.ensure_schema()
        self.source = self.root / 'synthetic.pdf'
        self.source.write_bytes(b'%PDF-synthetic local fixture')
        self.attachment = {'input_ref': 'attachment', 'local_path': str(self.source), 'paper_id': 'synthetic'}
        self.db.create_task('owner', None, 'carrier-owner', '读取合成论文，不保存', 'deny', [self.attachment])
        self.db.create_task('foreign', None, 'carrier-foreign', '无材料授权，不保存', 'deny', [])
        self.files = FileStore(self.root / 'data', self.root / 'out', self.root)
        self.client = httpx.Client(transport=httpx.MockTransport(lambda request: self.fail('no remote request allowed')))
        self.addCleanup(self.client.close)
        self.provider = PDFProvider(self.files, client=self.client)
        self.stack.enter_context(patch.object(self.provider, '_parse', return_value=[
            PageEvidence(evidence_id='', document_id='', page_number=n, text='Controlled synthetic methods and results.\n' * 40)
            for n in (1, 2)]))
        self.service = PaperService(self.db, None, self.provider)
        self.registry = ToolRegistry(specs())
        self.first = self.execute()
        self.assertTrue(self.first.ok, self.first.message)
        self.eid = self.first.data['pages'][0]['evidence_id']

    def execute(self, task='owner'):
        """真正经过工具权限、服务与provider边界，不使用补读替身。"""
        attachments = {self.attachment['input_ref']: self.attachment}
        context = ToolContext(task, SavePolicy.deny, attachments=attachments, services={'papers': self.service})
        return self.registry.execute('read_paper', {'target_ref': 'attachment', 'start_page': 1, 'page_count': 1}, context)

    def readback(self):
        """导航生成过程保持已存在的SQL页授权。"""
        return self.service.authorized_source('owner', self.eid)['readback']

    def test_normal_navigation_executes_and_keeps_authorized_page(self):
        """当前本地入口可尝试，实际工具成功并保持相同证据版本。"""
        before = self.readback()
        self.assertTrue(before['available'])
        self.assertEqual(before['target_ref'], 'attachment')
        result = self.execute()
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['pages'][0]['evidence_id'], self.eid)

    def test_missing_file_navigation_unavailable_but_sql_page_retained(self):
        """本地载体失踪不虚报入口，也不删除已有授权原文。"""
        self.source.unlink()
        self.assertFalse(self.readback()['available'])
        self.assertEqual(self.readback()['reason'], 'PDF_LOCAL_MISSING')
        bundle = self.service.evidence_bundle('owner', [self.eid])
        self.assertEqual(bundle[0]['text'], self.first.data['pages'][0]['text'])
        self.assertTrue(self.db.task_has_evidence('owner', self.eid))
        result = self.execute()
        self.assertFalse(result.ok)
        self.assertEqual(result.data['error_code'], 'PDF_LOCAL_MISSING')

    def test_directory_cannot_be_advertised_as_readable_pdf(self):
        """路径存在但为目录时仍不可用。"""
        self.source.unlink()
        self.source.mkdir()
        self.assertFalse(self.readback()['available'])
        self.assertEqual(self.execute().data['error_code'], 'PDF_LOCAL_MISSING')

    def test_open_permission_failure_is_unavailable_and_precise(self):
        """模拟实际open拒绝；不依赖当前用户是否可绕过chmod。"""
        original = Path.open
        def denied(path, *args, **kwargs):
            """仅拒绝附件文件，SQLite和缓存保持正常。"""
            if path == self.source.resolve():
                raise PermissionError('synthetic access denied')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', denied):
            self.assertFalse(self.readback()['available'])
            self.assertEqual(self.readback()['reason'], 'PDF_LOCAL_UNREADABLE')
            self.assertEqual(self.execute().data['error_code'], 'PDF_LOCAL_UNREADABLE')

    def test_missing_local_plus_url_keeps_local_priority(self):
        """同附件带URL仍优先本地，导航和执行均不偷偷联网替换。"""
        self.attachment['url'] = 'https://fixture.invalid/alternate.pdf'
        self.db.execute('UPDATE tasks SET attachments_json=? WHERE task_id=?', (json.dumps([self.attachment]), 'owner'))
        self.source.unlink()
        self.assertFalse(self.readback()['available'])
        self.assertEqual(self.execute().data['error_code'], 'PDF_LOCAL_MISSING')

    def test_advertised_then_deleted_fails_at_execution(self):
        """TOCTOU重新检查；导航成功不保证后续文件仍存在。"""
        self.assertTrue(self.readback()['available'])
        self.source.unlink()
        result = self.execute()
        self.assertFalse(result.ok)
        self.assertEqual(result.data['error_code'], 'PDF_LOCAL_MISSING')

    def test_same_name_candidate_requires_paper_grant_like_read(self):
        """持久题录优先于附件；缺paper grant时导航与read均拒绝。"""
        candidate = PaperCandidate(paper_id='attachment', title='Other catalog identity', source='fixture', pdf_url='https://fixture.invalid/unrelated.pdf')
        self.db.save_paper(candidate.paper_id, candidate.model_dump(mode='json'))
        self.assertFalse(self.db.task_has_paper('owner', 'attachment'))
        self.assertFalse(self.readback()['available'])
        self.assertFalse(self.execute().ok)

    def test_revoked_input_does_not_erase_page_or_invent_access(self):
        """撤销附件入口后SQL页仍可见，当前角色无入口且跨角色不能读页。"""
        self.db.execute('DELETE FROM task_inputs WHERE task_id=? AND input_ref=?', ('owner', 'attachment'))
        self.assertFalse(self.readback()['available'])
        self.assertTrue(self.db.task_has_evidence('owner', self.eid))
        with self.assertRaises(PermissionError):
            self.service.authorized_source('foreign', self.eid)

    def test_current_candidate_source_has_same_priority_as_actual_read(self):
        """当前返回候选优先于持久题录时，不能用题录URL掩盖其坏本地来源。"""
        self.db.execute('DELETE FROM task_inputs WHERE task_id=? AND input_ref=?', ('owner', 'attachment'))
        stored = PaperCandidate(paper_id='synthetic', title='Synthetic', source='fixture', pdf_url='https://fixture.invalid/paper.pdf')
        self.db.save_paper(stored.paper_id, stored.model_dump(mode='json'))
        self.db.grant_paper('owner', stored.paper_id, 'owner')
        self.service.candidates[stored.paper_id] = stored.model_copy(update={'pdf_url': str(self.root / 'missing.pdf')})
        self.assertFalse(self.readback()['available'])
        self.assertEqual(self.readback()['reason'], 'PDF_LOCAL_MISSING')
        context = ToolContext('owner', SavePolicy.deny, attachments={}, services={'papers': self.service})
        result = self.registry.execute('read_paper', {'target_ref': 'synthetic', 'page_count': 1}, context)
        self.assertFalse(result.ok)
        self.assertEqual(result.data['error_code'], 'PDF_LOCAL_MISSING')

    def test_changed_content_version_refuses_source_window(self):
        """载体导航不能绕过既有来源hash窗口检查。"""
        changed = dict(self.first.data)
        changed['content_sha256'] = hashlib.sha256(b'different version').hexdigest()
        with self.assertRaises(ValueError):
            self.service.validate_source_observation('owner', self.first.data['pages'][0], changed)

    def test_acquire_tool_does_not_start_fallback_for_corrupt_snapshot(self):
        """坏不可变缓存准确拒绝，不触发备用定位或重新授予坏页。"""
        stored = PaperCandidate(paper_id='synthetic', title='Synthetic', source='fixture', pdf_url=str(self.source))
        self.db.save_paper(stored.paper_id, stored.model_dump(mode='json'))
        self.db.grant_paper('owner', stored.paper_id, 'owner')
        self.service.web = Mock()
        self.service.web.locate_pdf.side_effect = AssertionError('corrupt snapshot is not a fallback reason')
        path = Path(self.db.fetchone('SELECT cache_path FROM documents WHERE document_id=?', (self.first.data['document_id'],))['cache_path'])
        raw = json.loads(path.read_text())
        raw['pages'][0]['page_number'] = 99
        path.write_text(json.dumps(raw))
        frozen = path.read_bytes()
        before = self.db.list_task_evidence('owner')
        context = ToolContext('owner', SavePolicy.deny, attachments={}, services={'papers': self.service})
        registry = ToolRegistry(specs(web_search_enabled=True, web_fetch_enabled=True))
        result = registry.execute('acquire_paper_pdf', {'target_ref': 'synthetic', 'page_count': 1}, context)
        self.assertFalse(result.ok)
        self.assertFalse(result.retryable)
        self.assertEqual(result.data['error_code'], 'PDF_CACHE_INVALID')
        self.assertEqual(path.read_bytes(), frozen)
        self.assertEqual(self.db.list_task_evidence('owner'), before)
        self.service.web.locate_pdf.assert_not_called()
