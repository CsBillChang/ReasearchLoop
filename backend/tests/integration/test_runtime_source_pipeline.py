"""取文工具→服务→provider的隔离链；所有HTTP响应与论文均为合成。"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pymupdf

from research_loop.config import Settings
from research_loop.domain.evidence import PaperCandidate
from research_loop.domain.tasks import SavePolicy
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs
from research_loop.tools.registry import ToolRegistry
from scripts.manage import diagnose


class NoBackupRequests:
    """原路径成功或取消时，不应启动备用定位请求。"""

    def locate_pdf(self, *args, **kwargs):
        """记录错误路径为测试失败，而不是发起任何真实请求。"""
        raise AssertionError('no backup Search/Fetch is authorized in this test')


class RuntimeSourcePipeline(unittest.TestCase):
    """临时SQLite为默认；显式专用测试库可验证相同业务链。"""

    def setUp(self):
        """阻断网络，数据库只允许SQLite临时文件或严格测试库名。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        supplied = os.environ.get('FIVE_ROOT_TEST_DATABASE_URL')
        database_url = supplied or 'sqlite:///' + str(self.root / 'test.db')
        if supplied:
            parsed = urlsplit(supplied)
            if (parsed.scheme != 'mysql' or not parsed.path.lstrip('/').startswith('researchloop_test_')
                    or parsed.hostname not in {'localhost', '127.0.0.1'}):
                raise AssertionError('only dedicated researchloop_test_ MySQL schema is allowed')
        # 唯一网络例外是调用者显式批准的本机专用测试MySQL端点。
        original_connect = socket.socket.connect
        allowed = (parsed.hostname, parsed.port or 3306) if supplied else None
        def guarded_connect(sock, address):
            """限制实际socket端点，防止HTTP绕过MockTransport。"""
            if allowed and isinstance(address, tuple) and address[:2] == allowed:
                return original_connect(sock, address)
            raise AssertionError('network prohibited')
        self.stack.enter_context(patch.object(socket.socket, 'connect', guarded_connect))
        self.settings = Settings(project_root=self.root, data_dir=self.root / 'data', output_dir=self.root / 'out',
                                 database_url=database_url, deepseek_model='deepseek-flash')
        self.db = Database(database_url)
        self.db.ensure_schema()
        self.files = FileStore(self.settings.data_dir, self.settings.output_dir, self.root)
        self.task_id = uuid4().hex
        self.db.create_task(self.task_id, None, uuid4().hex, '读取合成论文，不保存', 'deny', [], token_settings=self.settings)
        self.paper_id = 'synthetic-' + uuid4().hex
        candidate = PaperCandidate(paper_id=self.paper_id, title='Synthetic fixture', source='fixture',
                                   paper_url='https://fixture.invalid/' + self.paper_id + '.pdf')
        self.db.save_paper(self.paper_id, candidate.model_dump(mode='json'))
        self.db.grant_paper(self.task_id, self.paper_id, self.task_id)
        self.registry = ToolRegistry(specs(web_search_enabled=True, web_fetch_enabled=True))
        self.text = '# Synthetic paper ' + self.paper_id + '\n' + ('Controlled synthetic scientific methods and results.\n' * 65)
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((40, 40), self.text, fontsize=7)
        self.pdf_bytes = document.tobytes()
        document.close()
        self.calls = []

    def provider(self, handler, **kwargs):
        """装配真实服务与内存HTTP；禁用所有备用Search/Fetch。"""
        def route(request):
            """保存本次实际请求，返回合成响应。"""
            self.calls.append(request)
            return handler(request)
        client = httpx.Client(transport=httpx.MockTransport(route))
        self.addCleanup(client.close)
        provider = PDFProvider(self.files, client=client, **kwargs)
        return provider, PaperService(self.db, None, provider, web=NoBackupRequests())

    def context(self, service, flag=None):
        """使用当前任务权限，取消信号只来自此测试的事件。"""
        return ToolContext(self.task_id, SavePolicy.deny, cancelled=flag.is_set if flag else lambda: False,
                           services={'papers': service})

    def test_acquire_jina_returns_text_identity_not_physical_pdf(self):
        """备用工具可首先获得Jina原文，页面与成功说明必须保留未知物理页。"""
        _, service = self.provider(lambda request: httpx.Response(200, text=self.text))
        result = self.registry.execute('acquire_paper_pdf', {'target_ref': self.paper_id}, self.context(service))
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['read_coverage']['locator_kind'], 'document_text')
        self.assertIn('document text evidence returned', result.message)
        self.assertTrue(all(page['page_number'] is None for page in result.data['pages']))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.db.list_task_evidence(self.task_id)), 1)
        self.assertEqual(list(self.files.pdf_dir.glob('*.pdf')), [])

    def test_acquire_explicit_pages_returns_physical_evidence(self):
        """明确物理页沿用PDF入口，成功信息受实际coverage控制。"""
        _, service = self.provider(lambda request: httpx.Response(200, content=self.pdf_bytes))
        result = self.registry.execute('acquire_paper_pdf', {'target_ref': self.paper_id, 'page_count': 1}, self.context(service))
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['read_coverage']['locator_kind'], 'physical_page')
        self.assertEqual(result.message, 'physical PDF evidence returned')
        self.assertEqual(result.data['pages'][0]['page_number'], 1)
        self.assertEqual(len(self.calls), 1)
        self.assertNotEqual(self.calls[0].url.host, 'r.jina.ai')

    def test_acquire_cancelled_during_backoff_neither_retries_nor_grants(self):
        """取消类型贯通可选获取工具，原失败不变且不授正文权限。"""
        flag = threading.Event()
        _, service = self.provider(lambda request: httpx.Response(503))
        with patch('research_loop.providers.pdf.time.sleep', side_effect=lambda seconds: flag.set()):
            result = self.registry.execute('acquire_paper_pdf', {'target_ref': self.paper_id}, self.context(service, flag))
        self.assertEqual(result.kind, 'cancelled', result.message)
        self.assertFalse(result.retryable)
        self.assertEqual(result.data['error_code'], 'PAPER_READ_CANCELLED')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.db.list_task_evidence(self.task_id), [])
        self.assertEqual(list(self.files.parse_dir.glob('*.json')), [])
        self.assertEqual(list(self.files.pdf_dir.glob('*.pdf')), [])

    def test_url_cache_new_pdf_contract_creates_separate_document(self):
        """服务读回也不能复用不兼容URL缓存，旧版本原文和授权继续保留。"""
        provider, service = self.provider(lambda request: httpx.Response(403) if request.url.host == 'r.jina.ai'
                                         else httpx.Response(200, content=self.pdf_bytes))
        first = service.read(self.task_id, self.paper_id)
        old = self.db.fetchone('SELECT * FROM documents WHERE document_id=?', (first['document_id'],))
        old_cache = Path(old['cache_path']).read_bytes()
        provider.parser_version = 'synthetic-next'
        second = service.read(self.task_id, self.paper_id)
        self.assertNotEqual(first['document_id'], second['document_id'])
        self.assertEqual(self.db.fetchone('SELECT * FROM documents WHERE document_id=?', (first['document_id'],)), old)
        self.assertEqual(Path(old['cache_path']).read_bytes(), old_cache)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(len(self.db.list_task_evidence(self.task_id)), 2)

    def test_diagnosis_reports_local_missing_component_without_model_probe(self):
        """诊断分开数据库与本机RAG就绪，缺权重不伪装全局服务失败。"""
        with patch('scripts.manage.ROOT', self.root), patch('scripts.manage.probe', return_value={'ok': False}), \
                patch('importlib.import_module', side_effect=ImportError('synthetic missing dependency')):
            report = diagnose(self.settings)
        self.assertTrue(report['database_ready'])
        self.assertEqual(report['model_requests'], 0)
        self.assertEqual(report['local_components']['embedding']['status'], 'unavailable')
        self.assertEqual(report['runtime_fingerprint']['effective_settings']['context_output_reserve'], 65536)
        self.assertFalse(report['model_key_present'])
