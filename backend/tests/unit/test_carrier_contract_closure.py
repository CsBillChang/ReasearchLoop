"""载体缓存闭环：全合成内容与内存HTTP，不接触外站或生产文件。"""
from contextlib import ExitStack
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import httpx

from research_loop.domain.evidence import PageEvidence
from research_loop.providers.paper_acquisition import PaperAcquisitionError
from research_loop.providers.pdf import PDFProvider
from research_loop.storage.files import FileStore, sha256_bytes


class CarrierCacheClosureTests(unittest.TestCase):
    """同一份坏快照在所有消费入口准确拒绝，保持不可变内容。"""

    def setUp(self):
        """阻断socket，全部载体与解析结果只存在临时目录。"""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('NETWORK_FORBIDDEN')))
        self.stack.enter_context(patch.object(socket, 'create_connection', side_effect=AssertionError('NETWORK_FORBIDDEN')))
        self.files = FileStore(self.root / 'data', self.root / 'out', self.root)
        self.source = self.root / 'synthetic.pdf'
        self.source.write_bytes(b'%PDF-synthetic carrier bytes')
        self.text = 'Synthetic controlled methods and bounded experimental results.\n' * 40
        self.calls = []
        self.url = 'https://fixture.invalid/synthetic.pdf'
        self.client = httpx.Client(transport=httpx.MockTransport(self.route))
        self.addCleanup(self.client.close)
        self.provider = PDFProvider(self.files, client=self.client)
        self.stack.enter_context(patch.object(self.provider, '_parse', side_effect=self.parsed))

    def route(self, request):
        """Jina拒绝后返回同一合成PDF；不含任何socket请求。"""
        self.calls.append(request)
        return httpx.Response(403) if request.url.host == 'r.jina.ai' else httpx.Response(200, content=self.source.read_bytes())

    def parsed(self, path):
        """固定两页，身份由真实provider按现有规则赋值。"""
        return [PageEvidence(evidence_id='', document_id='', page_number=n, text=self.text) for n in (1, 2)]

    def corrupt(self, kind):
        """保留原[99,2]与short失败形状，其他结构破坏分别测量。"""
        doc = self.provider.ingest(self.url, 'synthetic')
        path = Path(doc.cache_path)
        raw = json.loads(path.read_text())
        if kind == 'page_order':
            raw['pages'][0]['page_number'] = 99
        elif kind == 'short':
            for page in raw['pages']:
                page['text'] = 'short'
        elif kind == 'document':
            raw['pages'][0]['document_id'] = 'foreign-document'
        elif kind == 'evidence':
            raw['pages'][1]['evidence_id'] = raw['pages'][0]['evidence_id']
        elif kind == 'page_type':
            raw['pages'][0]['page_number'] = '1'
        path.write_text(json.dumps(raw))
        return doc, path, path.read_bytes()

    def assert_rejected(self, operation, path, frozen):
        """错误必须有不可重试分类，且不能覆写或删除坏快照。"""
        with self.assertRaises(PaperAcquisitionError) as caught:
            operation()
        self.assertEqual(caught.exception.code, 'PDF_CACHE_INVALID')
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(path.read_bytes(), frozen)

    def test_url_hit_rejects_original_page_order_fixture(self):
        """URL入口不得把坏页序当作缓存miss后绕过。"""
        _, path, frozen = self.corrupt('page_order')
        before = len(self.calls)
        self.assert_rejected(lambda: self.provider.ingest(self.url, 'synthetic'), path, frozen)
        self.assertEqual(len(self.calls), before)

    def test_content_hit_rejects_original_page_order_fixture(self):
        """相同字节重新获取后，内容缓存仍执行完整校验。"""
        _, path, frozen = self.corrupt('page_order')
        self.assert_rejected(lambda: self.provider.ingest_pages(str(self.source), 'synthetic'), path, frozen)

    def test_pages_rejects_original_page_order_fixture(self):
        """按页消费不能靠切片漏掉坏页序。"""
        doc, path, frozen = self.corrupt('page_order')
        self.assert_rejected(lambda: self.provider.pages(doc, 2, 1), path, frozen)

    def test_url_hit_rejects_original_short_fixture(self):
        """短文本不能在URL缓存和同内容回退间绕过。"""
        _, path, frozen = self.corrupt('short')
        self.assert_rejected(lambda: self.provider.ingest(self.url, 'synthetic'), path, frozen)

    def test_content_hit_rejects_original_short_fixture(self):
        """内容缓存的基本文本门与新解析一致。"""
        _, path, frozen = self.corrupt('short')
        self.assert_rejected(lambda: self.provider.ingest_pages(str(self.source), 'synthetic'), path, frozen)

    def test_stale_url_index_cannot_bypass_corrupt_content_after_download(self):
        """原失败链：URL索引miss后重新取得同内容，也不能放行旧坏快照。"""
        for kind in ('page_order', 'short'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                self.provider.file_store = FileStore(Path(temp) / 'data', Path(temp) / 'out', temp)
                _, path, frozen = self.corrupt(kind)
                index = self.provider.file_store.parse_dir / ('source-' + sha256_bytes(self.url.encode()) + '.json')
                raw = json.loads(index.read_text())
                raw['parser_version'] = 'stale-version'
                index.write_text(json.dumps(raw))
                frozen_index = index.read_bytes()
                before = len(self.calls)
                self.assert_rejected(lambda: self.provider.ingest(self.url, 'synthetic'), path, frozen)
                self.assertEqual(len(self.calls), before + 2)
                self.assertEqual(index.read_bytes(), frozen_index)

    def test_pages_rejects_original_short_fixture(self):
        """直接消费短文本快照时也拒绝。"""
        doc, path, frozen = self.corrupt('short')
        self.assert_rejected(lambda: self.provider.pages(doc), path, frozen)

    def test_page_identity_and_types_cannot_be_coerced(self):
        """页身份、重复证据ID及字符串页号不静默修复。"""
        for kind in ('document', 'evidence', 'page_type'):
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as temp:
                    self.provider.file_store = FileStore(Path(temp) / 'data', Path(temp) / 'out', temp)
                    doc, path, frozen = self.corrupt(kind)
                    self.assert_rejected(lambda: self.provider.pages(doc), path, frozen)

    def test_new_parse_invalid_locator_never_commits_snapshot(self):
        """新解析与缓存消费共用页序合同，验证失败不落解析快照。"""
        with patch.object(self.provider, '_parse', return_value=[PageEvidence(evidence_id='', document_id='', page_number=99, text=self.text)]):
            with self.assertRaises(PaperAcquisitionError):
                self.provider.ingest_pages(str(self.source), 'synthetic')
        self.assertEqual(list(self.files.parse_dir.glob('*.json')), [])

    def test_url_index_cannot_relabel_cached_content_hash(self):
        """URL索引不能给已有解析页换一个内容hash，既有文档键规则保持。"""
        doc = self.provider.ingest(self.url, 'synthetic')
        path = Path(doc.cache_path)
        frozen = path.read_bytes()
        index = self.files.parse_dir / ('source-' + sha256_bytes(self.url.encode()) + '.json')
        raw = json.loads(index.read_text())
        raw['content_sha256'] = sha256_bytes(b'different content')
        index.write_text(json.dumps(raw))
        self.assert_rejected(lambda: self.provider.ingest(self.url, 'synthetic'), path, frozen)

    def test_valid_pdf_url_content_and_page_hits_preserve_snapshot(self):
        """合法URL、内容和逐页命中均保持身份及原快照。"""
        doc = self.provider.ingest(self.url, 'synthetic')
        frozen = Path(doc.cache_path).read_bytes()
        with patch.object(self.provider, '_parse', side_effect=AssertionError('cache must avoid parse')):
            self.assertEqual(self.provider.ingest(self.url, 'alias').document_id, doc.document_id)
            self.assertEqual(self.provider.ingest_pages(str(self.source), 'alias').document_id, doc.document_id)
        self.assertEqual([p.page_number for p in self.provider.pages(doc)], [1, 2])
        self.assertEqual(Path(doc.cache_path).read_bytes(), frozen)

    def test_missing_parse_file_keeps_existing_miss_route(self):
        """仅缺失解析文件时沿原路线获取，不将可重建索引当坏内容。"""
        doc = self.provider.ingest(self.url, 'synthetic')
        Path(doc.cache_path).unlink()
        before = len(self.calls)
        self.assertEqual(self.provider.ingest(self.url, 'synthetic').document_id, doc.document_id)
        self.assertEqual(len(self.calls), before + 2)

    def test_stale_url_index_uses_current_contract(self):
        """过期索引miss，旧不可变内容仍保留。"""
        doc = self.provider.ingest(self.url, 'synthetic')
        frozen = Path(doc.cache_path).read_bytes()
        self.provider.parser_version = 'next'
        changed = self.provider.ingest(self.url, 'synthetic')
        self.assertNotEqual(changed.document_id, doc.document_id)
        self.assertEqual(Path(doc.cache_path).read_bytes(), frozen)

    def test_jina_url_content_and_page_validation(self):
        """Jina同样验证文本hash、无物理页及不可变内容命中。"""
        with patch.object(self.provider, '_jina', return_value=(self.text, [])):
            doc = self.provider.ingest(self.url, 'synthetic')
            self.assertIsNone(self.provider.pages(doc)[0].page_number)
            self.assertEqual(self.provider.ingest(self.url, 'alias').document_id, doc.document_id)
            path = Path(doc.cache_path)
            raw = json.loads(path.read_text())
            raw['pages'][0]['text'] = self.text + ' altered'
            path.write_text(json.dumps(raw))
            frozen = path.read_bytes()
            self.assert_rejected(lambda: self.provider.pages(doc), path, frozen)
            self.assert_rejected(lambda: self.provider.ingest(self.url, 'synthetic'), path, frozen)
            index = self.files.parse_dir / ('source-' + sha256_bytes(self.url.encode()) + '.json')
            index.unlink()
            self.assert_rejected(lambda: self.provider.ingest(self.url, 'synthetic'), path, frozen)

    def test_jina_short_or_physical_page_rejected_at_all_consumers(self):
        """Jina坏文本/物理页不能从URL、同内容命中或pages入口消费。"""
        for kind in ('short', 'physical_page'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                self.provider.file_store = FileStore(Path(temp) / 'data', Path(temp) / 'out', temp)
                with patch.object(self.provider, '_jina', return_value=(self.text, [])):
                    doc = self.provider.ingest(self.url, 'synthetic')
                    path = Path(doc.cache_path)
                    raw = json.loads(path.read_text())
                    raw['pages'][0]['text' if kind == 'short' else 'page_number'] = 'short' if kind == 'short' else 99
                    path.write_text(json.dumps(raw))
                    frozen = path.read_bytes()
                    self.assert_rejected(lambda: self.provider.pages(doc), path, frozen)
                    self.assert_rejected(lambda: self.provider.ingest(self.url, 'synthetic'), path, frozen)
                    index = self.provider.file_store.parse_dir / ('source-' + sha256_bytes(self.url.encode()) + '.json')
                    index.unlink()
                    self.assert_rejected(lambda: self.provider.ingest(self.url, 'synthetic'), path, frozen)
