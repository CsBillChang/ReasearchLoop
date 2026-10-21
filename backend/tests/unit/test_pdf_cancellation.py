"""正文获取取消的合成流回归；禁止外网，不改变超时或重试预算。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
import pymupdf

from research_loop.providers.paper_acquisition import PaperAcquisitionError
from research_loop.providers.pdf import PDFProvider, PaperReadCancelled
from research_loop.storage.files import FileStore


class CancelStream(httpx.SyncByteStream):
    """在第一块之后触发取消，记录后续内容是否仍被读取。"""

    def __init__(self, flag):
        """保存本次合成流的取消事件。"""
        self.flag, self.closed, self.after_cancel = flag, False, False

    def __iter__(self):
        """第二块返回时取消已就绪，不伪造阻塞线程已停止。"""
        yield b"%PDF-first-block"
        self.flag.set()
        yield b"cancel-observed-on-this-block"
        self.after_cancel = True
        yield b"must-not-read"

    def close(self):
        """记录真实response上下文的退出。"""
        self.closed = True


class PDFCancellationTests(unittest.TestCase):
    """验证请求、重试、回退与缓存提交边界。"""

    def setUp(self):
        """所有文件均在临时目录，所有HTTP均用MockTransport。"""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.files = FileStore(self.root / "data", self.root / "notes", self.root)
        for owner, name in ((socket.socket, "connect"), (socket, "create_connection")):
            blocker = patch.object(owner, name, side_effect=AssertionError("network prohibited"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def provider(self, handler, **kwargs):
        """创建不持有真实socket的provider。"""
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return PDFProvider(self.files, client=client, **kwargs)

    def test_cancel_jina_stream_has_no_retry_fallback_or_cache(self):
        """取消从文本流向上传播，不能被ValueError/OSError回退吞掉。"""
        flag, calls = threading.Event(), []
        stream = CancelStream(flag)
        provider = self.provider(lambda request: calls.append(request) or httpx.Response(200, stream=stream))
        with self.assertRaises(PaperReadCancelled), provider.cancellation_scope(flag.is_set):
            provider.ingest("https://fixture.invalid/p.pdf", "paper")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].url.host, "r.jina.ai")
        self.assertTrue(stream.closed)
        self.assertFalse(stream.after_cancel)
        self.assertEqual(list(self.root.rglob("*.json")), [])
        self.assertEqual(list(self.root.rglob("*.pdf")), [])

    def test_cancel_pdf_stream_does_not_parse_or_commit(self):
        """物理页下载也检查每块，未完成流不能成为PDF缓存。"""
        flag, calls = threading.Event(), []
        stream = CancelStream(flag)
        provider = self.provider(lambda request: calls.append(request) or httpx.Response(200, stream=stream))
        with patch.object(provider, "_parse", side_effect=AssertionError("cancelled parse")), \
                self.assertRaises(PaperReadCancelled), provider.cancellation_scope(flag.is_set):
            provider.ingest_pages("https://fixture.invalid/p.pdf", "paper")
        self.assertEqual(len(calls), 1)
        self.assertTrue(stream.closed)
        self.assertFalse(stream.after_cancel)
        self.assertEqual(list(self.root.rglob("*.pdf")), [])

    def test_cancel_at_response_header_prevents_retry(self):
        """503响应到来时已取消，不能新增第二次Jina请求。"""
        flag, calls = threading.Event(), []

        def handle(request):
            calls.append(request)
            flag.set()
            return httpx.Response(503)

        provider = self.provider(handle)
        with self.assertRaises(PaperReadCancelled), provider.cancellation_scope(flag.is_set):
            provider.ingest("https://fixture.invalid/p.pdf", "paper")
        self.assertEqual(len(calls), 1)

    def test_cancel_after_existing_backoff_prevents_next_attempt(self):
        """退避中及时检查取消，不开始下一次请求。"""
        flag, calls, waits = threading.Event(), [], []
        provider = self.provider(lambda request: calls.append(request) or httpx.Response(503))

        def wait(seconds):
            waits.append(seconds)
            flag.set()

        with patch("research_loop.providers.pdf.time.sleep", side_effect=wait), \
                self.assertRaises(PaperReadCancelled), provider.cancellation_scope(flag.is_set):
            provider.ingest("https://fixture.invalid/p.pdf", "paper")
        self.assertEqual(len(waits), 1)
        self.assertGreater(waits[0], 0)
        self.assertLess(waits[0], 2.0)
        self.assertEqual(len(calls), 1)

    def test_no_cancel_keeps_jina_attempts_timeout_and_fallback(self):
        """三次Jina及原PDF回退均保留，不因新作用域增减请求。"""
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((40, 40), ("Synthetic scientific paper evidence.\n" * 45), fontsize=9)
        content = document.tobytes()
        document.close()
        calls, waits = [], []

        def handle(request):
            calls.append(request)
            return httpx.Response(503) if request.url.host == "r.jina.ai" else httpx.Response(200, content=content)

        provider = self.provider(handle)
        with patch("research_loop.providers.pdf.time.sleep", side_effect=waits.append), provider.cancellation_scope(lambda: False):
            result = provider.ingest("https://fixture.invalid/p.pdf", "paper")
        self.assertEqual(result.total_pages, 1)
        self.assertEqual(len(calls), 4)
        self.assertAlmostEqual(sum(waits), 6.0)
        self.assertTrue(all(r.extensions["timeout"]["read"] == 60.0 for r in calls))
        self.assertEqual(provider.max_bytes, 80 * 1024 * 1024)

    def test_scope_is_thread_local_and_restores_nested_callback(self):
        """并行读取不继承取消，嵌套异常也恢复外层回调。"""
        provider = self.provider(lambda request: httpx.Response(200))
        flag = threading.Event()
        with provider.cancellation_scope(flag.is_set):
            with self.assertRaises(PaperReadCancelled), provider.cancellation_scope(lambda: True):
                self.fail("cancelled nested scope must not yield")
            provider._check_cancelled()
            flag.set()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(provider._check_cancelled).result(timeout=3)
            with self.assertRaises(PaperReadCancelled):
                provider._check_cancelled()
            flag.clear()
        provider._check_cancelled()

    def test_cancel_before_parse_has_no_file_commit(self):
        """下载返回后取消，PDF文件与解析缓存均不提交。"""
        flag = threading.Event()
        provider = self.provider(lambda request: httpx.Response(200))

        def download(url):
            flag.set()
            return b"%PDF-synthetic", url, "application/pdf"

        with patch.object(provider, "_download", side_effect=download), \
                self.assertRaises(PaperReadCancelled), provider.cancellation_scope(flag.is_set):
            provider.ingest_pages("https://fixture.invalid/p.pdf", "paper")
        self.assertEqual(list(self.root.rglob("*.pdf")), [])

    def test_existing_byte_limit_still_rejects_oversized_stream(self):
        """无取消时仍使用原字节限制和原错误码。"""
        provider = self.provider(lambda request: httpx.Response(200, content=b"%PDF-too-long"), max_bytes=5)
        with self.assertRaises(PaperAcquisitionError) as caught, provider.cancellation_scope(lambda: False):
            provider.ingest_pages("https://fixture.invalid/p.pdf", "paper")
        self.assertEqual(caught.exception.code, "PDF_TOO_LARGE")
