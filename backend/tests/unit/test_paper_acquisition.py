"""获取层固定响应回归；检查 论文 地址、错误分类与真实本地解析，不调用模型。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import pymupdf

from research_loop.domain.evidence import PaperCandidate
from research_loop.domain.tasks import SavePolicy
from research_loop.providers.discovery import DiscoveryProvider
from research_loop.providers.paper_acquisition import PaperAcquisitionError, pdf_from_detail, resolve_pdf_url
from research_loop.providers.paper_sources import resolve_backends
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs
from research_loop.tools.registry import ToolRegistry


class PaperURLTests(unittest.TestCase):
    def test_upstream_conference_routes(self):
        """论文 支持的详情地址应确定性转换；编号不是某篇论文的特判。"""
        pairs = [
            ("https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3396_ECCV_2024_paper.php", "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03396.pdf"),
            ("https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/91_ECCV_2024_paper.php", "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00091.pdf"),
            ("https://openaccess.thecvf.com/content/CVPR2025/html/Test_paper.html", "https://openaccess.thecvf.com/content/CVPR2025/papers/Test_paper.pdf"),
            ("https://openreview.net/forum?id=abc", "https://openreview.net/pdf?id=abc"),
            ("https://aclanthology.org/2025.emnlp-main.384/", "https://aclanthology.org/2025.emnlp-main.384.pdf"),
            ("https://paper.example/main.pdf?download=1", "https://paper.example/main.pdf?download=1"),
            ("https://paper.example/detail", "https://paper.example/detail"),
        ]
        for original, expected in pairs:
            with self.subTest(url=original):
                self.assertEqual(resolve_pdf_url(None, original), expected)
        self.assertEqual(resolve_pdf_url("https://other.example/v2.pdf", pairs[0][0]), "https://other.example/v2.pdf")

    def test_no_substring_host_or_supplement_guess(self):
        url = "https://ecva.net.evil.example/html/99_paper.php"
        self.assertEqual(resolve_pdf_url(None, url), url)
        self.assertEqual(pdf_from_detail('<a href="appendix.pdf">PDF</a>', "https://paper.example/a"), "")

    def test_detail_metadata_relative_and_main_link(self):
        base = "https://proceedings.iclr.cc/paper/2026/hash/a.html"
        self.assertEqual(pdf_from_detail('<meta name="citation_pdf_url" content="../file/a.pdf"><a href="a-supp.pdf">PDF</a>', base),
                         "https://proceedings.iclr.cc/paper/2026/file/a.pdf")
        self.assertEqual(pdf_from_detail('<a href="/paper/main.pdf"><span>Download PDF</span></a>', base),
                         "https://proceedings.iclr.cc/paper/main.pdf")
        self.assertEqual(pdf_from_detail('<a href="https://raw.githubusercontent.com/mlresearch/v1/main/a.pdf">Paper</a>', base),
                         "https://raw.githubusercontent.com/mlresearch/v1/main/a.pdf")

    def test_ambiguous_or_external_pdf_not_chosen(self):
        for html in ['<a href="a.pdf">PDF</a><a href="b.pdf">PDF</a>',
                     '<meta name="citation_pdf_url" content="https://unrelated.example/a.pdf">',
                     '<meta name="citation_pdf_url" content="file:///tmp/p.pdf">']:
            with self.subTest(html=html):
                self.assertEqual(pdf_from_detail(html, "https://paper.example/detail"), "")


class PaperPDFTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = FileStore(self.root / "data", self.root / "out", self.root)
        document = pymupdf.open()
        for number in range(1, 4):
            page = document.new_page()
            page.insert_text((40, 40), (f"Research paper page {number}. Evidence remains on its physical page.\n" * 12))
        self.pdf = document.tobytes()
        document.close()

    def provider(self, handler, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        provider = PDFProvider(self.files, client=client, **kwargs)
        provider.jina_retry_wait = 0
        return provider

    def test_detail_to_pdf_full_pages_and_content_cache(self):
        """PDF 回退使用 PyMuPDF 原始提取，默认全页，重复正文命中既有解析缓存。"""
        calls = []
        def handler(request):
            if request.url.host == "r.jina.ai": return httpx.Response(200, text="short error")
            calls.append(str(request.url))
            if request.url.path == "/detail":
                return httpx.Response(200, text='<meta name="citation_pdf_url" content="/main.pdf">', headers={"content-type": "text/html"})
            return httpx.Response(200, content=self.pdf, headers={"content-type": "application/pdf"})
        provider = self.provider(handler)
        first = provider.ingest_pages("https://paper.example/detail", "paper-1")
        pages = provider.pages(first)
        self.assertEqual([p.page_number for p in pages], [1, 2, 3])
        self.assertEqual(first.parser_name, "pymupdf")
        self.assertTrue(all(p.text and p.parsed_markdown and p.document_id == first.document_id for p in pages))
        self.assertEqual(provider.last_fetch["resolved_pdf_url"], "https://paper.example/main.pdf")
        self.assertEqual(calls, ["https://paper.example/detail", "https://paper.example/main.pdf"])
        with patch.object(provider, "_parse", side_effect=AssertionError("must reuse parsed content")):
            second = provider.ingest_pages("https://paper.example/detail", "paper-1")
        self.assertEqual(first.cache_path, second.cache_path)
        self.assertEqual(first.content_sha256, second.content_sha256)

    def test_redirect_and_pdf_magic_over_content_type(self):
        def handler(request):
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "/main.pdf"})
            return httpx.Response(200, content=self.pdf, headers={"content-type": "application/octet-stream"})
        provider = self.provider(handler)
        self.assertEqual(provider.ingest_pages("https://paper.example/start", "p").total_pages, 3)
        self.assertEqual(provider.last_fetch["attempts"][0]["redirects"][0]["status"], 302)

    def test_http_network_and_not_pdf_are_distinct(self):
        cases = [(403, b"Forbidden", "text/plain", "PDF_HTTP_ERROR", False),
                 (429, b"Limit", "text/plain", "PDF_HTTP_ERROR", False),
                 (503, b"Unavailable", "text/plain", "PDF_HTTP_ERROR", True),
                 (200, b'<html><body>abstract only</body></html>', "text/html", "PDF_LINK_UNAVAILABLE", False),
                 (200, b'{"message":"error"}', "application/json", "PDF_NOT_PDF", False)]
        for status, content, content_type, code, retryable in cases:
            with self.subTest(code=code, status=status):
                provider = self.provider(lambda req: httpx.Response(status, content=content, headers={"content-type": content_type}))
                with self.assertRaises(PaperAcquisitionError) as raised:
                    provider.ingest_pages("https://paper.example/test", "p")
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.retryable, retryable)
                self.assertEqual(len(provider.last_fetch["attempts"]), 1)
        def timeout(request):
            raise httpx.ReadTimeout("test", request=request)
        provider = self.provider(timeout)
        with self.assertRaises(PaperAcquisitionError) as raised:
            provider.ingest_pages("https://paper.example/timeout", "p")
        self.assertEqual(raised.exception.code, "PDF_NETWORK_ERROR")
        self.assertEqual(provider.last_fetch["attempts"][0]["error"], "ReadTimeout")

    def test_size_and_parse_error_not_published_as_evidence(self):
        provider = self.provider(lambda req: httpx.Response(200, content=self.pdf), max_bytes=20)
        with self.assertRaises(PaperAcquisitionError) as raised:
            provider.ingest_pages("https://paper.example/test.pdf", "p")
        self.assertEqual(raised.exception.code, "PDF_TOO_LARGE")
        provider = self.provider(lambda req: httpx.Response(200, content=b"%PDF-1.7 invalid"))
        with self.assertRaises(PaperAcquisitionError) as raised:
            provider.ingest_pages("https://paper.example/test.pdf", "p")
        self.assertEqual(raised.exception.code, "PDF_PARSE_FAILED")
        self.assertEqual(list(self.files.parse_dir.glob("*.json")), [])

    def test_service_reads_known_paper_without_pdf_field(self):
        """复现 LaMI 类问题：合法身份只有详情链接，工具应取得全页而非报身份错误。"""
        calls = []
        def handler(request):
            if request.url.host == "r.jina.ai": return httpx.Response(200, text="short error")
            calls.append(str(request.url))
            return httpx.Response(200, content=self.pdf)
        provider = self.provider(handler)
        db = Database("sqlite:///" + str(self.root / "test.db"))
        db.ensure_schema()
        discovery = DiscoveryProvider()
        self.addCleanup(discovery.close)
        papers = PaperService(db, discovery, provider)
        paper = PaperCandidate(paper_id="eccv-fixture-91", title="Test", source="eccv",
                               paper_url="https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/91_ECCV_2024_paper.php")
        db.save_paper(paper.paper_id, paper.model_dump(mode="json"))
        db.grant_paper("task", paper.paper_id, "task")
        ctx = ToolContext(task_id="task", save_policy=SavePolicy.deny, services={"papers": papers})
        registry = ToolRegistry(specs())
        result = registry.execute("read_paper", {"target_ref": paper.paper_id}, ctx)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data["returned_pages"], [1, 2, 3])
        self.assertEqual(calls, ["https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00091.pdf"])
        self.assertTrue(all(db.task_has_evidence("task", p["evidence_id"]) for p in result.data["pages"]))
        missing = PaperCandidate(paper_id="known-no-url", title="Test", source="eccv")
        db.save_paper(missing.paper_id, missing.model_dump(mode="json"))
        db.grant_paper("task", missing.paper_id, "task")
        result = registry.execute("read_paper", {"target_ref": missing.paper_id}, ctx)
        self.assertEqual(result.data["error_code"], "PDF_URL_MISSING")
        self.assertFalse(result.retryable)
        result = registry.execute("read_paper", {"target_ref": "invented-paper"}, ctx)
        self.assertEqual(result.data["error_code"], "PAPER_REFERENCE_INVALID")
        other = ToolContext(task_id="unauthorized", save_policy=SavePolicy.deny, services={"papers": papers})
        self.assertFalse(registry.execute("read_paper", {"target_ref": paper.paper_id}, other).ok)
        self.assertEqual(len(calls), 1)


class PaperRetryTests(unittest.TestCase):
    def provider(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        backends = resolve_backends({"supabase": {"url": "https://papers.example", "anon_key": "private-test-key"}})
        return DiscoveryProvider(backends=backends, client=client, retry_wait_seconds=0)

    def test_transient_retry_then_success_with_rpc_timeout(self):
        calls = []
        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(503)
            if len(calls) == 2:
                raise httpx.ReadTimeout("test", request=request)
            return httpx.Response(200, json=[{"id": "2601.12345", "title": "P"}])
        provider = self.provider(handler)
        result = provider.search("object detection", sources=["arxiv"])
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0].extensions["timeout"]["read"], 45)
        self.assertEqual([r.get("http_status", r.get("error")) for r in provider.last_requests[0]["attempts"]], [503, "ReadTimeout", 200])
        self.assertNotIn("private-test-key", str(provider.last_requests))

    def test_four_attempt_cap_and_network_diagnostic(self):
        calls = []
        def handler(request):
            calls.append(request)
            raise httpx.ConnectTimeout("test", request=request)
        provider = self.provider(handler)
        result = provider.search("query", sources=["cvpr"])
        failed = next(s for s in result.source_results if s.source == "cvpr")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(len(calls), 4)
        self.assertEqual(failed.request["timeout_seconds"], 20)
        self.assertEqual(len(failed.request["attempts"]), 4)
        self.assertNotIn("http_status", failed.request)

    def test_only_rate_limit_retries_access_denial_or_statement_timeout_stops(self):
        for status, body in [(429, {}), (403, {}), (500, {"code": "57014"})]:
            with self.subTest(status=status):
                provider = self.provider(lambda req: httpx.Response(status, json=body))
                result = provider.search("query", sources=["cvpr"])
                failed = next(s for s in result.source_results if s.source == "cvpr")
                self.assertEqual(failed.status, "failed")
                self.assertEqual(len(failed.request["attempts"]), 4 if status == 429 else 1)
                self.assertEqual(failed.request["http_status"], status)


if __name__ == "__main__":
    unittest.main()
