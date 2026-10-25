"""访问挑战与备用衔接回归：合成 HTTP、临时 SQLite 或指定隔离 MySQL。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import httpx

from research_loop.domain.evidence import PaperCandidate
from research_loop.providers.paper_acquisition import (
    PaperAcquisitionError, access_challenge_reason, blocked_primary_hosts,
)
from research_loop.providers.pdf import PDFProvider, is_usable_paper_text
from research_loop.providers.web_fetch import WebFetchProvider
from research_loop.providers.web_search import WebSearchProvider
from research_loop.services.web_acquisition import WebAcquisitionService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database


ORIGINAL = "https://openreview.net/forum?id=SyntheticAccessPaper"
ORIGINAL_PDF = "https://openreview.net/pdf?id=SyntheticAccessPaper"
CHALLENGE = "https://openreview.net/challenge?redirect=%2Fpdf%3Fid%3DSyntheticAccessPaper"
ALTERNATE = "https://arxiv.org/abs/2601.12345v1"
ALTERNATE_PDF = "https://arxiv.org/pdf/2601.12345v1"
TITLE = "Synthetic Visual Reasoning"


def old_challenge_failure():
    """复现 F1 的旧诊断形状，不包含私有正文、模型输出或真实论文身份。"""
    return {"error_code": "PDF_LINK_UNAVAILABLE", "acquisition": {
        "requested_source": ORIGINAL_PDF, "final_url": CHALLENGE, "status": 200,
        "content_type": "text/html; charset=utf-8", "attempts": [{
            "requested_url": ORIGINAL_PDF, "final_url": CHALLENGE, "status": 200,
            "content_type": "text/html; charset=utf-8",
            "redirects": [{"url": ORIGINAL_PDF, "status": 302}],
        }],
    }}


class AccessChallengeTests(unittest.TestCase):
    """纯检测与下载测试，不向真实地址发送请求。"""

    def test_original_redirect_challenge_is_not_a_paper_detail(self):
        """真实响应形状 302→200 HTML 输出准确阻断及可复用原址诊断。"""
        calls = []

        def respond(request):
            """用本地响应重放重定向，其他请求直接报错。"""
            calls.append(str(request.url))
            if str(request.url) == ORIGINAL_PDF:
                return httpx.Response(302, headers={"location": CHALLENGE})
            if str(request.url) == CHALLENGE:
                return httpx.Response(200, text="<html><title>Security Verification</title>Verify you are human</html>",
                                      headers={"content-type": "text/html"})
            raise AssertionError("unexpected synthetic request: " + str(request.url))

        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(respond)) as client:
            files = FileStore(Path(directory) / "data", Path(directory) / "out", directory)
            provider = PDFProvider(files, client=client)
            with self.assertRaises(PaperAcquisitionError) as raised:
                provider.ingest_pages(ORIGINAL, "synthetic")
            failure = raised.exception
            self.assertEqual(failure.code, "PDF_ACCESS_CHALLENGE")
            self.assertFalse(failure.retryable)
            self.assertEqual(failure.acquisition["failure_kind"], "access_challenge")
            self.assertEqual(failure.acquisition["final_url"], CHALLENGE)
            self.assertEqual(failure.acquisition["attempts"][0]["redirects"][0]["status"], 302)
            self.assertEqual(blocked_primary_hosts({"error_code": failure.code, "acquisition": failure.acquisition}), {"openreview.net"})
            self.assertEqual(list(files.pdf_dir.glob("*.pdf")), [])
        self.assertEqual(calls, [ORIGINAL_PDF, CHALLENGE])

    def test_old_failure_is_recognized_without_rewriting_it(self):
        """旧失败无需伪造新 HTTP 或重写原收据即可识别为验证页。"""
        failure = old_challenge_failure()
        before = json.dumps(failure, sort_keys=True)
        self.assertEqual(blocked_primary_hosts(failure), {"openreview.net"})
        self.assertEqual(json.dumps(failure, sort_keys=True), before)

    def test_challenge_word_in_paper_is_not_a_verification_page(self):
        """安全研究论文正常讨论 CAPTCHA 时不可被字面关键词阻断。"""
        content = "<html><title>CAPTCHA as Visual Reasoning</title><h1>Method</h1>" + (
            "We compare captcha solving and access denied classifications in visual reasoning. " * 30) + "</html>"
        self.assertIsNone(access_challenge_reason(ALTERNATE, content, content_type="text/html"))
        self.assertIsNone(access_challenge_reason("https://example.org/challenge", content, content_type="text/html"))
        self.assertIsNone(access_challenge_reason(ALTERNATE, "We analyze captcha and checking your browser messages.",
                                                  title="CAPTCHA as Visual Reasoning"))
        self.assertTrue(is_usable_paper_text("CAPTCHA as Visual Reasoning\n" + "We evaluate captcha tasks. " * 80))

    def test_verified_page_structure_and_plain_prompt_are_detected(self):
        """HTML 标题及纯文本开头的验证指令均保留明确原因。"""
        self.assertEqual(access_challenge_reason(ALTERNATE, "<html><title>Just a moment...</title>Enable JavaScript and cookies</html>"),
                         "verification_page")
        self.assertEqual(access_challenge_reason(ALTERNATE, "Verify you are human\n[PDF](https://example.org/a.pdf)"),
                         "verification_prompt")
        self.assertFalse(is_usable_paper_text("Verify you are human\n" + "Page unavailable. " * 100))
        self.assertIsNone(access_challenge_reason(CHALLENGE, "%PDF-1.7 synthetic", content_type="application/pdf"))

    def test_scholarly_title_with_verification_prefix_is_usable_text(self):
        """学术标题的验证短语前缀不能覆盖副标题、作者与正常正文。"""
        title = "Verify You Are Human: A Survey of CAPTCHA Solving"
        article = title + "\nAlice Example\nAbstract\n" + "This survey compares visual challenge recognition methods. " * 40
        self.assertIsNone(access_challenge_reason(ALTERNATE, article, title=title))
        self.assertTrue(is_usable_paper_text(article))
        self.assertIsNone(access_challenge_reason(ALTERNATE, "Verify you are human is a phrase analyzed in the study."))

    def test_jina_failure_stays_on_jina_and_does_not_block_publisher(self):
        """代理认证/验证失败不应被归因到原 PDF 域名。"""
        primary = {"error_code": "PDF_HTTP_ERROR", "acquisition": {
            "requested_source": ORIGINAL_PDF, "final_url": ORIGINAL_PDF, "status": 404,
            "attempts": [{"requested_url": ORIGINAL_PDF, "status": 404}],
            "jina_attempts": [{"provider": "jina", "url": "https://r.jina.ai/" + ORIGINAL_PDF,
                               "http_status": 401, "failure_kind": "access_challenge"}],
        }}
        self.assertEqual(blocked_primary_hosts(primary), set())
        self.assertEqual(blocked_primary_hosts({"acquisition": {"status": 403, "final_url": ORIGINAL_PDF}}), {"openreview.net"})

    def test_jina_verification_returns_to_direct_pdf_without_proxy_retry(self):
        """Jina 自身返回验证内容时只记该尝试，不把内容当论文缓存。"""
        calls = []

        def respond(request):
            """只测试 Jina 子路径，没有真实正文下载。"""
            calls.append(str(request.url))
            return httpx.Response(200, text="Title: Security Verification\nVerify you are human\n" + "Unavailable. " * 120)

        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(respond)) as client:
            provider = PDFProvider(FileStore(Path(directory) / "data", Path(directory) / "out", directory), client=client)
            text, attempts = provider._jina(ORIGINAL_PDF)
        self.assertEqual(text, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(attempts[0]["provider"], "jina")
        self.assertEqual(attempts[0]["retry_stopped_reason"], "access_challenge")


class ChallengeFallbackTests(unittest.TestCase):
    """持久 Search/Fetch 替身与受阻域名、授权来源边界的集成验证。"""

    def setUp(self):
        """每例创建独立临时文件与任务，MySQL 仅使用显式测试连接。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Database(os.environ.get("RESEARCHLOOP_TEST_WEB_DATABASE_URL") or "sqlite:///" + temporary.name + "/challenge.db")
        self.db.ensure_schema()
        self.task = "challenge-test-" + uuid4().hex
        self.db.create_task(self.task, None, "synthetic", "Try OpenReview first; use a public alternate if blocked.", "deny", [])
        self.calls = []
        self.fetch_title = TITLE
        self.fetch_content = "Synthetic Author\nWe analyze captcha in visual reasoning.\n[PDF](" + ALTERNATE_PDF + ")"
        self.results = [{"title": TITLE, "url": ORIGINAL}, {"title": TITLE, "url": ALTERNATE}]
        self.client = httpx.Client(transport=httpx.MockTransport(self.respond))
        self.addCleanup(self.client.close)
        self.service = WebAcquisitionService(self.db, WebSearchProvider("fixture", client=self.client),
                                             WebFetchProvider("fixture", client=self.client))
        self.candidate = PaperCandidate(paper_id="synthetic", title=TITLE, authors=["Synthetic Author"],
                                        source="openreview", paper_url=ORIGINAL)
        # 对齐检索返回候选的持久化合同，只授予本例的确切论文。
        self.db.save_paper(self.candidate.paper_id, self.candidate.model_dump(mode="json"))
        self.db.grant_paper(self.task, self.candidate.paper_id, self.task)

    def respond(self, request):
        """只模拟付费组件接口，不发送任何网络请求。"""
        payload = json.loads(request.content)
        self.calls.append((request.url.path, payload))
        if request.url.path.endswith("/search"):
            return httpx.Response(200, json={"results": self.results})
        if request.url.path.endswith("/fetch"):
            return httpx.Response(200, json={"title": self.fetch_title, "content": self.fetch_content})
        raise AssertionError("unexpected synthetic request")

    def test_old_challenge_skips_original_and_locates_authorized_alternate(self):
        """旧验证页诊断阻止原站代理重取；备用页正常提及 captcha 不误拒。"""
        first = self.service.locate_pdf(self.task, self.candidate, primary_failure=old_challenge_failure())
        self.assertEqual(first["status"], "located")
        self.assertEqual(first["pdf_url"], ALTERNATE_PDF)
        self.assertEqual(first["diagnostics"][0]["status"], "original_access_denied_no_proxy_bypass")
        self.assertEqual([payload["url"] for path, payload in self.calls if path.endswith("/fetch")], [ALTERNATE])
        second = self.service.locate_pdf(self.task, self.candidate, primary_failure=old_challenge_failure())
        self.assertEqual(second["pdf_url"], ALTERNATE_PDF)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(second["fetch_receipt"]["reused"])

    def test_explicit_host_scope_is_not_bypassed_for_alternate(self):
        """用户硬域名限制仍优先；受阻后没有合法备用就明确结束。"""
        self.db.execute("UPDATE tasks SET prompt=? WHERE task_id=?", ("Only use openreview.net for downloads.", self.task))
        result = self.service.locate_pdf(self.task, self.candidate, primary_failure=old_challenge_failure())
        self.assertEqual(result["status"], "unconfirmed_identity_or_pdf")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["search"]["rejected"][0]["reason"], "outside_explicit_host_scope")
        self.assertIsNone(result["pdf_url"])

    def test_fetch_verification_is_reported_with_receipt_without_pdf(self):
        """备用站也验证时保留其成功 HTTP 收据和不可取文事实，不继续伪造全文。"""
        self.fetch_title = "Security Verification"
        self.fetch_content = "Verify you are human\n[PDF](" + ALTERNATE_PDF + ")"
        result = self.service.locate_pdf(self.task, self.candidate, primary_failure=old_challenge_failure())
        self.assertEqual(result["status"], "unconfirmed_identity_or_pdf")
        rejection = result["diagnostics"][-1]
        self.assertEqual(rejection["status"], "access_challenge")
        self.assertEqual(rejection["request_state"], "completed")
        self.assertTrue(rejection["receipt_id"])
        self.assertIsNone(result["pdf_url"])


if __name__ == "__main__":
    unittest.main()
