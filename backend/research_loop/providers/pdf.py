"""正文获取：有效缓存→Jina→PDF/PyMuPDF；保留真实物理页码。

不将远端文本段落推断为 PDF 页；第三方适配许可见仓库根 NOTICE。"""
from __future__ import annotations

import hashlib
import json
import re
import time
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import httpx

from research_loop.domain.evidence import AssetReference, Document, PageEvidence
from research_loop.providers.paper_acquisition import PaperAcquisitionError, pdf_from_detail, resolve_pdf_url
from research_loop.storage.files import FileStore, sha256_bytes


class PDFProvider:
    def __init__(self, file_store: FileStore, parser_name: str = "dpr",
                 parser_version: str = "1", parser_config: dict[str, Any] | None = None,
                 timeout: float = 60.0, max_bytes: int = 80 * 1024 * 1024,
                 client: httpx.Client | None = None):
        self.file_store = file_store
        self.parser_name = parser_name
        self.parser_version = parser_version
        self.parser_config = parser_config or {"text_route": "cache-jina-pymupdf", "min_text_chars": 1200}
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self._owns_client = client is None
        # Researcher 使用线程池共享 provider；诊断必须与当前读取线程绑定，不能串到其他论文。
        self._fetch_state = threading.local()
        self.jina_retry_wait = 2.0

    @property
    def last_fetch(self) -> dict[str, Any]:
        """返回当前同步读取线程的下载诊断，供 PaperService 在同线程取回。"""
        return getattr(self._fetch_state, "diagnostics", {})

    @last_fetch.setter
    def last_fetch(self, value: dict[str, Any]) -> None:
        """替换当前线程诊断；不影响其他并行 Researcher 的读取记录。"""
        self._fetch_state.diagnostics = value

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def ingest(self, source: str, paper_id: str) -> Document:
        """使用缓存→Jina→PDF/PyMuPDF 路线，本地附件不发送远端。

        Args:
            source: 已授权的 URL 或本地 PDF。
            paper_id: 业务论文身份，不从文本猜测身份。

        Returns:
            可用正文快照；Jina 文本的 total_pages 为 None，不冒充物理页覆盖。
        """
        if not source.startswith(("https://", "http://")):
            return self._ingest_pdf(source, paper_id)
        source = resolve_pdf_url(None, source)
        key = sha256_bytes(source.encode())
        index_path = self.file_store.parse_dir / ("source-" + key + ".json")
        if index_path.exists():
            try:
                document = Document.model_validate(json.loads(index_path.read_text()))
                cached = json.loads(Path(document.cache_path).read_text())
                if is_usable_paper_text("\n\n".join(p["text"] for p in cached["pages"])):
                    if document.parser_name == "jina" and sha256_bytes(cached["pages"][0]["text"].encode()) != document.content_sha256:
                        raise ValueError("text cache hash mismatch")
                    self.last_fetch = {"kind": "text_cache", "source": source, "attempts": [], "cache_path": document.cache_path}
                    return document.model_copy(update={"paper_id": paper_id})
            except (ValueError, OSError, KeyError, TypeError):
                pass  # 损坏或过期格式不能当成有效全文，重新走同论文获取路线。
        text, attempts = self._jina(source)
        if is_usable_paper_text(text):
            digest = sha256_bytes(text.encode())
            document_id = "text:" + digest[:32]
            page = PageEvidence(evidence_id="evidence:" + digest[:40], document_id=document_id,
                                page_number=None, text=text, parsed_markdown=text)
            cache_path = self.file_store.cache_parse_path("jina-" + digest)
            payload = {"total_pages": None, "pages": [page.model_dump(mode="json")],
                       "gaps": ["Jina 正文无可靠物理页码；全文覆盖和图表公式解释未逐页验证"]}
            self.file_store.write_bytes_atomic(cache_path, json.dumps(payload, ensure_ascii=False).encode())
            document = Document(document_id=document_id, paper_id=paper_id, content_sha256=digest,
                                parser_name="jina", parser_version="reader-v1", config_sha256=sha256_bytes(b"dpr-jina-v1"),
                                total_pages=None, cache_path=str(cache_path), gaps=payload["gaps"])
            self.last_fetch = {"kind": "jina", "source": source, "attempts": attempts, "hash_kind": "extracted_text"}
        else:
            try:
                document = self._ingest_pdf(source, paper_id)
            finally:
                self.last_fetch["jina_attempts"] = attempts
        # 只更新可重建的 URL 缓存索引；正文/解析快照由内容 hash 命名并保留历史。
        fd, temporary = tempfile.mkstemp(dir=index_path.parent, prefix=".source-")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(document.model_dump_json())
            os.replace(temporary, index_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return document

    def ingest_pages(self, source: str, paper_id: str) -> Document:
        """物理页请求必须直接解析 PDF，不能用 Jina 正文片段冒充指定页。"""
        return self._ingest_pdf(source, paper_id)

    def _jina(self, source: str) -> tuple[str, list[dict[str, Any]]]:
        """最多三次只读 Jina 请求；有效性门失败交给 PDF 回退。

        Args:
            source: 当前论文的解析地址。

        Returns:
            正文或空串，以及全部 HTTP/传输尝试；不包含认证信息。
        """
        attempts = []
        for number in range(1, 4):
            attempt = {"provider": "jina", "attempt": number, "url": "https://r.jina.ai/" + source}
            attempts.append(attempt)
            try:
                with self.client.stream("GET", attempt["url"], timeout=60, follow_redirects=True) as response:
                    attempt["http_status"] = response.status_code
                    parts, size = [], 0
                    for part in response.iter_bytes():
                        size += len(part)
                        if size > self.max_bytes:
                            raise ValueError("Jina response exceeds byte limit")
                        parts.append(part)
                    text = b"".join(parts).decode("utf-8", errors="replace")
                    attempt["usable"] = is_usable_paper_text(text)
                    if response.status_code == 200 and text.strip():
                        return text, attempts
            except (httpx.HTTPError, ValueError) as exc:
                attempt["error"] = type(exc).__name__
            if number < 3:
                time.sleep(self.jina_retry_wait * number)
        return "", attempts

    def _ingest_pdf(self, source: str, paper_id: str) -> Document:
        """读取完整 PDF，以内容 hash、解析器版本和配置组成缓存键；命中缓存时不重复解析。"""
        content = self._read_source(source)
        content_hash = sha256_bytes(content)
        pdf_path = self.file_store.cache_pdf_path(content_hash)
        if not pdf_path.exists():
            self.file_store.write_bytes_atomic(pdf_path, content)
        config_hash = hashlib.sha256(json.dumps(self.parser_config, sort_keys=True).encode()).hexdigest()
        cache_key = hashlib.sha256(f"{content_hash}:pymupdf:{self.parser_version}:{config_hash}".encode()).hexdigest()
        cache_path = self.file_store.cache_parse_path(cache_key)
        document_id = f"doc:{cache_key[:32]}"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return Document(document_id=document_id, paper_id=paper_id, content_sha256=content_hash,
                            parser_name="pymupdf", parser_version=self.parser_version,
                            config_sha256=config_hash, total_pages=cached["total_pages"],
                            cache_path=str(cache_path), gaps=cached.get("gaps", []))
        try:
            pages = self._parse(pdf_path)
        except Exception as exc:
            raise PaperAcquisitionError("PDF_PARSE_FAILED", f"PDF 已取得，解析失败：{type(exc).__name__}",
                                        acquisition=self.last_fetch) from exc
        if not is_usable_paper_text("\n\n".join(page.text for page in pages)):
            raise PaperAcquisitionError("PDF_TEXT_UNUSABLE", "全文文本未通过 基本有效性检查", acquisition=self.last_fetch)
        for page in pages:
            page.document_id = document_id
            page.evidence_id = f"evidence:{hashlib.sha256(f'{document_id}:{page.page_number}'.encode()).hexdigest()[:40]}"
        payload = {"total_pages": len(pages), "pages": [page.model_dump(mode="json") for page in pages],
                   "gaps": sorted({gap for page in pages for gap in page.gaps})}
        self.file_store.write_bytes_atomic(cache_path, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        return Document(document_id=document_id, paper_id=paper_id, content_sha256=content_hash,
                        parser_name="pymupdf", parser_version=self.parser_version,
                        config_sha256=config_hash, total_pages=len(pages), cache_path=str(cache_path),
                        gaps=payload["gaps"])

    def pages(self, document: Document, start_page: int = 1,
              page_count: int | None = None) -> list[PageEvidence]:
        """从解析缓存读取完整正文或指定的物理页片段。

        Args:
            document: 带有解析缓存路径和总页数的文档。
            start_page: 从 1 起算的物理起始页。
            page_count: 返回页数；None 表示读取至末页，没有 10 页上限。

        Returns:
            按原文顺序排列、保留证据身份的页列表。

        Raises:
            ValueError: 起始页越界或页数非正数。
        """
        cached = json.loads(Path(document.cache_path).read_text(encoding="utf-8"))
        if document.total_pages is None:
            if start_page != 1 or page_count is not None:
                raise ValueError("Jina text has no physical page mapping; request direct PDF pages")
            return [PageEvidence.model_validate(raw) for raw in cached["pages"]]
        if start_page < 1 or start_page > document.total_pages or (page_count is not None and page_count < 1):
            raise ValueError("requested physical pages are outside the document")
        start = start_page
        end = document.total_pages if page_count is None else min(document.total_pages, start + page_count - 1)
        out: list[PageEvidence] = []
        for raw in cached["pages"][start - 1:end]:
            out.append(PageEvidence.model_validate(raw))
        return out

    def _read_source(self, source: str) -> bytes:
        """获取 PDF 字节；来源地址转换后允许一次详情页解析，正文仍交本地解析器。

        Args:
            source: 已授权的本地附件、PDF URL 或同论文详情页。

        Returns:
            通过文件头和大小检查的 PDF 字节。

        Raises:
            PaperAcquisitionError: 无主文链接、HTTP/网络失败、错误文件或大小超限。
        """
        from urllib.parse import urlsplit
        self.last_fetch = {"requested_source": source, "attempts": []}
        is_url = urlsplit(source).scheme in {"http", "https"}
        if not is_url:
            path = Path(source).expanduser().resolve()
            if not path.is_file():
                raise PaperAcquisitionError("PDF_LOCAL_MISSING", "授权的本地 PDF 不存在", acquisition=self.last_fetch)
            if path.suffix.lower() != ".pdf":
                raise PaperAcquisitionError("PDF_NOT_PDF", "本地附件不是 PDF", acquisition=self.last_fetch)
            if path.stat().st_size > self.max_bytes:
                raise PaperAcquisitionError("PDF_TOO_LARGE", f"PDF exceeds configured byte limit {self.max_bytes}", acquisition=self.last_fetch)
            content = path.read_bytes()
            self.last_fetch.update(kind="local", path=str(path), bytes=len(content))
        else:
            target = resolve_pdf_url(None, source)
            content, final_url, content_type = self._download(target)
            if not content.lstrip().startswith(b"%PDF-"):
                is_html = "html" in content_type.lower() or content.lstrip().lower().startswith((b"<!doctype html", b"<html"))
                if not is_html:
                    raise PaperAcquisitionError("PDF_NOT_PDF", "下载结果不是 PDF 或可解析的论文详情页", acquisition=self.last_fetch)
                pdf_url = pdf_from_detail(content.decode("utf-8", errors="replace"), final_url)
                if not pdf_url or pdf_url == final_url:
                    raise PaperAcquisitionError("PDF_LINK_UNAVAILABLE", "已找到论文详情，但页面没有唯一明确的主文 PDF 链接；论文身份仍有效", acquisition=self.last_fetch)
                self.last_fetch["detail_url"] = final_url
                self.last_fetch["resolved_pdf_url"] = pdf_url
                content, _, _ = self._download(pdf_url)
        if not content.lstrip().startswith(b"%PDF-"):
            raise PaperAcquisitionError("PDF_NOT_PDF", "下载结果不是 PDF，不把 HTML/摘要充当全文", acquisition=self.last_fetch)
        self.last_fetch["bytes"] = len(content)
        return content

    def _download(self, url: str) -> tuple[bytes, str, str]:
        """有界下载并记录每次实际请求；保持原始 PDF 的 60 秒默认超时。

        Args:
            url: 直接 PDF 或详情页地址。

        Returns:
            字节、重定向后的地址和 Content-Type。

        Raises:
            PaperAcquisitionError: HTTP 错误、网络异常或文件超限。此层不自动重发。
        """
        attempt = {"requested_url": url, "timeout_seconds": self.timeout}
        self.last_fetch["attempts"].append(attempt)
        self.last_fetch["kind"] = "http"
        try:
            with self.client.stream("GET", url, headers={"Accept": "application/pdf,text/html;q=0.9",
                                                        "User-Agent": "ResearchLoop/AG2"},
                                    timeout=self.timeout, follow_redirects=True) as response:
                attempt.update(final_url=str(response.url), status=response.status_code,
                               content_type=response.headers.get("content-type", ""),
                               redirects=[{"url": str(r.url), "status": r.status_code} for r in response.history])
                self.last_fetch.update({key: value for key, value in attempt.items() if key != "redirects"})
                if response.status_code >= 300:
                    raise PaperAcquisitionError("PDF_HTTP_ERROR", f"PDF 获取返回 HTTP {response.status_code}",
                                                retryable=response.status_code >= 500, acquisition=self.last_fetch)
                chunks, size = [], 0
                # 在流读取过程中限制大小，避免响应全部进入内存后才发现超限。
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise PaperAcquisitionError("PDF_TOO_LARGE", f"PDF exceeds configured byte limit {self.max_bytes}", acquisition=self.last_fetch)
                    chunks.append(chunk)
                attempt["bytes"] = size
                return b"".join(chunks), str(response.url), response.headers.get("content-type", "")
        except httpx.HTTPError as exc:
            attempt["error"] = type(exc).__name__
            self.last_fetch["error"] = type(exc).__name__
            raise PaperAcquisitionError("PDF_NETWORK_ERROR", f"PDF 网络请求失败：{type(exc).__name__}",
                                        retryable=True, acquisition=self.last_fetch) from exc

    def _parse(self, path: Path) -> list[PageEvidence]:
        """逐页调用 get_text('text')，保留物理页；不额外运行 Markdown 转换。

        Args:
            path: 已验证文件头并缓存的 PDF。

        Returns:
            所有物理页的原始文本及图像存在性说明；不保证表格、公式和图意完整。
        """
        import pymupdf
        pages = []
        with pymupdf.open(path) as doc:
            for index, page in enumerate(doc):
                text = page.get_text("text")
                gaps = []
                if not text.strip():
                    gaps.append("page has no extractable text; no OCR or image interpretation performed")
                assets = [AssetReference(asset_id=f"image:{index+1}:{n}", page_number=index+1, kind="image")
                          for n, _ in enumerate(page.get_images(full=True))]
                if assets:
                    gaps.append("page contains images; text extraction does not establish image understanding")
                pages.append(PageEvidence(evidence_id="", document_id="", page_number=index+1,
                                          text=text, parsed_markdown=text, assets=assets, gaps=gaps))
        return pages


def is_usable_paper_text(text: str) -> bool:
    """沿用 1200 字符基本检查，拒绝代理错误页；这不是语义质量评分。"""
    value = str(text or "").strip()
    if value.startswith("%PDF-") or len(value) < 1200:
        return False
    head = value[:1000].lower()
    if re.search(r"warning: target url returned error|has been withdrawn and is unavailable", head):
        return False
    if re.search(r"<!doctype\s+html|<html\b|<body\b", head):
        return False
    if value.startswith(("{", "[")) and re.search(r'"(?:error|message|status)"\s*:', head):
        return False
    return not bool(re.match(r"(?:error|access denied|service unavailable|rate limit|upstream error)\b", head))
