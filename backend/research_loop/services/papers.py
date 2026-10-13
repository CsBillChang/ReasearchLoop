"""论文服务：串接检索、授权、PDF 解析与页证据持久化；只向任务返回它有权读取的内容。"""
from __future__ import annotations

import json
from typing import Any

from research_loop.domain.evidence import CitationStats, Document, PaperCandidate, SearchResult
from research_loop.domain.reading import UserPageRange, user_page_range
from research_loop.providers.discovery import DiscoveryProvider
from research_loop.providers.dpr_acquisition import PaperAcquisitionError, resolve_pdf_url
from research_loop.providers.pdf import PDFProvider
from research_loop.storage.mysql import Database


class PaperService:
    def __init__(self, db: Database, discovery: DiscoveryProvider, pdf: PDFProvider, citations: Any = None):
        self.db = db
        self.discovery = discovery
        self.pdf = pdf
        self.citations = citations
        self.candidates: dict[str, PaperCandidate] = {}

    def search(self, query: str, date_from: str | None = None, date_to: str | None = None,
               sources: list[str] | None = None, limit: int = 5, task_id: str | None = None) -> SearchResult:
        """输入查询和筛选，输出候选及逐源状态；把返回候选存入数据库并授予当前任务读取权限。"""
        result = self.discovery.search(query, date_from, date_to, sources, limit)
        for candidate in result.candidates:
            self.candidates[candidate.paper_id] = candidate
            self.db.save_paper(candidate.paper_id, candidate.model_dump(mode="json"))
            if task_id:
                self.db.grant_paper(task_id, candidate.paper_id, task_id)
        return result

    def citation_stats(self, task_id: str, paper_ids: list[str]) -> list[CitationStats]:
        if not self.citations:
            return [CitationStats(paper_id=paper_id, status="unavailable", message="OpenAlex not configured")
                    for paper_id in paper_ids]
        output = []
        for paper_id in paper_ids:
            if not self.db.task_has_paper(task_id, paper_id):
                output.append(CitationStats(paper_id=paper_id, status="unavailable", message="paper is not authorized"))
                continue
            row = self.db.get_paper(paper_id)
            if not row:
                output.append(CitationStats(paper_id=paper_id, status="unmatched", message="paper metadata missing"))
                continue
            stat = self.citations.lookup(PaperCandidate.model_validate(row["metadata"]))
            self.db.save_citation_stats(stat.model_dump(mode="json"))
            output.append(stat)
        return output

    def user_scope(self, task_id: str) -> UserPageRange | None:
        """读取根任务中的用户限制，使三个角色遵守相同边界。

        Args:
            task_id: 当前根任务或子任务 ID。

        Returns:
            用户明确限定的连续物理页范围，没有限制时返回 None。

        Raises:
            ValueError: 用户限制有歧义，需要澄清。
        """
        task = self.db.get_task(task_id)
        if not task:
            return None
        root_id = task.get("root_task_id") or task_id
        root = self.db.get_task(root_id) or task
        answers = self.db.fetchall(
            "SELECT answer FROM clarifications WHERE task_id=? AND status='answered' ORDER BY answered_at,clarification_id",
            (root_id,),
        )
        return user_page_range([root["prompt"]] + [row["answer"] for row in answers if row["answer"]])

    def read(self, task_id: str, target_ref: str, start_page: int = 1, page_count: int | None = None,
             attachments: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        """读取授权论文，默认向调用方返回完整的逐页解析正文。

        Args:
            task_id: 接收证据的任务 ID。
            target_ref: 已授权的 paper_id 或附件 input_ref。
            start_page: 本次读取的物理起始页，默认第一页。
            page_count: 可选片段长度；省略时读到末页或用户指定边界。
            attachments: 后端已绑定的附件信息。

        Returns:
            页级正文、源 hash、实际覆盖范围及解析缺口；缓存完整和本次返回完整分别记录。

        Raises:
            PermissionError: 论文未授权或请求超出用户明确限定页范围。
            PaperAcquisitionError: 论文身份不存在、获取地址缺失或 PDF 获取/解析失败。
            ValueError: 页码非法或用户范围存在歧义。
        """
        user_scope = self.user_scope(task_id)
        if user_scope:
            if start_page == 1 and page_count is None:
                start_page = user_scope.start
            if page_count is None:
                page_count = user_scope.end - start_page + 1
            if start_page < user_scope.start or page_count < 1 or start_page + page_count - 1 > user_scope.end:
                raise PermissionError("requested pages exceed the user-specified physical page range")
        attached = attachments and attachments.get(target_ref)
        attached_authorized = bool(attached and (self.db.task_has_input(task_id, target_ref)
                                                 or not self.db.get_paper(target_ref)))
        candidate = None if attached_authorized else self.candidates.get(target_ref)
        if candidate and not self.db.task_has_paper(task_id, target_ref):
            raise PermissionError("paper is not authorized for this task")
        if not candidate:
            stored = self.db.get_paper(target_ref)
            if stored:
                if not self.db.task_has_paper(task_id, target_ref):
                    raise PermissionError("paper is not authorized for this task")
                candidate = PaperCandidate.model_validate(stored["metadata"])
        source = None
        paper_id = target_ref
        if candidate:
            # DPR 的显式 PDF 优先及来源转换规则；合法详情页不再被误报为非法身份。
            source = resolve_pdf_url(candidate.pdf_url, candidate.paper_url)
        elif attached_authorized:
            if not self.db.task_has_input(task_id, target_ref):
                self.db.grant_input(task_id, target_ref, task_id)
            source = attached.get("local_path") or attached.get("url")
            paper_id = attached.get("paper_id") or target_ref
        if not source:
            if candidate or attached_authorized:
                raise PaperAcquisitionError("PDF_URL_MISSING", "论文/附件身份有效，但未提供 PDF 或详情页地址")
            raise PaperAcquisitionError("PAPER_REFERENCE_INVALID", "target_ref is not a returned paper_id or an authorized input_ref")
        page_request = user_scope is not None or start_page != 1 or page_count is not None
        ingest = getattr(self.pdf, "ingest_pages", self.pdf.ingest) if page_request else self.pdf.ingest
        document = ingest(source, paper_id)
        self.db.save_document(document.model_dump(mode="json"))
        count = page_count if page_count is not None else (document.total_pages - start_page + 1 if document.total_pages else None)
        pages = self.pdf.pages(document, start_page, count)
        for page in pages:
            self.db.save_page(page.model_dump(mode="json"))
            self.db.mark_task_evidence(task_id, page.evidence_id)
        next_page = pages[-1].page_number + 1 if pages and document.total_pages and pages[-1].page_number < document.total_pages else None
        return {
            "paper_id": paper_id,
            "target_ref": target_ref,
            "document_id": document.document_id,
            "content_sha256": document.content_sha256,
            "total_pages": document.total_pages,
            "returned_pages": [page.page_number for page in pages],
            "pages": [page.model_dump(mode="json") for page in pages],
            "next_page": next_page,
            "acquisition": dict(getattr(self.pdf, "last_fetch", {})),
            "parse_coverage": {"total_pages": document.total_pages, "cached": True},
            "read_coverage": {
                "returned_pages": [page.page_number for page in pages],
                "full_document_returned": ([page.page_number for page in pages] == list(range(1, document.total_pages + 1))) if document.total_pages else None,
                "all_extracted_text_returned": True if not document.total_pages else count == document.total_pages,
                "locator_kind": "physical_page" if document.total_pages else "document_text",
                "scope_origin": "user" if user_scope else "full_document",
                "user_page_range": {"start": user_scope.start, "end": user_scope.end,
                                    "instruction": user_scope.instruction} if user_scope else None,
            },
            "known_gaps": document.gaps + sorted({gap for page in pages for gap in page.gaps}),
        }

    def evidence_bundle(self, task_id: str, evidence_ids: list[str]) -> list[dict[str, Any]]:
        """组装授权原文及解析限制；仅选中的页进入包，不宣称整份 PDF 已理解。"""
        output = []
        documents = {}
        attachments = json.loads((self.db.get_task(task_id) or {}).get("attachments_json") or "[]")
        for ref in dict.fromkeys(evidence_ids):
            if not self.db.task_has_evidence(task_id, ref):
                raise PermissionError("evidence was not granted to task")
            page = self.db.get_page(ref)
            if not page:
                raise ValueError("evidence page is missing")
            document = self.db.fetchone("SELECT * FROM documents WHERE document_id=?", (page["document_id"],))
            if not document:
                raise ValueError("evidence document is missing")
            paper_id = document["paper_id"]
            documents[page["document_id"]] = document
            target = next((a["input_ref"] for a in attachments if (a.get("paper_id") or a["input_ref"]) == paper_id), paper_id)
            output.append({"evidence_id": ref, "paper_id": paper_id, "target_ref": target,
                           "document_id": page["document_id"], "source_sha256": document["content_sha256"],
                           "page_number": page["page_number"], "text": page["text"],
                           "parsed_markdown": page.get("parsed_markdown", ""),
                           "assets": json.loads(page.get("assets_json") or "[]"),
                           "gaps": json.loads(document.get("gaps_json") or "[]") + json.loads(page.get("gaps_json") or "[]")})
        # 授权池是当前角色累计取得/继承的证据，选中包只是本次核验输入，二者不能互相替代。
        available = {}
        for ref in self.db.list_task_evidence(task_id):
            if evidence := self.db.get_page(ref):
                available.setdefault(evidence['document_id'], set()).add(evidence['page_number'])
        for page in output:
            doc = documents[page["document_id"]]
            selected = sorted({p["page_number"] for p in output if p["document_id"] == page["document_id"] and p["page_number"] is not None})
            total = doc["total_pages"] or None
            page["read_scope"] = {"total_physical_pages": total, "included_physical_pages": selected,
                "all_physical_pages_included": selected == list(range(1, total + 1)) if total else None,
                "locator_kind": "physical_page" if total else "document_text",
                "visual_content_interpreted": False,
                "scope_kind": "selected_evidence_bundle",
                "limitation": "这是本次选入核验包的页，不是任务累计阅读范围；含全部页也不代表图片、图表、公式完整可读。"}
            authorized = sorted(n for n in available.get(page['document_id'], set()) if n is not None)
            page['task_evidence_scope'] = {'authorized_physical_pages': authorized,
                'all_physical_pages_available': authorized == list(range(1,total+1)) if total else None,
                'provenance': 'persisted_task_evidence',
                'limitation': '任务已取得或从子任务继承的文字证据；不等于每个Agent均阅读过或理解所有图表。'}
        return output

    def evidence_scope(self, task_id: str) -> list[dict[str, Any]]:
        """从真实授权页汇总交接范围，不采用模型自报的全文/图表理解程度。"""
        docs = {}
        for page in self.evidence_bundle(task_id, self.db.list_task_evidence(task_id)):
            item = docs.setdefault(page["document_id"], {"document_id": page["document_id"],
                "source_sha256": page["source_sha256"], **page["read_scope"], "gaps": [], "asset_pages": []})
            item["gaps"] = list(dict.fromkeys(item["gaps"] + page["gaps"]))
            if page["assets"] and page["page_number"] not in item["asset_pages"]:
                item["asset_pages"].append(page["page_number"])
        return list(docs.values())

    def inspect_media(self, task_id: str, target_ref: str, start_page: int, page_count: int | None, attachments: dict) -> dict:
        """复用授权读取取得物理 PDF，再调用 DPR 图表适配；无视觉模型时如实返回限制。"""
        from research_loop.providers.paper_media import PaperMediaProvider
        # 显式起始范围触发 PDF 页级路径，不能把 Jina 文本 hash 当 PDF hash。
        scope = self.user_scope(task_id)
        start = scope.start if scope and start_page == 1 else start_page
        first = self.read(task_id, target_ref, start, 1, attachments)
        total = first['total_pages']
        end = start + page_count - 1 if page_count is not None else (scope.end if scope else total)
        if not total or end > total or end < start or (scope and end > scope.end):
            raise PermissionError('media pages exceed PDF or user scope')
        files = self.pdf.file_store
        result = PaperMediaProvider(files).extract(files.cache_pdf_path(first['content_sha256']), first['content_sha256'], list(range(start, end + 1)))
        return {**result, 'target_ref': target_ref, 'document_id': first['document_id'],
                'gaps': ['图表文件已提取，但当前文字模型没有接收图片；不能宣称视觉核验通过。']}
