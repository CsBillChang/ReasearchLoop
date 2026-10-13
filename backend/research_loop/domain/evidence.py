"""原文数据契约：候选元数据、PDF 缓存身份与页级证据分开表达；有链接不等于已读取正文。"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


# 定向核验的判断与依据类型；没有找到支持不等于找到了反证。
class ClaimAssessment(BaseModel):
    claim: str = Field(min_length=1)
    status: Literal["supported", "contradicted", "insufficient"]
    basis: Literal["direct_statement", "not_found", "incomplete_or_ambiguous"]
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    source_quote: str | None = None

    @model_validator(mode="after")
    def validate_basis(self):
        """阻止以缺证作肯定/否定判断；引句的实际存在性由服务检查。"""
        if self.status != "insufficient" and (self.basis != "direct_statement" or
                not self.evidence_ids or not (self.source_quote or "").strip()):
            raise ValueError("supported/contradicted require direct_statement, source_quote and evidence_ids; absence or incomplete coverage means insufficient")
        return self


# 检索候选元数据；标题摘要和 PDF URL 都不等于已读正文。
class PaperCandidate(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    source: str
    published: str | None = None
    date_precision: Literal["day", "month", "year", "unknown"] = "unknown"
    paper_url: str | None = None
    pdf_url: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    doi: str | None = None
    source_id: str | None = None
    version: str | None = None


# 逐源诊断：成功非空、成功为空、未查、未配置和失败分别记录。
class SourceResult(BaseModel):
    source: str
    status: Literal["queried_success", "queried_empty", "not_queried", "not_configured", "failed", "partial_results"]
    message: str = ""
    candidate_count: int = 0
    request: dict[str, Any] = Field(default_factory=dict)


# 一次有界检索返回；truncated 和 query_scope 用于说明覆盖限制。
class SearchResult(BaseModel):
    candidates: list[PaperCandidate] = Field(default_factory=list)
    source_results: list[SourceResult] = Field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    truncated: bool = False
    query_scope: str = ""


# 历史引用统计结构；当前产品不再把被引量作为任务必须实现的能力。
class CitationStats(BaseModel):
    paper_id: str
    provider: str = "openalex"
    status: Literal["matched", "unmatched", "ambiguous", "unavailable", "not_queried"]
    count: int | None = None
    provider_id: str | None = None
    matched_by: Literal["doi", "title_authors_year", "none"] = "none"
    matched_title: str | None = None
    queried_at: str | None = None
    scope: str = "OpenAlex cited_by_count at query time"
    message: str = ""
    http_status: int | None = None
    retry_after_seconds: int | None = None
    rate_limit_remaining: int | None = None
    access_mode: Literal["api_key", "anonymous", "unknown"] = "unknown"


# 图像等资产定位；bbox 可缺失，资产记录不代表已理解图意。
class AssetReference(BaseModel):
    asset_id: str
    page_number: int
    kind: Literal["image", "drawing", "unknown"] = "unknown"
    bbox: list[float] | None = None
    note: str = "图片位置已保留；未代表模型已理解图意。"


# 一页原文证据：保存物理页号、原文、Markdown 和解析缺口。
class PageEvidence(BaseModel):
    evidence_id: str
    document_id: str
    page_number: int | None
    text: str
    parsed_markdown: str = ""
    assets: list[AssetReference] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


# 解析文档身份：PDF hash、解析器版本及配置共同决定缓存复用。
class Document(BaseModel):
    document_id: str
    paper_id: str
    content_sha256: str
    parser_name: str
    parser_version: str
    config_sha256: str
    total_pages: int | None
    cache_path: str
    gaps: list[str] = Field(default_factory=list)
