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


def valid_source_reference(value: dict) -> bool:
    """检查页来源完整版本合同；显示题录身份不参与内容位置推断。"""
    import re
    fields = {'evidence_id', 'document_id', 'paper_id', 'source_sha256', 'page_number', 'text_sha256'}
    if not isinstance(value, dict) or set(value) != fields:
        return False
    if any(not isinstance(value[key], str) or not value[key].strip() for key in fields - {'page_number'}):
        return False
    if any(re.fullmatch(r'[0-9a-f]{64}', value[key]) is None for key in ('source_sha256', 'text_sha256')):
        return False
    page = value['page_number']
    return page is None or (type(page) is int and page > 0)


def source_reference(page: dict, document: dict) -> dict:
    """从权威页和文档生成引用；不接受模型自报内容hash或读取权限。"""
    import hashlib
    value = {'evidence_id': page['evidence_id'], 'document_id': document['document_id'],
             'paper_id': document['paper_id'], 'source_sha256': document['content_sha256'],
             'page_number': page['page_number'] or None,
             'text_sha256': hashlib.sha256(page['text'].encode()).hexdigest()}
    if page['document_id'] != document['document_id'] or not valid_source_reference(value):
        raise ValueError('SOURCE_REFERENCE_INVALID')
    return value


def validate_source_window(item: dict, data: dict, page: dict, document: dict, *, kind: str = 'page') -> None:
    """验证实际返回的页或连续片段；缺字段和无版本窗口均拒绝。

    Args:
        item: 原始工具观察中的页或命中，压缩前逐字校验。
        data: 公共来源封装；页可继承content_sha256。
        page: 当前角色已授权的数据库页。
        document: 当前解析版本。
        kind: page为整页，hit为明确字符窗口。

    Raises:
        ValueError: 缺字段、位置越界、正文或hash不一致。
    """
    import hashlib
    if (kind not in {'page', 'hit'} or not isinstance(item, dict)
            or any(key not in item for key in ('evidence_id', 'document_id', 'page_number', 'text'))
            or any(item[key] != page[key] for key in ('evidence_id', 'document_id', 'page_number'))
            or type(item['page_number']) is not type(page['page_number']) or not isinstance(item['text'], str)):
        raise ValueError('SOURCE_WINDOW_INVALID')
    version = item.get('source_sha256') or data.get('content_sha256')
    versions = [value[key] for value, key in ((item, 'source_sha256'), (data, 'content_sha256')) if key in value]
    if not version or any(current != document['content_sha256'] for current in versions):
        raise ValueError('SOURCE_VERSION_CHANGED')
    text = page['text']
    if kind == 'hit':
        start, end = item.get('char_start'), item.get('char_end')
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError('SOURCE_WINDOW_INVALID')
        text = text[start:end]
    if text != item['text'] or ('text_sha256' in item and item['text_sha256'] != hashlib.sha256(text.encode()).hexdigest()):
        raise ValueError('SOURCE_TEXT_CHANGED')


def explicit_versioned_paper_locator(paper_id: str, metadata: dict) -> tuple | None:
    """仅用存储题录证明同一明确arXiv版本，保持348全部负例边界。"""
    import re
    from research_loop.domain.source_access import canonical_paper_url
    try:
        if not isinstance(metadata, dict) or metadata.get('paper_id') != paper_id:
            return None
        locators = []
        for field in ('paper_url', 'pdf_url'):
            url = metadata.get(field)
            if url is None or url == '':
                continue
            if not isinstance(url, str):
                return None
            locator = canonical_paper_url(url)
            if (not locator or locator[0] != 'arxiv' or not locator[2]
                    or re.fullmatch(r'v[1-9]\d*', locator[2]) is None):
                return None
            locators.append(locator)
        if not locators or len(set(locators)) != 1:
            return None
        # 可解析的题录ID也是身份约束，不能用v1链接掩盖显式v2或另一论文ID。
        identifier = canonical_paper_url('https://arxiv.org/abs/' + paper_id)
        if identifier and (identifier[:2] != locators[0][:2]
                           or (identifier[2] is not None and identifier[2] != locators[0][2])):
            return None
        version = metadata.get('version')
        if version is not None:
            if not isinstance(version, str) or re.fullmatch(r'v?[1-9]\d*', version) is None:
                return None
            if locators[0][2] != (version if version.startswith('v') else 'v' + version):
                return None
        return locators[0]
    except (ValueError, TypeError, KeyError):
        return None
