"""原文索引契约：身份和定位先于检索评分；不授予论文或页级读取权限。"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Agent与服务共用输入；document_id就是不可变的解析版本，不自动选择同论文的其他版本。
class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=2, max_length=1000)
    document_ids: list[str] = Field(min_length=1, max_length=20)
    limit: int = Field(default=5, ge=1, le=8)
    candidate_limit: int = Field(default=24, ge=8, le=64)

    @field_validator("document_ids")
    @classmethod
    def unique_versions(cls, values: list[str]) -> list[str]:
        """拒绝空身份与重复版本，避免扩大或重复计算检索范围。"""
        if any(not v.strip() or len(v) > 255 for v in values) or len(set(values)) != len(values):
            raise ValueError("RETRIEVAL_DOCUMENT_IDS_INVALID")
        return values


def fingerprint(value: object) -> str:
    """对稳定JSON或原始字符串计算完整hash，用于内容与配置版本身份。"""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 内部契约并不作为Agent工具schema暴露，拒绝拼写错误和隐含的多余配置。
class ChunkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str = "paragraph-offset-v1"
    max_tokens: int = Field(default=448, ge=16, le=448)


class SourceChunk(BaseModel):
    """绑定一份原文快照的连续字符片段；char_end为不包含的右边界。"""
    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk_id: str
    document_id: str
    evidence_id: str
    source_sha256: str
    page_number: int | None
    char_start: int
    char_end: int
    text: str
    text_sha256: str
    section: str | None = None
    section_origin: str = "unavailable"


_HEADING = re.compile(
    r"^(?:#{1,6}\s+[^\n]{1,100}|(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:Introduction|Related Work|Methods?|Methodology|Experiments?|Results|"
    r"Conclusions?|Limitations|References|Appendix))$", re.IGNORECASE)


def split_page(page: dict, source: dict, config: ChunkConfig, tokenizer_id: str,
               offsets: Callable[[str], list[tuple[int, int]]]) -> list[SourceChunk]:
    """按段落及token边界切分实际页文字，不改写公式或虚构章节。

    Args:
        page: 已校验的页级证据；未知物理页号为None。
        source: document_id、content_sha256和解析器身份。
        config: 冻结切分配置。
        tokenizer_id: tokenizer内容身份，防止换分词器混用旧chunk。
        offsets: 返回原字符串中的token字符区间，不得截断输入。

    Returns:
        连续原文片段；所有非空白文字均被覆盖，章节不可靠时为None。

    Raises:
        ValueError: 分词器无法覆盖非空原文或返回非法区间。
    """
    text = page["text"]
    # 单独的明确标题成为分段边界；普通短行不是标题，避免把表格行当章节。
    boundaries = {0, len(text)}
    headings = {}
    for match in re.finditer(r"[^\n]+", text):
        title = match.group().strip()
        if _HEADING.fullmatch(title):
            boundaries.update((match.start(), match.end()))
            headings[match.start()] = title.lstrip("# ")
    for match in re.finditer(r"\n\s*\n", text):
        boundaries.update((match.start(), match.end()))
    cuts = sorted(boundaries)
    section = None
    result = []
    for left, right in zip(cuts, cuts[1:]):
        if left in headings:
            section = headings[left]
        if not text[left:right].strip():
            continue
        positions = offsets(text[left:right])
        if not positions or any(a < 0 or b < a or b > right-left for a, b in positions):
            raise ValueError("TOKEN_OFFSETS_INVALID")
        # 以后一组首token起点切分；即使tokenizer忽略空白，也保留完整原文子串。
        starts = [left]
        for n in range(config.max_tokens, len(positions), config.max_tokens):
            cut = left + positions[n][0]
            if cut > starts[-1]:
                starts.append(cut)
        starts.append(right)
        for start, end in zip(starts, starts[1:]):
            body = text[start:end]
            if not body.strip():
                continue
            identity = {"source": source, "evidence_id": page["evidence_id"],
                        "page_hash": fingerprint(text), "config": config.model_dump(),
                        "tokenizer": tokenizer_id, "start": start, "end": end}
            result.append(SourceChunk(chunk_id=fingerprint(identity), document_id=source["document_id"],
                evidence_id=page["evidence_id"], source_sha256=source["content_sha256"],
                page_number=page.get("page_number") or None, char_start=start, char_end=end,
                text=body, text_sha256=fingerprint(body), section=section,
                section_origin="explicit_heading" if section else "unavailable"))
    return result
