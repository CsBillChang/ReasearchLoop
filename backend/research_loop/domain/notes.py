"""笔记数据契约：草稿、核验结论和文件保存结果相互独立，均以当前内容 hash 绑定。"""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


# 创建草稿的输入；evidence_ids 必须已返回给当前任务。
class CreateNoteInput(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    markdown: str = Field(min_length=1, max_length=100000)
    evidence_ids: list[str] = Field(default_factory=list)


# 保存指定版本的输入；hash 防止把旧操作应用到新稿。
class SaveNoteInput(BaseModel):
    draft_id: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


# 数据库草稿；status 不代替实际文件收据。
class Draft(BaseModel):
    draft_id: str
    task_id: str
    title: str
    markdown: str
    content_hash: str
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["draft", "partial", "ready", "pending_review"] = "draft"


# 单条审校判断：主张、支持/冲突/缺证、原因及原文定位。
class ReviewFinding(BaseModel):
    claim: str
    status: Literal["supported", "contradicted", "insufficient"]
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_quote: str | None = None
    # 历史审校记录可缺省；新审校入口使用更严格的 ClaimAssessment。
    basis: Literal["direct_statement", "not_found", "incomplete_or_ambiguous"] | None = None


# 当前稿的独立审校记录；结果与 draft_id/content_hash 绑定。
class ReviewResult(BaseModel):
    review_id: str
    task_id: str
    draft_id: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["supported", "issues", "insufficient", "not_run"]
    findings: list[ReviewFinding] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    model_requests: int = 0
    created_at: str | None = None


# 文件保存结果；file_sha256 可因待核实说明而不同于原稿 content_hash。
class SavedNote(BaseModel):
    draft_id: str
    path: str
    content_hash: str
    receipt_id: str
    file_sha256: str | None = None
    status: Literal["saved", "already_saved", "conflict", "pending_review"]
    review_status: str | None = None


# 一条精确局部修改，finding_index 为本次审校 findings 的零基索引。
class NotePatch(BaseModel):
    finding_index: int = Field(ge=0)
    old_text: str = Field(min_length=1, max_length=10000)
    new_text: str = Field(max_length=10000)
    reason: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list)


# 修订绑定当前审校、稿件及 hash，不接收完整替换稿或任意文件路径。
class RevisionInput(BaseModel):
    draft_id: str
    content_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    review_id: str
    patches: list[NotePatch] = Field(min_length=1, max_length=20)


# 网页显式操作的输入合同；不会自动提升旧任务的权限。
class SaveDraft(BaseModel):
    content_hash: str=Field(pattern=r'^[a-f0-9]{64}$')
    confirm: bool=False
