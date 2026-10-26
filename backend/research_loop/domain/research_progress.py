"""阶段研究进展合同：候选发现与原文身份分开，结构合法不代表语义通过。"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# 只要求模型给出发现及已有证据，版本、位置、核验状态由业务层派生。
class ProgressFinding(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    finding_id: str | None = Field(default=None, description=(
        "Omit for a new finding; the system assigns its ID. "
        "To update, copy an existing finding_id from this task's current research progress. "
        "Do not invent an ID or reuse one from another task."))
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


# None 表示沿用上次缺口，空列表表示明确清空；不能因省略字段而丢掉工作。
class ResearchProgressInput(BaseModel):
    findings: list[ProgressFinding] = Field(default_factory=list)
    gaps: list[str] | None = None
    next_question: str | None = None
    base_revision: int | None = Field(default=None, ge=0)
