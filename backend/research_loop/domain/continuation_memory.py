"""实验接续记忆参数；候选由模型生成，来源与关系由程序另行校验。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from research_loop.domain.tasks import FinishTaskInput


class ContinuationInput(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


# 当前输入和本轮回答的身份由程序绑定；历史记录必须使用目录中的准确身份。
class ContinuationSource(ContinuationInput):
    source_type: Literal['current_user', 'current_answer', 'research_history']
    task_id: str | None = Field(default=None, max_length=100)
    quote: str = Field(min_length=1, max_length=2000)


class ContinuationConclusion(ContinuationInput):
    key: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_-]{0,39}$')
    text: str = Field(min_length=1, max_length=1500)
    conditions: list[str] = Field(default_factory=list, max_length=6)
    sources: list[ContinuationSource] = Field(min_length=1, max_length=6)


# 更新关系保留旧记录；关系存在不等于新说法已经过原文核验。
class ConclusionCorrection(ContinuationInput):
    previous_task_id: str = Field(min_length=1, max_length=100)
    previous_key: str = Field(min_length=1, max_length=40)
    previous_content_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    replacement_key: str = Field(min_length=1, max_length=40)
    relation: Literal['corrects', 'qualifies', 'updates']
    reason: str = Field(min_length=1, max_length=1000)


class ContinuationCandidate(ContinuationInput):
    research_goal: str = Field(min_length=1, max_length=2000)
    conclusions: list[ContinuationConclusion] = Field(default_factory=list, max_length=12)
    unresolved: list[str] = Field(default_factory=list, max_length=8)
    next_steps: list[str] = Field(default_factory=list, max_length=8)
    corrections: list[ConclusionCorrection] = Field(default_factory=list, max_length=12)


# 仅实验 ToolSpec 使用此类；坏旁路候选不阻挡既有回答交付。
class ContinuationFinishTaskInput(FinishTaskInput):
    continuation_memory: SkipValidation[ContinuationCandidate | None] = None
