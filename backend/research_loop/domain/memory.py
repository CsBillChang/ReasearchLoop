"""会话、个人记忆和笔记版本参数合同；Pydantic类说明不进入工具提示。"""
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


# 统一拒绝拼写错误字段，防止操作默默落到错误版本。
class MemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationCreate(MemoryInput):
    title: str = Field(default="新对话", min_length=1, max_length=200)
    idempotency_key: str = Field(min_length=1, max_length=120)


# 人工改名使用旧标题作并发条件；不更改会话消息和活跃时间。
class ConversationRename(MemoryInput):
    title: str = Field(min_length=1, max_length=200)
    expected_title: str = Field(min_length=1, max_length=200)
    idempotency_key: str = Field(min_length=1, max_length=120)

    @field_validator('title', mode='before')
    @classmethod
    def clean_title(cls, value):
        """折叠换行与首尾空白，避免空白标题和多行侧栏。"""
        return ' '.join(value.split()) if isinstance(value, str) else value


# 直接记录用户自然语言，不用生成模型猜测分类或新增保存授权。
class PreferenceStatement(MemoryInput):
    content: str = Field(min_length=1, max_length=1000)
    idempotency_key: str = Field(min_length=1, max_length=120)

    @field_validator('content', mode='before')
    @classmethod
    def clean_content(cls, value):
        """拒绝纯空白，保留内部换行、否定词及完整表达。"""
        return value.strip() if isinstance(value, str) else value


class RememberInput(MemoryInput):
    slot: str = Field(min_length=1, max_length=100, description="同一偏好的稳定键，如language或interest:目标检测；不得用于保存权限")
    kind: Literal["explicit", "inferred"]
    content: str = Field(min_length=1, max_length=1000)
    source_quote: str = Field(min_length=1, max_length=2000, description="当前真实用户消息或澄清中的逐字原句，不引用论文或模型输出")
    expected_version: int = Field(default=0, ge=0)


class ProfileSet(MemoryInput):
    slot: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=1000)
    expected_version: int = Field(default=0, ge=0)
    idempotency_key: str = Field(min_length=1, max_length=120)


class ForgetInput(MemoryInput):
    memory_id: str = Field(min_length=1, max_length=64)
    expected_version: int = Field(ge=1)


class RecallInput(MemoryInput):
    query: str = Field(min_length=1, max_length=500)
    source_type: Literal["research_history", "user_note", "all"] = "all"
    limit: int = Field(default=5, ge=1, le=10)


class HistoryRead(MemoryInput):
    source_type: Literal["research_history", "message", "conversation", "context_archive", "user_note"]
    source_id: str = Field(min_length=1, max_length=100)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=6000, ge=100, le=16000)


class NoteUpdate(MemoryInput):
    base_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    markdown: str = Field(min_length=1, max_length=200000)
    idempotency_key: str = Field(min_length=1, max_length=120)
    confirm: bool = False


class NoteToolUpdate(NoteUpdate):
    note_id: str = Field(min_length=1, max_length=64)


class ProfileRead(MemoryInput):
    pass
