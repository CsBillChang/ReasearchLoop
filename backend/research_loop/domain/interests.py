"""行为兴趣候选合同：模型理解主题，代码绑定当前真实用户来源。"""
from __future__ import annotations

import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def normalize_topic_key(value: str) -> str:
    """规范化主题键的字形、大小写和空白，不猜测领域或语义同义词。"""
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


class InterestCandidate(BaseModel):
    model_config = ConfigDict(extra='forbid')

    topic_key: str = Field(min_length=1, max_length=100, description='稳定主题键；已有同义主题沿用其键，不能改名恢复撤销主题')
    content: str = Field(min_length=1, max_length=1000, description='本次行为体现的研究兴趣线索，不声称已确认长期偏好')
    reason: str = Field(min_length=1, max_length=1000, description='说明提问、阅读选择或比较维度如何支持该线索，不引用模型推荐充当用户行为')
    source_quote: str = Field(min_length=1, max_length=2000, description='当前根用户问题或已回答澄清中的逐字片段；不引用旧消息、论文或模型输出')
    merge_memory_id: str | None = Field(default=None, max_length=64, description='合并已有兴趣时使用上下文中的真实inferred memory_id')
    expected_version: int = Field(default=0, ge=0)

    @field_validator('topic_key', 'content', 'reason', 'source_quote')
    @classmethod
    def nonblank(cls, value: str) -> str:
        """拒绝纯空白；来源引句保留原字词，只清理边界空白。"""
        value = value.strip()
        if not value:
            raise ValueError('interest fields must not be blank')
        return value

    @model_validator(mode='after')
    def merge_version(self):
        """显式合并必须绑定版本，新增不能自报一个未读取的版本。"""
        if bool(self.merge_memory_id) != bool(self.expected_version):
            raise ValueError('merge_memory_id and positive expected_version must be provided together')
        return self
