"""按需核验合同：对象快照、依据层级、局部修改和明确结束分开表达。"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


def fingerprint(value) -> str:
    """返回规范化数据身份；身份变化不表示语义得到修复。"""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def target_fingerprint(value: dict) -> str:
    """以真实内容/来源版本标识对象，排除行动ID、进展计数等审计包装。"""
    items = []
    for item in value['items']:
        if value['target_type'] == 'candidates':
            items.append({'item_id': item['item_id'], 'metadata': item['metadata']})
        else:
            items.append({'item_id': item['item_id'], 'text': item['text'],
                'sources': sorted(({k: r[k] for k in ('source_sha256', 'page_number', 'text_sha256')}
                                   for r in item.get('sources', [])), key=fingerprint)})
    return fingerprint([value['target_type'], value['source_task_id'], sorted(items, key=lambda i: i['item_id']),
                        value.get('requirements', {})])


# 新对象从权威成功行动加载；继续处理必须绑定工具返回的target_id/hash。
class ResearchReviewInput(BaseModel):
    target_type: Literal['candidates', 'findings', 'answer']
    source_task_id: str | None = None
    item_ids: list[str] = Field(default_factory=list)
    answer_markdown: str | None = Field(default=None, max_length=80000)
    target_id: str | None = None
    content_hash: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    refresh_source: bool = False
    reason: str = Field(min_length=1, max_length=2000)
    verification_questions: list[str] = Field(min_length=1, max_length=10)
    evidence_ids: list[str] = Field(default_factory=list)
    goal_ids: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def check_identity(self):
        """后续操作必须带精确版本；空问题不能用于重置处理身份。"""
        if bool(self.target_id) != bool(self.content_hash):
            raise ValueError('target_id and content_hash must be supplied together')
        if any(not q.strip() for q in self.verification_questions):
            raise ValueError('verification_questions cannot be blank')
        return self


# 元数据判断必须点明实际字段；全文判断必须给已授权引句。
class TargetAssessment(BaseModel):
    # 与执行校验相同：两个判断字段必须且仅一个非空；允许旧合法响应携带另一空字段。
    model_config = ConfigDict(json_schema_extra={'oneOf': [
        {'required': ['status'], 'properties': {
            'status': {'type': 'string', 'enum': ['supported', 'contradicted', 'insufficient']},
            'suitability': {'type': 'null'}}},
        {'required': ['suitability'], 'properties': {
            'suitability': {'type': 'string', 'enum': ['suitable', 'unsuitable', 'uncertain']},
            'status': {'type': 'null'}}},
    ]})
    item_id: str = Field(min_length=1)
    status: Literal['supported', 'contradicted', 'insufficient'] | None = Field(default=None,
        description='findings/answer必填非空判断；范围披露也必须明确判断，不能由basis/reason/action代替。')
    suitability: Literal['suitable', 'unsuitable', 'uncertain'] | None = Field(default=None,
        description='candidates必填非空适用性判断；与status必须且仅一个非空。')
    basis: Literal['direct_statement', 'not_found', 'incomplete_or_ambiguous', 'metadata', 'scope_disclosure']
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    source_quote: str | None = None
    metadata_fields: list[str] = Field(default_factory=list)
    suggested_action: Literal['keep', 'supplement', 'revise', 'exclude', 'end']

    @model_validator(mode='after')
    def check_basis(self):
        """严格分离适用性与事实核验；缺证不能构造肯定或否定结论。"""
        if (self.status is None) == (self.suitability is None):
            raise ValueError('use exactly one non-null verdict: status for findings/answer, suitability for candidates; '
                             'reason, basis and suggested_action do not supply a verdict')
        certain = self.status in {'supported', 'contradicted'}
        if certain and self.basis != 'scope_disclosure' and (self.basis != 'direct_statement' or not self.source_quote or not self.evidence_ids):
            raise ValueError('factual verdict requires direct source quote and evidence_ids')
        if self.basis == 'scope_disclosure' and (self.status != 'supported' or not self.evidence_ids):
            raise ValueError('scope_disclosure supports an accurate limitation with actual evidence scope only')
        if self.status == 'insufficient' and self.basis not in {'not_found', 'incomplete_or_ambiguous'}:
            raise ValueError('insufficient requires an absence or coverage limitation')
        if self.suitability and self.basis != 'metadata':
            raise ValueError('candidate suitability must cite actual metadata fields')
        return self


# 每个对象至少一条意见；同对象多个问题可分别判断，不能用支持意见覆盖缺证。
class FinishResearchReview(BaseModel):
    findings: list[TargetAssessment] = Field(min_length=1, max_length=30)
    gaps: list[str] = Field(default_factory=list, max_length=30)

    @staticmethod
    def describe_validation_error(error: ValidationError, arguments: dict) -> str:
        """将原校验位置与对象身份交回Reviewer，不补全或删除任何意见。

        Args:
            error: 参数模型的真实校验失败。
            arguments: 模型提交的原始参数，仅用于报告位置。

        Returns:
            具体意见下标、item_id及原校验原因；整批仍未接收。
        """
        details = []
        rows = arguments.get('findings', []) if isinstance(arguments, dict) else []
        for item in error.errors(include_url=False):
            location = item['loc']
            path = ''.join(f'[{part}]' if isinstance(part, int) else ('.' if i else '') + str(part)
                           for i, part in enumerate(location))
            identity = ''
            if (len(location) > 1 and location[0] == 'findings' and isinstance(location[1], int)
                    and isinstance(rows, list) and location[1] < len(rows) and isinstance(rows[location[1]], dict)):
                identity = f" item_id={rows[location[1]].get('item_id')!r}"
            details.append(f"{path}{identity}: {item['msg']}")
        return 'finish_review未接收整批意见；请明确填写判断，不根据其他字段推断。' + '；'.join(details)


# 精确patch引用原句和稳定item_id，不覆盖整个研究对象。
class ResearchPatch(BaseModel):
    item_id: str
    old_text: str = Field(min_length=1, max_length=10000)
    new_text: str = Field(max_length=10000)
    reason: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list)


class ResearchRevisionInput(BaseModel):
    target_id: str
    content_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    review_id: str
    patches: list[ResearchPatch] = Field(min_length=1, max_length=20)


# 结束是Supervisor的处理选择，不会把Reviewer改成supported。
class CloseResearchIssueInput(BaseModel):
    target_id: str
    content_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    review_id: str
    action: Literal['exclude', 'disclose', 'end']
    reason: str = Field(min_length=1, max_length=2000)


def assessment_state(item: dict) -> str:
    """将两类判断转换为进展比较标签，仍保留原始判断字段。"""
    return item.get('status') or item.get('suitability') or 'unreviewed'


def review_progress(previous: dict | None, current: dict) -> dict:
    """按稳定项目比较核验结果，不以意见总数减少宣称解决问题。

    Args:
        previous: 同一对象上一次已结算的逐项判断。
        current: 当前版本的独立判断。

    Returns:
        实际解决的项目、取得反证的项目及保留问题；不是准确率估计。
    """
    def states(result):
        """保留同对象的全部状态；重复意见和排列变化不制造进展。"""
        grouped = {}
        for finding in (result or {}).get('findings', []):
            grouped.setdefault(finding['item_id'], set()).add(assessment_state(finding))
        return grouped

    old, new = states(previous), states(current)
    resolved, clarified = [], []
    clear = {'supported', 'suitable'}
    for key, values in new.items():
        before = old.get(key, set())
        if before.intersection({'contradicted', 'insufficient', 'uncertain', 'unsuitable'}) and values <= clear:
            resolved.append(key)
        if (before.intersection({'insufficient', 'uncertain'}) and before <= clear | {'insufficient', 'uncertain'}
                and values.intersection({'contradicted', 'unsuitable'}) and values <= clear | {'contradicted', 'unsuitable'}):
            clarified.append(key)
    return {'resolved_item_ids': resolved, 'clarified_item_ids': clarified,
            'effective_change': bool(resolved or clarified), 'semantic_guarantee': False}
