"""任务数据契约：区分运行状态、保存权限、工具观察和模型收据，供 API、服务与 Agent 共用。"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, SkipValidation, field_validator, model_validator

from research_loop.domain.interests import InterestCandidate


# 任务生命周期：partial 表示有限交付，不可直接当作完整成功。
class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    partial = "partial"
    failed = "failed"
    cancelled = "cancelled"
    needs_input = "needs_input"


# 文件保存授权；deny 是后端约束，不能靠提示词或模型意愿绕过。
class SavePolicy(str, Enum):
    allow = "allow"
    deny = "deny"


# 当前三个协作角色；具体可用工具由注册表限制。
class Role(str, Enum):
    supervisor = "supervisor"
    researcher = "researcher"
    reviewer = "reviewer"


# 可观察目标；语义相关性由 Agent 判断，程序只验证来源、数量、实际读取与身份。
class ResearchGoal(BaseModel):
    goal_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,40}$")
    description: str = Field(min_length=1, max_length=1000)
    # legacy仅兼容无归属的旧计划；不从描述猜测旧目标是否可撤回。
    origin: Literal["user", "strategy", "legacy"] = "legacy"
    status: Literal["active", "withdrawn"] = "active"
    # user目标引用真实用户原句；程序只核对出处，不证明语义归属正确。
    source_quote: str | None = Field(default=None, max_length=12000)
    change_reason: str | None = Field(default=None, max_length=2000)
    min_papers: int = Field(default=1, ge=0, le=20)
    requires_fulltext: bool = False
    # 目标内来源是本目标资格条件；真实用户硬限制须放在origin=user目标。
    sources: list[str] = Field(default_factory=list)
    input_refs: list[str] = Field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None

    @field_validator("date_from", "date_to")
    @classmethod
    def validate_date(cls, value):
        """计划中的日期用明确的ISO日历日期，拒绝模糊年份字符串。"""
        if value is not None:
            datetime.strptime(value, "%Y-%m-%d")
        return value

    @model_validator(mode="after")
    def ordered_dates(self):
        """核对日期和目标归属；撤回只适用于有理由的自拟步骤。"""
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not exceed date_to")
        if self.origin == "user" and not (self.source_quote and self.source_quote.strip()):
            raise ValueError("user goal requires source_quote from the actual user request")
        if self.status == "withdrawn" and (self.origin != "strategy" or not (self.change_reason and self.change_reason.strip())):
            raise ValueError("only strategy goals may be withdrawn, with change_reason")
        return self


# 研究目标结构；计划经工具观察和checkpoint持久化，区别于长期Memory。
class ResearchBrief(BaseModel):
    goal: str = Field(min_length=1, max_length=12000)
    requested_count: int | None = Field(default=None, ge=1, le=20)
    count_confirmed: bool = False
    suggested_count: int = Field(default=5, ge=1, le=20)
    # 新建计划顶层来源/日期仅检索偏好；无服务器语义标记的历史计划保留原硬范围。
    date_from: str | None = None
    date_to: str | None = None
    sources: list[str] = Field(default_factory=list)
    input_refs: list[str] = Field(default_factory=list)
    comparison_dimensions: list[str] = Field(default_factory=list)
    save_policy: SavePolicy = SavePolicy.deny
    # memory也用于仅依当前对话给定材料的零论文讨论；论文问答仍沿研究材料合同。
    task_kind: Literal["search", "compare", "survey", "memory"] = "search"
    goals: list[ResearchGoal] = Field(default_factory=list, max_length=12)
    save_requested: bool = False
    deep_reading_requested: bool = Field(default=False, description="True only when the user requests a deep-reading note or eight-point summary. Full-text reading, comparison, answers, and saving a short research note do not imply this deliverable.")
    evidence_matrix_requested: bool = Field(default=False, description="True only when this plan includes a structured evidence matrix. Selects non-thinking generation after source evidence is available; does not require a matrix for ordinary comparisons or change save permission.")

    @field_validator("date_from", "date_to")
    @classmethod
    def validate_date(cls, value):
        """根任务时间窗使用与子目标相同的日历日期合同。"""
        return ResearchGoal.validate_date(value)

    @model_validator(mode="after")
    def unique_goals(self):
        """目标身份不得重复；历史未使用目标清单的ResearchBrief仍可读取。"""
        ids = [goal.goal_id for goal in self.goals]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate goal_id")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not exceed date_to")
        return self


# 最终逐目标判断；子任务身份必须来自本次根任务，未解决项必须说明原因。
class GoalDelivery(BaseModel):
    goal_id: str
    status: Literal["satisfied", "unresolved"]
    subtask_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=2000)


# 根任务结束：可使用已存在的草稿，避免模型重复传输/改写正文。
class FinishTaskInput(BaseModel):
    answer_markdown: str | None = Field(default=None, max_length=80000)
    # 处理说明与正文分离；绑定快照时省略answer_markdown，勿追加说明后重传原文。
    processing_summary: str | None = Field(default=None, max_length=80000)
    draft_id: str | None = None
    reflection_target_id: str | None = None
    reflection_content_hash: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    matrix_hash: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    goals: list[GoalDelivery] = Field(min_length=1, max_length=12)
    # 整个旁路容错，坏画像候选不得阻挡研究交付；终态后独立逐项校验。
    interest_candidates: SkipValidation[list[InterestCandidate]] = Field(default_factory=list)


# 持久化澄清问题：回答绑定 clarification_id，避免恢复到错误暂停点。
class Clarification(BaseModel):
    clarification_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    question: str
    fields: list[str] = Field(default_factory=list)
    suggested_values: list[str] = Field(default_factory=list)
    status: Literal["pending", "answered", "expired"] = "pending"
    answer: str | None = None


# 恢复输入：问题 ID、用户回答和幂等键缺一不可。
class ResumeInput(BaseModel):
    clarification_id: str
    answer: str = Field(min_length=1, max_length=4000)
    idempotency_key: str = Field(min_length=1, max_length=200)


# 子任务目标与论文授权；start_page/page_count 仅为阅读建议，不能制造用户未指定的页权限。
class SubtaskSpec(BaseModel):
    objective: str = Field(min_length=1, max_length=4000)
    paper_ids: list[str] = Field(default_factory=list)
    input_refs: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    start_page: int | None = Field(default=None, ge=1)
    page_count: int | None = Field(default=None, ge=1)
    comparison_dimensions: list[str] = Field(default_factory=list)
    position: int = Field(default=0, ge=0)
    mode: Literal["screen", "investigate", "deep_read"] = Field(default="investigate", description="screen for search; investigate for questions/comparison; deep_read only for a user-requested deep-reading note or eight-point summary.")
    goal_ids: list[str] = Field(default_factory=list, max_length=12)
    verification_claims: list[str] = Field(default_factory=list, max_length=10, description="Specific claims to verify, not assumed true. In investigate mode return an assessment for each; in screen/deep_read they are optional follow-up questions, not a scientific-verification completion gate. Leave empty for ordinary search/comparison.")


# 子任务交付：发现、证据和缺口并存；failed/partial 不冒充完整成功。
class SubtaskResult(BaseModel):
    subtask_id: str
    parent_task_id: str
    objective: str
    status: Literal["queued", "running", "completed", "partial", "failed", "cancelled"]
    findings: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    error: str | None = None
    role: str = "researcher"
    review_status: str | None = None
    review_result: dict[str, Any] | None = None
    mode: str = "deep_read"
    goal_ids: list[str] = Field(default_factory=list)
    deep_reading_status: str | None = None
    assessments: list[dict[str, Any]] = Field(default_factory=list)
    evidence_scope: list[dict[str, Any]] = Field(default_factory=list)


# 附件由 API 绑定本地路径或 URL；Agent 使用 input_ref，不能任意指定服务器文件。
class AttachmentInput(BaseModel):
    input_ref: str = Field(min_length=1, max_length=200)
    local_path: str | None = None
    url: str | None = None
    filename: str | None = None
    # 可选的固定论文身份由服务端绑定到合法附件；模型仍通过 input_ref 读取。
    paper_id: str | None = None

    @field_validator("local_path", "url", "filename", mode="before")
    @classmethod
    def blank_is_none(cls, value: Any) -> Any:
        return value or None


# 自然语言任务提交合同；默认禁止保存，调用方须显式表达写入授权。
class TaskCreate(BaseModel):
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)
    prompt: str = Field(min_length=1, max_length=12000)
    idempotency_key: str | None = Field(default=None, max_length=200)
    attachments: list[AttachmentInput] = Field(default_factory=list)
    save_policy: SavePolicy = SavePolicy.deny

    @field_validator("idempotency_key")
    @classmethod
    def normalize_key(cls, value: str | None) -> str | None:
        return value.strip() if value else None


# 用户结果：正文、草稿、文件路径与缺口独立；有 draft_id 不等于 answer_markdown 非空。
class TaskResult(BaseModel):
    evidence_matrix: dict[str, Any] | None = None
    answer_kind: str = "research_result"
    note_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reflection: dict[str, Any] = Field(default_factory=dict)
    research_reviews: list[dict[str, Any]] = Field(default_factory=list)
    review_notices: list[dict[str, Any]] = Field(default_factory=list)
    review_status: str | None = None
    save_status: str | None = None
    source_warnings: list[str] = Field(default_factory=list)
    answer_markdown: str | None = None
    answer_body_markdown: str | None = None
    processing_summary: str | None = None
    retained_answer_source: dict[str, Any] | None = None
    draft_id: str | None = None
    saved_path: str | None = None
    gaps: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str | None = None
    coverage: dict[str, Any] | None = None


# API 返回视图：将根任务状态、子任务、澄清和已记录 usage 一起交给调用方。
class TaskView(BaseModel):
    task_id: str
    status: TaskStatus
    prompt: str
    save_policy: SavePolicy
    result: TaskResult | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    parent_task_id: str | None = None
    role: Role = Role.supervisor
    subtasks: list[SubtaskResult] = Field(default_factory=list)
    clarification: Clarification | None = None


# 一次工具调用的持久记录，顺序号用于恢复时定位已执行行动。
class ActionRecord(BaseModel):
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    sequence: int
    tool_name: str
    arguments: dict[str, Any]
    status: Literal["started", "succeeded", "failed", "cancelled"] = "started"
    observation: dict[str, Any] = Field(default_factory=dict)


# 工具给 Agent 的统一观察；ok、kind、retryable 分开表达执行结果。
class Observation(BaseModel):
    ok: bool
    kind: Literal["success", "empty", "needs_review", "validation_error", "execution_error", "cancelled"]
    data: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    retryable: bool = False


# 模型请求收据合同；unknown 表示结果不明，不能据此补零用量或自动重发。
class ModelCallReceipt(BaseModel):
    task_id: str
    sequence: int
    request_id: str | None = None
    status: Literal["started", "succeeded", "failed", "unknown"]
    usage: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None


# 网页显式操作的输入合同；不会自动提升旧任务的权限。
class Feedback(BaseModel):
    conversation_id: str
    source_task_id: str
    source_draft_id: str | None = None
    content_hash: str=Field(pattern=r'^[a-f0-9]{64}$')
    prompt: str=Field(min_length=1,max_length=10000)
    idempotency_key: str
    save_policy: SavePolicy=SavePolicy.deny
