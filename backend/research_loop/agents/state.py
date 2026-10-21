"""LangGraph 共享状态：跨节点传递的消息、待执行工具、证据与结束原因都须在此声明。"""
from __future__ import annotations
from typing import Any, TypedDict


# 图运行状态合同；新增跨节点字段必须在此声明，否则可能无法随图状态传递。
class AgentState(TypedDict, total=False):
    task_id: str
    role: str
    review_basis_required: bool
    prompt: str
    save_policy: str
    attachments: dict[str, dict[str, Any]]
    messages: list[dict[str, Any]]
    pending_tool_calls: list[dict[str, Any]]
    # 与待执行响应对应的冻结工具范围；恢复优先读原请求，不能扩大到日常注册表。
    pending_tool_scope: list[str]
    actions: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    model_requests: int
    tool_actions: int
    # 拒绝回执有独立行动序号，但不消耗工具额度；恢复时两种计数分别沿用。
    action_sequence: int
    # 新任务共享根冻结策略；历史快照缺失时保持其原额度。
    execution_policy: dict[str, Any]
    budget_tail: bool
    max_model_requests: int
    max_tool_actions: int
    reserve_model_requests: int
    final_answer: str | None
    final_reason: str | None
    draft_id: str | None
    saved_path: str | None
    evidence_ids: list[str]
    gaps: list[str]
    source_warnings: list[str]
    cancelled: bool
    finished: bool
    worker_id: str
    lease_token: str
    parent_task_id: str | None
    root_task_id: str | None
    subtask_results: list[dict[str, Any]]
    review_result: dict[str, Any] | None
    structured_result: dict[str, Any] | None
    research_mode: str
    research_plan: dict[str, Any] | None
    delivery: dict[str, Any] | None
    # 合法失败候选的精确行动引用独立于已解除的资格阻断；正文不进入此字段。
    finish_candidate_ref: dict[str, str] | None
    coverage_required: bool
    conversation_id: str
    context_metrics: dict[str, Any]
    # 成功行动的派生投影；恢复仍以数据库日志为准。
    research_progress: dict[str, Any]
    # 重复取证或明确资格阻断的交接；候选只存行动/hash引用，正文保留原观察。
    research_handoff: dict[str, Any] | None

    deep_reading: dict[str, Any]
