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
    actions: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    model_requests: int
    tool_actions: int
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
    coverage_required: bool
    conversation_id: str
    context_metrics: dict[str, Any]

    deep_reading: dict[str, Any]
