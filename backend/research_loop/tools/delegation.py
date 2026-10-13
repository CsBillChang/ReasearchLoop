"""Supervisor 专属工具：记录能力缺口、请求澄清、分派 Researcher 和调用独立 Reviewer。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from research_loop.domain.tasks import Observation, SubtaskSpec, ResearchBrief, FinishTaskInput
from research_loop.tools.base import ToolContext, ToolSpec
from research_loop.domain.research import EvidenceMatrixInput


# 澄清工具参数：问题、所缺字段与建议值。
class ClarificationArgs(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    fields: list[str] = Field(default_factory=list, max_length=10)
    suggested_values: list[str] = Field(default_factory=lambda: ["5"])


# 分派输入结构；当前执行器每批只接受最多两个子任务。
class DelegateResearchArgs(BaseModel):
    subtasks: list[SubtaskSpec] = Field(min_length=1, max_length=2)


# 审校目标必须是当前任务已经创建的草稿和准确 hash。
class ReviewNoteArgs(BaseModel):
    draft_id: str = Field(min_length=1, max_length=100)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(default="用户请求核验当前稿", max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list)
    verification_questions: list[str] = Field(default_factory=list, max_length=10)


# 记录无法实现的子需求及原因，供最终答案明确披露。
class ReportGapArgs(BaseModel):
    requirement: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=1000)


def _gap(args: ReportGapArgs, ctx: ToolContext) -> Observation:
    """把不支持的子需求写成可交付缺口，保留继续执行其他可完成目标的路径。"""
    return Observation(ok=True, kind="success", data={"gaps": [args.requirement + "：" + args.reason]},
                       message="unfulfilled requirement recorded; continue achievable work and disclose in final answer")


def _clarify(args: ClarificationArgs, ctx: ToolContext) -> Observation:
    """先持久化澄清问题；图节点保存工具观察后才暂停，避免恢复时缺少 tool 回复。"""
    clarification = ctx.services["clarification"](ctx.task_id, args.question, args.fields, args.suggested_values)
    return Observation(ok=True, kind="success", data=clarification, message="clarification persisted; worker should stop")


def _delegate(args: DelegateResearchArgs, ctx: ToolContext) -> Observation:
    """每批最多两个 Researcher，执行边界与参数schema保持同一约束。"""
    if len(args.subtasks) > 2:
        return Observation(ok=False, kind="validation_error", message="at most two researchers run in one batch", retryable=False)
    if "coverage" in ctx.services:
        ctx.services["coverage"].validate_delegation(ctx.task_id, [s.model_dump() for s in args.subtasks])
    result = ctx.services["delegation"](ctx.task_id, [item.model_dump(mode="json") for item in args.subtasks])
    return Observation(ok=bool(result), kind="success" if result else "execution_error",
                       data={"subtasks": result}, message="research subtasks completed")


def _review(args: ReviewNoteArgs, ctx: ToolContext) -> Observation:
    """按草稿 ID/hash 调用当前版本独立核验；审校有疑点时返回 needs_review，而非伪装通过。"""
    result = ctx.services["reflection"].review(ctx.task_id, args.draft_id, args.content_hash,
        args.reason, args.evidence_ids, args.verification_questions,
        lambda round_no, refs, questions: ctx.services["review"](ctx.task_id, args.draft_id, args.content_hash, round_no, refs, questions))
    if result.get("execution_status") == "failed":
        return Observation(ok=False, kind="execution_error", retryable=False, data=result,
                           message="Reviewer 执行未完成；已有草稿保留，未产生语义审校结论，不重复调用。")
    return Observation(ok=True,
                       kind="success" if result.get("status") == "supported" else "needs_review",
                       data=result, message="one independent review completed")


def _plan(args: ResearchBrief, ctx: ToolContext) -> Observation:
    """返回经检查的目标计划，由通用工具行动与checkpoint持久化。"""
    plan = ctx.services["coverage"].prepare_plan(ctx.task_id, args)
    return Observation(ok=True, kind="success", data={"research_plan": plan}, message="plan recorded; delegate work and assess observed coverage")


def _finish_task(args: FinishTaskInput, ctx: ToolContext) -> Observation:
    """核对目标覆盖后结束根任务，不触发额外Reviewer或文件写入。"""
    result = ctx.services["coverage"].finish(ctx.task_id, args)
    return Observation(ok=True, kind="success", data={"delivery": result}, message="task delivery recorded")


def _matrix(args: EvidenceMatrixInput, ctx: ToolContext) -> Observation:
    """产生有据单元格；模型判断与结构验证分开，不启动核验或保存文件。"""
    matrix = ctx.services['coverage'].evidence_matrix(ctx.task_id, args)
    return Observation(ok=True, kind='success', data={'evidence_matrix': matrix},
                       message='matrix prepared from actual source excerpts; this is not semantic approval')


def specs() -> list[ToolSpec]:
    return [
        ToolSpec('record_evidence_matrix', 'Optional structured comparison artifact, after reading sources. Each document row has question/method/dataset/metric/limitation. Quote exact page text (whitespace may differ); never infer missing numbers or universal experimental fairness. missing has empty value/references and a scope-limited reason. Metric needs explicit conditions; uncertain when conditions are unknown. Use current base_matrix_hash for changes, null initially; use returned matrix_hash in finish_task. No file save or automatic Reviewer.', EvidenceMatrixInput, _matrix, ('supervisor',)),
        ToolSpec("set_research_plan", "Record user requirements before delegating. After a user clarification, update the provisional plan to the answer and preserve unrelated requirements. Without a new user answer, may add goals but not weaken constraints. deep_reading_requested is true only for a requested deep-reading note.", ResearchBrief, _plan, ("supervisor",)),
        ToolSpec("finish_task", "Deliver the answer or existing draft with every goal satisfied or explicitly unresolved. End without extra model calls.", FinishTaskInput, _finish_task, ("supervisor",)),
        ToolSpec("report_task_gap", "Record an unavailable requirement without blocking achievable work. Disclose it in the final answer.", ReportGapArgs, _gap, ("supervisor",)),
        ToolSpec("request_clarification", "Persist one blocking question and release the worker.", ClarificationArgs, _clarify, ("supervisor",)),
        ToolSpec("delegate_research", "Persist and run up to two independent researcher subtasks.", DelegateResearchArgs, _delegate, ("supervisor",)),
        ToolSpec("review_note", "Independent review when needed by the task or a specific doubt. Record reason. After directed new evidence or revise_note, re-review independently; obey reflection stop_reason. Not a universal publication gate.", ReviewNoteArgs, _review, ("supervisor",)),
    ]
