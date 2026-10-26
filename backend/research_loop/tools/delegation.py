"""Supervisor 调度与角色共用阶段进展：澄清、委派、按需核验及候选成果记录。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from research_loop.domain.tasks import Observation, SubtaskSpec, ResearchBrief, FinishTaskInput
from research_loop.tools.base import ToolContext, ToolSpec
from research_loop.services.coverage import CoverageEligibilityError
from research_loop.domain.research import EvidenceMatrixInput
from research_loop.domain.reflection import ResearchReviewInput, ResearchRevisionInput, CloseResearchIssueInput


# 澄清工具参数：问题、所缺字段与建议值。
class ClarificationArgs(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    fields: list[str] = Field(default_factory=list, max_length=10)
    suggested_values: list[str] = Field(default_factory=lambda: ["5"])


# 分派清单独立于执行器资源槽位，全部合法子任务先持久化再排队。
class DelegateResearchArgs(BaseModel):
    subtasks: list[SubtaskSpec] = Field(min_length=1)


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
    """校验整批范围后交给队列，物理并发由执行器资源配置控制。"""
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


def _review_research(args: ResearchReviewInput, ctx: ToolContext) -> Observation:
    """核验已授权研究对象，不要求创建笔记；执行失败保持独立。"""
    result = ctx.services['reflection'].review_research(ctx.task_id, args,
        lambda attempt, target, refs, questions: ctx.services['review_research'](ctx.task_id, attempt, target, refs, questions))
    failed = result.get('status') == 'execution_failed'
    return Observation(ok=not failed, kind='execution_error' if failed else
        ('success' if result.get('quality_status') == 'supported' else 'needs_review'),
        retryable=False, data=result, message=('；'.join(result.get('gaps', [])) if failed else
            '按需研究核验结果；查看具体问题、依据和reflection停止原因。'))


def _revise_research(args: ResearchRevisionInput, ctx: ToolContext) -> Observation:
    """提交精确局部版本，原子保存差异，不覆盖子任务原结果。"""
    result = ctx.services['reflection'].revise_research(ctx.task_id, args)
    return Observation(ok=True, kind='success', data=result, message='研究对象局部新版本已登记，当前版本尚未核验。')


def _close_research(args: CloseResearchIssueInput, ctx: ToolContext) -> Observation:
    """记录Supervisor明确结束选择，保留Reviewer原始意见。"""
    result = ctx.services['reflection'].close_research(ctx.task_id, args)
    return Observation(ok=True, kind='success', data={'research_disposition': result},
                       message='已记录排除/披露/结束原因；请交付已有成果和适用边界。')


def _plan(args: ResearchBrief, ctx: ToolContext) -> Observation:
    """返回经检查的目标计划，由通用工具行动与checkpoint持久化。"""
    plan = ctx.services["coverage"].prepare_plan(ctx.task_id, args)
    return Observation(ok=True, kind="success", data={"research_plan": plan}, message="plan recorded; delegate work and assess observed coverage")


def _finish_task(args: FinishTaskInput, ctx: ToolContext) -> Observation:
    """核对目标覆盖后结束根任务，不触发额外Reviewer或文件写入。"""
    try:
        identity = ctx.services.get('finish_action_identity')
        if identity is None:
            result = ctx.services["coverage"].finish(ctx.task_id, args)
        else:
            result = ctx.services["coverage"].finish(ctx.task_id, args, action_identity=identity)
    except CoverageEligibilityError as exc:
        data = {'coverage_blocker': exc.diagnostic}
        if exc.candidate is not None:
            data['finish_candidate'] = exc.candidate
        return Observation(ok=False, kind='execution_error', retryable=False, data=data, message=str(exc))
    return Observation(ok=True, kind="success", data={"delivery": result}, message="task delivery recorded")


def _matrix(args: EvidenceMatrixInput, ctx: ToolContext) -> Observation:
    """产生有据单元格；模型判断与结构验证分开，不启动核验或保存文件。"""
    matrix = ctx.services['coverage'].evidence_matrix(ctx.task_id, args)
    return Observation(ok=True, kind='success', data={'evidence_matrix': matrix},
                       message='matrix prepared from actual source excerpts; this is not semantic approval')


def specs(*, continuation_memory: bool = False) -> list[ToolSpec]:
    from research_loop.agents.researcher import progress_spec
    finish_model = FinishTaskInput
    if continuation_memory:
        from research_loop.domain.continuation_memory import ContinuationFinishTaskInput
        finish_model = ContinuationFinishTaskInput
    return [
        progress_spec(),
        ToolSpec('review_research', 'Optional independent review of actual candidates/findings or an answer snapshot. For new candidates/findings use source_task_id and actual item_ids. For an explicitly referenced prior assistant answer in this conversation, omit answer_markdown: the server loads its exact body and identity; do not copy it or re-read all sources first. For a newly composed answer supply answer_markdown. Continue with returned target_id/content_hash, omit source selectors. Give neutral verification_questions and reason; only authorized evidence_ids. No note creation or save required.', ResearchReviewInput, _review_research, ('supervisor',)),
        ToolSpec('revise_research', 'Patch only unresolved selected items using current target_id/content_hash/review_id and exact old_text/new_text. Preserve all other text and source records; re-review current version independently. Not a whole-answer rewrite.', ResearchRevisionInput, _revise_research, ('supervisor',)),
        ToolSpec('close_research_issue', 'End an unresolved review with a concrete reason: exclude unsuitable candidates, disclose a limitation, or end after no useful path. Does not approve content or authorize save. Use current target_id/content_hash/review_id.', CloseResearchIssueInput, _close_research, ('supervisor',)),
        ToolSpec('record_evidence_matrix', 'Optional structured comparison artifact, after reading sources. Each document row has question/method/dataset/metric/limitation. Quote exact page text (whitespace may differ); never infer missing numbers or universal experimental fairness. missing has empty value/references and a scope-limited reason. Metric needs explicit conditions; uncertain when conditions are unknown. Use current base_matrix_hash for changes, null initially; use returned matrix_hash in finish_task. No file save or automatic Reviewer.', EvidenceMatrixInput, _matrix, ('supervisor',)),
        ToolSpec("set_research_plan", "Put actual user hard sources/date constraints on the corresponding origin=user goals with verbatim source_quote. Copy one short contiguous span from the actual user request or resumed answer; do not paraphrase, add punctuation, or join separated spans with ellipses. Split separated requirements into goals with their own exact quotes. Top-level sources/date are changeable retrieval preferences in newly created plans. Keep root input_refs/count/save constraints. Unmarked historical plans preserve their prior hard scope until actual user clarification. Separate user deliverables (origin=user with verbatim source_quote) from optional self-devised research steps (origin=strategy); legacy only preserves old unclassified goals. Preserve user constraints without a real resumed answer. Adjust strategy goals with change_reason, or retain them as status=withdrawn; never delete their IDs. User clarification may redirect requirements. deep_reading_requested only for a requested deep-reading note.", ResearchBrief, _plan, ("supervisor",)),
        ToolSpec("finish_task", "Deliver retained work and report every active goal as satisfied or unresolved. Withdrawn strategy goals need no delivery entry; unresolved strategy steps are recorded separately from unmet user requirements. With reflection_target_id/hash omit answer_markdown: the server loads the exact current body. Put handling notes in processing_summary. End without extra model calls.", finish_model, _finish_task, ("supervisor",)),
        ToolSpec("report_task_gap", "Record an unavailable user requirement without blocking achievable work, then disclose it. Optional internal strategy steps use plan status/change_reason instead.", ReportGapArgs, _gap, ("supervisor",)),
        ToolSpec("request_clarification", "Persist one blocking question and release the worker.", ClarificationArgs, _clarify, ("supervisor",)),
        ToolSpec("delegate_research", "Persist all independent researcher subtasks and run them through the configured concurrency queue.", DelegateResearchArgs, _delegate, ("supervisor",)),
        ToolSpec("review_note", "Independent review when needed by the task or a specific doubt. Record reason. After directed new evidence or revise_note, re-review independently; obey reflection stop_reason. Not a universal publication gate.", ReviewNoteArgs, _review, ("supervisor",)),
    ]
