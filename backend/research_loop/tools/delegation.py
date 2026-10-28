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


def specs() -> list[ToolSpec]:
    from research_loop.agents.researcher import progress_spec
    return [
        progress_spec(),
        ToolSpec('review_research', 'Independently review a selected current object when the user requests it or there is a specific doubt. New candidates/findings use source_task_id and actual item_ids. For an explicitly referenced prior answer in this conversation, or its unique identity read through read_memory, omit answer_markdown so the server loads the exact body; a newly composed answer supplies it. Continue using returned target_id/content_hash without source selectors. Give reason, neutral verification_questions and authorized evidence_ids. Questions concern scientific claims or actual research scope, not save/format compliance. No note creation or full-source reread prerequisite.', ResearchReviewInput, _review_research, ('supervisor',)),
        ToolSpec('revise_research', 'Revise selected unresolved items using current target_id/content_hash/review_id and exact old_text/new_text. Each old_text must occur uniquely in its current item. Preserve unrelated text and sources. Success creates an unreviewed version: previous verdicts do not approve it. Independently re-review the new version with the original neutral questions and actual evidence, not historical criticism.', ResearchRevisionInput, _revise_research, ('supervisor',)),
        ToolSpec('close_research_issue', 'Record a disposition for the current target_id/content_hash/review_id: exclude an unsuitable candidate, disclose a limitation, or end with the concrete stopping reason. Exclude unsuitable candidates or disclose concrete limitations; choose end when no useful continuation path or progress remains. Obey reflection.can_continue/stop_reason. Success preserves the work and review history; it does not approve content or authorize saving. Deliver retained results and unresolved requirements.', CloseResearchIssueInput, _close_research, ('supervisor',)),
        ToolSpec('record_evidence_matrix', 'Create an optional evidence matrix when the user requests it or the comparison needs it; use evidence_matrix_requested in the plan. Rows have question/method/dataset/metric/limitation with actual document_id/evidence_id and exact page quotes (whitespace may differ). Keep metric conditions; unknown values use missing, empty value/references and a scope-limited reason, or uncertain when conditions are unclear. Parallel rows do not establish comparability or fairness. Initial base_matrix_hash is null; changes bind the current hash. Pass the returned matrix_hash to finish_task. Matrix and prose do not synchronize automatically. No file-save permission or independent approval.', EvidenceMatrixInput, _matrix, ('supervisor',)),
        ToolSpec("set_research_plan", 'Record the current user deliverables as origin=user with verbatim source_quote, distinct from optional origin=strategy steps; legacy only preserves old unclassified goals. Put actual hard sources/date_from/date_to on the corresponding user goals; new-plan top-level sources/dates are adjustable retrieval preferences. sources means conferences/sources, not attachment IDs: retain actual root input_refs, count, save requirements and permissions. Unmarked historical plans retain their prior hard scope; do not migrate them without actual user clarification. Without a user count, requested_count=null and count_confirmed=false; suggested_count is not a requirement. Adjust strategy with change_reason or retain its ID as withdrawn; do not delete IDs or withdraw user/legacy requirements to claim completion. After real clarification update changed requirements and retain the rest. deep_reading_requested is only for explicit deep-reading notes/eight-point summaries, not ordinary reading/comparison/saving.', ResearchBrief, _plan, ("supervisor",)),
        ToolSpec("finish_task", 'Deliver answer_markdown or bind the current draft_id or reflection_target_id/reflection_content_hash. With a bound existing object omit answer_markdown: the server loads its exact body; do not append handling notes to it and resend. Put processing/failure notes in processing_summary. Report every active goal as satisfied or unresolved with real subtask_ids and a reason. Withdrawn strategy goals need no delivery entry; unfinished strategies are distinct from unmet user requirements. If a user requirement remains unmet, retain useful work and report partial; old partial subtasks may be complemented by later results. A missing source still needs a readable explanation. Success ends the root task without another model call.', FinishTaskInput, _finish_task, ("supervisor",)),
        ToolSpec("report_task_gap", "Record an unavailable user requirement without blocking achievable work, then disclose it. Optional internal strategy steps use plan status/change_reason instead.", ReportGapArgs, _gap, ("supervisor",)),
        ToolSpec("request_clarification", "Persist one blocking question and release the worker.", ClarificationArgs, _clarify, ("supervisor",)),
        ToolSpec("delegate_research", 'Delegate independent work needed by the user goal; submit the complete independent subtask list for the configured concurrency queue, with valid plan goal_ids. Give each objective the original subquestions, necessary conditions and authorized materials. Attachment tasks use the actual input_refs, not a replacement search or guessed identity. screen selects recommendations from metadata/abstracts; investigate answers questions/comparisons; deep_read is only for an explicitly requested deep-reading note. Do not create another Researcher merely to summarize results already available.', DelegateResearchArgs, _delegate, ("supervisor",)),
        ToolSpec("review_note", 'Independently review an existing draft using its exact current draft_id/content_hash, reason, neutral verification_questions and authorized evidence_ids. Use when requested or when there is a specific doubt, not as a universal publication/save gate. Questions concern scientific claims or true research scope; save/unchanged-text requirements use execution checks. After directed new evidence or revise_note, re-review the current version without historical criticism; obey reflection.can_continue/stop_reason.', ReviewNoteArgs, _review, ("supervisor",)),
    ]
