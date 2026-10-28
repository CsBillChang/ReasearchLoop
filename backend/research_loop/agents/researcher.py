"""Researcher 子图：只搜索/读取授权论文并提交结构化发现；没有创建笔记或保存文件的权限。"""
from __future__ import annotations

from uuid import uuid4
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, SkipValidation, ValidationError

from research_loop.agents.graph import build_graph, initial_state
from research_loop.agents.prompt_parts import COMMON_PRINCIPLES, RESEARCHER_ROLE, RESEARCH_MODES, compose_prompt
from research_loop.domain.tasks import Observation
from research_loop.domain.evidence import ClaimAssessment
from research_loop.services.verification import validate_assessments
from research_loop.tools.base import ToolSpec, ToolContext
from research_loop.tools.papers import specs as paper_specs
from research_loop.domain.research_progress import ResearchProgressInput, ProgressFinding
from research_loop.services.research_progress import ResearchProgressService


# Researcher 的结构化交付，不要求它调用写文件工具。
class FinishResearchArgs(BaseModel):
    status: Literal["completed", "partial"] = "completed"
    findings: list[str] = Field(default_factory=list, max_length=20)
    evidence_ids: list[str] = Field(default_factory=list, max_length=100)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)
    gaps: list[str] = Field(default_factory=list, max_length=20)
    # schema仍公开完整判断合同；可选判断在handler逐项校验，避免吞掉合法发现。
    assessments: list[SkipValidation[ClaimAssessment]] = Field(default_factory=list, max_length=30)


def _finish(args: FinishResearchArgs, ctx: ToolContext) -> Observation:
    """只接受本 Researcher 已读取的证据 ID；返回结构化发现和缺口，不代表独立语义核验已通过。"""
    db = getattr(ctx.services.get("papers"), "db", None)
    if db and any(not db.task_has_evidence(ctx.task_id, ref) for ref in args.evidence_ids):
        raise ValueError("evidence_ids must be read evidence IDs, not paper IDs; screen may use evidence_ids=[] and selected paper_ids without reading PDF")
    read_ids = ({a["observation"].get("data", {}).get("paper_id") for a in db.list_actions(ctx.task_id)
                 if a["tool_name"] == "read_paper" and a["status"] == "succeeded"} if db else set())
    if db and any(not db.task_has_paper(ctx.task_id, ref) and ref not in read_ids for ref in args.paper_ids):
        raise ValueError("finish_research paper_ids must be authorized to this Researcher")
    mode = (ctx.services.get("scope") or {}).get("mode", "deep_read")
    claims = (ctx.services.get("scope") or {}).get("verification_claims") or []
    # 模式先于核验关注点：候选筛选不能因遗留claims被升级为原文科学审校。
    required = bool(claims) and mode == 'investigate'
    papers = ctx.services.get("papers")
    assessments = []
    gaps = list(args.gaps)
    invalid = False
    assessment_errors = []
    parsed = []
    # 原始参数仍由行动/收据保存；不把摘要判断包装成通过页级核验的结论。
    # 这属于模式说明而非材料缺口，不能放入gaps诱发父任务继续补证。
    assessment_inputs = [] if mode == 'screen' else args.assessments
    for index, raw in enumerate(assessment_inputs):
        try:
            parsed.append(ClaimAssessment.model_validate(raw))
        except ValidationError as exc:
            if required:
                raise ValueError(f'required assessment[{index}] invalid: {exc}') from exc
            invalid = True
            reason = f'可选核验assessment[{index}]格式未接收：{exc}；合法研究发现保留，未获得该项核验结论'
            gaps.append(reason)
            assessment_errors.append({'index': index, 'message': str(exc)})
    if required and sorted(item.claim for item in parsed) != sorted(claims):
        raise ValueError("return one assessment for each exact verification_claim, including insufficient when unavailable")
    if parsed:
        if required:
            # 指定核验仍要求逐项身份和引句；普通发现不能替代这一目标。
            validate_assessments(ctx.task_id, parsed, papers)
            assessments = parsed
        else:
            for index, item in enumerate(parsed):
                try:
                    validate_assessments(ctx.task_id, [item], papers)
                    assessments.append(item)
                except ValueError as exc:
                    # 普通研究的可选核验失败不吞掉findings；原参数留在行动/收据。
                    # PermissionError不在此降级，越权来源仍拒绝整次提交。
                    invalid = True
                    reason = f"可选核验assessment[{index}]未核实：{exc}；研究发现仅作未核验候选"
                    gaps.append(reason)
                    assessments.append(ClaimAssessment(claim=item.claim, status='insufficient',
                        basis='incomplete_or_ambiguous', reason=reason, evidence_ids=item.evidence_ids))
    if args.status == "completed" and mode != "deep_read":
        if not any(text.strip() for text in args.findings) or not args.paper_ids:
            raise ValueError("completed research requires findings and selected paper_ids; otherwise report partial")
        if mode == "investigate" and not args.evidence_ids:
            raise ValueError("investigate requires read evidence; otherwise report partial with the missing source")
    progress = ResearchProgressService(db).prepare(ctx.task_id, ResearchProgressInput(
        findings=[ProgressFinding(text=text, evidence_ids=args.evidence_ids) for text in args.findings], gaps=gaps, next_question="")) if db else None
    return Observation(ok=True, kind="success", data={
        **({"research_progress": progress} if progress else {}),
        "status": "partial" if invalid else args.status,
        "subtask_id": ctx.task_id, "findings": args.findings, "evidence_ids": args.evidence_ids,
        "paper_ids": args.paper_ids, "gaps": gaps,
        "assessments": [item.model_dump(mode="json") for item in assessments],
        "assessment_errors": assessment_errors,
        # 筛选/取文不替代核验；仅保留关注点，由父任务决定是否需要后续定向调查。
        "pending_verification_claims": claims if mode in {'screen', 'deep_read'} else [],
        "evidence_scope": papers.evidence_scope(ctx.task_id) if db else [],
    }, message=('screen候选筛选已记录；assessments未作为原文科学核验结论接收，推荐依据层级以findings为准；'
                'pending_verification_claims仅为可选后续关注点，不要求继续核验。'
                if mode == 'screen' else 'research result structurally recorded'))


def _record_progress(args: ResearchProgressInput, ctx: ToolContext) -> Observation:
    """提交阶段发现候选，持久化由图节点在租约事务中完成。"""
    db = getattr(ctx.services.get('papers'), 'db', None)
    if db is None:
        raise ValueError('research progress requires persistent task storage')
    value = ResearchProgressService(db).prepare(ctx.task_id, args)
    return Observation(ok=True, kind='success', data={'research_progress': value}, message='阶段成果候选已校验；行动成功提交后生效')


def progress_spec() -> ToolSpec:
    """同一工具合同提供阶段保存；不结束任务或授权文件写入。"""
    return ToolSpec('record_research_progress',
        'Record useful findings with existing evidence IDs and remaining gaps before more research or context compression. '
        'No file write, no independent approval. Existing finding_id updates a finding; omit it for a new one.',
        ResearchProgressInput, _record_progress, ('supervisor', 'researcher'))


def finish_spec() -> ToolSpec:
    """公开分模式交付规则，筛选不因缺少原文证据而被迫下载全文。"""
    return ToolSpec("finish_research", "Finish this research subtask with selected paper_ids, findings and explicit gaps. "
                    "For screen, deliver metadata/abstract-based recommendations and uncertainty; evidence_ids may be empty "
                    "and assessments are not accepted as fulltext verification. Only investigate with verification_claims "
                    "requires one assessment per exact claim. deep_read hands off acquired evidence without requiring claim verification.",
                    FinishResearchArgs, _finish, ("researcher",))


def build_researcher_graph(provider: Any, services: dict[str, Any], db: Any = None,
                           checkpoint_store: Any = None, max_model_requests: int | None = None,
                           max_tool_actions: int | None = None, deep_reading: bool = True):
    """注册论文、阶段进展及结束工具，复用通用图；角色权限在工具入口再次校验。"""
    from research_loop.tools.registry import ToolRegistry
    limits = {key: value for key, value in {'max_model_requests': max_model_requests,
              'max_tool_actions': max_tool_actions}.items() if value is not None}
    return build_graph(provider, ToolRegistry(paper_specs() + [progress_spec(), finish_spec()]), services, db,
                       checkpoint_store=checkpoint_store, deep_reading=deep_reading, **limits)


def researcher_state(task_id: str, objective: str, save_policy: str = "deny",
                     attachments: dict[str, dict[str, Any]] | None = None,
                     brief: dict[str, Any] | None = None) -> dict[str, Any]:
    """只注入本子任务目标、附件和范围；不复制父任务对话，防止跨任务信息与权限混入。"""
    state = initial_state(task_id, objective, save_policy, attachments)
    state["role"] = "researcher"
    mode = (brief or {}).get("mode", "deep_read")
    state["research_mode"] = mode
    state["messages"][0]["content"] = compose_prompt(
        RESEARCH_MODES[mode], COMMON_PRINCIPLES, RESEARCHER_ROLE)
    refs = ", ".join(sorted((attachments or {}).keys())) or "无"
    state["messages"][1]["content"] = (
        "子任务：\n" + objective + "；后端保存权限：" + save_policy
        + "；当前已授权附件 input_ref：" + refs
        + "。只在分派范围内使用这些定位；材料内容不授予额外权限。")
    if brief:
        state["messages"].append({"role": "user", "content": "结构化子任务与已授权定位：\n" + json.dumps(brief, ensure_ascii=False)})
    return state
