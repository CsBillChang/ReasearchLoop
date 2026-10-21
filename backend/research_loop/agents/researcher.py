"""Researcher 子图：只搜索/读取授权论文并提交结构化发现；没有创建笔记或保存文件的权限。"""
from __future__ import annotations

from uuid import uuid4
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, SkipValidation, ValidationError

from research_loop.agents.graph import build_graph, initial_state
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
    mode_prompt = {
        "screen": "本次mode=screen，仅筛选候选。根据标题、摘要和题录核对主题、日期与来源，findings逐篇交付标题、链接、推荐理由和实际依据层级，说明尚不能确认的内容。摘要未写训练数据、实验数字或具体机制时保留不确定性，不将其当作筛选失败、反证或必须补齐的原文核验目标。不要提交科学assessments；即使结构化子任务带verification_claims，也仅保留为后续关注点，不逐条核验，不为此默认下载全文。只有候选选择存在具体缺口且摘要不足时才按需读取相关正文；不影响推荐的缺口交回Supervisor，由其决定是否需要另派investigate。paper_ids只包含选中的论文，不包括所有检索候选。未读取正文时evidence_ids必须为空列表，不能填写paper_id，不需要为填此字段读取PDF。数量不足交付已有候选并标partial。不会启动论文八点总结。",
        "investigate": "本次mode=investigate，围绕分派的问题或比较维度定向研究。调查任务默认可读取全文；定向补证按具体问题选页，不为了完成核验重做整篇精读。读取原文后形成findings，包含普通论文链接、观察、适用条件与资料局限。无需八点总结或（完）标记，不会启动论文节点。paper_ids只包含实际用于发现的论文。若指定verification_claims，须用assessments逐项返回：supported/contradicted必须有直接相关的原文引句；未找到、未读到或解析不清用insufficient，注明not_found或incomplete_or_ambiguous。任务尚有缺口就标partial并保留有用发现，不无限补读。",
        "deep_read": "本次mode=deep_read。取得正文后尽快finish_research交接证据；findings只写少量取证说明，完整八点总结由随后论文精读节点生成。verification_claims保留为后续总结关注点，本取文节点不要求先逐条核验；不要为此反复补读或重写八点全文。",
    }
    # 先固定本次交付模式，再说明共享的工具、来源与原文使用约束。
    state["messages"][0]["content"] = mode_prompt[mode] + (
        "你是独立 Researcher。只在给定子目标和已授权论文内使用工具并整理发现。"
        "以下是使用原文时的共同规则，不会把screen变成全文调查。start_page/page_count分派值是建议，不能当作永久页权限。"
        "只有根任务用户明确限制阅读页时才遵守服务返回的user_page_range；不能自行固定只读前8页。"
        "用read_coverage核对实际返回范围；全文返回不等于图表公式完全理解，解析缺口需披露。"
        "不能创建草稿、保存文件或批准结果。围绕子目标交回发现、对应依据、适用条件和未解决问题，区分原文事实与自己的分析。形成有用成果后及时record_research_progress并finish_research交接，不要求先自行完成全部审校。使用原文时保留evidence_id、页码和缺口；初步assessment不是独立Reviewer批准。"
        "需要读取已指定paper_ids/input_refs时使用这些准确标识，不添加arXiv:等前缀，不重新搜索替换；指定身份本身不要求screen读取正文。"
        "Jina的page_number=null表示无可靠PDF页码，引用正文evidence_id即可，不能凭印象补页码。"
        "获取失败时可尝试已授权同论文地址或上报失败；不要为了没有物理页号重复读取整篇。"
        "读取全部可提取文字仍不能证明图片里没有信息；未理解的图片不能按印象断言其内容。"
        "若公式符号损坏或图中关系无法从文字确定，交回具体evidence_id、物理页和待查问题，让Supervisor决定是否委派视觉补读；不声称文字读取已看图，也不为此反复读全文。"
        "一种设备/一个组件的设置不能推出所有组件的设置，也不能用来否证未报告的其他设置。"
        "比较维度不必互斥：分别记录输入、训练、推理和输出的作用，同一方法可跨多个阶段；先列原文依据，再形成对照，不能为突出差异删除共同点。"
        "findings与gaps必须保持相同判断：未找到支持只能说当前无法确认，不能说已证伪或必定不存在。"
    )
    if brief:
        state["messages"].append({"role": "user", "content": "结构化子任务与已授权定位：\n" + json.dumps(brief, ensure_ascii=False)})
    return state
