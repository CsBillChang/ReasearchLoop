"""Researcher 子图：只搜索/读取授权论文并提交结构化发现；没有创建笔记或保存文件的权限。"""
from __future__ import annotations

from uuid import uuid4
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_loop.agents.graph import build_graph, initial_state
from research_loop.domain.tasks import Observation
from research_loop.domain.evidence import ClaimAssessment
from research_loop.services.verification import validate_assessments
from research_loop.tools.base import ToolSpec, ToolContext
from research_loop.tools.papers import specs as paper_specs


# Researcher 的结构化交付，不要求它调用写文件工具。
class FinishResearchArgs(BaseModel):
    status: Literal["completed", "partial"] = "completed"
    findings: list[str] = Field(default_factory=list, max_length=20)
    evidence_ids: list[str] = Field(default_factory=list, max_length=100)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)
    gaps: list[str] = Field(default_factory=list, max_length=20)
    assessments: list[ClaimAssessment] = Field(default_factory=list, max_length=30)


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
    if claims and sorted(item.claim for item in args.assessments) != sorted(claims):
        raise ValueError("return one assessment for each exact verification_claim, including insufficient when unavailable")
    papers = ctx.services.get("papers")
    if args.assessments:
        validate_assessments(ctx.task_id, args.assessments, papers)
    if args.status == "completed" and mode != "deep_read":
        if not any(text.strip() for text in args.findings) or not args.paper_ids:
            raise ValueError("completed research requires findings and selected paper_ids; otherwise report partial")
        if mode == "investigate" and not args.evidence_ids:
            raise ValueError("investigate requires read evidence; otherwise report partial with the missing source")
    return Observation(ok=True, kind="success", data={
        "status": args.status,
        "subtask_id": ctx.task_id, "findings": args.findings, "evidence_ids": args.evidence_ids,
        "paper_ids": args.paper_ids, "gaps": args.gaps,
        "assessments": [item.model_dump(mode="json") for item in args.assessments],
        "evidence_scope": papers.evidence_scope(ctx.task_id) if db else [],
    }, message="research result structurally recorded")


def finish_spec() -> ToolSpec:
    return ToolSpec("finish_research", "Finish this research subtask with evidence IDs and explicit gaps.",
                    FinishResearchArgs, _finish, ("researcher",))


def build_researcher_graph(provider: Any, services: dict[str, Any], db: Any = None,
                           checkpoint_store: Any = None, max_model_requests: int = 12,
                           max_tool_actions: int = 20, deep_reading: bool = True):
    """仅注册论文工具和 finish_research，复用通用图；角色权限在工具入口再次校验。"""
    from research_loop.tools.registry import ToolRegistry
    return build_graph(provider, ToolRegistry(paper_specs() + [finish_spec()]), services, db,
                       max_model_requests, max_tool_actions, 0, checkpoint_store, deep_reading=deep_reading)


def researcher_state(task_id: str, objective: str, save_policy: str = "deny",
                     attachments: dict[str, dict[str, Any]] | None = None,
                     brief: dict[str, Any] | None = None) -> dict[str, Any]:
    """只注入本子任务目标、附件和范围；不复制父任务对话，防止跨任务信息与权限混入。"""
    state = initial_state(task_id, objective, save_policy, attachments, 12, 20, 0)
    state["role"] = "researcher"
    mode = (brief or {}).get("mode", "deep_read")
    state["research_mode"] = mode
    state["messages"][0]["content"] = (
        "你是独立 Researcher。只在给定子目标和已授权论文内搜索、读取原文并整理发现。"
        "默认首次read_paper省略page_count读取全文；start_page/page_count分派值只是阅读建议，不能当作永久页权限。"
        "只有根任务用户明确限制阅读页时才遵守服务返回的user_page_range；不能自行固定只读前8页。"
        "用read_coverage核对实际返回范围；全文返回不等于图表公式完全理解，解析缺口需披露。"
        "不能创建草稿、保存文件或批准结果；结束时调用 finish_research，保留 evidence_id、页码和缺口。"
        "已指定paper_ids/input_refs时直接读取这些准确标识，不添加arXiv:等前缀，不重新搜索替换。"
        "Jina的page_number=null表示无可靠PDF页码，引用正文evidence_id即可，不能凭印象补页码。"
        "获取失败时可尝试已授权同论文地址或上报失败；不要为了没有物理页号重复读取整篇。"
        "定向核验须用assessments逐项返回verification_claims：supported/contradicted必须有直接相关的原文引句；"
        "未找到、未读到或解析不清用insufficient，注明not_found或incomplete_or_ambiguous。"
        "读取全部可提取文字仍不能证明图片里没有信息；未理解的图片不能按印象断言其内容。"
        "一种设备/一个组件的设置不能推出所有组件的设置，也不能用来否证未报告的其他设置。"
        "比较维度不必互斥：分别记录输入、训练、推理和输出的作用，同一方法可跨多个阶段；先列原文依据，再形成对照，不能为突出差异删除共同点。"
        "findings与gaps必须保持相同判断：未找到支持只能说当前无法确认，不能说已证伪或必定不存在。"
    )
    mode_prompt = {
        "screen": "本次仅筛选候选。优先根据标题/摘要核对主题、日期和来源，findings交付逐篇标题、链接、相关理由及依据层级。只有摘要不足才读取正文，不默认逐篇下载或精读。paper_ids只包含选中的论文，不包括所有检索候选。未读取正文时evidence_ids必须为空列表，不能填写paper_id，也不需要为填此字段而读取PDF。数量不足交付已有候选并标partial。不会启动DPR八点总结。",
        "investigate": "本次定向研究。读取原文后围绕分派的问题/维度形成findings，包含普通论文链接、观察、适用条件与资料局限。无需八点总结或（完）标记，不会启动DPR节点。paper_ids只包含实际用于发现的论文。任务尚有缺口就标partial并保留有用发现，不无限补读。",
        "deep_read": "取得正文后尽快finish_research交接证据；findings只写少量取证说明，完整八点总结由随后DPR精读节点生成。",
    }
    state["messages"][0]["content"] += mode_prompt[mode]
    if brief:
        state["messages"].append({"role": "user", "content": "结构化子任务与已授权定位：\n" + json.dumps(brief, ensure_ascii=False)})
    return state
