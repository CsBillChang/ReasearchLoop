"""Reviewer 子图：独立核对当前草稿及原文，允许补读；每个子图核验一次，不自动改写；修复轮次由根任务持久控制。"""
from __future__ import annotations

from typing import Any
import json
from uuid import uuid4

from pydantic import BaseModel, Field

from research_loop.agents.graph import build_graph, initial_state
from research_loop.agents.prompt_parts import (
    COMMON_PRINCIPLES, REVIEWER_ROLE, REVIEW_EVIDENCE_RULES,
    NOTE_REVIEW_CONTRACT, TARGET_REVIEW_CONTRACT, compose_prompt,
)
from research_loop.domain.evidence import ClaimAssessment
from research_loop.domain.notes import ReviewFinding
from research_loop.services.verification import validate_assessments
from research_loop.domain.tasks import Observation
from research_loop.tools.base import ToolContext, ToolSpec
from research_loop.tools.papers import ReadPaperArgs


# Reviewer 的一次核验结论；issues 与 insufficient 不应伪装成 supported。
class FinishReviewArgs(BaseModel):
    status: str = Field(pattern=r"^(supported|issues|insufficient)$")
    findings: list[ClaimAssessment] = Field(default_factory=list, max_length=30)
    gaps: list[str] = Field(default_factory=list, max_length=30)


# 仅用于已持久化旧上下文/收据恢复；不能用新合同拒绝旧成功响应后再发一次请求。
class LegacyFinishReviewArgs(FinishReviewArgs):
    findings: list[ReviewFinding] = Field(default_factory=list, max_length=30)


def _finish(args: FinishReviewArgs, ctx: ToolContext) -> Observation:
    """核对证据授权和结论一致性后返回当前稿的审校结果；不在此修改稿件或保存文件。"""
    db = getattr(ctx.services.get("papers"), "db", None)
    if db and any(not db.task_has_evidence(ctx.task_id, ref) for finding in args.findings for ref in finding.evidence_ids):
        raise ValueError("review evidence reference is not authorized")
    errors = []
    if args.status == "supported" and any(f.status != "supported" for f in args.findings):
        errors.append("supported verdict conflicts with unresolved findings")
    if args.findings and ctx.services.get("review_basis_required", True):
        try:
            validate_assessments(ctx.task_id, args.findings, ctx.services["papers"])
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("; ".join(errors))
    data = {
        "review_id": str(uuid4()), "task_id": ctx.services.get("review_parent_task_id", ctx.task_id),
        "draft_id": ctx.services.get("review_draft_id", ""),
        "content_hash": ctx.services.get("review_content_hash", ""),
        "status": args.status, "findings": [item.model_dump(mode="json") for item in args.findings],
        "gaps": args.gaps,
    }
    return Observation(ok=True, kind="success" if args.status == "supported" else "needs_review",
                       data=data, message="review result structurally recorded")


def finish_spec(require_basis: bool = True) -> ToolSpec:
    return ToolSpec("finish_review", "Finish one independent review without changing or saving the draft.",
                    FinishReviewArgs if require_basis else LegacyFinishReviewArgs, _finish, ("reviewer",))


def build_reviewer_graph(provider: Any, services: dict[str, Any], db: Any = None,
                         checkpoint_store: Any = None, max_model_requests: int | None = None,
                         max_tool_actions: int | None = None):
    """Reviewer仅有原文读取/检索与核验结束；不能自行开启无限修订。"""
    from research_loop.tools.registry import ToolRegistry
    read_only = [spec for spec in __import__("research_loop.tools.papers", fromlist=["specs"]).specs()
                     if spec.name in {"read_paper", "inspect_paper_media", "retrieve_evidence"}]
    from research_loop.domain.reflection import FinishResearchReview
    finish = (ToolSpec("finish_review", "Return one or more findings per selected item_id; cover every selected item. Each finding states a specific claim, reason and evidence. Repeated item_id for different issues is allowed. Do not edit or save.",
                       FinishResearchReview, _finish_research_review, ("reviewer",))
              if services.get("review_target") else finish_spec(services.get("review_basis_required", True)))
    limits = {key: value for key, value in {'max_model_requests': max_model_requests,
              'max_tool_actions': max_tool_actions}.items() if value is not None}
    return build_graph(provider, ToolRegistry(read_only + [finish]), services, db,
                       checkpoint_store=checkpoint_store, **limits)


def reviewer_state(task_id: str, draft_id: str, content_hash: str, markdown: str,
                   evidence_ids: list[str], evidence_bundle: list[dict[str, Any]] | None = None, questions: list[str] | None = None,
                   artifact_checks: dict[str, Any] | None = None) -> dict[str, Any]:
    """输入当前稿、内容 hash 和实际页级原文；构造独立审校上下文，不继承历史批评。"""
    state = initial_state(task_id, (
        "独立核验当前比较稿，不继承历史批评。稿件ID=" + draft_id + " hash=" + content_hash +
        "\n稿件:\n" + markdown + "\n可核对 evidence_ids=" + ",".join(evidence_ids)
    ), "deny", {})
    # 独立审校沿用当前稿数据，不继承父角色的委派说明。
    state["messages"][1]["content"] = state["prompt"] + "；后端保存权限：deny"
    if questions:
        state["messages"][1]["content"] += "\n本次中性核验问题：" + json.dumps(questions, ensure_ascii=False)
        state["messages"][1]["content"] += (
            "\n本次为定向核验：只报告这些问题涉及的科学主张及直接受影响的结论。"
            "不要扩展为其他论文或未修改段落的全稿重审。核验通过只表示这个范围，不能宣称全文已核验。"
            "若问题涉及逐字保留/保存状态，只引用程序检查或在gaps注明未检查，不把它列为缺原文的科学主张。"
        )
    if artifact_checks:
        state["messages"][1]["content"] += "\n程序只读差异检查（不是历史语义意见）：" + json.dumps(artifact_checks, ensure_ascii=False)
    state["role"] = "reviewer"
    state["review_basis_required"] = True
    state["messages"][0]["content"] = compose_prompt(
        COMMON_PRINCIPLES, REVIEWER_ROLE, REVIEW_EVIDENCE_RULES, NOTE_REVIEW_CONTRACT)
    state["messages"].append({"role": "user", "content": "权威原文证据包（不是历史意见）：\n" + json.dumps(evidence_bundle or [], ensure_ascii=False)})
    return state


def _finish_research_review(args, ctx: ToolContext) -> Observation:
    """覆盖选定对象并校验依据，生成通用结果，不触碰笔记批准状态。"""
    from research_loop.services.verification import validate_target_assessments
    target = ctx.services['review_target']
    validate_target_assessments(ctx.task_id, target, args.findings, ctx.services['papers'])
    clear = all(f.status == 'supported' or f.suitability == 'suitable' for f in args.findings)
    result = {'review_id': str(uuid4()), 'task_id': ctx.services['review_parent_task_id'],
              'target_id': target['target_id'], 'content_hash': target['content_hash'],
              'status': 'supported' if clear else 'issues',
              'findings': [f.model_dump(mode='json') for f in args.findings], 'gaps': args.gaps,
              'evidence_ids': sorted({ref for f in args.findings for ref in f.evidence_ids})}
    return Observation(ok=True, kind='success' if clear else 'needs_review', data=result,
                       message='selected research items independently reviewed; no note approval or file write')


def research_reviewer_state(task_id: str, target: dict, questions: list[str], bundle: list[dict],
                            execution_checks: dict | None = None) -> dict:
    """仅传当前内容、中性问题和原始证据，历史判词留在控制层审计中。"""
    state = initial_state(task_id, '独立核验选定研究对象。', 'deny', {})
    state['role'] = 'reviewer'
    state['messages'][0]['content'] = compose_prompt(
        COMMON_PRINCIPLES, REVIEWER_ROLE, REVIEW_EVIDENCE_RULES, TARGET_REVIEW_CONTRACT)
    state['messages'][1]['content'] = '当前权威对象和中性问题：\n' + json.dumps(
        {'target': target, 'verification_questions': questions,
         'execution_checks': execution_checks or {}}, ensure_ascii=False)
    state['messages'].append({'role': 'user', 'content': '实际原文证据包：\n' + json.dumps(bundle, ensure_ascii=False)})
    return state
