"""Reviewer 子图：独立核对当前草稿及原文，允许补读；每个子图核验一次，不自动改写；修复轮次由根任务持久控制。"""
from __future__ import annotations

from typing import Any
import json
from uuid import uuid4

from pydantic import BaseModel, Field

from research_loop.agents.graph import build_graph, initial_state
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
                         checkpoint_store: Any = None, max_model_requests: int = 4):
    """Reviewer仅有原文读取/检索与核验结束；不能自行开启无限修订。"""
    from research_loop.tools.registry import ToolRegistry
    read_only = [spec for spec in __import__("research_loop.tools.papers", fromlist=["specs"]).specs()
                     if spec.name in {"read_paper", "inspect_paper_media", "retrieve_evidence"}]
    from research_loop.domain.reflection import FinishResearchReview
    finish = (ToolSpec("finish_review", "Return one or more findings per selected item_id; cover every selected item. Each finding states a specific claim, reason and evidence. Repeated item_id for different issues is allowed. Do not edit or save.",
                       FinishResearchReview, _finish_research_review, ("reviewer",))
              if services.get("review_target") else finish_spec(services.get("review_basis_required", True)))
    return build_graph(provider, ToolRegistry(read_only + [finish]), services, db,
                       max_model_requests, 20, 0, checkpoint_store)


def reviewer_state(task_id: str, draft_id: str, content_hash: str, markdown: str,
                   evidence_ids: list[str], evidence_bundle: list[dict[str, Any]] | None = None, questions: list[str] | None = None,
                   artifact_checks: dict[str, Any] | None = None) -> dict[str, Any]:
    """输入当前稿、内容 hash 和实际页级原文；构造独立审校上下文，不继承历史批评。"""
    state = initial_state(task_id, (
        "独立核验当前比较稿，不继承历史批评。稿件ID=" + draft_id + " hash=" + content_hash +
        "\n稿件:\n" + markdown + "\n可核对 evidence_ids=" + ",".join(evidence_ids)
    ), "deny", {})
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
    state["max_model_requests"] = 4
    state["messages"][0]["content"] = (
        "你是独立 Reviewer，只核对当前稿中的原文支持性。可以 read_paper 补读已授权页，"
        "不能重写草稿、创建/保存笔记，也不能把引用统计当正文证据；结束时调用 finish_review。"
        "已提供的证据包是真实读取的原文，可直接核对。page_number=null是Jina正文，使用evidence_id定位，不要求凭空补物理页码。只有确需补读时使用其中target_ref和页码，不能把evidence_id、draft_id或hash传给read_paper。"
        "只检查当前已作出的核心陈述；准确披露未读范围/不提供被引量不是幻觉，也不要求草稿回答其未宣称覆盖的问题。"
        "缺依据与原文矛盾分开报告；一次核验后立即finish_review，无需重复核验或输出修改稿。"
        "supported/contradicted需direct_statement及绑定原文的source_quote，反证必须逻辑上排除当前主张，不能只是提到另一种设置。"
        "找不到支持用insufficient/not_found；解析不全、未读图片用insufficient/incomplete_or_ambiguous。"
        "准确说明在已读取范围无法确认本身可被支持，不因原论文缺证而要求该说明也改成有确定事实。"
        "不要把未找到提升为全篇不存在或已确定否证。证据包的read_scope与gaps、assets描述可用材料边界，不是图片内容。"
        "read_scope仅表示本次选入审校包的页；task_evidence_scope表示任务累计已取得/继承的文字页。不能因审校包未选某页就说此前任务没读该页；不能据assets存在推断图表只含图片或全部公式不可读。"
        "范围/格式/保存等元说明不是论文事实，不要求论文原文出现对应句子；只针对其中实际声称的科学事实提findings。"
        "文字是否保持由程序差异检查负责；它不证明科学事实正确，但不应被当作缺少论文来源的问题。"
        "每条source_quote用最短的连续原文，保留PDF字词；不要用省略号拼接多个句段。"
    )
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
    # 与既有Reviewer相同的角色请求配置，不改变根任务预算。
    state['max_model_requests'] = 4
    state['messages'][0]['content'] = (
        '你是独立Reviewer，负责选定研究对象的适用性、原文支持性和结论推理，不负责规划整个任务或生成新报告。'
        '输入是当前对象/版本、中性问题和已有依据。item_id标识对象，不是意见ID；每个选定对象至少一条判断。'
        '同一回答的多个问题可用相同item_id分别提交，reason明确对应主张和问题；保留不同判断，不用一条支持意见替代其他缺证或反证。'
        '先使用已有证据，只有具体缺口才定向补读；不重做整篇研究。'
        'candidates只检查元数据/摘要对研究目标的适用性，使用suitability=suitable/unsuitable/uncertain与basis=metadata，'
        '引用实际metadata_fields；摘要不是已读全文，不推断未提供的实验细节。'
        '每条意见必须且仅一个判断字段非空；不得遗漏判断或写null，reason/basis/suggested_action不能代替判断。'
        'findings/answer使用status=supported/contradicted/insufficient（包括scope_disclosure）；candidates使用suitability。'
        'supported或contradicted必须有direct_statement、'
        '实际evidence_ids和最短连续source_quote；存在另一种设置不等于排除了当前说法。'
        'source_quote保留原文字词，不用省略号或占位符拼接公式、句段；找不到可定位引句则说明缺口，不伪造引文。'
        '找不到依据用not_found，解析/图片/范围不足用incomplete_or_ambiguous，均为insufficient。'
        '准确披露在已读范围无法确认可用supported/scope_disclosure并绑定实际evidence_ids，不要求论文原句证明执行范围。'
        'scope_disclosure只用于查证范围说明，不能据此批准科学断言。'
        '仅检查中性问题涉及的当前主张及直接受影响结论，不扩展全稿或未提出的研究问题。'
        '未保存、未改其他段落、遵守用户授权属于程序执行约束，不是论文事实；不能用scope_disclosure或补论文证据ID把这些约束批准为科学判断。'
        '这类要求只依据execution_checks中的实际程序记录，在gaps说明检查范围；未提供差异检查就注明未检查，不能凭阅读回答或论文原文推断。'
        'findings只提交研究对象的科学主张/适用性及真实查证范围；纯程序问题放gaps，不自动删除或改写已提交的错误判断。'
        '检查结论是否保留证据中的条件、范围和例外；研究者的推断与作者报告分开。信息不足不是反证，另一种设置不自动否定当前说法。'
        '可read_paper/retrieve_evidence补读授权材料，inspect_paper_media仅提取定位，不代表理解图像。'
        '证据包read_scope只是本次入包页，task_evidence_scope才是任务已取得范围。'
        '已有依据足够就finish_review；没有可行补证路径时以insufficient说明已查范围和具体缺口并结束，不要求证明论文全篇不存在。'
        'reason写清当前说法、依据与判断之间的关系；suggested_action给keep/supplement/revise/exclude/end建议。'
        '不改文、不保存、不替Supervisor调度；Supervisor依据观察决定后续动作。'
    )
    state['messages'][1]['content'] = '当前权威对象和中性问题：\n' + json.dumps(
        {'target': target, 'verification_questions': questions,
         'execution_checks': execution_checks or {}}, ensure_ascii=False)
    state['messages'].append({'role': 'user', 'content': '实际原文证据包：\n' + json.dumps(bundle, ensure_ascii=False)})
    return state
