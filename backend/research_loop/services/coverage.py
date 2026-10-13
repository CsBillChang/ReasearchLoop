"""研究目标覆盖：复用行动日志保存计划，核对交付事实，不进行科学事实审校。"""
from __future__ import annotations

import calendar
import json
from datetime import date

from research_loop.domain.tasks import ResearchBrief, FinishTaskInput


class CoverageService:
    """从本任务的权威记录验证目标身份、资料覆盖和文件交付。"""

    def __init__(self, db):
        """注入已有数据库，不新建独立计划或审校存储。"""
        self.db = db

    def evidence_matrix(self, task_id, args):
        """整理可选比较产物；复用同根原文，提交由图的行动事务负责。"""
        from research_loop.services.evidence_matrix import EvidenceMatrixService
        return EvidenceMatrixService(self.db).prepare(task_id, args)

    def plan(self, task_id: str) -> dict | None:
        """返回最近成功的计划；同一工具批次内也能读取刚持久化的行动。"""
        for action in reversed(self.db.list_actions(task_id)):
            if action["tool_name"] == "set_research_plan" and action["status"] == "succeeded":
                return (action.get("observation") or {}).get("data", {}).get("research_plan")
        return None

    def prepare_plan(self, task_id: str, brief: ResearchBrief) -> dict:
        """依据有效用户回答更新计划；没有新回答时禁止模型削弱既定范围。

        Args:
            task_id: 根任务身份。
            brief: 模型从原始指令解释的目标；解释本身不冒充程序语义验证。

        Returns:
            可放入工具观察的计划，保存权限以数据库为准。
        """
        if not brief.goals:
            raise ValueError("research plan requires at least one observable goal")
        if brief.task_kind != "memory":
            # 汇总/局部改写子目标不额外索取一篇论文；全计划仍须有真实材料目标。
            # min_papers计不同材料，不要求为了修改旧稿去找“新论文”。
            if not any(g.min_papers > 0 for g in brief.goals):
                raise ValueError("研究计划至少需要一个min_papers>0的材料目标；纯历史回顾使用memory")
            if any(g.min_papers == 0 and g.requires_fulltext for g in brief.goals):
                raise ValueError("要求全文核对的目标需min_papers>0；零篇只用于不额外取材的汇总目标")
        if brief.task_kind == "memory":
            if not self.db.fetchone("SELECT task_id FROM conversation_links WHERE task_id=?", (task_id,)):
                raise ValueError("memory plan requires a conversation")
            if brief.requested_count or brief.deep_reading_requested or brief.save_requested:
                raise ValueError("新检索、精读或生成保存仍走原研究计划，不使用memory计划")
        task = self.db.get_task(task_id)
        value = brief.model_dump(mode="json")
        value["save_policy"] = task["save_policy"]
        attachments = {a["input_ref"] for a in json.loads(task["attachments_json"])}
        self._normalize_input_scopes(value, attachments)
        previous = self.plan(task_id)
        answer = self._latest_answer(task_id)
        # 用户回答由resume事务持久化，模型不能伪造回答ID来获取改计划权限。
        # 同一回答只触发一次范围重定；后续仍可增加目标，但不能任意降低要求。
        answered_revision = bool(answer and (not previous or previous.get("clarification_id") != answer["clarification_id"]))
        if previous and not answered_revision:
            # 旧计划可能把附件写在sources；归一后比较，不能把字段拆分误当成放宽范围。
            self._normalize_input_scopes(previous, attachments)
            old = {g["goal_id"]: g for g in previous["goals"]}
            new = {g["goal_id"]: g for g in value["goals"]}
            if any(new.get(key) != goal for key, goal in old.items()):
                raise ValueError(f"existing goals cannot be removed or weakened; keep current goal_ids {list(old)} and their constraints, add a new goal or report unresolved")
            for key in ("requested_count", "count_confirmed", "save_requested", "deep_reading_requested", "date_from", "date_to", "sources", "input_refs", "task_kind"):
                if previous.get(key, False if key == "deep_reading_requested" else None) != value.get(key):
                    raise ValueError("existing task constraint cannot be changed: " + key)
        if answer:
            value["clarification_id"] = answer["clarification_id"]
        return value

    def _latest_answer(self, task_id: str) -> dict | None:
        """读取同任务经resume提交的最新回答，排除待回答和过期问题。

        Returns:
            真实恢复记录绑定的回答；没有有效回答时返回None。
        """
        return self.db.fetchone(
            "SELECT c.clarification_id,c.answer FROM clarifications c "
            "JOIN resume_requests r ON r.task_id=c.task_id AND r.clarification_id=c.clarification_id "
            "WHERE c.task_id=? AND c.status='answered' AND r.answer=c.answer "
            "ORDER BY c.answered_at DESC,c.created_at DESC LIMIT 1", (task_id,))

    def _require_answered_plan(self, task_id: str, plan: dict) -> None:
        """新回答必须先进入计划，不能沿旧数量派发或结束；不解析自然语言。"""
        answer = self._latest_answer(task_id)
        if answer and plan.get("clarification_id") != answer["clarification_id"]:
            raise ValueError("new user clarification takes priority; update set_research_plan from the answer before delegating or finishing")

    @staticmethod
    def _normalize_input_scopes(value: dict, attachments: set[str]) -> None:
        """原位分离附件与会议来源；仅接受根任务授权的附件身份。"""
        for scope in [value, *value["goals"]]:
            # 来源会议与附件身份分开；兼容模型已经明确写出的input_ref:前缀。
            inline_refs = [s.split(":", 1)[1] for s in scope.get("sources", []) if s.startswith("input_ref:")]
            refs = list(dict.fromkeys(scope.get("input_refs", []) + inline_refs))
            if not set(refs).issubset(attachments):
                raise ValueError("plan input_refs must be authorized root attachments")
            scope["input_refs"] = refs
            scope["sources"] = [s for s in scope.get("sources", []) if not s.startswith("input_ref:")]

    def validate_delegation(self, task_id: str, specs: list[dict]) -> None:
        """新任务子目标须绑定计划；旧任务没有计划时仍走原合同。"""
        plan = self.plan(task_id)
        if not plan:
            return
        self._require_answered_plan(task_id, plan)
        known = {g["goal_id"] for g in plan["goals"]}
        for spec in specs:
            ids = spec.get("goal_ids") or []
            if not ids or not set(ids).issubset(known):
                raise ValueError(f"subtask goal_ids must reference the current research plan; valid goal_ids: {sorted(known)}; do not rename the plan to match invented IDs")
            if spec.get("mode", "deep_read") == "deep_read" and not plan.get("deep_reading_requested", False):
                raise ValueError("DPR deep_read requires a user-requested deep-reading note; use investigate for full-text analysis/comparison, or screen for search")

    @staticmethod
    def in_scope(meta: dict, goal: dict, plan: dict, inputs: dict | None = None, paper_id: str | None = None) -> bool:
        """按实际元数据核对来源/日期；未知日期不能满足明确时间范围。"""
        for scope in (plan, goal):
            raw_sources = scope.get("sources") or []
            sources = [s for s in raw_sources if not s.startswith("input_ref:")]
            refs = (scope.get("input_refs") or []) + [s.split(":", 1)[1] for s in raw_sources if s.startswith("input_ref:")]
            if refs and (paper_id is None or paper_id not in {(inputs or {}).get(ref) for ref in refs}):
                return False
            if sources and str(meta.get("source", "")).lower() not in {s.lower() for s in sources}:
                return False
            start, end = scope.get("date_from"), scope.get("date_to")
            if not start and not end:
                continue
            published = (meta.get("published") or "")[:10]
            try:
                if len(published) == 4:
                    low, high = published + "-01-01", published + "-12-31"
                elif len(published) == 7:
                    year, month = map(int, published.split("-"))
                    low, high = published + "-01", published + f"-{calendar.monthrange(year, month)[1]:02d}"
                else:
                    low = high = date.fromisoformat(published).isoformat()
                if (start and low < start) or (end and high > end):
                    return False
            except (ValueError, TypeError):
                return False
        return True

    def finish(self, task_id: str, payload: FinishTaskInput) -> dict:
        """验证覆盖报告并形成交付；不调用模型、不保存或修改笔记。

        Args:
            task_id: 当前根任务身份。
            payload: Supervisor最终正文/草稿以及逐目标结论。

        Returns:
            正文、覆盖记录、未完成项和可观察的completed/partial判断。

        Raises:
            ValueError: 目标漏报、身份越界或声称已覆盖而缺乏执行证据。
        """
        plan = self.plan(task_id)
        if not plan:
            raise ValueError("set_research_plan before finish_task")
        self._require_answered_plan(task_id, plan)
        expected = {g["goal_id"]: g for g in plan["goals"]}
        if len(payload.goals) != len(expected) or {g.goal_id for g in payload.goals} != set(expected):
            actual = [g.goal_id for g in payload.goals]
            raise ValueError("finish_task must account for every goal exactly once; "
                             f"expected goal_ids={list(expected)}, received={actual}. "
                             "Use the current plan IDs; report unfinished goals as unresolved. Do not reread history to guess IDs.")
        children = {c["subtask_id"]: c for c in self.db.list_subtasks(task_id)}
        if plan.get("task_kind") == "memory" and not any(a["status"] == "succeeded" and a["tool_name"] in {"remember_preference", "forget_preference", "search_memory", "read_memory", "get_profile", "update_saved_note"} for a in self.db.list_actions(task_id)):
            raise ValueError("memory任务必须有真实记忆操作或读回，不能只声称已经记住")
        inputs = {a["input_ref"]: a.get("paper_id") or a["input_ref"]
                  for a in json.loads(self.db.get_task(task_id)["attachments_json"])}
        resolved, gaps, paper_ids = [], [], set()
        bound=next((e['payload'] for e in self.db.list_events(task_id) if e['event_type']=='feedback_draft_bound'),None)
        # 当前稿接续已由API原子绑定原文，可满足材料覆盖；不能为重新计数强迫Researcher再读一遍。
        # 这里只证明材料身份，Reviewer语义结论和目标达成判断仍分别记录。
        inherited=[]
        if bound:
            for ref in bound['evidence_ids']:
                if self.db.task_has_evidence(task_id,ref):
                    row=self.db.fetchone('SELECT d.paper_id FROM page_evidence e JOIN documents d ON d.document_id=e.document_id WHERE e.evidence_id=?',(ref,))
                    if row:inherited.append(row['paper_id'])
        for item in payload.goals:
            matched = set()
            # Supervisor 已有直接 read_paper 工具。成功读取的当前任务材料也应
            # 计入覆盖，否则追问即便补读成功仍被迫再建重复 Researcher 子任务。
            # 这里只复用实际成功观察；历史摘要、仅有 document_id 均不算原文。
            for action in self.db.list_actions(task_id):
                if action['tool_name'] != 'read_paper' or action['status'] != 'succeeded':
                    continue
                paper_id=(action.get('observation') or {}).get('data',{}).get('paper_id')
                if not paper_id:
                    continue
                row=self.db.get_paper(paper_id)
                if self.in_scope(row['metadata'] if row else {},expected[item.goal_id],plan,inputs,paper_id):
                    matched.add(paper_id)
            for paper_id in set(inherited):
                row=self.db.get_paper(paper_id)
                if self.in_scope(row['metadata'] if row else {},expected[item.goal_id],plan,inputs,paper_id):
                    matched.add(paper_id)
            for child_id in item.subtask_ids:
                if child_id not in children:
                    raise ValueError("foreign subtask_id")
                child = children[child_id]
                result = (child.get("result") or {}).get("result", {})
                # 分篇研究也可支持父级综合目标；程序检查同根身份与材料范围，语义关联由Supervisor判断。
                if child["status"] not in {"completed", "partial"} or not result.get("findings"):
                    # 旧失败可与后续成功一同列为依据；只累计有用成果，最后再检查覆盖数量。
                    continue
                read_ids = set()
                for action in self.db.list_actions(child_id):
                    if action["tool_name"] == "read_paper" and action["status"] == "succeeded":
                        read_ids.add((action.get("observation") or {}).get("data", {}).get("paper_id"))
                for paper_id in result.get("paper_ids", []):
                    row = self.db.get_paper(paper_id)
                    # 合法附件以实际read_paper身份为准；没有元数据时不能满足明确日期/会议条件。
                    metadata = row["metadata"] if row else {}
                    if ((not self.db.task_has_paper(child_id, paper_id) and paper_id not in read_ids)
                            or not self.in_scope(metadata, expected[item.goal_id], plan, inputs, paper_id)):
                        continue
                    if expected[item.goal_id]["requires_fulltext"] and paper_id not in read_ids:
                        continue
                    matched.add(paper_id)
            if item.status == "satisfied" and len(matched) < expected[item.goal_id]["min_papers"]:
                if not item.subtask_ids and children:
                    raise ValueError(f"goal {item.goal_id} missing subtask_ids; bind supporting existing child IDs first: {list(children)}. Do not retrieve again merely because the IDs were omitted.")
                raise ValueError(f"goal {item.goal_id} lacks qualified papers; retrieve more or report unresolved")
            if item.status == "unresolved":
                gaps.append(expected[item.goal_id]["description"] + "：" + item.reason)
            paper_ids.update(matched)
            resolved.append({**item.model_dump(), "verified_paper_ids": sorted(matched)})
        count = plan.get("requested_count")
        if count and len(paper_ids) < count:
            gaps.append(f"要求{count}篇，当前满足目标及范围的不同论文为{len(paper_ids)}篇")
        draft = self.db.get_draft(payload.draft_id) if payload.draft_id else None
        if payload.draft_id and (not draft or draft["task_id"] != task_id):
            raise ValueError("draft_id must belong to this task")
        answer = draft["markdown"] if draft else payload.answer_markdown
        if not answer or not answer.strip():
            raise ValueError("deliver an answer or retained findings, even when partial")
        receipt = self.db.get_receipt(draft["draft_id"], draft["content_hash"]) if draft else None
        from research_loop.services.notes import NoteService
        # 聊天答复和笔记附件是不同产物；只用当前稿的真实、完整收据满足保存目标。
        artifacts = NoteService(self.db, None).artifacts(task_id)
        if plan.get("save_requested") and not receipt and not any(a["is_current_draft"] and a["integrity"] == "verified" for a in artifacts):
            gaps.append("要求保存的当前笔记尚无成功文件收据")
        for action in self.db.list_actions(task_id):
            if action["tool_name"] == "report_task_gap" and action["status"] == "succeeded":
                gaps.extend(action["observation"].get("data", {}).get("gaps", []))
        matrix = None
        if payload.matrix_hash:
            from research_loop.services.evidence_matrix import EvidenceMatrixService
            matrix = EvidenceMatrixService(self.db).delivery_ref(task_id, payload.matrix_hash)
        return {"answer_markdown": answer, "draft_id": payload.draft_id, "note_artifacts": artifacts,
                "evidence_matrix": matrix,
                "saved_path": receipt["path"] if receipt else None,
                "coverage": {"plan": plan, "goals": resolved, "paper_ids": sorted(paper_ids)},
                "gaps": gaps, "status": "partial" if gaps else "completed"}
