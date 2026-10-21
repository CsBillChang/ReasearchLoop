"""研究目标覆盖：复用行动日志保存计划，核对交付事实，不进行科学事实审校。"""
from __future__ import annotations

import calendar
import hashlib
import json
from copy import deepcopy
from datetime import date

from research_loop.domain.tasks import ResearchBrief, ResearchGoal, FinishTaskInput


class CoverageEligibilityError(ValueError):
    """携带局部完成资格诊断；不把对象权限错误当作可保留候选。"""

    def __init__(self, message: str, diagnostic: dict, candidate: dict | None = None):
        """保存经服务核对的缺口与可选行动身份，供调度工具消费。"""
        super().__init__(message)
        self.diagnostic = diagnostic
        self.candidate = candidate


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

    def normalize_references(self, task_id: str, spec: dict) -> dict:
        """只把当前任务唯一绑定的附件别名规范为input_ref，不授予新权限。

        Args:
            task_id: 持有附件或论文授权的父任务。
            spec: 模型分派参数；不原地修改旧行动记录。

        Returns:
            引用去重、排序后的副本，可用于稳定幂等键。

        Raises:
            PermissionError: 外来引用、附件已失权或同一别名绑定多个版本。
        """
        task = self.db.get_task(task_id)
        attachments = json.loads(task['attachments_json'])
        inputs = list(spec.get('input_refs') or [])
        papers = []
        for ref in spec.get('paper_ids') or []:
            matches = [a['input_ref'] for a in attachments if a.get('paper_id') == ref]
            if len(set(matches)) > 1:
                raise PermissionError('ambiguous attachment paper_id; use exact input_ref: ' + ref)
            if matches:
                inputs.append(matches[0])
            elif self.db.task_has_paper(task_id, ref):
                papers.append(ref)
            else:
                raise PermissionError('cannot delegate an unauthorized paper: ' + ref)
        bound = {a['input_ref'] for a in attachments}
        for ref in inputs:
            if ref not in bound or not self.db.task_has_input(task_id, ref):
                raise PermissionError('cannot delegate an unauthorized attachment: ' + ref)
        return {**spec, 'paper_ids': sorted(set(papers)), 'input_refs': sorted(set(inputs))}

    def source_coverage(self, task_id: str) -> list[dict]:
        """只累计成功观察实际返回且当前仍获授权的原文位置。

        Args:
            task_id: 实际执行读取或检索的角色任务。

        Returns:
            版本、页或片段范围及原行动身份。片段不升级为全篇阅读。
            空命中、仅索引就绪、失效版本及已撤销权限均不计入。
        """
        from research_loop.services.papers import PaperService
        papers = PaperService(self.db, None, None)
        output = []
        for action in self.db.list_actions(task_id):
            if action['status'] != 'succeeded' or action['tool_name'] not in {
                    'read_paper', 'acquire_paper_pdf', 'retrieve_evidence'}:
                continue
            observation = action.get('observation') or {}
            if not observation.get('ok'):
                continue
            data = observation.get('data') or {}
            for key, kind in (('pages', 'page'), ('hits', 'hit')):
                for item in data.get(key) or []:
                    try:
                        validated = papers.validate_source_observation(task_id, item, data, kind=kind)
                    except (PermissionError, ValueError, KeyError, TypeError):
                        # 旧观察保留原样；完成门只接受当前仍可验证的实际内容。
                        continue
                    reference = validated['reference']
                    identity = validated['display_identity']
                    output.append({**reference, 'paper_id': identity.get('paper_id'),
                        'display_identity': identity, 'action_id': action['action_id'],
                        'source_task_id': task_id, 'tool_name': action['tool_name'],
                        'material_scope': 'retrieved_chunk' if kind == 'hit' else 'returned_page',
                        'char_start': item.get('char_start') if kind == 'hit' else None,
                        'char_end': item.get('char_end') if kind == 'hit' else None,
                        'total_physical_pages': validated['document'].get('total_pages'),
                        'full_document_returned': False if kind == 'hit' else None})
        for source in output:
            if source['material_scope'] != 'returned_page':
                continue
            total = source['total_physical_pages']
            actual = {row['page_number'] for row in output if row['action_id'] == source['action_id']
                      and row['document_id'] == source['document_id'] and row['material_scope'] == 'returned_page'}
            source['full_document_returned'] = actual == set(range(1, total + 1)) if total else None
        return output

    def distinct_papers(self, paper_ids) -> set[str]:
        """仅按精确身份或已证明的明确同版别名去重，不按题名或内容hash合并。"""
        from research_loop.services.papers import PaperService
        papers = PaperService(self.db, None, None)
        unique = set()
        for paper_id in sorted(set(paper_ids)):
            if not any(papers.paper_alias_matches(paper_id, prior) for prior in unique):
                unique.add(paper_id)
        return unique

    def goal_achievements(self, task_id: str, plan: dict | None, progress) -> list[dict]:
        """按精确目标ID展示同根成果与未处理范围，不推断研究问题语义等价。

        Args:
            task_id: 当前根任务。
            plan: 当前成功计划，可能尚未建立。
            progress: 共同进展服务，用于读取有效发现和版本状态。

        Returns:
            每个目标的候选成果、材料范围、停止原因及已记录缺口；是否
            足以完成目标仍由Supervisor选择，候选目录本身不授予原文权限。
        """
        children = self.db.list_subtasks(task_id)
        records = []
        for goal in (plan or {}).get('goals', []):
            if goal.get('status', 'active') != 'active':
                continue
            candidates = []
            for child in children:
                if (self.db.get_task(child['subtask_id']) or {}).get('role') != 'researcher':
                    continue
                spec = child['spec']
                retained = progress.retained(child['subtask_id'])
                # 不同目标也展示为待选择材料，但不能冒充当前目标已完成。
                path = 'current_task.subtasks[subtask_id=' + child['subtask_id'] + ']'
                actual = self.source_coverage(child['subtask_id'])
                candidates.append({'subtask_id': child['subtask_id'],
                    'same_goal_id': goal['goal_id'] in (spec.get('goal_ids') or []),
                    'status': child['status'],
                    'finding_refs': [{'finding_id': f['finding_id'], 'revision': f.get('revision'),
                        'source_integrity': f.get('source_integrity')} for f in retained['findings']],
                    'findings_in': path + '.research_progress',
                    'scope_in': path + '.scope',
                    'material_coverage_in': path + '.material_coverage',
                    'material_summary': {'source_count': len(actual),
                        'document_ids': sorted({source['document_id'] for source in actual}),
                        'scope_kinds': sorted({source['material_scope'] for source in actual})},
                    'stop_reason_in': path + '.stop_reason', 'gaps_in': path + '.gaps'})
            records.append({'goal_id': goal['goal_id'], 'reuse_candidates': candidates,
                'unaddressed_scope': {'plan_goal_id': goal['goal_id'], 'semantic_completion': 'not_assessed',
                    'notice': '已有成果不自动证明当前问题完成；根据具体发现及缺口选择复用、继续或partial交付。'}})
        return records

    def current_context(self, task_id: str, state: dict) -> dict:
        """从持久业务记录装配不可被旧工具摘要替代的任务状态。

        Args:
            task_id: 当前根任务身份，不读取其他任务的授权页。
            state: 当前图状态，仅用于定位候选稿和待执行调用；权限以库为准。

        Returns:
            原始要求、有效澄清、计划、当前稿身份及已授权证据位置。
            页索引不是原文，也不因出现在上下文中而授予新权限。
        """
        from research_loop.services.research_progress import ResearchProgressService
        from research_loop.agents.reflection import ReflectionService
        progress = ResearchProgressService(self.db)
        task = self.db.get_task(task_id)
        if not task:
            raise ValueError("CONTEXT_TASK_MISSING: current task is unavailable")
        plan = self.plan(task_id)
        answer = self._latest_answer(task_id)
        # 成功行动可能早于checkpoint落盘；新请求优先沿用已落库的新稿身份。
        draft_id = state.get("draft_id")
        for action in reversed(self.db.list_actions(task_id)):
            candidate = (action.get("observation") or {}).get("data", {}).get("draft_id")
            if action["status"] == "succeeded" and candidate and action["tool_name"] in {"create_note", "revise_note"}:
                draft_id = candidate
                break
        draft = self.db.get_draft(draft_id) if draft_id else None
        draft_valid = bool(draft and draft["task_id"] == task_id)
        receipt = self.db.get_receipt(draft_id, draft["content_hash"]) if draft_valid else None
        refs = set(self.db.list_task_evidence(task_id))
        pages = []
        unavailable = set(state.get("evidence_ids") or []) - refs
        from research_loop.services.papers import PaperService
        papers = PaperService(self.db, None, None)
        for ref in sorted(refs):
            try:
                pages.append(papers.authorized_source(task_id, ref, with_readback=False)['reference'])
            except (PermissionError, ValueError):
                unavailable.add(ref)
        return {"task_id": task_id, "user_request": task["prompt"],
                "save_policy": task["save_policy"], "latest_clarification": answer,
                "research_progress": progress.retained(task_id),
                "research_reviews": ReflectionService(self.db).research_context(task_id),
                "research_plan": plan,
                "goal_achievements": self.goal_achievements(task_id, plan, progress),
                "material_coverage": self.source_coverage(task_id),
                "plan_needs_update": bool(answer and (not plan or plan.get("clarification_id") != answer["clarification_id"])),
                "attachments": [{**{k: a.get(k) for k in ("input_ref", "paper_id", "filename")},
                                 "delegate_using": {"input_refs": [a['input_ref']]},
                                 "notice": "paper_id仅为附件别名；用当前input_ref委派及读取"}
                                for a in json.loads(task["attachments_json"])],
                "current_draft": ({k: draft[k] for k in ("draft_id", "content_hash", "status", "evidence_ids")}
                                  if draft_valid else None),
                "unavailable_draft_id": draft_id if draft_id and not draft_valid else None,
                "save_status": task.get("save_status"),
                "save_receipt": ({k: receipt.get(k) for k in ("receipt_id", "draft_id", "content_hash", "status")}
                                 if receipt else None),
                "subtasks": [{"subtask_id": child["subtask_id"], "status": child["status"],
                              "goal_ids": child["spec"].get("goal_ids", []),
                              "research_progress": progress.retained(child["subtask_id"]),
                              "scope": {k: child['spec'].get(k) for k in ('mode', 'paper_ids', 'input_refs',
                                  'sources', 'date_from', 'date_to', 'start_page', 'page_count', 'comparison_dimensions', 'verification_claims')},
                              "material_coverage": self.source_coverage(child['subtask_id']),
                              "gaps": list(dict.fromkeys([*progress.retained(child['subtask_id'])['gaps'],
                                  *((((child.get('result') or {}).get('result') or {}).get('gaps')) or [])])),
                              # 旧往返压缩后仍保留真实停止原因；已读身份不参与完成门判断。
                              "stop_reason": ((child.get('result') or {}).get('result') or {}).get('stop_reason'),
                              "read_paper_ids": ((child.get('result') or {}).get('result') or {}).get('read_paper_ids', []),
                              "objective": child["objective"]} for child in self.db.list_subtasks(task_id)],
                "pending_tool_calls": [{k: c.get(k) for k in ("call_id", "name", "arguments")}
                                       for c in state.get("pending_tool_calls", [])],
                "authorized_evidence": pages, "unavailable_evidence_ids": sorted(unavailable),
                "notice": "当前任务/权限/目标ID以本记录为准。位置索引不是原文内容；旧摘要不替代证据。"}

    def prepare_plan(self, task_id: str, brief: ResearchBrief) -> dict:
        """保护用户要求，同时允许有理由地调整模型自拟研究步骤。

        Args:
            task_id: 根任务身份。
            brief: 模型从原始指令解释的目标；解释本身不冒充程序语义验证。

        Returns:
            可放入工具观察的计划，保存权限以数据库为准。
        """
        if not brief.goals:
            raise ValueError("research plan requires at least one observable goal")
        active = [g for g in brief.goals if g.status == "active"]
        if not any(g.origin != "strategy" for g in active):
            raise ValueError("research plan requires at least one active user or legacy goal")
        if brief.task_kind != "memory":
            # 汇总/局部改写子目标不额外索取一篇论文；用户交付仍须有材料目标。
            # min_papers计不同材料，不要求为了修改旧稿去找“新论文”。
            # strategy可未完成，不能单独承担必需材料门而使零取材被判completed。
            if not any(g.min_papers > 0 and g.origin != "strategy" for g in active):
                raise ValueError("研究计划至少需要一个active user/legacy且min_papers>0的材料目标；纯历史回顾使用memory")
            if any(g.min_papers == 0 and g.requires_fulltext for g in active):
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
        # 归一副本以比较旧默认字段，绝不回写已持久化的历史行动。
        previous = deepcopy(self.plan(task_id))
        if previous:
            previous["goals"] = [ResearchGoal.model_validate(g).model_dump(mode="json") for g in previous["goals"]]
            self._normalize_input_scopes(previous, attachments)
        old = {g["goal_id"]: g for g in previous["goals"]} if previous else {}
        new = {g["goal_id"]: g for g in value["goals"]}
        answer = self._latest_answer(task_id)
        # 用户回答由resume事务持久化，模型不能伪造回答ID来获取改计划权限。
        # 同一回答只触发一次用户范围重定；自拟步骤始终可带理由调整。
        answered_revision = bool(answer and (not previous or previous.get("clarification_id") != answer["clarification_id"]))
        for key, goal in new.items():
            if goal["origin"] != "user" or old.get(key) == goal:
                continue
            sources = [task["prompt"], answer["answer"] if answer else ""]
            if not any(goal["source_quote"] in source for source in sources):
                raise ValueError("user goal source_quote must occur verbatim in the actual user request or resumed answer")
        # 步骤退出仍留在新计划中；历史行动保存前版，理由保存本次决定。
        for key, goal in old.items():
            if goal["origin"] == "strategy":
                if key not in new:
                    raise ValueError("strategy goals cannot be removed; retain goal_id with status=withdrawn and change_reason")
                replacement = new[key]
                if replacement != goal and not (replacement["change_reason"] and replacement["change_reason"].strip()):
                    raise ValueError("changing a strategy goal requires change_reason")
        if previous and not answered_revision:
            for key, goal in old.items():
                replacement = new.get(key)
                if goal["origin"] == "strategy":
                    if replacement["origin"] != "strategy":
                        raise ValueError("existing strategy goal origin cannot be changed without a new user answer")
                elif replacement != goal:
                    raise ValueError(f"existing goals cannot be removed or weakened; preserve user/legacy goal {key} and its constraints, or report unresolved")
            protected = ["requested_count", "count_confirmed", "save_requested", "deep_reading_requested", "input_refs", "task_kind"]
            if previous.get("_scope_semantics") != "goal_owned_constraints":
                protected.extend(["date_from", "date_to", "sources"])
            for key in protected:
                if previous.get(key, False if key == "deep_reading_requested" else None) != value.get(key):
                    raise ValueError("existing task constraint cannot be changed: " + key)
        # 标记只由服务端写入成功计划观察；无标记旧任务沿用旧硬范围。
        # 真实resume可重定合同，但普通prepare不能默默迁移历史计划。
        if not previous or previous.get("_scope_semantics") == "goal_owned_constraints" or answered_revision:
            value["_scope_semantics"] = "goal_owned_constraints"
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
        known = {g["goal_id"] for g in plan["goals"] if g.get("status", "active") == "active"}
        for spec in specs:
            ids = spec.get("goal_ids") or []
            if not ids or not set(ids).issubset(known):
                raise ValueError(f"subtask goal_ids must reference the current research plan; valid goal_ids: {sorted(known)}; do not rename the plan to match invented IDs")
            if spec.get("mode", "deep_read") == "deep_read" and not plan.get("deep_reading_requested", False):
                raise ValueError("论文 deep_read requires a user-requested deep-reading note; use investigate for full-text analysis/comparison, or screen for search")

    @staticmethod
    def scope_failures(meta: dict, goal: dict, plan: dict, inputs: dict | None = None,
                       paper_id: str | None = None) -> list[dict]:
        """按条件归属核对实际事实；检索偏好不能剔除已授权上传原文。

        Args:
            meta: 实际题录元数据；缺项保持未知，不合成上传来源或日期。
            goal: 当前目标，其来源和日期是该目标的资格条件。
            plan: 服务端已持久化计划；无标记旧计划保留原范围。
            inputs: 根附件到实际同版身份的授权映射。
            paper_id: 当前材料的显示身份。

        Returns:
            不满足的具体条件和实际值；空列表表示范围满足。
        """
        failures = []
        modern = plan.get('_scope_semantics') == 'goal_owned_constraints'
        for owner, scope in (('plan', plan), ('goal', goal)):
            origin = scope.get('origin', 'legacy') if owner == 'goal' else ('strategy' if modern else 'legacy')
            raw_sources = scope.get('sources') or []
            refs = (scope.get('input_refs') or []) + [s.split(':', 1)[1] for s in raw_sources if s.startswith('input_ref:')]
            allowed = set()
            for ref in refs:
                value = (inputs or {}).get(ref)
                allowed.update(value if isinstance(value, (set, list, tuple)) else [value])
            if refs and (paper_id is None or paper_id not in allowed):
                failures.append({'scope': owner, 'origin': origin, 'field': 'input_refs',
                                 'expected': refs, 'actual': paper_id, 'fact': 'outside_authorized_input_scope'})
            if modern and owner == 'plan':
                continue
            sources = [s for s in raw_sources if not s.startswith('input_ref:')]
            if sources and str(meta.get('source', '')).lower() not in {s.lower() for s in sources}:
                failures.append({'scope': owner, 'origin': origin, 'field': 'sources',
                                 'expected': sources, 'actual': meta.get('source'),
                                 'fact': 'source_unknown' if not meta.get('source') else 'source_mismatch'})
            start, end = scope.get('date_from'), scope.get('date_to')
            if not start and not end:
                continue
            published = meta.get('published')
            try:
                value = (published or '')[:10]
                if len(value) == 4:
                    low, high = value + '-01-01', value + '-12-31'
                elif len(value) == 7:
                    year, month = map(int, value.split('-'))
                    low, high = value + '-01', value + f'-{calendar.monthrange(year, month)[1]:02d}'
                else:
                    low = high = date.fromisoformat(value).isoformat()
                # 题录缺少可靠日历值时保持未知，四位占位符也不能满足单侧时间窗。
                date.fromisoformat(low)
                date.fromisoformat(high)
                valid = not ((start and low < start) or (end and high > end))
                fact = 'published_outside_range'
            except (ValueError, TypeError):
                valid, fact = False, 'published_unknown'
            if not valid:
                failures.append({'scope': owner, 'origin': origin, 'field': 'published',
                                 'expected': {'date_from': start, 'date_to': end},
                                 'actual': published, 'fact': fact})
        return failures

    @staticmethod
    def in_scope(meta: dict, goal: dict, plan: dict, inputs: dict | None = None, paper_id: str | None = None) -> bool:
        """复用带条件归属的范围核对；未知硬来源或日期仍不能满足。"""
        return not CoverageService.scope_failures(meta, goal, plan, inputs, paper_id)

    def _qualification_environment(self, task_id: str, reviews: list[dict]) -> dict:
        """汇集完成门共用的实际授权材料，不以子任务ID或新行动计研究进展。"""
        from research_loop.services.papers import PaperService
        papers = PaperService(self.db, None, None)
        children = {c['subtask_id']: c for c in self.db.list_subtasks(task_id)}
        sources = {owner: self.source_coverage(owner) for owner in [task_id, *children]}
        inputs = {a['input_ref']: a.get('paper_id') or a['input_ref']
                  for a in json.loads(self.db.get_task(task_id)['attachments_json'])}
        actual_ids = {source['paper_id'] for rows in sources.values() for source in rows if source['paper_id']}
        for ref, paper_id in inputs.items():
            aliases = {current for current in actual_ids if papers.paper_alias_matches(paper_id, current)}
            if aliases:
                inputs[ref] = aliases
        inherited = []
        bound = next((e['payload'] for e in self.db.list_events(task_id) if e['event_type'] == 'feedback_draft_bound'), None)
        if bound:
            for ref in bound['evidence_ids']:
                try:
                    source = papers.authorized_source(task_id, ref, with_readback=False)['reference']
                    authorized = self.db.list_task_papers(task_id) + [
                        value for values in inputs.values() for value in (values if isinstance(values, set) else [values])]
                    if any(papers.paper_alias_matches(source['paper_id'], allowed) for allowed in authorized):
                        inherited.append(source)
                except (PermissionError, ValueError):
                    continue
        unsuitable = {f['item_id'] for review in reviews if review['target']['target_type'] == 'candidates'
                      for f in review.get('findings', []) if f.get('suitability') == 'unsuitable'}
        return {'papers': papers, 'children': children, 'sources': sources, 'inputs': inputs,
                'inherited': inherited, 'unsuitable': unsuitable}

    def _goal_qualification(self, task_id: str, goal: dict, plan: dict, environment: dict,
                            subtask_ids: list[str]) -> tuple[set[str], list[dict], list[str]]:
        """统一完成判定与阻断指纹的依赖，资格之外的材料变化不解锁。

        Args:
            task_id: 当前根任务。
            goal: 当前有效目标合同。
            plan: 已持久化计划，含服务器内部范围语义。
            environment: 当前实际材料、同根子成果、权限与核验排除记录。
            subtask_ids: 明确选择的支持子任务；指纹评估枚举全部同根可复用成果。

        Returns:
            合格不同论文、具体范围失败以及合格材料版本。子成果只有就绪且
            有发现才计入；重复页、改写objective或不合格来源不改变资格事实。
        """
        papers = environment['papers']
        matched, rejected, versions = set(), [], {}

        def consider(paper_id: str, actual: list[dict]) -> None:
            """只保存满足硬范围的当前版本；缺题录上传仍不伪造元数据。"""
            row = self.db.get_paper(paper_id)
            failures = self.scope_failures((row or {}).get('metadata') or {}, goal, plan,
                                          environment['inputs'], paper_id)
            rejected.extend(failures)
            if failures or any(papers.paper_alias_matches(paper_id, excluded) for excluded in environment['unsuitable']):
                return
            matched.add(paper_id)
            # 完成门按材料身份计数。已经有一页原文后读同材料其他页不改变其资格；
            # 新版本、由题录变实际原文及首次可复用子成果才形成新的资格事实。
            facts = [{key: source.get(key) for key in ('document_id', 'source_sha256')}
                     for source in actual]
            if not facts:
                facts = [{'paper_id': paper_id, 'material': 'authorized_metadata_only'}]
            versions.setdefault(paper_id, set()).update(json.dumps(fact, sort_keys=True, ensure_ascii=False) for fact in facts)

        for source in [*environment['sources'][task_id], *environment['inherited']]:
            if source.get('paper_id'):
                consider(source['paper_id'], [source])
        for child_id in subtask_ids:
            child = environment['children'][child_id]
            result = (child.get('result') or {}).get('result', {})
            if child['status'] not in {'completed', 'partial'} or not result.get('findings'):
                continue
            actual = environment['sources'][child_id]
            read_ids = {source['paper_id'] for source in actual if source['paper_id']}
            for paper_id in set(result.get('paper_ids', [])) | read_ids:
                if not self.db.task_has_paper(child_id, paper_id) and paper_id not in read_ids:
                    continue
                if goal['requires_fulltext'] and paper_id not in read_ids:
                    continue
                consider(paper_id, [source for source in actual if source['paper_id'] == paper_id])
        distinct = self.distinct_papers(matched)
        qualified_versions = set()
        for paper_id in matched:
            facts = versions[paper_id]
            actual = {fact for fact in facts if 'document_id' in json.loads(fact)}
            qualified_versions.update(actual or facts)
        qualified_versions = sorted(qualified_versions)
        return distinct, rejected, qualified_versions

    def qualification_fingerprint(self, task_id: str, goal_id: str) -> str | None:
        """散列与finish相同的资格事实，不因无关读回或子任务改写解锁。

        Args:
            task_id: 当前根任务；所有可复用子成果均与完成门采用同一判断。
            goal_id: 精确有效目标身份。

        Returns:
            硬合同、当前合格数量及版本摘要；目标退出时返回None。未知来源
            仍未知时读新页不解除来源阻断，子成果首次就绪则按实际贡献更新。
        """
        from research_loop.agents.reflection import ReflectionService
        plan = self.plan(task_id)
        goal = next((g for g in (plan or {}).get('goals', []) if g['goal_id'] == goal_id
                     and g.get('status', 'active') == 'active'), None)
        if goal is None:
            return None
        environment = self._qualification_environment(task_id, ReflectionService(self.db).research_context(task_id))
        matched, _, versions = self._goal_qualification(task_id, goal, plan, environment, list(environment['children']))
        fields = ('goal_id', 'origin', 'status', 'min_papers', 'requires_fulltext', 'sources', 'date_from', 'date_to', 'input_refs')
        root_fields = ('input_refs',) if plan.get('_scope_semantics') == 'goal_owned_constraints' else ('input_refs', 'sources', 'date_from', 'date_to')
        value = {'goal': {key: goal.get(key) for key in fields},
                 'root': {key: plan.get(key) for key in root_fields},
                 'qualified_count': len(matched), 'qualified_versions': versions}
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _delivery_candidate(self, task_id: str, payload: FinishTaskInput, action_identity: dict | None) -> tuple:
        """先核对交付对象及正文，只有合法服务器行动才绑定可保留候选。"""
        from research_loop.agents.reflection import ReflectionService
        task = self.db.get_task(task_id)
        if not task:
            raise ValueError('delivery task is unavailable')
        draft = self.db.get_draft(payload.draft_id) if payload.draft_id else None
        if payload.draft_id and (not draft or draft['task_id'] != task_id):
            raise ValueError('draft_id must belong to this task')
        answer = draft['markdown'] if draft else payload.answer_markdown
        if bool(payload.reflection_target_id) != bool(payload.reflection_content_hash):
            raise ValueError('reflection_target_id/hash must be provided together')
        if payload.reflection_target_id:
            record = ReflectionService(self.db).research_target(task_id, payload.reflection_target_id, payload.reflection_content_hash)
            target = record['current']
            if draft or target['target_type'] != 'answer':
                raise ValueError('reflection delivery requires an answer snapshot, without draft_id')
            reviewed_answer = target['items'][0]['text']
            if payload.answer_markdown is not None and payload.answer_markdown != reviewed_answer:
                raise ValueError('answer differs from selected current reflection snapshot; omit answer_markdown '
                                 'when delivering the bound snapshot, put processing notes in processing_summary '
                                 'and unresolved goals in goals[].reason; use revise_research for actual text changes')
            answer = reviewed_answer
        if not answer or not answer.strip():
            raise ValueError('deliver an answer or retained findings, even when partial')
        if payload.matrix_hash:
            from research_loop.services.evidence_matrix import EvidenceMatrixService
            EvidenceMatrixService(self.db).delivery_ref(task_id, payload.matrix_hash)
        children = {c['subtask_id'] for c in self.db.list_subtasks(task_id)}
        if any(child not in children for item in payload.goals for child in item.subtask_ids):
            raise ValueError('foreign subtask_id')
        candidate = None
        if (isinstance(action_identity, dict) and action_identity.get('task_id') == task_id
                and action_identity.get('action_id') and action_identity.get('input_hash')):
            candidate = {**{key: action_identity[key] for key in ('task_id', 'action_id', 'input_hash')},
                'answer_body_chars': len(answer), 'processing_summary_chars': len(payload.processing_summary or ''),
                'draft_id': payload.draft_id, 'reflection_target_id': payload.reflection_target_id,
                'reflection_content_hash': payload.reflection_content_hash,
                'answer_content_hash': hashlib.sha256(answer.encode()).hexdigest()}
        return draft, answer, candidate

    def resolve_finish_candidate(self, task_id: str, action: dict, candidate: dict) -> dict:
        """从精确失败行动读回合法正文，不在工具观察重复保存整篇回答。

        Args:
            task_id: 当前根任务。
            action: 已持久化的明确失败finish_task行动。
            candidate: 该行动观察中由服务验证的候选身份与正文hash。

        Returns:
            重新核对对象权限、版本及正文hash后的正文和独立处理说明。

        Raises:
            ValueError: 行动身份、原参数、对象版本或正文与候选不符。
        """
        if (action.get('task_id') != task_id or action.get('tool_name') != 'finish_task'
                or action.get('status') != 'failed' or candidate.get('task_id') != task_id
                or candidate.get('action_id') != action.get('action_id')):
            raise ValueError('finish candidate action identity mismatch')
        arguments = action.get('arguments')
        if not isinstance(arguments, dict):
            raise ValueError('finish candidate original arguments unavailable')
        input_hash = hashlib.sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if candidate.get('input_hash') != input_hash:
            raise ValueError('finish candidate arguments hash changed')
        payload = FinishTaskInput.model_validate(arguments)
        _, answer, actual = self._delivery_candidate(task_id, payload, {
            'task_id': task_id, 'action_id': action['action_id'], 'input_hash': input_hash})
        for key in ('task_id', 'action_id', 'input_hash', 'answer_content_hash', 'draft_id',
                    'reflection_target_id', 'reflection_content_hash', 'answer_body_chars', 'processing_summary_chars'):
            if candidate.get(key) != actual.get(key):
                raise ValueError('finish candidate identity or content changed: ' + key)
        return {**actual, 'answer_body_markdown': answer, 'processing_summary': payload.processing_summary}

    def finish(self, task_id: str, payload: FinishTaskInput, *, action_identity: dict | None = None) -> dict:
        """验证覆盖报告并形成交付；不调用模型、不保存或修改笔记。

        Args:
            task_id: 当前根任务身份。
            payload: Supervisor最终正文/草稿以及逐目标结论。
            action_identity: 图节点注入的实际行动身份，模型不能提供。

        Returns:
            正文、覆盖记录、未完成项和可观察的completed/partial判断。

        Raises:
            ValueError: 目标漏报、身份越界或声称已覆盖而缺乏执行证据。
        """
        draft, answer, candidate = self._delivery_candidate(task_id, payload, action_identity)
        plan = self.plan(task_id)
        if not plan:
            raise ValueError("set_research_plan before finish_task")
        self._require_answered_plan(task_id, plan)
        expected = {g["goal_id"]: g for g in plan["goals"] if g.get("status", "active") == "active"}
        if len(payload.goals) != len(expected) or {g.goal_id for g in payload.goals} != set(expected):
            actual = [g.goal_id for g in payload.goals]
            raise ValueError("finish_task must account for every goal exactly once; "
                             f"expected goal_ids={list(expected)}, received={actual}. "
                             "Use the current plan IDs; report unfinished goals as unresolved. Do not reread history to guess IDs.")
        from research_loop.agents.reflection import ReflectionService
        reviews = ReflectionService(self.db).research_context(task_id)
        environment = self._qualification_environment(task_id, reviews)
        children = environment['children']
        sources_by_owner = environment['sources']
        papers = environment['papers']
        if plan.get("task_kind") == "memory" and not any(a["status"] == "succeeded" and a["tool_name"] in {"remember_preference", "forget_preference", "search_memory", "read_memory", "get_profile", "update_saved_note"} for a in self.db.list_actions(task_id)):
            raise ValueError("memory任务必须有真实记忆操作或读回，不能只声称已经记住")
        resolved, gaps, paper_ids = [], [], set()
        for item in payload.goals:
            matched, rejected, _ = self._goal_qualification(task_id, expected[item.goal_id], plan,
                                                           environment, item.subtask_ids)
            if item.status == "satisfied" and len(matched) < expected[item.goal_id]["min_papers"]:
                missing_bindings = not item.subtask_ids and bool(children)
                unique_failures = {json.dumps(value, sort_keys=True, ensure_ascii=False): value for value in rejected}
                actions = ['bind_existing_subtask_ids'] if missing_bindings else []
                if any(f['origin'] == 'strategy' for f in rejected):
                    actions.append('revise_or_withdraw_strategy_with_change_reason')
                if any(f['origin'] in {'user', 'legacy'} for f in rejected):
                    actions.append('provide_verified_metadata_or_matching_authorized_material')
                if not rejected and not missing_bindings:
                    actions.append('obtain_actual_authorized_source_content')
                actions.append('report_goal_unresolved_with_specific_reason')
                diagnostic = {'code': 'GOAL_SUPPORT_BINDING_MISSING' if missing_bindings else 'GOAL_QUALIFICATION_BLOCKED',
                    'goal_id': item.goal_id, 'conditions': list(unique_failures.values()),
                    'missing_facts': {'required_papers': expected[item.goal_id]['min_papers'], 'qualified_papers': len(matched),
                                      'supporting_subtask_ids_missing': missing_bindings},
                    'suggested_actions': actions, 'state_fingerprint': self.qualification_fingerprint(task_id, item.goal_id)}
                message = (f'goal {item.goal_id} missing subtask_ids; bind supporting existing child IDs first: {list(children)}. '
                           'Do not retrieve again merely because the IDs were omitted.' if missing_bindings else
                           f'goal {item.goal_id} lacks qualified papers; inspect coverage_blocker conditions and correct their ownership or report unresolved')
                raise CoverageEligibilityError(message, diagnostic, candidate)
            if item.status == "unresolved" and expected[item.goal_id].get("origin", "legacy") != "strategy":
                gaps.append(expected[item.goal_id]["description"] + "：" + item.reason)
            # 可选步骤的资料不能替代用户交付数量，也不因步骤缺证制造用户缺口。
            if expected[item.goal_id].get("origin", "legacy") != "strategy":
                paper_ids.update(matched)
            resolved.append({**item.model_dump(), "origin": expected[item.goal_id].get("origin", "legacy"),
                             "verified_paper_ids": sorted(matched),
                             "material_coverage": [source for owner in [task_id, *item.subtask_ids]
                                 for source in sources_by_owner[owner] if source['paper_id'] and
                                 any(papers.paper_alias_matches(source['paper_id'], paper_id) for paper_id in matched)]})
        paper_ids = self.distinct_papers(paper_ids)
        count = plan.get("requested_count")
        if count and len(paper_ids) < count:
            gaps.append(f"要求{count}篇，当前满足目标及范围的不同论文为{len(paper_ids)}篇")
        from research_loop.agents.reflection import ReflectionService
        reviews = ReflectionService(self.db).research_context(task_id)
        # 仅已选择的核验结果随交付展示；不强制其他任务调用Reviewer。
        answer_body = answer
        notices = []
        for review in reviews:
            if review.get('quality_status') != 'supported':
                decision = (review.get('decisions') or [{}])[-1]
                notices.append({'target_id': review['target_id'], 'quality_status': review['quality_status'],
                    'goal_ids': review['target'].get('goal_ids', []), 'issues': review.get('issues', []),
                    'disposition': decision})
                if decision.get('action') != 'exclude':
                    pending = review.get('pending_checks', [])
                    execution_errors = [c.get('execution_error') for c in pending if c.get('execution_error')]
                    reasons = [f.get('reason', '') for check in pending for f in check.get('findings', [])]
                    reason = decision.get('reason') or '；'.join(dict.fromkeys(r for r in reasons if r)) or '所选内容仍有待核实事项'
                    if execution_errors:
                        # 技术诊断来自持久行动；模型的结束说明不能替代实际错误因果。
                        details = []
                        for error in execution_errors:
                            first = error.get('first_error')
                            details.append((f"首个工具错误[{first['tool_name']}]：{first['message']}；" if first else '')
                                           + '最后停止：' + error['stop_reason'])
                        reason = '独立核验执行未完成，已有正文保留；' + '；'.join(details) + '。未获得语义结论，不能据此认定原文缺证或答案错误。'
                    notices[-1]['message'] = reason
                    answer += '\n\n核验范围说明：' + reason
                    # 仅显式绑定的必要目标受此缺口影响；可选材料局限不强制整项失败。
                    for goal in resolved:
                        if goal['goal_id'] in review['target'].get('goal_ids', []):
                            goal['status'] = 'unresolved'
                            goal['reason'] = reason
                            if expected[goal['goal_id']].get('origin', 'legacy') != 'strategy':
                                gaps.append(expected[goal['goal_id']]['description'] + '：' + reason)
        if not answer or not answer.strip():
            raise ValueError('deliver readable retained work, including when reflection is incomplete')
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
        summary = payload.processing_summary
        if summary and summary.strip():
            answer += '\n\n处理说明：' + summary
        return {"answer_markdown": answer, "answer_body_markdown": answer_body,
                "processing_summary": summary, "draft_id": payload.draft_id, "note_artifacts": artifacts,
                "evidence_matrix": matrix, "research_reviews": reviews, "review_notices": notices,
                "saved_path": receipt["path"] if receipt else None,
                "coverage": {"plan": plan, "goals": resolved, "paper_ids": sorted(paper_ids)},
                "gaps": gaps, "status": "partial" if gaps else "completed"}
