"""执行编排：领取任务、维护租约、组装各角色工具、运行 LangGraph，再持久化用户可查询的结果。"""
from __future__ import annotations

from research_loop.agents.reflection import ReflectionService

import json
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from research_loop.agents.graph import build_graph, initial_state
from research_loop.agents.researcher import build_researcher_graph, researcher_state
from research_loop.agents.reviewer import build_reviewer_graph, reviewer_state
from research_loop.config import Settings
from research_loop.domain.tasks import SavePolicy, TaskResult, TaskStatus
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.providers.discovery import DiscoveryProvider
from research_loop.providers.pdf import PDFProvider
from research_loop.services.notes import NoteService
from research_loop.services.coverage import CoverageService
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.tools.notes import specs as note_specs
from research_loop.tools.delegation import specs as delegation_specs
from research_loop.tools.papers import specs as paper_specs
from research_loop.tools.registry import ToolRegistry


class TaskExecutor:
    def __init__(self, settings: Settings, db: Database | None = None,
                 provider_factory: Callable[[Settings], Any] | None = None,
                 pdf_factory: Callable[[Settings, FileStore], Any] | None = None,
                 *, supervisor_tools: set[str] | None = None):
        """注入依赖；隔离验收可收缩根角色工具，日常入口缺省不变。

        Args:
            settings: 当前实例的配置。
            db: 可选的隔离或生产数据库。
            provider_factory: 模型适配器工厂。
            pdf_factory: 原文适配器工厂。
            supervisor_tools: 根角色可见且可执行的工具名集合；不增加权限，
                使用者负责在验收清单中冻结。None不收缩日常工具。
        """
        self.settings = settings
        self.db = db or Database(settings.database_url)
        self.db.ensure_schema()
        self.provider_factory = provider_factory or self._default_provider
        self.pdf_factory = pdf_factory
        self.supervisor_tools = frozenset(supervisor_tools) if supervisor_tools is not None else None
        from research_loop.services.retrieval import RetrievalRuntime
        self.retrieval = RetrievalRuntime(self.db, settings)

    def _run_research_batch(self, parent_task_id: str, specs: list[dict[str, Any]],
                            worker_id: str, lease_token: str, settings: Settings,
                            papers: PaperService) -> list[dict[str, Any]]:
        """校验分派范围并持久化子任务；最多并行两个 Researcher，复用已完成结果并回传证据权限。"""
        root = self.db.get_task(parent_task_id).get("root_task_id") or parent_task_id
        parent = self.db.get_task(parent_task_id)
        attachments = {a["input_ref"]: a for a in json.loads(parent["attachments_json"])}
        children = []
        for position, spec in enumerate(specs):
            for ref in spec.get("paper_ids") or []:
                if not self.db.task_has_paper(parent_task_id, ref):
                    raise PermissionError("cannot delegate an unauthorized paper: " + ref)
            for ref in spec.get("input_refs") or []:
                if ref not in attachments or not self.db.task_has_input(parent_task_id, ref):
                    raise PermissionError("cannot delegate an unauthorized attachment: " + ref)
            key = "research:" + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            existing = self.db.fetchone("SELECT * FROM subtasks WHERE parent_task_id=? AND idempotency_key=?", (parent_task_id, key))
            next_position = len([c for c in self.db.list_subtasks(parent_task_id) if c["position_no"] != 999])
            child = existing or self.db.create_subtask(parent_task_id, root, next_position, spec["objective"], spec, key)
            for ref in spec.get("paper_ids") or []:
                self.db.grant_paper(child["subtask_id"], ref, parent_task_id)
            for ref in spec.get("input_refs") or []:
                self.db.grant_input(child["subtask_id"], ref, parent_task_id)
            # 子任务只持有获分派的附件，不继承父任务全部对话或无关附件。
            self.db.execute("UPDATE tasks SET attachments_json=? WHERE task_id=?", (json.dumps([attachments[r] for r in spec.get("input_refs") or []]), child["subtask_id"]))
            children.append((child, spec))

        def run_one(item: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
            """为一个子任务构造独立状态和工具，运行后先检查租约再提交结果；异常保留为失败观察。"""
            child, spec = item
            if child.get("status") in {"completed", "partial", "failed", "cancelled"} and child.get("result_json"):
                return json.loads(child["result_json"])["result"]
            child_id = child["subtask_id"]
            self.db.update_task(child_id, status="running", claimed_by=worker_id,
                                lease_token=lease_token, lease_until=self.db.get_task(parent_task_id).get("lease_until"))
            self.db.execute("UPDATE subtasks SET status='running',updated_at=? WHERE subtask_id=?", (self.db_now(), child_id))
            provider = self.provider_factory(settings)
            try:
                assigned = {r: attachments[r] for r in spec.get("input_refs") or []}
                state = CheckpointStore(self.db).load(child_id) or researcher_state(child_id, spec["objective"], "deny", assigned, spec)
                state.update({"worker_id": worker_id, "lease_token": lease_token,
                              "parent_task_id": parent_task_id, "root_task_id": root})
                graph = build_researcher_graph(provider, {"papers": papers, "scope": spec, "retrieval": self.retrieval}, self.db, CheckpointStore(self.db),
                                               max_model_requests=min(12, settings.max_model_requests),
                                               max_tool_actions=settings.max_tool_actions,
                                               deep_reading=state.get("research_mode", "deep_read") == "deep_read")
                result_state = graph.invoke(state, config={"recursion_limit": 2 * (settings.max_model_requests + settings.max_tool_actions) + 8})
                evidence_ids = result_state.get("evidence_ids") or []
                for evidence_id in evidence_ids:
                    self.db.grant_evidence(parent_task_id, evidence_id, child_id)
                for paper_id in self.db.list_task_papers(child_id):
                    self.db.grant_paper(parent_task_id, paper_id, child_id)
                structured = result_state.get("structured_result") or {}
                mode = result_state.get("research_mode", "deep_read")
                complete = (bool(result_state.get("final_answer")) and
                            ((result_state.get("deep_reading") or {}).get("status") == "complete"
                             if mode == "deep_read" else structured.get("status") == "completed"))
                status = ("cancelled" if result_state.get("cancelled") or self.db.get_task(child_id).get("cancel_requested")
                          else "completed" if complete else "partial")
                # 搜索授权的候选不等于最终选中或实际读取的论文。
                selected = structured.get("paper_ids") or []
                if not selected and mode == "deep_read":
                    selected = list(dict.fromkeys(a["observation"]["data"]["paper_id"]
                        for a in self.db.list_actions(child_id) if a["tool_name"] == "read_paper"
                        and a["status"] == "succeeded" and a["observation"].get("data", {}).get("paper_id")))
                result = {"subtask_id": child_id, "parent_task_id": parent_task_id,
                          "mode": mode, "goal_ids": spec.get("goal_ids") or [],
                          "objective": spec["objective"], "status": status,
                          "findings": structured.get("findings") or [result_state.get("final_answer") or ""],
                          "evidence_ids": evidence_ids, "paper_ids": selected,
                          "assessments": structured.get("assessments") or [],
                          "evidence_scope": structured.get("evidence_scope") or [],
                          "gaps": result_state.get("gaps") or [],
                          "deep_reading_status": (result_state.get("deep_reading") or {}).get("status") }
                if not self.db.is_owner(child_id, worker_id, lease_token):
                    raise RuntimeError("subtask lease lost before result commit")
                self.db.save_subtask_result(child_id, status, result)
                return result
            except Exception as exc:
                result = {"subtask_id": child_id, "parent_task_id": parent_task_id,
                          "mode": spec.get("mode", "deep_read"), "goal_ids": spec.get("goal_ids") or [],
                          "objective": spec["objective"], "status": "failed", "error": str(exc),
                          "findings": [], "evidence_ids": [], "paper_ids": [], "gaps": [str(exc)]}
                if self.db.get_task(child_id).get("cancel_requested"):
                    result["status"] = "cancelled"
                if self.db.is_owner(child_id, worker_id, lease_token):
                    self.db.save_subtask_result(child_id, result["status"], result, str(exc))
                return result
            finally:
                provider.close()

        with ThreadPoolExecutor(max_workers=min(2, max(1, len(children)))) as pool:
            return list(pool.map(run_one, children))

    def _review_once(self, parent_task_id: str, draft_id: str, content_hash: str,
                     worker_id: str, lease_token: str, settings: Settings,
                     papers: PaperService, notes: NoteService, round_no: int = 0,
                     evidence_ids: list[str] | None = None, questions: list[str] | None = None) -> dict[str, Any]:
        """输入父任务、当前草稿/hash，复用已有审校或执行一次独立 Reviewer；运行失败单列，不制造语义结论。"""
        draft = self.db.get_draft(draft_id)
        if not draft or draft["task_id"] != parent_task_id or draft["content_hash"] != content_hash:
            raise ValueError("draft hash is missing or expired")
        child = self.db.create_subtask(parent_task_id, parent_task_id, 999 + round_no, "independent review", {
            "draft_id": draft_id, "content_hash": content_hash, "round": round_no}, f"review:{round_no}:{draft_id}:{content_hash}")
        review_task_id = child["subtask_id"]
        if child.get("status") in {"failed", "completed"} and child.get("result_json"):
            return json.loads(child["result_json"])["result"]
        self.db.update_task(review_task_id, role="reviewer", status="running", claimed_by=worker_id,
                            lease_token=lease_token, lease_until=self.db.get_task(parent_task_id).get("lease_until"))
        for evidence_id in self.db.list_task_evidence(parent_task_id):
            self.db.grant_evidence(review_task_id, evidence_id, parent_task_id)
        for paper_id in self.db.list_task_papers(parent_task_id):
            self.db.grant_paper(review_task_id, paper_id, parent_task_id)
        parent_attachments = json.loads(self.db.get_task(parent_task_id)["attachments_json"])
        for attachment in parent_attachments:
            self.db.grant_input(review_task_id, attachment["input_ref"], parent_task_id)
        self.db.execute("UPDATE tasks SET attachments_json=? WHERE task_id=?", (json.dumps(parent_attachments), review_task_id))
        provider = self.provider_factory(settings)
        try:
            bundle = papers.evidence_bundle(parent_task_id, evidence_ids if evidence_ids is not None else draft.get("evidence_ids", []))
            state = CheckpointStore(self.db).load(review_task_id) or reviewer_state(review_task_id, draft_id, content_hash, draft["markdown"], evidence_ids if evidence_ids is not None else draft.get("evidence_ids", []), bundle, questions,
                ReflectionService(self.db).revision_checks(parent_task_id, draft_id))
            state["attachments"] = {a["input_ref"]: a for a in parent_attachments}
            state.update({"worker_id": worker_id, "lease_token": lease_token,
                          "parent_task_id": parent_task_id, "root_task_id": parent_task_id})
            reviewer_services = {"papers": papers, "review_draft_id": draft_id,
                                 "retrieval": self.retrieval,
                                 "review_content_hash": content_hash, "review_parent_task_id": parent_task_id,
                                 "review_basis_required": state.get("review_basis_required", False)}
            graph = build_reviewer_graph(provider, reviewer_services, self.db, CheckpointStore(self.db), max_model_requests=4)
            result_state = graph.invoke(state, config={"recursion_limit": 2 * (settings.max_model_requests + settings.max_tool_actions) + 8})
            review_data = next((action["observation"]["data"] for action in reversed(result_state.get("actions", []))
                                if action["tool_name"] == "finish_review" and action["status"] == "succeeded"), None)
            if not review_data:
                error = result_state.get("final_reason") or "reviewer did not finish structurally"
                review_data = {"task_id": parent_task_id, "draft_id": draft_id, "content_hash": content_hash,
                               "status": "execution_failed", "execution_status": "failed",
                               "gaps": ["审校执行失败，不能据此判断草稿正确或缺证：" + error]}
                if self.db.is_owner(review_task_id, worker_id, lease_token):
                    self.db.save_subtask_result(review_task_id, "failed", review_data, error)
                    self.db.update_task(parent_task_id, quality_status="review_failed")
                return review_data
            if not self.db.is_owner(review_task_id, worker_id, lease_token) or not self.db.is_owner(parent_task_id, worker_id, lease_token):
                raise RuntimeError("reviewer/root lease lost before result commit")
            for evidence_id in self.db.list_task_evidence(review_task_id):
                self.db.grant_evidence(parent_task_id, evidence_id, review_task_id)
            notes.save_review(__import__("research_loop.domain.notes", fromlist=["ReviewResult"]).ReviewResult.model_validate(review_data))
            self.db.save_subtask_result(review_task_id, "completed", review_data)
            return review_data
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
            result = {"task_id": parent_task_id, "draft_id": draft_id, "content_hash": content_hash,
                      "status": "execution_failed", "execution_status": "failed", "gaps": ["审校执行失败：" + error]}
            if self.db.is_owner(review_task_id, worker_id, lease_token):
                self.db.save_subtask_result(review_task_id, "failed", result, error)
                self.db.update_task(parent_task_id, quality_status="review_failed")
            return result
        finally:
            provider.close()

    def db_now(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _default_provider(settings: Settings) -> Any:
        return DeepSeekProvider(settings.deepseek_api_key or "", settings.deepseek_model,
                                settings.deepseek_base_url, settings.request_timeout_seconds)

    def execute(self, task_id: str, worker_id: str = "direct") -> TaskResult:
        """领取或恢复指定任务，运行角色图并提交持久结果。

        Args:
            task_id: 已入队的根任务 ID。
            worker_id: 用于租约所有权校验的执行者身份。

        Returns:
            任务交付结果；执行失败或缺证不伪装成完整成功。

        Raises:
            KeyError: 任务不存在。
            RuntimeError: 任务当前不可领取或租约归其他执行者持有。
        """
        task = self.db.get_task(task_id)
        if not task:
            raise KeyError("task not found")
        service = TaskService(self.db)
        # 网页取消queued任务可能已原子结算；Worker收到旧领取请求只读回，不再运行。
        if task["status"] == "cancelled" and task.get("result_json"):
            return TaskResult.model_validate_json(task["result_json"])
        if self.db.has_unknown_model_call(task_id) or self.db.has_unknown_family_call(task_id):
            result = self._deliverable_result(task_id, {}, "已有模型请求结果未知，未自动重发")
            if task["status"] not in {"failed", "completed", "partial", "cancelled"}:
                service.complete(task_id, TaskStatus.partial if result.answer_markdown and result.answer_kind != "failure_report" else TaskStatus.failed, result, result.reason)
            return result
        claimed = self.db.claim_task(worker_id, task_id=task_id)
        if not claimed:
            raise RuntimeError("task is not queued or this worker does not own its lease")
        task = claimed
        lease_token = task.get("lease_token")
        stop_heartbeat = threading.Event()
        def heartbeat():
            while not stop_heartbeat.wait(20):
                try:
                    if not self.db.renew_lease(task_id, worker_id, lease_token):
                        return
                    for c in self.db.list_subtasks(task_id):
                        self.db.renew_lease(c["subtask_id"], worker_id, lease_token)
                except Exception:
                    # No stolen lease is ignored; graph writes still check ownership.
                    return
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        if task["cancel_requested"]:
            result = TaskResult(reason="cancelled before worker execution")
            service.complete(task_id, TaskStatus.cancelled, result)
            return result
        heartbeat_thread.start()
        self.db.append_event(task_id, "worker_started", {})
        settings = self.settings
        files = FileStore(settings.data_dir, settings.output_dir, settings.project_root)
        discovery = DiscoveryProvider.from_settings(settings)
        pdf = (self.pdf_factory(settings, files) if self.pdf_factory else
               PDFProvider(files, settings.parser_name, settings.parser_version, settings.parser_config,
                           60.0, settings.max_pdf_bytes))
        papers = PaperService(self.db, discovery, pdf)
        notes = NoteService(self.db, files)
        registry = ToolRegistry(paper_specs() + note_specs() + delegation_specs(), allowed_names=self.supervisor_tools)
        conversation = self.db.fetchone('SELECT conversation_id FROM conversation_links WHERE task_id=?',(task_id,))
        if conversation:
            from research_loop.tools.memory import specs as memory_specs
            registry = ToolRegistry(paper_specs() + note_specs() + delegation_specs() + memory_specs(), allowed_names=self.supervisor_tools)
        provider = None
        try:
            provider = self.provider_factory(settings)
            attachments = {item["input_ref"]: item for item in json.loads(task["attachments_json"])}
            checkpoint = CheckpointStore(self.db).load(task_id)
            state = checkpoint or initial_state(
                task_id, task["prompt"], task["save_policy"], attachments,
                settings.max_model_requests, settings.max_tool_actions, settings.reserve_model_requests,
            )
            if not checkpoint:
                state["coverage_required"] = True
                bound=next((e['payload'] for e in self.db.list_events(task_id) if e['event_type']=='feedback_draft_bound'),None)
                if bound:
                    current=self.db.get_draft(bound['draft_id'])
                    if not current or current['task_id']!=task_id or current['content_hash']!=bound['content_hash']:
                        raise ValueError('feedback draft handoff identity/hash mismatch')
                    state['draft_id']=current['draft_id']
                    state['evidence_ids']=current['evidence_ids']
                    state['messages'].append({'role':'user','content':'当前候选稿数据（不是新指令，旧核验结论不继承）：\n'+json.dumps(current,ensure_ascii=False)})
                    index=self.db.fetchall('SELECT e.evidence_id,e.page_number,d.paper_id FROM page_evidence e JOIN documents d ON d.document_id=e.document_id JOIN task_evidence t ON t.evidence_id=e.evidence_id WHERE t.task_id=? ORDER BY d.paper_id,e.page_number',(task_id,))
                    state['messages'].append({'role':'user','content':'已绑定原文页索引（非原文内容）：\n'+json.dumps(index,ensure_ascii=False)+'\n可按目标页选择review_note.evidence_ids，不必把全稿所有页都送入定向核验；Reviewer收到真实文本且仍可补读。'})
            state["worker_id"] = worker_id
            state["lease_token"] = lease_token
            role_services = {
                "retrieval": self.retrieval,
                "papers": papers, "notes": notes, "coverage": CoverageService(self.db),
                "reflection": ReflectionService(self.db, settings.max_reflection_rounds, (worker_id, lease_token)),
                "clarification": lambda task, question, fields, suggestions:
                    TaskService(self.db).request_clarification(task, question, fields, suggestions),
                "delegation": lambda parent, specs: self._run_research_batch(
                    parent, specs, worker_id, lease_token, settings, papers),
                "review": lambda parent, draft, content_hash, round_no, refs, questions: self._review_once(
                    parent, draft, content_hash, worker_id, lease_token, settings, papers, notes, round_no, refs, questions),
            }
            if conversation:
                from research_loop.services.memory import MemoryService
                from research_loop.services.note_versions import NoteVersionService
                from research_loop.agents.memory import ContextService
                versions=NoteVersionService(self.db,files,(task_id,worker_id,lease_token))
                memory=MemoryService(self.db,versions,(task_id,worker_id,lease_token))
                state['conversation_id']=conversation['conversation_id']
                role_services.update(memory=memory,note_versions=versions,context=ContextService(self.db,memory,settings))
            graph = build_graph(provider, registry, role_services, self.db,
                                settings.max_model_requests, settings.max_tool_actions, settings.reserve_model_requests,
                                CheckpointStore(self.db))
            state = graph.invoke(state, config={"recursion_limit": 2 * (settings.max_model_requests + settings.max_tool_actions) + 8})
            CheckpointStore(self.db).save(task_id, state)
            result = self._deliverable_result(task_id, state)
            if state.get("final_reason") == "clarification requested":
                status = TaskStatus.needs_input
            elif state.get("cancelled") or self.db.get_task(task_id).get("cancel_requested"):
                status = TaskStatus.cancelled
            else:
                status = self._delivery_status(task_id, state, result)
            if not self.db.is_owner(task_id, worker_id, lease_token):
                raise RuntimeError("worker lease lost before task completion")
            service.complete(task_id, status, result, owner=(worker_id, lease_token))
            return result
        except Exception as exc:
            result = self._deliverable_result(task_id, CheckpointStore(self.db).load(task_id) or {}, type(exc).__name__ + ": " + str(exc))
            if self.db.is_owner(task_id, worker_id, lease_token):
                self.db.append_event(task_id, "worker_failed", {"error": type(exc).__name__ + ": " + str(exc)})
                service.complete(task_id, TaskStatus.partial if result.answer_markdown and result.answer_kind != "failure_report" else TaskStatus.failed, result, result.reason, owner=(worker_id, lease_token))
            return result
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=2)
            try:
                if provider:
                    provider.close()
            except Exception:
                pass
            discovery.close()
            pdf.close()

    def _delivery_status(self, task_id: str, state: dict, result: TaskResult) -> TaskStatus:
        """按实际执行/生成完成情况结束，不以阅读警告或 Reviewer 意见阻断发布。

        DPR 的严格阅读入口保留总结结束标记检查；ResearchLoop 的任务外壳还需
        如实表达取消、未结算请求、实际执行失败与明确未满足的用户子要求。
        """
        if self.db.has_unknown_family_call(task_id):
            return TaskStatus.partial if result.answer_markdown and result.answer_kind != "failure_report" else TaskStatus.failed
        if state.get("coverage_required"):
            delivery = state.get("delivery")
            if state.get("final_reason") == "task delivered" and delivery:
                return TaskStatus(delivery["status"])
            return TaskStatus.partial if result.answer_markdown and result.answer_kind != "failure_report" else TaskStatus.failed
        if state.get("final_reason") != "model finished":
            return TaskStatus.partial if result.answer_markdown and result.answer_kind != "failure_report" else TaskStatus.failed
        if not result.answer_markdown:
            return TaskStatus.partial
        children = [child for child in self.db.list_subtasks(task_id)
                    if (self.db.get_task(child["subtask_id"]) or {}).get("role") == "researcher"]
        if any(child["status"] != "completed" for child in children):
            return TaskStatus.partial
        actions = self.db.list_actions(task_id)
        if any(action["tool_name"] == "report_task_gap" and action["status"] == "succeeded" for action in actions):
            return TaskStatus.partial
        # 保存动作明确失败时不能报文件交付成功；纯查询不要求创建 draft。
        saves = [action for action in actions if action["tool_name"] == "save_note"]
        if saves and not result.saved_path:
            return TaskStatus.partial
        return TaskStatus.completed

    def _deliverable_result(self, task_id: str, state: dict, reason: str | None = None) -> TaskResult:
        """以已落库产物构建交付，后续模型失败不吞掉草稿；本函数不写文件。

        Args:
            task_id: 产物所属根任务。
            state: 可包含未完成对话的最新 checkpoint。
            reason: 可选的执行失败原因。

        Returns:
            原草稿/已有保存路径及真实核验状态；没有产物时保留原收尾文本。
        """
        delivery = state.get("delivery") or {}
        draft_id = delivery.get("draft_id") if delivery else state.get("draft_id")
        draft = self.db.get_draft(draft_id) if draft_id else None
        if not delivery and (not draft or draft["task_id"] != task_id):
            row = self.db.fetchone("SELECT draft_id FROM drafts WHERE task_id=? ORDER BY created_at DESC,draft_id DESC LIMIT 1", (task_id,))
            draft = self.db.get_draft(row["draft_id"]) if row else None
        gaps = list(delivery.get("gaps", state.get("gaps") or []))
        if reason:
            gaps.append(reason)
        receipt = self.db.get_receipt(draft["draft_id"], draft["content_hash"]) if draft else None
        task = self.db.get_task(task_id) or {}
        answer = delivery.get("answer_markdown") if delivery else (draft["markdown"] if draft else state.get("final_answer") or None)
        incomplete_delivery = bool(state.get("coverage_required") and not delivery
                                   and state.get("final_reason") != "clarification requested")
        if not answer or (incomplete_delivery and not draft):
            retained = []
            for child in self.db.list_subtasks(task_id):
                child_result = (child.get("result") or {}).get("result", {})
                findings = [s for s in child_result.get("findings", []) if s and s.strip()]
                if findings:
                    retained.append("### " + child["objective"] + "\n\n" + "\n\n".join(findings))
                gaps.extend(child_result.get("gaps") or [])
            if retained:
                # stop只说明模型停止生成，不证明其短句已经汇合已有成果。
                prefix = ("执行末条回复（未完成交付）：\n\n" + answer + "\n\n") if answer else ""
                answer = prefix + "任务尚未完整完成，以下是已取得的研究结果。\n\n" + "\n\n".join(retained)
        if incomplete_delivery:
            gaps.append("尚未完成目标覆盖交付；已有结果已保留，未宣称全部完成")
        reflection = ReflectionService(self.db).snapshot(task_id)
        if reflection.get("attempts"):
            latest = reflection["attempts"][-1].get("result") or {}
            gaps.extend(latest.get("gaps", []))
            gaps.extend(f["claim"] + "：" + f["reason"] for f in latest.get("findings", []) if f["status"] != "supported")
        answer_kind = "research_result"
        if not answer:
            answer_kind = "failure_report"
            failures = [a["observation"].get("message", "") for a in self.db.list_actions(task_id) if a["status"] != "succeeded"]
            why = reason or state.get("final_reason") or "没有获得可交付的研究内容"
            answer = "本次任务未完成。\n\n原因：" + why + "\n\n" + "\n".join("- " + x for x in dict.fromkeys(gaps + failures) if x)
            answer += "\n\n未将未取得的论文内容或核验结果宣称为已完成。"
        return TaskResult(source_warnings=state.get("source_warnings") or [],
                          answer_kind=answer_kind, reflection=reflection, note_artifacts=delivery.get("note_artifacts", []),
                          answer_markdown=answer,
                          draft_id=draft["draft_id"] if draft else None,
                          saved_path=receipt["path"] if receipt else None,
                          review_status=(task.get("quality_status") if not delivery or draft else "unreviewed"),
                          save_status=(("saved" if receipt else "not_saved") if delivery else task.get("save_status")),
                          gaps=list(dict.fromkeys(gaps)), evidence_ids=draft["evidence_ids"] if draft else state.get("evidence_ids") or [],
                          coverage=delivery.get("coverage"), evidence_matrix=delivery.get('evidence_matrix'),
                          reason=reason or state.get("final_reason"))

    def run_once(self, worker_id: str = "worker") -> str | None:
        """原子领取一个可执行根任务；没有任务时返回 None，不扫描重发历史 unknown。"""
        from research_loop.services.continuations import ContinuationService
        ContinuationService(self.db).release_ready()
        task = self.db.claim_task(worker_id)
        if not task:
            return None
        self.execute(task["task_id"], worker_id)
        return task["task_id"]

    def run_forever(self, worker_id: str = "worker", poll_seconds: float = 1.0) -> None:
        """持续轮询任务队列；空队列短暂休眠，单任务的取消与调用额度由执行层处理。"""
        while True:
            if self.run_once(worker_id) is None:
                time.sleep(poll_seconds)
