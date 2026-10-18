"""执行编排：领取任务、维护租约、组装各角色工具、运行 LangGraph，再持久化用户可查询的结果。"""
from __future__ import annotations

from research_loop.agents.reflection import ReflectionService
from research_loop.services.research_progress import ResearchProgressService

import json
import time
import hashlib
import threading
from uuid import uuid4
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


def research_stop_details(state: dict, status: str, read_paper_ids: list[str]) -> dict:
    """把真实结束原因和已读身份交给父角色，不凭空生成研究结论。

    Args:
        state: 子图最后的持久状态。
        status: Worker依据结构化交接及取消状态计算的结果状态。
        read_paper_ids: 成功read_paper行动读回的论文身份，不等于合格结论覆盖。

    Returns:
        当前停止原因、明确缺口和只读身份；不修改findings或paper_ids。
    """
    gaps = list(state.get('gaps') or [])
    reason = state.get('final_reason')
    if status != 'completed' and not state.get('structured_result'):
        reason = reason or 'researcher did not finish structurally'
        gaps.append('子任务未完成：' + reason + '；已读原文仍保留，不代表已形成研究结论。')
    return {'stop_reason': reason, 'read_paper_ids': list(dict.fromkeys(read_paper_ids)),
            'gaps': list(dict.fromkeys(gaps))}


def research_review_failure(actions: list[dict], terminal_error: str, stop_reason: str | None) -> dict:
    """把已持久化的工具错误与最终停止原因分开传递，不推断论文缺证。

    Args:
        actions: 本次Reviewer的已落库行动，包括失败的结果提交。
        terminal_error: 执行器捕获的最终异常。
        stop_reason: 图的最后停止原因；异常未返回图时可为空。

    Returns:
        全部历史错误、首错、最近错误、未成功提交的结果错误及最终停止状态。
    """
    failures = []
    for action in actions:
        observation = action.get('observation') or {}
        if action.get('status') != 'failed':
            continue
        failures.append({'action_id': action.get('action_id'), 'sequence': action.get('sequence', action.get('sequence_no')),
            'tool_name': action['tool_name'], 'kind': observation.get('kind'),
            'stage': 'result_validation' if action['tool_name'] == 'finish_review' else 'tool_execution',
            'message': observation.get('message', '')})
    final_submission = next((a for a in reversed(actions) if a['tool_name'] == 'finish_review'), None)
    blocking = (next((e for e in reversed(failures) if e['stage'] == 'result_validation'
                     and e['action_id'] == final_submission.get('action_id')), None)
                if final_submission and final_submission.get('status') == 'failed' else None)
    return {'first_error': failures[0] if failures else None, 'tool_errors': failures,
            'last_error': failures[-1] if failures else None, 'blocking_error': blocking,
            'terminal_error': terminal_error, 'stop_reason': stop_reason or terminal_error,
            'semantic_verdict_available': False}


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
        # 整批先校验；后续有歧义/越权时不能已创建一半子任务。
        specs = [CoverageService(self.db).normalize_references(parent_task_id, spec) for spec in specs]
        for position, spec in enumerate(specs):
            for ref in spec.get("paper_ids") or []:
                if not self.db.task_has_paper(parent_task_id, ref):
                    raise PermissionError("cannot delegate an unauthorized paper: " + ref)
            for ref in spec.get("input_refs") or []:
                if ref not in attachments or not self.db.task_has_input(parent_task_id, ref):
                    raise PermissionError("cannot delegate an unauthorized attachment: " + ref)
            key = "research:" + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            existing = self.db.fetchone("SELECT * FROM subtasks WHERE parent_task_id=? AND idempotency_key=?", (parent_task_id, key))
            if not existing:
                # 旧任务的引用顺序可能未规范化；恢复先匹配现存子任务，不改其旧身份。
                for prior in self.db.list_subtasks(parent_task_id):
                    if prior.get('position_no') == 999:
                        continue
                    if CoverageService(self.db).normalize_references(parent_task_id, prior['spec']) == spec:
                        # list_subtasks是API投影（已移除result_json）；恢复需原始收据行。
                        existing = self.db.fetchone('SELECT * FROM subtasks WHERE subtask_id=?', (prior['subtask_id'],))
                        break
            next_position = max((c["position_no"] for c in self.db.list_subtasks(parent_task_id)), default=-1) + 1
            child = existing or self.db.create_subtask(parent_task_id, root, next_position, spec["objective"], spec, key)
            if any(c['subtask_id'] == child['subtask_id'] for c, _ in children):
                continue
            for ref in spec.get("paper_ids") or []:
                self.db.grant_paper(child["subtask_id"], ref, parent_task_id)
            for ref in spec.get("input_refs") or []:
                self.db.grant_input(child["subtask_id"], ref, parent_task_id)
            # 子任务只持有获分派的附件，不继承父任务全部对话或无关附件。
            self.db.execute("UPDATE tasks SET attachments_json=? WHERE task_id=?", (json.dumps([attachments[r] for r in spec.get("input_refs") or []]), child["subtask_id"]))
            children.append((child, spec))

        def retain(child_id: str, result: dict) -> dict:
            """将已成功提交的阶段发现交回父任务，来源失效时保留内容并明确缺口。"""
            progress = ResearchProgressService(self.db).retained(child_id)
            result['research_progress'] = progress
            result['findings'] = list(dict.fromkeys([*result.get('findings', []),
                *(f['text'] for f in progress['findings'])]))
            result['gaps'] = list(dict.fromkeys([*result.get('gaps', []), *progress['gaps']]))
            refs = {ref['evidence_id'] for f in progress['findings'] if f['source_integrity'] == 'intact' for ref in f['sources']}
            result['evidence_ids'] = list(dict.fromkeys([*result.get('evidence_ids', []), *sorted(refs)]))
            for finding in progress['findings']:
                if finding['source_integrity'] != 'intact':
                    result['gaps'].append('阶段发现来源已失效，不能作为当前核验依据：' + finding['finding_id'])
                else:
                    result['paper_ids'] = list(dict.fromkeys([*result.get('paper_ids', []), *(r['paper_id'] for r in finding['sources'])]))
            if (self.db.is_owner(child_id, worker_id, lease_token)
                    and self.db.is_owner(parent_task_id, worker_id, lease_token)
                    and not self.db.get_task(parent_task_id).get('cancel_requested')
                    and not self.db.get_task(child_id).get('cancel_requested')):
                for ref in result['evidence_ids']:
                    if self.db.task_has_evidence(child_id, ref):
                        self.db.grant_evidence(parent_task_id, ref, child_id)
            return result

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
                from research_loop.agents.memory import ContextService
                graph = build_researcher_graph(provider, {"papers": papers, "scope": spec, "retrieval": self.retrieval,
                                               "context": ContextService(self.db, None, settings)}, self.db, CheckpointStore(self.db),
                                               max_model_requests=min(12, settings.max_model_requests),
                                               # 子角色沿用冻结的本地上限（新任务默认20）；48是根共享额度。
                                               max_tool_actions=state.get('max_tool_actions', 20),
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
                read_ids = list(dict.fromkeys(a["observation"]["data"]["paper_id"]
                    for a in self.db.list_actions(child_id) if a["tool_name"] == "read_paper"
                    and a["status"] == "succeeded" and a["observation"].get("data", {}).get("paper_id")))
                if not selected and mode == "deep_read":
                    selected = read_ids
                result = {"subtask_id": child_id, "parent_task_id": parent_task_id,
                          "mode": mode, "goal_ids": spec.get("goal_ids") or [],
                          "objective": spec["objective"], "status": status,
                          "findings": structured.get("findings") or ([result_state['final_answer']]
                                      if (result_state.get('final_answer') or '').strip() else []),
                          "evidence_ids": evidence_ids, "paper_ids": selected,
                          "assessments": structured.get("assessments") or [],
                          "evidence_scope": structured.get("evidence_scope") or [],
                          **research_stop_details(result_state, status, read_ids),
                          "deep_reading_status": (result_state.get("deep_reading") or {}).get("status") }
                if not self.db.is_owner(child_id, worker_id, lease_token):
                    raise RuntimeError("subtask lease lost before result commit")
                result = retain(child_id, result)
                self.db.save_subtask_result(child_id, status, result)
                return result
            except Exception as exc:
                result = {"subtask_id": child_id, "parent_task_id": parent_task_id,
                          "mode": spec.get("mode", "deep_read"), "goal_ids": spec.get("goal_ids") or [],
                          "objective": spec["objective"], "status": "failed", "error": str(exc),
                          "findings": [], "evidence_ids": [], "paper_ids": [], "gaps": [str(exc)]}
                result = retain(child_id, result)
                result['stop_reason'] = str(exc)
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
        position = max(999 + round_no, max((c["position_no"] for c in self.db.list_subtasks(parent_task_id)), default=-1) + 1)
        child = self.db.create_subtask(parent_task_id, parent_task_id, position, "independent review", {
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

    def _visual_once(self, parent_task_id: str, args, worker_id: str, lease_token: str,
                     settings: Settings, papers: PaperService) -> dict:
        """将显式视觉委派交给单次Researcher节点，复用同版结果而非父任务收据。

        Args:
            parent_task_id: 发起委派的Supervisor任务。
            args: 已校验的问题、已读页、原因和可选读回偏移。
            worker_id: 原父子链执行者。
            lease_token: 当前租约，失权不得发布观察。
            settings: 原日常配置；仅视觉适配器使用组件固定模型/超时。
            papers: 已具备真实读取证据的同一论文服务。

        Returns:
            带来源/收据的紧凑模型观察或具体失败；不会保存笔记或批准内容。
        """
        from research_loop.agents.visual_reading import VisualReadingNode, build_visual_graph
        from research_loop.domain.visual_reading import VisualConfig, compact_observation
        from research_loop.storage.memory import Transaction, check_owner, digest

        def guard():
            """事务内核对父租约/取消，再进行下一段执行或结果交接。"""
            with self.db.connection() as conn:
                check_owner(Transaction(self.db, conn), (parent_task_id, worker_id, lease_token))

        guard()
        item = papers.visual_input_for_evidence(parent_task_id, args)
        identity = 'visual:' + digest(item.model_dump(mode='json'))
        existing = self.db.fetchone('SELECT * FROM subtasks WHERE parent_task_id=? AND idempotency_key=?',
                                   (parent_task_id, identity))
        if args.offset and not existing:
            raise ValueError('visual readback requires the original retained result; no new request sent')
        visual_config = VisualConfig()
        if not existing and settings.deepseek_base_url.rstrip('/') != visual_config.base_url:
            raise ValueError('visual node unavailable for configured endpoint; no provider switch')
        parent = self.db.get_task(parent_task_id)
        root = parent.get('root_task_id') or parent_task_id
        position = max((c['position_no'] for c in self.db.list_subtasks(parent_task_id)), default=-1) + 1
        child = self.db.create_subtask(parent_task_id, root, position, args.question,
            {'mode': 'visual_reading', 'input': item.model_dump(mode='json'),
             'evidence_id': args.evidence_id, 'reason': args.reason, 'trigger': args.trigger}, identity)
        tid = child['subtask_id']
        if child.get('result_json') and child['status'] in {'completed', 'partial', 'failed', 'cancelled'}:
            guard()
            return compact_observation(json.loads(child['result_json'])['result'], args, tid, True)
        if args.offset:
            raise ValueError('visual result is not settled; readback cannot start or resume generation')
        child_task = self.db.get_task(tid)
        if child_task.get('cancel_requested'):
            raise PermissionError('visual child was cancelled; no automatic restart')
        if (child_task['status'] == 'running' and child_task.get('lease_until')
                and child_task['lease_until'] >= self.db_now()
                and not self.db.is_owner(tid, worker_id, lease_token)):
            raise PermissionError('visual child is owned by a different live lease')
        # 仅给子节点本次授权页和缓存文件；不继承父历史或其他论文/页。
        document = self.db.fetchone('SELECT paper_id FROM documents WHERE document_id=?', (item.document_id,))
        attached = {'input_ref': item.target_ref, 'paper_id': document['paper_id'],
                    'local_path': str(papers.pdf.file_store.cache_pdf_path(item.pdf_sha256))}
        guard()
        self.db.update_task(tid, status='running', claimed_by=worker_id, lease_token=lease_token,
            lease_until=parent['lease_until'])
        self.db.execute('UPDATE tasks SET attachments_json=? WHERE task_id=?',
                        (json.dumps([attached], ensure_ascii=False), tid))
        self.db.grant_input(tid, item.target_ref, parent_task_id)
        self.db.grant_paper(tid, document['paper_id'], parent_task_id)
        self.db.grant_evidence(tid, args.evidence_id, parent_task_id)
        self.db.execute("UPDATE subtasks SET status='running',updated_at=? WHERE subtask_id=?", (self.db_now(), tid))
        provider = None
        try:
            if settings.deepseek_base_url.rstrip('/') != visual_config.base_url:
                raise ValueError('visual node unavailable for configured endpoint; no provider switch')
            node_settings = settings.model_copy(update={'deepseek_model': visual_config.model,
                'request_timeout_seconds': visual_config.timeout_seconds})
            provider = self.provider_factory(node_settings)
            graph = build_visual_graph(VisualReadingNode(self.db, papers, provider), tid, item,
                worker_id=worker_id, lease_token=lease_token)
            result = graph.invoke({})
            status, error = ('completed' if result['status'] == 'complete' else 'partial'), None
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            result = {'status': 'execution_failed', 'answer': '', 'error': error,
                      'source': {**item.model_dump(mode='json'), 'evidence_id': args.evidence_id},
                      'processing_summary': '视觉节点执行失败，未产生事实判断；保留已有文字结果及具体缺口。'}
            status = 'failed'
        finally:
            if provider:
                provider.close()
        guard()
        if not self.db.is_owner(tid, worker_id, lease_token):
            raise PermissionError('visual child lease lost before handoff')
        # 完整观察留在子任务，父工具只接收分页窗口，不复制图片或全文。
        # 父角色后续异常时也能按原成果保留路径交付，不能把观察丢成空失败报告。
        result = {**result, 'mode': 'visual_reading',
                  'findings': ['视觉模型观察（未语义核验）：\n' + result['answer']] if result.get('answer') else [],
                  'evidence_ids': [args.evidence_id], 'paper_ids': [document['paper_id']],
                  'gaps': [error] if error else [], 'stop_reason': error or result['status']}
        # 交接写入与父/子租约检查同事务，防止检查后失权仍覆盖新执行者。
        with self.db.connection() as conn:
            tx = Transaction(self.db, conn)
            if self.db.kind == 'sqlite':
                tx.execute('BEGIN IMMEDIATE')
            check_owner(tx, (parent_task_id, worker_id, lease_token))
            check_owner(tx, (tid, worker_id, lease_token))
            now = self.db_now()
            tx.execute('UPDATE subtasks SET status=?,result_json=?,updated_at=? WHERE subtask_id=?',
                (status, json.dumps({'result': result, 'error': error}, ensure_ascii=False), now, tid))
            tx.execute('UPDATE tasks SET status=?,result_json=?,error=?,updated_at=? WHERE task_id=?',
                (status, json.dumps(result, ensure_ascii=False), error, now, tid))
        return compact_observation(result, args, tid, False)

    def _review_research_once(self, parent_task_id: str, attempt_id: str, target: dict,
                             refs: list[str], questions: list[str], worker_id: str,
                             lease_token: str, settings: Settings, papers: PaperService) -> dict:
        """复用独立Reviewer/收据/快照执行通用核验；原始结果不写笔记表。"""
        from research_loop.agents.reviewer import research_reviewer_state
        key = 'research-review:' + attempt_id
        position = max((c['position_no'] for c in self.db.list_subtasks(parent_task_id)), default=-1) + 1
        child = self.db.create_subtask(parent_task_id, parent_task_id, position,
            'independent research review', {'target_id': target['target_id'],
            'content_hash': target['content_hash'], 'attempt_id': attempt_id}, key)
        tid = child['subtask_id']
        if child.get('status') in {'completed', 'failed'} and child.get('result_json'):
            return json.loads(child['result_json'])['result']
        parent = self.db.get_task(parent_task_id)
        if not self.db.is_owner(parent_task_id, worker_id, lease_token):
            raise PermissionError('research review root lease lost')
        self.db.update_task(tid, role='reviewer', status='running', claimed_by=worker_id,
                            lease_token=lease_token, lease_until=parent.get('lease_until'))
        for eid in self.db.list_task_evidence(parent_task_id):
            self.db.grant_evidence(tid, eid, parent_task_id)
        for pid in self.db.list_task_papers(parent_task_id):
            self.db.grant_paper(tid, pid, parent_task_id)
        attachments = json.loads(parent['attachments_json'])
        for attachment in attachments:
            self.db.grant_input(tid, attachment['input_ref'], parent_task_id)
        self.db.execute('UPDATE tasks SET attachments_json=? WHERE task_id=?', (json.dumps(attachments), tid))
        provider = self.provider_factory(settings)
        end = None
        try:
            bundle = papers.evidence_bundle(parent_task_id, refs)
            execution_checks = {'save_policy': parent['save_policy'],
                'successful_save_actions': [{'action_id': a['action_id'], 'tool_name': a['tool_name']}
                    for a in self.db.list_actions(parent_task_id)
                    if a['tool_name'] == 'save_note' and a['status'] == 'succeeded'],
                'unrelated_text_preservation': 'not_checked',
                'notice': '仅当前父任务已记录的保存动作；不是全文件系统审计或科学证据，未提供段落差异检查。'}
            current = CheckpointStore(self.db).load(tid) or research_reviewer_state(tid, target, questions, bundle, execution_checks)
            current.update(worker_id=worker_id, lease_token=lease_token, parent_task_id=parent_task_id,
                           root_task_id=parent_task_id, attachments={a['input_ref']: a for a in attachments})
            graph = build_reviewer_graph(provider, {'papers': papers, 'retrieval': self.retrieval,
                'review_target': target, 'review_parent_task_id': parent_task_id}, self.db,
                CheckpointStore(self.db), max_model_requests=4)
            end = graph.invoke(current, config={'recursion_limit': 2 * (settings.max_model_requests + settings.max_tool_actions) + 8})
            result = next((a['observation']['data'] for a in reversed(end.get('actions', []))
                if a['tool_name'] == 'finish_review' and a['status'] == 'succeeded'), None)
            if not result:
                raise ValueError(end.get('final_reason') or 'reviewer did not finish structurally')
            for eid in self.db.list_task_evidence(tid):
                self.db.grant_evidence(parent_task_id, eid, tid)
            status, error = 'completed', None
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            diagnostic = research_review_failure(self.db.list_actions(tid), error,
                                                  (end or {}).get('final_reason'))
            primary = diagnostic['blocking_error'] or diagnostic['last_error']
            label = '结果提交错误' if diagnostic['blocking_error'] else '最近工具错误'
            detail = (f"；{label}[{primary['tool_name']}]: {primary['message']}" if primary else '')
            result = {'review_id': uuid4().hex, 'target_id': target['target_id'],
                'content_hash': target['content_hash'], 'status': 'execution_failed',
                'execution_status': 'failed', 'execution_error': diagnostic, 'findings': [],
                'gaps': ['核验执行未完成' + detail + '；最后停止：' + diagnostic['stop_reason'] +
                         '。这是执行状态，不代表原文缺证或内容错误。']}
            status = 'failed'
        finally:
            provider.close()
        if not self.db.is_owner(tid, worker_id, lease_token) or not self.db.is_owner(parent_task_id, worker_id, lease_token):
            raise PermissionError('research review lease lost before commit')
        self.db.save_subtask_result(tid, status, result, error)
        return result

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
            if checkpoint and not checkpoint.get('coverage_required'):
                # 旧快照沿用旧工具范围；不能让新增schema挤占其既有上下文或改变恢复合同。
                added = {'review_research', 'revise_research', 'close_research_issue', 'delegate_visual_reading'}
                registry = ToolRegistry([registry.get(name) for name in registry.names() if name not in added])
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
            from research_loop.agents.memory import ContextService
            role_services = {
                "context": ContextService(self.db, None, settings),
                "stream_public": bool(conversation),
                "retrieval": self.retrieval,
                "papers": papers, "notes": notes, "coverage": CoverageService(self.db),
                "reflection": ReflectionService(self.db, settings.max_reflection_rounds, (worker_id, lease_token)),
                "clarification": lambda task, question, fields, suggestions:
                    TaskService(self.db).request_clarification(task, question, fields, suggestions),
                "delegation": lambda parent, specs: self._run_research_batch(
                    parent, specs, worker_id, lease_token, settings, papers),
                "visual_reading": lambda parent, args: self._visual_once(
                    parent, args, worker_id, lease_token, settings, papers),
                "review_research": lambda parent, attempt, target, refs, questions: self._review_research_once(
                    parent, attempt, target, refs, questions, worker_id, lease_token, settings, papers),
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

        论文 的严格阅读入口保留总结结束标记检查；ResearchLoop 的任务外壳还需
        如实表达取消、未结算请求、实际执行失败与明确未满足的用户子要求。
        """
        if self.db.has_unknown_family_call(task_id):
            return TaskStatus.partial if result.answer_markdown and result.answer_kind != "failure_report" else TaskStatus.failed
        from research_loop.domain.context import textual_delivery
        if textual_delivery(state.get('final_answer') or '') is not None and not state.get('delivery'):
            return TaskStatus.partial if result.answer_kind != 'failure_report' else TaskStatus.failed
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
        from research_loop.domain.context import textual_delivery
        candidate = textual_delivery(answer) if answer and not draft and not delivery else None
        if candidate is not None:
            # 旧checkpoint恢复也必须区分文本化内部参数，不能只保护新模型响应。
            notice = "内部交付参数未执行；以下仅为未核验候选，未完成目标覆盖交付。"
            answer = notice + "\n\n" + candidate if candidate else None
            gaps.append(notice)
        incomplete_delivery = bool(state.get("coverage_required") and not delivery
                                   and state.get("final_reason") != "clarification requested")
        retained_answer_source, retained_body, retained_summary = None, None, None
        if not answer or (incomplete_delivery and not draft):
            retained = []
            retained_targets = []
            retention_notes = []
            superseded = set()
            review_state = ReflectionService(self.db).snapshot(task_id)
            for record in review_state.get('research_targets', {}).values():
                current_target = record['current']
                if current_target['target_type'] == 'candidates':
                    continue
                texts = [i['text'] for i in current_target['items'] if i.get('text')]
                if texts:
                    retained.append('\n\n'.join(texts))
                    retained_targets.append({k: current_target[k] for k in ('target_id', 'content_hash', 'target_type')})
                    retention_notes.append('保留当前核验对象正文，实际核验状态见记录；不表示内容已获批准。')
                for version in record['versions']:
                    superseded.update(i.get('text') for i in version['items'])
            own = ResearchProgressService(self.db).retained(task_id)
            own_texts = [f['text'] for f in own['findings'] if f['text'] not in superseded]
            if own_texts:
                retained.append('\n\n'.join(own_texts))
                retention_notes.append('已记录阶段成果未独立核验。')
            gaps.extend(own['gaps'])
            for child in self.db.list_subtasks(task_id):
                if (self.db.get_task(child['subtask_id']) or {}).get('role') == 'reviewer':
                    continue
                child_result = (child.get("result") or {}).get("result", {})
                progress = ResearchProgressService(self.db).retained(child['subtask_id'])
                findings = list(dict.fromkeys([s for s in child_result.get('findings', []) if isinstance(s, str) and s.strip() and s not in superseded] + [f['text'] for f in progress['findings'] if f['text'] not in superseded]))
                gaps.extend(progress['gaps'])
                if findings:
                    retained.append("\n\n".join(findings))
                    retention_notes.append("保留子任务成果：" + child["objective"])
                gaps.extend(child_result.get("gaps") or [])
            if retained:
                # 正文逐字来自当前版本；状态、模型末条说明和执行诊断不能混进正文。
                retained_body = "\n\n".join(dict.fromkeys(retained))
                if retained_targets:
                    retained_answer_source = {'source_type': 'research_targets', 'targets': retained_targets,
                        'is_original_paper_evidence': False, 'save_permission_inherited': False}
                diagnostics = []
                for review in ReflectionService(self.db).research_context(task_id):
                    error = review.get('execution_error') or {}
                    if error:
                        first = error.get('first_error') or {}
                        diagnostics.append('核验执行未完成；' +
                            (f"首个工具错误[{first.get('tool_name')}]：{first.get('message')}；" if first else '') +
                            '核验最后停止：' + str(error.get('stop_reason') or error.get('terminal_error')) +
                            '。未获得语义结论，不能据此认定原文缺证或答案错误。')
                retention_notes.extend(diagnostics)
                last_action = (self.db.list_actions(task_id) or [None])[-1]
                if last_action and last_action.get('status') == 'failed':
                    retention_notes.append('最后工具错误[' + last_action['tool_name'] + ']：' +
                                           (last_action.get('observation') or {}).get('message', ''))
                if answer and answer != retained_body:
                    retention_notes.append('执行末条回复（未完成交付）：' + answer)
                retained_summary = '任务尚未完整完成。' + '；'.join(dict.fromkeys(retention_notes)) + \
                    '停止原因：' + (reason or state.get('final_reason') or '执行未完成')
                answer = retained_body + '\n\n处理说明：' + retained_summary
            elif not answer and not draft and not delivery and state.get('final_reason') != 'clarification requested':
                from research_loop.services.conversations import ConversationService
                referenced = ConversationService(self.db).referenced_answer(task_id)
                if referenced:
                    retained_body = referenced['content']
                    retained_answer_source = {k: v for k, v in referenced.items() if k != 'content'}
                    retained_summary = ('本轮处理未完成，按明确来源保留已有回答原文；未完成本轮独立核验，不表示内容已获批准。'
                                        '停止原因：' + (reason or state.get('final_reason') or '执行未完成'))
                    answer = retained_body + '\n\n处理说明：' + retained_summary
                    gaps.append(retained_summary)
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
                          answer_kind=answer_kind, reflection=reflection,
                          research_reviews=ReflectionService(self.db).research_context(task_id),
                          review_notices=delivery.get("review_notices", []), note_artifacts=delivery.get("note_artifacts", []),
                          answer_markdown=answer,
                          answer_body_markdown=delivery.get('answer_body_markdown', retained_body or answer),
                          processing_summary=delivery.get('processing_summary', retained_summary),
                          retained_answer_source=retained_answer_source,
                          draft_id=draft["draft_id"] if draft else None,
                          saved_path=receipt["path"] if receipt else None,
                          review_status=('unreviewed' if retained_answer_source else
                                         (task.get("quality_status") if not delivery or draft else "unreviewed")),
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
