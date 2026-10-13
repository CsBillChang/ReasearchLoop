"""任务生命周期：幂等创建、只读视图、取消、澄清恢复及最终落库；不在此执行研究模型。"""
from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

from research_loop.domain.tasks import Clarification, ResumeInput, SavePolicy, TaskCreate, TaskResult, TaskStatus, TaskView
from research_loop.storage.mysql import Database
from research_loop.providers.deepseek import validate_messages


class IdempotencyConflict(ValueError):
    pass


class TaskService:
    def __init__(self, db: Database, settings=None):
        """复用配置并在新建时冻结预算，读回旧任务不回填策略。"""
        from research_loop.config import get_settings
        self.db = db
        self.settings = settings or get_settings()

    def create(self, payload: TaskCreate) -> TaskView:
        """输入已校验任务，输出任务视图；记录输入 hash 与幂等键，并建立根任务共享额度。"""
        if payload.conversation_id:
            from research_loop.services.conversations import ConversationService
            return self.view(self.db.get_task(ConversationService(self.db, self.settings).submit(payload)))
        canonical = payload.model_dump(mode="json")
        # 老任务的幂等指纹仍保留附件中的null，只移除新加的可选会话字段。
        canonical.pop("conversation_id", None)
        input_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if payload.idempotency_key:
            old = self.db.get_task_by_idempotency(payload.idempotency_key)
            if old:
                if old["input_hash"] != input_hash:
                    raise IdempotencyConflict("idempotency key was used with different input")
                return self.view(old)
        task_id = str(uuid4())
        try:
            self.db.create_task(task_id, payload.idempotency_key, input_hash, payload.prompt,
                                payload.save_policy.value, [item.model_dump(mode="json") for item in payload.attachments], token_settings=self.settings)
        except Exception as exc:
            if payload.idempotency_key and self.db.get_task_by_idempotency(payload.idempotency_key):
                return self.view(self.db.get_task_by_idempotency(payload.idempotency_key))
            raise exc
        for attachment in payload.attachments:
            self.db.grant_input(task_id, attachment.input_ref, task_id)
        self.db.ensure_budget(task_id)
        self.db.append_event(task_id, "task_created", {"status": "queued", "root_task_id": task_id})
        return self.view(self.db.get_task(task_id))

    def view(self, row: dict[str, Any] | None) -> TaskView:
        """聚合根任务及子任务收据、结果和澄清信息；unknown 单独计数，已知 token 合计不代表未知用量为零。"""
        if not row:
            raise KeyError("task not found")
        from research_loop.services.conversations import ConversationService
        try:
            ConversationService(self.db).reconcile(row["task_id"])
        except Exception as exc:
            # 派生记忆不可回滚已经完成的任务；读回时继续暴露可恢复告警。
            memory_warning = type(exc).__name__
        else:
            memory_warning = None
        result = json.loads(row["result_json"]) if row.get("result_json") else None
        calls = (self.db.list_model_calls(row["task_id"]) if row.get("parent_task_id") else
                 self.db.fetchall("SELECT c.* FROM model_calls c JOIN tasks t ON t.task_id=c.task_id WHERE t.root_task_id=?", (row["task_id"],)))
        # 本地发送前拒绝只有意图收据，不能计为一次远端模型调用；旧unknown保持原样。
        not_sent = sum(call.get("sent_state") == "not_sent" for call in calls)
        usage: dict[str, Any] = {"requests": len(calls) - not_sent, "not_sent": not_sent,
                                 "unknown": sum(call["status"] == "unknown" for call in calls)}
        totals = {}
        for call in calls:
            for key, value in json.loads(call.get("usage_json") or "{}").items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + value
        usage["tokens"] = totals
        if memory_warning:
            usage["memory_projection_warning"] = memory_warning
        root_task_id = row.get("root_task_id") or row["task_id"]
        usage["budget"] = self.db.budget_summary(root_task_id)
        subtasks = []
        for child in self.db.list_subtasks(row["task_id"]):
            raw = child.get("result") or {}
            if isinstance(raw, dict) and "result" in raw:
                raw = raw["result"] or {}
            child_row = self.db.get_task(child["subtask_id"])
            is_review = child_row and child_row.get("role") == "reviewer"
            detail = ({"role": "reviewer", "review_status": raw.get("status"), "review_result": raw,
                       "gaps": raw.get("gaps") or []} if is_review else raw)
            subtasks.append({**detail, "subtask_id": child["subtask_id"], "parent_task_id": row["task_id"],
                             "objective": child["objective"], "status": child["status"]})
        clarification = self.db.get_pending_clarification(row["task_id"])
        return TaskView(task_id=row["task_id"], status=TaskStatus(row["status"]), prompt=row["prompt"],
                        save_policy=SavePolicy(row["save_policy"]), result=TaskResult.model_validate(result) if result else None,
                        error=row.get("error"), created_at=row.get("created_at"), updated_at=row.get("updated_at"), usage=usage,
                        parent_task_id=row.get("parent_task_id"), role=row.get("role", "supervisor"),
                        subtasks=subtasks, clarification=clarification)

    def cancel(self, task_id: str, cancel_continuations: bool = True) -> TaskView:
        """取消根任务并通知未结束子任务；已结束任务返回原结果，不重写历史终态。"""
        if cancel_continuations:
            from research_loop.storage.memory import MemoryStore
            with MemoryStore(self.db).transaction() as tx:
                continuation=tx.one('SELECT * FROM task_continuations WHERE predecessor_id=?',(task_id,))
                tx.execute("UPDATE task_continuations SET status='cancelled' WHERE predecessor_id=? AND status='pending'",(task_id,))
            # 恰好已接续时，停止请求也传播到接续任务；不会重新开启已结束任务。
            if continuation and continuation['status']=='queued' and continuation['next_task_id']:
                self.cancel(continuation['next_task_id'])
        row = self.db.get_task(task_id)
        if not row:
            raise KeyError("task not found")
        if row["status"] in {"completed", "partial", "failed", "cancelled"}:
            return self.view(row)
        self.db.update_task(task_id, cancel_requested=1)
        for child in self.db.list_subtasks(task_id):
            if child["status"] in {"queued", "running"}:
                self.db.update_task(child["subtask_id"], cancel_requested=1, status="cancelled")
                self.db.execute("UPDATE subtasks SET status='cancelled',updated_at=? WHERE subtask_id=?",
                                (self.db_now(), child["subtask_id"]))
        # queued/needs_input没有正在执行的远端调用，可立即结算；running留给Worker安全收尾。
        if row["status"] in {"queued","needs_input"}:
            from research_loop.workers.executor import TaskExecutor
            from research_loop.storage.checkpoints import CheckpointStore
            from research_loop.config import get_settings
            executor=TaskExecutor(get_settings(),self.db)
            state=CheckpointStore(self.db).load(task_id) or {}
            result=executor._deliverable_result(task_id,state,"用户取消，保留已有成果")
            changed=self.db.execute("UPDATE tasks SET status='cancelled',result_json=?,updated_at=? WHERE task_id=? AND status IN ('queued','needs_input')",(result.model_dump_json(),self.db_now(),task_id))
            if changed:
                self.db.append_event(task_id,"task_finished",{"status":"cancelled","result":result.model_dump(mode="json")})
        self.db.append_event(task_id, "cancel_requested", {})
        return self.view(self.db.get_task(task_id))

    @staticmethod
    def db_now() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def request_clarification(self, task_id: str, question: str, fields: list[str],
                              suggested_values: list[str]) -> dict[str, Any]:
        """已有 pending 问题则复用，否则落库新问题与事件；这里尚不撤销 Worker 租约。"""
        pending = self.db.get_pending_clarification(task_id)
        if pending:
            return pending
        clarification = Clarification(task_id=task_id, question=question, fields=fields,
                                      suggested_values=suggested_values)
        self.db.save_clarification(clarification.model_dump(mode="json"))
        self.db.append_event(task_id, "clarification_requested", clarification.model_dump(mode="json"))
        return clarification.model_dump(mode="json")

    def resume(self, payload: ResumeInput, task_id: str) -> TaskView:
        """向原对话追加澄清回答并重新入队，不用回答替换原始任务。

        Args:
            payload: 问题 ID、用户回答及恢复幂等键。
            task_id: 处于 needs_input 的原任务 ID。

        Returns:
            恢复后的任务视图；原 prompt、保存权限和历史消息保持不变。

        Raises:
            KeyError: 任务不存在。
            IdempotencyConflict: 相同幂等键对应不同回答。
            ValueError: 问题过期、任务已取消或 checkpoint 的工具对话不完整。
        """
        # 并发回答串行处理，回答、checkpoint 与队列状态在同一事务提交。
        # 历史损坏的对话不能在此静默修补，否则可能重新执行先前工具或请求。
        with self.db.connection() as conn:
            cur = conn.cursor()
            def execute(sql, params=()):
                return cur.execute(sql if self.db.kind == "sqlite" else sql.replace("?", "%s"), params)
            if self.db.kind == "sqlite":
                execute("BEGIN IMMEDIATE")
            suffix = " FOR UPDATE" if self.db.kind == "mysql" else ""
            execute("SELECT * FROM tasks WHERE task_id=?" + suffix, (task_id,))
            row = cur.fetchone()
            if not row:
                raise KeyError("task not found")
            execute("SELECT * FROM resume_requests WHERE task_id=? AND idempotency_key=?", (task_id, payload.idempotency_key))
            existing = cur.fetchone()
            if existing:
                if existing["answer"] != payload.answer or existing["clarification_id"] != payload.clarification_id:
                    raise IdempotencyConflict("resume idempotency key was used with different input")
            else:
                if row["status"] != "needs_input" or row["cancel_requested"]:
                    raise ValueError("task is not safely paused for input or has been cancelled")
                execute("SELECT * FROM clarifications WHERE clarification_id=?", (payload.clarification_id,))
                clarification = cur.fetchone()
                if not clarification or clarification["task_id"] != task_id or clarification["status"] != "pending":
                    raise ValueError("clarification is missing, expired, or belongs to another task")
                execute("SELECT state_json FROM checkpoints WHERE task_id=?", (task_id,))
                checkpoint = cur.fetchone()
                state = json.loads(checkpoint["state_json"]) if checkpoint else {}
                if not state.get("messages") or state.get("pending_tool_calls") or state.get("final_reason") != "clarification requested":
                    raise ValueError("clarification checkpoint is incomplete; no resume or model resend")
                validate_messages(state["messages"])
                state["messages"] = state["messages"] + [{"role": "user", "content": "澄清回答：" + payload.answer}]
                # 业务恢复说明与用户原话分开；回答优先于暂定建议，已知上下文不丢弃。
                # 是否允许重定计划由CoverageService读取真实resume记录决定，而非此提示授权。
                state["messages"].append({"role": "system", "content":
                    "任务已收到本次澄清的用户回答。用户回答优先于此前的建议和暂定计划；先调用set_research_plan更新相关要求，保留用户未改变的原始目标和约束，再继续执行。不要为了满足旧建议数量额外检索。"})
                state.update(finished=False, final_answer=None, final_reason=None)
                state.pop("worker_id", None)
                state.pop("lease_token", None)
                now = self.db_now()
                execute("INSERT INTO resume_requests(request_id,task_id,clarification_id,idempotency_key,answer,created_at) VALUES(?,?,?,?,?,?)",
                        (str(uuid4()), task_id, payload.clarification_id, payload.idempotency_key, payload.answer, now))
                execute("UPDATE clarifications SET status='answered',answer=?,answered_at=? WHERE clarification_id=?",
                        (payload.answer, now, payload.clarification_id))
                execute("UPDATE checkpoints SET state_json=?,updated_at=? WHERE task_id=?",
                        (json.dumps(state, ensure_ascii=False), now, task_id))
                execute("UPDATE tasks SET status='queued',result_json=NULL,error=NULL,claimed_by=NULL,lease_token=NULL,lease_until=NULL,updated_at=? WHERE task_id=?", (now, task_id))
                execute("INSERT INTO task_events(event_id,task_id,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
                        (str(uuid4()), task_id, "task_resumed", json.dumps({"clarification_id": payload.clarification_id}), now))
        return self.view(self.db.get_task(task_id))

    def complete(self, task_id: str, status: TaskStatus, result: TaskResult, error: str | None = None,
                 owner: tuple[str, str] | None = None) -> TaskView:
        """带所有权参数时先检查租约，再写最终结果和结束事件；这是业务落库，不再调用模型。"""
        if owner and not self.db.is_owner(task_id, owner[0], owner[1]):
            raise RuntimeError("worker lease lost; task completion rejected")
        self.db.update_task(task_id, status=status.value, result_json=result.model_dump_json(), error=error,
                            cancel_requested=0, lease_until=None)
        self.db.append_event(task_id, "task_finished", {"status": status.value, "result": result.model_dump(mode="json")})
        return self.view(self.db.get_task(task_id))
