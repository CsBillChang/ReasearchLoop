"""持久化适配器：生产使用 MySQL，隔离测试可使用 SQLite；保存任务事实、行动、收据、证据与租约。"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4
from urllib.parse import urlparse


SCHEMA = """
CREATE TABLE IF NOT EXISTS reflection_sessions (task_id VARCHAR(64) PRIMARY KEY, state_json TEXT NOT NULL, updated_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS schema_version (version VARCHAR(64) PRIMARY KEY, applied_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (
 task_id VARCHAR(64) PRIMARY KEY, idempotency_key VARCHAR(255) UNIQUE, input_hash CHAR(64) NOT NULL,
 prompt TEXT NOT NULL, save_policy VARCHAR(16) NOT NULL, attachments_json TEXT NOT NULL,
 status VARCHAR(32) NOT NULL, result_json TEXT, error TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
 claimed_by VARCHAR(255), lease_token VARCHAR(64), lease_until VARCHAR(64), parent_task_id VARCHAR(64),
 root_task_id VARCHAR(64), role VARCHAR(32) NOT NULL DEFAULT 'supervisor', brief_json TEXT,
 quality_status VARCHAR(32) NOT NULL DEFAULT 'unreviewed', save_status VARCHAR(32) NOT NULL DEFAULT 'not_saved',
 created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS task_events (
 event_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, event_type VARCHAR(64) NOT NULL,
 payload_json TEXT NOT NULL, created_at VARCHAR(64) NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_task_events_task ON task_events(task_id, created_at);
CREATE TABLE IF NOT EXISTS actions (
 action_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, sequence_no INTEGER NOT NULL,
 tool_name VARCHAR(128) NOT NULL, arguments_json TEXT NOT NULL, status VARCHAR(32) NOT NULL,
 observation_json TEXT NOT NULL, created_at VARCHAR(64) NOT NULL, UNIQUE(task_id, sequence_no)
);
CREATE TABLE IF NOT EXISTS model_calls (
 call_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, sequence_no INTEGER NOT NULL,
 request_id VARCHAR(255), status VARCHAR(32) NOT NULL, request_json TEXT NOT NULL,
 response_json TEXT, usage_json TEXT NOT NULL, finish_reason VARCHAR(64), request_hash CHAR(64),
 sent_state VARCHAR(32) NOT NULL DEFAULT 'unknown', attempt_no INTEGER NOT NULL DEFAULT 1,
 created_at VARCHAR(64) NOT NULL, UNIQUE(task_id, sequence_no, attempt_no)
);
CREATE TABLE IF NOT EXISTS papers (
 paper_id VARCHAR(255) PRIMARY KEY, metadata_json TEXT NOT NULL, created_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
 document_id VARCHAR(255) PRIMARY KEY, paper_id VARCHAR(255) NOT NULL, content_sha256 CHAR(64) NOT NULL,
 parser_name VARCHAR(128) NOT NULL, parser_version VARCHAR(64) NOT NULL, config_sha256 CHAR(64) NOT NULL,
 total_pages INTEGER NOT NULL, cache_path TEXT NOT NULL, gaps_json TEXT NOT NULL, created_at VARCHAR(64) NOT NULL,
 UNIQUE(content_sha256, parser_name, parser_version, config_sha256)
);
CREATE TABLE IF NOT EXISTS page_evidence (
 evidence_id VARCHAR(255) PRIMARY KEY, document_id VARCHAR(255) NOT NULL, page_number INTEGER NOT NULL,
 text TEXT NOT NULL, parsed_markdown TEXT NOT NULL, assets_json TEXT NOT NULL, gaps_json TEXT NOT NULL,
 UNIQUE(document_id, page_number)
);
CREATE TABLE IF NOT EXISTS task_evidence (
 task_id VARCHAR(64) NOT NULL, evidence_id VARCHAR(255) NOT NULL, returned_at VARCHAR(64) NOT NULL,
 PRIMARY KEY(task_id, evidence_id)
);
CREATE TABLE IF NOT EXISTS drafts (
 draft_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, title TEXT NOT NULL,
 markdown TEXT NOT NULL, content_hash CHAR(64) NOT NULL, evidence_ids_json TEXT NOT NULL,
 status VARCHAR(32) NOT NULL, created_at VARCHAR(64) NOT NULL, UNIQUE(task_id, content_hash)
);
CREATE TABLE IF NOT EXISTS note_exports (
 draft_id VARCHAR(64) NOT NULL, content_hash CHAR(64) NOT NULL, file_sha256 CHAR(64) NOT NULL,
 review_status VARCHAR(32), PRIMARY KEY(draft_id, content_hash)
);
CREATE TABLE IF NOT EXISTS save_receipts (
 receipt_id VARCHAR(64) PRIMARY KEY, draft_id VARCHAR(64) NOT NULL, content_hash CHAR(64) NOT NULL,
 path TEXT NOT NULL, status VARCHAR(32) NOT NULL, created_at VARCHAR(64) NOT NULL,
 UNIQUE(draft_id, content_hash)
);
CREATE TABLE IF NOT EXISTS checkpoints (
 task_id VARCHAR(64) PRIMARY KEY, state_json TEXT NOT NULL, updated_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS subtasks (
 subtask_id VARCHAR(64) PRIMARY KEY, parent_task_id VARCHAR(64) NOT NULL, root_task_id VARCHAR(64) NOT NULL,
 position_no INTEGER NOT NULL, objective TEXT NOT NULL, spec_json TEXT NOT NULL, status VARCHAR(32) NOT NULL,
 result_json TEXT, idempotency_key VARCHAR(255) NOT NULL, created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL,
 UNIQUE(parent_task_id, idempotency_key), UNIQUE(parent_task_id, position_no)
);
CREATE TABLE IF NOT EXISTS task_papers (
 task_id VARCHAR(64) NOT NULL, paper_id VARCHAR(255) NOT NULL, permission VARCHAR(32) NOT NULL DEFAULT 'read',
 granted_by VARCHAR(64) NOT NULL, created_at VARCHAR(64) NOT NULL, PRIMARY KEY(task_id, paper_id)
);
CREATE TABLE IF NOT EXISTS task_inputs (
 task_id VARCHAR(64) NOT NULL, input_ref VARCHAR(200) NOT NULL, permission VARCHAR(32) NOT NULL DEFAULT 'read',
 granted_by VARCHAR(64) NOT NULL, created_at VARCHAR(64) NOT NULL, PRIMARY KEY(task_id, input_ref)
);
CREATE TABLE IF NOT EXISTS clarifications (
 clarification_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, question TEXT NOT NULL,
 fields_json TEXT NOT NULL, suggested_values_json TEXT NOT NULL, status VARCHAR(32) NOT NULL,
 answer TEXT, created_at VARCHAR(64) NOT NULL, answered_at VARCHAR(64)
);
CREATE TABLE IF NOT EXISTS review_results (
 review_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, draft_id VARCHAR(64) NOT NULL,
 content_hash CHAR(64) NOT NULL, status VARCHAR(32) NOT NULL, result_json TEXT NOT NULL, created_at VARCHAR(64) NOT NULL,
 UNIQUE(task_id, draft_id, content_hash)
);
CREATE TABLE IF NOT EXISTS resume_requests (
 request_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, clarification_id VARCHAR(64) NOT NULL,
 idempotency_key VARCHAR(200) NOT NULL, answer TEXT NOT NULL, created_at VARCHAR(64) NOT NULL,
 UNIQUE(task_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS task_budgets (
 root_task_id VARCHAR(64) PRIMARY KEY, model_limit INTEGER NOT NULL, tool_limit INTEGER NOT NULL,
 reserve_tail INTEGER NOT NULL, model_used INTEGER NOT NULL DEFAULT 0, tool_used INTEGER NOT NULL DEFAULT 0,
 reviewer_used INTEGER NOT NULL DEFAULT 0, created_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS citation_stats (
 paper_id VARCHAR(255) PRIMARY KEY, provider VARCHAR(64) NOT NULL, status VARCHAR(32) NOT NULL,
 count_value INTEGER, provider_id VARCHAR(255), matched_by VARCHAR(64) NOT NULL, result_json TEXT NOT NULL,
 queried_at VARCHAR(64), created_at VARCHAR(64) NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


class Database:
    def __init__(self, url: str):
        self.url = url
        parsed = urlparse(url)
        self.kind = "sqlite" if parsed.scheme in ("sqlite", "sqlite3") else "mysql"
        self.sqlite_path: str | None = None
        if self.kind == "sqlite":
            self.sqlite_path = str(Path(parsed.path or ":memory:").expanduser())
            if self.sqlite_path != ":memory:":
                Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Iterator[Any]:
        """连接上下文正常退出则提交，异常则回滚，最后关闭；调用方可在一次事务内组合多条 SQL。"""
        if self.kind == "sqlite":
            conn = sqlite3.connect(self.sqlite_path or ":memory:", timeout=30)
            conn.row_factory = sqlite3.Row
        else:
            import pymysql
            parsed = urlparse(self.url)
            conn = pymysql.connect(
                host=parsed.hostname or "127.0.0.1", port=parsed.port or 3306,
                user=parsed.username or "researchloop", password=parsed.password or "",
                database=(parsed.path or "/researchloop").lstrip("/"), charset="utf8mb4",
                autocommit=False, cursorclass=pymysql.cursors.DictCursor,
            )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _sql(self, statement: str) -> str:
        if self.kind == "sqlite":
            return statement
        return (statement.replace("TEXT", "LONGTEXT").replace("INTEGER", "BIGINT")
                .replace("CREATE INDEX IF NOT EXISTS", "CREATE INDEX"))

    def ensure_schema(self) -> None:
        """幂等创建基础表和索引；已有数据库的结构升级仍由显式迁移脚本管理。"""
        from research_loop.storage.memory import SCHEMA as MEMORY_SCHEMA
        from research_loop.storage.continuations import SCHEMA as CONTINUATION_SCHEMA
        from research_loop.storage.evidence_index import SCHEMA as INDEX_SCHEMA
        from research_loop.storage.subscriptions import SCHEMA as SUBSCRIPTIONS_SCHEMA
        from research_loop.storage.token_budget import SCHEMA as TOKEN_SCHEMA
        from research_loop.storage.reading_list import SCHEMA as READING_SCHEMA
        schema = SCHEMA + MEMORY_SCHEMA + CONTINUATION_SCHEMA + INDEX_SCHEMA + SUBSCRIPTIONS_SCHEMA + TOKEN_SCHEMA + READING_SCHEMA
        with self.connection() as conn:
            if self.kind == "sqlite":
                conn.executescript(schema)
            else:
                with conn.cursor() as cur:
                    for statement in schema.split(";"):
                        if statement.strip():
                            try:
                                cur.execute(self._sql(statement))
                            except Exception as exc:
                                message = str(exc).lower()
                                if "already exists" not in message and "duplicate key name" not in message:
                                    raise

    def execute(self, statement: str, params: tuple[Any, ...] = ()) -> int:
        """参数化执行写入并返回受影响行数；按后端转换占位符，不在调用方拼接用户值。"""
        if self.kind == "mysql" and "INSERT OR REPLACE" in statement:
            statement = statement.replace("INSERT OR REPLACE", "REPLACE")
        with self.connection() as conn:
            cur = conn.execute(statement, params) if self.kind == "sqlite" else conn.cursor()
            if self.kind != "sqlite":
                cur.execute(statement.replace("?", "%s"), params)
            return cur.rowcount

    def fetchone(self, statement: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.connection() as conn:
            if self.kind == "sqlite":
                row = conn.execute(statement, params).fetchone()
                return dict(row) if row else None
            with conn.cursor() as cur:
                cur.execute(statement.replace("?", "%s"), params)
                row = cur.fetchone()
                return dict(row) if row else None

    def fetchall(self, statement: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connection() as conn:
            if self.kind == "sqlite":
                return [dict(row) for row in conn.execute(statement, params).fetchall()]
            with conn.cursor() as cur:
                cur.execute(statement.replace("?", "%s"), params)
                return [dict(row) for row in cur.fetchall()]

    def create_task(self, task_id: str, idempotency_key: str | None, input_hash: str, prompt: str,
                    save_policy: str, attachments: Any, *, parent_task_id: str | None = None,
                    root_task_id: str | None = None, role: str = "supervisor", brief: Any = None, token_settings=None) -> None:
        """插入任务基本信息并登记附件权限；父子任务共用任务表，通过 role 和 root_task_id 关联。"""
        now = _now()
        from research_loop.storage.memory import Transaction
        from research_loop.storage.token_budget import create_policy
        with self.connection() as conn:
            tx = Transaction(self, conn)
            tx.execute(
                "INSERT INTO tasks(task_id,idempotency_key,input_hash,prompt,save_policy,attachments_json,status,parent_task_id,root_task_id,role,brief_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, idempotency_key, input_hash, prompt, save_policy, _json(attachments), "queued",
                 parent_task_id, root_task_id or task_id, role, _json(brief) if brief is not None else None, now, now),
            )
            if token_settings is not None and parent_task_id is None:
                create_policy(tx, task_id, token_settings)
                self.create_budget(tx, task_id, token_settings)
        for attachment in attachments or []:
            input_ref = attachment.get("input_ref") if isinstance(attachment, dict) else None
            if input_ref:
                try:
                    self.grant_input(task_id, input_ref, task_id)
                except Exception:
                    # 旧 AG1 表结构由显式迁移升级；这里保留原有兼容分支，不隐式迁移。
                    pass

    @staticmethod
    def create_budget(tx, root_task_id: str, settings) -> None:
        """在新根任务事务中冻结共享请求/工具额度；不更新任何历史策略。

        Args:
            tx: 创建该任务的事务，失败时任务和策略一并回滚。
            root_task_id: 新根任务身份。
            settings: 本次实例配置，收尾预留至少给根任务留一次执行机会。
        """
        limit = int(settings.max_model_requests)
        tools = int(settings.max_tool_actions)
        tail = max(0, min(int(settings.reserve_model_requests), limit - 1))
        if limit < 1 or tools < 1:
            raise ValueError('INVALID_REQUEST_BUDGET: 请求及工具上限必须为正数')
        tx.execute('INSERT INTO task_budgets(root_task_id,model_limit,tool_limit,reserve_tail,created_at) VALUES(?,?,?,?,?)',
                   (root_task_id, limit, tools, tail, _now()))

    def assert_budget_policy(self, root_task_id: str, settings) -> dict:
        """新验收发送前读回断言；只报告不一致，不替恢复任务改写旧策略。

        Returns:
            已验证的数据库共享策略及Token账本。

        Raises:
            ValueError: 实际策略缺失或与冻结配置不一致。
        """
        row = self.budget_summary(root_task_id)
        expected = (settings.max_model_requests, settings.max_tool_actions,
                    max(0, min(settings.reserve_model_requests, settings.max_model_requests - 1)))
        actual = tuple((row or {}).get(k) for k in ('model_limit', 'tool_limit', 'reserve_tail'))
        tokens = (row or {}).get('tokens') or {}
        policy = row.get('execution_policy') or {}
        mode_matches = not policy or policy.get('mode', 'enforce') == settings.execution_budget_mode
        if not mode_matches or actual != expected or (tokens.get('token_limit'), tokens.get('tail_tokens')) != (settings.task_token_limit, settings.task_token_tail):
            raise ValueError('BUDGET_POLICY_MISMATCH: 数据库预算与冻结配置不一致，未授权发送')
        return row

    def ensure_budget(self, root_task_id: str, model_limit: int = 48, tool_limit: int = 80,
                      reserve_tail: int = 10) -> None:
        """首次为根任务建立额度；子 Agent 使用同一根记录，不各自获得额外的整批额度。"""
        try:
            self.execute("INSERT INTO task_budgets(root_task_id,model_limit,tool_limit,reserve_tail,created_at) VALUES(?,?,?,?,?)",
                         (root_task_id, model_limit, tool_limit, reserve_tail, _now()))
        except Exception:
            pass

    def reserve_model_slot(self, root_task_id: str, role: str = "supervisor") -> bool:
        """条件 UPDATE 预留请求位置；Researcher 不得占用 Supervisor/Reviewer 的收尾预留。"""
        budget = self.fetchone("SELECT * FROM task_budgets WHERE root_task_id=?", (root_task_id,))
        if not budget:
            return True
        policy = self.execution_policy(root_task_id) or {}
        condition = ("1=1" if policy.get("mode") == "observe" else
                     "model_used < model_limit" if role in {"supervisor", "reviewer"} else
                     "model_used < model_limit - reserve_tail")
        updated = self.execute(f"UPDATE task_budgets SET model_used=model_used+1, reviewer_used=reviewer_used+? WHERE root_task_id=? AND {condition}",
                               (1 if role == "reviewer" else 0, root_task_id))
        return updated == 1

    def reserve_tool_slot(self, root_task_id: str, reserve_tail: int = 0) -> bool:
        """原子领取工具额度并保护收尾空间；恢复已有行动不重复领取。

        Args:
            root_task_id: 根任务身份。
            reserve_tail: 本次行动不可占用的剩余槽位，不改变冻结总上限。
        """
        if reserve_tail < 0:
            raise ValueError("reserve_tail must be nonnegative")
        budget = self.fetchone("SELECT * FROM task_budgets WHERE root_task_id=?", (root_task_id,))
        if not budget:
            return True
        if (self.execution_policy(root_task_id) or {}).get("mode") == "observe":
            return self.execute("UPDATE task_budgets SET tool_used=tool_used+1 WHERE root_task_id=?", (root_task_id,)) == 1
        return self.execute("UPDATE task_budgets SET tool_used=tool_used+1 WHERE root_task_id=? AND tool_used < tool_limit - ?",
                            (root_task_id, reserve_tail)) == 1

    def execution_policy(self, root_task_id: str) -> dict | None:
        """读取根任务冻结策略，历史任务缺失时不回填。"""
        from research_loop.storage.token_budget import TokenBudgetStore
        return TokenBudgetStore(self).execution_policy(root_task_id)

    def active_role_count(self, root_task_id: str) -> int:
        """仅计已占执行槽的子角色，排队项不预占其他角色交接槽。"""
        row = self.fetchone("SELECT COUNT(*) AS n FROM tasks WHERE root_task_id=? AND parent_task_id IS NOT NULL AND status='running'", (root_task_id,))
        return int(row["n"])

    def delivery_tool_reserve(self, root_task_id: str) -> int:
        """根据实际保存权限与成功行动保留尚未完成的创建、保存和交付步骤。"""
        task = self.get_task(root_task_id) or {}
        if task.get('save_policy') != 'allow':
            return 1
        completed = {a['tool_name'] for a in self.list_actions(root_task_id) if a['status'] == 'succeeded'}
        return 1 + int('create_note' not in completed) + int('save_note' not in completed)

    def budget_summary(self, root_task_id: str) -> dict[str, Any]:
        from research_loop.storage.token_budget import TokenBudgetStore
        result = self.fetchone("SELECT * FROM task_budgets WHERE root_task_id=?", (root_task_id,)) or {}
        if policy := self.execution_policy(root_task_id):
            result["execution_policy"] = policy
        if tokens := TokenBudgetStore(self).summary(root_task_id):
            result["tokens"] = tokens
        return result

    def save_citation_stats(self, stats: dict[str, Any]) -> None:
        self.execute("INSERT OR REPLACE INTO citation_stats(paper_id,provider,status,count_value,provider_id,matched_by,result_json,queried_at,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                     (stats["paper_id"], stats.get("provider", "openalex"), stats["status"], stats.get("count"),
                      stats.get("provider_id"), stats.get("matched_by", "none"), _json(stats), stats.get("queried_at"), _now()))

    def get_citation_stats(self, paper_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT result_json FROM citation_stats WHERE paper_id=?", (paper_id,))
        return json.loads(row["result_json"]) if row else None

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self.fetchone("SELECT * FROM tasks WHERE task_id=?", (task_id,))

    def get_task_by_idempotency(self, key: str) -> dict[str, Any] | None:
        return self.fetchone("SELECT * FROM tasks WHERE idempotency_key=?", (key,))

    def create_subtask(self, parent_task_id: str, root_task_id: str, position: int,
                       objective: str, spec: Any, idempotency_key: str) -> dict[str, Any]:
        """按父任务和分派幂等键复用或创建子任务，保存目标与范围；并非复制父任务全部对话。"""
        existing = self.fetchone("SELECT * FROM subtasks WHERE parent_task_id=? AND idempotency_key=?",
                                 (parent_task_id, idempotency_key))
        if existing:
            return existing
        subtask_id = str(uuid4())
        now = _now()
        self.execute("INSERT INTO subtasks(subtask_id,parent_task_id,root_task_id,position_no,objective,spec_json,status,idempotency_key,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (subtask_id, parent_task_id, root_task_id, position, objective, _json(spec), "queued", idempotency_key, now, now))
        import hashlib
        self.create_task(subtask_id, f"subtask:{parent_task_id}:{idempotency_key}",
                         hashlib.sha256(objective.encode()).hexdigest(), objective, "deny", [],
                         parent_task_id=parent_task_id, root_task_id=root_task_id, role="researcher", brief=spec)
        return self.fetchone("SELECT * FROM subtasks WHERE subtask_id=?", (subtask_id,)) or {}

    def list_subtasks(self, parent_task_id: str) -> list[dict[str, Any]]:
        rows = self.fetchall("SELECT * FROM subtasks WHERE parent_task_id=? ORDER BY position_no", (parent_task_id,))
        for row in rows:
            row["spec"] = json.loads(row.pop("spec_json"))
            row["result"] = json.loads(row.pop("result_json")) if row.get("result_json") else None
        return rows

    def get_subtask(self, subtask_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT * FROM subtasks WHERE subtask_id=?", (subtask_id,))
        if row:
            row["spec"] = json.loads(row.pop("spec_json"))
            row["result"] = json.loads(row.pop("result_json")) if row.get("result_json") else None
        return row

    def save_subtask_result(self, subtask_id: str, status: str, result: Any, error: str | None = None) -> None:
        self.execute("UPDATE subtasks SET status=?,result_json=?,updated_at=? WHERE subtask_id=?",
                     (status, _json({"result": result, "error": error}), _now(), subtask_id))
        self.update_task(subtask_id, status=status, result_json=_json(result), error=error)

    def grant_paper(self, task_id: str, paper_id: str, granted_by: str) -> None:
        """登记任务级论文读取授权；候选存在于全库并不等于任意任务都有权读取。"""
        try:
            self.execute("INSERT INTO task_papers(task_id,paper_id,permission,granted_by,created_at) VALUES(?,?,?,?,?)",
                         (task_id, paper_id, "read", granted_by, _now()))
        except Exception:
            pass

    def task_has_paper(self, task_id: str, paper_id: str) -> bool:
        return self.fetchone("SELECT 1 AS ok FROM task_papers WHERE task_id=? AND paper_id=?", (task_id, paper_id)) is not None

    def grant_input(self, task_id: str, input_ref: str, granted_by: str) -> None:
        """登记附件 input_ref 的授权，供父任务把选定附件传给子任务。"""
        try:
            self.execute("INSERT INTO task_inputs(task_id,input_ref,permission,granted_by,created_at) VALUES(?,?,?,?,?)",
                         (task_id, input_ref, "read", granted_by, _now()))
        except Exception:
            pass

    def task_has_input(self, task_id: str, input_ref: str) -> bool:
        return self.fetchone("SELECT 1 AS ok FROM task_inputs WHERE task_id=? AND input_ref=?", (task_id, input_ref)) is not None

    def save_clarification(self, clarification: dict[str, Any]) -> None:
        """保存问题但不在此发布 needs_input；必须先等 Worker 持久化工具回复和 checkpoint。"""
        self.execute("INSERT INTO clarifications(clarification_id,task_id,question,fields_json,suggested_values_json,status,created_at) VALUES(?,?,?,?,?,?,?)",
                     (clarification["clarification_id"], clarification["task_id"], clarification["question"],
                      _json(clarification.get("fields", [])), _json(clarification.get("suggested_values", [])), "pending", _now()))
        # Worker 先保存工具观察和 checkpoint，随后才发布 needs_input。
        # 如果在此先发布暂停状态，会提前失去自己的写入租约。

    def get_clarification(self, clarification_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT * FROM clarifications WHERE clarification_id=?", (clarification_id,))
        if row:
            row["fields"] = json.loads(row.pop("fields_json"))
            row["suggested_values"] = json.loads(row.pop("suggested_values_json"))
        return row

    def get_pending_clarification(self, task_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT * FROM clarifications WHERE task_id=? AND status='pending' ORDER BY created_at DESC LIMIT 1", (task_id,))
        if row:
            row["fields"] = json.loads(row.pop("fields_json"))
            row["suggested_values"] = json.loads(row.pop("suggested_values_json"))
        return row

    def answer_clarification(self, clarification_id: str, answer: str) -> bool:
        return self.execute("UPDATE clarifications SET status='answered',answer=?,answered_at=? WHERE clarification_id=? AND status='pending'",
                            (answer, _now(), clarification_id)) == 1

    def save_resume_request(self, task_id: str, clarification_id: str, idempotency_key: str, answer: str) -> dict[str, Any] | None:
        existing = self.fetchone("SELECT * FROM resume_requests WHERE task_id=? AND idempotency_key=?", (task_id, idempotency_key))
        if existing:
            return existing
        request_id = str(uuid4())
        self.execute("INSERT INTO resume_requests(request_id,task_id,clarification_id,idempotency_key,answer,created_at) VALUES(?,?,?,?,?,?)",
                     (request_id, task_id, clarification_id, idempotency_key, answer, _now()))
        return self.fetchone("SELECT * FROM resume_requests WHERE request_id=?", (request_id,))

    def save_review_result(self, result: dict[str, Any]) -> None:
        """更新当前稿投影；历史各轮的原始结论保留在 reflection_sessions 中。"""
        existing = self.get_review_result(result["task_id"], result["draft_id"], result["content_hash"])
        if existing:
            self.execute("UPDATE review_results SET review_id=?,status=?,result_json=?,created_at=? WHERE task_id=? AND draft_id=? AND content_hash=?",
                (result["review_id"], result["status"], _json(result), _now(), result["task_id"], result["draft_id"], result["content_hash"]))
            self.update_task(result["task_id"], quality_status=result["status"])
            return
        self.execute("INSERT INTO review_results(review_id,task_id,draft_id,content_hash,status,result_json,created_at) VALUES(?,?,?,?,?,?,?)",
                     (result["review_id"], result["task_id"], result["draft_id"], result["content_hash"],
                      result["status"], _json(result), _now()))
        self.update_task(result["task_id"], quality_status=result["status"])

    def get_review_result(self, task_id: str, draft_id: str, content_hash: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT result_json FROM review_results WHERE task_id=? AND draft_id=? AND content_hash=?",
                            (task_id, draft_id, content_hash))
        return json.loads(row["result_json"]) if row else None

    def update_task(self, task_id: str, **fields: Any) -> None:
        allowed = {"status", "result_json", "error", "cancel_requested", "claimed_by", "lease_token", "lease_until", "quality_status", "save_status", "brief_json", "role", "parent_task_id", "root_task_id"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        if not fields:
            return
        fields["updated_at"] = _now()
        assignments = ",".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE tasks SET {assignments} WHERE task_id=?", tuple(fields.values()) + (task_id,))

    def append_event(self, task_id: str, event_type: str, payload: Any) -> str:
        event_id = str(uuid4())
        self.execute("INSERT INTO task_events(event_id,task_id,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
                     (event_id, task_id, event_type, _json(payload), _now()))
        return event_id

    def list_events(self, task_id: str, after_id: str | None = None) -> list[dict[str, Any]]:
        rows = self.fetchall("SELECT * FROM task_events WHERE task_id=? ORDER BY created_at,event_id", (task_id,))
        if after_id:
            ids = [row["event_id"] for row in rows]
            if after_id in ids:
                rows = rows[ids.index(after_id) + 1:]
        for row in rows:
            row["payload"] = json.loads(row.pop("payload_json"))
        return rows

    def save_action(self, action: dict[str, Any]) -> None:
        """保存工具行动及观察，供恢复时按任务/顺序读取；结果成功与否均应保留。"""
        existing = self.fetchone("SELECT action_id,status FROM actions WHERE task_id=? AND sequence_no=?",
                                 (action["task_id"], action["sequence"]))
        if existing:
            if existing["status"] == "succeeded" and action["status"] != "succeeded":
                return
            self.execute("UPDATE actions SET status=?, observation_json=? WHERE task_id=? AND sequence_no=?",
                         (action["status"], _json(action.get("observation", {})), action["task_id"], action["sequence"]))
            return
        self.execute(
            "INSERT INTO actions(action_id,task_id,sequence_no,tool_name,arguments_json,status,observation_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (action["action_id"], action["task_id"], action["sequence"], action["tool_name"],
             _json(action["arguments"]), action["status"], _json(action.get("observation", {})), _now()),
        )

    def save_model_call(self, task_id: str, sequence: int, status: str, request: Any,
                        response: Any = None, usage: Any = None, request_id: str | None = None,
                        finish_reason: str | None = None, sent_state: str = "unknown",
                        request_hash: str | None = None) -> dict[str, Any]:
        """保存原始请求、响应及已知 usage；恢复时成功收据优先复用，unknown 不补成成功或零用量。"""
        from research_loop.storage.token_budget import TokenBudgetStore
        import hashlib
        request_hash = request_hash or hashlib.sha256(_json(request).encode()).hexdigest()
        old = self.get_model_call(task_id, sequence)
        if old:
            if old["status"] == "succeeded":
                TokenBudgetStore(self).settle(task_id, sequence)
                return old
            self.execute("UPDATE model_calls SET status=?, request_id=COALESCE(request_id,?), response_json=COALESCE(?,response_json), usage_json=?, finish_reason=COALESCE(?,finish_reason), sent_state=? WHERE call_id=?",
                         (status, request_id, _json(response) if response is not None else None,
                          _json(usage or {}), finish_reason, sent_state, old["call_id"]))
            TokenBudgetStore(self).settle(task_id, sequence)
            return self.get_model_call(task_id, sequence) or old
        call_id = str(uuid4())
        self.execute(
            "INSERT INTO model_calls(call_id,task_id,sequence_no,request_id,status,request_json,response_json,usage_json,finish_reason,request_hash,sent_state,attempt_no,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (call_id, task_id, sequence, request_id, status, _json(request), _json(response) if response is not None else None,
             _json(usage or {}), finish_reason, request_hash, sent_state, 1, _now()),
        )
        TokenBudgetStore(self).settle(task_id, sequence)
        return self.get_model_call(task_id, sequence) or {"call_id": call_id, "status": status}

    def begin_model_call(self, task_id: str, sequence: int, request: Any) -> dict[str, Any]:
        """先建立请求意图收据，再允许网络发送；尚未发送与远端结果未知需要分开表达。"""
        return self.save_model_call(task_id, sequence, "started", request, sent_state="not_sent")

    def mark_model_call_sent(self, task_id: str, sequence: int) -> None:
        """记录已进入发送窗口；进程在此之后退出时不能假定供应商没有执行。"""
        self.execute("UPDATE model_calls SET sent_state='sent' WHERE task_id=? AND sequence_no=? AND status='started'",
                     (task_id, sequence))

    def get_model_call(self, task_id: str, sequence: int) -> dict[str, Any] | None:
        return self.fetchone("SELECT * FROM model_calls WHERE task_id=? AND sequence_no=? ORDER BY attempt_no DESC LIMIT 1",
                             (task_id, sequence))

    def save_paper(self, paper_id: str, metadata: Any) -> None:
        self.execute("INSERT OR REPLACE INTO papers(paper_id,metadata_json,created_at) VALUES(?,?,?)",
                     (paper_id, _json(metadata), _now()))

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT * FROM papers WHERE paper_id=?", (paper_id,))
        if row:
            row["metadata"] = json.loads(row.pop("metadata_json"))
        return row

    def save_document(self, document: dict[str, Any]) -> None:
        """保存 PDF 解析缓存身份；页级正文由 save_page 单独保存，任务授权另记。"""
        self.execute(
            "INSERT OR REPLACE INTO documents(document_id,paper_id,content_sha256,parser_name,parser_version,config_sha256,total_pages,cache_path,gaps_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (document["document_id"], document["paper_id"], document["content_sha256"], document["parser_name"],
             document["parser_version"], document["config_sha256"], document["total_pages"] or 0, document["cache_path"],
             _json(document.get("gaps", [])), _now()))

    def get_document_by_cache(self, content_hash: str, parser_name: str, parser_version: str, config_hash: str) -> dict[str, Any] | None:
        return self.fetchone("SELECT * FROM documents WHERE content_sha256=? AND parser_name=? AND parser_version=? AND config_sha256=?",
                             (content_hash, parser_name, parser_version, config_hash))

    def save_page(self, page: dict[str, Any]) -> None:
        self.execute(
            "INSERT OR REPLACE INTO page_evidence(evidence_id,document_id,page_number,text,parsed_markdown,assets_json,gaps_json) VALUES(?,?,?,?,?,?,?)",
            (page["evidence_id"], page["document_id"], page["page_number"] or 0, page["text"], page.get("parsed_markdown", ""),
             _json(page.get("assets", [])), _json(page.get("gaps", []))))

    def get_page(self, evidence_id: str) -> dict[str, Any] | None:
        # 原表 NOT NULL：0 专用于无页码全文；业务接口始终返回 None，绝不显示为第 0 页。
        row = self.fetchone("SELECT * FROM page_evidence WHERE evidence_id=?", (evidence_id,))
        if row and row["page_number"] == 0:
            row["page_number"] = None
        return row

    def mark_task_evidence(self, task_id: str, evidence_id: str) -> None:
        """记录某页已返回给当前任务；页在全库存在并不能替代任务级证据授权。"""
        try:
            self.execute("INSERT INTO task_evidence(task_id,evidence_id,returned_at) VALUES(?,?,?)",
                         (task_id, evidence_id, _now()))
        except Exception:
            pass

    def task_has_evidence(self, task_id: str, evidence_id: str) -> bool:
        return self.fetchone("SELECT 1 AS ok FROM task_evidence WHERE task_id=? AND evidence_id=?", (task_id, evidence_id)) is not None

    def list_task_evidence(self, task_id: str) -> list[str]:
        return [row["evidence_id"] for row in self.fetchall("SELECT evidence_id FROM task_evidence WHERE task_id=?", (task_id,))]

    def grant_evidence(self, task_id: str, evidence_id: str, granted_by: str) -> None:
        """将父子协作允许共享的页证据授予接收任务，不修改原页面内容。"""
        try:
            self.execute("INSERT INTO task_evidence(task_id,evidence_id,returned_at) VALUES(?,?,?)",
                         (task_id, evidence_id, _now()))
        except Exception:
            pass

    def list_task_papers(self, task_id: str) -> list[str]:
        return [row["paper_id"] for row in self.fetchall("SELECT paper_id FROM task_papers WHERE task_id=?", (task_id,))]

    def save_draft(self, draft: dict[str, Any]) -> None:
        self.execute(
            "INSERT OR REPLACE INTO drafts(draft_id,task_id,title,markdown,content_hash,evidence_ids_json,status,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (draft["draft_id"], draft["task_id"], draft["title"], draft["markdown"], draft["content_hash"],
             _json(draft.get("evidence_ids", [])), draft.get("status", "draft"), _now()))

    def get_draft(self, draft_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT * FROM drafts WHERE draft_id=?", (draft_id,))
        if row:
            row["evidence_ids"] = json.loads(row.pop("evidence_ids_json"))
        return row

    def save_receipt(self, receipt_id: str, draft_id: str, content_hash: str, path: str, status: str) -> None:
        """文件写入后的业务收据；文件真实性仍需保存服务读回路径和内容 hash 核验。"""
        self.execute("INSERT OR REPLACE INTO save_receipts(receipt_id,draft_id,content_hash,path,status,created_at) VALUES(?,?,?,?,?,?)",
                     (receipt_id, draft_id, content_hash, path, status, _now()))

    def get_receipt(self, draft_id: str, content_hash: str) -> dict[str, Any] | None:
        return self.fetchone("SELECT * FROM save_receipts WHERE draft_id=? AND content_hash=?", (draft_id, content_hash))

    def save_checkpoint(self, task_id: str, state: Any) -> None:
        self.execute("INSERT OR REPLACE INTO checkpoints(task_id,state_json,updated_at) VALUES(?,?,?)",
                     (task_id, _json(state), _now()))

    def get_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT state_json FROM checkpoints WHERE task_id=?", (task_id,))
        return json.loads(row["state_json"]) if row else None

    def list_model_calls(self, task_id: str) -> list[dict[str, Any]]:
        return self.fetchall("SELECT * FROM model_calls WHERE task_id=? ORDER BY sequence_no", (task_id,))

    def has_unknown_model_call(self, task_id: str) -> bool:
        """检查单任务是否有不可安全重发的请求；只返回状态，不自动修复历史收据。"""
        return self.fetchone("SELECT 1 AS found FROM model_calls WHERE task_id=? AND (status='unknown' OR (status='started' AND sent_state <> 'not_sent')) LIMIT 1", (task_id,)) is not None

    def has_unknown_family_call(self, root_task_id: str) -> bool:
        # 其他 Researcher 正在执行的 started 请求不是遗留请求，不能仅凭该状态误判为失联。
        """根任务或任一子任务有未结算请求时阻断家族继续发模型请求，避免盲重发。"""
        return self.fetchone("SELECT 1 AS found FROM model_calls c JOIN tasks t ON t.task_id=c.task_id WHERE t.root_task_id=? AND c.status='unknown' LIMIT 1", (root_task_id,)) is not None

    def list_actions(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.fetchall("SELECT * FROM actions WHERE task_id=? ORDER BY sequence_no", (task_id,))
        for row in rows:
            row["arguments"] = json.loads(row.pop("arguments_json"))
            row["observation"] = json.loads(row.pop("observation_json"))
        return rows

    def get_action(self, task_id: str, sequence: int) -> dict[str, Any] | None:
        rows = self.list_actions(task_id)
        return next((row for row in rows if row["sequence_no"] == sequence), None)

    def is_owner(self, task_id: str, worker_id: str, lease_token: str | None = None) -> bool:
        """按 Worker 身份、租约 token 和有效期验证当前写入权；旧进程不能凭 task_id 继续提交。"""
        row = self.fetchone("SELECT claimed_by,lease_token,lease_until,status FROM tasks WHERE task_id=?", (task_id,))
        if not row or row["claimed_by"] != worker_id or row["status"] != "running":
            return False
        if lease_token and row.get("lease_token") != lease_token:
            return False
        return bool(row.get("lease_until") and row["lease_until"] >= _now())

    def renew_lease(self, task_id: str, worker_id: str, lease_token: str, lease_seconds: int = 300) -> bool:
        """仅当前持有者可延长运行租约；心跳不等于业务已经完成。"""
        until = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        return self.execute("UPDATE tasks SET lease_until=?,updated_at=? WHERE task_id=? AND status='running' AND claimed_by=? AND lease_token=?",
                            (until, _now(), task_id, worker_id, lease_token)) == 1

    def claim_task(self, worker_id: str, lease_seconds: int = 300, task_id: str | None = None) -> dict[str, Any] | None:
        """在数据库事务中领取根任务并生成租约；MySQL 使用行锁，防止多个 Worker 同时执行。"""
        if task_id:
            current = self.get_task(task_id)
            if current and current.get("status") == "running" and current.get("claimed_by") == worker_id and current.get("lease_until", "") >= _now():
                return current
        token = str(uuid4())
        now = _now()
        until = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        with self.connection() as conn:
            if self.kind == "sqlite":
                conn.execute("BEGIN IMMEDIATE")
                where = "task_id=? AND (status='queued' OR (status='running' AND lease_until < ?))" if task_id else "parent_task_id IS NULL AND (status='queued' OR (status='running' AND lease_until < ?))"
                params = (task_id, now) if task_id else (now,)
                row = conn.execute("SELECT * FROM tasks WHERE " + where + " ORDER BY created_at LIMIT 1", params).fetchone()
                if not row:
                    return None
                changed = conn.execute("UPDATE tasks SET status='running',claimed_by=?,lease_token=?,lease_until=?,updated_at=? WHERE task_id=? AND (status='queued' OR (status='running' AND lease_until < ?))",
                                       (worker_id, token, until, now, row["task_id"], now)).rowcount
                if changed != 1:
                    return None
                return dict(conn.execute("SELECT * FROM tasks WHERE task_id=?", (row["task_id"],)).fetchone())
            with conn.cursor() as cur:
                where = "task_id=%s AND (status='queued' OR (status='running' AND lease_until < %s))" if task_id else "parent_task_id IS NULL AND (status='queued' OR (status='running' AND lease_until < %s))"
                params = (task_id, now) if task_id else (now,)
                cur.execute("SELECT * FROM tasks WHERE " + where + " ORDER BY created_at LIMIT 1 FOR UPDATE", params)
                row = cur.fetchone()
                if not row:
                    return None
                cur.execute("UPDATE tasks SET status='running',claimed_by=%s,lease_token=%s,lease_until=%s,updated_at=%s WHERE task_id=%s AND (status='queued' OR (status='running' AND lease_until < %s))",
                            (worker_id, token, until, now, row["task_id"], now))
                if cur.rowcount != 1:
                    return None
                cur.execute("SELECT * FROM tasks WHERE task_id=%s", (row["task_id"],))
                return dict(cur.fetchone())
