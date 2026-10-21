"""AG4持久层：会话、记忆和版本记录共用事务，不替代任务收据。"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from research_loop.storage.mysql import Database

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_owner (owner_id VARCHAR(64) PRIMARY KEY, revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS conversations (conversation_id VARCHAR(64) PRIMARY KEY, title TEXT NOT NULL, created_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS conversation_links (task_id VARCHAR(64) PRIMARY KEY, conversation_id VARCHAR(64) NOT NULL, message_id VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS conversation_messages (message_id VARCHAR(64) PRIMARY KEY, conversation_id VARCHAR(64) NOT NULL, sequence_no INTEGER NOT NULL, message_key VARCHAR(200) NOT NULL UNIQUE, task_id VARCHAR(64) NOT NULL, role VARCHAR(32) NOT NULL, content TEXT NOT NULL, created_at VARCHAR(64) NOT NULL, UNIQUE(conversation_id,sequence_no));
CREATE TABLE IF NOT EXISTS memory_entries (memory_id VARCHAR(64) PRIMARY KEY, slot VARCHAR(200) NOT NULL, kind VARCHAR(32) NOT NULL, content TEXT NOT NULL, source_json TEXT NOT NULL, status VARCHAR(32) NOT NULL, version INTEGER NOT NULL, created_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS memory_operations (request_key VARCHAR(200) PRIMARY KEY, payload_hash CHAR(64) NOT NULL, result_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS research_memories (task_id VARCHAR(64) PRIMARY KEY, conversation_id VARCHAR(64) NOT NULL, record_json TEXT NOT NULL, search_text TEXT NOT NULL, created_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS interest_projections (task_id VARCHAR(64) PRIMARY KEY, policy_version INTEGER NOT NULL, status VARCHAR(32) NOT NULL, result_json TEXT, created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS interest_sources (interest_slot VARCHAR(200) NOT NULL, task_id VARCHAR(64) NOT NULL, added_version INTEGER NOT NULL, source_json TEXT NOT NULL, created_at VARCHAR(64) NOT NULL, PRIMARY KEY(interest_slot,task_id));
CREATE TABLE IF NOT EXISTS context_archives (snapshot_id CHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, sequence_no INTEGER NOT NULL, original_json TEXT NOT NULL, projected_json TEXT NOT NULL, metrics_json TEXT NOT NULL, created_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS note_heads (note_id VARCHAR(64) PRIMARY KEY, source_draft_id VARCHAR(64) NOT NULL, current_version_id VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS note_versions (version_id VARCHAR(64) PRIMARY KEY, note_id VARCHAR(64) NOT NULL, base_hash CHAR(64) NOT NULL, content_hash CHAR(64) NOT NULL, title TEXT NOT NULL, markdown TEXT NOT NULL, evidence_json TEXT NOT NULL, path TEXT, created_at VARCHAR(64) NOT NULL);
"""


class Transaction:
    """适配SQLite/MySQL游标；所有值使用参数绑定。"""

    def __init__(self, db: Database, connection: Any):
        """保存当前事务游标，不创建第二条数据库连接。"""
        self.db = db
        self.cursor = connection.cursor()

    def execute(self, sql: str, params: tuple = ()) -> Any:
        """执行同一事务内的参数化SQL并返回游标。"""
        self.cursor.execute(sql if self.db.kind == "sqlite" else sql.replace("?", "%s"), params)
        return self.cursor

    def one(self, sql: str, params: tuple = ()) -> dict | None:
        """返回一行字典或None。"""
        row = self.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def all(self, sql: str, params: tuple = ()) -> list[dict]:
        """返回当前事务中的查询结果。"""
        return [dict(r) for r in self.execute(sql, params).fetchall()]


class MemoryStore:
    """单用户记忆存储；写操作锁定local行，避免并发提交交叉覆盖。"""

    def __init__(self, db: Database):
        """复用调用方数据库，不在构造函数迁移或写生产数据。"""
        self.db = db

    @contextmanager
    def transaction(self) -> Iterator[Transaction]:
        """提供可恢复写入的事务边界；首次建立单用户串行锁。"""
        with self.db.connection() as conn:
            tx = Transaction(self.db, conn)
            if self.db.kind == "sqlite":
                tx.execute("BEGIN IMMEDIATE")
                tx.execute("INSERT OR IGNORE INTO memory_owner(owner_id,revision) VALUES('local',0)")
            else:
                # 重复INSERT IGNORE会先持共享锁再升级，并发时可能死锁；直接取排他锁。
                tx.execute("INSERT INTO memory_owner(owner_id,revision) VALUES('local',0) ON DUPLICATE KEY UPDATE revision=revision")
            tx.one("SELECT * FROM memory_owner WHERE owner_id='local'" + (" FOR UPDATE" if self.db.kind == "mysql" else ""))
            yield tx


def check_owner(tx: Transaction, actor: tuple | None) -> None:
    """在同一事务中围栏Worker写入；人工API没有Worker身份，不伪造租约。"""
    if actor is None:
        return
    task_id,worker,token=actor
    suffix=' FOR UPDATE' if tx.db.kind=='mysql' else ''
    row=tx.one('SELECT * FROM tasks WHERE task_id=?'+suffix,(task_id,))
    from research_loop.storage.mysql import _now
    if not row or row['status']!='running' or row['cancel_requested'] or row['claimed_by']!=worker or row['lease_token']!=token or not row.get('lease_until') or row['lease_until']<_now():
        raise PermissionError('MEMORY_LEASE_LOST: cancelled or no longer owned')


def dump(value: Any) -> str:
    """生成稳定JSON，用于幂等hash和审计。"""
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def digest(value: Any) -> str:
    """按稳定JSON计算内容指纹。"""
    import hashlib
    return hashlib.sha256(dump(value).encode()).hexdigest()


def operation(tx: Transaction, key: str, payload: Any) -> dict | None:
    """重放相同操作；相同键不同参数必须报冲突。"""
    import json
    old = tx.one('SELECT * FROM memory_operations WHERE request_key=?', (key,))
    if old and old['payload_hash'] != digest(payload):
        raise ValueError('IDEMPOTENCY_CONFLICT: same key, different payload')
    return json.loads(old['result_json']) if old else None


def record_operation(tx: Transaction, key: str, payload: Any, result: Any) -> None:
    """在业务事务中保存可重复读回的操作结果。"""
    tx.execute('INSERT INTO memory_operations(request_key,payload_hash,result_json) VALUES(?,?,?)', (key,digest(payload),dump(result)))
