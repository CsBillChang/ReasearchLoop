"""原文索引的MySQL事实与构建锁；不改写既有论文、页证据或任务记录。"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from research_loop.domain.retrieval import fingerprint
from research_loop.storage.mysql import Database, _now


SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_index_builds (
 index_id CHAR(64) PRIMARY KEY, document_id VARCHAR(255) NOT NULL,
 manifest_json TEXT NOT NULL, bm25_status VARCHAR(16) NOT NULL,
 vector_status VARCHAR(16) NOT NULL, last_error TEXT,
 created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_chunks (
 chunk_id CHAR(64) PRIMARY KEY, chunk_json TEXT NOT NULL,
 terms_json TEXT NOT NULL, token_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_index_members (
 index_id CHAR(64) NOT NULL, chunk_id CHAR(64) NOT NULL, ordinal_no INTEGER NOT NULL,
 embedding_json TEXT, PRIMARY KEY(index_id,chunk_id), UNIQUE(index_id,ordinal_no)
);
"""


class EvidenceIndexStore:
    """索引构建元数据、BM25词项与可回放向量；没有任务权限的全库读接口不对工具公开。"""

    def __init__(self, db: Database):
        """接收已有数据库适配器；建表仅由迁移或测试ensure_schema完成。"""
        self.db = db

    @contextmanager
    def lock(self, index_id: str):
        """串行构建同一索引，进程退出时自动释放；不占用长业务事务。"""
        key = "evidence:" + fingerprint([getattr(self.db, 'url', self.db.kind), index_id])[:48]
        if self.db.kind == "sqlite":
            path = Path(self.db.sqlite_path + ".index-locks")
            path.mkdir(exist_ok=True)
            with (path / index_id).open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("INDEX_BUILD_BUSY") from exc
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
        else:
            with self.db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT GET_LOCK(%s,0) AS acquired", (key,))
                    if cur.fetchone()["acquired"] != 1:
                        raise RuntimeError("INDEX_BUILD_BUSY")
                    try:
                        yield
                    finally:
                        cur.execute("SELECT RELEASE_LOCK(%s)", (key,))

    def prepare(self, index_id: str, manifest: dict, chunks: list, term_function) -> None:
        """事务保存不可变chunk/词项与成员清单，成功提交后BM25才为ready。

        Args:
            index_id: 完整manifest内容hash。
            manifest: 文档、页快照、切分、模型与集合身份。
            chunks: 与manifest一致的连续原文片段。
            term_function: 复用现有词项解析器，保持Memory接口不变。
        """
        from collections import Counter
        with self.db.connection() as conn:
            cur = conn.cursor()

            def execute(sql, values=()):
                """同一事务内执行跨SQLite/MySQL参数化SQL。"""
                return cur.execute(sql.replace("?", "%s") if self.db.kind == "mysql" else sql, values)

            execute("SELECT manifest_json FROM evidence_index_builds WHERE index_id=?", (index_id,))
            old = cur.fetchone()
            if old:
                if json.loads(old["manifest_json"]) != manifest:
                    raise ValueError("INDEX_MANIFEST_CONFLICT")
                return
            now = _now()
            execute("INSERT INTO evidence_index_builds VALUES(?,?,?,?,?,?,?,?)",
                (index_id, manifest["source"]["document_id"], json.dumps(manifest, ensure_ascii=False),
                 "pending", "pending", None, now, now))
            for n, chunk in enumerate(chunks):
                body = chunk.model_dump()
                counts = Counter(term_function(chunk.text))
                execute("SELECT chunk_json FROM evidence_chunks WHERE chunk_id=?", (chunk.chunk_id,))
                existing = cur.fetchone()
                if existing and json.loads(existing["chunk_json"]) != body:
                    raise ValueError("CHUNK_IDENTITY_CONFLICT")
                if not existing:
                    # 不同index可复用相同chunk；唯一键防止并发重复创建。
                    sql = "INSERT OR IGNORE" if self.db.kind == "sqlite" else "INSERT IGNORE"
                    execute(sql + " INTO evidence_chunks VALUES(?,?,?,?)", (chunk.chunk_id,
                        json.dumps(body, ensure_ascii=False), json.dumps(counts, ensure_ascii=False), sum(counts.values())))
                execute("INSERT INTO evidence_index_members(index_id,chunk_id,ordinal_no) VALUES(?,?,?)",
                        (index_id, chunk.chunk_id, n))
            execute("UPDATE evidence_index_builds SET bm25_status='ready' WHERE index_id=?", (index_id,))

    def get(self, index_id: str) -> dict:
        """读取构建状态与冻结清单，未知身份明确报错。"""
        row = self.db.fetchone("SELECT * FROM evidence_index_builds WHERE index_id=?", (index_id,))
        if not row:
            raise ValueError("INDEX_NOT_FOUND")
        row["manifest"] = json.loads(row.pop("manifest_json"))
        return row

    def members(self, index_id: str) -> list[dict]:
        """读回按序成员，向量日记用于中断恢复，调用方负责授权。"""
        rows = self.db.fetchall("SELECT m.*,c.chunk_json,c.terms_json,c.token_count FROM evidence_index_members m "
            "JOIN evidence_chunks c ON c.chunk_id=m.chunk_id WHERE m.index_id=? ORDER BY m.ordinal_no", (index_id,))
        for row in rows:
            row["chunk"] = json.loads(row.pop("chunk_json"))
            row["terms"] = json.loads(row.pop("terms_json"))
            row["embedding"] = json.loads(row.pop("embedding_json") or "null")
        return rows

    def vector_receipt(self, index_id: str, chunk_id: str, vector: list[float]) -> None:
        """先持久化成功编码结果，再发送Qdrant；恢复不重复已落盘的编码。"""
        self.db.execute("UPDATE evidence_index_members SET embedding_json=? WHERE index_id=? AND chunk_id=?",
                        (json.dumps(vector), index_id, chunk_id))

    def rebuild_terms(self, index_id: str, members: list[dict], term_function) -> None:
        """从已经验证的原文重建派生词项；事务中恢复BM25就绪状态。"""
        from collections import Counter
        with self.db.connection() as conn:
            cur = conn.cursor()
            for member in members:
                counts = Counter(term_function(member['chunk']['text']))
                if member['terms'] != counts or member['token_count'] != sum(counts.values()):
                    sql = "UPDATE evidence_chunks SET terms_json=?,token_count=? WHERE chunk_id=?"
                    cur.execute(sql.replace('?', '%s') if self.db.kind == 'mysql' else sql,
                        (json.dumps(counts, ensure_ascii=False), sum(counts.values()), member['chunk_id']))
            sql = "UPDATE evidence_index_builds SET bm25_status='ready' WHERE index_id=?"
            cur.execute(sql.replace('?', '%s') if self.db.kind == 'mysql' else sql, (index_id,))

    def state(self, index_id: str, vector_status: str, error: str | None = None) -> None:
        """只改新索引的向量状态，不回滚原文或已经可用的BM25。"""
        if vector_status not in {"pending", "partial", "ready", "failed"}:
            raise ValueError("invalid index status")
        self.db.execute("UPDATE evidence_index_builds SET vector_status=?,last_error=?,updated_at=? WHERE index_id=?",
                        (vector_status, error, _now(), index_id))

    def bm25_failed(self, index_id: str) -> None:
        """成员/原文身份损坏时撤销词项ready，不影响其他文档版本。"""
        self.db.execute("UPDATE evidence_index_builds SET bm25_status='failed',updated_at=? WHERE index_id=?",
                        (_now(), index_id))
