# 数据库迁移入口：显式执行版本迁移，避免应用启动时隐式重置已有任务和收据。
"""Explicit, idempotent migration entry point."""
from __future__ import annotations
from pathlib import Path
from research_loop.config import get_settings
from research_loop.storage.mysql import Database


def main() -> None:
    db = Database(get_settings().database_url)
    migration_dir = Path(__file__).resolve().parents[1] / "migrations"
    db.execute("CREATE TABLE IF NOT EXISTS schema_version (version VARCHAR(64) PRIMARY KEY, applied_at VARCHAR(64) NOT NULL)")
    applied = {row["version"] for row in db.fetchall("SELECT version FROM schema_version")}
    for path in sorted(migration_dir.glob("*.sql")):
        version = path.stem
        if version in applied:
            continue
        sql = path.read_text(encoding="utf-8")
        if db.kind == "sqlite":
            with db.connection() as conn:
                conn.executescript(sql)
        else:
            with db.connection() as conn:
                with conn.cursor() as cur:
                    clean_sql = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
                    for statement in (part.strip() for part in clean_sql.split(";")):
                        if statement:
                            cur.execute(statement)
        db.execute("INSERT INTO schema_version(version, applied_at) VALUES(?,CURRENT_TIMESTAMP)", (version,))
        print(f"applied {version}")
    print("researchloop migrations ready")


if __name__ == "__main__":
    main()
