"""最小阅读清单持久结构；论文状态不修改研究/核验/笔记记录。"""
SCHEMA = """
CREATE TABLE IF NOT EXISTS reading_list (
 paper_id VARCHAR(200) PRIMARY KEY, task_id VARCHAR(64) NOT NULL,
 metadata_json TEXT NOT NULL, status VARCHAR(32) NOT NULL,
 version INTEGER NOT NULL, created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS reading_list_operations (
 operation_id VARCHAR(200) PRIMARY KEY, payload_hash CHAR(64) NOT NULL,
 paper_id VARCHAR(200) NOT NULL, created_at VARCHAR(64) NOT NULL
);
"""
