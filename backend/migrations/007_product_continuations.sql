-- 网页取消后继续的持久请求；独立于旧任务、收据和保存权限。
CREATE TABLE IF NOT EXISTS task_continuations (
 continuation_id VARCHAR(64) PRIMARY KEY,
 predecessor_id VARCHAR(64) UNIQUE NOT NULL,
 conversation_id VARCHAR(64) NOT NULL,
 input_json LONGTEXT NOT NULL,
 input_hash CHAR(64) NOT NULL,
 status VARCHAR(32) NOT NULL,
 next_task_id VARCHAR(64),
 created_at VARCHAR(64) NOT NULL
);
