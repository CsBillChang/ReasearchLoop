-- 交付后追加累计Token账本与阅读清单；不回填旧任务。

CREATE TABLE IF NOT EXISTS task_token_budgets (
 root_task_id VARCHAR(64) PRIMARY KEY, token_limit BIGINT NOT NULL,
 tail_tokens BIGINT NOT NULL, used_tokens BIGINT NOT NULL DEFAULT 0,
 reserved_tokens BIGINT NOT NULL DEFAULT 0, created_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS token_reservations (
 reservation_id VARCHAR(160) PRIMARY KEY, root_task_id VARCHAR(64) NOT NULL,
 task_id VARCHAR(64) NOT NULL, sequence_no BIGINT NOT NULL,
 request_hash CHAR(64) NOT NULL, reserved_tokens BIGINT NOT NULL,
 actual_tokens BIGINT, status VARCHAR(32) NOT NULL, created_at VARCHAR(64) NOT NULL,
 UNIQUE(task_id,sequence_no)
);

CREATE TABLE IF NOT EXISTS reading_list (
 paper_id VARCHAR(200) PRIMARY KEY, task_id VARCHAR(64) NOT NULL,
 metadata_json LONGTEXT NOT NULL, status VARCHAR(32) NOT NULL,
 version BIGINT NOT NULL, created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS reading_list_operations (
 operation_id VARCHAR(200) PRIMARY KEY, payload_hash CHAR(64) NOT NULL,
 paper_id VARCHAR(200) NOT NULL, created_at VARCHAR(64) NOT NULL
);
