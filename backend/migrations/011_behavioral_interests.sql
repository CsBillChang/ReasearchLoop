-- 仅新任务提交时启用行为兴趣投影；不回填历史任务或画像。
CREATE TABLE IF NOT EXISTS interest_projections (
 task_id VARCHAR(64) PRIMARY KEY, policy_version BIGINT NOT NULL,
 status VARCHAR(32) NOT NULL, result_json LONGTEXT,
 created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS interest_sources (
 interest_slot VARCHAR(200) NOT NULL, task_id VARCHAR(64) NOT NULL,
 added_version BIGINT NOT NULL, source_json LONGTEXT NOT NULL, created_at VARCHAR(64) NOT NULL,
 PRIMARY KEY(interest_slot,task_id)
);
