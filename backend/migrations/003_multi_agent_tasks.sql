-- AG2.1 parent/child task identity, access grants, clarification and review state.
ALTER TABLE tasks ADD COLUMN parent_task_id VARCHAR(64);
ALTER TABLE tasks ADD COLUMN root_task_id VARCHAR(64);
ALTER TABLE tasks ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'supervisor';
ALTER TABLE tasks ADD COLUMN brief_json LONGTEXT;
ALTER TABLE tasks ADD COLUMN quality_status VARCHAR(32) NOT NULL DEFAULT 'unreviewed';
ALTER TABLE tasks ADD COLUMN save_status VARCHAR(32) NOT NULL DEFAULT 'not_saved';
CREATE TABLE subtasks (
 subtask_id VARCHAR(64) PRIMARY KEY, parent_task_id VARCHAR(64) NOT NULL, root_task_id VARCHAR(64) NOT NULL,
 position_no BIGINT NOT NULL, objective LONGTEXT NOT NULL, spec_json LONGTEXT NOT NULL, status VARCHAR(32) NOT NULL,
 result_json LONGTEXT, idempotency_key VARCHAR(255) NOT NULL, created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL,
 UNIQUE(parent_task_id, idempotency_key), UNIQUE(parent_task_id, position_no)
);
CREATE TABLE task_papers (
 task_id VARCHAR(64) NOT NULL, paper_id VARCHAR(255) NOT NULL, permission VARCHAR(32) NOT NULL DEFAULT 'read',
 granted_by VARCHAR(64) NOT NULL, created_at VARCHAR(64) NOT NULL, PRIMARY KEY(task_id, paper_id)
);
CREATE TABLE task_inputs (
 task_id VARCHAR(64) NOT NULL, input_ref VARCHAR(200) NOT NULL, permission VARCHAR(32) NOT NULL DEFAULT 'read',
 granted_by VARCHAR(64) NOT NULL, created_at VARCHAR(64) NOT NULL, PRIMARY KEY(task_id, input_ref)
);
CREATE TABLE clarifications (
 clarification_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, question LONGTEXT NOT NULL,
 fields_json LONGTEXT NOT NULL, suggested_values_json LONGTEXT NOT NULL, status VARCHAR(32) NOT NULL,
 answer LONGTEXT, created_at VARCHAR(64) NOT NULL, answered_at VARCHAR(64)
);
CREATE TABLE review_results (
 review_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, draft_id VARCHAR(64) NOT NULL,
 content_hash CHAR(64) NOT NULL, status VARCHAR(32) NOT NULL, result_json LONGTEXT NOT NULL, created_at VARCHAR(64) NOT NULL,
 UNIQUE(task_id, draft_id, content_hash)
);
CREATE TABLE resume_requests (
 request_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, clarification_id VARCHAR(64) NOT NULL,
 idempotency_key VARCHAR(200) NOT NULL, answer LONGTEXT NOT NULL, created_at VARCHAR(64) NOT NULL,
 UNIQUE(task_id, idempotency_key)
);
CREATE TABLE task_budgets (
 root_task_id VARCHAR(64) PRIMARY KEY, model_limit BIGINT NOT NULL, tool_limit BIGINT NOT NULL,
 reserve_tail BIGINT NOT NULL, model_used BIGINT NOT NULL DEFAULT 0, tool_used BIGINT NOT NULL DEFAULT 0,
 reviewer_used BIGINT NOT NULL DEFAULT 0, created_at VARCHAR(64) NOT NULL
);
CREATE TABLE citation_stats (
 paper_id VARCHAR(255) PRIMARY KEY, provider VARCHAR(64) NOT NULL, status VARCHAR(32) NOT NULL,
 count_value BIGINT, provider_id VARCHAR(255), matched_by VARCHAR(64) NOT NULL, result_json LONGTEXT NOT NULL,
 queried_at VARCHAR(64), created_at VARCHAR(64) NOT NULL
);
