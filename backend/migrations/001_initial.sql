-- AG1 minimal schema. Apply with backend/scripts/migrate.py; never runs on import.
CREATE TABLE IF NOT EXISTS schema_version (version VARCHAR(64) PRIMARY KEY, applied_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (
 task_id VARCHAR(64) PRIMARY KEY, idempotency_key VARCHAR(255) UNIQUE, input_hash CHAR(64) NOT NULL,
 prompt LONGTEXT NOT NULL, save_policy VARCHAR(16) NOT NULL, attachments_json LONGTEXT NOT NULL,
 status VARCHAR(32) NOT NULL, result_json LONGTEXT, error LONGTEXT, cancel_requested BIGINT NOT NULL DEFAULT 0,
 claimed_by VARCHAR(255), lease_until VARCHAR(64), created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS task_events (
 event_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, event_type VARCHAR(64) NOT NULL,
 payload_json LONGTEXT NOT NULL, created_at VARCHAR(64) NOT NULL
);
CREATE INDEX ix_task_events_task ON task_events(task_id, created_at);
CREATE TABLE IF NOT EXISTS actions (
 action_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, sequence_no BIGINT NOT NULL,
 tool_name VARCHAR(128) NOT NULL, arguments_json LONGTEXT NOT NULL, status VARCHAR(32) NOT NULL,
 observation_json LONGTEXT NOT NULL, created_at VARCHAR(64) NOT NULL, UNIQUE(task_id, sequence_no)
);
CREATE TABLE IF NOT EXISTS model_calls (
 call_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, sequence_no BIGINT NOT NULL,
 request_id VARCHAR(255), status VARCHAR(32) NOT NULL, request_json LONGTEXT NOT NULL,
 response_json LONGTEXT, usage_json LONGTEXT NOT NULL, finish_reason VARCHAR(64), created_at VARCHAR(64) NOT NULL,
 UNIQUE(task_id, sequence_no)
);
CREATE TABLE IF NOT EXISTS papers (paper_id VARCHAR(255) PRIMARY KEY, metadata_json LONGTEXT NOT NULL, created_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
 document_id VARCHAR(255) PRIMARY KEY, paper_id VARCHAR(255) NOT NULL, content_sha256 CHAR(64) NOT NULL,
 parser_name VARCHAR(128) NOT NULL, parser_version VARCHAR(64) NOT NULL, config_sha256 CHAR(64) NOT NULL,
 total_pages BIGINT NOT NULL, cache_path LONGTEXT NOT NULL, gaps_json LONGTEXT NOT NULL, created_at VARCHAR(64) NOT NULL,
 UNIQUE(content_sha256, parser_name, parser_version, config_sha256)
);
CREATE TABLE IF NOT EXISTS page_evidence (
 evidence_id VARCHAR(255) PRIMARY KEY, document_id VARCHAR(255) NOT NULL, page_number BIGINT NOT NULL,
 text LONGTEXT NOT NULL, parsed_markdown LONGTEXT NOT NULL, assets_json LONGTEXT NOT NULL, gaps_json LONGTEXT NOT NULL,
 UNIQUE(document_id, page_number)
);
CREATE TABLE IF NOT EXISTS task_evidence (
 task_id VARCHAR(64) NOT NULL, evidence_id VARCHAR(255) NOT NULL, returned_at VARCHAR(64) NOT NULL,
 PRIMARY KEY(task_id, evidence_id)
);
CREATE TABLE IF NOT EXISTS drafts (
 draft_id VARCHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, title LONGTEXT NOT NULL,
 markdown LONGTEXT NOT NULL, content_hash CHAR(64) NOT NULL, evidence_ids_json LONGTEXT NOT NULL,
 status VARCHAR(32) NOT NULL, created_at VARCHAR(64) NOT NULL, UNIQUE(task_id, content_hash)
);
CREATE TABLE IF NOT EXISTS save_receipts (
 receipt_id VARCHAR(64) PRIMARY KEY, draft_id VARCHAR(64) NOT NULL, content_hash CHAR(64) NOT NULL,
 path LONGTEXT NOT NULL, status VARCHAR(32) NOT NULL, created_at VARCHAR(64) NOT NULL,
 UNIQUE(draft_id, content_hash)
);
CREATE TABLE IF NOT EXISTS checkpoints (task_id VARCHAR(64) PRIMARY KEY, state_json LONGTEXT NOT NULL, updated_at VARCHAR(64) NOT NULL);
