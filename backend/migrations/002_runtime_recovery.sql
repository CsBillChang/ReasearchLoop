-- AG2.0 runtime recovery and worker ownership additions. Additive only.
ALTER TABLE tasks ADD COLUMN lease_token VARCHAR(64);
ALTER TABLE model_calls ADD COLUMN request_hash CHAR(64);
ALTER TABLE model_calls ADD COLUMN sent_state VARCHAR(32) NOT NULL DEFAULT 'unknown';
ALTER TABLE model_calls ADD COLUMN attempt_no BIGINT NOT NULL DEFAULT 1;
CREATE INDEX ix_model_calls_task_position ON model_calls(task_id, sequence_no, attempt_no);
