CREATE TABLE IF NOT EXISTS reflection_sessions (
 task_id VARCHAR(64) PRIMARY KEY,
 state_json LONGTEXT NOT NULL,
 updated_at VARCHAR(64) NOT NULL
);
