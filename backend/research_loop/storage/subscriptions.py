"""简报持久表；与研究任务、模型收据和原文索引分开，不改变它们的事实。"""
SCHEMA = '''
CREATE TABLE IF NOT EXISTS subscriptions (
 subscription_id VARCHAR(64) PRIMARY KEY, revision INTEGER NOT NULL, config_json TEXT NOT NULL,
 enabled INTEGER NOT NULL, next_due VARCHAR(64) NOT NULL, conversation_id VARCHAR(64) NOT NULL,
 created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS brief_runs (
 run_id VARCHAR(64) PRIMARY KEY, subscription_id VARCHAR(64) NOT NULL, revision INTEGER NOT NULL,
 slot VARCHAR(64) NOT NULL, config_json TEXT NOT NULL, windows_json TEXT NOT NULL,
 status VARCHAR(32) NOT NULL, lease_token VARCHAR(64), lease_until VARCHAR(64), result_json TEXT,
 created_at VARCHAR(64) NOT NULL, published_at VARCHAR(64),
 UNIQUE(subscription_id,revision,slot)
);
CREATE TABLE IF NOT EXISTS brief_requests (
 request_id VARCHAR(64) PRIMARY KEY, run_id VARCHAR(64) NOT NULL, lane CHAR(64) NOT NULL,
 attempt INTEGER NOT NULL, status VARCHAR(32) NOT NULL, result_json TEXT,
 started_at VARCHAR(64) NOT NULL, settled_at VARCHAR(64), UNIQUE(run_id,lane,attempt)
);
CREATE TABLE IF NOT EXISTS brief_cursors (
 subscription_id VARCHAR(64) NOT NULL, lane CHAR(64) NOT NULL, through_day VARCHAR(10) NOT NULL,
 run_id VARCHAR(64) NOT NULL, PRIMARY KEY(subscription_id,lane)
);
CREATE TABLE IF NOT EXISTS brief_candidates (
 subscription_id VARCHAR(64) NOT NULL, lane CHAR(64) NOT NULL, identity CHAR(64) NOT NULL,
 candidate_json TEXT NOT NULL, first_found VARCHAR(64) NOT NULL, last_found VARCHAR(64) NOT NULL,
 PRIMARY KEY(subscription_id,lane,identity)
);
CREATE TABLE IF NOT EXISTS brief_items (
 run_id VARCHAR(64) NOT NULL, identity CHAR(64) NOT NULL, item_json TEXT NOT NULL,
 displayed_at VARCHAR(64), PRIMARY KEY(run_id,identity)
);
CREATE TABLE IF NOT EXISTS brief_read_marks (
 identity CHAR(64) PRIMARY KEY, is_read INTEGER NOT NULL, updated_at VARCHAR(64) NOT NULL
);
'''
