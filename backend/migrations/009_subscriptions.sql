-- AG7.3只追加订阅、逐源窗口、简报请求与用户曝光标记；不更改既有论文和笔记。

CREATE TABLE IF NOT EXISTS subscriptions (
 subscription_id VARCHAR(64) PRIMARY KEY, revision BIGINT NOT NULL, config_json LONGTEXT NOT NULL,
 enabled BIGINT NOT NULL, next_due VARCHAR(64) NOT NULL, conversation_id VARCHAR(64) NOT NULL,
 created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS brief_runs (
 run_id VARCHAR(64) PRIMARY KEY, subscription_id VARCHAR(64) NOT NULL, revision BIGINT NOT NULL,
 slot VARCHAR(64) NOT NULL, config_json LONGTEXT NOT NULL, windows_json LONGTEXT NOT NULL,
 status VARCHAR(32) NOT NULL, lease_token VARCHAR(64), lease_until VARCHAR(64), result_json LONGTEXT,
 created_at VARCHAR(64) NOT NULL, published_at VARCHAR(64),
 UNIQUE(subscription_id,revision,slot)
);
CREATE TABLE IF NOT EXISTS brief_requests (
 request_id VARCHAR(64) PRIMARY KEY, run_id VARCHAR(64) NOT NULL, lane CHAR(64) NOT NULL,
 attempt BIGINT NOT NULL, status VARCHAR(32) NOT NULL, result_json LONGTEXT,
 started_at VARCHAR(64) NOT NULL, settled_at VARCHAR(64), UNIQUE(run_id,lane,attempt)
);
CREATE TABLE IF NOT EXISTS brief_cursors (
 subscription_id VARCHAR(64) NOT NULL, lane CHAR(64) NOT NULL, through_day VARCHAR(10) NOT NULL,
 run_id VARCHAR(64) NOT NULL, PRIMARY KEY(subscription_id,lane)
);
CREATE TABLE IF NOT EXISTS brief_candidates (
 subscription_id VARCHAR(64) NOT NULL, lane CHAR(64) NOT NULL, identity CHAR(64) NOT NULL,
 candidate_json LONGTEXT NOT NULL, first_found VARCHAR(64) NOT NULL, last_found VARCHAR(64) NOT NULL,
 PRIMARY KEY(subscription_id,lane,identity)
);
CREATE TABLE IF NOT EXISTS brief_items (
 run_id VARCHAR(64) NOT NULL, identity CHAR(64) NOT NULL, item_json LONGTEXT NOT NULL,
 displayed_at VARCHAR(64), PRIMARY KEY(run_id,identity)
);
CREATE TABLE IF NOT EXISTS brief_read_marks (
 identity CHAR(64) PRIMARY KEY, is_read BIGINT NOT NULL, updated_at VARCHAR(64) NOT NULL
);
