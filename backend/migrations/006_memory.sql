-- AG4会话、记忆和笔记版本；不改写旧任务。

CREATE TABLE IF NOT EXISTS memory_owner (owner_id VARCHAR(64) PRIMARY KEY, revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS conversations (conversation_id VARCHAR(64) PRIMARY KEY, title LONGTEXT NOT NULL, created_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS conversation_links (task_id VARCHAR(64) PRIMARY KEY, conversation_id VARCHAR(64) NOT NULL, message_id VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS conversation_messages (message_id VARCHAR(64) PRIMARY KEY, conversation_id VARCHAR(64) NOT NULL, sequence_no INTEGER NOT NULL, message_key VARCHAR(200) NOT NULL UNIQUE, task_id VARCHAR(64) NOT NULL, role VARCHAR(32) NOT NULL, content LONGTEXT NOT NULL, created_at VARCHAR(64) NOT NULL, UNIQUE(conversation_id,sequence_no));
CREATE TABLE IF NOT EXISTS memory_entries (memory_id VARCHAR(64) PRIMARY KEY, slot VARCHAR(200) NOT NULL, kind VARCHAR(32) NOT NULL, content LONGTEXT NOT NULL, source_json LONGTEXT NOT NULL, status VARCHAR(32) NOT NULL, version INTEGER NOT NULL, created_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS memory_operations (request_key VARCHAR(200) PRIMARY KEY, payload_hash CHAR(64) NOT NULL, result_json LONGTEXT NOT NULL);
CREATE TABLE IF NOT EXISTS research_memories (task_id VARCHAR(64) PRIMARY KEY, conversation_id VARCHAR(64) NOT NULL, record_json LONGTEXT NOT NULL, search_text LONGTEXT NOT NULL, created_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS context_archives (snapshot_id CHAR(64) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, sequence_no INTEGER NOT NULL, original_json LONGTEXT NOT NULL, projected_json LONGTEXT NOT NULL, metrics_json LONGTEXT NOT NULL, created_at VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS note_heads (note_id VARCHAR(64) PRIMARY KEY, source_draft_id VARCHAR(64) NOT NULL, current_version_id VARCHAR(64) NOT NULL);
CREATE TABLE IF NOT EXISTS note_versions (version_id VARCHAR(64) PRIMARY KEY, note_id VARCHAR(64) NOT NULL, base_hash CHAR(64) NOT NULL, content_hash CHAR(64) NOT NULL, title LONGTEXT NOT NULL, markdown LONGTEXT NOT NULL, evidence_json LONGTEXT NOT NULL, path LONGTEXT, created_at VARCHAR(64) NOT NULL);
