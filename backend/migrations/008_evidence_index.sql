-- AG7.1只新增可重建索引及不可变原文片段，不改变旧表或数据。
CREATE TABLE IF NOT EXISTS evidence_index_builds (
 index_id CHAR(64) PRIMARY KEY, document_id VARCHAR(255) NOT NULL,
 manifest_json LONGTEXT NOT NULL, bm25_status VARCHAR(16) NOT NULL,
 vector_status VARCHAR(16) NOT NULL, last_error LONGTEXT,
 created_at VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_chunks (
 chunk_id CHAR(64) PRIMARY KEY, chunk_json LONGTEXT NOT NULL,
 terms_json LONGTEXT NOT NULL, token_count BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_index_members (
 index_id CHAR(64) NOT NULL, chunk_id CHAR(64) NOT NULL, ordinal_no BIGINT NOT NULL,
 embedding_json LONGTEXT, PRIMARY KEY(index_id,chunk_id), UNIQUE(index_id,ordinal_no)
);
