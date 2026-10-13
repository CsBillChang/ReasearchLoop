-- Preserve reviewed draft identity separately from annotated export bytes.
CREATE TABLE note_exports (
 draft_id VARCHAR(64) NOT NULL, content_hash CHAR(64) NOT NULL, file_sha256 CHAR(64) NOT NULL,
 review_status VARCHAR(32), PRIMARY KEY(draft_id, content_hash)
);
