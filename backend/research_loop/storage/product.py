"""产品交互持久层：保存取消后继续请求；模型状态仍由任务和收据负责。"""
SCHEMA='''
CREATE TABLE IF NOT EXISTS task_continuations (
 continuation_id VARCHAR(64) PRIMARY KEY,
 predecessor_id VARCHAR(64) UNIQUE NOT NULL,
 conversation_id VARCHAR(64) NOT NULL,
 input_json TEXT NOT NULL,
 input_hash CHAR(64) NOT NULL,
 status VARCHAR(32) NOT NULL,
 next_task_id VARCHAR(64),
 created_at VARCHAR(64) NOT NULL
);
'''
