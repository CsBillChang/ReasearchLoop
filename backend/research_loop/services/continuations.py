"""取消后继续：持久化后续要求，旧任务结算后幂等入队，不依赖浏览器存活。"""
from __future__ import annotations
import json
from uuid import uuid4
from research_loop.storage.memory import MemoryStore,dump,digest
from research_loop.storage.mysql import _now
from research_loop.domain.tasks import TaskCreate




class ContinuationService:
    """同会话串行接续；一个前置任务只接受一份待执行的新要求。"""

    def __init__(self,db):
        """复用数据库事务；不发模型调用。"""
        self.db=db

    def request(self,predecessor_id,payload):
        """记录明确用户新要求；同输入重放复用，不同输入拒绝覆盖。"""
        from research_loop.services.tasks import TaskService
        canonical=payload.model_dump(mode='json');fingerprint=digest(canonical)
        with MemoryStore(self.db).transaction() as tx:
            link=tx.one('SELECT * FROM conversation_links WHERE task_id=? AND conversation_id=?',(predecessor_id,payload.conversation_id))
            if not link:raise KeyError('前置任务不属于当前会话')
            if old:=tx.one('SELECT * FROM task_continuations WHERE predecessor_id=?',(predecessor_id,)):
                if old['input_hash']!=fingerprint:raise ValueError('已有不同的后续要求，不能覆盖')
            else:
                if tx.one('SELECT task_id FROM tasks WHERE idempotency_key=?',(payload.idempotency_key,)):
                    raise ValueError('后续要求必须使用新的幂等键，不能指向既有任务')
                tx.execute('INSERT INTO task_continuations(continuation_id,predecessor_id,conversation_id,input_json,input_hash,status,created_at) VALUES(?,?,?,?,?,?,?)',(str(uuid4()),predecessor_id,payload.conversation_id,dump(canonical),fingerprint,'pending',_now()))
        TaskService(self.db).cancel(predecessor_id,cancel_continuations=False)
        self.release_ready()
        return self.db.fetchone('SELECT continuation_id,status,next_task_id FROM task_continuations WHERE predecessor_id=?',(predecessor_id,))

    def release_ready(self):
        """恢复取消请求并原子接续；刷新、崩溃或并发轮询不重复入队。"""
        from research_loop.services.tasks import TaskService
        from research_loop.services.conversations import ConversationService
        for candidate in self.db.fetchall("SELECT * FROM task_continuations WHERE status='pending'"):
            # 请求落库后进程可能在cancel前退出，Worker仍负责完成已接受的停止要求。
            TaskService(self.db).cancel(candidate['predecessor_id'],cancel_continuations=False)
            with MemoryStore(self.db).transaction() as tx:
                row=tx.one("SELECT c.* FROM task_continuations c JOIN tasks t ON t.task_id=c.predecessor_id WHERE c.continuation_id=? AND c.status='pending' AND t.status IN ('completed','partial','failed','cancelled')",(candidate['continuation_id'],))
                if not row:
                    continue
                payload=TaskCreate.model_validate(json.loads(row['input_json']))
                try:
                    tid=ConversationService(self.db).submit_tx(tx,payload)
                except ValueError:
                    # 同会话仍忙时下一次轮询继续，不重写输入、不扩大保存权限。
                    continue
                tx.execute("UPDATE task_continuations SET status='queued',next_task_id=? WHERE continuation_id=?",(tid,row['continuation_id']))
