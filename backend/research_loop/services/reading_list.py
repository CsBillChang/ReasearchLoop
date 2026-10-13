"""聊天中的个人阅读清单：只记录用户动作，不自动执行研究或改变核验状态。"""
from __future__ import annotations

import json
from research_loop.storage.memory import MemoryStore, digest, dump
from research_loop.storage.mysql import _now


class ReadingListService:
    """本地单用户清单服务，复用用户事务锁控制并发和幂等。"""

    def __init__(self, db):
        """持有数据库，不创建后台任务。"""
        self.db = db

    @staticmethod
    def public(row: dict) -> dict:
        """解码服务端元数据快照，不向客户端暴露文件路径。"""
        return {k: v for k, v in row.items() if k != 'metadata_json'} | {'paper': json.loads(row['metadata_json'])}

    def list(self) -> list[dict]:
        """按最近变更列出含移除墓碑的记录，供界面正确绑定版本。"""
        return [self.public(r) for r in self.db.fetchall('SELECT * FROM reading_list ORDER BY updated_at DESC,paper_id')]

    def change(self, args) -> dict:
        """原子添加或更新；来源必须是任务家族已获得的真实论文。

        Args:
            args: paper_id、task_id、action、expected_version及idempotency_key。

        Returns:
            当前清单记录；重放成功请求返回当前状态，不覆盖后来的修改。

        Raises:
            KeyError: 论文/来源任务不可见。
            ValueError: 版本过期或同幂等键不同输入。
        """
        payload = args.model_dump(exclude={'idempotency_key'})
        with MemoryStore(self.db).transaction() as tx:
            old = tx.one('SELECT * FROM reading_list WHERE paper_id=?', (args.paper_id,))
            operation = tx.one('SELECT * FROM reading_list_operations WHERE operation_id=?', (args.idempotency_key,))
            if operation:
                if operation['payload_hash'] != digest(payload):
                    raise ValueError('READING_LIST_IDEMPOTENCY_CONFLICT: 同一请求键的内容不同')
                return self.public(old)
            if args.action == 'add':
                task = tx.one('SELECT * FROM tasks WHERE task_id=?', (args.task_id,))
                if not task:
                    raise KeyError('来源任务不存在')
                row = tx.one('SELECT p.metadata_json FROM papers p JOIN task_papers tp ON tp.paper_id=p.paper_id JOIN tasks t ON t.task_id=tp.task_id WHERE t.root_task_id=? AND p.paper_id=?',
                             (task['root_task_id'] or task['task_id'], args.paper_id))
                if not row:
                    raise KeyError('论文不属于已取得的任务来源')
                if old and old['status'] != 'removed':
                    # 同论文重复加入不降级已读状态，也不覆盖初次来源快照。
                    tx.execute('INSERT INTO reading_list_operations(operation_id,payload_hash,paper_id,created_at) VALUES(?,?,?,?)',
                               (args.idempotency_key, digest(payload), args.paper_id, _now()))
                    return self.public(old)
            elif not old or old['status'] == 'removed':
                raise ValueError('READING_LIST_NOT_ACTIVE: 请先加入待读')
            version = old['version'] if old else 0
            if args.expected_version != version:
                raise ValueError('READING_LIST_STALE: 清单已变化，请刷新后再操作')
            now = _now()
            status = {'add': 'unread', 'read': 'read', 'unread': 'unread', 'remove': 'removed'}[args.action]
            if old:
                tx.execute('UPDATE reading_list SET status=?,version=?,updated_at=? WHERE paper_id=?',
                           (status, version + 1, now, args.paper_id))
            else:
                metadata = json.loads(row['metadata_json'])
                public = {key: metadata.get(key) for key in ('paper_id','title','authors','source','published','paper_url','pdf_url')}
                public['paper_id'] = args.paper_id
                tx.execute('INSERT INTO reading_list(paper_id,task_id,metadata_json,status,version,created_at,updated_at) VALUES(?,?,?,?,?,?,?)',
                           (args.paper_id, args.task_id, dump(public), status, 1, now, now))
            tx.execute('INSERT INTO reading_list_operations(operation_id,payload_hash,paper_id,created_at) VALUES(?,?,?,?)',
                       (args.idempotency_key, digest(payload), args.paper_id, now))
            return self.public(tx.one('SELECT * FROM reading_list WHERE paper_id=?', (args.paper_id,)))
