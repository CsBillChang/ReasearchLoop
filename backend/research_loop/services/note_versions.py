"""项目笔记的只读查询、差异预览和不可变版本保存；保留原文件与审校历史。"""
from __future__ import annotations
import difflib
import hashlib
import json
from pathlib import Path
from research_loop.storage.memory import MemoryStore, dump, digest, operation, record_operation, check_owner
from research_loop.storage.mysql import _now
from research_loop.storage.files import FileConflictError


class NoteVersionService:
    """使用数据库当前指针管理笔记版本，文件写入后可按同键恢复。"""

    def __init__(self, db, files, actor=None):
        """注入数据库与受限项目文件目录，不扫描用户外部目录。"""
        self.db, self.files, self.store = db, files, MemoryStore(db)
        self.actor=actor

    @staticmethod
    def _original(tx, note_id):
        """只读取已有保存收据的草稿，不把未保存draft伪装成笔记。"""
        row = tx.one('SELECT d.*,s.path FROM drafts d JOIN save_receipts s ON s.draft_id=d.draft_id AND s.content_hash=d.content_hash WHERE d.draft_id=?', (note_id,))
        if not row:
            raise KeyError('saved note not found')
        return {'note_id':note_id,'version_id':note_id,'base_hash':row['content_hash'],'content_hash':row['content_hash'],'title':row['title'],'markdown':row['markdown'],'evidence_json':row['evidence_ids_json'],'path':row['path'],'created_at':row['created_at']}

    @classmethod
    def _current(cls, tx, note_id):
        """在同一事务读当前指针和版本，防止预览与提交使用不同基线。"""
        head = tx.one('SELECT * FROM note_heads WHERE note_id=?', (note_id,))
        return tx.one('SELECT * FROM note_versions WHERE version_id=?', (head['current_version_id'],)) if head else cls._original(tx,note_id)

    @staticmethod
    def _check_file(row):
        """原文件丢失或被用户编辑时报告冲突，不覆盖或静默忽略。"""
        path = Path(row['path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['content_hash']:
            raise ValueError('NOTE_CONFLICT: 文件已改变或不存在')

    def get(self, note_id: str) -> dict:
        """返回当前内容、来源类型和版本；文件完整性与审校状态分别表达。"""
        with self.store.transaction() as tx:
            row = self._current(tx,note_id)
        try:
            self._check_file(row)
            integrity = 'verified'
        except ValueError:
            integrity = 'conflict'
        return {**row,'source_type':'user_note','integrity':integrity,'is_original_paper_evidence':False,'review_status':'not_inherited'}

    def list_current(self) -> list[dict]:
        """返回已保存笔记的当前版本，搜索不会混入旧版本。"""
        return [self.get(r['draft_id']) for r in self.db.fetchall('SELECT DISTINCT draft_id FROM save_receipts ORDER BY draft_id')]

    def versions(self, note_id: str) -> list[dict]:
        """按创建顺序列出历史版本；尚未更新时显示原始保存稿。"""
        with self.store.transaction() as tx:
            rows = tx.all('SELECT * FROM note_versions WHERE note_id=? ORDER BY created_at,version_id', (note_id,))
            return rows or [self._original(tx,note_id)]

    def update(self, note_id: str, args, save_allowed: bool) -> dict:
        """先预览差异，确认后在新路径保存；不覆盖已有笔记。

        Args:
            note_id: 原保存草稿ID，不能由路径代替。
            args: NoteUpdate，包含基线hash、全文新稿、幂等键和确认位。
            save_allowed: 当前请求授权；Agent来自本任务save_policy。

        Returns:
            差异预览或新版本收据。新版本不继承旧Reviewer通过状态。

        Raises:
            PermissionError: 禁止保存的任务尝试提交。
            ValueError: 基线过期、文件冲突或幂等键复用参数不同。
        """
        if args.confirm and not save_allowed:
            raise PermissionError('save denied by current task')
        payload = {'note_id':note_id,**args.model_dump()}
        key = 'note:'+note_id+':'+args.idempotency_key
        with self.store.transaction() as tx:
            check_owner(tx,self.actor)
            if args.confirm and (old := operation(tx,key,payload)):
                self._check_file(old)
                return old
            current = self._current(tx,note_id)
            if current['content_hash'] != args.base_hash:
                raise ValueError('NOTE_STALE_VERSION')
            self._check_file(current)
            delta = '\n'.join(difflib.unified_diff(current['markdown'].splitlines(),args.markdown.splitlines(),fromfile='current',tofile='candidate',lineterm=''))
            content_hash = hashlib.sha256(args.markdown.encode()).hexdigest()
            if not args.confirm:
                return {'status':'preview','note_id':note_id,'base_hash':args.base_hash,'content_hash':content_hash,'diff':delta,'saved':False}
            if not tx.one('SELECT note_id FROM note_heads WHERE note_id=?',(note_id,)):
                original = self._original(tx,note_id)
                tx.execute('INSERT INTO note_versions(version_id,note_id,base_hash,content_hash,title,markdown,evidence_json,path,created_at) VALUES(?,?,?,?,?,?,?,?,?)', tuple(original[k] for k in ('version_id','note_id','base_hash','content_hash','title','markdown','evidence_json','path','created_at')))
                tx.execute('INSERT INTO note_heads(note_id,source_draft_id,current_version_id) VALUES(?,?,?)',(note_id,note_id,note_id))
            version_id = digest([note_id,args.idempotency_key,content_hash])
            # 全局记忆锁覆盖本地原子写入；中断留下的同hash文件可复用，异hash拒绝覆盖。
            try:
                path, actual = self.files.save_markdown('versions-'+note_id,version_id,args.markdown,content_hash)
            except FileConflictError as exc:
                raise ValueError('NOTE_CONFLICT: 候选路径已有不同内容') from exc
            row = {'version_id':version_id,'note_id':note_id,'base_hash':args.base_hash,'content_hash':actual,'title':current['title'],'markdown':args.markdown,'evidence_json':current['evidence_json'],'path':str(path),'created_at':_now()}
            tx.execute('INSERT INTO note_versions(version_id,note_id,base_hash,content_hash,title,markdown,evidence_json,path,created_at) VALUES(?,?,?,?,?,?,?,?,?)',tuple(row.values()))
            tx.execute('UPDATE note_heads SET current_version_id=? WHERE note_id=?',(version_id,note_id))
            result = {**row,'status':'saved','saved':True,'diff':delta,'review_status':'not_inherited','source_type':'user_note'}
            record_operation(tx,key,payload,result)
            return result
