"""会话服务：用户消息和任务原子关联，澄清继续原任务，结束结果可幂等补投影。"""
from __future__ import annotations
import json
import hashlib
import re
from uuid import uuid4
from research_loop.storage.memory import MemoryStore, dump, digest, operation, record_operation
from research_loop.storage.mysql import Database, _now

TERMINAL = {'completed', 'partial', 'failed', 'cancelled'}


class ConversationService:
    """管理本地用户的对话；每个会话同一时刻只运行一个根任务。"""

    def __init__(self, db: Database, settings=None):
        """复用数据库，不发模型请求。"""
        from research_loop.config import get_settings
        self.settings = settings or get_settings()
        self.db = db
        self.store = MemoryStore(db)

    def create(self, title: str, key: str) -> dict:
        """按用户幂等键新建会话；已有会话不改名。"""
        with self.store.transaction() as tx:
            op = 'conversation:' + key
            if old := operation(tx, op, title):
                return old
            row = {'conversation_id': str(uuid4()), 'title': title, 'created_at': _now()}
            tx.execute('INSERT INTO conversations(conversation_id,title,created_at) VALUES(?,?,?)', tuple(row.values()))
            record_operation(tx, op, title, row)
            return row

    def list(self) -> list[dict]:
        """按最后消息时间排列，空会话使用创建时间；改名不会改变活跃日期。"""
        return self.db.fetchall('SELECT c.*,COALESCE(m.last_activity_at,c.created_at) AS last_activity_at '
            'FROM conversations c LEFT JOIN (SELECT conversation_id,MAX(created_at) AS last_activity_at '
            'FROM conversation_messages GROUP BY conversation_id) m ON m.conversation_id=c.conversation_id '
            'ORDER BY last_activity_at DESC,c.conversation_id')

    def answer_message(self, task_id: str, message_id: str) -> dict | None:
        """读回同会话、当前用户消息之前的assistant回答，不授予原文或保存权限。"""
        return self.db.fetchone('SELECT m.* FROM conversation_links l '
            'JOIN conversation_messages current_message ON current_message.message_id=l.message_id '
            'JOIN conversation_messages m ON m.conversation_id=l.conversation_id '
            "WHERE l.task_id=? AND m.message_id=? AND m.role='assistant' "
            'AND m.sequence_no<current_message.sequence_no', (task_id, message_id))

    def referenced_answer(self, task_id: str) -> dict | None:
        """解析明确内容身份或成功message读回，供异常交付保留已有回答。

        Args:
            task_id: 当前任务，不跨会话选择历史内容。

        Returns:
            唯一回答内容及原消息/版本身份；没有明确来源或存在多个不同候选时
            返回None。不默认采用最近回复，不继承旧核验或保存授权。
        """
        task = self.db.get_task(task_id)
        link = self.db.fetchone('SELECT * FROM conversation_links WHERE task_id=?', (task_id,))
        if not task or not link:
            return None
        wanted = set(re.findall(r'(?i)(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])', task['prompt']))
        wanted = {value.lower() for value in wanted}
        candidates = {}
        rows = self.db.fetchall('SELECT message_id FROM conversation_messages WHERE conversation_id=? '
                               "AND role='assistant' ORDER BY sequence_no DESC", (link['conversation_id'],))
        for entry in rows:
            row = self.answer_message(task_id, entry['message_id'])
            if row and hashlib.sha256(row['content'].encode()).hexdigest() in wanted:
                candidates.setdefault(hashlib.sha256(row['content'].encode()).hexdigest(), row)
        for action in self.db.list_actions(task_id):
            data = (action.get('observation') or {}).get('data') or {}
            args = action.get('arguments') or {}
            if (action['tool_name'] != 'read_memory' or action.get('status') != 'succeeded'
                    or args.get('source_type') != 'message' or data.get('source_type') != 'message'
                    or data.get('source_id') != args.get('source_id')):
                continue
            row = self.answer_message(task_id, data['source_id'])
            # read_memory的content_hash沿用JSON规范化身份；交付另提供原文字节SHA。
            if row and data.get('content_hash') == digest(row['content']):
                candidates.setdefault(hashlib.sha256(row['content'].encode()).hexdigest(), row)
        if len(candidates) != 1:
            return None
        content_hash, row = next(iter(candidates.items()))
        return {'content': row['content'], 'source_type': 'message', 'source_id': row['message_id'],
                'content_sha256': content_hash, 'conversation_id': link['conversation_id'],
                'source_task_id': row['task_id'], 'is_original_paper_evidence': False,
                'review_status': 'unreviewed', 'save_permission_inherited': False}

    def rename(self, conversation_id: str, args) -> dict:
        """按用户看到的旧标题修改会话名，保留消息和全部研究身份。

        Args:
            conversation_id: 已存在的本地会话身份。
            args: 新标题、预期旧标题和幂等键。

        Returns:
            数据库当前会话；旧成功请求重放不会覆盖后来的改名。

        Raises:
            KeyError: 会话不存在。
            ValueError: 幂等键冲突或其他窗口已修改标题。
        """
        payload = [conversation_id, args.title, args.expected_title]
        with self.store.transaction() as tx:
            row = tx.one('SELECT * FROM conversations WHERE conversation_id=?', (conversation_id,))
            if not row:
                raise KeyError('conversation not found')
            key = 'conversation-title:' + args.idempotency_key
            if operation(tx, key, payload):
                return row
            if row['title'] != args.expected_title:
                raise ValueError('CONVERSATION_STALE_TITLE: 对话名称已更新，请刷新后再修改')
            tx.execute('UPDATE conversations SET title=? WHERE conversation_id=?', (args.title, conversation_id))
            record_operation(tx, key, payload, {'conversation_id': conversation_id})
            return {**row, 'title': args.title}

    @staticmethod
    def add_message(tx, cid: str, task_id: str, key: str, role: str, content: str) -> str:
        """在已锁事务中追加消息；重放不产生第二条相同记录。"""
        if old := tx.one('SELECT message_id FROM conversation_messages WHERE message_key=?', (key,)):
            return old['message_id']
        seq = tx.one('SELECT MAX(sequence_no) AS n FROM conversation_messages WHERE conversation_id=?', (cid,))['n'] or 0
        mid = str(uuid4())
        tx.execute('INSERT INTO conversation_messages(message_id,conversation_id,sequence_no,message_key,task_id,role,content,created_at) VALUES(?,?,?,?,?,?,?,?)', (mid,cid,seq+1,key,task_id,role,content,_now()))
        return mid

    def submit(self, payload) -> str:
        """原子提交消息、queued任务、附件权限和共享额度。

        Args:
            payload: 带conversation_id的TaskCreate；save_policy仅来自本次请求。

        Returns:
            新任务或同输入幂等重放的task_id。

        Raises:
            ValueError: 会话忙、幂等冲突或已有澄清等待回答。
            KeyError: 会话不存在。
        """
        with self.store.transaction() as tx:
            return self.submit_tx(tx, payload)

    def submit_tx(self, tx, payload) -> str:
        """在调用方持有的单用户事务内入队；供持久接续原子提交使用。"""
        cid = payload.conversation_id
        canonical = payload.model_dump(mode='json')
        fingerprint = digest(canonical)
        if not tx.one('SELECT * FROM conversations WHERE conversation_id=?', (cid,)):
            raise KeyError('conversation not found')
        if payload.idempotency_key:
            old = tx.one('SELECT * FROM tasks WHERE idempotency_key=?', (payload.idempotency_key,))
            if old:
                if old['input_hash'] != fingerprint:
                    raise ValueError('IDEMPOTENCY_CONFLICT')
                return old['task_id']
        pending_next=tx.one("SELECT input_json FROM task_continuations WHERE conversation_id=? AND status='pending'",(cid,))
        if pending_next and json.loads(pending_next['input_json']).get('idempotency_key')!=payload.idempotency_key:
            raise ValueError('CONVERSATION_BUSY: 已有等待旧任务结束的后续要求')
        active = tx.one("SELECT t.task_id,t.status FROM tasks t JOIN conversation_links l ON l.task_id=t.task_id WHERE l.conversation_id=? AND t.status IN ('queued','running','needs_input')", (cid,))
        if active:
            raise ValueError('CONVERSATION_BUSY: ' + dump(active) + '; needs_input请调用原任务resume')
        # 在同一锁内补齐旧终态，再给新消息编号，防止恢复窗口颠倒上下文。
        for prior in tx.all('SELECT task_id FROM conversation_links WHERE conversation_id=?',(cid,)):
            self._reconcile_tx(tx,prior['task_id'])
        # 同一会话的附件像消息上下文一样可继续引用。只接续最近一组材料；
        # 新上传的明确材料优先，不跨会话继承，也不继承旧保存权限或旧页证据。
        # 幂等指纹仍绑定原始用户输入，恢复时复用已冻结的附件快照。
        attachments=canonical['attachments']
        inherited_from=None
        if not attachments:
            previous=tx.all('SELECT t.task_id,t.attachments_json FROM tasks t JOIN conversation_links l ON l.task_id=t.task_id JOIN conversation_messages m ON m.message_id=l.message_id WHERE l.conversation_id=? ORDER BY m.sequence_no DESC',(cid,))
            for item in previous:
                if candidate:=json.loads(item['attachments_json']):
                    attachments=candidate
                    inherited_from=item['task_id']
                    break
        tid, now = str(uuid4()), _now()
        tx.execute('INSERT INTO tasks(task_id,idempotency_key,input_hash,prompt,save_policy,attachments_json,status,root_task_id,role,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)', (tid,payload.idempotency_key,fingerprint,payload.prompt,payload.save_policy.value,dump(attachments),'queued',tid,'supervisor',now,now))
        from research_loop.storage.token_budget import create_policy
        create_policy(tx, tid, self.settings)
        mid = self.add_message(tx,cid,tid,'input:'+tid,'user',payload.prompt)
        tx.execute('INSERT INTO conversation_links(task_id,conversation_id,message_id) VALUES(?,?,?)', (tid,cid,mid))
        from research_loop.services.interests import InterestService
        InterestService(self.db).enable_tx(tx, tid)
        for attachment in attachments:
            tx.execute('INSERT INTO task_inputs(task_id,input_ref,granted_by,created_at) VALUES(?,?,?,?)', (tid,attachment['input_ref'],tid,now))
        if inherited_from:
            tx.execute('INSERT INTO task_events(event_id,task_id,event_type,payload_json,created_at) VALUES(?,?,?,?,?)',
                (str(uuid4()),tid,'conversation_attachments_bound',dump({'source_task_id':inherited_from,
                'input_refs':[a['input_ref'] for a in attachments],'save_permission_inherited':False,'page_evidence_inherited':False}),now))
        self.db.create_budget(tx, tid, self.settings)
        tx.execute('INSERT INTO task_events(event_id,task_id,event_type,payload_json,created_at) VALUES(?,?,?,?,?)', (str(uuid4()),tid,'task_created',dump({'conversation_id':cid,'status':'queued'}),now))
        return tid

    def reconcile(self, task_id: str) -> None:
        """从权威任务/澄清补齐会话与研究记录；恢复不调用模型，不重复计数。"""
        with self.store.transaction() as tx:
            self._reconcile_tx(tx,task_id)

    def _reconcile_tx(self, tx, task_id: str) -> None:
        """在调用方事务内补投影，使消息顺序与新提交原子一致。"""
        link = tx.one('SELECT * FROM conversation_links WHERE task_id=?', (task_id,))
        task = tx.one('SELECT * FROM tasks WHERE task_id=?', (task_id,))
        if not link or not task:
            return
        cid = link['conversation_id']
        for question in tx.all('SELECT * FROM clarifications WHERE task_id=? ORDER BY created_at', (task_id,)):
            self.add_message(tx,cid,task_id,'question:'+question['clarification_id'],'assistant',question['question'])
            if question['answer'] is not None:
                self.add_message(tx,cid,task_id,'clarified:'+question['clarification_id'],'user',question['answer'])
        if task['status'] not in TERMINAL:
            return
        result = json.loads(task['result_json'] or '{}')
        answer = result.get('answer_markdown') or result.get('reason') or task['error'] or task['status']
        self.add_message(tx,cid,task_id,'result:'+task_id,'assistant',answer)
        if tx.one('SELECT task_id FROM research_memories WHERE task_id=?', (task_id,)):
            return
        # 来源来自数据库，不从模型摘要猜论文ID或已完成精读数量。
        paper_ids = [r['paper_id'] for r in tx.all('SELECT DISTINCT p.paper_id FROM task_papers p JOIN tasks t ON t.task_id=p.task_id WHERE t.root_task_id=?', (task_id,))]
        papers = [json.loads(r['metadata_json']) for pid in paper_ids if (r := tx.one('SELECT metadata_json FROM papers WHERE paper_id=?', (pid,)))]
        evidence_ids = [r['evidence_id'] for r in tx.all('SELECT DISTINCT e.evidence_id FROM task_evidence e JOIN tasks t ON t.task_id=e.task_id WHERE t.root_task_id=?', (task_id,))]
        from research_loop.services.research_progress import ResearchProgressService
        stage = ResearchProgressService(self.db)
        progress = [stage.retained(row['task_id']) for row in tx.all('SELECT task_id FROM tasks WHERE root_task_id=?', (task_id,))]
        progress = [item for item in progress if item['revision']]
        record = {'research_progress': progress, 'source_type':'research_history','task_id':task_id,'conversation_id':cid,'prompt':task['prompt'],'status':task['status'],'answer':answer,'papers':papers,'paper_ids':paper_ids,'evidence_ids':evidence_ids,'gaps':result.get('gaps',[]),'note_artifacts':result.get('note_artifacts',[]),'created_at':task['created_at']}
        tx.execute('INSERT INTO research_memories(task_id,conversation_id,record_json,search_text,created_at) VALUES(?,?,?,?,?)', (task_id,cid,dump(record), task['prompt']+'\n'+answer+'\n'+dump(papers), _now()))

    def view(self, cid: str, offset: int = 0, limit: int = 100) -> dict:
        """按消息顺序读回；先补投影，服务重启后也可修复终态后的中断窗口。"""
        row = self.db.fetchone('SELECT * FROM conversations WHERE conversation_id=?', (cid,))
        if not row:
            raise KeyError('conversation not found')
        links = self.db.fetchall('SELECT task_id FROM conversation_links WHERE conversation_id=?', (cid,))
        for link in links:
            self.reconcile(link['task_id'])
        messages = self.db.fetchall('SELECT * FROM conversation_messages WHERE conversation_id=? ORDER BY sequence_no LIMIT ? OFFSET ?', (cid,limit,offset))
        total = self.db.fetchone('SELECT COUNT(*) AS n FROM conversation_messages WHERE conversation_id=?', (cid,))['n']
        active = self.db.fetchall("SELECT t.task_id,t.status FROM tasks t JOIN conversation_links l ON l.task_id=t.task_id WHERE l.conversation_id=? AND t.status IN ('queued','running','needs_input')", (cid,))
        pending = self.db.fetchall("SELECT predecessor_id,continuation_id FROM task_continuations WHERE conversation_id=? AND status='pending'", (cid,))
        return {**row,'messages':messages,'total':total,'offset':offset,'active_tasks':active,'pending_continuations':pending}
