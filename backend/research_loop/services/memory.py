"""用户画像和历史召回：保留来源、版本与撤销记录，不把记忆当作论文原文。"""
from __future__ import annotations
import json
from uuid import uuid4
from research_loop.storage.memory import MemoryStore, dump, digest, operation, record_operation, check_owner
from research_loop.storage.indexes import rank
from research_loop.storage.mysql import _now


class MemoryService:
    """单用户记忆服务；当前指令优先，保存授权永不从记忆继承。"""

    def __init__(self, db, notes=None, actor=None):
        """复用数据库和可选笔记版本服务，无模型依赖。"""
        self.db, self.notes, self.store = db, notes, MemoryStore(db)
        self.actor=actor

    def active(self) -> list[dict]:
        """返回有效显式偏好及推断；撤销/被替代版本不参与召回。"""
        return [self._entry(r) for r in self.db.fetchall("SELECT * FROM memory_entries WHERE status='active' ORDER BY slot,version")]

    @staticmethod
    def _entry(row: dict) -> dict:
        """解包来源字段，保留原始身份和版本。"""
        row = dict(row)
        row['source'] = json.loads(row.pop('source_json'))
        return row

    def remember(self, task_id: str, args) -> dict:
        """把当前用户原句绑定到偏好；禁止用模型输出或旧论文伪造用户表达。

        Args:
            task_id: 本次会话任务。
            args: RememberInput，包含原句、槽位和乐观版本。

        Raises:
            ValueError: 来源不符、过期版本或推断覆盖显式偏好。
        """
        task = self.db.get_task(task_id)
        texts = [task['prompt']] + [r['answer'] for r in self.db.fetchall("SELECT answer FROM clarifications WHERE task_id=? AND status='answered'", (task_id,))]
        if not any(args.source_quote in text for text in texts):
            raise ValueError('MEMORY_SOURCE: 引用必须来自当前用户指令或澄清')
        source = {'type':'user_message', 'task_id':task_id, 'quote':args.source_quote, 'task_created_at':task['created_at']}
        return self._put(args.slot,args.kind,args.content,source,args.expected_version,'remember:'+task_id+':'+digest(args.model_dump())[:48])

    def set_profile(self, args) -> dict:
        """个人面板的显式编辑；相同请求可重放，版本过期则拒绝。"""
        return self._put(args.slot,'explicit',args.content,{'type':'profile_edit','quote':args.content},args.expected_version,'profile:'+args.idempotency_key)

    def add_statement(self, args) -> dict:
        """保存一段明确的长期偏好，不要求用户提供内部槽位。

        Args:
            args: 自然语言原文和幂等键，已由接口校验长度及空白。

        Returns:
            持久记忆记录。重复原文复用有效记录；旧请求不会复活撤销项。

        Raises:
            ValueError: 同一幂等键被用于不同的原文。
        """
        key = 'profile-statement:' + args.idempotency_key
        with self.store.transaction() as tx:
            if old := operation(tx, key, args.content):
                return self._entry(tx.one('SELECT * FROM memory_entries WHERE memory_id=?', (old['memory_id'],)))
            # Python逐字比较，避免MySQL大小写不敏感排序规则合并不同原文。
            for row in tx.all("SELECT * FROM memory_entries WHERE status='active' AND kind='explicit'"):
                if row['content'] == args.content:
                    record_operation(tx, key, args.content, {'memory_id': row['memory_id']})
                    return self._entry(row)
            identity = str(uuid4())
            row = {'memory_id': identity, 'slot': 'statement:' + identity,
                   'kind': 'explicit', 'content': args.content,
                   'source_json': dump({'type': 'profile_statement', 'quote': args.content}),
                   'status': 'active', 'version': 1, 'created_at': _now()}
            tx.execute('INSERT INTO memory_entries(memory_id,slot,kind,content,source_json,status,version,created_at) VALUES(?,?,?,?,?,?,?,?)', tuple(row.values()))
            record_operation(tx, key, args.content, {'memory_id': identity})
            return self._entry(row)

    def _put(self, slot, kind, content, source, expected_version, key) -> dict:
        """在串行事务中建立新版本；被撤销的旧调用仅返回现状，不复活记录。"""
        payload = [slot,kind,content,source,expected_version]
        with self.store.transaction() as tx:
            check_owner(tx,self.actor)
            if old := operation(tx,key,payload):
                return self._entry(tx.one('SELECT * FROM memory_entries WHERE memory_id=?', (old['memory_id'],)))
            latest = tx.one('SELECT * FROM memory_entries WHERE slot=? ORDER BY version DESC LIMIT 1', (slot,))
            version = latest['version'] if latest else 0
            if expected_version != version:
                raise ValueError('MEMORY_STALE_VERSION: 当前版本='+str(version))
            if latest and latest['status'] == 'revoked':
                if kind == 'inferred' or source.get('task_created_at','9999') <= json.loads(latest['source_json']).get('revoked_at',''):
                    raise ValueError('MEMORY_REVOKED: 不从旧任务或推断恢复已撤销偏好')
            if latest and latest['kind'] == 'explicit' and kind == 'inferred':
                raise ValueError('MEMORY_PRIORITY: 推断不能覆盖显式偏好')
            tx.execute("UPDATE memory_entries SET status='superseded' WHERE slot=? AND status='active'", (slot,))
            row = {'memory_id':str(uuid4()), 'slot':slot,'kind':kind,'content':content,'source_json':dump(source),'status':'active','version':version+1,'created_at':_now()}
            tx.execute('INSERT INTO memory_entries(memory_id,slot,kind,content,source_json,status,version,created_at) VALUES(?,?,?,?,?,?,?,?)', tuple(row.values()))
            record_operation(tx,key,payload,{'memory_id':row['memory_id']})
            return self._entry(row)

    def forget(self, memory_id: str, expected_version: int) -> dict:
        """撤销指定当前版本并留墓碑；重复撤销幂等，不能撤销后来的修订。"""
        with self.store.transaction() as tx:
            check_owner(tx,self.actor)
            row = tx.one('SELECT * FROM memory_entries WHERE memory_id=?', (memory_id,))
            if not row:
                raise KeyError('memory not found')
            if row['version'] != expected_version or row['status'] == 'superseded':
                raise ValueError('MEMORY_STALE_VERSION')
            source = json.loads(row['source_json'])
            source.setdefault('revoked_at',_now())
            tx.execute("UPDATE memory_entries SET status='revoked',source_json=? WHERE memory_id=?", (dump(source),memory_id))
            return self._entry({**row,'status':'revoked','source_json':dump(source)})

    def search(self, query: str, source_type='all', limit=5, task_id=None) -> list[dict]:
        """按来源过滤后BM25召回；研究历史和个人笔记明确标注，不输出伪造原文。"""
        rows = []
        if source_type in {'all','research_history'}:
            for row in self.db.fetchall('SELECT * FROM research_memories ORDER BY created_at DESC'):
                record = json.loads(row['record_json'])
                rows.append({'source_type':'research_history','source_id':row['task_id'],'search_text':row['search_text'], 'record':record})
        if source_type in {'all','user_note'} and self.notes:
            rows.extend({'source_type':'user_note','source_id':n['note_id'],'search_text':n['title']+'\n'+n['markdown'],'record':n} for n in self.notes.list_current())
        hits = [{k:v for k,v in r.items() if k != 'search_text'} for r in rank(query,rows,limit)]
        if task_id:
            for hit in hits:
                if hit['source_type']=='research_history':
                    self._grant_history_papers(task_id,hit['record'])
        # 召回只返回索引卡；全文及大批候选仍可按source_id分页读回，避免一次命中塞满上下文。
        from research_loop.domain.context import excerpt
        for hit in hits:
            record = hit['record']
            if hit['source_type'] == 'research_history':
                hit['record'] = {k: record.get(k) for k in ('task_id','conversation_id','status','created_at')}
                hit['record'].update(prompt=excerpt(record.get('prompt',''),query,400),
                    answer=excerpt(record.get('answer',''),query,1000),
                    paper_ids=(record.get('paper_ids') or [])[:8],
                    papers=[{k:p.get(k) for k in ('paper_id','title','source','year')}
                            for p in (record.get('papers') or [])[:8]],
                    gaps=(record.get('gaps') or [])[:4])
                hit['record']['gaps'] = [excerpt(str(g),query,250) for g in hit['record']['gaps']]
            else:
                hit['record'] = {k:record.get(k) for k in ('note_id','title','version','content_hash')}
                hit['record']['markdown'] = excerpt(record.get('markdown',''),query,1000)
            hit['excerpted'] = True
            hit['readback'] = {'source_type':hit['source_type'],'source_id':hit['source_id']}
            # limit最多10条；每卡最多2400字节，使同一轮多次检索仍不会返回整库正文。
            while len(dump(hit).encode('utf-8')) > 2400:
                card=hit['record']
                lists=[k for k,v in card.items() if isinstance(v,list) and v]
                texts=[k for k,v in card.items() if isinstance(v,str) and len(v)>80 and k not in {'source_id','task_id','conversation_id'}]
                if lists:
                    card[max(lists,key=lambda k:len(dump(card[k])))].pop()
                elif texts:
                    key=max(texts,key=lambda k:len(card[k]));card[key]=excerpt(card[key],query,max(60,len(card[key])//2))
                else:
                    break
        return hits

    def _grant_history_papers(self, task_id, record):
        """显式读取历史后才授予论文读取权限；限定附件任务不自动扩展材料范围。"""
        task=self.db.get_task(task_id)
        if json.loads(task['attachments_json']):
            return
        for pid in record.get('paper_ids') or []:
            if self.db.get_paper(pid):
                self.db.grant_paper(task_id,pid,task_id)

    def read(self, task_id: str, args) -> dict:
        """分页读回原始记录；档案不可跨任务读取，消息不可跨会话误绑定。

        Returns:
            带来源身份、内容指纹及下一偏移的文本。旧摘要和笔记不授予原文证据权限。
        """
        kind, sid = args.source_type,args.source_id
        if kind == 'research_history':
            row = self.db.fetchone('SELECT record_json FROM research_memories WHERE task_id=?', (sid,))
            content = row['record_json'] if row else None
            if row:
                self._grant_history_papers(task_id,json.loads(content))
        elif kind == 'context_archive':
            row = self.db.fetchone('SELECT original_json FROM context_archives WHERE snapshot_id=? AND task_id=?', (sid,task_id))
            content = row['original_json'] if row else None
        elif kind == 'conversation':
            link = self.db.fetchone('SELECT conversation_id FROM conversation_links WHERE task_id=?', (task_id,))
            if not link or sid != link['conversation_id']:
                raise KeyError('conversation outside task scope')
            content = dump(self.db.fetchall('SELECT message_id,role,content,sequence_no FROM conversation_messages WHERE conversation_id=? ORDER BY sequence_no',(sid,)))
        elif kind == 'message':
            link = self.db.fetchone('SELECT conversation_id FROM conversation_links WHERE task_id=?', (task_id,))
            row = self.db.fetchone('SELECT content FROM conversation_messages WHERE message_id=? AND conversation_id=?', (sid,link['conversation_id'])) if link else None
            content = row['content'] if row else None
        else:
            content = dump(self.notes.get(sid)) if self.notes else None
        if content is None:
            raise KeyError('memory source not found in permitted scope')
        text = content[args.offset:args.offset+args.limit]
        return {'source_type':kind,'source_id':sid,'content':text,'content_hash':digest(content),'offset':args.offset,'total_chars':len(content),'next_offset':args.offset+len(text) if args.offset+len(text)<len(content) else None,'is_original_paper_evidence':False}

    def profile(self) -> dict:
        """从成功精读子任务统计不同论文；不把下载量、重试或用户已读混为一谈。"""
        papers = set()
        for row in self.db.fetchall("SELECT result_json FROM subtasks WHERE status='completed'"):
            result = json.loads(row['result_json'] or '{}').get('result',{})
            if result.get('mode') == 'deep_read' and result.get('deep_reading_status') == 'complete':
                papers.update(result.get('paper_ids') or [])
        records = self.active()
        return {'explicit_preferences':[r for r in records if r['kind']=='explicit'], 'inferred_interests':[r for r in records if r['kind']=='inferred'], 'revoked':[self._entry(r) for r in self.db.fetchall("SELECT * FROM memory_entries WHERE status='revoked'")], 'statistics':{'deep_read_distinct_papers':len(papers),'conversations':self.db.fetchone('SELECT COUNT(*) AS n FROM conversations')['n'],'history_tasks':self.db.fetchone('SELECT COUNT(*) AS n FROM research_memories')['n']}, 'statistics_basis':'成功完成论文精读的不同paper_id；不等于用户本人已读'}
