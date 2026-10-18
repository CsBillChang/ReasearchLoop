"""个性化简报服务：配置、逐源收据、窗口推进和聊天交付共用持久事务。"""
from __future__ import annotations
import json
from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo
from research_loop.domain.subscriptions import SubscriptionInput, now_utc, schedule_slot
from research_loop.storage.mysql import Database
from research_loop.storage.memory import MemoryStore, dump, digest, operation, record_operation
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.services.recommendations import paper_identity, rank_candidates, render_brief


def lane_key(source: str, query: str) -> str:
    """同一来源/基础查询共用覆盖游标，不受标题和数量编辑影响。"""
    return digest([source, query])


def request_lane(window: dict) -> str:
    """收据还绑定精确日期，停机合并窗口不误复用较窄窗口的响应。"""
    return digest(window)


class SubscriptionService:
    """本机单用户简报存储；外部请求不在数据库事务内执行。"""

    def __init__(self, db: Database, clock=now_utc):
        """注入数据库与时钟，测试日历边界而不改系统时间。"""
        self.db, self.clock, self.store = db, clock, MemoryStore(db)

    def configure(self, payload: SubscriptionInput, subscription_id: str | None = None) -> dict:
        """幂等新增/乐观版本编辑；禁用可取消活动批次，其他编辑等待批次结束。"""
        now = self.clock()
        value = payload.model_dump()
        config = {k:v for k,v in value.items() if k not in {'idempotency_key','expected_revision','enabled'}}
        with self.store.transaction() as tx:
            key = 'subscription:' + payload.idempotency_key
            fingerprint = {'subscription_id':subscription_id, **value}
            if old := operation(tx, key, fingerprint):
                return old
            old = tx.one('SELECT * FROM subscriptions WHERE subscription_id=?', (subscription_id,)) if subscription_id else None
            if subscription_id and not old:
                raise KeyError('subscription not found')
            if payload.expected_revision != (old['revision'] if old else 0):
                raise ValueError('SUBSCRIPTION_STALE: 请刷新后再编辑')
            if old and tx.one("SELECT run_id FROM brief_runs WHERE subscription_id=? AND status='running'", (subscription_id,)):
                if payload.enabled:
                    raise ValueError('SUBSCRIPTION_BUSY: 请等本轮结束，或先关闭订阅')
                tx.execute("UPDATE brief_runs SET status='cancelled' WHERE subscription_id=? AND status='running'", (subscription_id,))
            sid = subscription_id or str(uuid4())
            revision = (old['revision'] if old else 0) + 1
            next_due = schedule_slot(config, now, following=True).isoformat()
            if not old:
                cid = str(uuid4())
                tx.execute('INSERT INTO conversations(conversation_id,title,created_at) VALUES(?,?,?)', (cid,payload.title,now.isoformat()))
                tx.execute('INSERT INTO subscriptions(subscription_id,revision,config_json,enabled,next_due,conversation_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)', (sid,revision,dump(config),int(payload.enabled),next_due,cid,now.isoformat(),now.isoformat()))
            else:
                cid = old['conversation_id']
                # 未改变调度点/开关的配置编辑不跳过已经到期的窗口。
                previous = json.loads(old['config_json'])
                if old['enabled'] and payload.enabled and all(config[k] == previous[k] for k in ('timezone','at')):
                    next_due = old['next_due']
                tx.execute('UPDATE subscriptions SET revision=?,config_json=?,enabled=?,next_due=?,updated_at=? WHERE subscription_id=?', (revision,dump(config),int(payload.enabled),next_due,now.isoformat(),sid))
                tx.execute('UPDATE conversations SET title=? WHERE conversation_id=?', (payload.title,cid))
            result = {'subscription_id':sid,'revision':revision,'config':config,'enabled':payload.enabled,'next_due':next_due,'conversation_id':cid}
            record_operation(tx,key,fingerprint,result)
            return result

    def list(self) -> list[dict]:
        """读回配置及真实统计；生成、页面展示、精读和本人已读分别显示。"""
        output = []
        for row in self.db.fetchall('SELECT * FROM subscriptions ORDER BY created_at'):
            sid = row['subscription_id']
            row['config'] = json.loads(row.pop('config_json'))
            row['enabled'] = bool(row['enabled'])
            latest = self.db.fetchone('SELECT run_id,status,published_at FROM brief_runs WHERE subscription_id=? ORDER BY created_at DESC LIMIT 1', (sid,))
            row['latest'] = latest
            row['statistics'] = {
                'found':self.db.fetchone('SELECT COUNT(DISTINCT identity) AS n FROM brief_candidates WHERE subscription_id=?',(sid,))['n'],
                'delivered':self.db.fetchone('SELECT COUNT(DISTINCT i.identity) AS n FROM brief_items i JOIN brief_runs r ON r.run_id=i.run_id WHERE r.subscription_id=?',(sid,))['n'],
                'displayed':self.db.fetchone('SELECT COUNT(DISTINCT i.identity) AS n FROM brief_items i JOIN brief_runs r ON r.run_id=i.run_id WHERE r.subscription_id=? AND i.displayed_at IS NOT NULL',(sid,))['n'],
                'user_read':self.db.fetchone('SELECT COUNT(DISTINCT c.identity) AS n FROM brief_candidates c JOIN brief_read_marks m ON m.identity=c.identity WHERE c.subscription_id=? AND m.is_read=1',(sid,))['n']}
            output.append(row)
        return output

    def claim(self, subscription_id: str, lease_seconds: int = 120) -> dict | None:
        """原子领取到期批次；停机跨多日合并为最近调度点的一份简报。

        Args:
            subscription_id: 已持久化且启用的订阅。
            lease_seconds: 进程失联后的恢复时限；每次外部请求前续租。

        Returns:
            冻结配置、逐来源查询窗口和租约身份；未到期/别人持有则返回None。
        """
        now = self.clock()
        with self.store.transaction() as tx:
            sub = tx.one('SELECT * FROM subscriptions WHERE subscription_id=?',(subscription_id,))
            if not sub or not sub['enabled']:
                return None
            active = tx.one("SELECT * FROM brief_runs WHERE subscription_id=? AND status='running'",(subscription_id,))
            if active and active['lease_until'] and datetime.fromisoformat(active['lease_until']) > now:
                return None
            if not active and datetime.fromisoformat(sub['next_due']) > now:
                return None
            config = json.loads(sub['config_json'])
            slot = schedule_slot(config,now)
            end = (slot.astimezone(ZoneInfo(config['timezone'])).date() - timedelta(days=1)).isoformat()
            windows = []
            for source in config['sources']:
                for query in config['directions']:
                    lane = lane_key(source,query)
                    cursor = tx.one('SELECT through_day FROM brief_cursors WHERE subscription_id=? AND lane=?',(subscription_id,lane))
                    # 首次7天；成功后重叠最后一天。失败不越过原有游标，避免无声遗漏。
                    if cursor:
                        lower = cursor['through_day']
                    else:
                        # 从首次应执行的日期倒推完整日历天；09:00后创建不能多算一天。
                        first_slot = schedule_slot(config, datetime.fromisoformat(sub['created_at']), following=True)
                        origin = first_slot.astimezone(ZoneInfo(config['timezone'])).date()
                        lower = (origin - timedelta(days=config['lookback_days'])).isoformat()
                    windows.append({'source':source,'query':query,'from':min(lower,end),'to':end,'lane':lane})
            token, rid = str(uuid4()), active['run_id'] if active else str(uuid4())
            until = (now + timedelta(seconds=lease_seconds)).isoformat()
            if active:
                # 失联进程可能未收到上游响应；即使扩大了窗口也不能遗留started冒充待发。
                tx.execute("UPDATE brief_requests SET status='unknown',settled_at=? WHERE run_id=? AND status='started'",(now.isoformat(),rid))
                # 只扩展旧批次日期；旧请求及响应保留，精确相同窗口才会复用。
                previous = json.loads(active['windows_json'])
                starts = {w['lane']:w['from'] for w in previous}
                for w in windows:
                    w['from'] = min(w['from'],starts.get(w['lane'],w['from']))
                tx.execute('UPDATE brief_runs SET slot=?,windows_json=?,lease_token=?,lease_until=? WHERE run_id=?',(slot.isoformat(),dump(windows),token,until,rid))
            else:
                tx.execute('INSERT INTO brief_runs(run_id,subscription_id,revision,slot,config_json,windows_json,status,lease_token,lease_until,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)',(rid,subscription_id,sub['revision'],slot.isoformat(),dump(config),dump(windows),'running',token,until,now.isoformat()))
            tx.execute('UPDATE subscriptions SET next_due=? WHERE subscription_id=?',(schedule_slot(config,now,following=True).isoformat(),subscription_id))
            return {**tx.one('SELECT * FROM brief_runs WHERE run_id=?',(rid,)), 'windows':windows,'config':config}

    def _owned(self, tx, run_id: str, token: str) -> dict:
        """围栏业务提交，取消、失租或配置变化不能被旧进程覆盖。"""
        run = tx.one('SELECT * FROM brief_runs WHERE run_id=?',(run_id,))
        sub = tx.one('SELECT * FROM subscriptions WHERE subscription_id=?',(run['subscription_id'],)) if run else None
        if not run or run['status'] != 'running' or run['lease_token'] != token or datetime.fromisoformat(run['lease_until']) <= self.clock() or not sub['enabled'] or sub['revision'] != run['revision']:
            raise PermissionError('BRIEF_LEASE_LOST: 本轮已关闭或由其他进程恢复')
        return run

    def begin_request(self, run: dict, window: dict, max_requests: int) -> dict:
        """发送前持久化尝试；成功结果复用，started恢复记unknown并占用两次上限。

        Returns:
            cached结果、ready请求身份、或耗尽标志。这里只重试只读论文，不处理模型收据。
        """
        rid, token, lane = run['run_id'],run['lease_token'],request_lane(window)
        with self.store.transaction() as tx:
            self._owned(tx,rid,token)
            tx.execute('UPDATE brief_runs SET lease_until=? WHERE run_id=?',((self.clock()+timedelta(seconds=120)).isoformat(),rid))
            rows = tx.all('SELECT * FROM brief_requests WHERE run_id=? AND lane=? ORDER BY attempt',(rid,lane))
            for r in rows:
                if r['status']=='succeeded':
                    return {'cached':json.loads(r['result_json'])}
                if r['status']=='started':
                    tx.execute("UPDATE brief_requests SET status='unknown',settled_at=? WHERE request_id=?",(self.clock().isoformat(),r['request_id']))
            if len(rows)>=2 or tx.one('SELECT COUNT(*) AS n FROM brief_requests WHERE run_id=?',(rid,))['n']>=max_requests:
                return {'exhausted':True}
            request_id = str(uuid4())
            tx.execute('INSERT INTO brief_requests(request_id,run_id,lane,attempt,status,started_at) VALUES(?,?,?,?,?,?)',(request_id,rid,lane,len(rows)+1,'started',self.clock().isoformat()))
            return {'request_id':request_id}

    def settle_request(self, run: dict, request_id: str, result: dict) -> None:
        """先保存已知返回，随后才解析/交付；失权旧进程不得覆盖新所有者状态。"""
        with self.store.transaction() as tx:
            owner = tx.one('SELECT lease_token FROM brief_runs WHERE run_id=?',(run['run_id'],))
            if not owner or owner['lease_token']!=run['lease_token']:
                raise PermissionError('BRIEF_LEASE_LOST')
            status = 'failed' if result['coverage']['status']=='failed' else 'succeeded'
            tx.execute('UPDATE brief_requests SET status=?,result_json=?,settled_at=? WHERE request_id=? AND run_id=? AND status=?',(status,dump(result),self.clock().isoformat(),request_id,run['run_id'],'started'))

    def publish(self, run: dict, results: list[dict]) -> dict:
        """原子发布候选、覆盖游标、聊天消息和曝光收据；排名使用发布时有效画像。"""
        with self.store.transaction() as tx:
            rid, sid = run['run_id'], run['subscription_id']
            existing = tx.one('SELECT result_json FROM brief_runs WHERE run_id=?',(rid,))
            if existing and existing['result_json']:
                return json.loads(existing['result_json'])
            self._owned(tx,rid,run['lease_token'])
            config, now = run['config'],self.clock().isoformat()
            for result in results:
                coverage = result['coverage']
                lane = lane_key(coverage['source'],coverage['query'])
                for paper in result.get('candidates',[]):
                    identity = paper_identity(paper)
                    old = tx.one('SELECT identity FROM brief_candidates WHERE subscription_id=? AND lane=? AND identity=?',(sid,lane,identity))
                    if old:
                        tx.execute('UPDATE brief_candidates SET candidate_json=?,last_found=? WHERE subscription_id=? AND lane=? AND identity=?',(dump(paper),now,sid,lane,identity))
                    else:
                        tx.execute('INSERT INTO brief_candidates(subscription_id,lane,identity,candidate_json,first_found,last_found) VALUES(?,?,?,?,?,?)',(sid,lane,identity,dump(paper),now,now))
                if coverage['status'] in {'queried_success','queried_empty'}:
                    old = tx.one('SELECT through_day FROM brief_cursors WHERE subscription_id=? AND lane=?',(sid,lane))
                    if old:
                        tx.execute('UPDATE brief_cursors SET through_day=?,run_id=? WHERE subscription_id=? AND lane=?',(max(old['through_day'],coverage['to']),rid,sid,lane))
                    else:
                        tx.execute('INSERT INTO brief_cursors(subscription_id,lane,through_day,run_id) VALUES(?,?,?,?)',(sid,lane,coverage['to'],rid))
            current_lanes = {w['lane']:w['query'] for w in run['windows']}
            pool = {}
            for row in tx.all('SELECT * FROM brief_candidates WHERE subscription_id=?',(sid,)):
                if row['lane'] not in current_lanes:
                    continue
                paper = pool.setdefault(row['identity'],json.loads(row['candidate_json']))
                paper.setdefault('_queries',[]).append(current_lanes[row['lane']])
            # 与画像撤销共用MemoryStore锁，发布时不会注入已撤销的偏好。
            profile = MemoryService(self.db).profile()
            deep = set()
            for row in tx.all("SELECT result_json FROM subtasks WHERE status='completed'"):
                value=json.loads(row['result_json'] or '{}').get('result',{})
                if value.get('mode')=='deep_read' and value.get('deep_reading_status')=='complete':
                    deep.update(value.get('paper_ids',[]))
            delivered={r['identity'] for r in tx.all('SELECT DISTINCT identity FROM brief_items')}
            read={r['identity'] for r in tx.all('SELECT identity FROM brief_read_marks WHERE is_read=1')}
            items=rank_candidates(list(pool.values()),config,profile,delivered,read,deep)
            coverage=[r['coverage'] for r in results]
            period={'from':min(w['from'] for w in run['windows']),'to':max(w['to'] for w in run['windows'])}
            partial=any(c['status'] not in {'queried_success','queried_empty'} for c in coverage)
            result={'run_id':rid,'status':'partial' if partial else 'completed','period':period,'items':items,'coverage':coverage,'model_requests':0,'fulltext_downloads':0,'published_at':now,'markdown':render_brief(config,items,coverage,period)}
            sub=tx.one('SELECT conversation_id FROM subscriptions WHERE subscription_id=?',(sid,))
            result['conversation_id']=sub['conversation_id']
            ConversationService.add_message(tx,sub['conversation_id'],'','brief:'+rid,'assistant',result['markdown'])
            for item in items:
                tx.execute('INSERT INTO brief_items(run_id,identity,item_json) VALUES(?,?,?)',(rid,item['identity'],dump(item)))
            tx.execute('UPDATE brief_runs SET status=?,result_json=?,published_at=?,lease_until=NULL WHERE run_id=?',(result['status'],dump(result),now,rid))
            return result

    def brief(self, run_id: str) -> dict:
        """读回当前简报；只读不等于用户已展示/已读。"""
        row=self.db.fetchone('SELECT * FROM brief_runs WHERE run_id=?',(run_id,))
        if not row:
            raise KeyError('brief not found')
        result=json.loads(row['result_json']) if row['result_json'] else {'run_id':run_id,'status':row['status'],'items':[]}
        result['read_identities']=[r['identity'] for r in self.db.fetchall('SELECT identity FROM brief_read_marks WHERE is_read=1')]
        return result

    def mark_displayed(self, run_id: str) -> dict:
        """前端实际展开后显式记录曝光；重复上报不改变首次时间。"""
        self.brief(run_id)
        self.db.execute('UPDATE brief_items SET displayed_at=? WHERE run_id=? AND displayed_at IS NULL',(self.clock().isoformat(),run_id))
        return {'displayed':True}

    def mark_read(self, identity: str, value: bool) -> dict:
        """只有用户点击才设置已读；不改论文精读任务或Reviewer状态。"""
        with self.store.transaction() as tx:
            if not tx.one('SELECT identity FROM brief_items WHERE identity=?',(identity,)):
                raise KeyError('candidate was not delivered')
            if tx.one('SELECT identity FROM brief_read_marks WHERE identity=?',(identity,)):
                tx.execute('UPDATE brief_read_marks SET is_read=?,updated_at=? WHERE identity=?',(int(value),self.clock().isoformat(),identity))
            else:
                tx.execute('INSERT INTO brief_read_marks(identity,is_read,updated_at) VALUES(?,?,?)',(identity,int(value),self.clock().isoformat()))
        return {'identity':identity,'read':value}
