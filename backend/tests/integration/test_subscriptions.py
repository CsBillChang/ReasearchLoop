"""订阅、并发调度、逐源窗口和聊天交付合同；默认SQLite，显式隔离MySQL复验。"""
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch
from urllib.parse import urlparse
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.domain.subscriptions import SubscriptionInput, schedule_slot
from research_loop.domain.evidence import SearchResult, SourceResult, PaperCandidate
from research_loop.storage.mysql import Database
from research_loop.storage.memory import dump
from research_loop.services.subscriptions import SubscriptionService, lane_key
from research_loop.services.recommendations import paper_identity
from research_loop.workers.scheduler import BriefScheduler


class SourceStub:
    """明确标记的DPR替身；禁止触及PDF或LLM，仅返回候选元数据。"""
    def __init__(self, fail=(), truncated=False):
        """配置失败来源和饱和标志，记录实际查询参数。"""
        self.calls=[]; self.fail=fail; self.truncated=truncated; self.version='1'

    def search(self,query,start,end,sources,limit):
        """返回稳定ID，标题相同也不应跨ID去重。"""
        source=sources[0];self.calls.append((query,start,end,source))
        failed=source in self.fail
        return SearchResult(candidates=[] if failed else [PaperCandidate(paper_id=source+'-example',title='Language object detection',abstract='Uses language supervision.',source=source,published=end,date_precision='day',paper_url='https://example.org/paper',version=self.version)],source_results=[SourceResult(source=source,status='failed' if failed else 'queried_success',message='controlled source failure' if failed else '')],truncated=self.truncated)

    def close(self):
        """替身无网络连接。"""


class SubscriptionTests(unittest.TestCase):
    """在真实API和数据库合同上检查可恢复调度，不使用真实供应商。"""
    def setUp(self):
        """每例使用独立订阅和固定时钟；MySQL库必须明确以测试前缀命名。"""
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        url=os.getenv('AG73_TEST_DATABASE_URL') or 'sqlite:///'+str(root/'db')
        if url.startswith('mysql') and not urlparse(url).path.startswith('/researchloop_test_ag73_'):
            raise ValueError('isolated AG73 database required')
        self.config=Settings(project_root=root,data_dir=root/'data',output_dir=root/'notes',database_url=url)
        self.db=Database(url);self.db.ensure_schema()
        self.time=datetime(2026,10,11,0,tzinfo=timezone.utc)
        self.service=SubscriptionService(self.db,lambda:self.time)
        self.payload=SubscriptionInput(directions=['object detection'],sources=['arxiv','cvpr'],enabled=True,idempotency_key=uuid4().hex)
        self.sub=self.service.configure(self.payload)
        self.provider=SourceStub()
        self.worker=BriefScheduler(self.config,provider_factory=lambda:self.provider,clock=lambda:self.time)
        # 隔离MySQL共享测试库里为每个样本使用独特ID，避免上一例的曝光影响排序。
        original=self.provider.search;prefix=uuid4().hex
        def unique(*a,**kw):
            result=original(*a,**kw)
            for p in result.candidates:p.paper_id=prefix+'-'+p.paper_id
            return result
        self.provider.search=unique

    def due(self,days=0):
        """推进到上海09:01，模拟时钟不修改生产时间。"""
        self.time=datetime(2026,10,11,1,1,tzinfo=timezone.utc)+timedelta(days=days)

    def run_one(self):
        """只执行本例的订阅，不扫描其他测试记录。"""
        return self.worker.run_once(self.sub['subscription_id'])[0]

    def test_schedule_and_idempotent_config(self):
        """上海09:00前后创建均首次回溯7个整天；未到期/重复提交不触发请求。"""
        self.assertEqual(self.sub['next_due'],'2026-10-11T01:00:00+00:00')
        self.assertEqual(self.worker.run_once(self.sub['subscription_id']),[])
        self.assertEqual(self.service.configure(self.payload),self.sub)
        with self.assertRaisesRegex(ValueError,'STALE'):
            self.service.configure(self.payload.model_copy(update={'idempotency_key':uuid4().hex}),self.sub['subscription_id'])
        self.assertEqual(self.provider.calls,[])
        self.time=datetime(2026,10,11,2,tzinfo=timezone.utc)
        late=self.service.configure(self.payload.model_copy(update={'idempotency_key':uuid4().hex}))
        self.assertEqual(late['next_due'],'2026-10-12T01:00:00+00:00')
        self.due(1)
        claimed=self.service.claim(late['subscription_id'])
        self.assertEqual(claimed['windows'][0]['from'],'2026-10-05')
        self.assertEqual(claimed['windows'][0]['to'],'2026-10-11')

    def test_catchup_single_brief_and_repeated_tick(self):
        """停机四天只交付一份，重复tick不重复请求/消息，未把生成当成已展示。"""
        self.due(4);result=self.run_one()
        self.assertEqual(result['period'],{'from':'2026-10-04','to':'2026-10-14'})
        self.assertEqual(len(result['items']),2)
        self.assertEqual(len(self.provider.calls),2)
        self.assertEqual(self.worker.run_once(self.sub['subscription_id']),[])
        rows=self.db.fetchall('SELECT * FROM conversation_messages WHERE message_key=?',('brief:'+result['run_id'],))
        self.assertEqual(len(rows),1)
        self.assertTrue(all(r['displayed_at'] is None for r in self.db.fetchall('SELECT * FROM brief_items WHERE run_id=?',(result['run_id'],))))
        self.assertEqual(self.db.fetchone('SELECT COUNT(*) n FROM model_calls')['n'],0)

    def test_failure_cursor_overlap_and_backlog(self):
        """失败来源不前进；成功来源次日重叠最后一天；不足五篇不补造候选。"""
        self.provider.fail=['cvpr'];self.due();first=self.run_one()
        self.assertEqual(first['status'],'partial');self.assertEqual(len(self.provider.calls),3)
        self.assertEqual(len(first['items']),1)
        cursors=self.db.fetchall('SELECT * FROM brief_cursors WHERE subscription_id=?',(self.sub['subscription_id'],))
        self.assertEqual(len(cursors),1)
        self.provider.fail=[];self.due(1);second=self.run_one()
        self.assertEqual(self.provider.calls[-2][1],'2026-10-10')
        self.assertEqual(self.provider.calls[-1][1],'2026-10-04')
        self.assertEqual(len(second['items']),1)
        self.assertEqual(second['items'][0]['paper']['source'],'cvpr')

    def test_truncated_does_not_advance(self):
        """上游top-k饱和仍交付所得候选，但不宣称完整覆盖。"""
        self.provider.truncated=True;self.due();result=self.run_one()
        self.assertEqual(result['status'],'partial')
        self.assertEqual(self.db.fetchall('SELECT * FROM brief_cursors WHERE subscription_id=?',(self.sub['subscription_id'],)),[])
        self.assertIn('达到候选上限',result['markdown'])

    def test_concurrent_claim_and_stale_owner(self):
        """同窗口只被一个进程领取；旧token不能覆盖恢复进程。"""
        self.due()
        with ThreadPoolExecutor(2) as pool:
            claims=list(pool.map(lambda _:self.service.claim(self.sub['subscription_id']),range(2)))
        active=[x for x in claims if x];self.assertEqual(len(active),1)
        old=active[0];self.time+=timedelta(minutes=3)
        recovered=self.service.claim(self.sub['subscription_id']);self.assertEqual(old['run_id'],recovered['run_id'])
        with self.assertRaises(PermissionError):self.service.begin_request(old,old['windows'][0],4)

    def test_receipt_crash_reuse_and_atomic_delivery(self):
        """成功响应持久化后进程中断，恢复不再次查询该窗口，交付仍只有一条。"""
        self.due()
        def crash(*args):raise RuntimeError('controlled after receipt')
        self.worker.after_receipt=crash
        with self.assertRaisesRegex(RuntimeError,'controlled'):self.run_one()
        self.assertEqual(len(self.provider.calls),1)
        self.time+=timedelta(minutes=3);self.worker.after_receipt=None
        result=self.run_one();self.assertEqual(len(self.provider.calls),2)
        self.assertEqual(result['status'],'completed')
        count=self.db.fetchone('SELECT COUNT(*) n FROM brief_requests WHERE run_id=?',(result['run_id'],))['n']
        self.assertEqual(count,2)

    def test_started_unknown_counted_and_bounded(self):
        """中断的只读请求保留unknown且占用额度；最多再尝试一次而非无限补发。"""
        self.due();run=self.service.claim(self.sub['subscription_id'])
        self.service.begin_request(run,run['windows'][0],4)
        self.time+=timedelta(minutes=3)
        result=self.run_one()
        rows=self.db.fetchall('SELECT * FROM brief_requests WHERE run_id=?',(result['run_id'],))
        self.assertEqual(sum(r['status']=='unknown' for r in rows),1)
        self.assertEqual(len(rows),3)

    def test_resume_active_coalesces_multi_day(self):
        """停机时已存在未交付批次也只恢复一份，窄旧窗口不能冒充新窗口。"""
        self.due();run=self.service.claim(self.sub['subscription_id'])
        self.service.begin_request(run,run['windows'][0],4)
        self.due(4);result=self.run_one()
        self.assertEqual(result['run_id'],run['run_id'])
        self.assertEqual(result['period']['to'],'2026-10-14')
        self.assertFalse(self.db.fetchall("SELECT request_id FROM brief_requests WHERE run_id=? AND status='started'",(run['run_id'],)))
        self.assertEqual(self.worker.run_once(self.sub['subscription_id']),[])

    def test_disable_during_request_prevents_publication(self):
        """关闭订阅可以中止当前批次；保留已返回收据，但不交付旧配置简报。"""
        self.due()
        def disable(*args):
            self.service.configure(self.payload.model_copy(update={'enabled':False,'expected_revision':1,'idempotency_key':uuid4().hex}),self.sub['subscription_id'])
        self.worker.after_receipt=disable
        result=self.run_one()
        self.assertEqual(result['status'],'cancelled_or_reclaimed')
        self.assertFalse(self.db.fetchall('SELECT * FROM brief_items WHERE run_id=?',(result['run_id'],)))
        self.assertEqual(len(self.provider.calls),1)

    def test_new_version_and_display_read_separate(self):
        """同版本不重复曝光，新版本仍可推荐；展示与已读独立。"""
        self.due();first=self.run_one();identity=first['items'][0]['identity']
        self.service.mark_displayed(first['run_id']);self.service.mark_displayed(first['run_id'])
        self.service.mark_read(identity,True)
        self.assertIn(identity,self.service.brief(first['run_id'])['read_identities'])
        self.service.mark_read(identity,False)
        self.provider.version='2';self.due(1);second=self.run_one()
        self.assertEqual(len(second['items']),2)
        self.assertNotIn(identity,{x['identity'] for x in second['items']})

    def test_budget_delivers_honest_partial(self):
        """预算不足只返回已有成果与未查原因，不调用精读补候选。"""
        self.worker.request_cap=1;self.due();result=self.run_one()
        self.assertEqual(len(self.provider.calls),1)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['coverage'][1]['status'],'budget_exhausted')

    def test_profile_revocation_between_query_and_publish(self):
        """画像在查询后撤销时，发布排名读取当前状态，不留下已撤销推荐理由。"""
        from research_loop.services.memory import MemoryService
        from research_loop.domain.memory import ProfileSet
        memory=MemoryService(self.db)
        entry=memory.set_profile(ProfileSet(slot='interest-'+uuid4().hex,content='language',idempotency_key=uuid4().hex))
        self.due();revoked=[]
        def revoke(*args):
            if not revoked:
                memory.forget(entry['memory_id'],entry['version']);revoked.append(True)
        self.worker.after_receipt=revoke
        result=self.run_one()
        self.assertNotIn(entry['memory_id'],dump(result['items']))

    def test_api_validates_sources_versions_and_no_requests(self):
        """真实FastAPI配置入口不会调用供应商；未知来源、过期版本均明确拒绝。"""
        from research_loop.main import app
        with patch('research_loop.api.dependencies.get_settings',return_value=self.config),TestClient(app) as client:
            response=client.get('/api/subscriptions');self.assertEqual(response.status_code,200)
            self.assertEqual(len(response.json()['sources']),17)
            data=self.payload.model_dump();data.update(idempotency_key=uuid4().hex,sources=['bogus'])
            self.assertEqual(client.post('/api/subscriptions',json=data).status_code,422)
            data.update(sources=['arxiv'],expected_revision=0)
            self.assertEqual(client.post('/api/subscriptions/'+self.sub['subscription_id'],json=data).status_code,409)
        self.assertEqual(self.provider.calls,[])


if __name__=='__main__':unittest.main()
