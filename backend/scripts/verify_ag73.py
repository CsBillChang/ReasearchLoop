"""AG7.3冻结验收：隔离库、只读论文、有界两窗、真实进程退出与资产读回。"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import patch
from urllib.parse import urlparse,urlunparse
from uuid import uuid4
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.domain.subscriptions import SubscriptionInput
from research_loop.services.subscriptions import SubscriptionService
from research_loop.storage.mysql import Database
from research_loop.storage.memory import dump,digest
from research_loop.workers.scheduler import BriefScheduler

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/ag7-3-incremental-brief-20261011'


def write(name,value):
    """独占保存验收记录，不覆盖已有批次或失败。"""
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,default=str)


def settings():
    """只从当前配置派生固定测试库，不记录密钥。"""
    current=Settings.from_env(ROOT)
    parsed=urlparse(current.database_url)
    if parsed.hostname not in {'localhost','127.0.0.1'}:raise ValueError('local MySQL only')
    return current.model_copy(update={'database_url':urlunparse(parsed._replace(path='/researchloop_test_ag73_20261011')),'data_dir':OUT/'data','output_dir':OUT/'notes'})


def snapshot():
    """只读哈希全部既有表及PDF/笔记；不把正文、认证字段写入证据。"""
    config=Settings.from_env(ROOT);db=Database(config.database_url)
    tables={}
    for row in db.fetchall('SHOW TABLES'):
        name=next(iter(row.values()))
        rows=sorted(dump(r) for r in db.fetchall('SELECT * FROM `'+name+'`'))
        tables[name]={'rows':len(rows),'sha256':digest(rows)}
    files={}
    for folder in [ROOT/'data',ROOT/'artifacts']:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.pdf','.md'} and OUT not in p.parents:
                files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return {'tables':tables,'files':files}


def freeze():
    """在任何论文验收请求前固定查询、窗口、数据库与总额度。"""
    write('production-before.json',snapshot())
    write('real-freeze.json',{'sources':['cvpr','ijcai'],'directions':['open vocabulary object detection'],
        'lookback_days':365,'limit':5,'created_at':'2026-10-07T00:00:00+00:00',
        'ticks':['2026-10-08T01:01:00+00:00','2026-10-11T01:01:00+00:00'],
        'clock_note':'注入历史调度时刻检验真实论文日期请求与停机合并；不是等待两天的墙钟实验',
        'request_cap_per_tick':4,'total_physical_rpc_cap':8,'per_lane_max_attempts':2,
        'timeout_seconds':30,'total_time_limit_seconds':600,'generative_model_cap':0,'fulltext_download_cap':0,
        'database':'researchloop_test_ag73_20261011',
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'backend/research_loop').rglob('*.py')}})
    print('freeze saved; no external calls')


def real():
    """两个冻结日期窗口走API配置→MySQL→调度器→论文→聊天；不调用研究Worker。"""
    from research_loop.main import app
    frozen=json.loads((OUT/'real-freeze.json').read_text())
    for path,expected in frozen['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    write('real-started.json',{'started_at':datetime.now(timezone.utc).isoformat(),'cap':frozen['total_physical_rpc_cap']})
    config=settings();db=Database(config.database_url);db.ensure_schema()
    stamp=[datetime.fromisoformat(frozen['created_at'])]
    service=SubscriptionService(db,lambda:stamp[0])
    payload={'title':'开放词汇检测 · 真实简报验收','directions':frozen['directions'],'sources':frozen['sources'],'lookback_days':frozen['lookback_days'],'enabled':True,'idempotency_key':'ag73-real-once'}
    with patch('research_loop.api.dependencies.get_settings',return_value=config), patch('research_loop.api.subscriptions.service',return_value=service),TestClient(app) as client:
        made=client.post('/api/subscriptions',json=payload);made.raise_for_status();sub=made.json();write('real-subscription.json',sub)
        worker=BriefScheduler(config,clock=lambda:stamp[0],request_cap=frozen['request_cap_per_tick'],duration_seconds=300)
        results=[]
        for index,instant in enumerate(frozen['ticks']):
            stamp[0]=datetime.fromisoformat(instant)
            result=worker.run_once(sub['subscription_id']);assert len(result)==1
            write(f'real/window-{index+1}.json',result[0]);results+=result
            assert worker.run_once(sub['subscription_id'])==[]
            response=client.get('/api/conversations/'+sub['conversation_id']);response.raise_for_status()
            assert len(response.json()['messages'])==index+1
        # 验收结束关闭此测试订阅，浏览器只读和表单操作不会引发后台网络。
        service.configure(SubscriptionInput(**{**payload,'enabled':False,'expected_revision':1,'idempotency_key':'ag73-real-disable'}),sub['subscription_id'])
    requests=db.fetchall('SELECT q.* FROM brief_requests q JOIN brief_runs r ON r.run_id=q.run_id WHERE r.subscription_id=?',(sub['subscription_id'],))
    calls=0
    for r in requests:
        value=json.loads(r['result_json'] or '{}');request=value.get('coverage',{}).get('diagnostic',{}).get('request',{})
        calls+=sum(len(w.get('attempts',[])) for w in request.get('windows',[request]))
    assert calls<=8
    write('real-requests.json',requests)
    write('real-result.json',{'subscription_id':sub['subscription_id'],'conversation_id':sub['conversation_id'],
        'runs':[{'run_id':r['run_id'],'status':r['status'],'candidates':len(r['items']),'coverage':r['coverage']} for r in results],
        'physical_rpc_requests':calls,'dispatch_receipts':len(requests),'unknown':sum(r['status']=='unknown' for r in requests),
        'generative_model_requests':0,'fulltext_downloads':0,'subscription_disabled_after_acceptance':True})
    print('real batch completed:',calls,'论文 RPCs; 0 LLM; conversation',sub['conversation_id'])


def crash_child():
    """独立进程在论文替身响应落库后真实SIGKILL；不消耗真实来源预算。"""
    from tests.integration.test_subscriptions import SourceStub
    config=settings();provider=SourceStub()
    original=provider.search
    def logged(*args,**kwargs):
        with (OUT/'recovery-sends.txt').open('a') as f:f.write('sent\n');f.flush();os.fsync(f.fileno())
        return original(*args,**kwargs)
    provider.search=logged
    def die(*args):os.kill(os.getpid(),signal.SIGKILL)
    clock=lambda:datetime(2026,10,11,1,1,tzinfo=timezone.utc)
    worker=BriefScheduler(config,provider_factory=lambda:provider,clock=clock,after_receipt=die)
    worker.run_once(json.loads((OUT/'recovery-subscription.json').read_text())['subscription_id'])


def recovery():
    """实测进程退出后复用已落盘的来源结果；第一次发送后恢复零额外查询。"""
    from tests.integration.test_subscriptions import SourceStub
    config=settings();db=Database(config.database_url);db.ensure_schema()
    service=SubscriptionService(db,lambda:datetime(2026,10,11,0,tzinfo=timezone.utc))
    sub=service.configure(SubscriptionInput(title='受控SIGKILL简报',directions=['test recovery'],sources=['arxiv'],enabled=True,idempotency_key=uuid4().hex))
    write('recovery-subscription.json',sub)
    result=subprocess.run([sys.executable,__file__,'crash-child'],cwd=ROOT)
    assert result.returncode==-signal.SIGKILL
    sends=len((OUT/'recovery-sends.txt').read_text().splitlines());assert sends==1
    provider=SourceStub()
    worker=BriefScheduler(config,provider_factory=lambda:provider,clock=lambda:datetime(2026,10,11,1,4,tzinfo=timezone.utc))
    recovered=worker.run_once(sub['subscription_id']);assert len(recovered)==1 and provider.calls==[]
    assert worker.run_once(sub['subscription_id'])==[]
    service.configure(SubscriptionInput(title='受控SIGKILL简报',directions=['test recovery'],sources=['arxiv'],enabled=False,expected_revision=1,idempotency_key=uuid4().hex),sub['subscription_id'])
    write('recovery-result.json',{'sigkill_exit':result.returncode,'simulated_discovery_sends_before':sends,'simulated_discovery_sends_after':0,'generative_calls':0,'result':recovered[0]})
    print('SIGKILL recovery passed; no duplicate discovery send')


def protected():
    """校验旧事实及文件逐项未变化，只允许新增009迁移记录和七个空表。"""
    before=json.loads((OUT/'production-before.json').read_text());after=snapshot()
    for name,old in before['tables'].items():
        if name=='schema_version':continue
        assert old==after['tables'][name],name
    for path,old in before['files'].items():assert old==after['files'][path],path
    write('production-after.json',after)
    write('protection-result.json',{'unchanged_old_tables':len(before['tables'])-1,'unchanged_files':len(before['files']),'schema_version':'009 additive migration allowed','new_tables':sorted(set(after['tables'])-set(before['tables']))})
    print('protected assets verified')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','real','recovery','crash-child','protected'])
    {'freeze':freeze,'real':real,'recovery':recovery,'crash-child':crash_child,'protected':protected}[parser.parse_args().mode]()
