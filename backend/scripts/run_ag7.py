"""AG7.5本机交付验证：先隔离启停，再冻结已有PDF两问与最多12次真实请求。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse, urlunparse

import httpx
from research_loop.config import Settings
from research_loop.services.runtime import application_lock
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from scripts.manage import Manager, ROOT
from scripts.backup_restore import verify_backup, digest, snapshot
from scripts.evaluation_common import exclusive
from scripts.evaluate_reflection import FrozenLedger
from scripts.run_ag6 import EvaluationRunner, summarize_usage

OUT=ROOT/'artifacts/ag7-5-local-delivery-20261011'
SPEC={
    'physical_cap':12,'token_cap':600000,'wall_seconds':1800,
    'cases':[
        {'id':'Q1','request_cap':8,'prompt':'请只依据我上传的这篇论文，用中文简短说明它试图解决什么问题、核心方法是什么，控制在200字左右并给出原文页码。不要检索其他论文，不需要八点精读笔记，不保存文件。'},
        {'id':'Q2','request_cap':8,'prompt':'接着刚才这篇论文：你提到的方法有哪些关键策略？分别解决什么问题？请沿用同一份原文，简短回答并注明页码，不搜索、不生成精读笔记、不保存。'}
    ],
    'pdf':'artifacts/ag6-evaluation-20261011/materials/mllm3.pdf',
    'save_policy':'deny','retry_batch':False,'new_protocol':False,
}


def configuration() -> Settings:
    """最终验证使用独立本机库、输出和向量集合；不接管生产队列。"""
    current=Settings.from_env(ROOT)
    parsed=urlparse(current.database_url)
    if parsed.scheme!='mysql' or parsed.hostname not in {'127.0.0.1','localhost'}:
        raise ValueError('LOCAL_MYSQL_REQUIRED')
    return current.model_copy(update={
        'database_url':urlunparse(parsed._replace(path='/researchloop_test_ag75_20261011')),
        'data_dir':OUT/'real/data','output_dir':OUT/'real/notes',
        'retrieval_namespace':'researchloop_test_ag75_20261011',
        'max_model_requests':8,'max_tool_actions':20,'reserve_model_requests':2})


def migrate(config: Settings) -> None:
    """对显式隔离库执行正常迁移，不跳过数据库就绪检查。"""
    env={**os.environ,'DATABASE_URL':config.database_url,'PYTHONPATH':str(ROOT/'backend')}
    result=subprocess.run([sys.executable,str(ROOT/'backend/scripts/migrate.py')],env=env,cwd=ROOT,capture_output=True,text=True)
    (OUT/'isolated-migration.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError('ISOLATED_MIGRATION_FAILED')


def verify_runtime() -> None:
    """真实隔离进程启动、重复启动、停止和重启；不提交研究任务。"""
    config=configuration();migrate(config);db=Database(config.database_url)
    if db.fetchone('SELECT COUNT(*) AS n FROM tasks')['n']:
        raise ValueError('RUNTIME_TEST_REQUIRES_EMPTY_TASKS')
    manager=Manager(config,'ag75-validation');checks={}
    try:
        first=manager.start(18375,4175,scheduler=True)
        second=manager.start(18375,4175,scheduler=True)
        checks['idempotent_start']=first['services']==second['services'] or all(first['services'][k]['pid']==second['services'][k]['pid'] for k in first['services'])
        with httpx.Client(base_url='http://127.0.0.1:18375',timeout=10) as client:
            checks['health']=client.get('/api/health').json()
            checks['ready']=client.get('/api/ready').json()
        checks['first_runtime']=first
        stopped=manager.stop();checks['stop']=stopped
        assert stopped['stopped']
        again=manager.start(18375,4175,scheduler=True)
        checks['restart_new_pids']=all(again['services'][k]['pid']!=first['services'][k]['pid'] for k in first['services'])
        checks['stop_again']=manager.stop()
        checks['receipts']=db.fetchone('SELECT COUNT(*) AS n FROM model_calls')['n']
        checks['tasks']=db.fetchone('SELECT COUNT(*) AS n FROM tasks')['n']
        checks['passed']=all([checks['idempotent_start'],checks['restart_new_pids'],checks['ready']['database_ready'],checks['receipts']==0,checks['tasks']==0,checks['stop_again']['stopped']])
        exclusive(OUT/'runtime-check.json',checks)
        assert checks['passed']
    finally:
        manager.stop()


def freeze() -> None:
    """在真实调用之前独占封存材料、两问、当前源码、预算和先行证据。"""
    runtime=json.loads((OUT/'runtime-check.json').read_text())
    restore=json.loads((OUT/'restore-result.json').read_text())
    if not runtime['passed'] or not restore['model_receipts_unchanged']:
        raise ValueError('ZERO_AND_ISOLATED_GATES_REQUIRED')
    verify_backup(OUT/'private-backup')
    files=list((ROOT/'backend/research_loop').rglob('*.py'))+[Path(__file__),ROOT/'backend/scripts/run_ag6.py',ROOT/'backend/scripts/evaluate_reflection.py',ROOT/SPEC['pdf']]
    exclusive(OUT/'real/freeze.json',{'scope':SPEC,'files':{str(p.relative_to(ROOT)):digest(p) for p in files},
        'model':configuration().deepseek_model,'zero_tests':'../zero-tests.log','isolation':'../runtime-check.json',
        'source':'existing AG6 material, not a new-paper or old evaluation rerun',
        'stop':'unknown, missing usage or supplier error stops subsequent sends; no extra batch'})


def verify_freeze() -> dict:
    """运行前核对材料和源码未变；冻结后禁止静默换问题或预算。"""
    frozen=json.loads((OUT/'real/freeze.json').read_text())
    if frozen['scope']!=SPEC:
        raise ValueError('FROZEN_SCOPE_CHANGED')
    for name,expected in frozen['files'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('FROZEN_FILE_CHANGED: '+name)
    return frozen


def real() -> None:
    """HTTP API→隔离MySQL→当前Worker→真实供应商；同会话追问且最多12次物理发送。"""
    frozen=verify_freeze();config=configuration();db=Database(config.database_url)
    if db.fetchone('SELECT COUNT(*) AS n FROM tasks')['n']:
        raise ValueError('REAL_BATCH_ALREADY_HAS_TASKS')
    exclusive(OUT/'real/started.json',{'scope':frozen['scope']})
    ledger=FrozenLedger(OUT/'real/physical.json',SPEC)
    runner=object.__new__(EvaluationRunner);runner.ledger=ledger;runner.cancel_hook=False
    executor=TaskExecutor(config,db,provider_factory=runner.provider)
    manager=Manager(config,'ag75-validation');results=[]
    try:
        manager.start(18375,4175,worker=False)
        with application_lock(ROOT),httpx.Client(base_url='http://127.0.0.1:18375',timeout=30) as client:
            response=client.post('/api/conversations',json={'title':'AG7.5 本机交付问答','idempotency_key':'ag75-final'})
            response.raise_for_status();conversation=response.json()['conversation_id']
            upload=client.post('/api/uploads',params={'filename':'TVP.pdf'},content=(ROOT/SPEC['pdf']).read_bytes(),headers={'content-type':'application/pdf'})
            upload.raise_for_status()
            for case in SPEC['cases']:
                ledger.case=case['id']
                payload={'conversation_id':conversation,'prompt':case['prompt'],'idempotency_key':'ag75-'+case['id'],'save_policy':'deny'}
                if case['id']=='Q1':
                    payload['attachments']=[{'input_ref':upload.json()['input_ref'],'filename':'TVP.pdf'}]
                accepted=client.post('/api/tasks',json=payload);accepted.raise_for_status();tid=accepted.json()['task_id']
                exclusive(OUT/'real'/(case['id']+'-input.json'),{'payload':payload,'task_id':tid})
                db.execute('UPDATE task_budgets SET model_limit=8,tool_limit=20,reserve_tail=2 WHERE root_task_id=?',(tid,))
                started=time.monotonic();executor.execute(tid,'ag75-final-worker')
                response=client.get('/api/tasks/'+tid);response.raise_for_status()
                result={'case':case['id'],'task_id':tid,'view':response.json(),'seconds':time.monotonic()-started,
                    'context':db.fetchall('SELECT * FROM context_archives WHERE task_id=?',(tid,)),
                    'events':db.list_events(tid),'actions':db.list_actions(tid)}
                exclusive(OUT/'real'/(case['id']+'-result.json'),result);results.append(result)
                if any(r['status']!='succeeded' or not r.get('usage') for r in ledger.records):
                    break
                if result['view']['status'] not in {'completed','partial'}:
                    break
            conversation_view=client.get('/api/conversations/'+conversation);conversation_view.raise_for_status()
            exclusive(OUT/'real/conversation.json',conversation_view.json())
    finally:
        receipts=db.fetchall('SELECT * FROM model_calls ORDER BY created_at,call_id')
        exclusive(OUT/'real/receipts.json',receipts)
        exclusive(OUT/'real/evidence.json',db.fetchall('SELECT * FROM page_evidence'))
        exclusive(OUT/'real/summary.json',{'questions':len(results),'physical':len(ledger.records),
            'usage':summarize_usage(receipts),'saved_files':list(map(str,config.output_dir.rglob('*.md'))) if config.output_dir.exists() else [],
            'statuses':{r['case']:r['view']['status'] for r in results}})
        manager.stop()


def main() -> None:
    """分别运行零模型/冻结/真实三步，禁止默认隐式调用。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify-runtime','freeze','real'])
    args=parser.parse_args()
    {'verify-runtime':verify_runtime,'freeze':freeze,'real':real}[args.command]()


if __name__=='__main__':
    main()
