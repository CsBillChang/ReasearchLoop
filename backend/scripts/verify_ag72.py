"""AG7.2固定组件与Agent接线验收：已有两篇PDF，独立MySQL/Qdrant，拒绝覆盖批次。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import urlparse, urlunparse
from unittest.mock import patch
import uuid

import httpx

from fastapi.testclient import TestClient

from research_loop.config import Settings
from research_loop.domain.retrieval import RetrievalQuery, fingerprint
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.providers.embeddings import LocalEmbedding
from research_loop.providers.pdf import PDFProvider
from research_loop.providers.reranker import LocalReranker
from research_loop.services.papers import PaperService
from research_loop.services.retrieval import RetrievalService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.storage.vector_index import VectorIndex
from research_loop.workers.executor import TaskExecutor
from scripts.run_ag6 import EvaluationRunner, EvaluationLedger, summarize_usage, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/ag7-2-hybrid-retrieval-20261011'


def exclusive(path: Path, value) -> None:
    """独占建立冻结/结果，历史证据不覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def settings() -> Settings:
    """只允许本机显式测试库，默认固定本轮独立命名空间。"""
    current = Settings.from_env(ROOT)
    parsed = urlparse(current.database_url)
    test_url = os.getenv('AG72_TEST_DATABASE_URL') or urlunparse(parsed._replace(path='/researchloop_test_ag72_20261011'))
    p = urlparse(test_url)
    if p.hostname not in {'localhost','127.0.0.1'} or not p.path.startswith('/researchloop_test_ag72_'):
        raise ValueError('isolated AG72 MySQL required')
    return current.model_copy(update={'database_url':test_url,'data_dir':OUT/'data','output_dir':OUT/'notes',
        'retrieval_namespace':'researchloop_test_ag72_20261011','max_model_requests':12,
        'max_tool_actions':20,'reserve_model_requests':2})


def materials(config: Settings, db: Database):
    """复用AG6已验证PDF/解析缓存，注册测试任务及当前授权页；不联网找论文。"""
    selected=json.loads((ROOT/'backend/tests/acceptance/ag6_cases.json').read_text())['materials']
    files=FileStore(config.data_dir,config.output_dir,ROOT)
    pdf=PDFProvider(files)
    papers=PaperService(db,None,pdf)
    attachments=[{'input_ref':key,'local_path':str(ROOT/selected[key]['path']),'paper_id':selected[key]['paper_id']} for key in ['od1','mllm1']]
    task='ag72-components-'+uuid.uuid4().hex
    db.create_task(task,task,task,'读取两篇原文，允许定向检索，不保存文件。','deny',attachments)
    docs=[]
    for attachment in attachments:
        key=attachment['input_ref']; value=selected[key]
        assert hashlib.sha256(Path(attachment['local_path']).read_bytes()).hexdigest()==value['sha256']
        config_hash=hashlib.sha256(json.dumps(pdf.parser_config,sort_keys=True).encode()).hexdigest()
        cache=hashlib.sha256(f'{value["sha256"]}:pymupdf:1:{config_hash}'.encode()).hexdigest()+'.json'
        shutil.copy2(ROOT/'artifacts/ag6-evaluation-20261011/first-pass/data/cache/parsed'/cache,files.parse_dir/cache)
        result=papers.read(task,key,attachments={a['input_ref']:a for a in attachments})
        docs.append({'key':key,'document_id':result['document_id'],'source_sha256':value['sha256'],
            'paper_id':value['paper_id'],'evidence_ids':[p['evidence_id'] for p in result['pages']]})
    pdf.close()
    return task,docs,attachments


def components():
    """真实本机组件检查，数据来自旧PDF；不发生成模型请求。"""
    out=OUT/'components'; config=settings(); db=Database(config.database_url); db.ensure_schema()
    exclusive(out/'started.json',{'database':urlparse(db.url).path,'generative_requests':0})
    task,docs,attachments=materials(config,db)
    exclusive(out/'inputs.json',{'task_id':task,'documents':docs,'attachments':attachments})
    vector=VectorIndex(config.qdrant_url,config.retrieval_namespace)
    embedding=LocalEmbedding(ROOT/'data/ag71-preflight/models/manifest.json')
    reranker=LocalReranker(ROOT/'data/ag71-preflight/models/manifest.json')
    service=RetrievalService(db,embedding,vector,reranker)
    queries=[('chinese','GenerateU如何用语言生成识别物体？',[docs[0]['document_id']]),
        ('english','How are region features aligned with language in open ended detection?',[docs[0]['document_id']]),
        ('exact','region word alignment',[docs[0]['document_id']]),
        ('synonym','SEED-Bench怎样评测大模型的图像和视频理解能力？',[docs[1]['document_id']]),
        ('multi','language and visual understanding tasks',[d['document_id'] for d in docs]),
        ('missing','What exact GPU driver version did these authors use?',[d['document_id'] for d in docs])]
    exclusive(out/'query-freeze.json',queries)
    summary=[]
    for name,query,ids in queries:
        start=time.monotonic(); result=service.search(task,RetrievalQuery(query=query,document_ids=ids))
        exclusive(out/(name+'.json'),result)
        assert result['status']=='ready', result['degradations']
        assert result['factual_support']=='not_assessed'
        assert all(h['document_id'] in ids and h['text']==db.get_page(h['evidence_id'])['text'][h['char_start']:h['char_end']] for h in result['hits'])
        summary.append({'case':name,'seconds':round(time.monotonic()-start,3),
            'channels':result['channels'],'top_pages':[h['page_number'] for h in result['hits']],
            'top_documents':[h['document_id'] for h in result['hits']]})
    # 控制适配器异常测试降级，原服务与集合不关闭或破坏；不声称这是网络故障注入。
    with patch.object(vector,'query',side_effect=OSError('controlled Qdrant outage')):
        degraded=service.search(task,RetrievalQuery(query='region word alignment',document_ids=[docs[0]['document_id']]))
        assert degraded['status']=='degraded' and degraded['hits']
        exclusive(out/'vector-degraded.json',degraded)
    with patch.object(reranker,'score',side_effect=OSError('controlled reranker outage')):
        degraded=service.search(task,RetrievalQuery(query='region word alignment',document_ids=[docs[0]['document_id']]))
        assert all(h['reranker_score'] is None for h in degraded['hits'])
        exclusive(out/'reranker-degraded.json',degraded)
    # 另用本机未监听端口验证真实连接拒绝；不修改共享Qdrant的状态。
    with httpx.Client(base_url='http://127.0.0.1:1',timeout=2) as unavailable:
        with patch.object(vector,'client',unavailable):
            disconnected=service.search(task,RetrievalQuery(query='region word alignment',document_ids=[docs[0]['document_id']]))
            assert disconnected['status']=='degraded' and disconnected['hits']
            exclusive(out/'vector-connect-error.json',disconnected)
    stranger=task+'-stranger'; db.create_task(stranger,stranger,stranger,'阅读','deny',[])
    for target in [stranger]:
        try: service.search(target,RetrievalQuery(query='language',document_ids=[docs[0]['document_id']]))
        except PermissionError: pass
        else: raise AssertionError('cross-task leak')
    try: reranker.score('token '*600,['source'])
    except ValueError as exc: assert str(exc)=='RERANK_INPUT_TOO_LONG'
    else: raise AssertionError('silent truncation')
    exclusive(out/'result.json',{'passed':True,'queries':summary,'source_offsets':True,
        'cross_task_rejected':True,'reranker_capacity_rejected':True,'degradation_cases':3,
        'embedding':embedding.identity,'reranker':reranker.identity,'generative_requests':0})
    vector.close()
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def real():
    """只执行预先冻结的有限API→MySQL→Worker批次；范围/账本存在即拒绝重跑。"""
    from research_loop.main import app
    out=OUT/'real'; config=settings(); db=Database(config.database_url)
    manifest=json.loads((out/'freeze.json').read_text())
    for relative,digest in manifest['source_files'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, relative
    inputs=json.loads((OUT/'components/inputs.json').read_text())
    ledger=EvaluationLedger(out/'physical.json',manifest)
    runner=object.__new__(EvaluationRunner)
    runner.ledger=ledger; runner.cancel_hook=False
    executor=TaskExecutor(config,db,provider_factory=runner.provider)
    results=[]
    with patch('research_loop.api.dependencies.get_settings',return_value=config), TestClient(app) as client:
        for case in manifest['cases']:
            ledger.case=case['id']
            reply=client.post('/api/tasks',json={'prompt':case['prompt'],'save_policy':'deny',
                'idempotency_key':'ag72-real-'+case['id'],'attachments':inputs['attachments']})
            assert reply.status_code==202,reply.text
            task=reply.json()['task_id']
            executor.execute(task,'ag72-real-worker')
            family=[task]+[s['subtask_id'] for s in db.list_subtasks(task)]
            actions={tid:db.list_actions(tid) for tid in family}
            receipts=db.fetchall('SELECT * FROM model_calls WHERE task_id IN ('+','.join('?' for _ in family)+')',tuple(family))
            view=client.get('/api/tasks/'+task).json()
            result={'case':case['id'],'task_id':task,'view':view,'actions':actions,
                'receipts':receipts,'events':{tid:db.list_events(tid) for tid in family},'usage':summarize_usage(receipts)}
            exclusive(out/(case['id']+'.json'),result); results.append(result)
            before=len(ledger.records)
            duplicate=client.post('/api/tasks',json={'prompt':case['prompt'],'save_policy':'deny',
                'idempotency_key':'ag72-real-'+case['id'],'attachments':inputs['attachments']})
            assert duplicate.json()['task_id']==task and len(ledger.records)==before
            if result['usage']['unknown'] or result['usage']['provider_failures'] or result['usage']['missing_usage_call_ids']:
                break
    receipts=[r for result in results for r in result['receipts']]
    exclusive(out/'result.json',{'executed_cases':[r['case'] for r in results],
        'physical_requests':len(ledger.records),'usage':summarize_usage(receipts),'notes':list(map(str,config.output_dir.rglob('*.md'))),
        'quality_review':'pending independent source check'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['components','real'])
    args=parser.parse_args()
    (components if args.mode=='components' else real)()
