"""固定草稿的Reflection配对评测；零调用原稿基线与既有Agent局部修订链比较。"""
from __future__ import annotations
import argparse
import json
import time
import threading
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from research_loop.domain.notes import CreateNoteInput
from research_loop.providers.deepseek import RequestNotSent
from research_loop.providers.pdf import PDFProvider
from research_loop.services.notes import NoteService
from research_loop.services.papers import PaperService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from scripts.evaluation_common import ROOT, OUT, settings, exclusive, verify_freeze
from scripts.run_ag6 import EvaluationLedger, EvaluationRunner, summarize_usage


class FrozenLedger(EvaluationLedger):
    """复用现有物理计数；任何未结算或usage缺失都在下一次网络发送前停止。"""
    def __init__(self, path, manifest):
        """沿用共享额度，串行预留检查与父类预留共用可重入锁。"""
        super().__init__(path, manifest)
        self.lock=threading.RLock()

    def reserve(self, kind, messages, max_tokens, tools=None):
        """串行样本无并行在途允许；缺失usage不按零计，不追加恢复批次。"""
        with self.lock:
            if any(r['status'] != 'succeeded' or any(not isinstance((r.get('usage') or {}).get(k),int)
                    for k in ('prompt_tokens','completion_tokens','total_tokens')) for r in self.records):
                raise RequestNotSent('AG74_UNSETTLED_OR_USAGE_UNKNOWN')
            return super().reserve(kind, messages, max_tokens, tools)


def seed_case(client, db, config, case, prompt, corpus):
    """用当前稿服务绑定受控输入，再走真实Worker；gold标签绝不进入模型消息。

    Returns:
        任务ID、原稿身份与原稿hash。初稿为固定实验输入，不冒充模型首轮生成。
    """
    document=next(d for d in corpus['documents'] if d['key']==case['material'])
    source=document['material']
    attachment={'input_ref':case['material'],'local_path':str(ROOT/source['path']),'paper_id':source['paper_id']}
    response=client.post('/api/tasks',json={'prompt':prompt,'save_policy':'deny',
        'attachments':[attachment],'idempotency_key':'ag74-reflection-'+case['id']})
    response.raise_for_status();tid=response.json()['task_id']
    files=FileStore(config.data_dir,config.output_dir,ROOT);pdf=PDFProvider(files)
    try:
        read=PaperService(db,None,pdf).read(tid,case['material'],attachments={case['material']:attachment})
        draft=NoteService(db,files).create(tid,CreateNoteInput(title='固定草稿 '+case['id'],
            markdown=case['markdown'],evidence_ids=[p['evidence_id'] for p in read['pages']]))
    finally: pdf.close()
    db.append_event(tid,'feedback_draft_bound',{'draft_id':draft.draft_id,'content_hash':draft.content_hash,
        'evidence_ids':draft.evidence_ids,'origin':'controlled_evaluation_input','review_inherited':False})
    db.execute('UPDATE task_budgets SET model_limit=8,tool_limit=20,reserve_tail=2 WHERE root_task_id=?',(tid,))
    return tid,draft.model_dump(mode='json')


def run():
    """每份稿仅一次真实链；质量负向记录在报告，unknown/供应商失败阻止余下发送。"""
    from research_loop.main import app
    frozen=verify_freeze();spec=json.loads((ROOT/'backend/tests/acceptance/ag74_cases.json').read_text())
    corpus=json.loads((OUT/'corpus.json').read_text());config=settings();db=Database(config.database_url)
    exclusive(OUT/'reflection/started.json',{'cases':[c['id'] for c in spec['reflection']],
        'baseline':'frozen unchanged draft; zero model calls; not an end-to-end no-reflection agent run'})
    manifest={**spec['reflection_budget'],'cases':[{'id':c['id'],'request_cap':8} for c in spec['reflection']]}
    ledger=FrozenLedger(OUT/'reflection/physical.json',manifest)
    runner=object.__new__(EvaluationRunner);runner.ledger=ledger;runner.cancel_hook=False
    executor=TaskExecutor(config,db,provider_factory=runner.provider)
    results=[]
    with patch('research_loop.api.tasks.get_settings',return_value=config),TestClient(app) as client:
        for case in spec['reflection']:
            ledger.case=case['id'];started=time.monotonic()
            tid,draft=seed_case(client,db,config,case,spec['reflection_prompt'],corpus)
            exclusive(OUT/'reflection'/f'{case["id"]}-input.json',{'task_id':tid,'draft':draft,'source_case':case['id']})
            executor.execute(tid,'ag74-evaluation-worker')
            family=[tid]+[s['subtask_id'] for s in db.list_subtasks(tid)]
            receipts=db.fetchall('SELECT * FROM model_calls WHERE task_id IN ('+','.join('?' for _ in family)+')',tuple(family))
            view=client.get('/api/tasks/'+tid);view.raise_for_status()
            reflection=db.fetchone('SELECT * FROM reflection_sessions WHERE task_id=?',(tid,))
            result={'case':case['id'],'task_id':tid,'view':view.json(),'baseline':draft,
                'drafts':db.fetchall('SELECT * FROM drafts WHERE task_id=? ORDER BY created_at,draft_id',(tid,)),
                'reviews':db.fetchall('SELECT * FROM review_results WHERE task_id=?',(tid,)),
                'reflection':json.loads(reflection['state_json']) if reflection else {},
                'actions':{t:db.list_actions(t) for t in family},'receipts':receipts,
                'events':db.list_events(tid),'seconds':time.monotonic()-started,'usage':summarize_usage(receipts)}
            exclusive(OUT/'reflection'/f'{case["id"]}.json',result);results.append(result)
            print(case['id'],result['view']['status'],result['usage'],flush=True)
            if result['usage']['unknown'] or result['usage']['provider_failures'] or result['usage']['missing_usage_call_ids']:
                break
    receipts=[r for result in results for r in result['receipts']]
    assert not list(config.output_dir.rglob('*.md'))
    exclusive(OUT/'reflection/result.json',{'executed_cases':[r['case'] for r in results],
        'not_executed_cases':[c['id'] for c in spec['reflection'] if c['id'] not in {r['case'] for r in results}],
        'physical_requests':len(ledger.records),'usage':summarize_usage(receipts),
        'markdown_files':[], 'quality_review':'requires separate source audit; task terminal is not a semantic label'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['run'])
    parser.parse_args();run()
