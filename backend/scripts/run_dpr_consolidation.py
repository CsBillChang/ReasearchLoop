"""一次固定自然语言验收：真实 API→隔离 MySQL→Worker，不预选论文或替模型排步骤。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from unittest.mock import patch

from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.main import app
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor

PROMPT = '找两篇近三年的开放词汇目标检测论文，比较它们如何利用语言信息。基于实际读取的原文生成一份简短比较笔记，注明出处和未解决的问题，并保存到项目中。'


def snapshot(db: Database) -> dict:
    """只读生产内容身份，禁止把密钥或正文数据库 URL 写入 artifact。"""
    counts = {}
    for name in ('tasks','model_calls','drafts','save_receipts','documents','page_evidence'):
        counts[name] = db.fetchone('SELECT COUNT(*) AS n FROM '+name)['n']
    exports = []
    for row in db.fetchall('SELECT path FROM save_receipts'):
        path=Path(row['path'])
        exports.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None})
    return {'counts':counts,'export_hashes':exports}


def main() -> None:
    """显式 --run 才消耗一次批次；重复执行读取已有产物，绝不续跑未知任务。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--readback',action='store_true')
    args=parser.parse_args()
    if not (args.run or args.readback):parser.error('use --run once, or --readback')
    settings=Settings.from_env()
    root=settings.project_root
    artifact=root/'artifacts/ag12-dpr-consolidation-20261009'
    artifact.mkdir(parents=True,exist_ok=True)
    u=urlsplit(settings.database_url)
    url=urlunsplit((u.scheme,u.netloc,'/researchloop_test_dpr_live_20261009','',''))
    isolated=settings.model_copy(update={'database_url':url,'data_dir':artifact/'runtime','output_dir':artifact/'notes',
                                         'max_model_requests':32,'max_tool_actions':60,'reserve_model_requests':6})
    db=Database(url);db.ensure_schema()
    identity_path=artifact/'task.json'
    if args.run and identity_path.exists():raise SystemExit('batch already created; use --readback, never rerun')
    if args.run:
        source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (root/'backend/research_loop').rglob('*.py')}
        freeze={'prompt':PROMPT,'model':settings.deepseek_model,'model_limit':32,'tool_limit':60,'reserve_tail':6,
                'reviewer_limit':4,'summary_output_budgets':[4096,2048],'model_http_timeout':settings.request_timeout_seconds,
                'database':'researchloop_test_dpr_live_20261009','source_hashes':source_hashes}
        (artifact/'freeze.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2))
        (artifact/'production-before.json').write_text(json.dumps(snapshot(Database(settings.database_url)),ensure_ascii=False,indent=2))
        with patch('research_loop.api.tasks.get_settings',return_value=isolated),TestClient(app) as client:
            response=client.post('/api/tasks',json={'prompt':PROMPT,'idempotency_key':'dpr-consolidation-20261009-one-task','save_policy':'allow'})
            response.raise_for_status();created=response.json()
            identity_path.write_text(json.dumps(created,ensure_ascii=False,indent=2))
            task_id=created['task_id']
            # 仅修改本次尚未执行的隔离任务额度；不改变生产默认或历史账本。
            db.execute('UPDATE task_budgets SET model_limit=32,tool_limit=60,reserve_tail=6 WHERE root_task_id=? AND model_used=0',(task_id,))
            print(json.dumps({'task_id':task_id,'status':'started','cap':32}),flush=True)
            TaskExecutor(isolated,db).execute(task_id,'dpr-consolidation-once')
    else:
        task_id=json.loads(identity_path.read_text())['task_id']
    with patch('research_loop.api.tasks.get_settings',return_value=isolated),TestClient(app) as client:
        view=client.get('/api/tasks/'+task_id).json()
    children=db.list_subtasks(task_id)
    ids=[task_id]+[child['subtask_id'] for child in children]
    for tid in ids:
        payload={'task':db.get_task(tid),'actions':db.list_actions(tid),'events':db.list_events(tid),
                 'receipts':db.list_model_calls(tid),'checkpoint':db.get_checkpoint(tid)}
        (artifact/(tid+'.json')).write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str))
    result=view.get('result') or {}
    if result.get('answer_markdown'):(artifact/'answer.md').write_text(result['answer_markdown'])
    (artifact/'result.json').write_text(json.dumps(view,ensure_ascii=False,indent=2))
    (artifact/'production-after.json').write_text(json.dumps(snapshot(Database(settings.database_url)),ensure_ascii=False,indent=2))
    print(json.dumps({'task_id':task_id,'status':view['status'],'usage':view['usage'],
                      'saved_path':result.get('saved_path'),'review_status':result.get('review_status')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
