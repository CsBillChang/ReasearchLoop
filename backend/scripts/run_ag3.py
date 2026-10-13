"""AG3 有界真实验收：固定附件/问题，隔离库与产物，API→Worker，禁止自动语义重试。"""
from __future__ import annotations

import os
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
from unittest.mock import patch

from fastapi.testclient import TestClient

from research_loop.config import get_settings
from research_loop.main import app
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor


PROMPTS = {
    'T1': '''请读取附件 RARR，核对并修订我提供的下列待核查草稿。先将草稿原样登记，再独立核验；仅修改有问题的句子，修改后独立复核，并保存一份简短笔记到项目中。开头和结尾未涉及问题的文字逐字保留。只用这份附件，不搜索其他论文，不需要八点精读。
待核查草稿（含验收预置疑点，不代表正确事实）：
# 方法记录
RARR 研究生成文本的归因与修订。
RARR 的查询生成、agreement model 和 edit model 都通过微调 T5 实现。
以上记录仅针对这篇论文。''',
    'T2': '''仅依据附件 RARR 核验：论文实验的 GPU 驱动版本是否为 999.99？请自主读取原文并检查这一具体说法，不搜索外部来源，不要猜测。无法找到依据时，说明已经读到的范围、哪些内容可以确认、具体不能确认的内容与停止理由。不要保存文件，不需要八点精读。''',
}
LIMITS = {'T1': (24, 6), 'T2': (12, 4)}


def dump(path: Path, value) -> None:
    """写入 JSON 证据，不输出密钥或完整 Settings。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def run(output: Path, case: str) -> None:
    """在已冻结目录执行一次指定案例，不覆盖旧任务或复用 unknown。"""
    if (output/case/'started.json').exists():
        raise ValueError('case already started; no automatic repeat or resume')
    freeze=json.loads((output/'freeze.json').read_text())
    root=Path(__file__).resolve().parents[2]
    current={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in (root/'backend').rglob('*.py')}
    update_path = output/'freeze-after-engineering-fix.json'
    update = json.loads(update_path.read_text()) if update_path.exists() else {}
    expected = update['source_hashes'] if case in update.get('remaining_cases', []) else freeze['source_hashes']
    if current!=expected:
        raise ValueError('source changed after freeze')
    base=get_settings(); model_limit,reserve=freeze['limits'][case]
    settings=base.model_copy(update={'database_url':freeze['database_url'], 'data_dir':output/'data',
        'output_dir':output/'notes','max_model_requests':model_limit,'reserve_model_requests':reserve})
    if not settings.database_url.rsplit('/',1)[-1].startswith('researchloop_test_'):
        raise ValueError('isolated database required')
    if not settings.deepseek_api_key:
        raise ValueError('business model not configured')
    pdf=output/'inputs'/'RARR.pdf'
    if sha256(pdf.read_bytes()).hexdigest()!=freeze['pdf_sha256']:
        raise ValueError('source PDF changed')
    db=Database(settings.database_url);db.ensure_schema()
    with patch('research_loop.api.tasks.get_settings', return_value=settings),TestClient(app) as client:
        payload={'prompt':PROMPTS[case],'save_policy':'allow' if case=='T1' else 'deny','idempotency_key':str(output)+'-'+case,
                 'attachments':[{'input_ref':'RARR','local_path':str(pdf),'paper_id':'RARR-fixed-source'}]}
        response=client.post('/api/tasks',json=payload);response.raise_for_status();task=response.json()['task_id']
        # 独立验收预算同时进入根账本及图；不改变生产日常默认配置。
        db.execute('UPDATE task_budgets SET model_limit=?,reserve_tail=? WHERE root_task_id=?',(model_limit,reserve,task))
        dump(output/case/'started.json',{'task_id':task,'input':payload,'model_limit':model_limit,'reserve_tail':reserve})
        print(case,task,'started',flush=True)
        TaskExecutor(settings,db).execute(task,'ag3-bounded-acceptance')
        view=client.get('/api/tasks/'+task).json();dump(output/case/'view.json',view)
        for tid in [task]+[c['subtask_id'] for c in db.list_subtasks(task)]:
            dump(output/case/(tid+'.json'),{'task':db.get_task(tid),'actions':db.list_actions(tid),
                'receipts':db.list_model_calls(tid),'checkpoint':db.get_checkpoint(tid)})
        (output/case/'answer.md').write_text((view.get('result') or {}).get('answer_markdown') or '')
        print(json.dumps({'case':case,'status':view['status'],'usage':view['usage'],
                         'review_status':(view.get('result') or {}).get('review_status')},ensure_ascii=False),flush=True)
        if view['usage']['unknown'] or view['usage']['requests']>model_limit:
            dump(output/'STOP.json',{'case':case,'reason':'unknown or budget violation'})


def main() -> None:
    """冻结或执行一次；固定参数只用于本验收，不进入生产 Agent 的判断逻辑。"""
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=['freeze','T1','T2'],required=True)
    parser.add_argument('--database-url')
    parser.add_argument('--repair', action='store_true', help='Freeze the newly authorized boundary repair batch, T1<=16 and T2<=12; never modify an old freeze')
    args=parser.parse_args();output=args.output.resolve();root=Path(__file__).resolve().parents[2]
    if args.case=='freeze':
        if (output/'freeze.json').exists():raise ValueError('freeze already exists')
        if not args.database_url or not args.database_url.rsplit('/',1)[-1].startswith('researchloop_test_'):
            raise ValueError('isolated database required')
        (output/'inputs').mkdir(parents=True,exist_ok=True)
        shutil.copyfile((Path(os.environ.get("RESEARCHLOOP_TEST_PDF_DIR", str(Path.home() / "Downloads"))) / "RARR.pdf"),output/'inputs'/'RARR.pdf')
        limits = {'T1': (16, 4), 'T2': (12, 4)} if args.repair else LIMITS
        dump(output/'freeze.json',{'prompts':PROMPTS,'limits':limits,'total_physical_limit':sum(x[0] for x in limits.values()),'database_url':args.database_url,
             'pdf_sha256':sha256((output/'inputs'/'RARR.pdf').read_bytes()).hexdigest(),
             'source_hashes':{str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in (root/'backend').rglob('*.py')},
             'test_input_notice':'T1错误句是验收提供的用户草稿，不是模型自然错误率样本；T2为明确缺证问题。'})
        print('frozen; zero business model calls')
    else:
        if (output/'STOP.json').exists():raise ValueError('batch stopped')
        run(output,args.case)


if __name__=='__main__':main()
