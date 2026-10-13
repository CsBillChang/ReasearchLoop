"""AG5一次有界真实产品回归：上传PDF→问答→同会话反馈，只在显式隔离库运行。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.storage.mysql import Database
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.workers.executor import TaskExecutor


def main():
    """固定两任务各8次共享额度，记录失败和stream，不重试历史请求。"""
    parser=argparse.ArgumentParser();parser.add_argument('--real',action='store_true');parser.add_argument('--pdf',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    if not args.real:raise ValueError('需要--real显式运行')
    root=Path(__file__).resolve().parents[2];out=Path(args.output).resolve();out.relative_to(root/'artifacts');out.mkdir(exist_ok=False,parents=True)
    settings=Settings.from_env(root).model_copy(update={'database_url':'mysql://researchloop:researchloop_local@127.0.0.1:33316/researchloop_test_ag5_real_20261010','data_dir':out/'data','output_dir':out/'notes','max_model_requests':8,'reserve_model_requests':2})
    db=Database(settings.database_url);db.ensure_schema()
    if db.fetchone('SELECT COUNT(*) AS n FROM tasks')['n']:raise ValueError('本批次只允许运行一次，不重用已有任务库')
    freeze={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'backend/research_loop').rglob('*.py')}
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2))
    physical=[]
    class Counted(DeepSeekProvider):
        """记录物理发送；达到冻结上限立即阻断，不借余额重跑。"""
        def _count(self,kind):
            if len(physical)>=16:raise RuntimeError('AG5 frozen cap reached before send')
            physical.append({'number':len(physical)+1,'kind':kind})
            (out/'physical.json').write_text(json.dumps(physical,indent=2))
            print('physical',len(physical),kind,flush=True)
        def send(self,*a,**kw):
            self._count('json');return super().send(*a,**kw)
        def send_stream(self,*a,**kw):
            self._count('stream');return super().send_stream(*a,**kw)
    def factory(s):
        return Counted(s.deepseek_api_key,s.deepseek_model,s.deepseek_base_url,s.request_timeout_seconds)
    from research_loop.main import app
    patches=[patch('research_loop.api.'+n+'.get_settings',return_value=settings) for n in ('tasks','product','notes')]
    for p in patches:p.start()
    results=[]
    try:
        with TestClient(app) as c:
            upload=c.post('/api/uploads?filename='+Path(args.pdf).name,content=Path(args.pdf).read_bytes());upload.raise_for_status();attachment=upload.json()
            cid=c.post('/api/conversations',json={'title':'AG5真实PDF问答','idempotency_key':'ag5-real-chat'}).json()['conversation_id']
            prompts=['请阅读上传的论文，用不超过200字说明作者解决什么问题、核心方法是什么，并注明你实际读取的范围。不要搜索其他论文，不需要八点精读笔记，也不要保存文件。','这是验收测试反馈：请把上一条回答整理为两条简短要点，分别说明核心方法和证据边界，保留不确定性，不增加原文未支持的结论，不保存文件。']
            for i,prompt in enumerate(prompts):
                if i==0:
                    response=c.post('/api/tasks',json={'conversation_id':cid,'prompt':prompt,'attachments':[{'input_ref':attachment['input_ref'],'filename':Path(args.pdf).name}],'save_policy':'deny','idempotency_key':'ag5-real-1'})
                else:
                    previous=results[-1]['view'];text=previous['result']['answer_markdown']
                    response=c.post('/api/feedback',json={'conversation_id':cid,'source_task_id':previous['task_id'],'content_hash':hashlib.sha256(text.encode()).hexdigest(),'prompt':prompt,'save_policy':'deny','idempotency_key':'ag5-real-feedback'})
                response.raise_for_status();tid=response.json()['task_id']
                db.execute('UPDATE task_budgets SET model_limit=8,reserve_tail=2 WHERE root_task_id=?',(tid,))
                TaskExecutor(settings,db,factory).execute(tid)
                view=c.get('/api/tasks/'+tid).json();calls=db.fetchall('SELECT m.* FROM model_calls m JOIN tasks t ON t.task_id=m.task_id WHERE t.root_task_id=? ORDER BY m.created_at',(tid,))
                record={'input':prompt,'view':view,'receipts':calls,'events':db.list_events(tid),'artifacts':c.get('/api/tasks/'+tid+'/artifacts').json()}
                results.append(record);(out/f'case-{i+1}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str))
                print(json.dumps({'case':i+1,'status':view['status'],'answer':(view.get('result') or {}).get('answer_markdown'),'usage':view['usage']},ensure_ascii=False),flush=True)
                if view['status'] not in {'completed','partial'} or view['usage']['unknown'] or any(r['status']!='succeeded' for r in calls):break
    finally:
        for p in patches:p.stop()
        receipts=db.fetchall('SELECT * FROM model_calls ORDER BY created_at')
        summary={'cases':len(results),'physical_requests':len(physical),'usage':{k:sum(json.loads(r.get('usage_json') or '{}').get(k,0) for r in receipts) for k in ['prompt_tokens','completion_tokens','total_tokens']},'unknown':sum(r['status']=='unknown' for r in receipts),'provider_failure':sum(r['status'] in ['response_error','rejected'] for r in receipts),'statuses':[r['view']['status'] for r in results],'public_stream_events':sum(e['event_type']=='answer_stream' for r in results for e in r['events']),'saved_files':len(list(settings.output_dir.rglob('*.md'))),'frozen_core_unchanged':all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in freeze.items())}
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
