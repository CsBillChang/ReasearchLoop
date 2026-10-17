"""AG4固定真实会话验收：隔离MySQL，最多四任务/每题四请求，无论文搜索或DPR。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.storage.mysql import Database
from research_loop.storage.memory import dump
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.services.tasks import TaskService
from research_loop.domain.tasks import TaskCreate,TaskStatus,TaskResult
from research_loop.workers.executor import TaskExecutor
from research_loop.providers.deepseek import DeepSeekProvider

INPUTS=[
    '请记住：我长期关注开放词汇目标检测，默认用中文回答科研问题。这是长期偏好。不要检索论文、不要读PDF，也不要写笔记文件；仅记录偏好并简短确认。',
    '我们刚才确定的长期研究方向是什么？只回答方向即可。本次临时用英文回答，不把英文记成长期偏好。不要检索、读PDF或保存文件。',
    '请回顾我以前表达的开放词汇目标检测研究方向与默认语言，并指出你引用的历史任务记录ID。只回顾记忆，不检索、读PDF或保存文件。',
    '请找回本会话最开始确定的两个比较维度，并用中文回答。这一轮不保存文件，也不开展新研究。若摘要省略了，请读回本会话原记录，不要猜。',
]


def manifest(root):
    """冻结实际源码与提示，测试后核对没有中途改规则。"""
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'backend').rglob('*.py'))}


def main():
    """执行一次冻结批次并保留所有响应、用量及失败，禁止生产库和重复覆盖。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--real',action='store_true')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    url=os.environ.get('AG4_TEST_DATABASE_URL','')
    if not url.rsplit('/',1)[-1].startswith('researchloop_test_ag4_real_'):
        raise ValueError('独立researchloop_test_ag4_real_*数据库必需')
    if not args.real:
        raise ValueError('真实运行必须显式--real；离线检查请运行test_memory')
    out=Path(args.output).resolve()
    out.relative_to(root/'artifacts')
    out.mkdir(parents=True,exist_ok=False)
    settings=Settings.from_env(root).model_copy(update={'database_url':url,'data_dir':out/'data','output_dir':out/'notes','max_model_requests':6,'reserve_model_requests':0})
    if not settings.deepseek_api_key:
        raise ValueError('未配置模型密钥')
    db=Database(url);db.ensure_schema()
    if db.fetchone('SELECT COUNT(*) AS n FROM tasks')['n']:
        raise ValueError('真实固定批次要求空的独立测试库，不能删除已有记录后重跑')
    initial=manifest(root)
    (out/'freeze.json').write_text(dump({'inputs':INPUTS,'physical_cap':16,'per_task_cap':4,'code':initial,'model':settings.deepseek_model,'notes':'压缩案例使用明确标记的合成历史；不宣称新论文质量或前端通过'}))
    physical=[]
    class CountedProvider(DeepSeekProvider):
        """真实发送前记录计数，所有计数不包含恢复复用。"""
        def send(self,*a,**kw):
            """批次总闸门优先，供应商自动重试关闭。"""
            if len(physical)>=16:raise RuntimeError('AG4_BATCH_CAP')
            physical.append({'number':len(physical)+1})
            (out/'physical.json').write_text(dump(physical))
            return super().send(*a,**kw)
    def factory(s):
        """保持当前配置业务模型，不以开发模型代替DeepSeek。"""
        return CountedProvider(s.deepseek_api_key,s.deepseek_model,s.deepseek_base_url,s.request_timeout_seconds)
    conv=ConversationService(db)
    first=conv.create('AG4真实偏好及追问','case12')['conversation_id']
    second=conv.create('AG4新会话召回','case3')['conversation_id']
    long=conv.create('AG4合成历史压缩','case4')['conversation_id']
    # 这是零模型历史夹具，不计真实研究任务，不修改真人评价。
    seed=['本会话比较维度：语言输入位置与训练监督信号。仅本次使用，不作为长期偏好。']+['合成历史噪声%d：'%i+'中间过程占位内容。'*250 for i in range(18)]+['合成近期消息1','合成近期消息2']
    with conv.store.transaction() as tx:
        for i,text in enumerate(seed):
            conv.add_message(tx,long,'synthetic-'+str(i),'seed:'+str(i),'user' if i%2==0 else 'assistant',text)
    results=[]
    from research_loop.main import app
    def prohibited(*a,**kw):
        """本批次禁止网络检索和PDF读取，若误调工具则保留权限失败。"""
        raise PermissionError('AG4冻结范围仅记忆，不调用论文上游')
    try:
        with patch('research_loop.api.dependencies.get_settings',return_value=settings), TestClient(app) as client, patch('research_loop.providers.discovery.DiscoveryProvider.search',side_effect=prohibited), patch('research_loop.services.papers.PaperService.read',side_effect=prohibited):
            for i,(prompt,cid) in enumerate(zip(INPUTS,[first,first,second,long]),1):
                task=client.post('/api/tasks',json={'prompt':prompt,'conversation_id':cid,'save_policy':'deny','idempotency_key':'ag4-real-'+str(i)}).json()
                tid=task['task_id']
                db.execute('UPDATE task_budgets SET model_limit=4,reserve_tail=0 WHERE root_task_id=?',(tid,))
                current=settings.model_copy(update={'memory_history_tokens':3000}) if i==4 else settings
                TaskExecutor(current,db,factory).execute(tid)
                view=client.get('/api/tasks/'+tid).json()
                receipts=db.list_model_calls(tid)
                record={'case':i,'input':prompt,'task_id':tid,'view':view,'receipts':receipts,'actions':db.list_actions(tid),'context_archives':db.fetchall('SELECT snapshot_id,metrics_json FROM context_archives WHERE task_id=?',(tid,))}
                results.append(record)
                (out/f'case-{i}.json').write_text(dump(record))
                print(dump({'case':i,'status':view['status'],'answer':(view.get('result') or {}).get('answer_markdown'),'requests':view['usage']['requests']}),flush=True)
                if view['usage']['unknown'] or any(r['status']!='succeeded' for r in receipts):
                    break
    finally:
        receipts=db.fetchall('SELECT * FROM model_calls ORDER BY created_at,sequence_no')
        usage={k:sum(json.loads(r.get('usage_json') or '{}').get(k,0) for r in receipts) for k in ('prompt_tokens','completion_tokens','total_tokens')}
        summary={'cases':len(results),'physical_requests':len(physical),'usage':usage,'unknown':sum(r['status']=='unknown' for r in receipts),'source_manifest_unchanged':manifest(root)==initial,'saved_files':len(list(settings.output_dir.rglob('*.md'))),'profile':MemoryService(db).profile(),'task_statuses':[r['view']['status'] for r in results]}
        (out/'summary.json').write_text(dump(summary))
        print(dump({k:v for k,v in summary.items() if k!='profile'}),flush=True)


if __name__=='__main__':
    main()
