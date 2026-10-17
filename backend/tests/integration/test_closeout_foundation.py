"""交付后补齐接线验收：真实API/数据库/Worker，供应商明确替身且禁止网络。"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.domain.tasks import TaskCreate
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.storage.token_budget import TokenBudgetStore, estimate_request
from research_loop.workers.executor import TaskExecutor


class CloseoutFoundationTests(unittest.TestCase):
    """同组测试可显式切至研究专用MySQL库，所有任务使用独立身份。"""

    def setUp(self):
        """拒绝生产库并隔离文件目录；默认SQLite零外部依赖。"""
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        url=os.getenv('CLOSEOUT_TEST_DATABASE_URL') or 'sqlite:///'+str(root/'closeout.db')
        if not url.startswith('sqlite:') and not url.rsplit('/',1)[-1].startswith('researchloop_test_closeout_'):
            raise ValueError('isolated closeout database required')
        self.settings=Settings(project_root=root,data_dir=root/'data',output_dir=root/'notes',database_url=url,
                               deepseek_api_key='STUB',retrieval_enabled=False)
        self.db=Database(url);self.db.ensure_schema()
        from research_loop.main import app
        p=patch('research_loop.api.dependencies.get_settings',return_value=self.settings);p.start();self.addCleanup(p.stop)
        self.client=TestClient(app);self.addCleanup(self.client.close)
        self.task=self.create()

    def create(self):
        """API接受任务但不自动启动Worker。"""
        response=self.client.post('/api/tasks',json={'prompt':'报告已有材料，不检索，不保存','idempotency_key':str(uuid4())})
        self.assertEqual(response.status_code,202,response.text)
        return response.json()['task_id']

    def seed_paper(self):
        """固定合成元数据不是新的论文检索或真实质量样本。"""
        pid='fixture:'+str(uuid4())
        self.db.save_paper(pid,{'paper_id':pid,'title':'合成论文 <script>','source':'fixture',
                               'paper_url':'https://example.invalid/paper','local_path':'secret'})
        self.db.grant_paper(self.task,pid,self.task)
        return pid

    def test_api_worker_usage_and_success_receipt_replay(self):
        """真实Worker发送前有预留，原响应恢复无需第二次供应商调用。"""
        from research_loop.agents.graph import build_graph, initial_state
        from research_loop.tools.registry import ToolRegistry
        db=self.db;tid=self.task;calls=[]
        class Provider:
            def send(self,messages,tools,**kwargs):
                calls.append(kwargs)
                held=TokenBudgetStore(db).summary(tid)
                assert held['reserved_tokens']>0
                assert kwargs['max_tokens']>0
                return {'id':'stub-receipt','usage':{'total_tokens':80},'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'现有材料不足，未新增检索或保存。'}}]}
            def close(self):pass
        with patch('httpx.Client.send',side_effect=AssertionError('network forbidden')):
            TaskExecutor(self.settings,self.db,lambda _:Provider()).execute(tid)
            # 直接从请求序号1重入图验证原messages复用，不把终态重新入队。
            result=build_graph(Provider(),ToolRegistry([]),{},db).invoke(initial_state(tid,'原输入','deny'))
        self.assertEqual(len(calls),1);self.assertIn('不足',result['final_answer'])
        budget=self.client.get('/api/tasks/'+tid).json()['usage']['budget']['tokens']
        self.assertEqual((budget['used_tokens'],budget['reserved_tokens']),(80,0))
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))

    def test_api_worker_budget_denial_delivers_reason_without_send(self):
        """预算拒绝有可见输出，不新建模型收据或空转任务。"""
        self.db.execute('UPDATE task_token_budgets SET token_limit=1,tail_tokens=0 WHERE root_task_id=?',(self.task,))
        class Forbidden:
            def send(self,*args,**kwargs):raise AssertionError('must not send')
            def close(self):pass
        TaskExecutor(self.settings,self.db,lambda _:Forbidden()).execute(self.task)
        view=self.client.get('/api/tasks/'+self.task).json()
        self.assertEqual(view['status'],'failed')
        self.assertTrue(view['result']['answer_markdown'])
        self.assertIn('TOKEN_BUDGET',json.dumps(view['result']))
        self.assertEqual(self.db.list_model_calls(self.task),[])

    def test_parallel_root_budget_rejects_one_child(self):
        """MySQL行锁与SQLite写锁都不能超卖同根剩余额度。"""
        self.db.execute('UPDATE task_token_budgets SET token_limit=5000,tail_tokens=1000 WHERE root_task_id=?',(self.task,))
        request={'messages':[{'role':'user','content':'x'}],'max_tokens':1000}
        def reserve(i):
            try:TokenBudgetStore(self.db).reserve(self.task,self.task+'-'+str(i),1,request,'researcher');return True
            except ValueError:return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(pool.map(reserve,range(2))),1)
        self.assertEqual(TokenBudgetStore(self.db).summary(self.task)['reserved_tokens'],estimate_request(request))

    def test_receipt_recovery_in_new_database_instance(self):
        """响应已持久而未结算，重建服务只补计一次usage。"""
        request={'messages':[],'max_tokens':1000};store=TokenBudgetStore(self.db)
        store.reserve(self.task,self.task,1,request,'supervisor')
        with patch.object(TokenBudgetStore,'settle',side_effect=RuntimeError('controlled exit')):
            with self.assertRaises(RuntimeError):self.db.save_model_call(self.task,1,'succeeded',request,response={},usage={'total_tokens':75},sent_state='sent')
        fresh=TokenBudgetStore(Database(self.db.url));fresh.settle(self.task,1);fresh.settle(self.task,1)
        self.assertEqual((fresh.summary(self.task)['used_tokens'],fresh.summary(self.task)['reserved_tokens']),(75,0))

    def test_sigkill_after_response_does_not_resend_or_double_charge(self):
        """子进程落收据后真实SIGKILL，恢复只补结算，没有供应商网络。"""
        import subprocess
        import signal
        import sys
        code = '''
import os, signal
from unittest.mock import patch
from research_loop.storage.mysql import Database
from research_loop.storage.token_budget import TokenBudgetStore
db=Database(os.environ['CLOSEOUT_RECOVERY_URL']);tid=os.environ['CLOSEOUT_RECOVERY_TASK']
request={'messages':[], 'max_tokens':1000}
TokenBudgetStore(db).reserve(tid,tid,1,request,'supervisor')
def terminate(*args):os.kill(os.getpid(),signal.SIGKILL)
with patch.object(TokenBudgetStore,'settle',side_effect=terminate):
 db.save_model_call(tid,1,'succeeded',request,response={'id':'persisted-before-kill'},usage={'total_tokens':67},sent_state='sent')
'''
        result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'CLOSEOUT_RECOVERY_URL':self.db.url,
                              'CLOSEOUT_RECOVERY_TASK':self.task},capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,-signal.SIGKILL,result.stderr)
        fresh=TokenBudgetStore(Database(self.db.url));fresh.settle(self.task,1);fresh.settle(self.task,1)
        self.assertEqual((fresh.summary(self.task)['used_tokens'],fresh.summary(self.task)['reserved_tokens']),(67,0))
        self.assertEqual(len(self.db.list_model_calls(self.task)),1)

    def test_conversation_new_task_policy_is_frozen(self):
        """会话入口也冻结预算，幂等提交不因设置变化回填。"""
        from research_loop.services.conversations import ConversationService
        cid=ConversationService(self.db).create('预算会话',str(uuid4()))['conversation_id']
        payload={'conversation_id':cid,'prompt':'这次回答，不保存','idempotency_key':str(uuid4())}
        first=self.client.post('/api/tasks',json=payload).json()
        changed=self.settings.model_copy(update={'task_token_limit':2000000})
        with patch('research_loop.api.dependencies.get_settings',return_value=changed):second=self.client.post('/api/tasks',json=payload).json()
        self.assertEqual(first['task_id'],second['task_id'])
        self.assertEqual(second['usage']['budget']['tokens']['token_limit'],1000000)

    def test_reading_list_add_read_remove_replay_and_isolation(self):
        """加入去重、已读不改任务、移除重放不复活；清单不泄露路径、不生成文件。"""
        pid=self.seed_paper()
        add={'paper_id':pid,'task_id':self.task,'action':'add','expected_version':0,'idempotency_key':str(uuid4())}
        row=self.client.post('/api/reading-list',json=add).json()
        self.assertEqual(row['status'],'unread');self.assertNotIn('local_path',row['paper'])
        same=self.client.post('/api/reading-list',json={**add,'idempotency_key':str(uuid4())}).json()
        self.assertEqual(same['version'],1)
        read={**add,'task_id':None,'action':'read','expected_version':1,'idempotency_key':str(uuid4())}
        self.assertEqual(self.client.post('/api/reading-list',json=read).json()['status'],'read')
        remove={**read,'action':'remove','expected_version':2,'idempotency_key':str(uuid4())}
        self.assertEqual(self.client.post('/api/reading-list',json=remove).json()['status'],'removed')
        self.assertEqual(self.client.post('/api/reading-list',json=add).json()['status'],'removed')
        stale={**add,'idempotency_key':str(uuid4())}
        self.assertEqual(self.client.post('/api/reading-list',json=stale).status_code,409)
        self.assertEqual(self.client.post('/api/reading-list',json={**stale,'expected_version':3}).json()['version'],4)
        self.assertEqual(self.db.get_task(self.task)['status'],'queued')
        self.assertEqual(self.db.list_model_calls(self.task),[])
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))

    def test_reading_list_rejects_wrong_source_and_conflicting_key(self):
        """全库存在论文不代表任意任务可引用它；相同键不可换操作。"""
        pid=self.seed_paper();other=self.create()
        args={'paper_id':pid,'task_id':other,'action':'add','expected_version':0,'idempotency_key':str(uuid4())}
        self.assertEqual(self.client.post('/api/reading-list',json=args).status_code,404)
        args['task_id']=self.task
        self.assertEqual(self.client.post('/api/reading-list',json=args).status_code,200)
        self.assertEqual(self.client.post('/api/reading-list',json={**args,'action':'remove','expected_version':1}).status_code,409)

    def test_parallel_reading_add_is_single_row(self):
        """多个窗口同时加入同一论文，只保留一份当前记录。"""
        from research_loop.api.reading_list import ReadingChange
        from research_loop.services.reading_list import ReadingListService
        pid=self.seed_paper()
        def add(_):
            return ReadingListService(self.db).change(ReadingChange(paper_id=pid,task_id=self.task,action='add',expected_version=0,idempotency_key=str(uuid4())))
        with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(add,range(2)))
        self.assertEqual([r['version'] for r in rows],[1,1])
        self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM reading_list WHERE paper_id=?',(pid,))['n'],1)

    def test_deep_reading_wrapper_uses_same_root_policy(self):
        """DPR原函数不改，外层物理调用共享根预算并复用成功收据。"""
        from research_loop.agents.deep_reading import _ReceiptClient
        calls=[]
        class Provider:
            def send(self,*args,**kwargs):
                calls.append(kwargs)
                return {'id':'dpr-stub','usage':{'total_tokens':91},'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'完成（完）'}}]}
        state={'task_id':self.task,'root_task_id':self.task};stage={'base_sequence':0}
        for _ in range(2):
            client=_ReceiptClient(state,stage,Provider(),self.db,lambda _:None)
            client.kwargs.update(max_tokens=4096,temperature=.2)
            self.assertIn('（完）',client.chat([{'role':'user','content':'原文'}])['content'])
        self.assertEqual(len(calls),1)
        self.assertEqual(TokenBudgetStore(self.db).summary(self.task)['used_tokens'],91)
