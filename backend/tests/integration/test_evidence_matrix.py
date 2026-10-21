"""矩阵API到Worker集成；供应商和论文内容明确合成，支持专用MySQL库。"""
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.domain.research import EvidenceMatrixInput
from research_loop.domain.tasks import ActionRecord, Observation
from research_loop.services.evidence_matrix import EvidenceMatrixService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from backend.tests.unit.test_evidence_matrix import seed, persist


class MatrixProvider:
    """依据真实工具观察生成固定调用，不使用外部模型。"""

    def __init__(self, args, counter):
        """保存确定的合成材料及跨进程调用计数文件。"""
        self.args,self.counter=args,Path(counter)

    def send(self,messages,tools,**kwargs):
        """替身保留生产Provider形状，原始响应与usage仍走真实持久层。"""
        with self.counter.open('a') as f:f.write('send\n')
        observations=[json.loads(m['content']) for m in messages if m.get('role')=='tool']
        matrix=next((o['data']['evidence_matrix'] for o in reversed(observations) if 'evidence_matrix' in o.get('data',{})),None)
        if matrix:
            calls=[('finish_task',{'answer_markdown':'已整理矩阵；硬件细节仅在已读范围缺证，不直接跨条件排名。',
                'matrix_hash':matrix['matrix_hash'],'goals':[{'goal_id':'compare','status':'satisfied','reason':'已整理固定材料'}]})]
        elif any('research_plan' in o.get('data',{}) for o in observations):
            calls=[('record_evidence_matrix',self.args)]
        else:
            calls=[('set_research_plan',{'goal':'比较已读材料','task_kind':'compare','requested_count':1,'count_confirmed':True,'evidence_matrix_requested':True,
                'goals':[{'goal_id':'compare','description':'整理已有材料','min_papers':1,'requires_fulltext':True}]}),
                ('read_paper',{'target_ref':'fixture-input'})]
        return {'id':'matrix-stub','usage':{'total_tokens':40},'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
            'tool_calls':[{'id':'call-'+str(i)+'-'+name,'type':'function','function':{'name':name,'arguments':json.dumps(a)}} for i,(name,a) in enumerate(calls)]}}]}

    def close(self):
        """合成Provider无外部连接。"""


def fixture_read(db,args):
    """返回已建合成页；真实验证不能使用这个替身。"""
    row=args['rows'][0];doc=db.fetchone('SELECT * FROM documents WHERE document_id=?',(row['document_id'],))
    return {'paper_id':doc['paper_id'],'document_id':doc['document_id'],
            'content_sha256':doc['content_sha256'],'total_pages':doc['total_pages'] or None,
            'pages':[db.get_page(e) for e in row['inspected_evidence_ids']]}


class MatrixIntegrationTests(unittest.TestCase):
    """生产入口相同、数据与输出根目录隔离，不修改真实研究。"""

    def setUp(self):
        """仅接受researchloop_test_matrix_*或临时SQLite。"""
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        url=os.getenv('MATRIX_TEST_DATABASE_URL') or 'sqlite:///'+str(self.root/'test.db')
        if not url.startswith('sqlite:') and not url.rsplit('/',1)[-1].startswith('researchloop_test_matrix_'):
            raise ValueError('isolated matrix database required')
        self.settings=Settings(project_root=self.root,data_dir=self.root/'data',output_dir=self.root/'notes',database_url=url,
                               deepseek_api_key='STUB',retrieval_enabled=False)
        self.db=Database(url);self.db.ensure_schema()
        from research_loop.main import app
        p=patch('research_loop.api.dependencies.get_settings',return_value=self.settings);p.start();self.addCleanup(p.stop)
        self.client=TestClient(app);self.addCleanup(self.client.close)
        self.tid=self.client.post('/api/tasks',json={'prompt':'对固定材料整理矩阵，不保存文件','idempotency_key':str(uuid4())}).json()['task_id']
        self.args=seed(self.db,self.tid);self.service=EvidenceMatrixService(self.db)
        self.counter=self.root/'calls.txt'

    def execute(self):
        """零网络执行真实调度，只有已声明的原文替身和模型替身。"""
        with patch('httpx.Client.send',side_effect=AssertionError('network forbidden')),patch('research_loop.services.papers.PaperService.read',return_value=fixture_read(self.db,self.args)):
            return TaskExecutor(self.settings,self.db,lambda _:MatrixProvider(self.args,self.counter)).execute(self.tid)

    def test_api_worker_matrix_and_delivery(self):
        """矩阵进入下一次messages、最终结果和产物API，禁止保存仍无Markdown。"""
        result=self.execute();self.assertTrue(result.evidence_matrix,result)
        view=self.client.get('/api/tasks/'+self.tid).json();self.assertEqual(view['status'],'completed',view)
        matrices=self.client.get('/api/tasks/'+self.tid+'/artifacts').json()['evidence_matrices']
        self.assertEqual(matrices[0]['matrix_hash'],result.evidence_matrix['matrix_hash'])
        calls=self.db.list_model_calls(self.tid);self.assertEqual(len(calls),3)
        self.assertIn(matrices[0]['matrix_hash'],calls[-1]['request_json'])
        self.assertNotIn('thinking',json.loads(calls[0]['request_json']))
        for call in calls[1:]:
            self.assertEqual(json.loads(call['request_json'])['thinking'],{'type':'disabled'})
        self.assertEqual(len(self.counter.read_text().splitlines()),3)
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
        self.assertEqual(self.client.get('/api/tasks/'+self.tid+'/artifacts').json()['evidence_matrices'],matrices)

    def test_atomic_concurrent_change_and_immutable_success(self):
        """同前版两个修改只能提交一个；已经成功的行动不能覆盖。"""
        task=self.db.claim_task('w',task_id=self.tid);owner=(self.tid,'w',task['lease_token'])
        original=persist(self.db,self.tid,self.args,owner=owner)
        actions=[]
        for seq in (2,3):
            args=copy.deepcopy(self.args);args.update(base_matrix_hash=original['matrix_hash'],question='revision '+str(seq))
            matrix=self.service.prepare(self.tid,EvidenceMatrixInput(**args))
            action=ActionRecord(task_id=self.tid,sequence=seq,tool_name='record_evidence_matrix',arguments=args)
            self.db.save_action(action.model_dump(mode='json'));action.status='succeeded'
            action.observation=Observation(ok=True,kind='success',data={'evidence_matrix':matrix}).model_dump()
            actions.append(action.model_dump(mode='json'))
        def commit(action):
            try:self.service.commit_action(action,owner);return True
            except ValueError:return False
        with ThreadPoolExecutor(max_workers=2) as pool:self.assertEqual(sum(pool.map(commit,actions)),1)
        self.assertEqual(len(self.service.history(self.tid)),2)
        winner=next(a for a in actions if self.db.get_action(self.tid,a['sequence'])['status']=='succeeded')
        self.service.commit_action(winner,owner)
        tampered=copy.deepcopy(winner);tampered['observation']['data']['evidence_matrix']['matrix_hash']='a'*64
        with self.assertRaises(ValueError):self.service.commit_action(tampered,owner)

    def test_streaming_worker_parameters_reach_provider(self):
        """网页实际使用的流式Worker也传递局部参数，原始响应正常入库。"""
        # 与网页相同，只有会话任务才启用ContextService及真实流入口。
        cid = self.client.post('/api/conversations',json={'title':'矩阵替身','idempotency_key':str(uuid4())}).json()['conversation_id']
        self.tid = self.client.post('/api/tasks',json={'prompt':'整理矩阵','idempotency_key':str(uuid4()),
                                                       'conversation_id':cid}).json()['task_id']
        self.args = seed(self.db,self.tid)
        seen = []

        class StreamProvider(MatrixProvider):
            """仅供应商为替身；记录收到的请求参数。"""

            def send_stream(self, messages, tools, *, on_public, **kwargs):
                """复用合成响应，避免把模拟结果宣称为真实流验证。"""
                seen.append(kwargs.copy())
                return self.send(messages, tools, **kwargs)

        with patch('httpx.Client.send',side_effect=AssertionError('network forbidden')),patch(
                'research_loop.services.papers.PaperService.read',return_value=fixture_read(self.db,self.args)):
            result = TaskExecutor(self.settings,self.db,lambda _:StreamProvider(self.args,self.counter)).execute(self.tid)
        self.assertTrue(result.evidence_matrix)
        self.assertEqual(len(seen),3)
        self.assertNotIn('thinking',seen[0])
        self.assertTrue(all(v['thinking']=={'type':'disabled'} for v in seen[1:]))

    def test_frozen_scope_is_visible_in_worker_receipts(self):
        """走真实API/Worker链，冻结禁止的工具从第一次messages起就不可见。"""
        allowed={'set_research_plan','read_paper','record_evidence_matrix','finish_task','report_task_gap'}
        with patch('httpx.Client.send',side_effect=AssertionError('network forbidden')),patch(
                'research_loop.services.papers.PaperService.read',return_value=fixture_read(self.db,self.args)):
            result=TaskExecutor(self.settings,self.db,lambda _:MatrixProvider(self.args,self.counter),
                                supervisor_tools=allowed).execute(self.tid)
        self.assertTrue(result.evidence_matrix)
        for call in self.db.list_model_calls(self.tid):
            names={s['function']['name'] for s in json.loads(call['request_json'])['tools']}
            self.assertEqual(names,allowed)

    def test_cancellation_and_lease_rejected_after_prepare(self):
        """已计算的产物不得绕过取消和失权，之后仍可按正确租约提交。"""
        task=self.db.claim_task('owner',task_id=self.tid)
        with self.assertRaises(PermissionError):persist(self.db,self.tid,self.args,owner=(self.tid,'owner','stolen'))
        self.db.update_task(self.tid,cancel_requested=1)
        with self.assertRaises(PermissionError):persist(self.db,self.tid,self.args,owner=(self.tid,'owner',task['lease_token']))
        self.assertFalse(self.service.history(self.tid))

    def test_foreign_task_and_legacy_empty_projection(self):
        """矩阵不跨任务泄露，旧任务无需回填。"""
        other=self.client.post('/api/tasks',json={'prompt':'普通问答','idempotency_key':str(uuid4())}).json()['task_id']
        self.assertEqual(self.client.get('/api/tasks/'+other+'/artifacts').json()['evidence_matrices'],[])
        with self.assertRaises(ValueError):self.service.prepare(other,EvidenceMatrixInput(**self.args))
        self.assertIsNone(self.client.get('/api/tasks/'+other).json()['result'])

    def test_sigkill_after_matrix_commit_reuses_receipts(self):
        """真实进程在矩阵提交后退出；恢复重放成功观察，只调用尚未发送的收尾。"""
        config=self.root/'config.json';config.write_text(json.dumps({'settings':self.settings.model_dump(mode='json'),'tid':self.tid,'args':self.args,'counter':str(self.counter)}))
        code='''
import json,os,signal,sys
from unittest.mock import patch
from research_loop.config import Settings
from research_loop.storage.mysql import Database
from research_loop.services.evidence_matrix import EvidenceMatrixService
from research_loop.workers.executor import TaskExecutor
from backend.tests.integration.test_evidence_matrix import MatrixProvider,fixture_read
v=json.load(open(sys.argv[1]));settings=Settings(**v['settings']);db=Database(settings.database_url)
original=EvidenceMatrixService.commit_action
def terminate(self,action,owner):
 original(self,action,owner);os.kill(os.getpid(),signal.SIGKILL)
with patch('research_loop.services.papers.PaperService.read',return_value=fixture_read(db,v['args'])),patch.object(EvidenceMatrixService,'commit_action',terminate),patch('httpx.Client.send',side_effect=AssertionError('network forbidden')):
 TaskExecutor(settings,db,lambda _:MatrixProvider(v['args'],v['counter'])).execute(v['tid'])
'''
        proc=subprocess.run([sys.executable,'-c',code,str(config)],capture_output=True,text=True,timeout=30)
        self.assertEqual(proc.returncode,-signal.SIGKILL,proc.stderr)
        self.assertEqual(len(self.counter.read_text().splitlines()),2)
        before=self.service.latest(self.tid)['matrix_hash']
        self.db.execute('UPDATE tasks SET lease_until=? WHERE task_id=?',('2000-01-01',self.tid))
        result=self.execute()
        self.assertEqual(result.evidence_matrix['matrix_hash'],before)
        self.assertEqual(len(self.counter.read_text().splitlines()),3)
        self.assertEqual(len(self.db.list_model_calls(self.tid)),3)
        self.assertEqual(len(self.service.history(self.tid)),1)
