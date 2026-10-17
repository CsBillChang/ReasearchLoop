"""AG7.5暴露的同会话附件接续合同；真实API/Worker，供应商替身，零模型费用。"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import pymupdf
from research_loop.config import Settings
from research_loop.main import app
from research_loop.storage.mysql import Database
from research_loop.services.conversations import ConversationService
from research_loop.services.tasks import TaskService
from research_loop.services.artifacts import ArtifactService
from research_loop.domain.tasks import TaskCreate, TaskStatus, TaskResult
from research_loop.workers.executor import TaskExecutor


class AttachmentProvider:
    """仅替代模型的固定三步规划、真实读附件、交付；保留实际消息用于断言。"""

    def __init__(self, reference):
        """记录本例附件身份，不能从历史页证据绕过重新读取。"""
        self.reference=reference;self.messages=[]

    def close(self):
        """替身没有网络连接。"""

    def send(self,messages,tools,**kwargs):
        """返回确定性工具调用；实际读取、持久化和权限校验仍走生产实现。"""
        self.messages.append(copy.deepcopy(messages))
        calls=[c['function']['name'] for m in messages for c in m.get('tool_calls',[])]
        if 'set_research_plan' not in calls:
            name='set_research_plan';args={'goal':'读取同一会话附件回答','task_kind':'survey','requested_count':1,'count_confirmed':True,'save_requested':False,'save_policy':'deny','input_refs':[self.reference],'goals':[{'goal_id':'read','description':'读取原文','min_papers':1,'requires_fulltext':True,'input_refs':[self.reference]}]}
        elif 'read_paper' not in calls:
            name='read_paper';args={'target_ref':self.reference}
        else:
            name='finish_task';args={'answer_markdown':'已通过当前任务 read_paper 重新读取会话附件；本轮不保存文件。','goals':[{'goal_id':'read','status':'satisfied','reason':'实际读取成功'}]}
        return {'id':'stub-'+str(len(self.messages)),'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15},'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':'','tool_calls':[{'id':'call-'+str(len(self.messages)),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}]}


class ConversationAttachmentTests(unittest.TestCase):
    """附件绑定不是旧证据/保存授权继承，且保持版本化幂等。"""

    def setUp(self):
        """每例新会话与文件；可显式接入独立 MySQL 验证事务路径。"""
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        url=os.getenv('AG75_TEST_DATABASE_URL') or 'sqlite:///'+str(self.root/'db')
        if not url.startswith('sqlite:') and not url.rsplit('/',1)[-1].startswith('researchloop_test_ag75_zero_'):
            raise ValueError('AG75 isolated MySQL required')
        self.settings=Settings(project_root=self.root,data_dir=self.root/'data',output_dir=self.root/'notes',database_url=url,max_model_requests=4,reserve_model_requests=0,retrieval_enabled=False)
        self.db=Database(url);self.db.ensure_schema();self.conv=ConversationService(self.db)
        self.cid=self.conv.create('附件交付测试',uuid4().hex)['conversation_id'];self.service=TaskService(self.db)
        self.pdf=self.root/'sample.pdf'
        with pymupdf.open() as doc:
            page=doc.new_page()
            page.insert_textbox((40,40,550,790),'\n'.join(['Original source for follow-up: visual prompts are transferable.']*32),fontsize=10)
            doc.save(self.pdf)
        self.attachment={'input_ref':'fixture-'+uuid4().hex,'local_path':str(self.pdf)}

    def submit(self,attachments=None,policy='deny',cid=None,key=None):
        """创建测试消息，任务授权只来自本次请求。"""
        return self.service.create(TaskCreate(conversation_id=cid or self.cid,prompt='继续读同一论文',attachments=attachments or [],save_policy=policy,idempotency_key=key or uuid4().hex)).task_id

    def finish(self,tid):
        """测试前置任务仅设合成终态；不伪称做过真实模型研究。"""
        self.service.complete(tid,TaskStatus.completed,TaskResult(answer_markdown='合成前置结果'))

    def test_followup_snapshot_permissions(self):
        """接续材料但不复制 allow、页证据或模型收据；旧任务不变。"""
        first=self.submit([self.attachment],policy='allow');self.finish(first)
        second=self.submit();row=self.db.get_task(second)
        self.assertEqual(json.loads(row['attachments_json'])[0]['input_ref'],self.attachment['input_ref'])
        self.assertEqual(row['save_policy'],'deny')
        self.assertTrue(self.db.task_has_input(second,self.attachment['input_ref']))
        self.assertEqual(self.db.fetchall('SELECT * FROM task_evidence WHERE task_id=?',(second,)),[])
        self.assertEqual(self.db.list_model_calls(second),[])
        self.assertTrue(any(e['event_type']=='conversation_attachments_bound' for e in self.db.list_events(second)))

    def test_current_attachments_replace_previous_set(self):
        """当前明确上传另一份材料时，不把旧材料混入本轮。"""
        first=self.submit([self.attachment]);self.finish(first)
        new={**self.attachment,'input_ref':'new'};second=self.submit([new])
        self.assertEqual([a['input_ref'] for a in json.loads(self.db.get_task(second)['attachments_json'])],['new'])
        self.assertFalse(self.db.task_has_input(second,self.attachment['input_ref']))

    def test_no_cross_conversation(self):
        """新会话不会因相同文案取得其他会话附件。"""
        first=self.submit([self.attachment]);self.finish(first)
        cid=self.conv.create('新会话',uuid4().hex)['conversation_id']
        second=self.submit(cid=cid)
        self.assertEqual(json.loads(self.db.get_task(second)['attachments_json']),[])

    def test_idempotent_snapshot_survives_later_upload(self):
        """重复提交沿用当时快照，之后的新附件不改变旧任务。"""
        first=self.submit([self.attachment]);self.finish(first)
        key=uuid4().hex;second=self.submit(key=key);self.finish(second)
        third=self.submit([{**self.attachment,'input_ref':'later'}]);self.finish(third)
        self.assertEqual(self.submit(key=key),second)
        self.assertEqual(json.loads(self.db.get_task(second)['attachments_json'])[0]['input_ref'],self.attachment['input_ref'])

    def test_api_worker_actual_read(self):
        """本批真实失败的附件为空问题，在API→DB→Worker messages→PDF链中回归。"""
        first=self.submit([self.attachment]);self.finish(first)
        fake=AttachmentProvider(self.attachment['input_ref'])
        with patch('research_loop.api.dependencies.get_settings',return_value=self.settings),TestClient(app) as client:
            response=client.post('/api/tasks',json={'conversation_id':self.cid,'prompt':'沿用这篇原文，继续回答，不保存','save_policy':'deny','idempotency_key':uuid4().hex})
            self.assertEqual(response.status_code,202);tid=response.json()['task_id']
            TaskExecutor(self.settings,self.db,provider_factory=lambda _:fake).execute(tid,'ag75-zero-worker')
            view=client.get('/api/tasks/'+tid).json()
        self.assertIn(self.attachment['input_ref'],json.dumps(fake.messages[0],ensure_ascii=False))
        reads=[a for a in self.db.list_actions(tid) if a['tool_name']=='read_paper']
        self.assertEqual(len(reads),1);self.assertTrue(reads[0]['observation']['ok'],reads[0]['observation'])
        self.assertTrue(self.db.fetchall('SELECT * FROM task_evidence WHERE task_id=?',(tid,)))
        self.assertEqual(view['status'],'completed',view)
        self.assertEqual(list(self.settings.output_dir.rglob('*.md')) if self.settings.output_dir.exists() else [],[])
        # 上传PDF没有发现候选元数据时，网页仍须展示本任务已经读取的证据页。
        product=ArtifactService(self.db,self.settings)
        artifacts=product.artifacts(tid)
        self.assertEqual(len(artifacts['papers']),1)
        self.assertEqual(artifacts['papers'][0]['paper_id'],self.attachment['input_ref'])
        eid=artifacts['evidence'][0]['evidence_id']
        self.assertTrue(product.evidence(tid,eid)['pdf_available'])
        self.assertTrue(product.pdf(tid,eid).is_file())
        self.assertEqual(product.artifacts(first)['evidence'],[])
        with self.assertRaises(KeyError):product.evidence(first,eid)

    def test_deleted_source_is_not_fabricated(self):
        """附件路径消失时保留定位以报告真实文件缺失，不生成替代原文。"""
        first=self.submit([self.attachment]);self.finish(first);self.pdf.unlink()
        second=self.submit()
        row=self.db.get_task(second)
        self.assertEqual(json.loads(row['attachments_json'])[0]['local_path'],str(self.pdf))
        self.assertFalse(Path(json.loads(row['attachments_json'])[0]['local_path']).exists())
