"""API→数据库→Worker→检索→模型messages合同；模型/向量为明确替身。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlparse
import uuid

from fastapi.testclient import TestClient

from research_loop.config import Settings
from research_loop.main import app
from research_loop.services.retrieval import RetrievalService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from tests.unit.test_hybrid_retrieval import QueryEmbedding, QueryVector, TestReranker
from tests.unit.test_retrieval_contracts import seed_document


class RetrievalAgentTests(unittest.TestCase):
    """可在SQLite或显式隔离MySQL运行，禁止隐式连接生产库。"""
    def setUp(self):
        """每例创建独立任务/文档；模型发送计数仅为供应商替身次数。"""
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        url=os.getenv('AG72_INTEGRATION_DATABASE_URL') or 'sqlite:///'+str(self.root/'db')
        if url.startswith('mysql') and not urlparse(url).path.startswith('/researchloop_test_ag72_'):
            raise ValueError('isolated AG72 database required')
        self.db=Database(url); self.db.ensure_schema()
        self.config=Settings(project_root=self.root,data_dir=self.root/'data',output_dir=self.root/'notes',database_url=url)
        self.client=TestClient(app)
        self.patch=patch('research_loop.api.dependencies.get_settings',return_value=self.config)
        self.patch.start(); self.addCleanup(self.patch.stop); self.addCleanup(self.client.close)
        self.key='ag72-contract-'+uuid.uuid4().hex
        self.payload={'prompt':'基于已授权原文定位language证据并简短回答，不保存。','save_policy':'deny','idempotency_key':self.key}
        response=self.client.post('/api/tasks',json=self.payload)
        self.assertEqual(response.status_code,202)
        self.task=response.json()['task_id']
        self.doc,self.pages=seed_document(self.db,self.root,key=self.key,text='Language features identify objects.')
        for page in self.pages: self.db.grant_evidence(self.task,page['evidence_id'],'test-fixture')
        self.service=RetrievalService(self.db,QueryEmbedding(),QueryVector(),TestReranker())
        self.sent=[]
        owner=self

        class Provider:
            """按模型看到的消息返回工具调用，禁用网络。"""
            def send(self,messages,tools=None,force_text=False,**kwargs):
                """第二次必须真正收到检索观察，才返回固定回答。"""
                owner.sent.append({'messages':messages,'tools':tools})
                observed=[json.loads(m['content']) for m in messages if m.get('role')=='tool' and 'retrieved_chunks_only' in m.get('content','')]
                if not observed:
                    message={'content':None,'tool_calls':[{'id':'retrieve-'+owner.key,'type':'function',
                        'function':{'name':'retrieve_evidence','arguments':json.dumps({'query':'Language','document_ids':[owner.doc['document_id']]})}}]}
                    finish='tool_calls'
                else:
                    owner.assertEqual(observed[0]['data']['factual_support'],'not_assessed')
                    owner.assertIn('Language',observed[0]['data']['hits'][0]['text'])
                    message={'content':'原文片段说明语言特征用于识别物体。'}; finish='stop'
                return {'id':'synthetic-'+str(len(owner.sent)), 'choices':[{'message':message,'finish_reason':finish}],
                    'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}

            def close(self):
                """没有网络资源。"""
                pass

        self.executor=TaskExecutor(self.config,self.db,provider_factory=lambda _:Provider())
        self.executor.retrieval=self.service

    def test_actual_api_worker_messages_and_idempotency(self):
        """实际业务执行后读回模型请求与精确工具观察，不靠mock断言声称已接线。"""
        self.executor.execute(self.task,'ag72-test')
        actions=self.db.list_actions(self.task)
        hits=[a for a in actions if a['tool_name']=='retrieve_evidence']
        self.assertEqual(len(hits),1)
        self.assertEqual(hits[0]['status'],'succeeded')
        calls=self.db.list_model_calls(self.task)
        self.assertEqual(len(calls),2)
        self.assertIn('retrieved_chunks_only',calls[1]['request_json'])
        self.assertIn(hits[0]['observation']['data']['hits'][0]['chunk_id'],calls[1]['request_json'])
        self.assertEqual(self.client.post('/api/tasks',json=self.payload).json()['task_id'],self.task)
        self.assertEqual(len(self.sent),2)
        self.assertFalse(list(self.config.output_dir.rglob('*.md')))

    def test_success_receipt_recovery_does_not_resend(self):
        """在成功收据落盘后模拟进程退出；恢复复用收据并只发后续请求。"""
        original=self.db.save_model_call
        stopped=False
        def fail_after_receipt(*args,**kwargs):
            """故障只在第一份成功收据实际提交之后发生。"""
            nonlocal stopped
            result=original(*args,**kwargs)
            if not stopped and args[2]=='succeeded':
                stopped=True
                raise SystemExit('controlled exit after receipt')
            return result
        with patch.object(self.db,'save_model_call',side_effect=fail_after_receipt):
            with self.assertRaises(SystemExit): self.executor.execute(self.task,'ag72-recovery')
        self.assertEqual(len(self.sent),1)
        receipt=self.db.get_model_call(self.task,1)
        self.assertEqual(receipt['status'],'succeeded')
        self.executor.execute(self.task,'ag72-recovery')
        self.assertEqual(len(self.sent),2)
        self.assertEqual(self.db.get_model_call(self.task,1)['request_id'],receipt['request_id'])
        self.assertEqual(len([a for a in self.db.list_actions(self.task) if a['tool_name']=='retrieve_evidence']),1)

    def test_cancelled_task_never_loads_or_calls_retrieval(self):
        """取消在执行前生效，不因为可用索引而继续模型或工具调用。"""
        response=self.client.post('/api/tasks/'+self.task+'/cancel')
        self.assertEqual(response.status_code,200)
        self.executor.execute(self.task,'ag72-cancel')
        self.assertEqual(self.sent,[])
        self.assertEqual(self.db.list_actions(self.task),[])

    def test_reviewer_graph_receives_chunks_without_write_tools(self):
        """Reviewer独立图实际执行检索，观察进入模型消息，仍不开放搜索/保存。"""
        from research_loop.agents.reviewer import build_reviewer_graph, reviewer_state
        provider=self.executor.provider_factory(self.config)
        state=reviewer_state(self.task,'synthetic-draft','a'*64,'需核对的合成陈述',[],[])
        graph=build_reviewer_graph(provider,{'retrieval':self.service},self.db)
        result=graph.invoke(state)
        self.assertTrue(result['finished'])
        self.assertEqual(len(self.sent),2)
        names={s['function']['name'] for s in self.sent[0]['tools']}
        self.assertIn('retrieve_evidence',names)
        self.assertTrue(names.isdisjoint({'save_note','create_note','search_papers','delegate_research'}))
        self.assertIn('retrieved_chunks_only',self.db.get_model_call(self.task,2)['request_json'])
