# 协作链路测试：覆盖持久父子任务与交付边界；依赖环境门控的用例须单独记录跳过。
"""API -> actual role graphs/services/storage, synthetic model/PDF boundaries only."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
import httpx
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.domain.evidence import Document, PageEvidence
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor

class FixturePDF:
    def ingest(self, source, paper_id):
        return Document(document_id='doc-'+paper_id,paper_id=paper_id,content_sha256=hashlib.sha256(paper_id.encode()).hexdigest(),parser_name='fixture',parser_version='1',config_sha256='a'*64,total_pages=2,cache_path=source)
    def pages(self, document, start, count):
        return [PageEvidence(evidence_id=f'evidence:{document.paper_id}:{n}',document_id=document.document_id,page_number=n,text=f'{document.paper_id}: language prefix embeddings, physical page {n}',parsed_markdown=f'{document.paper_id}: language prefix embeddings') for n in range(start,min(start+count,3))]
    def close(self): pass

class Scenario:
    def __init__(self, verdict='supported'):
        self.verdict=verdict;self.requests=[]
    def response(self, body, name=None, args=None, content=None):
        m={'role':'assistant','content':content or ''}
        if name:m['tool_calls']=[{'id':str(uuid4()),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]
        return httpx.Response(200,json={'id':str(uuid4()),'choices':[{'finish_reason':'tool_calls' if name else 'stop','message':m}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
    def handler(self, request):
        body=json.loads(request.content);self.requests.append(body);messages=body['messages'];validate_messages(messages)
        if body['messages'][0]['content'].startswith('你是一名资深学术论文分析助手'):
            assert body['max_tokens']==4096 and body['temperature']==0.3
            return self.response(body, content='## DPR 精读（供应商替身）\n语言前缀方法来自提供的正文。\n（完）')
        names={x['function']['name'] for x in body.get('tools',[])}
        assert 'get_citation_stats' not in names
        obs=[json.loads(m['content']) for m in messages if m['role']=='tool']
        if 'finish_research' in names:
            brief=json.loads(messages[2]['content'].split('\n',1)[1]);ref=brief['input_refs'][0]
            assert brief['page_count']==1
            if not obs:return self.response(body,'read_paper',{'target_ref':ref,'start_page':1,'page_count':1})
            pages=obs[-1]['data']['pages']
            return self.response(body,'finish_research',{'findings':[f'{ref} uses language prefix embeddings on page 1.'],'evidence_ids':[p['evidence_id'] for p in pages],'paper_ids':[],'gaps':[]})
        if 'finish_review' in names:
            bundle=json.loads(messages[-1]['content'].split('\n',1)[1])
            assert len(bundle)==2 and all(p['text'] and p['target_ref'] in ['input-A','input-B'] for p in bundle)
            return self.response(body,'finish_review',{'status':self.verdict,'findings':[{'claim':'method','status':'supported' if self.verdict=='supported' else 'insufficient','basis':'direct_statement' if self.verdict=='supported' else 'incomplete_or_ambiguous','source_quote':bundle[0]['text'],'reason':'synthetic verification','evidence_ids':[bundle[0]['evidence_id']]}],'gaps':[]})
        done=[m['function']['name'] for message in messages for m in message.get('tool_calls',[])]
        if not done:return self.response(body,'request_clarification',{'question':'几篇？','fields':['count'],'suggested_values':['5']})
        if 'report_task_gap' not in done:return self.response(body,'report_task_gap',{'requirement':'高被引','reason':'没有被引量工具，无法核实'})
        if 'delegate_research' not in done:
            # 此夹具专测历史精读/可选审校恢复，显式选择精读，不依赖新任务默认模式。
            return self.response(body,'delegate_research',{'subtasks':[{'objective':f'read {ref}','mode':'deep_read','input_refs':[ref],'start_page':1,'page_count':1,'comparison_dimensions':['language use']} for ref in ['input-A','input-B']]})
        if 'create_note' not in done:
            ids=[e for c in obs[-1]['data']['subtasks'] for e in c['evidence_ids']]
            return self.response(body,'create_note',{'title':'comparison','markdown':'# Compare\nBoth papers discuss language prefix embeddings.\n高被引要求未完成。','evidence_ids':ids})
        draft=next(o['data'] for o in obs if o.get('message','').startswith('draft persisted'))
        if 'review_note' not in done:return self.response(body,'review_note',{'draft_id':draft['draft_id'],'content_hash':draft['content_hash']})
        if 'save_note' not in done:return self.response(body,'save_note',{'draft_id':draft['draft_id'],'content_hash':draft['content_hash']})
        return self.response(body,content='比较笔记已保存；高被引要求未完成，没有被引量工具。')
    def factory(self,settings):
        return DeepSeekProvider('fixture',client=httpx.Client(transport=httpx.MockTransport(self.handler)))

class CollaborationContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        url=os.getenv('AG2_COLLAB_TEST_DATABASE_URL') or 'sqlite:///'+str(self.root/'researchloop_test_collaboration.db')
        if not url.startswith('sqlite:') and 'researchloop_test_' not in url.rsplit('/',1)[-1]:raise ValueError('isolated DB required')
        self.settings=Settings(project_root=self.root,data_dir=self.root/'data',output_dir=self.root/'notes',database_url=url)
        self.db=Database(url);self.db.ensure_schema()
        for name in ['A','B']:(self.root/(name+'.pdf')).write_bytes(b'%PDF-fixture')
    def create(self,client):
        r=client.post('/api/tasks',json={'prompt':'比较高被引论文并保存','save_policy':'allow','idempotency_key':str(uuid4()),'attachments':[{'input_ref':'input-'+x,'local_path':str(self.root/(x+'.pdf')),'paper_id':'paper-'+x} for x in ['A','B']]})
        self.assertEqual(r.status_code,202,r.text);return r.json()['task_id']
    def run_scenario(self,verdict):
        from research_loop.main import app
        scenario=Scenario(verdict);executor=TaskExecutor(self.settings,self.db,scenario.factory,lambda s,f:FixturePDF())
        with patch('research_loop.api.tasks.get_settings',return_value=self.settings),TestClient(app) as client:
            tid=self.create(client);executor.execute(tid)
            q=client.get(f'/api/tasks/{tid}').json()['clarification']
            res=client.post(f'/api/tasks/{tid}/resume',json={'clarification_id':q['clarification_id'],'answer':'2篇','idempotency_key':str(uuid4())});self.assertEqual(res.status_code,200,res.text)
            executor.execute(tid,'resumed')
            res=client.get(f'/api/tasks/{tid}');self.assertEqual(res.status_code,200,res.text);view=res.json()
        self.assertEqual(view['status'],'partial') # explicit unavailable citation requirement
        self.assertTrue(view['result']['saved_path'])
        children=self.db.list_subtasks(tid);self.assertEqual(len(children),3)
        review=self.db.get_review_result(tid,view['result']['draft_id'],self.db.get_draft(view['result']['draft_id'])['content_hash'])
        self.assertEqual(review['status'],verdict);self.assertEqual(len(review['findings']),1)
        self.assertEqual(len([r for r in scenario.requests if 'finish_review' in {t['function']['name'] for t in r.get('tools',[])}]),1)
        self.assertEqual(view['usage']['requests'],len(scenario.requests))
        receipt=self.db.get_receipt(view['result']['draft_id'],review['content_hash'])
        self.assertEqual(receipt['status'],'saved')  # 核验意见独立记录，不再改写导出内容或阻断保存。
        for child in children:
            self.assertIsNotNone(self.db.get_checkpoint(child['subtask_id']))
            if child['position_no']!=999:
                cp=self.db.get_checkpoint(child['subtask_id']);self.assertEqual(cp['final_reason'],'structured role result')
                self.assertEqual(len(cp['attachments']),1)
                self.assertEqual(cp['deep_reading']['status'],'complete')
        return tid,scenario,executor
    def test_supported_api_worker_handoff_save(self):self.run_scenario('supported')
    def test_insufficient_keeps_findings_and_api_status(self):self.run_scenario('insufficient')
    def test_issues_keeps_findings_and_api_status(self):self.run_scenario('issues')
    def test_root_checkpoint_recovery_reuses_finished_children_without_calls(self):
        tid,scenario,executor=self.run_scenario('supported');count=len(scenario.requests)
        self.db.update_task(tid,status='queued',result_json=None)
        executor.execute(tid,'after-restart')
        self.assertEqual(len(scenario.requests),count)
        self.assertEqual(len(self.db.list_subtasks(tid)),3)
    def test_background_worker_does_not_claim_child_tasks(self):
        task=TaskService(self.db).create(__import__('research_loop.domain.tasks',fromlist=['TaskCreate']).TaskCreate(prompt='root'))
        self.db.update_task(task.task_id,status='needs_input')
        child=self.db.create_subtask(task.task_id,task.task_id,0,'child',{},'child')
        # An older child must never beat eligible roots, including fixtures from previous runs.
        self.db.update_task(child['subtask_id'],created_at='1900-01-01T00:00:00+00:00')
        candidate=self.db.claim_task('outsider')
        if candidate:
            self.assertIsNone(candidate['parent_task_id'])
            self.assertNotEqual(candidate['task_id'],child['subtask_id'])
        self.assertEqual(self.db.get_task(child['subtask_id'])['status'],'queued')
    def test_page_scope_and_cached_candidate_authorization(self):
        from research_loop.tools.papers import specs
        from research_loop.tools.registry import ToolRegistry
        from research_loop.tools.base import ToolContext
        from research_loop.domain.tasks import SavePolicy
        from research_loop.services.papers import PaperService
        from research_loop.domain.evidence import PaperCandidate
        papers=PaperService(self.db,None,FixturePDF());papers.candidates['private']=PaperCandidate(paper_id='private',title='private',source='fixture',pdf_url='private.pdf')
        with self.assertRaises(PermissionError):papers.read('other','private')
        # 页权限来自根任务的用户原话，而不是 Supervisor 自行分配的 page_count。
        self.db.create_task('other',None,'a'*64,'只看第1页','deny',[])
        ctx=ToolContext(task_id='other',save_policy=SavePolicy.deny,services={'papers':papers,'scope':{'input_refs':['input-A'],'start_page':1,'page_count':1}})
        r=ToolRegistry(specs()).execute('read_paper',{'target_ref':'input-A','start_page':2,'page_count':1},ctx)
        self.assertFalse(r.ok);self.assertIn('page range',r.message)
    def test_reviewer_completed_status_does_not_overwrite_runtime(self):
        tid,_,_=self.run_scenario('insufficient')
        view=TaskService(self.db).view(self.db.get_task(tid));v=next(c for c in view.subtasks if c.role=='reviewer')
        self.assertEqual(v.status,'completed');self.assertEqual(v.review_status,'insufficient')
    def test_sigkill_reviewer_receipt_restores_entire_family_without_resend(self):
        if not os.getenv('AG2_COLLAB_TEST_DATABASE_URL'):self.skipTest('explicit isolated MySQL test required')
        import subprocess,signal,time,sys
        from research_loop.domain.tasks import TaskCreate,ResumeInput
        task=TaskService(self.db).create(TaskCreate(prompt='比较高被引论文并保存',save_policy='allow',attachments=[{'input_ref':'input-'+x,'local_path':str(self.root/(x+'.pdf')),'paper_id':'paper-'+x} for x in ['A','B']]))
        scenario=Scenario();TaskExecutor(self.settings,self.db,scenario.factory,lambda s,f:FixturePDF()).execute(task.task_id)
        q=TaskService(self.db).view(self.db.get_task(task.task_id)).clarification
        TaskService(self.db).resume(ResumeInput(clarification_id=q.clarification_id,answer='2篇',idempotency_key='resume'),task.task_id)
        ready=self.root/'ready';counter=self.root/'calls.jsonl'
        env={**os.environ,'AG2_COLLAB_ROOT':str(self.root),'AG2_COLLAB_TASK':task.task_id,'AG2_COLLAB_COUNTER':str(counter),'AG2_COLLAB_READY':str(ready)}
        cmd=[sys.executable,str(Path(__file__).with_name('collaboration_recovery_process.py'))]
        child=subprocess.Popen(cmd+['crash'],env=env)
        try:
            deadline=time.monotonic()+35
            while not ready.exists() and time.monotonic()<deadline and child.poll() is None:time.sleep(.1)
            self.assertTrue(ready.exists(),'child did not reach reviewer receipt boundary')
            reviewer=ready.read_text();receipt=self.db.list_model_calls(reviewer)[0];budget_before=self.db.budget_summary(task.task_id)
            child.send_signal(signal.SIGKILL);child.wait(timeout=10)
            before=[json.loads(l) for l in counter.read_text().splitlines()]
            self.db.update_task(task.task_id,lease_until='2000-01-01T00:00:00+00:00')
            subprocess.run(cmd+['resume'],env=env,check=True,timeout=40)
            after=[json.loads(l) for l in counter.read_text().splitlines()]
            self.assertEqual(sum(x['reviewer'] for x in after),1)
            self.assertEqual(sum(x['researcher'] for x in after),sum(x['researcher'] for x in before))
            self.assertEqual(self.db.list_model_calls(reviewer)[0]['request_id'],receipt['request_id'])
            self.assertEqual(len(self.db.list_subtasks(task.task_id)),3)
            view=TaskService(self.db).view(self.db.get_task(task.task_id));self.assertTrue(view.result.saved_path)
            self.assertEqual(len(after)-len(before),2) # only previously unsent Supervisor save + final
            self.assertEqual(self.db.budget_summary(task.task_id)['model_used']-budget_before['model_used'],2)
        finally:
            if child.poll() is None:child.kill();child.wait()

    def test_optional_review_export_preserves_original_markdown_and_hash(self):
        tid,_,_=self.run_scenario('insufficient')
        view=TaskService(self.db).view(self.db.get_task(tid));draft=self.db.get_draft(view.result.draft_id)
        exported=Path(view.result.saved_path).read_text()
        self.assertEqual(exported, draft['markdown'])
        self.assertNotIn('> 待核实草稿', exported)
        self.assertEqual(hashlib.sha256(draft['markdown'].encode()).hexdigest(),draft['content_hash'])
        record=self.db.fetchone('SELECT * FROM note_exports WHERE draft_id=?',(draft['draft_id'],))
        self.assertEqual(hashlib.sha256(exported.encode()).hexdigest(),record['file_sha256'])
        self.assertEqual(record['file_sha256'],draft['content_hash'])
    def test_parent_cancel_during_child_response_cannot_save_or_restart_child(self):
        from research_loop.domain.tasks import TaskCreate,ResumeInput
        task=TaskService(self.db).create(TaskCreate(prompt='比较高被引论文并保存',save_policy='allow',attachments=[{'input_ref':'input-'+x,'local_path':str(self.root/(x+'.pdf')),'paper_id':'paper-'+x} for x in ['A','B']]))
        scenario=Scenario();executor=TaskExecutor(self.settings,self.db,scenario.factory,lambda s,f:FixturePDF())
        executor.execute(task.task_id)
        q=TaskService(self.db).view(self.db.get_task(task.task_id)).clarification
        TaskService(self.db).resume(ResumeInput(clarification_id=q.clarification_id,answer='2篇',idempotency_key='cancel-resume'),task.task_id)
        handler=scenario.handler
        def cancelling(request):
            names={t['function']['name'] for t in json.loads(request.content).get('tools',[])}
            if 'finish_research' in names:TaskService(self.db).cancel(task.task_id)
            return handler(request)
        scenario.handler=cancelling
        executor.execute(task.task_id,'cancel-test')
        self.assertEqual(self.db.get_task(task.task_id)['status'],'cancelled')
        self.assertTrue(all(c['status']=='cancelled' for c in self.db.list_subtasks(task.task_id)))
        self.assertFalse(self.db.fetchall('SELECT s.* FROM save_receipts s JOIN drafts d ON s.draft_id=d.draft_id WHERE d.task_id=?',(task.task_id,)))
