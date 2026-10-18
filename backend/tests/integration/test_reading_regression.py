"""论文 集中适配回归：真实图/服务/收据，供应商与 HTTP 均为明确替身。"""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import pymupdf

from research_loop.agents.deep_reading import SYSTEM_PROMPT
from research_loop.agents.graph import build_graph, initial_state
from research_loop.config import Settings
from research_loop.domain.evidence import Document, PageEvidence
from research_loop.domain.tasks import TaskCreate
from research_loop.providers.deepseek import DeepSeekProvider, ModelResponse, ToolCall
from research_loop.providers.discovery import DiscoveryProvider
from research_loop.providers.paper_sources import resolve_backends
from research_loop.providers.pdf import PDFProvider, is_usable_paper_text
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.tools.registry import ToolRegistry
from research_loop.workers.executor import TaskExecutor


class ReadingRegressionTests(unittest.TestCase):
    """通过输入输出验证正文、单节点及失败交付，不把替身结果当作真实模型验收。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = FileStore(self.root/'data', self.root/'notes', self.root)
        url = os.getenv('READING_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root/'test.db')
        if not url.startswith('sqlite:') and 'researchloop_test_' not in url.rsplit('/',1)[-1]:
            raise ValueError('explicit isolated MySQL required')
        self.db = Database(url)
        self.db.ensure_schema()
        self.settings = Settings(project_root=self.root, data_dir=self.root/'data', output_dir=self.root/'notes', database_url=self.db.url)

    def provider(self, handler):
        """创建零等待 HTTP 替身，避免失败用例真的联网或退避。"""
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        provider = PDFProvider(self.files, client=client)
        provider.jina_retry_wait = 0
        return provider

    def test_jina_valid_cache_null_page_and_evidence_delivery(self):
        calls = []
        text = '# Paper\n' + 'Our method uses textual category embeddings.\n' * 60
        def handler(req):
            calls.append(str(req.url))
            return httpx.Response(200, text=text)
        pdf = self.provider(handler)
        tid = TaskService(self.db).create(TaskCreate(prompt='阅读全文')).task_id
        papers = PaperService(self.db, None, pdf)
        args = dict(task_id=tid, target_ref='attached', attachments={'attached':{'url':'https://paper.example/main.pdf'}})
        first = papers.read(**args)
        second = papers.read(**args)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(first['total_pages'])
        self.assertEqual(first['returned_pages'], [None])
        self.assertIsNone(first['read_coverage']['full_document_returned'])
        self.assertTrue(first['read_coverage']['all_extracted_text_returned'])
        bundle = papers.evidence_bundle(tid, [first['pages'][0]['evidence_id']])
        self.assertEqual(bundle[0]['text'], text)
        self.assertIsNone(bundle[0]['page_number'])
        self.assertEqual(second['acquisition']['kind'], 'text_cache')
        from research_loop.services.notes import NoteService
        from research_loop.domain.notes import CreateNoteInput
        draft = NoteService(self.db, self.files).create(tid, CreateNoteInput(title='test',markdown='summary',evidence_ids=[bundle[0]['evidence_id']]))
        self.assertEqual(draft.markdown, 'summary')
        self.assertEqual(draft.evidence_ids, [bundle[0]['evidence_id']])
        self.assertNotIn('physical page None',draft.markdown)

    def test_jina_error_falls_back_pymupdf_and_explicit_pages_skip_jina(self):
        document = pymupdf.open()
        for n in range(3):
            page = document.new_page()
            page.insert_text((40,40), ('Evidence for textual category features.\n' * 20))
        data = document.tobytes();document.close()
        calls = []
        def handler(req):
            calls.append(str(req.url))
            return httpx.Response(200, text='Warning: Target URL returned error' + 'x'*1400) if req.url.host=='r.jina.ai' else httpx.Response(200,content=data)
        pdf = self.provider(handler)
        doc = pdf.ingest('https://paper.example/a.pdf','p')
        self.assertEqual(doc.parser_name,'pymupdf')
        self.assertEqual(doc.total_pages,3)
        self.assertEqual(len(calls),2)
        pdf.ingest('https://paper.example/a.pdf','p')
        self.assertEqual(len(calls),2)
        doc2 = pdf.ingest_pages('https://paper.example/a.pdf','p')
        self.assertEqual(pdf.pages(doc2,2,1)[0].page_number,2)
        self.assertEqual(len([c for c in calls if 'r.jina.ai' in c]),1)
        self.assertFalse(is_usable_paper_text('<html>'+'a'*2000))
        self.assertFalse(is_usable_paper_text('{"error":"'+'a'*2000))

    def test_date_splitting_is_bounded_deduplicated_and_disclosed(self):
        calls = []
        def handler(req):
            body=json.loads(req.content);calls.append(body)
            if len(calls)==1:return httpx.Response(500,json={'code':'57014'})
            return httpx.Response(200,json=[{'id':'paper','title':'paper'}])
        cfg=resolve_backends({'supabase':{'url':'https://papers.example','anon_key':'fixture'}})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result=DiscoveryProvider(backends=cfg,client=client).search('language','2026-01-01','2026-01-04',['arxiv'])
        self.assertEqual(len(result.candidates),1)
        self.assertEqual(len(calls),3)
        self.assertEqual(calls[1]['filter_published_end'],calls[2]['filter_published_start'])
        self.assertTrue(all(c['query_text']=='language' for c in calls))
        with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(500,json={'code':'57014'}))) as client:
            p=DiscoveryProvider(backends=cfg,client=client,max_date_windows=3)
            result=p.search('language','2023-01-01','2026-01-04',['arxiv'])
        source=next(s for s in result.source_results if s.source=='arxiv')
        self.assertEqual(source.status,'partial_results')
        self.assertEqual(len(source.request['windows']),3)
        self.assertTrue(source.request['unresolved_windows'])
        self.assertTrue(result.truncated)

    def test_parallel_download_diagnostics_stay_with_their_paper(self):
        """强制两个 HTTP 下载重叠，验证地址、正文和诊断始终属于同一论文。"""
        import copy
        import threading
        from concurrent.futures import ThreadPoolExecutor

        barrier = threading.Barrier(2)

        def handler(req):
            barrier.wait(timeout=5)
            return httpx.Response(200, content=b"%PDF-" + str(req.url).encode(),
                                  headers={"content-type": "application/pdf"})

        pdf = self.provider(handler)

        def read(source):
            content = pdf._read_source(source)
            return content, copy.deepcopy(pdf.last_fetch)

        urls = ["https://paper.example/A.pdf", "https://paper.example/B.pdf"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(read, urls))
        for source, (content, diagnostics) in zip(urls, outputs):
            self.assertEqual(content, b"%PDF-" + source.encode())
            self.assertEqual(diagnostics["requested_source"], source)
            self.assertEqual(diagnostics["final_url"], source)
            self.assertEqual(len(diagnostics["attempts"]), 1)
            self.assertEqual(diagnostics["attempts"][0]["requested_url"], source)

    def summary_fixture(self, handler):
        """准备一个已真实落库的取证交接状态，直接进入生产图的精读节点。"""
        tid=TaskService(self.db).create(TaskCreate(prompt='只基于已读原文精读，不保存')).task_id
        self.db.save_document(Document(document_id='doc',paper_id='paper',content_sha256='a'*64,parser_name='fixture',parser_version='1',config_sha256='b'*64,total_pages=1,cache_path='fixture').model_dump())
        page=PageEvidence(evidence_id='evidence:one',document_id='doc',page_number=1,text='Source-only fact. Do not use old summary.')
        self.db.save_page(page.model_dump());self.db.mark_task_evidence(tid,page.evidence_id)
        client=httpx.Client(transport=httpx.MockTransport(handler));self.addCleanup(client.close)
        provider=DeepSeekProvider('fixture',client=client)
        state=initial_state(tid,'精读','deny');state.update(role='researcher',finished=True,structured_result={'findings':['old terse observation']},evidence_ids=[page.evidence_id])
        papers=PaperService(self.db,None,None)
        graph=build_graph(provider,ToolRegistry([]),{'papers':papers},self.db,checkpoint_store=CheckpointStore(self.db),deep_reading=True)
        return tid,state,graph

    @staticmethod
    def raw(text, reason='stop'):
        return {'id':'fixed-receipt','choices':[{'finish_reason':reason,'message':{'role':'assistant','content':text}}],
                'usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20}}

    def test_single_node_upstream_continuation_budget_and_resume(self):
        calls=[]
        def handler(req):
            body=json.loads(req.content);calls.append(body)
            self.assertEqual(body['messages'][0]['content'],SYSTEM_PROMPT)
            if len(calls) == 1:
                self.assertIn('Source-only fact',body['messages'][1]['content'])
            else:
                self.assertNotIn('Source-only fact',json.dumps(body['messages']))
                self.assertIn('上一次输出如下',body['messages'][2]['content'])
            return httpx.Response(200,json=self.raw('## incomplete' if len(calls)==1 else 'continued\n（完）','length' if len(calls)==1 else 'stop'))
        tid,state,graph=self.summary_fixture(handler)
        result=graph.invoke(state)
        self.assertEqual([r['max_tokens'] for r in calls],[4096,2048])
        self.assertTrue(all(r['thinking'] == {'type': 'disabled'} for r in calls))
        receipts = self.db.list_model_calls(tid)
        self.assertTrue(all(json.loads(r['request_json'])['thinking'] == {'type': 'disabled'} for r in receipts))
        self.assertEqual(result['deep_reading']['status'],'complete')
        self.assertEqual(result['model_requests'],2)
        graph.invoke(self.db.get_checkpoint(tid))
        self.assertEqual(len(calls),2)
        self.assertEqual(len(self.db.list_model_calls(tid)),2)

    def test_summary_thinking_option_does_not_change_other_calls(self):
        """精读专用 thinking 参数不泄漏到 Supervisor 或 Reviewer 的普通调用。"""
        calls = []
        with httpx.Client(transport=httpx.MockTransport(
            lambda r: (calls.append(json.loads(r.content)) or httpx.Response(200, json=self.raw('text')))
        )) as client:
            provider = DeepSeekProvider('fixture', client=client)
            messages = [{'role': 'user', 'content': 'question'}]
            provider.send(messages, thinking={'type': 'disabled'}, max_tokens=4096)
            provider.send(messages)
        self.assertEqual(calls[0]['thinking'], {'type': 'disabled'})
        self.assertNotIn('thinking', calls[1])

    def test_receipt_before_checkpoint_crash_recovers_without_resend(self):
        calls=[]
        def handler(req):
            calls.append(req)
            return httpx.Response(200,json=self.raw('## summary\n（完）'))
        tid,state,graph=self.summary_fixture(handler)
        original=self.db.save_model_call
        def crash(*args,**kwargs):
            result=original(*args,**kwargs)
            if args[2]=='succeeded':raise SystemExit('simulated process exit after receipt')
            return result
        with patch.object(self.db,'save_model_call',side_effect=crash),self.assertRaises(SystemExit):graph.invoke(state)
        budget=self.db.budget_summary(tid)['model_used']
        result=graph.invoke(self.db.get_checkpoint(tid))
        self.assertEqual(len(calls),1)
        self.assertEqual(result['deep_reading']['status'],'complete')
        self.assertEqual(self.db.budget_summary(tid)['model_used'],budget)
        self.assertEqual(self.db.list_model_calls(tid)[0]['request_id'],'fixed-receipt')

    def test_continuation_unknown_retains_first_draft_and_does_not_resend(self):
        calls=[]
        def handler(req):
            calls.append(req)
            if len(calls)==2:raise httpx.ReadTimeout('controlled timeout',request=req)
            return httpx.Response(200,json=self.raw('## preserved incomplete draft','length'))
        tid,state,graph=self.summary_fixture(handler)
        result=graph.invoke(state)
        self.assertIn('preserved incomplete draft',result['final_answer'])
        self.assertEqual(result['deep_reading']['status'],'partial')
        self.assertEqual([c['status'] for c in self.db.list_model_calls(tid)],['succeeded','unknown'])
        graph.invoke(self.db.get_checkpoint(tid));self.assertEqual(len(calls),2)

    def test_three_rounds_length_ends_and_cancel_does_not_call(self):
        calls=[]
        tid,state,graph=self.summary_fixture(lambda r:(calls.append(r) or httpx.Response(200,json=self.raw('part','length'))))
        result=graph.invoke(state)
        self.assertEqual(len(calls),6)
        self.assertEqual(result['deep_reading']['status'],'partial')
        tid2,state2,graph2=self.summary_fixture(lambda r:self.fail('cancelled task sent a model request'))
        TaskService(self.db).cancel(tid2)
        result=graph2.invoke(state2)
        self.assertTrue(result['cancelled'])

    def test_reviewer_timeout_is_execution_failure_and_draft_survives(self):
        from integration.test_collaboration_contract import Scenario,FixturePDF
        from research_loop.domain.tasks import ResumeInput
        scenario=Scenario();handler=scenario.handler
        def fail_review(req):
            names={t['function']['name'] for t in json.loads(req.content).get('tools',[])}
            if 'finish_review' in names:raise httpx.ReadTimeout('controlled reviewer timeout',request=req)
            return handler(req)
        scenario.handler=fail_review
        task=TaskService(self.db).create(TaskCreate(prompt='比较并保存',save_policy='allow',attachments=[{'input_ref':'input-'+c,'url':'https://paper.example/'+c+'.pdf','paper_id':'paper-'+c} for c in ['A','B']]))
        executor=TaskExecutor(self.settings,self.db,scenario.factory,lambda s,f:FixturePDF())
        executor.execute(task.task_id)
        clarif=TaskService(self.db).view(self.db.get_task(task.task_id)).clarification
        TaskService(self.db).resume(ResumeInput(clarification_id=clarif.clarification_id,answer='2',idempotency_key='resume'),task.task_id)
        result=executor.execute(task.task_id)
        self.assertTrue(result.draft_id)
        self.assertIn('# Compare',result.answer_markdown)
        self.assertEqual(result.review_status,'review_failed')
        self.assertIsNone(result.saved_path)
        self.assertIsNone(self.db.get_review_result(task.task_id,result.draft_id,self.db.get_draft(result.draft_id)['content_hash']))
        reviewer=next(c for c in self.db.list_subtasks(task.task_id) if c['position_no']==999)
        self.assertEqual(reviewer['status'],'failed')
        count=len(self.db.list_model_calls(reviewer['subtask_id']))
        repeated=executor.execute(task.task_id)
        self.assertEqual(repeated.draft_id,result.draft_id)
        self.assertEqual(len(self.db.list_model_calls(reviewer['subtask_id'])),count)

    def test_nonretryable_duplicate_does_not_execute_tool_twice(self):
        from pydantic import BaseModel
        from research_loop.tools.base import ToolSpec
        from research_loop.domain.tasks import Observation
        class Args(BaseModel):pass
        calls=[]
        def tool(args,ctx):
            calls.append(1)
            return Observation(ok=False,kind='execution_error',retryable=False,message='missing source')
        class Provider:
            def __init__(self):self.n=0
            def complete(self,*a,**kw):
                self.n+=1
                return ModelResponse(None,[ToolCall(str(self.n),'read_test',{})]) if self.n<3 else ModelResponse('No source; stopped')
        state=initial_state('task','prompt','deny')
        result=build_graph(Provider(),ToolRegistry([ToolSpec('read_test','test',Args,tool)]),{}).invoke(state)
        self.assertEqual(len(calls),1)
        self.assertIn('未重复执行',str(result['observations']))

    def test_real_sigkill_after_reading_receipt_keeps_usage_without_resend(self):
        """真实终止子进程，供应商仍为替身；复验 MySQL 收据而非网络质量。"""
        import subprocess, signal, sys, time
        if not os.getenv('READING_TEST_DATABASE_URL'):
            self.skipTest('explicit isolated MySQL required')
        tid,state,_=self.summary_fixture(lambda r:self.fail('parent must not send'))
        self.db.save_checkpoint(tid,state)
        env={**os.environ,'READING_RECOVERY_DB':self.db.url,'READING_RECOVERY_TASK':tid,
             'READING_RECOVERY_COUNTER':str(self.root/'counter'),'READING_RECOVERY_READY':str(self.root/'ready')}
        command=[sys.executable,str(Path(__file__).with_name('reading_recovery_process.py'))]
        child=subprocess.Popen(command+['crash'],env=env)
        try:
            deadline=time.monotonic()+20
            while not (self.root/'ready').exists() and child.poll() is None and time.monotonic()<deadline:
                time.sleep(.1)
            self.assertTrue((self.root/'ready').exists())
            child.send_signal(signal.SIGKILL);child.wait(timeout=5)
            self.assertEqual(child.returncode,-9)
            before=self.db.list_model_calls(tid)
            budget=self.db.budget_summary(tid)['model_used']
            subprocess.run(command+['resume'],env=env,check=True,timeout=20)
            self.assertEqual((self.root/'counter').read_text().count('one physical request'),1)
            self.assertEqual(self.db.list_model_calls(tid),before)
            self.assertEqual(self.db.budget_summary(tid)['model_used'],budget)
            self.assertEqual(self.db.get_checkpoint(tid)['deep_reading']['status'],'complete')
        finally:
            if child.poll() is None:
                child.kill();child.wait()
