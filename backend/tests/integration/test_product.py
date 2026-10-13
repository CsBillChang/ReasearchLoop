"""AG5产品接口、流装配与恢复测试：本地真实API/数据库，供应商使用明确替身。"""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
import httpx
import pymupdf
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.storage.mysql import Database
from research_loop.services.tasks import TaskService
from research_loop.services.conversations import ConversationService
from research_loop.services.notes import NoteService
from research_loop.storage.files import FileStore
from research_loop.domain.tasks import TaskCreate,TaskResult,TaskStatus,SavePolicy
from research_loop.domain.notes import CreateNoteInput
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.providers.streaming import StreamAssembly,string_prefix
from research_loop.workers.executor import TaskExecutor
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.agents.graph import initial_state


class ProductTests(unittest.TestCase):
    """隔离接口成功/失败分支，避免UI成功掩盖身份和写入问题。"""

    def setUp(self):
        """默认SQLite，真实MySQL必须显式指定隔离库。"""
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        url=os.getenv('AG5_TEST_DATABASE_URL') or 'sqlite:///'+str(root/'test.db')
        if not url.startswith('sqlite:') and not url.rsplit('/',1)[-1].startswith('researchloop_test_ag5_'):
            raise ValueError('AG5 isolated database required')
        self.s=Settings(project_root=root,data_dir=root/'data',output_dir=root/'notes',database_url=url)
        self.db=Database(url);self.db.ensure_schema();self.ts=TaskService(self.db)
        self.cid=ConversationService(self.db).create('AG5测试',str(uuid4()))['conversation_id']
        self.files=FileStore(self.s.data_dir,self.s.output_dir,root)
        self.notes=NoteService(self.db,self.files)
        from research_loop.main import app
        for name in ['tasks','product','notes']:
            p=patch('research_loop.api.'+name+'.get_settings',return_value=self.s);p.start();self.addCleanup(p.stop)
        self.client=TestClient(app);self.addCleanup(self.client.close)

    def create(self,prompt='用户研究指令',**kw):
        """每例用独立会话任务和幂等键，不触发供应商。"""
        return self.client.post('/api/tasks',json={'conversation_id':self.cid,'prompt':prompt,'idempotency_key':str(uuid4()),**kw}).json()['task_id']

    def test_pre_send_budget_denial_has_no_remote_unknown_or_retry(self):
        """实际API→Worker经过评测闸门；确定未发送单列，既有未知请求行为不变。"""
        from scripts.run_ag6 import EvaluationLedger, EvaluationRunner, summarize_usage
        runner=object.__new__(EvaluationRunner)
        runner.cancel_hook=False
        runner.ledger=EvaluationLedger(self.s.data_dir.parent/'ledger.json',{
            'cases':[{'id':'fixture','request_cap':2}], 'physical_cap':2,
            'token_cap':1, 'wall_seconds':60})
        runner.ledger.case='fixture'
        task_id=self.create('报告已有成果，不检索，不保存')
        executor=TaskExecutor(self.s.model_copy(update={'deepseek_api_key':'fixture'}),self.db,runner.provider)
        with patch.object(httpx.Client,'send',side_effect=AssertionError('must not reach network')):
            executor.execute(task_id)
            # 已结算partial不能当作新任务重新领取；不绕过现有租约/终态保护。
            with self.assertRaisesRegex(RuntimeError,'not queued'):
                executor.execute(task_id)
        rows=self.db.list_model_calls(task_id)
        self.assertEqual(len(rows),1)
        self.assertEqual((rows[0]['status'],rows[0]['sent_state']),('not_sent','not_sent'))
        self.assertIn('AG6_TOKEN_CAP_BEFORE_SEND',rows[0]['response_json'])
        self.assertEqual(runner.ledger.records,[])
        self.assertFalse(self.db.has_unknown_family_call(task_id))
        self.assertEqual(summarize_usage(rows)['missing_usage_call_ids'],[])
        view=self.client.get('/api/tasks/'+task_id).json()
        self.assertEqual(view['usage']['unknown'],0)
        self.assertEqual(view['usage']['requests'],0)
        self.assertEqual(view['usage']['not_sent'],1)
        self.assertNotEqual(view['status'],'completed')
        self.assertFalse(list(self.s.output_dir.rglob('*.md')))

    def test_draft_review_status_is_bound_to_hash_not_body(self):
        """正文提前自称核验成功不构成结果；失败记录与新hash的未核验状态独立显示。"""
        tid=self.create()
        first=self.notes.create(tid,CreateNoteInput(title='候选',markdown='正文自称：已独立核验'))
        url=f'/api/tasks/{tid}/drafts/{first.draft_id}'
        self.assertEqual(self.client.get(url).json()['verification']['status'],'not_run')
        from research_loop.agents.reflection import ReflectionService
        ReflectionService(self.db).review(tid,first.draft_id,first.content_hash,'核验修改处',[],['修改句是否有据'],
            lambda *args:{'status':'execution_failed','gaps':['模型未完成核验']})
        value=self.client.get(url).json()
        self.assertEqual(value['verification']['status'],'execution_failed')
        self.assertEqual(value['verification']['content_hash'],first.content_hash)
        self.assertEqual(value['markdown'],first.markdown)
        second=self.notes.create(tid,CreateNoteInput(title='新稿',markdown='新版本'))
        self.assertEqual(self.client.get(f'/api/tasks/{tid}/drafts/{second.draft_id}').json()['verification']['status'],'not_run')

    def test_pdf_upload_identity_and_invalid(self):
        """PDF字节/大小/身份验证，重复上传复用hash，不泄露服务器路径。"""
        doc=pymupdf.open();doc.new_page().insert_text((40,40),'AG5 PDF attachment');raw=doc.tobytes();doc.close()
        response=self.client.post('/api/uploads?filename=paper.pdf',content=raw)
        self.assertEqual(response.status_code,201,response.text)
        a=response.json();self.assertEqual(a['pages'],1);self.assertNotIn('local_path',a)
        self.assertEqual(self.client.post('/api/uploads',content=raw).json()['input_ref'],a['input_ref'])
        tid=self.create(attachments=[{'input_ref':a['input_ref'],'filename':'paper.pdf'}])
        self.assertTrue(Path(json.loads(self.db.get_task(tid)['attachments_json'])[0]['local_path']).exists())
        self.assertEqual(self.client.post('/api/uploads',content=b'not pdf').status_code,422)
        small=self.s.model_copy(update={'max_pdf_bytes':3})
        with patch('research_loop.api.product.get_settings',return_value=small):
            self.assertEqual(self.client.post('/api/uploads',content=raw).status_code,413)
        self.assertEqual(self.client.post('/api/tasks',json={'prompt':'x','attachments':[{'input_ref':'upload-../oops'}]}).status_code,422)
        self.assertEqual(self.client.post('/api/tasks',json={'prompt':'x','attachments':[{'input_ref':a['input_ref'],'url':'https://other.invalid'}]}).status_code,422)

    def test_sse_replay_and_terminal_no_model(self):
        """已结算结果重连只读，不重复消息、模型或流式片段。"""
        tid=self.create();first=self.db.append_event(tid,'answer_stream',{'text':'第一段','stream_key':'answer','request_number':1,'provisional':True})
        second=self.db.append_event(tid,'answer_stream',{'text':'第一段和第二段','stream_key':'answer','request_number':1,'provisional':True})
        self.ts.complete(tid,TaskStatus.completed,TaskResult(answer_markdown='最终回答'))
        response=self.client.get('/api/tasks/'+tid+'/stream',headers={'Last-Event-ID':first})
        self.assertEqual(response.status_code,200);self.assertIn(second,response.text);self.assertNotIn('id: '+first,response.text)
        self.assertIn('event: settled',response.text);self.assertIn('最终回答',response.text)
        self.assertEqual(self.db.list_model_calls(tid),[])
        self.assertEqual(ConversationService(self.db).view(self.cid)['total'],2)
        self.assertEqual(self.client.get('/api/tasks/missing/stream').status_code,404)

    def test_cancel_retains_result_before_new_task(self):
        """queued取消立即终态，保留已有草稿，新任务不继承保存授权。"""
        tid=self.create(save_policy='allow')
        draft=self.notes.create(tid,CreateNoteInput(title='已有成果',markdown='保留这段',evidence_ids=[]))
        self.assertEqual(self.client.post('/api/tasks/'+tid+'/cancel').json()['status'],'cancelled')
        self.assertIn('保留这段',self.client.get('/api/tasks/'+tid).json()['result']['answer_markdown'])
        follow=self.create('按新要求继续')
        self.assertEqual(self.db.get_task(follow)['save_policy'],'deny')
        self.assertTrue(self.db.get_draft(draft.draft_id))

    def test_save_confirm_hash_and_conflict(self):
        """保存按钮是具体稿件授权，不改变旧任务deny；冲突/过期拒绝。"""
        tid=self.create();d=self.notes.create(tid,CreateNoteInput(title='精读稿',markdown='准确披露未知',evidence_ids=[]))
        self.ts.complete(tid,TaskStatus.partial,TaskResult(answer_markdown=d.markdown,draft_id=d.draft_id))
        url=f'/api/tasks/{tid}/drafts/{d.draft_id}/save';body={'content_hash':d.content_hash,'confirm':True}
        self.assertEqual(self.client.post(url,json={**body,'confirm':False}).status_code,422)
        a=self.client.post(url,json=body);self.assertEqual(a.status_code,200,a.text)
        self.assertEqual(self.client.post(url,json=body).json()['receipt_id'],a.json()['receipt_id'])
        self.assertEqual(self.db.get_task(tid)['save_policy'],'deny')
        self.assertEqual(self.client.post(url,json={**body,'content_hash':'0'*64}).status_code,409)
        Path(a.json()['path']).write_text('用户编辑')
        self.assertEqual(self.client.post(url,json=body).status_code,409)

    def test_feedback_binding_and_idempotency(self):
        """反馈匹配同会话准确结果hash，重复提交仅一个新任务。"""
        tid=self.create();answer='原回答'
        self.ts.complete(tid,TaskStatus.completed,TaskResult(answer_markdown=answer))
        payload={'conversation_id':self.cid,'source_task_id':tid,'content_hash':hashlib.sha256(answer.encode()).hexdigest(),'prompt':'请核对这一结论','idempotency_key':str(uuid4())}
        a=self.client.post('/api/feedback',json=payload);self.assertEqual(a.status_code,202,a.text)
        b=self.client.post('/api/feedback',json=payload);self.assertEqual(a.json()['task_id'],b.json()['task_id'])
        self.assertIn(tid,a.json()['prompt'])
        self.assertEqual(self.client.post('/api/feedback',json={**payload,'content_hash':'0'*64}).status_code,409)
        self.assertEqual(self.client.post('/api/feedback',json={**payload,'conversation_id':'other'}).status_code,404)

    def _current_draft_feedback(self, save_policy='deny'):
        """建立有真实身份绑定的微型原文夹具；不使用网络或真实论文结论。"""
        tid=self.create(save_policy='allow');p='paper-'+uuid4().hex;doc='doc-'+uuid4().hex
        self.db.save_paper(p,{'paper_id':p,'title':'Fixture','source':'cvpr'})
        self.db.save_document({'document_id':doc,'paper_id':p,'content_sha256':hashlib.sha256(doc.encode()).hexdigest(),
            'parser_name':'fixture','parser_version':'1','config_sha256':'0'*64,'total_pages':2,'cache_path':'unused'})
        refs=[]
        for page in (1,2):
            ref=doc+'-p'+str(page);refs.append(ref)
            self.db.save_page({'evidence_id':ref,'document_id':doc,'page_number':page,'text':'The model uses language embeddings.'})
            self.db.grant_evidence(tid,ref,tid)
        self.db.grant_paper(tid,p,tid)
        d=self.notes.create(tid,CreateNoteInput(title='当前稿',markdown='保留标题。\n模型使用语言嵌入。\n保留结尾。',evidence_ids=refs))
        self.ts.complete(tid,TaskStatus.partial,TaskResult(answer_markdown=d.markdown,draft_id=d.draft_id))
        body={'conversation_id':self.cid,'source_task_id':tid,'source_draft_id':d.draft_id,'content_hash':d.content_hash,
              'prompt':'仅核验语言用途并交付当前稿，不检索或重写','save_policy':save_policy,'idempotency_key':str(uuid4())}
        return d,body,refs

    def _run_current_draft_chain(self, save):
        """真实API/Worker配供应商替身，只核验指定页，不重新创建稿或Researcher。"""
        old,body,refs=self._current_draft_feedback('allow' if save else 'deny')
        response=self.client.post('/api/feedback',json=body);self.assertEqual(response.status_code,202,response.text)
        tid=response.json()['task_id']
        self.assertEqual(self.client.post('/api/feedback',json=body).json()['task_id'],tid)
        current=self.db.fetchone('SELECT * FROM drafts WHERE task_id=?',(tid,))
        calls=[]
        class Provider:
            """只按当前工具观察作确定决策；不连接任何模型服务。"""
            def send(inner,messages,tools,**kwargs):
                names={t['function']['name'] for t in tools};calls.append(messages)
                done={c['function']['name'] for m in messages for c in m.get('tool_calls',[])}
                if 'finish_review' in names:
                    bundle=json.loads(messages[-1]['content'].split('\n',1)[1])
                    self.assertEqual([p['evidence_id'] for p in bundle],[refs[0]])
                    name='finish_review';args={'status':'supported','findings':[{'claim':'模型使用语言嵌入','status':'supported',
                        'reason':'原文支持','basis':'direct_statement','source_quote':'The model uses language embeddings.','evidence_ids':[refs[0]]}]}
                elif 'set_research_plan' not in done:
                    current_message=next(m['content'] for m in messages if str(m.get('content','')).startswith('当前候选稿数据'))
                    self.assertEqual(json.loads(current_message.split('\n',1)[1])['markdown'],old.markdown)
                    name='set_research_plan';args={'goal':'核验当前稿','save_requested':save,
                        'goals':[{'goal_id':'verify','description':'核验语言用途','requires_fulltext':True,'min_papers':1}]}
                elif 'review_note' not in done:
                    name='review_note';args={'draft_id':current['draft_id'],'content_hash':current['content_hash'],
                        'evidence_ids':[refs[0]],'verification_questions':['语言用途是否被原文支持？']}
                elif save and 'save_note' not in done:
                    name='save_note';args={'draft_id':current['draft_id'],'content_hash':current['content_hash']}
                else:
                    name='finish_task';args={'draft_id':current['draft_id'],'goals':[{'goal_id':'verify','status':'satisfied','reason':'当前稿定向核验完成'}]}
                return {'id':uuid4().hex,'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2},'choices':[{'finish_reason':'tool_calls',
                    'message':{'role':'assistant','content':'','tool_calls':[{'id':uuid4().hex,'type':'function','function':{'name':name,'arguments':json.dumps(args,ensure_ascii=False)}}]}}]}
            send_stream=send
            def close(inner):
                """替身没有网络资源。"""
        TaskExecutor(self.s,self.db,lambda s:Provider()).execute(tid)
        view=self.client.get('/api/tasks/'+tid).json()
        self.assertEqual(view['status'],'completed',view)
        self.assertEqual(view['result']['answer_markdown'],old.markdown)
        self.assertEqual(view['result']['review_status'],'supported')
        self.assertEqual(bool(view['result']['saved_path']),save)
        self.assertEqual(len(self.db.list_subtasks(tid)),1)
        self.assertFalse({'search_papers','delegate_research','create_note','revise_note'}&{a['tool_name'] for a in self.db.list_actions(tid)})
        self.assertEqual(self.db.get_draft(old.draft_id)['markdown'],old.markdown)
        self.assertFalse(self.db.get_review_result(old.task_id,old.draft_id,old.content_hash))
        if not save:
            with self.assertRaises(PermissionError):self.notes.save(tid,SavePolicy.deny,current['draft_id'],old.content_hash)
        else:
            self.assertEqual(Path(view['result']['saved_path']).read_text(),old.markdown)
        self.assertEqual(len(calls),5 if save else 4)

    def test_current_draft_feedback_review_without_save(self):
        """禁存稿复审并完成交付，无整篇重生成或文件写入。"""
        from research_loop.domain.tasks import SavePolicy
        self._run_current_draft_chain(False)

    def test_current_draft_feedback_review_and_save(self):
        """原授权允许时在同hash核验后安全保存，旧任务及审校保持。"""
        self._run_current_draft_chain(True)

    def test_current_draft_feedback_rejects_stale_and_rolls_back(self):
        """过期/跨会话/跨稿被拒；复制事务失败不留下可被Worker领取的任务。"""
        from research_loop.storage.memory import Transaction
        d,body,refs=self._current_draft_feedback()
        self.assertEqual(self.client.post('/api/feedback',json={**body,'content_hash':'0'*64}).status_code,409)
        self.assertEqual(self.client.post('/api/feedback',json={**body,'conversation_id':'other'}).status_code,404)
        original=Transaction.execute
        def fail(tx,sql,params=()):
            """在任务已入队但稿件未插入的事务窗口注入异常。"""
            if sql.startswith('INSERT INTO drafts'):raise ValueError('CONTROLLED_COPY_FAILURE')
            return original(tx,sql,params)
        with patch.object(Transaction,'execute',fail):
            self.assertEqual(self.client.post('/api/feedback',json=body).status_code,409)
        self.assertIsNone(self.db.get_task_by_idempotency(body['idempotency_key']))
        self.notes.create(d.task_id,CreateNoteInput(title='新版',markdown='不同内容'))
        self.assertEqual(self.client.post('/api/feedback',json=body).status_code,409)

    def test_evidence_authorization_and_page_identity(self):
        """其他任务不能取原文/PDF；未知页码不会变成第0页。"""
        tid=self.create();self.assertEqual(self.client.get(f'/api/tasks/{tid}/evidence/other').status_code,404)
        self.assertEqual(self.client.get(f'/api/tasks/{tid}/evidence/other/pdf').status_code,404)
        self.assertEqual(self.client.get(f'/api/tasks/{tid}/drafts/other').status_code,404)
        self.assertEqual(self.client.get(f'/api/tasks/{tid}/artifacts').json()['evidence'],[])

    def test_feedback_retains_bound_attachment_and_page_scope_not_save_permission(self):
        """旧稿反馈可直接补读同材料；不会继承旧保存授权或暗中放宽页范围。"""
        from research_loop.services.papers import PaperService
        doc=pymupdf.open()
        for number in range(3):
            doc.new_page().insert_text((40,40),f'Authorized material page {number+1}')
        raw=doc.tobytes();doc.close()
        upload=self.client.post('/api/uploads?filename=source.pdf',content=raw).json()
        tid=self.create('只读物理第1至2页，写一份稿',save_policy='allow',attachments=[{'input_ref':upload['input_ref']}])
        self.ts.complete(tid,TaskStatus.completed,TaskResult(answer_markdown='原稿'))
        payload={'conversation_id':self.cid,'source_task_id':tid,'content_hash':hashlib.sha256('原稿'.encode()).hexdigest(),
                 'prompt':'核对原稿这一结论，不保存','idempotency_key':str(uuid4())}
        response=self.client.post('/api/feedback',json=payload)
        self.assertEqual(response.status_code,202,response.text)
        current=self.db.get_task(response.json()['task_id'])
        self.assertIn('"answer_markdown": "原稿"',current['prompt'])
        self.assertIn('"source_type": "research_history"',current['prompt'])
        self.assertIn('"source_id": "'+tid+'"',current['prompt'])
        self.assertEqual(json.loads(current['attachments_json']),json.loads(self.db.get_task(tid)['attachments_json']))
        self.assertEqual(current['save_policy'],'deny')
        self.assertTrue(self.db.task_has_input(current['task_id'],upload['input_ref']))
        scope=PaperService(self.db,None,None).user_scope(current['task_id'])
        self.assertEqual((scope.start,scope.end),(1,2))
        self.assertEqual(self.client.post('/api/feedback',json=payload).json()['task_id'],current['task_id'])

    def test_large_feedback_baseline_uses_exact_history_readback(self):
        """过长原稿不挤爆任务输入；指向原任务的既有分页入口，不截断冒充完整稿。"""
        answer='原稿材料内容。'*2000
        tid=self.create('已有长稿')
        self.ts.complete(tid,TaskStatus.completed,TaskResult(answer_markdown=answer))
        payload={'conversation_id':self.cid,'source_task_id':tid,'content_hash':hashlib.sha256(answer.encode()).hexdigest(),
                 'prompt':'只修最后一处结论，不保存','idempotency_key':str(uuid4())}
        response=self.client.post('/api/feedback',json=payload)
        self.assertEqual(response.status_code,202,response.text)
        current=self.db.get_task(response.json()['task_id'])
        self.assertIn('"source_type": "research_history"',current['prompt'])
        self.assertIn('"source_id": "'+tid+'"',current['prompt'])
        self.assertNotIn('"answer_markdown":',current['prompt'])
        self.assertLessEqual(len(current['prompt']),12000)


    def test_graph_stream_receipt_restore_no_resend(self):
        """实际API→Worker→流式供应商替身；成功收据回放不再次发送。"""
        from tests.integration.test_memory import MemoryProvider
        class Provider(MemoryProvider):
            """继承已验证记忆场景，用公开流回调记录真实接口链路。"""
            def send_stream(self,messages,tools,force_text=False,*,on_public,**kwargs):
                raw=self.send(messages,tools,force_text,**kwargs)
                if raw['choices'][0]['message']['tool_calls'][0]['function']['name']=='finish_task':
                    on_public({'text':'实际公开回答','stream_key':'answer','provisional':True})
                return raw
        p=Provider(None);tid=self.create('请回顾记录，不搜索或保存。')
        TaskExecutor(self.s,self.db,lambda s:p).execute(tid)
        self.assertEqual(self.db.get_task(tid)['status'],'completed')
        self.assertTrue(any(e['event_type']=='answer_stream' for e in self.db.list_events(tid)))
        count=len(p.calls)
        state=initial_state(tid,self.db.get_task(tid)['prompt'],'deny');state['coverage_required']=True
        CheckpointStore(self.db).save(tid,state)
        self.db.update_task(tid,status='queued',claimed_by=None,lease_token=None,lease_until=None)
        TaskExecutor(self.s,self.db,lambda s:p).execute(tid)
        self.assertEqual(len(p.calls),count)

    def test_continuation_persists_without_browser_and_replays(self):
        """页面关闭后Worker仍接续，双轮询只入队一次，原稿与新授权分离。"""
        from research_loop.services.continuations import ContinuationService
        tid=self.create(save_policy='allow')
        d=self.notes.create(tid,CreateNoteInput(title='已有成果',markdown='已完成部分',evidence_ids=[]))
        self.db.update_task(tid,status='running')
        body={'conversation_id':self.cid,'prompt':'新要求：只解释现有结果，不保存','save_policy':'deny','idempotency_key':str(uuid4())}
        url=f'/api/tasks/{tid}/continue'
        a=self.client.post(url,json=body);self.assertEqual(a.status_code,202,a.text)
        self.assertEqual(a.json()['status'],'pending')
        self.assertEqual(self.client.post(url,json=body).json()['continuation_id'],a.json()['continuation_id'])
        self.assertEqual(self.client.post(url,json={**body,'prompt':'另一要求'}).status_code,409)
        self.assertEqual(self.client.post('/api/tasks',json={**body,'idempotency_key':str(uuid4())}).status_code,409)
        self.ts.complete(tid,TaskStatus.cancelled,TaskResult(answer_markdown=d.markdown,draft_id=d.draft_id))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _:ContinuationService(self.db).release_ready(),range(2)))
        row=self.db.fetchone('SELECT * FROM task_continuations WHERE predecessor_id=?',(tid,))
        self.assertEqual(row['status'],'queued')
        self.assertEqual(self.client.post(f"/api/tasks/{row['next_task_id']}/continue",json=body).status_code,409)
        self.assertEqual(self.db.get_task(row['next_task_id'])['save_policy'],'deny')
        self.assertEqual(len(self.db.fetchall('SELECT * FROM tasks WHERE idempotency_key=?',(body['idempotency_key'],))),1)
        self.assertIn('已完成部分',str(ConversationService(self.db).view(self.cid)['messages']))

    def test_continuation_transaction_rolls_back_and_cancel_wins(self):
        """新任务与接续状态同事务；提交中断无半条任务，取消后不复活。"""
        from research_loop.services.continuations import ContinuationService
        from research_loop.storage.memory import Transaction
        tid=self.create();self.db.update_task(tid,status='running')
        body={'conversation_id':self.cid,'prompt':'继续','idempotency_key':str(uuid4())}
        self.client.post(f'/api/tasks/{tid}/continue',json=body)
        self.ts.complete(tid,TaskStatus.cancelled,TaskResult(answer_markdown='保留结果'))
        original=Transaction.execute
        def fail(tx,sql,params=()):
            if sql.startswith("UPDATE task_continuations SET status='queued'"):raise RuntimeError('CONTROLLED_COMMIT_INTERRUPTION')
            return original(tx,sql,params)
        with patch.object(Transaction,'execute',fail):
            with self.assertRaisesRegex(RuntimeError,'CONTROLLED'):ContinuationService(self.db).release_ready()
        self.assertIsNone(self.db.fetchone('SELECT task_id FROM tasks WHERE idempotency_key=?',(body['idempotency_key'],)))
        self.ts.cancel(tid)
        ContinuationService(self.db).release_ready()
        self.assertIsNone(self.db.fetchone('SELECT task_id FROM tasks WHERE idempotency_key=?',(body['idempotency_key'],)))
        self.assertEqual(self.db.fetchone('SELECT status FROM task_continuations WHERE predecessor_id=?',(tid,))['status'],'cancelled')

    def test_continuation_recovers_cancel_gap_and_stops_queued_next(self):
        """持久化后、通知取消前故障可恢复；晚到停止信号取消已入队接续。"""
        from research_loop.services.continuations import ContinuationService
        tid=self.create();body=TaskCreate(conversation_id=self.cid,prompt='继续但不保存',idempotency_key=str(uuid4()))
        with patch.object(TaskService,'cancel',side_effect=RuntimeError('CONTROLLED_CANCEL_GAP')):
            with self.assertRaises(RuntimeError):ContinuationService(self.db).request(tid,body)
        ContinuationService(self.db).release_ready()
        row=self.db.fetchone('SELECT * FROM task_continuations WHERE predecessor_id=?',(tid,))
        self.assertEqual(self.db.get_task(tid)['status'],'cancelled');self.assertEqual(row['status'],'queued')
        self.ts.cancel(tid)
        self.assertEqual(self.db.get_task(row['next_task_id'])['status'],'cancelled')
        self.assertEqual(self.db.list_model_calls(row['next_task_id']),[])

    def test_stream_disconnect_unknown_is_not_resent(self):
        """供应商断流后原请求保留unknown；恢复不追加网络请求。"""
        from tests.integration.test_memory import MemoryProvider
        class Broken(MemoryProvider):
            """受控流式故障，不真实访问供应商。"""
            def send_stream(self,*args,on_public,**kwargs):
                self.calls.append('sent');on_public({'text':'未完成片段','stream_key':'answer','provisional':True})
                raise RuntimeError('stream disconnected')
        p=Broken(None);tid=self.create('请直接回答，不保存')
        executor=TaskExecutor(self.s,self.db,lambda _:p)
        executor.execute(tid)
        self.assertEqual(len(p.calls),1)
        self.assertEqual(self.ts.view(self.db.get_task(tid)).usage['unknown'],1)
        self.db.update_task(tid,status='queued',claimed_by=None,lease_token=None,lease_until=None)
        executor.execute(tid)
        self.assertEqual(len(p.calls),1)


class StreamingTests(unittest.TestCase):
    """真实SSE协议解析，覆盖转义、usage、隐私及断流。"""
    def test_fragmented_tool_text(self):
        """碎片化转义在补齐后公开正文；推理不进入发布内容。"""
        out=[];a=StreamAssembly(out.append)
        args=json.dumps({'answer_markdown':'结论\n公式 \\(x=1\\)','goals':[]},ensure_ascii=True)
        a.feed({'choices':[{'delta':{'reasoning_content':'PRIVATE','tool_calls':[{'index':0,'id':'c','function':{'name':'finish_task','arguments':''}}]}}]})
        for c in args:a.feed({'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':c}}]}}]})
        a.feed({'usage':{'total_tokens':20},'choices':[{'delta':{},'finish_reason':'tool_calls'}]})
        raw=a.response();self.assertEqual(json.loads(raw['choices'][0]['message']['tool_calls'][0]['function']['arguments'])['answer_markdown'],'结论\n公式 \\(x=1\\)')
        self.assertEqual(out[-1]['text'],'结论\n公式 \\(x=1\\)');self.assertNotIn('PRIVATE',json.dumps(out))

    def test_missing_usage_fails(self):
        """finish并不等于完整收据，缺usage不能填零。"""
        a=StreamAssembly(lambda p:None);a.feed({'choices':[{'delta':{'content':'x'},'finish_reason':'stop'}]})
        with self.assertRaisesRegex(RuntimeError,'usage'):a.response()

    def test_provider_real_sse_contract_and_disconnect(self):
        """供应商SSE替身验证stream_options并装配usage，断流不能返回成功。"""
        chunks=[{'id':'r1','choices':[{'index':0,'delta':{'content':'真正的流式片段'}}]}, {'id':'r1','choices':[{'index':0,'delta':{},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':2,'total_tokens':3}}]
        def handle(request):
            self.assertTrue(json.loads(request.content)['stream_options']['include_usage'])
            data=''.join('data: '+json.dumps(c,ensure_ascii=False)+'\n\n' for c in chunks)
            return httpx.Response(200,text=data+'data: [DONE]\n\n')
        out=[];client=httpx.Client(transport=httpx.MockTransport(handle));self.addCleanup(client.close)
        p=DeepSeekProvider('fake',client=client)
        raw=p.send_stream([{'role':'user','content':'x'}],on_public=out.append)
        self.assertEqual(raw['usage']['total_tokens'],3);self.assertEqual(out[-1]['text'],'真正的流式片段')
        broken=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,text='data: '+json.dumps(chunks[0])+'\n\n')));self.addCleanup(broken.close)
        with self.assertRaisesRegex(RuntimeError,'disconnected'):
            DeepSeekProvider('fake',client=broken).send_stream([{'role':'user','content':'x'}],on_public=lambda p:None)
