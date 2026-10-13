"""AG4真实接口/Worker合同；默认SQLite，显式配置后运行隔离MySQL，模型用可控替身。"""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.domain.tasks import TaskCreate, TaskStatus, TaskResult, SavePolicy, ResumeInput
from research_loop.domain.memory import RememberInput, ProfileSet, NoteUpdate, HistoryRead
from research_loop.domain.notes import CreateNoteInput
from research_loop.storage.mysql import Database
from research_loop.storage.memory import dump
from research_loop.storage.files import FileStore
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.services.tasks import TaskService
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.services.notes import NoteService
from research_loop.services.note_versions import NoteVersionService
from research_loop.agents.memory import ContextService, PREFIX, estimate
from research_loop.agents.graph import initial_state
from research_loop.providers.deepseek import validate_messages
from research_loop.workers.executor import TaskExecutor


class MemoryProvider:
    """根据持久观察选择下一动作；记录真实生产图传入messages。"""
    def __init__(self, quote='以后请默认用中文回答科研问题。'):
        self.quote=quote
        self.calls=[]

    def close(self):
        """替身无网络资源。"""

    def send(self,messages,tools,force_text=False,**kwargs):
        """依次保存偏好、建立记忆计划、显式结束；不生成假论文事实。"""
        self.calls.append(copy.deepcopy(messages))
        done=[c['function']['name'] for m in messages for c in m.get('tool_calls',[])]
        if 'remember_preference' not in done and self.quote:
            name,args='remember_preference',{'slot':'language-'+self.quote[-5:],'kind':'explicit','content':'中文','source_quote':self.quote}
        elif 'get_profile' not in done and not self.quote:
            name,args='get_profile',{}
        elif 'set_research_plan' not in done:
            name,args='set_research_plan',{'goal':'记忆管理','task_kind':'memory','goals':[{'goal_id':'memory','description':'保存或回顾偏好','min_papers':0}]}
        else:
            name,args='finish_task',{'answer_markdown':'根据实际记录，默认中文；当前任务不保存文件。','goals':[{'goal_id':'memory','status':'satisfied','reason':'已实际读写记忆'}]}
        return {'id':'fixture-'+str(len(self.calls)),'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15},'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':'','tool_calls':[{'id':'c'+str(len(self.calls)),'type':'function','function':{'name':name,'arguments':dump(args)}}]}}]}


class MemoryTests(unittest.TestCase):
    """验证会话、长期偏好、压缩、文件版本和模型收据恢复。"""
    def setUp(self):
        """每例使用唯一身份及临时文件，MySQL只允许显式隔离库。"""
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        url=os.getenv('AG4_TEST_DATABASE_URL') or 'sqlite:///'+str(root/'test.db')
        if not url.startswith('sqlite:') and not url.rsplit('/',1)[-1].startswith('researchloop_test_ag4_'):
            raise ValueError('AG4 isolated database required')
        self.settings=Settings(project_root=root,data_dir=root/'data',output_dir=root/'notes',database_url=url,max_model_requests=6,reserve_model_requests=0)
        self.db=Database(url);self.db.ensure_schema()
        self.conv=ConversationService(self.db)
        self.cid=self.conv.create('AG4测试',str(uuid4()))['conversation_id']
        self.tasks=TaskService(self.db)
        self.files=FileStore(root/'data',root/'notes',root)
        self.notes=NoteVersionService(self.db,self.files)
        self.memory=MemoryService(self.db,self.notes)
        self.context=ContextService(self.db,self.memory,self.settings)
        self.prefix=uuid4().hex

    def task(self,prompt='测试记忆',cid=None,policy='deny'):
        """创建独立根任务，权限只取当前参数。"""
        return self.tasks.create(TaskCreate(prompt=prompt,conversation_id=cid or self.cid,idempotency_key=str(uuid4()),save_policy=policy)).task_id

    def done(self,tid,answer='已完成历史研究'):
        """合成任务终态用于派生投影测试，不冒充真实模型完成研究。"""
        self.tasks.complete(tid,TaskStatus.completed,TaskResult(answer_markdown=answer))

    def test_conversation_busy_idempotency_and_permission(self):
        """同会话不交叉运行，新一轮不继承allow。"""
        payload=TaskCreate(prompt='本轮保存',conversation_id=self.cid,idempotency_key=self.prefix,save_policy='allow')
        a=self.tasks.create(payload).task_id
        self.assertEqual(a,self.tasks.create(payload).task_id)
        with self.assertRaisesRegex(ValueError,'CONVERSATION_BUSY'):self.task()
        self.done(a)
        b=self.task()
        self.assertEqual(self.db.get_task(b)['save_policy'],'deny')
        self.assertEqual(len(self.conv.view(self.cid)['messages']),3)
        self.conv.view(self.cid)
        self.assertEqual(len(self.conv.view(self.cid)['messages']),3)

    def test_atomic_conversation_submission(self):
        """两连接并发提交只有一个根任务；另一个明确忙。"""
        def submit(i):
            try:return self.task('并发'+str(i))
            except ValueError:return None
        with ThreadPoolExecutor(2) as pool:results=list(pool.map(submit,[1,2]))
        self.assertEqual(sum(r is not None for r in results),1)

    def test_source_version_revocation_and_replay(self):
        """撤销后成功旧调用不能复活；伪造来源和过期版本拒绝。"""
        tid=self.task('以后默认中文，长期关注目标检测。')
        args=RememberInput(slot=self.prefix,kind='explicit',content='中文',source_quote='以后默认中文')
        entry=self.memory.remember(tid,args)
        self.memory.forget(entry['memory_id'],1)
        self.assertEqual(self.memory.remember(tid,args)['status'],'revoked')
        with self.assertRaisesRegex(ValueError,'MEMORY_SOURCE'):
            self.memory.remember(tid,args.model_copy(update={'source_quote':'我喜欢火星'}))
        with self.assertRaisesRegex(ValueError,'MEMORY_STALE'):
            self.memory.set_profile(ProfileSet(slot=self.prefix,content='English',idempotency_key=str(uuid4())))
        new=self.memory.set_profile(ProfileSet(slot=self.prefix,content='English',expected_version=1,idempotency_key=str(uuid4())))
        self.assertEqual(new['version'],2)
        with self.assertRaisesRegex(ValueError,'MEMORY_PRIORITY'):
            self.memory.remember(tid,args.model_copy(update={'kind':'inferred','expected_version':2}))

    def test_inference_separate_and_revocable(self):
        """推断有来源标签，不写入显式偏好，撤销后不用于提示。"""
        tid=self.task('这次想了解开放词汇目标检测')
        entry=self.memory.remember(tid,RememberInput(slot=self.prefix,kind='inferred',content='开放词汇目标检测',source_quote='开放词汇目标检测'))
        self.assertIn(entry,self.memory.profile()['inferred_interests'])
        self.memory.forget(entry['memory_id'],1)
        self.assertNotIn(entry['memory_id'],dump(self.context.history(tid)))

    def test_real_api_worker_receipts_and_cross_conversation(self):
        """真实API→DB→Worker；替身只代替供应商，消息注入和工具写入真实执行。"""
        from research_loop.main import app
        quote='以后请默认用中文回答科研问题。'+self.prefix
        provider=MemoryProvider(quote)
        with patch('research_loop.api.tasks.get_settings',return_value=self.settings), TestClient(app) as client:
            tid=client.post('/api/tasks',json={'prompt':quote,'conversation_id':self.cid,'save_policy':'deny'}).json()['task_id']
            TaskExecutor(self.settings,self.db,lambda s:provider).execute(tid)
            result=client.get('/api/tasks/'+tid).json()
            self.assertEqual(result['status'],'completed',result)
            self.assertEqual(len(provider.calls),3)
            self.assertTrue(any(PREFIX in m.get('content','') for m in provider.calls[0]))
            self.assertEqual(len(self.db.list_model_calls(tid)),3)
            self.assertEqual(client.get('/api/conversations/'+self.cid).json()['total'],2)
            # 同内容成功收据回放：回退checkpoint计数，不回退记忆和action业务事实。
            state=initial_state(tid,quote,'deny',max_model_requests=6,reserve_model_requests=0)
            state['coverage_required']=True
            CheckpointStore(self.db).save(tid,state)
            self.db.update_task(tid,status='queued',claimed_by=None,lease_token=None,lease_until=None)
            TaskExecutor(self.settings,self.db,lambda s:provider).execute(tid)
            self.assertEqual(len(provider.calls),3)
            other=client.post('/api/conversations',json={'title':'新会话','idempotency_key':str(uuid4())}).json()['conversation_id']
            next_id=self.task('继续讨论科研问题，本次英文',cid=other)
            p2=MemoryProvider(None)
            TaskExecutor(self.settings,self.db,lambda s:p2).execute(next_id)
            self.assertIn('中文',dump(p2.calls[0]))
            self.assertIn('本次英文',dump(p2.calls[0]))
            self.assertFalse(list(self.settings.output_dir.rglob('*.md')))

    def test_projection_rebuild_does_not_duplicate(self):
        """终态后投影中断可补齐，重复读回不增加消息或研究统计。"""
        tid=self.task('目标检测历史')
        self.db.update_task(tid,status='partial',result_json=dump({'answer_markdown':'保留已有发现','gaps':['缺PDF']}))
        self.conv.view(self.cid);self.conv.view(self.cid)
        self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM research_memories WHERE task_id=?',(tid,))['n'],1)
        self.assertEqual(self.conv.view(self.cid)['total'],2)
        match=self.memory.search('目标检测历史','research_history',10)
        self.assertIn(tid,[r['source_id'] for r in match])
        self.assertEqual(next(r for r in match if r['source_id']==tid)['record']['status'],'partial')

    def test_compression_pairs_archive_and_readback(self):
        """长工具观察压缩后配对仍合法；原文可读回，当前指令不变。"""
        tid=self.task('只比较两篇，不保存，保留物理页码')
        s=self.settings.model_copy(update={'context_window_tokens':18000,'context_output_reserve':1000,'context_safety_tokens':500})
        ctx=ContextService(self.db,self.memory,s)
        messages=[{'role':'system','content':'system'},{'role':'user','content':self.db.get_task(tid)['prompt']}]
        for i in range(3):
            messages += [{'role':'assistant','content':'','tool_calls':[{'id':str(i),'type':'function','function':{'name':'read_paper','arguments':'{}'}}]}, {'role':'tool','tool_call_id':str(i),'content':'目标检测正文 abc '*2000 if i<2 else '近期观察'}]
        projected,metrics=ctx.prepare({'task_id':tid,'prompt':'只比较两篇，不保存','model_requests':2},messages,[])
        validate_messages(projected)
        self.assertLess(metrics['after'],metrics['before'])
        self.assertEqual(projected[2],messages[1])
        self.assertEqual(projected[-1],messages[-1])
        archived=self.memory.read(tid,HistoryRead(source_type='context_archive',source_id=metrics['snapshot_id']))
        self.assertIn('目标检测正文',archived['content'])
        self.assertFalse(archived['is_original_paper_evidence'])
        self.assertIsNotNone(archived['next_offset'])
        before=self.db.fetchone('SELECT COUNT(*) AS n FROM context_archives WHERE task_id=?',(tid,))['n']
        ctx.prepare({'task_id':tid,'prompt':'只比较两篇，不保存'},messages,[])
        self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM context_archives WHERE task_id=?',(tid,))['n'],before)

    def test_latest_memory_page_shrinks_with_exact_readback_cursor(self):
        """最新记忆页超限时前移字符游标；工具配对/推理/原件完整保留，可无缝补读。"""
        old=self.task('历史材料')
        content='目标检测历史与条件。'*2000
        self.done(old,content)
        tid=self.task('请继续历史材料，不保存')
        source=self.db.fetchone("SELECT message_id FROM conversation_messages WHERE task_id=? AND role='assistant'",(old,))['message_id']
        args=HistoryRead(source_type='message',source_id=source,offset=100,limit=16000)
        observation={'ok':True,'data':self.memory.read(tid,args)}
        assistant={'role':'assistant','content':'','reasoning_content':'供应商既有推理字段',
                   'tool_calls':[{'id':'read1','type':'function','function':{'name':'read_memory','arguments':dump(args.model_dump())}}]}
        messages=[{'role':'system','content':'system'},{'role':'user','content':'请继续历史材料，不保存'},assistant,
                  {'role':'tool','tool_call_id':'read1','content':dump(observation)}]
        original=copy.deepcopy(messages)
        settings=self.settings.model_copy(update={'context_window_tokens':10000,'context_output_reserve':1000,'context_safety_tokens':500})
        ctx=ContextService(self.db,self.memory,settings)
        # 本例只隔离单轮工具投影；真实多轮历史仍由另一个测试覆盖。
        with patch.object(ctx,'history',return_value={}):
            projected,metrics=ctx.prepare({'task_id':tid,'prompt':'继续历史材料'},messages,[])
        validate_messages(projected)
        page=json.loads(projected[-1]['content'])['data']
        self.assertLessEqual(metrics['after'],metrics['input_capacity'])
        self.assertTrue(metrics['memory_pages_shortened'])
        self.assertEqual(projected[-2],assistant)
        self.assertEqual(messages,original)
        self.assertEqual(page['content_hash'],observation['data']['content_hash'])
        self.assertEqual(page['next_offset'],args.offset+len(page['content']))
        tail=self.memory.read(tid,args.model_copy(update={'offset':page['next_offset']}))
        self.assertEqual(page['content']+tail['content'],content[args.offset:page['next_offset']+len(tail['content'])])
        archive=self.db.fetchone('SELECT original_json FROM context_archives WHERE snapshot_id=?',(metrics['snapshot_id'],))
        self.assertEqual(json.loads(archive['original_json'])['messages'],original)
        paper_messages=copy.deepcopy(original)
        paper_messages[-2]['tool_calls'][0]['function']['name']='read_paper'
        with patch.object(ctx,'history',return_value={}):
            with self.assertRaisesRegex(ValueError,'CONTEXT_CAPACITY'):
                ctx.prepare({'task_id':tid,'prompt':'继续历史材料'},paper_messages,[])
        self.assertEqual(paper_messages[-1],original[-1])

    def test_uncompressible_request_stops_before_model(self):
        """必要输入过大时留下可读原因，零供应商请求。"""
        tid=self.task('需要保留的当前指令 '*700)
        s=self.settings.model_copy(update={'context_window_tokens':4096,'context_output_reserve':1000,'context_safety_tokens':500})
        p=MemoryProvider(None)
        TaskExecutor(s,self.db,lambda s:p).execute(tid)
        result=self.tasks.view(self.db.get_task(tid))
        self.assertEqual(len(p.calls),0)
        self.assertIn('CONTEXT_CAPACITY',dump(result.model_dump(mode='json')))
        self.assertTrue(result.result.answer_markdown)

    def test_history_compression_preserves_identity_and_recent_messages(self):
        """较早消息有稳定ID可回读，近期消息和新限制保持。"""
        for i in range(7):
            tid=self.task('历史输入'+str(i)+' detector '*250)
            self.done(tid,'历史输出'+str(i)+' analysis '*250)
        tid=self.task('第二篇论文是什么？这次禁止保存')
        s=self.settings.model_copy(update={'memory_history_tokens':4000})
        context=ContextService(self.db,self.memory,s).history(tid)
        self.assertTrue(context['older_messages_omitted'])
        self.assertEqual(len(context['messages']),4)
        self.assertTrue(all(not x['excerpted'] for x in context['messages']))
        args=HistoryRead(source_type='conversation',source_id=self.cid)
        self.assertIn('历史输入0',self.memory.read(tid,args)['content'])
        with self.assertRaises(KeyError):
            self.memory.read(tid,args.model_copy(update={'source_id':'foreign'}))

    def test_projection_order_after_crash_window(self):
        """旧结果尚未投影时提交新消息，旧结果必须排在新消息前。"""
        old=self.task('上一轮')
        self.db.update_task(old,status='completed',result_json=dump({'answer_markdown':'上一轮答案'}))
        current=self.task('追问上一轮')
        messages=self.conv.view(self.cid)['messages']
        self.assertEqual([m['content'] for m in messages],['上一轮','上一轮答案','追问上一轮'])
        self.assertIn('上一轮答案',dump(self.context.history(current)))

    def test_lost_lease_cannot_write_memory(self):
        """Worker失权或取消后，Memory写入在同一事务拒绝。"""
        tid=self.task('长期偏好中文')
        claimed=self.db.claim_task('worker',task_id=tid)
        service=MemoryService(self.db,actor=(tid,'worker',claimed['lease_token']))
        args=RememberInput(slot=self.prefix,kind='explicit',content='中文',source_quote='长期偏好中文')
        self.db.update_task(tid,cancel_requested=1)
        with self.assertRaisesRegex(PermissionError,'MEMORY_LEASE_LOST'):service.remember(tid,args)
        self.assertIsNone(self.db.fetchone('SELECT * FROM memory_entries WHERE slot=?',(self.prefix,)))
        self.db.update_task(tid,cancel_requested=0,lease_until='2000-01-01T00:00:00+00:00')
        with self.assertRaisesRegex(PermissionError,'MEMORY_LEASE_LOST'):service.remember(tid,args)

    def test_profile_counts_distinct_completed_deep_reading(self):
        """相同论文重跑不重复计数，普通阅读和partial不计完整精读。"""
        parent=self.task()
        before=self.memory.profile()['statistics']['deep_read_distinct_papers']
        for i,(mode,status) in enumerate([('deep_read','complete'),('deep_read','complete'),('investigate','complete'),('deep_read','partial')]):
            child=self.db.create_subtask(parent,parent,i,'合成计数案例',{},self.prefix+str(i))
            self.db.save_subtask_result(child['subtask_id'],'completed',{'mode':mode,'deep_reading_status':status,'paper_ids':[self.prefix if i<2 else self.prefix+str(i)]})
        self.assertEqual(self.memory.profile()['statistics']['deep_read_distinct_papers'],before+1)

    def test_clarification_remains_original_task_and_prompt(self):
        """会话中的澄清仍走原任务；不能用单独新任务替代最初指令。"""
        tid=self.task('先询问数量，近三年目标检测，禁止保存')
        question=self.tasks.request_clarification(tid,'比较几篇？',['count'],['5'])
        state=initial_state(tid,self.db.get_task(tid)['prompt'],'deny')
        state.update(final_reason='clarification requested',finished=True)
        CheckpointStore(self.db).save(tid,state)
        self.db.update_task(tid,status='needs_input')
        with self.assertRaisesRegex(ValueError,'CONVERSATION_BUSY'):self.task('5篇')
        self.tasks.resume(ResumeInput(clarification_id=question['clarification_id'],answer='5篇',idempotency_key=self.prefix),tid)
        messages=CheckpointStore(self.db).load(tid)['messages']
        self.assertIn('近三年目标检测',dump(messages))
        self.assertIn('5篇',dump(messages))
        self.assertEqual(self.db.get_task(tid)['save_policy'],'deny')
        self.assertEqual(len(self.conv.view(self.cid)['messages']),3)

    def saved_note(self):
        """创建明确标记的合成笔记，仅测试版本与文件，不宣称研究质量。"""
        tid=self.task(policy='allow')
        notes=NoteService(self.db,self.files)
        draft=notes.create(tid,CreateNoteInput(title='合成目标检测笔记',markdown='# 基线\n用户原有内容'))
        receipt=notes.save(tid,SavePolicy.allow,draft.draft_id,draft.content_hash)
        self.done(tid)
        return draft,receipt

    def test_note_preview_version_conflict_and_source(self):
        """预览零写入、新版本不覆盖旧文、冲突和过期阻断。"""
        draft,receipt=self.saved_note()
        args=NoteUpdate(base_hash=draft.content_hash,markdown=draft.markdown+'\n用户明确的新补充',idempotency_key=self.prefix)
        preview=self.notes.update(draft.draft_id,args,False)
        self.assertFalse(preview['saved'])
        with self.assertRaises(PermissionError):self.notes.update(draft.draft_id,args.model_copy(update={'confirm':True}),False)
        args=args.model_copy(update={'confirm':True})
        saved=self.notes.update(draft.draft_id,args,True)
        self.assertEqual(Path(receipt.path).read_text(),draft.markdown)
        self.assertEqual(self.notes.update(draft.draft_id,args,True),saved)
        self.assertEqual(len(self.notes.versions(draft.draft_id)),2)
        self.assertEqual(self.notes.get(draft.draft_id)['review_status'],'not_inherited')
        with self.assertRaisesRegex(ValueError,'NOTE_STALE'):
            self.notes.update(draft.draft_id,args.model_copy(update={'idempotency_key':'new'}),True)
        result=self.memory.search('用户明确的新补充','user_note',10)
        self.assertIn(saved['content_hash'],dump(result))
        self.assertNotIn('research_history',[r['source_type'] for r in result])
        Path(saved['path']).write_text('用户外部修改')
        with self.assertRaisesRegex(ValueError,'NOTE_CONFLICT'):self.notes.update(draft.draft_id,args,True)

    def test_note_file_before_commit_recovery(self):
        """文件写入后事务失败，重试同键复用同hash文件，只有两个版本。"""
        draft,receipt=self.saved_note()
        args=NoteUpdate(base_hash=draft.content_hash,markdown='# 新版本',idempotency_key=self.prefix,confirm=True)
        from research_loop.services import note_versions
        with patch.object(note_versions,'record_operation',side_effect=RuntimeError('after file before commit')):
            with self.assertRaises(RuntimeError):self.notes.update(draft.draft_id,args,True)
        saved=self.notes.update(draft.draft_id,args,True)
        self.assertEqual(len(self.notes.versions(draft.draft_id)),2)
        self.assertEqual(Path(saved['path']).read_text(),'# 新版本')
        self.assertEqual(Path(receipt.path).read_text(),draft.markdown)

    def test_cancel_and_unknown_no_model(self):
        """取消和未结算收据保护不因Memory入口而改变。"""
        tid=self.task()
        self.tasks.cancel(tid)
        p=MemoryProvider(None)
        TaskExecutor(self.settings,self.db,lambda s:p).execute(tid)
        self.assertEqual(len(p.calls),0)
        other=self.task()
        self.db.begin_model_call(other,1,{'messages':[],'tools':[]})
        self.db.mark_model_call_sent(other,1)
        TaskExecutor(self.settings,self.db,lambda s:p).execute(other)
        self.assertEqual(len(p.calls),0)
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))


if __name__=='__main__':
    unittest.main()
