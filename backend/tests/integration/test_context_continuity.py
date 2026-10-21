"""P0接续验证：生产API/Worker/存储，替身模型与合成原文，支持独立MySQL。"""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.agents.memory import ContextService, PREFIX
from research_loop.agents.graph import initial_state
from research_loop.domain.memory import HistoryRead
from research_loop.domain.notes import CreateNoteInput
from research_loop.domain.tasks import ActionRecord, TaskCreate, ResumeInput
from research_loop.providers.deepseek import validate_messages
from research_loop.services.conversations import ConversationService
from research_loop.services.coverage import CoverageService
from research_loop.services.memory import MemoryService
from research_loop.services.notes import NoteService
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.files import FileStore
from research_loop.storage.memory import dump
from research_loop.storage.mysql import Database
from research_loop.tools.memory import specs as memory_specs
from research_loop.workers.executor import TaskExecutor
from integration.test_collaboration_contract import FixturePDF


def memory_payload(messages):
    """提取实际供应商请求中的系统投影，不查看服务内部缓存。"""
    item=next(m['content'] for m in messages if m.get('role')=='system' and m.get('content','').startswith(PREFIX))
    return json.loads(item[item.index('\n{')+1:])


def exchange(name, args, data, call_id='read-source'):
    """生成完整供应商往返；data来自真实服务的合成PDF读回。"""
    return [{'role':'assistant','content':'','reasoning_content':'原推理字段保留',
             'tool_calls':[{'id':call_id,'type':'function','function':{'name':name,'arguments':dump(args)}}]},
            {'role':'tool','tool_call_id':call_id,'content':dump({'ok':True,'data':data})}]


class LongPDF(FixturePDF):
    """仅扩展合成正文长度触发真实压缩，不调用网络或模型。"""
    def pages(self, document, start, count):
        """沿用稳定页码/身份，正文包含需要保留的句子。"""
        pages=super().pages(document,start,count)
        for page in pages:
            page.text += '\nEXACT SOURCE sentence preserves language conditions. '*140
            page.parsed_markdown=page.text
        return pages


class ContinuityProvider:
    """分步返回固定行动，从真实当前任务投影读取目标和稿件。"""
    def __init__(self, readback=False):
        """是否在本例显式调用档案工具；记录每次真实传入的messages。"""
        self.readback=readback
        self.calls=[]

    def close(self):
        """替身没有网络资源。"""

    def send(self,messages,tools,**kwargs):
        """首读/补读/稿件/交付实际执行，供应商输出为可复现替身。"""
        self.calls.append(copy.deepcopy(messages))
        current=memory_payload(messages)['current_task']
        n=len(self.calls)
        if n==1:
            calls=[('set_research_plan',{'goal':'比较语言信息','task_kind':'compare','requested_count':2,'count_confirmed':True,
                    'goals':[{'goal_id':'language-comparison','description':'对照两份原文','min_papers':2,'requires_fulltext':True}]}),
                   ('read_paper',{'target_ref':'input-A'})]
        elif n==2:
            calls=[('read_paper',{'target_ref':'input-B'})]
        elif n==3 and self.readback:
            archive=next(json.loads(m['content']) for m in messages if m.get('role')=='tool' and 'archived_tool_result' in m.get('content',''))
            calls=[('read_memory',archive['readback'])]
        elif not current['current_draft']:
            calls=[('create_note',{'title':'合成比较','markdown':'# 已取得的比较\n两篇均提供语言信息，具体条件见原文。',
                    'evidence_ids':[e['evidence_id'] for e in current['authorized_evidence']]})]
        else:
            calls=[('finish_task',{'answer_markdown':'两份合成材料比较已完成，禁止保存。',
                    'goals':[{'goal_id':current['research_plan']['goals'][0]['goal_id'],'status':'satisfied','reason':'按实际两份原文完成'}]})]
        return {'id':'p0-stub-'+str(n),'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15},
                'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':'',
                 'tool_calls':[{'id':f'{n}-{i}','type':'function','function':{'name':name,'arguments':dump(args)}} for i,(name,args) in enumerate(calls)]}}]}


class ContextContinuityTests(unittest.TestCase):
    """不改生产数据库；分别验证接续、不可读回出口和成功收据恢复。"""
    def setUp(self):
        """每例唯一会话与临时目录；MySQL必须显式使用P0隔离库。"""
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        url=os.getenv('P0_TEST_DATABASE_URL') or 'sqlite:///'+str(self.root/'test.db')
        if not url.startswith('sqlite:') and not url.rsplit('/',1)[-1].startswith('researchloop_test_p0_'):
            raise ValueError('P0 isolated MySQL required')
        self.settings=Settings(project_root=self.root,data_dir=self.root/'data',output_dir=self.root/'notes',database_url=url,
                               retrieval_enabled=False,max_model_requests=5,reserve_model_requests=0,context_compress_ratio=0.1)
        self.db=Database(url);self.db.ensure_schema()
        self.conv=ConversationService(self.db,self.settings)
        self.cid=self.conv.create('P0接续',uuid4().hex)['conversation_id']
        self.tasks=TaskService(self.db,self.settings)
        self.files=FileStore(self.settings.data_dir,self.settings.output_dir,self.root)
        self.memory=MemoryService(self.db)
        self.context=ContextService(self.db,self.memory,self.settings)
        self.tools=[s.schema() for s in memory_specs() if s.name=='read_memory']

    def task(self, cid=None, client=None, prompt='比较附件两篇，不保存', settings=None):
        """提交固定材料与禁止保存，新一轮不继承文件权限。"""
        attachments=[]
        for name in ('A','B'):
            path=self.root/(name+'.pdf');path.write_bytes(b'%PDF-fixture')
            attachments.append({'input_ref':'input-'+name,'paper_id':'paper-'+name,'local_path':str(path)})
        payload=TaskCreate(prompt=prompt,conversation_id=cid or self.cid,
                    idempotency_key=uuid4().hex,attachments=attachments,save_policy='deny')
        if client:
            response=client.post('/api/tasks',json=payload.model_dump(mode='json'))
            self.assertEqual(response.status_code,202,response.text)
            return response.json()['task_id']
        return (TaskService(self.db,settings) if settings else self.tasks).create(payload).task_id

    def test_named_attachment_scope_through_read_service(self):
        """完整用户指令进入服务后读到指定页；工具参数不能扩张用户范围。"""
        tid=self.task(prompt='仅使用附件 Alpha 与 Beta 的物理第1至2页，比较语言信息，不保存。')
        papers=PaperService(self.db,None,LongPDF())
        attachments={a['input_ref']:a for a in json.loads(self.db.get_task(tid)['attachments_json'])}
        # 分材料合同要求真实名称映射，不能由两个不相干input_ref猜Alpha/Beta。
        attachments['input-A']['filename'] = 'Alpha.pdf'
        attachments['input-B']['filename'] = 'Beta.pdf'
        self.db.execute('UPDATE tasks SET attachments_json=? WHERE task_id=?', (dump(list(attachments.values())), tid))
        for ref in attachments:
            data=papers.read(tid,ref,start_page=1,page_count=2,attachments=attachments)
            self.assertEqual(data['returned_pages'],[1,2])
            self.assertEqual(data['read_coverage']['scope_origin'],'user')
            with self.assertRaises(PermissionError):
                papers.read(tid,ref,start_page=1,page_count=3,attachments=attachments)
        self.assertEqual(len(self.db.list_task_evidence(tid)),4)
        self.assertEqual(self.db.list_model_calls(tid),[])

    def source_messages(self,tid):
        """生成真实服务的页证据与多组观察，持久计划从实际行动日志读回。"""
        papers=PaperService(self.db,None,LongPDF())
        attachments={a['input_ref']:a for a in json.loads(self.db.get_task(tid)['attachments_json'])}
        data=papers.read(tid,'input-A',attachments=attachments)
        plan={'goal':'对照原文','goals':[{'goal_id':'exact-goal-id','description':'比较两份','min_papers':2}], 'save_policy':'deny'}
        action=ActionRecord(task_id=tid,sequence=1,tool_name='set_research_plan',arguments={},status='succeeded',
                            observation={'ok':True,'data':{'research_plan':plan}})
        self.db.save_action(action.model_dump(mode='json'))
        messages=[{'role':'system','content':'system'},{'role':'user','content':'比较两篇，不保存'}]
        messages+=exchange('set_research_plan',{}, {'research_plan':plan},'plan')
        messages+=exchange('read_paper',{'target_ref':'input-A'},data)
        messages+=exchange('report_task_gap',{}, {'gap':'限定已读页'},'latest')
        return messages,data

    def test_authoritative_goals_and_archive_offsets(self):
        """压缩保留真实目标，offset精确读回原工具组，旧checkpoint不能扩大权限。"""
        tid=self.task();messages,data=self.source_messages(tid)
        original=copy.deepcopy(messages)
        state={'task_id':tid,'prompt':'伪旧目标','save_policy':'allow','research_plan':{'goals':[{'goal_id':'stale'}]}}
        projected,metrics=self.context.prepare(state,messages,self.tools)
        validate_messages(projected)
        current=memory_payload(projected)['current_task']
        self.assertEqual(current['research_plan']['goals'][0]['goal_id'],'exact-goal-id')
        self.assertEqual(current['save_policy'],'deny')
        self.assertTrue(metrics['compressed_groups'])
        summary=next(json.loads(m['content']) for m in projected if m.get('role')=='tool' and 'archived_tool_result' in m.get('content','') and 'read-source' in m['content'])
        read=self.memory.read(tid,HistoryRead(**summary['readback']))
        self.assertTrue(read['content'].startswith(dump(messages[4])))
        self.assertFalse(read['is_original_paper_evidence'])
        self.assertEqual(messages,original)
        self.assertEqual(len(current['authorized_evidence']),2)

    def test_no_readback_retains_source_or_projects_exact_windows(self):
        """禁用档案工具时超大观察投影为定位原句，不建议不存在的工具。"""
        tid=self.task();messages,data=self.source_messages(tid)
        projected,metrics=self.context.prepare({'task_id':tid,'prompt':'比较'},messages,[])
        self.assertIn(messages[5],projected)
        self.assertNotIn('分页read_memory',dump(projected))
        limits=self.settings.model_copy(update={'context_window_tokens':12000,'context_output_reserve':1000,'context_safety_tokens':500})
        smaller_cid=self.conv.create('固定小容量会话',uuid4().hex)['conversation_id']
        tid=self.task(cid=smaller_cid,settings=limits);messages,data=self.source_messages(tid)
        small=ContextService(self.db,self.memory,limits)
        projected, metrics = small.prepare({'task_id':tid,'prompt':'比较'},messages,[])
        self.assertLessEqual(metrics['after'], metrics['input_capacity'])
        self.assertTrue(metrics['source_projected_calls'])
        view=json.loads(next(m['content'] for m in projected if m.get('tool_call_id')=='read-source'))['data']
        self.assertIn('无补读能力',view['context_notice'])
        for page, original in zip(view['pages'],data['pages']):
            for span in page['context_spans']:
                self.assertEqual(span['text'],original['text'][span['start']:span['end']])
            self.assertNotIn('context_readback',page)
        self.assertEqual(self.db.get_page(data['pages'][0]['evidence_id'])['text'],data['pages'][0]['text'])

    def test_closing_restores_source_from_exact_archive(self):
        """收尾仅开放finish时恢复此前取证往返，原文和目标仍完整。"""
        tid=self.task();messages,data=self.source_messages(tid)
        projected,_=self.context.prepare({'task_id':tid,'prompt':'比较'},messages,self.tools)
        closing,metrics=self.context.prepare({'task_id':tid,'prompt':'比较'},projected,[])
        self.assertEqual(metrics['restored_source_groups'],1)
        self.assertIn(messages[5],closing)
        validate_messages(closing)
        self.assertEqual(memory_payload(closing)['current_task']['research_plan']['goals'][0]['goal_id'],'exact-goal-id')

    def test_source_projection_rejects_stale_and_foreign_identity(self):
        """最新大观察也核对授权、版本与原句，不能以投影为由跳过校验。"""
        tid=self.task();messages,data=self.source_messages(tid)
        small=ContextService(self.db,None,self.settings.model_copy(update={
            'context_window_tokens':12000,'context_output_reserve':1000,'context_safety_tokens':500}))
        for field,value in (('evidence_id','foreign-page'),('text','stale text'),
                            ('document_id','foreign-document'),('page_number',999)):
            with self.subTest(field=field):
                changed=copy.deepcopy(messages)
                observation=json.loads(changed[5]['content'])
                observation['data']['pages'][0][field]=value
                changed[5]['content']=dump(observation)
                with self.assertRaisesRegex(ValueError,'CONTEXT_SOURCE_STALE'):
                    small.prepare_researcher({'task_id':tid,'prompt':'比较'},changed,[])
        changed=copy.deepcopy(messages)
        observation=json.loads(changed[5]['content'])
        observation['data']['content_sha256']='stale-version'
        changed[5]['content']=dump(observation)
        with self.assertRaisesRegex(ValueError,'CONTEXT_SOURCE_STALE'):
            small.prepare_researcher({'task_id':tid,'prompt':'比较'},changed,[])
        self.assertEqual(self.db.get_page(data['pages'][0]['evidence_id'])['text'],data['pages'][0]['text'])

    def test_foreign_or_stale_archive_rejected(self):
        """档案不能跨任务；原文版本或已撤销页不能借旧摘要继续流入。"""
        tid=self.task();messages,data=self.source_messages(tid)
        projected,_=self.context.prepare({'task_id':tid,'prompt':'比较'},messages,self.tools)
        other_cid=self.conv.create('其他会话',uuid4().hex)['conversation_id'];other=self.task(other_cid)
        with self.assertRaisesRegex(ValueError,'CONTEXT_ARCHIVE_UNAVAILABLE'):
            self.context.prepare({'task_id':other,'prompt':'越权'},projected,self.tools)
        with self.assertRaises(KeyError):
            self.memory.read(other,HistoryRead(source_type='context_archive',source_id=json.loads(next(m['content'] for m in projected if 'archived_tool_result' in m.get('content','')))['source_id']))
        self.db.execute('UPDATE page_evidence SET text=? WHERE evidence_id=?',('changed text',data['pages'][0]['evidence_id']))
        with self.assertRaisesRegex(ValueError,'CONTEXT_SOURCE_STALE'):
            self.context.prepare({'task_id':tid,'prompt':'比较'},projected,[])

    def test_draft_pending_calls_and_unavailable_ids(self):
        """只注入本任务真实稿件和授权页，跨任务ID不当作可用来源。"""
        small=self.settings.model_copy(update={'context_window_tokens':12000,'context_output_reserve':1000,'context_safety_tokens':500})
        tid=self.task(settings=small);messages,data=self.source_messages(tid)
        draft=NoteService(self.db,self.files).create(tid,CreateNoteInput(title='已有稿',markdown='已有成果',evidence_ids=[]))
        state={'task_id':tid,'draft_id':draft.draft_id,'evidence_ids':['foreign-id'],
               'pending_tool_calls':[{'call_id':'pending','name':'finish_task','arguments':{'answer_markdown':'pending'}}]}
        current=CoverageService(self.db).current_context(tid,state)
        self.assertEqual(current['current_draft']['content_hash'],draft.content_hash)
        self.assertEqual(current['unavailable_evidence_ids'],['foreign-id'])
        self.assertEqual(current['pending_tool_calls'][0]['call_id'],'pending')
        foreign=self.conv.create('其他',uuid4().hex)['conversation_id'];other=self.task(foreign)
        self.assertIsNone(CoverageService(self.db).current_context(other,state)['current_draft'])

    def test_user_clarification_survives_compression(self):
        """resume的真实用户回答优先于旧计划，提示更新计划而非猜测新目标。"""
        tid=self.task();messages,_=self.source_messages(tid)
        clarification=self.tasks.request_clarification(tid,'比较几篇？',['count'],['5'])
        state=initial_state(tid,self.db.get_task(tid)['prompt'],'deny')
        state.update(messages=messages,finished=True,final_reason='clarification requested')
        CheckpointStore(self.db).save(tid,state)
        self.db.update_task(tid,status='needs_input')
        self.tasks.resume(ResumeInput(clarification_id=clarification['clarification_id'],answer='只比较已上传的两篇，仍禁止保存',idempotency_key=uuid4().hex),tid)
        state=CheckpointStore(self.db).load(tid)
        projected,_=self.context.prepare(state,state['messages'],self.tools)
        current=memory_payload(projected)['current_task']
        self.assertEqual(current['latest_clarification']['answer'],'只比较已上传的两篇，仍禁止保存')
        self.assertTrue(current['plan_needs_update'])
        self.assertEqual(current['save_policy'],'deny')

    def test_successful_new_draft_wins_over_old_checkpoint(self):
        """工具行动已写入新稿但checkpoint仍旧时，不向下一请求注入旧hash。"""
        tid=self.task()
        notes=NoteService(self.db,self.files)
        old=notes.create(tid,CreateNoteInput(title='旧稿',markdown='旧稿',evidence_ids=[]))
        new=notes.create(tid,CreateNoteInput(title='新稿',markdown='局部更新',evidence_ids=[]))
        action=ActionRecord(task_id=tid,sequence=1,tool_name='revise_note',arguments={},status='succeeded',
                            observation={'ok':True,'data':new.model_dump(mode='json')})
        self.db.save_action(action.model_dump(mode='json'))
        current=CoverageService(self.db).current_context(tid,{'draft_id':old.draft_id})
        self.assertEqual(current['current_draft']['content_hash'],new.content_hash)
        child=self.db.create_subtask(tid,tid,0,'已有子目标',{'goal_ids':['exact-goal-id']},uuid4().hex)
        current=CoverageService(self.db).current_context(tid,{'draft_id':old.draft_id})
        self.assertEqual(current['subtasks'][0]['subtask_id'],child['subtask_id'])
        self.assertEqual(current['subtasks'][0]['goal_ids'],['exact-goal-id'])

    def test_actual_api_worker_readback_and_receipt_replay(self):
        """五次替身调用触发压缩/读回/收尾；重放成功收据不再发送。"""
        from research_loop.main import app
        provider=ContinuityProvider(readback=True)
        allowed={'set_research_plan','read_paper','read_memory','create_note','finish_task'}
        with (patch('research_loop.api.dependencies.get_settings',return_value=self.settings),TestClient(app) as client,
             patch('httpx.Client.send',side_effect=AssertionError('network forbidden'))):
            tid=self.task(client=client)
            executor=TaskExecutor(self.settings,self.db,lambda _:provider,lambda s,f:LongPDF(),supervisor_tools=allowed)
            result=executor.execute(tid)
            view=client.get('/api/tasks/'+tid).json()
            self.assertEqual(view['status'],'completed',view)
            self.assertEqual(len(provider.calls),5)
            self.assertTrue(any('archived_tool_result' in dump(m) for m in provider.calls))
            self.assertTrue(any(a['tool_name']=='read_memory' and a['status']=='succeeded' for a in self.db.list_actions(tid)))
            self.assertEqual(memory_payload(provider.calls[-1])['available_tools'],['finish_task'])
            self.assertIn('EXACT SOURCE',dump(provider.calls[-1]))
            receipts=copy.deepcopy(self.db.list_model_calls(tid))
            # 模拟成功响应已入库、图状态丢失；权威行动/收据不回滚。
            CheckpointStore(self.db).save(tid,initial_state(tid,self.db.get_task(tid)['prompt'],'deny',
                {a['input_ref']:a for a in json.loads(self.db.get_task(tid)['attachments_json'])},5,80,0)|{'coverage_required':True})
            self.db.update_task(tid,status='queued',claimed_by=None,lease_token=None,lease_until=None)
            executor.execute(tid)
            self.assertEqual(len(provider.calls),5)
            self.assertEqual(self.db.list_model_calls(tid),receipts)
            self.assertFalse(list(self.settings.output_dir.rglob('*.md')))

    def test_capacity_exit_keeps_existing_draft(self):
        """没有安全读回且容量不足时，Worker零请求并保留原稿交付partial。"""
        small=self.settings.model_copy(update={'context_window_tokens':12000,'context_output_reserve':1000,'context_safety_tokens':500})
        tid=self.task(settings=small);messages,data=self.source_messages(tid)
        draft=NoteService(self.db,self.files).create(tid,CreateNoteInput(title='已有稿',markdown='既有可读成果',evidence_ids=[]))
        state=initial_state(tid,self.db.get_task(tid)['prompt'],'deny')
        state.update(messages=messages,draft_id=draft.draft_id,coverage_required=True)
        # 不可裁剪的当前用户约束本身超容量；不能用已能投影的全文模拟该出口。
        state['messages']=messages+[{'role':'user','content':'当前必须保留的要求。'*3000}]
        CheckpointStore(self.db).save(tid,state)
        provider=ContinuityProvider()
        small=self.settings.model_copy(update={'context_window_tokens':12000,'context_output_reserve':1000,'context_safety_tokens':500})
        result=TaskExecutor(small,self.db,lambda _:provider,lambda s,f:LongPDF(),supervisor_tools={'finish_task'}).execute(tid)
        self.assertEqual(provider.calls,[])
        self.assertEqual(self.db.get_task(tid)['status'],'partial')
        self.assertEqual(result.answer_markdown,'既有可读成果')
        self.assertIn('CONTEXT_CAPACITY',dump(result.model_dump()))
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
