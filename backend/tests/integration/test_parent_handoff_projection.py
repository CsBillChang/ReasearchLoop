"""大型父子交接实际API链；生成模型与PDF均为显式合成替身。"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from research_loop.agents.memory import estimate
from research_loop.api.dependencies import get_runtime_settings
from research_loop.config import Settings
from research_loop.domain.evidence import PageEvidence
from research_loop.domain.memory import HistoryRead
from research_loop.main import app
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.services.research_progress import ResearchProgressService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_collaboration_contract import FixturePDF
from integration.test_context_continuity import memory_payload
from integration.test_research_progress_handoff import ProgressScenario, FollowupScenario


class ManyPagePDF(FixturePDF):
    """合成18页来源，放大真实身份重复开销，不打开本机PDF。"""

    def ingest(self, source, paper_id):
        """沿用稳定身份，仅改变合成页数。"""
        return super().ingest(source, paper_id).model_copy(update={'total_pages':18})

    def pages(self, document, start, count):
        """每页保留可精确核对的原句与位置。"""
        return [PageEvidence(evidence_id=f'evidence:{document.paper_id}:{n}',
            document_id=document.document_id,page_number=n,
            text=f'{document.paper_id} language prefix embeddings condition page {n}.',
            parsed_markdown=f'language method page {n}') for n in range(start,min(start+count,19))]


class LargeHandoffScenario(ProgressScenario):
    """规划与委派合并一次决策，子成果大量复用来源身份。"""

    def __init__(self, case, partial=False, large_history=False):
        """选择完整或有明确缺口的终态；记录实际待发投影。"""
        super().__init__()
        self.case,self.partial,self.parent_steps=case,partial,0
        self.large_history = large_history
        self.seen_pages = {}

    def send(self, messages, tools, **kwargs):
        """只产生固定行动，证明工程合同而非模型自主质量。"""
        validate_messages(messages)
        self.requests.append(copy.deepcopy({'messages':messages,'tools':tools}))
        names={t['function']['name'] for t in tools}
        if 'finish_research' in names:
            brief=next(json.loads(m['content'].split('\n',1)[1]) for m in messages
                if m.get('content','').startswith('结构化子任务与已授权定位：'))
            ref=brief['input_refs'][0];step=self.steps.get(ref,0);self.steps[ref]=step+1
            if step==0:return self.response('read_paper',{'target_ref':ref})
            visible=next((json.loads(m['content'])['data']['pages'] for m in messages
                if m.get('role')=='tool' and json.loads(m['content']).get('data',{}).get('pages')), None)
            if visible:
                self.seen_pages[ref] = visible
            # 合成行动可引用此前实际见过的页；旧往返压缩后不要求工具原件仍内联。
            pages = self.seen_pages[ref]
            if step==1:
                return self.response('record_research_progress',{'findings':[{'text':ref+'阶段发现保留语言嵌入条件',
                    'evidence_ids':[pages[0]['evidence_id']]}],'gaps':['阶段仍在调查']})
            return self.response('finish_research',{'status':'partial' if self.partial else 'completed',
                'findings':[f'{ref}方法细节{i}：语言信息参与训练；这些是不同限定，必须逐字保留。' for i in range(12)],
                'evidence_ids':[p['evidence_id'] for p in pages],
                'paper_ids':[pages[0]['document_id'].removeprefix('doc-')],
                'gaps':['图片内容尚不可解释'] if self.partial else []})
        step=self.parent_steps;self.parent_steps+=1
        if step==0:
            plan={'goal':'比较两个合成原文机制','task_kind':'compare','input_refs':['input-A','input-B'],
                'requested_count':2,'count_confirmed':True,'goals':[{'goal_id':'compare','description':'比较语言机制',
                'min_papers':2,'requires_fulltext':True,'input_refs':['input-A','input-B']}]}
            delegation={'subtasks':[{'objective':'调查'+r,'mode':'investigate','goal_ids':['compare'],
                'input_refs':[r]} for r in ['input-A','input-B']]}
            a=self.response('set_research_plan',plan);b=self.response('delegate_research',delegation)
            a['choices'][0]['message']['tool_calls']+=b['choices'][0]['message']['tool_calls']
            return a
        if step>1:raise AssertionError('no extra synthetic parent request expected')
        payload=memory_payload(messages);self.case.assertEqual(payload['current_task']['save_policy'],'deny')
        self.case.assertLessEqual(estimate({'messages':messages,'tools':tools}),115712)
        children=payload['current_task']['subtasks']
        for child in children:
            p=child['research_progress'];self.case.assertEqual(len(p['findings']),13)
            for f in p['findings']:
                self.case.assertEqual(f['verification_status'],'unreviewed')
                self.case.assertTrue(all(k in p['source_catalog'] for k in f['source_refs']))
        observations=[json.loads(m['content']) for m in messages if m.get('role')=='tool']
        delegated=next(o['data']['subtasks'] for o in observations if o.get('data',{}).get('subtasks'))
        self.case.assertTrue(all('findings_in' in c and 'findings' not in c for c in delegated))
        answer = 'input-A与input-B均使用语言嵌入，各自条件按来源区分。'+('图片内容未确认。' if self.partial else '')
        if self.large_history:
            answer += '\n历史答案的不同限定仅用于工程负载，不是真实论文结论。'*1800
        return self.response('finish_task',{'answer_markdown':answer,
            'goals':[{'goal_id':'compare','status':'unresolved' if self.partial else 'satisfied',
                'subtask_ids':[c['subtask_id'] for c in children],'reason':'图片未读明' if self.partial else '合成原文与发现已完整交接'}]})


class LargeHandoffTests(unittest.TestCase):
    """按真实服务链检查单组批次、压缩接续、成功收据恢复和合理partial。"""

    def run_case(self, partial=False, recovery=False, followup=False, large_history=False):
        """隔离目录/数据库，不授予文件保存权限，禁止供应商网络路径。"""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);url=os.getenv('CORE_LOOP_TEST_DATABASE_URL') or 'sqlite:///'+str(root/'test.db')
            if not url.startswith('sqlite:') and not url.rsplit('/',1)[-1].startswith('researchloop_test_core_loop_'):
                raise ValueError('isolated test database required')
            settings=Settings(project_root=root,data_dir=root/'data',output_dir=root/'notes',database_url=url)
            db=Database(url);db.ensure_schema();scenario=LargeHandoffScenario(self,partial,large_history)
            cid=ConversationService(db,settings).create('大型交接',uuid4().hex)['conversation_id']
            attachments=[]
            for name in ('A','B'):
                path=root/(name+'.pdf');path.write_bytes(b'%PDF-synthetic')
                attachments.append({'input_ref':'input-'+name,'paper_id':'paper-'+uuid4().hex,'local_path':str(path)})
            app.dependency_overrides[get_runtime_settings]=lambda:settings
            try:
                with patch.object(DeepSeekProvider,'send',side_effect=AssertionError('real model forbidden')), \
                     patch.object(DeepSeekProvider,'send_stream',side_effect=AssertionError('real stream forbidden')), TestClient(app) as client:
                    created=client.post('/api/tasks',json={'conversation_id':cid,'prompt':'比较附件两个机制，不保存',
                        'attachments':attachments,'save_policy':'deny','idempotency_key':uuid4().hex})
                    self.assertEqual(created.status_code,202,created.text);tid=created.json()['task_id']
                    worker=TaskExecutor(settings,db,lambda _:scenario,lambda *_:ManyPagePDF())
                    if recovery:
                        original=CheckpointStore.save
                        def interrupt(store, task_id, state):
                            """父任务成功响应已记收据、交付工具尚未执行时中断。"""
                            if task_id==tid and any(c['name']=='finish_task' for c in state.get('pending_tool_calls',[])):
                                raise SystemExit('AFTER_PARENT_RECEIPT')
                            return original(store,task_id,state)
                        with patch.object(CheckpointStore,'save',interrupt):
                            with self.assertRaises(SystemExit):worker.execute(tid)
                        frozen={c['call_id']:c for t in db.fetchall('SELECT task_id FROM tasks WHERE root_task_id=?',(tid,)) for c in db.list_model_calls(t['task_id'])}
                        sent=len(scenario.requests)
                        db.execute('UPDATE tasks SET lease_until=? WHERE root_task_id=?',('2000-01-01T00:00:00Z',tid))
                    worker.execute(tid)
                    self.assertEqual(client.get('/api/tasks/'+tid).json()['status'],'partial' if partial else 'completed')
                    children=db.list_subtasks(tid)
                    for child in children:
                        full=ResearchProgressService(db).retained(child['subtask_id'])
                        self.assertEqual(len(full['findings']),13)
                        self.assertTrue(all('sources' in f and 'source_refs' not in f for f in full['findings']))
                        self.assertNotIn('source_catalog',child['result']['result']['research_progress'])
                    archives=db.fetchall('SELECT * FROM context_archives WHERE task_id=? ORDER BY sequence_no',(tid,))
                    row=next(r for r in archives if json.loads(r['metrics_json']).get('handoff_projected_calls'))
                    metrics=json.loads(row['metrics_json'])
                    self.assertGreater(metrics['before'],metrics['input_capacity'])
                    self.assertLessEqual(metrics['after'],metrics['input_capacity'])
                    raw=json.loads(row['original_json']);projected=json.loads(row['projected_json'])
                    self.assertIn('sources',raw['current_task']['subtasks'][0]['research_progress']['findings'][0])
                    back=memory_payload(projected)['current_task']['handoff_readback']
                    args=HistoryRead.model_validate({k:v for k,v in back.items() if k!='notice'})
                    read=MemoryService(db).read(tid,args)
                    self.assertEqual(read['content'],row['original_json'][args.offset:args.offset+args.limit])
                    foreign=str(uuid4());db.create_task(foreign,None,uuid4().hex,'外来任务','deny',[])
                    with self.assertRaises(KeyError):MemoryService(db).read(foreign,args)
                    db.update_task(foreign,status='cancelled',cancel_requested=True)
                    if recovery:
                        self.assertEqual(sent,len(scenario.requests))
                        restored={c['call_id']:c for t in db.fetchall('SELECT task_id FROM tasks WHERE root_task_id=?',(tid,)) for c in db.list_model_calls(t['task_id'])}
                        self.assertEqual(frozen,restored)
                    if followup:
                        # 追问根创建前冻结测试压缩阈值；Worker不能回写既有策略。
                        settings=settings.model_copy(update={'context_compress_ratio':0.01})
                        prompt = '沿用 input-A 和 input-B 比较两个机制，不保存。' if large_history else '基于上次记录解释 input-A 的语言机制；只读取物理第1页，不保存。'
                        created=client.post('/api/tasks',json={'conversation_id':cid,'prompt':prompt,
                            'save_policy':'deny','idempotency_key':uuid4().hex})
                        self.assertEqual(created.status_code,202,created.text)
                        next_id=created.json()['task_id'];provider=LargeHandoffScenario(self) if large_history else FollowupScenario(self)
                        TaskExecutor(settings.model_copy(update={'context_compress_ratio':0.01}),db,lambda _:provider,lambda *_:ManyPagePDF()).execute(next_id)
                        result = client.get('/api/tasks/'+next_id).json()
                        self.assertEqual(result['status'],'completed',result)
                        if not large_history:
                            self.assertTrue(any(json.loads(r['metrics_json']).get('compressed_groups') for r in db.fetchall('SELECT metrics_json FROM context_archives WHERE task_id=?',(next_id,))))
                        if large_history:
                            rows = db.fetchall('SELECT * FROM context_archives WHERE task_id=?', (next_id,))
                            externalized = [r for r in rows if json.loads(r['metrics_json']).get('workspace_externalized')]
                            self.assertTrue(externalized)
                            for row in externalized:
                                payload = memory_payload(json.loads(row['projected_json']))
                                self.assertEqual(payload['current_task']['save_policy'], 'deny')
                                def read_all(value):
                                    """通过生产服务分页重建省略值，并拒绝其他任务读取。"""
                                    if isinstance(value, dict):
                                        for key, item in value.items():
                                            if key.endswith('_context_archive'):
                                                args = HistoryRead.model_validate(item['readback'])
                                                original = json.loads(row['original_json'])
                                                for part in item['original_path']: original = original[part]
                                                chunks, remaining = [], item['value_chars']
                                                while remaining:
                                                    got = MemoryService(db).read(next_id, args)
                                                    text = got['content'][:remaining]
                                                    self.assertTrue(text)
                                                    chunks.append(text); remaining -= len(text)
                                                    args = args.model_copy(update={'offset': args.offset+len(text)})
                                                self.assertEqual(json.loads(''.join(chunks)), original)
                                                with self.assertRaises(KeyError): MemoryService(db).read(tid, HistoryRead.model_validate(item['readback']))
                                            else: read_all(item)
                                    elif isinstance(value, list):
                                        for item in value: read_all(item)
                                read_all(payload)
                    self.assertFalse(list(root.rglob('*.md')))
                    self.assertEqual(db.get_task(tid)['save_policy'],'deny')
            finally:app.dependency_overrides.clear()

    def test_large_single_batch_and_compressed_followup(self):
        """同一决策规划/委派，大型交接后同会话压缩追问完成。"""
        self.run_case(followup=True)

    def test_large_partial_preserves_candidates_and_reason(self):
        """明确缺口下保留真实候选与状态，不改成核验通过。"""
        self.run_case(partial=True)

    def test_parent_success_receipt_is_reused(self):
        """大型交接后的成功父请求恢复不重发，旧收据逐字段不变。"""
        self.run_case(recovery=True)

    def test_large_history_and_new_handoff_share_one_request_capacity(self):
        """大历史会话后再次委派两材料，模型可见工作集与完整档案分别核验。"""
        self.run_case(followup=True, large_history=True)
