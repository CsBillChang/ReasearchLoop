"""真实API/数据库/Worker中的无进展交接；模型和PDF均为显式合成替身。"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from research_loop.api.dependencies import get_runtime_settings
from research_loop.config import Settings
from research_loop.main import app
from research_loop.providers.deepseek import validate_messages
from research_loop.services.research_progress import ResearchProgressService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_collaboration_contract import FixturePDF
from integration.test_research_progress_handoff import ProgressScenario


class ReadbackScenario(ProgressScenario):
    """依真实观察生成决策，证明接线，不证明DeepSeek会采用相同策略。"""

    def __init__(self, mode, case):
        """固定交付/继续补读/无效交接三类行为，不改变产品预算。"""
        super().__init__()
        self.mode, self.case = mode, case

    def send(self, messages, tools, **kwargs):
        """前三步模拟互补读页与压缩读回，再按可见工具选择交接。"""
        validate_messages(messages)
        self.requests.append(copy.deepcopy({'messages':messages,'tools':tools}))
        names={t['function']['name'] for t in tools}
        if 'finish_research' not in names:
            done=[c['function']['name'] for m in messages for c in m.get('tool_calls',[])]
            if 'delegate_research' not in done:
                return super().send(messages,tools,**kwargs)
            observations=[json.loads(m['content']) for m in messages if m.get('role')=='tool']
            children=[c for o in observations for c in o.get('data',{}).get('subtasks',[])]
            self.case.assertEqual(len(children),2)
            partial=any(c['status']!='completed' for c in children)
            texts=[text for c in children for text in c.get('findings',[])]
            reason='；'.join(c.get('stop_reason') or '' for c in children)
            return self.response('finish_task',{'answer_markdown':'\n'.join(texts) or '未形成比较结论：'+reason,
                'goals':[{'goal_id':'compare','status':'unresolved' if partial else 'satisfied',
                          'subtask_ids':[c['subtask_id'] for c in children],'reason':reason or '合成互补原文形成发现'}]})
        brief=next(json.loads(m['content'].split('\n',1)[1]) for m in messages if m.get('content','').startswith('结构化子任务与已授权定位：'))
        ref=brief['input_refs'][0]
        step=self.steps.get(ref,0);self.steps[ref]=step+1
        if self.mode=='retained':
            if step==1:
                first=next(json.loads(m['content'])['data']['pages'][0] for m in messages if m.get('role')=='tool' and json.loads(m['content']).get('data',{}).get('pages'))
                return self.response('record_research_progress',{'findings':[{'text':ref+'已有候选发现','evidence_ids':[first['evidence_id']]}]})
            if step in (2,3,4):
                return self.response('read_paper',{'target_ref':ref,'start_page':2 if step==2 else 1,'page_count':1})
            if step>=5:
                self.case.assertEqual(names,{'finish_research','record_research_progress'})
                return self.response('record_research_progress',{'findings':[]})
        if step<3:
            return self.response('read_paper',{'target_ref':ref,'start_page':2 if step==1 else 1,'page_count':1})
        if step==3:
            self.case.assertEqual(names,{'finish_research','record_research_progress'})
            self.case.assertTrue(any('研究进展交接' in m.get('content','') for m in messages if m.get('role')=='system'))
        pages={}
        for m in messages:
            if m.get('role')=='tool':
                for p in json.loads(m['content']).get('data',{}).get('pages',[]):pages[p['page_number']]=p
            if m.get('role')=='user' and m.get('content','').startswith('{"archived_tool_groups"'):
                for p in json.loads(m['content'])['sources']:pages[p['page_number']]=p
        self.case.assertEqual(set(pages),{1,2})
        if self.mode=='ignore':
            return self.response('read_paper',{'target_ref':ref,'start_page':1,'page_count':1})
        if self.mode=='empty':
            return self.response('record_research_progress',{'findings':[]})
        if self.mode=='continue' and step==3:
            return self.response('record_research_progress',{'findings':[{'text':ref+'候选机制来自第一页','evidence_ids':[pages[1]['evidence_id']]}],
                'gaps':['核对第二页条件'],'next_question':'用第二页核对条件'})
        if self.mode=='continue' and step==4:
            self.case.assertIn('read_paper',names)
            return self.response('read_paper',{'target_ref':ref,'start_page':2,'page_count':1})
        return self.response('finish_research',{'findings':[ref+'在两页互补原文中使用语言嵌入。'],
            'evidence_ids':[p['evidence_id'] for p in pages.values()], 'paper_ids':[pages[1]['document_id'].removeprefix('doc-')]})


class ReadbackHandoffFlowTests(unittest.TestCase):
    """全部场景禁止保存，验证失效调用、状态恢复与父子交接。"""

    def run_case(self, mode='finish', recovery=False):
        """隔离执行完整API链，必要时在成功收据后/执行前中断。"""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            url=os.getenv('READBACK_TEST_DATABASE_URL') or 'sqlite:///'+str(root/'test.db')
            if not url.startswith('sqlite:') and not url.rsplit('/',1)[-1].startswith('researchloop_test_core_loop_'):
                raise ValueError('explicit isolated database required')
            settings=Settings(project_root=root,database_url=url,data_dir=root/'data',output_dir=root/'notes',context_compress_ratio=0.01)
            db=Database(url);db.ensure_schema()
            scenario=ReadbackScenario(mode,self)
            attachments=[]
            for name in ('A','B'):
                p=root/(name+'.pdf');p.write_bytes(b'%PDF-synthetic')
                attachments.append({'input_ref':'input-'+name,'paper_id':'paper-'+uuid4().hex,'local_path':str(p)})
            app.dependency_overrides[get_runtime_settings]=lambda:settings
            try:
                with TestClient(app) as client:
                    created=client.post('/api/tasks',json={'prompt':'比较附件，不保存','save_policy':'deny','attachments':attachments,'idempotency_key':uuid4().hex})
                    self.assertEqual(created.status_code,202,created.text)
                    tid=created.json()['task_id']
                    executor=TaskExecutor(settings,db,lambda _:scenario,lambda *_:FixturePDF())
                    old_calls={}
                    if recovery:
                        original=CheckpointStore.save
                        interrupted=[]
                        def stop_before_handoff(store, task_id, state):
                            """成功响应已持久但行动未执行，旧Checkpoint恢复应复用响应。"""
                            if not interrupted and (state.get('research_handoff') or {}).get('phase')=='requested' and state.get('pending_tool_calls'):
                                interrupted.append(task_id)
                                raise SystemExit('CONTROLLED_BEFORE_HANDOFF')
                            return original(store,task_id,state)
                        with patch.object(CheckpointStore,'save',stop_before_handoff):
                            with self.assertRaises(SystemExit):executor.execute(tid)
                        old_calls={c['call_id']:c for t in db.fetchall('SELECT task_id FROM tasks WHERE root_task_id=?',(tid,)) for c in db.list_model_calls(t['task_id']) if c['status']=='succeeded'}
                        db.execute('UPDATE tasks SET lease_until=? WHERE root_task_id=?',('2000-01-01T00:00:00Z',tid))
                    executor.execute(tid)
                    view=client.get('/api/tasks/'+tid).json()
                    expected='partial' if mode in {'ignore','empty','retained'} else 'completed'
                    self.assertEqual(view['status'],expected,view)
                    children=db.list_subtasks(tid)
                    self.assertEqual(len(children),2)
                    for child in children:
                        actions=db.list_actions(child['subtask_id'])
                        executed_reads = [a for a in actions if a['tool_name'] == 'read_paper'
                                          and a['observation'].get('data', {}).get('tool_executed') is not False]
                        self.assertEqual(len(executed_reads), 4 if mode in {'continue','retained'} else 3)
                        self.assertFalse(any(a['tool_name'] in {'save_note','create_note'} for a in actions))
                        if mode in {'ignore','empty','retained'}:
                            if mode=='retained':self.assertTrue(child['result']['result']['findings'])
                            else:self.assertEqual(child['result']['result']['findings'],[])
                            self.assertIn('unavailable' if mode == 'ignore' else 'research',
                                          child['result']['result']['stop_reason'])
                        if mode=='continue':self.assertTrue(ResearchProgressService(db).snapshot(child['subtask_id'])['findings'])
                    calls=[c for t in db.fetchall('SELECT task_id FROM tasks WHERE root_task_id=?',(tid,)) for c in db.list_model_calls(t['task_id'])]
                    indexed={c['call_id']:c for c in calls}
                    for key,value in old_calls.items():self.assertEqual(indexed[key],value)
                    self.assertTrue(all(c['status']=='succeeded' for c in calls))
                    self.assertEqual(list(root.rglob('*.md')),[])
                    self.assertEqual(view['save_policy'],'deny')
                    prior=copy.deepcopy(calls)
                    with self.assertRaisesRegex(RuntimeError,'not queued'):
                        executor.execute(tid)
                    again=[c for t in db.fetchall('SELECT task_id FROM tasks WHERE root_task_id=?',(tid,)) for c in db.list_model_calls(t['task_id'])]
                    self.assertEqual(prior,again)
            finally:
                app.dependency_overrides.clear()

    def test_complementary_readback_hands_off(self):
        """两子角色被路由至已有交接接口，父角色完成比较。"""
        self.run_case()

    def test_specific_new_gap_can_continue(self):
        """提交真实阶段更新后可继续定向补读，不永久锁死研究工具。"""
        self.run_case('continue')

    def test_ignored_handoff_ends_honestly(self):
        """模型无视可见工具重复取证时不执行，父角色收到真实停止原因。"""
        self.run_case('ignore')

    def test_empty_stage_ends_without_fabricating_findings(self):
        """空阶段更新不算进展，partial且不伪造发现。"""
        self.run_case('empty')

    def test_success_receipt_survives_handoff_recovery(self):
        """交接窗口中断后，成功收据不重发且旧参数保持。"""
        self.run_case(recovery=True)

    def test_unresolved_handoff_keeps_submitted_findings(self):
        """重复读回仍无法结束时，父角色保留此前真实提交的候选成果。"""
        self.run_case('retained')
