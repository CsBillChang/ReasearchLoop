"""零模型隔离链：真实上下文、数据库、HTTP和Worker，供应商/PDF为替身。"""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from research_loop.agents.memory import ContextService
from research_loop.agents.researcher import build_researcher_graph, finish_spec, researcher_state
from research_loop.api.dependencies import get_runtime_settings
from research_loop.config import Settings
from research_loop.domain.tasks import ActionRecord, SavePolicy, TaskCreate
from research_loop.domain.research_progress import ResearchProgressInput
from research_loop.main import app
from research_loop.providers.deepseek import RequestNotSent, validate_messages
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.services.research_progress import ResearchProgressService, visible_evidence
from research_loop.storage.memory import dump
from research_loop.storage.mysql import Database
from research_loop.tools.base import ToolContext
from research_loop.tools.memory import specs as memory_specs
from research_loop.tools.registry import ToolRegistry
from research_loop.workers.executor import TaskExecutor
from integration.test_collaboration_contract import FixturePDF
from integration.test_context_continuity import LongPDF, exchange


def response(name, args, *, malformed=False):
    """构造可审计供应商替身响应，截断用真实finish_reason而非伪造成功内容。"""
    return {'id':uuid4().hex, 'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15},
            'choices':[{'finish_reason':'length' if malformed else 'tool_calls', 'message':{
                'role':'assistant','content':'','tool_calls':[{'id':uuid4().hex,'type':'function',
                    'function':{'name':name,'arguments':'{"findings": [' if malformed else dump(args)}}]}}]}


class ContinuousUseClosureTests(unittest.TestCase):
    """默认SQLite；真实MySQL必须显式绑定本轮专用新库。"""

    def setUp(self):
        """隔离数据、输出和服务依赖，并拒绝任何未替身的外网/模型发送。"""
        temp = tempfile.TemporaryDirectory(prefix='researchloop-closure-'); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        url = os.getenv('CONTINUITY_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root/'test.db')
        if not url.startswith('sqlite:') and not url.rsplit('/',1)[-1].startswith('researchloop_test_continuity_'):
            raise ValueError('explicit isolated continuity database required')
        self.settings = Settings(project_root=self.root, data_dir=self.root/'data', output_dir=self.root/'notes',
                                 database_url=url, retrieval_enabled=False, context_compress_ratio=0.1)
        self.db = Database(url); self.db.ensure_schema()
        self.memory = MemoryService(self.db)
        self.context = ContextService(self.db,self.memory,self.settings)
        self.tasks = TaskService(self.db,self.settings)
        self.cid = ConversationService(self.db,self.settings).create('合成研究',uuid4().hex)['conversation_id']
        self.tools = [s.schema() for s in memory_specs() if s.name == 'read_memory']
        self.addCleanup(app.dependency_overrides.clear)
        # TestClient也经过httpx.send，因此HTTP场景单独用项目模型入口拒绝外调。
        guard = patch('research_loop.providers.deepseek.DeepSeekProvider.send', side_effect=AssertionError('REAL_MODEL_FORBIDDEN'))
        guard.start(); self.addCleanup(guard.stop)

    def task(self, client=None):
        """创建仅含合成附件且禁止保存的当前会话任务。"""
        path = self.root/'fixture.pdf'; path.write_bytes(b'%PDF-synthetic')
        payload = TaskCreate(prompt='比较附件方法，不保存文件', idempotency_key=uuid4().hex,
            conversation_id=self.cid, save_policy='deny', attachments=[{
                'input_ref':'input-A','paper_id':'paper-A','local_path':str(path)}])
        if client:
            result = client.post('/api/tasks',json=payload.model_dump(mode='json'))
            self.assertEqual(result.status_code,202,result.text)
            return result.json()['task_id']
        return self.tasks.create(payload).task_id

    def papers(self, tid, long=False):
        """通过实际论文服务建立合成页与授权，不直接伪造可见性。"""
        papers = PaperService(self.db,None,LongPDF() if long else FixturePDF())
        attachments = {a['input_ref']:a for a in json.loads(self.db.get_task(tid)['attachments_json'])}
        data = papers.read(tid,'input-A',attachments=attachments)
        return papers, attachments, data

    def test_search_result_can_be_externalized_without_losing_identity(self):
        """大型最后一组候选/诊断可逆读回，所有标题和链接保留。"""
        tid = self.task()
        candidates = [{'paper_id':f'p{i}','title':f'Synthetic paper {i}', 'paper_url':f'https://fixture.invalid/{i}',
                       'published':'2025-01-01', 'abstract':('研究方法与明确条件。'*500), 'metadata':{'raw':'detail '*300}}
                      for i in range(20)]
        data = {'candidates':candidates,'source_results':[{'source':'cvpr','status':'queried_success',
                  'request':{'responses':['diagnostic '*2000]}}]}
        messages = [{'role':'system','content':'合成角色'}, {'role':'user','content':'筛选三篇，不保存'}]
        messages += exchange('search_papers',{'query':'synthetic'},data)
        original = copy.deepcopy(messages)
        projected, metrics = self.context.prepare({'task_id':tid,'prompt':'筛选方法','model_requests':1},messages,self.tools)
        validate_messages(projected)
        self.assertLessEqual(metrics['after'],metrics['input_capacity'])
        self.assertTrue(metrics['workspace_externalized'])
        result = json.loads(projected[-1]['content'])['data']['candidates']
        self.assertEqual([(p['paper_id'],p['title'],p['paper_url']) for p in result],
                         [(p['paper_id'],p['title'],p['paper_url']) for p in candidates])
        detail = next(p['abstract_context_archive'] for p in result if 'abstract_context_archive' in p)
        archive = self.db.fetchone('SELECT original_json FROM context_archives WHERE snapshot_id=?',
                                  (detail['readback']['source_id'],))['original_json']
        start = detail['readback']['offset']
        self.assertIn(json.loads(archive[start:start+detail['value_chars']]),[p['abstract'] for p in candidates])
        self.assertEqual(messages,original)

    def test_complementary_sources_survive_below_hard_capacity(self):
        """旧观察归档后即使已达容量，也保留两页互补原文且档案仍可恢复。"""
        tid = self.task(); papers,attachments,data = self.papers(tid,long=True)
        messages = [{'role':'system','content':'合成角色'},{'role':'user','content':'比较方法，不保存'}]
        messages += exchange('read_paper',{'target_ref':'input-A'},data,'whole')
        for page in (1,2):
            args = {'target_ref':'input-A','start_page':page,'page_count':1}
            messages += exchange('read_paper',args,papers.read(tid,attachments=attachments,**args),'page-'+str(page))
        projected, metrics = self.context.prepare({'task_id':tid,'prompt':'比较方法'},messages,self.tools)
        self.assertLessEqual(metrics['after'],metrics['input_capacity'])
        visible = visible_evidence(projected)
        for page in data['pages']:
            self.assertTrue(any(row[2] == page['page_number'] and row[4] == page['text'] for row in visible))
        self.assertTrue(metrics['compressed_groups'])

    def test_optional_assessment_does_not_swallow_findings(self):
        """格式错误的可选核验保留错误，不批准、不丢掉合法发现。"""
        tid = self.task(); papers,_,data = self.papers(tid)
        ctx = ToolContext(tid,SavePolicy.deny,services={'papers':papers,'scope':{'mode':'investigate'}},role='researcher')
        payload = {'status':'partial','findings':['语言信息来自合成页，实验条件尚未核实。'],
                   'evidence_ids':[data['pages'][0]['evidence_id']],
                   'assessments':[{'claim':'未提供引句','status':'supported','basis':'direct_statement'}]}
        result = ToolRegistry([finish_spec()]).execute('finish_research',payload,ctx)
        self.assertTrue(result.ok,result.message)
        self.assertEqual(result.data['findings'],payload['findings'])
        self.assertEqual(result.data['status'],'partial')
        self.assertEqual(result.data['assessments'],[])
        self.assertTrue(result.data['assessment_errors'])
        ctx.services['scope']['verification_claims'] = ['未提供引句']
        strict = ToolRegistry([finish_spec()]).execute('finish_research',payload,ctx)
        self.assertFalse(strict.ok)
        self.assertIn('required assessment',strict.message)

    def test_researcher_uses_same_single_progress_projection(self):
        """子角色只注入自身规范成果，工具副本引用同一路径且原件保留。"""
        tid = self.task(); _,_,data = self.papers(tid)
        args = ResearchProgressInput(findings=[{'text':'独特的合成方法发现。','evidence_ids':[data['pages'][0]['evidence_id']]}])
        value = ResearchProgressService(self.db).prepare(tid,args)
        action = ActionRecord(task_id=tid,sequence=1,tool_name='record_research_progress',
                              arguments=args.model_dump(),status='succeeded',
                              observation={'ok':True,'data':{'research_progress':value}})
        self.db.save_action(action.model_dump(mode='json'))
        state = researcher_state(tid,'合成任务',brief={'mode':'investigate'})
        messages = state['messages'] + exchange('record_research_progress',{}, {'research_progress':value},'own')
        original = copy.deepcopy(messages)
        projected,_ = self.context.prepare_researcher(state,messages,[])
        tool = json.loads(next(m['content'] for m in projected if m.get('tool_call_id')=='own'))
        self.assertEqual(tool['data']['research_progress']['context_path'],'current_task.research_progress')
        self.assertEqual(sum('独特的合成方法发现。' in (m.get('content') or '') for m in projected),1)
        self.assertEqual(messages,original)

    def test_deep_read_handoff_preserves_pending_checks(self):
        """取文节点可交接原文，保留关注点，不假称已完成独立核验。"""
        tid = self.task(); papers,_,data = self.papers(tid)
        claims = ['请核对模型条件','请核对推理耗时']
        ctx = ToolContext(tid,SavePolicy.deny,services={'papers':papers,
                          'scope':{'mode':'deep_read','verification_claims':claims}},role='researcher')
        result = ToolRegistry([finish_spec()]).execute('finish_research',{
            'evidence_ids':[p['evidence_id'] for p in data['pages']]},ctx)
        self.assertTrue(result.ok,result.message)
        self.assertEqual(result.data['pending_verification_claims'],claims)
        self.assertEqual(result.data['assessments'],[])

    def test_deep_read_handoff_enters_existing_summary_node(self):
        """真实角色图取文交接后进入既有精读节点，核验关注点不阻断路由。"""
        tid = self.task(); papers,_,data = self.papers(tid)
        claims = ['核对生成条件']
        class Provider:
            """仅提交已经读取的证据，不生成或伪造真实精读稿。"""
            def send(self,*args,**kwargs):
                """固定调用交接工具。"""
                return response('finish_research',{'evidence_ids':[p['evidence_id'] for p in data['pages']]})
        state = researcher_state(tid,'精读合成附件',brief={'mode':'deep_read','verification_claims':claims})
        state['evidence_ids'] = [p['evidence_id'] for p in data['pages']]
        observed = []
        def summary(state,provider,papers,db,record):
            """仅替身精读模型边界，检查真实图的入参和路由。"""
            observed.append(state['structured_result'])
            return record(state,{'deep_reading':{'done':True,'status':'complete'},'final_answer':'合成精读节点输出'})
        with patch('research_loop.agents.deep_reading.deep_reading_node',side_effect=summary):
            graph = build_researcher_graph(Provider(),{'papers':papers,'scope':{'mode':'deep_read',
                                          'verification_claims':claims}},self.db)
            result = graph.invoke(state)
        self.assertEqual(len(observed),1)
        self.assertEqual(observed[0]['pending_verification_claims'],claims)
        self.assertEqual(result['deep_reading']['status'],'complete')
        self.assertIsNone(result.get('saved_path'))

    def test_truncation_does_not_consume_terminal_tool_slot(self):
        """length只保留一次成功收据，截断工具不执行、不消耗槽位或自动补发。"""
        tid = self.task(); papers,_,_ = self.papers(tid)
        raw = response('finish_research', {}, malformed=True)
        class Provider:
            """只提供固定的长度截断响应，发送次数可检查。"""
            def __init__(self):
                """初始化替身发送计数。"""
                self.calls = 0
            def send(self,*args,**kwargs):
                """返回原始截断响应，不生成后续纠正响应。"""
                self.calls += 1
                return copy.deepcopy(raw)
        provider = Provider()
        state = researcher_state(tid,'合成任务',brief={'mode':'investigate'})
        state.update(max_model_requests=4,max_tool_actions=1)
        graph = build_researcher_graph(provider, {'papers':papers,'scope':{'mode':'investigate'}},self.db,
                                       max_model_requests=4,max_tool_actions=1,deep_reading=False)
        result = graph.invoke(state)
        self.assertTrue(result['finished'])
        self.assertTrue(result['final_reason'].startswith('MODEL_LENGTH_TOOL_CALL:'))
        self.assertIsNone(result.get('structured_result'))
        self.assertEqual(result['pending_tool_calls'],[])
        self.assertEqual(provider.calls,1)
        self.assertEqual(result['tool_actions'],0)
        self.assertEqual(self.db.budget_summary(tid)['tool_used'],0)
        self.assertEqual(self.db.list_actions(tid),[])
        calls = self.db.list_model_calls(tid)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['status'],'succeeded')
        self.assertEqual(calls[0]['finish_reason'],'length')
        self.assertEqual(json.loads(calls[0]['response_json']),raw)
        self.assertEqual(json.loads(calls[0]['usage_json']),raw['usage'])

    def test_malformed_non_length_parameters_can_use_terminal_slot_after_correction(self):
        """非length的畸形JSON未执行且不占槽，观察错误后仍可提交合法结束。"""
        tid = self.task(); papers,_,_ = self.papers(tid)
        class Provider:
            """仅产生两次固定替身响应。"""
            def __init__(self):
                """初始化替身调用计数。"""
                self.calls = 0
            def send(self,*args,**kwargs):
                """首次返回非length畸形参数，第二次提交合法partial。"""
                self.calls += 1
                raw = response('finish_research', {'status':'partial','findings':['保留合成发现。']}, malformed=self.calls==1)
                raw['choices'][0]['finish_reason'] = 'tool_calls'
                return raw
        provider = Provider()
        state = researcher_state(tid,'合成任务',brief={'mode':'investigate'})
        state.update(max_model_requests=4,max_tool_actions=1)
        graph = build_researcher_graph(provider, {'papers':papers,'scope':{'mode':'investigate'}},self.db,
                                       max_model_requests=4,max_tool_actions=1,deep_reading=False)
        result = graph.invoke(state)
        self.assertEqual(result['structured_result']['findings'],['保留合成发现。'])
        self.assertEqual(provider.calls,2)
        self.assertEqual(result['tool_actions'],1)
        self.assertEqual(self.db.budget_summary(tid)['tool_used'],1)
        calls = self.db.list_model_calls(tid)
        self.assertTrue(all(c['status']=='succeeded' for c in calls))

    def test_api_worker_failure_keeps_body_and_explanation(self):
        """已有进展后供应商明确未发送，HTTP结果仍保留正文与准确处理说明。"""
        class Provider:
            """固定取文、记录发现、受控未发送三步。"""
            def __init__(self):
                """仅保存固定步骤与实际证据身份。"""
                self.calls = 0; self.eid = None
            def close(self):
                """替身没有外部连接。"""
            def send(self,messages,tools,**kwargs):
                """从实际工具观察取得证据身份，之后模拟未发送错误。"""
                self.calls += 1
                if self.calls == 1:
                    return response('read_paper',{'target_ref':'input-A'})
                if self.calls == 2:
                    page = next(json.loads(m['content'])['data']['pages'][0] for m in messages
                                if m.get('role')=='tool' and '"pages"' in m.get('content',''))
                    self.eid = page['evidence_id']
                    return response('record_research_progress',{'findings':[{'text':'已读合成方法，实验未核实。','evidence_ids':[self.eid]}]})
                raise RequestNotSent('CONTROLLED_NOT_SENT_AFTER_PROGRESS')
        provider = Provider()
        app.dependency_overrides[get_runtime_settings] = lambda:self.settings
        with TestClient(app) as client:
            tid = self.task(client)
            TaskExecutor(self.settings,self.db,provider_factory=lambda _:provider,pdf_factory=lambda *_:FixturePDF()).execute(tid)
            view = client.get('/api/tasks/'+tid).json()
        result = view['result']
        self.assertIn('已读合成方法，实验未核实。',result['answer_body_markdown'])
        self.assertIn('CONTROLLED_NOT_SENT_AFTER_PROGRESS',result['processing_summary'])
        self.assertEqual(view['status'],'partial')
        self.assertIsNone(result['saved_path'])
        self.assertFalse(self.db.list_model_calls(tid)[-1]['status']=='unknown')
        self.assertFalse(list((self.root/'notes').rglob('*.md')))

    def test_failure_without_findings_reports_actual_read_scope(self):
        """无研究结论时仍列已读范围，不能声称论文缺证或独立核验已完成。"""
        tid = self.task(); self.papers(tid)
        result = TaskExecutor(self.settings,self.db)._deliverable_result(tid,{
            'evidence_ids':self.db.list_task_evidence(tid),'final_reason':'CONTROLLED_INVALID_REVIEW_JSON'})
        self.assertEqual(result.answer_kind,'failure_report')
        self.assertIn('物理页1, 2',result.answer_body_markdown)
        self.assertIn('CONTROLLED_INVALID_REVIEW_JSON',result.processing_summary)
        self.assertIsNone(result.saved_path)
