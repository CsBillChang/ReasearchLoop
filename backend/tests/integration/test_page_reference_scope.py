"""正常API与读取服务验证出处页码要求；模型和外部网络全部禁止。"""
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from integration.test_collaboration_contract import FixturePDF
from research_loop.agents.graph import initial_state
from research_loop.api.dependencies import get_runtime_settings
from research_loop.config import Settings
from research_loop.domain.evidence import PageEvidence
from research_loop.domain.tasks import SavePolicy
from research_loop.main import app
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs
from research_loop.tools.registry import ToolRegistry
from research_loop.storage.mysql import Database
from research_loop.storage.checkpoints import CheckpointStore
from unit.test_named_attachment_scope import ORIGINAL_TEXT_PROMPT


class NamedPageScopePDF(FixturePDF):
    """六页合成正文用于验证单页/相邻页权限，不模拟真实论文质量。"""

    def __init__(self):
        """保留读取次数，证明越界请求在获取正文前被拒绝。"""
        self.ingests = 0

    def ingest(self, source, paper_id):
        """沿用来源身份，提供第4页及其相邻页的明确测试边界。"""
        self.ingests += 1
        return super().ingest(source, paper_id).model_copy(update={'total_pages': 6})

    def pages(self, document, start, count):
        """只返回读取服务已经批准的合成页。"""
        return [PageEvidence(evidence_id=f'evidence:{document.paper_id}:{n}',
            document_id=document.document_id, page_number=n,
            text=f'{document.paper_id}: synthetic structure on physical page {n}')
            for n in range(start, min(start + count, 7))]


class PageReferenceReadingTests(unittest.TestCase):
    """真实任务/权限存储与工具入口，合成PDF不代替原材料读回验证。"""

    def setUp(self):
        """新建临时目录或显式隔离MySQL，每个测试使用独立任务。"""
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        url=os.getenv('PAGE_SCOPE_TEST_DATABASE_URL') or 'sqlite:///'+str(self.root/'test.db')
        if not url.startswith('sqlite:') and not url.rsplit('/',1)[-1].startswith('researchloop_test_page_scope_'):
            raise ValueError('page-scope tests require isolated MySQL')
        self.settings=Settings(project_root=self.root,data_dir=self.root/'data',output_dir=self.root/'notes',database_url=url,retrieval_enabled=False)
        self.db=Database(url);self.db.ensure_schema();self.roots=[]
        old=app.dependency_overrides.copy();app.dependency_overrides[get_runtime_settings]=lambda:self.settings
        def restore_overrides():
            """只恢复测试前依赖，不影响其他用例的配置。"""
            app.dependency_overrides.clear();app.dependency_overrides.update(old)
        self.addCleanup(restore_overrides)
        for method in ('send','send_stream'):
            guard=patch.object(DeepSeekProvider,method,side_effect=AssertionError('real models forbidden'))
            guard.start();self.addCleanup(guard.stop)
        network=patch.object(httpx.HTTPTransport,'handle_request',side_effect=AssertionError('external HTTP forbidden'))
        network.start();self.addCleanup(network.stop)
        self.api=TestClient(app);self.api.__enter__();self.addCleanup(self.api.__exit__,None,None,None)
        self.papers=PaperService(self.db,None,FixturePDF());self.registry=ToolRegistry(specs())
        self.addCleanup(self.close_tasks)

    def close_tasks(self):
        """正常取消未执行夹具，收据保持为空，不留下待执行任务。"""
        for tid in self.roots:TaskService(self.db).cancel(tid)

    def create(self, prompt, filenames=None):
        """由正常HTTP路由固定根用户原话与禁止保存权限。"""
        attachments=[]
        for name in ('A','B'):
            path=self.root/(name+'.pdf');path.write_bytes(b'%PDF-fixture')
            attachments.append({'input_ref':'input-'+name,'paper_id':'paper-'+uuid4().hex,'local_path':str(path), 'filename': (filenames or {}).get(name)})
        response=self.api.post('/api/tasks',json={'prompt':prompt,'save_policy':'deny','attachments':attachments,'idempotency_key':uuid4().hex})
        self.assertEqual(response.status_code,202,response.text)
        tid=response.json()['task_id'];self.roots.append(tid)
        return tid

    def context(self, tid, role):
        """子角色按真实根任务恢复阅读限制，不用模型分派文字替代用户权限。"""
        current=tid
        if role!='supervisor':
            child=self.db.create_subtask(tid,tid,1 if role=='researcher' else 2,'不影响用户阅读范围的合成子目标',{},uuid4().hex)
            current=child['subtask_id'];self.db.update_task(current,role=role)
        attachments={a['input_ref']:a for a in json.loads(self.db.get_task(tid)['attachments_json'])}
        if current != tid:
            for ref in attachments:
                self.db.grant_input(current, ref, tid)
            self.db.execute('UPDATE tasks SET attachments_json=? WHERE task_id=?', (json.dumps(list(attachments.values())), current))
        return ToolContext(task_id=current,root_task_id=tid,role=role,save_policy=SavePolicy.deny,attachments=attachments,
                           services={'papers':self.papers,'scope':{'input_refs':['input-A','input-B']}})

    def assert_no_side_effects(self, tid):
        """读页只保存证据，不调用模型或保存笔记。"""
        view=self.api.get('/api/tasks/'+tid).json();self.assertEqual(view['save_policy'],'deny')
        self.assertEqual(self.db.fetchall('SELECT m.call_id FROM model_calls m JOIN tasks t ON t.task_id=m.task_id WHERE t.root_task_id=?',(tid,)),[])
        self.assertEqual(list(self.root.rglob('*.md')),[])

    def test_reference_request_reads_fulltext_for_all_roles(self):
        """三角色读同一原话均默认全文，来源仍有物理页号。"""
        tid=self.create('只依据这两份附件并注明原文页码，不保存。')
        for role in ('supervisor','researcher','reviewer'):
            ctx=self.context(tid,role)
            for ref in ('input-A','input-B'):
                with self.subTest(role=role,ref=ref):
                    value=self.registry.execute('read_paper',{'target_ref':ref},ctx)
                    self.assertTrue(value.ok,value.message);self.assertEqual(value.data['returned_pages'],[1,2])
                    self.assertIsNone(value.data['read_coverage']['user_page_range'])
                    self.assertEqual(value.data['read_coverage']['scope_origin'],'full_document')
        self.assert_no_side_effects(tid)

    def test_explicit_limit_and_cross_role_overflow_stay_protected(self):
        """真实限制仍仅返回第1页，三角色请求第2页全部被拒绝。"""
        tid=self.create('只读物理第1页，并注明原文页码，不保存。')
        for role in ('supervisor','researcher','reviewer'):
            ctx=self.context(tid,role)
            value=self.registry.execute('read_paper',{'target_ref':'input-A'},ctx)
            self.assertTrue(value.ok,value.message);self.assertEqual(value.data['returned_pages'],[1])
            self.assertEqual(value.data['read_coverage']['scope_origin'],'user')
            rejected=self.registry.execute('read_paper',{'target_ref':'input-A','start_page':2,'page_count':1},ctx)
            self.assertFalse(rejected.ok);self.assertIn('user-specified',rejected.message)
        self.assert_no_side_effects(tid)

    def test_unclear_and_conflicting_ranges_do_not_grant_fulltext(self):
        """不明确、冲突、离散范围仍在获取正文前阻断。"""
        for text in ('只看某几页，并注明原文页码。','只读第1页和第3页，并注明原文页码。',
                     'A仅看第1页，A仅看第2页，并注明原文页码。'):
            tid=self.create(text);ctx=self.context(tid,'researcher')
            value=self.registry.execute('read_paper',{'target_ref':'input-A'},ctx)
            self.assertFalse(value.ok);self.assertIn('page range',value.message)
            self.assertEqual(self.db.list_task_evidence(ctx.task_id),[])
            self.assert_no_side_effects(tid)

    def test_reference_clarification_does_not_erase_actual_limit(self):
        """恢复澄清后增加出处要求，根任务限制仍向子角色传播。"""
        tid=self.create('只读物理第1页，不保存。')
        service=TaskService(self.db);question=service.request_clarification(tid,'补充要求？',['output'],[])
        self.db.update_task(tid,status='needs_input')
        response=self.api.post('/api/tasks/'+tid+'/resume',json={'clarification_id':question['clarification_id'],
            'answer':'请注明原文页码。','idempotency_key':uuid4().hex})
        self.assertEqual(response.status_code,409,response.text)
        self.assertIn('checkpoint is incomplete',response.text)
        state=initial_state(tid,self.db.get_task(tid)['prompt'],'deny')
        state.update(finished=True,final_reason='clarification requested')
        CheckpointStore(self.db).save(tid,state)
        response=self.api.post('/api/tasks/'+tid+'/resume',json={'clarification_id':question['clarification_id'],
            'answer':'请注明原文页码。','idempotency_key':uuid4().hex})
        self.assertEqual(response.status_code,200,response.text)
        ctx=self.context(tid,'reviewer');value=self.registry.execute('read_paper',{'target_ref':'input-A'},ctx)
        self.assertTrue(value.ok,value.message);self.assertEqual(value.data['returned_pages'],[1])
        self.assert_no_side_effects(tid)

    def test_attachment_scope_is_not_expanded_by_citation_wording(self):
        """允许全文仍只限分派的附件，取消也优先于读取。"""
        tid=self.create('只依据附件，注明原文页码，不保存。');ctx=self.context(tid,'researcher')
        ctx.services['scope']={'input_refs':['input-A']}
        rejected=self.registry.execute('read_paper',{'target_ref':'input-B'},ctx)
        self.assertFalse(rejected.ok);self.assertIn('assigned references',rejected.message)
        ctx.cancelled=lambda:True
        cancelled=self.registry.execute('read_paper',{'target_ref':'input-A'},ctx)
        self.assertEqual(cancelled.kind,'cancelled');self.assertEqual(self.db.list_task_evidence(ctx.task_id),[])
        self.assert_no_side_effects(tid)

    def test_original_named_instruction_reads_four_and_rejects_neighbours_for_all_roles(self):
        """完整原指令经API/工具/服务读取第4页，三角色均不能扩大范围。"""
        pdf = NamedPageScopePDF()
        self.papers = PaperService(self.db, None, pdf)
        tid = self.create(ORIGINAL_TEXT_PROMPT, {'A': 'LaVer.pdf'})
        for role in ('supervisor', 'researcher', 'reviewer'):
            ctx = self.context(tid, role)
            with self.subTest(role=role):
                value = self.registry.execute('read_paper', {'target_ref': 'input-A'}, ctx)
                self.assertTrue(value.ok, value.message)
                self.assertEqual(value.data['returned_pages'], [4])
                self.assertEqual(value.data['read_coverage']['user_page_range']['start'], 4)
                self.assertEqual(value.data['read_coverage']['user_page_range']['end'], 4)
                before = pdf.ingests
                for start, count in ((3, 1), (5, 1), (4, 2), (1, 4)):
                    rejected = self.registry.execute('read_paper',
                        {'target_ref': 'input-A', 'start_page': start, 'page_count': count}, ctx)
                    self.assertFalse(rejected.ok)
                    self.assertIn('user-specified', rejected.message)
                self.assertEqual(pdf.ingests, before)
                self.assertEqual([self.db.get_page(eid)['page_number']
                    for eid in self.db.list_task_evidence(ctx.task_id)], [4])
        self.assertEqual(self.db.get_task(tid)['prompt'], ORIGINAL_TEXT_PROMPT)
        self.assert_no_side_effects(tid)

    def test_named_wording_does_not_bypass_ambiguous_limit_or_attachment_scope(self):
        """新句式的歧义仍先拒绝；合法范围也不授权其他附件或取消后读取。"""
        pdf = NamedPageScopePDF()
        self.papers = PaperService(self.db, None, pdf)
        for prompt in (
            '仅根据附件 Alpha 物理前面几页，注明原文页码。',
            '仅根据附件 Alpha 物理第4页和第6页，不保存。',
            '仅根据附件 Alpha 物理第4页；只依据附件 Beta 的物理第5页。',
        ):
            tid = self.create(prompt)
            ctx = self.context(tid, 'researcher')
            before = pdf.ingests
            value = self.registry.execute('read_paper', {'target_ref': 'input-A'}, ctx)
            self.assertFalse(value.ok)
            self.assertIn('page range', value.message)
            self.assertEqual(pdf.ingests, before)
            self.assertEqual(self.db.list_task_evidence(ctx.task_id), [])
            self.assert_no_side_effects(tid)
        tid = self.create(ORIGINAL_TEXT_PROMPT)
        ctx = self.context(tid, 'researcher')
        ctx.services['scope'] = {'input_refs': ['input-A']}
        rejected = self.registry.execute('read_paper', {'target_ref': 'input-B'}, ctx)
        self.assertFalse(rejected.ok)
        self.assertIn('assigned references', rejected.message)
        ctx.cancelled = lambda: True
        cancelled = self.registry.execute('read_paper', {'target_ref': 'input-A'}, ctx)
        self.assertEqual(cancelled.kind, 'cancelled')
        self.assert_no_side_effects(tid)


if __name__=='__main__':unittest.main()
