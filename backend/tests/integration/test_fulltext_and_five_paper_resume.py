"""验证全文传递及五篇澄清恢复；外部模型/PDF 用替身，业务服务与图实际运行。"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from research_loop.agents.supervisor import append_tool_observation
from research_loop.config import Settings
from research_loop.domain.evidence import Document, PageEvidence
from research_loop.domain.reading import user_page_range
from research_loop.domain.tasks import SavePolicy, TaskCreate
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs
from research_loop.tools.registry import ToolRegistry
from research_loop.workers.executor import TaskExecutor


class CachedPDF:
    """创建 25 页可定位缓存，实际分页读取仍调用生产 PDFProvider.pages。"""

    def __init__(self, root):
        self.root = root
        self.reader = PDFProvider(FileStore(root/'cache', root/'out', root))

    def ingest(self, source, paper_id):
        digest = hashlib.sha256(paper_id.encode()).hexdigest()
        doc_id = 'doc-' + digest[:20]
        pages = [PageEvidence(
            evidence_id=f'evidence:{paper_id}:{n}', document_id=doc_id, page_number=n,
            text=f'{paper_id} 第 {n} 页。' + ('完整正文' * 6000 if n == 1 else '语言特征证据。') + f'末尾标记-{n}',
            parsed_markdown=f'# 物理页 {n}\n{paper_id} language evidence; LAST-{n}',
        ).model_dump() for n in range(1, 26)]
        cache = self.root/(digest + '.json')
        cache.write_text(json.dumps({'total_pages':25, 'pages':pages}))
        return Document(document_id=doc_id, paper_id=paper_id, content_sha256=digest,
                        parser_name='fixture', parser_version='1', config_sha256='a'*64,
                        total_pages=25, cache_path=str(cache))

    def pages(self, document, start, count):
        return self.reader.pages(document, start, count)

    def close(self):
        self.reader.close()


class FulltextBoundaryTests(unittest.TestCase):
    """覆盖默认全文、显式限制、子角色权限与真实模型消息中的长文本。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = Database('sqlite:///' + str(self.root/'researchloop_test_fulltext.db'))
        self.db.ensure_schema()
        self.pdf = CachedPDF(self.root)
        self.addCleanup(self.pdf.close)
        self.papers = PaperService(self.db, None, self.pdf)
        self.registry = ToolRegistry(specs())

    def context(self, prompt, scope=None):
        """创建持久根任务，避免直接用模型 scope 伪造用户授权。"""
        task = TaskService(self.db).create(TaskCreate(prompt=prompt))
        return ToolContext(task_id=task.task_id, save_policy=SavePolicy.deny,
                           attachments={'pdf':{'local_path':'fixture.pdf'}},
                           services={'papers':self.papers, 'scope':scope or {}})

    def test_default_full_document_and_untruncated_model_message(self):
        ctx = self.context('阅读并比较语言信息，不要保存。')
        result = self.registry.execute('read_paper', {'target_ref':'pdf'}, ctx)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['returned_pages'], list(range(1,26)))
        self.assertTrue(result.data['read_coverage']['full_document_returned'])
        messages = []
        append_tool_observation(messages, 'read-1', result.model_dump())
        sent = json.loads(messages[0]['content'])['data']['pages']
        self.assertEqual(sent, result.data['pages'])
        self.assertIn('末尾标记-1', sent[0]['text'])
        self.assertIn('LAST-25', sent[-1]['parsed_markdown'])
        self.assertNotIn('observation truncated', messages[0]['content'])
        self.assertEqual(len(self.db.list_task_evidence(ctx.task_id)), 25)

    def test_agent_batch_hint_does_not_forbid_page_nine(self):
        ctx = self.context('阅读原文并比较。', {'input_refs':['pdf'], 'start_page':1, 'page_count':8})
        result = self.registry.execute('read_paper', {'target_ref':'pdf','start_page':9,'page_count':17}, ctx)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['returned_pages'], list(range(9,26)))
        self.assertFalse(result.data['read_coverage']['full_document_returned'])

    def test_explicit_user_range_is_enforced_for_all_roles(self):
        root = self.context('只看第1至2页，不要保存。')
        for role in ['supervisor','researcher','reviewer']:
            with self.subTest(role=role):
                if role == 'supervisor':
                    ctx = root
                else:
                    child = self.db.create_subtask(root.task_id,root.task_id,1 if role=='researcher' else 2,
                                                  'read paper',{},role)
                    ctx = ToolContext(task_id=child['subtask_id'],save_policy=SavePolicy.deny,
                                      role=role,attachments=root.attachments,services=root.services)
                default = self.registry.execute('read_paper',{'target_ref':'pdf'},ctx)
                self.assertTrue(default.ok, default.message)
                self.assertEqual(default.data['returned_pages'],[1,2])
                rejected = self.registry.execute('read_paper',{'target_ref':'pdf','start_page':3,'page_count':1},ctx)
                self.assertFalse(rejected.ok)
                self.assertIn('user-specified',rejected.message)

    def test_range_formats_and_ambiguities(self):
        for prompt, expected in [('只读前十二页',(1,12)), ('仅物理第1页',(1,1)),
                                 ('只依据附件前两页生成短稿并保存',(1,2)),
                                 ('only read physical pages 9-25',(9,25))]:
            with self.subTest(prompt=prompt):
                scope = user_page_range([prompt])
                self.assertEqual((scope.start,scope.end),expected)
        self.assertIsNone(user_page_range(['不要只看前8页，请读取全文。']))
        self.assertIsNone(user_page_range(['只读第1页','取消页数限制，读取全文。']))
        self.assertIsNone(user_page_range(['比较两篇论文，研究近三年的方法。','5篇']))
        for prompt in ['只看某几页', '只读第1页和第3页', '只读第3至1页', '只读第1页，另一篇只读第2页']:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                user_page_range([prompt])


class FivePaperResumeTests(unittest.TestCase):
    """实际 API→数据库→新 Worker→角色图验证；不宣称替身能证明 DeepSeek 语义表现。"""

    def test_original_intent_survives_five_paper_resume_and_three_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(project_root=root,data_dir=root/'data',output_dir=root/'notes',
                                database_url='sqlite:///'+str(root/'researchloop_test_five.db'))
            db = Database(settings.database_url)
            db.ensure_schema()
            prompt = '比较近三年关于开放词汇目标检测的论文如何利用语言信息。请先询问比较几篇，得到回答前不要检索，也不要保存文件。'
            requests = []
            batch_sizes = []
            seen_refs = []

            def response(name=None,args=None,content=''):
                message = {'role':'assistant','content':content}
                if name:
                    message['tool_calls'] = [{'id':str(uuid4()),'type':'function',
                                             'function':{'name':name,'arguments':json.dumps(args)}}]
                return httpx.Response(200,json={'id':str(uuid4()),'choices':[{
                    'finish_reason':'tool_calls' if name else 'stop','message':message}],
                    'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})

            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                messages = body['messages']
                validate_messages(messages)
                if messages[0]['content'].startswith('你是一名资深学术论文分析助手'):
                    self.assertIn('末尾标记-25',messages[1]['content'])
                    return response(content='## DPR 精读（供应商替身）\n语言信息证据已读。\n（完）')
                names = {t['function']['name'] for t in body.get('tools',[])}
                observations = [json.loads(m['content']) for m in messages if m['role']=='tool']
                if 'finish_research' in names:
                    brief = json.loads(messages[2]['content'].split('\n',1)[1])
                    self.assertIn('开放词汇目标检测', brief['objective'])
                    ref = brief['input_refs'][0]
                    if not observations:
                        seen_refs.append(ref)
                        return response('read_paper',{'target_ref':ref})
                    pages = observations[-1]['data']['pages']
                    self.assertEqual([p['page_number'] for p in pages],list(range(1,26)))
                    self.assertIn('末尾标记-1',pages[0]['text'])
                    return response('finish_research',{'findings':[f'{ref} language evidence page 25'],
                                                       'evidence_ids':[p['evidence_id'] for p in pages],'gaps':[]})
                if 'finish_review' in names:
                    bundle = json.loads(messages[-1]['content'].split('\n',1)[1])
                    self.assertEqual(len(bundle),125)
                    return response('finish_review',{'status':'supported','findings':[],
                                                      'gaps':[]})
                # 每一轮 Supervisor 都能看到原始题意、时间条件及禁止保存。
                self.assertIn(prompt,messages[1]['content'])
                self.assertIn('后端保存权限：deny',messages[1]['content'])
                done = [call['function']['name'] for m in messages for call in m.get('tool_calls',[])]
                if not done:
                    return response('request_clarification',{'question':'比较几篇？','fields':['paper_count']})
                self.assertTrue(any(m['role']=='user' and m['content']=='澄清回答：5篇' for m in messages))
                batches = done.count('delegate_research')
                if batches < 3:
                    refs = ['p0','p1'] if batches==0 else ['p2','p3'] if batches==1 else ['p4']
                    batch_sizes.append(len(refs))
                    return response('delegate_research',{'subtasks':[{
                        'objective':'比较近三年开放词汇目标检测论文如何利用语言信息，不保存。',
                        # 本例显式保留旧精读恢复路径；新任务的意图分流另有覆盖。
                        'mode':'deep_read',
                        'input_refs':[ref]} for ref in refs]})
                if 'create_note' not in done:
                    ids = [e for o in observations for s in o.get('data',{}).get('subtasks',[]) for e in s['evidence_ids']]
                    return response('create_note',{'title':'五篇比较','markdown':'# 五篇语言信息比较（隔离替身）',
                                                   'evidence_ids':ids})
                draft = next(o['data'] for o in observations if 'draft_id' in o.get('data',{}))
                if 'review_note' not in done:
                    return response('review_note',{'draft_id':draft['draft_id'],'content_hash':draft['content_hash']})
                return response(content='已比较5篇的语言信息，未保存文件。')

            def factory(_):
                return DeepSeekProvider('fixture',client=httpx.Client(transport=httpx.MockTransport(handler)))

            def executor():
                return TaskExecutor(settings,db,factory,lambda s,f:CachedPDF(root))

            attachments = []
            for n in range(5):
                path = root/f'p{n}.pdf'
                path.write_bytes(b'%PDF-fixture')
                attachments.append({'input_ref':f'p{n}','local_path':str(path)})
            from research_loop.main import app
            with patch('research_loop.api.tasks.get_settings',return_value=settings),TestClient(app) as client:
                created = client.post('/api/tasks',json={'prompt':prompt,'save_policy':'deny',
                                                        'attachments':attachments}).json()
                task_id = created['task_id']
                # 保留旧五篇精读的澄清/恢复合同；新模式的覆盖交付在adaptive测试中验证。
                from research_loop.agents.graph import initial_state
                db.save_checkpoint(task_id, initial_state(task_id, prompt, 'deny',
                    {a['input_ref']: a for a in attachments}, 48, 80, 10))
                executor().execute(task_id)
                paused = client.get(f'/api/tasks/{task_id}').json()
                self.assertEqual(paused['status'],'needs_input')
                self.assertEqual(len(requests),1)
                resumed = client.post(f'/api/tasks/{task_id}/resume',json={
                    'clarification_id':paused['clarification']['clarification_id'],
                    'answer':'5篇','idempotency_key':'five-paper-answer'})
                self.assertEqual(resumed.status_code,200,resumed.text)
                executor().execute(task_id,'new-worker-after-clarification')
                result = client.get(f'/api/tasks/{task_id}').json()
            self.assertEqual(batch_sizes,[2,2,1])
            self.assertEqual(sorted(seen_refs),['p0','p1','p2','p3','p4'])
            self.assertEqual(result['prompt'],prompt)
            self.assertEqual(result['status'],'completed',result)
            self.assertEqual(len(result['subtasks']),6)
            self.assertIsNone(result['result']['saved_path'])
            self.assertEqual(len(list((root/'notes').rglob('*.md'))),0)
            self.assertEqual(result['usage']['unknown'],0)
