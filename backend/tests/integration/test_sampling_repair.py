"""采样失败修复的隔离链路：真实API/存储/图，替身PDF及供应商。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.services.coverage import CoverageService
from research_loop.services.tasks import TaskService
from research_loop.domain.tasks import TaskCreate
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_collaboration_contract import FixturePDF
from integration.test_adaptive_research import AdaptiveScenario


class LargeSourcePDF(FixturePDF):
    """可定位合成大页，触发子角色原文投影；不联网下载或编码。"""
    def pages(self, document, start, count):
        """重复文本仅用于容量边界，真实业务数据没有注入该内容。"""
        pages=super().pages(document,start,count)
        for page in pages:
            page.text='Language embeddings condition the decoder.\n' + 'Unrelated original details. '*2600 + '\nTail: the dataset is shared.'
            page.parsed_markdown=page.text
        return pages


class RepairScenario:
    """从实际角色观察选择固定行动，不声称模型自主质量。"""
    def __init__(self, attachments, save, invalid):
        """配置合成核验错误与保存权限，并收集实际messages。"""
        self.attachments=attachments;self.save=save;self.invalid=invalid;self.requests=[]

    def handler(self, request):
        """先验证消息及投影来源，响应经过真实收据和工具执行链。"""
        body=json.loads(request.content);self.requests.append(body)
        messages=body['messages'];validate_messages(messages)
        names={t['function']['name'] for t in body.get('tools',[])}
        done=[c['function']['name'] for m in messages for c in m.get('tool_calls',[])]
        obs=[json.loads(m['content']) for m in messages if m['role']=='tool']
        reply=AdaptiveScenario.response
        if 'finish_research' in names:
            brief=json.loads(next(m['content'] for m in messages if m['role']=='user' and m['content'].startswith('结构化子任务')).split('\n',1)[1])
            assert len(brief['input_refs'])==1 and not brief['paper_ids']
            assert not any(m['content'].startswith('[ResearchLoop Memory') for m in messages if isinstance(m.get('content'),str))
            ref=brief['input_refs'][0]
            if 'read_paper' not in done:return reply('read_paper',{'target_ref':ref})
            data=next(o['data'] for o in obs if o.get('data',{}).get('pages'))
            page=data['pages'][0]
            assert page['context_excerpted'] and page['context_readback']['target_ref']==ref
            assert page['context_spans'][0]['text'].startswith('Language embeddings')
            args={'status':'completed','findings':['语言嵌入作用于解码器；来自合成原文。'],
                'paper_ids':[data['paper_id']],'evidence_ids':[p['evidence_id'] for p in data['pages']]}
            if self.invalid:
                args['assessments']=[{'claim':'语言作用','status':'supported','basis':'direct_statement',
                    'reason':'测试错误摘录','source_quote':'Language ... decoder.', 'evidence_ids':[page['evidence_id']]}]
            return reply('finish_research',args)
        if 'set_research_plan' not in done:
            return reply('set_research_plan',{'goal':'比较两份已上传材料','task_kind':'compare','save_requested':self.save,
                'goals':[{'goal_id':'language','description':'比较语言信息','min_papers':2,'requires_fulltext':True}]})
        if 'delegate_research' not in done:
            specs=[{'objective':'比较语言信息 '+a['input_ref'],'paper_ids':[a['paper_id']],'mode':'investigate','goal_ids':['language']} for a in self.attachments]
            return reply('delegate_research',{'subtasks':specs})
        children=next(o['data']['subtasks'] for o in obs if o.get('data',{}).get('subtasks'))
        assert len(children)==2 and all(c['findings'] for c in children)
        if self.invalid:assert all(c['status']=='partial' and c['assessments'][0]['status']=='insufficient' for c in children)
        if self.save:
            if 'create_note' not in done:
                return reply('create_note',{'title':'合成比较候选','markdown':'# 比较\n两篇采用语言嵌入。'+('部分核验未完成。' if self.invalid else ''),
                    'evidence_ids':[e for c in children for e in c['evidence_ids']]})
            draft=next(o['data'] for o in obs if o.get('data',{}).get('draft_id'))
            if 'save_note' not in done:return reply('save_note',{'draft_id':draft['draft_id'],'content_hash':draft['content_hash']})
        else:draft=None
        return reply('finish_task',{'answer_markdown':'两篇采用语言嵌入。'+('引句核验有缺口，保留候选。' if self.invalid else ''),
            'draft_id':draft['draft_id'] if draft else None,
            'goals':[{'goal_id':'language','status':'unresolved' if self.invalid else 'satisfied',
                'subtask_ids':[c['subtask_id'] for c in children],'reason':'按当前候选与核验状态交付'}]})

    def factory(self, settings):
        """只用内存HTTP传输，不允许实际供应商网络。"""
        return DeepSeekProvider('fixture',client=httpx.Client(transport=httpx.MockTransport(self.handler)))


class SamplingRepairTests(unittest.TestCase):
    """既可本机SQLite，也可显式专用MySQL；不清理任何已有库。"""
    def setUp(self):
        """每例独立身份/输出目录，数据库名需明确隔离。"""
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        url=os.getenv('SAMPLING_REPAIR_DATABASE_URL') or 'sqlite:///'+str(self.root/'test.db')
        if not url.startswith('sqlite:') and not url.rsplit('/',1)[-1].startswith('researchloop_test_sampling_repair_'):
            raise ValueError('dedicated sampling repair database required')
        self.settings=Settings(project_root=self.root,database_url=url,data_dir=self.root/'data',output_dir=self.root/'notes',
            retrieval_enabled=False,max_model_requests=16,reserve_model_requests=1,task_token_limit=2000000)
        self.db=Database(url);self.db.ensure_schema();self.prefix=uuid4().hex
        self.attachments=[]
        for name in ('A','B'):
            path=self.root/(name+'.pdf');path.write_bytes(b'%PDF-fixture')
            self.attachments.append({'input_ref':self.prefix+name,'paper_id':'paper-'+self.prefix+name,'local_path':str(path)})

    def run_case(self,save,invalid):
        """测试实际API→Worker→双子任务→交付，终态恢复不重复调用或保存。"""
        from research_loop.main import app
        scenario=RepairScenario(self.attachments,save,invalid)
        executor=TaskExecutor(self.settings,self.db,scenario.factory,lambda s,f:LargeSourcePDF())
        with patch('research_loop.api.dependencies.get_settings',return_value=self.settings),TestClient(app) as client:
            r=client.post('/api/tasks',json={'prompt':'比较附件两篇。'+('保存待核实比较笔记。' if save else '不要保存。'),
                'save_policy':'allow' if save else 'deny','idempotency_key':self.prefix,'attachments':self.attachments})
            self.assertEqual(r.status_code,202,r.text);tid=r.json()['task_id'];executor.execute(tid)
            view=client.get('/api/tasks/'+tid).json()
        self.assertEqual(view['status'],'partial' if invalid else 'completed',view['result'])
        self.assertEqual(len(self.db.list_subtasks(tid)),2)
        for child in self.db.list_subtasks(tid):
            state=self.db.get_checkpoint(child['subtask_id'])
            self.assertTrue(state['context_metrics'])
            pages=[p for m in state['messages'] if m['role']=='tool' for p in json.loads(m['content']).get('data',{}).get('pages',[])]
            self.assertTrue(pages);self.assertGreater(len(pages[0]['text']),60000)
            self.assertNotIn('context_excerpted',pages[0])
        paths=list(self.settings.output_dir.rglob('*.md'));self.assertEqual(len(paths),1 if save else 0)
        original={str(p):p.read_bytes() for p in paths};n=len(scenario.requests)
        self.db.update_task(tid,status='queued',result_json=None);executor.execute(tid,'recovery')
        self.assertEqual(len(scenario.requests),n);self.assertEqual({str(p):p.read_bytes() for p in paths},original)
        # 已有同一子任务以别名或input_ref恢复，重复条目不得再次执行。
        child=self.db.list_subtasks(tid)[0];spec=child['spec'];ref=spec['input_refs'][0]
        alias=next(a['paper_id'] for a in self.attachments if a['input_ref']==ref)
        self.db.execute('UPDATE subtasks SET idempotency_key=? WHERE subtask_id=?',('legacy-'+self.prefix,child['subtask_id']))
        restored=executor._run_research_batch(tid,[spec,{**spec,'paper_ids':[alias],'input_refs':[]}],
            'no-new-worker','unused',self.settings,None)
        self.assertEqual(len(restored),1);self.assertEqual(len(scenario.requests),n)
        self.assertEqual(restored[0]['status'],'partial' if invalid else 'completed')
        self.assertTrue(restored[0]['findings'])
        return tid

    def test_normal_projection_handoff_and_saved_delivery(self):
        """大页正常候选可保存，两个角色均不因字节体积直接失败。"""
        self.run_case(True,False)

    def test_optional_quote_failure_keeps_partial_and_deny(self):
        """引句错误保留候选与原因，禁止保存仍无Markdown文件。"""
        self.run_case(False,True)

    def test_reference_ambiguity_foreign_and_equivalence(self):
        """别名与input_ref稳定等价，版本歧义及外来ID仍拒绝。"""
        task=TaskService(self.db,self.settings).create(TaskCreate(prompt='读取附件',attachments=self.attachments))
        service=CoverageService(self.db);a=self.attachments[0]
        self.assertEqual(service.normalize_references(task.task_id,{'paper_ids':[a['paper_id']]}),
            service.normalize_references(task.task_id,{'input_refs':[a['input_ref'],a['input_ref']]}))
        for spec in ({'paper_ids':['foreign']},{'input_refs':['foreign']}):
            with self.assertRaises(PermissionError):service.normalize_references(task.task_id,spec)
        same=[a,{**self.attachments[1],'paper_id':a['paper_id']}]
        self.db.execute('UPDATE tasks SET attachments_json=? WHERE task_id=?',(json.dumps(same),task.task_id))
        with self.assertRaisesRegex(PermissionError,'ambiguous'):service.normalize_references(task.task_id,{'paper_ids':[a['paper_id']]})

    def test_textual_delivery_api_is_partial_and_never_saved(self):
        """真实响应解析链拒绝执行文本JSON，保留可读候选与原receipt。"""
        from research_loop.main import app
        content=json.dumps({'goals':[{'goal_id':'invented','status':'satisfied'}],
                            'answer_markdown':'候选回答','draft_id':'foreign'},ensure_ascii=False)
        calls=[]
        def handler(request):
            """返回无tool_calls的文本化结束参数，不产生真实网络。"""
            calls.append(request)
            return AdaptiveScenario.response(content=content)
        executor=TaskExecutor(self.settings,self.db,lambda _:DeepSeekProvider('fixture',client=httpx.Client(transport=httpx.MockTransport(handler))),lambda s,f:FixturePDF())
        with patch('research_loop.api.dependencies.get_settings',return_value=self.settings),TestClient(app) as client:
            res=client.post('/api/tasks',json={'prompt':'根据已有信息回答，不保存','save_policy':'deny'})
            tid=res.json()['task_id'];executor.execute(tid)
            view=client.get('/api/tasks/'+tid).json()
        self.assertEqual(view['status'],'partial')
        self.assertIn('未核验候选',view['result']['answer_markdown'])
        self.assertIn('候选回答',view['result']['answer_markdown'])
        self.assertFalse(view['result']['saved_path']);self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
        self.assertEqual(len(calls),1)
        receipt=self.db.list_model_calls(tid)[0]
        self.assertEqual(json.loads(receipt['response_json'])['choices'][0]['message']['content'],content)
