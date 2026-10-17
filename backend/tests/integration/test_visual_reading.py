"""正常任务API→读取服务→视觉节点→HTTP替身→收据；可显式使用隔离MySQL。"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from research_loop.agents.visual_reading import VisualReadingNode
from research_loop.api.dependencies import get_runtime_settings
from research_loop.config import Settings
from research_loop.domain.visual_reading import build_request, VisualConfig
from research_loop.main import app
from research_loop.providers.deepseek import DeepSeekProvider, ProviderResponseError
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.storage.token_budget import TokenBudgetStore
from unit.test_visual_reading import fixture


class VisualIntegrationTests(unittest.TestCase):
    """验证权限/恢复与真实SQL；供应商返回固定替身，不代表视觉理解通过。"""

    def setUp(self):
        """默认SQLite或显式隔离MySQL；禁止外部HTTP发送。"""
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        self.files, self.pdf_path, self.item, self.pages = fixture(self.root)
        url = os.getenv('VISUAL_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root/'test.db')
        if not url.startswith('sqlite:') and not url.rsplit('/',1)[-1].startswith('researchloop_test_visual_'):
            raise ValueError('visual tests require isolated database')
        self.settings = Settings(project_root=self.root, data_dir=self.files.data_dir,
            output_dir=self.files.output_dir, database_url=url, retrieval_enabled=False, deepseek_model='deepseek-flash')
        self.db=Database(url); self.db.ensure_schema(); self.tasks=[]; self.calls=[]
        old=app.dependency_overrides.copy(); app.dependency_overrides[get_runtime_settings]=lambda:self.settings
        def restore():
            """恢复测试前依赖，不更改生产配置。"""
            app.dependency_overrides.clear(); app.dependency_overrides.update(old)
        self.addCleanup(restore)
        network=patch.object(httpx.HTTPTransport,'handle_request',side_effect=AssertionError('external HTTP forbidden'))
        network.start(); self.addCleanup(network.stop)
        self.api=TestClient(app); self.api.__enter__(); self.addCleanup(self.api.__exit__,None,None,None)
        self.pdf=PDFProvider(self.files); self.addCleanup(self.pdf.close)
        parse=patch.object(self.pdf,'_parse',side_effect=AssertionError('no full PDF reparse'))
        parse.start(); self.addCleanup(parse.stop)
        self.papers=PaperService(self.db,None,self.pdf)
        self.http=httpx.Client(transport=httpx.MockTransport(self.respond),timeout=120); self.addCleanup(self.http.close)
        self.provider=DeepSeekProvider('synthetic-key',model='deepseek-flash',client=self.http)
        self.node=VisualReadingNode(self.db,self.papers,self.provider)
        self.addCleanup(self.cleanup_tasks)

    def cleanup_tasks(self):
        """隔离夹具终态化；成功/失败收据均保留。"""
        for tid in self.tasks:
            self.db.update_task(tid,status='cancelled',cancel_requested=1)

    def respond(self, request):
        """固定HTTP响应，保留发送payload供A/B及恢复断言。"""
        self.calls.append(json.loads(request.content))
        return httpx.Response(200,json={'id':'synthetic-'+str(len(self.calls)),'model':'synthetic-flash',
            'choices':[{'message':{'content':'合成观察：A为12.0，B为24.0。'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}})

    def create(self, prompt='只依据附件，注明原文页码，不保存。'):
        """通过现有HTTP路由创建deny任务并领取专用租约。"""
        response=self.api.post('/api/tasks',json={'prompt':prompt,'save_policy':'deny','idempotency_key':uuid4().hex,
            'attachments':[{'input_ref':self.item.target_ref,'local_path':str(self.pdf_path),'paper_id':'synthetic-paper'}]})
        self.assertEqual(response.status_code,202,response.text)
        tid=response.json()['task_id']; self.tasks.append(tid)
        claimed=self.db.claim_task('visual-test',task_id=tid)
        self.assertIsNotNone(claimed)
        return tid, {'worker_id':'visual-test','lease_token':claimed['lease_token']}

    def test_normal_receipt_replay_and_no_source_overwrite(self):
        """真实服务准备输入，HTTP收据回放保持request_id/usage，禁止笔记写入。"""
        tid,actor=self.create(); before=self.pdf_path.read_bytes()
        first=self.node.run(tid,self.item,**actor); second=self.node.run(tid,self.item,**actor)
        self.assertEqual(len(self.calls),1); self.assertTrue(second['replayed'])
        self.assertEqual(first['request_id'],second['request_id']); self.assertEqual(first['usage'],second['usage'])
        checkpoint=self.db.get_checkpoint(tid)
        self.assertNotIn('base64',json.dumps(checkpoint)); self.assertFalse(first['source']['semantically_verified'])
        self.assertEqual(self.db.get_page(first['source']['evidence_id'])['text'],self.pages[0]['text'])
        self.assertEqual(self.pdf_path.read_bytes(),before); self.assertEqual(list(self.files.output_dir.rglob('*.md')),[])
        self.assertEqual(self.db.get_task(tid)['save_policy'],'deny')
        self.assertEqual(self.db.assert_budget_policy(tid,self.settings)['model_used'],1)
        self.assertEqual(TokenBudgetStore(self.db).summary(tid)['used_tokens'],120)

    def test_ab_same_text_through_actual_http_adapter(self):
        """配对两任务除图片之外实际供应商消息相同。"""
        for group in ('text','visual'):
            tid,actor=self.create(); self.node.run(tid,self.item.model_copy(update={'group':group}),**actor)
        a,b=deepcopy(self.calls)
        b['messages'][1]['content']=b['messages'][1]['content'][:1]
        self.assertEqual(a,b); self.assertEqual(a['thinking'],{'type':'disabled'}); self.assertNotIn('tools',a)

    def test_success_then_process_exit_recovers_without_request(self):
        """收据已落盘、checkpoint前退出；恢复只读旧收据。"""
        tid,actor=self.create()
        with patch.object(self.node,'_checkpoint',side_effect=SystemExit('controlled post-receipt exit')):
            with self.assertRaises(SystemExit): self.node.run(tid,self.item,**actor)
        result=self.node.run(tid,self.item,**actor)
        self.assertTrue(result['replayed']); self.assertEqual(len(self.calls),1)

    def test_started_sent_and_unknown_are_not_retried(self):
        """已进入发送窗口后中断与超时都不能重发；未知usage不填0。"""
        for exc in (SystemExit('controlled send exit'),TimeoutError('synthetic timeout')):
            tid,actor=self.create()
            with patch.object(self.provider,'send',side_effect=exc):
                with self.assertRaises((SystemExit,RuntimeError)): self.node.run(tid,self.item,**actor)
            with self.assertRaises(RuntimeError): self.node.run(tid,self.item,**actor)
            receipt=self.db.get_model_call(tid,1)
            self.assertIn(receipt['status'],{'started','unknown'}); self.assertEqual(json.loads(receipt['usage_json']),{})
        self.assertEqual(self.calls,[])

    def test_not_sent_intent_can_continue_once(self):
        """尚未进入发送窗口的意图可恢复，请求槽位不重复占用。"""
        tid,actor=self.create()
        with patch.object(self.node,'_mark_sent',side_effect=SystemExit('controlled pre-send exit')):
            with self.assertRaises(SystemExit): self.node.run(tid,self.item,**actor)
        self.node.run(tid,self.item,**actor)
        self.assertEqual(len(self.calls),1); self.assertEqual(self.db.budget_summary(tid)['model_used'],1)

    def test_version_and_question_changes_do_not_reuse_receipt(self):
        """原问题/页文/附件版本变化均拒绝，不冒用旧成功收据。"""
        tid,actor=self.create(); self.node.run(tid,self.item,**actor)
        for item in (self.item.model_copy(update={'question':'不同问题'}), self.item.model_copy(update={'page_text_sha256':'0'*64})):
            with self.assertRaises(ValueError): self.node.run(tid,item,**actor)
        self.pdf_path.write_bytes(b'%PDF changed')
        with self.assertRaises(ValueError): self.node.run(tid,self.item,**actor)
        self.assertEqual(len(self.calls),1)

    def test_page_scope_attachment_and_missing_cache_reject_before_send(self):
        """真正的页范围仍有效；未绑定附件及无缓存不回退获取全文。"""
        tid,actor=self.create('只读物理第2页，并注明原文页码，不保存。')
        with self.assertRaises(PermissionError): self.node.run(tid,self.item,**actor)
        tid,actor=self.create()
        with self.assertRaises(PermissionError): self.node.run(tid,self.item.model_copy(update={'target_ref':'other'}),**actor)
        next(self.files.parse_dir.glob('*.json')).unlink()
        with self.assertRaises(ValueError): self.node.run(tid,self.item,**actor)
        self.assertEqual(self.calls,[])

    def test_cancel_and_lost_lease_reject_before_send(self):
        """取消与失权阻止读取/请求，不写成功观察。"""
        tid,actor=self.create(); TaskService(self.db,self.settings).cancel(tid)
        with self.assertRaises(PermissionError): self.node.run(tid,self.item,**actor)
        tid,actor=self.create(); self.db.update_task(tid,lease_token='new-owner')
        with self.assertRaises(PermissionError): self.node.run(tid,self.item,**actor)
        self.assertEqual(self.calls,[])

    def test_loss_after_response_retains_receipt_but_not_checkpoint(self):
        """执行中失权仍保留已付费响应；拒绝覆盖当前任务状态。"""
        tid,actor=self.create(); original=self.provider.send
        def lose(*args,**kwargs):
            """模拟请求完成时租约已被其他执行者接管。"""
            raw=original(*args,**kwargs);self.db.update_task(tid,lease_token='new-owner');return raw
        with patch.object(self.provider,'send',side_effect=lose), self.assertRaises(PermissionError):
            self.node.run(tid,self.item,**actor)
        self.assertEqual(self.db.get_model_call(tid,1)['status'],'succeeded');self.assertIsNone(self.db.get_checkpoint(tid))

    def test_provider_error_stays_explicit_and_blocks_replay(self):
        """HTTP拒绝保留原错误与未知用量，恢复不变成新请求。"""
        tid,actor=self.create()
        with patch.object(self.provider,'send',side_effect=ProviderResponseError(400,'synthetic rejected','r-reject')):
            with self.assertRaises(ProviderResponseError): self.node.run(tid,self.item,**actor)
        with self.assertRaises(RuntimeError): self.node.run(tid,self.item,**actor)
        self.assertEqual(self.db.get_model_call(tid,1)['request_id'],'r-reject')

    def test_truncated_output_preserved_as_partial(self):
        """截断是一次结果，正文保留且不自动续写。"""
        tid,actor=self.create()
        raw={'id':'truncated','choices':[{'message':{'content':'尚未完成的观察'},'finish_reason':'length'}],'usage':{'total_tokens':100}}
        with patch.object(self.provider,'send',return_value=raw) as send:
            result=self.node.run(tid,self.item,**actor)
        self.assertEqual(send.call_count,1);self.assertEqual(result['status'],'partial');self.assertEqual(result['answer'],'尚未完成的观察')

    def test_concurrent_same_lease_sends_once(self):
        """同租约误并发重入仍由发送窗口CAS保证只发送一次。"""
        tid,actor=self.create(); entered=threading.Event(); release=threading.Event(); original=self.provider.send
        def send(*args,**kwargs):
            """把第一次请求停在网络窗口，第二次不能重发。"""
            entered.set();release.wait(10);return original(*args,**kwargs)
        with patch.object(self.provider,'send',side_effect=send), ThreadPoolExecutor(2) as pool:
            first=pool.submit(self.node.run,tid,self.item,**actor);self.assertTrue(entered.wait(10))
            try:
                with self.assertRaises(RuntimeError): self.node.run(tid,self.item,**actor)
            finally: release.set()
            self.assertEqual(first.result()['status'],'complete')
        self.assertEqual(len(self.calls),1)

    def test_existing_budget_rejections_leave_no_orphan_tokens(self):
        """只在隔离夹具耗尽旧额度；组件不能修改额度或留下未发送Token占用。"""
        for kind in ('requests','tokens'):
            tid,actor=self.create()
            if kind=='requests':
                self.db.execute('UPDATE task_budgets SET model_used=model_limit WHERE root_task_id=?',(tid,))
            else:
                self.db.execute('UPDATE task_token_budgets SET used_tokens=token_limit WHERE root_task_id=?',(tid,))
            with self.assertRaises(ValueError): self.node.run(tid,self.item,**actor)
            self.assertEqual(TokenBudgetStore(self.db).summary(tid)['reserved_tokens'],0)
        self.assertEqual(self.calls,[])


if __name__=='__main__': unittest.main()
