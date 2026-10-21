"""精读发送边界的冻结策略验证：临时数据库与HTTP替身，不改DPR生成行为。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import httpx

from research_loop.agents.deep_reading import _ReceiptClient, _RunInterrupted
from research_loop.config import Settings
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.storage.mysql import Database
from research_loop.storage.token_budget import TokenBudgetStore


class ReadingPolicyFreezeTests(unittest.TestCase):
    """可显式复验隔离MySQL；所有供应商响应均为MockTransport。"""

    def setUp(self):
        """只创建临时文件和唯一测试任务，不加载生产环境文件。"""
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        url = os.getenv('READING_POLICY_TEST_DATABASE_URL') or 'sqlite:///' + str(root/'reading.db')
        if not url.startswith('sqlite:') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_'):
            raise ValueError('explicit dedicated reading test database required')
        self.db = Database(url); self.db.ensure_schema()
        self.settings = Settings(project_root=root, data_dir=root/'data', output_dir=root/'out', database_url=url)
        self.sent = []
        def handler(request):
            """只记录本机替身实际收到的模型参数。"""
            self.sent.append(json.loads(request.content))
            return httpx.Response(200, json={'id': 'synthetic-reading', 'choices': [
                {'message': {'role': 'assistant', 'content': '合成总结（完）'}, 'finish_reason': 'stop'}],
                'usage': {'total_tokens': 2}})
        self.http = httpx.Client(transport=httpx.MockTransport(handler)); self.addCleanup(self.http.close)
        self.messages = [{'role': 'user', 'content': 'synthetic text'}]

    def make_client(self, settings=None, model=None, legacy=False):
        """创建新根任务或无新策略旧任务，并只装配精读收据边界。"""
        settings = settings or self.settings
        tid = 'reading-policy-' + uuid4().hex
        self.db.create_task(tid, None, 'synthetic', 'synthetic reading', 'deny', [],
                            token_settings=None if legacy else settings)
        state = {'task_id': tid, 'root_task_id': tid, 'max_model_requests': 48}
        if policy := self.db.execution_policy(tid):
            state['execution_policy'] = policy
        provider = DeepSeekProvider('fixture', model=model or settings.deepseek_model, client=self.http)
        client = _ReceiptClient(state, {'base_sequence': 0}, provider, self.db, lambda update: state.update(update))
        client.kwargs.update(max_tokens=4096, temperature=.3)
        return tid, client

    def request(self, client):
        """返回精读生成函数实际冻结的完整参数形状。"""
        return {'stage': 'dpr_deep_reading', 'messages': self.messages, 'tools': [], 'force_text': True,
                'max_tokens': 4096, 'temperature': .3, 'thinking': client.kwargs['thinking']}

    def test_new_send_rejects_changed_model_before_any_reservation(self):
        """新精读不能在旧模型计量策略下发送另一个模型。"""
        tid, client = self.make_client(model='deepseek-flash')
        with self.assertRaisesRegex(_RunInterrupted, 'TOKEN_POLICY_MODEL_CHANGED'):
            client.chat(self.messages)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.db.list_model_calls(tid), [])
        self.assertEqual(TokenBudgetStore(self.db).summary(tid)['reserved_tokens'], 0)

    def test_unsent_receipt_rejects_changed_model_without_rewriting_request(self):
        """尚未发送的旧请求也校验模型，原请求JSON保持不变。"""
        tid, client = self.make_client(model='deepseek-flash')
        self.db.begin_model_call(tid, 1, self.request(client))
        before = self.db.get_model_call(tid, 1)
        with self.assertRaisesRegex(_RunInterrupted, 'TOKEN_POLICY_MODEL_CHANGED'):
            client.chat(self.messages)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.db.get_model_call(tid, 1), before)

    def test_success_receipt_replays_before_current_model_checks(self):
        """成功响应先回放；当前模型变化不引发重发或旧请求改写。"""
        tid, client = self.make_client(model='deepseek-flash')
        raw = {'choices': [{'message': {'role': 'assistant', 'content': '旧合成成功（完）'}, 'finish_reason': 'stop'}]}
        self.db.save_model_call(tid, 1, 'succeeded', self.request(client), response=raw,
                                usage={'total_tokens': 2}, sent_state='sent')
        before = self.db.get_model_call(tid, 1)
        self.assertEqual(client.chat(self.messages)['content'], '旧合成成功（完）')
        self.assertEqual(self.sent, [])
        self.assertEqual(self.db.get_model_call(tid, 1), before)

    def test_actual_input_output_and_safety_enforce_window_in_both_modes(self):
        """监测累计成本也必须遵守模型窗口，按实际4096输出而非较小默认预留计数。"""
        for mode in ['enforce', 'observe']:
            with self.subTest(mode=mode):
                settings = self.settings.model_copy(update={'execution_budget_mode': mode,
                    'context_window_tokens': 4096, 'context_output_reserve': 512, 'context_safety_tokens': 256})
                tid, client = self.make_client(settings)
                with self.assertRaisesRegex(_RunInterrupted, 'CONTEXT_CAPACITY'):
                    client.chat([{'role': 'user', 'content': 'x'*12000}])
                self.assertEqual(self.db.list_model_calls(tid), [])
                self.assertEqual(TokenBudgetStore(self.db).summary(tid)['reserved_tokens'], 0)
        self.assertEqual(self.sent, [])

    def test_legacy_policy_missing_keeps_original_send_contract(self):
        """没有新冻结策略的历史任务不补写策略或改变原输出参数。"""
        tid, client = self.make_client(legacy=True)
        self.assertIn('（完）', client.chat(self.messages)['content'])
        self.assertEqual(self.sent[0]['max_tokens'], 4096)
        self.assertIsNone(TokenBudgetStore(self.db).counting_policy(tid))
