"""精读原生收据、供应商适配器和实验传输的本地串联合同。"""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from harness_ab_budget import Ledger
from harness_ab_transport import (BudgetTransport, bind_deep_reading_client,
    deep_reading_request_contract, deep_reading_request_scope, visual_request_contract)
from research_loop.agents.deep_reading import _ReceiptClient, _RunInterrupted
from research_loop.config import Settings
from research_loop.providers.deepseek import DeepSeekProvider, RequestNotSent
from research_loop.providers.paper_summary import generate_deep_summary
from research_loop.storage.mysql import Database
from research_loop.storage.token_budget import TokenBudgetStore


USAGE = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
    'prompt_cache_hit_tokens': 40, 'prompt_cache_miss_tokens': 60}
INITIAL = '\n'.join(f'## {i}. 合成要点\n只用于本地接线。' for i in range(1, 5))
CONTINUATION = '\n'.join(f'## {i}. 合成要点\n不代表真实论文质量。' for i in range(5, 9)) + '\n\n（完）'


class DeepReadContractTests(unittest.TestCase):
    """每例独立SQLite和内存HTTP；任何意外套接字连接直接使测试失败。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='dpr-contract-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.counter = 0
        self.sent = []
        self.socket_guards = []
        for name in ('connect', 'connect_ex'):
            guard = patch.object(socket.socket, name,
                side_effect=AssertionError('real socket forbidden'))
            self.socket_guards.append(guard.start())
            self.addCleanup(guard.stop)

    def tearDown(self):
        for guard in self.socket_guards:
            self.assertEqual(guard.call_count, 0, 'test attempted a real socket connection')

    @staticmethod
    def payload(maximum=4096):
        return {'model': 'deepseek-flash', 'temperature': .3, 'max_tokens': maximum,
            'thinking': {'type': 'disabled'},
            'messages': [{'role': 'system', 'content': '合成精读系统消息'},
                         {'role': 'user', 'content': '合成授权源文'}]}

    def chain(self, *, enabled=True, limits=None, usage=USAGE, handler=None):
        """创建真实原生任务收据层，HTTP替身只替换最底层响应。"""
        self.counter += 1
        directory = self.root / str(self.counter)
        directory.mkdir()
        db = Database('sqlite:///' + str(directory / 'task.sqlite3'))
        db.ensure_schema()
        settings = Settings(project_root=directory, data_dir=directory/'data',
            output_dir=directory/'output', database_url=db.url,
            deepseek_model='deepseek-flash', execution_budget_mode='observe',
            model_max_output_tokens=65536, context_output_reserve=8192)
        tid = 'synthetic-dpr-' + str(self.counter)
        db.create_task(tid, None, 'synthetic', 'synthetic deep reading', 'deny', [],
            token_settings=settings)
        ledger = Ledger(directory / 'budget.sqlite3', allowed_cells=[('DPR', 'A')],
            **(limits or {}))
        ledger.begin_root(tid, 'DPR', 'A')
        calls = []

        def respond(request):
            body = json.loads(request.content)
            calls.append(body)
            self.sent.append(body)
            if handler:
                return handler(request)
            raw = {'id': 'synthetic-' + str(len(calls)), 'model': 'deepseek-flash',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': '合成总结（完）'}}]}
            if usage is not None:
                raw['usage'] = copy.deepcopy(usage)
            return httpx.Response(200, json=raw)

        transport = BudgetTransport(ledger, tid, directory/'requests',
            inner=httpx.MockTransport(respond), visual_contract=visual_request_contract(),
            deep_reading_contract=deep_reading_request_contract() if enabled else None)
        http = httpx.Client(transport=transport, trust_env=False)
        self.addCleanup(http.close)
        provider = DeepSeekProvider('synthetic-key-not-a-secret', model='deepseek-flash', client=http)
        state = {'task_id': tid, 'root_task_id': tid, 'max_model_requests': 48,
                 'execution_policy': db.execution_policy(tid)}
        return {'db': db, 'tid': tid, 'ledger': ledger, 'http': http,
            'provider': provider, 'state': state, 'calls': calls, 'transport': transport}

    @staticmethod
    def native(chain, maximum=4096):
        state = chain['state']
        client = _ReceiptClient(state, {'base_sequence': 0}, chain['provider'], chain['db'],
            lambda update: state.update(update))
        client.kwargs.update(max_tokens=maximum, temperature=.3)
        return client

    def assert_not_sent(self, chain):
        snapshot = chain['ledger'].snapshot()
        self.assertEqual(chain['calls'], [])
        self.assertEqual(snapshot['physical_sent_requests'], 0)
        self.assertEqual(snapshot['reservations'], [])
        self.assertTrue(snapshot['halt_reason'])

    def test_contract_freezes_both_native_phases_and_rejects_drift(self):
        expected = {'version': 1, 'model': 'deepseek-flash', 'temperature': .3,
            'thinking': {'type': 'disabled'}, 'stream': False,
            'max_tokens_by_phase': {'initial': 4096, 'continuation': 2048}}
        self.assertEqual(deep_reading_request_contract(), expected)
        for mutation in ({'temperature': .1}, {'model': 'other'},
                {'max_tokens_by_phase': {'initial': 4096, 'continuation': 4096}}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                BudgetTransport(None, 'root', self.root/'invalid',
                    inner=httpx.MockTransport(lambda _: None),
                    deep_reading_contract={**expected, **mutation})

    def test_actual_summary_initial_continuation_and_all_receipts_replay_locally(self):
        replies = iter(enumerate((INITIAL, CONTINUATION), 1))

        def respond(_request):
            number, content = next(replies)
            return httpx.Response(200, json={'id': f'synthetic-dpr-{number}', 'model': 'deepseek-flash',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': content}}], 'usage': USAGE})

        chain = self.chain(handler=respond)
        md, text = self.root/'paper.md', self.root/'paper.txt'
        md.write_text('# Synthetic metadata')
        text.write_text('Synthetic source only. No real scientific assertion.')
        with bind_deep_reading_client(_ReceiptClient):
            first = generate_deep_summary(str(md), str(text), client=self.native(chain))
        # 回放不需要实验scope，也不得再次进入HTTP；请求身份仍由原生收据校验。
        before = chain['db'].list_model_calls(chain['tid'])
        second = generate_deep_summary(str(md), str(text), client=self.native(chain))
        self.assertEqual(first, INITIAL + '\n\n' + CONTINUATION)
        self.assertEqual(second, first)
        self.assertEqual(before, chain['db'].list_model_calls(chain['tid']))
        self.assertEqual([b['max_tokens'] for b in chain['calls']], [4096, 2048])
        for body in chain['calls']:
            self.assertEqual(set(body), {'model', 'messages', 'temperature', 'max_tokens', 'thinking'})
            self.assertEqual(body['temperature'], .3)
            self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertIn('Synthetic source only.', chain['calls'][0]['messages'][1]['content'])
        self.assertIn(INITIAL, chain['calls'][1]['messages'][-1]['content'])
        snapshot = chain['ledger'].snapshot()
        self.assertEqual(snapshot['physical_sent_requests'], 2)
        self.assertEqual(snapshot['actual_known_tokens'], 240)
        self.assertEqual([r['status'] for r in snapshot['reservations']], ['settled', 'settled'])
        metadata = [json.loads(r['request_json']) for r in snapshot['reservations']]
        self.assertEqual([m['request_kind'] for m in metadata], ['text', 'text'])
        self.assertEqual([m['component'] for m in metadata], ['dpr_deep_reading'] * 2)
        self.assertEqual([m['deep_reading_phase'] for m in metadata], ['initial', 'continuation'])
        self.assertEqual(TokenBudgetStore(chain['db']).summary(chain['tid'])['used_tokens'], 240)
        for file in chain['transport'].directory.iterdir():
            self.assertNotIn(b'synthetic-key-not-a-secret', file.read_bytes())

    def test_native_three_round_limit_uses_both_phases_and_replays_without_send(self):
        """三轮各一次续写；未出现结束标记时保留原生最终回退语义。"""
        for completed in (True, False):
            replies=[f'合成第{n}段' for n in range(1,7)]
            if completed:
                replies[-1]+='（完）'
            numbered=iter(enumerate(replies,1))

            def respond(_request):
                number,content=next(numbered)
                return httpx.Response(200,json={'id':f'synthetic-round-{number}',
                    'model':'deepseek-flash','choices':[{'index':0,'finish_reason':'stop',
                    'message':{'role':'assistant','content':content}}],'usage':USAGE})

            chain=self.chain(handler=respond)
            md,text=self.root/'retry.md',self.root/'retry.txt'
            md.write_text('# Synthetic metadata');text.write_text('Synthetic source only.')
            with self.subTest(completed=completed), bind_deep_reading_client(_ReceiptClient):
                first=generate_deep_summary(str(md),str(text),client=self.native(chain))
            expected=replies[4]+'\n\n'+replies[5] if completed else replies[4]
            self.assertEqual(first,expected)
            before=chain['db'].list_model_calls(chain['tid'])
            self.assertEqual(generate_deep_summary(str(md),str(text),client=self.native(chain)),expected)
            self.assertEqual(before,chain['db'].list_model_calls(chain['tid']))
            self.assertEqual([b['max_tokens'] for b in chain['calls']],[4096,2048]*3)
            self.assertEqual(chain['ledger'].snapshot()['physical_sent_requests'],6)
            self.assertEqual(chain['ledger'].snapshot()['actual_known_tokens'],720)

    def test_missing_explicit_contract_preserves_native_unknown_but_never_sends(self):
        for maximum in (4096, 2048):
            chain = self.chain(enabled=False)
            with self.subTest(maximum=maximum), bind_deep_reading_client(_ReceiptClient):
                with self.assertRaises(_RunInterrupted):
                    self.native(chain, maximum).chat(self.payload(maximum)['messages'])
            self.assert_not_sent(chain)
            receipt = chain['db'].get_model_call(chain['tid'], 1)
            self.assertEqual((receipt['status'], receipt['sent_state']), ('unknown', 'sent'))

    def test_dpr_shaped_native_calls_outside_scope_are_not_component_identity(self):
        for maximum in (4096, 2048):
            chain = self.chain()
            with self.subTest(maximum=maximum), self.assertRaises(_RunInterrupted):
                self.native(chain, maximum).chat(self.payload(maximum)['messages'])
            self.assert_not_sent(chain)

    def test_scope_is_not_inherited_by_worker_thread(self):
        chain = self.chain()
        with deep_reading_request_scope(4096), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(chain['http'].post, 'https://api.deepseek.com/chat/completions',
                json=self.payload())
            with self.assertRaises(RequestNotSent):
                future.result()
        self.assert_not_sent(chain)

    def test_scope_restores_outer_phase_and_cleans_up_after_exception(self):
        chain = self.chain()
        with deep_reading_request_scope(4096):
            chain['http'].post('https://api.deepseek.com/chat/completions', json=self.payload())
            with self.assertRaisesRegex(RuntimeError, 'synthetic scope exit'):
                with deep_reading_request_scope(2048):
                    chain['http'].post('https://api.deepseek.com/chat/completions', json=self.payload(2048))
                    raise RuntimeError('synthetic scope exit')
            chain['http'].post('https://api.deepseek.com/chat/completions', json=self.payload())
        with self.assertRaises(RequestNotSent):
            chain['http'].post('https://api.deepseek.com/chat/completions', json=self.payload())
        self.assertEqual([b['max_tokens'] for b in chain['calls']], [4096, 2048, 4096])
        self.assertEqual(chain['ledger'].snapshot()['physical_sent_requests'], 3)

    def test_scope_maximum_and_exact_parameters_tools_stream_and_images_are_enforced(self):
        mutations = [lambda p: p.update(max_tokens=2048), lambda p: p.update(max_tokens=True),
            lambda p: p.update(max_tokens=4097), lambda p: p.update(temperature=.1),
            lambda p: p.update(model='another-model'), lambda p: p.pop('thinking'),
            lambda p: p.update(thinking={'type': 'enabled'}), lambda p: p.update(stream=True),
            lambda p: p.update(tools=[]), lambda p: p.update(tool_choice='auto'),
            lambda p: p['messages'][1].update(content=[{'type': 'text', 'text': 'synthetic'}]),
            lambda p: p['messages'][1].update(content=[{'type': 'image_url',
                'image_url': {'detail': 'original', 'url': 'data:image/png;base64,c3ludGhldGlj'}}])]
        for mutate in mutations:
            chain = self.chain()
            payload = self.payload(); mutate(payload)
            with self.subTest(mutation=mutate), deep_reading_request_scope(4096):
                with self.assertRaises(RequestNotSent):
                    chain['http'].post('https://api.deepseek.com/chat/completions', json=payload)
            self.assert_not_sent(chain)
        for maximum in (None, True, 2049):
            chain = self.chain()
            with self.subTest(scope_maximum=maximum), deep_reading_request_scope(maximum):
                with self.assertRaises(RequestNotSent):
                    chain['http'].post('https://api.deepseek.com/chat/completions', json=self.payload())
            self.assert_not_sent(chain)

    def test_shared_budget_refusal_reaches_no_http_and_keeps_old_native_classification(self):
        chain = self.chain(limits={'batch_tokens': 100})
        with bind_deep_reading_client(_ReceiptClient), self.assertRaises(_RunInterrupted):
            self.native(chain).chat(self.payload()['messages'])
        self.assert_not_sent(chain)
        self.assertEqual(chain['db'].get_model_call(chain['tid'], 1)['status'], 'unknown')

    def test_sent_missing_usage_stays_reserved_and_native_reentry_never_resends(self):
        chain = self.chain(usage=None)
        with bind_deep_reading_client(_ReceiptClient), self.assertRaises(_RunInterrupted):
            self.native(chain).chat(self.payload()['messages'])
        before = chain['db'].get_model_call(chain['tid'], 1)
        snapshot = chain['ledger'].snapshot()
        self.assertEqual((before['status'], before['sent_state']), ('unknown', 'sent'))
        self.assertEqual(snapshot['physical_sent_requests'], 1)
        self.assertEqual(snapshot['reservations'][0]['status'], 'unknown')
        self.assertEqual(snapshot['charged_or_reserved']['tokens'], 1_048_576 + 4096)
        self.assertEqual(snapshot['actual_known_tokens'], 0)
        self.assertTrue(snapshot['halt_reason'])
        with bind_deep_reading_client(_ReceiptClient), self.assertRaises(_RunInterrupted):
            self.native(chain).chat(self.payload()['messages'])
        self.assertEqual(len(chain['calls']), 1)
        self.assertEqual(before, chain['db'].get_model_call(chain['tid'], 1))


if __name__ == '__main__':
    unittest.main()
