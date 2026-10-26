"""持久试点预算的纯MockTransport测试；禁止任何真实socket连接。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
import json
from pathlib import Path
import socket
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

import httpx

from research_loop.providers.deepseek import DeepSeekProvider, ProviderResponseError, RequestNotSent
from scripts.reflection_pilot_budget import PilotBudgetLedger, PilotBudgetStopped, wrap_provider


LIMITS = {'max_requests':48,'max_tokens':2_000_000,'max_cost_cny':'16'}
RATES = {'input_per_million_cny':'2','output_per_million_cny':'8','cached_input_per_million_cny':'0.04'}
MESSAGES = [{'role':'user','content':'核验合成稿，禁止保存。'}]


def usage(prompt=100,completion=5):
    return {'prompt_tokens':prompt,'completion_tokens':completion,'total_tokens':prompt+completion,
            'prompt_cache_hit_tokens':20,'prompt_cache_miss_tokens':prompt-20,
            'completion_tokens_details':{'reasoning_tokens':2}}


def result(number='request-1',*,used=None):
    return {'id':number,'model':'deepseek-flash','choices':[{'index':0,'finish_reason':'stop',
        'message':{'role':'assistant','content':'合成结果。'}}], 'usage':usage() if used is None else used}


class ReflectionPilotBudgetTests(unittest.TestCase):
    """每例独立SQLite，实际业务适配器仅访问显式MockTransport。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name in ('connect','connect_ex'):
            guard = patch.object(socket.socket,name,side_effect=AssertionError('real socket forbidden'))
            guard.start()
            self.addCleanup(guard.stop)
        guard = patch('socket.create_connection',side_effect=AssertionError('real socket forbidden'))
        guard.start()
        self.addCleanup(guard.stop)
        self.seen = []

    def ledger(self,**kwargs):
        return PilotBudgetLedger(self.root/'budget.sqlite3',limits=kwargs.pop('limits',LIMITS),
                                 rates=kwargs.pop('rates',RATES),**kwargs)

    def provider(self,ledger,handler):
        def capture(request):
            self.seen.append(json.loads(request.content))
            return handler(request)
        original = DeepSeekProvider('SYNTHETIC_TEST_SECRET','deepseek-flash',
            client=httpx.Client(transport=httpx.MockTransport(capture),trust_env=False))
        provider = wrap_provider(original,ledger)
        self.addCleanup(provider.close)
        return provider

    def test_actual_complete_http_payload_contract_bound_and_persistent_settlement(self):
        ledger = self.ledger()
        provider = self.provider(ledger,lambda _:httpx.Response(200,json=result()))
        tools = [{'type':'function','function':{'name':'finish_review','parameters':{'type':'object'}}}]
        provider.send(MESSAGES,tools,max_tokens=65536,thinking={'type':'enabled'})
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot['physical_requests'],1)
        self.assertEqual(snapshot['known_total_tokens'],105)
        row = snapshot['calls'][0]
        self.assertEqual(row['input_reserved'],1_048_576)
        self.assertEqual(row['output_reserved'],65536)
        self.assertEqual(self.seen[0]['thinking'],{'type':'enabled'})
        self.assertEqual(self.seen[0]['tools'],tools)
        self.assertEqual(self.seen[0]['max_tokens'],65536)
        self.assertGreater(json.loads(row['details_json'])['format_reserve_tokens'],8192)
        self.assertNotIn('SYNTHETIC_TEST_SECRET',json.dumps(snapshot))
        again = self.ledger()
        self.assertEqual(again.snapshot()['known_total_tokens'],105)
        self.assertIsNone(again.global_stop)

    def test_global_ledger_across_root_and_reviewer_providers(self):
        ledger = self.ledger()
        first = self.provider(ledger,lambda _:httpx.Response(200,json=result('root')))
        second = self.provider(ledger,lambda _:httpx.Response(200,json=result('reviewer')))
        first.send(MESSAGES,max_tokens=65536)
        second.send(MESSAGES,max_tokens=65536)
        self.assertEqual(ledger.snapshot()['physical_requests'],2)
        self.assertEqual(ledger.snapshot()['known_total_tokens'],210)

    def test_context_contract_reservation_stops_before_token_limit_without_shortening_output(self):
        ledger = self.ledger()
        provider = self.provider(ledger,lambda _:httpx.Response(200,json=result(used=usage(900000,10))))
        provider.send(MESSAGES,max_tokens=65536)
        with self.assertRaisesRegex(PilotBudgetStopped,'token_reservation_limit'):
            provider.send(MESSAGES,max_tokens=65536)
        self.assertEqual(len(self.seen),1)
        self.assertEqual(ledger.snapshot()['known_total_tokens'],900010)
        self.assertEqual(self.seen[0]['max_tokens'],65536)
        self.assertEqual(self.ledger().global_stop,'token_reservation_limit')

    def test_request_and_money_limits_are_persistent_pre_send_stops(self):
        for kind,limits,expected in [('request',{**LIMITS,'max_requests':1},'request_limit'),
                                     ('money',{**LIMITS,'max_cost_cny':'1'},'money_reservation_limit')]:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                ledger = PilotBudgetLedger(Path(folder).resolve()/'budget.sqlite3',limits=limits,rates=RATES)
                provider = self.provider(ledger,lambda _:httpx.Response(200,json=result()))
                before = len(self.seen)
                if kind=='request':
                    provider.send(MESSAGES,max_tokens=65536)
                with self.assertRaisesRegex(PilotBudgetStopped,expected):
                    provider.send(MESSAGES,max_tokens=65536)
                self.assertEqual(len(self.seen)-before,1 if kind=='request' else 0)
                self.assertEqual(ledger.global_stop,expected)

    def test_unknown_keeps_reserved_usage_and_stops_reopen(self):
        ledger = self.ledger()
        def timeout(_):
            raise httpx.ReadTimeout('synthetic')
        provider = self.provider(ledger,timeout)
        with self.assertRaises(httpx.ReadTimeout):
            provider.send(MESSAGES,max_tokens=65536)
        row = ledger.snapshot()['calls'][0]
        self.assertEqual(row['state'],'unknown')
        self.assertIsNone(row['usage_json'])
        self.assertIsNone(row['actual_tokens'])
        self.assertEqual(row['input_reserved'],1_048_576)
        again = self.ledger()
        with self.assertRaises(PilotBudgetStopped):
            provider.send(MESSAGES,max_tokens=65536)
        self.assertEqual(len(self.seen),1)
        self.assertEqual(again.global_stop,'transport_unknown:ReadTimeout')

    def test_explicit_request_not_sent_releases_reservation_without_filling_usage(self):
        ledger = self.ledger()
        def local_refusal(_):
            raise RequestNotSent('synthetic deterministic refusal')
        refusing = self.provider(ledger,local_refusal)
        with self.assertRaises(RequestNotSent):
            refusing.send(MESSAGES,max_tokens=65536)
        row = ledger.snapshot()['calls'][0]
        self.assertEqual(row['state'],'not_sent')
        self.assertIsNone(row['usage_json'])
        self.assertEqual(ledger.snapshot()['physical_requests'],0)
        self.assertIsNone(ledger.global_stop)
        good = self.provider(ledger,lambda _:httpx.Response(200,json=result()))
        good.send(MESSAGES,max_tokens=65536)
        self.assertEqual(ledger.snapshot()['physical_requests'],1)

    def test_402_retains_first_rejection_and_does_not_retry(self):
        ledger = self.ledger()
        provider = self.provider(ledger,lambda _:httpx.Response(402,json={'error':'synthetic insufficient balance'},headers={'x-request-id':'reject-1'}))
        with self.assertRaises(ProviderResponseError):
            provider.send(MESSAGES,max_tokens=65536)
        ledger.halt('later_error_must_not_replace_first')
        self.assertEqual(ledger.global_stop,'provider_http_402')
        row = ledger.snapshot()['calls'][0]
        self.assertEqual(row['state'],'rejected')
        self.assertEqual(row['request_id'],'reject-1')
        self.assertIsNone(row['actual_tokens'])
        with self.assertRaises(PilotBudgetStopped):
            provider.send(MESSAGES,max_tokens=65536)
        self.assertEqual(len(self.seen),1)

    def test_missing_and_invalid_usage_do_not_release_budget(self):
        ledger = self.ledger()
        raw = result();raw.pop('usage')
        provider = self.provider(ledger,lambda _:httpx.Response(200,json=raw))
        provider.send(MESSAGES,max_tokens=65536)
        self.assertEqual(ledger.global_stop,'usage_missing')
        self.assertEqual(ledger.snapshot()['unsettled_or_usage_missing'],1)
        self.assertIsNone(ledger.snapshot()['calls'][0]['actual_tokens'])
        with tempfile.TemporaryDirectory() as folder:
            other = PilotBudgetLedger(Path(folder).resolve()/'budget.sqlite3',limits=LIMITS,rates=RATES)
            invalid = usage();invalid['total_tokens']+=1
            bad = self.provider(other,lambda _:httpx.Response(200,json=result(used=invalid)))
            with self.assertRaisesRegex(ValueError,'mismatch'):
                bad.send(MESSAGES,max_tokens=65536)
            row=other.snapshot()['calls'][0]
            self.assertEqual(other.global_stop,'usage_invalid')
            self.assertEqual(json.loads(row['usage_json']),invalid)
            self.assertIsNone(row['actual_tokens'])

    def test_crash_recovery_keeps_unsettled_reservation_and_refuses_paid_resume(self):
        ledger = self.ledger()
        ledger.reserve('synthetic',1_048_576,65536)
        again = self.ledger()
        self.assertEqual(again.global_stop,'recovered_unsettled_call')
        with self.assertRaises(PilotBudgetStopped):
            again.reserve('another',1_048_576,65536)
        self.assertEqual(again.snapshot()['unsettled_or_usage_missing'],1)

    def test_single_pending_reservation_is_atomic_across_threads(self):
        ledger = self.ledger()
        def reserve(_):
            try:
                return ledger.reserve('synthetic',1_048_576,65536)
            except PilotBudgetStopped:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            ids = list(pool.map(reserve,range(2)))
        self.assertEqual(sum(x is not None for x in ids),1)
        self.assertEqual(ledger.global_stop,'physical_request_in_flight')

    def test_idempotent_settlement_cache_details_and_change_rejection(self):
        ledger = self.ledger()
        identity = ledger.reserve('synthetic',1000,65536)
        ledger.dispatched(identity)
        ledger.settle(identity,usage(),request_id='one')
        ledger.settle(identity,usage(),request_id='one')
        row = ledger.snapshot()['calls'][0]
        self.assertEqual(row['actual_tokens'],105)
        self.assertEqual(json.loads(row['usage_json'])['prompt_cache_hit_tokens'],20)
        self.assertLess(float(row['cache_adjusted_cost_estimate']),float(row['actual_cost_upper']))
        with self.assertRaisesRegex(ValueError,'SETTLEMENT_CHANGED'):
            ledger.settle(identity,usage(101,5),request_id='one')

    def test_actual_overrun_is_recorded_and_stops_next_send(self):
        ledger = self.ledger()
        identity = ledger.reserve('synthetic',100,10)
        ledger.dispatched(identity)
        ledger.settle(identity,usage(101,11),request_id='overflow')
        self.assertEqual(ledger.snapshot()['known_total_tokens'],112)
        self.assertEqual(ledger.global_stop,'reservation_upper_bound_breached')

    def test_config_changes_and_mutation_cannot_raise_approved_budget(self):
        ledger = self.ledger()
        limits = ledger.limits;limits['max_requests']=100
        self.assertEqual(ledger.limits['max_requests'],48)
        with self.assertRaisesRegex(ValueError,'CONFIGURATION_IS_FROZEN'):
            ledger.limits = limits
        with self.assertRaisesRegex(ValueError,'FROZEN_CONFIGURATION_CHANGED'):
            self.ledger(limits={**LIMITS,'max_requests':47})
        with self.assertRaisesRegex(ValueError,'EXCEEDS_APPROVED'):
            PilotBudgetLedger(self.root/'too-much.sqlite3',limits={**LIMITS,'max_tokens':2_000_001},rates=RATES)

    def test_model_endpoint_and_max_output_must_match_frozen_contract(self):
        ledger = self.ledger()
        provider = self.provider(ledger,lambda _:httpx.Response(200,json=result()))
        for kwargs in [{},{'max_tokens':1}]:
            with self.subTest(kwargs=kwargs),self.assertRaisesRegex(RequestNotSent,'MAX_OUTPUT_REQUIRED'):
                provider.send(MESSAGES,**kwargs)
        self.assertEqual(self.seen,[])
        self.assertEqual(ledger.snapshot()['attempt_rows'],0)
        original = DeepSeekProvider('synthetic','deepseek-flash',base_url='https://fixture.invalid',
            client=httpx.Client(transport=httpx.MockTransport(lambda _:httpx.Response(200,json=result())),trust_env=False))
        self.addCleanup(original.client.close)
        with self.assertRaisesRegex(ValueError,'UNSAFE_ENDPOINT'):
            wrap_provider(original,ledger)

    def test_direct_client_calls_cannot_bypass_budget_wrapper(self):
        ledger = self.ledger()
        provider = self.provider(ledger,lambda _:httpx.Response(200,json=result()))
        with self.assertRaisesRegex(RequestNotSent,'OUTSIDE_BUDGETED_PROVIDER_CALL'):
            provider.client.post(provider.endpoint,json={'model':'deepseek-flash','messages':MESSAGES,'max_tokens':65536})
        self.assertEqual(self.seen,[])

    def test_plain_text_cannot_borrow_vision_output_cap_to_fit_budget(self):
        ledger = self.ledger()
        provider = self.provider(ledger,lambda _:httpx.Response(200,json=result()))
        with self.assertRaisesRegex(RequestNotSent,'ORDINARY_OUTPUT_CHANGED'):
            provider.send(MESSAGES,max_tokens=8192)
        self.assertEqual(self.seen,[])
        self.assertEqual(ledger.snapshot()['attempt_rows'],0)

    def test_native_vision_png_preserves_8192_disabled_under_same_context_bound(self):
        def chunk(kind,data):
            return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
        png=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,8,6,0,0,0))+
             chunk(b'IDAT',zlib.compress(b'\x00\x00\x00\x00\xff'))+chunk(b'IEND',b''))
        messages=[{'role':'user','content':[{'type':'text','text':'核验合成图片。'},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode(),'detail':'original'}}]}]
        ledger=self.ledger()
        provider=self.provider(ledger,lambda _:httpx.Response(200,json=result()))
        provider.send(messages,max_tokens=8192,thinking={'type':'disabled'})
        row=ledger.snapshot()['calls'][0]
        self.assertEqual(row['input_reserved'],1_048_576)
        self.assertEqual(row['output_reserved'],8192)
        self.assertEqual(json.loads(row['details_json'])['image_count'],1)
        self.assertEqual(self.seen[0]['thinking'],{'type':'disabled'})
        self.assertEqual(self.seen[0]['max_tokens'],8192)
        self.assertEqual(self.seen[0]['messages'],messages)

    def test_transport_internal_retries_are_rejected(self):
        ledger = self.ledger()
        client = httpx.Client(transport=httpx.HTTPTransport(retries=1),trust_env=False)
        self.addCleanup(client.close)
        original = DeepSeekProvider('synthetic','deepseek-flash',client=client)
        with self.assertRaisesRegex(ValueError,'RETRIES_OR_PROXY'):
            wrap_provider(original,ledger)

    def test_stream_uses_one_physical_reservation_and_preserves_usage(self):
        ledger = self.ledger()
        chunks = [{'id':'stream-1','model':'deepseek-flash','choices':[{'index':0,'delta':{'content':'合成'},'finish_reason':None}]},
                  {'id':'stream-1','choices':[{'index':0,'delta':{'content':'完成。'},'finish_reason':'stop'}]},
                  {'id':'stream-1','choices':[],'usage':usage()}]
        body = ''.join('data: '+json.dumps(c,ensure_ascii=False)+'\n\n' for c in chunks)+'data: [DONE]\n\n'
        provider = self.provider(ledger,lambda _:httpx.Response(200,text=body,headers={'content-type':'text/event-stream'}))
        public=[]
        raw=provider.send_stream(MESSAGES,max_tokens=65536,on_public=public.append)
        self.assertEqual(raw['choices'][0]['message']['content'],'合成完成。')
        self.assertEqual(ledger.snapshot()['physical_requests'],1)
        self.assertEqual(ledger.snapshot()['known_total_tokens'],105)
        self.assertTrue(self.seen[0]['stream'])
        self.assertEqual(self.seen[0]['stream_options'],{'include_usage':True})
        self.assertTrue(public)

    def test_stream_disconnect_keeps_unknown_and_never_resends(self):
        ledger = self.ledger()
        body='data: '+json.dumps({'id':'stream-x','choices':[{'index':0,'delta':{'content':'未完'},'finish_reason':None}]})+'\n\n'
        provider=self.provider(ledger,lambda _:httpx.Response(200,text=body))
        with self.assertRaisesRegex(RuntimeError,'disconnected'):
            provider.send_stream(MESSAGES,max_tokens=65536,on_public=lambda _:None)
        self.assertEqual(ledger.snapshot()['calls'][0]['state'],'unknown')
        with self.assertRaises(PilotBudgetStopped):
            provider.send_stream(MESSAGES,max_tokens=65536,on_public=lambda _:None)
        self.assertEqual(len(self.seen),1)


if __name__=='__main__':
    unittest.main()
