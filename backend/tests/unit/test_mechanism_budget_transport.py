"""共享物理预算、取消与完整收据的合成资格；不构造真实连接池。"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from experiments.mechanism_pilot.budget import BudgetHalt, Ledger, pilot_ledger
from experiments.mechanism_pilot.transport import QualificationTransport
from research_loop.providers.deepseek import RequestNotSent


class MechanismBudgetTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root, self.now, self.calls = Path(self.temp.name), 100.0, []

    def ledger(self, **kwargs):
        value = Ledger(self.root/'ledger.sqlite3', allowed_cells=[('R01','current'),('M1','A')],
            clock=lambda:self.now, **kwargs)
        value.begin_root('family', 'R01', 'current')
        return value

    @staticmethod
    def usage():
        return {'prompt_tokens':30,'completion_tokens':2,'total_tokens':32,
                'prompt_cache_hit_tokens':10,'prompt_cache_miss_tokens':20}

    def handler(self, request):
        self.calls.append(json.loads(request.content))
        return httpx.Response(200,json={'id':'synthetic','usage':self.usage(),
            'choices':[{'message':{'role':'assistant','content':'synthetic'},'finish_reason':'stop'}]})

    def transport(self, ledger, handler=None, **kwargs):
        return QualificationTransport(lambda:httpx.MockTransport(handler or self.handler),
            ledger,'family',kwargs.pop('input_token_limit',100),kwargs.pop('max_output_tokens',10),**kwargs)

    @staticmethod
    def payload(stream=False):
        return {'model':'deepseek-flash','messages':[{'role':'user','content':'合成问题'}],
                'max_tokens':10,'stream':stream}

    def send(self, transport, *, stream=False, url='https://fixture.invalid/chat/completions'):
        request=httpx.Request('POST',url,json=self.payload(stream))
        response=transport.handle_request(request)
        response.read()
        return response

    def test_parent_child_share_one_physical_ledger_and_body_unchanged(self):
        ledger=self.ledger(batch_requests=2)
        parent=self.transport(ledger);child=self.transport(ledger)
        self.send(parent);self.send(child)
        with self.assertRaises(RequestNotSent):self.send(child)
        snapshot=ledger.snapshot()
        self.assertEqual(snapshot['physical_sent_requests'],2)
        self.assertEqual(snapshot['actual_known_tokens'],64)
        self.assertEqual(self.calls,[self.payload(),self.payload()])
        self.assertEqual(snapshot['charged_or_reserved']['requests'],2)

    def test_peak_pricing_phase_caps_and_cleanup_cutoff_are_frozen(self):
        cells=[(f'R{i:02}','current') for i in range(1,7)]+[(f'M{i}',v) for i in range(1,5) for v in ('A','B')]
        ledger=pilot_ledger(self.root/'pilot.sqlite3',allowed_cells=cells,clock=lambda:self.now)
        ledger.begin_root('memory','M1','A')
        request=ledger.reserve('memory',input_upper_bound=1_048_576,max_output_tokens=4096)
        ledger.mark_sent(request['reservation_id']);ledger.settle(request['reservation_id'],self.usage())
        snapshot=ledger.snapshot();self.assertEqual(snapshot['actual_known_estimated_nano_yuan'],20*2400+10*48+2*9600)
        self.assertEqual(snapshot['limits']['batch_tokens'],3_000_000)
        self.assertEqual(snapshot['limits']['phase_limits']['memory']['tokens'],3_000_000)
        self.now+=2400
        with self.assertRaisesRegex(BudgetHalt,'cleanup'):ledger.reserve('memory',input_upper_bound=1,max_output_tokens=1)
        self.assertEqual(ledger.snapshot()['batch_absolute_deadline'],2800)
        with self.assertRaisesRegex(ValueError,'cannot change'):
            Ledger(ledger.path,allowed_cells=cells,clock=lambda:self.now)

    def test_original_one_million_memory_hard_share_cannot_admit_capacity_reservation(self):
        ledger=self.ledger(phase_limits={'r':{'cells':[['R01','current']],'tokens':2_000_000,'requests':48,'seconds':1500,'max_output_tokens':8192},
            'm':{'cells':[['M1','A']],'tokens':1_000_000,'requests':48,'seconds':900,'max_output_tokens':4096}})
        ledger.begin_root('memory','M1','A')
        with self.assertRaisesRegex(BudgetHalt,'phase token'):
            ledger.reserve('memory',input_upper_bound=1_048_576,max_output_tokens=4096)
        self.assertEqual(ledger.snapshot()['physical_sent_requests'],0)

    def test_phase_requests_output_and_time_have_atomic_pre_send_gates(self):
        for dimension in ('requests','output','seconds'):
            with self.subTest(dimension=dimension), tempfile.TemporaryDirectory() as folder:
                limits={'cells':[['R01','current']],'tokens':1000,'requests':1,'seconds':3,'max_output_tokens':10}
                other={'cells':[['M1','A']],'tokens':1000,'requests':1,'seconds':3,'max_output_tokens':10}
                ledger=Ledger(Path(folder)/'budget.db',allowed_cells=[('R01','current'),('M1','A')],clock=lambda:self.now,phase_limits={'r':limits,'m':other})
                ledger.begin_root('family','R01','current')
                if dimension=='requests':
                    ledger.reserve('family',input_upper_bound=100,max_output_tokens=10)
                elif dimension=='seconds':self.now+=3
                with self.assertRaisesRegex(BudgetHalt,'phase'):
                    ledger.reserve('family',input_upper_bound=1,max_output_tokens=11 if dimension=='output' else 1)

    def test_cancellation_after_reservation_releases_only_confirmed_unsent(self):
        ledger=self.ledger();cancelled={'yes':False};original=ledger.reserve
        def reserve(*args,**kwargs):
            result=original(*args,**kwargs);cancelled['yes']=True;return result
        transport=self.transport(ledger,cancelled=lambda:cancelled['yes'])
        with patch.object(ledger,'reserve',reserve),self.assertRaises(RequestNotSent):self.send(transport)
        snapshot=ledger.snapshot();self.assertEqual(len(self.calls),0)
        self.assertEqual(snapshot['reservations'][0]['status'],'not_sent')
        self.assertEqual(snapshot['charged_or_reserved']['tokens'],0)

    def test_cancelled_late_complete_receipt_settles_and_blocks_next_send(self):
        ledger=self.ledger();holder={}
        def handler(request):
            holder['transport'].cancel('合成取消');return self.handler(request)
        transport=self.transport(ledger,handler);holder['transport']=transport
        self.send(transport)
        self.assertEqual(ledger.snapshot()['reservations'][0]['status'],'settled')
        self.assertEqual(ledger.snapshot()['actual_known_tokens'],32)
        with self.assertRaises(RequestNotSent):self.send(transport)
        self.assertEqual(len(self.calls),1)

    def test_invalid_usage_and_http_error_remain_unknown_without_retry(self):
        for usage in (None,{'prompt_tokens':None,'completion_tokens':None,'total_tokens':None},
                      {'prompt_tokens':30,'completion_tokens':2,'total_tokens':33},
                      {'prompt_tokens':30,'completion_tokens':2,'total_tokens':32,'prompt_cache_hit_tokens':31}):
            with self.subTest(usage=usage),tempfile.TemporaryDirectory() as folder:
                self.root=Path(folder);ledger=self.ledger()
                transport=self.transport(ledger,lambda r:httpx.Response(200,json={'usage':usage}))
                with self.assertRaises(BudgetHalt):self.send(transport)
                self.assertEqual(ledger.snapshot()['reservations'][0]['status'],'unknown')
                self.assertEqual(ledger.snapshot()['charged_or_reserved']['tokens'],110)
                with self.assertRaises(RequestNotSent):self.send(transport)
        with tempfile.TemporaryDirectory() as folder:
            self.root=Path(folder);ledger=self.ledger();transport=self.transport(ledger,lambda r:httpx.Response(400,json={'error':'synthetic'}))
            self.assertEqual(self.send(transport).status_code,400)
            self.assertEqual(ledger.snapshot()['reservations'][0]['status'],'unknown')

    def test_timeout_unknown_and_finalize_preserve_sent_and_settled_states(self):
        ledger=self.ledger();holder={}
        def timeout(request):raise httpx.ReadTimeout('synthetic unknown')
        transport=self.transport(ledger,timeout)
        with self.assertRaises(httpx.ReadTimeout):self.send(transport)
        self.assertEqual(ledger.snapshot()['reservations'][0]['status'],'unknown')
        with self.assertRaises(RequestNotSent):self.send(transport)
        with tempfile.TemporaryDirectory() as folder:
            self.root=Path(folder);ledger=self.ledger()
            ids=[ledger.reserve('family',input_upper_bound=100,max_output_tokens=10)['reservation_id'] for _ in range(3)]
            ledger.mark_sent(ids[0]);ledger.settle(ids[0],self.usage());ledger.mark_sent(ids[1])
            with self.assertRaises(ValueError):ledger.finalize_pending('done')
            result=ledger.finalize_pending('done',workers_stopped=True)
            by_id={r['reservation_id']:r for r in result['reservations']}
            self.assertEqual([by_id[x]['status'] for x in ids],['settled','unknown','not_sent'])
            self.assertEqual(result['charged_or_reserved']['tokens'],142)

    def test_json_and_complete_sse_usage_settle_once_and_compressed_truncation_unknown(self):
        for encoded in (False,True):
            with self.subTest(encoded=encoded),tempfile.TemporaryDirectory() as folder:
                self.root=Path(folder);ledger=self.ledger()
                body=b'data: '+json.dumps({'id':'sse','usage':self.usage()}).encode()+b'\n\ndata: [DONE]\n\n'
                raw=gzip.compress(body) if encoded else body
                def handler(request):return httpx.Response(200,headers={'content-type':'text/event-stream',**({'content-encoding':'gzip'} if encoded else {})},stream=httpx.ByteStream(raw))
                transport=self.transport(ledger,handler);self.send(transport,stream=True);transport.close()
                self.assertEqual(ledger.snapshot()['actual_known_tokens'],32)
                self.assertEqual(ledger.snapshot()['reservations'][0]['status'],'settled')
        with tempfile.TemporaryDirectory() as folder:
            self.root=Path(folder);ledger=self.ledger()
            raw=gzip.compress(body)[:-4]
            transport=self.transport(ledger,lambda r:httpx.Response(200,headers={'content-encoding':'gzip'},stream=httpx.ByteStream(raw)))
            with self.assertRaises(Exception):self.send(transport,stream=True)
            self.assertEqual(ledger.snapshot()['reservations'][0]['status'],'unknown')

    def test_stream_close_without_done_preserves_full_unknown_reservation(self):
        ledger=self.ledger();raw=b'data: '+json.dumps({'usage':self.usage()}).encode()+b'\n\n'
        transport=self.transport(ledger,lambda r:httpx.Response(200,stream=httpx.ByteStream(raw)))
        with self.assertRaisesRegex(RuntimeError,'without DONE'):self.send(transport,stream=True)
        self.assertEqual(ledger.snapshot()['charged_or_reserved']['tokens'],110)
        self.assertEqual(ledger.snapshot()['reservations'][0]['status'],'unknown')

    def test_real_mode_requires_authorizer_before_constructing_inner(self):
        ledger=self.ledger();created=[]
        for authorize in (None,lambda:{'allow_real_model':False},lambda:{'allow_real_model':True,'freeze_sha256':'wrong'}):
            with self.subTest(authorize=authorize),self.assertRaises(PermissionError):
                QualificationTransport(lambda:created.append(True),ledger,'family',1_048_576,8192,
                    mode='real',freeze_sha256='a'*64,authorize=authorize)
        self.assertEqual(created,[])
        grant={'allow_real_model':True,'freeze_sha256':'a'*64,'model':'deepseek-flash','base_url':'https://api.deepseek.com'}
        for key, wrong in (('input_token_limit', 100), ('max_output_tokens', 4096)):
            invalid = {**grant, key: wrong}
            with self.subTest(capacity=key), self.assertRaises(PermissionError):
                QualificationTransport(lambda:created.append(True),ledger,'family',1_048_576,8192,
                    mode='real',freeze_sha256='a'*64,authorize=lambda:deepcopy(invalid))
        self.assertEqual(created,[])
        transport=QualificationTransport(lambda:httpx.MockTransport(self.handler),ledger,'family',1_048_576,8192,
            mode='real',freeze_sha256='a'*64,authorize=lambda:deepcopy(grant))
        payload=self.payload();payload['max_tokens']=8192
        response=transport.handle_request(httpx.Request('POST','https://api.deepseek.com/chat/completions',json=payload));response.read()
        self.assertEqual(len(self.calls),1)
        grant['freeze_sha256']='b'*64
        with self.assertRaises(RequestNotSent):transport.handle_request(httpx.Request('POST','https://api.deepseek.com/chat/completions',json=payload))
        self.assertEqual(len(self.calls),1)

    def test_substitute_authorizer_drift_is_checked_before_physical_send(self):
        ledger=self.ledger();grant={'allow_real_model':False,'freeze_sha256':'a'*64,'model':'deepseek-flash','base_url':'https://fixture.invalid'}
        transport=self.transport(ledger,authorize=lambda:deepcopy(grant),freeze_sha256='a'*64)
        grant['freeze_sha256']='b'*64
        with self.assertRaises(RequestNotSent):self.send(transport)
        self.assertEqual(self.calls,[])
        self.assertEqual(ledger.snapshot()['physical_sent_requests'],0)


if __name__=='__main__':unittest.main()
