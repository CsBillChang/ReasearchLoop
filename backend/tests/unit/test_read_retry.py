"""论文限流退避合同：用HTTP替身固定状态和时钟，不访问来源。"""
from datetime import datetime, timezone
import unittest
from unittest.mock import patch
import httpx
from research_loop.providers.read_retry import read_with_backoff, retry_after
from research_loop.providers.discovery import DiscoveryProvider


class ReadRetryTests(unittest.TestCase):
    """验证退避、上限和永久错误不重发。"""

    def test_retry_after_and_recovered_empty_is_not_failure(self):
        """429后遵守服务端等待，成功空结果仍可识别。"""
        responses=iter([httpx.Response(429,headers={'Retry-After':'3'}),httpx.Response(200,json=[])])
        waits=[];log={}
        result=read_with_backoff(lambda:next(responses),log,sleep=waits.append)
        self.assertEqual((result.status_code,waits,len(log['attempts'])),(200,[3],2))
        self.assertEqual(log['attempts'][0]['http_status'],429)

    def test_after_too_long_stops_without_early_retry(self):
        """不能把服务端120秒等待裁成60秒后提前重试。"""
        log={};calls=[]
        result=read_with_backoff(lambda:(calls.append(1) or httpx.Response(503,headers={'Retry-After':'120'})),log,sleep=lambda _:self.fail('no sleep'))
        self.assertEqual(len(calls),1);self.assertEqual(result.status_code,503)
        self.assertEqual(log['retry_stopped_reason'],'retry_after_exceeds_wait_budget')

    def test_date_and_invalid_retry_after(self):
        """HTTP日期与非法头保持一致语义。"""
        now=datetime(2026,10,12,0,0,0,tzinfo=timezone.utc)
        self.assertEqual(retry_after('Mon, 12 Oct 2026 00:00:05 GMT',now),5)
        self.assertEqual(retry_after('Mon, 12 Oct 2026 00:00:00 GMT',now),0)
        self.assertIsNone(retry_after('bad',now));self.assertIsNone(retry_after('-3',now))

    def test_fixed_attempt_cap_and_exponential_wait(self):
        """连续503最多4次，等待随次数增长；上层单次调度不再叠加重试。"""
        log={};waits=[]
        read_with_backoff(lambda:httpx.Response(503),log,sleep=waits.append,jitter=lambda:1)
        self.assertEqual(waits,[1,2,4]);self.assertEqual(len(log['attempts']),4)
        log={};read_with_backoff(lambda:httpx.Response(429),log,attempts=1,sleep=lambda _:self.fail('no wait'))
        self.assertEqual(len(log['attempts']),1)

    def test_permanent_and_statement_errors_not_retried(self):
        """鉴权、参数和SQL超时不消耗多轮请求。"""
        for code,body in [(400,{}),(401,{}),(403,{}),(500,{'code':'57014'})]:
            log={};read_with_backoff(lambda:httpx.Response(code,json=body),log,sleep=lambda _:self.fail('no retry'))
            self.assertEqual(len(log['attempts']),1)

    def test_network_failure_diagnostic_does_not_reuse_old_status(self):
        """最后网络异常不能显示前次429为当前结果。"""
        events=iter([httpx.Response(429),httpx.ConnectError('offline')]);log={}
        def send():
            event=next(events)
            if isinstance(event,Exception):raise event
            return event
        with self.assertRaises(httpx.ConnectError):read_with_backoff(send,log,attempts=2,sleep=lambda _:None)
        self.assertNotIn('http_status',log)
        self.assertEqual(log['attempts'][-1]['error'],'ConnectError')

    def test_actual_source_adapter_wires_retry_without_credentials_in_log(self):
        """测试实际只读RPC接线，payload与认证头不进入退避诊断。"""
        calls=[];log={}
        def handler(req):
            calls.append(req)
            return httpx.Response(429 if len(calls)==1 else 200,json=[])
        with httpx.Client(transport=httpx.MockTransport(handler)) as client,patch('research_loop.providers.discovery.time.sleep'):
            provider=DiscoveryProvider(client=client)
            result=provider._request_with_retries('https://fixture.invalid/rpc',{'query_text':'q'},{'Authorization':'SECRET'},1,log)
        self.assertEqual(result.status_code,200);self.assertEqual(len(calls),2)
        self.assertNotIn('SECRET',str(log))
