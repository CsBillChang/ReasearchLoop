"""发现诊断归属的确定性并发回归；只有合成HTTP，不访问网络或模型。"""
from concurrent.futures import ThreadPoolExecutor
import json
import socket
import threading
import unittest
from unittest.mock import patch

import httpx

from research_loop.providers.discovery import DiscoveryProvider


class DiscoveryRequestContextTests(unittest.TestCase):
    """从真实序列化请求、返回诊断和日期恢复观察隔离合同。"""

    def setUp(self):
        """阻断真实socket，所有并发由事件控制，不靠调度概率复现。"""
        for name, owner in [('connect', socket.socket), ('create_connection', socket)]:
            blocker = patch.object(owner, name, side_effect=AssertionError('network prohibited'))
            blocker.start()
            self.addCleanup(blocker.stop)

    def provider(self, handler, **kwargs):
        """配置两个合成来源；认证值不得出现在诊断中。"""
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        backends = {source: {'url': 'https://fixture.invalid', 'anon_key': 'private-test-key',
                             'bm25_rpc': 'match_' + source + '_papers_bm25'} for source in ('cvpr', 'arxiv')}
        return DiscoveryProvider(backends=backends, client=client, retry_wait_seconds=0, **kwargs)

    @staticmethod
    def source(result, name='cvpr'):
        """取确切来源，不把其他来源的未配置状态算作请求结果。"""
        return next(item for item in result.source_results if item.source == name)

    def parallel(self, kind='timeout', *, source_b='cvpr', max_attempts=1):
        """令A等B整个search结束后才返回失败或成功。"""
        entered, release = threading.Event(), threading.Event()
        wire, lock = [], threading.Lock()

        def handle(request):
            """记录wire body，保证断言不依赖同一个诊断字典。"""
            body = json.loads(request.content)
            with lock:
                wire.append({'body': body, 'path': request.url.path})
            query = body['query_text']
            if query == 'query-A':
                entered.set()
                if not release.wait(5):
                    raise AssertionError('controlled concurrency timed out')
                if kind == 'timeout':
                    raise httpx.ReadTimeout('synthetic timeout', request=request)
                if kind == 'http':
                    return httpx.Response(403, json={'message': 'synthetic denial'})
                if kind == 'rate':
                    return httpx.Response(429)
                if kind == 'split' and (body['filter_published_start'], body['filter_published_end']) == ('2026-01-01T00:00:00+00:00', '2026-01-05T00:00:00+00:00'):
                    return httpx.Response(500, json={'code': '57014'})
            lower = body.get('filter_published_start', '2026-01-02')[:10]
            return httpx.Response(200, json=[{'id': query + ':' + lower, 'title': query, 'published': lower}])

        provider = self.provider(handle, max_attempts=max_attempts)
        dates = ('2026-01-01', '2026-01-04') if kind == 'split' else (None, None)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(provider.search, 'query-A', *dates, ['cvpr'], 2)
            try:
                self.assertTrue(entered.wait(5))
                second = provider.search('query-B', sources=[source_b], limit=2)
            finally:
                release.set()
            first = future.result(timeout=5)
        return provider, first, second, wire

    def test_concurrent_timeout_keeps_request_and_attempt_owner(self):
        """A/B实际搜索词正确，A失败不得绑定B的成功诊断。"""
        provider, first, second, wire = self.parallel()
        a, b = self.source(first), self.source(second)
        self.assertEqual([item['body']['query_text'] for item in wire], ['query-A', 'query-B'])
        self.assertEqual(a.status, 'failed')
        self.assertEqual(a.request['payload']['query_text'], 'query-A')
        self.assertEqual(a.request['attempts'], [{'attempt': 1, 'error': 'ReadTimeout'}])
        self.assertNotIn('http_status', a.request)
        self.assertEqual(b.status, 'queried_success')
        self.assertEqual(b.request['payload']['query_text'], 'query-B')
        self.assertEqual(second.candidates[0].title, 'query-B')
        self.assertNotIn('private-test-key', str(first.model_dump()))
        self.assertEqual(provider.last_requests[0]['payload']['query_text'], 'query-B')

    def test_concurrent_http_failure_keeps_its_status(self):
        """非重试403的诊断不被另一查询200覆盖。"""
        _, first, second, wire = self.parallel('http')
        a = self.source(first)
        self.assertEqual(a.request['http_status'], 403)
        self.assertEqual(a.request['retry_stopped_reason'], 'non_retryable_http')
        self.assertEqual(a.request['payload']['query_text'], 'query-A')
        self.assertEqual(self.source(second).request['http_status'], 200)
        self.assertEqual(len(wire), 2)

    def test_different_source_does_not_erase_failure_request(self):
        """另一线程换来源，也不能使当前来源失败诊断丢失。"""
        _, first, _, _ = self.parallel(source_b='arxiv')
        a = self.source(first)
        self.assertEqual(a.request['source'], 'cvpr')
        self.assertEqual(a.request['payload']['query_text'], 'query-A')

    def test_concurrent_statement_timeout_still_recovers_date_windows(self):
        """57014归属正确后按原预算分片；关键词和闭区间参数不变。"""
        _, first, second, wire = self.parallel('split')
        a = self.source(first)
        self.assertEqual(a.status, 'queried_success')
        self.assertEqual(len(first.candidates), 2)
        self.assertEqual(a.request['unresolved_windows'], [])
        self.assertEqual(len(a.request['windows']), 3)
        self.assertTrue(all(window['payload']['query_text'] == 'query-A' for window in a.request['windows']))
        self.assertEqual(a.request['windows'][0]['retry_stopped_reason'], 'database_statement_timeout')
        bodies = [item['body'] for item in wire if item['body']['query_text'] == 'query-A']
        self.assertEqual([(body['filter_published_start'], body['filter_published_end']) for body in bodies], [
            ('2026-01-01T00:00:00+00:00', '2026-01-05T00:00:00+00:00'),
            ('2026-01-01T00:00:00+00:00', '2026-01-03T00:00:00+00:00'),
            ('2026-01-03T00:00:00+00:00', '2026-01-05T00:00:00+00:00')])
        self.assertEqual(len(wire), 4)
        self.assertEqual(self.source(second).request['payload']['query_text'], 'query-B')

    def test_parallel_success_preserves_candidates_without_serializing(self):
        """两条成功结果仍对应自己的wire，不能用全局锁回避并发。"""
        _, first, second, wire = self.parallel('success')
        self.assertEqual([item.title for item in first.candidates], ['query-A'])
        self.assertEqual([item.title for item in second.candidates], ['query-B'])
        self.assertEqual(self.source(first).request['payload']['query_text'], 'query-A')
        self.assertEqual(len(wire), 2)

    def test_retry_budget_stays_per_request_under_concurrency(self):
        """429仍只执行原max_attempts；不因诊断隔离追加物理请求。"""
        _, first, _, wire = self.parallel('rate', max_attempts=2)
        a = self.source(first)
        self.assertEqual(a.request['payload']['query_text'], 'query-A')
        self.assertEqual(a.request['http_status'], 429)
        self.assertEqual(len(a.request['attempts']), 2)
        self.assertEqual(sum(item['body']['query_text'] == 'query-A' for item in wire), 2)

    def test_same_thread_nested_search_restores_outer_failure(self):
        """HTTP hook重入search后恢复外层诊断，内层结果仍可独立使用。"""
        inner = []

        def handle(request):
            """外层发送过程中执行内层，再让外层连接失败。"""
            query = json.loads(request.content)['query_text']
            if query == 'outer':
                inner.append(provider.search('inner', sources=['cvpr']))
                raise httpx.ReadTimeout('outer timeout', request=request)
            return httpx.Response(200, json=[{'id': query, 'title': query}])

        provider = self.provider(handle, max_attempts=1)
        outer = provider.search('outer', sources=['cvpr'])
        self.assertEqual(self.source(outer).request['payload']['query_text'], 'outer')
        self.assertEqual(self.source(outer).status, 'failed')
        self.assertEqual(self.source(inner[0]).request['payload']['query_text'], 'inner')
        self.assertEqual(provider.last_requests[0]['payload']['query_text'], 'outer')
        provider.search('next', sources=['cvpr'])
        self.assertEqual(len(provider.last_requests), 1)
        self.assertEqual(provider.last_requests[0]['payload']['query_text'], 'next')

    def test_thread_pool_reuse_close_and_fresh_thread_do_not_leak_records(self):
        """复用线程只保留当前查询；新线程和主线程不会继承旧工作线程记录。"""
        provider = self.provider(lambda request: httpx.Response(200, json=[]), max_attempts=1)
        provider.search('main', sources=['cvpr'])

        def run():
            """在同一工作线程顺序查询，验证返回对象和诊断历史不混合。"""
            self.assertEqual(provider.last_requests, [])
            first = provider.search('first', sources=['cvpr'])
            second = provider.search('second', sources=['cvpr'])
            self.assertEqual(self.source(first).request['payload']['query_text'], 'first')
            self.assertEqual(self.source(second).request['payload']['query_text'], 'second')
            self.assertEqual(len(provider.last_requests), 1)
            self.assertEqual(provider.last_requests[0]['payload']['query_text'], 'second')
            provider.search('unconfigured', sources=['iclr'])
            self.assertEqual(provider.last_requests, [])

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(run).result(timeout=5)
        self.assertEqual(provider.last_requests[0]['payload']['query_text'], 'main')
        with ThreadPoolExecutor(max_workers=1) as fresh:
            self.assertEqual(fresh.submit(lambda: provider.last_requests).result(timeout=5), [])


if __name__ == '__main__':
    unittest.main()
