"""以原生供应商适配器与内存HTTP传输检查真实发送预算边界。"""
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

import httpx

from research_loop.providers.deepseek import DeepSeekProvider, ProviderResponseError, RequestNotSent


SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


def load_module(name):
    """从实验源码载入组件，避免修改日常业务导入路径。"""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUDGET = load_module('harness_ab_budget')
TRANSPORT = load_module('harness_ab_transport')
USAGE = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
    'prompt_cache_hit_tokens': 40, 'prompt_cache_miss_tokens': 60,
    'completion_tokens_details': {'reasoning_tokens': 12}}


class Chunks(httpx.SyncByteStream):
    """提供任意SSE分块及读/关闭故障，不发起套接字连接。"""

    def __init__(self, chunks, *, read_error=None, close_error=None):
        self.chunks, self.read_error, self.close_error = chunks, read_error, close_error

    def __iter__(self):
        yield from self.chunks
        if self.read_error:
            raise self.read_error

    def close(self):
        if self.close_error:
            raise self.close_error


class HarnessAbTransportTests(unittest.TestCase):
    """每例使用独立账本与合成响应，所有角色归入同一根任务身份。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='h1-transport-test-')
        self.root = Path(self.temp.name)
        self.now = 100.0
        self.ledger = BUDGET.Ledger(self.root / 'ledger.db', clock=lambda: self.now)
        self.ledger.begin_root('root', 'H1-Q1', 'A')
        self.sent = []
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def normal_response(usage=USAGE):
        result = {'id': 'synthetic-response', 'model': 'deepseek-flash',
            'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': '中文测试'}}]}
        if usage is not None:
            result['usage'] = usage
        return result

    @staticmethod
    def sse(*, usage=USAGE, done=True, duplicate_usage=False):
        records = [{'id': 'synthetic-stream', 'model': 'deepseek-flash',
            'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': '中文测试'}, 'finish_reason': 'stop'}]}]
        if usage is not None:
            record = {'id': 'synthetic-stream', 'model': 'deepseek-flash', 'choices': [], 'usage': usage}
            records.append(record)
            if duplicate_usage:
                records.append(record)
        data = b''.join(('data: ' + json.dumps(r, ensure_ascii=False) + '\r\n\r\n').encode() for r in records)
        return data + (b'data: [DONE]\r\n\r\n' if done else b'')

    def client(self, handler, **transport_options):
        self.counter += 1
        def tracked(request):
            self.sent.append(json.loads(request.content))
            return handler(request)
        transport = TRANSPORT.BudgetTransport(self.ledger, 'root', self.root / str(self.counter),
            inner=httpx.MockTransport(tracked), **transport_options)
        client = httpx.Client(transport=transport, trust_env=False)
        self.addCleanup(client.close)
        return client, transport

    @staticmethod
    def provider(client, **options):
        return DeepSeekProvider('synthetic-not-a-secret', model='deepseek-flash', client=client, **options)

    def assert_unknown(self, *, requests=1):
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot['physical_sent_requests'], requests)
        self.assertEqual(snapshot['reservations'][-1]['status'], 'unknown')
        self.assertTrue(snapshot['halt_reason'])
        self.assertGreater(snapshot['charged_or_reserved']['tokens'], 1_000_000)
        with self.assertRaises(BUDGET.BudgetHalt):
            self.ledger.reserve('root', max_output_tokens=64)

    def test_native_json_response_settles_usage_once_without_auth_headers_in_files(self):
        client, transport = self.client(lambda _: httpx.Response(200, json=self.normal_response()))
        raw = self.provider(client).send([{'role': 'user', 'content': '合成问题'}], max_tokens=64)
        self.assertEqual(raw['usage'], USAGE)
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot['actual_known_tokens'], 120)
        self.assertEqual(snapshot['physical_sent_requests'], 1)
        self.assertEqual(snapshot['charged_or_reserved']['tokens'], 120)
        self.assertEqual(snapshot['actual_known_estimated_nano_yuan'], 60 * 2000 + 40 * 40 + 20 * 8000)
        for file in transport.directory.iterdir():
            self.assertNotIn(b'synthetic-not-a-secret', file.read_bytes())

    def test_native_sse_one_byte_chunks_preserve_utf8_usage_and_done(self):
        wire = self.sse()
        client, _ = self.client(lambda _: httpx.Response(200,
            stream=Chunks([wire[i:i + 1] for i in range(len(wire))])))
        previews = []
        raw = self.provider(client).send_stream([{'role': 'user', 'content': '合成问题'}],
            on_public=previews.append, max_tokens=64)
        self.assertEqual(raw['choices'][0]['message']['content'], '中文测试')
        self.assertEqual(raw['usage'], USAGE)
        self.assertTrue(previews)
        self.assertEqual(self.ledger.snapshot()['actual_known_tokens'], 120)
        self.assertEqual(self.ledger.snapshot()['reservations'][0]['status'], 'settled')

    def test_identical_repeated_stream_usage_and_repeated_close_do_not_double_charge(self):
        client, transport = self.client(lambda _: httpx.Response(200,
            stream=Chunks([self.sse(duplicate_usage=True)])))
        self.provider(client).send_stream([{'role': 'user', 'content': 'x'}], on_public=lambda _: None, max_tokens=64)
        transport.close()
        transport.close()
        snapshot = self.ledger.snapshot()
        self.assertEqual(snapshot['actual_known_tokens'], 120)
        self.assertIsNone(snapshot['halt_reason'])

    def compression_ledger(self, label):
        """每个压缩边界使用独立批账本，不让前一个未知结果污染后续断言。"""
        self.ledger = BUDGET.Ledger(self.root / (label + '-ledger.db'), clock=lambda: self.now)
        self.ledger.begin_root('root', 'H1-Q1', 'A')
        self.sent = []

    @staticmethod
    def compressed(encoding, wire):
        """构造实际HTTP会返回的三种完整压缩字节流。"""
        if encoding == 'gzip':
            return gzip.compress(wire)
        if encoding == 'deflate':
            return zlib.compress(wire)
        import zstandard
        return zstandard.ZstdCompressor(write_checksum=True).compress(wire)

    def compressed_client(self, encoding, wire, *, truncated=False):
        raw = self.compressed(encoding, wire)
        if truncated:
            raw = raw[:-5]
        return self.client(lambda _: httpx.Response(200, headers={
            'Content-Encoding': encoding, 'Content-Length': str(len(raw)),
            'Content-Type': 'text/event-stream'},
            stream=Chunks([raw[i:i + 3] for i in range(0, len(raw), 3)])))

    def test_native_compressed_sse_is_decoded_once_and_exact_usage_is_settled(self):
        for encoding in ('gzip', 'deflate', 'zstd'):
            with self.subTest(encoding=encoding):
                self.compression_ledger('complete-' + encoding)
                client, transport = self.compressed_client(encoding, self.sse())
                raw = self.provider(client).send_stream([{'role': 'user', 'content': 'x'}],
                    on_public=lambda _: None, max_tokens=64)
                self.assertEqual(raw['choices'][0]['message']['content'], '中文测试')
                self.assertEqual(raw['usage'], USAGE)
                snapshot = self.ledger.snapshot()
                self.assertEqual(snapshot['actual_known_tokens'], 120)
                self.assertEqual(snapshot['reservations'][0]['status'], 'settled')
                self.assertIsNone(snapshot['halt_reason'])
                archive = next(transport.directory.glob('*.response.sse')).read_bytes()
                self.assertEqual(archive, self.sse())
                metadata = json.loads(next(transport.directory.glob('*.response-meta.json')).read_text())
                self.assertEqual(metadata['archive_body_encoding'], 'httpx_decoded_bytes')
                self.assertEqual(metadata['source_content_encoding'], encoding)

    def test_decoded_outer_response_removes_compression_and_compressed_length_headers(self):
        client, _ = self.compressed_client('gzip', self.sse())
        request = client.build_request('POST', 'https://api.deepseek.com/chat/completions',
            json={'model': 'deepseek-flash', 'temperature': 0.1, 'stream': True, 'max_tokens': 64})
        response = client.send(request, stream=True)
        self.assertNotIn('content-encoding', response.headers)
        self.assertNotIn('content-length', response.headers)
        self.assertEqual(response.read(), self.sse())
        response.close()
        self.assertEqual(self.ledger.snapshot()['actual_known_tokens'], 120)

    def test_compressed_stream_missing_usage_is_unknown(self):
        for encoding in ('gzip', 'deflate', 'zstd'):
            with self.subTest(encoding=encoding):
                self.compression_ledger('missing-usage-' + encoding)
                client, _ = self.compressed_client(encoding, self.sse(usage=None))
                with self.assertRaises(BUDGET.BudgetHalt):
                    self.provider(client).send_stream([{'role': 'user', 'content': 'x'}],
                        on_public=lambda _: None, max_tokens=64)
                self.assert_unknown()

    def test_compressed_stream_truncation_never_settles_even_if_done_was_decoded(self):
        for encoding in ('gzip', 'deflate', 'zstd'):
            with self.subTest(encoding=encoding):
                self.compression_ledger('truncated-' + encoding)
                client, _ = self.compressed_client(encoding, self.sse(), truncated=True)
                with self.assertRaises((httpx.DecodingError, RuntimeError)):
                    self.provider(client).send_stream([{'role': 'user', 'content': 'x'}],
                        on_public=lambda _: None, max_tokens=64)
                self.assert_unknown()

    def test_compressed_response_closed_before_consumption_preserves_unknown(self):
        for encoding in ('gzip', 'deflate', 'zstd'):
            with self.subTest(encoding=encoding):
                self.compression_ledger('early-close-' + encoding)
                client, _ = self.compressed_client(encoding, self.sse())
                request = client.build_request('POST', 'https://api.deepseek.com/chat/completions',
                    json={'model': 'deepseek-flash', 'temperature': 0.1, 'stream': True, 'max_tokens': 64})
                response = client.send(request, stream=True)
                response.close()
                self.assert_unknown()

    def test_json_missing_usage_keeps_full_reservation_and_halts(self):
        client, _ = self.client(lambda _: httpx.Response(200, json=self.normal_response(usage=None)))
        with self.assertRaises(BUDGET.BudgetHalt):
            self.provider(client).send([{'role': 'user', 'content': 'x'}], max_tokens=64)
        self.assert_unknown()

    def test_stream_done_without_usage_keeps_full_reservation_and_halts(self):
        client, _ = self.client(lambda _: httpx.Response(200, stream=Chunks([self.sse(usage=None)])))
        with self.assertRaises(BUDGET.BudgetHalt):
            self.provider(client).send_stream([{'role': 'user', 'content': 'x'}], on_public=lambda _: None, max_tokens=64)
        self.assert_unknown()

    def test_stream_eof_without_done_is_unknown_even_with_usage(self):
        client, _ = self.client(lambda _: httpx.Response(200, stream=Chunks([self.sse(done=False)])))
        with self.assertRaisesRegex(RuntimeError, 'without DONE'):
            self.provider(client).send_stream([{'role': 'user', 'content': 'x'}], on_public=lambda _: None, max_tokens=64)
        self.assert_unknown()

    def test_stream_read_error_and_early_response_close_preserve_unknown(self):
        client, _ = self.client(lambda _: httpx.Response(200,
            stream=Chunks([b'data: {}\n\n'], read_error=httpx.ReadError('synthetic disconnect'))))
        with self.assertRaises(httpx.ReadError):
            self.provider(client).send_stream([{'role': 'user', 'content': 'x'}], on_public=lambda _: None, max_tokens=64)
        self.assert_unknown()

    def test_close_before_consuming_response_preserves_unknown(self):
        client, _ = self.client(lambda _: httpx.Response(200, stream=Chunks([self.sse()])))
        request = client.build_request('POST', 'https://api.deepseek.com/chat/completions',
            json={'model': 'deepseek-flash', 'temperature': 0.1, 'stream': True, 'max_tokens': 64})
        response = client.send(request, stream=True)
        response.close()
        self.assert_unknown()

    def test_inner_close_exception_still_marks_incomplete_stream_unknown(self):
        client, _ = self.client(lambda _: httpx.Response(200,
            stream=Chunks([self.sse()], close_error=RuntimeError('synthetic close error'))))
        request = client.build_request('POST', 'https://api.deepseek.com/chat/completions',
            json={'model': 'deepseek-flash', 'temperature': 0.1, 'stream': True, 'max_tokens': 64})
        response = client.send(request, stream=True)
        with self.assertRaisesRegex(RuntimeError, 'close error'):
            response.close()
        self.assert_unknown()

    def test_conflicting_stream_usage_is_unknown_and_never_settled(self):
        wire = self.sse(done=False)
        changed = dict(USAGE, completion_tokens=21, total_tokens=121)
        wire += ('data: ' + json.dumps({'usage': changed}) + '\n\ndata: [DONE]\n\n').encode()
        client, _ = self.client(lambda _: httpx.Response(200, stream=Chunks([wire])))
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            self.provider(client).send_stream([{'role': 'user', 'content': 'x'}], on_public=lambda _: None, max_tokens=64)
        self.assert_unknown()

    def test_stream_model_change_is_unknown(self):
        wire = self.sse(done=False) + b'data: {"model":"different-model"}\n\ndata: [DONE]\n\n'
        client, _ = self.client(lambda _: httpx.Response(200, stream=Chunks([wire])))
        with self.assertRaisesRegex(ValueError, 'model changed'):
            self.provider(client).send_stream([{'role': 'user', 'content': 'x'}], on_public=lambda _: None, max_tokens=64)
        self.assert_unknown()

    def test_malformed_stream_json_is_unknown(self):
        client, _ = self.client(lambda _: httpx.Response(200, stream=Chunks([b'data: {broken}\n\n'])))
        with self.assertRaises(json.JSONDecodeError):
            self.provider(client).send_stream([{'role': 'user', 'content': 'x'}], on_public=lambda _: None, max_tokens=64)
        self.assert_unknown()

    def test_http_402_is_retained_and_second_attempt_is_denied_before_inner(self):
        client, _ = self.client(lambda _: httpx.Response(402, json={'error': 'synthetic balance failure'}))
        provider = self.provider(client)
        with self.assertRaises(ProviderResponseError) as raised:
            provider.send([{'role': 'user', 'content': 'x'}], max_tokens=64)
        self.assertEqual(raised.exception.status_code, 402)
        with self.assertRaises(RequestNotSent):
            provider.send([{'role': 'user', 'content': 'x'}], max_tokens=64)
        self.assertEqual(len(self.sent), 1)
        self.assert_unknown()

    def test_inner_timeout_is_unknown_and_not_automatically_retried(self):
        def handler(_):
            raise httpx.ReadTimeout('synthetic timeout')
        client, _ = self.client(handler)
        with self.assertRaises(httpx.ReadTimeout):
            self.provider(client).send([{'role': 'user', 'content': 'x'}], max_tokens=64)
        self.assertEqual(len(self.sent), 1)
        self.assert_unknown()

    def test_non_frozen_urls_parameters_are_rejected_without_reservation_or_send(self):
        client, _ = self.client(lambda _: httpx.Response(200, json=self.normal_response()))
        ordinary = {'model': 'deepseek-flash', 'temperature': 0.1, 'max_tokens': 64}
        invalid = [('GET', 'https://api.deepseek.com/chat/completions', ordinary),
            ('POST', 'http://api.deepseek.com/chat/completions', ordinary),
            ('POST', 'https://example.invalid/chat/completions', ordinary),
            ('POST', 'https://api.deepseek.com:444/chat/completions', ordinary),
            ('POST', 'https://api.deepseek.com/v1/chat/completions', ordinary),
            ('POST', 'https://api.deepseek.com/chat/completions?x=1', ordinary)]
        for change in [{'model': 'different'}, {'temperature': 0.2}, {'thinking': {'type': 'enabled'}},
                {'max_tokens': 0}, {'max_tokens': 65537}, {'max_tokens': True}, {'max_tokens': None}]:
            invalid.append(('POST', 'https://api.deepseek.com/chat/completions', {**ordinary, **change}))
        for method, url, payload in invalid:
            with self.subTest(method=method, url=url, payload=payload):
                with self.assertRaises(RequestNotSent):
                    client.request(method, url, json=payload)
        self.assertFalse(self.sent)
        self.assertFalse(self.ledger.snapshot()['reservations'])

    def test_request_journal_failure_is_confirmed_not_sent(self):
        client, transport = self.client(lambda _: httpx.Response(200, json=self.normal_response()))
        with patch.object(transport, '_write', side_effect=OSError('synthetic disk full')):
            with self.assertRaises(OSError):
                self.provider(client).send([{'role': 'user', 'content': 'x'}], max_tokens=64)
        snapshot = self.ledger.snapshot()
        self.assertFalse(self.sent)
        self.assertEqual(snapshot['reservations'][0]['status'], 'not_sent')
        self.assertEqual(snapshot['charged_or_reserved']['requests'], 0)

    def test_native_image_content_is_rejected_before_reservation_or_send(self):
        client, _ = self.client(lambda _: httpx.Response(200, json=self.normal_response()))
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': '合成图像问题'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,c3ludGhldGlj'}}]}]
        with self.assertRaises(RequestNotSent):
            self.provider(client).send(messages, max_tokens=64)
        snapshot = self.ledger.snapshot()
        self.assertFalse(self.sent)
        self.assertEqual(snapshot['physical_sent_requests'], 0)
        self.assertFalse(snapshot['reservations'])

    def test_deadline_crossing_before_mark_sent_is_confirmed_not_sent(self):
        client, _ = self.client(lambda _: httpx.Response(200, json=self.normal_response()))
        original = self.ledger.reserve
        def delayed_reserve(*args, **kwargs):
            value = original(*args, **kwargs)
            self.now += 901
            return value
        with patch.object(self.ledger, 'reserve', side_effect=delayed_reserve):
            with self.assertRaises(RequestNotSent):
                self.provider(client).send([{'role': 'user', 'content': 'x'}], max_tokens=64)
        snapshot = self.ledger.snapshot()
        self.assertFalse(self.sent)
        self.assertEqual(snapshot['reservations'][0]['status'], 'not_sent')
        self.assertEqual(snapshot['charged_or_reserved']['requests'], 0)

    def test_parallel_child_role_clients_use_one_root_family_and_shared_actual_totals(self):
        providers = [self.provider(self.client(lambda _: httpx.Response(200,
            json=self.normal_response()))[0]) for _ in range(2)]
        def send(index):
            return providers[index % 2].send([{'role': 'system', 'content': f'child-role-{index % 2}'},
                {'role': 'user', 'content': 'x'}], max_tokens=64)
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(send, range(8)))
        snapshot = self.ledger.snapshot()
        self.assertEqual(len(responses), 8)
        self.assertEqual(len(self.sent), 8)
        self.assertEqual(snapshot['physical_sent_requests'], 8)
        self.assertEqual(snapshot['actual_known_tokens'], 8 * 120)
        self.assertEqual({r['root_id'] for r in snapshot['reservations']}, {'root'})
        self.assertEqual(len(snapshot['roots']), 1)


if __name__ == '__main__':
    unittest.main()
