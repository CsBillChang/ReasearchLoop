"""非流式原生响应也必须通过完整压缩EOF，禁止缺尾JSON提前结算。"""
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zlib

import httpx

from research_loop.providers.deepseek import DeepSeekProvider, RequestNotSent


SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


def load_script(name):
    """只加载本实验工作树的共同支架，不读取旧实验账本或业务配置。"""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUDGET = load_script('harness_ab_budget')
TRANSPORT = load_script('harness_ab_transport')
USAGE = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
    'prompt_cache_hit_tokens': 40, 'prompt_cache_miss_tokens': 60,
    'completion_tokens_details': {'reasoning_tokens': 12}}


class Chunks(httpx.SyncByteStream):
    """以任意网络分块交付合成字节，完全不创建套接字。"""

    def __init__(self, values):
        self.values = values

    def __iter__(self):
        yield from self.values

    def close(self):
        pass


class DedupAbTransportTests(unittest.TestCase):
    """每项压缩场景使用新合成账本与MockTransport，不启动正式实验。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dedup-json-transport-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def response():
        """提供原生解析器可以接受的JSON，子集Token不得重复计费。"""
        return {'id': 'synthetic-json-compressed', 'model': 'deepseek-flash',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                'role': 'assistant', 'content': '中文合成结果'}}], 'usage': USAGE}

    @staticmethod
    def compress(encoding, value):
        if encoding == 'gzip':
            return gzip.compress(value)
        if encoding == 'deflate':
            return zlib.compress(value)
        import zstandard
        return zstandard.ZstdCompressor(write_checksum=True).compress(value)

    def client(self, label, encoding, *, truncated=False):
        """显式保留压缩响应头，HTTPX必须实际执行解码而非接收明文替身。"""
        root = self.root / label
        ledger = BUDGET.Ledger(root / 'synthetic.sqlite3',
            allowed_cells=[('R1-Q1', 'A')], batch_tokens=16_000_000,
            batch_requests=256, batch_yuan='64', batch_seconds=5400)
        ledger.begin_root('root', 'R1-Q1', 'A')
        plain = json.dumps(self.response(), ensure_ascii=False).encode()
        compressed = self.compress(encoding, plain)
        if truncated:
            # 仅删压缩尾部；正文可能已完整解出，不能用JSON合法性代替帧完整性。
            compressed = compressed[:-2]
            if encoding in {'gzip', 'deflate'}:
                decoder = zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == 'gzip' else zlib.MAX_WBITS)
                self.assertEqual(decoder.decompress(compressed), plain)
                self.assertFalse(decoder.eof)
        sent = []

        def handle(request):
            sent.append(request)
            return httpx.Response(200, headers={'Content-Encoding': encoding,
                'Content-Length': str(len(compressed)), 'Content-Type': 'application/json'},
                stream=Chunks([compressed[i:i + 3] for i in range(0, len(compressed), 3)]))

        transport = TRANSPORT.BudgetTransport(ledger, 'root', root / 'requests',
            inner=httpx.MockTransport(handle))
        client = httpx.Client(transport=transport, trust_env=False)
        self.addCleanup(client.close)
        provider = DeepSeekProvider('synthetic-local-only', model='deepseek-flash', client=client)
        return provider, ledger, transport, sent, plain

    def test_native_json_gzip_deflate_zstd_complete_frame_settles_exact_usage(self):
        for encoding in ('gzip', 'deflate', 'zstd'):
            with self.subTest(encoding=encoding):
                provider, ledger, transport, sent, plain = self.client('complete-' + encoding, encoding)
                raw = provider.send([{'role': 'user', 'content': '合成问题'}], max_tokens=64)
                self.assertEqual(raw, self.response())
                snapshot = ledger.snapshot()
                self.assertEqual(len(sent), 1)
                self.assertEqual(snapshot['physical_sent_requests'], 1)
                self.assertEqual(snapshot['actual_known_tokens'], 120)
                self.assertEqual(snapshot['actual_known_estimated_nano_yuan'], 60 * 2000 + 40 * 40 + 20 * 8000)
                self.assertEqual(snapshot['reservations'][0]['status'], 'settled')
                self.assertIsNone(snapshot['halt_reason'])
                self.assertEqual(next(transport.directory.glob('*.response.json')).read_bytes(), plain)
                for artifact in transport.directory.iterdir():
                    self.assertNotIn(b'synthetic-local-only', artifact.read_bytes())

    def test_native_json_truncated_compressed_trailer_keeps_unknown_and_blocks_next_send(self):
        for encoding in ('gzip', 'deflate', 'zstd'):
            with self.subTest(encoding=encoding):
                provider, ledger, _, sent, _ = self.client('truncated-' + encoding, encoding, truncated=True)
                with self.assertRaises(httpx.DecodingError):
                    provider.send([{'role': 'user', 'content': '合成问题'}], max_tokens=64)
                snapshot = ledger.snapshot()
                self.assertEqual(len(sent), 1)
                self.assertEqual(snapshot['reservations'][0]['status'], 'unknown')
                self.assertEqual(snapshot['actual_known_tokens'], 0)
                self.assertEqual(snapshot['charged_or_reserved']['tokens'], 1_048_576 + 64)
                self.assertTrue(snapshot['halt_reason'])
                with self.assertRaises(RequestNotSent):
                    provider.send([{'role': 'user', 'content': '不得自动重试'}], max_tokens=64)
                self.assertEqual(len(sent), 1)


if __name__ == '__main__':
    unittest.main()
