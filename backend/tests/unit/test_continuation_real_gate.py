"""真实接续入口的纯离线发送门合同；临时账本与 MockTransport 不连接网络。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

from research_loop.providers.deepseek import RequestNotSent


EXPERIMENT = Path(__file__).resolve().parents[2] / 'experiments' / 'memory_continuation'


def load_experiment(name: str):
    """按文件加载实验模块，并还原 runner 的临时导入路径调整。"""
    spec = importlib.util.spec_from_file_location('continuation_test_' + name, EXPERIMENT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    previous = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    return module


gate_module = load_experiment('real_gate')
runner_module = load_experiment('run_real')


class ChunkedResponse(httpx.SyncByteStream):
    """返回固定原始字节，可模拟中途断流。"""

    def __init__(self, chunks: list[bytes], *, interrupted=False):
        """注入合成响应，不创建客户端或连接。"""
        self.chunks = chunks
        self.interrupted = interrupted

    def __iter__(self):
        """保留给定字节边界，断流时不伪造结束标记。"""
        yield from self.chunks
        if self.interrupted:
            raise RuntimeError('合成响应中断')


class ContinuationRealGateTests(unittest.TestCase):
    """验证发送前拒绝、持久预算、未知用量及响应原件保留。"""

    def setUp(self):
        """只使用当前测试临时目录，阻断所有真实 socket 与 HTTPTransport。"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for target in ('socket.socket.connect', 'socket.socket.connect_ex',
                       'httpx.HTTPTransport.handle_request',
                       'httpx.AsyncHTTPTransport.handle_async_request'):
            blocker = patch(target, side_effect=AssertionError('真实发送门单测禁止网络'))
            blocker.start()
            self.addCleanup(blocker.stop)

    @staticmethod
    def payload(**overrides) -> dict:
        """提供纯合成消息和冻结输出上限，不读取环境凭据。"""
        return {'model': 'synthetic-model', 'messages': [{'role': 'user', 'content': '合成请求'}],
                'tools': [], 'max_tokens': 8192, **overrides}

    def gate(self, name='gate', **kwargs):
        """每个拒绝分支使用独立账本，停止原因不能互相覆盖。"""
        return gate_module.SharedGate(self.root / name, **kwargs)

    def test_nonofficial_urls_are_rejected_before_mock_inner(self):
        """非官方、代理、loopback、带认证或查询参数的地址均不进入底层。"""
        urls = [
            'http://api.deepseek.com/chat/completions',
            'https://proxy.invalid/chat/completions',
            'https://127.0.0.1/chat/completions',
            'https://localhost/chat/completions',
            'https://api.deepseek.com.proxy.invalid/chat/completions',
            'https://user:password@api.deepseek.com/chat/completions',
            'https://api.deepseek.com/chat/completions?extra=1',
            'https://api.deepseek.com/chat/completions#fragment',
            'https://api.deepseek.com:8443/chat/completions',
            'https://api.deepseek.com/unexpected',
        ]
        for index, url in enumerate(urls):
            with self.subTest(url=url):
                gate = self.gate('endpoint-' + str(index))
                inner_calls = []
                inner = httpx.MockTransport(lambda request: inner_calls.append(request) or httpx.Response(200))
                transport = gate_module.GuardedTransport(gate, inner=inner)
                request = httpx.Request('POST', url, json=self.payload())
                with self.assertRaisesRegex(RequestNotSent, 'GATE_FORBIDDEN_ENDPOINT'):
                    transport.handle_request(request)
                self.assertEqual(inner_calls, [])
                self.assertEqual(gate.summary()['admitted_requests'], 0)
                self.assertEqual(gate.summary()['stop_reason'], 'GATE_FORBIDDEN_ENDPOINT')

    def test_wrong_method_is_rejected_before_inner(self):
        """合法主机上的非POST请求仍在发送前拒绝。"""
        gate = self.gate()
        inner = httpx.MockTransport(lambda request: self.fail('不应调用底层'))
        transport = gate_module.GuardedTransport(gate, inner=inner)
        with self.assertRaisesRegex(RequestNotSent, 'GATE_FORBIDDEN_ENDPOINT'):
            transport.handle_request(httpx.Request('GET', 'https://api.deepseek.com/chat/completions'))
        self.assertEqual(gate.summary()['admitted_requests'], 0)

    def test_request_limit_does_not_admit_an_extra_attempt(self):
        """达到发送次数上限后只记录停止，不产生新的物理准入号。"""
        gate = self.gate(max_requests=1)
        index = gate.before(self.payload())
        gate.settle(index, {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30})
        with self.assertRaisesRegex(RequestNotSent, 'GATE_REQUEST_LIMIT'):
            gate.before(self.payload())
        summary = gate.summary()
        self.assertEqual(summary['admitted_requests'], 1)
        self.assertEqual(summary['used_tokens'], 30)
        self.assertEqual(summary['reserved_tokens'], 0)

    def test_missing_or_inconsistent_usage_stays_unknown_and_reserved(self):
        """缺失、矛盾或异常usage保留未知预留，不结算成零。"""
        values = [None, {}, {'total_tokens': True},
                  {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 2}]
        for index, usage in enumerate(values):
            with self.subTest(usage=usage):
                gate = self.gate('usage-' + str(index))
                admitted = gate.before(self.payload())
                reservation = gate.summary()['reserved_tokens']
                gate.settle(admitted, usage)
                summary = gate.summary()
                self.assertEqual(summary['requests'][0]['status'], 'unknown')
                self.assertIsNone(summary['requests'][0]['actual_tokens'])
                self.assertEqual(summary['reserved_tokens'], reservation)
                self.assertGreater(reservation, 0)
                self.assertEqual(gate.active_index, admitted)
                self.assertEqual(summary['stop_reason'], 'GATE_USAGE_UNKNOWN')
                with self.assertRaises(RequestNotSent):
                    gate.before(self.payload())
                self.assertEqual(gate.summary()['admitted_requests'], 1)

    def test_ledger_reentry_keeps_usage_and_original_deadline(self):
        """新门实例沿用已用次数、用量与首次时钟，不刷新剩余时间。"""
        with patch.object(gate_module.time, 'monotonic', return_value=100.0):
            gate = self.gate()
            gate.settle(gate.before(self.payload()), {'total_tokens': 37})
            created = gate.summary()['created_monotonic']
        with patch.object(gate_module.time, 'monotonic', return_value=150.0):
            reopened = self.gate()
            summary = reopened.summary()
        self.assertEqual(summary['created_monotonic'], created)
        self.assertEqual(summary['admitted_requests'], 1)
        self.assertEqual(summary['used_tokens'], 37)
        self.assertEqual(summary['remaining_seconds'], 550.0)
        with self.assertRaisesRegex(ValueError, 'GATE_FROZEN_LIMITS_MISMATCH'):
            self.gate(max_requests=13)

    def test_two_instances_cannot_admit_concurrent_unsettled_requests(self):
        """两个独立实例竞争同一账本时，只允许一个未结算请求。"""
        first, second = self.gate(), self.gate()
        barrier = threading.Barrier(2)
        def attempt(gate):
            barrier.wait(timeout=5)
            try:
                return ('admitted', gate.before(self.payload()))
            except RequestNotSent as exc:
                return ('rejected', str(exc))
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, [first, second]))
        self.assertEqual(sum(kind == 'admitted' for kind, value in results), 1)
        self.assertEqual(sum(kind == 'rejected' for kind, value in results), 1)
        self.assertEqual(first.summary()['admitted_requests'], 1)
        self.assertEqual(first.active_index, 1)
        self.assertEqual(first.summary()['stop_reason'], 'GATE_UNSETTLED_REQUEST')

    def test_output_limit_must_equal_frozen_8192(self):
        """缺失、错型或不同输出上限都在准入前拒绝。"""
        for index, value in enumerate([None, True, '8192', 8191, 65536]):
            with self.subTest(value=value):
                gate = self.gate('output-' + str(index))
                with self.assertRaisesRegex(RequestNotSent, 'GATE_OUTPUT_LIMIT_MISMATCH'):
                    gate.before(self.payload(max_tokens=value))
                self.assertEqual(gate.summary()['admitted_requests'], 0)

    def test_token_reservation_rejects_before_admission(self):
        """输入与冻结输出预留超额时，不准入请求也不调用模型。"""
        gate = self.gate(token_limit=9000)
        with self.assertRaisesRegex(RequestNotSent, 'GATE_TOKEN_RESERVATION_LIMIT'):
            gate.before(self.payload())
        self.assertEqual(gate.summary()['admitted_requests'], 0)

    def test_expired_gate_does_not_reset_when_reopened(self):
        """截止后重新构造CLI账本实例仍然拒绝发送。"""
        with patch.object(gate_module.time, 'monotonic', return_value=100.0):
            self.gate(seconds=10)
        with patch.object(gate_module.time, 'monotonic', return_value=111.0):
            reopened = self.gate(seconds=10)
            with self.assertRaisesRegex(RequestNotSent, 'GATE_TIME_LIMIT'):
                reopened.before(self.payload())
        self.assertEqual(reopened.summary()['admitted_requests'], 0)

    def test_raw_response_stream_is_exact_and_authorization_is_not_saved(self):
        """原始流逐字保留，账本和响应文件均不包含请求Authorization。"""
        gate = self.gate()
        chunks = [b'data: {"id":"synthetic-1","choices":[]}\n\n',
                  b'data: {"usage":{"total_tokens":30}}\n\n', b'data: [DONE]\n\n']
        credential = 'UNIT_TEST_AUTHORIZATION_NEVER_SAVE'
        seen = []
        def respond(request):
            seen.append(request.headers['Authorization'])
            return httpx.Response(200, stream=ChunkedResponse(chunks))
        transport = gate_module.GuardedTransport(gate, inner=httpx.MockTransport(respond))
        with httpx.Client(transport=transport, trust_env=False) as client:
            with client.stream('POST', 'https://api.deepseek.com/v1/chat/completions',
                    headers={'Authorization': 'Bearer ' + credential}, json=self.payload()) as response:
                self.assertEqual(response.read(), b''.join(chunks))
        summary = gate.summary()
        row = summary['requests'][0]
        raw = gate.output / row['response_path']
        self.assertEqual(raw.read_bytes(), b''.join(chunks))
        self.assertEqual(row['response_sha256'], hashlib.sha256(b''.join(chunks)).hexdigest())
        self.assertTrue(row['response_stream_exhausted'])
        self.assertEqual(row['http_status'], 200)
        self.assertEqual(seen, ['Bearer ' + credential])
        self.assertEqual(stat.S_IMODE(raw.stat().st_mode), 0o600)
        self.assertIsNone(row['actual_tokens'])
        for path in gate.output.rglob('*'):
            if path.is_file():
                self.assertNotIn(credential.encode(), path.read_bytes())
        gate.settle(1, {'total_tokens': 30})
        self.assertEqual(gate.summary()['used_tokens'], 30)

    def test_interrupted_response_keeps_partial_bytes_and_unknown(self):
        """流中断保留已经收到的原件和未知预留，不写完成或零成本。"""
        gate = self.gate()
        chunks = [b'data: {"id":"partial"}\n\n']
        transport = gate_module.GuardedTransport(gate, inner=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=ChunkedResponse(chunks, interrupted=True))))
        with httpx.Client(transport=transport, trust_env=False) as client:
            with self.assertRaisesRegex(RuntimeError, '合成响应中断'):
                client.post('https://api.deepseek.com/chat/completions', json=self.payload())
        summary = gate.summary()
        row = summary['requests'][0]
        self.assertEqual((gate.output / row['response_path']).read_bytes(), b''.join(chunks))
        self.assertFalse(row['response_stream_exhausted'])
        self.assertIsNone(row['actual_tokens'])
        self.assertGreater(summary['reserved_tokens'], 0)
        self.assertEqual(summary['stop_reason'], 'GATE_RESPONSE_STREAM_FAILED')

    def test_stop_preserves_first_reason_and_pending_reservation(self):
        """停止调用不能清除在途预留，后续原因也不能覆盖首错。"""
        gate = self.gate()
        gate.before(self.payload())
        reserved = gate.summary()['reserved_tokens']
        gate.stop('first_failure')
        gate.stop('later_failure')
        summary = gate.summary()
        self.assertEqual(summary['stop_reason'], 'first_failure')
        self.assertEqual(summary['reserved_tokens'], reserved)
        self.assertEqual(summary['requests'][0]['status'], 'unknown')

    def test_credential_parser_rejects_proxy_and_loopback_without_real_credentials(self):
        """只读取测试生成的配置，并清空继承环境以免接触真实凭据。"""
        config = self.root / 'synthetic.env'
        endpoints = ['http://127.0.0.1:8080', 'https://127.0.0.1', 'https://localhost',
                     'https://proxy.invalid/v1', 'https://api.deepseek.com.proxy.invalid',
                     'https://user:password@api.deepseek.com', 'https://api.deepseek.com/v1?extra=1']
        with patch.dict(os.environ, {}, clear=True):
            for endpoint in endpoints:
                with self.subTest(endpoint=endpoint):
                    config.write_text('DEEPSEEK_API_KEY=unit-test-credential\n'
                                      'DEEPSEEK_MODEL=synthetic-model\nDEEPSEEK_BASE_URL=' + endpoint + '\n')
                    with self.assertRaisesRegex(ValueError, '只允许'):
                        runner_module.credential_values(config)
            config.write_text('DEEPSEEK_API_KEY=unit-test-credential\n'
                              'DEEPSEEK_MODEL=synthetic-model\nDEEPSEEK_BASE_URL=https://api.deepseek.com/v1\n')
            self.assertEqual(runner_module.credential_values(config)['DEEPSEEK_BASE_URL'],
                             'https://api.deepseek.com/v1')

    def test_isolated_settings_construct_only_pilot_sqlite_paths(self):
        """生产服务环境不能覆盖显式隔离设置，构造过程不创建数据库。"""
        output = self.root / 'pilot'
        with patch.dict(os.environ, {'DATABASE_URL': 'mysql://invalid.invalid/deployment',
                'RESEARCHLOOP_DATA_DIR': '/invalid-deployment-directory'}, clear=True):
            settings = runner_module.isolated_settings(output, {
                'DEEPSEEK_API_KEY': 'unit-test-credential', 'DEEPSEEK_MODEL': 'synthetic-model',
                'DEEPSEEK_BASE_URL': 'https://api.deepseek.com'})
        self.assertEqual(settings.database_url, 'sqlite:///' + str(output / 'runtime' / 'pilot.sqlite'))
        self.assertEqual(settings.project_root, output / 'runtime')
        self.assertEqual(settings.data_dir, output / 'runtime' / 'data')
        self.assertEqual(settings.output_dir, output / 'runtime' / 'notes')
        self.assertEqual(settings.context_output_reserve, 8192)
        self.assertFalse(settings.retrieval_enabled)
        self.assertFalse(settings.web_search_enabled)
        self.assertFalse(settings.web_fetch_enabled)
        self.assertEqual(settings.discovery_backends, {})
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
