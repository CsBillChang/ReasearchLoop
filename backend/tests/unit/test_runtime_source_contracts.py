"""运行与取文合同的零网络合成验证；不读取部署配置或真实PDF。"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
import pymupdf

from research_loop.config import Settings
from research_loop.providers.pdf import PDFProvider, PaperReadCancelled
from research_loop.providers.read_retry import read_with_backoff
from research_loop.services import runtime
from research_loop.storage.files import FileStore, sha256_bytes
from scripts.manage import Manager


class RuntimeSourceContracts(unittest.TestCase):
    """冻结配置、依赖与进程身份的正反例。"""

    def setUp(self):
        """每例只在临时目录生成合成文件，并阻断网络。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict('os.environ', {}, clear=True))
        self.stack.enter_context(patch.object(socket, 'create_connection', side_effect=AssertionError('network prohibited')))
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('network prohibited')))
        self.settings = Settings(project_root=self.root, data_dir=self.root / 'data', output_dir=self.root / 'out',
                                 database_url='sqlite:///' + str(self.root / 'db'))
        source = self.root / 'backend/research_loop/synthetic.py'
        source.parent.mkdir(parents=True)
        source.write_text('VALUE = 1\n')
        self.source = source

    def test_fingerprint_is_deterministic_and_separate_from_ownership(self):
        """模型/预算或源码变化影响运行指纹，不改变实例的停止权限身份。"""
        first = runtime.runtime_fingerprint(self.settings)
        self.assertEqual(first, runtime.runtime_fingerprint(self.settings))
        changed = self.settings.model_copy(update={'context_output_reserve': 2048})
        self.assertEqual(runtime.identity(self.settings), runtime.identity(changed))
        self.assertNotEqual(first['fingerprint_id'], runtime.runtime_fingerprint(changed)['fingerprint_id'])
        self.source.write_text('VALUE = 2\n')
        self.assertNotEqual(first['fingerprint_id'], runtime.runtime_fingerprint(self.settings)['fingerprint_id'])

    def test_lock_env_manifest_and_interpreter_changes_are_detected(self):
        """锁、配置文件、模型清单和错误解释器prefix均可区分。"""
        for name in ('backend/uv.lock', '.env', 'data/ag71-preflight/models/manifest.json'):
            with self.subTest(name=name):
                before = runtime.runtime_fingerprint(self.settings)['fingerprint_id']
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic-change')
                self.assertNotEqual(before, runtime.runtime_fingerprint(self.settings)['fingerprint_id'])
        before = runtime.runtime_fingerprint(self.settings)['fingerprint_id']
        with patch.object(sys, 'prefix', '/synthetic/wrong-venv'):
            self.assertNotEqual(before, runtime.runtime_fingerprint(self.settings)['fingerprint_id'])

    def test_report_never_outputs_secrets_or_claims_config_provenance(self):
        """所有可包含认证的地址及配置只输出摘要，文件hash不冒充值来源。"""
        settings = self.settings.model_copy(update={'deepseek_api_key': 'PRIVATE-KEY', 'supabase_key': 'SOURCE-KEY',
            'database_url': 'mysql://user:DB-PASSWORD@localhost/db',
            'deepseek_base_url': 'https://name:URL-PASSWORD@fixture.invalid/?token=QUERY-SECRET',
            'discovery_backends': {'fixture': {'anon_key': 'NESTED-SECRET'}}})
        (self.root / '.env').write_text('OTHER=DOTENV-SECRET\n')
        report = runtime.runtime_fingerprint(settings)
        rendered = json.dumps(report)
        for secret in ('PRIVATE-KEY', 'SOURCE-KEY', 'DB-PASSWORD', 'URL-PASSWORD', 'QUERY-SECRET', 'NESTED-SECRET', 'DOTENV-SECRET'):
            self.assertNotIn(secret, rendered)
        self.assertIn('do not prove', report['configuration_provenance'])
        changed = settings.model_copy(update={'deepseek_api_key': 'DIFFERENT-PRIVATE-KEY'})
        self.assertNotEqual(report['fingerprint_id'], runtime.runtime_fingerprint(changed)['fingerprint_id'])

    def test_disabled_rag_never_imports_or_reads_models(self):
        """关闭的组件无需依赖，也不能影响其他启动路径。"""
        with patch('importlib.import_module', side_effect=AssertionError('no import')):
            result = runtime.local_component_readiness(self.settings.model_copy(update={'retrieval_enabled': False}))
        self.assertEqual({row['status'] for row in result.values()}, {'disabled'})

    def _model_manifest(self):
        """构造轻量清单，文件只是合成字节，不是可运行模型。"""
        payload = {}
        for name, repo in (('embedding', 'intfloat/multilingual-e5-small'),
                           ('reranker', 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')):
            folder = self.root / name
            folder.mkdir()
            files = {}
            for filename in ('model.onnx', 'tokenizer.json', 'config.json'):
                data = ('synthetic ' + filename).encode()
                (folder / filename).write_bytes(data)
                files[filename] = {'sha256': hashlib.sha256(data).hexdigest()}
            payload[name] = {'repo': repo, 'directory': str(folder), 'weights': 'model.onnx', 'files': files}
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(payload))
        return path

    def test_readiness_import_and_file_hash_checks_without_inference(self):
        """就绪只由实际导入/文件核对支持，缺依赖或权重变化单独报告。"""
        settings = self.settings.model_copy(update={'retrieval_model_manifest': self._model_manifest(),
                                                   'retrieval_reranker_backend': 'onnx_cpu'})
        with patch('importlib.import_module', return_value=object()) as imports:
            report = runtime.local_component_readiness(settings)
        self.assertEqual({row['status'] for row in report.values()}, {'ready'})
        self.assertEqual(imports.call_count, 3)
        self.assertTrue(all(not row['inference_probed'] for row in report.values()))
        (self.root / 'embedding/model.onnx').write_bytes(b'changed')
        with patch('importlib.import_module', side_effect=ImportError('private path must not leak')):
            report = runtime.local_component_readiness(settings)
        self.assertFalse(report['embedding']['files']['model.onnx']['matches'])
        self.assertEqual({row['status'] for row in report.values()}, {'unavailable'})
        self.assertNotIn('private path', json.dumps(report))

    def test_heartbeat_freezes_actual_identity_once_and_rejects_wrong_expected(self):
        """心跳复用启动快照，不把后改源码误报为当前进程已加载。"""
        target = self.root / 'beat.json'
        with patch.dict('os.environ', {'RESEARCHLOOP_HEARTBEAT': str(target), 'RESEARCHLOOP_PROCESS_TOKEN': 'synthetic'}), \
                patch.object(runtime, '_PROCESS_FINGERPRINT', None):
            first = runtime.configure_runtime(self.settings)
            self.source.write_text('VALUE = 3\n')
            with patch.object(runtime, 'runtime_fingerprint', side_effect=AssertionError('heartbeat must not rescan')):
                runtime.heartbeat('ready')
                runtime.heartbeat('busy')
            self.assertEqual(json.loads(target.read_text())['runtime_fingerprint_id'], first['fingerprint_id'])
        with patch.dict('os.environ', {'RESEARCHLOOP_EXPECTED_RUNTIME_FINGERPRINT': 'different'}):
            with self.assertRaisesRegex(RuntimeError, 'RUNTIME_FINGERPRINT_MISMATCH'):
                runtime.configure_runtime(self.settings)

    def test_manager_reuse_requires_registered_and_actual_fingerprints(self):
        """旧记录未知、错误心跳或配置差异都不自动停止/启动活动进程。"""
        with patch('scripts.manage.ROOT', self.root), patch('scripts.manage.owned', return_value=True), \
                patch('scripts.manage.os.kill') as kill, patch('scripts.manage.subprocess.Popen') as spawn:
            manager = Manager(self.settings)
            state = {'configuration_id': runtime.identity(self.settings), 'api_port': 18082, 'frontend_port': 4173,
                'readonly': False, 'scheduler': False, 'worker': True,
                'services': {'worker': {'pid': 111, 'token': 'synthetic', 'signature': 'synthetic'}}}
            runtime.atomic_json(manager.state_path, state)
            with self.assertRaisesRegex(RuntimeError, 'RUNTIME_FINGERPRINT_UNKNOWN'):
                manager.start()
            fingerprint = runtime.runtime_fingerprint(self.settings)
            state['runtime_fingerprint'] = fingerprint
            runtime.atomic_json(manager.state_path, state)
            beat = {'pid': 111, 'token': 'synthetic', 'runtime_fingerprint_id': 'wrong'}
            runtime.atomic_json(manager.directory / 'worker.ready.json', beat)
            with self.assertRaisesRegex(RuntimeError, 'RUNTIME_FINGERPRINT_MISMATCH'):
                manager.start()
            beat['runtime_fingerprint_id'] = fingerprint['fingerprint_id']
            runtime.atomic_json(manager.directory / 'worker.ready.json', beat)
            with patch('scripts.manage.probe', return_value={'ok': True}):
                self.assertEqual(manager.start()['services']['worker']['runtime_consistency'], 'match')
            self.source.write_text('VALUE = 4\n')
            with self.assertRaisesRegex(RuntimeError, 'RUNTIME_FINGERPRINT_MISMATCH'):
                manager.start()
            kill.assert_not_called()
            spawn.assert_not_called()


class SourceBoundaryContracts(unittest.TestCase):
    """HTTP流、URL缓存合同和取消退避不触达外部服务。"""

    def setUp(self):
        """创建临时文件存储，阻断所有socket连接。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.files = FileStore(self.root / 'data', self.root / 'out', self.root)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(socket, 'create_connection', side_effect=AssertionError('network prohibited')))
        self.stack.enter_context(patch.object(socket.socket, 'connect', side_effect=AssertionError('network prohibited')))
        self.text = '# Synthetic scientific paper\n' + ('Methods and scientific evidence with synthetic controlled results.\n' * 60)
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((40, 40), self.text, fontsize=7)
        self.pdf_bytes = document.tobytes()
        document.close()

    def provider(self, handler, **kwargs):
        """仅通过内存HTTP响应构造provider。"""
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return PDFProvider(self.files, client=client, **kwargs)

    def test_oversized_jina_stops_same_path_and_keeps_pdf_fallback(self):
        """超字节不是临时故障；只发一次Jina，再走现有PDF路径。"""
        calls = []
        def handle(request):
            """每次Jina返回同一确定超限，PDF返回小合成文档。"""
            calls.append(request)
            return httpx.Response(200, content=b'x' * 100001 if request.url.host == 'r.jina.ai' else self.pdf_bytes)
        provider = self.provider(handle, max_bytes=100000)
        with patch('research_loop.providers.pdf.time.sleep', side_effect=AssertionError('no pointless retry')):
            result = provider.ingest('https://fixture.invalid/p.pdf', 'synthetic')
        self.assertEqual(result.parser_name, 'pymupdf')
        self.assertEqual(len(calls), 2)
        self.assertEqual(provider.last_fetch['jina_attempts'][0]['retry_stopped_reason'], 'response_byte_limit')
        self.assertEqual(provider.max_bytes, 100000)
        self.assertEqual(provider.timeout, 60)

    def test_url_pdf_cache_checks_version_and_config_retains_old_snapshot(self):
        """URL索引不绕过PDF解析版本/配置；不可变旧快照继续存在。"""
        calls = []
        def handle(request):
            """Jina不可用，PDF按原顺序回退。"""
            calls.append(request)
            return httpx.Response(403) if request.url.host == 'r.jina.ai' else httpx.Response(200, content=self.pdf_bytes)
        provider = self.provider(handle)
        first = provider.ingest('https://fixture.invalid/p.pdf', 'one')
        first_bytes = Path(first.cache_path).read_bytes()
        again = provider.ingest('https://fixture.invalid/p.pdf', 'alias')
        self.assertEqual(again.document_id, first.document_id)
        self.assertEqual(len(calls), 2)
        provider.parser_version = 'synthetic-v2'
        changed = provider.ingest('https://fixture.invalid/p.pdf', 'one')
        self.assertNotEqual(changed.document_id, first.document_id)
        self.assertEqual(len(calls), 4)
        provider.parser_config = {'text_route': 'cache-jina-pymupdf', 'min_text_chars': 1300}
        final = provider.ingest('https://fixture.invalid/p.pdf', 'one')
        self.assertNotEqual(final.document_id, changed.document_id)
        self.assertEqual(len(calls), 6)
        self.assertEqual(Path(first.cache_path).read_bytes(), first_bytes)

    def test_jina_has_independent_contract_and_no_physical_pages(self):
        """Jina缓存按自己的reader合同判断，PDF版本变动不伪造物理页。"""
        calls = []
        provider = self.provider(lambda request: calls.append(request) or httpx.Response(200, text=self.text))
        first = provider.ingest('https://fixture.invalid/p.pdf', 'one')
        self.assertIsNone(first.total_pages)
        provider.parser_version = 'pdf-v2'
        second = provider.ingest('https://fixture.invalid/p.pdf', 'alias')
        self.assertEqual(second.document_id, first.document_id)
        self.assertEqual(len(calls), 1)
        index = self.files.parse_dir / ('source-' + sha256_bytes(b'https://fixture.invalid/p.pdf') + '.json')
        cached = json.loads(index.read_text())
        cached['parser_version'] = 'stale-reader'
        index.write_text(json.dumps(cached))
        provider.ingest('https://fixture.invalid/p.pdf', 'one')
        self.assertEqual(len(calls), 2)
        with self.assertRaises(ValueError):
            provider.pages(first, 1, 1)

    def test_cancelled_backoff_never_starts_next_request(self):
        """通用只读退避在等待中检查现有取消信号，保留原取消类型。"""
        flag, calls, waits = threading.Event(), [], []
        def check():
            """抛出现有取消异常而非HTTP失败。"""
            if flag.is_set():
                raise PaperReadCancelled('synthetic cancellation')
        def sleep(seconds):
            """在首个等待片段取消。"""
            waits.append(seconds)
            flag.set()
        with self.assertRaises(PaperReadCancelled):
            read_with_backoff(lambda: calls.append(1) or httpx.Response(503), {},
                              sleep=sleep, check_cancelled=check)
        self.assertEqual(len(calls), 1)
        self.assertLess(sum(waits), 1)

    def test_active_backoff_preserves_wait_budget_and_attempt_count(self):
        """无取消的分段等待仍累加为原退避，不能改变四次尝试合同。"""
        waits, calls, report = [], [], {}
        read_with_backoff(lambda: calls.append(1) or httpx.Response(503), report,
                          sleep=waits.append, jitter=lambda: 1, check_cancelled=lambda: None)
        self.assertEqual(len(calls), 4)
        self.assertAlmostEqual(sum(waits), 7)
        self.assertEqual(report['total_wait_seconds'], 7)
        self.assertEqual(report['retry_stopped_reason'], 'attempt_limit')
