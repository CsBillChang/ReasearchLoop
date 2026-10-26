"""发布配置合同；不加载真实模型或读取本机凭据。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from research_loop import __version__
from research_loop.config import Settings
from research_loop.main import app


class ReleaseConfigurationTests(unittest.TestCase):
    """默认装配与显式回退均需要可审阅的固定值。"""

    def test_default_and_explicit_cpu_route(self):
        """环境缺省采用已批准方案，旧路线通过显式配置选择。"""
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env(directory)
            self.assertEqual(settings.retrieval_route, 'located_fp16')
            self.assertEqual(settings.retrieval_reranker_backend, 'mps_fp16')
            with patch.dict(os.environ, {'RESEARCHLOOP_RETRIEVAL_ROUTE': 'hybrid_rrf',
                                        'RESEARCHLOOP_RERANKER_BACKEND': 'onnx_cpu'}):
                old = Settings.from_env(directory)
            self.assertEqual(old.retrieval_route, 'hybrid_rrf')
            self.assertEqual(old.retrieval_reranker_backend, 'onnx_cpu')

    def test_misspelled_backend_is_rejected(self):
        """未知后端不能被当成CPU或GPU继续执行。"""
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'RESEARCHLOOP_RERANKER_BACKEND': 'mp_fp16'}, clear=True):
            with self.assertRaises(ValidationError):
                Settings.from_env(directory)

    def test_release_metadata_agrees(self):
        """前端、Python和OpenAPI显示同一发布版本。"""
        root = Path(__file__).resolve().parents[3]
        self.assertEqual(__version__, '1.5.5')
        self.assertEqual(app.version, __version__)
        self.assertEqual(json.loads((root / 'frontend/package.json').read_text())['version'], __version__)
        lock = json.loads((root / 'frontend/package-lock.json').read_text())
        self.assertEqual(lock['version'], __version__)
        self.assertEqual(lock['packages']['']['version'], __version__)
