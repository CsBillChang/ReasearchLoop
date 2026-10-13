# AG1 验收入口测试：固定成功、缺证和改路场景；默认不额外发起真实模型请求。
"""Evidence gate for the explicitly-run AG1 real acceptance.

The default test suite never spends model requests. Set RUN_AG1_REAL=1 only
after the bounded external cases have been executed and recorded.
"""
import json
import os
import unittest
from pathlib import Path


class AG1AcceptanceEvidenceTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv("RUN_AG1_REAL") == "1", "real AG1 acceptance is explicit and budgeted")
    def test_bounded_real_acceptance_manifest(self):
        root = Path(__file__).resolve().parents[3]
        manifest = root / "artifacts" / "ag1-implementation-20261007" / "acceptance-summary.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(data["scope"], "AG1 only")
        self.assertEqual(data["physical_requests"]["unknown"], 0)
        self.assertLessEqual(data["physical_requests"]["total"], 48)
        self.assertEqual(data["cases"]["gap_corrected"]["files_written"], False)
