# AG2 验收入口测试：核对固定场景与证据合同，真实调用须显式授权开启。
import json
import unittest
from pathlib import Path


class AG2AcceptanceEvidenceTests(unittest.TestCase):
    def test_real_acceptance_is_explicit_and_bounded(self):
        runner = Path(__file__).parents[2] / "scripts" / "run_ag2.py"
        self.assertTrue(runner.exists())
        self.assertIn("RESEARCHLOOP_RUN_AG2_REAL", runner.read_text(encoding="utf-8"))
        source = runner.read_text(encoding="utf-8")
        self.assertIn("找一下近三年关于开放词汇目标检测的高被引文献，比较它们如何利用语言信息，并保存一份简短比较笔记。", source)
        self.assertIn("inspect_task", source)
        self.assertIn("expected_input_refs_authorized", source)
        self.assertIn("--pdf-a", source)
        self.assertIn("--pdf-b", source)
        self.assertNotIn("必须调用某工具", source)

    def test_old_ag1_failure_evidence_is_not_rewritten(self):
        """有本机原件时核对旧失败；公开克隆不附带私人验收记录。"""
        result = Path(__file__).parents[3] / "artifacts" / "ag1-readiness-review-20261007" / "results.json"
        if not result.is_file():
            self.skipTest("historical local AG1 evidence is not distributed")
        data = json.loads(result.read_text(encoding="utf-8"))
        self.assertTrue(data["probes"]["receipt_before_checkpoint"]["defect_reproduced"])
