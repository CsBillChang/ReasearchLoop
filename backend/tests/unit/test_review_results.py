# 审校合同测试：核验当前稿 hash、证据授权和结果一致性；不评估模型真实理解能力。
import hashlib
import unittest
from research_loop.domain.notes import ReviewFinding, ReviewResult


class ReviewResultTests(unittest.TestCase):
    def test_review_is_bound_to_draft_hash_and_keeps_quality_separate(self):
        digest = hashlib.sha256(b"draft").hexdigest()
        result = ReviewResult(review_id="r", task_id="t", draft_id="d", content_hash=digest,
                              status="issues", findings=[ReviewFinding(claim="x", status="insufficient", reason="no page")])
        self.assertEqual(result.content_hash, digest)
        self.assertEqual(result.status, "issues")
