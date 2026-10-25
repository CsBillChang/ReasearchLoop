"""用可手算样例核对评价分母、截断和时延插值。"""
import unittest

from scripts.report_lexical_retrieval import percentile, quality


class LexicalReportTests(unittest.TestCase):
    def test_recall_counts_all_gold_instead_of_any_hit(self):
        result = quality(["a", "x", "y"], ["a", "x", "y"], {"a", "b"})
        self.assertEqual(result["recall5"], .5)
        self.assertEqual(result["candidate_recall24"], .5)
        self.assertEqual(result["mrr24"], 1)

    def test_eighth_and_ninth_positions_have_different_delivery_coverage(self):
        ids = [f"x{i}" for i in range(7)] + ["a", "b"]
        result = quality(ids, ids, {"a", "b"})
        self.assertEqual(result["recall5"], 0)
        self.assertEqual(result["recall8"], .5)
        self.assertEqual(result["recall10"], 1)
        self.assertEqual(result["mrr24"], 1/8)

    def test_unanswerable_is_unscored_but_answerable_miss_is_zero(self):
        self.assertIsNone(quality([], [], set())["mrr24"])
        self.assertEqual(quality([], [], {"a"})["mrr24"], 0)

    def test_time_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([1, 2, 3, 4], .5), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], .95), 3.85)
        self.assertIsNone(percentile([], .5))


if __name__ == "__main__":
    unittest.main()
