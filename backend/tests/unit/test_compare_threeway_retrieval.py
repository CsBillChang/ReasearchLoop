"""完整检索实验的预算、越界与失败留证合同；不导入学习模型。"""
from copy import deepcopy
import hashlib
import time
import unittest
from unittest.mock import Mock, patch

from scripts.compare_threeway_retrieval import (
    ROUTES, CellStopped, build_plan, project_selection, run_one,
)
from scripts.compare_reranker_mps import new_counters
from scripts.retrieval_experiment_guard import BoundaryViolation


def candidate(cid="chunk", document="doc"):
    text = "fixed source"
    return {"chunk_id": cid, "document_id": document, "index_id": "index",
            "source_sha256": "a" * 64, "evidence_id": "evidence", "page_number": 1,
            "char_start": 0, "char_end": len(text), "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "score": 1.0}


class PlanTests(unittest.TestCase):
    def fixtures(self):
        queries = [{"case_id": str(i)} for i in range(114)]
        pairs = [24] * 114
        pairs[1:7], pairs[7:15], pairs[-1] = [20] * 6, [23] * 8, 1
        feeds = {str(i): {"pairs": n, "batches": [{}] * ((n + 3) // 4)}
                 for i, n in enumerate(pairs)}
        return queries, feeds

    def test_exact_budget_and_balanced_hot_order(self):
        queries, feeds = self.fixtures()
        plan = build_plan(queries, feeds)
        self.assertEqual(len(plan["cells"]), 1029)
        self.assertEqual(plan["budget"]["max_pairs"], 8067)
        self.assertEqual(plan["budget"]["max_native_batches"], 2025)
        first_positions = {r: 0 for r in ROUTES}
        hot = plan["cells"][3:]
        for i in range(0, len(hot), 3):
            self.assertEqual({c["route"] for c in hot[i:i + 3]}, set(ROUTES))
            first_positions[hot[i]["route"]] += 1
        self.assertEqual(set(first_positions.values()), {114})

    def test_duplicate_or_missing_scope_plan_rejected(self):
        queries, feeds = self.fixtures()
        queries[-1]["case_id"] = queries[0]["case_id"]
        with self.assertRaises(BoundaryViolation):
            build_plan(queries, feeds)

    def test_changed_native_budget_rejected(self):
        queries, feeds = self.fixtures()
        feeds["0"]["pairs"] = 23
        with self.assertRaises(BoundaryViolation):
            build_plan(queries, feeds)


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.q = {"case_id": "case", "query": "query", "scope_document_ids": ["doc"]}
        self.rows = [candidate()]
        self.lexical = Mock()
        self.lexical.select.return_value = {"candidates": self.rows, "trace": {}}
        self.case = project_selection(self.q, {"candidates": self.rows})
        self.cell = {"phase": "hot", "repetition": 0, "ordinal": 0,
                     "case_id": "case", "route": "bm25", "state": "not_executed"}
        self.counters = new_counters()
        self.counters.update(query_calls=0, embedding_queries=0)
        self.budget = {"max_query_calls": 1029, "max_seconds": 3600}

    def call(self):
        return run_one(self.q, self.cell, self.lexical, None, None, self.case,
                       {}, {}, self.counters, self.budget, time.perf_counter())

    def test_baseline_returns_full_text_with_live_selection(self):
        result = self.call()
        self.lexical.select.assert_called_once_with("query", ["doc"], "bm25_rerank")
        self.assertEqual(result["ranked_ids"], ["chunk"])
        self.assertEqual(result["hits"][0]["text"], "fixed source")
        self.assertEqual(result["status"], "executed")
        self.assertGreater(result["timing"]["wall_seconds"], 0)
        self.assertEqual(self.counters["score_attempts"], 0)

    def test_empty_baseline_is_preserved(self):
        self.lexical.select.return_value = {"candidates": [], "trace": {}}
        result = self.call()
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["hits"], [])

    def test_scope_leak_is_stopped_and_recorded(self):
        self.lexical.select.return_value = {"candidates": [candidate(document="other")]}
        with self.assertRaises(CellStopped) as stopped:
            self.call()
        self.assertEqual(stopped.exception.record["status"], "stopped")
        self.assertEqual(stopped.exception.record["hits"], [])
        self.assertIn("SCOPE", stopped.exception.record["error"]["reason"])

    def test_text_mismatch_is_stopped(self):
        self.rows[0]["text"] = "changed content"
        with self.assertRaises(CellStopped) as stopped:
            self.call()
        self.assertIn("TEXT", stopped.exception.record["error"]["reason"])

    def test_query_projection_cannot_include_labels(self):
        self.q["gold_ids"] = ["chunk"]
        with self.assertRaises(CellStopped) as stopped:
            self.call()
        self.assertIn("PROJECTION", stopped.exception.record["error"]["reason"])
        self.lexical.select.assert_not_called()

    def test_budget_exhaustion_never_selects_candidates(self):
        self.counters["query_calls"] = 1029
        with self.assertRaises(CellStopped):
            self.call()
        self.lexical.select.assert_not_called()

    def test_model_failure_keeps_started_cell(self):
        self.lexical.select.side_effect = RuntimeError("selection failed")
        with self.assertRaises(CellStopped) as stopped:
            self.call()
        row = stopped.exception.record
        self.assertEqual(row["error"]["reason"], "selection failed")
        self.assertGreater(row["timing"]["wall_seconds"], 0)
        self.assertEqual(row["counters_after"]["query_calls"], 1)

    def mps_engine(self):
        engine = Mock()
        engine.identity = {"variant": "mps_fp16"}
        engine.session.snapshot.return_value = {
            "complete": False, "expected_batches": 1, "verified_batches": 0,
            "session_run_attempts": 0, "session_run_successes": 0, "batches": []}
        engine.last_stats = {"old": "must disappear"}
        return engine

    def test_selection_failure_clears_prior_model_proof(self):
        self.cell["route"] = "located_fp16"
        self.lexical.select.side_effect = RuntimeError("selection failed")
        engine = self.mps_engine()
        with self.assertRaises(CellStopped) as stopped:
            run_one(self.q, self.cell, self.lexical, None, engine, self.case,
                    {"sentinel": 1}, {}, self.counters, self.budget, time.perf_counter())
        engine.session.reset.assert_called_once_with({"sentinel": 1})
        self.assertEqual(stopped.exception.record["local_reranker_stats"], {})
        self.assertEqual(self.counters["score_attempts"], 0)

    def test_failed_score_counts_actual_attempt_without_success(self):
        self.cell["route"] = "located_fp16"
        engine = self.mps_engine()
        engine.score.side_effect = RuntimeError("native failed")
        self.budget.update(max_score_attempts=343, max_pairs=8067)
        with patch("scripts.compare_threeway_retrieval.validate_feed_projection"):
            with self.assertRaises(CellStopped) as stopped:
                run_one(self.q, self.cell, self.lexical, None, engine, self.case,
                        {}, {}, self.counters, self.budget, time.perf_counter())
        self.assertEqual(stopped.exception.record["error"]["reason"], "native failed")
        self.assertEqual(self.counters["score_attempts"], 1)
        self.assertEqual(self.counters["failed_score_calls"], 1)
        self.assertEqual(self.counters["successful_score_calls"], 0)


if __name__ == "__main__":
    unittest.main()
