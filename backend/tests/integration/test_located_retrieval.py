"""片段定位正式入口的离线集成合同；SQLite与模型替身不代表真实效果。"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_loop.config import Settings
from research_loop.domain.retrieval import RetrievalQuery
from research_loop.services.retrieval import RetrievalRuntime, RetrievalService
from research_loop.storage.mysql import Database
from tests.unit.test_hybrid_retrieval import QueryEmbedding, QueryVector
from tests.unit.test_retrieval_contracts import seed_document


class RecordingReranker:
    """记录确切query/原文输入并按可控logit排序，无真实模型。"""

    identity = {"model_id": "synthetic-only", "backend": "pytorch_mps", "dtype": "torch.float16"}

    def __init__(self):
        self.calls = []
        self.failure = None
        self.after_score = None

    def score(self, query, passages):
        """使用同分logit保留候选顺序，可注入评分期间权限变化。"""
        self.calls.append((query, list(passages)))
        if self.failure:
            raise ValueError(self.failure)
        if self.after_score:
            self.after_score()
        return [1.0] * len(passages)


class LocatedRetrievalTests(unittest.TestCase):
    """运行真实索引/授权/服务，检索质量与GPU速度另有固定样本证据。"""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db = Database("sqlite:///" + str(self.root / "test.db"))
        self.db.ensure_schema()
        for task in ("owner", "other"):
            self.db.create_task(task, task, task, "阅读全文", "deny", [])
        self.doc, self.pages = self.seed("alpha", "AlphaNet: Training Architecture",
            "Methods\n\nA method for training models.\n\nInference uses learned features.")
        self.embedding, self.vectors, self.reranker = QueryEmbedding(), QueryVector(), RecordingReranker()
        self.service = RetrievalService(self.db, self.embedding, self.vectors, self.reranker,
                                        route="located_fp16")

    def seed(self, key, title, text, *, jina=False):
        """仅把合成目录写入本例临时库，正文经真实splitter与索引验证。"""
        document, pages = seed_document(self.db, self.root, key, text=text, jina=jina)
        self.db.save_paper(document["paper_id"], {"paper_id": document["paper_id"],
            "title": title, "source": "synthetic"})
        return document, pages

    def search(self, query="AlphaNet方法", *, docs=None, task="owner", candidate_limit=24, cancelled=lambda: False):
        return self.service.search(task, RetrievalQuery(query=query,
            document_ids=docs or [self.doc["document_id"]], candidate_limit=candidate_limit, limit=8), cancelled)

    def test_located_route_keeps_original_query_text_and_source_readback(self):
        result = self.search()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["strategy"]["name"], "located_fp16")
        self.assertEqual(result["effective_path"], "located_bm25_rerank")
        self.assertEqual(result["channels"]["vector"]["status"], "not_used")
        self.assertIsNone(result["rrf_k"])
        self.assertEqual(self.reranker.calls[0][0], "AlphaNet方法")
        self.assertIn("method", result["selection_trace"]["expanded_terms"])
        self.assertEqual(result["selection_trace"]["routing"], "explicit_paper")
        for hit in result["hits"]:
            page = self.db.get_page(hit["evidence_id"])
            self.assertEqual(hit["text"], page["text"][hit["char_start"]:hit["char_end"]])
            self.assertIn(hit["text"], self.reranker.calls[0][1])
            self.assertIn("readback", hit)
            self.assertIn("target_ref", hit)
            self.assertNotIn("rrf_rank", hit)

    def test_repeated_hot_query_skips_query_encoding_vectors_and_rrf(self):
        self.search()
        encoded = self.embedding.encoded
        with patch.object(self.embedding, "encode", side_effect=AssertionError("query encoding forbidden")), \
                patch.object(self.vectors, "query", side_effect=AssertionError("vector query forbidden")), \
                patch("research_loop.services.retrieval.reciprocal_rank_fusion", side_effect=AssertionError("RRF forbidden")):
            result = self.search()
        self.assertEqual(self.embedding.encoded, encoded)
        self.assertEqual(len(self.reranker.calls), 2)
        self.assertEqual(result["channels"]["vector"]["status"], "not_used")

    def test_cached_words_never_grant_another_task_scope(self):
        self.search()
        with self.assertRaises(PermissionError):
            self.search(task="other")
        self.assertEqual(len(self.reranker.calls), 1)

    def test_cached_words_never_bypass_revoked_page_grants(self):
        self.search()
        self.db.execute("DELETE FROM task_evidence WHERE task_id=?", ("owner",))
        with self.assertRaises(PermissionError):
            self.search()

    def test_revocation_during_score_prevents_delivery(self):
        self.reranker.after_score = lambda: self.db.execute("DELETE FROM task_evidence WHERE task_id=?", ("owner",))
        with self.assertRaises(PermissionError):
            self.search()

    def test_cached_words_rebuild_for_changed_title(self):
        first = self.search()
        self.db.save_paper(self.doc["paper_id"], {"paper_id": self.doc["paper_id"],
            "title": "BetaNet: A Different Architecture", "source": "synthetic"})
        second = self.search()
        self.assertEqual(first["selection_trace"]["routing"], "explicit_paper")
        self.assertEqual(second["selection_trace"]["routing"], "no_paper_fallback")
        self.assertEqual(len(self.service._lexical_cache), 2)

    def test_word_cache_has_two_snapshots_and_does_not_cache_query_results(self):
        for n in range(3):
            self.db.save_paper(self.doc["paper_id"], {"paper_id": self.doc["paper_id"],
                "title": f"Model{n}: Architecture", "source": "synthetic"})
            self.search(query="method")
        self.assertEqual(len(self.service._lexical_cache), 2)
        self.assertEqual(len(self.reranker.calls), 3)

    def test_scope_shrink_creates_narrow_version_and_cache(self):
        first = self.search(query="method Results")
        self.db.execute("UPDATE tasks SET prompt=? WHERE task_id=?", ("只看第1页", "owner"))
        second = self.search(query="method Results")
        self.assertNotEqual(first["scope"][0]["index_id"], second["scope"][0]["index_id"])
        self.assertTrue(all(hit["page_number"] == 1 for hit in second["hits"]))

    def test_unknown_physical_page_and_empty_title_are_searchable(self):
        document, _ = self.seed("text", "", "A method for training models.", jina=True)
        result = self.search(query="方法", docs=[document["document_id"]])
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["hits"])
        self.assertTrue(all(hit["page_number"] is None for hit in result["hits"]))
        self.assertEqual(result["selection_trace"]["routing"], "no_paper_fallback")

    def test_reranker_failure_keeps_lexical_order_and_reports_degraded(self):
        self.reranker.failure = "RERANK_MPS_UNAVAILABLE"
        result = self.search()
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["effective_path"], "located_bm25")
        self.assertEqual(result["channels"]["reranker"]["status"], "unavailable")
        self.assertEqual(result["degradations"], [{"component": "reranker", "code": "RERANK_MPS_UNAVAILABLE"}])
        self.assertTrue(result["hits"])
        self.assertEqual([hit["candidate_rank"] for hit in result["hits"]],
                         list(range(1, len(result["hits"]) + 1)))

    def test_empty_selection_never_scores_or_fills_with_unrelated_chunks(self):
        result = self.search(query="xyzzy")
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["hits"], [])
        self.assertEqual(self.reranker.calls, [])

    def test_candidate_limit_is_a_visible_upper_bound(self):
        text = "\n\n".join(f"method training candidate {i}" for i in range(32))
        document, _ = self.seed("many", "ManyNet: Candidate Methods", text)
        for requested, effective in ((8, 8), (24, 24), (64, 24)):
            with self.subTest(requested=requested):
                result = self.search(query="method", docs=[document["document_id"]], candidate_limit=requested)
                self.assertEqual(result["requested_candidate_limit"], requested)
                self.assertEqual(result["effective_candidate_limit"], effective)
                self.assertEqual(result["channels"]["reranker"]["candidate_count"], effective)
                self.assertEqual(len(self.reranker.calls[-1][1]), effective)

    def test_cancel_before_index_and_before_score_stops_work(self):
        with self.assertRaisesRegex(RuntimeError, "CANCELLED"):
            self.search(cancelled=lambda: True)
        self.assertEqual(self.embedding.encoded, 0)
        self.assertEqual(self.reranker.calls, [])

    def test_changed_source_cannot_reuse_cached_candidates(self):
        self.search()
        page = self.pages[0]
        self.db.execute("UPDATE page_evidence SET text=? WHERE evidence_id=?", ("tampered", page["evidence_id"]))
        with self.assertRaises(ValueError):
            self.search()
        self.assertEqual(len(self.reranker.calls), 1)

    def test_request_cannot_select_other_document_version(self):
        other, _ = self.seed("beta", "BetaNet: Other Method", "method in a different document")
        result = self.search(query="method", docs=[other["document_id"]])
        self.assertEqual({hit["document_id"] for hit in result["hits"]}, {other["document_id"]})

    def test_unknown_route_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "RETRIEVAL_ROUTE_UNSUPPORTED"):
            RetrievalService(self.db, self.embedding, self.vectors, self.reranker, route="unknown")

    def test_corrupt_catalog_is_rejected_before_scoring(self):
        self.db.save_paper(self.doc["paper_id"], {"title": False})
        with self.assertRaisesRegex(ValueError, "RETRIEVAL_CATALOG_INVALID"):
            self.search()
        self.assertEqual(self.reranker.calls, [])

    def test_runtime_selects_explicit_mps_backend_and_located_route(self):
        settings = Settings(project_root=self.root, data_dir=self.root, output_dir=self.root,
                            retrieval_route="located_fp16", retrieval_reranker_backend="mps_fp16")
        request = RetrievalQuery(query="method", document_ids=[self.doc["document_id"]])
        with patch("research_loop.providers.embeddings.LocalEmbedding", return_value=self.embedding), \
                patch("research_loop.providers.mps_reranker.MpsReranker", return_value=self.reranker) as provider, \
                patch("research_loop.storage.vector_index.VectorIndex", return_value=self.vectors), \
                patch("research_loop.services.retrieval.RetrievalService", return_value=self.service) as service:
            result = RetrievalRuntime(self.db, settings).search("owner", request)
        provider.assert_called_once()
        self.assertEqual(service.call_args.kwargs["route"], "located_fp16")
        self.assertEqual(result["strategy"]["name"], "located_fp16")

    def test_runtime_cpu_rollback_is_explicit_and_uses_four_threads(self):
        settings = Settings(project_root=self.root, data_dir=self.root, output_dir=self.root,
                            retrieval_route="hybrid_rrf", retrieval_reranker_backend="onnx_cpu")
        request = RetrievalQuery(query="method", document_ids=[self.doc["document_id"]])
        hybrid = RetrievalService(self.db, self.embedding, self.vectors, self.reranker)
        with patch("research_loop.providers.embeddings.LocalEmbedding", return_value=self.embedding), \
                patch("research_loop.providers.reranker.LocalReranker", return_value=self.reranker) as provider, \
                patch("research_loop.storage.vector_index.VectorIndex", return_value=self.vectors), \
                patch("research_loop.services.retrieval.RetrievalService", return_value=hybrid) as service:
            RetrievalRuntime(self.db, settings).search("owner", request)
        self.assertEqual(provider.call_args.kwargs, {"threads": 4, "batch_size": 4, "sort_by_length": False})
        self.assertEqual(service.call_args.kwargs["route"], "hybrid_rrf")


if __name__ == "__main__":
    unittest.main()
