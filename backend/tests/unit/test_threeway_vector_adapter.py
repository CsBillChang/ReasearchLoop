"""合成向量与替身查询编码器检查纯文件边界，不导入模型或数据库。"""

from copy import deepcopy
import hashlib
from types import SimpleNamespace
import unittest
import uuid

import numpy as np

from scripts import threeway_vector_adapter as adapter


def unit(axis=0):
    values = [0.0] * 384;values[axis] = 1.0
    return values


def identity():
    return {"model_id": adapter.MODEL_ID, "revision": adapter.MODEL_REVISION,
        "dimension": 384, "max_tokens": 512, "pooling": "masked_mean_l2",
        "prefixes": {"document": "passage: ", "query": "query: "},
        "weights_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
        "onnxruntime": "synthetic", "tokenizers": "synthetic"}


def fixture(count=3):
    rows, entries = [], []
    for i in range(count):
        doc = "doc-a" if i % 2 == 0 else "doc-b"
        cid, index = f"synthetic-{i}", "index-" + doc
        text = "synthetic passage " + str(i)
        row = {"key": str(uuid.uuid5(uuid.NAMESPACE_URL, "researchloop:" + index + ":" + cid)),
            "chunk_id": cid, "index_id": index, "document_id": doc,
            "evidence_id": "evidence-" + str(i), "source_sha256": "c" * 64,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text": text, "char_start": 0, "char_end": len(text), "page_number": 1,
            "vector_payload": {"frozen": True}}
        rows.append(row)
        entries.append({**{k: row[k] for k in adapter.IDENTITY_FIELDS},
            "vector": unit(0 if i != 1 else 1), "origin": "local_onnx_encoded",
            "source_receipt": {"path": "/frozen/synthetic-receipt.json", "sha256": "d" * 64, "bytes": 10}})
    corpus = {"schema": "threeway-vector-corpus-v1", "dimension": 384, "metric": "cosine",
        "embedding_identity": identity(), "embedding_threads": 2,
        "embedding_inter_op_threads": 1, "entry_count": count, "entries": entries}
    return rows, corpus


class FakeSession:
    def __init__(self):
        self.providers = ["CPUExecutionProvider"]
        self.options = SimpleNamespace(intra_op_num_threads=2, inter_op_num_threads=1)

    def get_providers(self):return self.providers
    def get_session_options(self):return self.options


class FakeEmbedding:
    def __init__(self, model_identity=None):
        self.identity = deepcopy(model_identity or identity())
        self.threads = 2
        self.session = FakeSession()
        self.calls = []
        self.result = [unit()]
        self.error = None

    def encode(self, values, *, query):
        self.calls.append((list(values), query))
        if self.error:raise self.error
        return deepcopy(self.result)


class FrozenCorpusTests(unittest.TestCase):
    def test_constructor_is_model_free_and_matrix_is_readonly_fp32(self):
        chunks, vectors = fixture();embedding = FakeEmbedding()
        retriever = adapter.FrozenVectorRetriever(chunks, vectors, embedding)
        self.assertEqual(embedding.calls, [])
        self.assertEqual(retriever._matrix.shape, (3, 384))
        self.assertEqual(retriever._matrix.dtype, np.float32)
        self.assertFalse(retriever._matrix.flags.writeable)
        self.assertEqual(retriever.identity["backend"], "numpy_exact_cosine")
        self.assertFalse(retriever.identity["query_vector_cache"])
        self.assertFalse(retriever.identity["document_encoding"])

    def test_entry_order_is_independent_from_chunk_order(self):
        chunks, vectors = fixture();vectors["entries"].reverse()
        retriever = adapter.FrozenVectorRetriever(chunks, vectors, FakeEmbedding())
        result = retriever.select("same query", ["doc-b"])
        self.assertEqual(result["candidates"][0]["chunk_id"], chunks[1]["chunk_id"])
        self.assertEqual(result["candidates"][0]["vector_score"], 0)

    def test_missing_duplicate_extra_or_mismatched_entries_rejected(self):
        for kind in ("missing", "duplicate", "extra", "text", "document", "source", "point"):
            chunks, vectors = fixture()
            if kind == "missing":vectors["entries"].pop()
            elif kind == "extra":vectors["entries"].append(deepcopy(vectors["entries"][0]))
            elif kind == "duplicate":vectors["entries"][1] = deepcopy(vectors["entries"][0])
            else:
                key = {"text": "text_sha256", "document": "document_id", "source": "source_sha256", "point": "key"}[kind]
                vectors["entries"][0][key] = "e" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                adapter.FrozenVectorRetriever(chunks, vectors, FakeEmbedding())

    def test_changed_raw_text_ranges_or_point_identity_rejected(self):
        for field, value in (("text", "changed"), ("char_end", 1), ("page_number", 0),
                             ("key", "wrong-key"), ("source_sha256", "bad-hash")):
            chunks, vectors = fixture();chunks[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapter.FrozenVectorRetriever(chunks, vectors, FakeEmbedding())

    def test_invalid_vector_dimension_nonfinite_zero_and_bool_rejected(self):
        for kind in ("dimension", "nonfinite", "zero", "bool"):
            chunks, vectors = fixture();value = vectors["entries"][0]["vector"]
            if kind == "dimension":value.pop()
            elif kind == "nonfinite":value[0] = float("nan")
            elif kind == "zero":vectors["entries"][0]["vector"] = [0.0] * 384
            else:value[0] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                adapter.FrozenVectorRetriever(chunks, vectors, FakeEmbedding())

    def test_model_prefix_revision_and_threads_must_match_frozen_vectors(self):
        for kind in ("prefix", "revision", "pooling", "threads", "actual_identity"):
            chunks, vectors = fixture();embedding = FakeEmbedding()
            if kind == "prefix":vectors["embedding_identity"]["prefixes"]["query"] = "passage: "
            elif kind == "revision":vectors["embedding_identity"]["revision"] = "other"
            elif kind == "pooling":vectors["embedding_identity"]["pooling"] = "cls"
            elif kind == "threads":embedding.threads = 4
            else:embedding.identity["weights_sha256"] = "e" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                adapter.FrozenVectorRetriever(chunks, vectors, embedding)

    def test_receipt_lineage_must_have_absolute_path_hash_and_bytes(self):
        for field, value in (("path", "relative.json"), ("sha256", "bad"), ("bytes", -1)):
            chunks, vectors = fixture();vectors["entries"][0]["source_receipt"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapter.FrozenVectorRetriever(chunks, vectors, FakeEmbedding())


class SelectTests(unittest.TestCase):
    def build(self, count=3):
        chunks, vectors = fixture(count);embedding = FakeEmbedding()
        return adapter.FrozenVectorRetriever(chunks, vectors, embedding), embedding

    def test_real_query_call_preserves_exact_text_and_query_flag(self):
        retriever, embedding = self.build()
        query = "  原始问题，不重写  "
        result = retriever.select(query, ["doc-a", "doc-b"])
        self.assertEqual(embedding.calls, [([query], True)])
        self.assertEqual(result["trace"]["query_encode_calls"], 1)
        self.assertEqual(result["trace"]["query_encode_successes"], 1)
        self.assertGreaterEqual(result["trace"]["embedding_seconds"], 0)
        self.assertGreaterEqual(result["trace"]["vector_search_seconds"], 0)

    def test_scope_filter_and_deterministic_point_key_tie_break(self):
        retriever, _ = self.build()
        result = retriever.select("q", ["doc-a"])
        rows = result["candidates"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["document_id"] == "doc-a" for row in rows))
        self.assertEqual([row["key"] for row in rows], sorted(row["key"] for row in rows))
        self.assertTrue(all(row["vector_score"] == 1 for row in rows))

    def test_same_query_is_encoded_again_and_result_not_cached(self):
        retriever, embedding = self.build()
        first = retriever.select("q", ["doc-a", "doc-b"])
        embedding.result = [unit(1)]
        second = retriever.select("q", ["doc-a", "doc-b"])
        self.assertEqual(len(embedding.calls), 2)
        self.assertNotEqual(first["candidates"][0]["chunk_id"], second["candidates"][0]["chunk_id"])
        self.assertEqual(retriever.last_stats["query_encode_calls"], 1)

    def test_top24_limit_preserves_full_chunk_identity_and_text(self):
        retriever, _ = self.build(30)
        result = retriever.select("q", ["doc-a", "doc-b"])
        self.assertEqual(len(result["candidates"]), 24)
        for row in result["candidates"]:
            self.assertEqual(hashlib.sha256(row["text"].encode()).hexdigest(), row["text_sha256"])
            self.assertIn("evidence_id", row)
            self.assertIn("vector_payload", row)

    def test_bad_scope_or_limit_rejected_before_encoding(self):
        for query, scope, limit in (("q", [], 24), ("q", ["unknown"], 24),
                ("q", ["doc-a", "doc-a"], 24), ("q", "doc-a", 24),
                ("", ["doc-a"], 24), ("q", ["doc-a"], 10), ("q", ["doc-a"], True)):
            retriever, embedding = self.build()
            with self.subTest(query=query, scope=scope, limit=limit), self.assertRaises(ValueError):
                retriever.select(query, scope, limit)
            self.assertEqual(embedding.calls, [])
            self.assertEqual(retriever.last_stats["query_encode_calls"], 0)

    def test_encoder_failure_keeps_attempt_and_elapsed(self):
        retriever, embedding = self.build();embedding.error = RuntimeError("synthetic encode failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            retriever.select("q", ["doc-a"])
        self.assertEqual(retriever.last_stats["query_encode_calls"], 1)
        self.assertEqual(retriever.last_stats["query_encode_successes"], 0)
        self.assertGreaterEqual(retriever.last_stats["embedding_seconds"], 0)
        self.assertIsNotNone(retriever.last_stats["error"])

    def test_bad_actual_output_shape_values_or_norm_rejected(self):
        for value in ([], [unit(), unit()], [[0] * 384], [[1] * 383], [[float("inf")] * 384]):
            retriever, embedding = self.build();embedding.result = value
            with self.subTest(length=len(value)), self.assertRaises(ValueError):
                retriever.select("q", ["doc-a"])
            self.assertEqual(retriever.last_stats["query_encode_calls"], 1)
            self.assertEqual(retriever.last_stats["candidate_count"], 0)

    def test_actual_session_provider_and_thread_changes_rejected(self):
        for kind in ("provider", "intra", "inter", "missing"):
            retriever, embedding = self.build()
            if kind == "provider":embedding.session.providers = ["CUDAExecutionProvider"]
            elif kind == "intra":embedding.session.options.intra_op_num_threads = 4
            elif kind == "inter":embedding.session.options.inter_op_num_threads = 2
            else:embedding.session = None
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                retriever.select("q", ["doc-a"])

    def test_encoder_identity_change_before_request_is_rejected_without_call(self):
        retriever, embedding = self.build();embedding.identity["revision"] = "changed"
        with self.assertRaises(ValueError):retriever.select("q", ["doc-a"])
        self.assertEqual(embedding.calls, [])

    def test_input_and_result_mutation_cannot_change_frozen_state(self):
        chunks, vectors = fixture();embedding = FakeEmbedding()
        retriever = adapter.FrozenVectorRetriever(chunks, vectors, embedding)
        chunks[0]["text"] = "changed";vectors["entries"][0]["vector"] = unit(1)
        first = retriever.select("q", ["doc-a"])
        first["candidates"][0]["text"] = "changed again"
        second = retriever.select("q", ["doc-a"])
        self.assertNotIn("changed", second["candidates"][0]["text"])
        self.assertTrue(all(row["vector_score"] == 1 for row in second["candidates"]))


if __name__ == "__main__":
    unittest.main()
