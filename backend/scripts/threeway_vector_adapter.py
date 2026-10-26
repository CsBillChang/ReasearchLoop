"""冻结passage向量与真实E5查询的内存精确余弦基线，不连接数据库。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import time
import uuid


MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIMENSION = 384
IDENTITY_FIELDS = ("key", "index_id", "chunk_id", "document_id", "evidence_id",
                   "text_sha256", "source_sha256")


def _hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _finite_unit_vector(value) -> bool:
    return (isinstance(value, list) and len(value) == DIMENSION
            and all(type(x) in (float, int) and math.isfinite(x) for x in value)
            and abs(math.fsum(x * x for x in value) - 1) < .002)


def _array_reference(value) -> dict:
    return {"dtype": "float32", "byte_order": "little", "shape": list(value.shape),
            "bytes": value.nbytes, "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


class FrozenVectorRetriever:
    """只缓存冻结文档矩阵，每次select都真实编码完整原查询。

    这是377同期纯文件基线，未模拟Qdrant传输或逐位数值。只在构造时
    准备passage矩阵；查询模型调用、FP32归一化、范围过滤和排序均在select。
    """

    def __init__(self, chunks: list[dict], vectors: dict, embedding):
        import numpy as np

        if (not isinstance(chunks, list) or not chunks or not isinstance(vectors, dict)
                or vectors.get("schema") != "threeway-vector-corpus-v1"
                or vectors.get("dimension") != DIMENSION or vectors.get("metric") != "cosine"
                or vectors.get("embedding_threads") != 2
                or vectors.get("embedding_inter_op_threads") != 1):
            raise ValueError("THREEWAY_VECTOR_CORPUS_CONTRACT_CHANGED")
        identity = vectors.get("embedding_identity", {})
        if (not isinstance(identity, dict) or identity.get("model_id") != MODEL_ID
                or identity.get("revision") != MODEL_REVISION
                or identity.get("dimension") != DIMENSION
                or identity.get("max_tokens") != 512 or identity.get("pooling") != "masked_mean_l2"
                or identity.get("prefixes") != {"document": "passage: ", "query": "query: "}
                or not _hash(identity.get("weights_sha256"))
                or not _hash(identity.get("tokenizer_sha256"))):
            raise ValueError("THREEWAY_VECTOR_EMBEDDING_IDENTITY_CHANGED")
        self.embedding, self._embedding_identity = embedding, deepcopy(identity)
        self._check_embedding()
        self._chunks, keys, ids = [], set(), set()
        for raw in chunks:
            if (not isinstance(raw, dict)
                    or any(not isinstance(raw.get(k), str) or not raw[k] for k in IDENTITY_FIELDS)
                    or not _hash(raw.get("source_sha256")) or not _hash(raw.get("text_sha256"))
                    or not isinstance(raw.get("text"), str)
                    or hashlib.sha256(raw["text"].encode()).hexdigest() != raw["text_sha256"]
                    or type(raw.get("char_start")) is not int or type(raw.get("char_end")) is not int
                    or raw["char_start"] < 0 or raw["char_end"] - raw["char_start"] != len(raw["text"])
                    or type(raw.get("page_number")) is not int or raw["page_number"] < 1
                    or raw["chunk_id"] in ids or raw["key"] in keys
                    or raw["key"] != str(uuid.uuid5(uuid.NAMESPACE_URL,
                        "researchloop:" + raw["index_id"] + ":" + raw["chunk_id"]))):
                raise ValueError("THREEWAY_VECTOR_CHUNK_IDENTITY_CHANGED")
            keys.add(raw["key"]);ids.add(raw["chunk_id"])
            self._chunks.append(deepcopy(raw))
        entries = vectors.get("entries")
        if (not isinstance(entries, list) or len(entries) != len(chunks)
                or vectors.get("entry_count") != len(chunks)):
            raise ValueError("THREEWAY_VECTOR_ENTRY_COVERAGE_CHANGED")
        by_id = {}
        for item in entries:
            if (not isinstance(item, dict) or not isinstance(item.get("chunk_id"), str)
                    or item["chunk_id"] in by_id):
                raise ValueError("THREEWAY_VECTOR_DUPLICATE_ENTRY")
            cid = item.get("chunk_id")
            if cid not in ids or not _finite_unit_vector(item.get("vector")):
                raise ValueError("THREEWAY_VECTOR_VALUES_INVALID")
            receipt = item.get("source_receipt", {})
            if (item.get("origin") not in {"local_onnx_encoded", "reused_frozen_snapshot"}
                    or not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str)
                    or not Path(receipt["path"]).is_absolute()
                    or not _hash(receipt.get("sha256"))
                    or type(receipt.get("bytes")) is not int or receipt["bytes"] < 0):
                raise ValueError("THREEWAY_VECTOR_SOURCE_RECEIPT_INVALID")
            by_id[cid] = item
        if set(by_id) != ids:
            raise ValueError("THREEWAY_VECTOR_ENTRY_SET_CHANGED")
        for row in self._chunks:
            if any(by_id[row["chunk_id"]].get(k) != row[k] for k in IDENTITY_FIELDS):
                raise ValueError("THREEWAY_VECTOR_ENTRY_IDENTITY_CHANGED")
        self._matrix = np.asarray([by_id[row["chunk_id"]]["vector"] for row in self._chunks], dtype=np.float32)
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        if (self._matrix.shape != (len(chunks), DIMENSION) or not np.isfinite(self._matrix).all()
                or not np.isfinite(norms).all() or np.any(norms <= 0)):
            raise ValueError("THREEWAY_VECTOR_FP32_MATRIX_INVALID")
        self._matrix = np.ascontiguousarray(self._matrix / norms, dtype=np.float32)
        self._matrix.flags.writeable = False
        self._documents = frozenset(row["document_id"] for row in self._chunks)
        self.identity = {"backend": "numpy_exact_cosine", "public_name": "纯向量（内存精确余弦）",
            "dimension": DIMENSION, "matrix_dtype": "float32", "passage_count": len(chunks),
            "matrix_reference": _array_reference(self._matrix),
            "embedding_identity": deepcopy(self._embedding_identity), "embedding_threads": 2,
            "embedding_inter_op_threads": 1, "query_vector_cache": False, "query_result_cache": False,
            "document_encoding": False, "tie_break": "descending score, ascending frozen point key",
            "boundary": "Same-run file baseline; excludes Qdrant transport/storage and does not claim bitwise Qdrant parity."}
        self.last_stats = {}

    def _check_embedding(self):
        if (getattr(self.embedding, "identity", None) != self._embedding_identity
                or getattr(self.embedding, "threads", None) != 2):
            raise ValueError("THREEWAY_VECTOR_QUERY_ENCODER_CHANGED")

    def _session_identity(self):
        session = getattr(self.embedding, "session", None)
        if session is None:
            raise ValueError("THREEWAY_VECTOR_QUERY_SESSION_MISSING")
        providers, options = session.get_providers(), session.get_session_options()
        if (providers != ["CPUExecutionProvider"] or options.intra_op_num_threads != 2
                or options.inter_op_num_threads != 1):
            raise ValueError("THREEWAY_VECTOR_QUERY_SESSION_CHANGED")
        return {"providers": providers, "intra_op_num_threads": options.intra_op_num_threads,
                "inter_op_num_threads": options.inter_op_num_threads}

    def select(self, query: str, scope: list[str], limit: int = 24) -> dict:
        """实际查询编码后只搜索授权文档，返回至多24个完整原文片段。"""
        import numpy as np

        stats = {"embedding_seconds": 0.0, "vector_search_seconds": 0.0,
            "query_encode_calls": 0, "query_encode_successes": 0,
            "query_vector_cache": False, "query_result_cache": False,
            "matrix_reference": deepcopy(self.identity["matrix_reference"]),
            "candidate_count": 0, "error": None}
        self.last_stats = stats
        try:
            if not isinstance(query, str) or not query.strip() or type(limit) is not int or limit != 24:
                raise ValueError("THREEWAY_VECTOR_QUERY_OR_LIMIT_INVALID")
            if (not isinstance(scope, (list, tuple)) or not scope
                    or any(not isinstance(d, str) or not d for d in scope)
                    or len(set(scope)) != len(scope) or not set(scope) <= self._documents):
                raise ValueError("THREEWAY_VECTOR_DOCUMENT_SCOPE_INVALID")
            self._check_embedding()
            allowed = set(scope)
            indices = [i for i, row in enumerate(self._chunks) if row["document_id"] in allowed]
            if not indices:
                raise ValueError("THREEWAY_VECTOR_SCOPE_EMPTY")
            stats["scope_document_ids"] = list(scope)
            stats["scoped_passage_count"] = len(indices)
            start = time.perf_counter()
            stats["query_encode_calls"] = 1
            try:
                encoded = self.embedding.encode([query], query=True)
                stats["query_encode_successes"] = 1
            finally:
                stats["embedding_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            try:
                self._check_embedding()
                stats["encoder_session"] = self._session_identity()
                if (not isinstance(encoded, list) or len(encoded) != 1
                        or not _finite_unit_vector(encoded[0])):
                    raise ValueError("THREEWAY_VECTOR_QUERY_OUTPUT_INVALID")
                vector = np.asarray(encoded[0], dtype=np.float32)
                norm = np.linalg.norm(vector)
                if vector.shape != (DIMENSION,) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
                    raise ValueError("THREEWAY_VECTOR_FP32_QUERY_INVALID")
                vector = np.ascontiguousarray(vector / norm, dtype=np.float32)
                scores = self._matrix[indices] @ vector
                if scores.shape != (len(indices),) or not np.isfinite(scores).all():
                    raise ValueError("THREEWAY_VECTOR_COSINE_OUTPUT_INVALID")
                ranked = sorted(zip(indices, scores.tolist()), key=lambda value:
                                (-value[1], self._chunks[value[0]]["key"]))[:limit]
                candidates = [{**deepcopy(self._chunks[i]), "vector_score": float(score)}
                              for i, score in ranked]
                stats["query_vector_reference"] = _array_reference(vector)
                stats["candidate_count"] = len(candidates)
                stats["ranked_point_keys"] = [row["key"] for row in candidates]
            finally:
                stats["vector_search_seconds"] = time.perf_counter() - start
            return {"route": "vector", "candidates": candidates, "trace": deepcopy(stats)}
        except BaseException as exc:
            stats["error"] = {"type": type(exc).__name__, "reason": str(exc)}
            raise
