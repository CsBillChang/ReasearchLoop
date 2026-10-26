"""任务授权范围内的片段定位与本机重排；保留显式混合检索回退。"""
from __future__ import annotations

from collections import OrderedDict
import json
import math
import threading

from research_loop.domain.lexical_retrieval import LexicalRetriever
from research_loop.domain.retrieval import RetrievalQuery, fingerprint
from research_loop.services.indexing import IndexingService
from research_loop.services.papers import PaperService
from research_loop.storage.indexes import rank_terms, reciprocal_rank_fusion
from research_loop.storage.vector_index import point_id


def error_code(exc: Exception) -> str:
    """只返回稳定故障码，不把HTTP地址、凭据或本机路径作为模型观察。"""
    value = str(exc)
    return value if isinstance(exc, ValueError) and value.startswith(("RERANK_", "INDEX_", "RETRIEVAL_", "EMBEDDING_")) else type(exc).__name__


class RetrievalService:
    """复用不可变原文索引；所有通道从同一份有效页集合开始。"""

    def __init__(self, db, embedding, vectors, reranker, *, route="hybrid_rrf"):
        """注入已配置组件，测试可替换适配器而不改业务路径。"""
        if route not in {"hybrid_rrf", "located_fp16"}:
            raise ValueError("RETRIEVAL_ROUTE_UNSUPPORTED")
        self.db, self.embedding, self.vectors, self.reranker = db, embedding, vectors, reranker
        self.indexing = IndexingService(db, embedding, vectors)
        self.route = route
        self._lexical_cache = OrderedDict()
        self._lexical_lock = threading.RLock()

    def _located_selection(self, request: RetrievalQuery, views: list[dict], records: list[dict],
                           catalog: list[dict]) -> dict:
        """仅缓存两份已核验词频快照；每次查询仍重新核验授权并实时选择候选。"""
        cache_key = fingerprint({"records": records, "catalog": catalog,
            "scope": [{"index_id": view["index_id"], "source": view["manifest"]["source"],
                       "pages": view["manifest"]["pages"]} for view in views]})
        with self._lexical_lock:
            lexical = self._lexical_cache.get(cache_key)
            if lexical is None:
                lexical = LexicalRetriever(records, catalog)
                self._lexical_cache[cache_key] = lexical
                if len(self._lexical_cache) > 2:
                    self._lexical_cache.popitem(last=False)
            else:
                self._lexical_cache.move_to_end(cache_key)
        return lexical.select(request.query, request.document_ids, "located_bm25_rerank")

    def _view(self, task_id: str, document_id: str) -> dict:
        """从当前任务的页授权选择一个确切版本；不扫描其他任务或旧版本凑结果。"""
        pages = self.db.fetchall("SELECT e.evidence_id,e.page_number FROM page_evidence e "
            "JOIN task_evidence t ON t.evidence_id=e.evidence_id WHERE t.task_id=? AND e.document_id=? "
            "ORDER BY e.page_number,e.evidence_id", (task_id, document_id))
        papers = PaperService(self.db, None, None)
        valid = []
        for page in pages:
            try:
                papers.authorized_source(task_id, page['evidence_id'], with_readback=False)
                valid.append(page)
            except PermissionError:
                continue
        pages = valid
        if not pages:
            raise PermissionError("RETRIEVAL_DOCUMENT_NOT_AUTHORIZED: read_paper must first grant page evidence")
        refs = [p["evidence_id"] for p in pages]
        model_hash = fingerprint(self.embedding.identity)
        for row in self.db.fetchall("SELECT index_id FROM evidence_index_builds WHERE document_id=? ORDER BY index_id", (document_id,)):
            build = self.indexing.store.get(row["index_id"])
            m = build["manifest"]
            if (m["model_hash"] == model_hash and m["collection"] == self.vectors.collection(model_hash)
                and m["splitter"] == self.indexing.config.model_dump()
                and {p["evidence_id"] for p in m["pages"]} == set(refs)):
                return self.indexing.authorized_chunks(task_id, row["index_id"])
        built = self.indexing.build(task_id, document_id, refs)
        return self.indexing.authorized_chunks(task_id, built["index_id"])

    def search(self, task_id: str, request: RetrievalQuery, cancelled=lambda: False) -> dict:
        """检索并保留通道证据、原文定位和故障信息；外部检索/生成调用数恒为零。

        Args:
            task_id: 当前Agent实际任务，不能借父任务权限查询。
            request: 工具与服务共用的版本/查询合同。
            cancelled: 任务取消或失去租约时停止本地后续工作。

        Returns:
            ready/degraded/empty结果及命中片段；任何分数都不是事实支持概率。

        Raises:
            PermissionError: 缺少该版本的页授权。
            ValueError: 原文/索引身份损坏，不能用降级掩盖权限或完整性故障。
        """
        def check_cancel():
            """本机计算各边界重读取消状态，已完成索引不删除。"""
            if cancelled():
                raise RuntimeError("RETRIEVAL_CANCELLED")
        check_cancel()
        views = []
        for document_id in request.document_ids:
            check_cancel()
            views.append(self._view(task_id, document_id))
        records, by_key, points, bags, catalog = [], {}, {}, [], []
        for view in views:
            if view["bm25_status"] != "ready":
                raise ValueError("RETRIEVAL_BM25_NOT_READY")
            # 来源标签用于展示，不再从冻结manifest恢复过期附件别名；授权和版本
            # 仍由authorized_chunks按task_evidence、document及原件hash逐次验证。
            paper_id = self.db.fetchone("SELECT paper_id FROM documents WHERE document_id=?",
                                       (view["document_id"],))["paper_id"]
            if self.route == "located_fp16":
                metadata_row = self.db.fetchone("SELECT metadata_json FROM papers WHERE paper_id=?", (paper_id,))
                try:
                    metadata = json.loads(metadata_row["metadata_json"]) if metadata_row else {}
                except (TypeError, ValueError) as exc:
                    raise ValueError("RETRIEVAL_CATALOG_INVALID") from exc
                if not isinstance(metadata, dict):
                    raise ValueError("RETRIEVAL_CATALOG_INVALID")
                title = metadata.get("title")
                if title is None:
                    title = ""
                if not isinstance(title, str):
                    raise ValueError("RETRIEVAL_CATALOG_INVALID")
                catalog.append({"document_id": view["document_id"], "title": title})
            for item in view["chunks"]:
                c = item["chunk"]
                key = point_id(view["index_id"], c["chunk_id"])
                row = {**c, "key": key, "index_id": view["index_id"],
                    "paper_id": paper_id}
                records.append(row)
                by_key[key] = row
                bags.append(item["terms"])
                points[key] = self.indexing._point(view["index_id"], view["manifest"],
                    {"chunk": c, "chunk_id": c["chunk_id"], "embedding": None})["payload"]
        vector_rows, reasons = [], []
        selection = None
        if self.route == "located_fp16":
            check_cancel()
            selection = self._located_selection(request, views, records, catalog)
            effective_limit = min(request.candidate_limit, 24)
            bm25 = selection["candidates"][:effective_limit]
            vector_state = "not_used"
            fused = [{"key": row["key"], "candidate_rank": n, "reranker_score": None,
                "channels": {"located_bm25": {"rank": n, "score": row["score"]}}}
                for n, row in enumerate(bm25, 1)]
        else:
            effective_limit = request.candidate_limit
            bm25 = rank_terms(request.query, records, bags, effective_limit)
            vector_state = "ready"
            check_cancel()
            try:
                if any(v["vector_status"] != "ready" for v in views):
                    raise ValueError("RETRIEVAL_VECTOR_INDEX_PARTIAL")
                query_vector = self.embedding.encode([request.query], query=True)[0]
                for view in views:
                    m = view["manifest"]
                    if self.vectors.count(m["collection"], view["index_id"]) != len(view["chunks"]):
                        raise ValueError("RETRIEVAL_VECTOR_COVERAGE_MISMATCH")
                    ids = [point_id(view["index_id"], c["chunk"]["chunk_id"]) for c in view["chunks"]]
                    hits = self.vectors.query(m["collection"], query_vector, ids, effective_limit)
                    if len(hits) != min(len(ids), effective_limit) or len({str(h["id"]) for h in hits}) != len(hits):
                        raise ValueError("RETRIEVAL_VECTOR_COVERAGE_MISMATCH")
                    for h in hits:
                        key = str(h["id"])
                        if key not in ids or h.get("payload") != points[key] or not math.isfinite(h["score"]):
                            raise ValueError("RETRIEVAL_VECTOR_IDENTITY_MISMATCH")
                        vector_rows.append({"key": key, "score": h["score"]})
                vector_rows = sorted(vector_rows, key=lambda r: (-r["score"], r["key"]))[:effective_limit]
            except Exception as exc:
                # 任一版本故障则整条向量通道降级，避免只有部分论文获向量加分。
                vector_rows, vector_state = [], "unavailable"
                reasons.append({"component": "vector", "code": error_code(exc)})
            check_cancel()
            fused = reciprocal_rank_fusion({"bm25": bm25, "vector": vector_rows})[:effective_limit]
            for n, row in enumerate(fused, 1):
                row.update(rrf_rank=n, reranker_score=None)
        check_cancel()
        rerank_state = "empty"
        if fused:
            try:
                scores = self.reranker.score(request.query, [by_key[r["key"]]["text"] for r in fused])
                if len(scores) != len(fused) or not all(math.isfinite(s) for s in scores):
                    raise ValueError("RERANK_OUTPUT_INVALID")
                for row, score in zip(fused, scores):
                    row["reranker_score"] = score
                fused.sort(key=lambda r: (-r["reranker_score"],
                    r["candidate_rank"] if self.route == "located_fp16" else r["rrf_rank"]))
                rerank_state = "ready"
            except Exception as exc:
                reasons.append({"component": "reranker", "code": error_code(exc)})
                rerank_state = "unavailable"
        check_cancel()
        # 返回前再次验证当前权限、原文与不可变索引，避免取消/范围变化后的陈旧交付。
        for view in views:
            self.indexing.authorized_chunks(task_id, view["index_id"])
        hits = [{**by_key[row["key"]], **row, "rank": n} for n, row in enumerate(fused[:request.limit], 1)]
        papers = PaperService(self.db, None, None)
        for hit in hits:
            source = papers.validate_source_observation(task_id, hit, {}, kind='hit')
            hit['readback'] = source['readback']
            hit['target_ref'] = source['readback']['target_ref']
        result = {"status": "degraded" if reasons else "ready" if hits else "empty",
            "query": request.query, "hits": hits,
            "evidence_ids": list(dict.fromkeys(h["evidence_id"] for h in hits)),
            "scope": [{"index_id": v["index_id"], "source": v["manifest"]["source"],
                "pages": v["manifest"]["pages"], "chunk_count": len(v["chunks"])} for v in views],
            "channels": {"bm25": {"status": "ready", "hit_count": len(bm25)},
                "vector": {"status": vector_state, "hit_count": len(vector_rows)},
                "reranker": {"status": rerank_state, "candidate_count": len(fused)}},
            "embedding": self.embedding.identity, "reranker": self.reranker.identity,
            "rrf_k": None if self.route == "located_fp16" else 60,
            "candidate_limit": effective_limit, "degradations": reasons,
            "coverage": "retrieved_chunks_only", "factual_support": "not_assessed",
            "notice": "分数仅为相关性，不是事实支持概率。命中不表示全文已读；无支持时补读原文或报告缺证，不能根据未命中推断全文不存在。"}
        if selection is not None:
            result.update(strategy={"name": self.route, "candidate_selection": "located_bm25_rerank",
                "local_quota": 18, "original_global_reserve": 6,
                "reranker_input_policy": "original_query_and_original_text"},
                requested_candidate_limit=request.candidate_limit, effective_candidate_limit=effective_limit,
                selection_trace={key: selection["trace"][key] for key in
                    ("routing", "expanded_terms")},
                effective_path="located_bm25_rerank" if rerank_state == "ready" else "located_bm25")
        return result


class RetrievalRuntime:
    """Worker共享懒加载入口，串行本机模型计算以控制8GB机器上的峰值内存。"""

    def __init__(self, db, settings):
        """只记录配置，不在API启动或旧任务恢复时下载/加载模型。"""
        self.db, self.settings, self.service = db, settings, None
        self.lock = threading.RLock()

    def search(self, task_id: str, request: RetrievalQuery, cancelled=lambda: False) -> dict:
        """按需启用真实组件；缺依赖时给出可见降级，不伪装已完成向量检索。"""
        with self.lock:
            if cancelled():
                raise RuntimeError("RETRIEVAL_CANCELLED")
            if not self.settings.retrieval_enabled:
                return {"status": "unavailable", "hits": [], "degradations": [{"code": "RETRIEVAL_DISABLED"}]}
            if self.service is None:
                try:
                    from research_loop.providers.embeddings import LocalEmbedding
                    from research_loop.storage.vector_index import VectorIndex
                    manifest = self.settings.retrieval_model_manifest or self.settings.project_root/"data/ag71-preflight/models/manifest.json"
                    embedding = LocalEmbedding(manifest)
                    backend = getattr(self.settings, "retrieval_reranker_backend", "mps_fp16")
                    if backend == "mps_fp16":
                        from research_loop.providers.mps_reranker import MpsReranker
                        reranker = MpsReranker(manifest)
                    elif backend == "onnx_cpu":
                        from research_loop.providers.reranker import LocalReranker
                        reranker = LocalReranker(manifest, threads=4, batch_size=4, sort_by_length=False)
                    else:
                        raise ValueError("RETRIEVAL_RERANKER_BACKEND_UNSUPPORTED")
                    self.service = RetrievalService(self.db, embedding,
                        VectorIndex(self.settings.qdrant_url, self.settings.retrieval_namespace), reranker,
                        route=getattr(self.settings, "retrieval_route", "located_fp16"))
                except (ImportError, FileNotFoundError) as exc:
                    return {"status": "unavailable", "hits": [], "degradations": [{"component": "local_models", "code": type(exc).__name__}],
                        "notice": "本机检索依赖或固定模型未就绪；可继续用read_paper直接读取，不冒充向量检索。"}
            return self.service.search(task_id, request, cancelled)
