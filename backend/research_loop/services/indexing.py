"""已授权原文→不可变片段→持久词项/向量；索引不扩大任务的读取权限。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from research_loop.domain.retrieval import ChunkConfig, fingerprint, split_page
from research_loop.services.papers import PaperService
from research_loop.storage.evidence_index import EvidenceIndexStore
from research_loop.storage.indexes import terms
from research_loop.storage.vector_index import matches, point_id, valid_vector


class IndexingService:
    """构建指定任务的证据快照；AG7.2工具仅为当前已授权页按需建库。"""

    def __init__(self, db, embedding, vectors, config: ChunkConfig | None = None):
        """注入数据库、本地编码和Qdrant；替身仅供测试，不改变业务合同。"""
        self.db, self.embedding, self.vectors = db, embedding, vectors
        self.config = config or ChunkConfig()
        self.store = EvidenceIndexStore(db)

    def _source_pages(self, task_id: str, document_id: str, evidence_ids: list[str]):
        """核对任务、原件hash、解析缓存和最新页级权限后返回数据库正文。

        Raises:
            PermissionError: 任务未读过指定证据，或最新用户范围不允许。
            ValueError: 文档/缓存/页快照身份不一致。
        """
        if not self.db.get_task(task_id) or not evidence_ids or len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("INDEX_INPUT_INVALID")
        document = self.db.fetchone("SELECT * FROM documents WHERE document_id=?", (document_id,))
        if not document:
            raise ValueError("INDEX_DOCUMENT_NOT_FOUND")
        pages = []
        papers = PaperService(self.db, None, None)
        for evidence_id in evidence_ids:
            if not self.db.task_has_evidence(task_id, evidence_id):
                raise PermissionError("INDEX_EVIDENCE_NOT_AUTHORIZED")
            page = papers.authorized_source(task_id, evidence_id, with_readback=False)['page']
            if not page or page["document_id"] != document_id:
                raise ValueError("INDEX_EVIDENCE_DOCUMENT_MISMATCH")
            number = page["page_number"]
            if document["total_pages"] and (number is None or not 1 <= number <= document["total_pages"]):
                raise ValueError("INDEX_PAGE_INVALID")
            if not document["total_pages"] and number is not None:
                raise ValueError("INDEX_UNKNOWN_PAGE_FABRICATED")
            pages.append(page)
        pages.sort(key=lambda p: (p["page_number"] or 0, p["evidence_id"]))
        if len({p["page_number"] for p in pages}) != len(pages):
            raise ValueError("INDEX_DUPLICATE_PAGE")
        cache_path = Path(document["cache_path"])
        cached = json.loads(cache_path.read_text())
        lookup = {p["evidence_id"]: p for p in cached["pages"]}
        for page in pages:
            original = lookup.get(page["evidence_id"], {})
            if any(original.get(k) != page[k] for k in ("document_id", "page_number", "text")):
                raise ValueError("INDEX_PAGE_CACHE_MISMATCH")
        if document["parser_name"] == "jina":
            if len(cached["pages"]) != 1 or fingerprint(cached["pages"][0]["text"]) != document["content_sha256"]:
                raise ValueError("INDEX_SOURCE_HASH_MISMATCH")
        else:
            pdf = cache_path.parent.parent / "pdf" / (document["content_sha256"] + ".pdf")
            if hashlib.sha256(pdf.read_bytes()).hexdigest() != document["content_sha256"]:
                raise ValueError("INDEX_SOURCE_HASH_MISMATCH")
        # paper_id可能是本次上传的临时input_ref；内容版本和页权限不能由附件别名定义。
        source = {k: document[k] for k in ("document_id", "content_sha256",
            "parser_name", "parser_version", "config_sha256", "total_pages")}
        source["hash_kind"] = "extracted_text" if document["parser_name"] == "jina" else "pdf_bytes"
        return source, pages

    @staticmethod
    def _content_source(source: dict) -> dict:
        """兼容旧索引的来源标签，只排除paper_id，不放宽其他身份字段。

        旧manifest、chunk ID及向量payload继续按原字节对应的身份校验，
        此投影仅比较当前原件是否仍是同一内容/解析版本，不授予任务权限。
        """
        return {key: value for key, value in source.items() if key != "paper_id"}

    @staticmethod
    def _page_manifest(pages: list[dict]) -> list[dict]:
        """持久化文字身份和定位；不把缓存绝对路径纳入内容版本。"""
        return [{"evidence_id": p["evidence_id"], "page_number": p["page_number"],
                 "text_sha256": fingerprint(p["text"]), "char_count": len(p["text"])} for p in pages]

    def build(self, task_id: str, document_id: str, evidence_ids: list[str], batch_size: int = 4) -> dict:
        """创建或恢复一个精确授权范围的原文索引。

        Args:
            task_id: 已获页证据的实际任务，不从缓存推导授权。
            document_id: 解析文档身份。
            evidence_ids: 明确列出的已授权页；相同集合的不同顺序复用同一版本。
            batch_size: 本机内存控制，每批最多8个片段。

        Returns:
            索引版本、两个通道状态及本次编码/写入计数。Qdrant失败保留BM25与向量收据。

        Raises:
            ValueError: 前置身份或范围错误；不会创建越界索引。
        """
        if not 1 <= batch_size <= 8:
            raise ValueError("INDEX_BATCH_SIZE_INVALID")
        source, pages = self._source_pages(task_id, document_id, evidence_ids)
        model_hash = fingerprint(self.embedding.identity)
        manifest = {"schema": "original-evidence-v1", "source": source, "pages": self._page_manifest(pages),
            "splitter": self.config.model_dump(), "tokenizer": self.embedding.tokenizer_id,
            "model": self.embedding.identity, "model_hash": model_hash,
            "terms_version": "latin-cjk-bigram-v1", "collection": self.vectors.collection(model_hash)}
        # 显式重建也复用兼容旧索引，避免检索虽恢复却重新编码整篇。
        # 只沿用来源标签；其余配置、页快照、原manifest hash必须全部一致。
        for row in self.db.fetchall("SELECT index_id FROM evidence_index_builds WHERE document_id=? ORDER BY index_id", (document_id,)):
            old = self.store.get(row["index_id"])["manifest"]
            header = {key: value for key, value in old.items() if key != "chunk_ids"}
            header["source"] = self._content_source(old["source"])
            if header == manifest:
                if fingerprint(old) != row["index_id"]:
                    raise ValueError("INDEX_SOURCE_VERSION_CHANGED")
                manifest["source"] = old["source"]
                break
        chunks = [chunk for page in pages for chunk in split_page(page, manifest["source"], self.config,
            self.embedding.tokenizer_id, self.embedding.offsets)]
        if not chunks:
            raise ValueError("INDEX_EMPTY_SOURCE")
        manifest["chunk_ids"] = [c.chunk_id for c in chunks]
        index_id = fingerprint(manifest)
        encoded = written = 0
        with self.store.lock(index_id):
            self.store.prepare(index_id, manifest, chunks, terms)
            # 任何重建先撤销ready，避免服务失败后仍展示上轮状态。
            self.store.state(index_id, "partial")
            try:
                members = self._validated_members(index_id, manifest, pages, check_terms=False)
                self.store.rebuild_terms(index_id, members, terms)
                self.vectors.ensure_collection(manifest["collection"], self.embedding.dimension)
                for start in range(0, len(members), batch_size):
                    batch = members[start:start+batch_size]
                    missing = [m for m in batch if m["embedding"] is None]
                    if missing:
                        outputs = self.embedding.encode([m["chunk"]["text"] for m in missing])
                        if len(outputs) != len(missing) or not all(valid_vector(v, self.embedding.dimension) for v in outputs):
                            raise ValueError("INDEX_EMBEDDING_INVALID")
                        for member, vector in zip(missing, outputs):
                            self.store.vector_receipt(index_id, member["chunk_id"], vector)
                            member["embedding"] = vector
                        encoded += len(missing)
                    expected = [self._point(index_id, manifest, m) for m in batch]
                    actual = self.vectors.retrieve(manifest["collection"], [p["id"] for p in expected])
                    absent = [p for p in expected if not matches(actual.get(p["id"]), p)]
                    if absent:
                        self.vectors.upsert(manifest["collection"], absent)
                        written += len(absent)
                    final = self.vectors.retrieve(manifest["collection"], [p["id"] for p in expected])
                    if not all(matches(final.get(p["id"]), p) for p in expected):
                        raise ValueError("INDEX_VECTOR_READBACK_MISMATCH")
                if self.vectors.count(manifest["collection"], index_id) != len(members):
                    raise ValueError("INDEX_VECTOR_COUNT_MISMATCH")
                # 构建期间权限或原文发生变化时不得晋升ready。
                self.authorized_chunks(task_id, index_id)
                self.store.state(index_id, "ready")
            except Exception as exc:
                # 不记录含密钥/地址的HTTP异常详情；失败位置可由异常类型和合同码定位。
                code = str(exc) if isinstance(exc, ValueError) and str(exc).startswith("INDEX_") else type(exc).__name__
                self.store.state(index_id, "partial", code)
                if code in {"INDEX_CHUNK_SOURCE_MISMATCH", "INDEX_MEMBERS_MISMATCH", "INDEX_TERMS_MISMATCH"}:
                    self.store.bm25_failed(index_id)
        return {**self.store.get(index_id), "chunk_count": len(chunks), "encoded_chunks": encoded,
                "upserted_points": written}

    def _validated_members(self, index_id: str, manifest: dict, pages: list[dict], *, check_terms: bool = True) -> list[dict]:
        """验证SQL词项、精确片段和成员清单，损坏缓存不能冒充BM25就绪。"""
        members = self.store.members(index_id)
        if [m["chunk_id"] for m in members] != manifest["chunk_ids"]:
            raise ValueError("INDEX_MEMBERS_MISMATCH")
        source_pages = {p["evidence_id"]: p for p in pages}
        for m in members:
            c = m["chunk"]
            page = source_pages.get(c["evidence_id"])
            if (not page or c["document_id"] != manifest["source"]["document_id"]
                or c["source_sha256"] != manifest["source"]["content_sha256"]
                or c["chunk_id"] != m["chunk_id"] or c["page_number"] != page["page_number"]
                or not 0 <= c["char_start"] < c["char_end"] <= len(page["text"])
                or c["text"] != page["text"][c["char_start"]:c["char_end"]]
                or fingerprint(c["text"]) != c["text_sha256"]):
                raise ValueError("INDEX_CHUNK_SOURCE_MISMATCH")
            expected_id = fingerprint({"source": manifest["source"], "evidence_id": c["evidence_id"],
                "page_hash": fingerprint(page["text"]), "config": manifest["splitter"],
                "tokenizer": manifest["tokenizer"], "start": c["char_start"], "end": c["char_end"]})
            if expected_id != c["chunk_id"]:
                raise ValueError("INDEX_CHUNK_SOURCE_MISMATCH")
            bag = Counter(terms(c["text"]))
            if check_terms and (m["terms"] != bag or m["token_count"] != sum(bag.values())):
                raise ValueError("INDEX_TERMS_MISMATCH")
            if m["embedding"] is not None and not valid_vector(m["embedding"], manifest["model"]["dimension"]):
                raise ValueError("INDEX_EMBEDDING_RECEIPT_INVALID")
        return members

    @staticmethod
    def _point(index_id: str, manifest: dict, member: dict) -> dict:
        """生成精确payload；全文留在SQL，向量结果通过chunk ID回读授权正文。"""
        chunk = member["chunk"]
        payload = {k: chunk[k] for k in ("chunk_id", "document_id", "evidence_id", "source_sha256",
            "page_number", "char_start", "char_end", "text_sha256")}
        payload.update(index_id=index_id, model_hash=manifest["model_hash"], splitter=manifest["splitter"])
        return {"id": point_id(index_id, member["chunk_id"]), "payload": payload, "vector": member["embedding"]}

    def authorized_chunks(self, task_id: str, index_id: str) -> dict:
        """每次读回重新检查当前权限与源hash，不因索引已ready就跳过。

        Returns:
            原文片段、持久BM25词项、模型身份及通道状态；不包含未经授权的相邻页。
        """
        result = self.store.get(index_id)
        manifest = result["manifest"]
        source, pages = self._source_pages(task_id, result["document_id"],
            [p["evidence_id"] for p in manifest["pages"]])
        if (fingerprint(manifest) != index_id
            or self._content_source(source) != self._content_source(manifest["source"])
            or self._page_manifest(pages) != manifest["pages"]):
            raise ValueError("INDEX_SOURCE_VERSION_CHANGED")
        members = self._validated_members(index_id, manifest, pages)
        result["chunks"] = [{"chunk": m["chunk"], "terms": m["terms"], "token_count": m["token_count"]} for m in members]
        return result
