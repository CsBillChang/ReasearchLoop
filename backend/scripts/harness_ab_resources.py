"""为隔离 Harness 对照复制已有索引收据，保留原生权限和检索实现。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import httpx

from research_loop.domain.retrieval import SourceChunk, fingerprint
from research_loop.services.indexing import IndexingService
from research_loop.services.retrieval import RetrievalRuntime
from research_loop.storage.evidence_index import EvidenceIndexStore
from research_loop.storage.indexes import terms
from research_loop.storage.vector_index import VectorIndex, matches, valid_vector


def _digest(path: str | Path) -> str:
    """核对明确指定的来源文件，不载入日常配置。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_bundle(bundle: dict) -> dict:
    """逐片段验证已有来源、文字窗口、词项和向量身份，不重新编码。"""
    manifest = bundle['manifest']
    if fingerprint(manifest) != bundle['index_id']:
        raise ValueError('HARNESS_VECTOR_MANIFEST_CHANGED')
    if fingerprint(manifest['model']) != manifest['model_hash']:
        raise ValueError('HARNESS_VECTOR_MODEL_CHANGED')
    if IndexingService._page_manifest(bundle['pages']) != manifest['pages']:
        raise ValueError('HARNESS_VECTOR_PAGE_IDENTITY_CHANGED')
    pages = {page['evidence_id']: page for page in bundle['pages']}
    if len(pages) != len(bundle['pages']):
        raise ValueError('HARNESS_VECTOR_DUPLICATE_PAGE')
    members = bundle['members']
    if [member['chunk_id'] for member in members] != manifest['chunk_ids']:
        raise ValueError('HARNESS_VECTOR_MEMBER_ORDER_CHANGED')
    for member in members:
        chunk = member['chunk']
        page = pages[chunk['evidence_id']]
        identity = {'source': manifest['source'], 'evidence_id': chunk['evidence_id'],
            'page_hash': fingerprint(page['text']), 'config': manifest['splitter'],
            'tokenizer': manifest['tokenizer'], 'start': chunk['char_start'], 'end': chunk['char_end']}
        if (fingerprint(identity) != chunk['chunk_id'] or chunk['chunk_id'] != member['chunk_id']
                or chunk['document_id'] != manifest['source']['document_id']
                or chunk['source_sha256'] != manifest['source']['content_sha256']
                or chunk['page_number'] != page['page_number']
                or chunk['text'] != page['text'][chunk['char_start']:chunk['char_end']]
                or fingerprint(chunk['text']) != chunk['text_sha256']
                or member['terms'] != Counter(terms(chunk['text']))
                or member['token_count'] != sum(member['terms'].values())
                or not valid_vector(member['embedding'], manifest['model']['dimension'])):
            raise ValueError('HARNESS_VECTOR_EXACT_WINDOW_CHANGED')
    for page in pages.values():
        covered = [False] * len(page['text'])
        for member in members:
            chunk = member['chunk']
            if chunk['evidence_id'] == page['evidence_id']:
                covered[chunk['char_start']:chunk['char_end']] = [True] * (chunk['char_end'] - chunk['char_start'])
        if any(not mark and not char.isspace() for mark, char in zip(covered, page['text'])):
            raise ValueError('HARNESS_VECTOR_TEXT_COVERAGE_GAP')
    return {'pages': len(pages), 'chunks': len(members), 'dimension': manifest['model']['dimension']}


def load_source_bundles(project_root: str | Path, manifest_path: str | Path, *,
                        source_sha256s: list[str] | set[str], policy: str = 'B') -> dict[str, dict]:
    """读取明确冻结的既有收据，拒绝缺失、歧义、漂移和项目外路径。

    Args:
        project_root: 既有本机材料所属项目，仅作只读路径边界。
        manifest_path: 列有 resource_path/resource_sha256 的已有清单。
        source_sha256s: 本次允许复制的 PDF 内容身份，不按目录扩大语料。
        policy: 来源清单的已有策略标签，不表示本次 H1 实验分组。

    Returns:
        按 PDF 内容 hash 映射的完整已有向量资源包。
    """
    root = Path(project_root).resolve()
    path = Path(manifest_path).resolve()
    if not path.is_relative_to(root):
        raise PermissionError('HARNESS_SOURCE_MANIFEST_OUTSIDE_PROJECT')
    wanted = set(source_sha256s)
    if not wanted:
        raise ValueError('HARNESS_SOURCE_SELECTION_EMPTY')
    selected = {}
    manifest = json.loads(path.read_text())
    for entry in manifest['resources']:
        if entry['policy'] != policy:
            continue
        resource_path = Path(entry['resource_path']).resolve()
        if not resource_path.is_relative_to(root):
            raise PermissionError('HARNESS_SOURCE_RESOURCE_OUTSIDE_PROJECT')
        if _digest(resource_path) != entry['resource_sha256']:
            raise ValueError('HARNESS_SOURCE_RESOURCE_CHANGED')
        bundle = json.loads(resource_path.read_text())
        source_hash = bundle['manifest']['source']['content_sha256']
        if source_hash not in wanted:
            continue
        if source_hash in selected:
            raise ValueError('HARNESS_SOURCE_RESOURCE_AMBIGUOUS')
        validate_bundle(bundle)
        document = bundle['document_ref']
        for key, hash_key in [('pdf_path', 'pdf_sha256'), ('parsed_path', 'parsed_cache_sha256')]:
            original = Path(document[key]).resolve()
            if not original.is_relative_to(root) or _digest(original) != document[hash_key]:
                raise ValueError('HARNESS_ORIGINAL_MATERIAL_CHANGED')
        parsed = json.loads(Path(document['parsed_path']).read_text())
        if parsed['pages'] != bundle['pages']:
            raise ValueError('HARNESS_PARSED_PAGES_CHANGED')
        selected[source_hash] = bundle
    if set(selected) != wanted:
        raise ValueError('HARNESS_EXACT_SOURCE_RESOURCE_MISSING')
    return selected


def rebind_bundle(bundle: dict, namespace: str) -> dict:
    """把同一文字/向量收据绑定到新格 namespace，不复用旧可写集合。"""
    if not namespace.startswith('researchloop_h1_'):
        raise PermissionError('HARNESS_NEW_NAMESPACE_REQUIRED')
    validate_bundle(bundle)
    rebound = deepcopy(bundle)
    # 使用原生构造器的命名规则；此临时客户端不连接服务。
    with httpx.Client(base_url='http://127.0.0.1:1', trust_env=False) as client:
        vectors = VectorIndex('http://127.0.0.1:1', namespace, client)
        rebound['manifest']['collection'] = vectors.collection(rebound['manifest']['model_hash'])
    rebound['source_index_id'] = bundle['index_id']
    rebound['source_collection'] = bundle['manifest']['collection']
    rebound['namespace'] = namespace
    rebound['index_id'] = fingerprint(rebound['manifest'])
    for member in rebound['members']:
        member['index_id'] = rebound['index_id']
    validate_bundle(rebound)
    return rebound


def prepare_parsed_cache(file_store, document_ref: dict) -> dict:
    """原字节复制许可 PDF/解析缓存到本格数据根，不解析、不联网。"""
    copied = {}
    for source_key, hash_key in [('pdf_path', 'pdf_sha256'), ('parsed_path', 'parsed_cache_sha256')]:
        source = Path(document_ref[source_key])
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != document_ref[hash_key]:
            raise ValueError('HARNESS_CACHE_SOURCE_CHANGED')
        target = (file_store.cache_pdf_path(document_ref['pdf_sha256']) if source_key == 'pdf_path'
                  else file_store.cache_parse_path(source.stem))
        file_store.write_bytes_atomic(target, content)
        if _digest(target) != document_ref[hash_key]:
            raise ValueError('HARNESS_CACHE_COPY_CHANGED')
        copied[source_key] = str(target)
    return copied


def _check_target(db, settings, bundle: dict) -> None:
    """拒绝生产资源及本格根以外的数据库，不从索引恢复读取权限。"""
    url = urlparse(settings.qdrant_url)
    if (url.hostname != '127.0.0.1' or url.port in {None, 33316, 6339, 6351, 6357, 6361}
            or url.scheme != 'http' or url.username or url.password
            or not settings.retrieval_namespace.startswith('researchloop_h1_')
            or settings.retrieval_namespace != bundle['namespace']):
        raise PermissionError('HARNESS_EXPERIMENT_STORAGE_REQUIRED')
    if (db.kind != 'sqlite' or not Path(db.sqlite_path).resolve().is_relative_to(settings.project_root.resolve())):
        raise PermissionError('HARNESS_CELL_SQLITE_REQUIRED')
    expected = settings.retrieval_namespace + '_' + bundle['manifest']['model_hash']
    if bundle['manifest']['collection'] != expected:
        raise ValueError('HARNESS_COLLECTION_SCOPE_CHANGED')


def _install_metadata(db, bundle: dict) -> None:
    """安装原生索引事实，不创建 documents、pages 或 task_evidence 授权。"""
    store = EvidenceIndexStore(db)
    store.prepare(bundle['index_id'], bundle['manifest'],
        [SourceChunk.model_validate(member['chunk']) for member in bundle['members']], terms)
    for member in bundle['members']:
        store.vector_receipt(bundle['index_id'], member['chunk_id'], member['embedding'])
    store.state(bundle['index_id'], 'ready')


def install_bundle(db, settings, bundle: dict, client: httpx.Client | None = None) -> dict:
    """向本格集合复制既有点并逐点读回，然后安装无授权的 SQL 元数据。"""
    counts = validate_bundle(bundle)
    _check_target(db, settings, bundle)
    own_client = client is None
    client = client or httpx.Client(base_url=settings.qdrant_url, timeout=30, trust_env=False)
    try:
        vectors = VectorIndex(settings.qdrant_url, settings.retrieval_namespace, client)
        collection = bundle['manifest']['collection']
        vectors.ensure_collection(collection, counts['dimension'])
        points = [IndexingService._point(bundle['index_id'], bundle['manifest'], member) for member in bundle['members']]
        for start in range(0, len(points), 32):
            vectors.upsert(collection, points[start:start + 32])
        actual = vectors.retrieve(collection, [point['id'] for point in points])
        if (len(actual) != len(points) or not all(matches(actual.get(point['id']), point) for point in points)
                or vectors.count(collection, bundle['index_id']) != len(points)):
            raise ValueError('HARNESS_VECTOR_READBACK_CHANGED')
        _install_metadata(db, bundle)
        error = max((abs(a - b) for point in points
            for a, b in zip(actual[point['id']]['vector'], point['vector'])), default=0)
        return {**counts, 'index_id': bundle['index_id'], 'collection': collection,
            'points': len(points), 'max_abs_error': error, 'task_grants_created': False,
            'new_document_encode_calls': 0}
    finally:
        if own_client:
            client.close()


class _ReceiptEmbedding:
    """只提供授权/版本校验需要的模型身份，禁止以旧收据冒充新编码。"""

    def __init__(self, manifest):
        self.identity = manifest['model']
        self.tokenizer_id = manifest['tokenizer']
        self.dimension = manifest['model']['dimension']

    def encode(self, *args, **kwargs):
        raise PermissionError('HARNESS_NEW_DOCUMENT_ENCODING_FORBIDDEN')


class HarnessRetrievalRuntime(RetrievalRuntime):
    """在原生检索前按实际已读权限复用既有片段，不重算文档向量。"""

    def __init__(self, db, settings, bundles: list[dict] | dict[str, dict], *, client_factory=None):
        super().__init__(db, settings)
        values = bundles.values() if isinstance(bundles, dict) else bundles
        self.bundles = {bundle['manifest']['source']['document_id']: bundle for bundle in values}
        if not self.bundles:
            raise ValueError('HARNESS_RUNTIME_RESOURCES_EMPTY')
        for bundle in self.bundles.values():
            validate_bundle(bundle)
            _check_target(db, settings, bundle)
        self.client_factory = client_factory

    def ensure_task_resources(self, task_id: str, document_ids: list[str]) -> list[dict]:
        """派生本任务当前合法页集合的收据；未读、换版、越权立即拒绝。"""
        if not document_ids or len(set(document_ids)) != len(document_ids):
            raise ValueError('HARNESS_RETRIEVAL_DOCUMENTS_INVALID')
        if any(document_id not in self.bundles for document_id in document_ids):
            raise PermissionError('HARNESS_RETRIEVAL_DOCUMENT_OUTSIDE_CELL')
        results = []
        with self.lock:
            for document_id in document_ids:
                bundle = self.bundles[document_id]
                rows = self.db.fetchall('SELECT e.evidence_id,e.page_number FROM page_evidence e '
                    'JOIN task_evidence t ON t.evidence_id=e.evidence_id WHERE t.task_id=? '
                    'AND e.document_id=? ORDER BY e.page_number,e.evidence_id', (task_id, document_id))
                if not rows:
                    raise PermissionError('RETRIEVAL_DOCUMENT_NOT_AUTHORIZED: read_paper must first grant page evidence')
                client = (self.client_factory() if self.client_factory else
                    httpx.Client(base_url=self.settings.qdrant_url, timeout=30, trust_env=False))
                try:
                    vectors = VectorIndex(self.settings.qdrant_url, self.settings.retrieval_namespace, client)
                    indexing = IndexingService(self.db, _ReceiptEmbedding(bundle['manifest']), vectors)
                    refs = [row['evidence_id'] for row in rows]
                    source, pages = indexing._source_pages(task_id, document_id, refs)
                    if source != bundle['manifest']['source']:
                        raise ValueError('INDEX_SOURCE_VERSION_CHANGED')
                    by_page = {page['evidence_id']: page for page in bundle['pages']}
                    if any(page['evidence_id'] not in by_page or any(page[key] != by_page[page['evidence_id']][key]
                            for key in ('document_id', 'page_number', 'text')) for page in pages):
                        raise ValueError('INDEX_SOURCE_VERSION_CHANGED')
                    subset = deepcopy(bundle)
                    subset['pages'] = [by_page[ref] for ref in refs]
                    subset['members'] = [member for member in subset['members'] if member['chunk']['evidence_id'] in refs]
                    subset['manifest']['pages'] = IndexingService._page_manifest(subset['pages'])
                    subset['manifest']['chunk_ids'] = [member['chunk_id'] for member in subset['members']]
                    subset['index_id'] = fingerprint(subset['manifest'])
                    for member in subset['members']:
                        member['index_id'] = subset['index_id']
                    result = install_bundle(self.db, self.settings, subset, client)
                    indexing.authorized_chunks(task_id, subset['index_id'])
                    results.append({**result, 'new_document_encode_calls': 0})
                finally:
                    client.close()
        return results

    def search(self, task_id, request, cancelled=lambda: False):
        """权限资源准备后交回原生 BM25/查询编码/向量/RRF/重排实现。"""
        if cancelled():
            raise RuntimeError('RETRIEVAL_CANCELLED')
        self.ensure_task_resources(task_id, request.document_ids)
        return super().search(task_id, request, cancelled)
