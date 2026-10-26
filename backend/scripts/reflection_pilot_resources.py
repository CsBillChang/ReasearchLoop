"""Reflection六稿的同版原文与原向量资源；装配不发送生成或神经请求。

调用方显式传入只读来源与全新运行根；本模块不读取.env，不启动Qdrant，
也不恢复旧任务。只有新存储逐点读回完成后才报告vector_ready。
"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit

import httpx

from research_loop.domain.retrieval import ChunkConfig, SourceChunk, fingerprint, split_page
from research_loop.providers.embeddings import LocalEmbedding
from research_loop.providers.pdf import PDFProvider
from research_loop.services.indexing import IndexingService
from research_loop.services.papers import PaperService
from research_loop.services.retrieval import RetrievalRuntime, RetrievalService
from research_loop.storage.evidence_index import EvidenceIndexStore
from research_loop.storage.files import FileStore
from research_loop.storage.indexes import terms
from research_loop.storage.vector_index import VectorIndex, matches, point_id, valid_vector


class ResourcePending(RuntimeError):
    """资源仍待资格，调用方不能继续真实实验。"""


def file_hash(path: Path) -> str:
    """读取原件内容身份，不修改来源文件。"""
    return sha256(Path(path).read_bytes()).hexdigest()


class ReadOnlyIndex:
    """旧SQLite仅接受SELECT且mode=ro；不能经此适配器写旧向量收据。"""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError('PILOT_SOURCE_INDEX_MISSING')

    def fetchall(self, sql, parameters=()):
        if not sql.lstrip().upper().startswith('SELECT '):
            raise PermissionError('PILOT_SOURCE_INDEX_READ_ONLY')
        with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(sql, parameters)]

    def fetchone(self, sql, parameters=()):
        rows = self.fetchall(sql, parameters)
        return rows[0] if rows else None


def validate_case_input(case: dict) -> tuple[dict, dict]:
    """只消费中性model_input，禁止标签键或超出已批准单页范围。"""
    identity = case.get('case_id')
    allowed = {'R01': 6, 'R02': 4, 'R03': 5, 'R04': 4, 'R05': 10, 'R06': 5}
    if identity not in allowed:
        raise ValueError('PILOT_CASE_NOT_APPROVED')
    model = case.get('model_input')
    if not isinstance(model, dict):
        raise ValueError('PILOT_EXPLICIT_MODEL_INPUT_REQUIRED')

    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {'category', 'gold', 'expected', 'expected_text', 'expected_handling', 'claims', 'before_status'}:
                    raise ValueError('PILOT_LABEL_FIELD_IN_MODEL_INPUT')
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(model)
    if not isinstance(model.get('question'), str) or not isinstance(model.get('draft'), str):
        raise ValueError('PILOT_QUESTION_AND_DRAFT_REQUIRED')
    if len(model.get('documents', [])) != 1:
        raise ValueError('PILOT_SINGLE_DOCUMENT_REQUIRED')
    document = model['documents'][0]
    if document.get('allowed_pages') != [allowed[identity]]:
        raise PermissionError('PILOT_APPROVED_PAGE_RANGE_CHANGED')
    if [page.get('physical_page') for page in document.get('initial_pages', [])] != [allowed[identity]]:
        raise ValueError('PILOT_INITIAL_PAGE_MISSING')
    for page in document['initial_pages']:
        if fingerprint(page['text']) != page['page_text_sha256']:
            raise ValueError('PILOT_INPUT_PAGE_HASH_CHANGED')
    return model, document


def check_endpoint(endpoint: str | None) -> None:
    """仅允许显式新本机端口，拒绝部署与已存在历史实验端口。"""
    if endpoint is None:
        return
    value = urlsplit(endpoint)
    if (value.scheme != 'http' or value.hostname != '127.0.0.1' or value.username or value.password
            or not value.port or value.port in {6333, 6339, 6351, 6357}
            or value.path not in {'', '/'} or value.query or value.fragment):
        raise PermissionError('PILOT_NEW_LOOPBACK_VECTOR_ENDPOINT_REQUIRED')


def load_source_bundle(document: dict, catalog_path: Path, source_index: Path,
                       embedding: LocalEmbedding, collection: str) -> dict:
    """按版本和原文窗口选择旧向量；缺失即停止，不运行编码补齐。"""
    catalog = json.loads(Path(catalog_path).read_text())
    rows = [row for row in catalog if row['document_id'] == document['document_id']]
    if len(rows) != 1:
        raise ValueError('PILOT_SOURCE_DOCUMENT_AMBIGUOUS')
    source = rows[0]
    if source['content_sha256'] != document['source_sha256']:
        raise ValueError('PILOT_SOURCE_PDF_VERSION_CHANGED')
    parsed_path = Path(source['cache_path']).resolve()
    pdf_path = parsed_path.parent.parent / 'pdf' / (source['content_sha256'] + '.pdf')
    if file_hash(pdf_path) != source['content_sha256']:
        raise ValueError('PILOT_SOURCE_PDF_HASH_CHANGED')
    parsed = json.loads(parsed_path.read_text())
    if parsed['total_pages'] != source['total_pages']:
        raise ValueError('PILOT_SOURCE_PAGE_COUNT_CHANGED')
    lookup = {page['page_number']: page for page in parsed['pages']}
    pages = [lookup[number] for number in document['allowed_pages']]
    for given, page in zip(document['initial_pages'], pages):
        if (given['text'] != page['text'] or fingerprint(page['text']) != given['page_text_sha256']
                or page['document_id'] != source['document_id']):
            raise ValueError('PILOT_SOURCE_PAGE_TEXT_CHANGED')
    db = ReadOnlyIndex(source_index)
    store = EvidenceIndexStore(db)
    candidates = []
    for row in db.fetchall('SELECT index_id FROM evidence_index_builds WHERE document_id=?', (source['document_id'],)):
        build = store.get(row['index_id'])
        manifest = build['manifest']
        if (build['vector_status'] == build['bm25_status'] == 'ready'
                and manifest['source']['content_sha256'] == source['content_sha256']
                and manifest['model'] == embedding.identity
                and manifest['tokenizer'] == embedding.tokenizer_id
                and manifest['pages'] == IndexingService._page_manifest(parsed['pages'])):
            candidates.append(build)
    if len(candidates) != 1:
        raise ResourcePending('PILOT_EXACT_VECTOR_RECEIPTS_NOT_AVAILABLE')
    original = candidates[0]
    old_manifest = original['manifest']
    if fingerprint(old_manifest) != original['index_id']:
        raise ValueError('PILOT_SOURCE_INDEX_HASH_CHANGED')
    members = [deepcopy(row) for row in store.members(original['index_id'])
               if row['chunk']['evidence_id'] in {page['evidence_id'] for page in pages}]
    expected = [chunk.model_dump() for page in pages for chunk in split_page(
        page, old_manifest['source'], ChunkConfig(**old_manifest['splitter']),
        embedding.tokenizer_id, embedding.offsets)]
    if expected != [row['chunk'] for row in members]:
        raise ValueError('PILOT_SOURCE_CHUNK_WINDOWS_CHANGED')
    if not members or any(not valid_vector(row['embedding'], embedding.dimension) for row in members):
        raise ResourcePending('PILOT_SOURCE_VECTORS_INCOMPLETE')
    manifest = deepcopy(old_manifest)
    manifest.update(collection=collection, pages=IndexingService._page_manifest(pages),
                    chunk_ids=[row['chunk_id'] for row in members])
    index_id = fingerprint(manifest)
    for row in members:
        row['index_id'] = index_id
    return {'document': source, 'pdf_path': str(pdf_path), 'parsed_path': str(parsed_path),
            'parsed_sha256': file_hash(parsed_path), 'pages': pages, 'members': members,
            'manifest': manifest, 'index_id': index_id, 'old_index_id': original['index_id'],
            'source_index_sha256': file_hash(source_index)}


class ScopedCacheOnlyPDF(PDFProvider):
    """原生PDF缓存合同校验；只能返回批准单页，不解析或联网回退。"""

    def __init__(self, files, settings, bundle):
        self.bundle = bundle
        self.allowed_pages = {page['page_number'] for page in bundle['pages']}
        super().__init__(files, parser_name=settings.parser_name, parser_version=settings.parser_version,
                         parser_config=settings.parser_config, timeout=settings.request_timeout_seconds,
                         max_bytes=settings.max_pdf_bytes,
                         client=httpx.Client(transport=httpx.MockTransport(self._network_refused)))

    @staticmethod
    def _network_refused(request):
        raise PermissionError('PILOT_PDF_NETWORK_NOT_ALLOWED')

    def _parse(self, path):
        raise PermissionError('PILOT_NEW_PARSE_NOT_ALLOWED')

    def _read_source(self, source):
        expected = self.file_store.cache_pdf_path(self.bundle['document']['content_sha256'])
        if str(source).startswith(('http:', 'https:')) or Path(source).resolve() != expected.resolve():
            raise PermissionError('PILOT_PDF_SOURCE_NOT_BOUND')
        return super()._read_source(source)

    def _check_document(self, document):
        original = self.bundle['document']
        for key in ('document_id', 'content_sha256', 'parser_name', 'parser_version', 'config_sha256', 'total_pages'):
            if getattr(document, key) != original[key]:
                raise ValueError('PILOT_PDF_DOCUMENT_VERSION_CHANGED')
        if Path(document.cache_path).resolve() != self.file_store.cache_parse_path(Path(self.bundle['parsed_path']).stem).resolve():
            raise ValueError('PILOT_PDF_CACHE_PATH_CHANGED')

    def ingest(self, source, paper_id):
        document = super().ingest(source, paper_id)
        self._check_document(document)
        return document

    def ingest_pages(self, source, paper_id):
        document = super().ingest_pages(source, paper_id)
        self._check_document(document)
        return document

    def pages(self, document, start_page=1, page_count=None):
        self._check_document(document)
        if (type(start_page) is not int or type(page_count) is not int or page_count < 1
                or set(range(start_page, start_page + page_count)) != self.allowed_pages):
            raise PermissionError('PILOT_PAGE_RANGE_NOT_AUTHORIZED')
        pages = super().pages(document, start_page, page_count)
        if [page.model_dump(mode='json') for page in pages] != self.bundle['pages']:
            raise ValueError('PILOT_PDF_SELECTED_PAGE_CHANGED')
        return pages

    def close(self):
        self.client.close()


class _MeasuredEmbedding:
    """只统计本案例实际encode调用，原参数和原对象属性不变。"""

    def __init__(self, original, counters):
        self.original, self.counters = original, counters

    def __getattr__(self, name):
        return getattr(self.original, name)

    def encode(self, texts, *, query=False):
        kind = 'query_encode' if query else 'passage_encode'
        self.counters[kind + '_calls'] += 1
        self.counters[kind + '_texts'] += len(texts)
        try:
            return self.original.encode(texts, query=query)
        except Exception:
            self.counters[kind + '_failed_calls'] += 1
            raise


class _MeasuredReranker:
    """只统计原生score调用，不记录查询内容或改变候选顺序。"""

    def __init__(self, original, counters):
        self.original, self.counters = original, counters

    def __getattr__(self, name):
        return getattr(self.original, name)

    def score(self, query, passages):
        self.counters['reranker_score_calls'] += 1
        self.counters['reranker_candidates'] += len(passages)
        try:
            return self.original.score(query, passages)
        except Exception:
            self.counters['reranker_failed_calls'] += 1
            raise


class ScopedRetrievalRuntime(RetrievalRuntime):
    """每次原生检索前重验当前task权限和精确单页索引。"""

    def __init__(self, db, settings, resources):
        super().__init__(db, settings)
        self.resources = resources

    def search(self, task_id, request, cancelled=lambda: False):
        with self.lock:
            if cancelled():
                raise RuntimeError('RETRIEVAL_CANCELLED')
            self.resources.validate_task_scope(task_id, self.db, request.document_ids)
            if self.service is None:
                # 装配同一原生组件，只在实例转发边界计数；排序与参数仍由业务服务处理。
                from research_loop.providers.reranker import LocalReranker
                try:
                    embedding = LocalEmbedding(self.settings.retrieval_model_manifest)
                    reranker = LocalReranker(self.settings.retrieval_model_manifest)
                except (ImportError, FileNotFoundError) as exc:
                    return {'status': 'unavailable', 'hits': [], 'degradations': [
                        {'component': 'local_models', 'code': type(exc).__name__}],
                        'notice': '本机检索依赖或固定模型未就绪；可继续用read_paper直接读取，不冒充向量检索。'}
                self.service = RetrievalService(self.db,
                    _MeasuredEmbedding(embedding, self.resources.runtime_calls),
                    VectorIndex(self.settings.qdrant_url, self.settings.retrieval_namespace),
                    _MeasuredReranker(reranker, self.resources.runtime_calls))
            return super().search(task_id, request, cancelled)


class PilotResources:
    """一个独立案例的原件副本、索引清单及受限工厂。"""

    def __init__(self, case_dir, settings, case, *, source_documents_path, source_index_path,
                 vector_endpoint, namespace, model_manifest_path=None, protected_roots=()):
        self.case_dir = Path(case_dir).resolve()
        self.settings, self.case = settings, deepcopy(case)
        self.model_input, self.input_document = validate_case_input(case)
        source_checkout = Path(__file__).resolve().parents[2]
        if self.case_dir == source_checkout or source_checkout in self.case_dir.parents:
            raise PermissionError('PILOT_CASE_OUTSIDE_CHECKOUT_REQUIRED')
        for root in protected_roots:
            if self.case_dir == Path(root).resolve() or Path(root).resolve() in self.case_dir.parents:
                raise PermissionError('PILOT_PROTECTED_RESOURCE_ROOT')
        if settings.project_root.resolve() != self.case_dir:
            raise ValueError('PILOT_SETTINGS_ROOT_MISMATCH')
        for path in (settings.data_dir, settings.output_dir):
            if not Path(path).resolve().is_relative_to(self.case_dir):
                raise PermissionError('PILOT_SETTINGS_PATH_OUTSIDE_CASE')
        url = urlsplit(settings.database_url)
        if url.scheme not in {'sqlite', 'sqlite3'} or not Path(url.path).resolve().is_relative_to(self.case_dir):
            raise PermissionError('PILOT_NEW_SQLITE_REQUIRED')
        if not settings.retrieval_enabled:
            raise ValueError('PILOT_RETRIEVAL_CANNOT_BE_DISABLED')
        check_endpoint(vector_endpoint)
        if not re.fullmatch(r'researchloop_reflection_pilot_[a-z0-9_]{1,30}', namespace):
            raise ValueError('PILOT_ISOLATED_NAMESPACE_REQUIRED')
        if settings.retrieval_namespace != namespace or vector_endpoint is not None and settings.qdrant_url.rstrip('/') != vector_endpoint.rstrip('/'):
            raise ValueError('PILOT_VECTOR_SETTINGS_MISMATCH')
        self.endpoint, self.namespace = vector_endpoint, namespace
        model_manifest_path = model_manifest_path or settings.retrieval_model_manifest
        if model_manifest_path is None:
            raise ResourcePending('PILOT_EXPLICIT_MODEL_MANIFEST_REQUIRED')
        self.model_manifest_path = Path(model_manifest_path).resolve()
        if settings.retrieval_model_manifest is None or Path(settings.retrieval_model_manifest).resolve() != self.model_manifest_path:
            raise ValueError('PILOT_MODEL_MANIFEST_SETTINGS_MISMATCH')
        # LocalEmbedding构造只核固定资产与tokenizer，不分配ONNX session或编码。
        self.embedding_identity = LocalEmbedding(self.model_manifest_path)
        collection = namespace + '_' + fingerprint(self.embedding_identity.identity)
        self.bundle = load_source_bundle(self.input_document, Path(source_documents_path), Path(source_index_path),
                                         self.embedding_identity, collection)
        self.files = FileStore(settings.data_dir, settings.output_dir, settings.project_root)
        self.files.write_bytes_atomic(self.files.cache_pdf_path(self.bundle['document']['content_sha256']),
                                      Path(self.bundle['pdf_path']).read_bytes())
        self.files.write_bytes_atomic(self.files.cache_parse_path(Path(self.bundle['parsed_path']).stem),
                                      Path(self.bundle['parsed_path']).read_bytes())
        self.input_ref = 'input-' + case['case_id']
        self.attachments = [{'input_ref': self.input_ref, 'paper_id': self.input_document['paper_id'],
                             'local_path': str(self.files.cache_pdf_path(self.input_document['source_sha256']))}]
        self.vector_state = 'pending_new_vector_endpoint' if self.endpoint is None else 'not_installed'
        self.vector_readback = None
        self.bound_tasks = {}
        self.runtime_calls = {'query_encode_calls': 0, 'query_encode_texts': 0, 'query_encode_failed_calls': 0,
                              'passage_encode_calls': 0, 'passage_encode_texts': 0, 'passage_encode_failed_calls': 0,
                              'reranker_score_calls': 0, 'reranker_candidates': 0, 'reranker_failed_calls': 0}
        provider = self.pdf_factory(settings, self.files)
        try:
            document = provider.ingest_pages(self.attachments[0]['local_path'], self.input_document['paper_id'])
            pages = provider.pages(document, self.input_document['allowed_pages'][0], 1)
            self.pdf_document = document.model_dump(mode='json')
            self.pdf_pages = [page.model_dump(mode='json') for page in pages]
        finally:
            provider.close()

    def pdf_factory(self, settings, files):
        if (settings.project_root.resolve() != self.case_dir or files.data_dir != self.files.data_dir
                or files.output_dir != self.files.output_dir):
            raise PermissionError('PILOT_PDF_FACTORY_RESOURCE_MISMATCH')
        return ScopedCacheOnlyPDF(files, settings, self.bundle)

    def _vector_client(self):
        if self.endpoint is None:
            raise ResourcePending('PILOT_NEW_VECTOR_ENDPOINT_PENDING')
        collection = self.bundle['manifest']['collection']

        def guard(request):
            prefix = '/collections/' + collection
            if not request.url.path.startswith(prefix) or request.url.path[len(prefix):] not in {'', '/points', '/points/count'}:
                raise PermissionError('PILOT_QUALIFICATION_VECTOR_PATH_NOT_ALLOWED')
        return httpx.Client(base_url=self.endpoint, timeout=30, trust_env=False, event_hooks={'request': [guard]})

    def install_indexes(self, db, settings):
        """新存储upsert精确旧向量并读回；不计算query或文档向量。"""
        if db.kind != 'sqlite' or not Path(db.sqlite_path).resolve().is_relative_to(self.case_dir):
            raise PermissionError('PILOT_INDEX_OUTSIDE_NEW_SQLITE')
        if settings.qdrant_url.rstrip('/') != (self.endpoint or '').rstrip('/') or settings.retrieval_namespace != self.namespace:
            raise ValueError('PILOT_VECTOR_SETTINGS_MISMATCH')
        bundle = self.bundle
        points = [IndexingService._point(bundle['index_id'], bundle['manifest'], row) for row in bundle['members']]
        with self._vector_client() as client:
            vectors = VectorIndex(self.endpoint, self.namespace, client)
            vectors.ensure_collection(bundle['manifest']['collection'], self.embedding_identity.dimension)
            actual = vectors.retrieve(bundle['manifest']['collection'], [point['id'] for point in points])
            absent = [point for point in points if point['id'] not in actual]
            if any(not matches(actual.get(point['id']), point) for point in points if point['id'] in actual):
                raise ValueError('PILOT_EXISTING_VECTOR_POINT_CHANGED')
            if absent:
                vectors.upsert(bundle['manifest']['collection'], absent)
            actual = vectors.retrieve(bundle['manifest']['collection'], [point['id'] for point in points])
            count = vectors.count(bundle['manifest']['collection'], bundle['index_id'])
            if count != len(points) or len(actual) != len(points) or not all(matches(actual.get(p['id']), p) for p in points):
                raise ValueError('PILOT_VECTOR_READBACK_MISMATCH')
        store = EvidenceIndexStore(db)
        store.prepare(bundle['index_id'], bundle['manifest'], [SourceChunk.model_validate(row['chunk']) for row in bundle['members']], terms)
        for row in bundle['members']:
            store.vector_receipt(bundle['index_id'], row['chunk_id'], row['embedding'])
        store.state(bundle['index_id'], 'ready')
        self.vector_state = 'ready'
        self.vector_readback = {'index_id': bundle['index_id'], 'exact_points': len(points),
                                'payloads_exact': True, 'max_vector_abs_error': max(abs(x-y)
                                for point in points for x,y in zip(actual[point['id']]['vector'], point['vector'])),
                                'new_encode_calls': 0, 'new_rerank_calls': 0}
        return deepcopy(self.vector_readback)

    def bind(self, task_id, db):
        """经原PaperService读取批准页后安装索引；元数据不代替task授权。"""
        provider = self.pdf_factory(self.settings, self.files)
        try:
            observation = PaperService(db, None, provider).read(task_id, self.input_ref,
                start_page=self.input_document['allowed_pages'][0], page_count=1,
                attachments={self.input_ref: self.attachments[0]})
        finally:
            provider.close()
        refs = [page['evidence_id'] for page in observation['pages']]
        if observation['returned_pages'] != self.input_document['allowed_pages']:
            raise ValueError('PILOT_BIND_READ_SCOPE_CHANGED')
        self.bound_tasks[task_id] = refs
        if self.endpoint is not None:
            self.install_indexes(db, self.settings)
            self.validate_task_scope(task_id, db, [observation['document_id']])
        return {'evidence_ids': refs, 'document_id': observation['document_id'],
                'index_id': self.bundle['index_id'] if self.vector_state == 'ready' else None,
                'vector_status': self.vector_state, 'read_observation': observation}

    def validate_task_scope(self, task_id, db, document_ids):
        if document_ids != [self.bundle['manifest']['source']['document_id']]:
            raise PermissionError('PILOT_RETRIEVAL_DOCUMENT_NOT_AUTHORIZED')
        refs = [page['evidence_id'] for page in self.bundle['pages']]
        source, pages = IndexingService(db, self.embedding_identity, None)._source_pages(task_id, document_ids[0], refs)
        if source != self.bundle['manifest']['source'] or IndexingService._page_manifest(pages) != self.bundle['manifest']['pages']:
            raise ValueError('PILOT_AUTHORIZED_SOURCE_VERSION_CHANGED')
        # 拒绝同一task额外得到页后仍利用单页冻结成绩的情况。
        actual = db.fetchall('SELECT e.evidence_id FROM page_evidence e JOIN task_evidence t ON t.evidence_id=e.evidence_id '
                             'WHERE t.task_id=? AND e.document_id=?', (task_id, document_ids[0]))
        if {row['evidence_id'] for row in actual} != set(refs):
            raise PermissionError('PILOT_TASK_PAGE_SCOPE_EXPANDED')
        return {'task_id': task_id, 'evidence_ids': refs, 'source_identity_verified': True}

    def retrieval_factory(self, db, settings):
        if self.vector_state != 'ready':
            raise ResourcePending('PILOT_VECTOR_RESOURCES_NOT_READY')
        if not settings.retrieval_enabled:
            raise ValueError('PILOT_RETRIEVAL_CANNOT_BE_DISABLED')
        if settings.qdrant_url.rstrip('/') != self.endpoint.rstrip('/') or settings.retrieval_namespace != self.namespace:
            raise ValueError('PILOT_VECTOR_SETTINGS_MISMATCH')
        return ScopedRetrievalRuntime(db, settings, self)

    def report(self):
        return {'status': 'ready_for_resource_consumption' if self.vector_state == 'ready' else 'partial_resources',
                'case_id': self.case['case_id'], 'allowed_pages': self.input_document['allowed_pages'],
                'pdf': {'document': self.pdf_document, 'returned_pages': [p['page_number'] for p in self.pdf_pages],
                        'pdf_copy_sha256': file_hash(self.attachments[0]['local_path']),
                        'parse_copy_sha256': file_hash(self.pdf_document['cache_path']),
                        'source_pdf_sha256': self.input_document['source_sha256']},
                'retrieval': {'enabled': self.settings.retrieval_enabled, 'vector_status': self.vector_state,
                              'endpoint': self.endpoint, 'namespace': self.namespace,
                              'old_index_id': self.bundle['old_index_id'], 'new_index_id': self.bundle['index_id'],
                              'pages': len(self.bundle['pages']), 'chunks': len(self.bundle['members']),
                              'readback': self.vector_readback, 'model_identity': self.embedding_identity.identity,
                              'model_manifest_sha256': file_hash(self.model_manifest_path)},
                'real_generation_calls': 0, 'new_embedding_calls': 0, 'new_rerank_calls': 0,
                'new_pdf_parse_calls': 0, 'new_search_fetch_calls': 0,
                'counter_scope': 'The preceding zero counters apply only to resource preparation/install; actual Agent retrieval calls are below.',
                'agent_retrieval_calls': deepcopy(self.runtime_calls),
                'request_configuration': {key: getattr(self.settings, key) for key in (
                    'deepseek_model', 'deepseek_base_url', 'request_timeout_seconds', 'context_window_tokens',
                    'context_output_reserve', 'context_safety_tokens', 'execution_budget_mode', 'max_model_requests',
                    'max_tool_actions', 'reserve_model_requests', 'task_token_limit', 'task_token_tail',
                    'max_reflection_rounds', 'retrieval_enabled')},
                'limitations': ['Resource readiness does not measure model quality or live account billing.',
                               'Only the approved physical page may be returned; full cache presence is not authorization.',
                               'Resource qualification does not execute vector queries or neural inference; Agent-stage calls are recorded separately.']}


def prepare_case(case_dir, settings, case, **kwargs):
    """返回独立案例资源；不创建Task、不启动服务，也不读取评分标签。"""
    return PilotResources(case_dir, settings, case, **kwargs)
