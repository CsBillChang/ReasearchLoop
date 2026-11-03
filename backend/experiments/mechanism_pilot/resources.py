"""冻结原文资源复制：只读既有PDF/解析缓存，只向新实验任务授予批准单页。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import threading
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from research_loop.domain.evidence import Document
from research_loop.domain.retrieval import fingerprint, split_page
from research_loop.providers.pdf import PDFProvider
from research_loop.providers.embeddings import LocalEmbedding
from research_loop.services.indexing import IndexingService
from research_loop.services.papers import PaperService
from research_loop.storage.evidence_index import EvidenceIndexStore
from research_loop.storage.files import FileStore
from research_loop.storage.indexes import terms
from research_loop.storage.vector_index import VectorIndex
from experiments.mechanism_pilot.reflection import ScopedRetrievalRuntime, _validate_bound_material


def file_sha256(path):
    """流式校验本机文件，避免复制大PDF时同时保留多个内存副本。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime_namespace(root):
    """每个运行根拥有稳定独立的命名空间，不复用日常部署集合。"""
    return 'researchloop_mechanism_' + hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:16]


def validate_runtime(root, settings):
    """拒绝部署目录、共享库、外部数据目录及默认部署向量端口。"""
    root = Path(root).resolve()
    if any((p / '.git').exists() for p in (root, *root.parents)):
        raise ValueError('FROZEN_RUNTIME_INSIDE_CHECKOUT')
    if Path(settings.project_root).resolve() != root:
        raise ValueError('FROZEN_PROJECT_ROOT_MISMATCH')
    for path in (settings.data_dir, settings.output_dir):
        path = Path(path).resolve()
        if path == root or not path.is_relative_to(root):
            raise ValueError('FROZEN_DATA_OUTPUT_OUTSIDE_RUNTIME')
    url = urlsplit(settings.database_url)
    if url.scheme != 'sqlite' or url.netloc or not Path(url.path).resolve().is_relative_to(root):
        raise ValueError('FROZEN_ISOLATED_SQLITE_REQUIRED')
    if settings.web_search_enabled or settings.web_fetch_enabled:
        raise ValueError('FROZEN_WEB_DISABLED_REQUIRED')
    endpoint = urlsplit(settings.qdrant_url)
    if (endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1'
            or endpoint.username or endpoint.password or endpoint.path not in {'', '/'}
            or endpoint.query or endpoint.fragment or not endpoint.port
            or endpoint.port in {6333, 6334, 6339, 6351, 6357}):
        raise ValueError('FROZEN_ISOLATED_VECTOR_ENDPOINT_REQUIRED')
    if settings.retrieval_namespace != runtime_namespace(root):
        raise ValueError('FROZEN_VECTOR_NAMESPACE_MISMATCH')
    if (not settings.retrieval_enabled or settings.retrieval_route != 'located_fp16'
            or settings.retrieval_reranker_backend != 'mps_fp16'):
        raise ValueError('FROZEN_V160_RETRIEVAL_CONFIGURATION_REQUIRED')
    return root


def source_manifest_entry(case, pdf_path, parsed_path, settings):
    """从显式本机来源生成待冻结条目，不读取部署数据库、不解析PDF。"""
    doc = case['model_input']['documents'][0]
    config_sha = hashlib.sha256(json.dumps(settings.parser_config, sort_keys=True).encode()).hexdigest()
    cache_key = hashlib.sha256(f"{doc['source_sha256']}:pymupdf:{settings.parser_version}:{config_sha}".encode()).hexdigest()
    if doc['document_id'] != 'doc:' + cache_key[:32]:
        raise ValueError('FROZEN_PARSER_DOCUMENT_IDENTITY_MISMATCH')
    pdf_path, parsed_path = Path(pdf_path).resolve(), Path(parsed_path).resolve()
    if file_sha256(pdf_path) != doc['source_sha256']:
        raise ValueError('FROZEN_PDF_HASH_MISMATCH')
    payload = json.loads(parsed_path.read_text())
    number = doc['allowed_pages'][0]
    selected = [p for p in payload['pages'] if p['page_number'] == number]
    if len(selected) != 1 or selected[0]['document_id'] != doc['document_id'] or selected[0]['text'] != doc['initial_pages'][0]['text']:
        raise ValueError('FROZEN_SOURCE_PAGE_MISMATCH')
    return {'case_id': case['case_id'], 'pdf_path': str(pdf_path), 'parsed_path': str(parsed_path),
        'pdf_sha256': doc['source_sha256'], 'parsed_sha256': file_sha256(parsed_path),
        'document_id': doc['document_id'], 'paper_id': doc['paper_id'],
        'parser_name': 'pymupdf', 'parser_version': settings.parser_version,
        'parser_config': deepcopy(settings.parser_config), 'config_sha256': config_sha,
        'cache_key': cache_key, 'total_pages': payload['total_pages']}


class FrozenPagePDF(PDFProvider):
    """复用已核对缓存；不是PDF解析器，只允许批准页、禁止联网和重解析。"""

    def __init__(self, files, settings, document, source_path, allowed_pages):
        self.document, self.source_path = document, Path(source_path).resolve()
        self.allowed_pages = set(allowed_pages)
        super().__init__(files, parser_name=settings.parser_name, parser_version=settings.parser_version,
            parser_config=settings.parser_config, client=httpx.Client(transport=httpx.MockTransport(self._deny)))
        self._owns_client = True

    @staticmethod
    def _deny(request):
        raise PermissionError('FROZEN_PDF_NETWORK_FORBIDDEN')

    def _read_source(self, source):
        raise PermissionError('FROZEN_PDF_SOURCE_FETCH_FORBIDDEN')

    def _parse(self, path):
        raise PermissionError('FROZEN_PDF_PARSE_FORBIDDEN')

    def ingest(self, source, paper_id):
        self._check_cancelled()
        if str(source).startswith(('http:', 'https:')) or Path(source).resolve() != self.source_path or paper_id != self.document.paper_id:
            raise PermissionError('FROZEN_ATTACHMENT_IDENTITY_MISMATCH')
        if file_sha256(self.source_path) != self.document.content_sha256:
            raise ValueError('FROZEN_COPIED_PDF_DRIFT')
        self._cached_pages(self.document)
        self.last_fetch = {'kind': 'frozen_verified_parse_cache', 'executed_pdf_parse': False,
            'network_requests': 0, 'approved_physical_pages': sorted(self.allowed_pages)}
        return self.document

    ingest_pages = ingest

    def pages(self, document, start, count):
        if document != self.document or type(start) is not int or type(count) is not int or count <= 0:
            raise PermissionError('FROZEN_PAGE_SCOPE_REQUIRED')
        if not set(range(start, start + count)).issubset(self.allowed_pages):
            raise PermissionError('FROZEN_PAGE_SCOPE_EXCEEDED')
        return super().pages(document, start, count)


class FrozenPageResources:
    """把冻结原件复制进实验根，并通过PaperService建立真实任务级授权。"""

    def __init__(self, root, settings, case, entry, *, prepare_lexical=False):
        if type(prepare_lexical) is not bool:
            raise ValueError('FROZEN_EXPLICIT_LEXICAL_FLAG_REQUIRED')
        self.prepare_lexical = prepare_lexical
        self._lexical_embedding = None
        self._lexical_lock = threading.RLock()
        self.root = validate_runtime(root, settings)
        self.settings, self.case = settings, case
        doc = case['model_input']['documents'][0]
        current = source_manifest_entry(case, entry['pdf_path'], entry['parsed_path'], settings)
        if current != entry:
            raise ValueError('FROZEN_SOURCE_MANIFEST_DRIFT')
        self.files = FileStore(settings.data_dir, settings.output_dir, self.root)
        pdf = self.files.cache_pdf_path(entry['pdf_sha256'])
        parsed = self.files.cache_parse_path(entry['cache_key'])
        self.files.write_bytes_atomic(pdf, Path(entry['pdf_path']).read_bytes())
        self.files.write_bytes_atomic(parsed, Path(entry['parsed_path']).read_bytes())
        if file_sha256(pdf) != entry['pdf_sha256'] or file_sha256(parsed) != entry['parsed_sha256']:
            raise ValueError('FROZEN_COPIED_CACHE_DRIFT')
        self.document = Document(document_id=doc['document_id'], paper_id=doc['paper_id'],
            content_sha256=entry['pdf_sha256'], parser_name=entry['parser_name'],
            parser_version=entry['parser_version'], config_sha256=entry['config_sha256'],
            total_pages=entry['total_pages'], cache_path=str(parsed),
            gaps=list(json.loads(parsed.read_text()).get('gaps') or []) +
                  ['仅授权冻结单页；完整缓存存在不扩大任务取证范围。'])
        self.attachments = [{'input_ref': 'frozen-' + case['case_id'], 'paper_id': doc['paper_id'], 'local_path': str(pdf)}]
        self._pdf_path = pdf
        self.lexical_report = None
        provider = self.pdf_factory(settings, self.files)
        try:
            provider.ingest(str(pdf), doc['paper_id'])
            observed = provider.pages(self.document, doc['allowed_pages'][0], 1)[0]
            if observed.text != doc['initial_pages'][0]['text']:
                raise ValueError('FROZEN_COPIED_PAGE_DRIFT')
        finally:
            provider.close()

    def pdf_factory(self, settings, files):
        if (Path(settings.project_root).resolve() != self.root
                or files.data_dir != self.files.data_dir or files.output_dir != self.files.output_dir):
            raise PermissionError('FROZEN_PDF_FACTORY_ROOT_MISMATCH')
        return FrozenPagePDF(files, settings, self.document, self._pdf_path,
            self.case['model_input']['documents'][0]['allowed_pages'])

    def bind(self, task_id, db):
        """程序发起真实读取并在模型决策前暂停，保留原生行动及观察。"""
        if db.kind != 'sqlite' or not Path(db.sqlite_path).resolve().is_relative_to(self.root):
            raise ValueError('FROZEN_DATABASE_OUTSIDE_RUNTIME')
        from uuid import uuid4
        from research_loop.agents.graph import build_graph, initial_state
        from research_loop.domain.execution_policy import inherit_execution_policy
        from research_loop.tools.papers import specs as paper_specs
        from research_loop.tools.registry import ToolRegistry
        doc = self.case['model_input']['documents'][0]
        task = db.get_task(task_id)
        attachments = {item['input_ref']: item for item in self.attachments}
        state = initial_state(task_id, task['prompt'], task['save_policy'], attachments,
            self.settings.max_model_requests, self.settings.max_tool_actions,
            self.settings.reserve_model_requests)
        state.update(role=task.get('role', 'supervisor'),
            root_task_id=task.get('root_task_id') or task_id,
            parent_task_id=task.get('parent_task_id'))
        inherit_execution_policy(state, db.execution_policy(task_id), db.budget_summary(task_id))
        identity = 'programmatic-read-' + uuid4().hex
        arguments = {'target_ref': self.attachments[0]['input_ref'],
            'start_page': doc['allowed_pages'][0], 'page_count': 1}
        state.update(pending_tool_scope=['read_paper'], pending_tool_calls=[{
            'call_id': identity, 'name': 'read_paper', 'arguments': arguments}])
        state['messages'].append({'role': 'assistant',
            'content': '评测程序读取冻结授权原文；此动作不是模型生成。', 'reasoning_content': '',
            'tool_calls': [{'id': identity, 'type': 'function', 'function': {
                'name': 'read_paper', 'arguments': json.dumps(arguments, ensure_ascii=False)}}]})
        origin = {'origin': 'evaluation_program_explicit_frozen_page_read', 'model_generated': False,
            'model_receipt_created': False, 'tool_call_id': identity, 'arguments': arguments}
        db.append_event(task_id, 'reflection_pilot_programmatic_read', origin)
        provider = self.pdf_factory(self.settings, self.files)
        try:
            registry = ToolRegistry(paper_specs(), allowed_names={'read_paper'})
            graph = build_graph(None, registry, {'papers': PaperService(db, None, provider)}, db,
                self.settings.max_model_requests, self.settings.max_tool_actions,
                self.settings.reserve_model_requests)
            # 同一工具节点负责权限、行动序号与真实观察；中断阻止下一轮模型请求。
            state = graph.invoke(state, interrupt_after=['execute_tools'])
        finally:
            provider.close()
        actions = state.get('actions') or []
        action = actions[-1] if actions else None
        if (not action or action['tool_name'] != 'read_paper'
                or action['arguments'] != arguments or action['status'] != 'succeeded'
                or not action.get('observation', {}).get('ok')):
            raise ValueError('FROZEN_PROGRAMMATIC_READ_FAILED: ' + json.dumps(action, ensure_ascii=False))
        observation = action['observation']['data']
        bound = {'evidence_ids': [p['evidence_id'] for p in observation['pages']],
            'read_observation': observation, 'programmatic_read_state': state,
            'programmatic_read_origin': origin}
        bound['material_identity'] = _validate_bound_material(db, task_id, self.case, self.attachments, bound)
        db.save_paper(doc['paper_id'], {'paper_id': doc['paper_id'], 'title': doc.get('title', ''),
            'source': 'frozen_local_cache', 'metadata': {'source_kind': 'frozen_page_fixture'}})
        return bound

    def validate_scope(self, task_id, db, document_ids):
        doc = self.case['model_input']['documents'][0]
        if document_ids != [doc['document_id']]:
            raise PermissionError('FROZEN_RETRIEVAL_DOCUMENT_SCOPE')
        rows = db.fetchall('SELECT e.evidence_id,e.page_number FROM page_evidence e JOIN task_evidence t '
            'ON t.evidence_id=e.evidence_id WHERE t.task_id=? AND e.document_id=?', (task_id, doc['document_id']))
        if len(rows) != 1 or [r['page_number'] for r in rows] != doc['allowed_pages']:
            raise PermissionError('FROZEN_RETRIEVAL_PAGE_SCOPE')
        _validate_bound_material(db, task_id, self.case, self.attachments,
            {'evidence_ids': [r['evidence_id'] for r in rows]})

    def prepare_lexical_index(self, task_id, db, embedding):
        """只分词并准备当前索引manifest；不encode、不写Qdrant、不做重排。"""
        self.validate_scope(task_id, db, [self.document.document_id])
        mock = httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(
            PermissionError('FROZEN_VECTOR_NETWORK_FORBIDDEN'))))
        vectors = VectorIndex(self.settings.qdrant_url, self.settings.retrieval_namespace, mock)
        try:
            indexing = IndexingService(db, embedding, vectors)
            refs = db.list_task_evidence(task_id)
            source, pages = indexing._source_pages(task_id, self.document.document_id, refs)
            model_hash = fingerprint(embedding.identity)
            manifest = {'schema': 'original-evidence-v1', 'source': source,
                'pages': indexing._page_manifest(pages), 'splitter': indexing.config.model_dump(),
                'tokenizer': embedding.tokenizer_id, 'model': embedding.identity, 'model_hash': model_hash,
                'terms_version': 'latin-cjk-bigram-v1', 'collection': vectors.collection(model_hash)}
            chunks = [chunk for page in pages for chunk in split_page(page, source, indexing.config,
                embedding.tokenizer_id, embedding.offsets)]
            if not chunks:
                raise ValueError('FROZEN_EMPTY_CHUNKS')
            manifest['chunk_ids'] = [c.chunk_id for c in chunks]
            identity = fingerprint(manifest)
            EvidenceIndexStore(db).prepare(identity, manifest, chunks, terms)
            checked = indexing.authorized_chunks(task_id, identity)
            self.lexical_report = {'index_id': identity, 'chunk_count': len(chunks),
                'bm25_status': checked['bm25_status'], 'vector_status': checked['vector_status'],
                'embedding_calls': 0, 'reranker_calls': 0, 'qdrant_requests': 0,
                'neural_retrieval_qualified': False, 'route': self.settings.retrieval_route,
                'backend': self.settings.retrieval_reranker_backend}
            return dict(self.lexical_report)
        finally:
            mock.close()

    def _validate_and_prepare(self, task_id, db, document_ids):
        """每次核对本任务权限，正式路线在首次查询前准备无神经调用的词项索引。"""
        if db.kind != 'sqlite' or not Path(db.sqlite_path).resolve().is_relative_to(self.root):
            raise ValueError('FROZEN_DATABASE_OUTSIDE_RUNTIME')
        self.validate_scope(task_id, db, document_ids)
        if not self.prepare_lexical:
            return
        with self._lexical_lock:
            if self.lexical_report is not None:
                existing = EvidenceIndexStore(db).get(self.lexical_report['index_id'])
                if existing['bm25_status'] != 'ready':
                    raise ValueError('FROZEN_PREPARED_LEXICAL_NOT_READY')
                return
            manifest = self.settings.retrieval_model_manifest
            if manifest is None or not Path(manifest).is_file():
                raise ValueError('FROZEN_MODEL_MANIFEST_REQUIRED_FOR_LEXICAL')
            # 这里只验证固定模型/tokenizer并计算offset；不调用encode或重排。
            self._lexical_embedding = self._lexical_embedding or LocalEmbedding(manifest)
            self.prepare_lexical_index(task_id, db, self._lexical_embedding)

    def retrieval_factory(self, db, settings):
        validate_runtime(self.root, settings)
        return ScopedRetrievalRuntime(db, settings, self._validate_and_prepare)

    def report(self):
        return {'frozen_page_ready': True, 'source_pdf_copied_and_verified': True,
            'cached_parse_copied_and_verified': True, 'pdf_parse_calls': 0,
            'network_requests': 0, 'allowed_pages': self.case['model_input']['documents'][0]['allowed_pages'],
            'retrieval_route': self.settings.retrieval_route,
            'reranker_backend': self.settings.retrieval_reranker_backend,
            'namespace': self.settings.retrieval_namespace, 'lexical': self.lexical_report,
            'lexical_preparation_enabled': self.prepare_lexical,
            'neural_retrieval_qualified': False,
            'pending': ['live isolated vector endpoint and local model/runtime qualification before real execution']}


class FrozenResourcesFactory:
    """只消费外部冻结资格manifest，私有来源路径不写入公共源码。"""

    def __init__(self, manifest_path, expected_sha256, *, prepare_lexical=False):
        if type(prepare_lexical) is not bool:
            raise ValueError('FROZEN_EXPLICIT_LEXICAL_FLAG_REQUIRED')
        self.prepare_lexical = prepare_lexical
        path = Path(manifest_path).resolve()
        if file_sha256(path) != expected_sha256:
            raise ValueError('FROZEN_MANIFEST_HASH_MISMATCH')
        value = json.loads(path.read_text())
        if value.get('schema') != 'frozen-reflection-sources-v1' or not isinstance(value.get('cases'), list):
            raise ValueError('FROZEN_MANIFEST_SCHEMA')
        self.manifest_path, self.expected_sha256 = path, expected_sha256
        self.entries = {r['case_id']: r for r in value['cases']}
        if len(self.entries) != len(value['cases']):
            raise ValueError('FROZEN_MANIFEST_DUPLICATE_CASE')

    def __call__(self, root, settings, case):
        if file_sha256(self.manifest_path) != self.expected_sha256:
            raise ValueError('FROZEN_MANIFEST_DRIFT')
        entry = self.entries.get(case['case_id'])
        if entry is None:
            raise ValueError('FROZEN_CASE_NOT_IN_MANIFEST')
        return FrozenPageResources(root, settings, case, entry, prepare_lexical=self.prepare_lexical)
