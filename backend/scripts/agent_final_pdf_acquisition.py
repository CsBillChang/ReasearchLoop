"""固定语料的获取适配；仅改变实验载体定位，不授予原生材料或页权限。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from unittest.mock import patch
from urllib.parse import urlsplit


SOURCE_ACQUISITION_CONTRACT = {
    'version': 'frozen_pdf_acquisition_v1',
    'identity': 'exact_paper_id_and_explicit_frozen_locator',
    'remote_paper_fetch': 'deny',
    'integrity': 'pdf_and_parsed_sha256_and_page_identity',
    'qualification': 'native_catalog_search_upload0_full_and_page_read',
    'url_absent_policy': 'frozen_pdf_blob_only_when_original_locators_absent',
    'version_policy': 'explicit_arxiv_vN_or_exact_frozen_url_or_unrecorded_local_blob',
    'native_authorization_unchanged': True,
}


def validate_source_acquisition_contract(manifest):
    """V3不能借用旧上传资格；旧封存合同仍可按原协议只读解释。"""
    value = manifest.get('source_acquisition_contract')
    version = (manifest.get('completion_contract') or {}).get('version')
    if value is None:
        if version == 'agent_final_completion_v3':
            raise PermissionError('FINAL_SOURCE_ACQUISITION_CONTRACT_REQUIRED')
        return False
    try:
        exact = isinstance(value, dict) and _canonical(value) == _canonical(SOURCE_ACQUISITION_CONTRACT)
    except (TypeError, ValueError):
        exact = False
    if not exact or 'final_suite_policy' not in manifest:
        raise PermissionError('FINAL_SOURCE_ACQUISITION_CONTRACT_INVALID')
    return True


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


class ControlledPDFResolver:
    """只把当前格的精确候选/版本映射到独立副本，不归一化模型传来的ID。"""

    def __init__(self, files, case, bundles, *, parser_version, parser_config, on_denial=None):
        from prompt_bundle_cases import case_materials, validate_preconditions
        from research_loop.providers.paper_acquisition import resolve_pdf_url
        from research_loop.domain.evidence import explicit_versioned_paper_locator
        from research_loop.domain.source_access import canonical_paper_url
        self.files, self.on_denial = files, on_denial
        self.entries, self.by_id, self.by_url = [], {}, {}
        materials = case_materials(case)
        catalog = validate_preconditions(case)['catalog']
        config_hash = hashlib.sha256(json.dumps(parser_config, sort_keys=True).encode()).hexdigest()
        for index, material in enumerate(materials):
            bundle = bundles.get(material['pdf_sha256'])
            self._require(isinstance(bundle, dict), 'source_bundle_missing')
            ref = bundle['document_ref']
            metadata = ref['metadata']
            paper_id = ref['paper_id']
            self._require(metadata.get('paper_id') == paper_id and paper_id not in self.by_id,
                          'paper_identity_conflict')
            self._require(ref['pdf_sha256'] == ref['content_sha256'] == material['pdf_sha256']
                          == bundle['manifest']['source']['content_sha256'], 'pdf_identity_conflict')
            self._require(ref['parser_name'] == 'pymupdf' and ref['parser_version'] == parser_version
                          and ref['parser_config_sha256'] == config_hash, 'parser_contract_changed')
            cache_key = hashlib.sha256(f"{material['pdf_sha256']}:pymupdf:{parser_version}:{config_hash}".encode()).hexdigest()
            document_id = 'doc:' + cache_key[:32]
            self._require(Path(ref['parsed_path']).stem == cache_key
                          and ref['document_id_360'] == bundle['manifest']['source']['document_id'] == document_id,
                          'parsed_document_identity_conflict')
            pages = ref['page_identity_map']
            numbers = list(range(1, material['physical_pages'] + 1))
            self._require([p['physical_page'] for p in pages] == numbers
                          and ref['authorized_physical_pages'] == numbers, 'page_identity_map_changed')
            entry = {'material_index': index, 'material_id': material.get('material_id'), 'paper_id': paper_id,
                'pdf_sha256': material['pdf_sha256'], 'parsed_sha256': ref['parsed_cache_sha256'],
                'document_id': document_id, 'parser_version': parser_version,
                'parser_config_sha256': config_hash, 'physical_pages': material['physical_pages'],
                'page_identity_map': deepcopy(pages),
                'pdf_path': files.cache_pdf_path(material['pdf_sha256']),
                'parsed_path': files.cache_parse_path(cache_key), 'source_locator': None,
                'original_paper_url': metadata.get('paper_url'), 'original_pdf_url': metadata.get('pdf_url'),
                'original_metadata_version': metadata.get('version'), 'original_catalog_source_url': None,
                'source_kind': 'local_attachment', 'version_status': 'frozen_local_file'}
            self.entries.append(entry)
            self.by_id[paper_id] = entry
            self._validate_local(entry)
        for row in catalog:
            self._require(row['material_index'] < len(self.entries), 'catalog_material_unknown')
            entry = self.entries[row['material_index']]
            ref = bundles[entry['pdf_sha256']]['document_ref']
            metadata = ref['metadata']
            self._require(entry['source_locator'] is None, 'catalog_binding_conflict')
            entry['original_catalog_source_url'] = row['source_url']
            no_remote = (row['source_url'] == '' and 'paper_url' in metadata and 'pdf_url' in metadata
                         and metadata['paper_url'] is None and metadata['pdf_url'] is None
                         and metadata.get('version') is None)
            if no_remote:
                source = 'frozen-pdf:' + entry['pdf_sha256']
                aliases = {source}
                entry.update(source_kind='frozen_local_blob', version_status='not_recorded_frozen_local_blob')
            else:
                source = resolve_pdf_url(None, row['source_url'])
                aliases = {resolve_pdf_url(None, value) for value in
                    (metadata.get('paper_url'), metadata.get('pdf_url')) if isinstance(value, str) and value}
                parsed = urlsplit(source)
                self._require(source in aliases and parsed.scheme in {'http', 'https'}
                    and parsed.hostname and parsed.username is None and parsed.password is None,
                    'catalog_locator_changed')
                locator = canonical_paper_url(source)
                entry.update(source_kind='frozen_exact_url', version_status='not_applicable_exact_frozen_locator')
            if not no_remote and ((locator and locator[0] == 'arxiv')
                    or entry['paper_id'].startswith('arxiv:')):
                proven = explicit_versioned_paper_locator(entry['paper_id'], metadata)
                self._require(proven is not None and proven == locator
                              and re.fullmatch(r'v[1-9]\d*', locator[2] or '') is not None,
                              'explicit_arxiv_version_required')
                if entry['paper_id'].startswith('arxiv:'):
                    identifier = canonical_paper_url('https://arxiv.org/abs/' + entry['paper_id'][6:])
                    self._require(identifier and identifier[:2] == locator[:2]
                        and (identifier[2] is None or identifier[2] == locator[2]), 'arxiv_ID_locator_changed')
                entry.update(source_kind='versioned_arxiv', version_status='explicit_arxiv_version')
            entry['source_locator'] = source
            for alias in aliases:
                self._require((entry['paper_id'], alias) not in self.by_url, 'catalog_binding_conflict')
                self.by_url[(entry['paper_id'], alias)] = entry
        self.catalog_entries = [self.entries[row['material_index']] for row in catalog]
        self._require(len({e['paper_id'] for e in self.catalog_entries}) == len(catalog), 'catalog_paper_id_conflict')
        self.contract = {'version': SOURCE_ACQUISITION_CONTRACT['version'], 'material_count': len(self.entries),
            'catalog_bindings': len(self.catalog_entries), 'remote_paper_fetch': 'deny',
            'source_mapping_sha256': hashlib.sha256(_canonical([self._public(e) for e in self.entries]).encode()).hexdigest()}

    def _require(self, value, operation):
        if not value:
            if self.on_denial is not None:
                self.on_denial('controlled_pdf_' + operation)
            raise PermissionError('CONTROLLED_PDF_' + operation.upper())

    @staticmethod
    def _public(entry):
        return {k: deepcopy(v) for k, v in entry.items() if k not in {'pdf_path', 'parsed_path'}}

    def _validate_local(self, entry):
        for key, root, digest in (('pdf_path', self.files.pdf_dir, entry['pdf_sha256']),
                                  ('parsed_path', self.files.parse_dir, entry['parsed_sha256'])):
            path = entry[key]
            self._require(path.is_file() and not path.is_symlink() and path.resolve() == path
                          and path.is_relative_to(root.resolve()) and path.stat().st_nlink == 1,
                          'local_copy_path_changed')
            self._require(_hash(path) == digest, 'local_copy_hash_changed')
        payload = json.loads(entry['parsed_path'].read_text())
        self._require(payload['total_pages'] == entry['physical_pages']
                      and [p['page_number'] for p in payload['pages']] == list(range(1, entry['physical_pages'] + 1)),
                      'local_page_range_changed')
        expected = {p['physical_page']: p['page_text_sha256'] for p in entry['page_identity_map']}
        self._require(all(p['document_id'] == entry['document_id']
            and hashlib.sha256(p['text'].encode()).hexdigest() == expected[p['page_number']]
            for p in payload['pages']), 'local_page_identity_changed')

    def resolve(self, source, paper_id):
        self._require(isinstance(source, str) and isinstance(paper_id, str), 'source_type_invalid')
        if urlsplit(source).scheme in {'https', 'http', 'frozen-pdf'}:
            entry = self.by_url.get((paper_id, source))
            self._require(entry is not None, 'paper_or_version_not_frozen')
        else:
            entry = self.by_id.get(paper_id)
            self._require(entry is not None, 'paper_id_not_frozen')
            path = Path(source)
            allowed = {self.files.pdf_dir.resolve(), (self.files.data_dir / 'uploads').resolve()}
            self._require(path.is_absolute() and path.resolve() == path and path.parent in allowed
                          and path.is_file() and not path.is_symlink()
                          and path.name == entry['pdf_sha256'] + '.pdf', 'attachment_path_not_current_cell')
            self._require(_hash(path) == entry['pdf_sha256'], 'attachment_hash_changed')
        self._validate_local(entry)
        return entry

    def verify_document(self, entry, document, paper_id):
        self._require(document.paper_id == paper_id and document.document_id == entry['document_id']
            and document.content_sha256 == entry['pdf_sha256'] and document.total_pages == entry['physical_pages']
            and document.parser_name == 'pymupdf' and document.parser_version == entry['parser_version']
            and document.config_sha256 == entry['parser_config_sha256']
            and Path(document.cache_path) == entry['parsed_path'], 'native_document_identity_changed')

    def verify_read(self, entry, data, numbers):
        self._require(data['paper_id'] == entry['paper_id'] and data['target_ref'] == entry['paper_id']
            and data['document_id'] == entry['document_id'] and data['content_sha256'] == entry['pdf_sha256']
            and data['returned_pages'] == numbers and len(data['pages']) == len(numbers), 'native_read_identity_changed')
        expected = {p['physical_page']: p['page_text_sha256'] for p in entry['page_identity_map']}
        self._require(all(p['page_number'] == n and p['document_id'] == entry['document_id']
            and hashlib.sha256(p['text'].encode()).hexdigest() == expected[n]
            for p, n in zip(data['pages'], numbers, strict=True)), 'native_read_page_hash_changed')


def controlled_pdf_provider(native_type, files, case, bundles, *, on_denial=None, **kwargs):
    """用原生PDF解析和页返回实现，唯一注入点是已核验的本地载体定位。"""
    resolver = ControlledPDFResolver(files, case, bundles, parser_version=kwargs['parser_version'],
        parser_config=kwargs['parser_config'], on_denial=on_denial)

    class ControlledPDF(native_type):
        def _controlled(self, method, source, paper_id):
            entry = resolver.resolve(source, paper_id)
            document = method(str(entry['pdf_path']), paper_id)
            resolver.verify_document(entry, document, paper_id)
            # 没有HTTP下载，不能伪造final_url或successful remote attempts。
            self.last_fetch = {'kind': 'frozen_local_pdf', 'requested_source': source,
                'content_sha256': entry['pdf_sha256'], 'parsed_cache_sha256': entry['parsed_sha256'],
                'material_id': entry['material_id'], 'source_kind': entry['source_kind'],
                'version_status': entry['version_status'],
                'source_mapping_sha256': resolver.contract['source_mapping_sha256'],
                'attempts': [], 'network_requests': 0}
            return document

        def ingest(self, source, paper_id):
            return self._controlled(super().ingest, source, paper_id)

        def ingest_pages(self, source, paper_id):
            return self._controlled(super().ingest_pages, source, paper_id)

    instance = ControlledPDF(files, **kwargs)
    instance.controlled_resolver = resolver
    return instance


def inspect_controlled_mapping(case, bundles, *, pdf_dir, parsed_dir, parser_version, parser_config):
    """独立审计只读当前格副本，不构造Settings、Provider、FileStore或数据库。"""
    class ReadOnlyPaths:
        def __init__(self):
            self.pdf_dir = Path(pdf_dir)
            self.parse_dir = Path(parsed_dir)

        def cache_pdf_path(self, identity):
            return self.pdf_dir / (identity + '.pdf')

        def cache_parse_path(self, identity):
            return self.parse_dir / (identity + '.json')
    resolver = ControlledPDFResolver(ReadOnlyPaths(), case, bundles,
        parser_version=parser_version, parser_config=parser_config)
    return {'contract': deepcopy(resolver.contract), 'entries': [resolver._public(e) for e in resolver.entries]}


def install_runtime_catalog_sources(stack, discovery_type, resolver):
    """只为原始URL均缺失项补明确本地载体；不改冻结输入或猜测远端版本。"""
    original = discovery_type.search

    def search(provider, *args, **kwargs):
        result = original(provider, *args, **kwargs)
        candidates = []
        for candidate in result.candidates:
            entry = resolver.by_id.get(candidate.paper_id)
            resolver._require(entry is not None
                and candidate.metadata.get('source_pdf_sha256') == entry['pdf_sha256'], 'catalog_candidate_identity_changed')
            if entry['source_kind'] == 'frozen_local_blob':
                resolver._require(candidate.paper_url == '' and candidate.pdf_url is None
                    and candidate.version is None, 'local_blob_original_fields_changed')
                metadata = deepcopy(candidate.metadata)
                metadata['controlled_acquisition'] = {key: entry[key] for key in
                    ('original_paper_url', 'original_pdf_url', 'original_catalog_source_url',
                     'source_locator', 'source_kind', 'version_status', 'pdf_sha256', 'parsed_sha256')}
                candidate = candidate.model_copy(update={'pdf_url': entry['source_locator'], 'metadata': metadata})
            candidates.append(candidate)
        return result.model_copy(update={'candidates': candidates})

    stack.enter_context(patch.object(discovery_type, 'search', new=search))


def qualify_catalog(db, root_id, pdf, discovery, case):
    """用原生search授权及分页/全文read验证真实题录入口，不植入伪造grant。"""
    from research_loop.services.papers import PaperService
    from research_loop.providers.paper_acquisition import resolve_pdf_url
    resolver = pdf.controlled_resolver
    entries = resolver.catalog_entries
    resolver._require(bool(entries), 'qualification_catalog_empty')
    resolver._require(not json.loads(db.get_task(root_id)['attachments_json'])
                      and not db.fetchall('SELECT paper_id FROM task_papers WHERE task_id=?', (root_id,))
                      and not db.fetchall('SELECT input_ref FROM task_inputs WHERE task_id=?', (root_id,)),
                      'qualification_must_start_without_upload_or_pregrants')
    papers = PaperService(db, discovery, pdf)
    search = papers.search('research method', limit=len(entries), task_id=root_id)
    returned = {p.paper_id: p for p in search.candidates}
    resolver._require(len(search.candidates) == len(returned) == len(entries)
        and set(returned) == {e['paper_id'] for e in entries}
        and search.query_scope == 'frozen_local_candidate_pool_lexical_bm25_not_web_search'
        and all(db.task_has_paper(root_id, e['paper_id']) for e in entries)
        and not db.fetchall('SELECT evidence_id FROM task_evidence WHERE task_id=?', (root_id,)),
        'native_catalog_search_or_grants_changed')
    rows = []
    for entry in entries:
        candidate = returned[entry['paper_id']]
        resolver._require(resolve_pdf_url(candidate.pdf_url, candidate.paper_url) == entry['source_locator']
            and candidate.metadata['source_pdf_sha256'] == entry['pdf_sha256'], 'native_catalog_return_changed')
        page = papers.read(root_id, entry['paper_id'], 1, 1)
        resolver.verify_read(entry, page, [1])
        full = papers.read(root_id, entry['paper_id'])
        resolver.verify_read(entry, full, list(range(1, entry['physical_pages'] + 1)))
        rows.append({'entry': entry, 'data': full, 'page1_text_sha256': entry['page_identity_map'][0]['page_text_sha256']})
    summary = {'status': 'passed', 'candidate_count': len(entries),
        'native_search_grants_verified': len(entries), 'read_targets_verified': len(rows),
        'page_requests_verified': len(rows), 'full_reads_verified': len(rows),
        'expected_physical_pages': sum(e['physical_pages'] for e in entries),
        'observed_physical_pages': sum(len(row['data']['pages']) for row in rows),
        'model_calls': 0, 'source_mapping_sha256': resolver.contract['source_mapping_sha256'],
        'before_search_grants': 0, 'before_read_page_grants': 0,
        'query_scope': search.query_scope, 'search_query': 'research method',
        'source_kind_counts': {kind: sum(e['source_kind'] == kind for e in entries)
                              for kind in sorted({e['source_kind'] for e in entries})},
        'version_status_counts': {kind: sum(e['version_status'] == kind for e in entries)
                                 for kind in sorted({e['version_status'] for e in entries})}}
    return rows, summary
