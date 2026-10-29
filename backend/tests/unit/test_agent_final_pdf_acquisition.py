"""受控54篇的真实缓存回归；只读原件，原生授权/页登记使用临时SQLite。"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

BACKEND = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BACKEND / 'scripts'), str(BACKEND)]
from agent_final_pdf_acquisition import (ControlledPDFResolver, SOURCE_ACQUISITION_CONTRACT,
    controlled_pdf_provider, qualify_catalog, validate_source_acquisition_contract)
from agent_final_pdf_acquisition import inspect_controlled_mapping, install_runtime_catalog_sources
from harness_ab_resources import prepare_parsed_cache
from prompt_bundle_cases import install_catalog
from research_loop.config import Settings
from research_loop.providers.discovery import DiscoveryProvider
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database

_ANCHOR = os.environ.get('AGENT_FINAL_PDF_ANCHOR_ROOT')
SOURCE = Path(_ANCHOR).resolve() if _ANCHOR else None


def read(path):
    return json.loads(Path(path).read_text())


@unittest.skipUnless(SOURCE is not None and (SOURCE / 'inputs/vector-manifest.json').is_file(),
                     'requires explicit AGENT_FINAL_PDF_ANCHOR_ROOT with sealed local54 PDF receipts')
class FrozenAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='controlled-pdf54-')
        cls.root = Path(cls.temp.name).resolve()
        cls.files = FileStore(cls.root / 'data', cls.root / 'output', cls.root)
        manifest = read(SOURCE / 'manifest.formal.json')
        cls.case = next(c for c in read(manifest['cases_path'])['cases']
                        if len(c.get('materials', [])) == 54 and len((c.get('preconditions') or {}).get('catalog', [])) == 54)
        cls.bundles = {row['source_sha256']: read(row['resource_path'])
                       for row in read(manifest['vector_manifest_path'])['resources']}
        for bundle in cls.bundles.values():
            prepare_parsed_cache(cls.files, bundle['document_ref'])
        cls.parser = {key: manifest['settings'][key] for key in ('parser_name', 'parser_version', 'parser_config')}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.stack = ExitStack();self.addCleanup(self.stack.close)
        self.network = [self.stack.enter_context(patch.object(socket.socket, method,
            side_effect=AssertionError('real socket forbidden'))) for method in ('connect', 'connect_ex')]
        self.stack.enter_context(patch.object(socket, 'getaddrinfo', side_effect=AssertionError('real DNS forbidden')))
        self.stack.enter_context(patch.object(Settings, '__init__', side_effect=AssertionError('Settings instance forbidden')))
        self.stack.enter_context(patch.object(PDFProvider, '_parse', side_effect=AssertionError('new PDF parsing forbidden')))
        self.http = []
        def denied(request):
            self.http.append(str(request.url))
            raise AssertionError('remote PDF/discovery HTTP forbidden')
        self.client = self.stack.enter_context(httpx.Client(transport=httpx.MockTransport(denied), trust_env=False))
        self.pdf = controlled_pdf_provider(PDFProvider, self.files, self.case, self.bundles,
            client=self.client, **self.parser)
        self.discovery = DiscoveryProvider(client=self.client)
        install_catalog(self.stack, DiscoveryProvider, self.case, self.bundles)
        install_runtime_catalog_sources(self.stack, DiscoveryProvider, self.pdf.controlled_resolver)
        self.dbtemp = tempfile.TemporaryDirectory(prefix='controlled-pdf-db-');self.addCleanup(self.dbtemp.cleanup)
        self.db = Database('sqlite:///' + str(Path(self.dbtemp.name) / 'tasks.sqlite3'))
        self.db.ensure_schema()
        self.db.create_task('root', None, 'frozen54', self.case['prompt'], 'deny', [])

    def tearDown(self):
        self.assertEqual(self.http, [])
        for mock in self.network:
            self.assertEqual(mock.call_count, 0)

    def papers(self, *, task='root'):
        service = PaperService(self.db, self.discovery, self.pdf)
        service.search('research method', task_id=task, limit=54)
        return service

    def versioned(self):
        return next(e for e in self.pdf.controlled_resolver.catalog_entries
                    if e['source_kind'] == 'versioned_arxiv' and e['paper_id'].startswith('arxiv:'))

    def test_all54_exact_mappings_and_native_catalog_full973_plus_page1(self):
        rows, summary = qualify_catalog(self.db, 'root', self.pdf, self.discovery, self.case)
        self.assertEqual(summary['candidate_count'], 54)
        self.assertEqual(summary['expected_physical_pages'], 973)
        self.assertEqual(summary['observed_physical_pages'], 973)
        self.assertEqual(summary['native_search_grants_verified'], 54)
        self.assertEqual(summary['page_requests_verified'], 54)
        self.assertEqual(summary['full_reads_verified'], 54)
        self.assertEqual(len(self.db.fetchall('SELECT * FROM task_evidence WHERE task_id=?', ('root',))), 973)
        self.assertEqual(len(rows), 54)
        self.assertTrue(all(row['data']['acquisition']['kind'] == 'frozen_local_pdf' for row in rows))
        self.assertFalse(self.db.fetchall('SELECT * FROM task_inputs'))
        self.assertEqual(self.pdf.controlled_resolver.contract['catalog_bindings'], 54)

    def test_registered_tool_qualification_also_preserves_full973_read_contract(self):
        rows, summary = qualify_catalog(self.db, 'root', self.pdf, self.discovery, self.case, registered_search=True)
        proof = summary['registered_search']
        self.assertEqual(proof['status'], 'passed_registered_tool_contract')
        self.assertEqual(len(proof['exact_title_queries']), 54)
        self.assertEqual(len(proof['source_filtered_exact_title_queries']), 54)
        self.assertEqual(proof['illegal_limit54_probe']['kind'], 'validation_error')
        self.assertEqual(summary['observed_physical_pages'], 973)
        self.assertEqual(len(rows), 54)
        self.assertTrue(all(row['arguments']['limit'] <= 20 for row in proof['exact_title_queries']))

    def test_versioned_sources_keep_ID_versions_and_native_page_hashes(self):
        papers = self.papers()
        entries = [e for e in self.pdf.controlled_resolver.catalog_entries if e['source_kind'] == 'versioned_arxiv'][:3]
        self.assertTrue(entries)
        for entry in entries:
            ref = entry['paper_id']
            self.assertRegex(entry['source_locator'], r'v[1-9]\d*(?:\.pdf)?/?$')
            data = papers.read('root', ref, 1, 1)
            self.pdf.controlled_resolver.verify_read(entry, data, [1])
            self.assertEqual(data['paper_id'], ref)
            self.assertNotIn('final_url', data['acquisition'])
            self.assertEqual(data['acquisition']['network_requests'], 0)
            self.assertNotIn(str(self.root), json.dumps(data['acquisition']))

    def test_missing_URL_sources_use_blob_without_inventing_remote_version(self):
        papers = self.papers()
        entries = self.pdf.controlled_resolver.catalog_entries
        blobs = [e for e in entries if e['source_kind'] == 'frozen_local_blob']
        self.assertTrue(blobs)
        for entry in blobs:
            candidate = self.db.get_paper(entry['paper_id'])['metadata']
            self.assertEqual(candidate['paper_id'], entry['paper_id'])
            self.assertIsNone(candidate['version'])
            self.assertEqual(candidate['pdf_url'], 'frozen-pdf:' + entry['pdf_sha256'])
            provenance = candidate['metadata']['controlled_acquisition']
            self.assertIsNone(provenance['original_paper_url'])
            self.assertIsNone(provenance['original_pdf_url'])
            self.assertEqual(provenance['original_catalog_source_url'], '')
            self.assertEqual(provenance['version_status'], 'not_recorded_frozen_local_blob')
            data = papers.read('root', entry['paper_id'], 1, 1)
            self.pdf.controlled_resolver.verify_read(entry, data, [1])

    def test_wrong_blob_ID_hash_and_known_URL_fallback_are_denied(self):
        resolver = self.pdf.controlled_resolver
        blob = next(e for e in resolver.entries if e['source_kind'] == 'frozen_local_blob')
        other = next(e for e in resolver.entries if e['paper_id'] != blob['paper_id'])
        known = self.versioned()
        for identity, locator in [(blob['paper_id'], 'frozen-pdf:' + '0' * 64),
                                 (other['paper_id'], blob['source_locator']),
                                 (known['paper_id'], 'frozen-pdf:' + known['pdf_sha256'])]:
            with self.subTest(identity=identity), self.assertRaises(PermissionError):
                self.pdf.ingest_pages(locator, identity)
        changed = deepcopy(self.case)
        next(row for row in changed['preconditions']['catalog']
             if row['material_index'] == known['material_index'])['source_url'] = ''
        with self.assertRaises(PermissionError):
            ControlledPDFResolver(self.files, changed, self.bundles,
                parser_version=self.parser['parser_version'], parser_config=self.parser['parser_config'])

    def test_wrong_version_unversioned_casechanged_ID_and_unknown_URL_rejected(self):
        entry = self.versioned()
        url = entry['source_locator']
        for source, identity in [(re.sub(r'v\d+(?=(?:\.pdf)?/?$)', 'v999', url), entry['paper_id']),
                (re.sub(r'v\d+(?=(?:\.pdf)?/?$)', '', url), entry['paper_id']),
                (url, entry['paper_id'].replace('arxiv:', 'arXiv:')),
                ('https://example.com/frozen.pdf', entry['paper_id'])]:
            with self.subTest(source=source, identity=identity), self.assertRaises(PermissionError):
                self.pdf.ingest_pages(source, identity)

    def test_ungranted_candidate_and_wrong_model_ID_are_native_rejections(self):
        papers = self.papers()
        reference = self.versioned()['paper_id']
        self.db.create_task('other', None, 'ungranted', self.case['prompt'], 'deny', [])
        with self.assertRaises(PermissionError):
            papers.read('other', reference, 1, 1)
        for ref in (reference.replace('arxiv:', 'arXiv:', 1), 'arxiv:0000.00000'):
            with self.subTest(ref=ref), self.assertRaises(Exception):
                papers.read('root', ref, 1, 1)
        self.assertFalse(self.db.fetchall('SELECT * FROM task_evidence'))

    def test_user_page_limit_and_out_of_range_keep_native_checks(self):
        papers = self.papers()
        ref = self.versioned()['paper_id']
        entry = self.pdf.controlled_resolver.by_id[ref]
        self.db.create_task('limited', None, 'limited', '只读取 ' + entry['source_locator'] + ' 的第1页，不保存。', 'deny', [])
        self.db.grant_paper('limited', ref, 'limited')
        with self.assertRaises(PermissionError):
            papers.read('limited', ref, 2, 1)
        with self.assertRaises(ValueError):
            papers.read('root', ref, 999, 1)
        self.assertFalse(self.db.fetchall('SELECT * FROM task_evidence'))

    def test_changed_pdf_or_parsed_bytes_refused_before_native_read(self):
        papers = self.papers()
        entry = self.versioned()
        for key in ('pdf_path', 'parsed_path'):
            path = entry[key];original = path.read_bytes()
            try:
                path.write_bytes(original + b'altered')
                with self.subTest(key=key), self.assertRaises(PermissionError):
                    papers.read('root', entry['paper_id'], 1, 1)
            finally:
                path.write_bytes(original)
        self.assertFalse(self.db.fetchall('SELECT * FROM task_evidence'))

    def test_local_upload_preserves_input_ref_and_native_grants(self):
        material = self.case['materials'][0]
        ref = self.bundles[material['pdf_sha256']]['document_ref']
        upload = self.files.data_dir / 'uploads' / (material['pdf_sha256'] + '.pdf')
        self.files.write_bytes_atomic(upload, Path(material['pdf_path']).read_bytes())
        attachment = {'input_ref': 'upload-' + material['pdf_sha256'], 'paper_id': ref['paper_id'], 'local_path': str(upload)}
        self.db.create_task('attached', None, 'attached', '读取已上传材料，不保存。', 'deny', [attachment])
        data = PaperService(self.db, None, self.pdf).read('attached', attachment['input_ref'], 1, 1,
                                                       attachments={attachment['input_ref']: attachment})
        self.assertEqual(data['target_ref'], attachment['input_ref'])
        self.assertEqual(data['paper_id'], ref['paper_id'])
        self.assertTrue(self.db.task_has_input('attached', attachment['input_ref']))

    def test_qualification_refuses_preplanted_paper_grant(self):
        ref = self.pdf.controlled_resolver.catalog_entries[0]['paper_id']
        self.db.grant_paper('root', ref, 'root')
        with self.assertRaises(PermissionError):
            qualify_catalog(self.db, 'root', self.pdf, self.discovery, self.case)

    def test_mapping_digest_is_independent_of_cell_path(self):
        other = deepcopy(self.case)
        resolver = ControlledPDFResolver(self.files, other, self.bundles,
            parser_version=self.parser['parser_version'], parser_config=self.parser['parser_config'])
        public = [resolver._public(entry) for entry in resolver.entries]
        self.assertNotIn(str(self.root), json.dumps(public))
        self.assertEqual(resolver.contract['source_mapping_sha256'], self.pdf.controlled_resolver.contract['source_mapping_sha256'])
        inspected = inspect_controlled_mapping(self.case, self.bundles, pdf_dir=self.files.pdf_dir,
            parsed_dir=self.files.parse_dir, parser_version=self.parser['parser_version'], parser_config=self.parser['parser_config'])
        self.assertEqual(inspected['contract'], resolver.contract)
        self.assertEqual(inspected['entries'], public)


class CatalogQualificationAPITests(unittest.TestCase):
    def setUp(self):
        from unit.test_prompt_bundle_ab import BundleBackendInputTests
        BundleBackendInputTests.setUp(self)

    def test_qualify_mode_uses_native_HTTP_empty_attachments_then_native_search_grants(self):
        from run_read_paper_ab import create_bundle_input_task
        case = {'case_id': 'catalog-qualification-real-api', 'suite_kind': 'agent_final_v1',
            'prompt': '从本地候选题录给出建议，不保存。', 'materials': self.materials,
            'preconditions': {'catalog': [{'material_index': i, 'title': 'Candidate ' + str(i),
                'abstract': 'Scientific research metadata ' + str(i), 'source_url': ''} for i in range(2)]}}
        install_catalog(self.stack, DiscoveryProvider, case, self.bundles)
        report, attachments = create_bundle_input_task(self.client, self.db, case, 'A', 'qualify', self.bundles,
                                                      catalog_acquisition=True)
        self.assertEqual(attachments, [])
        self.assertEqual(report['upload_count'], 0)
        self.assertFalse(report['qualification_only_attachment_override'])
        root = report['root_id']
        self.assertEqual(json.loads(self.db.get_task(root)['attachments_json']), [])
        self.assertFalse(self.db.fetchall('SELECT * FROM task_inputs WHERE task_id=?', (root,)))
        self.assertFalse(self.db.fetchall('SELECT * FROM task_papers WHERE task_id=?', (root,)))
        discovery = DiscoveryProvider.from_settings(self.settings)
        self.addCleanup(discovery.close)
        result = PaperService(self.db, discovery, None).search('Scientific research', task_id=root, limit=2)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(len(self.db.fetchall('SELECT * FROM task_papers WHERE task_id=?', (root,))), 2)
        self.assertFalse(self.db.list_model_calls(root))


class SourceContractTests(unittest.TestCase):
    def test_v3_requires_exact_contract_and_legacy_qualification_bytes_are_unchanged(self):
        from run_read_paper_ab import qualification_configuration
        old = {'completion_contract': {'version': 'agent_final_completion_v2'}}
        self.assertFalse(validate_source_acquisition_contract(old))
        self.assertNotIn('source_acquisition_contract', qualification_configuration(old))
        new = {'completion_contract': {'version': 'agent_final_completion_v3'}, 'final_suite_policy': {}}
        with self.assertRaises(PermissionError):
            validate_source_acquisition_contract(new)
        new['source_acquisition_contract'] = deepcopy(SOURCE_ACQUISITION_CONTRACT)
        self.assertTrue(validate_source_acquisition_contract(new))
        self.assertEqual(qualification_configuration(new)['source_acquisition_contract'], SOURCE_ACQUISITION_CONTRACT)
        new['source_acquisition_contract']['remote_paper_fetch'] = 'allow'
        with self.assertRaises(PermissionError):
            validate_source_acquisition_contract(new)
        new['source_acquisition_contract'] = deepcopy(SOURCE_ACQUISITION_CONTRACT)
        new['source_acquisition_contract']['native_authorization_unchanged'] = 1
        with self.assertRaises(PermissionError):
            validate_source_acquisition_contract(new)


if __name__ == '__main__':
    unittest.main(verbosity=2)
