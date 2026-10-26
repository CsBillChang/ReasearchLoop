"""Reflection试点原文/原向量资源资格；显式本机材料门，不发送神经请求。"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from research_loop.config import Settings
from research_loop.domain.tasks import TaskCreate
from research_loop.providers.embeddings import LocalEmbedding
from research_loop.providers.reranker import LocalReranker
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from scripts.reflection_pilot_resources import (ResourcePending, ReadOnlyIndex, check_endpoint,
    file_hash, prepare_case, validate_case_input, _MeasuredEmbedding, _MeasuredReranker)


class PilotResourceGuardTests(unittest.TestCase):
    """无需私有语料即可验证输入标签、固定页范围及端口拒绝。"""

    def test_deployment_and_historical_endpoints_are_rejected(self):
        for url in ('http://127.0.0.1:6339', 'http://127.0.0.1:6351', 'http://127.0.0.1:6357',
                    'http://example.com:6361', 'https://127.0.0.1:6361', 'http://127.0.0.1:6361/path'):
            with self.subTest(url=url), self.assertRaises(PermissionError):
                check_endpoint(url)
        check_endpoint(None)
        check_endpoint('http://127.0.0.1:6361')

    def test_label_fields_and_expanded_page_ranges_are_rejected(self):
        sample = {'case_id': 'R06', 'model_input': {'question': '核验', 'draft': '当前稿',
            'documents': [{'allowed_pages': [5], 'initial_pages': []}]}}
        with self.assertRaisesRegex(ValueError, 'LABEL_FIELD'):
            validate_case_input({**sample, 'model_input': {**sample['model_input'], 'expected_text': '泄露'}})
        expanded = deepcopy(sample)
        expanded['model_input']['documents'][0]['allowed_pages'] = [5, 6]
        with self.assertRaisesRegex(PermissionError, 'PAGE_RANGE'):
            validate_case_input(expanded)

    def test_measurement_spies_preserve_original_arguments(self):
        observed = []

        class Encoding:
            identity = {'identity': 'synthetic-spy-only'}

            def encode(self, texts, *, query=False):
                observed.append(('encode', texts, query))
                return 'original-encoding-result'

        class Ranking:
            def score(self, query, passages):
                observed.append(('score', query, passages))
                return 'original-ranking-result'

        counters = {'query_encode_calls': 0, 'query_encode_texts': 0, 'query_encode_failed_calls': 0,
                    'passage_encode_calls': 0, 'passage_encode_texts': 0, 'passage_encode_failed_calls': 0,
                    'reranker_score_calls': 0, 'reranker_candidates': 0, 'reranker_failed_calls': 0}
        encoding = _MeasuredEmbedding(Encoding(), counters)
        ranking = _MeasuredReranker(Ranking(), counters)
        query, passages = ['same-query'], ['first', 'second']
        self.assertEqual(encoding.encode(query, query=True), 'original-encoding-result')
        self.assertEqual(encoding.encode(passages), 'original-encoding-result')
        self.assertEqual(ranking.score(query[0], passages), 'original-ranking-result')
        self.assertIs(observed[0][1], query)
        self.assertIs(observed[1][1], passages)
        self.assertIs(observed[2][2], passages)
        self.assertEqual((counters['query_encode_calls'], counters['passage_encode_calls'],
                          counters['reranker_score_calls'], counters['reranker_candidates']), (1, 1, 1, 2))
        self.assertEqual(encoding.identity, Encoding.identity)


class PilotActualResourceTests(unittest.TestCase):
    """显式新Qdrant和私有同版原件；无生成、编码、重排或旧资源写入。"""

    @classmethod
    def setUpClass(cls):
        source = os.getenv('REFLECTION_PILOT_DATA_ROOT')
        if not source:
            raise unittest.SkipTest('requires explicit private source root; no automatic resource discovery')
        cls.source = Path(source).resolve()
        cls.inputs = cls.source/'artifacts/reflection-baseline-preparation-20261026/pilot/inputs'
        cls.catalog = cls.source/'artifacts/360-retrieval-corpus-preparation-20261023/snapshot/documents.json'
        cls.old_index = cls.source/'artifacts/361-isolated-retrieval-runtime-20261023/isolated/evaluation.sqlite'
        cls.model = cls.source/'data/ag71-preflight/models/manifest.json'
        safe = os.getenv('REFLECTION_PILOT_SAFE_SETTINGS')
        if not safe:
            raise unittest.SkipTest('requires explicitly frozen safe settings')
        cls.safe = json.loads(Path(safe).read_text())['values']
        cls.endpoint = os.getenv('REFLECTION_PILOT_VECTOR_URL')
        if cls.endpoint:
            check_endpoint(cls.endpoint)

    def setUp(self):
        output = os.getenv('REFLECTION_PILOT_RESOURCE_OUTPUT')
        self.temp = None
        if output:
            self.root = Path(output).resolve() / (self._testMethodName + '-' + uuid4().hex[:8])
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            self.temp = tempfile.TemporaryDirectory()
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name).resolve()
        self.old_hash = file_hash(self.old_index)
        self.addCleanup(lambda: self.assertEqual(file_hash(self.old_index), self.old_hash))

    def make(self, case_id='R01', *, endpoint='selected'):
        case = json.loads((self.inputs/(case_id+'.json')).read_text())
        root = self.root / case_id
        root.mkdir(exist_ok=False)
        selected = self.endpoint if endpoint == 'selected' else endpoint
        namespace = 'researchloop_reflection_pilot_' + case_id.lower() + '_' + uuid4().hex[:8]
        values = {**self.safe, 'project_root': root, 'data_dir': root/'data', 'output_dir': root/'output',
                  'database_url': 'sqlite:///' + str(root/'case.sqlite'),
                  'qdrant_url': selected or 'http://127.0.0.1:1', 'retrieval_namespace': namespace,
                  'retrieval_model_manifest': self.model, 'deepseek_api_key': None,
                  'web_search_enabled': False, 'web_fetch_enabled': False}
        settings = Settings(**values)
        resource = prepare_case(root, settings, case, source_documents_path=self.catalog,
            source_index_path=self.old_index, vector_endpoint=selected, namespace=namespace,
            model_manifest_path=self.model)
        return resource, settings, case

    def task(self, resource, settings):
        db = Database(settings.database_url)
        db.ensure_schema()
        number = resource.input_document['allowed_pages'][0]
        task = TaskService(db, settings).create(TaskCreate(
            prompt=f'仅允许读取附件 {resource.input_ref} 的物理第{number}页；核验当前稿，不联网、不保存。',
            save_policy='deny', attachments=resource.attachments))
        return db, task

    def save(self, filename, value):
        with (self.root/filename).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)

    def test_all_six_native_pages_and_exact_reused_vectors(self):
        if not self.endpoint:
            self.skipTest('new isolated Qdrant endpoint not supplied')
        results = []
        with patch.object(LocalEmbedding, 'encode', side_effect=AssertionError('new encoding forbidden')), \
                patch.object(LocalReranker, 'score', side_effect=AssertionError('new rerank forbidden')), \
                patch('research_loop.config.Settings.from_env', side_effect=AssertionError('dotenv forbidden')):
            for case_id in ('R01', 'R02', 'R03', 'R04', 'R05', 'R06'):
                resource, settings, case = self.make(case_id)
                db, task = self.task(resource, settings)
                self.assertFalse(db.list_task_evidence(task.task_id))
                binding = resource.bind(task.task_id, db)
                self.assertEqual(binding['vector_status'], 'ready')
                self.assertEqual(binding['read_observation']['returned_pages'], case['model_input']['documents'][0]['allowed_pages'])
                self.assertFalse(binding['read_observation']['read_coverage']['full_document_returned'])
                self.assertFalse(binding['read_observation']['read_coverage']['all_extracted_text_returned'])
                self.assertEqual(settings.context_output_reserve, self.safe['context_output_reserve'])
                self.assertEqual(settings.execution_budget_mode, self.safe['execution_budget_mode'])
                self.assertEqual(settings.max_reflection_rounds, self.safe['max_reflection_rounds'])
                self.assertTrue(resource.retrieval_factory(db, settings).settings.retrieval_enabled)
                self.assertIsNone(resource.embedding_identity.session)
                self.assertEqual(len(db.list_task_evidence(task.task_id)), 1)
                self.assertFalse(db.fetchall('SELECT * FROM model_calls'))
                self.assertFalse(db.fetchall('SELECT * FROM save_receipts'))
                report = resource.report()
                self.assertEqual(report['pdf']['pdf_copy_sha256'], report['pdf']['source_pdf_sha256'])
                self.assertLess(report['retrieval']['readback']['max_vector_abs_error'], 1e-5)
                results.append({'case_id': case_id, 'binding': binding, 'report': report})
        self.save('all-six-resource-results.json', {'results': results, 'real_generation_calls': 0,
                   'new_encode_calls': 0, 'new_rerank_calls': 0})

    def test_pdf_foreign_page_full_range_and_changed_version_rejected(self):
        resource, settings, _ = self.make('R06', endpoint=None)
        provider = resource.pdf_factory(settings, resource.files)
        self.addCleanup(provider.close)
        document = provider.ingest_pages(resource.attachments[0]['local_path'], resource.input_document['paper_id'])
        for start, count in ((1, None), (4, 1), (5, 2), (6, 1)):
            with self.subTest(start=start, count=count), self.assertRaisesRegex(PermissionError, 'PAGE_RANGE'):
                provider.pages(document, start, count)
        with self.assertRaisesRegex(ValueError, 'VERSION_CHANGED'):
            provider.pages(document.model_copy(update={'content_sha256': '0'*64}), 5, 1)
        with self.assertRaisesRegex(PermissionError, 'SOURCE_NOT_BOUND'):
            provider.ingest_pages('/etc/hosts', resource.input_document['paper_id'])
        self.assertEqual([p.page_number for p in provider.pages(document, 5, 1)], [5])

    def test_unread_foreign_task_and_tampered_cache_cannot_use_scope(self):
        resource, settings, _ = self.make('R03', endpoint=None)
        db, task = self.task(resource, settings)
        doc = resource.input_document['document_id']
        with self.assertRaises((PermissionError, ValueError)):
            resource.validate_task_scope(task.task_id, db, [doc])
        binding = resource.bind(task.task_id, db)
        self.assertEqual(binding['vector_status'], 'pending_new_vector_endpoint')
        self.assertTrue(resource.validate_task_scope(task.task_id, db, [doc]))
        with self.assertRaises(PermissionError):
            resource.validate_task_scope(task.task_id, db, ['foreign-document'])
        other = TaskService(db, settings).create(TaskCreate(prompt='没有该页权限', save_policy='deny'))
        with self.assertRaises((PermissionError, ValueError)):
            resource.validate_task_scope(other.task_id, db, [doc])
        cache = Path(resource.pdf_document['cache_path'])
        value = json.loads(cache.read_text())
        next(p for p in value['pages'] if p['page_number'] == 5)['text'] += ' changed'
        cache.write_text(json.dumps(value, ensure_ascii=False))
        with self.assertRaises((PermissionError, ValueError)):
            resource.validate_task_scope(task.task_id, db, [doc])

    def test_missing_vector_service_stays_pending_without_disabling_retrieval(self):
        resource, settings, _ = self.make('R05', endpoint=None)
        self.assertTrue(settings.retrieval_enabled)
        self.assertEqual(resource.report()['retrieval']['vector_status'], 'pending_new_vector_endpoint')
        db, task = self.task(resource, settings)
        resource.bind(task.task_id, db)
        with self.assertRaises(ResourcePending):
            resource.retrieval_factory(db, settings)
        self.assertFalse(db.fetchall('SELECT * FROM evidence_index_builds'))

    def test_old_index_refuses_write_sql(self):
        with self.assertRaisesRegex(PermissionError, 'READ_ONLY'):
            ReadOnlyIndex(self.old_index).fetchall('DELETE FROM evidence_index_builds')


if __name__ == '__main__':
    unittest.main()
