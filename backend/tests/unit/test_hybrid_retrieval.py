"""混合检索离线合同：权限/降级/版本/排序，不把替身分数当真实模型质量。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from pydantic import ValidationError

from research_loop.config import Settings
from research_loop.domain.retrieval import RetrievalQuery
from research_loop.services.retrieval import RetrievalRuntime, RetrievalService
from research_loop.storage.indexes import rank, rank_terms, reciprocal_rank_fusion
from research_loop.storage.mysql import Database
from research_loop.storage.vector_index import VectorIndex
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs
from research_loop.tools.registry import ToolRegistry
from tests.unit.test_retrieval_contracts import FakeEmbedding, FakeVector, seed_document


class QueryEmbedding(FakeEmbedding):
    """使确定性离线替身支持query参数，非E5语义验证。"""
    def encode(self, texts, *, query=False):
        """沿用可复现单位向量，记录所有编码。"""
        return super().encode(texts)


class QueryVector(FakeVector):
    """按确切ID计算余弦，支持故障与越界回包注入。"""
    fail = False
    corrupt = False

    def query(self, collection, vector, ids, limit):
        """模拟服务返回，越界payload必须被业务层拒绝。"""
        if self.fail:
            raise OSError('simulated outage')
        self.last_ids = ids
        result = [{**copy.deepcopy(self.points[collection][i]), 'score': sum(a*b for a,b in zip(vector, self.points[collection][i]['vector']))} for i in ids]
        result.sort(key=lambda r: (-r['score'], r['id']))
        if self.corrupt:
            result[0]['payload']['document_id'] = 'not-authorized'
        return result[:limit]


class TestReranker:
    """稳定排序替身，用于观察失败时是否还保留RRF。"""
    identity = {'model_id': 'synthetic-only'}
    fail = False

    def score(self, query, passages):
        """按词项计算可预期分数，非语义重排。"""
        if self.fail:
            raise ValueError('RERANK_INPUT_TOO_LONG')
        return [float(p.lower().count(query.lower())) for p in passages]


class HybridRetrievalTests(unittest.TestCase):
    """通过真实业务服务验证混合检索的可见结果和访问控制。"""
    def setUp(self):
        """每例独立SQLite/原文快照，未连接生产服务。"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = Database('sqlite:///'+str(self.root/'test.db'))
        self.db.ensure_schema()
        for task in ['owner','other']:
            self.db.create_task(task, task, task, '阅读全文', 'deny', [])
        self.doc, self.pages = seed_document(self.db, self.root, text='Language embeddings identify unseen objects.\n\nExactTerm appears here.')
        self.embedding, self.vectors, self.reranker = QueryEmbedding(), QueryVector(), TestReranker()
        self.service = RetrievalService(self.db,self.embedding,self.vectors,self.reranker)

    def search(self, task='owner', docs=None, query='Language'):
        """使用与工具相同的Pydantic输入合同。"""
        return self.service.search(task, RetrievalQuery(query=query, document_ids=docs or [self.doc['document_id']]))

    def test_three_channels_and_original_offsets(self):
        result = self.search()
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['coverage'], 'retrieved_chunks_only')
        self.assertEqual(result['factual_support'], 'not_assessed')
        for hit in result['hits']:
            page = self.db.get_page(hit['evidence_id'])
            self.assertEqual(hit['text'], page['text'][hit['char_start']:hit['char_end']])
            self.assertIn('vector', hit['channels'])
            self.assertIsInstance(hit['reranker_score'], float)

    def test_repeated_query_reuses_stored_document_vectors(self):
        first = self.search()
        encoded = self.embedding.encoded
        second = self.search()
        self.assertEqual(first, second)
        self.assertEqual(self.embedding.encoded-encoded, 1)  # 只有query编码

    def test_other_task_cannot_query_known_index(self):
        self.search()
        with self.assertRaises(PermissionError):
            self.search('other')

    def test_version_filter_does_not_select_old_or_other_document(self):
        other, pages = seed_document(self.db, self.root, 'other-version', text='Language unrelated source')
        self.search(docs=[other['document_id']])
        result = self.search()
        self.assertEqual({h['document_id'] for h in result['hits']}, {self.doc['document_id']})
        for key in self.vectors.last_ids:
            self.assertTrue(any(key == h['key'] for h in result['hits']))
        with self.assertRaises(PermissionError):
            self.search(docs=['nonexistent-version'])

    def test_two_documents_share_lexical_corpus_and_vector_ranking(self):
        other, _ = seed_document(self.db, self.root, 'another', text='Language and images together')
        result = self.search(docs=[self.doc['document_id'], other['document_id']])
        self.assertEqual(len(result['scope']), 2)
        self.assertEqual({h['document_id'] for h in result['hits']}, {self.doc['document_id'], other['document_id']})

    def test_user_page_restriction_mints_narrow_index(self):
        first = self.search()
        self.db.execute("UPDATE tasks SET prompt='只看第1页' WHERE task_id='owner'")
        result = self.search()
        self.assertNotEqual(first['scope'][0]['index_id'], result['scope'][0]['index_id'])
        self.assertEqual({h['page_number'] for h in result['hits']}, {1})

    def test_vector_outage_reports_bm25_only(self):
        self.search()
        self.vectors.fail = True
        result = self.search()
        self.assertEqual(result['status'], 'degraded')
        self.assertEqual(result['channels']['vector']['status'], 'unavailable')
        self.assertTrue(result['hits'])
        self.assertTrue(all('vector' not in h['channels'] for h in result['hits']))

    def test_reranker_failure_preserves_rrf_order(self):
        self.reranker.fail = True
        result = self.search()
        self.assertEqual(result['degradations'][0]['code'], 'RERANK_INPUT_TOO_LONG')
        self.assertEqual([h['rrf_rank'] for h in result['hits']], list(range(1,len(result['hits'])+1)))

    def test_wrong_vector_payload_is_not_returned(self):
        self.vectors.corrupt = True
        result = self.search()
        self.assertEqual(result['degradations'][0]['code'], 'RETRIEVAL_VECTOR_IDENTITY_MISMATCH')
        self.assertNotIn('not-authorized', json.dumps(result))

    def test_vector_missing_points_does_not_bias_some_papers(self):
        first = self.search()
        points = next(iter(self.vectors.points.values()))
        del points[next(iter(points))]
        result = self.search()
        self.assertEqual(result['channels']['vector']['status'], 'unavailable')
        self.assertEqual(result['degradations'][0]['code'], 'RETRIEVAL_VECTOR_COVERAGE_MISMATCH')

    def test_source_corruption_cannot_be_bm25_fallback(self):
        self.search()
        self.db.execute("UPDATE page_evidence SET text='bad' WHERE evidence_id=?", (self.pages[0]['evidence_id'],))
        with self.assertRaisesRegex(ValueError, 'CACHE_MISMATCH'):
            self.search()

    def test_unsupported_query_never_claims_factual_support(self):
        result = self.search(query='undocumented GPU driver version')
        self.assertEqual(result['channels']['bm25']['hit_count'], 0)
        self.assertEqual(result['factual_support'], 'not_assessed')
        self.assertTrue(result['hits'])  # 向量近邻可能仍有返回，不能等同答案证据。

    def test_empty_query_or_duplicate_versions_rejected(self):
        for kwargs in [{'query':'  ','document_ids':['d']}, {'query':'ok','document_ids':['d','d']}]:
            with self.assertRaises(ValidationError):
                RetrievalQuery(**kwargs)

    def test_cancel_before_local_models_or_index(self):
        with self.assertRaisesRegex(RuntimeError, 'CANCELLED'):
            self.service.search('owner', RetrievalQuery(query='language',document_ids=[self.doc['document_id']]),lambda: True)
        self.assertEqual(self.embedding.encoded, 0)

    def test_rrf_deduplicates_channels_and_ties_deterministically(self):
        rows = reciprocal_rank_fusion({'bm25':[{'key':'a','score':20},{'key':'a','score':20},{'key':'b','score':1}],
            'vector':[{'key':'b','score':.9},{'key':'a','score':.8}]})
        self.assertEqual([r['key'] for r in rows], ['a','b'])
        self.assertAlmostEqual(rows[0]['rrf_score'], 1/61+1/62)

    def test_persisted_bm25_matches_memory_implementation(self):
        from collections import Counter
        from research_loop.storage.indexes import terms
        records = [{'search_text':'language language model'},{'search_text':'model images'}]
        self.assertEqual(rank('language',records), rank_terms('language',records,[Counter(terms(r['search_text'])) for r in records]))

    def test_qdrant_request_has_exact_authorized_ids(self):
        def transport(request):
            """断言过滤真正发送给Qdrant，不是检索后才过滤。"""
            body=json.loads(request.content)
            self.assertEqual(body['filter'], {'must':[{'has_id':['authorized']}]})
            self.assertEqual(body['params'], {'exact':True})
            return httpx.Response(200,json={'status':'ok','result':{'points':[]}})
        with httpx.Client(base_url='http://localhost', transport=httpx.MockTransport(transport)) as client:
            vector = VectorIndex('http://localhost','researchloop_test',client)
            self.assertEqual(vector.query('collection',[1.,0.],['authorized'],5), [])
            with self.assertRaises(ValueError): vector.query('collection',[1.,0.],[],5)

    def test_all_roles_see_same_tool_schema_and_execution(self):
        registry = ToolRegistry(specs())
        for role in ['supervisor','researcher','reviewer']:
            self.assertTrue(any(s['function']['name']=='retrieve_evidence' for s in registry.schemas(role)))
            ctx=ToolContext(task_id='owner',save_policy='deny',attachments={},cancelled=lambda:False,services={'retrieval':self.service},role=role)
            result=registry.execute('retrieve_evidence', {'query':'Language','document_ids':[self.doc['document_id']]},ctx)
            self.assertTrue(result.ok)

    def test_disabled_runtime_is_visible(self):
        settings=Settings(project_root=self.root,data_dir=self.root,output_dir=self.root,retrieval_enabled=False)
        result=RetrievalRuntime(self.db,settings).search('owner',RetrievalQuery(query='test',document_ids=['d']))
        self.assertEqual(result['status'], 'unavailable')
