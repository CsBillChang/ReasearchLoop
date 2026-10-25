"""候选服务的独立SQLite与可选新专用MySQL/Qdrant验证；神经计算全部为替身。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from research_loop.domain.retrieval import RetrievalQuery
from research_loop.domain.retrieval_strategy import get_retrieval_strategy
from research_loop.services.retrieval import RetrievalService
from research_loop.storage.mysql import Database
from research_loop.storage.vector_index import VectorIndex, point_id
from scripts.component_eval_adapters import RetrievalAdapter
from tests.unit.test_hybrid_retrieval import QueryEmbedding, QueryVector
from tests.unit.test_retrieval_contracts import seed_document


NAMES = ('baseline', 'fusion_bm25_2_1', 'efficiency_length_sort')


def configured_reranker(name):
    """按候选构造明确暴露组批参数的替身，不创建ONNX Session。"""
    strategy = get_retrieval_strategy(name)
    return SimpleNamespace(**strategy.reranker_kwargs, identity={'model_id': 'synthetic-only'},
        last_stats={}, score=Mock(side_effect=lambda query, texts: [float(t.count('Language')) for t in texts]))


def seed_tasks(db):
    """仅给专用测试库初始化两个相互独立的本地任务。"""
    for task in ('owner', 'other'):
        db.create_task(task, task, task, '阅读全文', 'deny', [])


def adapter_chunks(service, doc, task='owner'):
    """从当前已授权索引读回适配器输入，保持业务身份与payload一致。"""
    view = service._view(task, doc['document_id'])
    rows = []
    for item in view['chunks']:
        chunk = item['chunk']
        rows.append({**chunk, 'index_id': view['index_id'],
            'key': point_id(view['index_id'], chunk['chunk_id']),
            'collection': view['manifest']['collection'],
            'vector_payload': service.indexing._point(view['index_id'], view['manifest'],
                {'chunk': chunk, 'chunk_id': chunk['chunk_id'], 'embedding': None})['payload']})
    return rows


class CandidateSQLiteIsolationTests(unittest.TestCase):
    """实际权限/原文/索引读回与替身通道共用，临时库逐例销毁。"""

    def setUp(self):
        """拒绝网络、真实编码/CE与真正向量查询，所有原文均为合成。"""
        for target in ('socket.socket.connect', 'research_loop.providers.embeddings.LocalEmbedding.encode',
                       'research_loop.providers.reranker.LocalReranker.score',
                       'research_loop.providers.reranker.LocalReranker._load',
                       'research_loop.storage.vector_index.VectorIndex.query'):
            guard = patch(target, side_effect=AssertionError('真实组件禁止'))
            guard.start()
            self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = Database('sqlite:///' + str(self.root / 'candidate.db'))
        self.db.ensure_schema()
        seed_tasks(self.db)
        text = '\n\n'.join(f'Language method synthetic paragraph {n}.' for n in range(30))
        self.doc, self.pages = seed_document(self.db, self.root, text=text)
        self.embedding, self.vectors = QueryEmbedding(), QueryVector()

    def service(self, name=None):
        """构造显式候选或旧默认入口，二者共享同一份已授权索引。"""
        reranker = configured_reranker(name or 'baseline')
        service = RetrievalService(self.db, self.embedding, self.vectors, reranker,
            strategy=get_retrieval_strategy(name) if name else None)
        return service, reranker

    def request(self, **kwargs):
        """保留生产原请求合同，候选固定24检查由显式策略执行。"""
        return RetrievalQuery(query='Language', document_ids=[self.doc['document_id']], limit=8, **kwargs)

    def test_three_strategies_service_adapter_same_scope_ranking_and_identity(self):
        """两入口复用融合算法，全24候选与同版原文身份相互一致。"""
        identities = []
        for name in NAMES:
            service, reranker = self.service(name)
            result = service.search('owner', self.request())
            self.assertEqual(result['status'], 'ready')
            self.assertEqual(result['effective_path'], 'rrf_rerank')
            self.assertEqual(result['strategy'], get_retrieval_strategy(name).as_dict())
            self.assertTrue(result['reranker_configuration']['verified'])
            self.assertEqual(len(reranker.score.call_args.args[1]), 24)
            rows = adapter_chunks(service, self.doc)
            adapter = RetrievalAdapter(lambda: self.embedding, lambda: self.vectors, lambda: reranker,
                                       strategy=get_retrieval_strategy(name))
            observed = adapter.run('Language', rows, 'rrf_rerank', [r['chunk_id'] for r in rows])
            self.assertEqual([r['chunk_id'] for r in result['hits']], observed['ranked_ids'][:8])
            self.assertEqual([r['rrf_score'] for r in result['hits']], [r['rrf_score'] for r in observed['hits'][:8]])
            identities.append((result['scope'], result['embedding'], {r['chunk_id'] for r in rows}))
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(identities[0], identities[2])

    def test_default_unchanged_and_custom_candidate_limit_still_supported(self):
        """默认旧输出逐值保持且仍接受生产自定义候选量。"""
        default, legacy_reranker = self.service()
        explicit, _ = self.service('baseline')
        first = default.search('owner', self.request())
        second = explicit.search('owner', self.request())
        for key in ('strategy', 'reranker_configuration', 'effective_path'):
            second.pop(key)
        self.assertEqual(first, second)
        self.assertNotIn('strategy', first)
        custom = default.search('owner', self.request(candidate_limit=8))
        self.assertEqual(custom['candidate_limit'], 8)
        self.assertEqual(len(legacy_reranker.score.call_args.args[1]), 8)
        with self.assertRaisesRegex(ValueError, 'CANDIDATE_LIMIT_MISMATCH'):
            explicit.search('owner', self.request(candidate_limit=8))

    def test_mismatch_and_non_json_configuration_are_deliverable_rrf_fallback(self):
        """不匹配组件不得评分；坏对象和NaN观察仍可严格JSON序列化。"""
        for bad in (False, object(), float('nan')):
            service, reranker = self.service('efficiency_length_sort')
            reranker.sort_by_length = bad
            result = service.search('owner', self.request())
            reranker.score.assert_not_called()
            self.assertEqual(result['effective_path'], 'rrf')
            self.assertEqual(result['degradations'][0]['code'], 'RETRIEVAL_RERANKER_CONFIG_MISMATCH')
            self.assertFalse(result['reranker_configuration']['verified'])
            json.dumps(result, allow_nan=False)
            self.assertEqual([h['rrf_rank'] for h in result['hits']], list(range(1, 9)))

    def test_vector_and_reranker_failures_report_actual_paths(self):
        """候选回退仍为原完整RRF顺序，单路失效不伪称双路完成。"""
        service, reranker = self.service('fusion_bm25_2_1')
        service.search('owner', self.request())
        reranker.score.side_effect = ValueError('RERANK_INPUT_TOO_LONG')
        result = service.search('owner', self.request())
        self.assertEqual(result['effective_path'], 'rrf')
        self.assertEqual(result['degradations'][0]['code'], 'RERANK_INPUT_TOO_LONG')
        self.vectors.fail = True
        result = service.search('owner', self.request())
        self.assertEqual(result['effective_path'], 'bm25')
        self.assertEqual(result['channels']['vector']['status'], 'unavailable')
        self.assertTrue(all('vector' not in h['channels'] for h in result['hits']))
        self.assertTrue(all(h['rrf_score'] == 2 / (60 + h['channels']['bm25']['rank']) for h in result['hits']))

    def test_other_task_version_and_narrow_page_scope_do_not_leak(self):
        """已知文档ID不授予权限，页限制不会从旧全页索引补候选。"""
        service, _ = self.service('fusion_bm25_2_1')
        first = service.search('owner', self.request())
        with self.assertRaises(PermissionError):
            service.search('other', self.request())
        other, _ = seed_document(self.db, self.root, 'another-version', text='Language foreign version')
        foreign = service.search('owner', RetrievalQuery(query='Language', document_ids=[other['document_id']]))
        self.assertEqual({r['document_id'] for r in foreign['hits']}, {other['document_id']})
        self.db.execute("UPDATE tasks SET prompt='只看第1页' WHERE task_id='owner'")
        narrow = service.search('owner', self.request())
        self.assertEqual({h['page_number'] for h in narrow['hits']}, {1})
        self.assertNotEqual(first['scope'][0]['index_id'], narrow['scope'][0]['index_id'])

    def test_cancel_before_work_and_after_score_stops_delivery(self):
        """取消继续沿用边界检查，不因显式候选而追加工作或交付。"""
        service, reranker = self.service('efficiency_length_sort')
        with self.assertRaisesRegex(RuntimeError, 'CANCELLED'):
            service.search('owner', self.request(), lambda: True)
        self.assertEqual(self.embedding.encoded, 0)
        stopped = False

        def cancel_after_score(query, texts):
            """替身执行后触发取消，验证返回前仍检查。"""
            nonlocal stopped
            stopped = True
            return [0.] * len(texts)

        reranker.score.side_effect = cancel_after_score
        with self.assertRaisesRegex(RuntimeError, 'CANCELLED'):
            service.search('owner', self.request(), lambda: stopped)

    def test_revocation_or_source_corruption_during_score_cannot_be_delivered(self):
        """返回前真实授权与hash复验不被候选排序绕过。"""
        service, reranker = self.service('baseline')

        def revoke(query, texts):
            """在评分边界撤销页授权，不触碰其他任务。"""
            self.db.execute("DELETE FROM task_evidence WHERE task_id='owner'")
            return [0.] * len(texts)

        reranker.score.side_effect = revoke
        with self.assertRaises(PermissionError):
            service.search('owner', self.request())
        for page in self.pages:
            self.db.mark_task_evidence('owner', page['evidence_id'])

        def corrupt(query, texts):
            """仅篡改临时测试库的页文本，证明原文校验仍拒绝。"""
            self.db.execute('UPDATE page_evidence SET text=? WHERE evidence_id=?',
                            ('corrupted synthetic page', self.pages[0]['evidence_id']))
            return [0.] * len(texts)

        reranker.score.side_effect = corrupt
        with self.assertRaisesRegex(ValueError, 'CACHE_MISMATCH'):
            service.search('owner', self.request())


class PayloadReadbackVector(VectorIndex):
    """只在新隔离库写入并读回合成向量，query使用确定性替身分数。"""

    def query(self, collection, vector, ids, limit):
        """按明确ID读取payload后模拟排名，不发送相似度query请求。"""
        points = self.retrieve(collection, ids)
        return [{'id': key, 'score': float(len(ids) - n), 'payload': points[key]['payload']}
                for n, key in enumerate(sorted(ids)[:limit]) if key in points]


class CandidateExternalIsolationTests(unittest.TestCase):
    """仅显式环境启用全新专用MySQL/Qdrant，缺少环境时明确跳过。"""

    def test_new_mysql_qdrant_readback_and_three_strategies(self):
        """真实存储读回配合假编码/CE，不测神经质量、吞吐或真实向量检索。"""
        database_url = os.environ.get('RETRIEVAL_CANDIDATE_TEST_DATABASE_URL')
        vector_url = os.environ.get('RETRIEVAL_CANDIDATE_TEST_VECTOR_URL')
        if not database_url or not vector_url:
            self.skipTest('未显式配置363专用MySQL/Qdrant；SQLite另测')
        database, vector = urlsplit(database_url), urlsplit(vector_url)
        if (database.scheme != 'mysql' or database.hostname not in {'localhost', '127.0.0.1'}
                or database.port != 33330 or database.path != '/researchloop_test_component_eval_363'
                or database.query or database.fragment
                or vector.scheme != 'http' or vector.hostname not in {'localhost', '127.0.0.1'}
                or vector.port != 6353 or vector.path not in {'', '/'} or vector.username or vector.password
                or vector.query or vector.fragment):
            self.fail('必须使用明确新专用363测试库33330与向量端口6353')
        for target in ('research_loop.providers.embeddings.LocalEmbedding.encode',
                       'research_loop.providers.reranker.LocalReranker.score',
                       'research_loop.providers.reranker.LocalReranker._load',
                       'research_loop.storage.vector_index.VectorIndex.query'):
            guard = patch(target, side_effect=AssertionError('真实神经或相似度查询禁止'))
            guard.start()
            self.addCleanup(guard.stop)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            db = Database(database_url)
            db.ensure_schema()
            seed_tasks(db)
            document, _ = seed_document(db, root, uuid4().hex,
                text='\n\n'.join(f'Language synthetic external paragraph {n}.' for n in range(30)))
            embedding = QueryEmbedding()
            # 专用本机测试直连独立端口，不能继承进程代理而访问其他服务。
            vector_client = httpx.Client(base_url=vector_url, trust_env=False)
            self.addCleanup(vector_client.close)
            vectors = PayloadReadbackVector(vector_url, 'researchloop_eval_363_' + uuid4().hex[:12],
                                           client=vector_client)
            self.addCleanup(vectors.close)
            identities = []
            for name in NAMES:
                strategy = get_retrieval_strategy(name)
                reranker = configured_reranker(name)
                service = RetrievalService(db, embedding, vectors, reranker, strategy=strategy)
                request = RetrievalQuery(query='Language', document_ids=[document['document_id']], limit=8)
                result = service.search('owner', request)
                self.assertEqual(result['status'], 'ready')
                self.assertEqual(result['effective_path'], 'rrf_rerank')
                self.assertEqual(len(reranker.score.call_args.args[1]), 24)
                rows = adapter_chunks(service, document)
                observed = RetrievalAdapter(lambda: embedding, lambda: vectors, lambda: reranker,
                    strategy=strategy).run('Language', rows, 'rrf_rerank', [r['chunk_id'] for r in rows])
                self.assertEqual([h['chunk_id'] for h in result['hits']], observed['ranked_ids'][:8])
                self.assertTrue(all(r['collection'].startswith('researchloop_eval_363_') for r in rows))
                identities.append(result['scope'])
                with self.assertRaises(PermissionError):
                    service.search('other', request)
                reranker.score.side_effect = ValueError('RERANK_INPUT_TOO_LONG')
                self.assertEqual(service.search('owner', request)['effective_path'], 'rrf')
            self.assertEqual(identities[0], identities[1])
            self.assertEqual(identities[0], identities[2])


if __name__ == '__main__':
    unittest.main()
