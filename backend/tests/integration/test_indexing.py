"""AG7.1真实隔离索引读回；需要显式库和已完成报告，默认不访问生产数据。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest
from urllib.parse import urlparse

import fitz

from research_loop.providers.embeddings import LocalEmbedding
from research_loop.services.indexing import IndexingService
from research_loop.storage.mysql import Database
from research_loop.storage.vector_index import VectorIndex, matches


class MySQLIndexingTests(unittest.TestCase):
    """核验最终代码读取真实原件/SQL/向量的一致性，不产生生成模型请求。"""

    @classmethod
    def setUpClass(cls):
        """只使用显式AG7.1测试配置；拒绝用默认生产URL凑通过。"""
        url = os.getenv('AG71_TEST_DATABASE_URL', '')
        report = os.getenv('AG71_REAL_REPORT')
        manifest = os.getenv('AG71_MODEL_MANIFEST')
        if not url or not report or not manifest:
            raise unittest.SkipTest('explicit AG71_TEST_DATABASE_URL/REAL_REPORT/MODEL_MANIFEST required')
        if not urlparse(url).path.startswith('/researchloop_test_ag71_'):
            raise ValueError('isolated AG7.1 database required')
        cls.directory = Path(report)
        cls.freeze = json.loads((cls.directory/'freeze.json').read_text())
        cls.report = json.loads((cls.directory/'result.json').read_text())
        cls.docs = json.loads((cls.directory/'input-documents.json').read_text())
        cls.task_id = cls.freeze['task_id']
        cls.db = Database(url)
        cls.vector = VectorIndex(os.getenv('AG71_QDRANT_URL', 'http://127.0.0.1:6339'), cls.freeze['namespace'])
        cls.model = LocalEmbedding(manifest)
        cls.service = IndexingService(cls.db, cls.model, cls.vector)

    @classmethod
    def tearDownClass(cls):
        """释放测试客户端，不停止本机共享服务。"""
        cls.vector.close()

    def test_pdf_pages_and_stored_chunk_locations_match_raw_file(self):
        """重新从物理PDF逐页提取，只核对文本位置，不声称表格/语义理解通过。"""
        for case in self.report['cases']:
            result = self.service.authorized_chunks(self.task_id, case['index_id'])
            document = self.db.fetchone('SELECT * FROM documents WHERE document_id=?', (result['document_id'],))
            path = Path(document['cache_path']).parent.parent/'pdf'/(document['content_sha256']+'.pdf')
            with fitz.open(path) as pdf:
                self.assertEqual(len(pdf), case['pages'])
                text = [page.get_text('text') for page in pdf]
                for record in result['chunks']:
                    chunk = record['chunk']
                    self.assertEqual(chunk['text'], text[chunk['page_number']-1][chunk['char_start']:chunk['char_end']])

    def test_reimport_final_code_encodes_and_writes_zero_points(self):
        """最新代码对同身份重复建库没有新的本机编码或向量写入。"""
        for doc in self.docs:
            result = self.service.build(self.task_id, doc['document_id'], doc['evidence_ids'])
            self.assertEqual((result['bm25_status'], result['vector_status']), ('ready', 'ready'))
            self.assertEqual((result['encoded_chunks'], result['upserted_points']), (0, 0))

    def test_every_real_point_matches_mysql_receipt(self):
        """逐点验证source/page/hash/model身份与实际向量，不只检查数量。"""
        for case in self.report['cases']:
            result = self.service.authorized_chunks(self.task_id, case['index_id'])
            members = self.service.store.members(case['index_id'])
            points = [self.service._point(case['index_id'], result['manifest'], m) for m in members]
            actual = self.vector.retrieve(result['manifest']['collection'], [p['id'] for p in points])
            self.assertTrue(all(matches(actual.get(p['id']), p) for p in points))
            self.assertEqual(len(actual), case['chunks'])

    def test_real_task_without_page_grant_is_denied(self):
        """保存策略与证据授权分离：即使索引已ready也不能跨任务回读。"""
        for case in self.report['cases']:
            with self.assertRaises(PermissionError):
                self.service.authorized_chunks(self.task_id+'-stranger', case['index_id'])
        self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM model_calls')['n'], 0)
