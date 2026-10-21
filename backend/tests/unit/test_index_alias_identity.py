"""附件别名不改变原文版本：新旧索引复用及任务/页权限的零模型回归。"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlparse
from uuid import uuid4

from research_loop.domain.retrieval import RetrievalQuery
from research_loop.services.indexing import IndexingService
from research_loop.services.retrieval import RetrievalService
from research_loop.storage.mysql import Database
from research_loop.storage.vector_index import VectorIndex
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs
from research_loop.tools.registry import ToolRegistry
from tests.unit.test_hybrid_retrieval import QueryEmbedding, QueryVector, TestReranker
from tests.unit.test_retrieval_contracts import seed_document


class LegacySourceIndex(IndexingService):
    """仅构建冻结格式的旧manifest；不模拟真实编码或语义质量。"""

    def _source_pages(self, task_id, document_id, evidence_ids):
        """保留旧版将附件别名放入source的行为，生成兼容性fixture。"""
        source, pages = super()._source_pages(task_id, document_id, evidence_ids)
        source['paper_id'] = self.db.fetchone(
            'SELECT paper_id FROM documents WHERE document_id=?', (document_id,))['paper_id']
        return source, pages


class IndexAliasIdentityTests(unittest.TestCase):
    """默认SQLite/向量替身；显式隔离配置可复用同组案例验证MySQL/Qdrant。"""

    def setUp(self):
        """创建专属原件和任务，不连接默认生产配置或生成模型。"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        url = os.getenv('INDEX_ALIAS_TEST_DATABASE_URL', '')
        if url and not urlparse(url).path.startswith('/researchloop_test_index_alias_'):
            raise ValueError('explicit isolated index-alias database required')
        self.db = Database(url or 'sqlite:///' + str(self.root/'test.db'))
        self.db.ensure_schema()
        self.key = uuid4().hex
        self.owner = 'alias-owner-' + self.key
        self.other = 'alias-other-' + self.key
        self.stranger = 'alias-stranger-' + self.key
        for task in (self.owner, self.other, self.stranger):
            self.db.create_task(task, task, task, '阅读全文，不保存', 'deny', [])
        self.doc, self.pages = seed_document(self.db, self.root, self.key,
                                            text='Methods\n\nLanguage embeddings identify unseen objects.')
        # 公共fixture默认登记owner；此组显式移到唯一任务，避免隔离库测试间共享授权。
        for page in self.pages:
            self.db.execute('DELETE FROM task_evidence WHERE task_id=? AND evidence_id=?',
                            ('owner', page['evidence_id']))
            self.db.mark_task_evidence(self.owner, page['evidence_id'])
        self.embedding = QueryEmbedding()
        qdrant = os.getenv('INDEX_ALIAS_TEST_QDRANT_URL', '')
        namespace = os.getenv('INDEX_ALIAS_TEST_NAMESPACE', '')
        if qdrant:
            if not url or not namespace.startswith('researchloop_test_index_alias_'):
                raise ValueError('isolated database and vector namespace required')
            self.vector = VectorIndex(qdrant, namespace)
            self.addCleanup(self.vector.close)
        else:
            self.vector = QueryVector()
        self.service = IndexingService(self.db, self.embedding, self.vector)
        self.retrieval = RetrievalService(self.db, self.embedding, self.vector, TestReranker())

    def build(self, legacy=False):
        """通过真实构建服务生成新格式或带历史别名的索引。"""
        service = LegacySourceIndex(self.db, self.embedding, self.vector) if legacy else self.service
        return service.build(self.owner, self.doc['document_id'], [p['evidence_id'] for p in self.pages])

    def reattach(self, alias='input-B', grant=True):
        """复现同PDF读入新附件后更新来源标签，但权限单独授予的实际路径。"""
        self.db.save_document({**self.doc, 'paper_id': alias})
        if grant:
            for page in self.pages:
                self.db.mark_task_evidence(self.other, page['evidence_id'])

    def search(self, task=None, document_id=None):
        """沿用Agent工具入口和实际三路检索服务，编码与重排为确定性替身。"""
        ctx = ToolContext(task_id=task or self.other, save_policy='deny', attachments={},
                          services={'retrieval': self.retrieval}, role='supervisor')
        return ToolRegistry(specs()).execute('retrieve_evidence', {
            'query': 'Language', 'document_ids': [document_id or self.doc['document_id']]}, ctx)

    def test_new_manifest_reuses_same_content_across_aliases(self):
        """新身份不包含别名；重新附加仅编码query，不复制文档向量。"""
        old = self.build()
        self.assertNotIn('paper_id', old['manifest']['source'])
        count = self.embedding.encoded
        self.reattach()
        result = self.search()
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['scope'][0]['index_id'], old['index_id'])
        self.assertEqual(self.embedding.encoded-count, 1)
        self.assertEqual(result.data['status'], 'ready')
        self.assertTrue(result.data['hits'])

    def test_legacy_index_and_chunks_reused_without_history_rewrite(self):
        """旧索引hash和片段保持，别名不同不再触发版本拒绝。"""
        old = self.build(legacy=True)
        row = self.db.fetchone('SELECT * FROM evidence_index_builds WHERE index_id=?', (old['index_id'],))
        members = self.service.store.members(old['index_id'])
        self.reattach()
        result = self.search()
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['scope'][0]['index_id'], old['index_id'])
        self.assertEqual(self.db.fetchone('SELECT * FROM evidence_index_builds WHERE index_id=?',
                                         (old['index_id'],)), row)
        self.assertEqual(self.service.store.members(old['index_id']), members)

    def test_explicit_rebuild_reuses_legacy_receipts_and_points(self):
        """显式重复建库也不改旧身份或重复编码/写入向量。"""
        old = self.build(legacy=True)
        self.reattach()
        rebuilt = self.service.build(self.other, self.doc['document_id'],
                                     [p['evidence_id'] for p in self.pages][::-1])
        self.assertEqual(rebuilt['index_id'], old['index_id'])
        self.assertEqual(rebuilt['manifest'], old['manifest'])
        self.assertEqual((rebuilt['encoded_chunks'], rebuilt['upserted_points']), (0, 0))

    def test_alias_change_during_search_does_not_invalidate_current_pages(self):
        """模拟另一任务在重排期间重读同PDF；返回前仍按内容/授权验证。"""
        old = self.build(legacy=True)
        self.reattach()
        class ReattachReranker(TestReranker):
            """替身仅在服务返回前改变来源标签。"""
            def score(inner, query, passages):
                """保持原内容，模拟并发读取的确定性时序。"""
                self.reattach('input-C')
                return super().score(query, passages)
        self.retrieval.reranker = ReattachReranker()
        result = self.search()
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['scope'][0]['index_id'], old['index_id'])

    def test_same_alias_with_different_pdf_does_not_reuse_version(self):
        """附件名相同但PDF内容不同，任务不能通过旧授权读取新版本。"""
        old = self.build(legacy=True)
        self.reattach()
        other, pages = seed_document(self.db, self.root, self.key+'-v2', text='Changed material.')
        self.db.save_document({**other, 'paper_id': 'input-B'})
        denied = self.search(document_id=other['document_id'])
        self.assertFalse(denied.ok)
        self.assertIn('NOT_AUTHORIZED', denied.message)
        for page in pages:
            self.db.mark_task_evidence(self.other, page['evidence_id'])
        result = self.search(document_id=other['document_id'])
        self.assertTrue(result.ok, result.message)
        self.assertNotEqual(result.data['scope'][0]['index_id'], old['index_id'])
        self.assertEqual({h['source_sha256'] for h in result.data['hits']}, {other['content_sha256']})

    def test_parser_revision_creates_distinct_index(self):
        """相同PDF换解析版本时不能跨版本复用旧片段。"""
        old = self.build(legacy=True)
        other, pages = seed_document(self.db, self.root, self.key, parser_version='2')
        self.assertEqual(other['content_sha256'], self.doc['content_sha256'])
        for page in pages:
            self.db.mark_task_evidence(self.other, page['evidence_id'])
        result = self.search(document_id=other['document_id'])
        self.assertTrue(result.ok, result.message)
        self.assertNotEqual(result.data['scope'][0]['index_id'], old['index_id'])

    def test_unread_same_pdf_has_no_implicit_permission(self):
        """同PDF和合法附件标签都不能替代task_evidence的页授权。"""
        old = self.build(legacy=True)
        self.reattach(grant=False)
        self.assertFalse(self.search().ok)
        with self.assertRaises(PermissionError):
            self.service.authorized_chunks(self.other, old['index_id'])
        self.assertFalse(self.search(self.stranger).ok)

    def test_page_scope_stays_narrow_after_alias_change(self):
        """只授权/只允许第一页时，不复用包含第二页的旧索引。"""
        old = self.build(legacy=True)
        self.reattach(grant=False)
        self.db.mark_task_evidence(self.other, self.pages[0]['evidence_id'])
        self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', ('只看第1页，不保存', self.other))
        result = self.search()
        self.assertTrue(result.ok, result.message)
        self.assertEqual({h['page_number'] for h in result.data['hits']}, {1})
        self.assertNotEqual(result.data['scope'][0]['index_id'], old['index_id'])
        with self.assertRaises(PermissionError):
            self.service.authorized_chunks(self.other, old['index_id'])

    def test_other_source_fields_still_fail_closed(self):
        """仅兼容来源别名，解析配置与页数等真实身份变化仍拒绝旧版本。"""
        old = self.build(legacy=True)
        self.reattach()
        for field, value in (('parser_version', 'changed'), ('config_sha256', 'f'*64), ('total_pages', 3)):
            with self.subTest(field=field):
                self.db.execute('UPDATE documents SET '+field+'=? WHERE document_id=?',
                                (value, self.doc['document_id']))
                with self.assertRaisesRegex(ValueError, 'INDEX_SOURCE_VERSION_CHANGED'):
                    self.service.authorized_chunks(self.other, old['index_id'])
                self.db.save_document({**self.doc, 'paper_id':'input-B'})

    def test_tampered_legacy_manifest_is_not_alias_compatibility(self):
        """即使只篡改旧paper_id，原manifest hash也必须匹配原索引身份。"""
        old = self.build(legacy=True)
        changed = copy.deepcopy(old['manifest'])
        changed['source']['paper_id'] = 'tampered'
        self.db.execute('UPDATE evidence_index_builds SET manifest_json=? WHERE index_id=?',
                        (json.dumps(changed), old['index_id']))
        self.reattach()
        with self.assertRaisesRegex(ValueError, 'INDEX_SOURCE_VERSION_CHANGED'):
            self.service.authorized_chunks(self.other, old['index_id'])
        with self.assertRaisesRegex(ValueError, 'INDEX_SOURCE_VERSION_CHANGED'):
            self.service.build(self.other, self.doc['document_id'], [p['evidence_id'] for p in self.pages])

    def test_page_text_and_chunk_corruption_still_rejected(self):
        """别名兼容不能掩盖原文缓存或索引片段损坏。"""
        old = self.build(legacy=True)
        self.reattach()
        self.db.execute('UPDATE page_evidence SET text=? WHERE evidence_id=?',
                        ('corrupted', self.pages[0]['evidence_id']))
        with self.assertRaisesRegex(ValueError, 'INDEX_PAGE_CACHE_MISMATCH'):
            self.service.authorized_chunks(self.other, old['index_id'])
        self.db.save_page(self.pages[0])
        member = self.service.store.members(old['index_id'])[0]
        chunk = {**member['chunk'], 'text': 'corrupted'}
        self.db.execute('UPDATE evidence_chunks SET chunk_json=? WHERE chunk_id=?',
                        (json.dumps(chunk), member['chunk_id']))
        with self.assertRaisesRegex(ValueError, 'INDEX_CHUNK_SOURCE_MISMATCH'):
            self.service.authorized_chunks(self.other, old['index_id'])
