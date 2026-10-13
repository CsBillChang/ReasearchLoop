"""原文索引确定性测试：替身仅测合同，真实模型/Qdrant另由隔离验收执行。"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import unittest

import httpx

from research_loop.domain.retrieval import ChunkConfig, fingerprint, split_page
from research_loop.services.indexing import IndexingService
from research_loop.storage.mysql import Database
from research_loop.storage.vector_index import VectorIndex, matches, valid_vector


class FakeEmbedding:
    """明确的离线向量替身；不能作为模型质量或本机性能证据。"""
    dimension = 4
    tokenizer_id = 'test-word-offsets-v1'

    def __init__(self, version='test-v1'):
        """为版本隔离提供可控身份和编码计数。"""
        self.identity = {'model_id': 'synthetic-only', 'revision': version, 'dimension': 4}
        self.encoded = 0

    def offsets(self, text):
        """返回真实单词字符位置，包含中文的非空白连续词。"""
        return [m.span() for m in re.finditer(r'\S+', text)]

    def encode(self, texts):
        """生成内容确定的单位向量以验证恢复是否重复计算。"""
        self.encoded += len(texts)
        result = []
        for text in texts:
            raw = [x+1 for x in hashlib.sha256(text.encode()).digest()[:4]]
            norm = math.sqrt(sum(x*x for x in raw))
            result.append([x/norm for x in raw])
        return result


class FakeVector:
    """Qdrant写入后的异常和点丢失替身，不声称进行真实网络调用。"""
    def __init__(self):
        """构建隔离字典，支持仅一次写后异常。"""
        self.points = {}
        self.fail_after_write = False

    def collection(self, model_hash):
        """不同模型使用独立集合。"""
        return 'test_' + model_hash

    def ensure_collection(self, name, dimension):
        """创建空集合但不混入其他版本。"""
        self.points.setdefault(name, {})

    def upsert(self, collection, points):
        """先写入再抛异常，模拟响应前连接中断。"""
        self.points[collection].update({p['id']: copy.deepcopy(p) for p in points})
        if self.fail_after_write:
            self.fail_after_write = False
            raise OSError('simulated after write')

    def retrieve(self, collection, ids):
        """仅返回明确请求的点。"""
        return {i: copy.deepcopy(self.points[collection][i]) for i in ids if i in self.points[collection]}

    def count(self, collection, index_id):
        """按索引隔离计数。"""
        return sum(p['payload']['index_id'] == index_id for p in self.points[collection].values())


def seed_document(db, root, key='one', *, jina=False, text=None, parser_version='1'):
    """创建可核对hash的合成原件/解析快照，不能作为论文解析质量证据。"""
    source = ('synthetic PDF bytes ' + key).encode()
    texts = [text or '# Methods\n\n' + ' '.join('token'+str(i) for i in range(50)), 'Results\n\nAccuracy 0.8 ± 0.1.']
    if jina:
        texts, source = [texts[0]], texts[0].encode()
    digest = hashlib.sha256(source).hexdigest()
    doc_id = 'doc:' + fingerprint([key, digest, parser_version])[:32]
    pages = [{'evidence_id': doc_id+':'+str(n), 'document_id': doc_id,
        'page_number': None if jina else n+1, 'text': body, 'parsed_markdown': body} for n, body in enumerate(texts)]
    cache = root/'cache'/'parsed'/f'{doc_id[4:]}.json'
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({'total_pages': None if jina else len(pages), 'pages': pages}))
    pdf = root/'cache'/'pdf'/f'{digest}.pdf'
    pdf.parent.mkdir(parents=True, exist_ok=True)
    if not jina:
        pdf.write_bytes(source)
    doc = dict(document_id=doc_id, paper_id='paper:'+key, content_sha256=digest,
        parser_name='jina' if jina else 'pymupdf', parser_version=parser_version,
        config_sha256=fingerprint('test-config'), total_pages=None if jina else len(pages), cache_path=str(cache))
    db.save_document(doc)
    for p in pages:
        db.save_page(p)
        db.mark_task_evidence('owner', p['evidence_id'])
    return doc, pages


class RetrievalContractTests(unittest.TestCase):
    """覆盖权限、版本、原文定位及两种派生索引的恢复边界。"""
    def setUp(self):
        """每例使用独立SQLite、目录和两个替身。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = Database('sqlite:///' + str(self.root/'test.db'))
        self.db.ensure_schema()
        for task in ['owner', 'other']:
            self.db.create_task(task, task, task, '阅读原文', 'deny', [])
        self.doc, self.pages = seed_document(self.db, self.root)
        self.embedding, self.vector = FakeEmbedding(), FakeVector()
        self.service = IndexingService(self.db, self.embedding, self.vector, ChunkConfig(max_tokens=16))

    def build(self, pages=None):
        """通过实际服务入口构建指定页。"""
        return self.service.build('owner', self.doc['document_id'], [p['evidence_id'] for p in (pages or self.pages)], 2)

    def test_duplicate_import_and_order_reuses_index_and_vectors(self):
        first = self.build()
        second = self.build(self.pages[::-1])
        self.assertEqual(first['index_id'], second['index_id'])
        self.assertEqual((second['encoded_chunks'], second['upserted_points']), (0, 0))
        self.assertEqual(second['vector_status'], 'ready')

    def test_partial_write_resumes_receipts_without_duplicate_points(self):
        self.vector.fail_after_write = True
        first = self.build()
        self.assertEqual((first['bm25_status'], first['vector_status']), ('ready', 'partial'))
        second = self.build()
        self.assertEqual(second['vector_status'], 'ready')
        self.assertEqual(self.embedding.encoded, second['chunk_count'])
        self.assertEqual(first['upserted_points'] + second['upserted_points'], second['chunk_count']-2)

    def test_missing_qdrant_points_rebuild_from_mysql_receipt(self):
        first = self.build()
        points = self.vector.points[first['manifest']['collection']]
        del points[next(iter(points))]
        second = self.build()
        self.assertEqual((second['encoded_chunks'], second['upserted_points']), (0, 1))

    def test_terms_can_be_rebuilt_from_verified_source(self):
        first = self.build()
        self.db.execute("UPDATE evidence_chunks SET terms_json='{}',token_count=0")
        with self.assertRaisesRegex(ValueError, 'INDEX_TERMS_MISMATCH'):
            self.service.authorized_chunks('owner', first['index_id'])
        second = self.build()
        self.assertEqual(second['vector_status'], 'ready')
        self.service.authorized_chunks('owner', second['index_id'])

    def test_other_task_cannot_build_or_read_cache(self):
        with self.assertRaises(PermissionError):
            self.service.build('other', self.doc['document_id'], [p['evidence_id'] for p in self.pages])
        first = self.build()
        with self.assertRaises(PermissionError):
            self.service.authorized_chunks('other', first['index_id'])

    def test_known_paper_is_not_a_page_grant(self):
        self.db.grant_paper('other', self.doc['paper_id'], 'owner')
        with self.assertRaises(PermissionError):
            self.service.build('other', self.doc['document_id'], [self.pages[0]['evidence_id']])

    def test_user_scope_is_rechecked_after_index_ready(self):
        first = self.build()
        self.db.execute("UPDATE tasks SET prompt='只看第1页' WHERE task_id='owner'")
        with self.assertRaises(PermissionError):
            self.build()
        with self.assertRaises(PermissionError):
            self.service.authorized_chunks('owner', first['index_id'])
        scoped = self.build(self.pages[:1])
        self.assertEqual(len(scoped['manifest']['pages']), 1)
        self.assertNotEqual(scoped['index_id'], first['index_id'])

    def test_unknown_page_keeps_text_offsets_not_page_one(self):
        doc, pages = seed_document(self.db, self.root, 'jina', jina=True)
        result = self.service.build('owner', doc['document_id'], [pages[0]['evidence_id']])
        output = self.service.authorized_chunks('owner', result['index_id'])
        self.assertTrue(all(c['chunk']['page_number'] is None for c in output['chunks']))
        self.assertEqual(result['manifest']['source']['hash_kind'], 'extracted_text')

    def test_changed_model_creates_new_collection_without_destroying_old(self):
        first = self.build()
        old = copy.deepcopy(self.vector.points)
        self.service.embedding = FakeEmbedding('test-v2')
        second = self.build()
        self.assertNotEqual(first['index_id'], second['index_id'])
        self.assertNotEqual(first['manifest']['collection'], second['manifest']['collection'])
        self.assertEqual(old[first['manifest']['collection']], self.vector.points[first['manifest']['collection']])

    def test_changed_source_or_parser_produces_new_chunks(self):
        first = self.build()
        for key, version in [('new-content', '1'), ('one', '2')]:
            doc, pages = seed_document(self.db, self.root, key, parser_version=version)
            result = self.service.build('owner', doc['document_id'], [p['evidence_id'] for p in pages])
            self.assertTrue(set(result['manifest']['chunk_ids']).isdisjoint(first['manifest']['chunk_ids']))

    def test_changed_splitter_is_an_explicit_version(self):
        first = self.build()
        self.service.config = ChunkConfig(max_tokens=32)
        second = self.build()
        self.assertNotEqual(first['index_id'], second['index_id'])

    def test_raw_file_tampering_is_rejected(self):
        path = self.root/'cache'/'pdf'/(self.doc['content_sha256']+'.pdf')
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SOURCE_HASH_MISMATCH'):
            self.build()

    def test_page_cache_tampering_is_rejected(self):
        first = self.build()
        self.db.execute("UPDATE page_evidence SET text='changed' WHERE evidence_id=?", (self.pages[0]['evidence_id'],))
        with self.assertRaisesRegex(ValueError, 'PAGE_CACHE_MISMATCH'):
            self.service.authorized_chunks('owner', first['index_id'])

    def test_cross_document_reference_is_rejected(self):
        doc, pages = seed_document(self.db, self.root, 'another')
        with self.assertRaisesRegex(ValueError, 'DOCUMENT_MISMATCH'):
            self.build(pages)

    def test_wrong_vector_receipt_never_promotes_ready(self):
        first = self.build()
        self.db.execute("UPDATE evidence_index_members SET embedding_json='[0,0,0,0]' WHERE index_id=?", (first['index_id'],))
        second = self.build()
        self.assertEqual(second['vector_status'], 'partial')
        self.assertEqual(second['last_error'], 'INDEX_EMBEDDING_RECEIPT_INVALID')

    def test_exact_chunk_substrings_cover_all_nonspace_characters(self):
        first = self.build()
        chunks = self.service.authorized_chunks('owner', first['index_id'])['chunks']
        for page in self.pages:
            coverage = set()
            for record in chunks:
                c = record['chunk']
                if c['evidence_id'] != page['evidence_id']:
                    continue
                self.assertEqual(c['text'], page['text'][c['char_start']:c['char_end']])
                span = set(range(c['char_start'], c['char_end']))
                self.assertFalse(coverage & span)
                coverage |= span
            self.assertTrue(all(i in coverage for i, char in enumerate(page['text']) if not char.isspace()))

    def test_unknown_section_is_not_guessed_from_short_lines(self):
        page = {**self.pages[0], 'text': 'Widget baseline\n0.54 0.66 123\n\nUnlabelled prose.'}
        chunks = split_page(page, self.doc, ChunkConfig(), 'test', self.embedding.offsets)
        self.assertTrue(all(c.section is None for c in chunks))

    def test_no_silent_empty_or_invalid_batch_build(self):
        with self.assertRaises(ValueError):
            self.service.build('owner', self.doc['document_id'], [], 4)
        with self.assertRaises(ValueError):
            self.service.build('owner', self.doc['document_id'], [self.pages[0]['evidence_id']], 9)

    def test_qdrant_namespace_and_collection_configuration_fail_closed(self):
        with self.assertRaises(ValueError):
            VectorIndex('http://localhost', '../another')
        def response(request):
            """模拟现有集合维度错误，不允许put覆盖。"""
            self.assertEqual(request.method, 'GET')
            return httpx.Response(200, json={'status': 'ok', 'result': {'config': {'params': {'vectors': {'size': 999, 'distance': 'Cosine'}}}}})
        with httpx.Client(base_url='http://localhost', transport=httpx.MockTransport(response)) as client:
            with self.assertRaisesRegex(ValueError, 'CONFIG_CONFLICT'):
                VectorIndex('http://localhost', 'researchloop_test', client).ensure_collection('test', 4)

    def test_vector_readback_requires_matching_identity_and_numbers(self):
        expected = {'id': 'a', 'vector': [1., 0.], 'payload': {'source': 'one'}}
        self.assertTrue(matches(expected, expected))
        self.assertFalse(matches({**expected, 'payload': {'source': 'other'}}, expected))
        self.assertFalse(valid_vector([float('nan'), 0], 2))
        self.assertFalse(valid_vector([0, 0], 2))
        self.assertFalse(valid_vector([True, False], 2))

    def test_modified_location_cannot_reuse_old_chunk_identity(self):
        """篡改为同页另一合法子串也不能沿用旧chunk ID，失败不保留BM25 ready。"""
        first = self.build()
        member = self.service.store.members(first['index_id'])[0]
        c = member['chunk']
        c['char_start'] += 1
        c['text'] = c['text'][1:]
        c['text_sha256'] = fingerprint(c['text'])
        self.db.execute('UPDATE evidence_chunks SET chunk_json=? WHERE chunk_id=?', (json.dumps(c), c['chunk_id']))
        second = self.build()
        self.assertEqual(second['vector_status'], 'partial')
        self.assertEqual(second['bm25_status'], 'failed')
        self.assertEqual(second['last_error'], 'INDEX_CHUNK_SOURCE_MISMATCH')
