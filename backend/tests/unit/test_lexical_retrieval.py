"""词法候选路线的合成合同，不读取论文、标签或调用真实模型。"""
from copy import deepcopy
from hashlib import sha256
import unittest

from research_loop.domain.lexical_retrieval import LexicalRetriever


def chunk(cid, document_id, text, **extra):
    """生成完整且可核对的合成原文身份。"""
    return {'chunk_id': cid, 'key': 'key-' + cid, 'document_id': document_id,
        'index_id': 'index-' + document_id, 'evidence_id': 'page-' + document_id,
        'source_sha256': 'a' * 64, 'text_sha256': sha256(text.encode()).hexdigest(),
        'page_number': 1, 'char_start': 0, 'char_end': len(text), 'text': text, **extra}


def catalog(*pairs):
    """只向候选模块提供公开目录字段。"""
    return [{'document_id': doc, 'title': title} for doc, title in pairs]


class LexicalRetrieverTests(unittest.TestCase):
    """覆盖明确名称、跨篇、歧义和来源授权边界。"""

    def setUp(self):
        self.rows = [chunk('alpha', 'a', 'A method for training models.'),
            chunk('beta', 'b', 'A method for inference and architecture.'),
            chunk('other', 'c', 'Other content about experiment and dataset.')]
        self.catalog = catalog(('a', 'AlphaNet: A New Architecture'),
            ('b', 'BetaNet: An Efficient System'), ('c', 'Other Study'))
        self.retriever = LexicalRetriever(self.rows, self.catalog)

    def test_baseline_uses_original_body_query(self):
        out = self.retriever.select('training', ['a', 'b', 'c'], 'bm25_rerank')
        self.assertEqual([r['chunk_id'] for r in out['candidates']], ['alpha'])
        self.assertEqual(out['search_query'], 'training')
        self.assertEqual(out['trace']['routing'], 'baseline')

    def test_exact_name_is_removed_and_chinese_concept_expanded(self):
        out = self.retriever.select('AlphaNet方法与训练', ['a', 'b', 'c'], 'located_bm25_rerank')
        self.assertEqual(out['local_document_ids'], ['a'])
        self.assertEqual(out['trace']['routing'], 'explicit_paper')
        self.assertNotIn('alphanet', out['search_query'])
        self.assertIn('method', out['trace']['expanded_terms'])
        self.assertIn('training', out['trace']['expanded_terms'])
        self.assertEqual(out['candidates'][0]['chunk_id'], 'alpha')
        self.assertEqual(out['candidates'][0]['text'], self.rows[0]['text'])
        self.assertEqual(out['trace']['original_query'], 'AlphaNet方法与训练')

    def test_identifier_boundary_rejects_partial_name(self):
        out = self.retriever.select('AlphaNets方法', ['a', 'b', 'c'], 'located_bm25_rerank')
        self.assertEqual(out['local_document_ids'], [])
        self.assertEqual(out['trace']['routing'], 'no_paper_fallback')

    def test_full_title_and_case_normalization(self):
        out = self.retriever.select('ALPHANET: A NEW ARCHITECTURE 方法', ['a', 'b', 'c'], 'located_bm25_rerank')
        self.assertEqual(out['local_document_ids'], ['a'])
        self.assertEqual(out['trace']['alias_matches'][0]['alias'], 'alphanet: a new architecture')

    def test_cross_paper_round_robin_preserves_each_paper(self):
        rows = [chunk(f'a-{i}', 'a', f'method training alpha {i}') for i in range(22)]
        rows += [chunk(f'b-{i}', 'b', f'method inference beta {i}') for i in range(22)]
        retriever = LexicalRetriever(rows, self.catalog[:2])
        out = retriever.select('AlphaNet与BetaNet方法', ['a', 'b'], 'located_bm25_rerank')
        self.assertEqual(out['local_document_ids'], ['a', 'b'])
        self.assertEqual(len(out['candidates']), 24)
        self.assertEqual([r['document_id'] for r in out['candidates'][:18]], ['a', 'b'] * 9)
        self.assertEqual(len({r['chunk_id'] for r in out['candidates']}), 24)

    def test_original_global_reserve_retains_outside_paper(self):
        rows = [chunk(f'a-{i}', 'a', f'method training {i}') for i in range(25)]
        rows += [chunk('outside', 'b', 'AlphaNet original clue')]
        retriever = LexicalRetriever(rows, self.catalog[:2])
        out = retriever.select('AlphaNet方法', ['a', 'b'], 'located_bm25_rerank')
        self.assertIn('outside', [r['chunk_id'] for r in out['candidates']])
        self.assertEqual(out['trace']['scope_document_ids'], ['a', 'b'])
        self.assertEqual(out['trace']['authorized_chunk_count'], 26)

    def test_ambiguous_short_name_does_not_hard_filter(self):
        retriever = LexicalRetriever(self.rows[:2], catalog(
            ('a', 'SharedName: Alpha Method'), ('b', 'SharedName: Beta Method')))
        out = retriever.select('SharedName方法', ['a', 'b'], 'located_bm25_rerank')
        self.assertEqual(out['local_document_ids'], [])
        self.assertEqual(out['trace']['routing'], 'ambiguous_fallback')
        self.assertEqual({r['document_id'] for r in out['candidates']}, {'a', 'b'})

    def test_mixed_ambiguous_and_unique_names_preserve_all_names_on_fallback(self):
        retriever = LexicalRetriever(self.rows, catalog(
            ('a', 'SharedName: Alpha Method'), ('b', 'SharedName: Beta Method'),
            ('c', 'SoloNet: Distinct Architecture')))
        out = retriever.select('SharedName与SoloNet方法', ['a', 'b', 'c'], 'located_bm25_rerank')
        self.assertEqual(out['local_document_ids'], [])
        self.assertEqual(out['trace']['routing'], 'ambiguous_fallback')
        self.assertIn('sharedname', out['search_query'])
        self.assertIn('solonet', out['search_query'])
        self.assertIn('method', out['trace']['expanded_terms'])
        self.assertEqual(out['trace']['scope_document_ids'], ['a', 'b', 'c'])

    def test_chinese_multiline_title_is_public_routing_metadata_only(self):
        retriever = LexicalRetriever(self.rows[:1], catalog(
            ('a', '模型甲：\n一种公开方法')))
        out = retriever.select('模型甲： 一种公开方法的方法', ['a'], 'located_bm25_rerank')
        self.assertEqual(out['local_document_ids'], ['a'])
        self.assertEqual(out['trace']['alias_matches'][0]['alias'], '模型甲: 一种公开方法')
        self.assertEqual(out['candidates'][0]['text'], self.rows[0]['text'])
        self.assertNotIn('模型甲', out['candidates'][0]['text'])

    def test_full_title_resolves_overlapping_ambiguous_prefix(self):
        retriever = LexicalRetriever(self.rows[:2], catalog(
            ('a', 'SharedName: Alpha Method'), ('b', 'SharedName: Beta Method')))
        out = retriever.select('SharedName: Alpha Method方法', ['a', 'b'], 'located_bm25_rerank')
        self.assertEqual(out['local_document_ids'], ['a'])

    def test_generic_acronym_does_not_identify_paper(self):
        retriever = LexicalRetriever(self.rows[:2], catalog(
            ('a', 'AlphaNet: CLIP and LLM Architecture'), ('b', 'BetaNet: DETR VQA and VLM')))
        out = retriever.select('CLIP LLM DETR VQA VLM 方法', ['a', 'b'], 'located_bm25_rerank')
        self.assertEqual(out['local_document_ids'], [])

    def test_no_explicit_name_uses_expanded_global_query(self):
        out = self.retriever.select('实验和数据集', ['a', 'b', 'c'], 'located_bm25_rerank')
        self.assertEqual(out['local_document_ids'], [])
        self.assertEqual([r['chunk_id'] for r in out['candidates']], ['other'])
        self.assertIn('dataset', out['search_query'])

    def test_only_existing_explicit_heading_is_used(self):
        rows = [chunk('yes', 'a', 'unrelated', section='Experimental Setup', section_origin='explicit_heading'),
                chunk('no', 'b', 'unrelated', section='Experimental Setup', section_origin='guessed')]
        retriever = LexicalRetriever(rows, self.catalog[:2])
        out = retriever.select('实验', ['a', 'b'], 'located_bm25_rerank')
        self.assertEqual([r['chunk_id'] for r in out['candidates']], ['yes'])
        baseline = retriever.select('experimental', ['a', 'b'], 'bm25_rerank')
        self.assertEqual(baseline['candidates'], [])

    def test_zero_match_is_empty_and_never_filled_arbitrarily(self):
        for route in ('bm25_rerank', 'located_bm25_rerank'):
            out = self.retriever.select('xyzzy', ['a', 'b'], route)
            self.assertEqual(out['candidates'], [])

    def test_authorized_scope_limits_routing_and_candidates(self):
        out = self.retriever.select('AlphaNet方法', ['b'], 'located_bm25_rerank')
        self.assertEqual(out['local_document_ids'], [])
        self.assertTrue(all(r['document_id'] == 'b' for r in out['candidates']))
        self.assertEqual(out['trace']['scope_document_ids'], ['b'])

    def test_missing_duplicate_or_outside_scope_is_rejected(self):
        for scope in (None, [], ['a', 'a'], ['unknown'], 'a', [True]):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                self.retriever.select('method', scope, 'bm25_rerank')

    def test_identity_hash_offsets_and_duplicate_keys_are_rejected(self):
        invalid = []
        for field, value in [('text', 'tampered'), ('char_end', 999), ('char_start', -1),
                             ('page_number', True), ('source_sha256', 'bad'), ('chunk_id', '')]:
            rows = deepcopy(self.rows); rows[0][field] = value; invalid.append(rows)
        invalid.append([*deepcopy(self.rows), deepcopy(self.rows[0])])
        rows = deepcopy(self.rows); rows[1]['key'] = rows[0]['key']; invalid.append(rows)
        for rows in invalid:
            with self.subTest(rows=rows[0]['chunk_id']), self.assertRaises(ValueError):
                LexicalRetriever(rows, self.catalog)

    def test_mixed_versions_and_conflicting_page_identity_are_rejected(self):
        rows = [chunk('one', 'a', 'method'), chunk('two', 'a', 'training')]
        for field, value in [('source_sha256', 'b' * 64), ('index_id', 'other-index'),
                             ('evidence_id', 'other-evidence')]:
            changed = deepcopy(rows); changed[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                LexicalRetriever(changed, self.catalog[:1])
        changed = deepcopy(rows); changed[1]['page_number'] = 2
        with self.assertRaises(ValueError):
            LexicalRetriever(changed, self.catalog[:1])

    def test_catalog_rejects_labels_and_missing_or_duplicate_documents(self):
        for entries in [self.catalog[:-1], [*self.catalog, self.catalog[0]],
                        [{**self.catalog[0], 'gold_chunk_id': 'alpha'}, *self.catalog[1:]]]:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                LexicalRetriever(self.rows, entries)

    def test_inputs_and_frozen_snapshot_are_not_mutated(self):
        rows, entries = deepcopy(self.rows), deepcopy(self.catalog)
        retriever = LexicalRetriever(rows, entries)
        before = deepcopy((rows, entries))
        first = retriever.select('方法', ['a', 'b', 'c'], 'located_bm25_rerank')
        self.assertEqual((rows, entries), before)
        rows[0]['text'] = 'mutated by caller'
        first['candidates'][0]['text'] = 'mutated result'
        second = retriever.select('方法', ['a', 'b', 'c'], 'located_bm25_rerank')
        self.assertEqual(second['candidates'][0]['text'], self.rows[0]['text'])

    def test_ties_use_frozen_input_order_and_scope_order_cannot_change_them(self):
        rows = [chunk('second', 'b', 'method'), chunk('first', 'a', 'method')]
        retriever = LexicalRetriever(rows, self.catalog[:2])
        first = retriever.select('method', ['a', 'b'], 'bm25_rerank')
        second = retriever.select('method', ['b', 'a'], 'bm25_rerank')
        self.assertEqual([r['chunk_id'] for r in first['candidates']], ['second', 'first'])
        self.assertEqual(first['candidates'], second['candidates'])


if __name__ == '__main__':
    unittest.main()
