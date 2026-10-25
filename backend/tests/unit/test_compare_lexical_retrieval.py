"""纯合成核对文件检索实验的身份、评分映射、计量与降级。"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import unittest

from scripts.compare_lexical_retrieval import ROUTES, run_one
from scripts.retrieval_experiment_guard import BoundaryViolation


def candidate(number: int, document='synthetic-document') -> dict:
    """制造带完整可核对身份的原文片段。"""
    text = f'synthetic passage {number}'
    return {'chunk_id': f'chunk-{number}', 'document_id': document,
            'index_id': 'synthetic-index', 'evidence_id': f'page-{number}',
            'source_sha256': 'a' * 64, 'text_sha256': sha256(text.encode()).hexdigest(),
            'page_number': number + 1, 'char_start': 0, 'char_end': len(text), 'text': text}


class FakeRetriever:
    """显式返回输入列表；测试可注入坏候选，不读取任何gold。"""

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def select(self, query, scope_document_ids, route):
        self.calls.append((query, list(scope_document_ids), route))
        return {'candidates': deepcopy(self.rows), 'search_query': 'expanded selection only',
                'trace': {'source': 'synthetic'}, 'local_document_ids': ['synthetic-document']}


class FakeReranker:
    """固定合成分数或异常，不导入tokenizer/ONNX或执行真实模型。"""

    def __init__(self, values=None, error=None):
        self.values, self.error, self.calls = values, error, []
        self.last_stats = {'stale': 'must not survive'}
        self.identity = {'model_id': 'synthetic-only', 'batch_size': 4}

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        if self.error is not None:
            raise self.error
        self.last_stats = {'pairs': len(passages), 'status': 'succeeded'}
        return self.values


class CompareLexicalRetrievalTests(unittest.TestCase):
    """验证运行器合同；不把合成通过计为检索效果。"""

    def setUp(self):
        self.question = {'case_id': 'synthetic-query', 'query': 'original user query',
                         'scope_document_ids': ['synthetic-document']}
        self.counts = {'score_attempts': 0, 'pair_attempts': 0, 'successful_score_calls': 0,
                       'successful_pairs': 0, 'degraded': 0, 'hot_rows': 0, 'warmup_rows': 0}
        self.rows = [candidate(i) for i in range(3)]

    def test_success_restores_identity_and_original_query_text(self):
        """分数按原候选身份映射，选择扩展词不进入CE输入。"""
        before = deepcopy(self.rows)
        selector, model = FakeRetriever(self.rows), FakeReranker([-5.0, 10.0, 3.0])
        result = run_one(selector, model, self.question, ROUTES[1], self.counts)
        self.assertEqual(result['status'], 'executed')
        self.assertEqual(result['candidate_ids'], ['chunk-0', 'chunk-1', 'chunk-2'])
        self.assertEqual(result['ranked_ids'], ['chunk-1', 'chunk-2', 'chunk-0'])
        self.assertEqual([r['candidate_rank'] for r in result['hits']], [2, 3, 1])
        self.assertEqual([r['reranker_score'] for r in result['hits']], [10.0, 3.0, -5.0])
        for hit in result['hits']:
            row = next(r for r in self.rows if r['chunk_id'] == hit['chunk_id'])
            self.assertEqual((hit['document_id'], hit['text_sha256']), (row['document_id'], row['text_sha256']))
        self.assertEqual(model.calls, [('original user query', [r['text'] for r in before])])
        self.assertEqual(self.rows, before)
        self.assertEqual((self.counts['score_attempts'], self.counts['pair_attempts']), (1, 3))
        self.assertEqual((self.counts['successful_score_calls'], self.counts['successful_pairs']), (1, 3))
        self.assertIsNone(result['usage'])
        self.assertGreaterEqual(result['timing']['wall_seconds'], 0)

    def test_equal_scores_preserve_candidate_order(self):
        """同分使用输入顺序，避免集合遍历引入名次不稳定。"""
        result = run_one(FakeRetriever(self.rows), FakeReranker([1.0, 1.0, -2.0]),
                         self.question, ROUTES[0], self.counts)
        self.assertEqual(result['ranked_ids'], ['chunk-0', 'chunk-1', 'chunk-2'])

    def test_empty_candidates_never_score_or_reuse_stats(self):
        """空列表保留empty，既不记评分预算也不沿用上一题stats。"""
        model = FakeReranker([])
        result = run_one(FakeRetriever([]), model, self.question, ROUTES[0], self.counts)
        self.assertEqual(result['status'], 'empty')
        self.assertEqual(result['hits'], [])
        self.assertEqual(result['local_reranker_stats'], {})
        self.assertEqual(model.calls, [])
        self.assertEqual(self.counts['score_attempts'], 0)

    def test_overlong_degradation_preserves_candidates_and_clears_stats(self):
        """已有成功后发生容量拒绝，失败输出仍不带旧成功统计。"""
        model = FakeReranker([0.1, 0.2, 0.3])
        run_one(FakeRetriever(self.rows), model, self.question, ROUTES[0], self.counts)
        model.error = ValueError('RERANK_INPUT_TOO_LONG')
        result = run_one(FakeRetriever(self.rows), model, self.question, ROUTES[0], self.counts)
        self.assertEqual(result['status'], 'degraded')
        self.assertEqual(result['effective_path'], 'lexical_fallback')
        self.assertEqual(result['ranked_ids'], ['chunk-0', 'chunk-1', 'chunk-2'])
        self.assertEqual(result['local_reranker_stats'], {})
        self.assertEqual(model.last_stats, {})
        self.assertTrue(all(r['reranker_score'] is None for r in result['hits']))
        self.assertEqual(result['error']['reason'], 'RERANK_INPUT_TOO_LONG')
        self.assertEqual((self.counts['score_attempts'], self.counts['successful_score_calls'], self.counts['degraded']), (2, 1, 1))

    def test_invalid_scores_are_rejected_without_success_credit(self):
        """数目不符、布尔、字符串、NaN及无穷值只形成明确降级。"""
        for values in ([1.0], [0., 1., float('nan')], [0., 1., float('inf')],
                       [0., True, 1.], [0., 'bad', 1.], None):
            with self.subTest(values=values):
                counts = dict.fromkeys(self.counts, 0)
                result = run_one(FakeRetriever(self.rows), FakeReranker(values), self.question, ROUTES[0], counts)
                self.assertEqual(result['status'], 'degraded')
                self.assertEqual(result['ranked_ids'], ['chunk-0', 'chunk-1', 'chunk-2'])
                self.assertEqual(result['local_reranker_stats'], {})
                self.assertEqual(counts['successful_score_calls'], 0)
                self.assertEqual(counts['successful_pairs'], 0)
                self.assertEqual(counts['degraded'], 1)
                self.assertTrue(all(r['reranker_score'] is None for r in result['hits']))

    def test_budget_rejects_before_any_model_invocation(self):
        """评分次数和配对数分别到上限时，本次模型调用不发生。"""
        for field, limit in [('score_attempts', 686), ('pair_attempts', 16464)]:
            with self.subTest(field=field):
                counts = dict.fromkeys(self.counts, 0); counts[field] = limit
                model = FakeReranker([0., 0., 0.])
                with self.assertRaisesRegex(BoundaryViolation, 'BUDGET_EXCEEDED'):
                    run_one(FakeRetriever(self.rows), model, self.question, ROUTES[0], counts)
                self.assertEqual(model.calls, [])
                self.assertEqual(counts['successful_score_calls'], 0)

    def test_bad_scope_hash_and_duplicate_candidates_reject_before_score(self):
        """越权、原文hash篡改、空ID、重复及超量候选均在模型前拒绝。"""
        outside = candidate(0, 'unauthorized-document')
        corrupt = {**candidate(0), 'text': 'tampered'}
        empty_id = {**candidate(0), 'chunk_id': ''}
        invalid_sets = [[outside], [corrupt], [empty_id], [candidate(0), candidate(0)],
                        [candidate(i) for i in range(25)]]
        for rows in invalid_sets:
            with self.subTest(rows=rows):
                model = FakeReranker([0.] * len(rows))
                with self.assertRaises(BoundaryViolation):
                    run_one(FakeRetriever(rows), model, self.question, ROUTES[0], self.counts)
                self.assertEqual(model.calls, [])
                self.assertEqual(self.counts['score_attempts'], 0)

    def test_invalid_route_rejects_before_retriever(self):
        """路线不依赖假组件验证，非法值连候选选择也不执行。"""
        selector, model = FakeRetriever(self.rows), FakeReranker([0., 0., 0.])
        with self.assertRaisesRegex(BoundaryViolation, 'ROUTE_INVALID'):
            run_one(selector, model, self.question, 'not-a-route', self.counts)
        self.assertEqual(selector.calls, [])
        self.assertEqual(model.calls, [])

    def test_resource_and_boundary_failures_escape_without_fallback(self):
        """系统资源和隔离异常停止实验，不能当普通质量问题继续。"""
        for error in (MemoryError('synthetic'), SystemError('synthetic'),
                      OSError('synthetic'), BoundaryViolation('synthetic')):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    run_one(FakeRetriever(self.rows), FakeReranker(error=error),
                            self.question, ROUTES[0], dict.fromkeys(self.counts, 0))


if __name__ == '__main__':
    unittest.main()
