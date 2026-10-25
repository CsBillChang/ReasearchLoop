"""显式候选的纯合同与替身接线；不计算真实论文排名或神经推理收益。"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256
from itertools import permutations
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research_loop.domain.retrieval_strategy import RetrievalStrategy, get_retrieval_strategy
from research_loop.storage.indexes import reciprocal_rank_fusion
from scripts.component_eval_adapters import RetrievalAdapter


NAMES = ('baseline', 'fusion_bm25_2_1', 'efficiency_length_sort')


def legacy_fusion(channels, k=60):
    """独立重放修改前算法，逐值比较完整排名、分数与原通道名次。"""
    result = {}
    for name, rows in channels.items():
        seen = set()
        for row in rows:
            if row['key'] in seen:
                continue
            seen.add(row['key'])
            position = len(seen)
            target = result.setdefault(row['key'], {'key': row['key'], 'rrf_score': 0.0, 'channels': {}})
            target['rrf_score'] += 1 / (k + position)
            target['channels'][name] = {'rank': position, 'score': row['score']}
    return sorted(result.values(), key=lambda row: (-row['rrf_score'], row['key']))


def synthetic_chunks(count=30):
    """构造完整合成来源身份，不读取论文、生产数据库或原实验材料。"""
    result = []
    for n in range(count):
        text = 'language method ' + str(n)
        result.append({'chunk_id': f'chunk-{n:02}', 'key': f'point-{n:02}', 'text': text,
            'document_id': 'synthetic-doc', 'index_id': 'synthetic-index', 'evidence_id': 'synthetic-page',
            'source_sha256': 'a' * 64, 'text_sha256': sha256(text.encode()).hexdigest(),
            'page_number': 1, 'char_start': 0, 'char_end': len(text),
            'collection': 'researchloop_eval_363_synthetic',
            'vector_payload': {'index_id': 'synthetic-index', 'chunk_id': f'chunk-{n:02}'}})
    return result


class CandidateStrategyTests(unittest.TestCase):
    """固定三配置、数值合法性与旧算法完整对照。"""

    def test_exact_frozen_configs_and_independent_serialization(self):
        """只允许两项单变量候选，外部修改参数副本不改变配置。"""
        for name, bm25, sort in zip(NAMES, (1, 2, 1), (False, False, True)):
            strategy = get_retrieval_strategy(name)
            expected = {'name': name, 'fusion_weights': {'bm25': bm25, 'vector': 1},
                'reranker_kwargs': {'threads': 2, 'batch_size': 4, 'sort_by_length': sort},
                'candidate_limit': 24, 'rrf_k': 60}
            self.assertEqual(strategy.as_dict(), expected)
            strategy.fusion_weights['bm25'] = 99
            strategy.reranker_kwargs['batch_size'] = 99
            strategy.as_dict()['reranker_kwargs']['threads'] = 99
            self.assertEqual(strategy.as_dict(), expected)
            with self.assertRaises(FrozenInstanceError):
                strategy.name = 'other'
        for name in ('unknown', 'fusion_and_sort', '', True, {}, None):
            with self.assertRaisesRegex(ValueError, 'STRATEGY_UNKNOWN'):
                get_retrieval_strategy(name)
        with self.assertRaises(TypeError):
            RetrievalStrategy('baseline', batch_size=8)

    def test_default_full_output_equals_independent_old_algorithm(self):
        """排列、重复、缺路、空榜及非标准历史通道逐值兼容。"""
        for left in permutations(('a', 'b', 'c')):
            for right in permutations(('a', 'b', 'c')):
                channels = {'bm25': [{'key': k, 'score': n + 10} for n, k in enumerate(left)],
                    'vector': [{'key': k, 'score': n / 10} for n, k in enumerate(right)]}
                channels['bm25'].insert(1, dict(channels['bm25'][0]))
                self.assertEqual(reciprocal_rank_fusion(channels), legacy_fusion(channels))
                self.assertEqual(reciprocal_rank_fusion(channels, weights={'bm25': 1, 'vector': 1}),
                                 legacy_fusion(channels))
        for channels in ({}, {'bm25': []}, {'historic': [{'key': 'a', 'score': -2}]}):
            self.assertEqual(reciprocal_rank_fusion(channels), legacy_fusion(channels))

    def test_two_to_one_is_hand_calculable_with_dedup_and_stable_ties(self):
        """使用去重后1-based排名，缺路无额外分数。"""
        channels = {'bm25': [{'key': 'b', 'score': 99}, {'key': 'b', 'score': 88}, {'key': 'a', 'score': 2}],
                    'vector': [{'key': 'a', 'score': .9}, {'key': 'c', 'score': .8}]}
        actual = reciprocal_rank_fusion(channels, weights={'bm25': 2, 'vector': 1})
        self.assertEqual([r['key'] for r in actual], ['a', 'b', 'c'])
        self.assertEqual(actual[0]['rrf_score'], 2 / 62 + 1 / 61)
        self.assertEqual(actual[1]['rrf_score'], 2 / 61)
        self.assertEqual(actual[2]['rrf_score'], 1 / 62)
        self.assertEqual(actual[0]['channels']['bm25']['rank'], 2)
        equal = reciprocal_rank_fusion({'bm25': [{'key': 'b', 'score': 3}],
            'vector': [{'key': 'a', 'score': 4}]}, weights={'bm25': 1, 'vector': 1})
        self.assertEqual([r['key'] for r in equal], ['a', 'b'])
        self.assertEqual(reciprocal_rank_fusion({'bm25': channels['bm25']},
            weights={'bm25': 2, 'vector': 1})[0]['rrf_score'], 2 / 61)

    def test_invalid_weights_rejected_without_numeric_coercion(self):
        """bool、无穷、缺项、多余项和全零不可冒充合法融合配置。"""
        bad = [None, [], {'bm25': 1}, {'vector': 1}, {'bm25': 1, 'vector': 1, 'extra': 2},
               {'other': 1, 'vector': 1}, {'bm25': 0, 'vector': 0}]
        bad.extend({'bm25': v, 'vector': 1} for v in (True, False, -1, float('nan'), float('inf'), '2', None, 10 ** 1000))
        for weights in bad:
            if weights is None:
                continue  # 省略和None共同保持历史默认接口。
            with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, 'WEIGHTS_INVALID'):
                reciprocal_rank_fusion({}, weights=weights)
        with self.assertRaisesRegex(ValueError, 'CHANNEL_UNKNOWN'):
            reciprocal_rank_fusion({'other': []}, weights={'bm25': 1, 'vector': 1})
        self.assertEqual(reciprocal_rank_fusion({}, weights={'bm25': 0, 'vector': 1}), [])

    def test_actual_reranker_parameters_are_required_and_type_exact(self):
        """候选名不能替代实际线程、batch和排序属性。"""
        strategy = get_retrieval_strategy('baseline')
        self.assertEqual(strategy.inspect_reranker(SimpleNamespace(**strategy.reranker_kwargs)), strategy.reranker_kwargs)
        for actual in ({}, {'threads': True, 'batch_size': 4, 'sort_by_length': False},
                       {'threads': 2, 'batch_size': 8, 'sort_by_length': False},
                       {'threads': 2, 'batch_size': 4, 'sort_by_length': True}):
            with self.assertRaisesRegex(ValueError, 'CONFIG_MISMATCH'):
                strategy.inspect_reranker(SimpleNamespace(**actual))


class CandidateAdapterTests(unittest.TestCase):
    """合成24候选、惰性工厂及每次实际路径和统计。"""

    def setUp(self):
        """所有神经组件和网络真实入口均以守卫拒绝。"""
        for target in ('socket.socket.connect', 'research_loop.providers.embeddings.LocalEmbedding.encode',
                       'research_loop.providers.reranker.LocalReranker.score',
                       'research_loop.providers.reranker.LocalReranker._load',
                       'research_loop.storage.vector_index.VectorIndex.query'):
            guard = patch(target, side_effect=AssertionError('真实组件禁止'))
            guard.start()
            self.addCleanup(guard.stop)
        self.rows = synthetic_chunks()
        self.scope = [r['chunk_id'] for r in self.rows]
        self.embedding = SimpleNamespace(encode=Mock(return_value=[[1., 0.]]))
        by_key = {r['key']: r for r in self.rows}
        self.vector = SimpleNamespace(query=Mock(side_effect=lambda collection, vector, ids, limit: [
            {'id': key, 'score': float(n), 'payload': by_key[key]['vector_payload']}
            for n, key in enumerate(ids[-limit:])][::-1]))

    def adapter(self, strategy, reranker=None):
        """用显式工厂装配替身，不继承生产Runtime默认组件。"""
        reranker = reranker or SimpleNamespace(**strategy.reranker_kwargs,
            score=Mock(side_effect=lambda query, texts: list(range(len(texts)))), last_stats={'synthetic': True})
        self.reranker = reranker
        self.factories = [Mock(return_value=x) for x in (self.embedding, self.vector, reranker)]
        return RetrievalAdapter(*self.factories, strategy=strategy)

    def test_all_strategies_pass_same_24_candidates_and_record_actual(self):
        """CE候选完整24条，融合和效率单变量配置均可追溯。"""
        for name in NAMES:
            strategy = get_retrieval_strategy(name)
            adapter = self.adapter(strategy)
            fused = adapter.run('language', self.rows, 'rrf', self.scope)
            self.factories[2].assert_not_called()
            result = adapter.run('language', self.rows, 'rrf_rerank', self.scope)
            self.assertEqual(result['strategy'], strategy.as_dict())
            self.assertEqual(result['reranker_configuration']['actual'], strategy.reranker_kwargs)
            self.assertTrue(result['reranker_configuration']['verified'])
            self.assertEqual(len(self.reranker.score.call_args.args[1]), 24)
            self.assertEqual(self.reranker.score.call_args.args[1], [r['text'] for r in fused['hits']])
            self.assertEqual(len(result['hits']), 24)
            self.assertEqual(result['ranked_ids'], list(reversed(fused['ranked_ids'])))

    def test_default_and_explicit_baseline_full_rankings_scores_match(self):
        """旧调用无需假组件参数属性，显式baseline仍给相同完整排名。"""
        strategy = get_retrieval_strategy('baseline')
        adapter = self.adapter(strategy)
        default = RetrievalAdapter(*self.factories)
        for variant in ('bm25', 'vector', 'rrf', 'rrf_rerank'):
            explicit = adapter.run('language', self.rows, variant, self.scope)
            legacy = default.run('language', self.rows, variant, self.scope)
            self.assertEqual(explicit['hits'], legacy['hits'])
            self.assertEqual(explicit['ranked_ids'], legacy['ranked_ids'])
            self.assertNotIn('strategy', legacy)
        legacy_reranker = SimpleNamespace(score=Mock(return_value=[1.] * 24))
        self.assertEqual(RetrievalAdapter(*self.factories[:2], lambda: legacy_reranker).run(
            'language', self.rows, 'rrf_rerank', self.scope)['effective_path'], 'rrf_rerank')

    def test_mismatch_refuses_score_and_reports_rrf_path(self):
        """组批配置不符时推理零次，准确降级不伪称候选完成。"""
        strategy = get_retrieval_strategy('efficiency_length_sort')
        bad = SimpleNamespace(threads=2, batch_size=4, sort_by_length=False, score=Mock())
        adapter = self.adapter(strategy, bad)
        expected = adapter.run('language', self.rows, 'rrf', self.scope)
        result = adapter.run('language', self.rows, 'rrf_rerank', self.scope)
        bad.score.assert_not_called()
        self.assertEqual(result['ranked_ids'], expected['ranked_ids'])
        self.assertEqual(result['effective_path'], 'rrf')
        self.assertEqual(result['degradations'][0]['code'], 'RETRIEVAL_RERANKER_CONFIG_MISMATCH')
        self.assertFalse(result['reranker_configuration']['verified'])

    def test_failures_do_not_reuse_prior_success_stats_or_escape_scope(self):
        """共享实例第二次失败不带旧stats，向量失败保留真实BM25路径。"""
        adapter = self.adapter(get_retrieval_strategy('fusion_bm25_2_1'))
        scope = self.scope[:5]
        first = adapter.run('language', self.rows, 'rrf_rerank', scope)
        self.assertEqual(set(first['ranked_ids']), set(scope))
        self.reranker.score.side_effect = ValueError('RERANK_INPUT_TOO_LONG')
        second = adapter.run('language', self.rows, 'rrf_rerank', scope)
        self.assertEqual(second['effective_path'], 'rrf')
        self.assertEqual(second['degradations'][0]['code'], 'RERANK_INPUT_TOO_LONG')
        self.assertFalse(any('stats' in item for item in second['trace']))
        self.vector.query.side_effect = OSError('synthetic outage')
        third = adapter.run('language', self.rows, 'rrf_rerank', scope)
        self.assertEqual(third['effective_path'], 'bm25')
        self.assertEqual(set(third['ranked_ids']), set(scope))
        self.assertEqual(third['degradations'][-1]['code'], 'RERANK_INPUT_TOO_LONG')

    def test_bm25_is_lazy_and_frozen_identity_rechecked(self):
        """纯BM25不构造模型，版本/hash损坏在工厂前拒绝。"""
        adapter = self.adapter(get_retrieval_strategy('baseline'))
        adapter.run('language', self.rows, 'bm25', self.scope)
        for factory in self.factories:
            factory.assert_not_called()
        frozen = {r['chunk_id']: dict(r) for r in self.rows}
        self.rows[0]['document_id'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'FROZEN_IDENTITY'):
            adapter.run('language', self.rows, 'rrf_rerank', self.scope, frozen_identity=frozen)
        for factory in self.factories:
            factory.assert_not_called()

    def test_non_json_actual_configuration_and_post_score_change_are_rejected(self):
        """配置观察不泄露对象repr，推理后参数漂移也准确回退原排名。"""
        strategy = get_retrieval_strategy('baseline')
        for bad in (object(), float('nan')):
            adapter = self.adapter(strategy)
            self.reranker.threads = bad
            result = adapter.run('language', self.rows, 'rrf_rerank', self.scope)
            json.dumps(result, allow_nan=False)
            self.reranker.score.assert_not_called()
            self.assertEqual(result['degradations'][0]['code'], 'RETRIEVAL_RERANKER_CONFIG_MISMATCH')
        adapter = self.adapter(strategy)
        expected = adapter.run('language', self.rows, 'rrf', self.scope)

        def change_configuration(query, texts):
            """只在替身评分后改变配置，验证候选身份不会虚标成功。"""
            self.reranker.batch_size = 8
            return list(range(len(texts)))

        self.reranker.score.side_effect = change_configuration
        result = adapter.run('language', self.rows, 'rrf_rerank', self.scope)
        self.assertEqual(result['ranked_ids'], expected['ranked_ids'])
        self.assertEqual(result['effective_path'], 'rrf')
        self.assertEqual(result['reranker_configuration']['actual']['batch_size'], 8)
        self.assertFalse(result['reranker_configuration']['verified'])

    def test_statistics_failure_preserves_original_fusion_ranking(self):
        """评分后的诊断失败不能把CE排序伪装成RRF回退。"""
        class BadStats:
            """专门注入不可读统计，不触碰真实模型。"""

            threads, batch_size, sort_by_length = 2, 4, False

            def score(self, query, texts):
                """提供与输入相反的确定性排序分数。"""
                return list(range(len(texts)))

            @property
            def last_stats(self):
                """模拟仅诊断读回失败。"""
                raise ValueError('RERANK_STATS_UNAVAILABLE')

        adapter = self.adapter(get_retrieval_strategy('baseline'), BadStats())
        expected = adapter.run('language', self.rows, 'rrf', self.scope)
        result = adapter.run('language', self.rows, 'rrf_rerank', self.scope)
        self.assertEqual(result['ranked_ids'], expected['ranked_ids'])
        self.assertEqual(result['effective_path'], 'rrf')
        self.assertEqual(result['degradations'][0]['code'], 'RERANK_STATS_UNAVAILABLE')
        self.assertFalse(any('stats' in item for item in result['trace']))


if __name__ == '__main__':
    unittest.main()
