"""检索优化的零模型合同：排名、组批身份、容量及评测分母。"""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from research_loop.providers.reranker import LocalReranker
from research_loop.storage.indexes import fuse_reranker_ranks
from scripts.optimize_retrieval import metric, no_regression


class TokenizerStub:
    """用长度构造可验证身份，不模拟语义相关性。"""
    def encode(self, query, text):
        """返回长度即身份的token序列。"""
        return SimpleNamespace(ids=[len(text)]*len(text))


class SessionStub:
    """记录padding大小和推理次数，返回首token对应分数。"""
    def __init__(self):
        """每例独立推理记录。"""
        self.shapes = []

    def get_inputs(self):
        """替身不要求token_type_ids。"""
        return []

    def run(self, output, feed):
        """与输入身份相关的确定性分数，非真实模型验证。"""
        self.shapes.append(feed['input_ids'].shape)
        return [feed['input_ids'][:, :1].astype(float)]


class RetrievalOptimizationTests(unittest.TestCase):
    """不下载模型或访问上游；真实分数变化留给冻结对照。"""
    def test_rank_endpoints_and_no_mutation(self):
        """权重端点分别复现旧重排和原RRF，原对象保持。"""
        rows = [{'key': str(i), 'text': '原文'+str(i)} for i in range(4)]
        before = copy.deepcopy(rows)
        self.assertEqual([r['key'] for r in fuse_reranker_ranks(rows, [2., 4., 4., 1.])], ['1', '2', '0', '3'])
        self.assertEqual([r['key'] for r in fuse_reranker_ranks(rows, [2., 4., 4., 1.], weight=0)], ['0', '1', '2', '3'])
        self.assertEqual(rows, before)

    def test_partial_rerank_keeps_tail_and_source(self):
        """仅评分前缀，尾部与原文定位逐字保留。"""
        rows = [{'key': str(i), 'page_number': i+1, 'document_id': 'same'} for i in range(5)]
        got = fuse_reranker_ranks(rows, [0., 1.], weight=.75)
        self.assertEqual([r['key'] for r in got], ['1', '0', '2', '3', '4'])
        self.assertEqual([r['page_number'] for r in got], [2, 1, 3, 4, 5])
        self.assertIsNone(got[-1]['reranker_score'])

    def test_invalid_scores_scope_and_weight(self):
        """重复身份、多出分数、NaN和越界权重被拒。"""
        for rows, scores, weight in [([{'key':'a'}]*2, [1.,2.], .5),
            ([{'key':'a'}], [1.,2.], .5), ([{'key':'a'}], [float('nan')], .5),
            ([{'key':'a'}], [1.], 2), ([{'key':'a'}], [1.], float('nan'))]:
            with self.assertRaises(ValueError): fuse_reranker_ranks(rows, scores, weight=weight)

    def model(self, sorted_rows=False):
        """注入替身，避免接触模型路径。"""
        model = LocalReranker(Path('unused.json'), batch_size=4, sort_by_length=sorted_rows)
        model.session = SessionStub(); model.tokenizer = TokenizerStub()
        return model

    def test_length_bucket_restores_identity_and_reduces_padding(self):
        """重排计算顺序不可改变返回分数对应的片段。"""
        texts = ['x'*n for n in (20,2,21,3,22,4,23,5)]
        baseline, candidate = self.model(), self.model(True)
        self.assertEqual(baseline.score('query', texts), candidate.score('query', texts))
        self.assertLess(candidate.last_stats['padded_tokens'], baseline.last_stats['padded_tokens'])
        self.assertEqual(candidate.last_stats['input_tokens'], 100)

    def test_capacity_rejected_before_inference(self):
        """超容量不截断，不混入部分输出。"""
        model = self.model(True)
        with self.assertRaisesRegex(ValueError, 'INPUT_TOO_LONG'): model.score('q', ['x'*513])
        self.assertEqual(model.session.shapes, [])
        with self.assertRaisesRegex(ValueError, 'CANDIDATE_LIMIT'): model.score('q', ['x']*65)
        self.assertEqual(model.score('q', []), [])

    def test_invalid_configuration(self):
        """仅接受明确的有限批量与线程配置。"""
        for args in ({'batch_size': 100}, {'threads': 0}, {'sort_by_length':'yes'}):
            with self.assertRaisesRegex(ValueError, 'CONFIG'): LocalReranker('unused', **args)

    def test_metrics_missing_and_tradeoff_gate(self):
        """缺证不计满分；Recall上升不能掩盖MRR下降。"""
        result = metric(['a','b','c'], ['b','c'])
        self.assertEqual(result, {'recall_at_10':1.,'recall_at_8':1.,'mrr_at_24':.5})
        self.assertIsNone(metric(['a'], [])['mrr_at_24'])
        self.assertFalse(no_regression({**result,'mrr_at_24':.4}, result))
        with self.assertRaises(ValueError): metric(['a','a'], ['a'])


if __name__ == '__main__':
    unittest.main()
