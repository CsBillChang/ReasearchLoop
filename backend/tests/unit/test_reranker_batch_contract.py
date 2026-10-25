"""重排组批的零模型合同：只使用合成tokenizer与内存Session。"""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from research_loop.providers.reranker import LocalReranker


class FakeTokenizer:
    """以片段中的长度与序号制造token，不执行真实分词。"""

    def __init__(self):
        """记录调用次数，约束每对仅编码一次。"""
        self.calls = []

    def encode(self, query, text):
        """返回首token携带原序号的合成序列。"""
        self.calls.append((query, text))
        index, size = map(int, text.split(':'))
        return SimpleNamespace(ids=[index + 10] * size)


class FakeSession:
    """只读入数组并从首token还原分数，不载入权重。"""

    def __init__(self, failure=None, token_types=True):
        """保存可选异常或坏输出模式。"""
        self.failure, self.token_types, self.feeds = failure, token_types, []

    def get_inputs(self):
        """显式声明模型输入。"""
        names = ['input_ids', 'attention_mask']
        return [SimpleNamespace(name=name) for name in names + (['token_type_ids'] if self.token_types else [])]

    def run(self, outputs, feed):
        """返回可追溯到原序号的合成logit或指定故障。"""
        self.feeds.append(feed)
        if isinstance(self.failure, BaseException):
            raise self.failure
        values = feed['input_ids'][:, :1].astype(float)
        if self.failure == 'nan':
            values[0, 0] = np.nan
        elif self.failure == 'count':
            values = values[:-1]
        return [values]


class RerankerBatchContractTests(unittest.TestCase):
    """验证身份、动态padding、容量和失败清理，不评价真实排名质量。"""

    def make_model(self, sort=False, failure=None, token_types=True):
        """注入假组件，禁止首次真实模型装载。"""
        model = LocalReranker('/nonexistent/never-loaded.json', sort_by_length=sort)
        model.tokenizer = FakeTokenizer()
        model.session = FakeSession(failure, token_types)
        return model

    def test_all_24_same_order_and_shapes_without_second_tokenization(self):
        """两种组批均完整还原24个输入，并记录真实feed形状。"""
        sizes = [8, 3, 17, 5, 12, 1, 9, 4] * 3
        passages = [f'{i}:{size}' for i, size in enumerate(sizes)]
        for sorted_batches in (False, True):
            with self.subTest(sorted_batches=sorted_batches), patch.object(LocalReranker, '_load', side_effect=AssertionError('real model forbidden')):
                model = self.make_model(sorted_batches)
                self.assertEqual(model.score('query', passages), list(map(float, range(10, 34))))
                self.assertEqual(len(model.tokenizer.calls), 24)
                stats = model.last_stats
                expected = sorted(range(24), key=lambda i: sizes[i]) if sorted_batches else list(range(24))
                self.assertEqual(stats['execution_order'], expected)
                self.assertEqual(stats['input_pair_lengths'], sizes)
                self.assertEqual(len(stats['batches']), 6)
                self.assertEqual(stats['pairs'], 24)
                self.assertEqual(stats['threads'], 2)
                self.assertEqual(stats['batch_size'], 4)
                for batch, feed in zip(stats['batches'], model.session.feeds):
                    lengths = [sizes[i] for i in batch['input_indices']]
                    self.assertEqual(batch['pair_lengths'], lengths)
                    self.assertEqual(batch['feed_shapes']['input_ids'], [4, max(lengths)])
                    self.assertEqual(feed['attention_mask'].sum(axis=1).tolist(), lengths)
                    self.assertTrue((feed['token_type_ids'] == 0).all())
                self.assertEqual(stats['padded_tokens'], sum(4 * max(b['pair_lengths']) for b in stats['batches']))

    def test_short_final_batch_and_optional_token_types(self):
        """不足四条的尾批按本批最长值padding，输入不要求token_type_ids。"""
        model = self.make_model(True, token_types=False)
        self.assertEqual(model.score('q', ['0:5', '1:2', '2:9', '3:7', '4:1']), [10., 11., 12., 13., 14.])
        self.assertEqual(model.last_stats['batches'][-1]['feed_shapes'], {'input_ids': [1, 9], 'attention_mask': [1, 9]})

    def test_length_512_allowed_513_rejected_before_session_run(self):
        """512边界保持；超长整批拒绝且清理历史stats，不静默截断。"""
        model = self.make_model()
        self.assertEqual(model.score('q', ['0:512']), [10.])
        model.session.feeds.clear()
        with self.assertRaisesRegex(ValueError, 'INPUT_TOO_LONG'):
            model.score('q', ['0:3', '1:513'])
        self.assertEqual(model.session.feeds, [])
        self.assertEqual(model.last_stats, {})

    def test_bad_output_exception_and_cancellation_clear_previous_success(self):
        """坏值、数量错误、Session异常和中断均不残留前次成功观察。"""
        for failure in ('nan', 'count', RuntimeError('fake failure'), KeyboardInterrupt('fake cancel')):
            with self.subTest(failure=str(failure)):
                model = self.make_model()
                model.score('q', ['0:4'])
                model.session.failure = failure
                with self.assertRaises(BaseException):
                    model.score('q', ['0:4'])
                self.assertEqual(model.last_stats, {})

    def test_empty_and_over_capacity_clear_stats_without_load(self):
        """空输入及超过64候选不触发模型装载，也不保留旧成功。"""
        model = self.make_model()
        for passages, error in (([], False), (['0:1'] * 65, True)):
            model.last_stats = {'status': 'old_success'}
            model.session = None
            with patch.object(LocalReranker, '_load', side_effect=AssertionError('must stay lazy')):
                if error:
                    with self.assertRaisesRegex(ValueError, 'CANDIDATE_LIMIT'):
                        model.score('q', passages)
                else:
                    self.assertEqual(model.score('q', passages), [])
            self.assertEqual(model.last_stats, {})
