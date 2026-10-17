"""证据报表的独立口径测试；不读取私人材料或执行模型。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.report_evidence import Reader, accounting, latency, mean_quality, quality


class EvidenceReportTests(unittest.TestCase):
    """检查多相关片段、缺证分母、计时聚合及未知用量。"""

    def test_recall_is_not_hit_rate(self):
        """两个相关片段只命中一个时Recall为0.5，不能记为命中率1。"""
        result = quality(['x', 'a'], ['a', 'b'], True, {'x', 'a', 'b'})
        self.assertEqual(result['recall_at_10'], .5)
        self.assertEqual(result['mrr_at_24'], .5)

    def test_cutoff_and_unreturned_gold(self):
        """第11名不计Recall@10，但仍参与MRR@24；无命中为0。"""
        keys = [str(i) for i in range(24)]
        result = quality(keys, ['10'], True, set(keys))
        self.assertEqual(result['recall_at_10'], 0)
        self.assertEqual(result['mrr_at_24'], 1 / 11)
        self.assertEqual(quality([], ['10'], True, set(keys))['mrr_at_24'], 0)

    def test_missing_is_excluded_not_perfect(self):
        """缺证题单列，不改变可回答题分母；全部缺证时返回None。"""
        missing = quality(['a'], [], False, {'a'})
        present = quality([], ['a'], True, {'a'})
        self.assertEqual(mean_quality([missing, present]),
                         {'answerable': 1, 'excluded_missing': 1, 'recall_at_10': 0, 'mrr_at_24': 0})
        self.assertIsNone(mean_quality([missing])['mrr_at_24'])

    def test_invalid_identity_and_labels_rejected(self):
        """重复、越界、标注冲突与超过24候选均拒绝，不静默修正分母。"""
        for keys, labels, answerable in [(['a', 'a'], ['a'], True), (['b'], ['a'], True),
                                          (['a'], [], True), (['a'], ['a'], False),
                                          (['a'], ['a', 'a'], True)]:
            with self.subTest(keys=keys, labels=labels, answerable=answerable):
                with self.assertRaises(ValueError):
                    quality(keys, labels, answerable, {'a'})
        with self.assertRaises(ValueError):
            quality(list(map(str, range(25))), ['0'], True, set(map(str, range(25))))

    def test_latency_uses_question_medians(self):
        """不把重复测量次数更多的题隐式加权，P95使用nearest-rank。"""
        result = latency([[1, 2, 100], [4]])
        self.assertEqual(result, {'timed_questions': 2, 'measurements': 4,
                                 'median_seconds': 3, 'p95_seconds': 4})
        for values in [[], [[]], [[float('nan')]], [[-1]]]:
            with self.assertRaises(ValueError):
                latency(values)

    def test_unknown_usage_not_zero(self):
        """未知请求仍在物理分母，已知tokens只作为小计。"""
        result = accounting([{'request_id': 'a', 'status': 'succeeded',
                              'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}},
                             {'request_id': 'b', 'status': 'unknown', 'usage': None}])
        self.assertEqual(result['physical_requests'], 2)
        self.assertEqual(result['known_tokens']['total_tokens'], 5)
        self.assertEqual(result['missing_usage_request_ids'], ['b'])
        self.assertFalse(result['all_usage_known'])

    def test_duplicate_receipt_and_invalid_total_rejected(self):
        """同请求不可重复累计，token小计不一致时停止报表。"""
        row = {'request_id': 'a', 'status': 'succeeded', 'usage': None}
        with self.assertRaises(ValueError):
            accounting([row, row])
        with self.assertRaises(ValueError):
            accounting([{**row, 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 6}}])

    def test_source_drift_separate_from_changed_material(self):
        """源码漂移明确报告，材料被更改则拒绝继续重算，不能混为历史全复现。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'backend').mkdir()
            (root / 'batch').mkdir()
            (root / 'backend/code.py').write_text('new source')
            material = root / 'batch/material.json'
            material.write_text('{}')
            frozen = {'files': {'backend/code.py': 'oldhash',
                                'batch/material.json': hashlib.sha256(b'{}').hexdigest()}}
            (root / 'batch/freeze.json').write_text(json.dumps(frozen))
            result = Reader(root).freeze_status('batch')
            self.assertFalse(result['historical_file_set_available'])
            self.assertEqual(result['mismatches'][0]['kind'], 'source_drift')
            material.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'FROZEN_MATERIAL_CHANGED'):
                Reader(root).freeze_status('batch')


if __name__ == '__main__':
    unittest.main()
