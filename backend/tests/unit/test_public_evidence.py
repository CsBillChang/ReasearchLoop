"""公开报表的负向校验；复用合成夹具，不读取原文或调用业务服务。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.report_public_evidence import reflection_report, report, retrieval_report, verify_usage


class PublicEvidenceTests(unittest.TestCase):
    """防止漏题、越权排名、重复计数及错误分母被静默接受。"""

    def setUp(self):
        """以两题合成数据固定真实质量与计时分母。"""
        self.data = {
            'scope': {'chunks': 2, 'documents': 2, 'questions': 2, 'answerable': 1,
                      'missing': 1, 'recall_cutoff': 10, 'mrr_cutoff': 24},
            'chunks': [{'id': 'C1', 'material': 'P1'}, {'id': 'C2', 'material': 'P2'}],
            'cases': [{'id': 'Q1', 'materials': ['P1'], 'answerable': True, 'relevant_chunks': ['C1']},
                      {'id': 'Q2', 'materials': ['P2'], 'answerable': False, 'relevant_chunks': []}],
            'batches': [{'id': 'synthetic', 'repeats': 1,
                         'runs': [{'case_id': 'Q1', 'mode': 'vector', 'repeat': 0, 'ranking': ['C1'], 'seconds': 1},
                                  {'case_id': 'Q2', 'mode': 'vector', 'repeat': 0, 'ranking': [], 'seconds': 3}],
                         'expected': {'vector': {'answerable': 1, 'excluded_missing': 1, 'recall_at_10': 1,
                                                'mrr_at_24': 1, 'timed_questions': 2, 'measurements': 2,
                                                'median_seconds': 2, 'p95_seconds': 3}}}]
        }

    def test_missing_question_excluded_from_quality_but_kept_in_time(self):
        """缺证题不能成为质量满分题，也不能从计时分母消失。"""
        result = retrieval_report(self.data)
        self.assertEqual(result['answerable'], 1)
        self.assertEqual(result['batches']['synthetic']['vector']['timed_questions'], 2)

    def test_deleted_or_duplicated_measurement_fails(self):
        """移除失败/缺证测量或重复排名都会破坏完整设计。"""
        for rows in (self.data['batches'][0]['runs'][:1], self.data['batches'][0]['runs'] * 2):
            data = copy.deepcopy(self.data)
            data['batches'][0]['runs'] = rows
            with self.assertRaisesRegex(ValueError, 'INCOMPLETE_OR_DUPLICATE'):
                retrieval_report(data)

    def test_unauthorized_chunk_and_changed_denominator_fail(self):
        """另一材料的命中不能替本题凑分；更改质量分母也会报错。"""
        data = copy.deepcopy(self.data)
        data['batches'][0]['runs'][0]['ranking'] = ['C2']
        with self.assertRaisesRegex(ValueError, 'INVALID_RANKING_OR_LABELS'):
            retrieval_report(data)
        self.data['scope']['answerable'] = 2
        with self.assertRaisesRegex(ValueError, 'DENOMINATOR'):
            retrieval_report(self.data)

    def test_wrong_aggregate_usage_fails(self):
        """缺失或不相等的Token不补零。"""
        with self.assertRaises(ValueError):
            verify_usage({'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 6})
        with self.assertRaises(ValueError):
            verify_usage({'prompt_tokens': 2, 'completion_tokens': 3})

    def test_supported_sentence_preservation_not_inferred_from_label(self):
        """前后标签都是supported时，仍必须保留原句的独立检查。"""
        data = {'scope': {'claim_slots': 1, 'drafts': 1},
                'claims': [{'draft_id': 'F1', 'slot': 1, 'before': 'supported', 'after': 'supported',
                            'original_supported_text_preserved': False}],
                'expected_transitions': {'supported -> supported': 1}, 'expected_preserved_supported': 1}
        with self.assertRaisesRegex(ValueError, 'SUPPORTED_TEXT_PRESERVATION'):
            reflection_report(data)

    def test_manifest_path_injection_rejected(self):
        """清单只能指定三份约定文件，不能用相对路径读取包外资料。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'provenance.json').write_text(json.dumps({'files': {'../private.json': 'x'}}))
            with self.assertRaisesRegex(ValueError, 'PUBLIC_FILE_MANIFEST'):
                report(root)


if __name__ == '__main__':
    unittest.main()
