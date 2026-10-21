"""行为兴趣旁路的容错与纯合同测试；不调用业务模型。"""
import unittest

from pydantic import ValidationError

from research_loop.domain.interests import InterestCandidate, normalize_topic_key
from research_loop.domain.tasks import FinishTaskInput


class InterestContractTests(unittest.TestCase):
    """验证兴趣候选不能把已可交付的研究答案变成参数错误。"""

    def test_malformed_sidecar_does_not_reject_delivery(self):
        """完整旁路及单条候选错误保留原值，交由独立投影诊断。"""
        for value in (None, 'bad', 7, {'topic_key': 'x'}, [None, {'unexpected': 1}]):
            with self.subTest(value=value):
                result = FinishTaskInput.model_validate({'answer_markdown': '保留回答',
                    'goals': [{'goal_id': 'g', 'status': 'unresolved', 'reason': '资料缺口'}],
                    'interest_candidates': value})
                self.assertEqual(result.answer_markdown, '保留回答')
                self.assertEqual(result.interest_candidates, value)

    def test_candidate_uses_ordinary_question_without_explicit_interest(self):
        """候选来源可以是普通研究提问，不要求出现声明兴趣的句式。"""
        result = InterestCandidate.model_validate({'topic_key': '图检索', 'content': '图检索研究线索',
            'reason': '当前问题比较了图检索方法', 'source_quote': '比较这两种图检索方法'})
        self.assertIsNone(result.merge_memory_id)

    def test_merge_requires_actual_version(self):
        """不接受无版本合并或凭空给新增条目填写旧版本。"""
        base = {'topic_key': 'topic', 'content': '线索', 'reason': '理由', 'source_quote': '问题'}
        for extra in ({'merge_memory_id': 'm'}, {'expected_version': 2}, {'content': ' '}):
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                InterestCandidate.model_validate({**base, **extra})

    def test_key_normalization_preserves_semantic_distinctions(self):
        """只统一字形大小写空白；不同重音或研究主题不猜测等价。"""
        self.assertEqual(normalize_topic_key(' ＧＲＡＰＨ  Retrieval '), 'graph retrieval')
        self.assertNotEqual(normalize_topic_key('resume'), normalize_topic_key('résumé'))


if __name__ == '__main__':
    unittest.main()
