"""多意见、原文校验与错误因果的合成回归；不代表模型语义正确率。"""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from research_loop.agents.reviewer import _finish_research_review
from research_loop.domain.reflection import FinishResearchReview, review_progress
from research_loop.services.verification import validate_target_assessments
from research_loop.workers.executor import research_review_failure


class FixturePages:
    """只接受固定授权页，避免测试替身放过越权引文。"""

    def evidence_bundle(self, task_id, refs):
        """返回合成原句，遇到未知来源直接拒绝。"""
        if any(ref != 'page-A' for ref in refs):
            raise PermissionError('foreign evidence')
        return [{'text': 'The method uses language embeddings.'} for _ in refs]


class ReflectionResultTests(unittest.TestCase):
    """核验对象可以包含多个意见，结构接收不等于全部支持。"""

    def setUp(self):
        """生成一个对象及七个不同问题，保持与失败轨迹相同的基数。"""
        self.target = {'target_id': 'target-A', 'target_type': 'answer', 'content_hash': 'a'*64,
                       'items': [{'item_id': 'answer', 'text': '待核对的方法和条件。'}]}
        self.findings = [{'item_id': 'answer', 'status': 'supported', 'basis': 'direct_statement',
            'reason': f'问题{index}的合成判断', 'source_quote': 'The method uses language embeddings.',
            'evidence_ids': ['page-A'], 'suggested_action': 'keep'} for index in range(7)]
        self.ctx = SimpleNamespace(task_id='reviewer', services={'review_target': self.target,
            'review_parent_task_id': 'root', 'papers': FixturePages()})

    def test_multiple_findings_remain_distinct_and_unresolved(self):
        """同对象支持/反证/缺证全部接收；末尾支持不会抹去前面的问题。"""
        self.findings[0].update(status='contradicted', suggested_action='revise')
        self.findings[1].update(status='insufficient', basis='not_found', source_quote=None,
                                suggested_action='supplement')
        args = FinishResearchReview(findings=self.findings)
        before = args.model_dump()
        result = _finish_research_review(args, self.ctx)
        self.assertTrue(result.ok)
        self.assertEqual(result.data['status'], 'issues')
        self.assertEqual(result.data['findings'], before['findings'])
        self.assertEqual(args.model_dump(), before)

    def test_missing_foreign_and_invalid_quotes_are_still_rejected(self):
        """放开意见基数不放开对象覆盖、来源权限或精确引句。"""
        cases = [({'item_id': 'foreign'}, ValueError, 'unknown'),
                 ({'evidence_ids': ['foreign']}, PermissionError, 'foreign evidence'),
                 ({'source_quote': 'The method ... embeddings.'}, ValueError, 'findings[3]')]
        for changes, error, message in cases:
            with self.subTest(changes=changes):
                rows = deepcopy(self.findings); rows[3].update(changes)
                with self.assertRaisesRegex(error, message.replace('[', r'\[').replace(']', r'\]')):
                    _finish_research_review(FinishResearchReview(findings=rows), self.ctx)
        target = deepcopy(self.target); target['items'].append({'item_id': 'another', 'text': '另一对象'})
        with self.assertRaisesRegex(ValueError, "missing=.*another"):
            validate_target_assessments('reviewer', target,
                FinishResearchReview(findings=self.findings).findings, FixturePages())

    def test_progress_never_uses_last_duplicate_item_verdict(self):
        """顺序变化、重复条目和数量减少不等于对象全部问题已解决。"""
        old = {'findings': [{'item_id': 'answer', 'status': 'insufficient'},
                            {'item_id': 'answer', 'status': 'supported'}]}
        same = {'findings': list(reversed(old['findings']))}
        self.assertFalse(review_progress(old, same)['effective_change'])
        self.assertFalse(review_progress(old, {'findings': old['findings'][:1]})['effective_change'])
        mixed = {'findings': [{'item_id': 'answer', 'status': 'contradicted'},
                              {'item_id': 'answer', 'status': 'insufficient'},
                              {'item_id': 'answer', 'status': 'supported'}]}
        self.assertFalse(review_progress(old, mixed)['effective_change'])
        supported = {'findings': [{'item_id': 'answer', 'status': 'supported'}]}
        self.assertEqual(review_progress(mixed, supported)['resolved_item_ids'], ['answer'])

    def test_first_error_and_terminal_stop_survive_together(self):
        """局部额度停止不能遮住此前结果校验错误，也不变成缺证判断。"""
        actions = [{'action_id': 'a1', 'sequence': 4, 'tool_name': 'finish_review', 'status': 'failed',
                    'observation': {'kind': 'execution_error', 'message': 'findings[1]: invalid quote'}}]
        original = deepcopy(actions)
        result = research_review_failure(actions, 'ValueError: model request budget exhausted',
                                        'model request budget exhausted')
        self.assertEqual(result['first_error']['action_id'], 'a1')
        self.assertEqual(result['first_error']['stage'], 'result_validation')
        self.assertIn('invalid quote', result['first_error']['message'])
        self.assertIn('budget', result['stop_reason'])
        self.assertFalse(result['semantic_verdict_available'])
        self.assertEqual(actions, original)
        self.assertIsNone(research_review_failure([], 'unknown', None)['first_error'])
