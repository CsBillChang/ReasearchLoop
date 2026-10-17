"""判断字段与失败身份合同的合成验证；不把schema合法当作语义正确。"""
from copy import deepcopy
import unittest

from pydantic import ValidationError
from research_loop.agents.graph import failed_review_handoffs, reflection_handoffs
from research_loop.domain.reflection import FinishResearchReview
from research_loop.tools.base import ToolContext, ToolSpec, execute_spec
from research_loop.domain.tasks import SavePolicy


class VerdictContractTests(unittest.TestCase):
    """与真实失败相同的八意见基数，不使用私人回答或原文。"""

    def rows(self):
        """返回合成范围说明，判断值由夹具明确填写而非程序推断。"""
        return [{'item_id': 'answer', 'status': 'supported', 'basis': 'scope_disclosure',
                 'reason': f'合成范围说明{n}', 'evidence_ids': ['page-A'], 'suggested_action': 'keep'} for n in range(8)]

    def test_schema_requires_exactly_one_non_null_verdict(self):
        """发布给工具的合同要求二选一，不能只靠隐藏的运行时校验。"""
        spec = ToolSpec('finish_review', 'fixture', FinishResearchReview, lambda a, c: None)
        schema = spec.schema()['function']['parameters']['$defs']['TargetAssessment']
        branches = schema['oneOf']
        self.assertEqual([b['required'] for b in branches], [['status'], ['suitability']])
        self.assertEqual(branches[0]['properties']['suitability'], {'type': 'null'})
        self.assertEqual(branches[1]['properties']['status'], {'type': 'null'})
        self.assertIn('必填非空', schema['properties']['status']['description'])

    def test_null_verdict_preserved_and_whole_batch_rejected(self):
        """第六条空判断不能从keep/范围披露猜成supported，也不接收其余七条。"""
        args = {'findings': self.rows()}; args['findings'][5]['status'] = None
        original = deepcopy(args); calls = []
        spec = ToolSpec('finish_review', 'fixture', FinishResearchReview, lambda a, c: calls.append(a))
        result = execute_spec(spec, args, ToolContext(task_id='t', save_policy=SavePolicy.deny))
        self.assertFalse(result.ok); self.assertEqual(calls, [])
        self.assertEqual(args, original)
        self.assertIn('findings[5]', result.message)
        self.assertIn("item_id='answer'", result.message)
        self.assertIn('non-null', result.message)

    def test_legal_verdicts_and_old_null_other_field_compatible(self):
        """合法事实判断、适用性和旧响应另一空字段保持兼容。"""
        for status in ['supported', 'contradicted', 'insufficient']:
            row = self.rows()[0]
            row.update(status=status, suitability=None, basis='not_found' if status == 'insufficient' else 'direct_statement',
                       source_quote=None if status == 'insufficient' else 'Actual fixture sentence.')
            self.assertEqual(FinishResearchReview(findings=[row]).findings[0].status, status)
        for suitability in ['suitable', 'unsuitable', 'uncertain']:
            row = {**self.rows()[0], 'status': None, 'suitability': suitability, 'basis': 'metadata', 'metadata_fields': ['title']}
            self.assertEqual(FinishResearchReview(findings=[row]).findings[0].suitability, suitability)
        self.assertEqual(len(FinishResearchReview(findings=self.rows()).findings), 8)

    def test_missing_conflicting_and_invalid_enum_never_coerced(self):
        """缺字段、冲突和拼写错误维持拒绝；不靠删除意见修复。"""
        for changes in [{'status': None}, {'status': 'pass'}, {'suitability': 'suitable'}]:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                FinishResearchReview(findings=[{**self.rows()[0], **changes}])
        row = self.rows()[0]; row.pop('status')
        with self.assertRaises(ValidationError): FinishResearchReview(findings=[row])

    def test_failed_observation_is_settled_but_not_research_progress(self):
        """相同失败身份重放不产生科研进展；普通参数拒绝也不冒充Reviewer结果。"""
        action = {'tool_name': 'review_research', 'status': 'failed', 'observation': {
            'ok': False, 'message': 'finish_review null verdict', 'data': {'status': 'execution_failed',
            'review_id': 'review-A', 'target_id': 'target-A', 'content_hash': 'a'*64}}}
        self.assertEqual(len(failed_review_handoffs([action, deepcopy(action)])), 1)
        self.assertEqual(reflection_handoffs([action]), (set(), -1))
        invalid = deepcopy(action); invalid['observation']['data'] = {}
        self.assertEqual(failed_review_handoffs([invalid]), {})

    def test_malformed_argument_container_returns_validation_error(self):
        """错误格式器不能让非对象参数升级成内部异常。"""
        spec = ToolSpec('finish_review', 'fixture', FinishResearchReview, lambda a, c: None)
        for args in [None, [], {'findings': None}]:
            result = execute_spec(spec, args, ToolContext(task_id='t', save_policy=SavePolicy.deny))
            self.assertFalse(result.ok)
            self.assertEqual(result.kind, 'validation_error')
