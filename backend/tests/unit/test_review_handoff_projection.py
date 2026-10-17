"""核验投影的公开合成合同：精确引用、完整意见、版本边界与可逆性。"""
from copy import deepcopy
import json
import unittest

from research_loop.domain.context import project_review_handoffs
from research_loop.providers.deepseek import validate_messages


def review_fixture():
    """生成同一item的多条不同判断，包含工具/状态中重复的展示字段。"""
    findings = [{'item_id': 'answer', 'status': status, 'basis': 'direct_statement',
        'reason': '不同限定' + str(i), 'source_quote': 'Exact source ' + str(i),
        'evidence_ids': ['e' + str(i)], 'suggested_action': 'keep' if status == 'supported' else 'revise'}
        for i, status in enumerate(['supported', 'contradicted', 'insufficient'])]
    result = {'review_id': 'r1', 'target_id': 't1', 'content_hash': 'a'*64,
        'status': 'issues', 'quality_status': 'issues', 'current_version_reviewed': True,
        'target': {'target_id': 't1', 'content_hash': 'a'*64, 'items': [{'item_id': 'answer', 'text': '原正文'}]},
        'findings': findings, 'issues': [{**f, 'issue_id': 'i'+str(i), 'state': f['status']}
            for i, f in enumerate(findings)],
        'pending_checks': [{'review_id': 'r1', 'content_hash': 'a'*64, 'findings': deepcopy(findings)}],
        'reflection': {'stop_reason': None, 'can_continue': True}, 'gaps': ['保留未知']}
    messages = [{'role': 'system', 'content': '系统'}, {'role': 'user', 'content': '核验，不保存'},
        {'role': 'assistant', 'content': '', 'reasoning_content': '保持供应商字段', 'tool_calls': [
            {'id': 'review-call', 'type': 'function', 'function': {'name': 'review_research', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'review-call', 'content': json.dumps({'ok': True, 'kind': 'success', 'data': result})}]
    return messages, {'save_policy': 'deny', 'research_reviews': [result]}


class ReviewProjectionTests(unittest.TestCase):
    """纯投影不调用模型，不改工具schema、版本状态或持久原件。"""

    def test_single_copy_keeps_every_judgment_and_identity(self):
        """状态中每条意见完整保留，工具精确引用，展示副本可逐字段还原。"""
        messages, context = review_fixture(); original = deepcopy([messages, context])
        out, current, metrics = project_review_handoffs(messages, context)
        validate_messages(out)
        self.assertEqual([messages, context], original)
        self.assertEqual(out[:-1], messages[:-1])
        data = json.loads(out[-1]['content'])['data']
        self.assertEqual(data['context_path'], 'current_task.research_reviews[0]')
        result = current['research_reviews'][0]; raw = context['research_reviews'][0]
        self.assertEqual(result['findings'], raw['findings'])
        self.assertEqual(result['target'], raw['target'])
        for i, issue in enumerate(result['issues']):
            self.assertEqual(issue['finding_in'], f'current_task.research_reviews[0].findings[{i}]')
            restored = {**result['findings'][i], **{k:v for k,v in issue.items() if k != 'finding_in'}}
            self.assertEqual(restored, raw['issues'][i])
        self.assertEqual(result['pending_checks'][0]['findings_in'], 'current_task.research_reviews[0].findings')
        self.assertNotIn('findings', result['pending_checks'][0])
        self.assertEqual(current['save_policy'], 'deny')
        self.assertEqual(metrics['review_projected_calls'], ['review-call'])

    def test_stale_version_and_extra_diagnostic_do_not_reuse_tool_result(self):
        """同ID不同版本/状态/理由/额外字段不能被错误的新对象替代。"""
        for key, value in [('content_hash', 'b'*64), ('status', 'execution_failed'),
                           ('extra_diagnostic', '必须保留')]:
            with self.subTest(key=key):
                messages, context = review_fixture()
                old = json.loads(messages[-1]['content']); old['data'][key] = value
                messages[-1]['content'] = json.dumps(old)
                out, _, metrics = project_review_handoffs(messages, context)
                self.assertEqual(out, messages)
                self.assertEqual(metrics['review_projected_calls'], [])

    def test_different_pending_review_remains_independent(self):
        """上一版待核验记录即使措辞相同，也不指向当前版本的判断。"""
        messages, context = review_fixture()
        context['research_reviews'][0]['pending_checks'][0]['content_hash'] = 'c'*64
        out, current, _ = project_review_handoffs(messages, context)
        self.assertEqual(out, messages)
        self.assertIn('findings', current['research_reviews'][0]['pending_checks'][0])

    def test_conflicting_issue_extras_are_not_lost(self):
        """只有完整finding字段相等才建立引用，不吞掉冲突值或独立补充字段。"""
        messages, context = review_fixture()
        context['research_reviews'][0]['issues'][0]['reason'] = '不同理由'
        context['research_reviews'][0]['issues'][1]['extra'] = '原样保留'
        _, current, _ = project_review_handoffs(messages, context)
        self.assertEqual(current['research_reviews'][0]['issues'][0], context['research_reviews'][0]['issues'][0])
        self.assertEqual(current['research_reviews'][0]['issues'][1]['extra'], '原样保留')

    def test_projection_is_idempotent(self):
        """不会对已有引用重复包裹或改变意见数量。"""
        messages, context = review_fixture()
        a, b, _ = project_review_handoffs(messages, context)
        c, d, _ = project_review_handoffs(a, b)
        self.assertEqual((a, b), (c, d))

    def test_failed_or_wrong_tool_and_missing_identity_stay_raw(self):
        """不从失败观察或普通JSON猜测对象身份、语义判断或批准。"""
        for variant in ['failed', 'wrong_tool', 'no_id']:
            messages, context = review_fixture()
            if variant == 'wrong_tool':
                messages[2]['tool_calls'][0]['function']['name'] = 'read_memory'
            elif variant == 'failed':
                value = json.loads(messages[-1]['content']); value['ok'] = False
                messages[-1]['content'] = json.dumps(value)
            else:
                context['research_reviews'][0].pop('review_id')
            out, _, _ = project_review_handoffs(messages, context)
            self.assertEqual(out, messages)
