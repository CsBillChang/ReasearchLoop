"""资格阻断和精确失败候选的合成合同，所有socket连接均禁止。"""
from copy import deepcopy
from unittest.mock import patch
import hashlib
import json
import os
import unittest

from research_loop.agents.graph import finish_input_hash, qualification_handoff, qualification_handoff_active
from research_loop.domain.tasks import ActionRecord, TaskStatus
from research_loop.workers.executor import TaskExecutor
from unit import test_research_closure_contracts as fixtures


class QualificationHandoffContracts(unittest.TestCase):
    """使用临时SQLite及固定PDF替身，未运行真实模型或生产数据库。"""

    def setUp(self):
        """固定临时SQLite，不继承外部测试数据库配置。"""
        with patch.dict(os.environ, {'FIVE_ROOT_TEST_DATABASE_URL': ''}):
            fixtures.ClosureContracts.setUp(self)

    record = fixtures.ClosureContracts.record

    def failure(self, answer='当前明确候选正文。', summary='独立处理说明。', **changes):
        """构造与真实finish入口一致的失败行动和合法服务器身份。"""
        arguments = {'answer_markdown': answer, 'processing_summary': summary, 'goals': [{'goal_id': 'mechanism', 'status': 'satisfied', 'reason': '合成'}], **changes}
        action = ActionRecord(task_id=self.tid, sequence=len(self.db.list_actions(self.tid))+1,
            tool_name='finish_task', arguments=arguments, status='failed')
        candidate = {'task_id': self.tid, 'action_id': action.action_id, 'input_hash': finish_input_hash(arguments),
            'answer_body_chars': len(answer), 'processing_summary_chars': len(summary),
            'answer_content_hash': hashlib.sha256(answer.encode()).hexdigest()}
        action.observation = {'ok': False, 'kind': 'execution_error', 'message': 'synthetic qualification blocked',
            'data': {'finish_candidate': candidate, 'coverage_blocker': {'goal_id': 'mechanism',
                'code': 'GOAL_QUALIFICATION_BLOCKED', 'state_fingerprint': self.coverage.qualification_fingerprint(self.tid, 'mechanism')}}}
        self.db.save_action(action.model_dump(mode='json'))
        handoff = qualification_handoff(self.db.list_actions(self.tid))
        return action, {'coverage_required': True, 'research_handoff': handoff, 'final_reason': 'qualification handoff unresolved'}

    def test_full_arguments_hash_keeps_objective_and_all_fields(self):
        """完整输入精确幂等；objective改写不被当作相同输入。"""
        first = {'subtasks': [{'objective': '问题甲', 'goal_ids': ['mechanism']}]}
        second = deepcopy(first); second['subtasks'][0]['objective'] = '问题乙'
        self.assertNotEqual(finish_input_hash(first), finish_input_hash(second))
        self.assertEqual(finish_input_hash({'b': 2, 'a': 1}), finish_input_hash({'a': 1, 'b': 2}))

    def test_handoff_only_keeps_reference_and_is_active_until_dependencies_change(self):
        """资格交接不复制正文，改写目标描述不会解除资格事实。"""
        action, state = self.failure()
        handoff = state['research_handoff']
        self.assertNotIn('answer_body_markdown', repr(handoff))
        self.assertEqual(handoff['finish_candidate_ref']['action_id'], action.action_id)
        self.assertTrue(qualification_handoff_active(handoff, self.coverage, self.tid))
        plan = deepcopy(self.coverage.plan(self.tid)); plan['goals'][0]['description'] = '换一种描述'
        self.record(self.tid, 'set_research_plan', {'research_plan': plan})
        self.assertTrue(qualification_handoff_active(handoff, self.coverage, self.tid))
        plan['goals'][0]['sources'] = ['cvpr']
        self.record(self.tid, 'set_research_plan', {'research_plan': plan})
        self.assertFalse(qualification_handoff_active(handoff, self.coverage, self.tid))

    def test_bound_candidate_body_has_priority_and_summary_is_separate(self):
        """后续失败文本不能吞掉当前绑定候选；处理说明独立表达。"""
        action, state = self.failure()
        state['final_answer'] = '后续失败回复'
        result = TaskExecutor(self.settings, self.db)._deliverable_result(self.tid, state, 'synthetic failure')
        self.assertEqual(result.answer_body_markdown, '当前明确候选正文。')
        self.assertEqual(result.answer_markdown.count('当前明确候选正文。'), 1)
        self.assertNotIn('独立处理说明', result.answer_body_markdown)
        self.assertIn('独立处理说明', result.processing_summary)
        self.assertEqual(result.retained_answer_source['action_id'], action.action_id)
        self.assertEqual(TaskExecutor(self.settings, self.db)._delivery_status(self.tid, state, result), TaskStatus.partial)

    def test_multiple_candidates_require_exact_reference(self):
        """多个候选不按时间挑选，仍使用明确绑定的先前行动。"""
        first, state = self.failure('第一候选。')
        self.failure('第二候选。')
        worker = TaskExecutor(self.settings, self.db)
        self.assertEqual(worker._deliverable_result(self.tid, state).answer_body_markdown, '第一候选。')
        self.assertIsNone(worker._bound_finish_candidate(self.tid, {'coverage_required': True}))

    def test_forged_candidate_identity_arguments_and_body_are_rejected(self):
        """task/action/hash/正文任一篡改均不能恢复候选。"""
        action, state = self.failure()
        worker = TaskExecutor(self.settings, self.db)
        for key in ('task_id', 'action_id', 'input_hash'):
            with self.subTest(key=key):
                forged = deepcopy(state); forged['research_handoff']['finish_candidate_ref'][key] = 'foreign'
                self.assertIsNone(worker._bound_finish_candidate(self.tid, forged))
        for mutate in ('arguments', 'body'):
            with self.subTest(mutate=mutate):
                altered = action.model_dump(mode='json')
                if mutate == 'arguments':
                    altered['arguments']['answer_markdown'] = '篡改参数'
                else:
                    altered['observation']['data']['finish_candidate']['answer_content_hash'] = '0'*64
                self.db.save_action(altered)
                if mutate == 'arguments':
                    self.db.execute('UPDATE actions SET arguments_json=? WHERE action_id=?',
                        (json.dumps(altered['arguments']), action.action_id))
                self.assertIsNone(worker._bound_finish_candidate(self.tid, state))
                self.db.execute('UPDATE actions SET arguments_json=? WHERE action_id=?',
                    (json.dumps(action.arguments), action.action_id))
                self.db.save_action(action.model_dump(mode='json'))

    def test_cancellation_stays_cancelled_even_with_candidate(self):
        """合法候选是交付内容，不把取消状态提升为completed。"""
        _, state = self.failure(); state['cancelled'] = True
        worker = TaskExecutor(self.settings, self.db)
        result = worker._deliverable_result(self.tid, state)
        self.assertEqual(worker._delivery_status(self.tid, state, result), TaskStatus.cancelled)

    def test_unvalidated_failure_does_not_create_candidate_reference(self):
        """一般权限失败或无服务器身份候选不能进入恢复通道。"""
        action, _ = self.failure()
        action.observation['data']['finish_candidate']['task_id'] = 'foreign-task'
        self.db.save_action(action.model_dump(mode='json'))
        handoff = qualification_handoff(self.db.list_actions(self.tid))
        self.assertIsNone(handoff['finish_candidate_ref'])

    def test_precommit_snapshot_uses_exact_pending_action_instead_of_latest(self):
        """观察先落库的异常只按冻结待执行序号/hash恢复，不能取另一失败候选。"""
        action, _ = self.failure('精确在途候选。')
        self.failure('后续无关候选。')
        state = {'action_sequence': action.sequence-1,
            'pending_tool_calls': [{'name': 'finish_task', 'arguments': action.arguments}]}
        worker = TaskExecutor(self.settings, self.db)
        self.assertEqual(worker._bound_finish_candidate(self.tid, state)['answer_body_markdown'], '精确在途候选。')
        state['pending_tool_calls'][0]['arguments'] = {'answer_markdown': '改写参数'}
        self.assertIsNone(worker._bound_finish_candidate(self.tid, state))

    def test_unknown_recovery_keeps_bound_candidate_and_cancelled_status(self):
        """旧unknown不重发，恢复读精确快照候选，取消仍为cancelled。"""
        _, state = self.failure()
        self.db.save_checkpoint(self.tid, state)
        self.db.save_model_call(self.tid, 1, 'unknown', {}, sent_state='sent')
        self.db.update_task(self.tid, cancel_requested=1)
        def forbidden_provider(_settings):
            """unknown早退不允许实例化或请求供应商。"""
            raise AssertionError('provider must not be created')
        worker = TaskExecutor(self.settings, self.db, forbidden_provider)
        result = worker.execute(self.tid)
        self.assertEqual(result.answer_body_markdown, '当前明确候选正文。')
        self.assertEqual(self.db.get_task(self.tid)['status'], 'cancelled')
        self.assertEqual(len(self.db.list_model_calls(self.tid)), 1)

    def test_resolved_newer_other_goal_cannot_hide_older_active_blocker(self):
        """每个目标独立保留阻断，另一目标后来的已解除失败不能抹掉旧阻断。"""
        first, _ = self.failure()
        second, _ = self.failure('另一目标候选。')
        second.observation['data']['coverage_blocker'].update(goal_id='other', state_fingerprint='old-other')
        self.db.save_action(second.model_dump(mode='json'))
        class CurrentQualification:
            """仅供应当前资格事实，不按action/objective制造进展。"""
            def qualification_fingerprint(inner, task_id, goal_id):
                """原目标保持，另一目标事实已变化。"""
                return self.coverage.qualification_fingerprint(task_id, goal_id) if goal_id == 'mechanism' else 'new-other'
        handoff = qualification_handoff(self.db.list_actions(self.tid), CurrentQualification(), self.tid)
        self.assertEqual(handoff['coverage_blocker']['goal_id'], 'mechanism')
        self.assertEqual(handoff['finish_candidate_ref']['action_id'], first.action_id)
