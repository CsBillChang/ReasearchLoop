"""来源/日期条件归属与合法失败候选的零网络单元合同。"""
from copy import deepcopy
import json
import socket
import unittest
from unittest.mock import Mock, patch

from research_loop.agents.memory import project_finish_candidates
from research_loop.domain.tasks import FinishTaskInput, ResearchBrief, SavePolicy
from research_loop.services.coverage import CoverageEligibilityError, CoverageService
from research_loop.tools.base import ToolContext, execute_spec
from research_loop.tools.delegation import specs


class ScopeContractTests(unittest.TestCase):
    """使用数据库替身隔离计划归属；实际授权链由集成测试覆盖。"""

    def setUp(self):
        """固定用户原句，禁止任何socket连接。"""
        network = patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden'))
        network.start(); self.addCleanup(network.stop)
        self.db = Mock()
        self.db.get_task.return_value = {'prompt': '解释上传材料，不保存。只用ICLR 2025年的论文。',
            'save_policy': 'deny', 'attachments_json': '[{"input_ref":"upload-a","paper_id":"upload-p"}]'}
        self.db.list_actions.return_value = []
        self.db.list_subtasks.return_value = []
        self.service = CoverageService(self.db)
        self.service.plan = Mock(return_value=None)
        self.service._latest_answer = Mock(return_value=None)
        self.user = {'goal_id': 'explain', 'description': '解释上传材料', 'origin': 'user',
                     'source_quote': '解释上传材料', 'min_papers': 1, 'requires_fulltext': True}
        self.brief = ResearchBrief(goal='解释上传材料', sources=['arxiv'], date_from='2026-01-01',
            input_refs=['upload-a'], goals=[self.user])

    def install(self):
        """用真实prepare返回观察合同，后续按持久计划重放。"""
        plan = self.service.prepare_plan('root', self.brief)
        self.service.plan.return_value = plan
        return plan

    def test_new_retrieval_preferences_do_not_filter_upload_with_unknown_metadata(self):
        """新合同偏好不伪造上传来源；真实授权附件仍满足范围。"""
        plan = self.install()
        self.assertEqual(plan['_scope_semantics'], 'goal_owned_constraints')
        self.assertTrue(self.service.in_scope({}, plan['goals'][0], plan, {'upload-a': 'upload-p'}, 'upload-p'))
        self.assertFalse(self.service.in_scope({}, plan['goals'][0], plan, {'upload-a': 'upload-p'}, 'foreign'))

    def test_new_preferences_change_but_user_and_root_contracts_stay(self):
        """可改顶层检索偏好，不能随之弱化用户目标、数量、附件及保存合同。"""
        plan = self.install()
        changed = self.service.prepare_plan('root', ResearchBrief.model_validate(plan | {
            'sources': ['iclr'], 'date_from': None, 'date_to': '2025-12-31'}))
        self.assertEqual(changed['sources'], ['iclr'])
        self.assertEqual(plan['sources'], ['arxiv'])
        for change in ({'input_refs': []}, {'save_requested': True}, {'requested_count': 2}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'task constraint'):
                self.service.prepare_plan('root', ResearchBrief.model_validate(plan | change))
        hard = deepcopy(plan); hard['goals'][0]['sources'] = ['iclr']
        with self.assertRaisesRegex(ValueError, 'existing goals'):
            self.service.prepare_plan('root', ResearchBrief.model_validate(hard))

    def test_unknown_real_user_sources_dates_cannot_satisfy(self):
        """硬条件在对应用户目标，分别给出实际未知事实及归属。"""
        goal = self.user | {'sources': ['iclr'], 'date_from': '2025-01-01', 'date_to': '2025-12-31'}
        plan = self.install()
        failures = self.service.scope_failures({}, goal, plan, {'upload-a': 'upload-p'}, 'upload-p')
        self.assertEqual({f['fact'] for f in failures}, {'source_unknown', 'published_unknown'})
        self.assertTrue(all(f['scope'] == 'goal' and f['origin'] == 'user' for f in failures))
        self.assertTrue(self.service.in_scope({'source': 'ICLR', 'published': '2025'}, goal, plan,
                                             {'upload-a': 'upload-p'}, 'upload-p'))
        self.assertFalse(self.service.in_scope({'source': 'ICLR', 'published': '2024'}, goal, plan,
                                              {'upload-a': 'upload-p'}, 'upload-p'))

    def test_invalid_published_placeholder_does_not_satisfy_one_sided_date(self):
        """非日历题录不能因字符串排序通过仅起始日期的硬限制。"""
        plan = self.install()
        goal = self.user | {'date_from': '2025-01-01'}
        for published in ('xxxx', '0000', '2025-99', None):
            with self.subTest(published=published):
                self.assertFalse(self.service.in_scope({'published': published}, goal, plan,
                    {'upload-a': 'upload-p'}, 'upload-p'))

    def test_unmarked_old_plan_cannot_be_migrated_by_model_or_prepare(self):
        """模型伪造标记会被参数模型忽略，旧观察和顶层硬范围保持原样。"""
        old = self.brief.model_dump(mode='json'); original = deepcopy(old)
        self.service.plan.return_value = old
        updated = self.service.prepare_plan('root', ResearchBrief.model_validate(old | {
            '_scope_semantics': 'goal_owned_constraints'}))
        self.assertNotIn('_scope_semantics', updated)
        self.assertFalse(self.service.in_scope({}, old['goals'][0], old, {'upload-a': 'upload-p'}, 'upload-p'))
        self.assertEqual(old, original)
        with self.assertRaisesRegex(ValueError, 'task constraint'):
            self.service.prepare_plan('root', ResearchBrief.model_validate(old | {'sources': []}))

    def test_real_resume_can_change_legacy_contract_once(self):
        """真实恢复答案重定范围后使用新语义，随后再保护user硬限制。"""
        old = self.brief.model_dump(mode='json'); self.service.plan.return_value = old
        answer = {'clarification_id': 'resume-actual', 'answer': '只用ICLR 2025年的论文。'}
        self.service._latest_answer.return_value = answer
        hard_goal = self.user | {'source_quote': answer['answer'], 'sources': ['iclr'],
                                'date_from': '2025-01-01', 'date_to': '2025-12-31'}
        new = self.service.prepare_plan('root', ResearchBrief.model_validate(old | {'sources': [], 'goals': [hard_goal]}))
        self.assertEqual(new['_scope_semantics'], 'goal_owned_constraints')
        self.assertEqual(new['clarification_id'], answer['clarification_id'])
        self.service.plan.return_value = new
        with self.assertRaisesRegex(ValueError, 'existing goals'):
            self.service.prepare_plan('root', ResearchBrief.model_validate(new | {'goals': [self.user]}))

    def test_internal_marker_is_not_a_model_protocol_field(self):
        """模型schema未新增内部语义字段。"""
        self.assertNotIn('_scope_semantics', ResearchBrief.model_json_schema()['properties'])
        self.assertNotIn('action_identity', FinishTaskInput.model_json_schema()['properties'])

    def test_failed_candidate_projection_preserves_original_and_separate_summary(self):
        """模型投影去掉重复正文，原观察完整保留供Worker按hash恢复。"""
        candidate = {'task_id': 'root', 'action_id': 'action-a', 'input_hash': 'a'*64,
            'answer_content_hash': 'b'*64, 'answer_body_markdown': '正文'*200,
            'processing_summary': '处理说明'}
        messages = [{'role': 'assistant', 'tool_calls': [{'id': 'call-a', 'function': {'name': 'finish_task'}}]},
                    {'role': 'tool', 'tool_call_id': 'call-a', 'content': json.dumps({'ok': False,
                     'data': {'coverage_blocker': {'code': 'GOAL_QUALIFICATION_BLOCKED'}, 'finish_candidate': candidate}})}]
        original = deepcopy(messages)
        projected = project_finish_candidates(messages)
        value = json.loads(projected[1]['content'])['data']['finish_candidate']
        self.assertNotIn('answer_body_markdown', value)
        self.assertNotIn('processing_summary', value)
        self.assertEqual(value['answer_body_chars'], 400)
        self.assertEqual(value['action_id'], 'action-a')
        self.assertEqual(messages, original)

    def test_tool_emits_only_local_eligibility_diagnostic(self):
        """资格错误有结构化观察；普通权限错误不携带候选。"""
        spec = next(s for s in specs() if s.name == 'finish_task')
        self.service.finish = Mock(side_effect=CoverageEligibilityError('资格缺口',
            {'code': 'GOAL_QUALIFICATION_BLOCKED', 'goal_id': 'explain'}, {'action_id': 'a'}))
        context = ToolContext('root', SavePolicy.deny, services={'coverage': self.service,
            'finish_action_identity': {'task_id': 'root', 'action_id': 'a', 'input_hash': 'b'*64}})
        arguments = {'answer_markdown': '正文', 'goals': [{'goal_id': 'explain', 'status': 'satisfied', 'reason': '已有材料'}]}
        value = execute_spec(spec, arguments, context)
        self.assertFalse(value.ok); self.assertFalse(value.retryable)
        self.assertEqual(value.data['coverage_blocker']['goal_id'], 'explain')
        self.assertIn('finish_candidate', value.data)
        self.service.finish.side_effect = PermissionError('foreign draft')
        value = execute_spec(spec, arguments, context)
        self.assertFalse(value.ok); self.assertEqual(value.data, {})
