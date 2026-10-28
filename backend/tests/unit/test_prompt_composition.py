"""新角色装配与直接结束的零模型合同；原文、供应商和数据库均为合成输入。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from research_loop.agents.prompt_parts import COMMON_PRINCIPLES
from research_loop.agents.researcher import researcher_state
from research_loop.agents.reviewer import reviewer_state, research_reviewer_state
from research_loop.agents.supervisor import initial_messages, SYSTEM_PROMPT


class PromptCompositionTests(unittest.TestCase):
    """检查角色交接输入及不同模式合同，而非要求模型遵从提示词。"""

    def test_supervisor_keeps_original_task_permission_and_one_common_section(self):
        """共享原则不依赖Memory头；原始问题与附件身份仍可直接取得。"""
        prompt = '比较两份附件；不保存。'
        attachments = {'input-B': {'paper_id': 'paper-B'}, 'input-A': {'paper_id': 'paper-A'}}
        original = deepcopy(attachments)
        messages = initial_messages(prompt, 'deny', attachments)
        self.assertEqual(messages[0]['content'].count(COMMON_PRINCIPLES), 1)
        self.assertTrue(messages[0]['content'].startswith(SYSTEM_PROMPT))
        self.assertIn(prompt, messages[1]['content'])
        self.assertIn('后端保存权限：deny', messages[1]['content'])
        self.assertIn('input-A, input-B', messages[1]['content'])
        self.assertEqual(attachments, original)

    def test_researcher_modes_keep_their_contract_and_only_child_input(self):
        """子任务保留精确brief，不携带没有权限执行的父角色委派指引。"""
        for mode in ('screen', 'investigate', 'deep_read'):
            with self.subTest(mode=mode):
                brief = {'mode': mode, 'input_refs': ['input-A'], 'verification_claims': ['合成待查主张'],
                         'start_page': 2, 'page_count': 1}
                original = deepcopy(brief)
                state = researcher_state('child', '合成目标原句', attachments={'input-A': {'paper_id': 'paper-A'}}, brief=brief)
                messages = state['messages']
                self.assertEqual(state['prompt'], '合成目标原句')
                self.assertEqual(state['research_mode'], mode)
                self.assertTrue(messages[0]['content'].startswith('本次mode=' + mode))
                self.assertEqual(messages[0]['content'].count(COMMON_PRINCIPLES), 1)
                self.assertNotIn('delegate_research', json.dumps(messages, ensure_ascii=False))
                self.assertIn('合成目标原句', messages[1]['content'])
                self.assertIn('后端保存权限：deny', messages[1]['content'])
                self.assertEqual(json.loads(messages[2]['content'].split('\n', 1)[1]), original)
                self.assertEqual(brief, original)
                self.assertIn('直接finish_research', messages[0]['content'])
                if mode == 'screen':
                    self.assertIn('不逐条核验', messages[0]['content'])
                    self.assertNotIn('须用assessments逐项返回', messages[0]['content'])
                elif mode == 'investigate':
                    self.assertIn('须用assessments逐项返回', messages[0]['content'])
                else:
                    self.assertIn('完整八点总结由随后论文精读节点生成', messages[0]['content'])

    def test_reviewer_entries_preserve_exact_objects_and_distinct_contracts(self):
        """新消息不污染证据和对象版本，旧稿与通用对象仍用各自结束合同。"""
        bundle = [{'evidence_id': 'e-synthetic', 'text': '合成原句', 'page_number': None}]
        target = {'target_id': 't-synthetic', 'content_hash': 'a' * 64, 'items': [{'item_id': 'answer', 'text': '合成结论'}]}
        questions = ['合成结论是否受原句支持？']
        checks = {'non_target_text_preserved': True}
        original = deepcopy((bundle, target, questions, checks))
        note = reviewer_state('note-child', 'draft-synthetic', 'b' * 64, '合成稿正文', ['e-synthetic'], bundle, questions, checks)
        general = research_reviewer_state('object-child', target, questions, bundle, checks)
        for state in (note, general):
            messages = state['messages']
            self.assertEqual(state['role'], 'reviewer')
            self.assertEqual(state['save_policy'], 'deny')
            self.assertEqual(messages[0]['content'].count(COMMON_PRINCIPLES), 1)
            self.assertNotIn('delegate_research', json.dumps(messages, ensure_ascii=False))
            self.assertEqual(json.loads(messages[-1]['content'].split('\n', 1)[1]), bundle)
        self.assertIn('合成稿正文', note['messages'][1]['content'])
        self.assertIn('draft-synthetic', note['messages'][1]['content'])
        self.assertIn('b' * 64, note['messages'][1]['content'])
        self.assertIn('笔记合同', note['messages'][0]['content'])
        self.assertIn('每个选定对象至少一条判断', general['messages'][0]['content'])
        self.assertEqual(json.loads(general['messages'][1]['content'].split('\n', 1)[1]),
                         {'target': target, 'verification_questions': questions, 'execution_checks': checks})
        self.assertEqual((bundle, target, questions, checks), original)


class DirectFinishHandoffTests(unittest.TestCase):
    """真实图与持久层接线，供应商/PDF替身不调用网络或模型。"""

    def test_finish_without_stage_record_persists_and_reaches_parent(self):
        """子角色直接结束仍事务保存；父角色拿到两篇成果并完成交付。"""
        from integration import test_research_progress_handoff as base
        from research_loop.services.research_progress import ResearchProgressService

        class DirectFinishScenario(base.ProgressScenario):
            """复用已有接线场景，仅去掉单独阶段提交这一步。"""

            def send(self, messages, tools, **kwargs):
                """从真实工具观察生成finish参数，不能直接伪造持久结果。"""
                if any(tool['function']['name'] == 'finish_research' for tool in tools):
                    brief = next(json.loads(message['content'].split('\n', 1)[1]) for message in messages
                                 if message.get('content', '').startswith('结构化子任务与已授权定位：'))
                    ref = brief['input_refs'][0]
                    if self.steps.get(ref) == 1:
                        self.steps[ref] = 2
                return super().send(messages, tools, **kwargs)

        scenario = DirectFinishScenario()
        committed = []
        commit = ResearchProgressService.commit_action

        def checked_commit(service, action, owner):
            """执行真实事务后读取持久快照，验证只有结束动作而无预备阶段动作。"""
            self.assertEqual(action['tool_name'], 'finish_research')
            self.assertFalse(any(row['tool_name'] == 'record_research_progress'
                                 for row in service.db.list_actions(action['task_id'])))
            commit(service, action, owner)
            snapshot = service.snapshot(action['task_id'])
            self.assertEqual(snapshot['revision'], 1)
            self.assertEqual(len(snapshot['findings']), 1)
            self.assertTrue(snapshot['findings'][0]['sources'])
            committed.append(action['task_id'])

        with patch.dict('os.environ', {'CORE_LOOP_TEST_DATABASE_URL': ''}), \
                patch.object(base, 'ProgressScenario', return_value=scenario), \
                patch('research_loop.main.get_settings', lambda: base.app.dependency_overrides[base.get_runtime_settings]()), \
                patch.object(ResearchProgressService, 'commit_action', checked_commit):
            base.ProgressHandoffTests.run_case(self)
        self.assertEqual(len(set(committed)), 2)
        self.assertFalse(any(call['function']['name'] == 'record_research_progress'
                             for request in scenario.requests for message in request['messages']
                             for call in message.get('tool_calls', [])))


if __name__ == '__main__':
    unittest.main()
