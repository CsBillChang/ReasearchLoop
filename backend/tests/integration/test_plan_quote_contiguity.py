"""连续规划引句的隔离服务回归；合成输入不依赖私有试验工件。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from research_loop.agents.supervisor import SYSTEM_PROMPT
from research_loop.config import Settings
from research_loop.domain.tasks import ResearchBrief, ResumeInput, TaskCreate
from research_loop.services.conversations import ConversationService
from research_loop.services.coverage import CoverageService
from research_loop.services.tasks import TaskService
from research_loop.storage.memory import MemoryStore
from research_loop.storage.mysql import Database
from research_loop.tools.delegation import specs


class PlanQuoteContiguityTests(unittest.TestCase):
    """保护逐字来源合同，区分连续引句、用户原生省略号和拼接伪引句。"""

    def setUp(self):
        """每例创建独立SQLite和目录，阻断真实模型及部署服务连接。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.settings = Settings(project_root=root, data_dir=root / 'data', output_dir=root / 'notes',
            database_url='sqlite:///' + str(root / 'test.sqlite'),
            deepseek_api_key=None, retrieval_enabled=False, discovery_backends={})
        for target in ('socket.socket.connect', 'socket.socket.connect_ex'):
            blocker = patch(target, side_effect=AssertionError('规划引句回归禁止网络连接'))
            blocker.start()
            self.addCleanup(blocker.stop)
        configuration = patch('research_loop.config.get_settings', return_value=self.settings)
        configuration.start()
        self.addCleanup(configuration.stop)
        self.db = Database(self.settings.database_url)
        self.db.ensure_schema()
        self.conversations = ConversationService(self.db, self.settings)
        self.tasks = TaskService(self.db, self.settings)
        self.service = CoverageService(self.db)
        self.prompt = ('请建立合成记录 DEMO-QUOTE，不代表论文结论。\n'
                       '记录共有7条断言，请计算当前保留数量。\n'
                       '请说明计数口径，保留讨论供后续继续。\n'
                       '不要找论文、精读或保存文件。')
        self.task_id = self.create_task(self.prompt)

    def create_task(self, prompt: str) -> str:
        """通过原生服务建立新会话根，不与其他例共享数据库或权限。"""
        cid = self.conversations.create('合成规划引句', uuid4().hex)['conversation_id']
        return self.tasks.create(TaskCreate(prompt=prompt, conversation_id=cid,
            idempotency_key=uuid4().hex, save_policy='deny')).task_id

    @staticmethod
    def goal(identity: str, quote: str) -> dict:
        """构造有来源锚点的零取材目标，避免引句测试受到论文门干扰。"""
        return {'goal_id': identity, 'description': '按本轮明确要求整理合成讨论',
                'origin': 'user', 'source_quote': quote, 'min_papers': 0}

    @staticmethod
    def brief(goals: list[dict]) -> ResearchBrief:
        """只使用普通memory计划，不改变来源检查或保存权限。"""
        return ResearchBrief(goal='继续本轮合成记录', task_kind='memory', goals=goals)

    def resume_answer(self, answer: str) -> None:
        """走真实澄清与resume事务，使服务能够核对回答所属任务。"""
        clarification = self.tasks.request_clarification(
            self.task_id, '请补充本轮输出形式。', ['format'], [])
        self.db.save_checkpoint(self.task_id, {
            'messages': [{'role': 'system', 'content': '合成规划测试。'},
                         {'role': 'user', 'content': self.prompt}],
            'pending_tool_calls': [], 'final_reason': 'clarification requested'})
        self.db.update_task(self.task_id, status='needs_input')
        self.tasks.resume(ResumeInput(clarification_id=clarification['clarification_id'],
            answer=answer, idempotency_key=uuid4().hex), self.task_id)

    def test_fourth_spliced_goal_rejects_entire_plan_without_rewriting_input(self):
        """前三引句连续而第四条用省略号拼接时，整份计划仍拒绝。"""
        goals = [self.goal('count', '请计算当前保留数量。'),
                 self.goal('explain', '请说明计数口径，'),
                 self.goal('continue', '保留讨论供后续继续。'),
                 self.goal('no_retrieval',
                     '请建立合成记录 DEMO-QUOTE，不代表论文结论。……不要找论文、精读或保存文件。')]
        self.assertEqual([item['source_quote'] in self.prompt for item in goals],
                         [True, True, True, False])
        candidate = self.brief(goals)
        original = deepcopy(candidate.model_dump())
        with self.assertRaisesRegex(ValueError, 'verbatim'):
            self.service.prepare_plan(self.task_id, candidate)
        self.assertEqual(candidate.model_dump(), original)
        self.assertIsNone(self.service.plan(self.task_id))
        self.assertEqual(self.db.list_actions(self.task_id), [])

    def test_one_short_contiguous_quote_is_accepted(self):
        """目标仅引用直接对应的连续短片段，无需把整段背景拼入出处。"""
        quote = '不要找论文、精读或保存文件。'
        plan = self.service.prepare_plan(self.task_id, self.brief([self.goal('scope', quote)]))
        self.assertEqual(plan['goals'][0]['source_quote'], quote)
        self.assertEqual(plan['save_policy'], 'deny')

    def test_separated_requirements_use_independent_contiguous_quotes(self):
        """相隔的用户要求各有独立目标与连续出处，均可通过原合同。"""
        quotes = ['请计算当前保留数量。', '不要找论文、精读或保存文件。']
        goals = [self.goal('requirement_' + str(index), quote) for index, quote in enumerate(quotes)]
        plan = self.service.prepare_plan(self.task_id, self.brief(goals))
        self.assertEqual([item['source_quote'] for item in plan['goals']], quotes)

    def test_paraphrase_added_punctuation_and_reordered_text_are_rejected(self):
        """不对引句做同义、标点或顺序容错，避免伪造逐字出处。"""
        for quote in ('请勿找论文、精读或保存文件。', '不要找论文、精读或保存文件！',
                      '不要保存文件、精读或找论文。', '请说明计数口径，……保留讨论供后续继续。'):
            with self.subTest(quote=quote), self.assertRaisesRegex(ValueError, 'verbatim'):
                self.service.prepare_plan(self.task_id, self.brief([self.goal('scope', quote)]))

    def test_literal_ellipsis_in_actual_user_message_is_accepted(self):
        """用户原文确有省略号时允许逐字引用，不以字符黑名单放宽或误杀来源。"""
        quote = '我还没确定……先只整理已有记录。'
        task_id = self.create_task(quote + ' 不保存文件。')
        plan = self.service.prepare_plan(task_id, self.brief([self.goal('scope', quote)]))
        self.assertEqual(plan['goals'][0]['source_quote'], quote)

    def test_other_user_message_and_assistant_text_cannot_supply_current_source(self):
        """数据库存在的其他用户话或助手话不能冒充本轮用户出处。"""
        other_quote = '其他任务要求先阅读所有附录。'
        self.create_task(other_quote)
        assistant_quote = '助手建议额外搜索三篇。'
        link = self.db.fetchone('SELECT conversation_id FROM conversation_links WHERE task_id=?',
                               (self.task_id,))
        with MemoryStore(self.db).transaction() as tx:
            ConversationService.add_message(tx, link['conversation_id'], self.task_id,
                                            'synthetic-assistant', 'assistant', assistant_quote)
        for quote in (other_quote, assistant_quote):
            with self.subTest(quote=quote), self.assertRaisesRegex(ValueError, 'verbatim'):
                self.service.prepare_plan(self.task_id, self.brief([self.goal('scope', quote)]))

    def test_resumed_answer_is_valid_but_cross_source_join_is_rejected(self):
        """真实澄清可独立提供出处，不能与原问题拼成不存在的跨来源引句。"""
        answer = '只使用表格说明，仍然不保存文件。'
        self.resume_answer(answer)
        plan = self.service.prepare_plan(self.task_id, self.brief([self.goal('format', answer)]))
        self.assertEqual(plan['goals'][0]['source_quote'], answer)
        self.assertTrue(plan['clarification_id'])
        joined = self.prompt.splitlines()[0] + '\n' + answer
        with self.assertRaisesRegex(ValueError, 'verbatim'):
            self.service.prepare_plan(self.task_id, self.brief([self.goal('format', joined)]))

    def test_multiline_quote_preserves_original_newline(self):
        """相邻句段连同原始换行可引用，移除换行即不再逐字匹配。"""
        quote = '\n'.join(self.prompt.splitlines()[-2:])
        plan = self.service.prepare_plan(self.task_id, self.brief([self.goal('scope', quote)]))
        self.assertEqual(plan['goals'][0]['source_quote'], quote)
        with self.assertRaisesRegex(ValueError, 'verbatim'):
            self.service.prepare_plan(self.task_id,
                                      self.brief([self.goal('scope', quote.replace('\n', ''))]))

    def test_supervisor_and_tool_describe_contiguous_verbatim_quotes(self):
        """主提示与实际工具描述均解释连续来源，具体例句只留在测试夹具。"""
        self.assertIn('连续、逐字相同', SYSTEM_PROMPT)
        self.assertIn('不要用省略号拼接', SYSTEM_PROMPT)
        description = next(item.description for item in specs() if item.name == 'set_research_plan')
        self.assertIn('short contiguous span', description)
        self.assertIn('join separated spans with ellipses', description)
        self.assertNotIn('DEMO-QUOTE', SYSTEM_PROMPT + description)


if __name__ == '__main__':
    unittest.main()
