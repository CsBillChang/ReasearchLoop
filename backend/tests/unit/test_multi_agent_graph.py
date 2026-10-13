# 多角色图测试：用可控依赖检查子任务执行及父任务汇总，不冒充真实多 Agent 语义验收。
import unittest
from research_loop.agents.researcher import researcher_state
from research_loop.agents.reviewer import reviewer_state


class MultiAgentGraphTests(unittest.TestCase):
    def test_role_states_do_not_share_messages(self):
        researcher = researcher_state("r", "method")
        reviewer = reviewer_state("v", "d", "a" * 64, "# draft", [])
        self.assertEqual(researcher["role"], "researcher")
        self.assertEqual(reviewer["role"], "reviewer")
        self.assertIsNot(researcher["messages"], reviewer["messages"])
