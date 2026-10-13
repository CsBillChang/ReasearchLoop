# 角色工具测试：Supervisor、Researcher、Reviewer 各自的可见/可执行权限应一致。
import unittest
from research_loop.domain.tasks import SavePolicy
from research_loop.tools.base import ToolContext
from research_loop.tools.delegation import specs
from research_loop.tools.registry import ToolRegistry


class RoleToolTests(unittest.TestCase):
    def test_researcher_cannot_delegate_or_save(self):
        registry = ToolRegistry(specs())
        ctx = ToolContext(task_id="s", save_policy=SavePolicy.deny, role="researcher")
        result = registry.execute("delegate_research", {"subtasks": [{"objective": "x"}]}, ctx)
        self.assertFalse(result.ok)
        self.assertEqual(result.kind, "validation_error")
