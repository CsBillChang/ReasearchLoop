"""冻结工具范围的schema与执行保持一致，日常缺省不收缩。"""
import unittest
from unittest.mock import Mock
from pydantic import BaseModel
from research_loop.domain.tasks import Observation, SavePolicy
from research_loop.tools.base import ToolContext, ToolSpec
from research_loop.tools.registry import ToolRegistry


class EmptyArgs(BaseModel):
    pass


class ToolScopeTests(unittest.TestCase):
    """使用明确替身验证禁用工具永不进入handler。"""

    def test_schema_names_execution_and_role_share_scope(self):
        """名称清单、可见schema和调用拒绝一致，根范围不能提升子角色权限。"""
        allowed=Mock(return_value=Observation(ok=True,kind='success'))
        blocked=Mock(return_value=Observation(ok=True,kind='success'))
        specs=[ToolSpec('read','read',EmptyArgs,allowed,('supervisor',)),
               ToolSpec('search','search',EmptyArgs,blocked)]
        registry=ToolRegistry(specs,allowed_names={'read'})
        self.assertEqual(registry.names(),['read'])
        self.assertEqual([s['function']['name'] for s in registry.schemas('supervisor')],['read'])
        self.assertEqual(registry.schemas('researcher'),[])
        ctx=ToolContext('t',SavePolicy.deny)
        self.assertTrue(registry.execute('read',{},ctx).ok)
        self.assertFalse(registry.execute('search',{},ctx).ok)
        blocked.assert_not_called()
        self.assertEqual(len(ToolRegistry(specs).schemas('supervisor')),2)
        self.assertEqual(ToolRegistry(specs,allowed_names=set()).schemas(),[])
