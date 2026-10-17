"""工具合同测试：模型schema、参数验证和handler共用同一注册入口。"""
import tempfile
import unittest
from pathlib import Path

from research_loop.domain.tasks import SavePolicy
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.services.notes import NoteService
from research_loop.tools.base import ToolContext
from research_loop.tools.notes import specs as note_specs
from research_loop.tools.papers import specs as paper_specs
from research_loop.tools.registry import ToolRegistry


class ToolContractTests(unittest.TestCase):
    """完整登记清单变化时显式更新期望，参数和禁存断言保持。"""

    def test_paper_and_note_tools_have_matching_schemas_and_validation(self):
        """视觉委派加入登记清单，所有工具仍共用参数模型与执行合同。"""
        registry = ToolRegistry(paper_specs() + note_specs())
        self.assertEqual(registry.names(), ["delegate_visual_reading", "retrieve_evidence", "inspect_paper_media", "search_papers", "read_paper", "revise_note", "create_note", "save_note"])
        for schema in registry.schemas():
            self.assertIn("name", schema["function"])
            self.assertIn("properties", schema["function"]["parameters"])
        ctx = ToolContext(task_id="t", save_policy=SavePolicy.deny, services={})
        result = registry.execute("read_paper", {"start_page": 0}, ctx)
        self.assertEqual(result.kind, "validation_error")

    def test_unknown_tool_and_save_permission_are_explicit(self):
        registry = ToolRegistry(note_specs())
        ctx = ToolContext(task_id="t", save_policy=SavePolicy.deny, services={})
        self.assertEqual(registry.execute("missing", {}, ctx).kind, "validation_error")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = Database("sqlite:///" + str(root / "test.db"))
            db.ensure_schema()
            notes = NoteService(db, FileStore(root / "data", root / "out", root))
            ctx.services["notes"] = notes
            draft = registry.execute("create_note", {"title": "x", "markdown": "# x"}, ctx)
            self.assertTrue(draft.ok)
            denied = registry.execute("save_note", {"draft_id": draft.data["draft_id"], "content_hash": draft.data["content_hash"]}, ctx)
            self.assertFalse(denied.ok)
            self.assertEqual(denied.kind, "execution_error")
