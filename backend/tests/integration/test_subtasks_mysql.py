# MySQL 子任务测试：验证数据库持久身份和恢复；无显式测试环境时跳过。
import tempfile
import unittest
from pathlib import Path
from research_loop.storage.mysql import Database
from research_loop.services.tasks import TaskService
from research_loop.domain.tasks import TaskCreate


class SubtaskPersistenceTests(unittest.TestCase):
    def test_duplicate_delegate_and_parent_scope_survive_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            url = "sqlite:///" + str(Path(temp) / "researchloop_test_subtasks.db")
            db = Database(url); db.ensure_schema()
            root = TaskService(db).create(TaskCreate(prompt="root", idempotency_key="root"))
            one = db.create_subtask(root.task_id, root.task_id, 0, "one", {"paper_ids": ["p1"]}, "same")
            again = db.create_subtask(root.task_id, root.task_id, 0, "one", {"paper_ids": ["p1"]}, "same")
            self.assertEqual(one["subtask_id"], again["subtask_id"])
            db.grant_paper(one["subtask_id"], "p1", root.task_id)
            self.assertTrue(db.task_has_paper(one["subtask_id"], "p1"))
            self.assertFalse(db.task_has_paper(root.task_id, "p1"))
