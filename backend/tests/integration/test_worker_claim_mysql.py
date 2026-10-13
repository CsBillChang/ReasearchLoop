# 真实 MySQL 并发领取测试：使用显式测试环境，不允许多个 Worker 同时持有同一有效租约。
import os
import unittest
from research_loop.storage.mysql import Database
from research_loop.domain.tasks import TaskCreate
from research_loop.services.tasks import TaskService


class MySQLClaimTests(unittest.TestCase):
    def test_two_connections_only_one_claims(self):
        if os.getenv("RESEARCHLOOP_RUN_MYSQL_TESTS") != "1":
            self.skipTest("explicit isolated MySQL test required")
        url = os.getenv("RESEARCHLOOP_TEST_DATABASE_URL", "mysql://researchloop:researchloop_local@127.0.0.1:33316/researchloop")
        db = Database(url); db.ensure_schema()
        key = "claim-test-" + os.urandom(6).hex()
        task = TaskService(db).create(TaskCreate(prompt="claim", idempotency_key=key))
        first = Database(url).claim_task("mysql-worker-a", task_id=task.task_id)
        second = Database(url).claim_task("mysql-worker-b", task_id=task.task_id)
        self.assertIsNotNone(first)
        self.assertIsNone(second)
