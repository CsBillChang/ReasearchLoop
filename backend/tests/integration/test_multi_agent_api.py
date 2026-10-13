# API 多角色测试：提交与查询接口需返回持久化的子任务、澄清及结果状态。
import os
import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient


class MultiAgentAPITests(unittest.TestCase):
    def test_resume_endpoint_preserves_task_id(self):
        with tempfile.TemporaryDirectory() as temp:
            os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(temp) / "researchloop_test_api.db")
            os.environ["RESEARCHLOOP_DATA_DIR"] = str(Path(temp) / "data")
            os.environ["RESEARCHLOOP_OUTPUT_DIR"] = str(Path(temp) / "out")
            from research_loop.main import app
            from research_loop.storage.mysql import Database
            from research_loop.services.tasks import TaskService
            db = Database(os.environ["DATABASE_URL"]); db.ensure_schema()
            with TestClient(app) as client:
                created = client.post("/api/tasks", json={"prompt": "count", "idempotency_key": "api-resume"}).json()
                task_id = created["task_id"]
                question = TaskService(db).request_clarification(task_id, "count?", ["count"], ["5"])
                db.save_checkpoint(task_id, {"messages": [{"role": "user", "content": "count"}, {"role": "assistant", "content": "how many?"}], "pending_tool_calls": [], "finished": True, "final_reason": "clarification requested"})
                db.update_task(task_id, status="needs_input")
                response = client.post(f"/api/tasks/{task_id}/resume", json={"clarification_id": question["clarification_id"], "answer": "2", "idempotency_key": "api-r1"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["task_id"], task_id)
