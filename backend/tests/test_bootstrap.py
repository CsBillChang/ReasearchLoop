# 启动检查：验证基础目录、配置与应用装配；不把可导入当成真实模型可用。
"""HTTP contract tests use an isolated SQLite database and never start a worker."""
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from research_loop.main import app


class BootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        os.environ["DATABASE_URL"] = "sqlite:///" + str(root / "researchloop_test_ag1.db")
        os.environ["RESEARCHLOOP_DATA_DIR"] = str(root / "data")
        os.environ["RESEARCHLOOP_OUTPUT_DIR"] = str(root / "notes")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.client = TestClient(app)

    def test_health_reports_ag1_configuration_without_claiming_worker(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["stage"], "AG7.5")
        self.assertFalse(response.json()["model_probed"])
        self.assertTrue(response.json()["worker_required"])

    def test_submit_idempotency_cancel_and_events(self):
        payload = {"prompt": "test", "idempotency_key": "bootstrap-one", "save_policy": "deny"}
        first = self.client.post("/api/tasks", json=payload)
        self.assertEqual(first.status_code, 202)
        task_id = first.json()["task_id"]
        again = self.client.post("/api/tasks", json=payload)
        self.assertEqual(again.status_code, 202)
        self.assertEqual(again.json()["task_id"], task_id)
        cancelled = self.client.post("/api/tasks/" + task_id + "/cancel")
        self.assertEqual(cancelled.status_code, 200)
        events = self.client.get("/api/tasks/" + task_id + "/events")
        self.assertEqual(events.status_code, 200)
        self.assertGreaterEqual(len(events.json()["events"]), 2)

    def test_backend_does_not_serve_frontend(self):
        self.assertEqual(self.client.get("/").status_code, 404)


if __name__ == "__main__":
    unittest.main()
