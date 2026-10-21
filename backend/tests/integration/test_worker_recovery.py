# Worker 恢复测试：按持久收据恢复任务，保护已发生的模型/文件副作用。
import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from pydantic import BaseModel

from research_loop.config import Settings
from research_loop.domain.tasks import TaskCreate
from research_loop.providers.deepseek import ModelResponse, ToolCall
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor


class DynamicProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools=None, force_text=False):
        """合成供应商给出明确测试usage；缺usage阻断由新预算测试独立覆盖。"""
        response = self._complete(messages, tools, force_text)
        response.usage = {"total_tokens": 1}
        return response

    def _complete(self, messages, tools=None, force_text=False):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(None, [ToolCall("r", "read_paper", {"target_ref": "rarr", "start_page": 1, "page_count": 2})])
        if self.calls == 2:
            observation = json.loads(next(message["content"] for message in reversed(messages) if message.get("role") == "tool"))
            ids = [page["evidence_id"] for page in observation["data"]["pages"]]
            return ModelResponse(None, [ToolCall("c", "create_note", {
                "title": "RARR page note",
                "markdown": "# RARR\n\n## Research question\nWhat is the task?\n\n## Core method\nA constrained method.\n\n## Conclusion\nThe returned pages support only this scoped statement.",
                "evidence_ids": ids,
            })])
        if self.calls == 3:
            observation = json.loads(next(message["content"] for message in reversed(messages) if message.get("role") == "tool"))
            return ModelResponse(None, [ToolCall("s", "save_note", {
                "draft_id": observation["data"]["draft_id"], "content_hash": observation["data"]["content_hash"],
            })])
        return ModelResponse("# 完成\n\n证据范围已标明。", [], "stop")

    def close(self):
        pass


class UnknownProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools=None, force_text=False):
        self.calls += 1
        raise RuntimeError("controlled unknown response")

    def close(self):
        pass


class WorkerRecoveryTests(unittest.TestCase):
    def _settings(self, root: Path) -> Settings:
        return Settings(project_root=root, data_dir=root / "data", output_dir=root / "notes",
                        database_url="sqlite:///" + str(root / "researchloop_test_worker.db"),
                        deepseek_api_key="fake", max_model_requests=6, max_tool_actions=10)

    def test_worker_runs_api_created_task_and_persists_receipts(self):
        source = (Path(os.environ.get("RESEARCHLOOP_TEST_PDF_DIR", str(Path.home() / "Downloads"))) / "RARR.pdf")
        if not source.exists():
            self.skipTest("fixed PDF sample unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = self._settings(root)
            db = Database(settings.database_url)
            db.ensure_schema()
            task = TaskService(db, settings).create(TaskCreate(
                prompt="只依据附件前两页生成短稿并保存", idempotency_key="worker-success",
                save_policy="allow", attachments=[{"input_ref": "rarr", "local_path": str(source)}],
            ))
            provider = DynamicProvider()
            # 此夹具是AG1旧单Agent合同，恢复时不追加新AG2覆盖完成条件。
            from research_loop.agents.graph import initial_state
            db.save_checkpoint(task.task_id, initial_state(task.task_id, task.prompt, 'allow',
                {'rarr': {'input_ref': 'rarr', 'local_path': str(source)}}, 6, 10))
            result = TaskExecutor(settings, db, lambda _: provider).execute(task.task_id)
            view = TaskService(db, settings).view(db.get_task(task.task_id))
            self.assertEqual(view.status.value, "completed")
            self.assertEqual(provider.calls, 4)
            self.assertTrue(result.saved_path)
            self.assertEqual(len(db.list_model_calls(task.task_id)), 4)
            self.assertTrue(Path(result.saved_path).exists())

    def test_unknown_model_result_is_not_resent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = self._settings(root)
            db = Database(settings.database_url)
            db.ensure_schema()
            task = TaskService(db, settings).create(TaskCreate(prompt="unknown", idempotency_key="worker-unknown"))
            provider = UnknownProvider()
            executor = TaskExecutor(settings, db, lambda _: provider)
            executor.execute(task.task_id)
            executor.execute(task.task_id)
            self.assertEqual(provider.calls, 1)
            self.assertTrue(db.has_unknown_model_call(task.task_id))
            self.assertEqual(TaskService(db, settings).view(db.get_task(task.task_id)).status.value, "failed")

    def test_sigkill_after_receipt_before_checkpoint_recovers_without_new_provider_call(self):
        if os.getenv("RESEARCHLOOP_RUN_MYSQL_TESTS") != "1":
            self.skipTest("explicit isolated MySQL test required")
        url = os.getenv("RESEARCHLOOP_TEST_DATABASE_URL")
        if not url:
            self.skipTest("RESEARCHLOOP_TEST_DATABASE_URL is required")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = self._settings(root).model_copy(update={"database_url": url})
            db = Database(url); db.ensure_schema()
            key = "sigkill-" + os.urandom(6).hex()
            task = TaskService(db, settings).create(TaskCreate(prompt="synthetic recovery", idempotency_key=key))
            counter = root / "provider.calls"
            ready = root / "ready"
            env = {**os.environ, "DATABASE_URL": url, "PYTHONPATH": str(Path(__file__).parents[2]),
                   "AG2_RECOVERY_ROOT": str(root), "AG2_RECOVERY_COUNTER": str(counter),
                   "AG2_RECOVERY_READY": str(ready), "AG2_RECOVERY_TASK": task.task_id}
            child = subprocess.Popen([str(Path(__file__).parents[3] / ".venv/bin/python"),
                                      str(Path(__file__).with_name("recovery_process.py")), "crash"], env=env)
            deadline = time.time() + 15
            while time.time() < deadline and not ready.exists():
                time.sleep(0.1)
            self.assertTrue(ready.exists(), "child did not reach receipt/checkpoint boundary")
            child.send_signal(signal.SIGKILL)
            child.wait(timeout=10)
            self.assertEqual(counter.read_text(encoding="utf-8").count("call"), 1)
            db.update_task(task.task_id, lease_until="2000-01-01T00:00:00+00:00")
            subprocess.run([str(Path(__file__).parents[3] / ".venv/bin/python"),
                            str(Path(__file__).with_name("recovery_process.py")), "resume"], env=env,
                           check=True, timeout=30)
            self.assertEqual(counter.read_text(encoding="utf-8").count("call"), 1)
            self.assertEqual(db.list_model_calls(task.task_id)[0]["request_id"], "child-request-1")
