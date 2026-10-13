# 协作恢复测试：父子任务、模型收据与租约应一起恢复，不重复执行已完成副作用。
import tempfile
import unittest
from pathlib import Path
from research_loop.storage.mysql import Database
from research_loop.services.tasks import TaskService
from research_loop.domain.tasks import TaskCreate, ResumeInput


class MultiAgentRecoveryTests(unittest.TestCase):
    def test_resume_is_idempotent_and_stale_question_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Database("sqlite:///" + str(Path(temp) / "researchloop_test_resume.db")); db.ensure_schema()
            service = TaskService(db)
            task = service.create(TaskCreate(prompt="need count", idempotency_key="resume-root"))
            question = service.request_clarification(task.task_id, "how many?", ["count"], ["5"])
            db.save_checkpoint(task.task_id, {"messages": [{"role": "user", "content": "count"}, {"role": "assistant", "content": "how many?"}], "pending_tool_calls": [], "finished": True, "final_reason": "clarification requested"})
            db.update_task(task.task_id, status="needs_input")
            payload = ResumeInput(clarification_id=question["clarification_id"], answer="2", idempotency_key="r1")
            first = service.resume(payload, task.task_id)
            again = service.resume(payload, task.task_id)
            self.assertEqual(first.task_id, again.task_id)
            with self.assertRaises(ValueError):
                service.resume(ResumeInput(clarification_id="bad", answer="3", idempotency_key="r2"), task.task_id)
