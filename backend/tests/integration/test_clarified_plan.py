"""澄清计划与精读交付路由：实际API/Worker，模型与PDF使用可控替身。"""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from research_loop.config import Settings
from research_loop.domain.tasks import FinishTaskInput, ResearchBrief, TaskCreate
from research_loop.services.coverage import CoverageService
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_adaptive_research import AdaptiveScenario
from integration.test_collaboration_contract import FixturePDF


class ClarifiedScenario(AdaptiveScenario):
    """复现先存建议5篇、再按真实resume回答重定范围的接口调用。"""

    def __init__(self, count=2, note=False):
        super().__init__()
        self.count, self.note = count, note

    def search(self, *args, **kwargs):
        """产生独立论文身份，候选数量与冻结测试答案相同。"""
        result = super().search(*args, **kwargs)
        first = result.candidates[0]
        result.candidates = [first.model_copy(update={"paper_id": f"{self.prefix}-{i}", "title": f"Paper {i}"}) for i in range(self.count)]
        return result

    def handler(self, request):
        """模型替身按实际工具观察决策；DPR输出只在明确精读笔记用例出现。"""
        body = json.loads(request.content)
        self.requests.append(body)
        messages = body["messages"]
        names = {t["function"]["name"] for t in body.get("tools", [])}
        observations = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        done = [c["function"]["name"] for m in messages for c in m.get("tool_calls", [])]
        if not names and body.get("max_tokens") == 4096:
            return self.response(content="# 精读笔记\n原文方法利用语言嵌入。\n（完）")
        if "finish_research" in names:
            if not observations:
                return self.response("search_papers", {"query": "language detection", "limit": self.count})
            papers = observations[0]["data"]["candidates"]
            reads = [o["data"] for o in observations if "document_id" in o.get("data", {})]
            if len(reads) < len(papers):
                return self.response("read_paper", {"target_ref": papers[len(reads)]["paper_id"]})
            return self.response("finish_research", {"status": "completed", "findings": ["基于原文比较语言利用。"],
                "paper_ids": [p["paper_id"] for p in papers], "evidence_ids": [p["evidence_id"] for r in reads for p in r["pages"]], "gaps": []})
        plans = [o["data"]["research_plan"] for o in observations if "research_plan" in o.get("data", {})]
        answered = any(m["role"] == "user" and m.get("content", "").startswith("澄清回答：") for m in messages)
        if not plans or (answered and not plans[-1].get("clarification_id")):
            confirmed = answered or self.note
            return self.response("set_research_plan", {"goal": "原文语言利用，2025年，不改变来源与保存权限",
                "task_kind": "compare", "requested_count": self.count if confirmed else None,
                "count_confirmed": confirmed, "suggested_count": 5,
                "deep_reading_requested": self.note, "save_requested": self.note,
                "sources": ["cvpr"], "date_from": "2025-01-01", "date_to": "2025-12-31",
                "goals": [{"goal_id": "language", "description": "确认范围内的语言利用比较" if confirmed else "待确认的语言比较",
                           "min_papers": self.count if confirmed else 5, "requires_fulltext": True}]})
        if not answered and not self.note:
            return self.response("request_clarification", {"question": "希望比较几篇？建议5篇。", "fields": ["count"], "suggested_values": ["5"]})
        if "delegate_research" not in done:
            return self.response("delegate_research", {"subtasks": [{"objective": "根据原文比较", "goal_ids": ["language"],
                "mode": "deep_read" if self.note else "investigate", "sources": ["cvpr"],
                "date_from": "2025-01-01", "date_to": "2025-12-31"}]})
        children = [s for o in observations for s in o.get("data", {}).get("subtasks", [])]
        if self.note and "create_note" not in done:
            return self.response("create_note", {"title": "精读笔记", "markdown": "# 精读笔记\n原文方法利用语言嵌入。\n（完）",
                "evidence_ids": [e for s in children for e in s["evidence_ids"]]})
        draft = next((o["data"] for o in observations if o.get("message", "").startswith("draft persisted")), None)
        if self.note and "save_note" not in done:
            return self.response("save_note", {"draft_id": draft["draft_id"], "content_hash": draft["content_hash"]})
        return self.response("finish_task", {"answer_markdown": "按用户确认数量完成原文比较。", "draft_id": draft["draft_id"] if draft else None,
            "goals": [{"goal_id": "language", "status": "satisfied", "subtask_ids": [c["subtask_id"] for c in children], "reason": "实际原文覆盖用户数量"}]})


class ClarifiedPlanTests(unittest.TestCase):
    """新旧范围、身份、一次消费、恢复及DPR节点准入的组合回归。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        url = os.getenv("ADAPTIVE_TEST_DATABASE_URL") or "sqlite:///" + str(root / "test.db")
        if not url.startswith("sqlite:") and not url.rsplit("/", 1)[-1].startswith("researchloop_test_"):
            raise ValueError("isolated test database required")
        self.settings = Settings(project_root=root, data_dir=root / "data", output_dir=root / "notes", database_url=url)
        self.db = Database(url)
        self.db.ensure_schema()

    def run_case(self, count=2, note=False, replay=False):
        """经API暂停/恢复、生产图执行和读回；可回放已成功receipt验证不重发。"""
        from research_loop.main import app
        scenario = ClarifiedScenario(count, note)
        executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda s, f: FixturePDF())
        with patch("research_loop.api.dependencies.get_settings", return_value=self.settings), TestClient(app) as client:
            payload = {"prompt": "生成一篇精读笔记并保存" if note else "先问数量再比较2025年CVPR原文，不保存",
                       "save_policy": "allow" if note else "deny", "idempotency_key": str(uuid4())}
            tid = client.post("/api/tasks", json=payload).json()["task_id"]
            with patch("research_loop.providers.discovery.DiscoveryProvider.search", side_effect=scenario.search):
                executor.execute(tid)
                if not note:
                    before = client.get(f"/api/tasks/{tid}").json()
                    self.assertEqual(before["status"], "needs_input", before)
                    self.assertEqual(scenario.searches, [])
                    self.assertEqual(self.db.list_subtasks(tid), [])
                    self.assertFalse(list(self.settings.output_dir.rglob("*.md")))
                    resume = {"clarification_id": before["clarification"]["clarification_id"], "answer": f"{count}篇，仍不保存。", "idempotency_key": "resume"}
                    self.assertEqual(client.post(f"/api/tasks/{tid}/resume", json=resume).status_code, 200)
                    resumed = copy.deepcopy(self.db.get_checkpoint(tid))
                    calls = len(scenario.requests)
                    self.assertEqual(client.post(f"/api/tasks/{tid}/resume", json=resume).status_code, 200)
                    self.assertEqual(len(scenario.requests), calls)
                    self.assertEqual(self.db.get_checkpoint(tid), resumed)
                    self.assertEqual(client.post(f"/api/tasks/{tid}/resume", json={**resume, "idempotency_key": "stale"}).status_code, 409)
                    # 新回答已入库、计划仍旧时不能直接委派或结束。
                    with self.assertRaisesRegex(ValueError, "new user clarification"):
                        CoverageService(self.db).validate_delegation(tid, [{"mode": "investigate", "goal_ids": ["language"]}])
                    executor.execute(tid, "after-answer")
                    if replay:
                        count_before = len(self.db.list_actions(tid)), len(scenario.requests)
                        self.db.save_checkpoint(tid, resumed)
                        self.db.update_task(tid, status="queued", result_json=None)
                        executor.execute(tid, "receipt-replay")
                        self.assertEqual((len(self.db.list_actions(tid)), len(scenario.requests)), count_before)
                view = client.get(f"/api/tasks/{tid}").json()
            self.assertEqual(client.post("/api/tasks", json=payload).json()["task_id"], tid)
        self.assertEqual(view["status"], "completed", view)
        self.assertEqual(view["usage"]["unknown"], 0)
        plan = view["result"]["coverage"]["plan"]
        self.assertEqual(plan["requested_count"], count)
        self.assertTrue(plan["count_confirmed"])
        self.assertEqual(plan["goals"][0]["min_papers"], count)
        self.assertEqual(plan["sources"], ["cvpr"])
        self.assertEqual(plan["date_from"], "2025-01-01")
        self.assertEqual(len(view["result"]["coverage"]["paper_ids"]), count)
        dpr_calls = [r for r in scenario.requests if r.get("max_tokens") == 4096 and not r.get("tools")]
        self.assertEqual(len(dpr_calls), 1 if note else 0)
        self.assertEqual(bool(view["result"]["saved_path"]), note)
        if note:
            self.assertEqual(Path(view["result"]["saved_path"]).read_text(), view["result"]["answer_markdown"])
        else:
            self.assertEqual(plan["save_policy"], "deny")
            self.assertFalse(list(self.settings.output_dir.rglob("*.md")))
        return tid, plan

    def test_suggest_five_answer_two(self):
        self.run_case(2)

    def test_suggest_five_answer_five(self):
        self.run_case(5)

    def test_answered_plan_and_success_receipts_replay_without_resend(self):
        self.run_case(2, replay=True)

    def test_explicit_reading_note_enters_dpr_and_saves(self):
        self.run_case(1, note=True)

    def test_answer_cannot_be_reused_to_weaken_the_plan(self):
        tid, plan = self.run_case(2)
        plan["requested_count"] = 1
        plan["goals"][0]["min_papers"] = 1
        with self.assertRaisesRegex(ValueError, "weakened"):
            CoverageService(self.db).prepare_plan(tid, ResearchBrief.model_validate(plan))

    def test_plan_without_user_note_request_rejects_deep_read(self):
        tid, _ = self.run_case(2)
        with self.assertRaisesRegex(ValueError, "DPR deep_read requires"):
            CoverageService(self.db).validate_delegation(tid, [{"goal_ids": ["language"], "mode": "deep_read"}])

    def test_foreign_pending_or_fake_answers_do_not_allow_scope_change(self):
        tid, plan = self.run_case(2)
        other = TaskService(self.db).create(TaskCreate(prompt="另一个任务"))
        self.db.execute("UPDATE clarifications SET task_id=? WHERE clarification_id=?", (other.task_id, plan["clarification_id"]))
        plan["goals"][0]["min_papers"] = 1
        plan["clarification_id"] = "model-forged-answer"
        with self.assertRaisesRegex(ValueError, "weakened"):
            CoverageService(self.db).prepare_plan(tid, ResearchBrief.model_validate(plan))

    def test_count_confirmed_flag_without_answer_cannot_change_goals(self):
        tid, plan = self.run_case(2)
        self.db.execute("UPDATE clarifications SET status='pending' WHERE clarification_id=?", (plan["clarification_id"],))
        plan["goals"][0]["min_papers"] = 1
        plan["count_confirmed"] = True
        with self.assertRaisesRegex(ValueError, "weakened"):
            CoverageService(self.db).prepare_plan(tid, ResearchBrief.model_validate(plan))
