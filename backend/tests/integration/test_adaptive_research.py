"""三个可观察场景的API→Worker回归；模型/论文边界为替身，不冒充真实语义验收。"""
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
from research_loop.domain.evidence import PaperCandidate, SearchResult, SourceResult
from research_loop.domain.tasks import FinishTaskInput, ResearchBrief, TaskCreate
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.services.coverage import CoverageService
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_collaboration_contract import FixturePDF


class AdaptiveScenario:
    """根据实际工具观察返回可复现决策，测试动态路径及恢复而非模型智力。"""

    def __init__(self, kind="S"):
        self.kind = kind
        self.prefix = uuid4().hex
        self.requests = []
        self.searches = []

    def search(self, query, date_from, date_to, sources, limit):
        """返回独立候选；模拟缺口只用于隔离测试，不进入生产工具。"""
        self.searches.append(query)
        count = 3 if self.kind == "S" else 1
        number = len(self.searches)
        return SearchResult(candidates=[PaperCandidate(
            paper_id=f"{self.prefix}-{number}-{n}", title=f"Language method {number}/{n}",
            source="cvpr", published="2025-06-01", abstract="Uses language embeddings.",
            pdf_url="https://fixture.invalid/p.pdf", paper_url="https://fixture.invalid/p")
            for n in range(count)], source_results=[SourceResult(source="cvpr", status="queried_success")])

    @staticmethod
    def response(name=None, args=None, content=""):
        """组装真实供应商形状的响应，走生产解析和持久化链。"""
        message = {"role": "assistant", "content": content}
        if name:
            message["tool_calls"] = [{"id": str(uuid4()), "type": "function", "function": {
                "name": name, "arguments": json.dumps(args)}}]
        return httpx.Response(200, json={"id": str(uuid4()), "choices": [{
            "finish_reason": "tool_calls" if name else "stop", "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    def handler(self, request):
        """从历史工具结果选择后续行动：缺口先补充，再汇总交付。"""
        body = json.loads(request.content)
        self.requests.append(body)
        messages = body["messages"]
        validate_messages(messages)
        names = {t["function"]["name"] for t in body.get("tools", [])}
        observations = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        done = [c["function"]["name"] for m in messages for c in m.get("tool_calls", [])]
        if "finish_research" in names:
            brief = json.loads(messages[2]["content"].split("\n", 1)[1])
            if not observations:
                return self.response("search_papers", {"query": brief["objective"], "limit": 3})
            candidates = observations[0]["data"]["candidates"]
            if brief["mode"] == "investigate" and "read_paper" not in done:
                return self.response("read_paper", {"target_ref": candidates[0]["paper_id"]})
            pages = observations[-1]["data"].get("pages", [])
            incomplete = self.kind == "P" or (self.kind == "C" and "initial" in brief["objective"])
            return self.response("finish_research", {"status": "partial" if incomplete else "completed",
                "findings": ["论文利用语言嵌入。"], "paper_ids": [p["paper_id"] for p in candidates],
                "evidence_ids": [p["evidence_id"] for p in pages],
                "gaps": ["另一研究维度尚无材料"] if incomplete else []})
        if "set_research_plan" not in done:
            goals = [{"goal_id": "language", "description": "语言利用方式", "min_papers": 3 if self.kind == "S" else 1,
                      "requires_fulltext": self.kind != "S", "sources": ["cvpr"],
                      "date_from": "2025-01-01", "date_to": "2025-12-31"}]
            if self.kind == "C":
                goals.append({**goals[0], "goal_id": "robustness", "description": "鲁棒性维度"})
            return self.response("set_research_plan", {"goal": "语言方法研究", "task_kind": "search" if self.kind == "S" else "survey",
                "save_requested": self.kind == "C", "requested_count": 3 if self.kind == "S" else 2,
                "goals": goals})
        batches = done.count("delegate_research")
        if batches == 0 or (self.kind == "C" and batches == 1):
            return self.response("delegate_research", {"subtasks": [{"objective": "initial language" if batches == 0 else "supplement missing dimension",
                "mode": "screen" if self.kind == "S" else "investigate", "goal_ids": ["language" if batches == 0 else "robustness"],
                "sources": ["cvpr"], "date_from": "2025-01-01", "date_to": "2025-12-31"}]})
        children = [s for o in observations for s in o.get("data", {}).get("subtasks", [])]
        if self.kind == "C":
            if "create_note" not in done:
                return self.response("create_note", {"title": "定向比较", "markdown": "# 比较\n两篇实际读取的论文采用语言嵌入；这是抽样观察，不代表全领域趋势。",
                    "evidence_ids": [e for s in children for e in s["evidence_ids"]]})
            draft = next(o["data"] for o in observations if o.get("message", "").startswith("draft persisted"))
            if "save_note" not in done:
                return self.response("save_note", {"draft_id": draft["draft_id"], "content_hash": draft["content_hash"]})
        else:
            draft = None
        return self.response("finish_task", {"answer_markdown": "候选列表基于标题摘要。" if self.kind == "S" else "已读论文使用语言嵌入；缺少另一维度证据，停止检索。",
            "draft_id": draft["draft_id"] if draft else None,
            "goals": [{"goal_id": goal, "status": "unresolved" if self.kind == "P" else "satisfied",
                       "subtask_ids": [s["subtask_id"] for s in children if goal in s["goal_ids"]],
                       "reason": "基于实际观察，缺口已披露或已补足"}
                      for goal in (["language", "robustness"] if self.kind == "C" else ["language"])]})

    def factory(self, settings):
        """所有角色仍用生产DeepSeekProvider，只替换HTTP传输。"""
        return DeepSeekProvider("fixture", client=httpx.Client(transport=httpx.MockTransport(self.handler)))


class AdaptiveResearchTests(unittest.TestCase):
    """既验证正常结果，也验证错范围、漏目标、恢复及已有成果保护。"""

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

    def run_scenario(self, kind):
        """通过实际API建任务，执行独立角色图，随后从API读回结果。"""
        from research_loop.main import app
        scenario = AdaptiveScenario(kind)
        executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda s, f: FixturePDF())
        with patch("research_loop.api.dependencies.get_settings", return_value=self.settings), TestClient(app) as client:
            response = client.post("/api/tasks", json={"prompt": "找三篇论文" if kind == "S" else "比较不同研究维度",
                "save_policy": "allow" if kind == "C" else "deny"})
            self.assertEqual(response.status_code, 202)
            tid = response.json()["task_id"]
            with patch("research_loop.providers.discovery.DiscoveryProvider.search", side_effect=scenario.search):
                executor.execute(tid)
            view = client.get(f"/api/tasks/{tid}").json()
        self.assertEqual(view["usage"]["unknown"], 0, view)
        self.assertEqual(view["usage"]["requests"], len(scenario.requests))
        return tid, view, scenario, executor

    def test_simple_screen_without_deep_reading_or_files(self):
        tid, view, scenario, _ = self.run_scenario("S")
        self.assertEqual(view["status"], "completed", view)
        self.assertEqual(len(view["result"]["coverage"]["paper_ids"]), 3)
        self.assertIsNone(view["result"]["draft_id"])
        self.assertEqual(len(view["subtasks"]), 1)
        child = view["subtasks"][0]
        self.assertEqual(child["mode"], "screen")
        self.assertFalse(child["deep_reading_status"])
        self.assertEqual([a["tool_name"] for a in self.db.list_actions(child["subtask_id"])], ["search_papers", "finish_research"])
        self.assertFalse(list(self.settings.output_dir.rglob("*.md")))

    def test_complex_supplement_resolves_historical_partial_and_saves(self):
        tid, view, scenario, executor = self.run_scenario("C")
        self.assertEqual(view["status"], "completed", view)
        self.assertEqual([s["status"] for s in view["subtasks"]], ["partial", "completed"])
        self.assertEqual(len(view["result"]["coverage"]["paper_ids"]), 2)
        path = Path(view["result"]["saved_path"])
        self.assertEqual(path.read_text(), view["result"]["answer_markdown"])
        self.assertEqual(view["result"]["gaps"], [])
        self.assertFalse(any(r["messages"][0]["content"].startswith("你是一名资深学术论文分析助手") for r in scenario.requests))
        calls = len(scenario.requests)
        before = path.stat().st_mtime_ns
        self.db.update_task(tid, status="queued", result_json=None)
        executor.execute(tid, "restarted")
        self.assertEqual(len(scenario.requests), calls)
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertEqual(self.db.get_task(tid)["status"], "completed")

    def test_partial_end_keeps_findings_without_files(self):
        tid, view, scenario, _ = self.run_scenario("P")
        self.assertEqual(view["status"], "partial", view)
        self.assertIn("语言嵌入", view["result"]["answer_markdown"])
        self.assertTrue(view["result"]["gaps"])
        self.assertEqual(len(view["subtasks"]), 1)
        self.assertIsNone(view["result"]["saved_path"])
        self.assertLess(len(scenario.requests), 10)

    def test_synthesis_goal_does_not_require_an_extra_paper(self):
        """有材料核对目标时允许零额外材料汇总；不能把所有取证要求改成零。"""
        task=TaskService(self.db, self.settings).create(TaskCreate(prompt='核对已有材料并局部改写',save_policy='deny'))
        service=CoverageService(self.db)
        goals=[{'goal_id':'verify','description':'核对已有论文','min_papers':1,'requires_fulltext':True},
               {'goal_id':'deliver','description':'汇总已验证内容','min_papers':0}]
        value=service.prepare_plan(task.task_id,ResearchBrief(goal='局部修订',task_kind='compare',goals=goals))
        self.assertEqual(value['goals'][1]['min_papers'],0)
        with self.assertRaisesRegex(ValueError,'材料目标'):
            service.prepare_plan(task.task_id,ResearchBrief(goal='绕过材料',task_kind='compare',goals=goals[1:]))
        with self.assertRaisesRegex(ValueError,'全文核对'):
            service.prepare_plan(task.task_id,ResearchBrief(goal='矛盾范围',task_kind='compare',goals=[goals[0],{**goals[1],'requires_fulltext':True}]))

    def test_coverage_rejects_foreign_missing_and_undercovered_goals(self):
        tid, view, _, _ = self.run_scenario("S")
        coverage = CoverageService(self.db)
        child_id = view["subtasks"][0]["subtask_id"]
        valid = {"answer_markdown": "已有观察", "goals": [{"goal_id": "language", "status": "satisfied", "subtask_ids": [child_id], "reason": "候选符合范围"}]}
        for goal_id, refs in [("unknown", [child_id]), ("language", ["foreign"]), ("language", [])]:
            with self.subTest(goal_id=goal_id, refs=refs), self.assertRaises(ValueError):
                coverage.finish(tid, FinishTaskInput.model_validate({**valid, "goals": [{**valid["goals"][0], "goal_id": goal_id, "subtask_ids": refs}]}))
        with self.assertRaisesRegex(ValueError, "expected goal_ids=\\['language'\\], received=\\['g1'\\]"):
            coverage.finish(tid, FinishTaskInput.model_validate({**valid, 'goals':[
                {**valid['goals'][0], 'goal_id':'g1'}]}))
        plan = coverage.plan(tid)
        plan["goals"][0]["min_papers"] = 1
        with self.assertRaisesRegex(ValueError, "weakened"):
            coverage.prepare_plan(tid, ResearchBrief.model_validate(plan))
        paper_id = view["result"]["coverage"]["paper_ids"][0]
        metadata = self.db.get_paper(paper_id)["metadata"]
        metadata["source"] = "other"
        self.db.save_paper(paper_id, metadata)
        with self.assertRaisesRegex(ValueError, "lacks qualified"):
            coverage.finish(tid, FinishTaskInput.model_validate(valid))

    def test_budget_stop_retains_successful_child_findings(self):
        tid, view, _, executor = self.run_scenario("P")
        result = executor._deliverable_result(tid, {"coverage_required": True}, "shared budget exhausted")
        self.assertIn("语言嵌入", result.answer_markdown)
        self.assertIn("shared budget exhausted", result.gaps)
        self.assertIsNone(result.saved_path)

    def test_premature_short_final_response_keeps_findings(self):
        """供应商stop且只有进度短句，不能吞掉已经落库的子任务结果。"""
        tid, view, _, executor = self.run_scenario("P")
        result = executor._deliverable_result(tid, {"coverage_required": True,
            "final_answer": "部分完成。", "final_reason": "model finished"})
        self.assertIn("语言嵌入", result.answer_markdown)
        self.assertIn("部分完成。", result.answer_markdown)
        self.assertTrue(result.gaps)
        self.assertEqual(executor._delivery_status(tid, {"coverage_required": True}, result).value, "partial")

    def test_explicit_final_answer_not_replaced_by_earlier_draft(self):
        """最终选择只回答时，不被任务较早保存的草稿替换或冒称新答案已保存。"""
        tid, view, _, executor = self.run_scenario("C")
        delivery = {"answer_markdown": "最新的定向答复", "draft_id": None, "gaps": [],
                    "coverage": view["result"]["coverage"], "status": "completed"}
        result = executor._deliverable_result(tid, {"delivery": delivery,
            "draft_id": view["result"]["draft_id"], "final_answer": "旧文本"})
        self.assertEqual(result.answer_markdown, "最新的定向答复")
        self.assertIsNone(result.saved_path)
        self.assertIsNone(result.draft_id)
        self.assertEqual(result.save_status, "not_saved")

    def test_no_evidence_final_text_cannot_claim_complete(self):
        scenario = AdaptiveScenario()
        scenario.handler = lambda r: scenario.response(content="凭空声称已经找到三篇")
        task = TaskService(self.db, self.settings).create(TaskCreate(prompt="找三篇论文"))
        result = TaskExecutor(self.settings, self.db, scenario.factory).execute(task.task_id)
        self.assertEqual(self.db.get_task(task.task_id)["status"], "partial")
        self.assertTrue(result.gaps)
        self.assertFalse(result.saved_path)

    def test_unknown_after_child_result_retains_work_and_does_not_resend(self):
        """模型汇总失联时，子Agent已提交的发现仍能读回，恢复不得补发。"""
        scenario = AdaptiveScenario("P")
        handler = scenario.handler
        attempts = []

        def fail_after_delegation(request):
            body = json.loads(request.content)
            names = [c["function"]["name"] for m in body["messages"] for c in m.get("tool_calls", [])]
            if "delegate_research" in names:
                attempts.append(1)
                raise httpx.ReadTimeout("controlled uncertain remote response", request=request)
            return handler(request)

        scenario.handler = fail_after_delegation
        task = TaskService(self.db, self.settings).create(TaskCreate(prompt="研究不同方法，不保存"))
        executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda s, f: FixturePDF())
        with patch("research_loop.providers.discovery.DiscoveryProvider.search", side_effect=scenario.search):
            result = executor.execute(task.task_id)
        self.assertIn("语言嵌入", result.answer_markdown)
        self.assertEqual(self.db.get_task(task.task_id)["status"], "partial")
        self.assertEqual(TaskService(self.db, self.settings).view(self.db.get_task(task.task_id)).usage["unknown"], 1)
        replay = executor.execute(task.task_id)
        self.assertIn("语言嵌入", replay.answer_markdown)
        self.assertEqual(len(attempts), 1)

    def test_date_scope_unknown_and_partial_precision_are_honest(self):
        goal = {"date_from": "2025-04-01", "date_to": "2025-12-31"}
        for published, expected in [(None, False), ("2025", False), ("2025-03", False), ("2025-04", True), ("2025-06-01", True)]:
            with self.subTest(published=published):
                self.assertEqual(CoverageService.in_scope({"published": published}, goal, {}), expected)

    def test_attachment_scope_normalizes_legacy_plan_without_weakening(self):
        """会议与附件不能混为一谈；旧计划重放仍拒绝越权附件和范围放宽。"""
        paper = self.settings.project_root / "attachment.pdf"
        paper.write_bytes(b"%PDF-fixture")
        task = TaskService(self.db, self.settings).create(TaskCreate(prompt="分析附件", attachments=[
            {"input_ref": "primary", "paper_id": "same-paper", "local_path": str(paper)},
            {"input_ref": "backup", "paper_id": "same-paper", "local_path": str(paper)}]))
        coverage = CoverageService(self.db)
        brief = ResearchBrief(goal="分析附件", sources=["cvpr", "input_ref:primary"],
            goals=[{"goal_id": "method", "description": "方法", "sources": ["input_ref:primary"]}])
        legacy = brief.model_dump(mode="json")
        # 保存历史供应商曾真实使用的字段形状，再走当前同一服务读取。
        self.db.save_action({"action_id": str(uuid4()), "task_id": task.task_id, "sequence": 1,
            "tool_name": "set_research_plan", "arguments": legacy, "status": "succeeded",
            "observation": {"data": {"research_plan": legacy}}})
        normalized = coverage.prepare_plan(task.task_id, brief)
        self.assertEqual(normalized["sources"], ["cvpr"])
        self.assertEqual(normalized["input_refs"], ["primary"])
        self.assertEqual(normalized["goals"][0]["input_refs"], ["primary"])
        inputs = {"primary": "same-paper", "backup": "same-paper"}
        self.assertTrue(coverage.in_scope({"source": "cvpr"}, legacy["goals"][0], legacy, inputs, "same-paper"))
        self.assertFalse(coverage.in_scope({"source": "cvpr"}, legacy["goals"][0], legacy, inputs, "different-paper"))
        self.assertFalse(coverage.in_scope({"source": "iclr"}, legacy["goals"][0], legacy, inputs, "same-paper"))
        for refs, expected_error in [(["unauthorized"], "authorized"), ([], "weakened")]:
            modified = json.loads(json.dumps(normalized))
            modified["goals"][0]["input_refs"] = refs
            with self.assertRaisesRegex(ValueError, expected_error):
                coverage.prepare_plan(task.task_id, ResearchBrief.model_validate(modified))

    def test_synthesis_reuses_per_paper_results_and_ignores_failed_attempt(self):
        """综合目标复用分篇阅读；保留失败尝试不应否定后来取得的有效材料。"""
        tid, view, _, _ = self.run_scenario("C")
        coverage = CoverageService(self.db)
        plan = coverage.plan(tid)
        plan["goals"].append({**plan["goals"][0], "goal_id": "synthesis", "description": "跨篇综合", "min_papers": 2})
        plan = coverage.prepare_plan(tid, ResearchBrief.model_validate(plan))
        actions = self.db.list_actions(tid)
        self.db.save_action({"action_id": str(uuid4()), "task_id": tid,
            "sequence": max(a["sequence_no"] for a in actions) + 1,
            "tool_name": "set_research_plan", "arguments": plan, "status": "succeeded",
            "observation": {"data": {"research_plan": plan}}})
        failed = self.db.create_subtask(tid, tid, 20, "旧来源失败", {"goal_ids": ["synthesis"]}, str(uuid4()))
        self.db.save_subtask_result(failed["subtask_id"], "failed", {"findings": [], "paper_ids": []}, "upstream unavailable")
        children = [s["subtask_id"] for s in view["subtasks"]]
        goals = [{k: g[k] for k in ("goal_id", "status", "subtask_ids", "reason")}
                 for g in view["result"]["coverage"]["goals"]]
        goals.append({"goal_id": "synthesis", "status": "satisfied",
            "subtask_ids": [failed["subtask_id"], *children], "reason": "综合两个分篇结果"})
        result = coverage.finish(tid, FinishTaskInput(draft_id=view["result"]["draft_id"], goals=goals))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["coverage"]["goals"][-1]["verified_paper_ids"]), 2)
        goals[-1]["subtask_ids"] = [failed["subtask_id"], children[0]]
        with self.assertRaisesRegex(ValueError, "lacks qualified"):
            coverage.finish(tid, FinishTaskInput(draft_id=view["result"]["draft_id"], goals=goals))
