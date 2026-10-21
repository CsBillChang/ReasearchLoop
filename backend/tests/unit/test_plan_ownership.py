"""计划归属与撤回的零模型服务合同；数据库及可选产物查询均为替身。"""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from research_loop.domain.tasks import FinishTaskInput, ResearchBrief, ResearchGoal
from research_loop.services.coverage import CoverageService


class PlanOwnershipTests(unittest.TestCase):
    """区分用户交付门和可调整策略，不从自然语言关键词推断归属。"""

    def setUp(self):
        """只在内存建立用户问题和可重放计划，不访问真实数据库。"""
        self.prompt = "推荐近三年两篇相关论文，给出链接和理由，之后我再选一篇细看。"
        self.db = Mock()
        self.db.get_task.return_value = {"prompt": self.prompt, "save_policy": "deny", "attachments_json": "[]"}
        self.db.list_actions.return_value = []
        self.db.list_subtasks.return_value = []
        self.db.list_events.return_value = []
        self.db.get_draft.return_value = None
        self.service = CoverageService(self.db)
        self.service.plan = Mock(return_value=None)
        self.service._latest_answer = Mock(return_value=None)
        self.user = {"goal_id": "recommend", "description": "两篇链接和理由", "origin": "user",
                     "source_quote": "两篇相关论文，给出链接和理由", "min_papers": 2}
        self.strategy = {"goal_id": "extra_read", "description": "可选全文核对", "origin": "strategy",
                         "min_papers": 4, "requires_fulltext": True}

    def brief(self, goals=None, **kwargs):
        """用同一合成用户合同构造计划输入。"""
        return ResearchBrief(goal="本轮推荐", requested_count=2, count_confirmed=True,
                             goals=goals if goals is not None else [self.user, self.strategy], **kwargs)

    def install(self):
        """把首次成功计划作为只读旧版，后续修改不能回写原对象。"""
        plan = self.service.prepare_plan("root", self.brief())
        self.service.plan.return_value = plan
        return plan

    def finish(self, goals):
        """隔离可选Reflection与笔记投影，运行真实覆盖及数量判断。"""
        with patch("research_loop.agents.reflection.ReflectionService.research_context", return_value=[]), \
             patch("research_loop.services.notes.NoteService.artifacts", return_value=[]):
            return self.service.finish("root", FinishTaskInput(answer_markdown="两篇链接及推荐理由。", goals=goals))

    def reads(self, count=2):
        """替换共同来源校验的输出，专测计划归属而不伪造原文授权。

        三种原文入口的真实页/窗口校验由research_closure_pipeline覆盖。
        """
        self.service.source_coverage = Mock(return_value=[{'paper_id': f'paper-{i}',
            'document_id': f'doc-{i}', 'evidence_id': f'evidence-{i}',
            'material_scope': 'returned_page', 'full_document_returned': False}
            for i in range(count)])
        self.db.get_paper.return_value = {"metadata": {"source": "cvpr", "published": "2025-06-01"}}

    def test_legacy_default_cannot_be_reclassified_without_resume(self):
        """旧无字段目标默认受保护，不自动猜测其实际是否来自用户。"""
        self.assertEqual(ResearchGoal(goal_id="old", description="旧全文要求").origin, "legacy")
        old = self.service.prepare_plan("root", self.brief([{k: v for k, v in self.user.items()
            if k not in {"origin", "source_quote"}}]))
        self.service.plan.return_value = old
        changed = deepcopy(old)
        changed["goals"][0]["origin"] = "strategy"
        changed["goals"].append({**self.user, "goal_id": "new_user"})
        with self.assertRaisesRegex(ValueError, "existing goals"):
            self.service.prepare_plan("root", ResearchBrief.model_validate(changed))

    def test_persisted_old_action_is_normalized_without_rewriting(self):
        """真实旧观察缺少新字段时可重放，保留原件并继续保护旧目标。"""
        legacy = {k: v for k, v in self.user.items() if k not in {"origin", "source_quote"}}
        raw = self.brief([legacy]).model_dump(mode="json")
        for key in ("origin", "status", "source_quote", "change_reason"):
            raw["goals"][0].pop(key)
        original = deepcopy(raw)
        self.db.list_actions.return_value = [{"tool_name": "set_research_plan", "status": "succeeded",
            "observation": {"data": {"research_plan": raw}}}]
        self.service.plan = CoverageService.plan.__get__(self.service)
        normalized = self.service.prepare_plan("root", ResearchBrief.model_validate(raw))
        self.assertEqual(normalized["goals"][0]["origin"], "legacy")
        self.assertEqual(raw, original)
        changed = deepcopy(raw)
        changed["goals"][0]["min_papers"] = 1
        with self.assertRaisesRegex(ValueError, "existing goals"):
            self.service.prepare_plan("root", ResearchBrief.model_validate(changed))
        self.db.list_task_evidence.return_value = []
        with patch("research_loop.services.research_progress.ResearchProgressService.retained", return_value={}), \
             patch("research_loop.agents.reflection.ReflectionService.research_context", return_value=[]):
            # 恢复状态里的任意新归属不能覆盖已持久化的旧计划。
            context = self.service.current_context("root", {"research_plan": {"goals": [self.strategy]}})
        self.assertEqual(context["research_plan"], original)

    def test_user_origin_needs_actual_source(self):
        """来源逐字存在只证明出处；不把它当模型语义判断正确的证据。"""
        with self.assertRaises(ValidationError):
            ResearchGoal(goal_id="u", description="用户目标", origin="user")
        with self.assertRaisesRegex(ValueError, "verbatim"):
            self.service.prepare_plan("root", self.brief([{**self.user, "source_quote": "用户没有说的全文要求"}]))
        self.assertEqual(self.service.prepare_plan("root", self.brief())["goals"][0]["origin"], "user")

    def test_strategy_only_plan_is_rejected(self):
        """所有目标都标内部步骤不能绕开用户交付门。"""
        with self.assertRaisesRegex(ValueError, "active user or legacy"):
            self.service.prepare_plan("root", self.brief([self.strategy]))

    def test_optional_strategy_cannot_be_the_only_material_goal(self):
        """可选步骤不能单独承接材料门，防止用户零篇目标无资料完成。"""
        for origin in ("user", "legacy"):
            with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, "user/legacy.*min_papers>0"):
                self.service.prepare_plan("root", ResearchBrief(goal="交付简短总结", requested_count=None,
                    goals=[self.user | {"origin": origin, "min_papers": 0}, self.strategy]))

    def test_zero_extra_material_synthesis_keeps_required_material_goal(self):
        """允许汇总不额外取材，但同一计划须保留用户材料覆盖目标。"""
        synthesis = self.user | {"goal_id": "synthesis", "description": "据已有材料汇总", "min_papers": 0}
        for origin in ("user", "legacy"):
            with self.subTest(origin=origin):
                plan = self.service.prepare_plan("root", ResearchBrief(goal="交付简短总结", requested_count=None,
                    goals=[self.user | {"origin": origin}, synthesis, self.strategy]))
                self.assertEqual(plan["goals"][1]["min_papers"], 0)
                self.assertEqual(plan["goals"][0]["min_papers"], 2)

    def test_strategy_withdrawal_keeps_history_and_rejects_delegation(self):
        """撤回步骤带理由保留原ID，不能再委派，旧版对象不变。"""
        old = self.install()
        withdrawn = {**self.strategy, "status": "withdrawn", "change_reason": "题录已足够支撑当前推荐"}
        value = self.service.prepare_plan("root", self.brief([self.user, withdrawn]))
        self.assertEqual(old["goals"][1]["status"], "active")
        self.assertEqual(value["goals"][1]["status"], "withdrawn")
        self.service.plan.return_value = value
        with self.assertRaisesRegex(ValueError, "valid goal_ids"):
            self.service.validate_delegation("root", [{"goal_ids": ["extra_read"], "mode": "investigate"}])
        self.service.validate_delegation("root", [{"goal_ids": ["recommend"], "mode": "screen"}])

    def test_strategy_changes_need_reason_and_cannot_delete_id(self):
        """实质修改需原因；原样重放与省略可选默认字段不会误触变更门。"""
        self.install()
        self.service.prepare_plan("root", self.brief())
        for value in (self.strategy | {"min_papers": 1}, self.strategy | {"min_papers": 1, "change_reason": " "}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "change_reason"):
                self.service.prepare_plan("root", self.brief([self.user, value]))
        changed = self.strategy | {"min_papers": 1, "change_reason": "只补一个具体机制缺口"}
        self.assertEqual(self.service.prepare_plan("root", self.brief([self.user, changed]))["goals"][1]["min_papers"], 1)
        with self.assertRaisesRegex(ValueError, "cannot be removed"):
            self.service.prepare_plan("root", self.brief([self.user]))

    def test_user_goal_and_root_constraints_stay_protected(self):
        """自拟步骤可撤回不会赋予修改用户数量、范围或目标的权限。"""
        old = self.install()
        for changes in ({"min_papers": 1}, {"requires_fulltext": True}, {"origin": "strategy"}):
            goals = [self.user | changes, self.strategy]
            if changes.get("origin") == "strategy":
                goals.append(self.user | {"goal_id": "other"})
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "existing goals"):
                self.service.prepare_plan("root", self.brief(goals))
        for changes in ({"requested_count": 1}, {"save_requested": True}, {"deep_reading_requested": True}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "task constraint"):
                self.service.prepare_plan("root", ResearchBrief.model_validate(old | changes))

    def test_new_root_preferences_adjust_while_legacy_root_constraints_stay(self):
        """新顶层检索偏好可改；无内部标记的历史合同仍受硬范围保护。"""
        plan = self.install()
        for changes in ({"date_from": "2026-01-01"}, {"sources": ["iclr"]}):
            with self.subTest(changes=changes):
                updated = self.service.prepare_plan("root", ResearchBrief.model_validate(plan | changes))
                self.assertEqual(updated["goals"], plan["goals"])
                self.assertEqual(updated["_scope_semantics"], "goal_owned_constraints")
        legacy = deepcopy(plan)
        legacy.pop("_scope_semantics")
        self.service.plan.return_value = legacy
        for changes in ({"date_from": "2026-01-01"}, {"sources": ["iclr"]}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "task constraint"):
                self.service.prepare_plan("root", ResearchBrief.model_validate(legacy | changes))

    def test_real_resume_can_redirect_user_goal_once(self):
        """只使用服务读到的真实恢复记录，不接受模型自填澄清身份。"""
        self.install()
        answer = {"clarification_id": "actual-resume", "answer": "改成一篇，要看全文机制。"}
        self.service._latest_answer.return_value = answer
        value = self.service.prepare_plan("root", ResearchBrief(goal="改为定向全文", requested_count=1,
            goals=[self.user | {"min_papers": 1, "requires_fulltext": True, "source_quote": answer["answer"]},
                   self.strategy]))
        self.assertEqual(value["clarification_id"], "actual-resume")
        self.service.plan.return_value = value
        changed = deepcopy(value)
        changed["goals"][0]["min_papers"] = 2
        with self.assertRaisesRegex(ValueError, "existing goals"):
            self.service.prepare_plan("root", ResearchBrief.model_validate(changed))

    def test_real_resume_still_requires_strategy_change_reason(self):
        """真实澄清允许重定用户目标，但不能抹去自拟步骤变更的依据。"""
        self.install()
        self.service._latest_answer.return_value = {"clarification_id": "actual-resume", "answer": "保持推荐要求。"}
        with self.assertRaisesRegex(ValueError, "change_reason"):
            self.service.prepare_plan("root", self.brief([self.user, self.strategy | {"min_papers": 1}]))

    def test_current_context_preserves_ownership_and_withdrawal(self):
        """压缩前的权威上下文保留完整归属、原句、撤回状态和原因。"""
        self.install()
        value = self.service.prepare_plan("root", self.brief([self.user,
            self.strategy | {"status": "withdrawn", "change_reason": "当前推荐已足够"}]))
        self.service.plan.return_value = value
        self.db.list_task_evidence.return_value = []
        with patch("research_loop.services.research_progress.ResearchProgressService.retained", return_value={}), \
             patch("research_loop.agents.reflection.ReflectionService.research_context", return_value=[]):
            context = self.service.current_context("root", {})
        self.assertEqual(context["research_plan"]["goals"], value["goals"])
        self.assertEqual(context["research_plan"]["goals"][0]["source_quote"], self.user["source_quote"])
        self.assertEqual(context["research_plan"]["goals"][1]["status"], "withdrawn")
        from research_loop.agents.memory import RULES
        self.assertIn("撤回不算用户未完成，不沿旧待办重新委派", RULES)

    def test_unresolved_strategy_is_not_unmet_user_requirement(self):
        """用户交付已满足时，可选全文步骤缺证保留记录而不制造partial。"""
        self.install()
        self.reads()
        result = self.finish([{"goal_id": "recommend", "status": "satisfied", "reason": "两个已授权材料支持推荐"},
                              {"goal_id": "extra_read", "status": "unresolved", "reason": "没有必要全面核对"}])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["gaps"], [])
        self.assertEqual(result["coverage"]["goals"][1]["origin"], "strategy")

    def test_withdrawn_strategy_needs_no_finish_entry(self):
        """结束仅枚举有效目标，撤回记录仍保存在交付的计划中。"""
        self.install()
        value = self.service.prepare_plan("root", self.brief([self.user,
            self.strategy | {"status": "withdrawn", "change_reason": "用户本轮先筛选"}]))
        self.service.plan.return_value = value
        self.reads()
        result = self.finish([{"goal_id": "recommend", "status": "satisfied", "reason": "资料支持"}])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["coverage"]["plan"]["goals"][1]["status"], "withdrawn")

    def test_strategy_material_does_not_replace_user_count(self):
        """自拟步骤的异范围资料不能凑齐用户交付论文数。"""
        user = self.user | {"sources": ["iclr"]}
        self.service.plan.return_value = self.service.prepare_plan("root", self.brief([user, self.strategy]))
        self.reads()
        result = self.finish([{"goal_id": "recommend", "status": "unresolved", "reason": "缺合格候选"},
                              {"goal_id": "extra_read", "status": "unresolved", "reason": "只有其他来源资料"}])
        self.assertEqual(result["coverage"]["paper_ids"], [])
        self.assertEqual(len(result["gaps"]), 2)
        self.assertEqual(result["status"], "partial")


if __name__ == "__main__":
    unittest.main()
