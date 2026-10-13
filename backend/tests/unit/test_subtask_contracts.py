# 子任务数据契约测试：验证分派、范围及结构化返回。
import unittest
from research_loop.domain.tasks import ResearchBrief, SavePolicy, SubtaskSpec, SubtaskResult


class SubtaskContractTests(unittest.TestCase):
    def test_brief_default_count_is_a_suggestion_not_confirmation(self):
        brief = ResearchBrief(goal="compare papers", save_policy=SavePolicy.deny)
        self.assertEqual(brief.suggested_count, 5)
        self.assertFalse(brief.count_confirmed)

    def test_subtask_has_explicit_scope_and_result_gap(self):
        spec = SubtaskSpec(objective="method", paper_ids=["p1"], start_page=1, page_count=2)
        result = SubtaskResult(subtask_id="s", parent_task_id="p", objective=spec.objective,
                               status="partial", gaps=["missing hardware detail"])
        self.assertEqual(result.paper_ids, [])
        self.assertEqual(result.status, "partial")
