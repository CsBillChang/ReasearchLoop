"""第1条目标解释的隔离合同；继承配对保护，材料和收据全部合成。"""
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unit import test_supervisor_ab_contract as scope_tests
from harness_ab_experiment import (GOAL_KIND, GOAL_CONTRACT_VERSION, NEW_GOAL_RULE,
    OLD_GOAL_RULE, NEW_RULE_4, NEW_RULE_7, OLD_RULE_4, SUPERVISOR, SCOPE_KIND,
    build_supervisor_contract, goal_candidate, supervisor_parts, validate_scope_manifest)
from run_read_paper_ab import qualification_configuration, validate_manifest


class SupervisorGoalAbContractTests(scope_tests.SupervisorAbContractTests):
    def setUp(self):
        super().setUp()
        self.baseline = 'SYSTEM_PROMPT = ' + repr('合成职责。\n1. ' + OLD_GOAL_RULE
            + '\n4. ' + NEW_RULE_4 + '\n7. ' + NEW_RULE_7) + '\n\ndef helper():\n    return 1\n'
        for variant in ('A', 'B'):
            (Path(self.roots[variant]) / SUPERVISOR).write_text(
                self.baseline if variant == 'A' else goal_candidate(self.baseline))
        self.manifest['experiment_kind'] = GOAL_KIND
        self.manifest['supervisor_prompt_contract'] = build_supervisor_contract(
            self.roots, experiment_kind=GOAL_KIND)

    def test_goal_kind_uses_independent_contract_and_preserves_rules_4_and_7(self):
        self.assertEqual(self.manifest['supervisor_prompt_contract']['version'], GOAL_CONTRACT_VERSION)
        a, _ = supervisor_parts(self.baseline)
        b, _ = supervisor_parts(goal_candidate(self.baseline))
        for number in (4, 7):
            pattern = rf'(?m)^{number}\. .*$'
            self.assertEqual(re.search(pattern, a).group(), re.search(pattern, b).group())
        self.assertIn(NEW_GOAL_RULE, b)
        self.assertNotIn(OLD_GOAL_RULE, b)

    def test_new_kind_receipt_keeps_pair_policy_and_prompt_identity(self):
        receipt = qualification_configuration(self.manifest)
        for key in ('supervisor_prompt_contract', 'semantic_gate_root', 'pair_review_policy', 'execution_order'):
            self.assertEqual(receipt[key], self.manifest[key])
        self.assertEqual(receipt['experiment_kind'], GOAL_KIND)

    def test_goal_treatment_cannot_be_labeled_as_previous_scope_experiment(self):
        self.manifest['experiment_kind'] = SCOPE_KIND
        with self.assertRaises(PermissionError):
            validate_manifest(self.manifest, 'SP1-Q1', 'A', 'qualify')

    def test_unknown_kind_cannot_select_arbitrary_prompt_treatment(self):
        with self.assertRaises(ValueError):
            build_supervisor_contract(self.roots, experiment_kind='arbitrary_prompt')

    def test_old_rules_4_and_7_baseline_cannot_be_used_for_goal_trial(self):
        before_previous_round = self.baseline.replace(NEW_RULE_4, OLD_RULE_4)
        with self.assertRaises(ValueError):
            goal_candidate(before_previous_round)

    def test_approved_replacement_must_belong_to_numbered_rule_1(self):
        moved = self.baseline.replace('1. ', '2. ')
        with self.assertRaises(ValueError):
            goal_candidate(moved)

    def test_candidate_does_not_allow_second_application_or_duplicate_clause(self):
        for text in (goal_candidate(self.baseline), self.baseline.replace(OLD_GOAL_RULE, OLD_GOAL_RULE * 2)):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    goal_candidate(text)

    def test_rule_4_or_7_drift_in_candidate_is_rejected(self):
        path = Path(self.roots['B']) / SUPERVISOR
        original = path.read_text()
        for clause in (NEW_RULE_4, NEW_RULE_7):
            with self.subTest(clause=clause):
                path.write_text(original.replace(clause, clause + '未批准扩展。'))
                with self.assertRaises(PermissionError):
                    validate_scope_manifest(self.manifest)
                path.write_text(original)

    def test_arbitrary_goal_wording_even_with_new_digest_is_rejected(self):
        path = Path(self.roots['B']) / SUPERVISOR
        path.write_text(path.read_text().replace(NEW_GOAL_RULE, NEW_GOAL_RULE + '额外硬要求。'))
        with self.assertRaises(ValueError):
            build_supervisor_contract(self.roots, experiment_kind=GOAL_KIND)


if __name__ == '__main__':
    unittest.main()
