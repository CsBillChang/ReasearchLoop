"""两文件交接候选的精确源码与配对门；不导入候选业务异常。"""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unit import test_supervisor_ab_contract as scope_tests
from harness_ab_experiment import (GOAL_KIND, HANDOFF_CONTRACT_VERSION, HANDOFF_FILES,
    HANDOFF_KIND, HANDOFF_RESEARCHER, HANDOFF_SERVICE, NEW_RULE_4, NEW_RULE_7,
    OLD_GOAL_RULE, OLD_PROGRESS_CALL, OLD_PROGRESS_IMPORT, PAIR_REVIEW_POLICY,
    PAIRED_KINDS, SCOPE_KIND, SUPERVISOR, SUPERVISOR_KINDS, UNKNOWN_ID_RAISE,
    build_handoff_contract, build_supervisor_contract, canonical, handoff_candidate,
    paired_source_contract, sha256, validate_paired_manifest, validate_scope_manifest)
from harness_ab_pair_gate import initialize_scope_session, issue_cell_grant, validate_prior_pair_gates
from run_read_paper_ab import qualification_configuration, run_cell, validate_active_variant, validate_manifest


class HandoffAbContractTests(unittest.TestCase):
    """合成源码在H基线即可验证合同，不依赖真实材料或新业务类。"""

    def setUp(self):
        support = scope_tests.SupervisorAbContractTests()
        support.setUp()
        self.addCleanup(support.doCleanups)
        self.support = support
        self.manifest, self.roots, self.root = support.manifest, support.roots, support.root
        prompt = '\n1. ' + OLD_GOAL_RULE + '\n4. ' + NEW_RULE_4 + '\n7. ' + NEW_RULE_7
        self.supervisor = 'SYSTEM_PROMPT = ' + repr(prompt) + '\n'
        self.baseline = {
            HANDOFF_SERVICE: 'class ResearchProgressService:\n    def prepare(self, task_id, args):\n        '
                + UNKNOWN_ID_RAISE + '\n',
            HANDOFF_RESEARCHER: OLD_PROGRESS_IMPORT + '\n\ndef _record_progress(args, ctx):\n'
                + OLD_PROGRESS_CALL + '\n    return value\n',
        }
        candidate = handoff_candidate(self.baseline)
        for variant in ('A', 'B'):
            source = Path(self.roots[variant])
            (source / SUPERVISOR).write_text(self.supervisor)
            texts = self.baseline if variant == 'A' else candidate
            for name, body in texts.items():
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)
            (source / 'backend/scripts/harness_ab_budget.py').write_text(
                "class Ledger:\n    def __init__(self):\n        self.config = {'first_error_policy_version': 1}\n")
        self.manifest.pop('supervisor_prompt_contract')
        self.manifest.update(experiment_kind=HANDOFF_KIND,
            handoff_source_contract=build_handoff_contract(self.roots))

    def test_handoff_kind_is_independent_of_old_supervisor_kinds(self):
        self.assertIn(HANDOFF_KIND, PAIRED_KINDS)
        self.assertNotIn(HANDOFF_KIND, SUPERVISOR_KINDS)
        self.assertEqual(SUPERVISOR_KINDS, {SCOPE_KIND, GOAL_KIND})
        self.assertEqual(self.manifest['handoff_source_contract']['version'], HANDOFF_CONTRACT_VERSION)
        self.assertEqual(set(self.manifest['handoff_source_contract']['business_files']), HANDOFF_FILES)
        with self.assertRaises(PermissionError):
            validate_scope_manifest(self.manifest)

    def test_exact_two_file_candidate_keeps_full_supervisor_and_both_dedup_on(self):
        for variant in ('A', 'B'):
            validate_manifest(self.manifest, 'SP1-Q1', variant, 'qualify')
        variants = self.manifest['handoff_source_contract']['variants']
        self.assertEqual(variants['A']['supervisor_file_sha256'], variants['B']['supervisor_file_sha256'])
        self.assertEqual(variants['A']['system_prompt_sha256'], variants['B']['system_prompt_sha256'])
        self.assertNotEqual(variants['A']['business_file_sha256'], variants['B']['business_file_sha256'])
        self.manifest['variant_settings']['A']['context_deduplicate_read_paper'] = False
        with self.assertRaises(PermissionError):
            validate_paired_manifest(self.manifest)

    def test_qualification_binds_handoff_contract_and_all_pair_controls(self):
        receipt = qualification_configuration(self.manifest)
        self.assertNotIn('supervisor_prompt_contract', receipt)
        for name in ('handoff_source_contract', 'semantic_gate_root', 'pair_review_policy', 'execution_order'):
            self.assertEqual(receipt[name], self.manifest[name])
        self.assertEqual(receipt['experiment_kind'], HANDOFF_KIND)

    def test_old_kind_cannot_relabel_handoff_business_change(self):
        for kind in SUPERVISOR_KINDS:
            manifest = copy.deepcopy(self.manifest)
            manifest['experiment_kind'] = kind
            with self.assertRaises(PermissionError):
                validate_manifest(manifest, 'SP1-Q1', 'A', 'qualify')
        with self.assertRaises(ValueError):
            build_supervisor_contract(self.roots, experiment_kind=HANDOFF_KIND)

    def test_missing_or_wrong_field_or_digest_is_rejected(self):
        original = copy.deepcopy(self.manifest)
        for mutation in ('missing', 'old_field', 'digest'):
            manifest = copy.deepcopy(original)
            if mutation == 'missing':
                manifest.pop('handoff_source_contract')
            elif mutation == 'old_field':
                manifest['supervisor_prompt_contract'] = manifest['handoff_source_contract']
            else:
                manifest['handoff_source_contract']['variants']['B']['source_tree_sha256'] = '0' * 64
            with self.subTest(mutation=mutation), self.assertRaises(PermissionError):
                validate_paired_manifest(manifest)

    def test_additional_handler_change_or_broad_error_mapping_is_rejected(self):
        path = Path(self.roots['B']) / HANDOFF_RESEARCHER
        original = path.read_text()
        for changed in (original.replace('except UnknownFindingIdError', 'except ValueError'),
                original + '\nUNAPPROVED = True\n',
                original.replace('new findings omit finding_id', 'silently remove unknown ID')):
            with self.subTest(change=changed[-100:]):
                path.write_text(changed)
                with self.assertRaises(ValueError):
                    build_handoff_contract(self.roots)
        path.write_text(original)

    def test_prompt_schema_or_other_business_drift_is_rejected(self):
        source = Path(self.roots['B'])
        for name in (SUPERVISOR, 'backend/research_loop/domain/research_progress.py',
                'backend/research_loop/agents/graph.py'):
            path = source / name
            existed = path.exists()
            old = path.read_text() if existed else None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((old or '') + '\nUNAPPROVED = True\n')
            with self.subTest(name=name), self.assertRaises(PermissionError):
                validate_paired_manifest(self.manifest)
            if existed:
                path.write_text(old)
            else:
                path.unlink()

    def test_shared_first_error_script_must_match_and_keep_policy(self):
        a = Path(self.roots['A']) / 'backend/scripts/harness_ab_budget.py'
        b = Path(self.roots['B']) / 'backend/scripts/harness_ab_budget.py'
        b.write_text(b.read_text() + '\nDRIFT = True\n')
        with self.assertRaises(ValueError):
            build_handoff_contract(self.roots)
        for version in ('0', 'True'):
            for path in (a, b):
                path.write_text("class Ledger:\n    def __init__(self):\n        self.config = {'first_error_policy_version': " + version + "}\n")
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'first-error'):
                build_handoff_contract(self.roots)

    def test_source_roots_must_be_independent_and_treatment_not_reapplied(self):
        with self.assertRaises(ValueError):
            build_handoff_contract({'A': self.roots['A'], 'B': self.roots['A']})
        with self.assertRaises(ValueError):
            handoff_candidate(handoff_candidate(self.baseline))

    def test_policy_duplicate_keys_expansions_or_multiple_assignments_are_rejected(self):
        bodies = [
            "        self.config = {'first_error_policy_version': 1, 'first_error_policy_version': 0}\n",
            "        self.config = {'first_error_policy_version': 1, **{'first_error_policy_version': 0}}\n",
            "        self.config = {'first_error_policy_version': 1}\n        self.config = {'first_error_policy_version': 0}\n",
        ]
        for body in bodies:
            for variant in ('A', 'B'):
                path = Path(self.roots[variant]) / 'backend/scripts/harness_ab_budget.py'
                path.write_text('class Ledger:\n    def __init__(self):\n' + body)
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, 'first-error'):
                build_handoff_contract(self.roots)

    def test_pair_policy_and_order_cannot_be_removed(self):
        for mutation in ('policy', 'order', 'control'):
            manifest = copy.deepcopy(self.manifest)
            if mutation == 'policy':
                manifest['pair_review_policy'] = {'required': False}
            elif mutation == 'order':
                manifest['execution_order'][1], manifest['execution_order'][2] = manifest['execution_order'][2], manifest['execution_order'][1]
            else:
                manifest['semantic_gate_root'] = manifest['runtime_root']
            with self.subTest(mutation=mutation), self.assertRaises(PermissionError):
                validate_manifest(manifest, 'SP1-Q1', 'A', 'qualify')

    def test_one_time_grant_binds_new_contract_and_prior_review_is_required(self):
        initialize_scope_session(self.manifest, 100.0, clock=lambda: 100.0)
        grant = issue_cell_grant(self.manifest, 'SP1-Q1', 'A', 200.0)
        self.assertEqual(grant['source_contract_sha256'], sha256(canonical(paired_source_contract(self.manifest))))
        current = {**self.manifest, 'cell_absolute_deadline': 200.0}
        validate_prior_pair_gates(current, 'SP1-Q1', 'A', clock=lambda: 101.0)
        with self.assertRaises(FileExistsError):
            issue_cell_grant(self.manifest, 'SP1-Q1', 'A', 200.0)
        issue_cell_grant(self.manifest, 'SP1-Q2', 'B', 200.0)
        with self.assertRaises(PermissionError):
            validate_prior_pair_gates(current, 'SP1-Q2', 'B', clock=lambda: 101.0)

    def test_execute_entry_cannot_bypass_parent_grant(self):
        with patch('run_read_paper_ab.select_variant_source', return_value=Path(self.roots['A'])), \
                patch('run_read_paper_ab.validate_manifest'), \
                patch('harness_ab_pair_gate.validate_prior_pair_gates', side_effect=PermissionError('synthetic grant denied')) as gate:
            with self.assertRaisesRegex(PermissionError, 'synthetic grant denied'):
                run_cell(self.manifest, 'SP1-Q2', 'A', 'execute')
            gate.assert_called_once_with(self.manifest, 'SP1-Q2', 'A')
        self.assertFalse((self.support.runtime / 'cells/SP1-Q2-A').exists())

    def test_per_send_contract_revalidation_rejects_drift(self):
        path = Path(self.roots['B']) / HANDOFF_SERVICE
        path.write_text(path.read_text() + '\nDRIFT = True\n')
        with self.assertRaises(PermissionError):
            validate_active_variant(self.manifest, 'B')

    def test_existing_semantic_gate_behavior_is_retained(self):
        challenge = self.support.challenge()
        result = self.support.evaluate(challenge, self.support.review(challenge))
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['quality_equivalent'])


if __name__ == '__main__':
    unittest.main()
