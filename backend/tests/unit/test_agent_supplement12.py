"""显式12格补测的合同反例；合成锚替代真实财务数据，不调用模型或服务。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import agent_supplement12 as supplement
import agent_final_completion as completion
from agent_final_observation_recovery import OBSERVATION_POLICY, validate_observation_policy
from harness_ab_pair_gate import FINAL_SUITE_POLICY


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Supplement12Tests(unittest.TestCase):
    """绑定问题、来源、控制目录、业务字节与不可重新分配的财务子额度。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        self.old, self.new = root/'old', root/'new'
        self.old.mkdir(); self.new.mkdir()
        cases = [{'case_id': f'{name}-R{turn}', 'suite_kind': 'agent_final_v1',
            'prompt': 'Synthetic exact user question '+name, 'materials': [{'pdf_path': '/old/paper.pdf',
                'pdf_sha256': 'a'*64}], 'preconditions': {'catalog': [], 'save_policy': 'deny'}}
            for turn in (1, 2) for name in supplement.CASE_IDS]
        materials = [{'pdf_sha256': hashlib.sha256(str(i).encode()).hexdigest(),
            'pdf_path': '/old/pdf/'+str(i), 'parsed_path': '/old/parsed/'+str(i),
            'title': 'Synthetic '+str(i), 'parsed_cache_sha256': 'b'*64} for i in range(54)]
        self.base = {'root': str(self.old), 'seal': {}, 'cases': {'cases': cases},
            'materials': {'materials': materials},
            'carried_usage': {'tokens': 100, 'requests': 2, 'nano_yuan': 500, 'seconds': 10},
            'manifest': {'source_roots': {v: str(self.old/'snapshots'/v) for v in ('A','B')},
                'settings': {'model': 'synthetic'},
                'completion_contract': {'clock': {'policy': 'financial_only', 'round_started_at': 1,
                    'round_absolute_deadline': None, 'model_absolute_deadline': None}},
                'ledger_limits': {'root_tokens': 4000000, 'root_requests': 32, 'root_seconds': 900,
                    'max_output_tokens': 65536, 'batch_seconds': 70723}}}
        for v in ('A', 'B'):
            name = 'snapshots/'+v+'/backend/research_loop/example.py'
            old = self.old/name; old.parent.mkdir(parents=True); old.write_text('SYNTHETIC = 1\n')
            new = self.new/name; new.parent.mkdir(parents=True); new.write_bytes(old.read_bytes())
            self.base['seal'][name] = {'sha256': digest(old)}
        ledger = self.old/'runtime/ledger-final.json'; write(ledger, {'synthetic': True})
        self.scope = self.new/'audit/scope.json'
        write(self.scope, {'authorized_by_user': True, 'cancel_global_time_limit': True,
            'user_request': 'Run fixed supplemental tasks.', 'generation_limits': supplement.CAPS})
        self.anchor_patch = patch.object(supplement, '_anchor', return_value=self.base)
        self.anchor_patch.start(); self.addCleanup(self.anchor_patch.stop)
        contract = supplement.build_contract(self.old, self.scope)
        for name in ('runtime', 'semantic-gates'): (self.new/name).mkdir()
        self.manifest = {'completion_contract': contract, 'experiment_kind': 'prompt_bundle_v1',
            'experiment_root': str(self.new), 'source_roots': {v: str(self.new/'snapshots'/v) for v in ('A','B')},
            'runtime_root': str(self.new/'runtime'), 'semantic_gate_root': str(self.new/'semantic-gates'),
            'prepared_input_root': str(self.new/'inputs'), 'cases_path': str(self.new/'inputs/cases.json'),
            'model_manifest_path': str(self.new/'inputs/model-manifest.json'),
            'authorization_path': str(self.new/'authorization.json'), 'execution_order': deepcopy(supplement.ORDER),
            'qualification_cells': deepcopy(supplement.ORDER), 'settings': {'model': 'synthetic'},
            'variant_settings': {'A': {}, 'B': {}}, 'ledger_limits': supplement.expected_limits(self.base),
            'evaluation_policy': deepcopy(completion.EVALUATION_POLICY),
            'observation_policy': deepcopy(OBSERVATION_POLICY), 'final_suite_policy': deepcopy(FINAL_SUITE_POLICY),
            'carried_budget_source': {'version': supplement.VERSION, 'anchor_root': str(self.old),
                'seal_sha256': supplement.ANCHOR_SEAL, 'ledger_path': str(ledger), 'ledger_sha256': digest(ledger)}}
        self.save_inputs({'cases': deepcopy(cases)}, {'materials': deepcopy(materials)})

    def save_inputs(self, cases, materials):
        values = [('cases.json', 'final_cases_sha256', cases),
                  ('material-snapshot.json', 'material_snapshot_sha256', materials),
                  ('execution-plan.json', 'final_plan_sha256', {'execution_order': supplement.ORDER})]
        for name, key, value in values:
            path = self.new/'inputs'/name; write(path, value); self.manifest[key] = digest(path)

    def test_explicit_dispatch_returns_empty_retained_and_financial_only_context(self):
        context = completion.validate_completion(self.manifest)
        self.assertEqual(context['new_execution_order'], supplement.ORDER)
        self.assertEqual(context['retained_count'], 0)
        self.assertEqual(context['retained_rows'], [])
        self.assertEqual(context['retry_positions'], [])
        self.assertEqual(context['clock']['round_started_at'], 1)
        self.assertEqual(context['carried_usage'], self.base['carried_usage'])
        self.assertTrue(validate_observation_policy(self.manifest))

    def test_budget_adds_only_fixed_suballowance_to_conservative_carry(self):
        limits = self.manifest['ledger_limits']
        self.assertEqual(limits['batch_tokens'], 12000100)
        self.assertEqual(limits['batch_requests'], 290)
        self.assertEqual(limits['batch_yuan'], '32.0000005')
        self.assertFalse(limits['enforce_batch_seconds'])
        for key in ('tokens', 'requests', 'nano_yuan'):
            m = deepcopy(self.manifest); m['ledger_limits']['carried_usage'][key] = 0
            with self.subTest(key=key), self.assertRaises(PermissionError): supplement.validate(m)

    def test_missing_reordered_or_extra_task_rejected(self):
        for order in (supplement.ORDER[:-1], list(reversed(supplement.ORDER)), supplement.ORDER+[supplement.ORDER[0]]):
            m = deepcopy(self.manifest); m['execution_order'] = order
            with self.assertRaises(PermissionError): supplement.validate(m)

    def test_scope_hash_or_budget_change_cannot_extend_authorization(self):
        scope = json.loads(self.scope.read_text()); scope['generation_limits']['tokens'] += 1
        write(self.scope, scope)
        with self.assertRaises(PermissionError): supplement.validate(self.manifest)

    def test_new_input_hash_cannot_authorize_changed_question(self):
        cases = deepcopy(self.base['cases']); cases['cases'][0]['prompt'] += ' altered'
        self.save_inputs(cases, deepcopy(self.base['materials']))
        with self.assertRaises(PermissionError): supplement.validate(self.manifest)

    def test_material_copy_path_can_change_but_identity_cannot(self):
        cases, materials = deepcopy(self.base['cases']), deepcopy(self.base['materials'])
        cases['cases'][0]['materials'][0]['pdf_path'] = '/new/copy.pdf'
        materials['materials'][0]['pdf_path'] = '/new/copy.pdf'
        self.save_inputs(cases, materials)
        supplement.validate(self.manifest)
        materials['materials'][0]['title'] += ' altered'
        self.save_inputs(cases, materials)
        with self.assertRaises(PermissionError): supplement.validate(self.manifest)

    def test_changed_business_bytes_are_rejected(self):
        (self.new/'snapshots/B/backend/research_loop/example.py').write_text('SYNTHETIC = 2\n')
        with self.assertRaisesRegex(PermissionError, 'BUSINESS_SOURCE'): supplement.validate(self.manifest)

    def test_runtime_cannot_write_scope_or_reuse_old_control(self):
        for path in (self.new/'runtime/auth.json', self.old/'auth.json', self.new/'inputs/auth.json'):
            m = deepcopy(self.manifest); m['authorization_path'] = str(path)
            with self.assertRaises(PermissionError): supplement.validate(m)

    def test_unknown_supplement_version_does_not_fall_back(self):
        m = deepcopy(self.manifest); m['completion_contract']['version'] += '_typo'
        with self.assertRaises(PermissionError): completion.validate_completion(m)

    def test_settings_or_qualification_subset_change_rejected(self):
        for key, value in (('settings', {'model': 'other'}), ('qualification_cells', supplement.ORDER[:2])):
            m = deepcopy(self.manifest); m[key] = value
            with self.assertRaises(PermissionError): supplement.validate(m)

    def test_existing_observation_versions_still_require_original_proofs(self):
        m = deepcopy(self.manifest); m['completion_contract'] = {'version': 'agent_final_completion_v6'}
        with self.assertRaises(PermissionError): validate_observation_policy(m)

    def test_real_anchor_rejects_unapproved_seal_before_reading_history(self):
        self.anchor_patch.stop()
        (self.old/'SEAL.json').write_text('{}')
        with self.assertRaisesRegex(PermissionError, 'APPROVED_ANCHOR_SEAL'): supplement._anchor(self.old)


class SupplementRuntimeMatrixTests(unittest.TestCase):
    """复用完整54材料副本边界，只隔离已在上组检查的锚合同。"""

    def setUp(self):
        from unit import test_agent_final_runtime as fixtures
        fixtures.FinalManifestTests.setUp(self)
        self.cases = self.cases[:6]
        for case, name in zip(self.cases, dict.fromkeys(cell[0] for cell in supplement.ORDER)):
            case['case_id'] = name
        self.m['completion_contract'] = {'version': supplement.VERSION}
        self.m['execution_order'] = deepcopy(supplement.ORDER)
        self.m['ledger_limits']['allowed_cells'] = deepcopy(supplement.ORDER)
        write(self.inputs/'cases.json', {'cases': self.cases})
        write(self.inputs/'execution-plan.json', {'execution_order': supplement.ORDER})
        self.m['final_cases_sha256'] = digest(self.inputs/'cases.json')
        self.m['final_plan_sha256'] = digest(self.inputs/'execution-plan.json')

    def test_explicit_twelve_dispatch_keeps_all_material_copy_checks(self):
        from agent_final_runtime import validate_suite_manifest
        with patch.object(supplement, 'validate', return_value={}) as validated:
            self.assertEqual(validate_suite_manifest(self.m)['cells'], 12)
            validated.assert_called_once_with(self.m)
            (self.inputs/'materials/53.json').write_text('changed')
            with self.assertRaises(PermissionError): validate_suite_manifest(self.m)

    def test_missing_explicit_version_never_relaxes_original_96_matrix(self):
        from agent_final_runtime import validate_suite_manifest
        self.m.pop('completion_contract')
        with self.assertRaises(PermissionError): validate_suite_manifest(self.m)


if __name__ == '__main__':
    unittest.main()
