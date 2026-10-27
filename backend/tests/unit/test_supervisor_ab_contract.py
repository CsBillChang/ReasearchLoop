"""Supervisor单变量合同与外部配对门；全部材料、收据和判词为合成。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from harness_ab_experiment import (OLD_RULE_4, OLD_RULE_7, PAIR_REVIEW_POLICY, SUPERVISOR,
    build_supervisor_contract, canonical, scope_candidate, sha256, validate_scope_manifest)
from harness_ab_pair_gate import (assert_fresh_gate_root, evaluate_pair_review, initialize_scope_session,
    issue_cell_grant, manifest_identity, prepare_pair_gate, validate_prior_pair_gates, wait_pair_review)
from run_read_paper_ab import qualification_configuration, select_variant_source, validate_manifest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


class SupervisorAbContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='supervisor-contract-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.runtime = self.root / 'runtime'
        self.runtime.mkdir()
        self.roots = {v: str(self.root / v) for v in ('A', 'B')}
        source = "SYSTEM_PROMPT = " + repr('合成职责。' + OLD_RULE_4 + OLD_RULE_7) + "\n\ndef helper():\n    return 1\n"
        for variant in ('A', 'B'):
            package = Path(self.roots[variant]) / SUPERVISOR
            package.parent.mkdir(parents=True)
            package.write_text(source if variant == 'A' else scope_candidate(source))
            scripts = Path(self.roots[variant]) / 'backend/scripts'
            scripts.mkdir()
            (scripts / 'common.py').write_text('VALUE = 1\n')
        self.material = self.root / 'synthetic.pdf'
        self.material.write_bytes(b'synthetic fixture, not a real PDF parser input')
        self.case = {'case_id': 'SP1-Q1', 'prompt': '合成研究问题',
            'material': {'pdf_path': str(self.material), 'pdf_sha256': sha256(self.material.read_bytes()),
                'authorized_physical_pages': [1], 'physical_pages': 1}}
        self.cases = self.root / 'cases.json'
        write(self.cases, {'cases': [self.case, {**self.case, 'case_id': 'SP1-Q2'}]})
        self.manifest = {'experiment_kind': 'supervisor_scope_v1', 'runtime_root': str(self.runtime),
            'source_roots': self.roots, 'cases_path': str(self.cases),
            'supervisor_prompt_contract': build_supervisor_contract(self.roots),
            'semantic_gate_root': str(self.root / 'semantic-gates'),
            'pair_review_policy': copy.deepcopy(PAIR_REVIEW_POLICY),
            'variant_settings': {v: {'context_deduplicate_read_paper': True} for v in ('A', 'B')},
            'settings': {'retrieval_route': 'located_fp16', 'retrieval_reranker_backend': 'mps_fp16'},
            'execution_order': [['SP1-Q1','A'],['SP1-Q1','B'],['SP1-Q2','B'],['SP1-Q2','A']],
            'ledger_limits': {'allowed_cells': [['SP1-Q1','A'],['SP1-Q1','B'],['SP1-Q2','A'],['SP1-Q2','B']]}}
        self.rows = []
        for variant in ('A','B'):
            cell = self.runtime / 'cells' / ('SP1-Q1-' + variant)
            root_id = 'synthetic-root-' + variant
            write(cell / 'result.json', {'status':'executed','root_id':root_id})
            write(cell / 'task-view.json', {'task_id':root_id,'prompt':self.case['prompt'],
                'status':'completed','result':{'answer_markdown':'合成有用答案' + variant}})
            write(cell / 'model-calls.json', [])
            write(cell / 'actions.json', {'hidden-root-id': [{'observation':{'data':{'pages':[
                {'evidence_id':'synthetic-evidence','page_number':1,'text':'合成原文，供引用定位。'}]}}}]})
            self.rows.append({'case_id':'SP1-Q1','variant':variant,'root_id':root_id,'issues':[]})

    def challenge(self):
        return prepare_pair_gate(self.manifest, 'SP1-Q1', self.rows, manifest_identity(self.manifest), clock=lambda:100.0)

    def review(self, challenge, *, a='full', b='full'):
        return {'version':1,'challenge_nonce':challenge['nonce'],
            'challenge_sha256':sha256(canonical(challenge)),'reviewed_at':101.0,
            'reviewer':{'kind':'agent_assisted','identity':'synthetic-reviewer','blinded':True,
                'full_authorized_source_checked':True,'paid_model_calls':0},
            'reviews':[{'anonymous_id':challenge['anonymous_ids'][v],
                'answer_sha256':challenge['identity']['answers'][v]['answer_sha256'],
                'source_sha256s':[self.case['material']['pdf_sha256']], 'core_complete':True,
                'critical_core_error':False,'requested_scope_complete':True,
                'whole_answer_quality':a if v=='A' else b,'evidence_notes':'仅为合成门验证，不是真实语义评分。'} for v in ('A','B')]}

    def evaluate(self, challenge, review):
        return evaluate_pair_review(self.manifest,'SP1-Q1',self.rows,manifest_identity(self.manifest),challenge,review,clock=lambda:102.0)

    def test_scope_variants_require_both_dedup_on(self):
        validate_manifest(self.manifest,'SP1-Q1','A','qualify')
        validate_manifest(self.manifest,'SP1-Q1','B','qualify')
        self.manifest['variant_settings']['A']['context_deduplicate_read_paper']=False
        with self.assertRaises(PermissionError):validate_manifest(self.manifest,'SP1-Q1','A','qualify')

    def test_scope_digest_cannot_be_rebound_to_arbitrary_field_or_missing_contract(self):
        self.manifest['supervisor_prompt_contract']['variants']['B']['system_prompt_sha256']='0'*64
        with self.assertRaises(PermissionError):validate_scope_manifest(self.manifest)
        self.manifest.pop('supervisor_prompt_contract')
        with self.assertRaises(PermissionError):validate_scope_manifest(self.manifest)

    def test_source_package_second_variable_is_rejected(self):
        for variant,value in [('A','VALUE = 1'),('B','VALUE = 2')]:
            (Path(self.roots[variant])/'backend/research_loop/other.py').write_text(value)
        with self.assertRaises(PermissionError):validate_scope_manifest(self.manifest)

    def test_common_script_drift_is_rejected(self):
        (Path(self.roots['B'])/'backend/scripts/common.py').write_text('VALUE = 2\n')
        with self.assertRaises(PermissionError):validate_scope_manifest(self.manifest)

    def test_additional_supervisor_function_change_is_rejected(self):
        path=Path(self.roots['B'])/SUPERVISOR
        path.write_text(path.read_text().replace('return 1','return 2'))
        with self.assertRaises(PermissionError):validate_scope_manifest(self.manifest)

    def test_additional_prompt_example_is_rejected(self):
        path=Path(self.roots['B'])/SUPERVISOR
        path.write_text(path.read_text().replace('合成职责。','合成职责。增加未批准示例。'))
        with self.assertRaises(PermissionError):validate_scope_manifest(self.manifest)

    def test_same_source_roots_cannot_hide_prompt_treatment(self):
        self.manifest['source_roots']['B']=self.roots['A']
        with self.assertRaises(PermissionError):validate_scope_manifest(self.manifest)

    def test_gate_control_inside_runtime_or_source_is_rejected(self):
        for value in [str(self.runtime),str(self.runtime/'semantic-gates'),str(Path(self.roots['A'])/'semantic-gates')]:
            with self.subTest(value=value):
                self.manifest['semantic_gate_root']=value
                with self.assertRaises(PermissionError):validate_scope_manifest(self.manifest)

    def test_runtime_named_gate_root_does_not_alias_child_write_access(self):
        self.manifest['runtime_root']=str(self.root/'semantic-gates')
        with self.assertRaises(PermissionError):validate_scope_manifest(self.manifest)

    def test_unpaired_order_is_rejected(self):
        self.manifest['execution_order'][1],self.manifest['execution_order'][2]=self.manifest['execution_order'][2],self.manifest['execution_order'][1]
        with self.assertRaises(PermissionError):validate_scope_manifest(self.manifest)

    def test_qualification_binds_prompt_contract_and_gate_policy(self):
        frozen=qualification_configuration(self.manifest)
        self.assertEqual(frozen['supervisor_prompt_contract'],self.manifest['supervisor_prompt_contract'])
        self.assertEqual(frozen['pair_review_policy'],PAIR_REVIEW_POLICY)
        self.assertEqual(frozen['semantic_gate_root'],self.manifest['semantic_gate_root'])

    def test_preloaded_controller_package_cannot_be_reused_for_other_variant(self):
        fake=SimpleNamespace(__file__=str(Path(self.roots['B'])/'backend/research_loop/__init__.py'))
        with patch.dict(sys.modules,{'research_loop':fake}):
            with self.assertRaises(PermissionError):select_variant_source(self.manifest,'A')

    def test_review_control_is_outside_child_boundary(self):
        from harness_ab_boundary import Boundary
        boundary=Boundary(self.runtime)
        self.assertFalse(boundary.allowed_path(Path(self.manifest['semantic_gate_root'])/'approval.json'))
        self.assertTrue(boundary.allowed_path(self.runtime/'cells/answer.json'))

    def test_gate_root_must_be_new_before_any_session_or_task(self):
        assert_fresh_gate_root(self.manifest)
        Path(self.manifest['semantic_gate_root']).mkdir()
        with self.assertRaises(PermissionError):assert_fresh_gate_root(self.manifest)

    def test_anonymous_packet_has_no_variant_root_or_cost_and_has_evidence_locator(self):
        self.challenge()
        packet=json.loads((Path(self.manifest['semantic_gate_root'])/'SP1-Q1/review-packet.json').read_text())
        for answer in packet['answers']:
            self.assertNotIn('variant',answer)
            self.assertNotIn('root_id',answer)
            evidence=answer['evidence_locators']['evidence'][0]
            self.assertEqual(evidence['physical_page'],1)
            self.assertEqual(evidence['source_sha256'],self.case['material']['pdf_sha256'])
        self.assertNotIn('hidden-root-id',canonical(packet))

    def test_complete_full_pair_passes_and_declares_agent_kind(self):
        challenge=self.challenge();result=self.evaluate(challenge,self.review(challenge))
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['reviewer_kind'],'agent_assisted')
        self.assertTrue(result['quality_equivalent'])

    def test_candidate_full_to_partial_stops_without_hiding_costs(self):
        challenge=self.challenge();result=self.evaluate(challenge,self.review(challenge,b='partial_usable'))
        self.assertEqual(result['status'],'stopped')
        self.assertIn('candidate_quality_regression',result['stop_reasons'])

    def test_both_partial_remain_diagnostic_not_quality_equivalent(self):
        challenge=self.challenge();result=self.evaluate(challenge,self.review(challenge,a='partial_usable',b='partial_usable'))
        self.assertEqual(result['status'],'passed')
        self.assertFalse(result['quality_equivalent'])

    def test_missing_core_or_scope_stops_even_when_useful(self):
        challenge=self.challenge();review=self.review(challenge,a='partial_usable',b='partial_usable')
        review['reviews'][1]['requested_scope_complete']=False
        review['reviews'][0]['core_complete']=False
        result=self.evaluate(challenge,review)
        self.assertEqual(result['status'],'stopped')
        self.assertIn('candidate_requested_scope_missing',result['stop_reasons'])

    def test_full_label_cannot_hide_missing_core(self):
        challenge=self.challenge();review=self.review(challenge)
        review['reviews'][0]['core_complete']=False
        with self.assertRaises(PermissionError):self.evaluate(challenge,review)

    def test_stale_challenge_and_wrong_material_or_answer_are_rejected(self):
        challenge=self.challenge()
        for change in [lambda r:r.update(challenge_nonce='old'),
                lambda r:r['reviews'][0].update(answer_sha256='wrong'),
                lambda r:r['reviews'][0].update(source_sha256s=['wrong']),
                lambda r:r.update(reviewed_at=99.0)]:
            review=self.review(challenge);change(review)
            with self.assertRaises(PermissionError):self.evaluate(challenge,review)

    def test_post_review_raw_answer_or_receipt_drift_is_rejected(self):
        challenge=self.challenge();review=self.review(challenge)
        path=self.runtime/'cells/SP1-Q1-A/model-calls.json'
        path.write_text('[{"changed":true}]')
        with self.assertRaises(PermissionError):self.evaluate(challenge,review)

    def test_reviewer_declaration_requires_kind_identity_source_and_zero_paid_calls(self):
        challenge=self.challenge()
        for key,value in [('kind','certified_human_gold'),('identity',''),('blinded',False),('paid_model_calls',1)]:
            review=self.review(challenge);review['reviewer'][key]=value
            with self.assertRaises(PermissionError):self.evaluate(challenge,review)

    def test_duplicate_anonymous_review_cannot_replace_other_answer(self):
        challenge=self.challenge();review=self.review(challenge)
        review['reviews'][1]=copy.deepcopy(review['reviews'][0])
        with self.assertRaises(PermissionError):self.evaluate(challenge,review)

    def test_wait_deadline_stops_without_any_external_or_next_cell_call(self):
        now=[100.0]
        def sleep(seconds):now[0]+=seconds
        result=wait_pair_review(self.manifest,'SP1-Q1',self.rows,manifest_identity(self.manifest),101.0,
            clock=lambda:now[0],sleep=sleep)
        self.assertEqual(result['stop_reasons'],['semantic_review_deadline'])
        self.assertEqual(now[0],101.0)
        self.assertFalse((self.runtime/'cells/SP1-Q2-B').exists())

    def test_direct_later_cell_requires_current_scheduler_grant_and_prior_review(self):
        initialize_scope_session(self.manifest,100.0,clock=lambda:100.0)
        self.manifest['cell_absolute_deadline']=200.0
        with self.assertRaises(PermissionError):validate_prior_pair_gates(self.manifest,'SP1-Q2','B',clock=lambda:102.0)
        issue_cell_grant(self.manifest,'SP1-Q2','B',200.0)
        with self.assertRaises(PermissionError):validate_prior_pair_gates(self.manifest,'SP1-Q2','B',clock=lambda:102.0)
        self.assertFalse((self.runtime/'cells/SP1-Q2-B').exists())

    def test_passed_prior_pair_allows_next_granted_cell_but_tamper_stops(self):
        initialize_scope_session(self.manifest,100.0,clock=lambda:100.0)
        challenge=self.challenge();review=self.review(challenge);decision=self.evaluate(challenge,review)
        directory=Path(self.manifest['semantic_gate_root'])/'SP1-Q1'
        write(directory/'review.json',review);write(directory/'decision.json',decision)
        self.manifest['cell_absolute_deadline']=200.0
        issue_cell_grant(self.manifest,'SP1-Q2','B',200.0)
        validate_prior_pair_gates(self.manifest,'SP1-Q2','B',clock=lambda:102.0)
        decision['review_sha256']='tampered';write(directory/'decision.json',decision)
        with self.assertRaises(PermissionError):validate_prior_pair_gates(self.manifest,'SP1-Q2','B',clock=lambda:102.0)

    def test_finalized_batch_or_extended_grant_deadline_cannot_resume(self):
        initialize_scope_session(self.manifest,100.0,clock=lambda:100.0)
        self.manifest['cell_absolute_deadline']=200.0
        issue_cell_grant(self.manifest,'SP1-Q1','A',200.0)
        validate_prior_pair_gates(self.manifest,'SP1-Q1','A',clock=lambda:102.0)
        self.manifest['cell_absolute_deadline']=201.0
        with self.assertRaises(PermissionError):validate_prior_pair_gates(self.manifest,'SP1-Q1','A',clock=lambda:102.0)
        self.manifest['cell_absolute_deadline']=200.0
        write(self.runtime/'batch-results.json',{'status':'stopped'})
        with self.assertRaises(PermissionError):validate_prior_pair_gates(self.manifest,'SP1-Q1','A',clock=lambda:102.0)


if __name__=='__main__':
    unittest.main()
