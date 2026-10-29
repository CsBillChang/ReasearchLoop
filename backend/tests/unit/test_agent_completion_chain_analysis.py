"""V3原生工件与配对评分归属分离，不能借新manifest覆盖旧观测。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import agent_final_analysis as analysis


class CompletionChainAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve();self.owners={}
        for name in ('N','R2','R3','current'):
            root=self.base/name;root.mkdir()
            owner={'experiment_root':str(root),'runtime_root':str(root/'runtime'),'semantic_gate_root':str(root/'reviews')}
            (root/'manifest.formal.json').write_text(json.dumps(owner));self.owners[name]=owner
        cell=self.base/'R2/runtime/cells/AFE-CROSS-03-R1-B'
        path=self.base/'R2/manifest.formal.json';score=self.base/'R3/manifest.formal.json'
        self.item={'case_id':'AFE-CROSS-03-R1','variant':'B','quality':'partial_usable',
            'source_manifest_path':str(path),'source_manifest_sha256':analysis.sha(path.read_bytes()),
            'score_manifest_path':str(score),'score_manifest_sha256':analysis.sha(score.read_bytes()),
            'provenance':{'anchor_root':str(self.base/'R2'),'cell_root':str(cell)}}
        self.context={'closed_ledger_roots':[str(self.base/x) for x in ('N','R2','R3')],
                      'retained_rows':[self.item],'technical_attempts':[]}

    def artifacts(self):
        ledger={'metadata':{},'roots':[],'reservations':[]}
        calls=[]
        def cell(reader,root,owner,case,variant,*args,**kwargs):
            calls.append({'root':root,'owner':owner,**kwargs})
            return {'score':{'whole_answer_quality':'partial_usable'},'status':'completed'}
        with patch.object(analysis,'read_ledger',return_value=ledger), \
             patch.object(analysis,'closure_status',return_value={'controller':{}}), \
             patch.object(analysis,'analyze_cell',side_effect=cell):
            result=analysis.completion_artifacts(analysis.Reader(),self.owners['current'],self.context)
        return result,calls

    def test_old_B_artifacts_stay_R2_while_score_is_original_R3_pair(self):
        result,calls=self.artifacts()
        self.assertEqual(calls[0]['owner'],self.owners['R2'])
        self.assertEqual(calls[0]['score_manifest'],self.owners['R3'])
        self.assertEqual(result[0][('AFE-CROSS-03-R1','B')]['continuation_provenance']['anchor_root'],str(self.base/'R2'))

    def test_missing_foreign_or_changed_score_identity_rejected(self):
        for change in ('missing-path','missing-hash','wrong-hash','current-owner','wrong-name'):
            before=deepcopy(self.item)
            if change=='missing-path':del self.item['score_manifest_path']
            elif change=='missing-hash':del self.item['score_manifest_sha256']
            elif change=='wrong-hash':self.item['score_manifest_sha256']='0'*64
            elif change=='current-owner':self.item['score_manifest_path']=str(self.base/'current/manifest.formal.json')
            else:self.item['score_manifest_path']=str(self.base/'R3/not-a-manifest.json')
            try:
                with self.subTest(change=change),self.assertRaises(ValueError):self.artifacts()
            finally:self.item.clear();self.item.update(before)

    def test_legacy_V2_owner_rule_unchanged_without_new_fields(self):
        del self.item['score_manifest_path'];del self.item['score_manifest_sha256']
        _,calls=self.artifacts();self.assertEqual(calls[0]['score_manifest'],self.owners['R2'])
        self.item['quality']=None
        _,calls=self.artifacts();self.assertEqual(calls[0]['score_manifest'],self.owners['current'])


    def original_review_origin(self, foreign=False, recorded_retained=False, observed_stop=False, forged_full=False):
        source=deepcopy(self.owners['R2' if foreign else 'R3'])
        score=deepcopy(self.owners['R3'])
        source['loaded_prompt_contract']={'A':{'role':'same-frozen-role'}}
        if not foreign:score=deepcopy(source)
        score['completion_contract']={'version':'agent_final_completion_v4' if observed_stop else 'agent_final_completion_v3'}
        if observed_stop:
            from agent_final_observation_recovery import OBSERVATION_POLICY
            score.update(observation_policy=deepcopy(OBSERVATION_POLICY),experiment_kind='prompt_bundle_v1',
                final_suite_policy={'version':'agent_final_v1','semantic_failures':'record_and_continue'})
        if not foreign:source=deepcopy(score)
        case='SYNTH-QA-01-R1';cell=Path(source['runtime_root'])/'cells'/(case+'-A')
        directory=Path(score['semantic_gate_root'])/case;directory.mkdir(parents=True,exist_ok=True)
        answer=None if observed_stop else 'Synthetic source-backed answer.';root='synthetic-root';material='1'*64
        contract={'source_owner':source['experiment_root']}
        identity={'root_id':root,'answer_sha256':analysis.sha(answer if answer is not None else ''),'backend_status':'completed','files':{},
            'source_provenance':{'cell_root':str(cell),'manifest_sha256':analysis.sha(analysis.canonical(source)),
                'source_contract_sha256':analysis.sha(analysis.canonical(contract)),
                'loaded_prompt_contract_sha256':analysis.sha(analysis.canonical(source['loaded_prompt_contract']['A'])),
                'retained':recorded_retained}}
        if observed_stop:
            identity.update(backend_status='needs_input',observed_status='needs_input',answer_absent=True)
            cell.mkdir(parents=True,exist_ok=True)
            (cell/'task-view.json').write_text(json.dumps({'task_id':root,'status':'needs_input',
                                                         'result':{'answer_markdown':None}}))
        challenge={'nonce':'synthetic-nonce','manifest_sha256':analysis.sha(analysis.canonical(score)),
            'identity':{'answers':{'A':identity},'materials':[{'pdf_sha256':material}]},
            'anonymous_ids':{'A':'anonymous'}}
        grade={'anonymous_id':'anonymous','answer_sha256':analysis.sha(answer if answer is not None else ''),'source_sha256s':[material],
               'whole_answer_quality':'partial_usable','evidence_notes':'Synthetic independent source check.'}
        if observed_stop:
            grade.update(whole_answer_quality='full' if forged_full else 'failed',execution_complete=forged_full,
                core_complete=forged_full,requested_scope_complete=forged_full,critical_core_error=False,
                unauthorized_delivery_write=False,forged_success=False)
        review={'challenge_nonce':challenge['nonce'],'challenge_sha256':analysis.sha(analysis.canonical(challenge)),
                'reviews':[grade]}
        decision={'review_sha256':analysis.sha(analysis.canonical(review)),
            'challenge_sha256':review['challenge_sha256'],'quality':{'A':grade['whole_answer_quality']}}
        for name,value in [('challenge-private.json',challenge),('review.json',review),('decision.json',decision)]:
            (directory/name).write_text(json.dumps(value))
        with patch('harness_ab_experiment.paired_source_contract',return_value=contract):
            return analysis.semantic_score(analysis.Reader(),Path(source['experiment_root']),score,case,'A',root,answer,
                                           artifact_cell=cell,source_manifest=source)

    def test_currently_retained_artifact_can_have_been_new_at_original_review(self):
        self.assertEqual(self.original_review_origin()['whole_answer_quality'],'partial_usable')
        wrong=self.original_review_origin(recorded_retained=True)
        self.assertEqual(wrong['issues'],['review_artifact_source_provenance_mismatch'])

    def test_foreign_artifact_retained_flag_is_bound_to_original_score_owner(self):
        self.assertEqual(self.original_review_origin(foreign=True,recorded_retained=True)['whole_answer_quality'],
                         'partial_usable')
        wrong=self.original_review_origin(foreign=True,recorded_retained=False)
        self.assertEqual(wrong['issues'],['review_artifact_source_provenance_mismatch'])

    def test_absent_non_delivery_answer_is_measured_without_fake_success(self):
        actual=self.original_review_origin(observed_stop=True)
        self.assertEqual(actual['whole_answer_quality'],'failed')
        self.assertFalse(actual['execution_complete'])
        forged=self.original_review_origin(observed_stop=True,forged_full=True)
        self.assertEqual(forged['score_status'],'invalid_review_binding')
        self.assertIn('non_delivery_grade_contradiction',forged['issues'])


if __name__=='__main__':unittest.main()
