"""单次准入的owner context复用；合成文件与计数，不连接真实模型或旧库。"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

BACKEND=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(BACKEND/'scripts'),str(BACKEND),str(BACKEND/'tests/unit')]
import agent_final_completion as completion
import agent_final_continuation as continuation
import harness_ab_pair_gate as gate


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')


class GateContextReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='gate-owner-context-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.now=1000.;self.calls=Counter();self.fail_owner=None
        self.order=[[case,v] for case in ('OLD0','CROSS','OLD2','OLD3','NEXT') for v in ('A','B')]
        self.cases=[{'case_id':case,'suite_kind':'agent_final_v1','prompt':'Synthetic '+case,'materials':[],
            'preconditions':{'catalog':[],'save_policy':'deny','history_messages':[{'role':'user','content':'Synthetic history.'}]}}
            for case in ('OLD0','CROSS','OLD2','OLD3','NEXT')]
        self.manifests={};self.paths={}
        for name in ('N','R2','R3','CURRENT'):
            root=self.root/name;root.mkdir()
            m={'owner_label':name,'experiment_kind':'prompt_bundle_v1','experiment_root':str(root),
                'runtime_root':str(root/'runtime'),'semantic_gate_root':str(root/'semantic-gates'),
                'cases_path':str(root/'cases.json'),'execution_order':self.order,
                'final_suite_policy':deepcopy(gate.FINAL_SUITE_POLICY),
                'pair_review_policy':{'poll_seconds':.01},'ledger_limits':{'root_seconds':900},
                'prompt_bundle_source_contract':{'owner':name},
                'loaded_prompt_contract':{v:{'owner':name,'variant':v} for v in ('A','B')}}
            if name in {'R3','CURRENT'}:m['completion_contract']={'version':'synthetic-owner-contract'}
            self.manifests[name]=m;self.paths[name]=root/'manifest.json'
            write(self.paths[name],m);write(Path(m['cases_path']),{'cases':self.cases})
        self.current=self.manifests['CURRENT']
        self.retained=[];self.rows={};self.cell={}
        sources={'OLD0':{'A':'N','B':'N'},'CROSS':{'A':'R3','B':'R2'},
                 'OLD2':{'A':'R3','B':'R3'},'OLD3':{'A':'N','B':'N'}}
        score_owner={'OLD0':'N','CROSS':'R3','OLD2':'R3','OLD3':'N'}
        for position,(case,variant) in enumerate(self.order[:8]):
            source=sources[case][variant];m=self.manifests[source]
            cell=Path(m['runtime_root'])/'cells'/(case+'-'+variant);root_id='synthetic-'+case+'-'+variant
            result={'case_id':case,'variant':variant,'root_id':root_id,'status':'executed',
                    'task_status':'completed','loaded_supervisor':m['loaded_prompt_contract'][variant]}
            view={'task_id':root_id,'status':'completed','prompt':'Synthetic '+case,
                  'result':{'answer_markdown':'Synthetic answer only, not a real grade.'}}
            for filename,value in [('result.json',result),('task-view.json',view),
                ('model-calls.json',[{'status':'succeeded'}]),('actions.json',{})]:write(cell/filename,value)
            row={'case_id':case,'variant':variant,'root_id':root_id,'status':'completed','issues':[],
                'original_position':position,'source_manifest_path':str(self.paths[source]),
                'score_manifest_path':str(self.paths[score_owner[case]]),
                'provenance':{'cell_root':str(cell)},'physical_receipts_verified':True,
                'native_status_counts':{'succeeded':1},'physical_status_counts':{'settled':1}}
            self.retained.append(row);self.rows[(case,variant)]=row;self.cell[(case,variant)]=cell
        clock={'policy':'financial_only','model_absolute_deadline':None,'round_absolute_deadline':None}
        # R3当时新A尚非retained；当前轮A/B都已保留，二者不可串用。
        old_b=deepcopy(self.rows[('CROSS','B')])
        self.contexts={'N':None,'R2':None,
            'R3':{'retained_count':1,'retained_rows':[old_b],'clock':clock},
            'CURRENT':{'retained_count':8,'retained_rows':self.retained,'clock':clock}}
        for target in ((continuation,'validate_continuation'),(completion,'validate_completion')):
            item=patch.object(*target,side_effect=self.validate);item.start();self.addCleanup(item.stop)
        timer=patch.object(gate.time,'time',return_value=self.now);timer.start();self.addCleanup(timer.stop)
        for method in ('connect','connect_ex'):
            item=patch.object(socket.socket,method,side_effect=AssertionError('real network forbidden'))
            observed=item.start();self.addCleanup(item.stop)
            self.addCleanup(lambda observed=observed:self.assertEqual(observed.call_count,0))
        for case in ('OLD0','CROSS','OLD2','OLD3'):
            owner=self.manifests[score_owner[case]];rows=[self.rows[(case,v)] for v in ('A','B')]
            challenge=gate.prepare_pair_gate(owner,case,rows,gate.manifest_identity(owner),clock=lambda:self.now-10)
            review={'version':1,'challenge_nonce':challenge['nonce'],'challenge_sha256':gate.sha256(gate.canonical(challenge)),
                'reviewed_at':self.now-5,'reviewer':{'kind':'agent_assisted','identity':'synthetic-fixture',
                    'blinded':True,'full_authorized_source_checked':True,'paid_model_calls':0},
                'reviews':[{'anonymous_id':challenge['anonymous_ids'][v],
                    'answer_sha256':challenge['identity']['answers'][v]['answer_sha256'],'source_sha256s':[],
                    'whole_answer_quality':'partial_usable','core_complete':True,'critical_core_error':False,
                    'requested_scope_complete':True,'execution_complete':True,'unauthorized_delivery_write':False,
                    'forged_success':False,'evidence_notes':'Synthetic only; no real semantic review.'} for v in ('A','B')]}
            decision=gate.evaluate_pair_review(owner,case,rows,gate.manifest_identity(owner),challenge,review,clock=lambda:self.now)
            directory=Path(owner['semantic_gate_root'])/case
            write(directory/'review.json',review);write(directory/'decision.json',decision)
        gate.initialize_scope_session(self.current,self.now,clock=lambda:self.now)
        self.calls.clear()

    def validate(self,manifest):
        name=manifest['owner_label'];self.calls[name]+=1
        if name==self.fail_owner:raise PermissionError('synthetic owner validation rejected')
        return deepcopy(self.contexts[name])

    def grant(self):
        return gate.issue_cell_grant(self.current,'NEXT','A',self.now+100)

    def admission(self):
        return gate.validate_prior_pair_gates({**self.current,'cell_absolute_deadline':self.now+100},
            'NEXT','A',clock=lambda:self.now)

    def test_each_owner_validated_once_per_public_grant_and_admission(self):
        self.grant();self.assertEqual(self.calls,{'CURRENT':1,'N':1,'R3':1})
        self.calls.clear();self.admission()
        self.assertEqual(self.calls,{'CURRENT':1,'N':1,'R3':1})
        self.assertIsNone(gate._ACTIVE_GATE_CONTEXTS.get())

    def test_old_R3_new_A_retained_false_is_not_rewritten_by_current_context(self):
        old=Path(self.manifests['R3']['semantic_gate_root'])/'CROSS/challenge-private.json'
        original=json.loads(old.read_text())
        self.assertIs(original['identity']['answers']['A']['source_provenance']['retained'],False)
        self.assertIs(original['identity']['answers']['B']['source_provenance']['retained'],True)
        self.grant();self.calls.clear();self.admission()
        self.assertEqual(json.loads(old.read_text()),original)
        current,_,_=gate.pair_identity(self.current,'CROSS',[self.rows[('CROSS',v)] for v in ('A','B')])
        self.assertIs(current['answers']['A']['source_provenance']['retained'],True)

    def test_external_pair_identity_still_performs_real_validation(self):
        rows=[self.rows[('CROSS',v)] for v in ('A','B')]
        gate.pair_identity(self.current,'CROSS',rows);first=self.calls['CURRENT']
        gate.pair_identity(self.current,'CROSS',rows)
        self.assertGreaterEqual(first,1);self.assertEqual(self.calls['CURRENT'],2*first)
        self.assertIsNone(gate._ACTIVE_GATE_CONTEXTS.get())

    def test_changed_per_pair_artifacts_or_old_review_remain_rejected(self):
        self.grant()
        paths=[self.cell[('CROSS','A')]/name for name in ('result.json','model-calls.json','actions.json')]
        paths.append(Path(self.manifests['R3']['semantic_gate_root'])/'CROSS/review.json')
        for path in paths:
            before=path.read_bytes();value=json.loads(before)
            if path.name=='result.json':value['loaded_supervisor']={'changed':True}
            elif path.name=='model-calls.json':value[0]['status']='failed'
            elif path.name=='actions.json':value['changed']=[]
            else:value['reviews'][0]['critical_core_error']=True
            try:
                write(path,value)
                with self.subTest(path=path.name),self.assertRaises(PermissionError):self.admission()
            finally:path.write_bytes(before)
        self.assertIsNone(gate._ACTIVE_GATE_CONTEXTS.get())

    def test_owner_validation_failure_restores_scope_and_is_not_reused(self):
        self.grant();self.calls.clear();self.fail_owner='R3'
        with self.assertRaises(PermissionError):self.admission()
        self.assertIsNone(gate._ACTIVE_GATE_CONTEXTS.get())
        self.fail_owner=None;self.calls.clear();self.admission()
        self.assertEqual(self.calls,{'CURRENT':1,'N':1,'R3':1})

    def test_positions94_and95_recheck47_pairs_but_validate_each_owner_once(self):
        # 仅合成完整矩阵压力夹具；占位旧grant不冒充真实任务或评分。
        extra=['SYNTH%02d'%i for i in range(43)]
        names=['OLD0','CROSS','OLD2','OLD3']+extra+['NEXT']
        self.current['execution_order']=[[case,v] for case in names for v in ('A','B')]
        cases={c['case_id']:c for c in self.cases}
        for case in extra:
            cases[case]={**deepcopy(cases['NEXT']),'case_id':case,'prompt':'Synthetic '+case}
        write(self.paths['CURRENT'],self.current)
        write(Path(self.current['cases_path']),{'cases':[cases[x] for x in names]})
        control=Path(self.current['semantic_gate_root'])
        (control/'session.json').unlink()
        control.rmdir()
        gate.initialize_scope_session(self.current,self.now,clock=lambda:self.now)

        def cell_row(case,variant):
            cell=Path(self.current['runtime_root'])/'cells'/(case+'-'+variant)
            root_id='synthetic-'+case+'-'+variant
            result={'case_id':case,'variant':variant,'root_id':root_id,'status':'executed',
                'task_status':'completed','loaded_supervisor':self.current['loaded_prompt_contract'][variant]}
            view={'task_id':root_id,'status':'completed','prompt':'Synthetic '+case,
                'result':{'answer_markdown':'Synthetic pressure fixture; not a real observation.'}}
            for filename,value in [('result.json',result),('task-view.json',view),
                    ('model-calls.json',[{'status':'succeeded'}]),('actions.json',{})]:
                write(cell/filename,value)
            return {'case_id':case,'variant':variant,'root_id':root_id,'issues':[]}

        for case in extra:
            rows=[cell_row(case,v) for v in ('A','B')]
            challenge=gate.prepare_pair_gate(self.current,case,rows,gate.manifest_identity(self.current),clock=lambda:self.now-10)
            review={'version':1,'challenge_nonce':challenge['nonce'],
                'challenge_sha256':gate.sha256(gate.canonical(challenge)),'reviewed_at':self.now-5,
                'reviewer':{'kind':'agent_assisted','identity':'synthetic-pressure-fixture','blinded':True,
                    'full_authorized_source_checked':True,'paid_model_calls':0},
                'reviews':[{'anonymous_id':challenge['anonymous_ids'][v],
                    'answer_sha256':challenge['identity']['answers'][v]['answer_sha256'],'source_sha256s':[],
                    'whole_answer_quality':'partial_usable','core_complete':True,'critical_core_error':False,
                    'requested_scope_complete':True,'execution_complete':True,'unauthorized_delivery_write':False,
                    'forged_success':False,'evidence_notes':'Synthetic only; no semantic review.'} for v in ('A','B')]}
            decision=gate.evaluate_pair_review(self.current,case,rows,gate.manifest_identity(self.current),challenge,review,clock=lambda:self.now)
            write(control/case/'review.json',review);write(control/case/'decision.json',decision)
        for position in range(8,94):
            write(control/'grants'/(str(position)+'.json'),{'synthetic_pressure_placeholder':True})
        for variant,position in [('A',94),('B',95)]:
            self.calls.clear()
            with patch.object(gate,'evaluate_pair_review',wraps=gate.evaluate_pair_review) as checked:
                grant=gate.issue_cell_grant(self.current,'NEXT',variant,self.now+100)
                self.assertEqual(grant['position'],position);self.assertEqual(checked.call_count,47)
            self.assertEqual(self.calls,{'CURRENT':1,'N':1,'R3':1})
            self.calls.clear()
            with patch.object(gate,'evaluate_pair_review',wraps=gate.evaluate_pair_review) as checked:
                gate.validate_prior_pair_gates({**self.current,'cell_absolute_deadline':self.now+100},
                    'NEXT',variant,clock=lambda:self.now)
                self.assertEqual(checked.call_count,47)
            self.assertEqual(self.calls,{'CURRENT':1,'N':1,'R3':1})
            cell_row('NEXT',variant)
        self.assertIsNone(gate._ACTIVE_GATE_CONTEXTS.get())

    def test_user_dict_cannot_replace_internal_capsule(self):
        with self.assertRaises(PermissionError):gate._GateContexts(self.current,self.contexts['CURRENT'])
        token=gate._ACTIVE_GATE_CONTEXTS.set({'CURRENT':self.contexts['CURRENT']})
        try:
            with self.assertRaises(PermissionError):gate._retained_cell(self.current,'CROSS','A')
        finally:gate._ACTIVE_GATE_CONTEXTS.reset(token)


if __name__=='__main__':unittest.main(verbosity=2)
