"""补测主96格、多锚历史与未决成本的合成零模型回归。"""
from copy import deepcopy
from pathlib import Path
import json
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'scripts'),str(Path(__file__).resolve().parents[2])]
import agent_final_analysis as analysis


class CompletionAnalysisTests(unittest.TestCase):
    """全部使用临时材料与合成时钟，禁止真实连接。"""

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name in ('connect','connect_ex'):
            guard=patch.object(socket.socket,name,side_effect=AssertionError('network forbidden'))
            mocked=guard.start();self.addCleanup(guard.stop)
            self.addCleanup(lambda mocked=mocked:self.assertEqual(mocked.call_count,0))

    def write(self,path,obj):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(obj));return path

    @staticmethod
    def order():
        order=[[f'AFE-QA-{i:02d}-R{r}',v] for r in (1,2) for i in range(1,25) for v in ('A','B')]
        order[10],order[11]=order[11],order[10]
        return order

    @staticmethod
    def row(case,variant):
        base,r=case.rsplit('-R',1);tokens=3 if variant=='A' else 2
        return {'case_id':base,'round':int(r),'expanded_case_id':case,'variant':variant,
            'root_id':case+'-'+variant,'category':'synthetic','status':'completed',
            'actual_task_family_tokens':tokens,'verified_settled_tokens_lower_bound':tokens,
            'verified_settled_nano_yuan_lower_bound':tokens+1,'usage_reconciled':True,'issues':[],
            'score':{'whole_answer_quality':'full'},'pending_reservations':[],
            'end_to_end':{'seconds':10 if variant=='A' else 8,'delivery_ready_observed':True}}

    @staticmethod
    def ledger(order,carry):
        rows=[];roots=[]
        for index,(case,variant) in enumerate(order):
            tokens=3 if variant=='A' else 2;identity=case+'-'+variant
            roots.append({'root_id':identity,'case_id':case,'variant':variant})
            rows.append({'reservation_id':'r'+str(index),'root_id':identity,'status':'settled','sent_at':100,
                'actual_input':tokens-1,'actual_output':1,'actual_tokens':tokens,'actual_nano':tokens+1})
        return {'metadata':{'config':{'carried_usage':carry},'started_at':100},'roots':roots,'reservations':rows}

    def fixture(self):
        order=self.order();history={'tokens':7,'requests':3,'nano_yuan':11,'seconds':5}
        roots=[self.root/name for name in ('n','r2','new')]
        limits={'batch_tokens':10000,'batch_requests':1000,'batch_yuan':'1','batch_seconds':1000}
        owners=[{'experiment_root':str(root),'runtime_root':str(root/'runtime'),
                 'semantic_gate_root':str(root/'reviews'),'execution_order':execution,
                 'ledger_limits':{**limits,'carried_usage':history},
                 'loaded_prompt_contract':{'A':{'synthetic':'A'},'B':{'synthetic':'B'}}}
                for root,execution in zip(roots,(order,order[4:],order[11:]))]
        n,r2,current=owners
        nledger=self.ledger(order[:5],history)
        rledger=self.ledger(order[4:12],history)
        for number in range(2):
            rledger['reservations'].append({'reservation_id':'pending'+str(number),
                'root_id':order[11][0]+'-'+order[11][1],'status':'unknown','sent_at':101,
                'reserved_tokens':9,'reserved_nano':11})
        prior=analysis.combined_ledger_totals([nledger,rledger])
        carry={key:history[key]+prior['held'][key] for key in prior['held']}|{'seconds':history['seconds']}
        current.update(completion_contract={'synthetic':True},evaluation_policy=deepcopy(analysis.COMPLETION_EVALUATION_POLICY),
            formal_controller_contract={'policy':'financial_only','cleanup_reserve_seconds':300})
        current['ledger_limits']={**limits,'carried_usage':carry}
        for owner in owners:self.write(Path(owner['experiment_root'])/'manifest.formal.json',owner)
        def item(position,owner,quality='full',technical=False):
            case,variant=order[position];source=Path(owner['experiment_root'])/'manifest.formal.json'
            value={'original_position':position,'case_id':case,'variant':variant,'quality':quality,
                'source_manifest_path':str(source),'source_manifest_sha256':analysis.sha(source.read_bytes()),
                'provenance':{'anchor_root':owner['experiment_root'],
                    'cell_root':str(Path(owner['runtime_root'])/'cells'/(case+'-'+variant))}}
            if technical:value.update(status='running',attempt_outcome='interrupted')
            return value
        retained=[item(i,n if i<4 else r2,None if i==10 else 'full') for i in range(11)]
        ctx={'anchor_manifest':n,'latest_anchor_manifest':r2,'closed_ledger_roots':[str(p) for p in roots[:2]],
             'full_execution_order':order,'retained_rows':retained,
             'technical_attempts':[item(4,n,None,True),item(11,r2,None,True)],
             'new_execution_order':order[11:],'carried_usage':carry,
             'clock':{'round_started_at':100,'pre_round_carried_seconds':5,'policy':'financial_only',
                      'round_absolute_deadline':None,'model_absolute_deadline':None}}
        ledgers={str(roots[0]):nledger,str(roots[1]):rledger,str(roots[2]):self.ledger(order[11:],carry)}
        ledgers[str(roots[2])]['metadata']['config']['enforce_batch_seconds']=False
        closure={'closed':True,'closed_at_unix':200000,'cleanup_all_passed':True,'controller':{},'checks':{}}
        return current,ctx,history,ledgers,closure

    def result(self,alter=None,row_alter=None,progress=False):
        manifest,context,history,ledgers,closure=self.fixture()
        if alter:alter(manifest,context,ledgers,closure)
        calls=[]
        def cell(reader,root,owner,case,variant,ledger,controller,**kwargs):
            row=self.row(case,variant);position=self.order().index([case,variant]);origin=Path(root).name
            technical=(origin=='n' and position==4) or (origin=='r2' and position==11)
            if technical:
                row.update(status='running',actual_task_family_tokens=None,usage_reconciled=False,
                           issues=['native_unknown'],score={'whole_answer_quality':None})
                row['end_to_end']={'seconds':None,'delivery_ready_observed':False}
            calls.append({'root':str(root),'case':case,'variant':variant,'kwargs':kwargs})
            if row_alter:row_alter(row,technical)
            return row
        module=types.ModuleType('agent_final_completion');module.validate_completion=lambda _:context
        previous=types.ModuleType('agent_final_continuation')
        previous.validate_continuation=lambda _:self.fail('v2 must use completion validation')
        def read_ledger(reader,path):return ledgers[str(Path(path).parent.parent)]
        with patch.dict(sys.modules,{'agent_final_completion':module,'agent_final_continuation':previous}), \
             patch.object(analysis,'historical_usage',return_value=history), \
             patch.object(analysis,'read_ledger',side_effect=read_ledger), \
             patch.object(analysis,'closure_status',return_value=closure), \
             patch.object(analysis,'analyze_cell',side_effect=cell):
            result=analysis.summarize(analysis.Reader(),Path(manifest['experiment_root']),manifest,progress)
        return result,calls

    def test_primary96_and_all98_finance_counts_each_anchor_once(self):
        result,calls=self.result();accounting=result['accounting']
        self.assertEqual(len(result['rows']),96)
        self.assertEqual(result['continuation']['retained_roots'],11)
        self.assertEqual(result['continuation']['new_roots_planned'],85)
        self.assertEqual(result['attempt_reliability']['all_attempts_planned'],98)
        self.assertTrue(result['primary_measurement_complete'])
        self.assertEqual(sum(row['actual_task_family_tokens'] for row in result['rows']),240)
        self.assertEqual(accounting['cumulative_known_settled_record_usage']['tokens'],253)
        self.assertEqual(accounting['cumulative_charged_or_reserved']['tokens'],271)
        self.assertEqual(accounting['cumulative_known_settled_record_usage']['requests'],101)
        self.assertEqual(accounting['cumulative_charged_or_reserved']['requests'],103)
        self.assertEqual(accounting['unknown_or_pending_count'],2)
        self.assertEqual(accounting['new_unknown_or_pending_count'],0)
        self.assertFalse(accounting['all_attempt_actual_cost_complete'])
        self.assertTrue(accounting['all_new_actual_usage_known'])
        self.assertTrue(result['efficiency_conclusion']['primary_comparison_supported'])
        self.assertFalse(result['efficiency_conclusion']['authorized_by_complete_evidence'])
        self.assertEqual(len(calls),98)
        for row in result['technical_attempts']:
            self.assertEqual(row['status'],'running')
            self.assertEqual(row['observed_attempt_outcome'],'interrupted')
            self.assertIsNone(row['actual_task_family_tokens'])

    def test_old_graded_observations_keep_owner_and_only_ungraded_B_uses_new_pair(self):
        result,calls=self.result()
        retained=[row for row in calls if row['kwargs'].get('artifact_cell')]
        self.assertEqual(len(retained),13)
        for row in retained:
            owner=row['kwargs']['score_manifest']['experiment_root']
            target='new' if [row['case'],row['variant']]==self.order()[10] else Path(row['root']).name
            self.assertEqual(Path(owner).name,target)
        self.assertTrue(all(row['original_batch_record'] for row in result['technical_attempts']))

    def test_removed_44_floor_does_not_remove_relative_quality_comparison(self):
        def quality(a_full,b_full):
            def alter(row,technical):
                if technical:return
                i=(row['round']-1)*24+int(row['case_id'].rsplit('-',1)[1])-1
                if i>=(a_full if row['variant']=='A' else b_full):row['score']['whole_answer_quality']='partial_usable'
            return alter
        result,_=self.result(row_alter=quality(40,40))
        self.assertIsNone(result['quality']['fixed_minimum_B_full'])
        self.assertEqual(result['quality']['required_B_full'],40)
        self.assertTrue(result['quality']['quality_gate_passed'])
        result,_=self.result(row_alter=quality(40,39))
        self.assertFalse(result['quality']['relative_quality_non_regression'])
        self.assertFalse(result['efficiency_conclusion']['primary_comparison_supported'])

    def test_partial_terminal_duration_is_not_success_latency(self):
        def partial(row,technical):
            if not technical and row['expanded_case_id']==self.order()[0][0] and row['variant']=='A':
                row.update(status='partial',score={'whole_answer_quality':'partial_usable'})
                row['end_to_end']={'seconds':259.7,'delivery_ready_observed':False}
        def grade(manifest,context,ledgers,closure):context['retained_rows'][0]['quality']='partial_usable'
        result,_=self.result(alter=grade,row_alter=partial)
        self.assertTrue(result['primary_measurement_complete'])
        self.assertIsNone(result['statistics']['overall']['latency_seconds']['change'])
        self.assertIsNotNone(result['terminal_outcome_statistics']['overall']['latency_seconds']['change'])
        self.assertIsNotNone(result['statistics']['overall']['tokens']['change'])
        self.assertFalse(result['efficiency_conclusion']['primary_comparison_supported'])
        report=analysis.report_text(result)
        self.assertIn('包含partial/failed，非成功交付时延',report)
        self.assertIn('本轮不设B full固定下限',report)

    def test_financial_only_keeps_original_clock_without_expired_global_cap(self):
        result,_=self.result();a=result['accounting']
        self.assertEqual(a['cumulative_known_settled_record_usage']['seconds'],199905)
        self.assertIsNone(a['original_cumulative_limits']['seconds'])
        self.assertIsNone(a['within_original_cumulative_limits']['seconds'])
        self.assertIsNone(a['new_allowance']['seconds'])
        self.assertEqual(a['historical_batch_seconds_not_a_current_cap'],1000)
        self.assertNotIn('original_cumulative_budget_exceeded',result['issues'])

    def test_missing_primary_or_unknown_main_usage_cannot_be_hidden_by_old_pending(self):
        def missing(row,technical):
            if not technical and [row['expanded_case_id'],row['variant']]==self.order()[-1]:
                row.update(root_id=None,status='not_created',actual_task_family_tokens=None,
                    verified_settled_tokens_lower_bound=None,usage_reconciled=False,score={'whole_answer_quality':None})
                row['end_to_end']={'seconds':None,'delivery_ready_observed':False}
        result,_=self.result(row_alter=missing)
        self.assertEqual(result['created_roots'],95)
        self.assertFalse(result['primary_measurement_complete'])
        self.assertIsNone(result['statistics']['overall']['tokens']['change'])
        self.assertIsNone(result['terminal_outcome_statistics']['overall']['latency_seconds']['change'])

    def test_wrong_carry_or_policy_blocks_results(self):
        def carry(manifest,context,ledgers,closure):context['carried_usage']['tokens']+=1
        result,_=self.result(alter=carry)
        self.assertIn('anchor_financial_carry_mismatch',result['issues'])
        self.assertFalse(result['efficiency_conclusion']['primary_comparison_supported'])
        def policy(manifest,context,ledgers,closure):manifest['evaluation_policy']['minimum_B_full']=44
        with self.assertRaisesRegex(PermissionError,'fixed full-count floor'):self.result(alter=policy)
        def old_timer(manifest,context,ledgers,closure):
            ledgers[manifest['experiment_root']]['metadata']['config']['enforce_batch_seconds']=True
        result,_=self.result(alter=old_timer)
        self.assertIn('financial_only_ledger_policy_mismatch',result['issues'])

    def test_progress_still_never_authorizes_final_conclusion(self):
        result,_=self.result(progress=True)
        self.assertEqual(result['status'],'progress_only')
        self.assertFalse(result['efficiency_conclusion']['primary_comparison_supported'])

    def test_source_manifest_hash_and_original_grade_are_not_rewritten(self):
        def changed(manifest,context,ledgers,closure):context['retained_rows'][0]['source_manifest_sha256']='bad'
        with self.assertRaisesRegex(ValueError,'source manifest identity'):self.result(alter=changed)
        def grade(manifest,context,ledgers,closure):context['retained_rows'][0]['quality']='partial_usable'
        with self.assertRaisesRegex(ValueError,'original grade changed'):self.result(alter=grade)

    def test_unknown_reservations_stay_held_with_settled_requests_distinguished(self):
        ledger=self.ledger(self.order()[:1],{})
        ledger['reservations']+=[{'status':'unknown','sent_at':1,'reserved_tokens':20,'reserved_nano':30},
                               {'status':'not_sent','sent_at':None}]
        totals=analysis.combined_ledger_totals([ledger])
        self.assertEqual(totals['known'],{'tokens':3,'requests':1,'nano_yuan':4})
        self.assertEqual(totals['held'],{'tokens':23,'requests':2,'nano_yuan':34})
        self.assertEqual(totals['physical_sent_request_count'],2)
        self.assertEqual(totals['pending'][0]['status'],'unknown')

    def test_financial_only_closure_needs_actual_bounded_cleanup_and_closed_scope(self):
        root=self.root/'closure';auth_path=root/'authorization.json'
        manifest={'completion_contract':{'clock':{'round_started_at':50}},'authorization_path':str(auth_path),
                  'formal_controller_contract':{'policy':'financial_only','cleanup_reserve_seconds':300}}
        controller={'policy':'financial_only','status':'financial_budget_halt','started_at':100,
            'cleanup_started_at':1000,'model_processes_stopped_at':1005,'cleanup_deadline_after_stop':1300,
            'finished_at':1010,'wrapper_finalized_at':1012,'batch_group_cleaned':True,'cleanup_all_passed':True}
        self.write(root/'formal-control/controller-result.json',controller)
        self.write(auth_path,{'approved':False,'allowed_modes':[],'round_started_at':50})
        auth={'status':'closed','policy':'financial_only','round_started_at':50,'no_resume_or_resend':True,'at':1013}
        self.write(root/'formal-preflight/authorization-closed.json',auth)
        self.write(root/'formal-preflight/vector-close.json',{'identity':{'running':False},'at':1014})
        result=analysis.closure_status(analysis.Reader(),root,manifest)
        self.assertTrue(result['closed'])
        for change in ({'cleanup_started_at':1005},{'wrapper_finalized_at':1301},
                       {'model_processes_stopped_at':1301},{'cleanup_all_passed':False}):
            self.write(root/'formal-control/controller-result.json',{**controller,**change})
            self.assertFalse(analysis.closure_status(analysis.Reader(),root,manifest)['closed'])
        self.write(root/'formal-control/controller-result.json',controller)
        self.write(root/'formal-preflight/authorization-closed.json',{**auth,'no_resume_or_resend':False})
        self.assertFalse(analysis.closure_status(analysis.Reader(),root,manifest)['closed'])

    def test_mixed_pair_review_binds_old_cell_and_old_prompt_identity(self):
        case='AFE-QA-01-R1';variant='B';answer='synthetic retained answer';identity='old-B'
        old={'loaded_prompt_contract':{'B':{'prompt':'oldB'}}}
        current={'completion_contract':{},'semantic_gate_root':str(self.root/'reviews'),
            'runtime_root':str(self.root/'new-runtime'),'loaded_prompt_contract':{'B':{'prompt':'newB'}}}
        cell=self.root/'old-runtime/cells'/('AFE-QA-01-R1-B')
        artifact=self.write(cell/'result.json',{'root_id':identity})
        source={'cell_root':str(cell),'manifest_sha256':analysis.sha(analysis.canonical(old)),
            'source_contract_sha256':analysis.sha(analysis.canonical({'source':'oldB'})),
            'loaded_prompt_contract_sha256':analysis.sha(analysis.canonical(old['loaded_prompt_contract']['B'])),'retained':True}
        answer_id={'root_id':identity,'answer_sha256':analysis.sha(answer),'source_provenance':source,
                   'files':{'result.json':analysis.sha(artifact.read_bytes())}}
        challenge={'manifest_sha256':analysis.sha(analysis.canonical(current)),'nonce':'synthetic',
            'anonymous_ids':{'B':'answer-one'},'identity':{'answers':{'B':answer_id},'materials':[{'pdf_sha256':'pdf'}]}}
        grade={'anonymous_id':'answer-one','answer_sha256':analysis.sha(answer),'source_sha256s':['pdf'],
            'whole_answer_quality':'full','core_complete':True,'requested_scope_complete':True,
            'critical_core_error':False,'execution_complete':True,'unauthorized_delivery_write':False,'forged_success':False}
        review={'challenge_nonce':'synthetic','challenge_sha256':analysis.sha(analysis.canonical(challenge)),'reviews':[grade]}
        decision={'review_sha256':analysis.sha(analysis.canonical(review)),
            'challenge_sha256':analysis.sha(analysis.canonical(challenge)),'quality':{'B':'full'}}
        for name,obj in [('challenge-private.json',challenge),('review.json',review),('decision.json',decision)]:
            self.write(self.root/'reviews'/case/name,obj)
        module=types.ModuleType('harness_ab_experiment');module.paired_source_contract=lambda _: {'source':'oldB'}
        with patch.dict(sys.modules,{'harness_ab_experiment':module}):
            score=analysis.semantic_score(analysis.Reader(),self.root,current,case,variant,identity,answer,
                                         artifact_cell=cell,source_manifest=old)
            self.assertEqual(score['whole_answer_quality'],'full')
            score=analysis.semantic_score(analysis.Reader(),self.root,current,case,variant,identity,answer,
                                         artifact_cell=cell,source_manifest=current)
            self.assertIn('review_artifact_source_provenance_mismatch',score['issues'])
            self.write(artifact,{'root_id':'changed'})
            score=analysis.semantic_score(analysis.Reader(),self.root,current,case,variant,identity,answer,
                                         artifact_cell=cell,source_manifest=old)
            self.assertIn('review_export_file_mismatch:result.json',score['issues'])


if __name__=='__main__':unittest.main()
