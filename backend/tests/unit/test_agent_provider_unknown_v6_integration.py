"""V6实验入口集成合成回归；证明内核由专用SQLite单测独立覆盖。"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import agent_final_completion as completion
import agent_final_completion_chain as chain
import agent_final_observation_recovery as observation
import agent_final_analysis as analysis
import harness_ab_pair_gate as gate

V6='agent_final_completion_v6'
ISSUES=['ledger_halted','native_response_unknown','unresolved_physical_receipt']


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


class ProviderUnknownV6IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.case='SYNTH-QA-01-R1';self.rid='unknown-root'
        self.proof={'version':'provider_unknown_stop_proof_v1','path':str(self.root/'proof.json'),'sha256':'a'*64}
        oldproof={'version':'root_budget_stop_proof_v1','path':str(self.root/'old-proof.json'),'sha256':'b'*64}
        self.manifest={'experiment_kind':'prompt_bundle_v1','completion_contract':{'version':V6,
            'observation_recovery_lineage':[{'anchor_root':str(self.root/'old'),'proof_path':str(self.root/'old'/observation.PROOF_PATH),'proof_sha256':'c'*64,'case_id':'SYNTH-OLD-R1'}],
            'root_budget_recovery_proof':oldproof,'root_budget_recovery_lineage':[{'anchor_root':str(self.root/'old'),'proof_ref':oldproof,'root_id':'old-budget','position':1}],
            'provider_unknown_recovery_proof':deepcopy(self.proof),'provider_unknown_recovery_lineage':[{'anchor_root':str(self.root/'closed'),'proof_ref':deepcopy(self.proof),'root_id':self.rid,'position':3}]},
            'observation_policy':deepcopy(observation.OBSERVATION_POLICY),'final_suite_policy':deepcopy(gate.FINAL_SUITE_POLICY)}
        self.proofs={'provider_unknown_stop_verified':True,'receipt_state_verified':True,'physical_receipts_verified':False,
            'actual_usage_known':False,'unconfirmed_request_count':1,'native_status_counts':{'succeeded':1,'unknown':1},'physical_status_counts':{'settled':1,'unknown':1}}
        self.cell=self.root/'closed/runtime/cells'/(self.case+'-A')
        self.retained={'case_id':self.case,'variant':'A','root_id':self.rid,'status':'failed','issues':list(ISSUES),
            'provenance':{'anchor_root':str(self.root/'closed'),'cell_root':str(self.cell)},**self.proofs}
        self.result={'case_id':self.case,'variant':'A','root_id':self.rid,'status':'executed','task_status':'failed',
            'ledger_halt':'transport_or_receipt_RemoteProtocolError','loaded_supervisor':{},
            'isolation':{'denials':[],'denied_connections':[]},
            'live_isolation_guard':{'first_error':None,'closed':False,'ledger_bound':True,'persistence_errors':[]}}
        self.view={'task_id':self.rid,'status':'failed','prompt':'Synthetic question','result':{'answer_markdown':'Synthetic failure report.','answer_kind':'failure_report'}}
        self.make_cell(self.cell,self.result,self.view)
        self.module=ModuleType('agent_final_provider_unknown_recovery');self.module._PARSED_ANCHORS={}
        self.context={**deepcopy(self.manifest['completion_contract']),'retained_rows':[deepcopy(self.retained)]}
        self.module.validate_provider_unknown_recovery=Mock(return_value=self.context)
        self.module.retained_provider_unknown_receipts=Mock(return_value=deepcopy(self.proofs))
        self.modules=patch.dict(sys.modules,{'agent_final_provider_unknown_recovery':self.module});self.modules.start();self.addCleanup(self.modules.stop)
        completion.clear_validation_caches();self.addCleanup(completion.clear_validation_caches)

    def make_cell(self,cell,result,view):
        at=lambda d:datetime.fromtimestamp(1000+d,timezone.utc).isoformat()
        values={'result.json':result,'task-view.json':view,'actions.json':{},
            'events.json':[{'task_id':result['root_id'],'event_type':'task_created','created_at':at(0),'payload':{}},
                {'task_id':result['root_id'],'event_type':'task_finished','created_at':at(1),'payload':{'status':view['status'],'result':view['result']}}],
            'model-calls.json':[{'status':'succeeded'},{'status':'unknown'}],
            'ledger-snapshot.json':{'roots':[{'root_id':result['root_id'],'case_id':result['case_id'],'variant':result['variant']}],
                'reservations':[{'root_id':result['root_id'],'status':'settled'},{'root_id':result['root_id'],'status':'unknown'}]}}
        for name,value in values.items():write(cell/name,value)

    def test_v6_shared_dispatch_parent_recursion_and_cache_clear(self):
        self.assertEqual(completion.validate_completion(self.manifest),self.context)
        self.assertEqual(chain._parent(self.manifest,{'ancestor'}),self.context)
        self.module.validate_provider_unknown_recovery.assert_called_with(self.manifest,_seen={'ancestor'})
        self.module._PARSED_ANCHORS['synthetic']='cached';completion.clear_validation_caches()
        self.assertEqual(self.module._PARSED_ANCHORS,{})
        bad=deepcopy(self.manifest);bad['continuation_contract']={}
        with self.assertRaises(PermissionError):completion.validate_completion(bad)

    def test_v6_policy_requires_typed_proof_and_all_lineages(self):
        self.assertTrue(observation.validate_observation_policy(self.manifest))
        for field in ('provider_unknown_recovery_proof','provider_unknown_recovery_lineage','root_budget_recovery_proof','observation_recovery_lineage'):
            bad=deepcopy(self.manifest);bad['completion_contract'].pop(field)
            with self.subTest(field=field),self.assertRaises(PermissionError):observation.validate_observation_policy(bad)
        bad=deepcopy(self.manifest);bad['completion_contract']['provider_unknown_recovery_lineage'][0]['proof_ref']['version']='other'
        with self.assertRaises(PermissionError):observation.validate_observation_policy(bad)

    def test_only_authority_retained_unknown_passes_and_projection_has_no_private_identifiers(self):
        stop=gate.observation_stop(self.manifest,self.cell,self.result,self.view,self.retained)
        self.assertEqual(stop['observed_status'],'failed');self.assertFalse(stop['actual_usage_known'])
        self.assertEqual(stop['unconfirmed_request_count'],1)
        private={**stop,'root_id':self.rid};public=gate.review_observation(private)
        fact=public['provider_receipt_observation']
        self.assertEqual(set(fact),{'verified_provider_unknown_stop','response_usage_unconfirmed','unconfirmed_request_count','no_resend_or_release','original_root_status','execution_complete','actual_usage_known'})
        self.assertFalse(fact['execution_complete']);self.assertFalse(fact['actual_usage_known'])
        self.assertNotIn(self.rid,json.dumps(public));self.assertNotIn(str(self.root),json.dumps(public))
        with self.assertRaises(PermissionError):gate.observation_stop(self.manifest,self.cell,self.result,self.view,None)
        self.module.retained_provider_unknown_receipts.side_effect=PermissionError('changed actual proof')
        with self.assertRaises(PermissionError):gate.observation_stop(self.manifest,self.cell,self.result,self.view,self.retained)

    def pair(self,issues):
        self.manifest['cases_path']=str(self.root/'cases.json')
        write(self.manifest['cases_path'],{'cases':[{'case_id':self.case,'prompt':self.view['prompt']}]})
        other=self.cell.with_name(self.case+'-B');rb={**deepcopy(self.result),'variant':'B','root_id':'clean-root','ledger_halt':None,'task_status':'completed'}
        vb={**deepcopy(self.view),'task_id':'clean-root','status':'completed'};self.make_cell(other,rb,vb)
        usage={'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}
        native=[{'call_id':'known','task_id':'clean-root','status':'succeeded','sent_state':'sent','request_id':'response',
            'response_json':json.dumps({'id':'response'}),'usage_json':json.dumps(usage)}]
        write(other/'model-calls.json',native)
        snapshot=json.loads((other/'ledger-snapshot.json').read_text());snapshot['reservations']=[{'root_id':'clean-root','status':'settled',
            'response_json':json.dumps({'response_id':'response'}),'usage_json':json.dumps(usage),'actual_input':1,'actual_output':1,'actual_tokens':2}]
        write(other/'ledger-snapshot.json',snapshot)
        source={'experiment_kind':'prompt_bundle_v1','prompt_bundle_source_contract':{},'loaded_prompt_contract':{'A':{},'B':{}}}
        row={'case_id':self.case,'variant':'A','root_id':self.rid}
        if issues is not None:row['issues']=issues
        rows=[row,{'case_id':self.case,'variant':'B','root_id':'clean-root','issues':[]}]
        def origin(_manifest,_case,variant):return (self.cell,source,self.retained) if variant=='A' else (other,source,None)
        with patch.object(gate,'_cell_origin',side_effect=origin),patch('prompt_bundle_cases.case_materials',return_value=[]),patch('prompt_bundle_cases.validate_preconditions',return_value={}):
            return gate.pair_identity(self.manifest,self.case,rows)

    def test_pair_issues_empty_missing_replaced_and_extra_are_rejected(self):
        identity,_,_=self.pair(list(ISSUES));self.assertTrue(identity['answers']['A']['provider_unknown_stop_verified'])
        for issues in ([],None,['replacement'],ISSUES+['extra']):
            with self.subTest(issues=issues),self.assertRaises(PermissionError):self.pair(issues)

    def test_review_reconstruction_uses_current_authority_not_caller_issues(self):
        rows=gate._review_identity_rows(self.manifest,self.case,{'A':{'root_id':self.rid}})
        self.assertEqual(rows[0]['issues'],ISSUES)
        self.module.retained_provider_unknown_receipts.return_value=None
        with self.assertRaises(PermissionError):gate._review_identity_rows(self.manifest,self.case,{'A':{'root_id':self.rid}})

    def test_analysis_keeps_unknown_actual_none_and_known_lower_bound(self):
        owner_root=self.root/'closed';owner={'experiment_root':str(owner_root),'runtime_root':str(owner_root/'runtime')}
        path=owner_root/'manifest.formal.json';write(path,owner)
        item={**self.retained,'quality':None,'source_manifest_path':str(path),'source_manifest_sha256':analysis.sha(path.read_bytes())}
        context={'version':V6,'closed_ledger_roots':[str(owner_root)],'retained_rows':[item],'technical_attempts':[]}
        cell_row={'status':'failed','score':{'whole_answer_quality':None},'issues':['ledger_halted'],
            'usage_reconciled':False,'actual_task_family_tokens':None,'actual_task_family_nano_yuan':None,
            'verified_settled_tokens_lower_bound':123456,'verified_settled_nano_yuan_lower_bound':123456}
        ledger={'metadata':{},'roots':[],'reservations':[]}
        with patch.object(analysis,'read_ledger',return_value=ledger),patch.object(analysis,'closure_status',return_value={'controller':{}}),patch.object(analysis,'analyze_cell',return_value=deepcopy(cell_row)):
            retained,technical,_,_=analysis.completion_artifacts(analysis.Reader(),self.manifest,context)
        row=retained[(self.case,'A')]
        self.assertIsNone(row['actual_task_family_tokens']);self.assertFalse(row['usage_reconciled'])
        self.assertEqual(row['verified_settled_tokens_lower_bound'],123456);self.assertEqual(row['original_reconciliation_issues'],ISSUES)
        self.assertTrue(row['provider_unknown_stop_verified']);self.assertFalse(row['actual_usage_known'])
        self.assertFalse(row['original_reconciliation_passed']);self.assertEqual(row['status'],'failed');self.assertEqual(technical,[])

    def test_summary_report_preserves_failure_and_unknown_limitations(self):
        import test_agent_completion_analysis as fixture
        f=fixture.CompletionAnalysisTests(methodName='runTest');f.setUp();self.addCleanup(f.doCleanups)
        f.order=lambda:[[f'SYNTH-QA-{i:02d}-R{r}',v] for r in (1,2) for i in range(1,25) for v in ('A','B')]
        case,variant=f.order()[0];identity=case+'-'+variant
        def alter(manifest,context,ledgers,closure):
            manifest['completion_contract']['version']=context['version']=V6
            context['retained_rows'][0]['quality']='failed'
            context['provider_unknown_recovery_lineage']=[{'root_id':identity,'proof_ref':deepcopy(self.proof)}]
        def row_alter(row,technical):
            if not technical and row['expanded_case_id']==case and row['variant']==variant:
                row.update(status='failed',provider_unknown_stop_verified=True,receipt_state_verified=True,
                    actual_usage_known=False,usage_reconciled=False,actual_task_family_tokens=None,
                    original_reconciliation_passed=False,original_reconciliation_issues=list(ISSUES),
                    provider_unknown_stop_outcome='failed_with_sent_unknown_provider_receipt')
                row['score']['whole_answer_quality']='failed';row['end_to_end'].update(delivery_ready_observed=False,successful_delivery_seconds=None)
        result,_=f.result(alter=alter,row_alter=row_alter)
        self.assertEqual(result['issues'],[]);self.assertFalse(result['primary_token_measurement_complete'])
        self.assertEqual(result['expected_provider_unknown_observation_count'],1)
        self.assertEqual(result['continuation']['primary_denominator'],96)
        entry=result['expected_provider_unknown_observations'][0]
        self.assertFalse(entry['actual_usage_known']);self.assertIsNone(entry['actual_task_family_tokens']);self.assertFalse(entry['technical_retry'])
        report=analysis.report_text(result)
        for text in ('Provider未知收据失败主观测 1 个','原failed与issues保留','不转technical、不重跑','实际Token/费用为N/A','native_response_unknown'):
            self.assertIn(text,report)


if __name__=='__main__':unittest.main(verbosity=2)
