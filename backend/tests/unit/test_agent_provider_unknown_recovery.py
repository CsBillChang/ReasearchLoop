"""V6合成收据回归；不构造Ledger/Settings，不读取真实实验或访问网络。"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import agent_final_completion as c
import agent_final_completion_chain as chain
import agent_final_provider_unknown_recovery as recovery
from test_agent_root_budget_recovery import anchor_fixture, write, write_sql, seal, sha, load


def fixture(base):
    f = anchor_fixture(base); root = f['root']; ledger = f['ledger']; m = f['manifest']
    m['completion_contract']['version'] = 'agent_final_completion_v5'
    m['settings'] = {'deepseek_model': 'synthetic-model'}
    parent = f['parent']; parent['pending_reservations'] = [
        {'reservation_id': 'prior-' + str(i), 'root_id': 'prior-root-' + str(i), 'status': 'unknown',
         'reserved_tokens': 10, 'reserved_nano': 20, 'actual_tokens': None, 'actual_nano': None}
        for i in range(2)]
    wire = {'model': 'synthetic-model', 'messages': [{'role': 'user', 'content': 'Synthetic question'}],
            'tools': [{'type': 'function', 'function': {'name': 'synthetic_tool'}}],
            'max_tokens': 64, 'temperature': .1, 'tool_choice': 'auto'}
    request = {k: deepcopy(wire[k]) for k in ('messages','tools','max_tokens')}
    request.update(force_text=False, terminal_only=False)
    native_text = json.dumps(request, ensure_ascii=False)
    unknown = deepcopy(ledger['reservations'][1]); unknown.update(reservation_id='physical-unknown', status='unknown',
        created_at=1061., sent_at=1062., finished_at=1065., reason=recovery.STOP_REASON,
        actual_input=None, actual_output=None, actual_tokens=None, actual_nano=None,
        usage_json='null', response_json=None,
        request_json=json.dumps({'model':wire['model'],'stream':False,'request_kind':'text','image_count':0,
            'body_sha256':hashlib.sha256(json.dumps(wire,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()}))
    ledger['reservations'].append(unknown)
    for row in ledger['reservations']:
        for key in unknown: row.setdefault(key, None)
    error = {'reason':recovery.STOP_REASON,'operation':'receipt_unknown','detected_at':1065.01,
        'root_id':'root-A','reservation_id':unknown['reservation_id'],'details':{},'limits':deepcopy(ledger['metadata']['config'])}
    error['root_occupancy'] = recovery._totals([r for r in ledger['reservations'] if r['root_id']=='root-A'],error['detected_at'])
    error['batch_occupancy'] = recovery._totals(ledger['reservations'],error['detected_at'])
    for key,src in [('tokens','tokens'),('requests','requests'),('nano','nano_yuan')]:
        error['batch_occupancy'][key] += ledger['metadata']['config']['carried_usage'][src]
    ledger['metadata'].update(first_error=error,halt_reason=recovery.STOP_REASON)
    write(root/'manifest.formal.json',m)
    batch = json.loads((root/'runtime/batch-results.json').read_text())
    batch['rows'][:2] = deepcopy(parent['retained_rows'])
    session = {'manifest_sha256':c._digest(m),'nonce':'synthetic-session'}
    write(root/'semantic-gates/session.json',session)
    for position in (2,3):
        row=batch['rows'][position]; variant=row['variant']; cell=root/'runtime/cells'/(row['case_id']+'-'+variant)
        grant=json.loads((root/f'semantic-gates/grants/{position}.json').read_text())
        grant.update(manifest_sha256=c._digest(m),continuation_sha256=c._digest(m['completion_contract']))
        write(root/f'semantic-gates/grants/{position}.json',grant)
        result=json.loads((cell/'result.json').read_text())
        result['experiment_attempt']={k:grant[k] for k in ('attempt_id','attempt_kind','position','continuation_sha256')}
        if variant=='A':
            row.update(status='failed',issues=deepcopy(recovery._ERRORS),
                native_status_counts={'succeeded':1,'unknown':1},physical_status_counts={'settled':1,'unknown':1})
            result.update(task_status='failed',ledger_halt=recovery.STOP_REASON)
            body={'answer_markdown':'Synthetic failure report.','answer_kind':'failure_report'}
            view={'task_id':'root-A','status':'failed','prompt':'synthetic prompt','result':body}
            at=lambda value:datetime.fromtimestamp(value,timezone.utc).isoformat()
            stored={'task_id':'root-A','status':'failed','prompt':'synthetic prompt','result_json':json.dumps(body),
                'parent_task_id':None,'root_task_id':'root-A','created_at':at(1030.),'updated_at':at(1066.)}
            sql_events=[{'task_id':'root-A','event_type':kind,'created_at':at(value),
                'payload_json':json.dumps({'status':'failed','result':body})}
                for kind,value in [('task_created',1030.),('task_finished',1066.)]]
            events={'task_id':'root-A','events':[{k:v for k,v in e.items() if k!='payload_json'}|
                {'payload':json.loads(e['payload_json'])} for e in sql_events]}
            success=json.loads((cell/'model-calls.json').read_text())[0]
            native=[success,{'call_id':'unknown-call','task_id':'root-A','status':'unknown','sent_state':'sent',
                'attempt_no':1,
                'request_id':None,'usage_json':'{}','response_json':json.dumps({'error':'Synthetic server disconnected.'}),
                'request_json':native_text,'request_hash':hashlib.sha256(native_text.encode()).hexdigest()}]
            for row in native:
                for key in set().union(*(set(item) for item in native)): row.setdefault(key, None)
            write(cell/'task-view.json',view);write(cell/'root-row.json',stored);write(cell/'events.json',events)
            write(cell/'model-calls.json',native);write(cell/'ledger-snapshot.json',ledger)
            (cell/'requests/physical-unknown.request.json').write_text(json.dumps(wire,ensure_ascii=False,indent=2))
            (cell/'task.sqlite3').unlink()
            write_sql(cell/'task.sqlite3',{'tasks':[stored],'model_calls':native,'task_events':sql_events})
        write(cell/'result.json',result)
    write(root/'runtime/batch-results.json',batch)
    write(root/'runtime/SYNTH-PAIR-1-A-reconciliation.json',{'issues':recovery._ERRORS,
        'ledger_halt_reason':recovery.STOP_REASON,'first_error':error,'native_not_sent':[],
        'unmatched_native_response_ids':[],'unmatched_physical_response_ids':[]})
    charge,pending=chain._charged(ledger)
    final={'limits':deepcopy(ledger['metadata']['config']),'carried_usage':parent['carried_usage'],
        'actual_known_tokens':10,'physical_sent_requests':3,'actual_known_estimated_nano_yuan':44,
        'halt_reason':recovery.STOP_REASON,'first_error':error,
        'charged_or_reserved':{'tokens':parent['carried_usage']['tokens']+charge['tokens'],
            'requests':parent['carried_usage']['requests']+charge['requests'],
            'nano':parent['carried_usage']['nano_yuan']+charge['nano_yuan']}}
    write(root/'runtime/ledger-final.json',final)
    (root/'runtime/budget.sqlite3').unlink()
    write_sql(root/'runtime/budget.sqlite3',{'metadata':[{'key':k,'value':json.dumps(v)} for k,v in ledger['metadata'].items()],
        'roots':ledger['roots'],'reservations':ledger['reservations']})
    for relative in ('audit/post-run-safety-supplement.json','audit/post-run-protection-corrected.json',
                     'audit/failures/post-audit-missing-github-readback.json'):
        (root/relative).unlink()
    scopes={'closure','containers','dataset','git_refs','old_failure','physical_budget_readonly',
        'processes','protected','seals','selected_production_materials','sources'}
    write(root/'audit/post-run-protection.json',{'status':'passed','safety_closed':True,'measurement_complete':False,
        'errors':[],'data_integrity_errors':[],'sections':dict.fromkeys(scopes,{}),
        'checks_by_scope':{'data':1,'safety':1},'checks_total':2,'checks_passed':2,
        'actions':{**dict.fromkeys(('Ledger_instantiations','business_Settings_instantiations','model_calls',
            'business_SQLite_writes','pending_releases','service_mutations'),0),'only_new_report_written':True}})
    write(root/'unused-synthetic-binding.json',{'synthetic':True})
    proof={'version':recovery.PROOF_VERSION,'status':'passed_exact_provider_unknown_observation_stop',
        'anchor_root':str(root),'parent_contract_sha256':c._digest(m['completion_contract']),
        'case_id':'SYNTH-PAIR-1','variant':'A','position':3,'root_id':'root-A',
        'original_batch_stop_reason':recovery.BATCH_STOP,'original_ledger_halt_reason':recovery.STOP_REASON,
        'original_reconciliation_issues':recovery._ERRORS,'original_records_unchanged':True,
        'original_reconciliation_passed':False,'no_retry_or_reclassification':True,
        'actions':dict.fromkeys(recovery._ACTIONS,0),'provider_receipt_evidence':recovery._provider_evidence(ledger,'root-A')}
    proof['anchor_seal_sha256']=seal(root)
    proof['bindings']={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and not p.name.startswith('SEAL')}
    path=base/'provider-proof/proof.json';write(path,proof)
    f.update(manifest=m,parent=parent,proof=proof,ref={'version':recovery.PROOF_VERSION,'path':str(path),'sha256':sha(path)})
    return f


class ProviderUnknownRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=fixture(Path(self.tmp.name).resolve());c.clear_validation_caches();self.addCleanup(c.clear_validation_caches)

    def build(self):
        def clean(root,manifest,position,ledger,load,**kwargs):
            row=deepcopy(load('runtime/batch-results.json')['rows'][position]);row.update(quality=None,retained=True)
            return row
        with patch.object(chain,'_parent',return_value=self.f['parent']),patch.object(chain,'_closure'),\
                patch.object(chain,'_ledger_limits'),patch.object(c,'_row',side_effect=clean):
            return recovery.build_provider_unknown_recovery_contract(self.f['root'],None,proof_ref=self.f['ref'])

    def change_proof(self,mutate):
        mutate(self.f['proof']);write(Path(self.f['ref']['path']),self.f['proof']);self.f['ref']['sha256']=sha(Path(self.f['ref']['path']))

    def test_failed_primary_and_three_unknown_holds_preserved_without_retry(self):
        value=self.build();row=value['retained_rows'][-1]
        self.assertEqual((value['retained_count'],len(value['new_execution_order']),len(value['technical_attempts'])),(4,92,3))
        self.assertEqual(value['minimum_complete_attempts'],99);self.assertEqual(value['retry_positions'],[])
        self.assertEqual(row['status'],'failed');self.assertEqual(row['issues'],recovery._ERRORS)
        self.assertFalse(row['physical_receipts_verified']);self.assertTrue(row['receipt_state_verified'])
        self.assertFalse(row['actual_usage_known']);self.assertIsNone(row['actual_task_family_tokens'])
        self.assertEqual(row['verified_settled_tokens_lower_bound'],5)
        self.assertEqual(row['verified_settled_nano_yuan_lower_bound'],22)
        charge,_=chain._charged(self.f['ledger'])
        self.assertEqual(value['carried_usage']['tokens'],self.f['parent']['carried_usage']['tokens']+charge['tokens'])
        self.assertEqual(value['carried_usage']['requests'],self.f['parent']['carried_usage']['requests']+charge['requests'])
        self.assertEqual(value['carried_usage']['nano_yuan'],self.f['parent']['carried_usage']['nano_yuan']+charge['nano_yuan'])
        self.assertEqual(value['carried_usage']['seconds'],self.f['parent']['carried_usage']['seconds'])
        self.assertEqual(len(value['pending_reservations']),3)
        self.assertEqual(value['pending_reservations'][:2],self.f['parent']['pending_reservations'])

    def test_proof_variant_and_self_reported_evidence_must_match_actual_sources(self):
        for field,value in [('variant','B'),('position',2),('root_id','other'),('provider_receipt_evidence',{})]:
            old=deepcopy(self.f['proof']);self.change_proof(lambda p:p.__setitem__(field,value))
            with self.subTest(field=field),self.assertRaises(PermissionError):self.build()
            self.f['proof']=old;self.change_proof(lambda p:None)

    def test_provider_evidence_rejects_unknown_release_and_post_halt_activity(self):
        for mutation in ('status','actual','reason','root','reservation','post_create','post_send','other_send','fee','bounds','duplicate','caps'):
            ledger=deepcopy(self.f['ledger']);u=ledger['reservations'][-1];error=ledger['metadata']['first_error']
            if mutation=='status':u['status']='not_sent'
            elif mutation=='actual':u['actual_tokens']=0
            elif mutation=='reason':u['reason']='request budget exhausted'
            elif mutation=='root':error['root_id']='root-B'
            elif mutation=='reservation':error['reservation_id']='other'
            elif mutation=='post_create':u['created_at']=error['detected_at']+1
            elif mutation=='post_send':u['sent_at']=error['detected_at']+1
            elif mutation=='other_send':ledger['reservations'][0].update(sent_at=1066.,finished_at=1067.)
            elif mutation=='fee':ledger['reservations'][0]['actual_nano']+=1
            elif mutation=='bounds':u['reserved_tokens']-=1
            elif mutation=='duplicate':ledger['reservations'].append(deepcopy(u))
            elif mutation=='caps':ledger['metadata']['config']['batch_tokens']=1;error['limits']=deepcopy(ledger['metadata']['config'])
            with self.subTest(mutation=mutation),self.assertRaises(PermissionError):recovery._provider_evidence(ledger,'root-A')

    def test_native_unknown_cannot_be_not_sent_success_or_zero_usage(self):
        cell=self.f['root']/'runtime/cells/SYNTH-PAIR-1-A'
        native=json.loads((cell/'model-calls.json').read_text());physical=[r for r in self.f['ledger']['reservations'] if r['root_id']=='root-A']
        for field,value in [('status','not_sent'),('sent_state','not_sent'),('attempt_no',2),('attempt_no',True),('request_id','made-up'),('usage_json','{"total_tokens":0}'),('response_json','{"error_type":"RemoteProtocolError"}')]:
            changed=deepcopy(native);changed[-1][field]=value
            with self.subTest(field=field),self.assertRaises(PermissionError):recovery._receipts('root-A',changed,physical)

    def test_actual_sql_event_payload_cannot_be_replaced_at_same_timestamp(self):
        cell=self.f['root']/'runtime/cells/SYNTH-PAIR-1-A';stored=json.loads((cell/'root-row.json').read_text())
        events=json.loads((cell/'events.json').read_text());view=json.loads((cell/'task-view.json').read_text())
        sql=[{k:v for k,v in row.items() if k!='payload'}|{'payload_json':json.dumps(row['payload'])} for row in events['events']]
        recovery._native_stop(stored,[stored],sql,view,events)
        for event in ('task_created','task_finished'):
            changed=deepcopy(events);next(r for r in changed['events'] if r['event_type']==event)['payload']={'status':'completed'}
            with self.subTest(event=event),self.assertRaises(PermissionError):recovery._native_stop(stored,[stored],sql,view,changed)

    def test_normal_passed_post_audit_has_no_R7_diagnostic_dependency(self):
        self.assertFalse((self.f['root']/'audit/post-run-safety-supplement.json').exists());self.build()
        report=json.loads((self.f['root']/'audit/post-run-protection.json').read_text())
        for field,value in [('status','failed'),('data_integrity_errors',[{'error':'real'}]),('sections',{}),('actions',{})]:
            bad=deepcopy(report);bad[field]=value
            with self.subTest(field=field),self.assertRaises(PermissionError):recovery._normal_safety(self.f['root'],lambda _:bad)

    def test_retained_helper_requires_current_authority_and_exact_original_issues(self):
        contract=self.build();context={**deepcopy(contract),'anchor_manifest':{},'latest_anchor_manifest':{}}
        manifest={'completion_contract':contract};row=contract['retained_rows'][-1];cell=Path(row['provenance']['cell_root'])
        result=json.loads((cell/'result.json').read_text());view=json.loads((cell/'task-view.json').read_text())
        with patch.object(c,'validate_completion',return_value=context):
            value=recovery.retained_provider_unknown_receipts(manifest,cell,result,view,row,context=context)
            self.assertEqual(len(value),7);self.assertFalse(value['actual_usage_known'])
            for issues in ([],None,['extra'],recovery._ERRORS+['extra']):
                bad=deepcopy(row)
                if issues is None:bad.pop('issues')
                else:bad['issues']=issues
                with self.subTest(issues=issues),self.assertRaises(PermissionError):recovery.retained_provider_unknown_receipts(manifest,cell,result,view,bad)
            with self.assertRaises(PermissionError):recovery.retained_provider_unknown_receipts(manifest,cell,result,view,row,context={'untrusted':True})

    def test_missing_even_unused_sealed_binding_is_rejected(self):
        self.change_proof(lambda p:p['bindings'].pop('unused-synthetic-binding.json'))
        with self.assertRaisesRegex(PermissionError,'EXACT_COMPLETE_SEALED_BINDINGS_REQUIRED'):self.build()

    def test_warm_proof_and_native_dependency_changes_are_not_cached_as_passed(self):
        contract=self.build();context={**deepcopy(contract),'anchor_manifest':{},'latest_anchor_manifest':{}}
        row=contract['retained_rows'][-1];cell=Path(row['provenance']['cell_root'])
        result=json.loads((cell/'result.json').read_text());view=json.loads((cell/'task-view.json').read_text())
        write(cell/'model-calls.json',[])
        with patch.object(c,'validate_completion',return_value=context),self.assertRaisesRegex(PermissionError,'DEPENDENCY_CHANGED'):
            recovery.retained_provider_unknown_receipts({'completion_contract':contract},cell,result,view,row)
        with self.assertRaises(PermissionError):self.build()

    def test_native_wire_and_archive_hashes_are_distinct_producer_identities(self):
        cell=self.f['root']/'runtime/cells/SYNTH-PAIR-1-A';native=json.loads((cell/'model-calls.json').read_text())
        physical=[r for r in self.f['ledger']['reservations'] if r['root_id']=='root-A']
        recovery._unknown_request(self.f['root'],self.f['manifest'],native,physical,self.f['proof'],load(self.f['root']))
        wire_file=cell/'requests/physical-unknown.request.json';metadata=json.loads(physical[-1]['request_json'])
        for wrong in (sha(wire_file),metadata['body_sha256']):
            changed=deepcopy(native);changed[-1]['request_hash']=wrong
            with self.subTest(wrong=wrong),self.assertRaisesRegex(PermissionError,'PRODUCER_SPECIFIC_REQUEST_HASH_CHANGED'):
                recovery._unknown_request(self.f['root'],self.f['manifest'],changed,physical,self.f['proof'],load(self.f['root']))

    def test_actual_task_family_must_be_connected_and_owned(self):
        cell=self.f['root']/'runtime/cells/SYNTH-PAIR-1-A';stored=json.loads((cell/'root-row.json').read_text())
        events=json.loads((cell/'events.json').read_text());view=json.loads((cell/'task-view.json').read_text())
        sql=[{k:v for k,v in r.items() if k!='payload'}|{'payload_json':json.dumps(r['payload'])} for r in events['events']]
        bad=deepcopy(stored);bad['root_task_id']='foreign'
        with self.assertRaises(PermissionError):recovery._native_stop(stored,[bad],sql,view,events)
        children=[{**deepcopy(stored),'task_id':key,'parent_task_id':parent,'status':'partial'} for key,parent in [('child-a','child-b'),('child-b','child-a')]]
        with self.assertRaisesRegex(PermissionError,'CYCLE_OR_DISCONNECTED'):
            recovery._native_stop(stored,[stored,*children],sql,view,events)

    def test_current_manifest_caps_and_ledger_driver_bytes_cannot_change(self):
        import agent_final_observation_recovery as observation
        import agent_final_pdf_acquisition as pdf
        contract=self.build();root=Path(self.tmp.name).resolve()/'next'
        original=deepcopy(self.f['parent']['anchor_manifest']);original['ledger_limits']={'root_tokens':123456}
        base={**contract,'anchor_manifest':original,'latest_anchor_manifest':self.f['manifest'],'old_cases':{},'old_materials':{}}
        manifest={'completion_contract':contract,'experiment_kind':'prompt_bundle_v1','evaluation_policy':c.EVALUATION_POLICY,
            'execution_order':contract['full_execution_order'],'source_acquisition_contract':pdf.SOURCE_ACQUISITION_CONTRACT,
            'ledger_limits':{'root_tokens':123456,'carried_usage':contract['carried_usage'],'enforce_batch_seconds':False},
            'experiment_root':str(root),'authorization_path':str(root/'new-auth.json'),'real_execution_released':False,'live_qualification_released':False,
            'source_roots':{v:str(root/'snapshots'/v) for v in ('A','B')},
            'prompt_bundle_source_contract':deepcopy(self.f['manifest']['prompt_bundle_source_contract']),
            'carried_budget_source':{'version':recovery.VERSION,'anchor_root':contract['anchor_root'],
                'ledger_path':str(Path(contract['anchor_root'])/'runtime/budget.sqlite3'),
                'ledger_sha256':contract['anchors']['runtime/budget.sqlite3'],'seal_sha256':contract['anchor_seals'][contract['anchor_root']]}}
        for key,rel in {'runtime_root':'runtime','semantic_gate_root':'semantic-gates','prepared_input_root':'inputs',
            'cases_path':'inputs/cases.json','model_manifest_path':'inputs/model-manifest.json'}.items():manifest[key]=str(root/rel)
        with patch.object(recovery,'build_provider_unknown_recovery_contract',return_value=contract),patch.object(recovery,'_anchor',return_value=base),\
                patch.object(observation,'validate_observation_policy',return_value=True),patch.object(pdf,'validate_source_acquisition_contract',return_value=True),\
                patch.object(c._helpers(),'_check_inputs'),patch.object(c,'_sources'):
            self.assertEqual(recovery.validate_provider_unknown_recovery(manifest)['version'],recovery.VERSION)
            bad=deepcopy(manifest);bad['ledger_limits']['root_tokens']+=1
            with self.assertRaisesRegex(PermissionError,'ORIGINAL_FINANCIAL_AND_ROOT_CAPS_REQUIRED'):recovery.validate_provider_unknown_recovery(bad)
            for name in ('harness_ab_budget.py','run_read_paper_batch.py'):
                bad=deepcopy(manifest);bad['prompt_bundle_source_contract']['variants']['A']['source_files']['backend/scripts/'+name]='changed'
                with self.subTest(name=name),self.assertRaisesRegex(PermissionError,'LEDGER_OR_BATCH_DRIVER_CHANGED'):recovery.validate_provider_unknown_recovery(bad)


if __name__=='__main__':unittest.main()
