"""V3通用叶追加的只读合成回归；父V2和原评分复算另有真实锚检查。"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'scripts'),str(Path(__file__).resolve().parent)]
import agent_final_completion as completion
import agent_final_completion_chain as chain
from harness_ab_budget import Ledger
from test_agent_final_completion import synthetic_completion_fixture
from test_agent_final_continuation import write_json,digest


def sealed(root):
    entries={str(p.relative_to(root)):{'sha256':digest(p),'bytes':p.stat().st_size}
             for p in root.rglob('*') if p.is_file() and p.name not in {'SEAL.json','SEAL.sha256','SEAL-READBACK.json'}}
    write_json(root/'SEAL.json',{'files':entries})
    value=digest(root/'SEAL.json');(root/'SEAL.sha256').write_text(value+'  SEAL.json\n')
    write_json(root/'SEAL-READBACK.json',{'status':'passed','seal_sha256':value})


def make_leaf(directory):
    f=synthetic_completion_fixture(directory);m=f['manifest'];parent=f['base'];root=f['new_root']
    parent['clock']=deepcopy(m['completion_contract']['clock'])
    for i,item in enumerate(parent['technical_attempts']):item['root_id']='old-technical-'+str(i)
    m['real_execution_released']=True;m['live_qualification_released']=False
    write_json(root/'manifest.formal.json',m)
    ledger=Ledger(root/'runtime/budget.sqlite3',clock=lambda:2000,start_time=parent['original_clock']['round_started_at'],
                  **m['ledger_limits'])
    rows=[]
    for position in (11,12):
        case,variant=m['execution_order'][position];identity='leaf-root-'+str(position)
        ledger.begin_root(identity,case,variant)
        reservation=ledger.reserve(identity,max_output_tokens=10,input_upper_bound=10)
        ledger.mark_sent(reservation['reservation_id'])
        usage={'prompt_tokens':3,'completion_tokens':2,'total_tokens':5,'prompt_cache_hit_tokens':0}
        ledger.settle(reservation['reservation_id'],usage,response_metadata={'response_id':'provider-'+str(position)})
        grant={'position':position,'case_id':case,'variant':variant,'root_id':identity,
               'manifest_sha256':completion._digest(m),'continuation_sha256':completion._digest(m['completion_contract']),
               'attempt_kind':'technical_retry' if position==11 else 'first_attempt','attempt_id':str(position)*32}
        write_json(root/'semantic-gates/grants'/f'{position}.json',grant)
        cell=root/'runtime/cells'/(case+'-'+variant)
        write_json(cell/'result.json',{'case_id':case,'variant':variant,'root_id':identity,
            'status':'executed' if position==11 else 'failed',
            'experiment_attempt':{k:grant[k] for k in ('attempt_id','attempt_kind','position','continuation_sha256')},
            'isolation':{'denials':[] if position==11 else [{'phase':'execution','operation':'network'}]}})
        write_json(cell/'task-view.json',{'task_id':identity,'status':'completed' if position==11 else 'partial'})
        write_json(cell/'model-calls.json',[{'status':'succeeded','sent_state':'sent','request_id':'provider-'+str(position),
            'response_json':json.dumps({'id':'provider-'+str(position)}),'usage_json':json.dumps(usage)}])
        write_json(cell/'actions.json',{})
        db=sqlite3.connect(cell/'task.sqlite3');db.execute('CREATE TABLE tasks(task_id TEXT,status TEXT)')
        db.execute('INSERT INTO tasks VALUES (?,?)',(identity,'completed' if position==11 else 'partial'));db.commit();db.close()
        rows.append({'case_id':case,'variant':variant,'root_id':identity,'status':'completed' if position==11 else 'partial',
            'issues':[] if position==11 else ['cell_execution_not_complete'],'returncode':0,'original_position':position,'retained':False})
    ledger.halt('cell_receipt_or_isolation_gate',operation='continuation_controller',root_id=rows[-1]['root_id'])
    decision={'status':'passed','stop_reasons':[],'quality':{'A':'partial_usable','B':'partial_usable'}}
    write_json(root/'semantic-gates'/rows[0]['case_id']/'decision.json',decision)
    write_json(root/'runtime/batch-results.json',{'status':'stopped','stop_reason':'cell_receipt_or_isolation_gate',
        'rows':parent['retained_rows']+rows,'retained_observations':11,'new_roots_planned':85,'new_attempted_count':2,
        'not_created':m['execution_order'][13:],'limits':m['ledger_limits']})
    write_json(root/'formal-execution-auth.json',{'approved':False,'allowed_modes':[],'status':'closed_no_resume',
        'manifest_sha256':completion._digest(m)})
    write_json(root/'formal-preflight/authorization-closed.json',{'status':'closed','no_resume_or_resend':True,
        'round_started_at':parent['original_clock']['round_started_at'],'round_absolute_deadline':None,'policy':'financial_only'})
    write_json(root/'formal-preflight/vector-close.json',{'status':'passed','identity':{'running':False}})
    write_json(root/'formal-control/controller-result.json',{'status':'exited','returncode':0,'batch_group_cleaned':True,
        'cleanup_all_passed':True,'policy':'financial_only','global_deadline':None,
        'original_T0':parent['original_clock']['round_started_at'],'original_clock_reset':False,
        'wrapper_finished_before_cleanup_deadline':True,'cleanup_started_at':2000,'model_processes_stopped_at':2001,
        'finished_at':2002,'wrapper_finalized_at':2003,'cleanup_deadline_after_stop':2300,
        'cleanup_steps_deadline_after_stop':2290,'final_process_groups':[{'alive_group_members':[]}]})
    write_json(root/'audit/post-run-protection.json',{'status':'passed','safety_closed':True,'errors':[]})
    sealed(root)
    return f,decision


class CompletionChainTests(unittest.TestCase):
    def setUp(self):
        chain._PARSED_ANCHORS.clear();self.addCleanup(chain._PARSED_ANCHORS.clear)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.fixture,self.decision=make_leaf(Path(self.temp.name).resolve())
        self.root=self.fixture['new_root'];self.parent=self.fixture['base']
        self.parent_patch=patch.object(chain,'_parent',return_value=self.parent);self.parent_mock=self.parent_patch.start();self.addCleanup(self.parent_patch.stop)
        self.grade_patch=patch.object(completion,'_old_grade',return_value=self.decision);self.grade_patch.start();self.addCleanup(self.grade_patch.stop)

    def build(self):
        return chain.build_completion_chain_contract(self.root,self.fixture['timing'])

    def test_prefix12_three_technical_and_84_suffix_fold_cost_once(self):
        value=self.build()
        self.assertEqual([value['retained_count'],len(value['technical_attempts']),len(value['new_execution_order']),value['minimum_complete_attempts']],
                         [12,3,84,99])
        self.assertEqual(value['new_execution_order'],value['full_execution_order'][12:])
        self.assertEqual(value['carried_usage']['tokens'],self.parent['carried_usage']['tokens']+10)
        self.assertEqual(value['carried_usage']['requests'],self.parent['carried_usage']['requests']+2)
        self.assertEqual(value['carried_usage']['seconds'],self.parent['carried_usage']['seconds'])
        self.assertEqual(value['pending_reservations'],self.parent['pending_reservations'])
        self.assertEqual(value['closed_ledger_roots'],self.parent['closed_ledger_roots']+[str(self.root)])
        self.assertEqual(value['retained_rows'][7]['status'],'partial')
        self.assertEqual(value['technical_attempts'][-1]['result_status'],'failed')
        self.assertTrue(value['technical_attempts'][-1]['physical_receipts_verified'])

    def test_cross_old_artifact_gets_separate_new_score_owner(self):
        value=self.build();old_b,new_a=value['retained_rows'][10:12]
        self.assertEqual(old_b['source_manifest_path'],self.parent['retained_rows'][10]['source_manifest_path'])
        self.assertEqual(old_b['score_manifest_path'],str(self.root/'manifest.formal.json'))
        self.assertEqual(new_a['source_manifest_path'],new_a['score_manifest_path'])
        self.assertEqual(old_b['quality'],'partial_usable')
        self.assertEqual(value['retained_rows'][:10],self.parent['retained_rows'][:10])
        self.assertIsNone(self.parent['retained_rows'][10]['quality'])

    def test_hot_cache_skips_SQL_but_rechecks_parent_seal_and_closure(self):
        with patch.object(completion,'_ledger',wraps=completion._ledger) as ledger, \
             patch.object(completion,'_sealed',wraps=completion._sealed) as seals, \
             patch.object(chain,'_closure',wraps=chain._closure) as closure:
            first=self.build();second=self.build()
        self.assertEqual(first,second);self.assertEqual(ledger.call_count,1)
        self.assertEqual(seals.call_count,2);self.assertEqual(closure.call_count,2)
        self.assertEqual(self.parent_mock.call_count,2)

    def test_cached_context_returns_deepcopies_and_never_accepts_parent_identity_drift(self):
        value=self.build();expected=deepcopy(value)
        value['retained_rows'][7]['quality']='full';value['pending_reservations'].clear()
        self.assertEqual(self.build(),expected)
        with patch.object(completion,'_ledger',wraps=completion._ledger) as ledger:
            self.parent['carried_usage']['tokens']+=1
            with self.assertRaises(PermissionError):self.build()
        self.assertEqual(ledger.call_count,1)
        self.assertEqual(len(chain._PARSED_ANCHORS),1)

    def test_warm_cache_still_rejects_seal_leaf_and_parent_proof_changes(self):
        self.build()
        path=self.root/'formal-execution-auth.json';before=path.read_bytes()
        try:
            path.write_bytes(before+b' ')
            with self.assertRaises(PermissionError):self.build()
        finally:path.write_bytes(before)
        with patch.object(chain,'_parent',side_effect=PermissionError('ancestor seal changed')):
            with self.assertRaises(PermissionError):self.build()
        self.assertEqual(len(chain._PARSED_ANCHORS),1)

    def test_external_protection_binding_rechecked_before_warm_cache_return(self):
        outside=self.root.parent/'external-proof.json';outside.write_text('{}');expected=digest(outside)
        original=chain._protection
        def protection(root,load):
            original(root,load)
            chain._require(digest(outside)==expected,'EXTERNAL_PROOF_CHANGED')
        with patch.object(chain,'_protection',side_effect=protection):
            self.build();outside.write_text('{"changed":true}')
            with patch.object(completion,'_ledger',side_effect=AssertionError('historical SQL must not be reached')):
                with self.assertRaises(PermissionError):self.build()
        self.assertEqual(len(chain._PARSED_ANCHORS),1)

    def test_incomplete_validation_never_writes_cache(self):
        with patch.object(completion,'_receipts',side_effect=PermissionError('receipt mismatch')):
            with self.assertRaises(PermissionError):self.build()
        self.assertEqual(chain._PARSED_ANCHORS,{})

    def test_current_source_auth_and_declared_prompt_checks_remain_live_after_cache_warmup(self):
        from agent_final_pdf_acquisition import SOURCE_ACQUISITION_CONTRACT
        from harness_ab_pair_gate import manifest_identity
        from prompt_bundle_ab import validate_activation
        contract=self.build();m=deepcopy(self.fixture['manifest']);owned=self.root.parent/'live-candidate'
        shutil.copytree(self.root/'inputs',owned/'inputs')
        for v in ('A','B'):shutil.copytree(self.root/'snapshots'/v,owned/'snapshots'/v)
        def rebind(value,key=''):
            if isinstance(value,dict):return {k:rebind(v,k) for k,v in value.items()}
            if isinstance(value,list):return [rebind(v,key) for v in value]
            if key in {'copy','pdf_path','parsed_path'} and isinstance(value,str) and value.startswith(str(self.root)+'/'):
                return str(owned/Path(value).relative_to(self.root))
            return value
        for path in (owned/'inputs').glob('*.json'):
            original=json.loads(path.read_text());changed=rebind(original)
            if changed!=original:write_json(path,changed)
        m.update(completion_contract=contract,experiment_root=str(owned),runtime_root=str(owned/'runtime'),
            semantic_gate_root=str(owned/'semantic-gates'),prepared_input_root=str(owned/'inputs'),cases_path=str(owned/'inputs/cases.json'),
            model_manifest_path=str(owned/'inputs/model-manifest.json'),authorization_path=str(owned/'auth.json'),
            source_roots={v:str(owned/'snapshots'/v) for v in ('A','B')},real_execution_released=True,
            live_qualification_released=False,evaluation_policy=deepcopy(completion.EVALUATION_POLICY),
            source_acquisition_contract=deepcopy(SOURCE_ACQUISITION_CONTRACT),batch_id='synthetic-live-cache-auth')
        m['ledger_limits']['carried_usage']=deepcopy(contract['carried_usage'])
        m['carried_budget_source']={'version':chain.VERSION,'anchor_root':str(self.root),
            'ledger_path':str(self.root/'runtime/budget.sqlite3'),'ledger_sha256':contract['anchors']['runtime/budget.sqlite3'],
            'seal_sha256':contract['anchor_seals'][str(self.root)]}
        auth={'approved':True,'human_approved':True,'policy':'financial_only','expires_at':None,
            'round_started_at':contract['clock']['round_started_at'],'round_absolute_deadline':None,
            'experiment_kind':m['experiment_kind'],'batch_id':m['batch_id'],'manifest_sha256':manifest_identity(m),
            'ledger_limits':m['ledger_limits'],'execution_order':m['execution_order'],'allowed_modes':['execute']}
        path=Path(m['authorization_path']);write_json(path,auth)
        self.assertEqual(validate_activation(m,'execute')['approved'],True)
        auth['approved']=False;write_json(path,auth)
        with self.assertRaises(PermissionError):validate_activation(m,'execute')
        auth['approved']=True;write_json(path,auth)
        source=owned/'snapshots/A/backend/research_loop/__init__.py';original=source.read_bytes()
        try:
            source.write_bytes(original+b'# changed current source\n')
            with self.assertRaises(PermissionError):validate_activation(m,'execute')
        finally:source.write_bytes(original)
        self.assertTrue(validate_activation(m,'execute')['approved'])
        m['loaded_prompt_contract']['A']['assembled_prompt_sha256']['supervisor']='changed'
        with self.assertRaises(PermissionError):completion.validate_completion(m)

    def test_unsealed_missing_changed_or_extra_anchor_files_rejected(self):
        self.build()
        for relative in ('SEAL.json','runtime/budget.sqlite3','runtime/cells/'+self.fixture['order'][11][0]+'-'+self.fixture['order'][11][1]+'/result.json'):
            path=self.root/relative;before=path.read_bytes()
            try:
                path.write_bytes(before+b' ')
                with self.subTest(relative=relative),self.assertRaises(PermissionError):self.build()
            finally:path.write_bytes(before)
        extra=self.root/'unsealed.txt';extra.write_text('extra')
        with self.assertRaises(PermissionError):self.build()
        extra.unlink()
        seal=self.root/'SEAL.json';before=seal.read_bytes();seal.unlink()
        try:
            with self.assertRaisesRegex(PermissionError,'ANCHOR_PENDING_SEAL'):self.build()
        finally:seal.write_bytes(before)

    def test_stage_halt_root_grade_receipts_order_and_carry_not_rewritten(self):
        self.build()
        for change in ('halt','old-grade','native-id','order','carry','controller','auth','denial'):
            if change=='old-grade':
                previous=self.parent['retained_rows'][10]['quality'];self.parent['retained_rows'][10]['quality']='full'
                try:
                    with self.assertRaises(PermissionError):self.build()
                finally:self.parent['retained_rows'][10]['quality']=previous
                continue
            if change in ('halt','carry'):
                path=self.root/'runtime/budget.sqlite3';before=path.read_bytes()
                db=sqlite3.connect(path)
                if change=='halt':db.execute('UPDATE metadata SET value=? WHERE key=?',(json.dumps(None),'first_error'))
                else:
                    config=json.loads(db.execute('SELECT value FROM metadata WHERE key="config"').fetchone()[0]);config['carried_usage']['tokens']+=1
                    db.execute('UPDATE metadata SET value=? WHERE key=?',(json.dumps(config),'config'))
                db.commit();db.close()
            else:
                if change=='native-id':path=next((self.root/'runtime/cells').glob('*/model-calls.json'))
                elif change=='order':path=self.root/'runtime/batch-results.json'
                elif change=='controller':path=self.root/'formal-control/controller-result.json'
                elif change=='auth':path=self.root/'formal-execution-auth.json'
                else:path=self.root/'runtime/cells'/(self.fixture['order'][11][0]+'-'+self.fixture['order'][11][1])/'result.json'
                before=path.read_bytes();value=json.loads(before)
                if change=='native-id':value[0]['request_id']='foreign'
                elif change=='order':value['rows'][-1]['original_position']=13
                elif change=='controller':value['batch_group_cleaned']=False
                elif change=='auth':value['approved']=True
                else:value['isolation']['denials']=[{'phase':'execution'}]
                write_json(path,value)
            sealed(self.root)
            try:
                with self.subTest(change=change),self.assertRaises(PermissionError):self.build()
            finally:path.write_bytes(before);sealed(self.root)

    def test_v3_candidate_explicit_dispatch_caps_owner_and_pending_release(self):
        from agent_final_pdf_acquisition import SOURCE_ACQUISITION_CONTRACT
        contract=self.build();candidate=deepcopy(self.fixture['manifest']);owned=self.root.parent/'third-candidate'
        candidate.update(completion_contract=contract,experiment_root=str(owned),runtime_root=str(owned/'runtime'),
            semantic_gate_root=str(owned/'semantic-gates'),prepared_input_root=str(owned/'inputs'),cases_path=str(owned/'inputs/cases.json'),
            model_manifest_path=str(owned/'inputs/model-manifest.json'),authorization_path=str(owned/'auth.json'),
            source_roots={v:str(owned/'snapshots'/v) for v in ('A','B')},real_execution_released=False,
            live_qualification_released=False,evaluation_policy=deepcopy(completion.EVALUATION_POLICY))
        candidate['source_acquisition_contract']=deepcopy(SOURCE_ACQUISITION_CONTRACT)
        candidate['ledger_limits']['carried_usage']=deepcopy(contract['carried_usage'])
        candidate['carried_budget_source']={'version':chain.VERSION,'anchor_root':str(self.root),
            'ledger_path':str(self.root/'runtime/budget.sqlite3'),'ledger_sha256':contract['anchors']['runtime/budget.sqlite3'],
            'seal_sha256':contract['anchor_seals'][str(self.root)]}
        with patch.object(completion,'_sources'),patch.object(completion._helpers(),'_check_inputs'):
            value=completion.validate_completion(candidate)
            self.assertEqual(value['retained_count'],12)
            for change in ('version','double','cap','score','prefix','pending-release','acquisition','boolean-acquisition'):
                m=deepcopy(candidate)
                if change=='version':m['completion_contract']['version']='agent_final_completion_v999'
                elif change=='double':m['continuation_contract']={}
                elif change=='cap':m['ledger_limits']['batch_tokens']+=1
                elif change=='score':m['completion_contract']['retained_rows'][10]['quality']='full'
                elif change=='prefix':m['execution_order']=m['execution_order'][12:]
                elif change=='acquisition':del m['source_acquisition_contract']
                elif change=='boolean-acquisition':m['source_acquisition_contract']['native_authorization_unchanged']=1
                else:
                    m['completion_contract']=chain.build_completion_chain_contract(self.root,None);m['real_execution_released']=True
                with self.subTest(change=change),self.assertRaises(PermissionError):completion.validate_completion(m)


@unittest.skipUnless(os.environ.get('AGENT_FINAL_COMPLETION_CHAIN_ANCHOR_ROOT'),'explicit sealed chain anchor not supplied')
class CompletionRealChainReadonlyTests(unittest.TestCase):
    def test_sealed_real_R3_derives_12_3_84_and_preserves_mixed_grade_owner(self):
        root=Path(os.environ['AGENT_FINAL_COMPLETION_CHAIN_ANCHOR_ROOT'])
        before={n:digest(root/n) for n in ('SEAL.json','runtime/budget.sqlite3','formal-execution-auth.json')}
        x=chain.build_completion_chain_contract(root,None)
        self.assertEqual([x['retained_count'],len(x['technical_attempts']),len(x['new_execution_order']),x['minimum_complete_attempts']],[12,3,84,99])
        self.assertEqual(x['retained_rows'][10]['score_manifest_path'],str(root/'manifest.formal.json'))
        self.assertNotEqual(x['retained_rows'][10]['source_manifest_path'],x['retained_rows'][10]['score_manifest_path'])
        self.assertEqual(len(x['pending_reservations']),2)
        self.assertEqual(before,{n:digest(root/n) for n in before})

    @unittest.skipUnless(os.environ.get('AGENT_FINAL_COMPLETION_GUARD_MANIFEST'),'explicit frozen live-shape manifest not supplied')
    def test_real_prewarmed_V3_has_zero_old_SQL_inside_CellBoundary_and_miss_stays_closed(self):
        code=r'''
from pathlib import Path
import hashlib,json,sys,tempfile
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
from agent_final_continuation import validate_continuation
from agent_final_runtime import CellBoundary
import agent_final_completion_chain as chain
m=json.loads(Path(sys.argv[2]).read_text())
ledger=Path(m['completion_contract']['anchor_root'])/'runtime/budget.sqlite3'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=digest(ledger);state={'phase':'warmup'};sql=[];network=[]
def observer(event,args):
    if event=='sqlite3.connect':sql.append({'phase':state['phase'],'path':str(args[0])})
    elif event in {'socket.connect','socket.getaddrinfo','socket.bind'}:
        network.append(event);raise AssertionError('network forbidden')
sys.addaudithook(observer)
value=validate_continuation(m)
assert chain._PARSED_ANCHORS
with tempfile.TemporaryDirectory() as temp:
    runtime=Path(temp).resolve();cell=runtime/'current-cell';cell.mkdir()
    guard=CellBoundary(cell,runtime);guard.enabled=False;sys.addaudithook(guard.audit)
    guard.phase='live_guard_replay';state['phase']='live_guard_replay';guard.enabled=True
    try:
        assert validate_continuation(m)==value
    finally:guard.enabled=False
    hot=[x for x in sql if x['phase']=='live_guard_replay']
    assert hot==[] and guard.denials==[]
    chain._PARSED_ANCHORS.clear();state['phase']='unprimed_guard_replay';guard.enabled=True
    rejected=None
    try:validate_continuation(m)
    except RuntimeError as exc:rejected=type(exc).__name__
    finally:guard.enabled=False
    assert rejected=='IsolationError' and not chain._PARSED_ANCHORS
    assert guard.denials[-1]['operation']=='sqlite_not_current_cell_or_shared_ledger'
assert not network and digest(ledger)==before
print(json.dumps({'status':'passed','warmup_sqlite_attempts':sum(x['phase']=='warmup' for x in sql),
    'hot_guard_sqlite_attempts':len(hot),'hot_guard_denials':0,'unprimed_denied':True,
    'network_calls':0,'model_calls':0,'Ledger_instantiations':0,'anchor_ledger_unchanged':True}))
'''
        scripts=Path(__file__).resolve().parents[2]/'scripts'
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(scripts),
            os.environ['AGENT_FINAL_COMPLETION_GUARD_MANIFEST']],capture_output=True,text=True,timeout=300)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        value=json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(value['hot_guard_sqlite_attempts'],0);self.assertTrue(value['unprimed_denied'])
        self.assertEqual(value['network_calls'],0)


if __name__=='__main__':unittest.main()
