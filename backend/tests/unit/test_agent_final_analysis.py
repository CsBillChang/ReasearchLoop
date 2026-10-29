"""固定主矩阵、续跑attempt与金融累计的合成离线回归。"""
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


class AnalysisTests(unittest.TestCase):
    """只使用合成材料和临时本地工件，不访问真实服务。"""

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.network=[]
        for name in ('connect','connect_ex'):
            guard=patch.object(socket.socket,name,side_effect=AssertionError('real network forbidden'))
            self.network.append(guard.start());self.addCleanup(guard.stop)

    def tearDown(self):
        for guard in self.network:self.assertEqual(guard.call_count,0)

    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value));return p

    @staticmethod
    def order():
        return [[f'AFE-QA-{i:02d}-R{r}',v] for r in (1,2) for i in range(1,25) for v in ('A','B')]

    @staticmethod
    def history():return {'tokens':7,'requests':3,'nano_yuan':11,'seconds':5}

    def test_history_is_derived_from_hash_anchored_sources(self):
        prior={'tokens':2,'requests':1,'nano_yuan':4,'seconds':2}
        ledger={'reservations':[{'status':'settled','actual_tokens':5,'actual_nano':7}],
            'actual_known_tokens':5,'actual_known_estimated_nano_yuan':7,'physical_sent_requests':1,
            'carried_usage':prior,'charged_or_reserved':{'tokens':7,'requests':2,'nano':11}}
        batch={'actual_known_tokens':5,'actual_known_estimated_nano_yuan':7,'physical_requests':1,'execution_seconds':'2.2'}
        manifest={'ledger_limits':{'carried_usage':{'tokens':7,'requests':2,'nano_yuan':11,'seconds':5}},'carried_budget_source':{}}
        for name,obj in [('ledger_final',ledger),('batch_results',batch)]:
            p=self.write(name+'.json',obj);manifest['carried_budget_source'].update({name+'_path':str(p),name+'_sha256':analysis.sha(p.read_bytes())})
        self.assertEqual(analysis.historical_usage(analysis.Reader(),manifest),manifest['ledger_limits']['carried_usage'])
        manifest['ledger_limits']['carried_usage']['tokens']+=1
        with self.assertRaisesRegex(ValueError,'manifest historical'):analysis.historical_usage(analysis.Reader(),manifest)
        self.write('ledger_final.json',{})
        with self.assertRaisesRegex(ValueError,'hash mismatch'):analysis.historical_usage(analysis.Reader(),manifest)

    def test_nonzero_history_cannot_be_unanchored(self):
        with self.assertRaisesRegex(ValueError,'anchored source'):
            analysis.historical_usage(analysis.Reader(),{'ledger_limits':{'carried_usage':self.history()}})
        self.assertEqual(analysis.historical_usage(analysis.Reader(),{'ledger_limits':{}}),dict.fromkeys(self.history(),0))

    def test_usage_counts_cached_and_reasoning_only_once(self):
        usage={'prompt_tokens':9,'completion_tokens':3,'total_tokens':12,'prompt_cache_hit_tokens':4,
               'prompt_cache_miss_tokens':5,'completion_tokens_details':{'reasoning_tokens':2}}
        parsed=analysis.parsed_usage(usage)
        self.assertEqual(parsed,{'input':9,'output':3,'tokens':12,'cached_input':4,'uncached_input':5})
        self.assertIsNone(analysis.parsed_usage({**usage,'total_tokens':14}))
        self.assertIsNone(analysis.parsed_usage({**usage,'prompt_tokens_details':'invalid'}))
        config={'cached_input_nano_per_token':1,'input_nano_per_token':3,'output_nano_per_token':4}
        self.assertEqual(analysis.rate_cost(parsed,config),31)
        self.assertIsNone(analysis.rate_cost(parsed,{}))

    def test_unique_events_and_full_family_define_latency(self):
        task={'task_id':'root','status':'completed','created_at':100,'updated_at':999,
              'result_json':json.dumps({'answer_markdown':'synthetic answer'})}
        events=[{'task_id':'root','event_type':'task_created','created_at':100},
                {'task_id':'root','event_type':'task_finished','created_at':110}]
        measured=analysis.latency(task,events,[task],{},{});self.assertEqual(measured['seconds'],10)
        self.assertTrue(measured['delivery_ready_observed'])
        no_terminal=analysis.latency(task,events[:1],[task],{},{});self.assertIsNone(no_terminal['seconds'])
        self.assertFalse(no_terminal['delivery_ready_observed'])
        duplicate=analysis.latency(task,events+[events[0]],[task],{},{});self.assertIsNone(duplicate['seconds'])
        child={'task_id':'child','status':'running','updated_at':111}
        family=analysis.latency(task,events,[task,child],{},{});self.assertFalse(family['delivery_ready_observed'])
        self.assertEqual(family['seconds'],10)

    def test_missing_ledger_is_null_and_unknown_reservations_are_held(self):
        self.assertIsNone(analysis.ledger_totals(None)['known']['tokens'])
        ledger={'reservations':[{'status':'unknown','sent_at':1,'reserved_tokens':8,'reserved_nano':9}]}
        totals=analysis.ledger_totals(ledger)
        self.assertEqual(totals['known']['tokens'],0)
        self.assertEqual(totals['held']['tokens'],8)
        self.assertEqual(len(totals['pending']),1)

    @staticmethod
    def row(case,v,technical=False):
        base,r=case.rsplit('-R',1);tokens=3 if v=='A' else 2
        return {'case_id':base,'round':int(r),'expanded_case_id':case,'variant':v,
            'root_id':('old-failed-' if technical else '')+case+'-'+v,'category':'synthetic',
            'status':'technical_failed_before_terminal' if technical else 'completed',
            'actual_task_family_tokens':None if technical else tokens,
            'verified_settled_tokens_lower_bound':tokens,
            'verified_settled_nano_yuan_lower_bound':tokens+1,
            'usage_reconciled':not technical,'issues':['native_unknown'] if technical else [],
            'score':{'whole_answer_quality':None if technical else 'full'},
            'end_to_end':{'seconds':None if technical else 10 if v=='A' else 8,
                          'delivery_ready_observed':not technical},'pending_reservations':[]}

    @staticmethod
    def ledger(order,carry,started=100):
        rows=[];roots=[]
        for i,(c,v) in enumerate(order):
            t=3 if v=='A' else 2
            rows.append({'reservation_id':'res-'+str(i),'root_id':c+'-'+v,'status':'settled','sent_at':100,
                         'actual_input':t-1,'actual_output':1,'actual_tokens':t,'actual_nano':t+1})
            roots.append({'root_id':c+'-'+v,'case_id':c,'variant':v})
        return {'metadata':{'config':{'carried_usage':carry},'started_at':started},'roots':roots,'reservations':rows}

    def fixture(self,continuation=True):
        order=self.order();old=self.root/'old';new=self.root/'new';history=self.history()
        anchor={'experiment_root':str(old),'runtime_root':str(old/'runtime'),
            'execution_order':order,'ledger_limits':{'carried_usage':history,'batch_tokens':1000,
            'batch_requests':200,'batch_yuan':'1','batch_seconds':1000}}
        old_ledger=self.ledger(order[:5],history)
        old_totals=analysis.ledger_totals(old_ledger)['known']
        carry={k:history[k]+old_totals[k] for k in old_totals}|{'seconds':history['seconds']}
        manifest={**anchor,'experiment_root':str(new),'runtime_root':str(new/'runtime')}
        manifest['ledger_limits']={**anchor['ledger_limits'],'carried_usage':carry if continuation else history}
        manifest['execution_order']=order[4:] if continuation else order
        if continuation:manifest['continuation_contract']={'synthetic':True}
        retained=[{'case_id':c,'variant':v,'quality':'full','provenance':{'anchor_root':str(old)}} for c,v in order[:4]]
        failed={'case_id':order[4][0],'variant':order[4][1],'status':'partial','provenance':{'anchor_root':str(old)}}
        ctx={'anchor_manifest':anchor,'anchor_root':str(old),'full_execution_order':order,'retained_rows':retained,
             'failed_attempt':failed,'new_execution_order':order[4:],'carried_usage':carry,
             'clock':{'round_started_at':100,'round_absolute_deadline':1000,'model_absolute_deadline':900,
                      'batch_seconds':1000,'pre_round_carried_seconds':5}} if continuation else None
        current=self.ledger(manifest['execution_order'],carry if continuation else history)
        closure={'closed':True,'closed_at_unix':150,'cleanup_all_passed':True,'controller':{},'checks':{}}
        module=types.ModuleType('agent_final_continuation');module.validate_continuation=lambda _:ctx
        def analyze(reader,root,man,c,v,ledger,controller):
            return self.row(c,v,Path(root)==old and (c,v)==tuple(order[4]))
        return manifest,ctx,current,old_ledger,closure,module,analyze

    def summarize_fixture(self,continuation=True,alter=None,progress=False,row_alter=None):
        m,ctx,current,old,closure,module,analyze=self.fixture(continuation)
        if alter:alter(m,ctx,current,closure)
        original_analyze=analyze
        def analyze(*args):
            row=original_analyze(*args)
            if row_alter:row_alter(row)
            return row
        def ledger(reader,path):return old if Path(path).parent.parent.name=='old' else current
        with patch.dict(sys.modules,{'agent_final_continuation':module}), \
             patch.object(analysis,'historical_usage',return_value=self.history()), \
             patch.object(analysis,'read_ledger',side_effect=ledger), \
             patch.object(analysis,'closure_status',return_value=closure), \
             patch.object(analysis,'analyze_cell',side_effect=analyze):
            return analysis.summarize(analysis.Reader(),Path(m['experiment_root']),m,progress)

    def test_continuation_primary96_all_attempt97_and_finance_once(self):
        result=self.summarize_fixture();a=result['accounting']
        self.assertEqual(len(result['rows']),96);self.assertEqual(result['counts']['A']['planned'],48)
        self.assertEqual(result['counts']['B']['planned'],48)
        self.assertEqual(result['continuation']['retained_roots'],4)
        self.assertEqual(result['continuation']['new_roots_planned'],92)
        self.assertEqual(len(result['technical_attempts']),1)
        self.assertEqual(result['technical_attempts'][0]['original_batch_record']['status'],'partial')
        self.assertIsNone(result['technical_attempts'][0]['actual_task_family_tokens'])
        self.assertEqual(result['attempt_reliability']['all_attempts_planned'],97)
        self.assertEqual(result['attempt_reliability']['first_attempts']['delivery_ready_observed'],95)
        self.assertEqual(a['new_known_settled_record_usage']['tokens'],230)
        self.assertEqual(a['round_all_attempt_known_usage']['tokens'],243)
        self.assertEqual(a['cumulative_known_settled_record_usage']['tokens'],250)
        self.assertEqual(a['cumulative_known_settled_record_usage']['requests'],100)
        self.assertEqual(a['cumulative_known_settled_record_usage']['seconds'],55)
        self.assertEqual(sum(r['actual_task_family_tokens'] for r in result['rows']),240)
        self.assertTrue(result['efficiency_conclusion']['authorized_by_complete_evidence'])
        self.assertEqual(len(result['failure_costs']),1)

    def test_legacy_complete96_has_no_additional_attempt_or_double_clock(self):
        result=self.summarize_fixture(False)
        self.assertEqual(result['attempt_reliability']['all_attempts_planned'],96)
        self.assertEqual(result['accounting']['cumulative_known_settled_record_usage']['tokens'],247)
        self.assertEqual(result['accounting']['cumulative_known_settled_record_usage']['seconds'],55)
        self.assertTrue(result['efficiency_conclusion']['authorized_by_complete_evidence'])

    def test_progress_forbids_success_even_when_all_synthetic_rows_full(self):
        self.assertFalse(self.summarize_fixture(progress=True)['efficiency_conclusion']['authorized_by_complete_evidence'])

    def test_changed_carry_or_missing_origin_blocks_conclusion(self):
        def alter(m,ctx,current,closure):current['metadata']['config']['carried_usage']={}
        result=self.summarize_fixture(alter=alter)
        self.assertIn('historical_carry_mismatch',result['issues'])
        self.assertFalse(result['efficiency_conclusion']['authorized_by_complete_evidence'])
        def no_clock(m,ctx,current,closure):ctx['clock']['round_started_at']=None
        result=self.summarize_fixture(alter=no_clock)
        self.assertIsNone(result['accounting']['cumulative_known_settled_record_usage']['seconds'])
        self.assertFalse(result['efficiency_conclusion']['authorized_by_complete_evidence'])

    def test_original_cumulative_time_cap_does_not_reset(self):
        def alter(m,ctx,current,closure):closure['closed_at_unix']=1200
        result=self.summarize_fixture(alter=alter)
        self.assertFalse(result['accounting']['within_original_cumulative_limits']['seconds'])
        self.assertIn('original_cumulative_budget_exceeded',result['issues'])
        self.assertFalse(result['efficiency_conclusion']['authorized_by_complete_evidence'])

    def test_quality_unknown_and_partial_do_not_silently_become_full(self):
        rows=[self.row(c,v) for c,v in self.order()]
        rows[0]['score']={'whole_answer_quality':'partial_usable'}
        rows[1]['score']={'whole_answer_quality':'unknown'}
        stats=analysis.aggregate(rows)
        self.assertEqual(stats['matched_full_sensitivity']['matched_full_pairs'],47)
        rows[0]['actual_task_family_tokens']=None
        stats=analysis.aggregate(rows)
        self.assertIsNone(stats['overall']['tokens']['change'])
        with self.assertRaisesRegex(ValueError,'duplicate'):analysis.aggregate(rows+[rows[0]])

    def test_quality_threshold_relative_floor_unknown_and_severe_flags(self):
        def quality(a_count,b_count,unknown=False,severe=False):
            def alter(row):
                index=(row['round']-1)*24+int(row['case_id'].rsplit('-',1)[1])-1
                threshold=a_count if row['variant']=='A' else b_count
                if index>=threshold:row['score']['whole_answer_quality']='unknown' if unknown else 'partial_usable'
                if severe and row['variant']=='B':row['score']['forged_success']=True
            return alter
        result=self.summarize_fixture(row_alter=quality(44,44))
        self.assertTrue(result['quality']['quality_gate_passed'])
        self.assertTrue(result['efficiency_conclusion']['authorized_by_complete_evidence'])
        for alter in (quality(44,43),quality(45,44),quality(44,44,unknown=True),quality(48,48,severe=True)):
            with self.subTest(alter=alter):
                result=self.summarize_fixture(row_alter=alter)
                self.assertFalse(result['quality']['quality_gate_passed'])
                self.assertFalse(result['efficiency_conclusion']['authorized_by_complete_evidence'])

    def test_full_pair_sensitivity_can_block_apparent_overall_improvement(self):
        def alter(row):
            index=(row['round']-1)*24+int(row['case_id'].rsplit('-',1)[1])-1
            partial=index>=44
            if partial:row['score']['whole_answer_quality']='partial_usable'
            row['end_to_end']['seconds']=(1000 if partial else 10) if row['variant']=='A' else (1 if partial else 11)
        result=self.summarize_fixture(row_alter=alter)
        self.assertTrue(result['quality']['quality_gate_passed'])
        self.assertTrue(result['efficiency_conclusion']['whole_set_both_lower'])
        self.assertFalse(result['efficiency_conclusion']['matched_full_pair_direction_supports_both_lower'])
        self.assertFalse(result['efficiency_conclusion']['authorized_by_complete_evidence'])

    def test_missing_primary_keeps_null_and_prohibits_complete_set_conclusion(self):
        def missing(row):
            if row['expanded_case_id']=='AFE-QA-24-R2' and row['variant']=='B':
                row.update(root_id=None,status='not_created',actual_task_family_tokens=None,
                           verified_settled_tokens_lower_bound=None,usage_reconciled=False,
                           score={'whole_answer_quality':None},end_to_end={'seconds':None})
        result=self.summarize_fixture(row_alter=missing)
        row=result['rows'][-1]
        self.assertIsNone(row['actual_task_family_tokens'])
        self.assertIsNone(result['statistics']['overall']['tokens']['change'])
        self.assertFalse(result['efficiency_conclusion']['authorized_by_complete_evidence'])
        self.assertEqual(result['created_roots'],95)
        self.assertIsNone(result['accounting']['new_actual_tokens_complete'])
        self.assertFalse(result['accounting']['all_new_actual_usage_known'])

    def test_full_grade_cannot_override_incomplete_delivery(self):
        def alter(row):
            if row['variant']=='B':row['end_to_end']['delivery_ready_observed']=False
        result=self.summarize_fixture(row_alter=alter)
        self.assertTrue(any('full_grade_without_observed_complete_delivery' in i for i in result['issues']))
        self.assertFalse(result['efficiency_conclusion']['authorized_by_complete_evidence'])

    def test_cli_uses_explicit_manifest_root_and_refuses_overwrite(self):
        result=self.summarize_fixture(False)
        manifest={'experiment_root':str(self.root)}
        path=self.write('manifest.json',manifest)
        output=self.root/'formal-analysis'/'synthetic'
        with patch.object(analysis,'summarize',return_value=result):
            self.assertEqual(analysis.main(['--manifest',str(path),'--output-dir',str(output)]),0)
            self.assertEqual(json.loads((output/'metrics.json').read_text())['formal_roots_planned'],96)
            self.assertTrue((output/'SEAL.json').exists())
            with self.assertRaises(FileExistsError):analysis.main(['--manifest',str(path),'--output-dir',str(output)])
            with self.assertRaises(PermissionError):analysis.main(['--manifest',str(path),'--output-dir',str(self.root/'outside')])

    def test_closed_snapshot_sqlite_and_output_are_read_only(self):
        db_path=self.root/'synthetic.sqlite3'
        import sqlite3
        db=sqlite3.connect(db_path)
        try:
            with db:db.execute('CREATE TABLE tasks(task_id TEXT)');db.execute("INSERT INTO tasks VALUES('synthetic')")
        finally:db.close()
        before=db_path.read_bytes();r=analysis.Reader()
        self.assertEqual(r.table(db_path,'tasks'),[{'task_id':'synthetic'}]);self.assertEqual(before,db_path.read_bytes())
        Path(str(db_path)+'-wal').write_bytes(b'active')
        with self.assertRaisesRegex(RuntimeError,'WAL'):r.table(db_path,'tasks')
        secret=self.write('.env.synthetic',{})
        with self.assertRaises(PermissionError):r.bytes(secret)


if __name__=='__main__':unittest.main()
