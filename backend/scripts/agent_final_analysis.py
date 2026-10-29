"""只读汇总冻结96格、物理/原生收据与外部评分，不导入业务或启动服务。"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))

CATEGORIES = {'QA':'单篇问答','NUM':'表格与实验分析','CROSS':'跨论文比较','REC':'论文筛选推荐',
              'REV':'定向核验修订','MEM':'Memory接续','DEEP':'完整精读保存'}
TERMINAL = {'completed','partial','failed','cancelled','needs_input'}
FINAL_CONTROLLER = {'exited','model_hard_deadline','cancelled','controller_error','refused_model_deadline_already_passed'}
COMPLETION_EVALUATION_POLICY = {'version':'agent_completion_descriptive_v1',
    'minimum_B_full':None,'retain_quality_comparison':True}


def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()


def decoded(value, default=None):
    if isinstance(value,str):
        try:return json.loads(value)
        except (TypeError,ValueError):return default
    return value if value is not None else default


def nonnegative(value):
    return type(value) is int and value >= 0


def stamp(value):
    if type(value) in (int,float) and math.isfinite(value):return float(value)
    if isinstance(value,str):
        try:return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
        except ValueError:pass
    return None


def parsed_usage(value):
    obj=decoded(value,{})
    if not isinstance(obj,dict):return None
    a,b,t=(obj.get(k) for k in ['prompt_tokens','completion_tokens','total_tokens'])
    if not all(nonnegative(v) for v in (a,b,t)) or a+b != t:return None
    details=obj.get('prompt_tokens_details') or {}
    if not isinstance(details,dict):return None
    nested=details.get('cached_tokens')
    hit=obj.get('prompt_cache_hit_tokens',nested if nested is not None else 0)
    miss=obj.get('prompt_cache_miss_tokens',a-hit if nonnegative(hit) else None)
    if not nonnegative(hit) or not nonnegative(miss) or hit+miss != a:return None
    if nested is not None and nested != hit:return None
    return {'input':a,'output':b,'tokens':t,'cached_input':hit,'uncached_input':miss}


class Reader:
    """只读本机文件和已关闭SQLite，记录输入身份；绝不读取.env或发起网络。"""
    def __init__(self):self.hashes={};self.connections=0
    def bytes(self,path):
        path=Path(path)
        if path.name.startswith('.env'):raise PermissionError('credential file forbidden')
        data=path.read_bytes();self.hashes[str(path)]=sha(data);return data
    def json(self,path,default=None):
        path=Path(path)
        return json.loads(self.bytes(path)) if path.exists() else default
    def table(self,path,table,where='',params=()):
        path=Path(path)
        if not path.exists():return None
        if table not in {'metadata','roots','reservations','tasks','task_events','model_calls','actions','save_receipts'}:
            raise PermissionError('table not allowed')
        # 收尾后应无活跃WAL；不以immutable读丢WAL，也不让只读连接创建原目录SHM。
        wal=Path(str(path)+'-wal')
        if wal.exists() and wal.stat().st_size:
            raise RuntimeError('closed SQLite still has WAL; preserve a complete offline snapshot before analysis')
        before=sha(self.bytes(path))
        db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True,timeout=1)
        self.connections+=1;db.row_factory=sqlite3.Row
        try:
            db.execute('PRAGMA query_only=ON')
            present=db.execute('SELECT name FROM sqlite_master WHERE type=? AND name=?',('table',table)).fetchone()
            rows=[dict(r) for r in db.execute('SELECT * FROM '+table+(' WHERE '+where if where else ''),params)] if present else []
        finally:db.close()
        if sha(path.read_bytes()) != before:raise RuntimeError('SQLite changed during read-only analysis')
        return rows


def read_ledger(reader,path):
    metadata=reader.table(path,'metadata')
    if metadata is None:return None
    meta={x['key']:decoded(x['value']) for x in metadata}
    return {'metadata':meta,'roots':reader.table(path,'roots'),'reservations':reader.table(path,'reservations')}


def closure_status(reader,root,manifest):
    controller=reader.json(root/'formal-control/controller-result.json')
    auth=reader.json(root/'formal-preflight/authorization-closed.json')
    vector=reader.json(root/'formal-preflight/vector-close.json')
    current=reader.json(Path(manifest['authorization_path']))
    checks={'controller_terminal':bool(controller and controller.get('status') in FINAL_CONTROLLER),
            'batch_group_cleaned':bool(controller and controller.get('batch_group_cleaned') is True),
            'execute_authorization_closed':bool(auth and auth.get('status')=='closed' and current
                and current.get('approved') is False and current.get('allowed_modes')==[]),
            'owned_vector_stopped':bool(vector and (vector.get('identity') or {}).get('running') is False)}
    ends=[stamp(x) for x in [(controller or {}).get('wrapper_finalized_at'),(controller or {}).get('finished_at'),
                             (auth or {}).get('at'),(vector or {}).get('at')]]
    end=max((v for v in ends if v is not None),default=None)
    financial_only='completion_contract' in manifest
    if financial_only:
        stopped=stamp((controller or {}).get('model_processes_stopped_at'))
        cleanup_started=stamp((controller or {}).get('cleanup_started_at'))
        cleanup_deadline=stamp((controller or {}).get('cleanup_deadline_after_stop'))
        started=stamp((controller or {}).get('started_at'))
        checks['controller_terminal']=bool(controller and controller.get('status') in FINAL_CONTROLLER|{'financial_budget_halt'})
        checks.update(financial_only_controller_policy=bool(controller and controller.get('policy')=='financial_only'),
            bounded_cleanup_contract=((manifest.get('formal_controller_contract') or {}).get('policy')=='financial_only'
                and (manifest.get('formal_controller_contract') or {}).get('cleanup_reserve_seconds')==300),
            bounded_cleanup_observed=bool(stopped is not None and cleanup_deadline is not None
                and cleanup_started is not None and started is not None and started<=cleanup_started<=stopped
                and abs(cleanup_deadline-cleanup_started-300)<.000001
                and end is not None and stopped<=end<=cleanup_deadline),
            cleanup_steps_passed=bool(controller and controller.get('cleanup_all_passed') is True),
            financial_only_authorization_closed=bool(auth and current and auth.get('policy')=='financial_only'
                and auth.get('no_resume_or_resend') is True and stamp(auth.get('round_started_at')) is not None
                and auth.get('round_started_at')==current.get('round_started_at')
                and auth.get('round_started_at')==(manifest['completion_contract'].get('clock') or {}).get('round_started_at')))
    closed=all(checks.values())
    return {'closed':closed,'checks':checks,'closed_at_unix':end,'controller_status':(controller or {}).get('status'),
            'cleanup_all_passed':(controller or {}).get('cleanup_all_passed'),
            'wrapper_finished_before_original_total_deadline':(controller or {}).get('wrapper_finished_before_original_total_deadline'),
            'original_absolute_total_deadline':(controller or {}).get('original_absolute_total_deadline'),
            'controller_cleanup_deadline_is_not_new_authorized_deadline':True,
            'timing_policy':'financial_only' if financial_only else 'absolute_deadline',
            'cleanup_deadline_after_stop':(controller or {}).get('cleanup_deadline_after_stop'),
            'controller':controller}


def archive_usage(reader,cell,reservation):
    base=cell/'requests'/str(reservation['reservation_id'])
    path=Path(str(base)+'.response.json')
    if path.exists():
        obj=reader.json(path,{})
        return {'response_id':obj.get('id'),'usage':parsed_usage(obj.get('usage')),'complete_archive':True,'path':str(path)}
    path=Path(str(base)+'.response.sse')
    if path.exists():
        identity=None;usage=None;done=False
        for line in reader.bytes(path).decode('utf-8').splitlines():
            if not line.startswith('data:'):continue
            text=line[5:].strip()
            if text=='[DONE]':done=True;continue
            obj=decoded(text)
            if isinstance(obj,dict):
                identity=obj.get('id') or identity
                if obj.get('usage') is not None:usage=parsed_usage(obj['usage'])
        return {'response_id':identity,'usage':usage,'complete_archive':done,'path':str(path)}
    return {'response_id':None,'usage':None,'complete_archive':False,'path':None}


def latency(root,events,all_tasks,export,controller):
    identity=root.get('task_id')
    starts=[x for x in events if x.get('task_id')==identity and x.get('event_type')=='task_created']
    ends=[x for x in events if x.get('task_id')==identity and x.get('event_type')=='task_finished']
    a=stamp(starts[0].get('created_at')) if len(starts)==1 else None
    b=stamp(ends[0].get('created_at')) if len(ends)==1 else None
    own_created=stamp(root.get('created_at'));own_updated=stamp(root.get('updated_at'))
    issue=[]
    if a is not None and own_created is not None and abs(own_created-a)>.01:issue.append('root_creation_event_timestamp_mismatch')
    if a is None:issue.append('unique_creation_event_missing')
    if b is not None and (a is None or b<a):issue.append('invalid_terminal_timestamp');b=None
    active=[x['task_id'] for x in all_tasks if x.get('status') not in TERMINAL]
    later=[x['task_id'] for x in all_tasks if b is not None and stamp(x.get('updated_at')) is not None
           and stamp(x['updated_at'])>b+.01 and x['task_id']!=identity]
    answer=decoded(root.get('result_json'),{}) or {}
    has_answer=isinstance(answer.get('answer_markdown'),str) and bool(answer.get('answer_markdown','').strip())
    ready=bool(b is not None and root.get('status')=='completed' and has_answer and not active and not later and not issue)
    observed_stops=[stamp((controller or {}).get('model_processes_stopped_at')),stamp(export.get('finished_at'))]
    stop=min((v for v in observed_stops if v is not None),default=None)
    duration=b-a if a is not None and b is not None and not issue else None
    return {'seconds':duration,'source':'unique_root_task_created_to_task_finished_events',
            'measurement_kind':'delivery_terminal_duration' if ready else 'terminal_or_failure_duration_not_quality_success',
            'delivery_ready_observed':ready,'task_created_event_count':len(starts),'task_finished_event_count':len(ends),
            'task_created_at':starts[0].get('created_at') if len(starts)==1 else None,
            'task_finished_at':ends[0].get('created_at') if len(ends)==1 else None,
            'root_row_created_delta_seconds':own_created-a if own_created is not None and a is not None else None,
            'root_updated_delta_from_finish_diagnostic_only':own_updated-b if own_updated is not None and b is not None else None,
            'active_descendant_ids':active,'descendants_updated_after_root_finish':later,
            'interrupted_time_to_observed_stop_seconds':stop-a if b is None and stop is not None and a is not None and stop>=a else None,
            'updated_at_never_substitutes_for_delivery':True,'issues':issue}


def semantic_score(reader,root,manifest,case_id,variant,root_id,answer,*,artifact_cell=None,
                   source_manifest=None):
    directory=Path(manifest['semantic_gate_root'])/case_id
    challenge=reader.json(directory/'challenge-private.json')
    review=reader.json(directory/'review.json');decision=reader.json(directory/'decision.json')
    empty={'whole_answer_quality':None,'score_status':'not_scored','execution_complete':None,
           'unauthorized_delivery_write':None,'forged_success':None,'issues':[]}
    if not challenge or not review or not decision:return empty
    issues=[]
    manifest_id=sha(canonical({k:v for k,v in manifest.items() if k!='cell_absolute_deadline'}))
    identity=(challenge.get('identity') or {}).get('answers',{}).get(variant,{})
    expected=sha(answer) if isinstance(answer,str) else None
    if challenge.get('manifest_sha256')!=manifest_id:issues.append('review_manifest_mismatch')
    if review.get('challenge_nonce')!=challenge.get('nonce') or review.get('challenge_sha256')!=sha(canonical(challenge)):
        issues.append('review_challenge_mismatch')
    if decision.get('review_sha256')!=sha(canonical(review)) or decision.get('challenge_sha256')!=sha(canonical(challenge)):
        issues.append('decision_review_mismatch')
    if identity.get('root_id')!=root_id or identity.get('answer_sha256')!=expected:issues.append('review_answer_identity_mismatch')
    cell=Path(artifact_cell) if artifact_cell is not None else Path(manifest['runtime_root'])/'cells'/(case_id+'-'+variant)
    if 'completion_contract' in manifest:
        # 混合配对的旧B保留原路径与原源码身份，不能借当前manifest改名为新运行。
        owner=source_manifest or manifest
        from harness_ab_experiment import paired_source_contract
        origin={'cell_root':str(cell),'manifest_sha256':sha(canonical({k:v for k,v in owner.items()
                 if k!='cell_absolute_deadline'})),
                'source_contract_sha256':sha(canonical(paired_source_contract(owner))),
                'loaded_prompt_contract_sha256':sha(canonical(owner['loaded_prompt_contract'][variant])),
                'retained':artifact_cell is not None}
        if identity.get('source_provenance')!=origin:issues.append('review_artifact_source_provenance_mismatch')
    for name,expected_file in identity.get('files',{}).items():
        path=cell/name
        if not path.exists() or sha(reader.bytes(path))!=expected_file:issues.append('review_export_file_mismatch:'+name)
    grades=[g for g in review.get('reviews',[]) if g.get('anonymous_id')==challenge.get('anonymous_ids',{}).get(variant)]
    if len(grades)!=1:return dict(empty,score_status='invalid_review_binding',issues=issues+['review_grade_missing_or_duplicate'])
    grade=grades[0]
    sources=[v['pdf_sha256'] for v in challenge.get('identity',{}).get('materials',[])]
    if grade.get('answer_sha256')!=expected or grade.get('source_sha256s')!=sources:issues.append('grade_answer_or_material_mismatch')
    if decision.get('quality',{}).get(variant)!=grade.get('whole_answer_quality'):issues.append('decision_grade_label_mismatch')
    quality=grade.get('whole_answer_quality')
    if quality=='full' and (grade.get('core_complete') is not True or grade.get('requested_scope_complete') is not True
         or grade.get('critical_core_error') is not False or grade.get('execution_complete') is not True
         or grade.get('unauthorized_delivery_write') is not False or grade.get('forged_success') is not False):issues.append('full_grade_contradiction')
    if issues:return dict(empty,score_status='invalid_review_binding',issues=issues)
    return {k:grade.get(k) for k in ['whole_answer_quality','core_complete','requested_scope_complete','critical_core_error',
          'execution_complete','unauthorized_delivery_write','forged_success','answer_sha256','evidence_notes']} | {
          'score_status':'artifact_bound_external_source_review','reviewer_kind':decision.get('reviewer_kind'),
          'pair_quality_equivalent':decision.get('quality_equivalent') is True,'issues':[]}


def analyze_cell(reader,root,manifest,case_id,variant,ledger,controller,*,artifact_cell=None,
                 score_manifest=None):
    rates=(ledger or {}).get('metadata',{}).get('config',{})
    cell=Path(artifact_cell) if artifact_cell is not None else Path(manifest['runtime_root'])/'cells'/(case_id+'-'+variant)
    result=reader.json(cell/'result.json',{})
    exported_root=reader.json(cell/'root-row.json',{})
    view=reader.json(cell/'task-view.json',{})
    db=cell/'task.sqlite3'
    tasks=reader.table(db,'tasks')
    root_candidates=[x for x in tasks or [] if not x.get('parent_task_id')]
    roots=[x for x in (ledger or {}).get('roots',[]) if x.get('case_id')==case_id and x.get('variant')==variant]
    root_id=result.get('root_id') or exported_root.get('task_id') or view.get('task_id') or (roots[0]['root_id'] if len(roots)==1 else None)
    if root_id is None and len(root_candidates)==1:root_id=root_candidates[0]['task_id']
    base,round_s=case_id.rsplit('-R',1)
    row={'case_id':base,'round':int(round_s),'expanded_case_id':case_id,'variant':variant,'root_id':root_id,
         'category':CATEGORIES.get(base.split('-')[1],'unknown'),'status':'not_created','cell_directory_exists':cell.exists(),
         'actual_task_family_tokens':None,'actual_input_tokens':None,'actual_output_tokens':None,
         'actual_cached_input_tokens':None,'physical_sent_requests':None,'end_to_end':{'seconds':None},
         'score':{'whole_answer_quality':None,'score_status':'not_created'},'issues':[]}
    if root_id is None:return row
    candidates=[x for x in tasks or [] if x.get('task_id')==root_id]
    task=candidates[0] if len(candidates)==1 else exported_root
    row['backend_task_status']=task.get('status')
    row['execution_attempt_status']=result.get('status')
    row['exception_type']=result.get('exception_type')
    row['status']=('technical_failed_before_terminal' if result.get('status')=='failed' and task.get('status') not in TERMINAL
                   else task.get('status',result.get('task_status','unknown')))
    if len(candidates)!=1:row['issues'].append('root_sqlite_record_missing_or_ambiguous')
    family=[x for x in tasks or [] if x.get('root_task_id')==root_id or x.get('task_id')==root_id]
    events=reader.table(db,'task_events') or (reader.json(cell/'events.json',{}) or {}).get('events',[])
    native=reader.table(db,'model_calls')
    native_export=reader.json(cell/'model-calls.json')
    if native is None:native=native_export
    elif native_export is not None and sorted((x['call_id'],x.get('status'),x.get('usage_json')) for x in native)!=sorted((x['call_id'],x.get('status'),x.get('usage_json')) for x in native_export):row['issues'].append('native_sqlite_export_disagreement')
    row['end_to_end']=latency(task,events,family,result,controller)
    row['issues']+=row['end_to_end']['issues']
    if exported_root and (exported_root.get('task_id')!=root_id or exported_root.get('result_json')!=task.get('result_json')):row['issues'].append('root_sqlite_export_disagreement')
    own=None if ledger is None else [x for x in ledger['reservations'] if x.get('root_id')==root_id]
    settled=[x for x in own or [] if x.get('status')=='settled']
    pending=[x for x in own or [] if x.get('status') not in {'settled','not_sent'}]
    receipt_checks=[];native_by_id=defaultdict(list)
    for call in native or []:
        response=decoded(call.get('response_json'),{}) or {}
        identity=call.get('request_id') or response.get('id')
        if identity:native_by_id[identity].append(call)
        if call.get('task_id') not in {x['task_id'] for x in family}:row['issues'].append('native_call_outside_root_family')
    seen=set();valid=[]
    for reservation in settled:
        usage=parsed_usage(reservation.get('usage_json'))
        meta=decoded(reservation.get('response_json'),{}) or {};identity=meta.get('response_id')
        archive=archive_usage(reader,cell,reservation)
        matches=native_by_id.get(identity,[])
        checks={'reservation_id':reservation['reservation_id'],'response_id':identity,
                'unique_response_id':bool(identity and identity not in seen),
                'usage_valid':usage is not None,'native_match_count':len(matches),
                'archive_complete':archive['complete_archive'],'archive_path':archive['path']}
        seen.add(identity)
        if usage is not None:
            checks.update(physical_fields_match=all(reservation.get(k)==usage[u] for k,u in
                          [('actual_input','input'),('actual_output','output'),('actual_tokens','tokens')]),
                          native_usage_match=len(matches)==1 and parsed_usage(matches[0].get('usage_json'))==usage
                            and matches[0].get('status')=='succeeded',
                          archive_usage_match=archive['usage']==usage and archive['response_id']==identity,
                          peak_cost_match=rate_cost(usage,rates) is not None and reservation.get('actual_nano')==rate_cost(usage,rates))
        passed=all(checks.get(k) is True for k in ['unique_response_id','usage_valid','archive_complete','physical_fields_match',
                                                 'native_usage_match','archive_usage_match','peak_cost_match'])
        checks['passed']=passed;receipt_checks.append(checks)
        if passed:valid.append((reservation,usage))
    native_success=[x for x in native or [] if x.get('status')=='succeeded']
    native_pending=[x for x in native or [] if x.get('status') not in {'succeeded','not_sent'}]
    reconciled=bool(own is not None and native is not None and len(valid)==len(settled)
                     and len(native_success)==len(settled) and not pending and not native_pending and not row['issues'])
    known={k:sum(u[k] for _,u in valid) for k in ['input','output','tokens','cached_input','uncached_input']}
    row.update(actual_task_family_tokens=known['tokens'] if reconciled else None,
               actual_input_tokens=known['input'] if reconciled else None,actual_output_tokens=known['output'] if reconciled else None,
               actual_cached_input_tokens=known['cached_input'] if reconciled else None,
               verified_settled_tokens_lower_bound=known['tokens'] if own is not None else None,
               verified_settled_nano_yuan_lower_bound=sum(r['actual_nano'] for r,_ in valid) if own is not None else None,
               actual_task_family_nano_yuan=sum(r['actual_nano'] for r,_ in valid) if reconciled else None,
               usage_reconciled=reconciled,token_status='verified_actual' if reconciled else 'unknown_or_unreconciled',
               physical_sent_requests=sum(x.get('sent_at') is not None for x in own) if own is not None else None,
               physical_status_counts=dict(Counter(x.get('status') for x in own)) if own is not None else None,
               native_status_counts=dict(Counter(x.get('status') for x in native)) if native is not None else None,
               receipt_checks=receipt_checks,pending_reservations=[{'reservation_id':x['reservation_id'],'status':x['status'],
                     'reserved_tokens':x['reserved_tokens'],'reserved_nano_yuan':x['reserved_nano'],'outcome':'unconfirmed_no_resend'} for x in pending],
               request_kinds=dict(Counter((decoded(x.get('request_json'),{}) or {}).get('request_kind','missing') for x in own or [])),
               family_roles=dict(Counter(x.get('role','unspecified') for x in family)),
               family_task_ids=[x['task_id'] for x in family])
    answer=(decoded(task.get('result_json'),{}) or {}).get('answer_markdown')
    scoring=score_manifest or manifest
    row['score']=semantic_score(reader,root,scoring,case_id,variant,root_id,answer,
        artifact_cell=cell if artifact_cell is not None else None,source_manifest=manifest)
    row['issues']+=row['score']['issues']
    actions=reader.table(db,'actions') or []
    row['failed_actions']=[]
    for action in actions:
        if action.get('status') in {'succeeded','success'}:continue
        observation=decoded(action.get('observation_json'),{}) or {}
        row['failed_actions'].append({k:action.get(k) for k in ['action_id','task_id','sequence_no','tool_name','status']} | {
                'observation_kind':observation.get('kind'),'observation_sha256':sha(canonical(observation)),
                'error_text_omitted':True})
    row['noncompleted_descendants']=[{'task_id':x['task_id'],'role':x.get('role'),'status':x.get('status')} for x in family
                                     if x['task_id']!=root_id and x.get('status')!='completed']
    return row


def delta(a,b):
    if not all(type(v) in (int,float) and math.isfinite(v) for v in (a,b)) or a<=0:return None
    return {'A':a,'B':b,'B_minus_A':b-a,'B_minus_A_percent':100*(b-a)/a,'reduction_percent':100*(1-b/a)}


def aggregate(rows,*,require_delivery_ready=False):
    """固定题等权，不平均各题百分比；不把缺失格排除后冒充全矩阵。"""
    by_case=defaultdict(dict)
    for r in rows:
        key=(r['round'],r['variant'])
        if key in by_case[r['case_id']]:raise ValueError('duplicate primary case/round/variant')
        by_case[r['case_id']][key]=r
    means=[]
    for case,group in sorted(by_case.items()):
        record={'case_id':case,'category':next(iter(group.values()))['category']}
        for variant in ['A','B']:
            own=[group.get((round_,variant)) for round_ in [1,2]]
            for name,get in [('latency_seconds',lambda r:r['end_to_end'].get('seconds')
                    if not require_delivery_ready or r['end_to_end'].get('delivery_ready_observed') is True else None),
                    ('tokens',lambda r:r.get('actual_task_family_tokens'))]:
                vals=[get(x) if x else None for x in own]
                record[variant+'_'+name]=statistics.mean(vals) if len(vals)==2 and all(v is not None for v in vals) else None
        record['latency_change']=delta(record['A_latency_seconds'],record['B_latency_seconds'])
        record['token_change']=delta(record['A_tokens'],record['B_tokens'])
        means.append(record)
    def summarize(items):
        result={'fixed_cases':len(items)}
        for name in ['latency_seconds','tokens']:
            totals={v:statistics.mean([x[v+'_'+name] for x in items]) if items and all(x[v+'_'+name] is not None for x in items) else None for v in ['A','B']}
            result[name]={'equal_case_weight_mean':totals,'change':delta(totals['A'],totals['B'])}
        return result
    overall=summarize(means)
    categories={category:summarize([x for x in means if x['category']==category]) for category in sorted({x['category'] for x in means})}
    matched=[]
    for case,group in by_case.items():
        for round_ in [1,2]:
            a,b=group.get((round_,'A')),group.get((round_,'B'))
            if a and b and all(x['score'].get('whole_answer_quality')=='full' and x.get('usage_reconciled')
                 and (not require_delivery_ready or x['end_to_end'].get('delivery_ready_observed') is True) for x in [a,b]):
                matched.append({'case_id':case,'round':round_,'latency_change':delta(a['end_to_end'].get('seconds'),b['end_to_end'].get('seconds')),
                                'token_change':delta(a.get('actual_task_family_tokens'),b.get('actual_task_family_tokens'))})
    sensitivity={'matched_full_pairs':len(matched),'pairs':matched,'meaning':'双方full的共同题轮敏感性，不能替换固定96格主分母。'}
    for metric,key in [('latency_seconds','latency_change'),('tokens','token_change')]:
        changes=[x[key] for x in matched]
        sensitivity[metric]=delta(sum(x['A'] for x in changes),sum(x['B'] for x in changes)) if changes and all(x is not None for x in changes) else None
    local=[{'case_id':x['case_id'],'category':x['category'],'latency_change':x['latency_change'],'token_change':x['token_change']} for x in means
           if any(x[k] and x[k]['B_minus_A']>0 for k in ['latency_change','token_change'])]
    return {'overall':overall,'categories':categories,'case_means':means,'matched_full_sensitivity':sensitivity,
            'latency_population':'observed_complete_delivery_only' if require_delivery_ready else 'observed_terminal_duration',
            'comparable_case_count':sum(x['latency_change'] is not None and x['token_change'] is not None for x in means),
            'local_regressions':local}


def rate_cost(usage, config):
    """按物理账本冻结费率复核金额，不在分析脚本另设价格。"""
    fields={'cached_input':'cached_input_nano_per_token','uncached_input':'input_nano_per_token',
            'output':'output_nano_per_token'}
    if not all(nonnegative(config.get(k)) for k in fields.values()):return None
    return sum(usage[key]*config[field] for key,field in fields.items())


def historical_usage(reader,manifest):
    """从hash锚定的旧账本与收尾时间推导历史，拒绝凭常量补齐。"""
    source=manifest.get('carried_budget_source')
    expected=manifest.get('ledger_limits',{}).get('carried_usage')
    keys={'tokens','requests','nano_yuan','seconds'}
    if not source:
        if expected and (set(expected)!=keys or any(not nonnegative(v) for v in expected.values()) or any(expected.values())):
            raise ValueError('nonzero historical carry requires anchored source')
        return dict.fromkeys(keys,0)
    values={}
    for name in ('batch_results','ledger_final'):
        data=reader.bytes(source[name+'_path'])
        if sha(data)!=source[name+'_sha256']:raise ValueError('historical receipt hash mismatch')
        values[name]=json.loads(data)
    ledger,batch=values['ledger_final'],values['batch_results']
    raw=ledger['reservations']
    if any(r['status'] not in {'settled','not_sent'} for r in raw):raise ValueError('unresolved historical reservation')
    settled=[r for r in raw if r['status']=='settled']
    if any(not nonnegative(r.get(k)) for r in settled for k in ('actual_tokens','actual_nano')):
        raise ValueError('invalid historical settled usage')
    tokens=sum(r['actual_tokens'] for r in settled);nano=sum(r['actual_nano'] for r in settled)
    if (tokens!=ledger['actual_known_tokens'] or nano!=ledger['actual_known_estimated_nano_yuan']
            or len(settled)!=ledger['physical_sent_requests']
            or batch['actual_known_tokens']!=tokens or batch['physical_requests']!=len(settled)
            or batch['actual_known_estimated_nano_yuan']!=nano):raise ValueError('historical totals disagree')
    prior=ledger.get('carried_usage') or dict.fromkeys(keys,0)
    if set(prior)!=keys or any(not nonnegative(v) for v in prior.values()):raise ValueError('invalid historical prior carry')
    elapsed=Decimal(str(batch['execution_seconds']))
    if not elapsed.is_finite() or elapsed<0:raise ValueError('invalid historical time')
    result={'tokens':tokens+prior['tokens'],'requests':len(settled)+prior['requests'],
            'nano_yuan':nano+prior['nano_yuan'],'seconds':math.ceil(elapsed)+prior['seconds']}
    if ledger['charged_or_reserved']!={k:result[v] for k,v in [('tokens','tokens'),('requests','requests'),('nano','nano_yuan')]}:
        raise ValueError('historical cumulative amounts disagree')
    if result!=expected:raise ValueError('manifest historical carry mismatch')
    return result


def ledger_totals(ledger):
    """已结算与未知预留分别汇总；账本不存在时保持null。"""
    if ledger is None:
        return {'known':dict.fromkeys(('tokens','requests','nano_yuan'),None),
                'held':dict.fromkeys(('tokens','requests','nano_yuan'),None),'pending':None,'status_counts':None,
                'settled_request_count':None,'physical_sent_request_count':None}
    raw=ledger['reservations'];settled=[r for r in raw if r['status']=='settled']
    valid=all(all(nonnegative(r.get(k)) for k in ('actual_input','actual_output','actual_tokens','actual_nano'))
              and r['actual_input']+r['actual_output']==r['actual_tokens'] for r in settled)
    pending=[r for r in raw if r['status'] not in {'settled','not_sent'}]
    held_valid=valid and all(nonnegative(r.get(k)) for r in pending for k in ('reserved_tokens','reserved_nano'))
    return {'known':{'tokens':sum(r['actual_tokens'] for r in settled) if valid else None,
                     'requests':sum(r.get('sent_at') is not None for r in raw),
                     'nano_yuan':sum(r['actual_nano'] for r in settled) if valid else None},
            'held':{'tokens':sum(r['actual_tokens'] if r['status']=='settled' else r['reserved_tokens'] for r in raw if r['status']!='not_sent') if held_valid else None,
                    'requests':sum(r['status']!='not_sent' for r in raw),
                    'nano_yuan':sum(r['actual_nano'] if r['status']=='settled' else r['reserved_nano'] for r in raw if r['status']!='not_sent') if held_valid else None},
            'pending':pending,'status_counts':dict(Counter(r['status'] for r in raw)),
            'settled_request_count':len(settled),'physical_sent_request_count':sum(r.get('sent_at') is not None for r in raw)}


def combined_ledger_totals(ledgers):
    """只合并各批自身reservations；不重复叠加其metadata中的历史carry。"""
    items=[ledger_totals(value) for value in ledgers]
    def total(group,key):
        values=[row[group][key] for row in items]
        return sum(values) if all(value is not None for value in values) else None
    known={key:total('known',key) for key in ('tokens','requests','nano_yuan')}
    settled=[row['settled_request_count'] for row in items]
    known['requests']=sum(settled) if all(value is not None for value in settled) else None
    counts=Counter()
    for row in items:counts.update(row['status_counts'] or {})
    return {'known':known,'held':{key:total('held',key) for key in ('tokens','requests','nano_yuan')},
            'pending':[reservation for row in items for reservation in row['pending'] or []]
                       if all(row['pending'] is not None for row in items) else None,
            'status_counts':dict(counts),'settled_request_count':known['requests'],
            'physical_sent_request_count':sum(row['physical_sent_request_count'] for row in items)
                if all(row['physical_sent_request_count'] is not None for row in items) else None}


def completion_artifacts(reader,manifest,context):
    """按每条旧格的原manifest/SQLite/账本读取，未评分旧B才使用新混合配对审核。"""
    owners={};ledgers={};closures={}
    for value in context['closed_ledger_roots']:
        root=Path(value)
        owner=reader.json(root/'manifest.formal.json')
        if not isinstance(owner,dict) or Path(owner['experiment_root'])!=root:
            raise ValueError('completion source manifest geometry mismatch')
        owners[str(root)]=owner
        ledgers[str(root)]=read_ledger(reader,Path(owner['runtime_root'])/'budget.sqlite3')
        closures[str(root)]=closure_status(reader,root,owner)
    def artifact(item,technical=False):
        provenance=item['provenance'];root=Path(provenance['anchor_root'])
        owner=owners[str(root)];cell=Path(provenance['cell_root']);key=(item['case_id'],item['variant'])
        if cell!=Path(owner['runtime_root'])/'cells'/(key[0]+'-'+key[1]):
            raise ValueError('completion cell source geometry mismatch')
        source_path=item.get('source_manifest_path',provenance.get('source_manifest_path'))
        source_sha=item.get('source_manifest_sha256',provenance.get('source_manifest_sha256'))
        if source_path is not None and (Path(source_path)!=root/'manifest.formal.json'
                or source_sha is None or sha(reader.bytes(source_path))!=source_sha):
            raise ValueError('completion source manifest identity mismatch')
        score_owner=owner
        if not technical and item.get('quality') is None:
            score_owner=manifest
        row=analyze_cell(reader,root,owner,*key,ledgers[str(root)],closures[str(root)]['controller'],
                         artifact_cell=cell,score_manifest=score_owner)
        row['continuation_provenance']={'retained':not technical,**provenance}
        row['original_batch_record']=item
        if not technical and item.get('quality') is not None and row['score'].get('whole_answer_quality')!=item['quality']:
            raise ValueError('retained original grade changed')
        if technical:
            row.update(attempt_kind='technical_attempt',observed_attempt_outcome=item.get('attempt_outcome',item.get('status')),
                       native_status_preserved=row['status'])
        return row
    retained={(item['case_id'],item['variant']):artifact(item) for item in context['retained_rows']}
    technical=[artifact(item,True) for item in context['technical_attempts']]
    prior=combined_ledger_totals(list(ledgers.values()))
    pending=[]
    for root,ledger in ledgers.items():
        for item in ledger_totals(ledger)['pending'] or []:
            pending.append({'anchor_root':root,**item,'outcome':'unconfirmed_no_resend'})
    prior['pending']=pending if prior['pending'] is not None else None
    return retained,technical,prior,closures


def new_allowance(limits,history):
    """原总限额扣原历史；续跑不构造新的增量授权。"""
    result={k:limits[lk]-history[k] for k,lk in [('tokens','batch_tokens'),('requests','batch_requests'),('seconds','batch_seconds')]}
    result['nano_yuan']=int(Decimal(str(limits['batch_yuan']))*Decimal(1000000000))-history['nano_yuan']
    return result


def validate_matrix(order):
    """主分母始终为24题两轮每版48格，续跑只改变工件来源。"""
    if (len(order)!=96 or len({tuple(x) for x in order})!=96
            or Counter(v for _,v in order)!={'A':48,'B':48}):raise ValueError('fixed96 unique rows,48 pervariant required')
    groups=Counter(case.rsplit('-R',1)[0] for case,_ in order)
    if len(groups)!=24 or any(n!=4 for n in groups.values()):raise ValueError('fixed24 cases x2rounds x2variants required')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+-R[12]',c) for c,_ in order):raise ValueError('invalid frozen round identity')


def attempt_reliability(primary,technical):
    """重试保留为额外attempt，不覆盖原第一次失败或改变主96格。"""
    replacements={(r['expanded_case_id'],r['variant']):r for r in technical}
    first=[replacements.get((r['expanded_case_id'],r['variant']),r) for r in primary]
    def counts(rows):
        return {'planned':len(rows),'created':sum(r.get('root_id') is not None for r in rows),
                'delivery_ready_observed':sum(r.get('end_to_end',{}).get('delivery_ready_observed') is True for r in rows),
                'full':sum(r.get('score',{}).get('whole_answer_quality')=='full' for r in rows),
                'status_counts':dict(Counter(r['status'] for r in rows))}
    return {'all_attempts_planned':len(primary)+len(technical),
            'all_attempts_created':sum(r.get('root_id') is not None for r in primary+technical),
            'first_attempts':counts(first),'first_attempts_by_variant':{v:counts([r for r in first if r['variant']==v]) for v in ('A','B')},
            'technical_attempts':len(technical),'retry_does_not_erase_first_attempt':True,
            'technical_attempt_outcomes':dict(Counter(r.get('observed_attempt_outcome') or r['status'] for r in technical)),
            'delivery_ready_is_not_quality_full':True}


def summarize(reader,root,manifest,progress=False):
    """统一汇总原完整矩阵和显式续跑，金融累计与主指标来源分别核对。"""
    from agent_final_continuation import validate_continuation
    completion='completion_contract' in manifest
    if completion:
        from agent_final_completion import validate_completion
        context=validate_completion(manifest)
        if manifest.get('evaluation_policy')!=COMPLETION_EVALUATION_POLICY:
            raise PermissionError('completion evaluation policy must explicitly remove the fixed full-count floor')
    else:context=validate_continuation(manifest)
    root=Path(root)
    closure=closure_status(reader,root,manifest)
    if not progress and not closure['closed']:raise RuntimeError('formal batch not independently closed; final metrics forbidden')
    anchor=context['anchor_manifest'] if context else manifest
    history=historical_usage(reader,anchor)
    order=context['full_execution_order'] if context else manifest['execution_order']
    validate_matrix(order)
    ledger=read_ledger(reader,Path(manifest['runtime_root'])/'budget.sqlite3')
    config=(ledger or {}).get('metadata',{}).get('config',{})
    expected_carry=context['carried_usage'] if context else history
    issues=[]
    if ledger is None:issues.append('generation_ledger_not_created')
    elif config.get('carried_usage',dict.fromkeys(history,0))!=expected_carry:issues.append('historical_carry_mismatch')
    if completion and ledger is not None and config.get('enforce_batch_seconds') is not False:
        issues.append('financial_only_ledger_policy_mismatch')
    if manifest['ledger_limits'].get('carried_usage',dict.fromkeys(history,0))!=expected_carry:issues.append('manifest_carry_mismatch')
    old_ledger=None;technical=[];retained={};prior_closures={}
    if completion:
        retained,technical,old_totals,prior_closures=completion_artifacts(reader,manifest,context)
    elif context:
        old_root=Path(context['anchor_root'])
        old_ledger=read_ledger(reader,Path(anchor['runtime_root'])/'budget.sqlite3')
        old_closure=closure_status(reader,old_root,anchor)
        for item in context['retained_rows']:
            key=(item['case_id'],item['variant'])
            row=analyze_cell(reader,old_root,anchor,*key,old_ledger,old_closure['controller'])
            row['continuation_provenance']={'retained':True,**item['provenance']}
            row['original_batch_record']=item
            if row['score'].get('whole_answer_quality')!=item['quality']:raise ValueError('retained original grade changed')
            retained[key]=row
        item=context['failed_attempt'];key=(item['case_id'],item['variant'])
        failed=analyze_cell(reader,old_root,anchor,*key,old_ledger,old_closure['controller'])
        failed.update(attempt_kind='technical_attempt',original_batch_record=item,continuation_provenance={'retained':False,**item['provenance']})
        technical.append(failed)
    if context:
        expected_new={tuple(x) for x in context['new_execution_order']}
        if set(retained)&expected_new or set(retained)|expected_new!={tuple(x) for x in order}:
            raise ValueError('retained/new primary matrix does not partition frozen96')
    rows=[]
    for case,v in order:
        row=retained.get((case,v))
        if row is None:
            row=analyze_cell(reader,root,manifest,case,v,ledger,closure['controller'])
            row['continuation_provenance']={'retained':False,'source_root':str(root)}
        rows.append(row)
    current_rows=[r for r in rows if not r.get('continuation_provenance',{}).get('retained')]
    raw=ledger['reservations'] if ledger else None
    root_ids={r['root_id'] for r in current_rows if r['root_id']}
    unassigned=[r['reservation_id'] for r in raw or [] if r['root_id'] not in root_ids]
    if unassigned:issues.append('physical_reservation_root_unassigned')
    for r in rows:
        if r['root_id'] and (r['issues'] or not r.get('usage_reconciled')):
            issues.append(r['expanded_case_id']+'-'+r['variant']+':incomplete_or_inconsistent_evidence')
        if r['score'].get('whole_answer_quality')=='full' and r['end_to_end'].get('delivery_ready_observed') is not True:
            issues.append(r['expanded_case_id']+'-'+r['variant']+':full_grade_without_observed_complete_delivery')
    new_rows_complete=all(r['root_id'] is not None and r.get('usage_reconciled') is True for r in current_rows)
    totals=ledger_totals(ledger)
    if not completion:old_totals=ledger_totals(old_ledger) if context else None
    known=totals['known'].copy();charged=totals['held'].copy();pending=totals['pending']
    if completion:known['requests']=totals['settled_request_count']
    if known['tokens'] is not None:
        settled_cells=sum(r.get('verified_settled_tokens_lower_bound') or 0 for r in current_rows if r['root_id'])
        if settled_cells!=known['tokens']:issues.append('verified_cell_sum_does_not_equal_settled_ledger_sum')
    if context:
        clock=context['clock'];origin=stamp(clock['round_started_at']);prior_seconds=clock['pre_round_carried_seconds']
        if not completion and (old_totals['pending'] is None or old_totals['pending']):issues.append('anchor_physical_usage_incomplete')
        financial_basis=old_totals['held'] if completion else old_totals['known']
        if financial_basis!= {k:expected_carry[k]-history[k] for k in ('tokens','requests','nano_yuan')}:
            issues.append('anchor_financial_carry_mismatch')
    else:
        origin=stamp((ledger or {}).get('metadata',{}).get('started_at'))
        prior_seconds=history['seconds']
    close=closure['closed_at_unix']
    elapsed=math.ceil(close-origin) if closure['closed'] and close is not None and origin is not None and close>=origin else None
    if origin is None:issues.append('original_round_clock_origin_missing')
    if closure['closed'] and elapsed is None:issues.append('closed_round_wall_clock_unmeasurable')
    known['seconds']=elapsed;charged['seconds']=elapsed
    known_basis={k:history[k]+old_totals['known'][k] if old_totals['known'][k] is not None else None
                 for k in ('tokens','requests','nano_yuan')} if completion else expected_carry
    cumulative={k:known_basis[k]+v if v is not None and known_basis[k] is not None else None
                for k,v in known.items() if k!='seconds'}
    cumulative_held={k:expected_carry[k]+v if v is not None else None for k,v in charged.items() if k!='seconds'}
    cumulative['seconds']=prior_seconds+elapsed if elapsed is not None else None
    cumulative_held['seconds']=cumulative['seconds']
    round_known={k:((old_totals['known'][k] if context else 0)+known[k])
                 if known[k] is not None and (not context or old_totals['known'][k] is not None) else None
                 for k in ('tokens','requests','nano_yuan')}
    round_known['seconds']=elapsed
    limits=manifest['ledger_limits'] if completion else anchor['ledger_limits']
    financial_only=bool(completion and context['clock'].get('policy')=='financial_only')
    cap={'tokens':limits['batch_tokens'],'requests':limits['batch_requests'],
         'nano_yuan':int(Decimal(str(limits['batch_yuan']))*Decimal(1000000000)),
         'seconds':None if financial_only else limits['batch_seconds']}
    cap_checks={k:None if cumulative_held[k] is None or cap[k] is None else cumulative_held[k]<=cap[k] for k in cap}
    if any(value is False for value in cap_checks.values()):issues.append('original_cumulative_budget_exceeded')
    counts={v:{'planned':48,'created':sum(r['root_id'] is not None for r in rows if r['variant']==v),
              'execution_status':dict(Counter(r['status'] for r in rows if r['variant']==v)),
              'quality':dict(Counter(r['score'].get('whole_answer_quality') or ('not_created' if not r['root_id'] else 'not_scored') for r in rows if r['variant']==v))} for v in ('A','B')}
    stats=aggregate(rows,require_delivery_ready=completion)
    terminal_stats=aggregate(rows) if completion else None
    complete=all(r['root_id'] is not None and r['status'] in TERMINAL for r in rows)
    full={v:counts[v]['quality'].get('full',0) for v in ('A','B')}
    scores_complete=all(r['score'].get('whole_answer_quality') in {'full','partial_usable','unusable','failed'} for r in rows)
    severe=[r['expanded_case_id']+'-'+r['variant'] for r in rows if any(r['score'].get(k) is True for k in ('unauthorized_delivery_write','forged_success'))]
    required_full=full['A'] if completion else max(44,full['A'])
    relative_quality=bool(scores_complete and full['B']>=full['A'])
    quality_ok=bool(complete and scores_complete and full['B']>=required_full and not severe)
    changes={k:stats['overall'][k]['change'] for k in ('latency_seconds','tokens')}
    both_lower=all(changes[k] and changes[k]['B_minus_A']<0 for k in changes)
    sensitivity=stats['matched_full_sensitivity']
    sensitivity_lower=bool(sensitivity['matched_full_pairs'] and all(sensitivity[k] and sensitivity[k]['B_minus_A']<0 for k in ('latency_seconds','tokens')))
    carried_pending=old_totals['pending'] if completion else []
    primary_supported=bool(not progress and closure['closed'] and closure['cleanup_all_passed'] and complete and quality_ok
                 and not issues and pending==[] and both_lower and sensitivity_lower)
    all_attempt_actual_complete=bool(pending==[] and (carried_pending==[] and new_rows_complete
        and all(r.get('usage_reconciled') is True for r in technical)
        and all(known[key] is not None for key in ('tokens','requests','nano_yuan')) if completion else True))
    success=bool(primary_supported and all_attempt_actual_complete)
    primary_token_complete=bool(complete and all(r.get('usage_reconciled') is True for r in rows))
    primary_terminal_complete=bool(complete and all(type(r['end_to_end'].get('seconds')) in (int,float)
        and math.isfinite(r['end_to_end']['seconds']) and r['end_to_end']['seconds']>=0 for r in rows))
    primary_measurement_complete=bool(primary_token_complete and primary_terminal_complete)
    allowance=new_allowance(limits,history)
    if financial_only:allowance['seconds']=None
    pending_all=None if pending is None or carried_pending is None else carried_pending+pending
    return {'status':'progress_only' if progress else 'closed_complete_matrix' if complete else 'closed_incomplete_matrix',
            'analysis_at':datetime.now(timezone.utc).isoformat(),'formal_roots_planned':96,'base_cases':24,'rounds':2,
            'baseline':'frozen deployment-equivalent source under common controlled conditions',
            'candidate':'frozen prompt candidate under the same controlled conditions',
            'counts':counts,'created_roots':sum(r['root_id'] is not None for r in rows),
            'primary_measurement_complete':primary_measurement_complete,
            'primary_token_measurement_complete':primary_token_complete,
            'primary_terminal_duration_measurement_complete':primary_terminal_complete,
            'evaluation_policy':manifest.get('evaluation_policy') if completion else None,
            'batch_status':(reader.json(Path(manifest['runtime_root'])/'batch-results.json',{}) or {}).get('status'),
            'batch_stop_reason':(reader.json(Path(manifest['runtime_root'])/'batch-results.json',{}) or {}).get('stop_reason'),
            'closure':{k:v for k,v in closure.items() if k!='controller'},
            'accounting':{'history_carried_once':history,'current_ledger_carried_once':expected_carry,
                'prior_round_known_settled_usage':old_totals['known'] if context else None,
                'prior_round_charged_or_reserved':old_totals['held'] if context else None,
                'new_known_settled_record_usage':known,'new_charged_or_reserved':charged,
                'round_all_attempt_known_usage':round_known,
                'cumulative_known_settled_record_usage':cumulative,'cumulative_charged_or_reserved':cumulative_held,
                'unknown_or_pending_count':len(pending_all) if pending_all is not None else None,
                'new_unknown_or_pending_count':len(pending) if pending is not None else None,
                'prior_unknown_or_pending_count':len(carried_pending) if carried_pending is not None else None,
                'prior_unknown_or_pending_reservations':carried_pending,
                'physical_status_counts':totals['status_counts'],
                'prior_physical_status_counts':old_totals['status_counts'] if context else None,
                'new_physical_sent_requests':totals['physical_sent_request_count'],
                'prior_physical_sent_requests':old_totals['physical_sent_request_count'] if context else None,
                'all_attempt_actual_cost_complete':all_attempt_actual_complete,
                'all_new_actual_usage_known':bool(raw is not None and new_rows_complete and not pending and not issues),
                'new_actual_tokens_complete':known['tokens'] if raw is not None and new_rows_complete and not pending and not issues else None,
                'origin_qualification_start':origin,'closure_at':close,'elapsed_ceil_includes_qualification_wait_scoring_cleanup':elapsed,
                'cumulative_seconds_formula':'pre_round_carried_seconds + ceil(final_close - original_round_T0)',
                'cost_kind':'configured_frozen_rate_estimate_not_supplier_invoice','unassigned_reservations':unassigned,
                'ledger_halt_reason':(ledger or {}).get('metadata',{}).get('halt_reason'),
                'new_allowance':allowance,
                'original_cumulative_limits':cap,'within_original_cumulative_limits':cap_checks,
                'timing_policy':'financial_only' if financial_only else 'absolute_deadline',
                'global_time_limit_applicable':not financial_only,
                'historical_batch_seconds_not_a_current_cap':limits['batch_seconds'] if financial_only else None,
                'known_usage_excludes_unresolved_prior_reservations':completion,
                'old_balance_does_not_authorize_replay':True},
            'quality':{'A_full':full['A'],'B_full':full['B'],'required_B_full':required_full,
                       'fixed_minimum_B_full':None if completion else 44,
                       'relative_quality_non_regression':relative_quality,
                       'scores_complete':scores_complete,'quality_gate_passed':quality_ok,'severe_violation_cells':severe,
                       'completed_status_does_not_imply_full':True},
            'statistics':stats,'terminal_outcome_statistics':terminal_stats,
            'efficiency_conclusion':{'authorized_by_complete_evidence':success,
                'primary_comparison_supported':primary_supported,
                'primary_comparison_scope':'fixed96_primary_observations_excluding_extra_technical_attempt_costs',
                'all_attempt_actual_cost_complete':all_attempt_actual_complete,
                'whole_set_both_lower':bool(both_lower),'matched_full_pair_direction_supports_both_lower':sensitivity_lower,
                'no_success_claim_on_incomplete_matrix_or_quality_failure':True,'no_population_or_P95_claim':True},
            'rows':rows,'technical_attempts':technical,'attempt_reliability':attempt_reliability(rows,technical),
            'failure_costs':[{'case_id':r['expanded_case_id'],'variant':r['variant'],
                'attempt_kind':r.get('attempt_kind','primary'),'status':r['status'],
                'observed_attempt_outcome':r.get('observed_attempt_outcome'),
                'actual_task_family_tokens':r.get('actual_task_family_tokens'),
                'verified_settled_tokens_lower_bound':r.get('verified_settled_tokens_lower_bound'),
                'verified_settled_nano_yuan_lower_bound':r.get('verified_settled_nano_yuan_lower_bound'),
                'pending_reservations':r.get('pending_reservations')} for r in rows+technical if r.get('root_id')
                and (r['status']!='completed' or r.get('attempt_kind')=='technical_attempt'
                     or r.get('score',{}).get('whole_answer_quality')!='full')],
            'continuation':{'enabled':bool(context),'retained_roots':len(retained),
                            'new_roots_planned':len(current_rows),'primary_denominator':96,
                            'multiple_immutable_anchors':completion,
                            'prior_closures_preserved':{name:{k:v for k,v in old.items() if k!='controller'}
                                                       for name,old in prior_closures.items()},
                            'technical_failure_kept_separately':bool(technical)},
            'issues':sorted(set(issues)),'input_sha256':dict(reader.hashes),'read_only_sqlite_connections':reader.connections,
            'model_calls':0,'network_calls':0,'service_operations':0,'business_source_modified':False}


def report_text(summary):
    """展示固定分母、额外失败、财务和结论边界，不在缺格时填零。"""
    a=summary['accounting'];q=summary['quality'];s=summary['statistics'];rel=summary['attempt_reliability']
    lines=['# 固定任务集A/B结果（只读汇总）','',
        '状态：'+summary['status']+'；主分母已创建 '+str(summary['created_roots'])+'/96。',
        '保留旧格 '+str(summary['continuation']['retained_roots'])+'；新格计划 '+str(summary['continuation']['new_roots_planned'])+'；全attempt计划 '+str(rel['all_attempts_planned'])+'。未创建、未知和不完整用量保留null。',
        '', '|版本|已创建/48|full|质量标签|','|---|---:|---:|---|']
    for v in ('A','B'):
        c=summary['counts'][v]
        lines.append('|'+v+'|'+str(c['created'])+'|'+str(c['quality'].get('full',0))+'|'+json.dumps(c['quality'],ensure_ascii=False)+'|')
    quality_line=('本轮不设B full固定下限；A/B完整质量标签均披露。相对质量未退化 '+str(q['relative_quality_non_regression'])+
                  '；用于等质效率结论的质量条件 '+str(q['quality_gate_passed'])+'。') if summary.get('evaluation_policy') else (
                  'B full门槛 '+str(q['required_B_full'])+'/48；质量门 '+str(q['quality_gate_passed'])+'。')
    lines+=['',quality_line,
             '', '|指标|A每题等权均值|B每题等权均值|B相对A变化|','|---|---:|---:|---:|']
    for key,label in [('latency_seconds','成功交付端到端秒' if summary.get('evaluation_policy') else '后端端到端秒'),('tokens','主96格任务族总Token')]:
        obj=s['overall'][key];means=obj['equal_case_weight_mean'];d=obj['change']
        show=lambda x:'N/A' if x is None else f'{x:.3f}'
        lines.append('|'+label+'|'+show(means['A'])+'|'+show(means['B'])+'|'+('N/A' if d is None else f"{d['B_minus_A_percent']:+.2f}%")+'|')
    if summary.get('terminal_outcome_statistics') is not None:
        terminal=summary['terminal_outcome_statistics']['overall']['latency_seconds']
        means=terminal['equal_case_weight_mean'];d=terminal['change']
        lines+=['','主96格任务创建→终态耗时（包含partial/failed，非成功交付时延）：A '+show(means['A'])+
                ' 秒；B '+show(means['B'])+' 秒；变化 '+('N/A' if d is None else f"{d['B_minus_A_percent']:+.2f}%")+'。',
                '主96格测量完整 '+str(summary['primary_measurement_complete'])+'；全部attempt实际成本完整 '+
                str(a['all_attempt_actual_cost_complete'])+'。旧技术attempt未决不改变主96格分母。',
                '主96格等质效率对比得到支持 '+str(summary['efficiency_conclusion']['primary_comparison_supported'])+
                '。该判断只覆盖主96格，额外技术attempt费用和未知预留单独披露。']
    lines+=['','效率结论：'+('完整证据满足质量门，两个整体资源指标均下降，full配对方向一致。' if summary['efficiency_conclusion']['authorized_by_complete_evidence'] else '现有完整性、质量、收据或方向条件不足，不能宣称同等质量下两个整体资源指标均下降。'),
        '', '原历史仅计一次：'+json.dumps(a['history_carried_once'],ensure_ascii=False)+'。',
        '新运行已结算：'+json.dumps(a['new_known_settled_record_usage'],ensure_ascii=False)+'；含未决预留：'+json.dumps(a['new_charged_or_reserved'],ensure_ascii=False)+'。',
        '同轮全部attempt已知结算下限（含旧失败；未知不补零）：'+json.dumps(a['round_all_attempt_known_usage'],ensure_ascii=False)+'。',
        '累计已知结算下限（各源仅计一次）：'+json.dumps(a['cumulative_known_settled_record_usage'],ensure_ascii=False)+'。费用为冻结费率估算。',
        '未决请求：本轮 '+str(a['new_unknown_or_pending_count'])+'；历史 '+str(a['prior_unknown_or_pending_count'])+
            '。保守累计占用：'+json.dumps(a['cumulative_charged_or_reserved'],ensure_ascii=False)+'。',
        '首次attempt可靠性：'+json.dumps(rel,ensure_ascii=False)+'。技术失败保留于technical_attempts，不改写旧记录、不作新的语义评分。',
        '失败或非full运行的成本与未知下限：'+json.dumps(summary['failure_costs'],ensure_ascii=False)+'。',
        '', '唯一时延口径为根task_created→task_finished；updated_at不能代替，失败终止耗时不代表成功交付。完整任务族计入全部角色和缓存输入Token，reasoning明细不重复叠加。',
        '每题R1/R2均值后24题等权；缺格不按0填充、不通过删格宣称全矩阵。双方full的共同配对仅供敏感性检查。',
        '局部退化：'+json.dumps(s['local_regressions'],ensure_ascii=False)+'。',
        '证据问题：'+json.dumps(summary['issues'],ensure_ascii=False)+'。',
        '不外推P95或真实外部流量；只读已关闭本地工件和SQLite，真实模型/网络/服务操作0。']
    if a.get('timing_policy')=='financial_only':
        lines+=['本轮按财务上限或全部任务终态停止；全局时间上限不适用。累计耗时仍从原T0计算，历史时间配置不构成新时间授权。每根任务与收尾的边界仍由独立控制器证据核验。']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    """从显式manifest定位私有结果，公开脚本不包含机器路径和历史数值。"""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--progress',action='store_true')
    args=p.parse_args(argv);reader=Reader();manifest=reader.json(args.manifest)
    root=Path(manifest['experiment_root']).resolve(strict=True);output=args.output_dir.resolve()
    if not output.is_relative_to(root/'formal-analysis'):raise PermissionError('output must stay under new private formal-analysis root')
    if output.exists():raise FileExistsError('refuse overwrite; choose a fresh output directory')
    result=summarize(reader,root,manifest,args.progress)
    output.mkdir(parents=True,exist_ok=False)
    def write(name,obj):
        with (output/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
    write('metrics.json',result);write('accounting.json',result['accounting'])
    write('failures.json',{'issues':result['issues'],'technical_attempts':result['technical_attempts'],
        'cells':[r for r in result['rows'] if r['root_id'] and (r['status']!='completed'
            or r['score'].get('whole_answer_quality')!='full' or r['issues'] or r.get('failed_actions') or r.get('pending_reservations'))]})
    (output/'REPORT.md').write_text(report_text(result))
    write('SEAL.json',{'kind':'read_only_formal_analysis','files':{x.name:{'sha256':sha(x.read_bytes()),'bytes':x.stat().st_size} for x in output.iterdir() if x.is_file()}})
    print(json.dumps({'status':result['status'],'created':result['created_roots'],'planned':96,
        'efficiency_claim':result['efficiency_conclusion']['authorized_by_complete_evidence'],'output':str(output)},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
