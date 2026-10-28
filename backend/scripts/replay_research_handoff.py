"""只读历史输入，在私有副本上回放合同与单个真实工具节点。"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import socket
import argparse
from datetime import datetime
import sqlite3
import sys
from types import SimpleNamespace

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace', type=Path, required=True)
parser.add_argument('--source-db', type=Path, required=True)
parser.add_argument('--task-id', required=True)
parser.add_argument('--model-sequence', type=int, required=True)
parser.add_argument('--action-sequence', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
options=parser.parse_args()
ROOT=options.workspace.resolve()
SOURCE=options.source_db.resolve()
OUT=options.output.resolve()
CHILD=options.task_id
assert SOURCE.is_file() and (ROOT/'backend/research_loop').is_dir()
assert SOURCE.parent not in [OUT,*OUT.parents], 'output must be outside historical source directory'
assert options.action_sequence>0 and options.model_sequence>0
OUT.mkdir(parents=True,exist_ok=False)
network_attempts=[]
def blocked(*args,**kwargs):
    network_attempts.append(True)
    raise AssertionError('network forbidden in offline replay')
socket.create_connection=blocked
socket.getaddrinfo=blocked
socket.socket.connect=blocked
sys.path.insert(0,str(ROOT/'backend'))
from research_loop.storage.mysql import Database
from research_loop.services.research_progress import ResearchProgressService
from research_loop.domain.research_progress import ResearchProgressInput
from research_loop.domain.tasks import SavePolicy
from research_loop.agents.researcher import progress_spec
from research_loop.agents.graph import build_graph
from research_loop.tools.registry import ToolRegistry
from research_loop.tools.base import ToolContext

class NoModel:
    requests=0
    def send(self,*args,**kwargs):
        self.requests+=1
        raise AssertionError('model forbidden in offline replay')
    send_stream=send

model=NoModel()
source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
with sqlite3.connect(SOURCE.as_uri()+'?mode=ro&immutable=1',uri=True) as conn:
    conn.row_factory=sqlite3.Row
    call=dict(conn.execute('SELECT * FROM model_calls WHERE task_id=? AND sequence_no=?',(CHILD,options.model_sequence)).fetchone())
    request=json.loads(call['request_json'])
    response=json.loads(call['response_json'])
    original_state=json.loads(conn.execute('SELECT state_json FROM checkpoints WHERE task_id=?',(CHILD,)).fetchone()[0])
    original_action=dict(conn.execute('SELECT * FROM actions WHERE task_id=? AND sequence_no=?',(CHILD,options.action_sequence)).fetchone())
with sqlite3.connect(SOURCE.as_uri()+'?mode=ro&immutable=1',uri=True) as conn:
    cutoff=datetime.fromisoformat(original_action['created_at'].replace('Z','+00:00'))
    evidence_times=[datetime.fromisoformat(r[0].replace('Z','+00:00')) for r in conn.execute('SELECT returned_at FROM task_evidence WHERE task_id=?',(CHILD,))]
    assert all(t<=cutoff for t in evidence_times), 'future evidence grants cannot be used as precondition'
original=json.loads(original_action['arguments_json'])
corrected=deepcopy(original)
for finding in corrected['findings']:
    finding.pop('finding_id',None)
finding_count=len(original['findings'])
assert finding_count>0 and all(f.get('finding_id') for f in original['findings'])
assert request['research_handoff']['phase']=='requested'
assert request['tools'][0]['function']['name']=='record_research_progress'
assert 'omit it for a new one' in request['tools'][0]['function']['description']

def clone(name):
    path=OUT/(name+'.sqlite3')
    assert not path.exists()
    with sqlite3.connect(SOURCE.as_uri()+'?mode=ro&immutable=1',uri=True) as src,sqlite3.connect(path) as dst:
        src.backup(dst)
        dst.execute('DELETE FROM actions WHERE task_id=? AND sequence_no>=?',(CHILD,options.action_sequence))
        dst.execute('DELETE FROM task_events WHERE task_id=? AND created_at>=?',(CHILD,original_action['created_at']))
    return Database('sqlite:///'+str(path))

def context(db):
    task=db.get_task(CHILD)
    return ToolContext(task_id=CHILD,save_policy=SavePolicy.deny,role='researcher',
        root_task_id=task['root_task_id'],parent_task_id=task['parent_task_id'],services={'papers':SimpleNamespace(db=db)})

results={}
db=clone('tool-contract')
progress=ResearchProgressService(db)
pre=progress.snapshot(CHILD)
assert pre['revision']==0 and len(pre['findings'])==0
obs=ToolRegistry([progress_spec()]).execute('record_research_progress',original,context(db))
assert not obs.ok and obs.kind=='execution_error' and obs.retryable
assert obs.message=='unknown finding_id; omit it for a new finding'
assert pre==progress.snapshot(CHILD)
results['original_tool']={'kind':obs.kind,'retryable':obs.retryable,'message':obs.message,'snapshot_unchanged':True,'new_findings_persisted':0}
obs2=ToolRegistry([progress_spec()]).execute('record_research_progress',corrected,context(db))
assert obs2.ok
candidate=obs2.data['research_progress']
assert len(candidate['findings'])==finding_count and candidate['revision']==1
assert all(f['verification_status']=='unreviewed' for f in candidate['findings'])
assert pre==progress.snapshot(CHILD)
results['corrected_prepare']={'only_change':'remove finding_id from all new findings','candidate_findings':finding_count,'candidate_revision':1,'not_committed_until_graph_action':True,'scientific_quality_assessed':False,'source_permissions_and_identity_from_actual_db':True}

def state(db,args,request_number=None):
    request_number=options.model_sequence if request_number is None else request_number
    s=deepcopy(original_state)
    s.update(actions=s['actions'][:options.action_sequence-1],observations=s['observations'][:options.action_sequence-1],tool_actions=options.action_sequence-1,action_sequence=options.action_sequence-1,
        research_handoff=deepcopy(request['research_handoff']),finished=False,final_reason=None,
        worker_id=None,lease_token=None,gaps=[],model_requests=request_number)
    assistant=deepcopy(response['choices'][0]['message'])
    assistant['tool_calls'][0]['function']['arguments']=json.dumps(args,ensure_ascii=False)
    s['messages']=deepcopy(request['messages'])+[assistant]
    s['pending_tool_calls']=[{'call_id':assistant['tool_calls'][0]['id'],'name':'record_research_progress','arguments':deepcopy(args)}]
    s['pending_tool_scope']=['record_research_progress','finish_research']
    s['research_progress']=ResearchProgressService(db).snapshot(CHILD)
    return s

def node(db,registry):
    graph=build_graph(model,registry,{'papers':SimpleNamespace(db=db)},db=db)
    return graph.nodes['execute_tools'].bound

raw_db=clone('original-graph')
raw_state=state(raw_db,original)
raw_before=ResearchProgressService(raw_db).snapshot(CHILD)
raw_update=node(raw_db,ToolRegistry([progress_spec()])).invoke(raw_state)
assert raw_update['finished'] is True and raw_update['final_reason']=='research readback made no progress'
assert raw_update['research_handoff']['phase']=='unresolved'
assert 'tool_error' not in raw_update['research_handoff']
assert ResearchProgressService(raw_db).snapshot(CHILD)==raw_before
results['original_graph_node']={'finished':True,'final_reason':raw_update['final_reason'],'handoff_phase':'unresolved','tool_error_feedback_slot_used':False,'same_as_historical_stop':True,'findings_committed':0}

fixed_db=clone('corrected-graph')
fixed_state=state(fixed_db,corrected)
fixed_update=node(fixed_db,ToolRegistry([progress_spec()])).invoke(fixed_state)
committed=ResearchProgressService(fixed_db).snapshot(CHILD)
assert fixed_update.get('finished') is not True
assert fixed_update['research_handoff']['phase']=='progress_recorded'
assert committed['revision']==1 and len(committed['findings'])==finding_count
results['corrected_graph_node']={'finished':fixed_update.get('finished',False),'handoff_phase':fixed_update['research_handoff']['phase'],'committed_findings':finding_count,'committed_revision':1,'all_unreviewed':all(f['verification_status']=='unreviewed' for f in committed['findings']),'counterfactual_not_model_behavior':True}

class ClassificationCounterfactual(ToolRegistry):
    def execute(self,name,args,ctx):
        result=super().execute(name,args,ctx)
        if name=='record_research_progress' and not result.ok and result.retryable and result.message=='unknown finding_id; omit it for a new finding':
            return result.model_copy(update={'kind':'validation_error'})
        return result

classified_db=clone('classification-only-counterfactual')
classified_state=state(classified_db,original)
classified_update=node(classified_db,ClassificationCounterfactual([progress_spec()])).invoke(classified_state)
assert classified_update.get('finished') is not True
assert classified_update['research_handoff']['tool_error']['message']==obs.message
assert classified_update['research_handoff']['phase']=='requested'
assert len(ResearchProgressService(classified_db).snapshot(CHILD)['findings'])==0
retry_state={**classified_state,**classified_update}
retry_state.update(model_requests=options.model_sequence+1,pending_tool_scope=['record_research_progress','finish_research'],pending_tool_calls=[{'call_id':'offline-legal-resubmit','name':'record_research_progress','arguments':deepcopy(corrected)}])
# 单个工具节点只回放参数，不发起“下一次模型请求”；新响应体是明确构造的离线反事实。
retry_update=node(classified_db,ClassificationCounterfactual([progress_spec()])).invoke(retry_state)
assert retry_update.get('finished') is not True
assert retry_update['research_handoff']['phase']=='progress_recorded'
assert len(ResearchProgressService(classified_db).snapshot(CHILD)['findings'])==finding_count
results['classification_only_counterfactual']={'original_bad_request_still_rejected':True,'new_findings_after_bad_request':0,'one_error_feedback_handoff_retained':True,'corrected_offline_resubmit_commits':finding_count,'no_new_model_receipt_or_request':True,'counterfactual_only':True}

# 在反事实重试槽内仍提交同一错误，应终止；不允许无限放宽。
repeat_db=clone('classification-repeat-counterfactual')
repeat_state=state(repeat_db,original)
first=node(repeat_db,ClassificationCounterfactual([progress_spec()])).invoke(repeat_state)
repeat_state={**repeat_state,**first}
repeat_state.update(model_requests=options.model_sequence+1,pending_tool_scope=['record_research_progress','finish_research'],pending_tool_calls=[{'call_id':'offline-repeated-invalid','name':'record_research_progress','arguments':deepcopy(original)}])
second=node(repeat_db,ClassificationCounterfactual([progress_spec()])).invoke(repeat_state)
assert second['finished'] is True
assert len(ResearchProgressService(repeat_db).snapshot(CHILD)['findings'])==0
results['classification_repeat_counterfactual']={'finished':True,'reason':second['final_reason'],'findings_committed':0,'bounded_error_feedback':True}

assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
assert model.requests==0 and not network_attempts
summary={'status':'passed','source_database_sha256':source_hash,'source_database_unchanged':True,'child_evidence_grants_before_failure':len(evidence_times),'future_child_evidence_grants_used':0,
    'original_findings_in_failed_payload':finding_count,'paid_or_stub_model_send_invocations':0,'network_attempts':0,
    'replay_scope':'actual tools and execute_tools node only; persisted original response; no model decision execution',
    'fixture_adjustments':['SQLite backup in private directory','remove child actions at or after rejected sequence and subsequent events','reconstruct requested handoff from the original native request','restore pre-tool counters/history','disable expired historical Worker lease for offline tool-node scope'],
    'results':results,'limits':['No scientific correctness scored','No original model retry behavior proven','No latency or Token savings estimated','No business code changed']}
(OUT/'replay-results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
