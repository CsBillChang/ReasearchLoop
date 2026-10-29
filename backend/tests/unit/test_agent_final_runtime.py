"""验证逐格存储边界及全矩阵身份，不接触部署资源或模型。"""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'scripts'),
              str(Path(__file__).resolve().parents[2])]
from agent_final_runtime import CellBoundary, cell_directory, digest, validate_suite_manifest
from harness_ab_boundary import IsolationError
from harness_ab_transport import (deep_reading_request_contract, matrix_request_contract,
                                  visual_request_contract)


class CellBoundaryTests(unittest.TestCase):
    """模拟实际审计事件，覆盖越界路径和数据库及唯一共享例外。"""

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve(); self.runtime=self.root/'runtime'
        self.cell=self.runtime/'cells/S1-R1-A';self.cell.mkdir(parents=True)
        self.boundary=CellBoundary(self.cell,self.runtime)

    def test_current_outputs_and_shared_ledger_allowed(self):
        for path in (self.cell/'data/cache/paper.pdf',self.cell/'output/note.md',self.runtime/'budget.sqlite3',self.runtime/'budget.sqlite3-journal'):
            self.boundary.audit('open',(str(path),'w',os.O_WRONLY|os.O_CREAT))
        for path in (self.cell/'task.sqlite3',self.runtime/'budget.sqlite3'):
            self.boundary.audit('sqlite3.connect',(str(path),))

    def test_other_cells_source_and_control_not_writable(self):
        for path in (self.runtime/'cells/S1-R1-B/task.sqlite3',self.runtime/'cells/S1-R2-A/output/note.md',self.root/'source/backend/x.py',self.root/'semantic-gates/review.json'):
            with self.subTest(path=str(path)),self.assertRaises(IsolationError):
                self.boundary.audit('open',(str(path),'w',os.O_WRONLY|os.O_CREAT))

    def test_only_current_task_database_and_ledger_may_connect(self):
        for path in (self.cell/'unexpected.db',self.runtime/'cells/S2-A/task.sqlite3',self.root/'deployment/task.sqlite3'):
            with self.assertRaises(IsolationError):self.boundary.audit('sqlite3.connect',(str(path),))

    def test_symbolic_redirect_and_environment_are_rejected(self):
        target=self.root/'protected';target.mkdir()
        link=self.cell/'escaped';link.symlink_to(target,target_is_directory=True)
        with self.assertRaises(IsolationError):self.boundary.audit('open',(str(link/'file'),'w',os.O_WRONLY))
        with self.assertRaises(IsolationError):self.boundary.audit('open',(str(self.cell/'.env'),'r',0))
        ledger=self.runtime/'budget.sqlite3';ledger.symlink_to(target/'task.db')
        with self.assertRaises(IsolationError):self.boundary.audit('sqlite3.connect',(str(ledger),))

    def test_stage_traversal_and_constructor_escape_before_creation(self):
        target=self.root/'protected'
        with self.assertRaises(PermissionError):
            cell_directory({'runtime_root':str(self.runtime),'qualification_run':'../../protected'},'S1','A','qualify')
        self.assertFalse(target.exists())
        with self.assertRaises(PermissionError):CellBoundary(target,self.runtime)
        (self.runtime/'redirect').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(PermissionError):
            cell_directory({'runtime_root':str(self.runtime),'qualification_run':'redirect'},'S1','A','qualify')


class FinalManifestTests(unittest.TestCase):
    """冻结矩阵不能被同名路径、旧实例或新顺序替换。"""

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.inputs=self.root/'inputs';self.inputs.mkdir()
        pdf=self.inputs/'paper.pdf';pdf.write_bytes(b'%PDF-synthetic')
        self.cases=[{'case_id':f'S{i}-R{n}','suite_kind':'agent_final_v1','materials':[{'pdf_path':str(pdf),'pdf_sha256':digest(pdf)}]} for n in (1,2) for i in range(24)]
        order=[[c['case_id'],v] for c in self.cases for v in ('A','B')]
        copies=[];materials=[]
        (self.inputs/'materials').mkdir()
        for i in range(54):
            paths=[]
            for ext in ('pdf','json'):
                path=self.inputs/'materials'/f'{i}.{ext}';path.write_bytes(f'synthetic-{i}-{ext}'.encode())
                copies.append({'copy':str(path),'sha256':digest(path)});paths.append(path)
            materials.append({'pdf_path':str(paths[0]),'pdf_sha256':digest(paths[0]),'parsed_path':str(paths[1]),'parsed_cache_sha256':digest(paths[1])})
        for c in self.cases:
            c['materials']=[{'pdf_path':materials[0]['pdf_path'],'pdf_sha256':materials[0]['pdf_sha256']}]
        for name,value in [('cases.json',{'cases':self.cases}),('execution-plan.json',{'execution_order':order}),('material-snapshot.json',{'copies':copies,'materials':materials})]:
            (self.inputs/name).write_text(json.dumps(value))
        self.m={'experiment_kind':'prompt_bundle_v1','final_suite_policy':{'version':'agent_final_v1','semantic_failures':'record_and_continue'},
            'experiment_root':str(self.root),'prepared_input_root':str(self.inputs),'material_project_root':str(self.inputs),'cases_path':str(self.inputs/'cases.json'),
            'material_snapshot_sha256':digest(self.inputs/'material-snapshot.json'),'final_cases_sha256':digest(self.inputs/'cases.json'),'final_plan_sha256':digest(self.inputs/'execution-plan.json'),
            'runtime_root':str(self.root/'runtime'),'semantic_gate_root':str(self.root/'semantic-gates'),'source_roots':{'A':str(self.root/'snapshots/A'),'B':str(self.root/'snapshots/B')},
            'execution_order':order,'ledger_limits':{'allowed_cells':order},'vector_endpoint':'http://127.0.0.1:6431','vector_port':6431,
            'request_contracts':{'visual':visual_request_contract(),
                'deep_reading':deep_reading_request_contract(),'matrix_delivery':matrix_request_contract()}}

    def test_full_matrix_passes_without_release(self):
        self.assertEqual(validate_suite_manifest(self.m)['cells'],96)

    def test_shortened_or_reordered_matrix_rejected(self):
        for order in (self.m['execution_order'][:48],list(reversed(self.m['execution_order']))):
            with self.assertRaises(PermissionError):validate_suite_manifest({**self.m,'execution_order':order})

    def test_changed_input_and_shared_runtime_rejected(self):
        with self.assertRaises(PermissionError):validate_suite_manifest({**self.m,'final_cases_sha256':'0'*64})
        with self.assertRaises(PermissionError):validate_suite_manifest({**self.m,'runtime_root':str(self.inputs/'runtime')})

    def test_deployment_endpoint_or_original_material_root_rejected(self):
        for patch in ({'vector_endpoint':'http://127.0.0.1:6339','vector_port':6339},{'material_project_root':str(self.root/'deployment')}):
            with self.assertRaises(PermissionError):validate_suite_manifest({**self.m,**patch})

    def test_wrong_suite_marker_does_not_fall_back_to_legacy(self):
        with self.assertRaises(PermissionError):validate_suite_manifest({**self.m,'final_suite_policy':{'version':'typo'}})

    def test_mistyped_runtime_and_parsed_cache_mutation_rejected(self):
        with self.assertRaises(PermissionError):validate_suite_manifest({**self.m,'runtime_root':str(self.root/'deployment/data')})
        (self.inputs/'materials/53.json').write_bytes(b'changed')
        with self.assertRaises(PermissionError):validate_suite_manifest(self.m)

    def test_qualification_binds_suite_contract_fields(self):
        from run_read_paper_ab import qualification_configuration
        m={**self.m,'loaded_prompt_contract':{},'batch_id':'synthetic'}
        config=qualification_configuration(m)
        self.assertEqual(config['final_suite_policy'],m['final_suite_policy'])
        self.assertNotEqual(config,qualification_configuration({**m,'final_cases_sha256':'0'*64}))


class ActualLedgerBoundaryTests(unittest.TestCase):
    """在真实Python审计钩子下串联预算，不用独立audit事件代替初始化。"""

    def setUp(self):
        """只创建新临时根，审计关闭后才由测试框架清理。"""
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.runtime=self.root/'runtime'
        self.cell=self.runtime/'cells/synthetic-A';self.cell.mkdir(parents=True)
        self.ledger=self.runtime/'budget.sqlite3'

    @contextmanager
    def guarded(self):
        """注册实际审计钩子，结束时停用该钩子避免污染其它测试。"""
        boundary=CellBoundary(self.cell,self.runtime);boundary.phase='execution'
        sys.addaudithook(boundary.audit)
        try:yield boundary
        finally:boundary.enabled=False

    def test_existing_parent_initialization_reopen_and_settle_under_real_hook(self):
        """复现首格构造顺序并走根登记、预留、发送和结算的真实事务。"""
        from harness_ab_budget import Ledger
        options={'allowed_cells':[['synthetic','A']]}
        with self.guarded() as boundary:
            first=Ledger(self.ledger,**options);first.begin_root('root','synthetic','A')
            item=first.reserve('root',input_upper_bound=16,max_output_tokens=8)
            first.mark_sent(item['reservation_id'])
            first.settle(item['reservation_id'],{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15})
            snapshot=Ledger(self.ledger,**options).snapshot()
        self.assertEqual(snapshot['actual_known_tokens'],15)
        self.assertEqual(snapshot['physical_sent_requests'],1)
        self.assertEqual(snapshot['reservations'][0]['status'],'settled')
        self.assertEqual(boundary.denials,[])
        self.assertEqual({Path(p).resolve() for p in boundary.sqlite_paths},{self.ledger})

    def test_runtime_mkdir_and_other_writes_still_rejected(self):
        """已有目录的直接mkdir与其它共享文件仍不能借修复绕过隔离。"""
        with self.guarded():
            for operation in (lambda:self.runtime.mkdir(exist_ok=True),
                              lambda:(self.runtime/'other.txt').write_text('forbidden'),
                              lambda:(self.runtime/'cells/other/task.sqlite3').write_bytes(b'forbidden'),
                              lambda:sqlite3.connect(self.root/'deployment.sqlite3')):
                with self.assertRaises(IsolationError):operation()
        self.assertFalse((self.runtime/'other.txt').exists())
        self.assertFalse((self.root/'deployment.sqlite3').exists())

    def test_ledger_symbolic_redirect_still_rejected(self):
        """共享账本文件名不能通过链接指向部署或其它目录。"""
        from harness_ab_budget import Ledger
        target=self.root/'protected.sqlite3';self.ledger.symlink_to(target)
        with self.guarded(),self.assertRaises(IsolationError):Ledger(self.ledger)
        self.assertFalse(target.exists())

    def test_unknown_retains_reservation_and_blocks_reopen_send(self):
        """审计与真实SQLite事务并存时未知结果仍完整占用且不得重发。"""
        from harness_ab_budget import Ledger, BudgetHalt
        options={'allowed_cells':[['synthetic','A']]}
        with self.guarded():
            ledger=Ledger(self.ledger,**options);ledger.begin_root('root','synthetic','A')
            item=ledger.reserve('root',input_upper_bound=16,max_output_tokens=8)
            ledger.mark_sent(item['reservation_id']);ledger.unknown(item['reservation_id'],'synthetic no network')
            reopened=Ledger(self.ledger,**options)
            with self.assertRaises(BudgetHalt):reopened.reserve('root',input_upper_bound=16,max_output_tokens=8)
            snapshot=reopened.snapshot()
        self.assertEqual(snapshot['reservations'][0]['status'],'unknown')
        self.assertEqual(snapshot['charged_or_reserved']['tokens'],24)
        self.assertEqual(snapshot['actual_known_tokens'],0)
        self.assertEqual(len(snapshot['reservations']),1)

    def test_carry_caps_and_original_deadline_under_real_hook(self):
        """历史只计一次，重开不能重新开始截止或扩大预算。"""
        from harness_ab_budget import Ledger, BudgetHalt
        now=[10.0];carry={'tokens':100,'requests':1,'nano_yuan':1000,'seconds':5}
        options={'allowed_cells':[['synthetic','A']],'clock':lambda:now[0],'start_time':10.0,
                 'batch_tokens':130,'batch_requests':3,'batch_seconds':30,'carried_usage':carry}
        with self.guarded():
            ledger=Ledger(self.ledger,**options);ledger.begin_root('root','synthetic','A')
            item=ledger.reserve('root',input_upper_bound=16,max_output_tokens=8)
            ledger.mark_sent(item['reservation_id']);ledger.settle(item['reservation_id'],
                {'prompt_tokens':10,'completion_tokens':5,'total_tokens':15})
            reopened=Ledger(self.ledger,**options)
            self.assertEqual(reopened.snapshot()['batch_absolute_deadline'],35)
            self.assertEqual(reopened.snapshot()['charged_or_reserved']['tokens'],115)
            with self.assertRaises(BudgetHalt):reopened.reserve('root',input_upper_bound=16,max_output_tokens=8)
            self.assertEqual(len(reopened.snapshot()['reservations']),1)

    def test_controller_can_create_missing_budget_parent(self):
        """未进入格审计的控制器仍可准备全新账本，保留原创建能力。"""
        from harness_ab_budget import Ledger
        path=self.root/'controller/new/budget.sqlite3'
        self.assertFalse(path.parent.exists());ledger=Ledger(path)
        self.assertTrue(path.is_file());self.assertFalse(ledger.snapshot()['reservations'])


class ExecuteBoundaryChainTests(unittest.TestCase):
    """独立进程串联真实execute与Worker，外部HTTP和神经RAG使用合成替身。"""

    def run_fixture(self,scenario):
        """隔离run_cell的cwd、环境与不可移除审计钩子。"""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            run=subprocess.run([sys.executable,'-I','-B',str(Path(__file__).resolve()),
                '--execute-fixture',scenario,str(root)],capture_output=True,text=True,timeout=90)
            self.assertEqual(run.returncode,0,run.stdout+'\n'+run.stderr)
            return json.loads((root/'fixture-result.json').read_text())

    def test_memory_execute_enters_real_ledger_worker_and_transport(self):
        """实际API创建、格审计、账本和业务Worker均运行，不调用供应商。"""
        result=self.run_fixture('memory')
        self.assertEqual(result['status'],'executed',result)
        self.assertEqual(result['task_status'],'completed',result)
        self.assertGreaterEqual(result['physical_settled'],3)
        self.assertEqual(result['native_success'],result['physical_settled'])
        self.assertEqual(result['actual_tokens'],15*result['physical_settled'])
        self.assertEqual(result['denial_phases'],['guard_selfcheck'])
        self.assertEqual(result['real_socket_connections'],0)

    def test_pdf_visual_child_uses_same_real_ledger_and_native_receipts(self):
        """合成PDF经真实读页与视觉子图，父子原生收据均归同一预算根。"""
        result=self.run_fixture('visual')
        self.assertEqual(result['status'],'executed',result)
        self.assertEqual(result['task_status'],'completed',result)
        self.assertEqual(result['visual_requests'],1,result)
        self.assertGreater(result['descendant_native_success'],0,result)
        self.assertEqual(result['native_success'],result['physical_settled'])
        self.assertEqual(result['budget_root_count'],1)
        self.assertEqual(result['reservation_root_count'],1)
        self.assertEqual(result['actual_tokens'],15*result['physical_settled'])
        self.assertEqual(result['denial_phases'],['guard_selfcheck'])
        self.assertEqual(result['real_socket_connections'],0)

    def test_pdf_deep_read_initial_continuation_exports_exact_completed_note(self):
        """真实精读子图续写后原样建稿、保存和交付，全部原生收据归同一根。"""
        result=self.run_fixture('deep_read')
        self.assertEqual(result['status'],'executed',result)
        self.assertEqual(result['task_status'],'completed',result)
        self.assertEqual(result['deep_max_tokens'],[4096,2048],result)
        self.assertEqual(result['deep_phases'],['initial','continuation'],result)
        self.assertEqual(result['deep_statuses'],['complete'],result)
        self.assertEqual(result['saved_note_count'],1,result)
        self.assertTrue(result['note_exact_merged'],result)
        self.assertTrue(result['note_hashes_match'],result)
        self.assertTrue(result['delivered_note_matches'],result)
        self.assertEqual(result['note_heading_count'],8,result)
        self.assertTrue(result['note_ends_complete'],result)
        self.assertEqual(result['native_success'],result['physical_settled'])
        self.assertGreaterEqual(result['descendant_native_success'],4,result)
        self.assertEqual(result['budget_root_count'],1)
        self.assertEqual(result['reservation_root_count'],1)
        self.assertEqual(result['actual_tokens'],15*result['physical_settled'])
        self.assertEqual(result['denial_phases'],['guard_selfcheck'])
        self.assertEqual(result['real_socket_connections'],0)

    def test_matrix_plan_and_existing_page_evidence_allow_native_text_sse(self):
        """真实计划与读页证据触发既有矩阵非思考请求，仍走原生SSE及请求证明。"""
        result=self.run_fixture('matrix')
        self.assertEqual(result['status'],'executed',result)
        self.assertEqual(result['task_status'],'completed',result)
        self.assertEqual(result['matrix_disabled_sse_requests'],1,result)
        self.assertEqual(result['matrix_plan_requested'],True,result)
        self.assertGreater(result['matrix_source_evidence_count'],0,result)
        self.assertEqual(result['native_success'],result['physical_settled'])
        self.assertEqual(result['actual_tokens'],15*result['physical_settled'])
        self.assertEqual(result['denial_phases'],['guard_selfcheck'])
        self.assertEqual(result['real_socket_connections'],0)


def _execute_fixture(scenario,root):
    """用合成资格门运行实际execute核心路径，不冒充正式96格身份资格。"""
    source=Path(__file__).resolve().parents[3]
    sys.path[:0]=[str(source/'backend'),str(source/'backend/tests')]
    import httpx
    import run_read_paper_ab as runner
    from harness_ab_budget import Ledger
    from research_loop.storage.mysql import Database
    from unit.test_harness_deep_read_contract import INITIAL, CONTINUATION
    root=Path(root).resolve();runtime=root/'runtime';runtime.mkdir()
    case={'case_id':'SYNTHETIC-R1','suite_kind':'agent_final_v1',
          'prompt':'回顾历史偏好，使用中文回答，不保存文件。','materials':[],
          'preconditions':{'history_messages':[{'role':'user','content':'以后默认中文，不保存文件。'}],
                           'save_policy':'deny'}}
    bundle=None
    if scenario in {'visual','deep_read','matrix'}:
        from unit.test_visual_reading import fixture
        fixture_files,pdf,visual_input,pages=fixture(root/'source-fixture')
        parsed=next(fixture_files.parse_dir.glob('*.json'))
        case.update(prompt='只依据附件读取图中表格，必要时查看图像，不保存文件。',
            materials=[{'pdf_path':str(pdf),'pdf_sha256':digest(pdf),'physical_pages':2}],
            preconditions={'save_policy':'deny'})
        bundle={'document_ref':{'paper_id':'synthetic-paper','pdf_path':str(pdf),
             'pdf_sha256':digest(pdf),'parsed_path':str(parsed),'parsed_cache_sha256':digest(parsed),
             'metadata':{'title':'Synthetic fixture only','source':'upload'}},
             'manifest':{'source':{'document_id':visual_input.document_id}}}
        if scenario=='deep_read':
            case.update(prompt='请精读附件，写成八点完整笔记并保存文件，保留结尾（完）。',
                        preconditions={'save_policy':'allow'})
        elif scenario=='matrix':
            case.update(prompt='请读取附件，用结构化证据矩阵列出A与B，不保存文件。')
    cases=root/'cases.json';cases.write_text(json.dumps({'cases':[case]}))
    auth=root/'synthetic-auth.json';auth.write_text(json.dumps({'approved':True}))
    frozen=root/'synthetic-frozen.txt';frozen.write_text('synthetic chain only')
    options={'allowed_cells':[[case['case_id'],'A']],'root_seconds':60,
             'root_requests':16,'max_output_tokens':65536}
    Ledger(runtime/'budget.sqlite3',**options)
    manifest={'experiment_kind':'prompt_bundle_v1','final_suite_policy':{
       'version':'agent_final_v1','semantic_failures':'record_and_continue'},
       'source_roots':{'A':str(source),'B':str(source)},'runtime_root':str(runtime),
       'cases_path':str(cases),'real_execution_released':True,'frozen_files':[{
       'path':str(frozen),'sha256':digest(frozen)}],'github_commit':'synthetic-not-publication',
       'qualification_receipt':str(root/'synthetic-qualification.json'),
       'authorization_path':str(auth),'cell_absolute_deadline':time.time()+60,
       'vector_endpoint':'http://127.0.0.1:16431','vector_port':16431,'official_ip_addresses':[],
       'settings':{'retrieval_enabled':False,'deepseek_model':'deepseek-flash',
                   'model_max_output_tokens':65536,'context_output_reserve':8192,
                   'parser_name':'pymupdf','parser_version':'1','execution_budget_mode':'observe'},
       'variant_settings':{'A':{},'B':{}},'model_manifest_path':str(root/'unused-model.json'),
       'material_project_root':str(root),'vector_manifest_path':str(root/'unused-vectors.json'),
       'ledger_limits':options,'request_contracts':{'visual':visual_request_contract(),
           'deep_reading':deep_reading_request_contract(),'matrix_delivery':matrix_request_contract()}}
    requests=[];connections=[];input_refs=[]
    actual_create=runner.create_bundle_input_task

    def capture_input(*args,**kwargs):
        """保留真实API输入创建，仅记录本例实际取得的附件引用。"""
        report,attachments=actual_create(*args,**kwargs)
        input_refs.extend(a['input_ref'] for a in attachments)
        return report,attachments

    class FixtureRetrieval:
        """外部向量资源替身；本例不测试检索排序或神经推理。"""
        def __init__(self,*args,**kwargs):self.service=None
        def close(self):pass
        def ensure_task_resources(self,*args,**kwargs):return []
        def search(self,*args,**kwargs):raise AssertionError('fixture does not request retrieval')

    class FixtureEmbedding:
        """无权重的外部编码器替身，仅满足入口中性预热形状。"""
        identity={'backend':'synthetic-no-neural'}
        def __init__(self,*args):pass
        def encode(self,texts,**kwargs):return [[1.0,0.0] for _ in texts]

    class FixtureReranker:
        """无权重的外部重排器替身，不报告真实FP16资格。"""
        identity={'backend':'synthetic-no-neural'}
        def __init__(self,*args):pass
        def score(self,query,texts):return [1.0 for _ in texts]

    class FixtureService:
        """仅给未用的检索接口标识，视觉读取仍走原PaperService及节点。"""
        def __init__(self,*args,**kwargs):self.route='synthetic-no-rag'

    def reply(name,args=None,content=''):
        """构造原适配器要求的合成响应及明确usage。"""
        message={'role':'assistant','content':content}
        if name:message['tool_calls']=[{'id':'call-'+uuid4().hex,'type':'function',
            'function':{'name':name,'arguments':json.dumps(args,ensure_ascii=False)}}]
        return {'id':'synthetic-'+uuid4().hex,'model':'deepseek-flash',
                'choices':[{'index':0,'message':message,'finish_reason':'tool_calls' if name else 'stop'}],
                'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}

    def handler(_transport,request):
        """仅替换最底层HTTP响应，BudgetTransport.handle_request及原生适配器保持。"""
        body=json.loads(request.read());requests.append(body)
        done=[c['function']['name'] for m in body['messages'] for c in m.get('tool_calls',[])]
        observations=[json.loads(m['content']) for m in body['messages'] if m['role']=='tool']
        images=any(isinstance(m.get('content'),list) and any(x.get('type')=='image_url'
            for x in m['content']) for m in body['messages'])
        tools={t['function']['name'] for t in body.get('tools',[])}
        if scenario=='deep_read' and body.get('temperature')==.3:
            value=reply(None,content=INITIAL if body['max_tokens']==4096 else CONTINUATION)
        elif scenario=='deep_read' and 'finish_research' in tools:
            if 'read_paper' not in done:
                value=reply('read_paper',{'target_ref':input_refs[0],'start_page':1,'page_count':2})
            else:
                read=next(o['data'] for o in observations if o.get('data',{}).get('pages'))
                value=reply('finish_research',{'status':'completed','findings':['已读合成全文'],
                    'paper_ids':[read['paper_id']],
                    'evidence_ids':[p['evidence_id'] for p in read['pages']]})
        elif scenario=='deep_read' and 'set_research_plan' not in done:
            value=reply('set_research_plan',{'goal':'八点精读并保存','requested_count':1,
                'count_confirmed':True,'task_kind':'compare','input_refs':input_refs,
                'save_policy':'allow','save_requested':True,'deep_reading_requested':True,
                'goals':[{'goal_id':'deep','description':'八点精读','min_papers':1,'requires_fulltext':True}]})
        elif scenario=='deep_read' and 'delegate_research' not in done:
            value=reply('delegate_research',{'subtasks':[{'objective':'八点精读','input_refs':input_refs,
                'goal_ids':['deep'],'mode':'deep_read','start_page':1,'page_count':2}]})
        elif scenario=='deep_read':
            delegated=next(o['data']['subtasks'][0] for o in observations
                if o.get('data',{}).get('subtasks'))
            if 'create_note' not in done:
                value=reply('create_note',{'title':'合成八点精读笔记',
                    'markdown':INITIAL+'\n\n'+CONTINUATION,'evidence_ids':delegated['evidence_ids']})
            else:
                draft=next(o['data'] for o in observations if o.get('data',{}).get('draft_id'))
                if 'save_note' not in done:
                    value=reply('save_note',{'draft_id':draft['draft_id'],'content_hash':draft['content_hash']})
                else:
                    value=reply('finish_task',{'draft_id':draft['draft_id'],
                        'goals':[{'goal_id':'deep','status':'satisfied','reason':'精读完成并保存',
                                  'subtask_ids':[delegated['subtask_id']]}]})
        elif scenario=='matrix' and 'set_research_plan' not in done:
            value=reply('set_research_plan',{'goal':'合成证据矩阵','requested_count':1,
                'count_confirmed':True,'task_kind':'compare','input_refs':input_refs,
                'evidence_matrix_requested':True,
                'goals':[{'goal_id':'matrix','description':'按原文列出证据矩阵',
                          'min_papers':1,'requires_fulltext':True}]})
        elif scenario=='matrix' and 'read_paper' not in done:
            value=reply('read_paper',{'target_ref':input_refs[0],'start_page':1,'page_count':2})
        elif scenario=='matrix':
            value=reply('finish_task',{'answer_markdown':'| 合成项目 | 值 |\n|---|---|\n| A | 12 |\n| B | 24 |',
                'goals':[{'goal_id':'matrix','status':'satisfied','reason':'已读取合成页证据并制表'}]})
        elif images:value=reply(None,content='合成图表观察：A为12，B为24；不代表真实模型质量。')
        elif scenario=='visual' and 'set_research_plan' not in done:
            value=reply('set_research_plan',{'goal':'回答附件图表','requested_count':1,
                'count_confirmed':True,'task_kind':'compare','input_refs':input_refs,
                'goals':[{'goal_id':'visual','description':'读取合成图表','min_papers':1,'requires_fulltext':True}]})
        elif scenario=='visual' and 'read_paper' not in done:
            value=reply('read_paper',{'target_ref':input_refs[0],'start_page':1,'page_count':1})
        elif scenario=='visual' and 'delegate_visual_reading' not in done:
            eid=next(o['data']['pages'][0]['evidence_id'] for o in observations if o.get('data',{}).get('pages'))
            value=reply('delegate_visual_reading',{'evidence_id':eid,'question':'读出A与B的数值。',
                'reason':'核对表格空间关系','trigger':'missing_spatial_relation'})
        elif scenario=='visual':
            value=reply('finish_task',{'answer_markdown':'合成定向回答：A为12，B为24。',
                'goals':[{'goal_id':'visual','status':'satisfied','reason':'原文及视觉观察已取得','subtask_ids':[]}]})
        elif 'set_research_plan' not in done:
            value=reply('set_research_plan',{'goal':'回顾偏好','task_kind':'memory',
                'goals':[{'goal_id':'memory','description':'回顾偏好','min_papers':0}]})
        elif 'get_profile' not in done:value=reply('get_profile',{})
        else:value=reply('finish_task',{'answer_markdown':'已调用记忆读取工具；本例只验证接线，不保存文件。',
            'goals':[{'goal_id':'memory','status':'satisfied','reason':'已实际读取记忆'}]})
        if not body.get('stream'):return httpx.Response(200,json=value)
        choice=value['choices'][0];delta=choice['message']
        for index,call in enumerate(delta.get('tool_calls',[])):call['index']=index
        chunk={**value,'choices':[{'index':0,'delta':delta,'finish_reason':choice['finish_reason']}]}
        return httpx.Response(200,headers={'content-type':'text/event-stream'},
            content='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n')

    def no_socket(event,args):
        """额外拒绝所有实际连接，HTTP替身调用不计网络请求。"""
        if event=='socket.connect':
            connections.append(repr(args[1]));raise AssertionError('real socket forbidden')
    sys.addaudithook(no_socket)
    with ExitStack() as stack:
        # 合成资格/身份门仅供单元接线；不替换被测执行、隔离、账本或Worker。
        for name in ('validate_manifest','validate_active_variant','verify_loaded_supervisor'):
            stack.enter_context(patch.object(runner,name,return_value={'synthetic_identity_only':True}))
        stack.enter_context(patch('harness_ab_pair_gate.validate_prior_pair_gates',return_value={}))
        stack.enter_context(patch.object(httpx.HTTPTransport,'handle_request',handler))
        stack.enter_context(patch.object(runner,'create_bundle_input_task',capture_input))
        if scenario in {'visual','deep_read','matrix'}:
            stack.enter_context(patch('harness_ab_resources.load_source_bundles',return_value={digest(pdf):bundle}))
            stack.enter_context(patch('harness_ab_resources.rebind_bundle',side_effect=lambda value,*_:value))
            stack.enter_context(patch('harness_ab_resources.install_bundle',return_value={'synthetic_external_vectors':True}))
            stack.enter_context(patch('harness_ab_resources.HarnessRetrievalRuntime',FixtureRetrieval))
            stack.enter_context(patch('research_loop.providers.embeddings.LocalEmbedding',FixtureEmbedding))
            stack.enter_context(patch('research_loop.providers.mps_reranker.MpsReranker',FixtureReranker))
            stack.enter_context(patch('research_loop.services.retrieval.RetrievalService',FixtureService))
        report=runner.run_cell(manifest,case['case_id'],'A','execute',credential='synthetic-key-no-network')
    cell=runtime/'cells'/(case['case_id']+'-A')
    snapshot=Ledger(runtime/'budget.sqlite3',**options).snapshot()
    native=json.loads((cell/'model-calls.json').read_text())
    result={'status':report['status'],'task_status':report.get('task_status'),
        'exception_type':report.get('exception_type'),'physical_settled':sum(r['status']=='settled' for r in snapshot['reservations']),
        'native_success':sum(r['status']=='succeeded' for r in native),
        'actual_tokens':snapshot['actual_known_tokens'],'real_socket_connections':len(connections),
        'denial_phases':sorted({r['phase'] for r in report['isolation']['denials']}),
        'synthetic_identity_gates':True,'not_formal_qualification_or_real_model_measurement':True,
        'external_generation_requests':0,'mock_http_requests':len(requests),
        'reservation_receipts_are_synthetic':True}
    root_id=report.get('root_id')
    result.update(visual_requests=sum(any(isinstance(m.get('content'),list) and any(
        x.get('type')=='image_url' for x in m['content']) for m in body['messages']) for body in requests),
        descendant_native_success=sum(r['status']=='succeeded' and r['task_id']!=root_id for r in native),
        budget_root_count=len(snapshot['roots']),
        reservation_root_count=len({r['root_id'] for r in snapshot['reservations']}),
        mocked_external_rag=scenario in {'visual','deep_read','matrix'})
    if scenario=='matrix':
        db=Database('sqlite:///'+str(cell/'task.sqlite3'))
        checkpoint=db.fetchone('SELECT state_json FROM checkpoints WHERE task_id=?',(root_id,))
        state=json.loads(checkpoint['state_json']) if checkpoint else {}
        result.update(matrix_disabled_sse_requests=sum(b.get('stream') is True
            and b.get('temperature')==.1 and b.get('thinking')=={'type':'disabled'} for b in requests),
            matrix_plan_requested=state.get('research_plan',{}).get('evidence_matrix_requested'),
            matrix_source_evidence_count=len(state.get('evidence_ids',[])))
    if scenario=='deep_read':
        db=Database('sqlite:///'+str(cell/'task.sqlite3'))
        rows=db.fetchall('SELECT d.markdown,d.content_hash,s.path,e.file_sha256 FROM drafts d '
            'JOIN save_receipts s ON s.draft_id=d.draft_id AND s.content_hash=d.content_hash '
            'JOIN note_exports e ON e.draft_id=d.draft_id AND e.content_hash=d.content_hash WHERE d.task_id=?',(root_id,))
        note=rows[0] if rows else {}
        markdown=note.get('markdown','')
        children=db.list_subtasks(root_id)
        task=db.get_task(root_id)
        delivered=json.loads(task['result_json']) if task and task.get('result_json') else {}
        result.update(deep_max_tokens=[b['max_tokens'] for b in requests if b.get('temperature')==.3],
            deep_phases=[json.loads(r['request_json']).get('deep_reading_phase') for r in snapshot['reservations']
                         if json.loads(r['request_json']).get('component')=='dpr_deep_reading'],
            deep_statuses=[(c.get('result') or {}).get('result',{}).get('deep_reading_status') for c in children],
            saved_note_count=len(rows),note_exact_merged=markdown==INITIAL+'\n\n'+CONTINUATION,
            note_heading_count=markdown.count('## '),note_ends_complete=markdown.endswith('（完）'),
            note_hashes_match=bool(note) and digest(Path(note['path']))==note['file_sha256']==note['content_hash'],
            delivered_note_matches=delivered.get('answer_markdown')==markdown)
    (root/'fixture-result.json').write_text(json.dumps(result))
    if report['status']!='executed':print((cell/'failure.log').read_text())


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--execute-fixture':_execute_fixture(sys.argv[2],Path(sys.argv[3]))
    else:unittest.main()
