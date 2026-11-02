"""冻结缓存/历史工厂的合成资格，不读取真实论文、模型权重或部署数据库。"""
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx

from experiments.mechanism_pilot.resources import (
    FrozenResourcesFactory, FrozenPageResources, file_sha256, runtime_namespace,
    source_manifest_entry, validate_runtime,
)
from experiments.mechanism_pilot.memory import execute_memory_case, load_memory_case
from research_loop.config import Settings
from research_loop.domain.memory import HistoryRead
from research_loop.domain.retrieval import RetrievalQuery
from research_loop.domain.tasks import TaskCreate
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.services.memory import MemoryService
from research_loop.services.retrieval import RetrievalRuntime
from research_loop.services.tasks import TaskService
from research_loop.storage.files import FileStore
from research_loop.storage.memory import dump
from research_loop.storage.mysql import Database


def settings_for(root):
    return Settings(project_root=root, data_dir=root/'data', output_dir=root/'output',
        database_url='sqlite:///'+str(root/'runtime.sqlite3'),
        deepseek_api_key=None, deepseek_base_url='https://fixture.invalid',
        deepseek_model='deepseek-flash', model_max_output_tokens=4096, context_output_reserve=4096,
        discovery_backends={}, web_search_enabled=False, web_fetch_enabled=False,
        retrieval_enabled=True, retrieval_route='located_fp16', retrieval_reranker_backend='mps_fp16',
        qdrant_url='http://127.0.0.1:16491', retrieval_namespace=runtime_namespace(root),
        max_model_requests=8, reserve_model_requests=0)


class FrozenResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for target in ('socket.socket.connect', 'socket.socket.connect_ex', 'socket.getaddrinfo'):
            blocker = patch(target, side_effect=AssertionError('NETWORK_FORBIDDEN'))
            blocker.start(); self.addCleanup(blocker.stop)

    def material(self):
        origin = self.root/'origin'; origin.mkdir()
        pdf = origin/'original.pdf'; pdf.write_bytes(b'%PDF-EXPLICIT-SYNTHETIC-NOT-A-REAL-PDF')
        content_hash = file_sha256(pdf)
        cfg = hashlib.sha256(json.dumps(settings_for(self.root/'runtime').parser_config, sort_keys=True).encode()).hexdigest()
        key = hashlib.sha256(f'{content_hash}:pymupdf:1:{cfg}'.encode()).hexdigest()
        docid = 'doc:'+key[:32]
        texts = ['First synthetic original page; no science claim. '*50,
                 'Second synthetic original page: a precise statement about synthetic conditions. '*50]
        pages = [{'evidence_id': 'evidence:'+hashlib.sha256((str(i)+text).encode()).hexdigest()[:40],
                  'document_id':docid, 'page_number':i, 'text':text,'parsed_markdown':text,
                  'assets':[],'gaps':[]} for i,text in enumerate(texts,1)]
        parsed=origin/(key+'.json');parsed.write_text(json.dumps({'total_pages':2,'pages':pages,'gaps':[]}))
        case={'case_id':'R01','model_input':{'question':'核验合成条件','draft':'合成稿件。','documents':[
            {'document_id':docid,'paper_id':'synthetic-paper','title':'Synthetic paper',
             'source_sha256':content_hash,'allowed_pages':[2], 'initial_pages':[
                {'physical_page':2,'text':texts[1],'page_text_sha256':hashlib.sha256(texts[1].encode()).hexdigest()}]}]}}
        entry=source_manifest_entry(case,pdf,parsed,settings_for(self.root/'runtime'))
        return case,entry,pages

    def test_copy_bind_only_approved_page_and_exact_original_cache(self):
        case,entry,pages=self.material(); before={p:file_sha256(p) for p in [entry['pdf_path'],entry['parsed_path']]}
        runtime=self.root/'runtime'; runtime.mkdir(); settings=settings_for(runtime)
        manifest=self.root/'manifest.json';manifest.write_text(json.dumps({'schema':'frozen-reflection-sources-v1','cases':[entry]}))
        factory=FrozenResourcesFactory(manifest,file_sha256(manifest))
        resources=factory(runtime,settings,case)
        db=Database(settings.database_url);db.ensure_schema()
        with patch('research_loop.config.get_settings',return_value=settings):
            task=TaskService(db,settings).create(TaskCreate(prompt='只读取附件 frozen-R01 的物理第2页，核验合成稿，不保存。',
                attachments=resources.attachments,save_policy='deny')).task_id
            bound=resources.bind(task,db)
        self.assertEqual(db.list_task_evidence(task),[pages[1]['evidence_id']])
        self.assertEqual(bound['material_identity']['source_sha256'],entry['pdf_sha256'])
        self.assertEqual(bound['read_observation']['returned_pages'],[2])
        self.assertEqual(bound['read_observation']['acquisition']['executed_pdf_parse'],False)
        self.assertEqual(before,{p:file_sha256(p) for p in before})
        self.assertNotEqual(Path(db.get_page(pages[1]['evidence_id'])['document_id']).name,'')
        provider=resources.pdf_factory(settings,resources.files)
        self.addCleanup(provider.close)
        with self.assertRaisesRegex(PermissionError,'PAGE_SCOPE_EXCEEDED'):provider.pages(resources.document,1,1)
        with self.assertRaisesRegex(PermissionError,'ATTACHMENT_IDENTITY'):provider.ingest('https://example.invalid/paper.pdf','synthetic-paper')
        with self.assertRaisesRegex(PermissionError,'DOCUMENT_SCOPE'):resources.validate_scope(task,db,['foreign'])
        with patch.object(RetrievalRuntime,'search',return_value={'native':True}) as native:
            self.assertEqual(resources.retrieval_factory(db,settings).search(task,RetrievalQuery(query='synthetic',document_ids=[resources.document.document_id])),{'native':True})
            self.assertEqual(native.call_count,1)
        self.assertFalse(resources.report()['neural_retrieval_qualified'])

    def test_lexical_manifest_preparation_never_encodes_or_calls_vectors(self):
        case,entry,_=self.material();runtime=self.root/'runtime';runtime.mkdir()
        settings=settings_for(runtime);resources=FrozenPageResources(runtime,settings,case,entry)
        db=Database(settings.database_url);db.ensure_schema()
        with patch('research_loop.config.get_settings',return_value=settings):
            tid=TaskService(db,settings).create(TaskCreate(prompt='只读取附件 frozen-R01 的物理第2页，不保存。',
                attachments=resources.attachments,save_policy='deny')).task_id
            resources.bind(tid,db)
        class Embedding:
            identity={'model_id':'synthetic-local-only','dimension':2}
            tokenizer_id='synthetic-tokenizer'
            @staticmethod
            def offsets(text):return [m.span() for m in re.finditer(r'\S+',text)]
            @staticmethod
            def encode(*args,**kwargs):raise AssertionError('NEURAL_FORBIDDEN')
        report=resources.prepare_lexical_index(tid,db,Embedding())
        self.assertEqual(report['bm25_status'],'ready')
        self.assertEqual(report['vector_status'],'pending')
        self.assertGreater(report['chunk_count'],0)
        self.assertEqual(report['embedding_calls'],0)
        self.assertEqual(report['qdrant_requests'],0)
        self.assertFalse(report['neural_retrieval_qualified'])

    def test_common_factory_lazy_prepares_before_native_search_without_expanding_scope(self):
        case,entry,pages=self.material();runtime=self.root/'runtime';runtime.mkdir()
        model=self.root/'model-manifest.json';model.write_text('{}')
        settings=settings_for(runtime).model_copy(update={'retrieval_model_manifest':model})
        manifest=self.root/'sources.json';manifest.write_text(json.dumps({'schema':'frozen-reflection-sources-v1','cases':[entry]}))
        resources=FrozenResourcesFactory(manifest,file_sha256(manifest),prepare_lexical=True)(runtime,settings,case)
        db=Database(settings.database_url);db.ensure_schema()
        def make_task():
            with patch('research_loop.config.get_settings',return_value=settings):
                return TaskService(db,settings).create(TaskCreate(prompt='只读取附件 frozen-R01 的物理第2页，不保存。',
                    attachments=resources.attachments,save_policy='deny')).task_id
        tid=make_task();resources.bind(tid,db)
        class Embedding:
            identity={'model_id':'synthetic-local-only','dimension':2}
            tokenizer_id='synthetic-tokenizer'
            @staticmethod
            def offsets(text):return [m.span() for m in re.finditer(r'\S+',text)]
            @staticmethod
            def encode(*args,**kwargs):raise AssertionError('NEURAL_FORBIDDEN')
        def native_search(_runtime,owner,request,cancelled):
            self.assertEqual(owner,tid)
            rows=db.fetchall('SELECT bm25_status,vector_status FROM evidence_index_builds')
            self.assertEqual(rows,[{'bm25_status':'ready','vector_status':'pending'}])
            self.assertEqual(db.list_task_evidence(tid),[pages[1]['evidence_id']])
            return {'native':True}
        wrapped=resources.retrieval_factory(db,settings)
        request=RetrievalQuery(query='synthetic',document_ids=[resources.document.document_id])
        self.assertIsNone(resources.lexical_report)
        with patch('experiments.mechanism_pilot.resources.LocalEmbedding',return_value=Embedding()) as loader, \
                patch.object(RetrievalRuntime,'search',autospec=True,side_effect=native_search) as native:
            self.assertEqual(wrapped.search(tid,request),{'native':True})
            self.assertEqual(wrapped.search(tid,request),{'native':True})
            self.assertEqual(loader.call_count,1)
            self.assertEqual(native.call_count,2)
            unauthorized=make_task()
            with self.assertRaisesRegex(PermissionError,'PAGE_SCOPE'):
                wrapped.search(unauthorized,request)
            self.assertEqual(native.call_count,2)
            self.assertEqual(db.list_task_evidence(unauthorized),[])
        self.assertTrue(resources.report()['lexical_preparation_enabled'])
        self.assertEqual(resources.lexical_report['embedding_calls'],0)
        self.assertEqual(resources.lexical_report['qdrant_requests'],0)

    def test_source_or_manifest_drift_rejected_before_copy(self):
        case,entry,_=self.material(); settings=settings_for(self.root/'runtime')
        Path(entry['parsed_path']).write_text('{}')
        with self.assertRaises((ValueError,KeyError)):
            FrozenPageResources(self.root/'runtime',settings,case,entry)
        self.assertFalse(settings.data_dir.exists())
        with self.assertRaises(FileNotFoundError):
            FrozenResourcesFactory(self.root/'absent.json','0'*64)
        manifest=self.root/'manifest.json';manifest.write_text('{}')
        with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):
            FrozenResourcesFactory(manifest,'0'*64)

    def test_runtime_rejects_shared_paths_namespace_route_and_port(self):
        runtime=self.root/'runtime';settings=settings_for(runtime)
        for fields in ({'database_url':'mysql://user:placeholder@127.0.0.1/deployment'},
            {'data_dir':self.root/'outside'},{'qdrant_url':'http://127.0.0.1:6333'},
            {'retrieval_namespace':'researchloop_daily'},{'retrieval_route':'hybrid_rrf'},
            {'web_fetch_enabled':True}):
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                validate_runtime(runtime,settings.model_copy(update=fields))


class MemoryFixtureScenario:
    """替身自行回传计划；入口不预填路线，不把替身当语义质量证据。"""
    def __init__(self,mode='normal'):
        self.mode,self.requests=mode,[]

    def handler(self,request):
        body=json.loads(request.content);self.requests.append(body)
        observations=[]
        for message in body['messages']:
            if message.get('role')=='tool':observations.append(json.loads(message['content']))
        completed=[call['function']['name'] for m in body['messages'] for call in m.get('tool_calls',[])]
        prompt=next(m['content'] for m in body['messages'] if m['role']=='user')
        if 'get_profile' not in completed and 'search_memory' not in completed:
            name,args='search_memory',{'query':'synthetic-v2','source_type':'research_history','limit':5}
        elif 'set_research_plan' not in completed:
            name,args='set_research_plan',{'goal':'按本题数据讨论','task_kind':'memory','goals':[
                {'goal_id':'answer','description':'回答当前问题','min_papers':0}]}
        else:
            name,args='finish_task',{'answer_markdown':'合成替身仅验证历史读取和交付接线。',
                'goals':[{'goal_id':'answer','status':'satisfied','reason':'已真实读取历史工具'}]}
        if self.mode=='tool_error' and name=='set_research_plan':args['goals'][0]['min_papers']=-1
        raw={'id':'synthetic-'+str(len(self.requests)),'choices':[{'index':0,'finish_reason':'tool_calls',
            'message':{'role':'assistant','content':'','reasoning_content':'合成推理',
              'tool_calls':[{'id':'call-'+str(len(self.requests)),'type':'function','function':
                {'name':name,'arguments':dump(args)}}]}}],
            'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}}
        if self.mode=='missing_usage':raw['usage']=None
        if body.get('stream'):
            message=raw['choices'][0]['message'];call=message['tool_calls'][0]
            chunks=[{'id':raw['id'],'choices':[{'index':0,'delta':{'role':'assistant','reasoning_content':'合成推理','tool_calls':[{'index':0,**call}]},'finish_reason':None}]},
                {'id':raw['id'],'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}]},
                {'id':raw['id'],'choices':[],'usage':raw['usage']}]
            return httpx.Response(200,headers={'content-type':'text/event-stream'},content=''.join('data: '+json.dumps(c,ensure_ascii=False)+'\n\n' for c in chunks)+'data: [DONE]\n\n')
        return httpx.Response(200,json=raw)

    def factory(self,settings):
        return DeepSeekProvider('fixture',settings.deepseek_model,settings.deepseek_base_url,
            client=httpx.Client(transport=httpx.MockTransport(self.handler)))


class MemoryResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def case(self,identity='M4'):
        value={'case_id':identity,'prompt':'新会话继续synthetic-v2，说明已确认与未决事项，不找论文、不保存。',
            'history_records':[{'record_key':'synthetic-v1','text':'V1：1/8=12.5%。'},
                {'record_key':'synthetic-v2','text':'V2：1/6≈16.67%；是否绘图尚未决定。'}]}
        if identity!='M4':value={'case_id':identity,'prompt':'沿用上次结果，只讨论，不找论文、不保存。','history':['旧口径固定为1/8。'] if identity!='M1' else []}
        p=self.root/(identity+'.json');p.write_text(json.dumps(value,ensure_ascii=False))
        return p,file_sha256(p)

    def run_case(self,identity='M4',mode='normal'):
        path,sha=self.case(identity);runtime=self.root/(identity+'-'+mode);settings=settings_for(runtime)
        scenario=MemoryFixtureScenario(mode)
        result=execute_memory_case(path,runtime,expected_input_sha256=sha,settings=settings,provider_factory=scenario.factory)
        return result,scenario

    def test_worker_installs_synthetic_history_and_selects_route_via_provider(self):
        result,scenario=self.run_case()
        self.assertIsNone(result['error'],result)
        self.assertIsNone(result['global_stop'],result)
        self.assertEqual(result['view']['status'],'completed',result)
        self.assertEqual(result['actual_plan']['task_kind'],'memory')
        self.assertFalse(result['route_preselected'])
        self.assertTrue(result['history_read_actions'])
        self.assertEqual(len(result['synthetic_history_seeds']),2)
        self.assertTrue(all(not r['model_generated'] for r in result['synthetic_history_seeds']))
        self.assertEqual(len(result['physical_receipts']),len(scenario.requests))
        self.assertEqual(result['strict_usage_summary']['total_tokens'],30*len(scenario.requests))
        self.assertEqual(result['semantic_quality'],'not_automatically_evaluated')
        db=Database('sqlite:///'+str(self.root/'M4-normal'/'runtime.sqlite3'))
        self.assertEqual(db.fetchone('SELECT COUNT(*) AS n FROM papers')['n'],0)
        self.assertEqual(db.fetchone('SELECT COUNT(*) AS n FROM save_receipts')['n'],0)
        self.assertEqual(db.fetchone('SELECT COUNT(*) AS n FROM memory_entries')['n'],0)
        for seed in result['synthetic_history_seeds']:
            self.assertEqual(db.list_model_calls(seed['task_id']),[])
            self.assertNotEqual(seed['conversation_id'],db.fetchone('SELECT conversation_id FROM conversation_links WHERE task_id=?',(result['task_id'],))['conversation_id'])
        read=MemoryService(db).read(result['task_id'],HistoryRead(source_type='research_history',source_id=result['synthetic_history_seeds'][1]['task_id']))
        self.assertIn('V2：1/6',read['content'])

    def test_same_conversation_and_empty_history_use_frozen_setup(self):
        for identity,expected in [('M1',0),('M2',1),('M3',1)]:
            result,_=self.run_case(identity)
            self.assertIsNone(result['error'],result)
            self.assertEqual(len(result['synthetic_history_seeds']),expected)
            self.assertEqual(result['real_model_calls'],0)

    def test_tool_or_usage_first_error_stops_later_send(self):
        for mode,expected in [('missing_usage',1),('tool_error',2)]:
            result,scenario=self.run_case(mode=mode)
            self.assertTrue(result['global_stop'],result)
            self.assertEqual(len(scenario.requests),expected)
            if mode=='missing_usage':self.assertIsNone(result['strict_usage_summary']['total_tokens'])
            else:self.assertEqual(result['first_failed_action']['tool_name'],'set_research_plan')

    def test_real_mode_without_new_authorization_and_labels_rejected(self):
        path,sha=self.case();runtime=self.root/'denied';settings=settings_for(runtime)
        with self.assertRaises(PermissionError):
            execute_memory_case(path,runtime,expected_input_sha256=sha,settings=settings,
                provider_factory=lambda _:self.fail('provider constructed'),execution_kind='real')
        self.assertFalse(runtime.exists())
        value=json.loads(path.read_text());value['expected']='forbidden';path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):load_memory_case(path,file_sha256(path))


if __name__=='__main__':unittest.main()
