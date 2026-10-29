"""注册工具的真实参数校验和54题录可达性；只用合成目录与独立SQLite。"""
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BACKEND=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(BACKEND/'scripts'),str(BACKEND)]
from agent_final_pdf_acquisition import qualify_registered_catalog_search,validate_source_acquisition_contract
from prompt_bundle_cases import install_catalog
from research_loop.services.papers import PaperService
from research_loop.storage.mysql import Database
from research_loop.tools.base import ToolContext,execute_spec
from research_loop.tools.papers import specs
from research_loop.domain.tasks import SavePolicy


class Discovery:
    def search(self,*args,**kwargs):
        raise AssertionError('external discovery forbidden')


class RegisteredCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        materials=[];catalog=[];bundles={};self.entries=[]
        for i in range(54):
            digest=sha256(('synthetic-catalog-'+str(i)).encode()).hexdigest()
            materials.append({'material_id':f'S{i}','pdf_path':f'/synthetic/not-opened/{i}.pdf',
                              'pdf_sha256':digest,'physical_pages':1})
            source='local' if i>=52 else 'arxiv' if i%2 else 'cvpr'
            bundles[digest]={'document_ref':{'paper_id':f'paper-{i:02}','metadata':{'source':source}}}
            catalog.append({'material_index':i,'title':f'Neutral study {i} distincttopic{i}',
                'abstract':f'Synthetic research method with distincttopic{i}.','authors':['Synthetic Author'],
                'source_url':f'https://example.invalid/{i}'})
            self.entries.append({'material_index':i,'paper_id':f'paper-{i:02}','pdf_sha256':digest})
        self.case={'suite_kind':'agent_final_v1','materials':materials,
                   'preconditions':{'catalog':catalog,'save_policy':'deny'}}
        install_catalog(self.stack,Discovery,self.case,bundles)
        self.discovery=Discovery()
        self.db=Database('sqlite:///'+str(Path(self.temp.name)/'task.sqlite3'))
        self.db.ensure_schema();self.db.create_task('root',None,'synthetic','neutral task','deny',[])
        self.papers=PaperService(self.db,self.discovery,object())

    @staticmethod
    def require(value,reason):
        if not value:raise PermissionError(reason)

    def qualify(self):
        return qualify_registered_catalog_search(self.papers,'root',self.case,self.entries,self.require)

    def test_registered_legal_queries_reach_all54_with_native_grants(self):
        proof=self.qualify()
        self.assertEqual(proof['status'],'passed_registered_tool_contract')
        self.assertEqual(len(proof['exact_title_queries']),54)
        self.assertEqual(len(proof['source_filtered_exact_title_queries']),54)
        self.assertEqual(proof['source_counts']['local'],2)
        self.assertEqual(proof['model_calls'],0)
        self.assertTrue(proof['local_is_not_all54_alias'])
        self.assertEqual({r['paper_id'] for r in proof['exact_title_queries']},
                         {r['paper_id'] for r in self.entries})
        self.assertTrue(all(self.db.task_has_paper('root',r['paper_id']) for r in self.entries))
        self.assertFalse(self.db.fetchall('SELECT * FROM task_evidence'))
        self.assertFalse(self.db.fetchall('SELECT * FROM task_inputs'))
        self.assertTrue(all(r['arguments']['limit']<=20 for r in proof['exact_title_queries']))

    def test_native_bulk54_does_not_pass_registered_limit54(self):
        self.assertEqual(len(self.papers.search('research method',limit=54,task_id='root').candidates),54)
        spec=next(s for s in specs() if s.name=='search_papers')
        result=execute_spec(spec,{'query':'research method','limit':54},
            ToolContext(task_id='root',save_policy=SavePolicy.deny,services={'papers':self.papers}))
        self.assertEqual(result.kind,'validation_error');self.assertFalse(result.ok)

    def test_wrong_pdf_binding_rejected_before_claiming_population(self):
        self.entries[0]['pdf_sha256']='0'*64
        with self.assertRaisesRegex(PermissionError,'source_identity_changed'):self.qualify()

    def test_tail_missing_from_real_handler_cannot_claim54(self):
        original=self.discovery.search
        def missing(*args,**kwargs):
            value=original(*args,**kwargs)
            return value.model_copy(update={'candidates':[c for c in value.candidates if c.paper_id!='paper-53']})
        with patch.object(self.discovery,'search',side_effect=missing):
            with self.assertRaisesRegex(PermissionError,'exact_title_not_reachable'):self.qualify()

    def test_ignoring_origin_filter_is_rejected(self):
        original=self.discovery.search
        def ignore_filter(query,date_from=None,date_to=None,sources=None,limit=5):
            return original(query,date_from,date_to,None,limit)
        with patch.object(self.discovery,'search',side_effect=ignore_filter):
            with self.assertRaisesRegex(PermissionError,'source_filter_changed'):self.qualify()

    def test_v4_cannot_omit_pdf_acquisition_contract(self):
        with self.assertRaisesRegex(PermissionError,'CONTRACT_REQUIRED'):
            validate_source_acquisition_contract({'completion_contract':{'version':'agent_final_completion_v4'},
                                                  'final_suite_policy':{'version':'agent_final_v1'}})

    def test_qualification_configuration_binds_observation_policy(self):
        from run_read_paper_ab import qualification_configuration
        old={'experiment_kind':'unpaired'}
        self.assertNotIn('observation_policy',qualification_configuration(old))
        new={**old,'observation_policy':{'version':'bound-test','non_delivery_execution_complete':False}}
        bound=deepcopy(qualification_configuration(new))
        new['observation_policy']['non_delivery_execution_complete']=True
        self.assertNotEqual(bound,qualification_configuration(new))


if __name__=='__main__':unittest.main()
