"""用真实后端API校验54题录和独立轮次输入，生成模型不参与。"""
import hashlib
import json
from pathlib import Path
import unittest


class FinalBackendInputTests(unittest.TestCase):
    """复用合成后端夹具，重点验证共同入口与新合同的组合。"""

    def setUp(self):
        from unit.test_prompt_bundle_ab import BundleBackendInputTests
        BundleBackendInputTests.setUp(self)

    def test_full54_catalog_reaches_api_and_grants_only_actual_results(self):
        from run_read_paper_ab import create_bundle_input_task
        from prompt_bundle_cases import install_catalog
        from research_loop.providers.discovery import DiscoveryProvider
        from research_loop.services.papers import PaperService
        materials=[];catalog=[];bundles={}
        for i in range(54):
            content=f'%PDF-synthetic-{i}'.encode();sha=hashlib.sha256(content).hexdigest()
            path=self.root/f'catalog-{i}.pdf';path.write_bytes(content)
            materials.append({'material_id':f'P{i+1:02d}','pdf_path':str(path),'pdf_sha256':sha,'physical_pages':1})
            catalog.append({'material_index':i,'title':f'Item {i}','abstract':f'unique{i} research topic','source_url':''})
            bundles[sha]={'document_ref':{'paper_id':f'paper-{i}','metadata':{'source':'arxiv'}}}
        case={'case_id':'REC-R1','suite_kind':'agent_final_v1','prompt':'Select relevant candidates, do not save',
            'materials':materials,'preconditions':{'catalog':catalog,'history_messages':[],'save_policy':'deny'}}
        install_catalog(self.stack,DiscoveryProvider,case,bundles)
        report,attachments=create_bundle_input_task(self.client,self.db,case,'A','execute',bundles)
        self.assertEqual(attachments,[])
        self.assertEqual(json.loads(self.db.get_task(report['root_id'])['attachments_json']),[])
        self.assertEqual(self.db.list_model_calls(report['root_id']),[])
        self.assertFalse(self.db.task_has_paper(report['root_id'],'paper-53'))
        result=PaperService(self.db,DiscoveryProvider.from_settings(self.settings),None).search('unique53',limit=1,task_id=report['root_id'])
        self.assertEqual(result.candidates[0].paper_id,'paper-53')
        self.assertTrue(self.db.task_has_paper(report['root_id'],'paper-53'))
        self.assertFalse(self.db.task_has_paper(report['root_id'],'paper-0'))
        self.assertEqual(self.db.list_task_evidence(report['root_id']),[])

    def test_rounds_create_distinct_conversations_and_no_fake_model_history(self):
        from run_read_paper_ab import create_bundle_input_task
        ids=[];conversations=[]
        for n in (1,2):
            case={'case_id':f'MEM-R{n}','suite_kind':'agent_final_v1','prompt':'Read our agreed constraints',
                'materials':[],'preconditions':{'history_messages':[{'role':'user','content':'No saving'}],'save_policy':'deny'}}
            report,_=create_bundle_input_task(self.client,self.db,case,'B','execute',{})
            ids.append(report['root_id']);conversations.append(report['conversation_id'])
            rows=self.db.fetchall('SELECT task_id,content FROM conversation_messages WHERE conversation_id=? ORDER BY sequence_no',(report['conversation_id'],))
            self.assertEqual(rows[0]['task_id'],'');self.assertEqual(rows[0]['content'],'No saving')
            self.assertEqual(self.db.list_model_calls(report['root_id']),[])
        self.assertEqual(len(set(ids)),2);self.assertEqual(len(set(conversations)),2)


if __name__=='__main__':unittest.main()
