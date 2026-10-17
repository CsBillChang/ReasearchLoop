"""评测草稿必须能走到真实收尾服务，不能只验证API创建成功。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pymupdf
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.storage.mysql import Database
from research_loop.services.coverage import CoverageService
from research_loop.domain.tasks import ResearchBrief,FinishTaskInput
from scripts.evaluate_reflection import seed_case


class EvaluationSeedTests(unittest.TestCase):
    """合成单页PDF、临时SQLite及真实服务；不运行模型、不改冻结实验记录。"""

    def test_seed_finish_preserves_source_and_unresolved(self):
        """绑定事件含完整来源身份；缺证仍是partial，不能靠补字段自动改为已核验。"""
        from research_loop.main import app
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pdf=root/'sample.pdf'
            with pymupdf.open() as document:
                # 满足DPR现有1200字符正文检查，不能因测试短文绕开生产解析器。
                document.new_page().insert_text((40,40),'\n'.join(
                    f'Fixture line {i}: A method with no GPU driver version details.' for i in range(30)),fontsize=9)
                document.save(pdf)
            config=Settings(project_root=root,database_url='sqlite:///'+str(root/'db.sqlite'),
                            data_dir=root/'data',output_dir=root/'notes')
            db=Database(config.database_url);db.ensure_schema()
            corpus={'documents':[{'key':'sample','material':{'path':str(pdf),'paper_id':'sample-paper'}}]}
            with patch('research_loop.api.dependencies.get_settings',return_value=config),TestClient(app) as client:
                tid,draft=seed_case(client,db,config,{'id':'seed-contract','material':'sample','markdown':'已有短稿'},'核验当前稿，不保存',corpus)
            event=next(e['payload'] for e in db.list_events(tid) if e['event_type']=='feedback_draft_bound')
            self.assertTrue(event['evidence_ids'])
            self.assertEqual(event['evidence_ids'],draft['evidence_ids'])
            self.assertTrue(all(db.task_has_evidence(tid,e) for e in event['evidence_ids']))
            service=CoverageService(db)
            plan=service.prepare_plan(tid,ResearchBrief(goal='核验当前稿',input_refs=['sample'],
                goals=[{'goal_id':'g1','description':'核验原句','min_papers':1,'requires_fulltext':True,'input_refs':['sample']}]))
            with patch.object(service,'plan',return_value=plan):
                result=service.finish(tid,FinishTaskInput(draft_id=draft['draft_id'],
                    goals=[{'goal_id':'g1','status':'unresolved','reason':'还没有完成独立复审'}]))
            self.assertEqual(result['status'],'partial')
            self.assertEqual(result['answer_markdown'],'已有短稿')
            self.assertEqual(result['coverage']['paper_ids'],['sample-paper'])
            self.assertTrue(result['gaps'])
            self.assertEqual(db.fetchall('SELECT * FROM model_calls'),[])
            self.assertFalse(list(config.output_dir.rglob('*.md')))
