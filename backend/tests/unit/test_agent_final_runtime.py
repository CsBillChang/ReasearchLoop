"""验证逐格存储边界及全矩阵身份，不接触部署资源或模型。"""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from agent_final_runtime import CellBoundary, cell_directory, digest, validate_suite_manifest
from harness_ab_boundary import IsolationError


class CellBoundaryTests(unittest.TestCase):
    """模拟实际审计事件，覆盖越界路径和数据库及唯一共享例外。"""

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.runtime=self.root/'runtime'
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
            'execution_order':order,'ledger_limits':{'allowed_cells':order},'vector_endpoint':'http://127.0.0.1:6431','vector_port':6431}

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


if __name__=='__main__':unittest.main()
