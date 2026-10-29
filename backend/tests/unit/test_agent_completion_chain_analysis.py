"""V3原生工件与配对评分归属分离，不能借新manifest覆盖旧观测。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import agent_final_analysis as analysis


class CompletionChainAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve();self.owners={}
        for name in ('N','R2','R3','current'):
            root=self.base/name;root.mkdir()
            owner={'experiment_root':str(root),'runtime_root':str(root/'runtime'),'semantic_gate_root':str(root/'reviews')}
            (root/'manifest.formal.json').write_text(json.dumps(owner));self.owners[name]=owner
        cell=self.base/'R2/runtime/cells/AFE-CROSS-03-R1-B'
        path=self.base/'R2/manifest.formal.json';score=self.base/'R3/manifest.formal.json'
        self.item={'case_id':'AFE-CROSS-03-R1','variant':'B','quality':'partial_usable',
            'source_manifest_path':str(path),'source_manifest_sha256':analysis.sha(path.read_bytes()),
            'score_manifest_path':str(score),'score_manifest_sha256':analysis.sha(score.read_bytes()),
            'provenance':{'anchor_root':str(self.base/'R2'),'cell_root':str(cell)}}
        self.context={'closed_ledger_roots':[str(self.base/x) for x in ('N','R2','R3')],
                      'retained_rows':[self.item],'technical_attempts':[]}

    def artifacts(self):
        ledger={'metadata':{},'roots':[],'reservations':[]}
        calls=[]
        def cell(reader,root,owner,case,variant,*args,**kwargs):
            calls.append({'root':root,'owner':owner,**kwargs})
            return {'score':{'whole_answer_quality':'partial_usable'},'status':'completed'}
        with patch.object(analysis,'read_ledger',return_value=ledger), \
             patch.object(analysis,'closure_status',return_value={'controller':{}}), \
             patch.object(analysis,'analyze_cell',side_effect=cell):
            result=analysis.completion_artifacts(analysis.Reader(),self.owners['current'],self.context)
        return result,calls

    def test_old_B_artifacts_stay_R2_while_score_is_original_R3_pair(self):
        result,calls=self.artifacts()
        self.assertEqual(calls[0]['owner'],self.owners['R2'])
        self.assertEqual(calls[0]['score_manifest'],self.owners['R3'])
        self.assertEqual(result[0][('AFE-CROSS-03-R1','B')]['continuation_provenance']['anchor_root'],str(self.base/'R2'))

    def test_missing_foreign_or_changed_score_identity_rejected(self):
        for change in ('missing-path','missing-hash','wrong-hash','current-owner','wrong-name'):
            before=deepcopy(self.item)
            if change=='missing-path':del self.item['score_manifest_path']
            elif change=='missing-hash':del self.item['score_manifest_sha256']
            elif change=='wrong-hash':self.item['score_manifest_sha256']='0'*64
            elif change=='current-owner':self.item['score_manifest_path']=str(self.base/'current/manifest.formal.json')
            else:self.item['score_manifest_path']=str(self.base/'R3/not-a-manifest.json')
            try:
                with self.subTest(change=change),self.assertRaises(ValueError):self.artifacts()
            finally:self.item.clear();self.item.update(before)

    def test_legacy_V2_owner_rule_unchanged_without_new_fields(self):
        del self.item['score_manifest_path'];del self.item['score_manifest_sha256']
        _,calls=self.artifacts();self.assertEqual(calls[0]['score_manifest'],self.owners['R2'])
        self.item['quality']=None
        _,calls=self.artifacts();self.assertEqual(calls[0]['score_manifest'],self.owners['current'])


if __name__=='__main__':unittest.main()
