"""推理独立对照的零模型合同：原排序、标注分母与采用门。"""
import copy
import unittest

from scripts.verify_reranker_efficiency import legacy_ranking, quality_by_group, adoption_gate


class RerankerEfficiencyTests(unittest.TestCase):
    """只验证规则，不用替身分数冒充真实质量通过。"""
    def test_legacy_order_and_ties(self):
        """始终按CE降序，同分保留原RRF顺序，不混合另一种排名。"""
        rows=[{'key':str(i),'page':i+1} for i in range(24)]
        original=copy.deepcopy(rows)
        keys=legacy_ranking(rows,[3.,3.,5.]+[0.]*21)
        self.assertEqual(keys[:3],['2','0','1']); self.assertEqual(rows,original)
        self.assertEqual(set(keys),{r['key'] for r in rows})

    def test_identity_and_invalid_score_rejected(self):
        """丢分数、重复身份、非有限数不能形成合法排名。"""
        for rows,scores in [([{'key':'x'}],[]),([{'key':'x'}]*2,[1.,2.]),([{'key':'x'}],[float('inf')])]:
            with self.assertRaises(ValueError): legacy_ranking(rows,scores)

    def test_groups_exclude_unanswerable(self):
        """21题分母的规则：空标注不加满分，中英文分别汇总。"""
        cases=[{'id':'a','answerable':True,'language':'zh'}, {'id':'b','answerable':True,'language':'en'}, {'id':'c','answerable':False,'language':'zh'}]
        rows={k:{'recall_at_10':v,'recall_at_8':v,'mrr_at_24':v} for k,v in [('a',1.),('b',.5),('c',None)]}
        result=quality_by_group(cases,rows)
        self.assertEqual(result['all']['mrr_at_24'],.75); self.assertEqual(result['zh']['mrr_at_24'],1.)

    def baseline(self):
        """构造门槛边界，无真实性能含义。"""
        return {'quality':{g:{'recall_at_10':.9,'recall_at_8':.8,'mrr_at_24':.4} for g in ('all','zh','en')},'median_seconds':2.,'p95_seconds':3.,'ranking_drift':[]}

    def test_gate_accepts_quality_preserved_speedup(self):
        """质量相等而速度达标可采用，不要求凭空提高MRR。"""
        base=self.baseline(); candidate=copy.deepcopy(base); candidate['median_seconds']=1.8
        self.assertTrue(adoption_gate(base,candidate)['eligible'])

    def test_gate_rejects_each_tradeoff(self):
        """不能用速度、某语言或均值掩盖其他退化。"""
        base=self.baseline()
        for field,value in [('median_seconds',1.81),('p95_seconds',3.1),('ranking_drift',['Q01'])]:
            candidate=copy.deepcopy(base); candidate['median_seconds']=1.; candidate[field]=value
            self.assertFalse(adoption_gate(base,candidate)['eligible'])
        candidate=copy.deepcopy(base);candidate['median_seconds']=1.;candidate['quality']['zh']['mrr_at_24']=.39
        self.assertFalse(adoption_gate(base,candidate)['eligible'])
