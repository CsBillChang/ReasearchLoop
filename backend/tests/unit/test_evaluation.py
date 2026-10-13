"""评测口径不受输出成绩影响；覆盖分母、越界、重复、缺证与冻结失败。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.evaluation_common import ranked_metrics, exclusive, sha, verify_freeze
from scripts.evaluate_reflection import FrozenLedger
from research_loop.providers.deepseek import RequestNotSent


class EvaluationTests(unittest.TestCase):
    """仅合成排序和临时文件，不触发业务模型或论文网络。"""
    def test_metrics_at_cutoff(self):
        """第11条不能算Recall@10，MRR仍按明确的24候选深度计算。"""
        keys=[str(i) for i in range(24)]
        result=ranked_metrics(keys,['0','10'],answerable=True,corpus=set(keys))
        self.assertEqual(result['recall_at_10'],.5)
        self.assertEqual(result['reciprocal_rank_at_24'],1)
        result=ranked_metrics(keys,['10'],answerable=True,corpus=set(keys))
        self.assertEqual(result['recall_at_10'],0)
        self.assertEqual(result['reciprocal_rank_at_24'],1/11)

    def test_missing_does_not_score_one(self):
        """缺证题Recall/MRR必须为null，返回相似片段不等于编造答案。"""
        result=ranked_metrics(['a'],[],answerable=False,corpus={'a'})
        self.assertIsNone(result['recall_at_10'])
        self.assertIsNone(result['reciprocal_rank_at_24'])

    def test_empty_answerable_is_zero(self):
        """可回答题无命中是零分，不能从分母中删去。"""
        result=ranked_metrics([],['a'],answerable=True,corpus={'a'})
        self.assertEqual(result['recall_at_10'],0)
        self.assertEqual(result['reciprocal_rank_at_24'],0)

    def test_bad_identity_and_labels_rejected(self):
        """越界、重复身份和错误标注不能产生貌似正常的指标。"""
        for ranked,relevant,answerable in [(['a','a'],['a'],True),(['outside'],['a'],True),([],[],True),([],['a'],False),([],['a','a'],True)]:
            with self.subTest(ranked=ranked,relevant=relevant):
                with self.assertRaises(ValueError):ranked_metrics(ranked,relevant,answerable=answerable,corpus={'a'})

    def test_freeze_and_exclusive(self):
        """旧成绩不能覆盖；冻结文件有改动时拒绝执行。"""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'questions.json';p.write_text('{}')
            exclusive(root/'freeze.json',{'files':{'questions.json':sha(p)}})
            with self.assertRaises(FileExistsError):exclusive(root/'freeze.json',{})
            with patch('scripts.evaluation_common.OUT',root),patch('scripts.evaluation_common.ROOT',root):
                verify_freeze();p.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'FROZEN_FILE_CHANGED'):verify_freeze()

    def test_unknown_and_missing_usage_stop_before_send(self):
        """未结算和成功但无usage都拒绝后续发送；不把未知量写成零。"""
        manifest={'physical_cap':2,'token_cap':10000,'wall_seconds':60,'cases':[{'id':'F','request_cap':2}]}
        with tempfile.TemporaryDirectory() as directory:
            ledger=FrozenLedger(Path(directory)/'ledger.json',manifest);ledger.case='F'
            args=('json',[{'role':'user','content':'test'}],100)
            n=ledger.reserve(*args)
            with self.assertRaises(RequestNotSent):ledger.reserve(*args)
            ledger.settle(n,{'id':'test','usage':None})
            with self.assertRaises(RequestNotSent):ledger.reserve(*args)
            self.assertEqual(len(ledger.records),1)

    def test_settled_cap_is_shared_and_has_no_retry(self):
        """成功收据允许下一例，共享物理额度耗尽不能靠切换case绕过。"""
        manifest={'physical_cap':1,'token_cap':10000,'wall_seconds':60,
                  'cases':[{'id':i,'request_cap':1} for i in ['A','B']]}
        with tempfile.TemporaryDirectory() as directory:
            ledger=FrozenLedger(Path(directory)/'ledger.json',manifest);ledger.case='A'
            args=('json',[{'role':'user','content':'test'}],100)
            n=ledger.reserve(*args);ledger.settle(n,{'id':'ok','usage':{'prompt_tokens':5,'completion_tokens':5,'total_tokens':10}})
            ledger.case='B'
            with self.assertRaises(RequestNotSent):ledger.reserve(*args)
            self.assertEqual(len(ledger.records),1)
