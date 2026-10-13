"""累计预算的可重放、并发和保守结算测试；不调用真实模型。"""
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from research_loop.config import Settings
from research_loop.storage.mysql import Database
from research_loop.storage.token_budget import TokenBudgetStore, actual_usage, estimate_request


class TokenBudgetTests(unittest.TestCase):
    """SQLite快速验证，共享余额在MySQL隔离套件复验。"""

    def setUp(self):
        """每个案例独立数据库，避免旧账本污染本例。"""
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.db = Database('sqlite:///' + str(root/'budget.db')); self.db.ensure_schema()
        self.settings = Settings(project_root=root,data_dir=root/'data',output_dir=root/'notes',task_token_limit=5000,task_token_tail=1000)
        self.db.create_task('root',None,'h','prompt','deny',[],token_settings=self.settings)
        self.tokens = TokenBudgetStore(self.db)
        self.request = {'messages':[{'role':'user','content':'x'}],'tools':[],'max_tokens':1000}

    def test_parallel_children_share_limit_and_tail(self):
        """并发预留不能都读到旧余额；Supervisor可使用子角色收尾余量。"""
        def reserve(i):
            try:self.tokens.reserve('root',str(i),1,self.request,'researcher'); return True
            except ValueError:return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(pool.map(reserve,range(2))),1)
        self.tokens.reserve('root','root',1,self.request,'supervisor')
        self.assertEqual(self.tokens.summary('root')['reserved_tokens'],2*estimate_request(self.request))

    def test_response_before_settlement_crash_replay_once(self):
        """模拟响应落盘后退出：新存储实例从原收据补结算一次。"""
        self.tokens.reserve('root','root',1,self.request,'supervisor')
        with patch.object(TokenBudgetStore,'settle',side_effect=RuntimeError('crash after receipt')):
            with self.assertRaises(RuntimeError):
                self.db.save_model_call('root',1,'succeeded',self.request,response={'choices':[]},usage={'total_tokens':120},sent_state='sent')
        fresh=TokenBudgetStore(Database(self.db.url))
        # 查询即可修复派生账本；再显式结算也不重复计入。
        self.assertEqual(fresh.summary('root')['used_tokens'],120)
        fresh.settle('root',1);fresh.settle('root',1)
        self.assertEqual((fresh.summary('root')['used_tokens'],fresh.summary('root')['reserved_tokens']),(120,0))
        with self.assertRaisesRegex(ValueError,'REPLAY_REQUIRED'):
            fresh.reserve('root','root',1,self.request,'supervisor')

    def test_reservation_identity_and_not_sent(self):
        """未发送可以重入原预留；参数不可改变，明确本地拒绝才释放。"""
        self.tokens.reserve('root','root',1,self.request,'supervisor')
        self.tokens.reserve('root','root',1,self.request,'supervisor')
        with self.assertRaisesRegex(ValueError,'CONFLICT'):
            self.tokens.reserve('root','root',1,{**self.request,'max_tokens':9},'supervisor')
        self.db.save_model_call('root',1,'not_sent',self.request,sent_state='not_sent')
        self.assertEqual(self.tokens.summary('root')['reserved_tokens'],0)
        self.assertEqual(self.tokens.summary('root')['used_tokens'],0)

    def test_missing_usage_and_http_error_hold_instead_of_zero(self):
        """成功缺usage或供应商拒绝均不能推定免费，后续任务请求阻断。"""
        for status in ['succeeded','unknown','rejected']:
            with self.subTest(status=status):
                self.db.execute('DELETE FROM token_reservations')
                self.db.execute('DELETE FROM model_calls')
                self.db.execute('UPDATE task_token_budgets SET reserved_tokens=0')
                self.tokens.reserve('root','root',1,self.request,'supervisor')
                self.db.save_model_call('root',1,status,self.request,response={},sent_state='sent')
                self.assertEqual(self.tokens.summary('root')['unknown_reservations'],1)
                self.assertGreater(self.tokens.summary('root')['reserved_tokens'],0)
                with self.assertRaisesRegex(ValueError,'UNKNOWN'):
                    self.tokens.reserve('root','root',2,self.request,'supervisor')

    def test_actual_overrun_reported_and_no_next_send(self):
        """真实usage超出估计如实计入，不用截断账本伪装预算通过。"""
        self.tokens.reserve('root','root',1,self.request,'supervisor')
        self.db.save_model_call('root',1,'succeeded',self.request,usage={'total_tokens':6000},sent_state='sent')
        self.assertEqual(self.tokens.summary('root')['overrun_tokens'],1000)
        with self.assertRaisesRegex(ValueError,'BUDGET'):
            self.tokens.reserve('root','root',2,self.request,'supervisor')

    def test_legacy_no_retroactive_policy(self):
        """无策略历史任务恢复时不回填新的预算。"""
        self.db.create_task('legacy',None,'h','old','deny',[])
        self.tokens.reserve('legacy','legacy',1,{},'researcher')
        self.assertIsNone(self.tokens.summary('legacy'))

    def test_invalid_usage_never_fabricated(self):
        """缺一项、布尔或负数/冲突usage不是有效用量。"""
        for value in [{},{'total_tokens':True},{'prompt_tokens':3},{'total_tokens':-2},
                      {'total_tokens':3,'prompt_tokens':3,'completion_tokens':2}]:
            self.assertIsNone(actual_usage(value))
        self.assertEqual(actual_usage({'prompt_tokens':3,'completion_tokens':2}),5)
