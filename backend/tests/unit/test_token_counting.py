"""共享容量与预留估算的零网络测试；不验证供应商真实Tokenizer精度。"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_loop.config import Settings
from research_loop.providers.token_counting import counting_policy, estimate, estimate_details, use_policy
from research_loop.storage.mysql import Database
from research_loop.storage.token_budget import TokenBudgetStore, estimate_request
from research_loop.agents.memory import ContextService


class TokenCountingTests(unittest.TestCase):
    """用合成中英文、工具与临时SQLite检查统一口径和历史边界。"""

    def setUp(self):
        """每例独立路径，任何网络调用都不是本测试步骤。"""
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = Settings(project_root=self.root,data_dir=self.root,output_dir=self.root,
                                 deepseek_model='deepseek-flash')

    def database(self, settings=None):
        """只创建本例临时SQLite，冻结新任务策略。"""
        db=Database('sqlite:///'+str(self.root/'test.db')); db.ensure_schema()
        db.create_task('root',None,'h','只读材料','deny',[],token_settings=settings or self.settings)
        return db

    def test_capabilities_and_environment_override(self):
        """能力上限、请求输出与显式旧配置彼此独立。"""
        self.assertEqual((self.settings.context_window_tokens,self.settings.model_max_output_tokens,
                          self.settings.context_output_reserve),(1000000,393216,65536))
        with patch.dict(os.environ, {'DEEPSEEK_MODEL':'deepseek-flash','RESEARCHLOOP_CONTEXT_WINDOW':'128000'},clear=True):
            configured=Settings.from_env(self.root)
        self.assertEqual(configured.context_window_tokens,128000)
        self.assertEqual(configured.context_output_reserve,65536)
        self.assertEqual(configured.researcher_concurrency,2)
        self.assertFalse(configured.web_search_enabled)
        with self.assertRaisesRegex(ValueError,'known model capability'):
            Settings(project_root=self.root,data_dir=self.root,output_dir=self.root,deepseek_model='deepseek-flash',context_window_tokens=2000000)

    def test_explicit_output_old_unknown_models_and_small_windows(self):
        """显式输出优先；旧模型保守兼容，小窗口不得被默认预留静默挤占。"""
        for model in ('deepseek-flash', 'deepseek-v4-flash', 'deepseek-v4-pro',
                      'deepseek-v4-flash-vision-exp'):
            with self.subTest(model=model):
                configured = Settings(project_root=self.root, data_dir=self.root, output_dir=self.root,
                                      deepseek_model=model)
                self.assertEqual(configured.context_output_reserve, 65536)
                self.assertNotIn('context_output_reserve', configured.model_fields_set)
        for model in ('deepseek-chat', 'deepseek-reasoner', 'unknown-model'):
            with self.subTest(model=model):
                configured = Settings(project_root=self.root, data_dir=self.root, output_dir=self.root,
                                      deepseek_model=model)
                self.assertEqual(configured.context_output_reserve, 8192)
        with patch.dict(os.environ, {'DEEPSEEK_MODEL': 'deepseek-flash',
                                   'RESEARCHLOOP_CONTEXT_OUTPUT_RESERVE': '8192'}, clear=True):
            configured = Settings.from_env(self.root)
        self.assertEqual(configured.context_output_reserve, 8192)
        self.assertIn('context_output_reserve', configured.model_fields_set)
        self.assertNotIn('context_output_reserve', self.settings.model_fields_set)
        with self.assertRaisesRegex(ValueError, 'consume the model window'):
            Settings(project_root=self.root, data_dir=self.root, output_dir=self.root,
                     deepseek_model='deepseek-flash', context_window_tokens=32768)
        with self.assertRaisesRegex(ValueError, 'exceeds model capability'):
            Settings(project_root=self.root, data_dir=self.root, output_dir=self.root,
                     deepseek_model='deepseek-flash', context_output_reserve=393217)
        valid = Settings(project_root=self.root, data_dir=self.root, output_dir=self.root,
                         deepseek_model='deepseek-flash', context_window_tokens=12000,
                         context_output_reserve=1024)
        self.assertEqual(valid.context_output_reserve, 1024)
        for window, cap, reserve in ((16000, 393216, 8192), (16000, 4096, 4096)):
            configured = Settings(project_root=self.root, data_dir=self.root, output_dir=self.root,
                                  deepseek_model='deepseek-flash', context_window_tokens=window,
                                  model_max_output_tokens=cap, context_output_reserve=reserve)
            self.assertEqual(configured.context_output_reserve, reserve)
        with self.assertRaisesRegex(ValueError, 'exceeds model capability'):
            Settings(project_root=self.root, data_dir=self.root, output_dir=self.root,
                     deepseek_model='deepseek-flash', model_max_output_tokens=4096)
        self.assertEqual(self.settings.context_output_reserve, 65536)

    def test_new_root_context_and_budget_share_64k(self):
        """三角色新根的输入容量、输出预留和账本使用同一65536。"""
        db = self.database()
        context = ContextService(db, None, self.settings)
        for sequence, role in enumerate(('supervisor', 'researcher', 'reviewer'), 1):
            state = {'task_id': 'root', 'root_task_id': 'root', 'role': role,
                     'prompt': '合成只读研究', 'model_requests': sequence - 1}
            messages = [{'role': 'user', 'content': '合成文字'}]
            prepare = context.prepare if role == 'supervisor' else context.prepare_role
            projected, metrics = prepare(state, messages, [])
            self.assertEqual(metrics['output_reserve_tokens'], 65536)
            self.assertEqual(metrics['input_capacity'], 1000000 - 65536 - 4096)
            request = {'messages': projected, 'tools': [], 'max_tokens': metrics['output_reserve_tokens']}
            store = TokenBudgetStore(db)
            before = store.summary('root')['reserved_tokens']
            store.reserve('root', 'root', sequence, request, role)
            self.assertEqual(store.summary('root')['reserved_tokens'] - before,
                             metrics['estimated_input_tokens'] + 65536)

    def test_shared_schema_and_format_accounting(self):
        """英文/中文不再等同字节；内容、工具和格式均计入同一输入值。"""
        request={'messages':[{'role':'user','content':'word '*200+'中文'*100}],
                 'tools':[{'type':'function','function':{'name':'lookup','parameters':{'type':'object'}}}],
                 'max_tokens':8192}
        before=deepcopy(request)
        inputs={k:request[k] for k in ('messages','tools')}
        details=estimate_details(inputs)
        self.assertEqual(estimate_request(request),estimate(inputs)+8192)
        self.assertEqual(details['format_overhead_tokens'],1048)
        self.assertNotEqual(details['text_tokens'],details['transport_bytes'])
        self.assertIn('not_a_guaranteed_upper_bound',details['accuracy'])
        self.assertFalse(details['is_exact'])
        self.assertGreater(estimate(inputs),estimate({'messages':request['messages'],'tools':[]}))
        self.assertEqual(request,before)

    def test_local_tokenizer_requires_compatibility_and_freezes_hash(self):
        """只加载显式本机JSON；文本可离线分词但供应商封装仍为估算。"""
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers
        except ImportError:
            self.skipTest('optional local-index tokenizers dependency absent')
        tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'hello':1},unk_token='[UNK]'))
        tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
        path=self.root/'tokenizer.json';tokenizer.save(str(path))
        configured=self.settings.model_copy(update={'token_counting_tokenizer_path':path})
        with self.assertRaisesRegex(ValueError,'COMPATIBILITY'):counting_policy(configured)
        configured=configured.model_copy(update={'token_counting_tokenizer_model':'deepseek-flash'})
        policy=counting_policy(configured)
        self.assertEqual(policy['tokenizer_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(estimate('hello hello',policy),4) # JSON quotes are separate unknown tokens.
        self.assertFalse(estimate_details({'messages':[]},policy)['is_exact'])
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'IDENTITY'):estimate('hello',policy)

    def test_frozen_root_capacity_reviewer_and_reservation_agree(self):
        """Reviewer使用冻结输入/输出配置；后来的Settings变化不改旧根。"""
        db=self.database(self.settings.model_copy(update={'context_output_reserve':8192}))
        current=self.settings.model_copy(update={'context_window_tokens':128000,'context_output_reserve':1000})
        context=ContextService(db,None,current)
        messages=[{'role':'system','content':'核验当前稿'},{'role':'user','content':'中文 '*500}]
        state={'task_id':'root','root_task_id':'root','role':'reviewer','prompt':'核验','model_requests':0}
        projected,metrics=context.prepare_role(state,messages,[])
        request={'messages':projected,'tools':[],'max_tokens':metrics['output_reserve_tokens']}
        store=TokenBudgetStore(db);policy=store.counting_policy('root')
        store.reserve('root','root',1,request,'reviewer')
        self.assertEqual(metrics['output_reserve_tokens'],8192)
        self.assertEqual(metrics['input_capacity'],1000000-8192-4096)
        self.assertEqual(store.summary('root')['reserved_tokens'],metrics['estimated_input_tokens']+8192)
        self.assertEqual(metrics['estimated_input_tokens'],estimate({'messages':projected,'tools':[]},policy))
        self.assertNotIn('Memory v1',str(projected))

    def test_model_change_cannot_silently_reuse_old_counting_profile(self):
        """新请求拒绝跨模型使用旧计量策略；历史收据由图原样回放。"""
        db=self.database()
        changed=self.settings.model_copy(update={'deepseek_model':'deepseek-v4-pro'})
        context=ContextService(db,None,changed)
        with self.assertRaisesRegex(ValueError,'TOKEN_POLICY_MODEL_CHANGED'):
            context.prepare_role({'task_id':'root','prompt':'核验','role':'reviewer'},[],[])
        self.assertFalse(db.list_model_calls('root'))

    def test_legacy_policy_and_existing_reservation_unchanged(self):
        """无新计量策略的旧根继续旧公式，已经预留的值永不重算。"""
        db=self.database(); db.execute('DELETE FROM token_counting_policies')
        store=TokenBudgetStore(db)
        request={'messages':[{'role':'user','content':'旧文字中文'}],'tools':[],'max_tokens':1000}
        from research_loop.storage.memory import dump
        expected=len(dump({'messages':request['messages'],'tools':[]}).encode())+1024+1000
        store.reserve('root','root',1,request,'supervisor')
        self.assertEqual(store.summary('root')['reserved_tokens'],expected)
        with patch('research_loop.storage.token_budget.estimate_request',side_effect=AssertionError('no recalc')):
            store.reserve('root','root',1,request,'supervisor')
        self.assertIsNone(store.execution_policy('root'))

    def test_frozen_tail_matches_database_clamping(self):
        """单请求根任务的收尾预留归零，JSON与共享额度行不能矛盾。"""
        configured=self.settings.model_copy(update={'max_model_requests':1,'reserve_model_requests':10})
        db=self.database(configured)
        policy=TokenBudgetStore(db).execution_policy('root')
        self.assertEqual(policy['reserve_model_requests'],0)
        self.assertEqual(db.budget_summary('root')['reserve_tail'],0)

    def test_observe_counts_but_unknown_still_blocks(self):
        """仅监测不以本地额度终止，但收据未知不能盲重发。"""
        configured=self.settings.model_copy(update={'execution_budget_mode':'observe','task_token_limit':1})
        db=self.database(configured);store=TokenBudgetStore(db)
        request={'messages':[{'role':'user','content':'x'}],'max_tokens':8192}
        store.reserve('root','root',1,request,'researcher')
        self.assertGreater(store.summary('root')['reserved_tokens'],1)
        db.save_model_call('root',1,'unknown',request,response={},sent_state='sent')
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):store.reserve('root','root',2,request,'researcher')
