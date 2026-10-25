"""组件适配器的离线路径与真实隔离SQLite服务测试；不评测模型或语义收益。"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import httpx

from scripts.component_eval_adapters import IsolationGuard, MemoryAdapter, ReflectionAdapter, RetrievalAdapter, validate_isolation
from research_loop.agents.reflection import ReflectionService
from research_loop.config import Settings
from research_loop.domain.evidence import Document, PageEvidence
from research_loop.domain.memory import ProfileSet
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.storage.memory import MemoryStore, dump
from research_loop.storage.mysql import Database, _now


def chunks(count=30):
    """生成完整同版身份；向量payload固定，不代表真实论文。"""
    output = []
    for n in range(count):
        text = 'language method ' + str(n)
        output.append({'chunk_id': f'chunk-{n:02}', 'key': f'point-{n:02}', 'text': text,
            'document_id': 'doc', 'index_id': 'index', 'evidence_id': 'page', 'source_sha256': 'a'*64,
            'text_sha256': sha256(text.encode()).hexdigest(), 'page_number': 1,
            'char_start': n*100, 'char_end': n*100+len(text), 'collection': 'researchloop_eval_unit_model',
            'vector_payload': {'index_id': 'index', 'chunk_id': f'chunk-{n:02}', 'source_sha256': 'a'*64}})
    return output


class RetrievalAdapterTests(unittest.TestCase):
    """用spy确认四条计算路径互不隐式启用组件。"""

    def setUp(self):
        """每例单独工厂和冻结片段。"""
        self.rows = chunks()
        self.scope = [r['chunk_id'] for r in self.rows]
        self.embedding = Mock()
        self.embedding.encode.return_value = [[1.0, 0.0]]
        self.vectors = Mock()
        by_key = {r['key']: r for r in self.rows}
        self.vectors.query.side_effect = lambda collection, vector, ids, limit: [
            {'id': key, 'score': float(n), 'payload': by_key[key]['vector_payload']}
            for n, key in enumerate(ids[-limit:])][::-1]
        self.reranker = Mock()
        self.reranker.score.side_effect = lambda query, texts: list(range(len(texts)))
        self.factories = [Mock(return_value=x) for x in (self.embedding, self.vectors, self.reranker)]
        self.adapter = RetrievalAdapter(*self.factories)

    def run_variant(self, variant):
        """调用统一授权范围，不使用生产最终命中limit。"""
        return self.adapter.run('language', self.rows, variant, self.scope)

    def test_bm25_never_constructs_models_or_vectors(self):
        """词项路线返回24名且模型工厂零调用。"""
        result = self.run_variant('bm25')
        self.assertEqual(len(result['ranked_ids']), 24)
        self.assertEqual(result['status'], 'executed')
        self.assertIsNone(result['usage'])
        for factory in self.factories:
            factory.assert_not_called()

    def test_vector_never_computes_bm25_or_reranks(self):
        """纯向量不偷偷混入词项或重排。"""
        with patch('scripts.component_eval_adapters.rank_terms', side_effect=AssertionError('bm25 forbidden')):
            result = self.run_variant('vector')
        self.assertEqual(len(result['ranked_ids']), 24)
        self.factories[2].assert_not_called()
        self.embedding.encode.assert_called_once_with(['language'], query=True)
        self.assertEqual(set(self.vectors.query.call_args.args[2]), {r['key'] for r in self.rows})

    def test_rrf_uses_shared_scope_without_reranker(self):
        """融合复用现有算法且不构造CE。"""
        result = self.run_variant('rrf')
        self.assertEqual(len(result['ranked_ids']), 24)
        self.assertEqual(result['effective_path'], 'rrf')
        self.factories[2].assert_not_called()
        self.assertEqual(set(result['corpus']), set(self.scope))

    def test_reranks_only_fixed_fused_top24(self):
        """CE只接收固定融合候选，不重排全部语料。"""
        result = self.run_variant('rrf_rerank')
        self.assertEqual(len(self.reranker.score.call_args.args[1]), 24)
        self.assertEqual(len(result['ranked_ids']), 24)
        self.assertEqual(len(set(result['ranked_ids'])), 24)
        self.assertIn('encoding_seconds', result['timing'])
        self.assertIn('fusion_seconds', result['timing'])
        self.assertIn('rerank_seconds', result['timing'])

    def test_hash_or_identity_rejects_before_factories(self):
        """hash及独立冻结版本不能靠向量降级掩盖。"""
        self.rows[0]['text'] = 'tampered'
        with self.assertRaisesRegex(ValueError, 'HASH'):
            self.run_variant('rrf_rerank')
        for factory in self.factories:
            factory.assert_not_called()
        self.rows = chunks()
        frozen = {r['chunk_id']: deepcopy(r) for r in self.rows}
        self.rows[0]['document_id'] = 'another-version'
        with self.assertRaisesRegex(ValueError, 'FROZEN_IDENTITY'):
            self.adapter.run('language', self.rows, 'bm25', self.scope, frozen_identity=frozen)

    def test_foreign_scope_is_not_scored_and_missing_scope_rejects(self):
        """范围外材料不进入任一通道；遗漏明确授权ID不可静默成功。"""
        result = self.adapter.run('language', self.rows, 'rrf', self.scope[:3])
        self.assertEqual(set(result['ranked_ids']), set(self.scope[:3]))
        self.assertEqual(len(self.vectors.query.call_args.args[2]), 3)
        with self.assertRaisesRegex(ValueError, 'SCOPE_INCOMPLETE'):
            self.adapter.run('language', self.rows, 'bm25', ['missing'])

    def test_vector_failure_is_explicit_and_does_not_run_bm25(self):
        """纯向量失败不伪造受控BM25基线。"""
        self.vectors.query.side_effect = OSError('offline')
        with patch('scripts.component_eval_adapters.rank_terms', side_effect=AssertionError('forbidden')):
            result = self.run_variant('vector')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['effective_path'], 'unavailable')
        self.assertEqual(result['ranked_ids'], [])
        self.assertTrue(result['degradations'])

    def test_foreign_payload_is_channel_failure(self):
        """同point ID也必须核对完整冻结payload。"""
        self.vectors.query.side_effect = lambda *args: [
            {'id': r['key'], 'score': 1.0, 'payload': {'foreign': True}} for r in self.rows[:24]]
        result = self.run_variant('rrf')
        self.assertEqual(result['effective_path'], 'bm25')
        self.assertEqual(result['execution_status'], 'degraded')

    def test_reranker_failure_retains_rrf_with_diagnostics(self):
        """CE失败保留完整RRF候选并指明实际路径。"""
        expected = self.run_variant('rrf')['ranked_ids']
        self.reranker.score.side_effect = RuntimeError('offline')
        actual = self.run_variant('rrf_rerank')
        self.assertEqual(actual['ranked_ids'], expected)
        self.assertEqual(actual['effective_path'], 'rrf')

    def test_candidate_limit_cannot_change_by_variant(self):
        """生产5/8条limit不能误用于MRR@24。"""
        with self.assertRaisesRegex(ValueError, 'LIMIT'):
            self.adapter.run('language', self.rows, 'bm25', self.scope, candidate_limit=8)


class IsolatedServiceTests(unittest.TestCase):
    """复用真实服务与临时SQLite，只替换独立语义判断。"""

    def setUp(self):
        """建立独立目录、库、显式Settings和任务，不加载环境配置。"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.config = {'root': str(self.root), 'database_url': 'sqlite:///' + str(self.root/'isolated.db'),
                       'collection_prefix': 'researchloop_eval_unit', 'vector_url': None}
        self.guard = IsolationGuard(self.config)
        self.db = Database(self.config['database_url'])
        self.db.ensure_schema()
        self.settings = Settings(project_root=self.root, data_dir=self.root/'data', output_dir=self.root/'output',
            database_url=self.config['database_url'], retrieval_namespace=self.config['collection_prefix'])
        self.task = uuid4().hex
        self.db.create_task(self.task, None, 'synthetic', 'language 当前问题，用中文，不保存。', 'deny', [])
        self.service = ReflectionService(self.db)
        self.calls = []
        document = Document(document_id=uuid4().hex, paper_id='synthetic', content_sha256='a'*64,
            parser_name='fixture', parser_version='1', config_sha256='b'*64, total_pages=1, cache_path='unused')
        self.db.save_document(document.model_dump())
        self.ref = uuid4().hex
        self.db.save_page(PageEvidence(evidence_id=self.ref, document_id=document.document_id,
                                      page_number=1, text='The method uses language information.').model_dump())
        self.db.mark_task_evidence(self.task, self.ref)

    def execute(self, status='contradicted'):
        """固定回调只产生已披露的测试判断，服务仍负责来源/版本/事务。"""
        def callback(attempt_id, target, refs, questions):
            """返回冻结测试判断，供真实服务验证身份和局部修订。"""
            self.calls.append(attempt_id)
            return {'review_id': uuid4().hex, 'target_id': target['target_id'],
                'content_hash': target['content_hash'], 'status': 'supported' if status == 'supported' else 'issues',
                'findings': [{'item_id': 'answer', 'status': status, 'reason': 'synthetic reviewer',
                              'evidence_ids': refs, 'source_quote': 'The method uses language information.',
                              'basis': 'direct_statement', 'suggested_action': 'revise'}], 'gaps': []}
        return callback

    def test_reflection_baseline_calls_neither_service_nor_callback(self):
        """off是原稿直接交付，不是把rounds设零。"""
        service, execute = Mock(), Mock()
        result = ReflectionAdapter(service, execute).run(self.task, '保留原稿', 'baseline')
        self.assertEqual(result['output'], '保留原稿')
        self.assertIsNone(result['review'])
        service.review_research.assert_not_called()
        execute.assert_not_called()

    def test_real_reflection_with_authorized_source_and_local_patch(self):
        """真实服务核验同稿并只修改一个唯一片段。"""
        draft = '保留开头。错误句。保留结尾。'
        adapter = ReflectionAdapter(self.service, self.execute())
        with patch.object(Settings, 'from_env', side_effect=AssertionError('production config forbidden')):
            result = adapter.run(self.task, draft, 'reflection', evidence_ids=[self.ref], patches=[{
                'item_id': 'answer', 'old_text': '错误句。', 'new_text': '修正句。',
                'reason': 'synthetic local edit', 'evidence_ids': [self.ref]}])
        self.assertEqual(result['status'], 'executed')
        self.assertEqual(result['output'], '保留开头。修正句。保留结尾。')
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(result['revision']['revision']['non_target_text_preserved'])
        self.assertEqual(self.db.get_task(self.task)['save_policy'], 'deny')
        self.assertIsNone(result['usage'])

    def test_rounds_zero_still_executes_first_review(self):
        """零轮配置仍进行首次核验，不能充当基线。"""
        result = ReflectionAdapter(ReflectionService(self.db, max_rounds=0), self.execute()).run(
            self.task, '错误句。', 'reflection', evidence_ids=[self.ref])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result['review']['reflection']['stop_reason'], 'round_limit')

    def test_reflection_failure_preserves_original_and_reason(self):
        """回调发送后的普通异常保留unknown，不能猜测零用量或失败。"""
        adapter = ReflectionAdapter(self.service, Mock(side_effect=RuntimeError('review unavailable')))
        result = adapter.run(self.task, '原稿正文。', 'reflection', evidence_ids=[self.ref])
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['output'], '原稿正文。')
        self.assertTrue(result['processing_summary'])
        self.assertTrue(result['degradations'])

    def test_unauthorized_evidence_rejects_before_reviewer(self):
        """非法来源不能经组件评测绕过权限。"""
        result = ReflectionAdapter(self.service, self.execute()).run(
            self.task, '原稿。', 'reflection', evidence_ids=['foreign'])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.calls, [])

    def test_review_timeout_stays_unknown_without_retry(self):
        """不确定执行状态与缺失usage保持unknown，不盲目重发。"""
        callback = Mock(side_effect=TimeoutError('receipt unknown'))
        adapter = ReflectionAdapter(self.service, callback)
        result = adapter.run(self.task, '当前稿。', 'reflection', evidence_ids=[self.ref])
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['usage'])
        self.assertEqual(callback.call_count, 1)
        snapshot = self.service.snapshot(self.task)
        pending = next(iter(snapshot['research_targets'].values()))['attempts'][0]
        self.assertEqual(pending['status'], 'started')
        self.assertNotIn('result', pending)
        repeated = adapter.run(self.task, '当前稿。', 'reflection', evidence_ids=[self.ref])
        self.assertEqual(repeated['status'], 'unknown')
        self.assertEqual(callback.call_count, 1)
        fresh = ReflectionAdapter(self.service, callback)
        repeated = fresh.run(self.task, '当前稿。', 'reflection', evidence_ids=[self.ref])
        self.assertEqual(repeated['status'], 'unknown')
        self.assertEqual(callback.call_count, 1)

    def test_httpx_read_timeout_is_unknown_and_not_resent(self):
        """生产ReadTimeout不是内建TimeoutError，也必须保留未知和禁止重发。"""
        callback = Mock(side_effect=httpx.ReadTimeout('physical request may have been sent'))
        adapter = ReflectionAdapter(self.service, callback)
        first = adapter.run(self.task, '冻结当前稿。', 'reflection', evidence_ids=[self.ref])
        second = adapter.run(self.task, '冻结当前稿。', 'reflection', evidence_ids=[self.ref])
        self.assertEqual(first['status'], 'unknown')
        self.assertEqual(second['status'], 'unknown')
        self.assertIsNone(first['usage'])
        self.assertIsNone(first['review'])
        self.assertEqual(callback.call_count, 1)
        self.assertTrue(first['trace'][-1]['callback_started'])
        self.assertFalse(first['trace'][-1]['service_review_finished'])
        self.assertTrue(second['trace'][-1]['blocked_unsettled'])

    def test_patch_rejection_after_settled_review_is_failed_with_review_preserved(self):
        """已正常结算核验后patch拒绝是failed，不能丢失正文或已得判断。"""
        adapter = ReflectionAdapter(self.service, self.execute())
        result = adapter.run(self.task, '原稿。', 'reflection', evidence_ids=[self.ref], patches=[{
            'item_id': 'answer', 'old_text': '不存在的句子', 'new_text': '新句',
            'reason': '合成非法patch', 'evidence_ids': [self.ref]}])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['output'], '原稿。')
        self.assertEqual(result['review']['status'], 'issues')
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(adapter._unknown_tasks)
        self.assertTrue(result['trace'][-1]['service_review_finished'])

    def memory_adapter(self):
        """创建真实会话关联及冻结基本历史，长期偏好/历史保持在真实服务。"""
        memory = MemoryService(self.db)
        memory.set_profile(ProfileSet(slot='language', content='以前优先英文', idempotency_key=uuid4().hex))
        conversation = ConversationService(self.db, self.settings)
        cid = conversation.create('synthetic', uuid4().hex)['conversation_id']
        with MemoryStore(self.db).transaction() as tx:
            mid = conversation.add_message(tx, cid, self.task, uuid4().hex, 'user', self.db.get_task(self.task)['prompt'])
            tx.execute('INSERT INTO conversation_links(task_id,conversation_id,message_id) VALUES(?,?,?)', (self.task, cid, mid))
        record = {'task_id': 'old', 'conversation_id': 'other', 'prompt': 'language old',
                  'answer': '历史研究language', 'status': 'completed', 'paper_ids': [], 'papers': [], 'gaps': []}
        self.db.execute('INSERT INTO research_memories(task_id,conversation_id,record_json,search_text,created_at) VALUES(?,?,?,?,?)',
                        ('old', 'other', dump(record), 'language method', _now()))
        basic = {'conversation_id': cid, 'messages': [{'message_id': 'fixed', 'role': 'user',
                    'content': '基本会话必须保持', 'excerpted': False}], 'recent_research': []}
        return MemoryAdapter(self.db, self.settings, memory_service=memory, basic_history=basic,
            context_factory=lambda cls, db, controlled, settings: cls(db, controlled, settings))

    def prepare_memory(self, adapter, variant, boundary, **kwargs):
        """经真实上下文服务处理，不调用模型或读取生产Settings。"""
        state = {'task_id': self.task, 'prompt': self.db.get_task(self.task)['prompt'], 'model_requests': 0}
        messages = [{'role': 'system', 'content': '基础身份'}, {'role': 'user', 'content': state['prompt']}]
        tools = [{'type': 'function', 'function': {'name': 'read_memory', 'description': 'read',
                  'parameters': {'type': 'object', 'properties': {}}}}]
        with patch.object(Settings, 'from_env', side_effect=AssertionError('production config forbidden')), \
                patch('research_loop.config.get_settings', return_value=self.settings):
            return adapter.run(state, messages, tools, variant, boundary=boundary, **kwargs)

    def test_history_only_boundary_preserves_preference_current_goal_and_capacity(self):
        """相关历史开关不删除基本会话、当前目标、偏好或容量检查。"""
        adapter = self.memory_adapter()
        before = self.db.get_task(self.task)
        baseline = self.prepare_memory(adapter, 'baseline', 'history_recall')
        enabled = self.prepare_memory(adapter, 'history_recall', 'history_recall')
        self.assertEqual(baseline['status'], 'executed', baseline)
        self.assertEqual(enabled['status'], 'executed', enabled)
        self.assertFalse(any(r['kind'] == 'automatic_research_history' for r in baseline['trace']))
        self.assertTrue(any(r['kind'] == 'automatic_research_history' for r in enabled['trace']))
        self.assertTrue(any(r['kind'] == 'preference' for r in baseline['trace']))
        for result in (baseline, enabled):
            text = dump(result['output'])
            self.assertIn('基本会话必须保持', text)
            self.assertIn('当前问题', text)
            self.assertIn('current_task', text)
            self.assertFalse(result['token_estimate_is_exact'])
            self.assertIsNone(result['usage'])
        self.assertEqual(baseline['context_metrics']['input_capacity'], enabled['context_metrics']['input_capacity'])
        self.assertEqual(self.db.get_task(self.task), before)

    def test_preference_boundary_preserves_history_and_current_instruction(self):
        """偏好关闭只影响指定画像曝光，当前中文要求保持。"""
        adapter = self.memory_adapter()
        baseline = self.prepare_memory(adapter, 'baseline', 'preference_context')
        enabled = self.prepare_memory(adapter, 'preference_context', 'preference_context')
        self.assertEqual(baseline['status'], 'executed', baseline)
        self.assertNotIn('以前优先英文', dump(baseline['output']))
        self.assertIn('以前优先英文', dump(enabled['output']))
        self.assertIn('用中文', dump(baseline['output']))
        self.assertTrue(any(r['kind'] == 'automatic_research_history' for r in baseline['trace']))
        self.assertEqual(adapter.bind_memory('preference_context', False).profile()['explicit_preferences'], [])
        self.assertTrue(adapter.bind_memory('preference_context', True).profile()['explicit_preferences'])

    def test_context_archive_readback_survives_history_off(self):
        """关闭相关历史仍可合法读取当前任务压缩档案。"""
        adapter = self.memory_adapter()
        self.prepare_memory(adapter, 'baseline', 'history_recall')
        archive = self.db.fetchone('SELECT snapshot_id FROM context_archives WHERE task_id=?', (self.task,))
        result = self.prepare_memory(adapter, 'baseline', 'history_recall', explicit_reads=[{
            'source_type': 'context_archive', 'source_id': archive['snapshot_id'], 'offset': 0, 'limit': 6000}])
        self.assertEqual(result['status'], 'executed', result)
        self.assertTrue(result['readbacks'][0]['content'])
        self.assertTrue(any(r['kind'] == 'explicit_context_archive' for r in result['trace']))
        self.assertEqual(self.db.get_task(self.task)['save_policy'], 'deny')

    def test_explicit_history_exposure_requires_manifest_choice(self):
        """允许替代显式历史读回时照实曝光，不冒称完全无历史。"""
        adapter = self.memory_adapter()
        reads = [{'source_type': 'research_history', 'source_id': 'old', 'offset': 0, 'limit': 6000}]
        rejected = self.prepare_memory(adapter, 'baseline', 'history_recall', explicit_reads=reads)
        self.assertEqual(rejected['status'], 'failed')
        allowed = self.prepare_memory(adapter, 'baseline', 'history_recall', explicit_reads=reads, allow_explicit_history=True)
        self.assertEqual(allowed['status'], 'executed')
        self.assertTrue(any(r['kind'] == 'explicit_research_history' for r in allowed['trace']))

    def test_context_projection_is_unexecuted_not_fake_implementation(self):
        """没有合法baseline策略接点时不伪造开关对照。"""
        adapter = self.memory_adapter()
        for variant in ('baseline', 'context_projection'):
            result = self.prepare_memory(adapter, variant, 'context_projection')
            self.assertEqual(result['status'], 'unexecuted')
            self.assertIn('pending', result['reason'])
            self.assertIn('reason', result['output'])
            self.assertIsNone(result['effective_path'])

    def test_memory_requires_explicit_safe_context_factory(self):
        """没有安全设置接缝时不运行隐藏的生产配置装配。"""
        adapter = MemoryAdapter(self.db, self.settings)
        with patch.object(Settings, 'from_env', side_effect=AssertionError('forbidden')):
            result = adapter.run({'task_id': self.task}, [], [], 'baseline', boundary='history_recall')
        self.assertEqual(result['status'], 'unexecuted')
        self.assertIsNone(result['effective_path'])
        self.assertIn('pending', result['output']['reason'])

    def test_preference_finish_interest_branch_is_pending(self):
        """未受控的行为兴趣旁路不能被记成偏好关闭。"""
        adapter = self.memory_adapter()
        tools = [{'type': 'function', 'function': {'name': 'finish_task'}}]
        result = adapter.run({'task_id': self.task}, [], tools, 'baseline', boundary='preference_context')
        self.assertEqual(result['status'], 'unexecuted')
        self.assertIsNone(result['effective_path'])

    def test_capacity_rejection_does_not_send_or_expand_window(self):
        """超合法窗口保留拒绝，不能继续发模型请求或扩大配置。"""
        adapter = self.memory_adapter()
        adapter.settings = self.settings.model_copy(update={'context_window_tokens': 4096,
            'context_output_reserve': 256, 'context_safety_tokens': 256})
        state = {'task_id': self.task, 'prompt': self.db.get_task(self.task)['prompt']}
        with patch('research_loop.config.get_settings', return_value=adapter.settings):
            result = adapter.run(state, [{'role': 'system', 'content': '基础'},
                {'role': 'user', 'content': '当前要求' * 20000}], [], 'baseline', boundary='history_recall')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('CAPACITY', result['error_code'])
        self.assertEqual(adapter.settings.context_window_tokens, 4096)


class IsolationGuardTests(unittest.TestCase):
    """初始化组件前拦截生产路径、schema和索引命名空间。"""

    def test_isolation_validation_is_pure_and_rejects_production(self):
        """纯配置校验既不创建目录也不读取生产配置。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()/'new'
            config = {'root': str(root), 'database_url': 'sqlite:///' + str(root/'isolated.db'),
                      'collection_prefix': 'researchloop_eval_new', 'vector_url': 'http://127.0.0.1:6349'}
            with patch.object(Settings, 'from_env', side_effect=AssertionError('forbidden')):
                guard = IsolationGuard(config)
            self.assertFalse(root.exists())
            guard.check_collection('researchloop_eval_new_modelhash')
            with self.assertRaises(ValueError):
                guard.check_collection('researchloop_evidence_v1_modelhash')
            for changes in ({'database_url': 'mysql://u:p@127.0.0.1:33316/researchloop'},
                            {'database_url': 'mysql://u:p@external/researchloop_test_component_eval_new'},
                            {'database_url': 'sqlite:////tmp/production.db'},
                            {'collection_prefix': 'researchloop_evidence_v1'},
                            {'database_url': 'mysql://u:p@127.0.0.1:33316/researchloop_test_component_eval_new'},
                            {'database_url': 'mysql://u:p@127.0.0.1:3306/researchloop_test_component_eval_new'},
                            {'database_url': 'mysql://u:p@127.0.0.1/researchloop_test_component_eval_new'},
                            {'database_url': 'mysql+pymysql://u:p@127.0.0.1:33328/researchloop_test_component_eval_new'},
                            {'collection_prefix': 'researchloop_eval_' + 'x'*60},
                            {'vector_url': 'http://127.0.0.1:6339'},
                            {'vector_url': 'http://127.0.0.1'},
                            {'vector_url': 'http://127.0.0.1:6349/collections'},
                            {'vector_url': 'http://127.0.0.1:6349/'},
                            {'vector_url': 'https://external:6333'}):
                with self.assertRaises(ValueError):
                    validate_isolation({**config, **changes})
            mysql = validate_isolation({**config, 'database_url': 'mysql://u:p@127.0.0.1:33328/researchloop_test_component_eval_new'})
            self.assertEqual(mysql['collection_prefix'], config['collection_prefix'])
            longest = validate_isolation({**config, 'collection_prefix': 'researchloop_eval_' + 'x'*59})
            self.assertEqual(len(longest['collection_prefix'].removeprefix('researchloop_')), 64)
            with self.assertRaisesRegex(ValueError, 'OVERLAP'):
                validate_isolation(config, protected_roots=[root.parent])
            production = {'root': str(Path(temp).resolve()/'data'/'run'), 'database_url': config['database_url'],
                          'collection_prefix': config['collection_prefix']}
            with self.assertRaisesRegex(ValueError, 'OVERLAP'):
                validate_isolation(production, project_root=temp)

    def test_symlink_and_parent_segments_are_rejected(self):
        """拒绝符号链接与父目录别名，不能靠resolve落到受保护资产。"""
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            (base/'actual').mkdir()
            (base/'alias').symlink_to(base/'actual', target_is_directory=True)
            config = {'database_url': 'sqlite:///' + str(base/'actual'/'test.db'),
                      'collection_prefix': 'researchloop_eval_alias'}
            for root in (base/'alias', base/'actual'/'..'/'actual'):
                with self.assertRaisesRegex(ValueError, 'PATH_ALIAS'):
                    validate_isolation({**config, 'root': str(root)})
            (base/'actual'/'file.db').touch()
            (base/'actual'/'alias.db').symlink_to(base/'actual'/'file.db')
            with self.assertRaisesRegex(ValueError, 'DATABASE_PATH_ALIAS'):
                validate_isolation({**config, 'root': str(base/'actual'),
                    'database_url': 'sqlite:///' + str(base/'actual'/'alias.db')})


if __name__ == '__main__':
    unittest.main()
