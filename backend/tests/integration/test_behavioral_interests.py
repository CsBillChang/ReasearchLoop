"""行为兴趣的真实API/Worker链；合成附件与供应商替身，不访问生产或外网。"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlparse
from uuid import uuid4

import pymupdf
from fastapi.testclient import TestClient

from research_loop.agents.graph import initial_state
from research_loop.api.dependencies import get_runtime_settings
from research_loop.config import Settings
from research_loop.main import app
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.memory import dump
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor


class BehavioralProvider:
    """只替换供应商响应，规划、PDF读取、结束行动及画像投影均真实执行。"""

    def __init__(self, reference, candidates=(), *, partial=False, answer=None, include_candidates=True):
        """注入本例合成候选，不替业务代码写行动或画像。"""
        self.reference = reference
        self.candidates = copy.deepcopy(candidates)
        self.partial = partial
        self.include_candidates = include_candidates
        self.answer = answer or '附件描述了视觉与语言表征的匹配；本轮没有保存文件。'
        self.calls = []

    def close(self):
        """替身没有网络资源。"""

    def send(self, messages, tools, **kwargs):
        """三次固定工具响应；保留实际请求供断言，不生成额外画像请求。"""
        self.calls.append(copy.deepcopy({'messages': messages, 'tools': tools}))
        names = [c['function']['name'] for m in messages for c in m.get('tool_calls', [])]
        if 'set_research_plan' not in names:
            name = 'set_research_plan'
            args = {'goal': '解释附件的方法', 'task_kind': 'survey', 'input_refs': [self.reference],
                    'goals': [{'goal_id': 'read', 'description': '从附件解释研究方法',
                               'min_papers': 2 if self.partial else 1,
                               'requires_fulltext': True, 'input_refs': [self.reference]}]}
        elif 'read_paper' not in names:
            name, args = 'read_paper', {'target_ref': self.reference}
        else:
            name = 'finish_task'
            args = {'answer_markdown': self.answer,
                    'goals': [{'goal_id': 'read', 'status': 'unresolved' if self.partial else 'satisfied',
                               'reason': '只有一份附件，保留已读结果' if self.partial else '实际读到附件原文'}],
                    'interest_candidates': self.candidates}
            if not self.include_candidates:
                args.pop('interest_candidates')
        seq = len(self.calls)
        return {'id': 'behavioral-fixture-' + str(seq),
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                'choices': [{'finish_reason': 'tool_calls', 'message': {
                    'role': 'assistant', 'content': '', 'tool_calls': [{
                        'id': 'behavioral-call-' + str(seq), 'type': 'function',
                        'function': {'name': name, 'arguments': dump(args)}}]}}]}


class BehavioralInterestTests(unittest.TestCase):
    """验证自动投影的用户可见行为；可显式配置独立MySQL复验。"""

    def setUp(self):
        """每例临时文件与唯一主题；数据库只接受明确命名的隔离库。"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        url = os.getenv('BEHAVIORAL_INTEREST_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root / 'test.db')
        if not url.startswith('sqlite:') and not urlparse(url).path.lstrip('/').startswith('researchloop_test_interests_'):
            raise ValueError('explicit researchloop_test_interests_* database required')
        self.settings = Settings(project_root=self.root, data_dir=self.root / 'data',
            output_dir=self.root / 'notes', database_url=url, max_model_requests=6,
            reserve_model_requests=0, retrieval_enabled=False, discovery_backends={})
        self.db = Database(url)
        self.db.ensure_schema()
        self.key = 'vision-' + uuid4().hex
        self.quote = '解释附件里的开放词汇目标检测方法'
        self.attachment = {'input_ref': 'synthetic-' + uuid4().hex,
                           'local_path': str(self.root / 'synthetic.pdf')}
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_textbox((40, 40, 550, 790), '\n'.join([
                'Synthetic academic source: visual and language features are matched.'] * 28), fontsize=10)
            document.save(self.attachment['local_path'])
        for target in ('httpx.HTTPTransport.handle_request', 'httpx.AsyncHTTPTransport.handle_async_request'):
            blocker = patch(target, side_effect=AssertionError('real HTTP forbidden in behavioral interest tests'))
            blocker.start()
            self.addCleanup(blocker.stop)
        main_settings = patch('research_loop.main.get_settings', return_value=self.settings)
        main_settings.start()
        self.addCleanup(main_settings.stop)
        old_override = app.dependency_overrides.get(get_runtime_settings)
        app.dependency_overrides[get_runtime_settings] = lambda: self.settings
        self.addCleanup(self._restore_override, old_override)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.cid = self.conversation()

    @staticmethod
    def _restore_override(old):
        """恢复调用方依赖覆盖，避免污染其他集成测试。"""
        if old is None:
            app.dependency_overrides.pop(get_runtime_settings, None)
        else:
            app.dependency_overrides[get_runtime_settings] = old

    def conversation(self):
        """从正式HTTP入口新建会话。"""
        response = self.client.post('/api/conversations', json={
            'title': '合成科研使用', 'idempotency_key': uuid4().hex})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()['conversation_id']

    def submit(self, prompt=None, cid=None):
        """提交普通研究问题；不声明兴趣，不授权文件保存。"""
        response = self.client.post('/api/tasks', json={'prompt': prompt or self.quote,
            'conversation_id': cid or self.cid, 'attachments': [self.attachment],
            'save_policy': 'deny', 'idempotency_key': uuid4().hex})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['task_id']

    def candidate(self, **changes):
        """建立从当前真实用户问题提出的可撤销候选。"""
        return {'topic_key': self.key, 'content': '开放词汇目标检测',
                'reason': '本次主动提问附件中视觉语言匹配的研究方法',
                'source_quote': self.quote, **changes}

    def execute(self, tid, candidates=(), **options):
        """只用替身供应商运行正式Worker，不启动后台工作进程。"""
        provider = BehavioralProvider(self.attachment['input_ref'], candidates, **options)
        TaskExecutor(self.settings, self.db, lambda _: provider).execute(tid)
        result = self.client.get('/api/tasks/' + tid).json()
        self.assertEqual(len(provider.calls), 3, result)
        self.assertEqual(len(self.db.list_model_calls(tid)), 3)
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
        return provider, result

    def interests(self):
        """从正式个人面板接口筛选本例主题，允许MySQL套件共用隔离库。"""
        response = self.client.get('/api/profile')
        self.assertEqual(response.status_code, 200, response.text)
        return [row for row in response.json()['inferred_interests']
                if row['source'].get('topic_key') == self.key]

    def sources(self, entry):
        """按画像身份读完整来源，个人面板不重复携带增长中的原句目录。"""
        from research_loop.services.interests import InterestService
        return InterestService(self.db).sources(entry['memory_id'])

    def test_natural_question_projects_without_preference_tool_or_extra_request(self):
        """普通问题结束即有兴趣线索，来源绑定用户消息而非附件或回答。"""
        tid = self.submit()
        provider, result = self.execute(tid, [self.candidate()])
        self.assertEqual(result['status'], 'completed', result)
        entry, = self.interests()
        self.assertEqual(entry['kind'], 'inferred')
        self.assertEqual(entry['source']['strength'], 'tentative')
        self.assertEqual(entry['source']['distinct_task_count'], 1)
        link = self.db.fetchone('SELECT * FROM conversation_links WHERE task_id=?', (tid,))
        evidence, = self.sources(entry)
        self.assertEqual(evidence['task_id'], tid)
        self.assertEqual(evidence['message_id'], link['message_id'])
        self.assertEqual(evidence['quote'], self.quote)
        self.assertEqual(evidence['content_hash'], hashlib.sha256(self.quote.encode()).hexdigest())
        self.assertNotIn('evidence', entry['source'])
        self.assertNotIn('remember_preference', [a['tool_name'] for a in self.db.list_actions(tid)])
        self.assertEqual(self.db.get_task(tid)['save_policy'], 'deny')
        finish_schema = next(tool['function'] for tool in provider.calls[0]['tools']
                             if tool['function']['name'] == 'finish_task')
        self.assertIn('interest_candidates', finish_schema['parameters']['properties'])

    def test_cross_round_merge_counts_tasks_not_replayed_reads(self):
        """同主题跨轮累积；重复投影和读回不扩大来源数量。"""
        from research_loop.services.interests import InterestService
        first = self.submit()
        self.execute(first, [self.candidate(), self.candidate()])
        before, = self.interests()
        second = self.submit(self.quote + '，进一步解释语言编码器。')
        self.execute(second, [self.candidate()])
        entry, = self.interests()
        self.assertEqual(entry['source']['distinct_task_count'], 2)
        self.assertEqual(entry['source']['strength'], 'repeated')
        self.assertEqual({e['task_id'] for e in self.sources(entry)}, {first, second})
        self.assertEqual(len(self.sources(before)), 1)
        InterestService(self.db).apply(first)
        InterestService(self.db).apply(second)
        for _ in range(2):
            self.client.get('/api/conversations/' + self.cid)
            self.client.get('/api/tasks/' + second)
        self.assertEqual(self.interests(), [entry])
        self.assertGreaterEqual(entry['version'], before['version'])
        context = InterestService(self.db).context(second)
        known = next(row for row in context['known_interests'] if row['memory_id'] == entry['memory_id'])
        self.assertEqual(known['distinct_task_count'], 2)
        self.assertNotIn('evidence', known)
        self.assertNotIn('source_quote', known)
        self.assertNotIn('reason', known)

    def test_cross_conversation_same_topic_merges_without_inheriting_save_permission(self):
        """本地用户跨会话主题可累计，文件授权仍只取本轮。"""
        self.execute(self.submit(), [self.candidate()])
        second = self.submit(cid=self.conversation())
        self.execute(second, [self.candidate()])
        entry, = self.interests()
        self.assertEqual(entry['source']['distinct_task_count'], 2)
        self.assertEqual(self.db.get_task(second)['save_policy'], 'deny')

    def test_assistant_foreign_user_and_paper_quotes_are_rejected(self):
        """可见历史回答、其他会话用户和论文文字都不能替代本轮用户行为。"""
        assistant_quote = '只在助手回答出现的天体物理主题'
        self.execute(self.submit(), [], answer=assistant_quote)
        foreign = '只在其他会话用户出现的量子主题'
        self.execute(self.submit(foreign, cid=self.conversation()), [])
        paper_quote = 'visual and language features are matched'
        for quote in (assistant_quote, foreign, paper_quote):
            with self.subTest(quote=quote):
                tid = self.submit()
                _, result = self.execute(tid, [self.candidate(source_quote=quote)])
                self.assertEqual(result['status'], 'completed', result)
                self.assertEqual(self.interests(), [])

    def test_explicit_preference_is_not_overwritten(self):
        """显式改写同主题槽位后，后续行为推断不能覆盖它。"""
        self.execute(self.submit(), [self.candidate()])
        inferred, = self.interests()
        response = self.client.post('/api/profile/preferences', json={
            'slot': inferred['slot'], 'content': '只把该主题用于课程作业',
            'expected_version': inferred['version'], 'idempotency_key': uuid4().hex})
        self.assertEqual(response.status_code, 200, response.text)
        explicit = response.json()
        self.execute(self.submit(), [self.candidate()])
        self.assertEqual(self.interests(), [])
        profile = self.client.get('/api/profile').json()
        self.assertIn(explicit, profile['explicit_preferences'])

    def test_distinct_accented_topic_keys_survive_database_collation(self):
        """MySQL默认重音排序不能把程序判定不同的主题键合并。"""
        keys = [self.key + '-resume', self.key + '-résumé']
        for key in keys:
            self.execute(self.submit(), [self.candidate(topic_key=key)])
        entries = [row for row in self.client.get('/api/profile').json()['inferred_interests']
                   if row['source'].get('topic_key') in keys]
        self.assertEqual(len(entries), 2)
        self.assertEqual(len({row['slot'] for row in entries}), 2)
        self.assertTrue(all(row['source']['distinct_task_count'] == 1 for row in entries))

    def test_revocation_blocks_same_topic_and_explicit_merge_attempt(self):
        """撤销后新任务同主题和指定旧memory_id都不能复活。"""
        self.execute(self.submit(), [self.candidate()])
        old, = self.interests()
        response = self.client.post('/api/profile/forget', json={
            'memory_id': old['memory_id'], 'expected_version': old['version']})
        self.assertEqual(response.status_code, 200, response.text)
        for candidate in (self.candidate(), self.candidate(merge_memory_id=old['memory_id'],
                                                          expected_version=old['version'])):
            self.execute(self.submit(), [candidate])
            self.assertEqual(self.interests(), [])

    def test_malformed_side_field_does_not_block_answer(self):
        """整个旁路字段和单候选的坏格式均不得阻断研究交付。"""
        for candidates in (None, 'not-a-list', {'topic_key': 'bad'}, [None],
                           [self.candidate(source_quote=''), self.candidate()]):
            with self.subTest(candidates=candidates):
                _, result = self.execute(self.submit(), candidates)
                self.assertEqual(result['status'], 'completed', result)
                self.assertIn('附件描述', result['result']['answer_markdown'])

    def test_forged_source_identity_and_stale_merge_do_not_update_interest(self):
        """模型不能自报任务来源或用过期画像版本改写合并目标。"""
        forged = self.candidate(task_id='foreign-task', message_id='foreign-message')
        _, result = self.execute(self.submit(), [forged])
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(self.interests(), [])
        self.execute(self.submit(), [self.candidate()])
        original, = self.interests()
        stale = self.candidate(merge_memory_id=original['memory_id'],
                               expected_version=original['version'] + 1)
        self.execute(self.submit(), [stale])
        self.assertEqual(self.interests(), [original])
        self.assertEqual(len(self.sources(original)), 1)

    def test_partial_with_successful_finish_still_projects(self):
        """有成功结束行动的部分交付可以保留明确来源的兴趣。"""
        tid = self.submit()
        _, result = self.execute(tid, [self.candidate()], partial=True)
        self.assertEqual(result['status'], 'partial', result)
        entry, = self.interests()
        self.assertEqual(entry['source']['distinct_task_count'], 1)

    def test_successful_receipt_replay_does_not_send_or_duplicate_interest(self):
        """回退图快照重放已成功收据，不能重发请求或新增兴趣来源。"""
        tid = self.submit()
        provider, result = self.execute(tid, [self.candidate()])
        entry, = self.interests()
        state = initial_state(tid, self.quote, 'deny', {self.attachment['input_ref']: self.attachment},
                              max_model_requests=6, reserve_model_requests=0)
        state['coverage_required'] = True
        CheckpointStore(self.db).save(tid, state)
        self.db.update_task(tid, status='queued', claimed_by=None, lease_token=None, lease_until=None)
        TaskExecutor(self.settings, self.db, lambda _: provider).execute(tid)
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(len(self.db.list_model_calls(tid)), 3)
        self.assertEqual(self.client.get('/api/tasks/' + tid).json()['status'], result['status'])
        self.assertEqual(self.interests(), [entry])

    def test_successful_old_finish_receipt_without_side_field_is_compatible(self):
        """旧响应不含画像字段仍正常完成，恢复不补发画像或研究请求。"""
        tid = self.submit()
        provider, result = self.execute(tid, include_candidates=False)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(self.interests(), [])
        state = initial_state(tid, self.quote, 'deny', {self.attachment['input_ref']: self.attachment},
                              max_model_requests=6, reserve_model_requests=0)
        state['coverage_required'] = True
        CheckpointStore(self.db).save(tid, state)
        self.db.update_task(tid, status='queued', claimed_by=None, lease_token=None, lease_until=None)
        TaskExecutor(self.settings, self.db, lambda _: provider).execute(tid)
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(len(self.db.list_model_calls(tid)), 3)
        self.assertEqual(self.interests(), [])

    def test_real_clarification_answer_is_a_valid_user_source(self):
        """候选可引用真实resume答案，并保留clarification身份。"""
        answer = '重点解释视觉语言匹配。'
        tid = self.submit('请读附件，我稍后补充问题。')
        provider = BehavioralProvider(self.attachment['input_ref'], [self.candidate(source_quote=answer)])
        original_send = provider.send

        def send(messages, tools, **kwargs):
            """先真实澄清一次，恢复后仍走同一供应商替身与正式图。"""
            called = [c['function']['name'] for m in messages for c in m.get('tool_calls', [])]
            if 'request_clarification' in called:
                return original_send(messages, tools, **kwargs)
            provider.calls.append(copy.deepcopy({'messages': messages, 'tools': tools}))
            return {'id': 'clarification-fixture',
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': '',
                    'tool_calls': [{'id': 'clarification-call', 'type': 'function', 'function': {
                        'name': 'request_clarification', 'arguments': dump({
                            'question': '需要解释哪个方面？', 'fields': ['scope'], 'suggested_values': []})}}]}}]}

        with patch.object(provider, 'send', side_effect=send):
            TaskExecutor(self.settings, self.db, lambda _: provider).execute(tid)
            self.assertEqual(self.db.get_task(tid)['status'], 'needs_input')
            question = self.db.get_pending_clarification(tid)
            response = self.client.post('/api/tasks/' + tid + '/resume', json={
                'clarification_id': question['clarification_id'], 'answer': answer,
                'idempotency_key': uuid4().hex})
            self.assertEqual(response.status_code, 200, response.text)
            TaskExecutor(self.settings, self.db, lambda _: provider).execute(tid)
        self.assertEqual(self.db.get_task(tid)['status'], 'completed')
        self.assertEqual(len(provider.calls), 4)
        entry, = self.interests()
        evidence, = self.sources(entry)
        self.assertEqual(evidence['quote'], answer)
        self.assertEqual(evidence['source_id'], question['clarification_id'])
        self.assertEqual(evidence['source_type'], 'clarification')

    def test_projection_failure_keeps_answer_then_database_only_recovery(self):
        """完成后画像投影故障不改答案，恢复只消费已有成功行动。"""
        from research_loop.services.interests import InterestService
        tid = self.submit()
        with patch.object(InterestService, 'apply', side_effect=RuntimeError('synthetic projection outage')):
            provider, result = self.execute(tid, [self.candidate()])
        self.assertEqual(result['status'], 'completed', result)
        self.assertIn('附件描述', result['result']['answer_markdown'])
        self.assertEqual(self.interests(), [])
        InterestService(self.db).recover_pending()
        entry, = self.interests()
        self.assertEqual(entry['source']['distinct_task_count'], 1)
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(self.client.get('/api/conversations/' + self.cid).json()['total'], 2)

    def test_get_routes_neither_apply_interests_nor_send_provider_requests(self):
        """GET只读画像；不把后台恢复隐藏在面板或会话读回里。"""
        from research_loop.services.interests import InterestService
        tid = self.submit()
        provider, _ = self.execute(tid, [self.candidate()])
        entry, = self.interests()
        with patch.object(InterestService, 'apply', side_effect=AssertionError('GET must not apply')):
            for path in ('/api/profile', '/api/conversations', '/api/conversations/' + self.cid,
                         '/api/tasks/' + tid):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.interests(), [entry])
        self.assertEqual(len(provider.calls), 3)

    def test_legacy_task_without_enable_marker_is_not_backfilled(self):
        """无新版启用标记的旧任务即使有成功候选行动也不回填画像。"""
        from research_loop.services.interests import InterestService
        tid = self.submit()
        self.db.execute('DELETE FROM interest_projections WHERE task_id=?', (tid,))
        provider, result = self.execute(tid, [self.candidate()])
        self.assertEqual(result['status'], 'completed', result)
        InterestService(self.db).apply(tid)
        InterestService(self.db).recover_pending()
        self.assertEqual(self.interests(), [])
        self.assertEqual(len(provider.calls), 3)

    def test_cancelled_and_exception_without_finish_do_not_infer(self):
        """取消或异常且无成功结束行动，不追加抽取调用。"""
        from research_loop.services.interests import InterestService
        tid = self.submit()
        response = self.client.post('/api/tasks/' + tid + '/cancel')
        self.assertEqual(response.status_code, 200, response.text)
        provider = BehavioralProvider(self.attachment['input_ref'], [self.candidate()])
        TaskExecutor(self.settings, self.db, lambda _: provider).execute(tid)
        self.assertEqual(provider.calls, [])
        failed = self.submit()
        broken = BehavioralProvider(self.attachment['input_ref'], [self.candidate()])
        with patch.object(broken, 'send', side_effect=RuntimeError('synthetic provider failure')) as send:
            TaskExecutor(self.settings, self.db, lambda _: broken).execute(failed)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(self.db.get_task(failed)['status'], 'failed')
        InterestService(self.db).recover_pending()
        self.assertEqual(self.interests(), [])


if __name__ == '__main__':
    unittest.main()
