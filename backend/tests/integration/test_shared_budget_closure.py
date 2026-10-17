"""冻结预算读回与父子提前收尾：API/事务/Worker真实，模型与PDF显式替身。"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from research_loop.config import Settings
from research_loop.domain.tasks import TaskCreate
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.services.conversations import ConversationService
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_adaptive_research import AdaptiveScenario
from integration.test_collaboration_contract import FixturePDF


class ClosingProvider:
    """等到预算收尾schema再交接，确保测试不会靠模型自行提前结束掩盖缺陷。"""

    def __init__(self, save, partial=False):
        """同步两个子角色首读，制造真实的并发共享额度消耗。"""
        self.save = save
        self.partial = partial
        self.requests = []
        self.barrier = threading.Barrier(2, timeout=10)

    def handler(self, request):
        """仅响应内存HTTP传输，所有工具仍走生产执行入口。"""
        body = json.loads(request.content)
        self.requests.append(body)
        messages = body['messages']; validate_messages(messages)
        names = {t['function']['name'] for t in body['tools']}
        done = [c['function']['name'] for m in messages for c in m.get('tool_calls', [])]
        observations = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
        def reply(name, args):
            """会话父角色返回真实SSE形状，子角色返回普通JSON；二者均为替身。"""
            response = AdaptiveScenario.response(name, args)
            if not body.get('stream'):
                return response
            value = response.json()
            choice = value['choices'][0]
            message = choice['message']
            for index, call in enumerate(message.get('tool_calls', [])):
                call['index'] = index
            chunk = {**value, 'choices': [{'index': 0, 'delta': message,
                                          'finish_reason': choice['finish_reason']}]}
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                content='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n')
        if 'finish_research' in names:
            if names == {'finish_research'}:
                data = next(o['data'] for o in observations if o.get('data', {}).get('pages'))
                return reply('finish_research', {'status': 'partial' if self.partial else 'completed',
                    'gaps': ['合成材料未提供训练条件'] if self.partial else [],
                    'findings': ['合成原文使用语言嵌入。'],
                    'paper_ids': [data['paper_id']], 'evidence_ids': [p['evidence_id'] for p in data['pages']]})
            assert 'read_paper' not in done, 'child should enter closing after its first read'
            brief = json.loads(next(m['content'] for m in messages if m['role'] == 'user' and
                                   m['content'].startswith('结构化子任务')).split('\n', 1)[1])
            self.barrier.wait()
            return reply('read_paper', {'target_ref': brief['input_refs'][0]})
        if 'set_research_plan' not in done:
            return reply('set_research_plan', {'goal': '比较两篇', 'task_kind': 'compare',
                'save_requested': self.save, 'goals': [{'goal_id': 'compare', 'description': '对照语言信息',
                                                       'min_papers': 2, 'requires_fulltext': True}]})
        if 'delegate_research' not in done:
            return reply('delegate_research', {'subtasks': [{'mode': 'investigate', 'objective': '语言信息'+name,
                'input_refs': ['input-'+name], 'goal_ids': ['compare']} for name in ('A', 'B')]})
        assert not names & {'read_paper', 'retrieve_evidence', 'delegate_research', 'review_note'}
        children = next(o['data']['subtasks'] for o in observations if o.get('data', {}).get('subtasks'))
        for child in children:
            if child.get('findings_in'):
                from integration.test_context_continuity import memory_payload
                canonical = next(c for c in memory_payload(messages)['current_task']['subtasks']
                                 if c['subtask_id'] == child['subtask_id'])
                child['findings'] = [f['text'] for f in canonical['research_progress']['findings']] + child.get('findings', [])
        assert all(c['findings'] for c in children)
        if self.save:
            if 'create_note' not in done:
                return reply('create_note', {'title': '预算收尾测试', 'markdown': '# 比较\n两份合成材料使用语言嵌入。',
                    'evidence_ids': [e for c in children for e in c['evidence_ids']]})
            draft = next(o['data'] for o in observations if o.get('data', {}).get('draft_id'))
            if 'save_note' not in done:
                return reply('save_note', {'draft_id': draft['draft_id'], 'content_hash': draft['content_hash']})
        return reply('finish_task', {'answer_markdown': '两份合成材料的比较已交付。'+('训练条件未提供，保留缺口。' if self.partial else ''),
            'draft_id': draft['draft_id'] if self.save else None,
            'goals': [{'goal_id': 'compare', 'status': 'unresolved' if self.partial else 'satisfied', 'reason': '依据已读取的合成原文',
                       'subtask_ids': [c['subtask_id'] for c in children]}]})

    def factory(self, settings):
        """强制使用MockTransport；不连接任何真实供应商。"""
        return DeepSeekProvider('fixture', client=httpx.Client(transport=httpx.MockTransport(self.handler)))


class SharedBudgetClosureTests(unittest.TestCase):
    """默认临时SQLite；MySQL只接受显式新建的专用测试库。"""

    def setUp(self):
        """独立文件目录和幂等身份，禁止沿用生产库或清理旧数据。"""
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        url = os.getenv('SOURCE_BUDGET_DATABASE_URL') or 'sqlite:///' + str(self.root/'test.db')
        if not url.startswith('sqlite:') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_source_budget_'):
            raise ValueError('dedicated source budget test database required')
        self.settings = Settings(project_root=self.root, data_dir=self.root/'data', output_dir=self.root/'notes',
            database_url=url, max_model_requests=20, max_tool_actions=48, reserve_model_requests=4,
            retrieval_enabled=False, task_token_limit=800000)
        self.db = Database(url); self.db.ensure_schema(); self.key = uuid4().hex
        self.tasks = TaskService(self.db, self.settings)

    def test_api_both_entries_persist_budget_and_replay_does_not_rewrite(self):
        """直接和会话API均读回20/48/4；旧任务不会被后来的配置覆盖。"""
        from research_loop.main import app
        cid = ConversationService(self.db, self.settings).create('预算', self.key)['conversation_id']
        with patch('research_loop.api.dependencies.get_settings', return_value=self.settings), TestClient(app) as client:
            for suffix, conversation in [('direct', None), ('chat', cid)]:
                payload = {'prompt': '比较已有附件，不保存', 'idempotency_key': self.key+suffix, 'conversation_id': conversation}
                response = client.post('/api/tasks', json=payload)
                self.assertEqual(response.status_code, 202, response.text)
                tid = response.json()['task_id']
                budget = self.db.assert_budget_policy(tid, self.settings)
                self.assertEqual([budget[k] for k in ('model_limit', 'tool_limit', 'reserve_tail')], [20, 48, 4])
                old = dict(budget)
                changed = self.settings.model_copy(update={'max_model_requests': 30})
                TaskService(self.db, changed).create(TaskCreate(**payload))
                self.assertEqual(self.db.budget_summary(tid), old)
                with self.assertRaisesRegex(ValueError, 'BUDGET_POLICY_MISMATCH'):
                    self.db.assert_budget_policy(tid, changed)
        self.assertFalse(self.db.list_model_calls(tid))

    def test_concurrent_children_cannot_take_parent_tail(self):
        """数据库原子条件保护16个子请求槽位，根角色另外保留4个。"""
        tid = self.tasks.create(TaskCreate(prompt='并发预算', idempotency_key=self.key)).task_id
        with ThreadPoolExecutor(max_workers=4) as pool:
            accepted = list(pool.map(lambda _: self.db.reserve_model_slot(tid, 'researcher'), range(24)))
        self.assertEqual(sum(accepted), 16)
        self.assertEqual([self.db.reserve_model_slot(tid) for _ in range(5)], [True]*4+[False])
        self.db.assert_budget_policy(tid, self.settings)

    def test_cancelled_task_does_not_call_provider(self):
        """取消不因新增预算检查而触发模型或重新开启任务。"""
        tid = self.tasks.create(TaskCreate(prompt='取消', idempotency_key=self.key)).task_id
        self.tasks.cancel(tid)
        forbidden = lambda _: self.fail('provider must not be constructed')
        TaskExecutor(self.settings, self.db, forbidden).execute(tid)
        self.assertFalse(self.db.list_model_calls(tid))

    def test_invalid_policy_rolls_back_task_and_budgets(self):
        """非法额度不得留下已创建但无一致策略的半成品任务。"""
        settings = self.settings.model_copy(update={'max_tool_actions': 0})
        before = {table: self.db.fetchone('SELECT COUNT(*) AS n FROM '+table)['n']
                  for table in ('tasks', 'task_budgets', 'task_token_budgets')}
        with self.assertRaisesRegex(ValueError, 'INVALID_REQUEST_BUDGET'):
            TaskService(self.db, settings).create(TaskCreate(prompt='非法配置', idempotency_key=self.key))
        for table, count in before.items():
            self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM '+table)['n'], count)

    def run_closing(self, save, partial=False, tool_bound=False):
        """API→两个并行Researcher→父角色收尾保存/交付，恢复不多扣额度。"""
        from research_loop.main import app
        settings = self.settings.model_copy(update={'max_model_requests': 20 if tool_bound else 10,
                                                    'max_tool_actions': 10 if tool_bound else 48})
        cid = ConversationService(self.db, settings).create('收尾', self.key)['conversation_id']
        attachments = []
        for name in ('A', 'B'):
            path = self.root/(name+'.pdf'); path.write_bytes(b'%PDF-fixture')
            attachments.append({'input_ref': 'input-'+name, 'paper_id': 'paper-'+name, 'local_path': str(path)})
        provider = ClosingProvider(save, partial)
        read_barrier = threading.Barrier(2, timeout=10)

        class ConcurrentPDF(FixturePDF):
            """两次读页均已扣除工具额度再返回，确定性复现工具额度先耗尽。"""
            def pages(self, document, start, count):
                """不改合成正文，只同步两个并发工具观察的交接点。"""
                read_barrier.wait()
                return super().pages(document, start, count)

        executor = TaskExecutor(settings, self.db, provider.factory,
                                lambda s, f: ConcurrentPDF() if tool_bound else FixturePDF())
        with patch('research_loop.api.dependencies.get_settings', return_value=settings), TestClient(app) as client:
            response = client.post('/api/tasks', json={'prompt': '比较两篇附件'+('并保存' if save else '不要保存'),
                'conversation_id': cid, 'attachments': attachments, 'save_policy': 'allow' if save else 'deny'})
            self.assertEqual(response.status_code, 202, response.text)
            tid = response.json()['task_id']
            self.db.assert_budget_policy(tid, settings)  # 发送任何替身请求前就检查，后验不代替预检。
            executor.execute(tid)
            view = client.get('/api/tasks/'+tid).json()
        self.assertEqual(view['status'], 'partial' if partial else 'completed', view)
        self.assertEqual(bool(view['result']['gaps']), partial)
        self.assertEqual(len(provider.requests), 9 if save else 7)
        self.assertEqual(self.db.budget_summary(tid)['model_used'], len(provider.requests))
        self.assertEqual(view['usage']['unknown'], 0)
        self.assertEqual(view['usage']['not_sent'], 0)
        self.assertEqual(len(list(settings.output_dir.rglob('*.md'))), int(save))
        calls = self.db.list_model_calls(tid)
        self.assertTrue(all(json.loads(c['request_json']).get('settling') for c in calls[2:]))
        original = json.dumps(calls, sort_keys=True)
        self.db.update_task(tid, status='queued', result_json=None)
        executor.execute(tid, 'recovery')
        self.assertEqual(json.dumps(self.db.list_model_calls(tid), sort_keys=True), original)
        self.assertEqual(len(provider.requests), 9 if save else 7)

    def test_parallel_handoff_and_authorized_save_before_cap(self):
        """预留段允许已有成果落稿/保存/结束，不被过早finish-only吞掉。"""
        self.run_closing(True)

    def test_parallel_handoff_and_deny_delivery_before_cap(self):
        """相同父子路径在deny下交付正文，无保存文件。"""
        self.run_closing(False)

    def test_partial_children_keep_findings_and_gaps_at_tail(self):
        """共享额度收尾不把缺证升级为成功，已有正文与缺口一并交付。"""
        self.run_closing(False, partial=True)

    def test_tool_limit_also_preserves_parent_delivery(self):
        """模型额度充足但工具额度将尽时，子角色交接、父角色保存并结束。"""
        self.run_closing(True, tool_bound=True)
