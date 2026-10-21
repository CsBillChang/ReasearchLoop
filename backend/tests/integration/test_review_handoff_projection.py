"""大型核验回传实际API/Worker验证，供应商HTTP和PDF使用公开合成替身。"""
from copy import deepcopy
import json
import re
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from integration import test_research_reflection as base
from integration.test_context_continuity import memory_payload
from integration.test_handoff_archive_delivery import HandoffArchiveScenario
from research_loop.agents.memory import estimate
from research_loop.domain.memory import HistoryRead
from research_loop.api.dependencies import get_runtime_settings
from research_loop.main import app
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.storage.memory import dump
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.tools.registry import ToolRegistry
from research_loop.workers.executor import TaskExecutor


def resolve_path(value, path):
    """仅解释当前请求中明确标出的本地JSON路径，不读取服务内部状态。"""
    for key, index in re.findall(r'([a-z_]+)|\[(\d+)\]', path):
        value = value[int(index)] if index else value[key]
    return value


class LargeReviewScenario(base.ResearchReflectionScenario):
    """固定大意见负载，依据实际可见投影选择局部修订或准确结束。"""

    def __init__(self, case, kind):
        """保存实际发送请求，合成替身不代表模型语义能力。"""
        super().__init__(kind); self.case = case; self.actual_requests = []; self.compacted_seen = False

    def factory(self, settings):
        """复用已有完整SSE/JSON编码，避免用JSON冒充同会话流式供应商。"""
        return HandoffArchiveScenario.factory(self, settings)

    def handler(self, request):
        """解析上下文中的精确引用，再用已有合成流程返回工具调用。"""
        body = json.loads(request.content); self.actual_requests.append(deepcopy(body))
        if not base.is_reviewer_request(body):
            self.case.assertLessEqual(estimate({'messages':body['messages'], 'tools':body.get('tools', [])}), 115712)
            for message in body['messages']:
                if message.get('role') != 'tool':
                    continue
                observation = json.loads(message['content']); data = observation.get('data') or {}
                if 'context_path' not in data:
                    continue
                context = memory_payload(body['messages'])
                review = deepcopy(resolve_path(context, data['context_path']))
                self.case.assertEqual(len(review['findings']), 7)
                self.case.assertEqual(context['current_task']['save_policy'], 'deny')
                self.compacted_seen = True
                # 替身仅按模型可见的引用重建，不替业务落库或强制修改判断。
                for issue in review['issues']:
                    if 'finding_in' in issue:
                        finding = resolve_path(context, issue.pop('finding_in')); issue.update(finding)
                for check in review['pending_checks']:
                    if 'findings_in' in check:
                        check['findings'] = resolve_path(context, check.pop('findings_in'))
                observation['data'] = review; message['content'] = json.dumps(observation)
            request = httpx.Request(request.method, request.url, content=json.dumps(body).encode())
        response = super().handler(request)
        if base.is_reviewer_request(body):
            raw = response.json()
            call = raw['choices'][0]['message']['tool_calls'][0]
            args = json.loads(call['function']['arguments']); single = args['findings'][0]
            args['findings'] = [{**single, 'reason': f'条件{i}：'+('逐字保留不同核验限定，合成负载不是真实研究结论。'*36)} for i in range(7)]
            call['function']['arguments'] = json.dumps(args)
            response = httpx.Response(200, json=raw)
        return response


class ReviewHandoffProjectionTests(unittest.TestCase):
    """SQLite默认隔离，显式环境变量可改用专用researchloop_test_* MySQL。"""

    def setUp(self):
        """复用合成材料和隔离数据库入口，不触及生产文件。"""
        base.ResearchReflectionTests.setUp(self)
        # Token估算小于旧字节估算，显式冻结较早触发投影的测试阈值。
        self.settings=self.settings.model_copy(update={'context_compress_ratio':0.1})

    def run_projection_case(self, kind, without_readback=False, recovery=False):
        """通过当前API→Worker完成大意见交接，并逐字读回完整原件。"""
        scenario = LargeReviewScenario(self, kind)
        cid = ConversationService(self.db, self.settings).create('合成核验回传', uuid4().hex)['conversation_id']
        app.dependency_overrides[get_runtime_settings] = lambda:self.settings
        self.addCleanup(app.dependency_overrides.pop, get_runtime_settings, None)
        executor = TaskExecutor(self.settings, self.db, scenario.factory, lambda *_:base.ScopedPDF())
        original_init = ToolRegistry.__init__
        def scoped_init(registry, specs=None, *, allowed_names=None):
            """仅收缩合成用例工具范围，不给生产或其他角色增加权限。"""
            if without_readback:
                names = {s.name for s in specs or []} - {'read_memory'}
                allowed_names = names if allowed_names is None else names & set(allowed_names)
            return original_init(registry, specs, allowed_names=allowed_names)
        with TestClient(app) as client, patch.object(ToolRegistry, '__init__', scoped_init), \
                patch.object(httpx.HTTPTransport, 'handle_request', side_effect=AssertionError('external network forbidden')):
            created = client.post('/api/tasks', json={'conversation_id':cid, 'prompt':'按需核验当前问题，禁止保存。',
                'save_policy':'deny', 'attachments':[{'input_ref':'input-A','paper_id':'paper-A','local_path':str(self.root/'A.pdf')}]})
            self.assertEqual(created.status_code, 202, created.text)
            tid = created.json()['task_id']
            saved = {}
            if recovery:
                original_save = CheckpointStore.save
                stopped = []
                def interrupt(store, task_id, state):
                    """核验后模型选择修订的成功收据已写入、工具尚未执行时中断。"""
                    if not stopped and task_id == tid and any(c['name']=='revise_research' for c in state.get('pending_tool_calls', [])):
                        stopped.append(task_id); raise SystemExit('AFTER_COMPACT_REVIEW_RECEIPT')
                    return original_save(store, task_id, state)
                with patch.object(CheckpointStore, 'save', interrupt), self.assertRaises(SystemExit):
                    executor.execute(tid)
                saved = {row['call_id']:row for row in self.db.fetchall('SELECT * FROM model_calls WHERE task_id=?', (tid,))}
                self.db.execute('UPDATE tasks SET lease_until=? WHERE root_task_id=?', ('2000-01-01T00:00:00Z',tid))
            executor.execute(tid, 'restored' if recovery else 'direct')
            view = client.get('/api/tasks/'+tid).json()
            self.assertEqual(view['status'], 'partial' if kind=='missing' else 'completed', view.get('error') or view['result'].get('reason'))
            self.assertTrue(all(a['status']=='succeeded' for a in self.db.list_actions(tid)))
            before = deepcopy(self.db.fetchall('SELECT * FROM model_calls WHERE task_id=? OR task_id IN (SELECT task_id FROM tasks WHERE root_task_id=?)', (tid,tid)))
            count = len(scenario.actual_requests)
            self.db.update_task(tid, status='queued', result_json=None)
            executor.execute(tid, 'restored')
            self.assertEqual(len(scenario.actual_requests), count)
            self.assertEqual(client.get('/api/tasks/'+tid).json()['result'], view['result'])
            self.assertEqual(self.db.fetchall('SELECT * FROM model_calls WHERE task_id=? OR task_id IN (SELECT task_id FROM tasks WHERE root_task_id=?)', (tid,tid)), before)
            for row in self.db.list_model_calls(tid):
                if row['call_id'] in saved: self.assertEqual(row, saved[row['call_id']])
        self.assertTrue(scenario.compacted_seen)
        rows = self.db.fetchall('SELECT * FROM context_archives WHERE task_id=? ORDER BY sequence_no', (tid,))
        projected = [row for row in rows if json.loads(row['metrics_json']).get('review_projected_calls')]
        self.assertTrue(projected)
        for row in projected:
            context = memory_payload(json.loads(row['projected_json']))
            for review in context['current_task']['research_reviews']:
                if without_readback:
                    self.assertNotIn('original_result_readback', review)
                    continue
                pointer = review['original_result_readback']; args = HistoryRead.model_validate(pointer['readback'])
                chunks = []; remaining = pointer['value_chars']
                while remaining:
                    value = MemoryService(self.db).read(tid, args)['content'][:remaining]
                    self.assertTrue(value); chunks.append(value); remaining -= len(value)
                    args = args.model_copy(update={'offset': args.offset + len(value)})
                expected = json.loads(row['original_json'])
                for key in pointer['original_path']: expected = expected[key]
                self.assertEqual(json.loads(''.join(chunks)), expected)
                self.assertEqual(len(expected['findings']), 7)
                self.assertTrue(all('source_quote' in f for f in expected['findings']))
                with self.assertRaises(KeyError): MemoryService(self.db).read('foreign-task', HistoryRead.model_validate(pointer['readback']))
        result = view['result']
        self.assertEqual(result['saved_path'], None)
        self.assertEqual(self.db.get_task(tid)['save_policy'], 'deny')
        for request in scenario.actual_requests:
            self.assertEqual(estimate({'messages': request['messages'], 'tools': request.get('tools', [])}) <= 115712, True)
        return tid, result

    def test_large_review_local_revision_and_receipt_reuse(self):
        """大核验结果后可局部修订再核验，已成功收据恢复不重发。"""
        _, result = self.run_projection_case('answer')
        self.assertEqual(result['answer_body_markdown'], '保留开头。方法使用语言前缀。保留结尾。')

    def test_large_review_uncertainty_can_finish_partial(self):
        """七条缺证意见保持原判断，允许结束并保留成果，不要求全部通过。"""
        _, result = self.run_projection_case('missing')
        self.assertEqual(result['research_reviews'][0]['quality_status'], 'issues')
        self.assertTrue(result['answer_body_markdown'])

    def test_no_memory_tool_keeps_lossless_local_links_without_readback(self):
        """去重引用仍在当前输入内，工具未授权时不假造档案读取入口。"""
        self.run_projection_case('missing', without_readback=True)

    def test_post_review_success_receipt_survives_process_exit(self):
        """紧凑输入生成的修订响应恢复后复用原收据，不再次发送。"""
        self.run_projection_case('answer', recovery=True)
