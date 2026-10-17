"""API→Worker/Supervisor→单次视觉子图→HTTP替身→交付，支持显式隔离MySQL。"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from research_loop.agents.visual_reading import VisualReadingNode
from research_loop.api.dependencies import get_runtime_settings
from research_loop.config import Settings
from research_loop.domain.evidence import Document, PageEvidence, PaperCandidate
from research_loop.domain.visual_reading import VisualDelegationInput
from research_loop.main import app
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.storage.token_budget import TokenBudgetStore
from research_loop.workers.executor import TaskExecutor
from unit.test_visual_reading import fixture


class VisualDelegationIntegrationTests(unittest.TestCase):
    """实际角色图和SQL执行；自然语言选择与视觉事实质量由替身控制，不能冒充真实语义验收。"""

    def setUp(self):
        """每例独立材料/任务，所有外部HTTP一律拦截。"""
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.files, self.pdf_path, self.item, self.pages = fixture(self.root)
        url = os.getenv('VISUAL_AGENT_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root / 'test.db')
        if not url.startswith('sqlite:') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_visual_agent_'):
            raise ValueError('explicit isolated visual Agent database required')
        self.settings = Settings(project_root=self.root, data_dir=self.files.data_dir,
            output_dir=self.files.output_dir, database_url=url, retrieval_enabled=False,
            deepseek_model='deepseek-chat', deepseek_api_key='synthetic-key')
        self.db = Database(url); self.db.ensure_schema()
        self.tasks, self.requests = [], []
        self.visual, self.failure, self.long_answer, self.reference = True, None, False, self.item.target_ref
        old = app.dependency_overrides.copy()
        app.dependency_overrides[get_runtime_settings] = lambda: self.settings
        def restore():
            """只恢复当前测试的依赖覆盖。"""
            app.dependency_overrides.clear(); app.dependency_overrides.update(old)
        self.addCleanup(restore)
        self.api = TestClient(app); self.api.__enter__(); self.addCleanup(self.api.__exit__, None, None, None)
        guard = patch.object(httpx.HTTPTransport, 'handle_request', side_effect=AssertionError('real HTTP forbidden'))
        guard.start(); self.addCleanup(guard.stop)
        no_parse = patch.object(PDFProvider, '_parse', side_effect=AssertionError('no full PDF reparse'))
        no_parse.start(); self.addCleanup(no_parse.stop)
        self.pdf = PDFProvider(self.files); self.addCleanup(self.pdf.close)
        self.papers = PaperService(self.db, None, self.pdf)
        self.executor = TaskExecutor(self.settings, self.db, self.factory)
        self.addCleanup(self.cleanup_tasks)

    def cleanup_tasks(self):
        """保留隔离记录，终态化本例所有父子任务。"""
        for tid in self.tasks:
            for child in self.db.list_subtasks(tid):
                self.db.update_task(child['subtask_id'], status='cancelled', cancel_requested=1)
            self.db.update_task(tid, status='cancelled', cancel_requested=1)

    def factory(self, settings):
        """实际适配器仅通过MockTransport；节点独立模式不改父配置。"""
        return DeepSeekProvider('synthetic-key', model=settings.deepseek_model,
            base_url=settings.deepseek_base_url, timeout=settings.request_timeout_seconds,
            client=httpx.Client(transport=httpx.MockTransport(self.transport), timeout=settings.request_timeout_seconds))

    def transport(self, request):
        """按实际聊天流合同包装同一替身响应，视觉节点仍是普通单次HTTP。"""
        response = self.respond(request)
        if not json.loads(request.content).get('stream') or response.status_code >= 400:
            return response
        raw = response.json(); choice = raw['choices'][0]
        delta = deepcopy(choice['message'])
        for index, call in enumerate(delta.get('tool_calls') or []):
            call['index'] = index
        chunk = {**{k: raw[k] for k in ('id', 'model', 'usage')},
                 'choices': [{'index': 0, 'delta': delta, 'finish_reason': choice['finish_reason']}]}
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
            content='data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\ndata: [DONE]\n\n')

    def response(self, name=None, arguments=None, content=''):
        """产生带真实形状usage的合成响应，不将其计为供应商调用。"""
        message = {'role': 'assistant', 'content': content}
        if name:
            message['tool_calls'] = [{'id': uuid4().hex, 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}}]
        return httpx.Response(200, json={'id': 'synthetic-' + uuid4().hex, 'model': 'deepseek-flash',
            'choices': [{'message': message, 'finish_reason': 'tool_calls' if name else 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})

    def respond(self, request):
        """固定Supervisor先读文字再可选委派；只验证实际代码接线，不测试模型理解能力。"""
        body = json.loads(request.content); self.requests.append(body)
        validate_messages(body['messages'])
        if body['messages'][0]['content'].startswith('依据提供的论文页'):
            if self.failure == 'timeout':
                raise httpx.ReadTimeout('synthetic visual timeout')
            if self.failure == 'rejected':
                return httpx.Response(400, json={'error': 'synthetic image rejection'})
            if self.failure == 'exit':
                raise SystemExit('controlled send-window exit')
            return self.response(content='甲' * 7001 if self.long_answer else '合成视觉观察：式(2)为逐元素相乘；此结果未语义核验。')
        calls = [c['function']['name'] for m in body['messages'] for c in m.get('tool_calls', [])]
        observations = [json.loads(m['content']) for m in body['messages'] if m['role'] == 'tool']
        if 'set_research_plan' not in calls:
            return self.response('set_research_plan', {'goal': '回答附件的定向问题', 'requested_count': 1,
                'count_confirmed': True, 'task_kind': 'compare', 'input_refs': [self.reference] if self.reference == self.item.target_ref else [],
                'goals': [{'goal_id': 'g1', 'description': '回答方法问题', 'min_papers': 1, 'requires_fulltext': True}]})
        if 'read_paper' not in calls:
            return self.response('read_paper', {'target_ref': self.reference, 'start_page': 1, 'page_count': 1})
        if self.visual and 'delegate_visual_reading' not in calls:
            eid = next(o['data']['pages'][0]['evidence_id'] for o in observations if o.get('data', {}).get('pages'))
            if self.failure == 'missing-cache':
                next(self.files.parse_dir.glob('*.json')).unlink()
            return self.response('delegate_visual_reading', self.args(evidence_id=eid).model_dump(mode='json'))
        failed = next((o for o in observations if not o.get('ok', False)), None)
        if self.failure == 'parent-timeout':
            raise httpx.ReadTimeout('synthetic parent failure after visual result')
        answer = '视觉读取不可用，保留文字成果并披露具体失败。' if failed else '合成定向回答已交付，视觉观察不是原文核验通过。'
        return self.response('finish_task', {'answer_markdown': answer, 'goals': [{'goal_id': 'g1',
            'status': 'unresolved' if failed else 'satisfied', 'subtask_ids': [],
            'reason': failed['message'] if failed else '原页文字及所需观察已取得'}]})

    def args(self, **updates):
        """使用原页定位，不让模型控制source hash或本机路径。"""
        return VisualDelegationInput.model_validate({'evidence_id': self.item.model_dump().get('evidence_id', self.pages[0]['evidence_id']),
            'question': self.item.question, 'reason': '公式符号损坏', 'trigger': 'damaged_formula', **updates})

    def create(self, prompt='只依据附件，回答式(2)的符号，不保存。', attachments=True, conversation=False):
        """正常HTTP入口创建任务；真实Worker随后领取。"""
        payload = {'prompt': prompt, 'save_policy': 'deny',
            'idempotency_key': uuid4().hex, 'attachments': [{'input_ref': self.item.target_ref,
                'local_path': str(self.pdf_path), 'paper_id': 'synthetic-paper'}] if attachments else []}
        if conversation:
            created = self.api.post('/api/conversations', json={'title': '合成视觉接线', 'idempotency_key': uuid4().hex})
            self.assertEqual(created.status_code, 201, created.text)
            payload['conversation_id'] = created.json()['conversation_id']
        response = self.api.post('/api/tasks', json=payload)
        self.assertEqual(response.status_code, 202, response.text)
        tid = response.json()['task_id']; self.tasks.append(tid)
        return tid

    def prepare(self, **kwargs):
        """为边界测试实际领取任务并读取页，跳过替身规划。"""
        tid = self.create(**kwargs)
        task = self.db.claim_task('visual-direct-test', task_id=tid)
        attached = {a['input_ref']: a for a in json.loads(task['attachments_json'])}
        self.papers.read(tid, self.item.target_ref, 1, 1, attached)
        return tid, ('visual-direct-test', task['lease_token'])

    def dispatch(self, tid, actor, args=None):
        """通过实际Worker委派回调运行单次子图。"""
        return self.executor._visual_once(tid, args or self.args(), *actor, self.settings, self.papers)

    def visual_requests(self):
        """只统计替身的图片请求，不把规划请求混作视觉调用。"""
        return [r for r in self.requests if r['messages'][0]['content'].startswith('依据提供的论文页')]

    def test_text_default_never_runs_visual_node(self):
        """普通任务经正常Supervisor/Worker交付，视觉可用也不强制调用。"""
        self.visual = False
        tid = self.create(); result = self.executor.execute(tid)
        self.assertTrue(result.answer_markdown)
        self.assertEqual(self.db.get_task(tid)['status'], 'completed')
        self.assertEqual(self.visual_requests(), [])
        self.assertEqual(self.db.list_subtasks(tid), [])
        self.assertEqual(len(self.requests), 3)

    def test_api_worker_visual_child_handoff_and_budget(self):
        """实际图委派一次视觉节点，父角色继续交付，来源与额度都可读回。"""
        tid = self.create(conversation=True); before = self.pdf_path.read_bytes()
        result = self.executor.execute(tid)
        self.assertTrue(result.answer_markdown)
        self.assertEqual(self.db.get_task(tid)['status'], 'completed')
        children = self.db.list_subtasks(tid)
        self.assertEqual(len(children), 1)
        child = children[0]; self.assertEqual(child['spec']['mode'], 'visual_reading')
        cp = self.db.get_checkpoint(child['subtask_id'])['visual_reading']
        self.assertFalse(cp['source']['semantically_verified'])
        self.assertEqual(cp['source']['evidence_id'], self.pages[0]['evidence_id'])
        self.assertEqual(len(self.visual_requests()), 1)
        self.assertEqual(len(self.requests), 5)
        budget = self.db.assert_budget_policy(tid, self.settings)
        self.assertEqual(budget['model_used'], 5)
        self.assertEqual(TokenBudgetStore(self.db).summary(tid)['used_tokens'], 75)
        self.assertEqual(self.db.get_task(child['subtask_id'])['save_policy'], 'deny')
        for request in self.requests:
            if request not in self.visual_requests():
                self.assertNotIn('data:image/', json.dumps(request))
                self.assertEqual(request['model'], 'deepseek-chat')
        self.assertEqual(self.visual_requests()[0]['model'], 'deepseek-flash')
        self.assertEqual(self.visual_requests()[0]['thinking'], {'type': 'disabled'})
        self.assertTrue(any(e['event_type'] == 'answer_stream' for e in self.db.list_events(tid)))
        self.assertIn('合成定向回答', self.api.get('/api/tasks/' + tid + '/stream').text)
        self.assertEqual(self.pdf_path.read_bytes(), before)
        self.assertEqual(list(self.files.output_dir.rglob('*.md')), [])

    def test_repeated_request_and_pagination_reuse_one_receipt(self):
        """长观察按原委派分页读回，重复原因不制造额外模型请求。"""
        tid, actor = self.prepare(); self.long_answer = True
        first = self.dispatch(tid, actor)
        second = self.dispatch(tid, actor, self.args(offset=first['next_offset'], reason='继续查看同一观察'))
        again = self.dispatch(tid, actor)
        self.assertEqual(first['answer_excerpt'] + second['answer_excerpt'], '甲' * 7001)
        self.assertTrue(second['reused']); self.assertTrue(again['reused'])
        self.assertEqual(len(self.visual_requests()), 1)
        self.assertEqual(len(self.db.list_subtasks(tid)), 1)
        self.assertNotIn('base64', json.dumps(first))
        self.assertEqual(self.db.budget_summary(tid)['model_used'], 1)

    def test_receipt_after_process_exit_is_not_resent(self):
        """受控收据后退出，恢复相同委派只回放成功响应。"""
        tid, actor = self.prepare()
        with patch.object(VisualReadingNode, '_checkpoint', side_effect=SystemExit('controlled receipt exit')):
            with self.assertRaises(SystemExit):
                self.dispatch(tid, actor)
        result = self.dispatch(tid, actor)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(self.visual_requests()), 1)
        self.assertEqual(len(self.db.list_subtasks(tid)), 1)

    def test_parent_action_after_child_success_replays_without_visual_send(self):
        """父行动落盘后退出，恢复重放原观察，不再委派或发送图片。"""
        tid = self.create(); original = self.db.save_action
        def interrupt(action):
            """在真实视觉结果和父行动都持久化后模拟进程退出。"""
            original(action)
            if action['tool_name'] == 'delegate_visual_reading' and action['status'] == 'succeeded':
                raise SystemExit('controlled parent action exit')
        with patch.object(self.db, 'save_action', side_effect=interrupt):
            with self.assertRaises(SystemExit): self.executor.execute(tid)
        self.db.update_task(tid, status='queued')
        result = self.executor.execute(tid, 'restored-worker')
        self.assertTrue(result.answer_markdown)
        self.assertEqual(self.db.get_task(tid)['status'], 'completed')
        self.assertEqual(len(self.visual_requests()), 1)
        self.assertEqual(len(self.db.list_subtasks(tid)), 1)
        self.assertEqual(self.db.budget_summary(tid)['model_used'], 5)

    def test_visual_result_is_retained_if_parent_later_fails(self):
        """父角色后续响应未知时，已落库的视觉观察仍交付并明确未核验。"""
        self.failure = 'parent-timeout'
        tid = self.create(); result = self.executor.execute(tid)
        self.assertEqual(self.db.get_task(tid)['status'], 'partial')
        self.assertIn('合成视觉观察', result.answer_markdown)
        self.assertIn('未语义核验', result.answer_markdown)
        self.assertEqual(len(self.visual_requests()), 1)
        child = self.db.list_subtasks(tid)[0]
        self.assertTrue(child['result']['result']['findings'])

    def test_sent_window_exit_cannot_create_another_visual_request(self):
        """发送窗口中断后started/sent保留，重复委派也不能新开子任务重发。"""
        tid, actor = self.prepare(); self.failure = 'exit'
        with self.assertRaises(SystemExit): self.dispatch(tid, actor)
        child = self.db.list_subtasks(tid)[0]['subtask_id']
        self.assertEqual(self.db.get_model_call(child, 1)['sent_state'], 'sent')
        self.failure = None
        failed = self.dispatch(tid, actor)
        self.assertEqual(failed['status'], 'execution_failed')
        self.assertIn('blocks resend', failed['error'])
        self.assertEqual(len(self.visual_requests()), 1)

    def test_cancel_or_lose_child_after_response_does_not_publish(self):
        """响应已付费时先保留收据，取消/失权后不得发布新的观察。"""
        original = VisualReadingNode._checkpoint
        for failure in ('cancel-parent', 'lose-child'):
            tid, actor = self.prepare()
            def lose(node, task_id, worker, token, result):
                """只在收据成功保存后撤销执行权。"""
                if failure == 'cancel-parent':
                    TaskService(self.db, self.settings).cancel(tid)
                else:
                    self.db.update_task(task_id, lease_token='new-owner')
                return original(node, task_id, worker, token, result)
            with patch.object(VisualReadingNode, '_checkpoint', new=lose):
                with self.assertRaises(PermissionError): self.dispatch(tid, actor)
            child = self.db.list_subtasks(tid)[0]['subtask_id']
            self.assertEqual(self.db.get_model_call(child, 1)['status'], 'succeeded')
            self.assertIsNone(self.db.get_checkpoint(child))
            with self.assertRaises(PermissionError): self.dispatch(tid, actor)
        self.assertEqual(len(self.visual_requests()), 2)

    def test_missing_cache_and_readback_without_result_do_not_generate(self):
        """缺缓存不重解析，读回偏移不能用来隐式启动新模型请求。"""
        tid, actor = self.prepare()
        with self.assertRaisesRegex(ValueError, 'original retained result'):
            self.dispatch(tid, actor, self.args(offset=1))
        next(self.files.parse_dir.glob('*.json')).unlink()
        with self.assertRaisesRegex(ValueError, 'cache'):
            self.dispatch(tid, actor)
        self.assertEqual(self.db.list_subtasks(tid), [])
        self.assertEqual(self.visual_requests(), [])

    def test_spatial_relation_trigger_is_retained_in_child_scope(self):
        """图关系缺口使用同一单次节点，原因留档而非新增角色或审校循环。"""
        tid, actor = self.prepare()
        result = self.dispatch(tid, actor, self.args(trigger='missing_spatial_relation',
            reason='文字丢失箭头方向', question='图中箭头方向是什么？'))
        child = self.db.list_subtasks(tid)[0]
        self.assertEqual(child['spec']['trigger'], 'missing_spatial_relation')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(self.visual_requests()), 1)

    def test_unknown_and_rejected_results_do_not_retry(self):
        """失败仍保留具体错误；相同委派不通过新子任务重发。"""
        for failure in ('timeout', 'rejected'):
            self.failure = failure
            tid, actor = self.prepare()
            first = self.dispatch(tid, actor); second = self.dispatch(tid, actor)
            self.assertEqual(first['status'], 'execution_failed'); self.assertTrue(second['reused'])
            self.assertIn('unknown' if failure == 'timeout' else '400', first['error'])
            child = self.db.list_subtasks(tid)[0]['subtask_id']
            receipt = self.db.list_model_calls(child)[0]
            self.assertEqual(receipt['status'], 'unknown' if failure == 'timeout' else 'rejected')
            self.assertEqual(json.loads(receipt['usage_json']), {})
        self.assertEqual(len(self.visual_requests()), 2)

    def test_failure_returns_to_supervisor_and_delivers_partial(self):
        """发送前确定的能力缺口交回Supervisor后准确交付，不制造语义通过。"""
        self.failure = 'missing-cache'
        tid = self.create(); result = self.executor.execute(tid)
        self.assertEqual(self.db.get_task(tid)['status'], 'partial')
        self.assertIn('视觉读取不可用', result.answer_markdown)
        self.assertEqual(len(self.visual_requests()), 0)
        self.assertTrue(any(a['tool_name'] == 'finish_task' and a['status'] == 'succeeded' for a in self.db.list_actions(tid)))

    def test_http_rejection_retains_error_and_usage_unknown_blocks_more_calls(self):
        """真实形状HTTP错误无usage，沿原策略结束，不能为让Supervisor收尾而补零。"""
        self.failure = 'rejected'
        tid = self.create(); result = self.executor.execute(tid)
        self.assertEqual(self.db.get_task(tid)['status'], 'failed')
        self.assertIn('400', result.answer_markdown)
        self.assertIn('TOKEN_USAGE_UNKNOWN', result.reason)
        self.assertEqual(len(self.visual_requests()), 1)
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(TokenBudgetStore(self.db).summary(tid)['unknown_reservations'], 1)

    def test_unread_foreign_page_and_changed_version_reject(self):
        """未读页、外部页身份、改版和无缓存均在子节点发送前拒绝。"""
        tid, actor = self.prepare()
        for eid in (self.pages[1]['evidence_id'], 'evidence:foreign'):
            with self.assertRaises((PermissionError, ValueError)):
                self.dispatch(tid, actor, self.args(evidence_id=eid))
        self.pdf_path.write_bytes(b'%PDF changed')
        with self.assertRaises(ValueError):
            self.dispatch(tid, actor)
        self.assertEqual(self.visual_requests(), [])
        self.assertEqual(self.db.list_subtasks(tid), [])

    def test_page_scope_and_text_only_source_reject(self):
        """即使已有页授权，用户指定范围和无物理页边界仍有效。"""
        tid, actor = self.prepare()
        self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', ('只看物理第2页，不保存。', tid))
        with self.assertRaises(PermissionError):
            self.dispatch(tid, actor)
        self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', ('不保存。', tid))
        self.db.save_document(Document(document_id='text-only-' + tid, paper_id='jina',
            content_sha256='a' * 64, parser_name='jina', parser_version='1', config_sha256='b' * 64,
            total_pages=None, cache_path='').model_dump(mode='json'))
        page = PageEvidence(evidence_id='text-only-' + tid, document_id='text-only-' + tid,
                            page_number=None, text='无物理页的正文')
        self.db.save_page(page.model_dump(mode='json')); self.db.mark_task_evidence(tid, page.evidence_id)
        with self.assertRaisesRegex(ValueError, 'physical PDF page'):
            self.dispatch(tid, actor, self.args(evidence_id=page.evidence_id))
        self.assertEqual(self.visual_requests(), [])

    def test_cancel_lost_lease_and_non_official_endpoint_reject(self):
        """节点委派不绕过取消/租约，不将密钥发送到另一个供应商。"""
        tid, actor = self.prepare(); TaskService(self.db, self.settings).cancel(tid)
        with self.assertRaises(PermissionError): self.dispatch(tid, actor)
        tid, actor = self.prepare(); self.db.update_task(tid, lease_token='other')
        with self.assertRaises(PermissionError): self.dispatch(tid, actor)
        tid, actor = self.prepare()
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            self.executor._visual_once(tid, self.args(), *actor,
                self.settings.model_copy(update={'deepseek_base_url': 'https://fixture.invalid'}), self.papers)
        self.assertEqual(self.visual_requests(), [])

    def test_acquired_paper_without_attachment_uses_cached_page(self):
        """检索取得的论文和附件均支持，不因paper_id在子任务里回退联网。"""
        self.reference = 'paper-' + uuid4().hex
        tid = self.create(attachments=False)
        candidate = PaperCandidate(paper_id=self.reference, title='Synthetic', source='fixture', pdf_url=str(self.pdf_path))
        self.db.save_paper(self.reference, candidate.model_dump(mode='json'))
        self.db.grant_paper(tid, self.reference, tid)
        result = self.executor.execute(tid)
        self.assertEqual(self.db.get_task(tid)['status'], 'completed')
        self.assertTrue(result.answer_markdown)
        self.assertEqual(len(self.visual_requests()), 1)


if __name__ == '__main__':
    unittest.main()
