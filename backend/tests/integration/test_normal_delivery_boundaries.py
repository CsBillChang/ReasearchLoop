"""正常论文入口与断流交付回归；仅外部HTTP、生成模型及PDF使用合成替身。"""
import copy
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from research_loop.api.dependencies import get_runtime_settings
from research_loop.config import Settings
from research_loop.main import app
from research_loop.providers.discovery import DiscoveryProvider
from research_loop.services.conversations import ConversationService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_collaboration_contract import FixturePDF
from integration.test_context_continuity import memory_payload
from integration.test_research_progress_handoff import ProgressScenario


PAPER_ID = '2601.12345v1'
TITLE = 'Synthetic Detector with Language Features'
LINK = 'https://arxiv.org/abs/' + PAPER_ID
FINDING = TITLE + '：摘要描述语言特征。链接：' + LINK + '。尚未核对正文实验。'


class NormalScenario:
    """合成决策沿实际工具结果取身份，不注入实验材料编号或候选池。"""

    def __init__(self, mode):
        self.mode = mode
        self.root_step = self.child_step = 0
        self.requests = []
        self.handoff_request = None
        self.selected = None

    def send(self, messages, tools, **kwargs):
        self.requests.append(copy.deepcopy({'messages': messages, 'tools': tools}))
        names = {item['function']['name'] for item in tools}
        respond = ProgressScenario.response
        observations = [json.loads(item['content']) for item in messages if item['role'] == 'tool']
        if 'finish_research' in names:
            step = self.child_step
            self.child_step += 1
            if self.mode.startswith('child_'):
                if step == 0:
                    return respond('read_paper', {'target_ref': 'input-A'})
                if self.mode == 'child_progress_disconnect' and step == 1:
                    pages = next(item['data']['pages'] for item in observations if item.get('data', {}).get('pages'))
                    return respond('record_research_progress', {'findings': [
                        {'text': FINDING, 'evidence_ids': [pages[0]['evidence_id']]}], 'gaps': ['实验条件尚未核验']})
                raise httpx.RemoteProtocolError('synthetic child disconnected after reading')
            if step == 0:
                return respond('search_papers', {'query': 'language detector', 'sources': ['arxiv'], 'limit': 3})
            candidates = next(item['data']['candidates'] for item in observations
                              if item.get('data', {}).get('candidates'))
            self.selected = candidates[0]['paper_id']
            if step != 1:
                raise AssertionError('unexpected additional researcher request')
            return respond('finish_research', {'findings': [FINDING], 'paper_ids': [self.selected], 'evidence_ids': []})
        step = self.root_step
        self.root_step += 1
        if step == 0:
            return respond('set_research_plan', {'goal': '推荐论文', 'task_kind': 'search',
                'requested_count': 1, 'count_confirmed': True,
                'goals': [{'goal_id': 'recommend', 'description': '推荐论文', 'min_papers': 1}]})
        if self.mode == 'read_then_disconnect':
            if step == 1:
                return respond('read_paper', {'target_ref': 'input-A'})
            raise httpx.RemoteProtocolError('synthetic connection closed after reading')
        if step == 1:
            if self.mode.startswith('child_'):
                return respond('delegate_research', {'subtasks': [{'objective': '解释附件方法',
                    'mode': 'investigate', 'goal_ids': ['recommend'], 'input_refs': ['input-A']}]})
            return respond('delegate_research', {'subtasks': [{'objective': '推荐语言特征检测论文',
                'mode': 'screen', 'goal_ids': ['recommend'], 'sources': ['arxiv']}]})
        if step == 2:
            self.handoff_request = copy.deepcopy({'messages': messages, 'tools': tools})
            child = next(child for item in observations for child in item.get('data', {}).get('subtasks', []))
            if child['paper_ids'] != [self.selected]:
                raise AssertionError('selected identity missing from native handoff')
            if self.mode == 'handoff_then_disconnect':
                raise httpx.RemoteProtocolError('synthetic connection closed after handoff')
            if self.mode == 'read_selected':
                return respond('read_paper', {'target_ref': child['paper_ids'][0], 'page_count': 1})
        if step > 3:
            raise AssertionError('unexpected additional supervisor request')
        children = memory_payload(messages)['current_task']['subtasks']
        return respond('finish_task', {'answer_markdown': FINDING,
            'goals': [{'goal_id': 'recommend', 'status': 'satisfied',
                'subtask_ids': [child['subtask_id'] for child in children], 'reason': '已有摘要推荐及真实链接'}]})

    def close(self):
        """替身没有外部连接。"""


class NormalDeliveryBoundaryTests(unittest.TestCase):
    """复现普通RPC检索和传输中断，检查实际HTTP任务视图与持久收据。"""

    def run_case(self, mode='screen', recovery=False, catalog=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(project_root=root, data_dir=root/'data', output_dir=root/'output',
                database_url='sqlite:///' + str(root/'case.db'),
                discovery_backends={'arxiv': {'url': 'https://source.invalid', 'anon_key': 'synthetic',
                    'bm25_rpc': 'match_arxiv_papers_bm25', 'enabled': True, 'use_bm25_rpc': True}})
            db = Database(settings.database_url)
            db.ensure_schema()
            scenario = NormalScenario(mode)
            rpc_requests = []
            def rpc(request):
                rpc_requests.append(json.loads(request.content))
                return httpx.Response(200, json=[{'id': PAPER_ID, 'title': TITLE, 'link': LINK,
                    'published': '2026-01-02', 'abstract': 'Synthetic language detector.', 'authors': ['Synthetic Author']}])
            transport = httpx.Client(transport=httpx.MockTransport(rpc))
            discovery = DiscoveryProvider(backends=settings.discovery_backends, client=transport)
            cid = ConversationService(db, settings).create('normal route', uuid4().hex)['conversation_id']
            attachments = []
            if mode == 'read_then_disconnect' or mode.startswith('child_'):
                pdf = root/'synthetic.pdf'
                pdf.write_bytes(b'%PDF-synthetic')
                attachments = [{'input_ref': 'input-A', 'paper_id': PAPER_ID, 'local_path': str(pdf)}]
            app.dependency_overrides[get_runtime_settings] = lambda: settings
            try:
                # 只替换HTTP客户端，Discovery.search及候选标准化保持生产实现。
                with ExitStack() as scope, patch.object(DiscoveryProvider, 'from_settings', return_value=discovery), \
                     patch('research_loop.main.get_settings', return_value=settings), \
                     patch('research_loop.config.get_settings', return_value=settings), \
                     patch('research_loop.main.heartbeat'), TestClient(app) as client:
                    if catalog:
                        from prompt_bundle_cases import install_catalog
                        pdf = root/'catalog.pdf'
                        pdf.write_bytes(b'%PDF-catalog-fixture')
                        digest = sha256(pdf.read_bytes()).hexdigest()
                        case = {'suite_kind': 'agent_final_v1', 'materials': [{'pdf_path': str(pdf),
                            'pdf_sha256': digest, 'physical_pages': 1, 'material_id': 'P-synthetic'}],
                            'preconditions': {'catalog': [{'material_index': 0, 'title': TITLE,
                                'abstract': 'Synthetic language detector.', 'source_url': LINK}]}}
                        install_catalog(scope, DiscoveryProvider, case, {digest: {
                            'document_ref': {'paper_id': PAPER_ID, 'metadata': {'source': 'arxiv'}}}})
                    response = client.post('/api/tasks', json={'conversation_id': cid,
                        'prompt': '推荐一篇语言特征检测论文，不保存', 'save_policy': 'deny',
                        'attachments': attachments, 'idempotency_key': uuid4().hex})
                    self.assertEqual(response.status_code, 202, response.text)
                    task_id = response.json()['task_id']
                    executor = TaskExecutor(settings, db, lambda _: scenario, lambda *_: FixturePDF())
                    executor.execute(task_id)
                    before = len(scenario.requests)
                    if recovery:
                        db.update_task(task_id, status='queued', result_json=None)
                        executor.execute(task_id, 'recovery-worker')
                        self.assertEqual(len(scenario.requests), before)
                    view = client.get('/api/tasks/' + task_id).json()
                    calls = db.list_model_calls(task_id)
                    children = db.list_subtasks(task_id)
                    self.assertEqual(list(settings.output_dir.glob('**/*.md')), [])
                    return {'view': view, 'calls': calls, 'children': children,
                        'rpc_requests': rpc_requests, 'requests': scenario.requests,
                        'handoff': scenario.handoff_request, 'parent_papers': db.list_task_papers(task_id),
                        'parent_evidence': db.list_task_evidence(task_id)}
            finally:
                app.dependency_overrides.clear()
                transport.close()

    def test_normal_search_handoff_delivers_without_experiment_labels(self):
        result = self.run_case()
        self.assertEqual(result['view']['status'], 'completed', result['view'])
        self.assertIn(LINK, result['view']['result']['answer_markdown'])
        self.assertIn(PAPER_ID, result['parent_papers'])
        self.assertEqual(len(result['rpc_requests']), 1)
        self.assertEqual(result['parent_evidence'], [])
        payload = json.dumps(result['handoff'], ensure_ascii=False)
        self.assertIn(PAPER_ID, payload)
        self.assertIn(TITLE, payload)
        self.assertNotIn('controlled_local_catalog', payload)
        self.assertNotIn('material_id', payload)

    def test_parent_reads_exact_selected_paper_after_normal_handoff(self):
        result = self.run_case('read_selected')
        self.assertEqual(result['view']['status'], 'completed', result['view'])
        self.assertTrue(result['parent_evidence'])
        self.assertEqual(result['parent_papers'], [PAPER_ID])

    def test_experiment_notice_reaches_first_root_and_child_requests(self):
        result = self.run_case(catalog=True)
        self.assertEqual(result['view']['status'], 'completed', result['view'])
        self.assertFalse(result['rpc_requests'])
        first_by_role = {}
        for request in result['requests']:
            names = {item['function']['name'] for item in request['tools']}
            role = 'researcher' if 'finish_research' in names else 'supervisor'
            first_by_role.setdefault(role, request)
        self.assertEqual(set(first_by_role), {'researcher', 'supervisor'})
        descriptions = []
        for request in first_by_role.values():
            text = next(item['function']['description'] for item in request['tools']
                        if item['function']['name'] == 'search_papers')
            self.assertIn('共1条', text)
            self.assertIn('metadata.material_id', text)
            descriptions.append(text)
        self.assertEqual(descriptions[0], descriptions[1])

    def test_completed_handoff_recovery_does_not_resend(self):
        result = self.run_case(recovery=True)
        self.assertEqual(result['view']['status'], 'completed')
        self.assertEqual(len(result['rpc_requests']), 1)

    def test_disconnect_preserves_existing_child_findings_as_partial(self):
        result = self.run_case('handoff_then_disconnect')
        self.assertEqual(result['view']['status'], 'partial', result['view'])
        self.assertIn(FINDING, result['view']['result']['answer_markdown'])
        self.assertIn('synthetic connection closed', result['view']['result']['answer_markdown'])
        unknown = [call for call in result['calls'] if call['status'] == 'unknown']
        self.assertEqual(len(unknown), 1)
        self.assertEqual(json.loads(unknown[0].get('usage_json') or '{}'), {})

    def test_disconnect_recovery_does_not_resend_unknown_request(self):
        result = self.run_case('handoff_then_disconnect', recovery=True)
        self.assertEqual(result['view']['status'], 'partial')
        self.assertEqual(sum(call['status'] == 'unknown' for call in result['calls']), 1)

    def test_read_pages_without_findings_remains_failed_not_usable(self):
        result = self.run_case('read_then_disconnect')
        self.assertEqual(result['view']['status'], 'failed', result['view'])
        self.assertEqual(result['view']['result']['answer_kind'], 'failure_report')
        self.assertIn('尚未形成可交付的研究结论', result['view']['result']['answer_markdown'])
        self.assertTrue(result['parent_evidence'])
        self.assertFalse(result['rpc_requests'])

    def test_child_read_without_findings_does_not_invent_parent_answer(self):
        result = self.run_case('child_read_disconnect')
        self.assertEqual(result['view']['status'], 'failed', result['view'])
        self.assertEqual(result['view']['result']['answer_kind'], 'failure_report')
        self.assertIn('尚未形成可交付的研究结论', result['view']['result']['answer_markdown'])
        self.assertFalse(result['rpc_requests'])

    def test_child_disconnect_retains_committed_progress(self):
        result = self.run_case('child_progress_disconnect')
        self.assertEqual(result['view']['status'], 'partial', result['view'])
        self.assertIn(FINDING, result['view']['result']['answer_markdown'])
        # 正式任务视图将缺口独立返回，前端在“未解决的问题与阅读边界”展示。
        self.assertIn('实验条件尚未核验', result['view']['result']['gaps'])


if __name__ == '__main__':
    unittest.main()
