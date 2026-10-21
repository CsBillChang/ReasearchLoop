"""API→Worker执行覆盖与精确复用；供应商/PDF/检索计算采用合成替身。"""
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from research_loop.api.dependencies import get_runtime_settings
from research_loop.config import Settings
from research_loop.main import app
from research_loop.providers.deepseek import validate_messages
from research_loop.services.papers import PaperService
from research_loop.services.coverage import CoverageService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_collaboration_contract import FixturePDF


def reply(name, args):
    """固定供应商JSON形状，所有语义结论明确来自测试脚本。"""
    return {'id': uuid4().hex, 'choices': [{'finish_reason': 'tool_calls', 'message': {
        'role': 'assistant', 'content': '', 'tool_calls': [{'id': uuid4().hex, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args)}}]}}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}


class ScriptedClosure:
    """模型边界替身，只选择既有工具，验证执行接线而非模型科研能力。"""

    def __init__(self, pid, path):
        """每个测试固定一篇合成材料和一种合法取证入口。"""
        self.pid, self.path, self.calls, self.root_step = pid, path, [], 0

    def send(self, messages, tools, **kwargs):
        """依照已返回的事实派发固定下一步，不产生外部请求。"""
        validate_messages(messages)
        self.calls.append({'messages': messages, 'tools': tools})
        names = {t['function']['name'] for t in tools}
        observations = [json.loads(m['content']) for m in messages if m.get('role') == 'tool']
        if 'finish_research' in names:
            data = next((o['data'] for o in observations if o.get('data', {}).get('pages')), None)
            if data is None:
                return reply('read_paper', {'target_ref': self.pid, 'start_page': 1, 'page_count': 1})
            return reply('finish_research', {'findings': ['合成材料方法发现。'], 'paper_ids': [self.pid],
                'evidence_ids': [data['pages'][0]['evidence_id']]})
        step = self.root_step; self.root_step += 1
        if step == 0:
            return reply('set_research_plan', {'goal': '解释合成方法', 'task_kind': 'search',
                'requested_count': 1, 'count_confirmed': True,
                'goals': [{'goal_id': 'method', 'description': '解释合成方法', 'min_papers': 1, 'requires_fulltext': True}]})
        if self.path == 'delegate':
            if step in (1, 2, 3):
                return reply('delegate_research', {'subtasks': [{'objective': '解释方法' if step < 3 else '解释不同问题',
                    'mode': 'investigate', 'goal_ids': ['method'], 'paper_ids': [self.pid], 'start_page': 1, 'page_count': 1}]})
            children = [c for o in observations for c in o.get('data', {}).get('subtasks', [])]
            return reply('finish_task', {'answer_markdown': '基于明确选择的子成果交付。',
                'goals': [{'goal_id': 'method', 'status': 'satisfied', 'reason': '使用已取得成果',
                    'subtask_ids': list(dict.fromkeys(c['subtask_id'] for c in children))}]})
        if step == 1:
            if self.path == 'retrieve_evidence':
                return reply(self.path, {'query': 'language mechanism', 'document_ids': ['doc-'+self.pid]})
            return reply(self.path, {'target_ref': self.pid, 'start_page': 1, 'page_count': 1})
        return reply('finish_task', {'answer_markdown': '依据实际返回的授权原文解释机制。',
            'goals': [{'goal_id': 'method', 'status': 'satisfied', 'reason': '已实际返回原文窗口'}]})

    def close(self):
        """替身没有网络连接。"""


class ClosurePipelineTests(unittest.TestCase):
    """默认临时SQLite，显式MySQL只能使用researchloop_test_专用库。"""

    def run_pipeline(self, path):
        """建立API任务，经真实Worker执行并由API读回；禁止外网和真实模型。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = os.getenv('FIVE_ROOT_TEST_DATABASE_URL') or 'sqlite:///' + str(root/'closure.db')
            if not url.startswith('sqlite:') and not url.rsplit('/', 1)[-1].split('?', 1)[0].startswith('researchloop_test_'):
                raise ValueError('only isolated researchloop_test_ database is allowed')
            settings = Settings(project_root=root, data_dir=root/'data', output_dir=root/'notes',
                database_url=url, web_search_enabled=True, web_fetch_enabled=True,
                web_search_api_key='synthetic', web_fetch_api_key='synthetic', retrieval_enabled=False)
            db = Database(url); db.ensure_schema()
            pid = 'closure-' + uuid4().hex
            scenario = ScriptedClosure(pid, path)
            worker = TaskExecutor(settings, db, lambda _: scenario, lambda *_: FixturePDF())
            app.dependency_overrides[get_runtime_settings] = lambda: settings
            # SQLite无需网络；专用MySQL允许数据库连接，其余工具仍全为替身。
            blocker = patch.object(socket.socket, 'connect', side_effect=AssertionError('real network forbidden')) if url.startswith('sqlite:') else patch('socket.getfqdn', return_value='isolated-test')
            try:
                with blocker, TestClient(app) as client:
                    response = client.post('/api/tasks', json={'prompt': '解释合成材料方法，不保存',
                        'save_policy': 'deny', 'idempotency_key': uuid4().hex})
                    self.assertEqual(response.status_code, 202, response.text)
                    tid = response.json()['task_id']
                    db.save_paper(pid, {'paper_id': pid, 'title': 'Synthetic source', 'source': 'arxiv',
                        'paper_url': 'https://arxiv.org/abs/2601.12345v1', 'pdf_url': 'https://arxiv.org/pdf/2601.12345v1'})
                    db.grant_paper(tid, pid, tid)
                    if path == 'retrieve_evidence':
                        # 预先取得授权页但不写read行动；检索本次只返回固定连续片段。
                        data = PaperService(db, None, FixturePDF()).read(tid, pid, 1, 1)
                        page = data['pages'][0]; text = page['text'][:12]
                        hit = {**page, 'paper_id': pid, 'source_sha256': data['content_sha256'],
                            'chunk_id': 'synthetic-chunk', 'char_start': 0, 'char_end': len(text), 'text': text,
                            'text_sha256': hashlib.sha256(text.encode()).hexdigest()}
                        worker.retrieval.search = lambda *_args, **_kwargs: {'status': 'ready', 'hits': [hit],
                            'evidence_ids': [page['evidence_id']], 'coverage': 'retrieved_chunks_only'}
                    result = worker.execute(tid)
                    view = client.get('/api/tasks/'+tid).json()
                    self.assertEqual(view['status'], 'completed', view)
                    self.assertEqual(view['result']['coverage']['paper_ids'], [pid])
                    self.assertIsNone(result.saved_path)
                    self.assertFalse(list((root/'notes').rglob('*.md')))
                    actions = db.list_actions(tid)
                    if path != 'delegate':
                        self.assertEqual([a['tool_name'] for a in actions], ['set_research_plan', path, 'finish_task'])
                        self.assertEqual(db.list_subtasks(tid), [])
                        coverage = view['result']['coverage']['goals'][0]['material_coverage']
                        self.assertEqual(len(coverage), 1)
                        self.assertFalse(coverage[0]['full_document_returned'])
                        if path == 'retrieve_evidence':
                            self.assertEqual(coverage[0]['material_scope'], 'retrieved_chunk')
                        self.assertEqual(actions[1]['observation']['data']['research_increment']['kind'], 'new_evidence')
                    else:
                        children = db.list_subtasks(tid)
                        self.assertEqual(len(children), 2)
                        delegated = [a['observation']['data']['subtasks'] for a in actions if a['tool_name'] == 'delegate_research']
                        self.assertEqual(delegated[0][0]['subtask_id'], delegated[1][0]['subtask_id'])
                        self.assertNotEqual(delegated[0][0]['subtask_id'], delegated[2][0]['subtask_id'])
                        context = CoverageService(db).current_context(tid, {})
                        candidates = context['goal_achievements'][0]['reuse_candidates']
                        self.assertEqual(len(candidates), 2)
                        self.assertTrue(all(item['finding_refs'] for item in candidates))
                        self.assertTrue(all(item['material_summary']['source_count'] for item in candidates))
                        self.assertEqual(sum(a['tool_name'] == 'read_paper' for child in children
                            for a in db.list_actions(child['subtask_id'])), 2)
                    return len(scenario.calls)
            finally:
                app.dependency_overrides.pop(get_runtime_settings, None)

    def test_acquire_returned_pages_finish_without_duplicate_child(self):
        """acquire成功原文经生产覆盖门，不能强迫再调用read或Researcher。"""
        self.assertEqual(self.run_pipeline('acquire_paper_pdf'), 3)

    def test_retrieved_chunk_finishes_targeted_question_without_wholepaper_claim(self):
        """已有授权页的检索片段计入材料身份，仍保留片段真实范围。"""
        self.assertEqual(self.run_pipeline('retrieve_evidence'), 3)

    def test_exact_delegation_reuses_child_and_rewritten_question_stays_explicit(self):
        """相同spec幂等，改写问题不被代码猜测等价；已有成果目录供Supervisor选择。"""
        self.assertEqual(self.run_pipeline('delegate'), 9)
