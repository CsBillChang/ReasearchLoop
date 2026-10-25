"""档案读回的工作集回退回归；默认SQLite，显式隔离MySQL，供应商为替身。"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from research_loop.agents.graph import initial_state
from research_loop.agents.memory import ContextService
from research_loop.config import Settings
from research_loop.domain.memory import HistoryRead
from research_loop.domain.tasks import TaskCreate
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.services.tasks import TaskService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.memory import dump
from research_loop.storage.mysql import Database, _now
from research_loop.tools.memory import specs as memory_specs
from research_loop.workers.executor import TaskExecutor


class ArchiveProvider:
    """通过生产图返回三个固定行动，真实读取/装配/收据逻辑不替换。"""

    def __init__(self, archive_id):
        """绑定本任务合成档案，保存每次模型可见请求用于断言。"""
        self.archive_id = archive_id
        self.calls = []

    def close(self):
        """替身无网络连接。"""

    def send(self, messages, tools, **kwargs):
        """生成计划、批量读回、交付；不是语义质量或真实模型验证。"""
        validate_messages(messages)
        self.calls.append(deepcopy(messages))
        step = len(self.calls)
        if step == 1:
            calls = [('set_research_plan', {'goal': '读回历史材料', 'task_kind': 'memory',
                      'goals': [{'goal_id': 'history', 'description': '保留历史材料', 'min_papers': 0}]}),
                     ('get_profile', {})]
        elif step == 2:
            calls = [('read_memory', {'source_type': 'context_archive', 'source_id': self.archive_id,
                      'offset': offset, 'limit': 16000}) for offset in (100, 18000)]
        elif step == 3:
            calls = [('finish_task', {'answer_markdown': '已读回历史材料；它不是原文证据，本任务不保存文件。',
                      'goals': [{'goal_id': 'history', 'status': 'satisfied', 'reason': '按实际档案读回'}]})]
        else:
            raise AssertionError('unexpected extra simulated request')
        return {'id': 'archive-fixture-' + str(step), 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': '',
                  'reasoning_content': 'r' * 8000,
                  'tool_calls': [{'id': str(step) + '-' + str(i), 'type': 'function',
                     'function': {'name': name, 'arguments': dump(args)}} for i, (name, args) in enumerate(calls)]}}]}


class MemoryPageFallbackTests(unittest.TestCase):
    """校验实际ContextService和API→DB→Worker，不修改生产配置或保存权限。"""

    def setUp(self):
        """每例独立目录和会话；外部库必须明确标为本轮测试库。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        url = os.getenv('MEMORY_FALLBACK_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root / 'test.sqlite')
        if url.startswith('mysql') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_memory_fallback_'):
            raise ValueError('explicit isolated memory fallback database required')
        self.settings = Settings(project_root=self.root, data_dir=self.root / 'data', output_dir=self.root / 'notes',
                                 database_url=url, retrieval_enabled=False)
        self.db = Database(url)
        self.db.ensure_schema()
        self.conversation = ConversationService(self.db, self.settings)
        self.cid = self.conversation.create('合成档案读回', uuid4().hex)['conversation_id']
        self.tasks = TaskService(self.db, self.settings)
        self.memory = MemoryService(self.db)

    def task(self, client=None, cid=None, settings=None):
        """通过同一创建合同固定当前禁止保存要求。"""
        payload = TaskCreate(prompt='读回历史材料，保留身份与条件，不保存文件。', conversation_id=cid or self.cid,
                             idempotency_key=uuid4().hex, save_policy='deny')
        if client:
            response = client.post('/api/tasks', json=payload.model_dump(mode='json'))
            self.assertEqual(response.status_code, 202, response.text)
            return response.json()['task_id']
        return (TaskService(self.db,settings) if settings else self.tasks).create(payload).task_id

    def archive(self, tid):
        """合成含Unicode、反斜线与换行的原始档案；不伪造真实论文或模型轨迹。"""
        raw = dump({'fixture': ('历史α\\引号"\n😀条件。' * 5000)})
        sid = hashlib.sha256((tid + raw).encode()).hexdigest()
        self.db.execute('INSERT INTO context_archives(snapshot_id,task_id,sequence_no,original_json,projected_json,metrics_json,created_at) VALUES(?,?,?,?,?,?,?)',
                        (sid, tid, 0, raw, '[]', '{}', _now()))
        return sid, raw

    def messages(self, tid, sid, count=2, length=16000):
        """旧组大推理迫使回退，最新批量档案页必须保留截短副本。"""
        args = [HistoryRead(source_type='context_archive', source_id=sid, offset=100 + i * 18000, limit=length)
                for i in range(count)]
        latest = {'role': 'assistant', 'content': '', 'reasoning_content': '当前推理原样保留',
                  'tool_calls': [{'id': 'page-' + str(i), 'type': 'function',
                    'function': {'name': 'read_memory', 'arguments': dump(a.model_dump())}} for i, a in enumerate(args)]}
        return [{'role': 'system', 'content': '系统规则'}, {'role': 'user', 'content': self.db.get_task(tid)['prompt']},
                {'role': 'assistant', 'content': '', 'reasoning_content': 'prior' * 4000,
                 'tool_calls': [{'id': 'old', 'type': 'function', 'function': {'name': 'get_profile', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'old', 'content': dump({'ok': True, 'data': {'profile': '合成历史'}})},
                latest, *[{'role': 'tool', 'tool_call_id': 'page-' + str(i),
                           'content': dump({'ok': True, 'data': self.memory.read(tid, a)})} for i, a in enumerate(args)]]

    def check_pages(self, tid, projected, messages):
        """逐字验证来源版本和字符游标；补读能重建截短前观察。"""
        original = {m['tool_call_id']: json.loads(m['content'])['data'] for m in messages
                    if m['role'] == 'tool' and m['tool_call_id'].startswith('page-')}
        found = []
        for m in projected:
            if m['role'] != 'tool' or m.get('tool_call_id') not in original:
                continue
            page = json.loads(m['content'])['data']
            before = original[m['tool_call_id']]
            self.assertTrue(page['context_excerpted'])
            self.assertLess(len(page['content']), len(before['content']))
            self.assertFalse(page['is_original_paper_evidence'])
            for key in ('source_type', 'source_id', 'content_hash', 'offset', 'total_chars'):
                self.assertEqual(page[key], before[key])
            self.assertEqual(page['next_offset'], page['offset'] + len(page['content']))
            tail = self.memory.read(tid, HistoryRead(source_type=page['source_type'], source_id=page['source_id'],
                                    offset=page['next_offset'], limit=16000))
            self.assertEqual(page['content'] + tail['content'][:len(before['content'])-len(page['content'])], before['content'])
            found.append(page)
        return found

    def run_projection(self, count):
        """实际调用prepare触发回退，输入与完整档案保持不变。"""
        settings = self.settings.model_copy(update={'context_window_tokens':13000,'context_output_reserve':1000,'context_safety_tokens':500})
        tid = self.task(settings=settings)
        sid, raw = self.archive(tid)
        messages = self.messages(tid, sid, count)
        before = deepcopy(messages)
        settings = self.settings.model_copy(update={'context_window_tokens': 13000, 'context_output_reserve': 1000,
                                                    'context_safety_tokens': 500})
        ctx = ContextService(self.db, self.memory, settings)
        tools = [s.schema() for s in memory_specs() if s.name in {'read_memory', 'get_profile'}]
        # SQLite每例新库，MySQL共享库；固定无关历史召回，避免前例成果改变本例容量输入。
        # 完整API/Worker场景不替换history，仍检查实际历史装配。
        history = ctx.history(tid)
        history['related_history'] = []
        with (patch.object(ctx, 'history', return_value=history),
              patch.object(ctx, '_working_set', wraps=ctx._working_set) as fallback):
            projected, metrics = ctx.prepare(initial_state(tid, self.db.get_task(tid)['prompt'], 'deny'), messages, tools)
        self.assertTrue(fallback.called)
        self.assertTrue(metrics['memory_pages_shortened'])
        self.assertLessEqual(metrics['after'], metrics['input_capacity'])
        self.assertEqual(len(self.check_pages(tid, projected, messages)), count)
        self.assertEqual(messages, before)
        self.assertEqual(projected[-count-1], messages[-count-1])
        validate_messages(projected)
        saved = self.db.fetchone('SELECT original_json FROM context_archives WHERE snapshot_id=?', (metrics['snapshot_id'],))
        self.assertEqual(json.loads(saved['original_json'])['messages'], before)
        self.assertEqual(self.db.fetchone('SELECT original_json FROM context_archives WHERE snapshot_id=?', (sid,))['original_json'], raw)
        self.assertEqual(self.db.get_task(tid)['save_policy'], 'deny')
        self.assertEqual(self.db.list_model_calls(tid), [])

    def test_one_memory_page_survives_working_set_fallback(self):
        """单页也不得在回退重建时恢复大原件。"""
        self.run_projection(1)

    def test_two_memory_pages_survive_working_set_fallback(self):
        """复现真实失败的同批两页，保留每页身份及续读游标。"""
        self.run_projection(2)

    def test_three_unicode_pages_survive_working_set_fallback(self):
        """多页及多字节/转义字符不依赖两页或固定偏移的特判。"""
        self.run_projection(3)

    def run_mixed_projection(self, count=24, capacity_failure=False):
        """同批档案、两组命中和读页共同装配，逐字位置与档案续读均可验证。"""
        from research_loop.services.papers import PaperService
        from research_loop.tools.papers import specs as paper_specs
        from integration.test_context_continuity import LongPDF
        settings = self.settings.model_copy(update={'context_window_tokens':35000,'context_output_reserve':4096,'context_safety_tokens':1024})
        tid = self.task(settings=settings)
        sid, _ = self.archive(tid)
        path = self.root / 'mixed.pdf'
        path.write_bytes(b'%PDF-synthetic')
        paper_id = 'mixed-' + uuid4().hex
        attachment = {'input_ref': 'input-mixed', 'paper_id': paper_id, 'local_path': str(path)}
        # 混合批次只持有当前任务附件，不向其他历史或foreign任务授予权限。
        self.db.grant_input(tid, 'input-mixed', tid)
        pages = PaperService(self.db, None, LongPDF()).read(tid, 'input-mixed', attachments={'input-mixed': attachment})
        source_hash = self.db.fetchone('SELECT content_sha256 FROM documents WHERE document_id=?',
                                      (pages['document_id'],))['content_sha256']
        messages = self.messages(tid, sid, 1)
        latest = messages[-2]
        for n in range(2):
            page = pages['pages'][n]
            hits = [{'document_id': page['document_id'], 'paper_id': paper_id, 'source_sha256': source_hash,
                     'evidence_id': page['evidence_id'], 'page_number': page['page_number'],
                     'chunk_id': f'{n}-{i}', 'char_start': i * 100, 'char_end': (i + 1) * 100,
                     'text': page['text'][i * 100:(i + 1) * 100], 'section_origin': 'page_fallback'} for i in range(count)]
            latest['tool_calls'].append({'id': f'hits-{n}', 'type': 'function', 'function': {
                'name': 'retrieve_evidence', 'arguments': dump({'query': 'language conditions', 'document_ids': [page['document_id']]})}})
            messages.append({'role': 'tool', 'tool_call_id': f'hits-{n}', 'content': dump({'ok': True, 'data': {'hits': hits}})})
        latest['tool_calls'].append({'id': 'mixed-pages', 'type': 'function', 'function': {
            'name': 'read_paper', 'arguments': dump({'target_ref': 'input-mixed'})}})
        messages.append({'role': 'tool', 'tool_call_id': 'mixed-pages', 'content': dump({'ok': True, 'data': pages})})
        before = deepcopy(messages)
        # 只在本合成例设置小窗口，同时覆盖可容纳与明确超容量两侧；不改变运行配置。
        settings = self.settings.model_copy(update={'context_window_tokens': 35000, 'context_output_reserve': 4096,
                                                    'context_safety_tokens': 1024})
        ctx = ContextService(self.db, self.memory, settings)
        tools = [s.schema() for s in memory_specs() + paper_specs() if s.name in {'read_memory', 'read_paper', 'retrieve_evidence'}]
        history = ctx.history(tid)
        history['related_history'] = []
        state = initial_state(tid, self.db.get_task(tid)['prompt'], 'deny', {'input-mixed': attachment})
        with patch.object(ctx, 'history', return_value=history):
            if capacity_failure:
                with self.assertRaisesRegex(ValueError, 'CONTEXT_CAPACITY'):
                    ctx.prepare(state, messages, tools)
                self.assertEqual(messages, before)
                self.assertEqual(self.db.list_model_calls(tid), [])
                self.assertEqual(self.db.get_task(tid)['save_policy'], 'deny')
                return
            projected, metrics = ctx.prepare(state, messages, tools)
        self.assertLessEqual(metrics['after'], metrics['input_capacity'])
        self.assertEqual(len(self.check_pages(tid, projected, messages)), 1)
        self.assertIn(latest, projected)  # 工具批次、参数及reasoning原样保留。
        factored = 0
        for message in projected:
            if message.get('role') != 'tool' or not message.get('tool_call_id', '').startswith('hits-'):
                continue
            data = json.loads(message['content'])['data']
            factored += bool(data.get('hits_defaults'))
            for hit in data.get('hits', []):
                hit = {**data.get('hits_defaults', {}), **hit}
                page = self.db.get_page(hit['evidence_id'])
                self.assertEqual(hit['document_id'], page['document_id'])
                self.assertEqual(hit['source_sha256'], source_hash)
                raw = page['text'][hit['char_start']:hit['char_end']]
                for span in hit.get('context_spans', []):
                    self.assertEqual(span['text'], raw[span['start']:span['end']])
        self.assertGreater(factored, 0)
        validate_messages(projected)
        self.assertEqual(messages, before)
        saved = self.db.fetchone('SELECT original_json FROM context_archives WHERE snapshot_id=?', (metrics['snapshot_id'],))
        self.assertEqual(json.loads(saved['original_json'])['messages'], before)
        self.assertEqual(self.db.get_task(tid)['save_policy'], 'deny')
        self.assertEqual(self.db.list_model_calls(tid), [])
        self.assertFalse(list(self.root.rglob('*.md')))

    def test_mixed_archive_retrieval_and_pages_keep_identity(self):
        """常规24条命中的混合批次保持档案、版本、原文区间与工具配对。"""
        self.run_mixed_projection()

    def test_oversized_protected_mixed_batch_still_reports_capacity(self):
        """小测试窗口无法容纳大量必要命中时明确报错，不删除身份伪装通过。"""
        # 新计量比字节计量更贴近容量；增加合成身份数以继续覆盖必要状态超限。
        self.run_mixed_projection(count=64, capacity_failure=True)

    def test_no_readback_tool_preserves_honest_capacity_failure(self):
        """没有档案工具时不裁原件、不制造无法执行的续读提示。"""
        settings = self.settings.model_copy(update={'context_window_tokens':13000,'context_output_reserve':1000,'context_safety_tokens':500})
        tid = self.task(settings=settings)
        sid, _ = self.archive(tid)
        messages = self.messages(tid, sid)
        original = deepcopy(messages)
        settings = self.settings.model_copy(update={'context_window_tokens': 13000, 'context_output_reserve': 1000,
                                                    'context_safety_tokens': 500})
        with self.assertRaisesRegex(ValueError, 'CONTEXT_CAPACITY'):
            ContextService(self.db, self.memory, settings).prepare(initial_state(tid, '读回历史，不保存', 'deny'), messages, [])
        self.assertEqual(messages, original)

    def test_archive_cannot_be_read_by_another_task(self):
        """装配修复不扩张档案权限或把历史当作原文。"""
        tid = self.task()
        sid, _ = self.archive(tid)
        other = self.conversation.create('另一会话', uuid4().hex)['conversation_id']
        other_tid = self.task(cid=other)
        with self.assertRaisesRegex(KeyError, 'permitted scope'):
            self.memory.read(other_tid, HistoryRead(source_type='context_archive', source_id=sid))

    def run_worker(self, recovery=False):
        """生产API/Worker走完整读回交付，供应商和故障点明确为替身。"""
        from research_loop.main import app
        from research_loop.api.dependencies import get_runtime_settings
        settings = self.settings.model_copy(update={'context_window_tokens': 40000, 'context_output_reserve': 4096,
                                                    'context_safety_tokens': 1024})
        original_overrides = app.dependency_overrides.copy()
        app.dependency_overrides[get_runtime_settings] = lambda: settings
        try:
            with (patch('research_loop.main.get_settings', return_value=settings),
                  TestClient(app) as client,
                  patch.object(DeepSeekProvider, 'send', side_effect=AssertionError('real supplier forbidden')),
                  patch.object(DeepSeekProvider, 'send_stream', side_effect=AssertionError('real supplier forbidden'))):
                tid = self.task(client)
                sid, raw = self.archive(tid)
                provider = ArchiveProvider(sid)
                executor = TaskExecutor(settings, self.db, lambda s: provider,
                    supervisor_tools={'set_research_plan', 'get_profile', 'read_memory', 'finish_task'})
                if recovery:
                    original_save = CheckpointStore.save
                    def crash(store, task_id, state):
                        """第三条成功收据后、结束工具执行前退出；恢复不得再次发送。"""
                        if len(provider.calls) == 3 and state.get('pending_tool_calls'):
                            raise SystemExit('fixture exit after successful receipt')
                        return original_save(store, task_id, state)
                    with patch.object(CheckpointStore, 'save', crash):
                        with self.assertRaises(SystemExit):
                            executor.execute(tid, 'fixture-before-exit')
                    before = deepcopy(self.db.list_model_calls(tid))
                    self.db.update_task(tid, status='queued', lease_until='2000-01-01T00:00:00+00:00')
                    executor.execute(tid, 'fixture-after-exit')
                    self.assertEqual(self.db.list_model_calls(tid), before)
                else:
                    executor.execute(tid, 'fixture-worker')
                view = client.get('/api/tasks/' + tid).json()
                self.assertEqual(view['status'], 'completed', view)
                self.assertEqual(len(provider.calls), 3)
                pages = [json.loads(m['content'])['data'] for m in provider.calls[-1] if m['role'] == 'tool'
                         and json.loads(m['content']).get('data', {}).get('source_type') == 'context_archive']
                self.assertEqual(len(pages), 2)
                self.assertTrue(all(p.get('context_excerpted') for p in pages))
                self.assertEqual(self.db.get_task(tid)['save_policy'], 'deny')
                self.assertEqual(self.db.fetchone('SELECT original_json FROM context_archives WHERE snapshot_id=?', (sid,))['original_json'], raw)
                self.assertEqual(self.db.fetchall('SELECT * FROM drafts WHERE task_id=?', (tid,)), [])
                self.assertEqual(list(self.root.rglob('*.md')), [])
                self.assertEqual(view['usage']['unknown'], 0)
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(original_overrides)

    def test_api_worker_delivers_after_batch_archive_readback(self):
        """普通Worker可从已截短的批量档案页继续交付，禁止保存保持。"""
        self.run_worker()

    def test_success_receipt_recovery_after_archive_readback(self):
        """成功响应后退出并恢复，沿用原request_id/usage且不多发替身请求。"""
        self.run_worker(recovery=True)
