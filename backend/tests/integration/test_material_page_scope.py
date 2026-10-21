"""真实持久服务和合成PDF/索引替身；不代表真实模型或原论文验证。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from research_loop.config import Settings
from research_loop.agents.graph import initial_state
from research_loop.agents.memory import ContextService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.domain.evidence import Document, PageEvidence, PaperCandidate
from research_loop.domain.notes import CreateNoteInput
from research_loop.domain.retrieval import RetrievalQuery
from research_loop.domain.tasks import Feedback, TaskCreate, ResumeInput
from research_loop.domain.source_access import exact_material_required
from research_loop.domain.visual_reading import VisualInput, VisualDelegationInput
from research_loop.domain.context import compact_tool_exchange
from research_loop.storage.memory import dump
from research_loop.storage.mysql import _now
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.providers.paper_acquisition import PaperAcquisitionError
from research_loop.services.conversations import ConversationService
from research_loop.services.feedback import submit_feedback
from research_loop.services.indexing import IndexingService
from research_loop.services.notes import NoteService
from research_loop.services.papers import PaperService
from research_loop.services.research_progress import ResearchProgressService
from research_loop.services.retrieval import RetrievalService
from research_loop.services.tasks import TaskService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from unit.test_retrieval_contracts import FakeEmbedding, FakeVector


class MaterialPDF:
    """六页合成载体，保持真实字节hash和可核对的原文索引快照。"""

    def __init__(self, root):
        """记录读取次数，越权请求不应触发解析。"""
        self.root, self.ingests = root, 0
        self.file_store = FileStore(root, root/'notes', root)

    def ingest(self, source, paper_id):
        """同内容产生同文档身份，专测共享hash不能合并权限。"""
        self.ingests += 1
        digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        document_id = 'doc:' + digest[:32]
        cache = self.root / 'cache' / 'parsed' / (digest + '.json')
        cache.parent.mkdir(parents=True, exist_ok=True)
        pdf = self.root / 'cache' / 'pdf' / (digest + '.pdf')
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(Path(source).read_bytes())
        document = Document(document_id=document_id, paper_id=paper_id, content_sha256=digest,
            parser_name='fixture', parser_version='1', config_sha256='a'*64, total_pages=6, cache_path=str(cache))
        cache.write_text(json.dumps({'pages': [p.model_dump() for p in self.pages(document, 1, 6)]}))
        return document

    def pages(self, document, start, count):
        """页身份只依赖文档和物理位置，正文不伪装真实论文。"""
        return [PageEvidence(evidence_id=f'{document.document_id}:{n}', document_id=document.document_id,
                page_number=n, text=f'Synthetic methods evidence page {n}. Results and conditions.')
                for n in range(start, min(start+count, 7))]


class Embedding(FakeEmbedding):
    """加入查询参数的离线编码替身。"""

    def encode(self, texts, query=False):
        """仅验证检索数据流，不评价语义效果。"""
        return super().encode(texts)


class Vectors(FakeVector):
    """只在授权候选ID中返回确定的点。"""

    def query(self, collection, vector, ids, limit):
        """测试真实SQL候选过滤是否先于排序。"""
        return [{'id': key, 'payload': self.points[collection][key]['payload'], 'score': 0.5}
                for key in ids[:limit]]


class Reranker:
    """明确的重排替身。"""
    identity = {'model_id': 'synthetic-only'}

    def score(self, query, texts):
        """返回等分以免排序掩盖授权问题。"""
        return [0.5] * len(texts)


class MaterialPageScopeIntegrationTests(unittest.TestCase):
    """同套服务在SQLite或显式隔离MySQL执行，无模型与外网调用。"""

    def setUp(self):
        """为每例建立独立材料/任务；不清理共享库或旧工件。"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        url = os.getenv('MATERIAL_PAGE_SCOPE_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root/'test.db')
        if not url.startswith('sqlite:') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_'):
            raise ValueError('isolated MySQL required')
        self.db = Database(url)
        self.db.ensure_schema()
        self.settings = Settings(project_root=self.root, data_dir=self.root/'data', output_dir=self.root/'notes', database_url=url)
        self.pdf = MaterialPDF(self.root)
        self.papers = PaperService(self.db, None, self.pdf)
        self.embedding, self.vectors = Embedding(), Vectors()
        self.retrieval = RetrievalService(self.db, self.embedding, self.vectors, Reranker())
        self.roots = []
        for name in ('send', 'send_stream'):
            guard = patch.object(DeepSeekProvider, name, side_effect=AssertionError('real model forbidden'))
            guard.start()
            self.addCleanup(guard.stop)
        network = patch('httpx.HTTPTransport.handle_request', side_effect=AssertionError('external HTTP forbidden'))
        network.start()
        self.addCleanup(network.stop)
        self.addCleanup(self.close_tasks)

    def close_tasks(self):
        """取消本例未执行根，禁止测试任务被真实Worker接走。"""
        for task in self.roots:
            self.db.update_task(task, status='cancelled', cancel_requested=1)

    def create(self, prompt, *, shared=False, refs=('A', 'B'), filenames=None, conversation=False):
        """按实际任务入口持久保存服务端材料身份。"""
        self.attachments = {}
        for index, ref in enumerate(refs):
            path = self.root/(ref + '.pdf')
            path.write_bytes(b'%PDF synthetic shared' if shared else ('%PDF synthetic '+ref).encode())
            self.attachments[ref] = {'input_ref': ref, 'local_path': str(path), 'filename': (filenames or {}).get(ref)}
        cid = ConversationService(self.db, self.settings).create('范围测试', uuid4().hex)['conversation_id'] if conversation else None
        view = TaskService(self.db, self.settings).create(TaskCreate(prompt=prompt, attachments=list(self.attachments.values()),
            conversation_id=cid, idempotency_key=uuid4().hex))
        self.roots.append(view.task_id)
        return view.task_id

    def clarify(self, tid, answer):
        """经正式澄清resume路径更新同根真实用户要求。"""
        service = TaskService(self.db, self.settings)
        item = service.request_clarification(tid, '请确认物理页范围', ['pages'], [])
        self.db.update_task(tid, status='needs_input')
        state = initial_state(tid, self.db.get_task(tid)['prompt'], 'deny')
        state.update(finished=True, final_reason='clarification requested')
        CheckpointStore(self.db).save(tid, state)
        service.resume(ResumeInput(clarification_id=item['clarification_id'], answer=answer, idempotency_key=uuid4().hex), tid)

    def read(self, tid, ref='A', **kwargs):
        """复用服务端附件映射，不手工授予页证据。"""
        return self.papers.read(tid, ref, attachments=self.attachments, **kwargs)

    def child(self, tid, refs, role='researcher'):
        """只给子角色指定材料，页建议不能影响用户权限。"""
        child = self.db.create_subtask(tid, tid, len(self.db.list_subtasks(tid)), '仅为模型生成的建议：只读第1页',
            {'input_refs': refs, 'start_page': 1, 'page_count': 1}, uuid4().hex)['subtask_id']
        self.db.update_task(child, role=role)
        for ref in refs:
            self.db.grant_input(child, ref, tid)
        self.db.execute('UPDATE tasks SET attachments_json=? WHERE task_id=?',
            (json.dumps([self.attachments[r] for r in refs]), child))
        return child

    def test_pending_b_allows_a_read_index_and_retrieval(self):
        tid = self.create('附件A只读前两页；附件B只允许前面几页。')
        data = self.read(tid)
        self.assertEqual(data['returned_pages'], [1, 2])
        before = self.pdf.ingests
        with self.assertRaises(ValueError):
            self.read(tid, 'B')
        self.assertEqual(self.pdf.ingests, before)
        result = self.retrieval.search(tid, RetrievalQuery(query='methods', document_ids=[data['document_id']]))
        self.assertTrue(result['hits'])
        self.assertEqual({h['page_number'] for h in result['hits']}, {1, 2})

    def test_real_r2_resume_updates_existing_roles(self):
        tid = self.create('比较本对话附件A（D09 LoRASculpt）和附件B（D10 LaVer）的核心方法，只使用用户许可的物理页。仅使用附件A的物理第1至2页；附件B只允许前面几页。B的具体页范围尚未确定，请先向我澄清B的物理页范围，在收到回答前不要读取B或猜测其范围。保留A的明确硬范围。不要重新检索，不做八点精读，不保存文件。')
        researcher, reviewer = self.child(tid, ['A']), self.child(tid, ['B'], 'reviewer')
        self.assertEqual(self.read(researcher)['returned_pages'], [1, 2])
        self.clarify(tid, 'B的范围补充为：仅使用附件B的物理第1至3页。同时调整A原来的硬范围：仅使用附件A的物理第2至3页。两份材料按各自范围比较；不要把两篇统一成同一页范围，不读取A第1页或B第4页及以后。不重新检索，不做八点精读，不保存文件。')
        self.assertEqual(self.read(researcher)['returned_pages'], [2, 3])
        self.assertEqual(self.read(reviewer, 'B')['returned_pages'], [1, 2, 3])
        for task, ref, page in ((tid, 'A', 1), (researcher, 'A', 1), (reviewer, 'B', 4)):
            with self.subTest(role=task, ref=ref), self.assertRaises(PermissionError):
                self.read(task, ref, start_page=page, page_count=1)

    def test_local_release_and_global_update(self):
        tid = self.create('附件A只读前两页；附件B只读前3页。')
        self.clarify(tid, '解除附件A的页数限制。')
        self.assertEqual(self.read(tid)['returned_pages'], list(range(1, 7)))
        self.assertEqual(self.read(tid, 'B')['returned_pages'], [1, 2, 3])
        self.clarify(tid, '所有材料只读物理第2至3页。')
        self.assertEqual(self.read(tid)['returned_pages'], [2, 3])
        self.assertEqual(self.read(tid, 'B')['returned_pages'], [2, 3])

    def test_duplicate_names_rejected_before_ingest(self):
        tid = self.create('仅使用附件报告的物理前两页。', filenames={'A': '报告.pdf', 'B': '报告.pdf'})
        with self.assertRaises(ValueError):
            self.read(tid)
        self.assertEqual(self.pdf.ingests, 0)

    def test_explicit_user_label_binds_unique_filename_not_order(self):
        tid = self.create('附件A（D09 LoRASculpt）和附件B（D10 LaVer）用于比较。仅使用附件A的物理第2至3页；附件B只读前面几页。',
            refs=('opaque-second', 'opaque-first'), filenames={'opaque-first': 'D09.pdf', 'opaque-second': 'D10.pdf'})
        self.assertEqual(self.read(tid, 'opaque-first')['returned_pages'], [2, 3])
        with self.assertRaises(ValueError):
            self.read(tid, 'opaque-second')

    def test_changed_version_rejected_without_replacing_binding(self):
        tid = self.create('附件A只读前两页。')
        first = self.read(tid)
        old = self.db.get_task(tid)['brief_json']
        Path(self.attachments['A']['local_path']).write_bytes(b'%PDF changed version')
        with self.assertRaisesRegex(ValueError, 'version changed'):
            self.read(tid)
        self.assertEqual(self.db.get_task(tid)['brief_json'], old)
        self.assertEqual(len(self.db.list_task_evidence(tid)), len(first['pages']))

    def test_shared_hash_keeps_role_material_scope(self):
        tid = self.create('附件A只读第1页；附件B只读第1至4页。', shared=True)
        a = self.read(tid)
        b = self.read(tid, 'B')
        self.assertEqual(a['document_id'], b['document_id'])
        child = self.child(tid, ['A'])
        for page in b['pages']:
            self.db.grant_evidence(child, page['evidence_id'], tid)
        with self.assertRaises(PermissionError):
            self.papers.authorized_source(child, b['pages'][3]['evidence_id'])
        source = self.papers.authorized_source(child, a['pages'][0]['evidence_id'])
        self.assertEqual(source['readback']['target_ref'], 'A')
        with self.assertRaises(PermissionError):
            self.papers.validate_source_observation(tid, b['pages'][3], {'target_ref': 'A'})
        # 无新映射的历史夹具也不能借B的题录身份扩大只派A的子角色。
        self.db.update_task(tid, brief_json='{}')
        with self.assertRaises(PermissionError):
            self.papers.authorized_source(child, b['pages'][3]['evidence_id'])

    def test_authorization_precedes_scope_parse(self):
        tid = self.create('只读前面几页。')
        child = self.child(tid, ['A'])
        with self.assertRaises(PermissionError):
            self.papers.user_scope(child, target_ref='B')
        with self.assertRaises(PaperAcquisitionError):
            self.read(child, 'B')
        self.assertFalse(self.db.task_has_input(child, 'B'))
        with self.assertRaises(PermissionError):
            self.papers.authorized_source(child, 'unknown-evidence')
        self.assertEqual(self.pdf.ingests, 0)

    def test_cached_read_rechecks_material_after_local_change(self):
        tid = self.create('附件A只读前两页；附件B只读前3页。')
        data = self.read(tid)
        args = {'target_ref': 'A'}
        self.db.save_action({'action_id': uuid4().hex, 'task_id': tid, 'sequence': 1, 'tool_name': 'read_paper', 'arguments': args, 'status': 'succeeded', 'observation': {'ok': True, 'data': data}})
        service = ResearchProgressService(self.db)
        self.assertIsNotNone(service.cached_read(tid, args, self.attachments))
        self.clarify(tid, '附件A只读物理第2至3页。')
        self.assertIsNone(service.cached_read(tid, args, self.attachments))
        valid_args = {'target_ref': 'A', 'start_page': 2, 'page_count': 1}
        legal = self.read(tid, start_page=2, page_count=1)
        self.db.save_action({'action_id': uuid4().hex, 'task_id': tid, 'sequence': 2, 'tool_name': 'read_paper', 'arguments': valid_args, 'status': 'succeeded', 'observation': {'ok': True, 'data': legal}})
        self.assertIsNotNone(service.cached_read(tid, valid_args, self.attachments))

    def test_old_wide_index_filters_current_scope_and_reuses_valid_view(self):
        tid = self.create('附件A只读前4页；附件B只允许前面几页。')
        data = self.read(tid)
        broad = self.retrieval.indexing.build(tid, data['document_id'], [p['evidence_id'] for p in data['pages']])
        self.clarify(tid, '附件A只读物理第2至3页。')
        with self.assertRaises(PermissionError):
            self.retrieval.indexing.authorized_chunks(tid, broad['index_id'])
        result = self.retrieval.search(tid, RetrievalQuery(query='methods', document_ids=[data['document_id']]))
        self.assertEqual({h['page_number'] for h in result['hits']}, {2, 3})
        count = self.embedding.encoded
        second = self.retrieval.search(tid, RetrievalQuery(query='methods', document_ids=[data['document_id']]))
        self.assertEqual(result['scope'], second['scope'])
        self.assertEqual(self.embedding.encoded, count+1)  # 第二次仅编码查询，合法索引复用。
        self.assertEqual({p['included_physical_pages'][0] for p in self.papers.evidence_scope(tid)}, {2})

    def test_scope_tightening_filters_context_without_losing_legal_history(self):
        tid = self.create('附件A只读前两页；附件B只读前面几页。')
        data = self.read(tid)
        messages = [{'role': 'user', 'content': '基于授权材料回答'},
            {'role': 'assistant', 'content': '', 'reasoning_content': '合成测试字段',
             'tool_calls': [{'id': 'actual-fixture-read', 'type': 'function', 'function': {'name': 'read_paper', 'arguments': '{"target_ref":"A"}'}}]},
            {'role': 'tool', 'tool_call_id': 'actual-fixture-read', 'content': json.dumps({'ok': True, 'data': data})}]
        original = json.dumps(messages)
        self.clarify(tid, '附件A只读物理第2至3页；附件B只读物理第1至3页。')
        context = ContextService(self.db, None, self.settings)
        state = {'task_id': tid, 'prompt': '继续解释', 'role': 'researcher'}
        projected, _ = context.prepare_researcher(state, messages, [])
        tool = json.loads(next(m['content'] for m in projected if m.get('tool_call_id') == 'actual-fixture-read'))
        self.assertEqual([p['page_number'] for p in tool['data']['pages']], [2])
        self.assertEqual(tool['data']['excluded_historical_sources'][0]['page_number'], 1)
        self.assertEqual(json.dumps(messages), original)
        # 真正撤销页grant与范围更新不同，必须继续失败，不默默过滤。
        self.db.execute('DELETE FROM task_evidence WHERE task_id=? AND evidence_id=?', (tid, data['pages'][1]['evidence_id']))
        with self.assertRaisesRegex(ValueError, 'CONTEXT_SOURCE_STALE'):
            context.prepare_researcher(state, messages, [])

    def test_acquire_resolves_only_authorized_candidate_scope(self):
        tid = self.create('候选甲只读物理第2至3页；附件B只允许前面几页。')
        candidate = PaperCandidate(paper_id='candidate-' + uuid4().hex, title='候选甲', source='fixture',
            pdf_url=self.attachments['A']['local_path'])
        self.db.save_paper(candidate.paper_id, candidate.model_dump())
        self.db.grant_paper(tid, candidate.paper_id, tid)
        self.papers.web = object()  # 原PDF成功，不应走备用Fetch或Search。
        result = self.papers.acquire_pdf(tid, candidate.paper_id, allowed_refs=[candidate.paper_id])
        self.assertEqual(result['returned_pages'], [2, 3])
        with self.assertRaises(PermissionError):
            self.papers.acquire_pdf(tid, candidate.paper_id, start_page=1, page_count=1, allowed_refs=[candidate.paper_id])

    def test_media_and_visual_keep_current_material_ranges(self):
        tid = self.create('附件A只读物理第2至3页；附件B只允许前面几页。')
        with patch('research_loop.providers.paper_media.PaperMediaProvider.extract', return_value={'assets': []}) as extract:
            self.papers.inspect_media(tid, 'A', 2, 2, self.attachments)
            self.assertEqual(extract.call_args.args[-1], [2, 3])
            with self.assertRaises(ValueError):
                self.papers.inspect_media(tid, 'B', 1, 1, self.attachments)
        data = self.read(tid, start_page=2, page_count=1)
        visual = self.child(tid, ['A'])
        self.db.update_task(visual, status='running')
        self.db.grant_evidence(visual, data['pages'][0]['evidence_id'], tid)
        self.clarify(tid, '附件A只读物理第3页。')
        request = VisualInput(target_ref='A', document_id=data['document_id'], pdf_sha256=data['content_sha256'],
            page_text_sha256=hashlib.sha256(data['pages'][0]['text'].encode()).hexdigest(), physical_page=2,
            question='合成页验证', group='visual')
        with self.assertRaises(PermissionError):
            self.papers.prepare_visual(visual, request)
        self.db.update_task(tid, status='running')
        args = VisualDelegationInput(evidence_id=data['pages'][0]['evidence_id'], question='合成页验证', reason='测试范围', trigger='damaged_formula')
        with self.assertRaises(PermissionError):
            self.papers.visual_input_for_evidence(tid, args)

    def test_archive_restore_filters_only_scope_and_keeps_original_offsets(self):
        tid = self.create('附件A只读前两页；附件B只读前面几页。')
        data = self.read(tid)
        group = [{'role': 'assistant', 'content': '', 'reasoning_content': '合成推理字段原样保留',
            'tool_calls': [{'id': 'fixture-read', 'type': 'function', 'function': {'name': 'read_paper', 'arguments': '{"target_ref":"A"}'}}]},
            {'role': 'tool', 'tool_call_id': 'fixture-read', 'content': dump({'ok': True, 'data': data})}]
        original = dump({'messages': group})
        source_id = hashlib.sha256((tid + original).encode()).hexdigest()
        self.db.execute('INSERT INTO context_archives(snapshot_id,task_id,sequence_no,original_json,projected_json,metrics_json,created_at) VALUES(?,?,?,?,?,?,?)',
            (source_id, tid, 1, original, '[]', '{}', _now()))
        summary = {'source_type': 'context_archive', 'source_id': source_id, 'original_message_range': [0, 1],
            'tools': group[0]['tool_calls'], 'observations': [{'historical': True}], 'archived_tool_exchange': True}
        previews = compact_tool_exchange(group, summary)
        self.clarify(tid, '附件A只读物理第2至3页。')
        restored, count = ContextService(self.db, None, self.settings)._restore_sources(tid, previews, {'read_memory'})
        self.assertEqual(count, 1)
        self.assertEqual(restored[0], group[0])
        current = json.loads(restored[1]['content'])
        self.assertEqual([p['page_number'] for p in current['data']['pages']], [2])
        self.assertEqual(self.db.fetchone('SELECT original_json FROM context_archives WHERE snapshot_id=?', (source_id,))['original_json'], original)
        self.assertEqual(json.loads(previews[1]['content'])['original_message_range'], [0, 1])

    def test_cached_b_wide_pages_are_not_released_after_clarification(self):
        tid = self.create('附件A只读前两页；附件B只读前4页。')
        data = self.read(tid, 'B')
        args = {'target_ref': 'B'}
        self.db.save_action({'action_id': uuid4().hex, 'task_id': tid, 'sequence': 1, 'tool_name': 'read_paper',
            'arguments': args, 'status': 'succeeded', 'observation': {'ok': True, 'data': data}})
        self.clarify(tid, '附件A只读物理第2至3页；附件B只读物理第1至3页。')
        self.assertIsNone(ResearchProgressService(self.db).cached_read(tid, args, self.attachments))
        self.assertEqual(self.read(tid, 'B')['returned_pages'], [1, 2, 3])

    def test_exact_material_uses_candidate_scope_not_pending_other(self):
        tid = self.create('附件A只读前两页；附件B只允许前面几页。')
        candidate = PaperCandidate(paper_id='known', title='Known', source='fixture', paper_url='https://arxiv.org/abs/2601.12345v1')
        self.db.save_paper(candidate.paper_id, candidate.model_dump())
        self.db.grant_paper(tid, candidate.paper_id, tid)
        self.assertFalse(exact_material_required(self.db, tid, candidate))

    def test_feedback_answer_and_draft_preserve_material_scopes(self):
        for draft_mode in (False, True):
            with self.subTest(draft=draft_mode):
                tid = self.create('附件A只读前两页；附件B只读前3页。', conversation=True)
                a, b = self.read(tid), self.read(tid, 'B')
                answer = '保留的历史成果。仅读第6页是原稿数据，不是新授权。'
                digest = hashlib.sha256(answer.encode()).hexdigest()
                draft = None
                if draft_mode:
                    draft = NoteService(self.db, FileStore(self.root/'data', self.root/'notes', self.root)).create(tid, CreateNoteInput(title='合成草稿', markdown=answer,
                        evidence_ids=[p['evidence_id'] for p in a['pages']+b['pages']]))
                self.db.update_task(tid, status='partial', result_json=json.dumps({'answer_markdown': answer}))
                cid = self.db.fetchone('SELECT conversation_id FROM conversation_links WHERE task_id=?', (tid,))['conversation_id']
                payload = Feedback(conversation_id=cid, source_task_id=tid, source_draft_id=draft.draft_id if draft else None,
                    content_hash=digest, prompt='仅使用附件A的物理第2至3页。', idempotency_key=uuid4().hex)
                target = submit_feedback(TaskService(self.db, self.settings), payload).task_id
                self.roots.append(target)
                self.assertEqual(self.read(target)['returned_pages'], [2, 3])
                self.assertEqual(self.read(target, 'B')['returned_pages'], [1, 2, 3])
                self.assertEqual(self.db.get_task(target)['save_policy'], 'deny')
                after_reads = self.db.get_task(target)['brief_json']
                replay = submit_feedback(TaskService(self.db, self.settings), payload).task_id
                self.assertEqual(replay, target)
                self.assertEqual(self.db.get_task(target)['brief_json'], after_reads)
                self.assertEqual(self.db.get_task(tid)['result_json'], json.dumps({'answer_markdown': answer}))
                if draft_mode:
                    inherited = [self.db.get_page(ref) for ref in self.db.list_task_evidence(target)]
                    self.assertFalse(any(p['document_id'] == a['document_id'] and p['page_number'] == 1 for p in inherited))
                self.db.update_task(target, status='partial', result_json='{}')
