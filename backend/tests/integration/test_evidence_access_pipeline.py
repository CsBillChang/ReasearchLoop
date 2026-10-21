"""来源别名、授权读回和档案恢复隔离链；向量/解析用确定性替身。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlparse
from uuid import uuid4

from research_loop.agents.memory import ContextService
from research_loop.config import Settings
from research_loop.domain.context import compact_tool_exchange, project_research_progress, project_research_handoffs
from research_loop.domain.evidence import Document, PageEvidence
from research_loop.domain.memory import HistoryRead
from research_loop.domain.research_progress import ResearchProgressInput
from research_loop.domain.retrieval import RetrievalQuery
from research_loop.domain.tasks import ActionRecord, TaskCreate
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.services.indexing import IndexingService
from research_loop.services.papers import PaperService
from research_loop.services.research_progress import ResearchProgressService
from research_loop.services.retrieval import RetrievalService
from research_loop.storage.mysql import Database, _now
from research_loop.tools.base import ToolContext
from research_loop.tools.memory import specs as memory_specs
from research_loop.tools.registry import ToolRegistry
from tests.unit.test_hybrid_retrieval import QueryEmbedding, QueryVector, TestReranker


class EvidenceAccessPipelineTests(unittest.TestCase):
    """默认临时SQLite；显式隔离MySQL同组复验不删除已有表或数据。"""

    def setUp(self):
        """只构造合成文件和唯一任务；除显式隔离数据库外阻断全部网络。"""
        temporary = tempfile.TemporaryDirectory(prefix='researchloop-evidence-access-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        url = os.getenv('FIVE_ROOT_TEST_DATABASE_URL', '')
        if url and (urlparse(url).scheme != 'mysql' or not urlparse(url).path.startswith('/researchloop_test_')):
            raise ValueError('explicit isolated researchloop_test_ MySQL database required')
        self.db = Database(url or 'sqlite:///' + str(self.directory / 'test.db'))
        self.db.ensure_schema()
        key = uuid4().hex
        self.owner, self.page_only, self.foreign = ['evidence-' + name + '-' + key for name in ('owner', 'page-only', 'foreign')]
        for task in (self.owner, self.page_only, self.foreign):
            self.db.create_task(task, None, hashlib.sha256(task.encode()).hexdigest(), '比较指定论文，不保存', 'deny', [])
        self.old_id, self.new_id = 'user-url:' + key, '2603.00595v1:' + key
        # 使用不可解析别名，双方元数据必须各自证明同一明确v1。
        for paper_id in (self.old_id, self.new_id):
            self.db.save_paper(paper_id, {'paper_id': paper_id, 'title': 'Synthetic metric source', 'source': 'arxiv',
                'paper_url': 'https://arxiv.org/abs/2603.00595v1', 'pdf_url': 'https://arxiv.org/pdf/2603.00595v1', 'version': 'v1'})
        self.db.grant_paper(self.owner, self.old_id, self.owner)
        content = ('synthetic-pdf-' + key).encode()
        source_hash = hashlib.sha256(content).hexdigest()
        cache = self.directory / 'cache' / 'parsed' / (key + '.json')
        cache.parent.mkdir(parents=True)
        pdf = cache.parent.parent / 'pdf' / (source_hash + '.pdf')
        pdf.parent.mkdir()
        pdf.write_bytes(content)
        self.document = {'document_id': 'doc:' + key, 'paper_id': self.old_id, 'content_sha256': source_hash,
            'parser_name': 'pymupdf', 'parser_version': '1', 'config_sha256': 'a' * 64,
            'total_pages': 2, 'cache_path': str(cache)}
        self.pages = [{'evidence_id': 'page:' + key + ':' + str(number), 'document_id': self.document['document_id'],
                       'page_number': number, 'text': text} for number, text in enumerate((
                           'Language conditioning reports bounded performance under stated conditions.',
                           'A second page remains outside the page-only task authorization.'), 1)]
        cache.write_text(json.dumps({'total_pages': 2, 'pages': self.pages}))
        self.db.save_document(self.document)
        for page in self.pages:
            self.db.save_page(page)
            self.db.mark_task_evidence(self.owner, page['evidence_id'])
        self.eid = self.pages[0]['evidence_id']
        self.db.mark_task_evidence(self.page_only, self.eid)
        self.papers = PaperService(self.db, None, None)
        self.progress = ResearchProgressService(self.db)
        self.context = ContextService(self.db, None, Settings(project_root=self.directory, data_dir=self.directory/'data', output_dir=self.directory/'output'))
        # MySQL仅放行显式连接的已冻结数据库端点；论文/模型全部拒绝。
        allowed_host, allowed_port = (urlparse(url).hostname, urlparse(url).port or 3306) if url else (None, None)
        original = socket.socket.connect
        def isolated_connect(sock, address):
            """只放行显式测试数据库，不为测试开放外网。"""
            if url and address[0] in {allowed_host, '127.0.0.1' if allowed_host == 'localhost' else allowed_host} and address[1] == allowed_port:
                return original(sock, address)
            raise AssertionError('NETWORK_FORBIDDEN')
        blocker = patch.object(socket.socket, 'connect', isolated_connect)
        blocker.start()
        self.addCleanup(blocker.stop)

    def commit(self, payload):
        """走实际intent和事务提交，成功行动后续只能幂等读回。"""
        value = self.progress.prepare(self.owner, payload)
        sequence = len(self.db.list_actions(self.owner)) + 1
        action = ActionRecord(task_id=self.owner, sequence=sequence, tool_name='record_research_progress',
            arguments=payload.model_dump(), status='started', observation={'ok': True, 'data': {'research_progress': value}}).model_dump(mode='json')
        self.db.save_action(action)
        self.progress.commit_action(action, None)
        return action

    def rebind(self):
        """同内容缓存只改显示别名，不授予第二个paper_id。"""
        self.db.save_document({**self.document, 'paper_id': self.new_id})

    def test_alias_readback_uses_old_authorized_target_and_actual_read_succeeds(self):
        """C1：旧授权入口可读；当前显示题录不能误当成新授权。"""
        stored = self.progress.source(self.owner, self.eid)
        self.rebind()
        bundle = self.papers.evidence_bundle(self.owner, [self.eid])[0]
        self.assertEqual(bundle['target_ref'], self.old_id)
        self.assertTrue(bundle['readback']['available'])
        self.assertTrue(self.progress.source_version_matches(self.owner, stored))
        document, pages = self.document, self.pages
        class LocalPDF:
            """零网络解析替身，真实服务仍执行授权和持久化。"""
            def ingest(self, source, paper_id):
                """返回冻结内容版本，模拟相同PDF从原授权入口读取。"""
                return Document(**{**document, 'paper_id': paper_id})
            def pages(self, value, start, count):
                """仅返回请求页，不夹带第二页。"""
                return [PageEvidence(**page) for page in pages if start <= page['page_number'] < start + (count or 2)]
        result = PaperService(self.db, None, LocalPDF()).read(self.owner, bundle['target_ref'], 1, 1)
        self.assertEqual([p['evidence_id'] for p in result['pages']], [self.eid])
        self.assertFalse(self.db.task_has_paper(self.owner, self.new_id))
        self.assertEqual(self.db.get_task(self.owner)['save_policy'], 'deny')

    def test_page_only_grant_has_no_invented_target_or_new_page(self):
        """C1/C9：继承一页不获得题录或相邻页，明确没有补读入口。"""
        self.rebind()
        bundle = self.papers.evidence_bundle(self.page_only, [self.eid])[0]
        self.assertIsNone(bundle['target_ref'])
        self.assertFalse(bundle['readback']['available'])
        with self.assertRaises(PermissionError):
            self.papers.authorized_source(self.page_only, self.pages[1]['evidence_id'])
        self.assertFalse(self.db.task_has_paper(self.page_only, self.new_id))
        self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM save_receipts r JOIN drafts d ON d.draft_id=r.draft_id WHERE d.task_id=?', (self.page_only,))['n'], 0)

    def test_index_retrieval_progress_share_alias_and_revocation_checks(self):
        """C1/C9：旧manifest不改写；检索命中通过同一来源合同，撤授权均拒绝。"""
        embedding, vectors = QueryEmbedding(), QueryVector()
        indexer = IndexingService(self.db, embedding, vectors)
        first = indexer.build(self.owner, self.document['document_id'], [self.eid])
        frozen = copy.deepcopy(first['manifest'])
        stored = self.progress.source(self.owner, self.eid)
        self.rebind()
        result = RetrievalService(self.db, embedding, vectors, TestReranker()).search(self.page_only,
            RetrievalQuery(query='Language', document_ids=[self.document['document_id']]))
        self.assertTrue(result['hits'])
        self.assertTrue(all(hit['target_ref'] is None and not hit['readback']['available'] for hit in result['hits']))
        self.assertEqual(indexer.store.get(first['index_id'])['manifest'], frozen)
        self.assertTrue(self.progress.source_version_matches(self.owner, stored))
        self.db.execute('DELETE FROM task_evidence WHERE task_id=? AND evidence_id=?', (self.page_only, self.eid))
        with self.assertRaises(PermissionError):
            indexer.authorized_chunks(self.page_only, first['index_id'])
        with self.assertRaises(PermissionError):
            self.papers.validate_source_observation(self.page_only, result['hits'][0], {}, kind='hit')

    def test_progress_plan_delta_and_existing_finding_analysis_keep_history(self):
        """C3/C5：计划改写与分析修订分开，旧成功行动与来源引用保持。"""
        first = self.commit(ResearchProgressInput(findings=[{'text': 'Bounded initial finding.', 'evidence_ids': [self.eid]}]))
        frozen = copy.deepcopy(self.db.list_actions(self.owner)[0])
        finding = first['observation']['data']['research_progress']['findings'][0]
        plan = self.commit(ResearchProgressInput(gaps=['A reworded plan gap.']))['observation']['data']['research_progress']
        self.assertEqual(plan['progress_change']['kind'], 'plan_only')
        analysis = self.commit(ResearchProgressInput(findings=[{'finding_id': finding['finding_id'],
            'text': 'Same evidence permits a narrower analysis.'}]))['observation']['data']['research_progress']
        self.assertEqual(analysis['progress_change']['kind'], 'finding_revision')
        self.assertEqual(analysis['findings'][0]['sources'], finding['sources'])
        self.assertEqual(analysis['findings'][0]['revision'], 2)
        retained = self.progress.retained(self.owner)
        projected = project_research_progress(retained)
        self.assertEqual(projected['findings'][0]['finding_id'], finding['finding_id'])
        self.assertEqual(projected['findings'][0]['source_integrity'], 'intact')
        self.assertIn('action_id', projected)
        self.assertEqual(self.db.list_actions(self.owner)[0], frozen)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            changed = copy.deepcopy(first)
            changed['observation']['data']['research_progress']['gaps'] = ['tampered']
            self.progress.commit_action(changed, None)

    def test_archive_hit_restore_rejects_missing_version_bounds_and_revocation(self):
        """C3/C9：压缩与档案恢复同时检查命中，不能从旧档案绕过授权。"""
        hit = {**self.pages[0], 'char_start': 2, 'char_end': 12,
               'text': self.pages[0]['text'][2:12], 'source_sha256': self.document['content_sha256']}
        def archived(item):
            """创建精确档案组，恢复从数据库读回原件。"""
            archive_id = 'archive-' + uuid4().hex
            call = {'id': 'hit-call', 'type': 'function', 'function': {'name': 'retrieve_evidence', 'arguments': '{}'}}
            group = [{'role': 'assistant', 'content': '', 'tool_calls': [call]},
                     {'role': 'tool', 'tool_call_id': 'hit-call', 'content': json.dumps({'ok': True, 'data': {'hits': [item]}})}]
            self.db.execute('INSERT INTO context_archives(snapshot_id,task_id,sequence_no,original_json,projected_json,metrics_json,created_at) VALUES(?,?,?,?,?,?,?)',
                (archive_id, self.owner, 1, json.dumps({'messages': group}), '[]', '{}', _now()))
            return compact_tool_exchange(group, {'source_id': archive_id, 'original_message_range': [0, 1],
                                                 'tools': [call], 'observations': ['source preview']})
        view = archived(hit)
        restored, count = self.context._restore_sources(self.owner, view, {'read_memory'})
        self.assertEqual(count, 1)
        self.assertEqual(json.loads(restored[1]['content'])['data']['hits'][0]['text'], hit['text'])
        for change in ({'source_sha256': None}, {'source_sha256': 'b' * 64}, {'char_start': -1}, {'char_end': 999}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'CONTEXT_SOURCE_STALE'):
                self.context._restore_sources(self.owner, archived({**hit, **change}), {'read_memory'})
        self.db.execute('DELETE FROM task_evidence WHERE task_id=? AND evidence_id=?', (self.owner, self.eid))
        with self.assertRaisesRegex(ValueError, 'CONTEXT_SOURCE_STALE'):
            self.context._restore_sources(self.owner, view, {'read_memory'})

    def test_exact_live_handoff_compacts_with_legacy_revision_but_not_changed_findings(self):
        """C3/F5：实际commit后当前正文只呈现一份；真正版本/来源变化不折叠。"""
        action = self.commit(ResearchProgressInput(findings=[{'text': 'One precise finding.', 'evidence_ids': [self.eid]}]))
        retained = self.progress.retained(self.owner)
        self.assertNotIn('readback', retained['findings'][0])
        call = {'id': 'progress-call', 'function': {'name': 'record_research_progress', 'arguments': '{}'}}
        def exchange(value):
            """将同一成功行动表示为真实消息往返。"""
            return [{'role': 'assistant', 'content': '', 'tool_calls': [call]}, {'role': 'tool',
                'tool_call_id': 'progress-call', 'content': json.dumps({'ok': True, 'data': {'research_progress': value}})}]
        raw = action['observation']['data']['research_progress']
        projected, current, metrics = project_research_handoffs(exchange(raw), {'research_progress': retained})
        self.assertEqual(metrics['handoff_projected_calls'], ['progress-call'])
        self.assertNotIn('One precise finding.', projected[1]['content'])
        self.assertEqual(current['research_progress']['findings'][0]['text'], 'One precise finding.')
        legacy = copy.deepcopy(raw)
        legacy['findings'][0].pop('revision')
        _, _, metrics = project_research_handoffs(exchange(legacy), {'research_progress': retained})
        self.assertEqual(metrics['handoff_projected_calls'], ['progress-call'])
        for change in ({'text': 'A different claim.'}, {'revision': 2},
                       {'sources': [{**raw['findings'][0]['sources'][0], 'source_sha256': 'b' * 64}]}):
            different = copy.deepcopy(raw)
            different['findings'][0].update(change)
            with self.subTest(change=change):
                _, _, metrics = project_research_handoffs(exchange(different), {'research_progress': retained})
                self.assertEqual(metrics['handoff_projected_calls'], [])

    def test_actual_history_finding_readback_matches_tool_schema_and_reads_original_record(self):
        """C3：实际会话历史投影给出可直接执行的read_memory参数，身份元数据单独保留。"""
        self.commit(ResearchProgressInput(findings=[{'text': 'A historical bounded finding.', 'evidence_ids': [self.eid]}]))
        stage = self.progress.retained(self.owner)
        conversation = ConversationService(self.db, self.context.settings)
        cid = conversation.create('Synthetic history readback', uuid4().hex)['conversation_id']
        record = {'task_id': self.owner, 'status': 'partial', 'paper_ids': [self.old_id], 'evidence_ids': [self.eid],
                  'papers': [], 'gaps': [], 'research_progress': [stage], 'answer': 'Synthetic historical draft.'}
        encoded = json.dumps(record)
        self.db.execute('INSERT INTO research_memories(task_id,conversation_id,record_json,search_text,created_at) VALUES(?,?,?,?,?)',
                        (self.owner, cid, encoded, 'historical bounded finding', _now()))
        current_task = conversation.submit(TaskCreate(prompt='解释上次候选发现，不保存', conversation_id=cid,
                                                       idempotency_key=uuid4().hex, save_policy='deny'))
        memory = MemoryService(self.db)
        context = ContextService(self.db, memory, self.context.settings).history(current_task)
        metadata = context['recent_research'][0]['stage_findings'][0]['finding_metadata'][0]
        self.assertEqual(metadata['finding_id'], stage['findings'][0]['finding_id'])
        self.assertEqual(metadata['revision'], stage['findings'][0]['revision'])
        self.assertEqual(metadata['source_task_id'], self.owner)
        self.assertEqual(metadata['source_integrity'], 'intact')
        self.assertFalse(metadata['current_source_authorized'])
        request = HistoryRead.model_validate(metadata['readback'])
        tool_context = ToolContext(task_id=current_task, role='supervisor', save_policy='deny',
                                   attachments={}, services={'memory': memory})
        result = ToolRegistry(memory_specs()).execute('read_memory', request.model_dump(), tool_context)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['content'], encoded)
        self.assertFalse(result.data['is_original_paper_evidence'])
        self.assertFalse(self.db.task_has_evidence(current_task, self.eid))
        self.assertEqual(self.db.get_task(current_task)['save_policy'], 'deny')

    def test_display_identity_requires_explicit_alias_proof_for_metadata_ownership(self):
        """C1：已授权内容可用，但未知/跨版显示身份不能替新题录证明全文覆盖。"""
        data = {'paper_id': self.old_id, 'content_sha256': self.document['content_sha256']}
        self.rebind()
        source = self.papers.validate_source_observation(self.owner, self.pages[0], data)
        self.assertEqual(source['display_identity']['status'], 'explicit_same_version_alias')
        self.assertEqual(source['display_identity']['paper_id'], self.new_id)
        self.db.save_paper(self.new_id, {'paper_id': self.new_id, 'title': 'Synthetic source', 'source': 'arxiv',
            'paper_url': 'https://arxiv.org/abs/2603.00595v2', 'version': 'v2'})
        source = self.papers.validate_source_observation(self.owner, self.pages[0], data)
        self.assertEqual(source['page']['text'], self.pages[0]['text'])
        self.assertEqual(source['display_identity']['status'], 'unverified_difference')
        self.assertIsNone(source['display_identity']['paper_id'])
        self.assertFalse(self.db.task_has_paper(self.owner, self.new_id))

    def test_successful_read_receipt_reuse_respects_provider_parse_contract(self):
        """C8/C9：原成功行动不变，解析合同过期时交还正常受控工具路径。"""
        target = 'input-' + uuid4().hex
        self.db.grant_input(self.owner, target, self.owner)
        arguments = {'target_ref': target, 'start_page': 1, 'page_count': 1}
        observation = {'ok': True, 'data': {'document_id': self.document['document_id'],
            'content_sha256': self.document['content_sha256'], 'pages': [self.pages[0]]}}
        action = ActionRecord(task_id=self.owner, sequence=1, tool_name='read_paper', arguments=arguments,
                              status='succeeded', observation=observation).model_dump(mode='json')
        self.db.save_action(action)
        path = Path(self.document['cache_path']).parent.parent / 'pdf' / (self.document['content_sha256'] + '.pdf')
        attachments = {target: {'local_path': str(path)}}
        self.assertIsNotNone(self.progress.cached_read(self.owner, arguments, attachments))
        class Provider:
            """仅检查确定性解析版本，无取文、推理或写入。"""
            def parse_contract_matches(self, document):
                """现配置与历史解析不匹配。"""
                return False
        self.assertIsNone(self.progress.cached_read(self.owner, arguments, attachments, provider=Provider()))
        self.assertEqual(self.db.list_actions(self.owner)[0]['observation'], observation)

    def test_cross_version_and_current_user_range_changes_fail_closed(self):
        """C1：明确跨版或用户收紧页面后不沿旧来源继续。"""
        stored = self.progress.source(self.owner, self.eid)
        self.rebind()
        self.db.save_paper(self.new_id, {'paper_id': self.new_id, 'title': 'Synthetic source', 'source': 'arxiv',
            'paper_url': 'https://arxiv.org/abs/2603.00595v2', 'version': 'v2'})
        self.assertFalse(self.progress.source_version_matches(self.owner, stored))
        self.assertFalse(self.papers.authorized_source(self.owner, self.eid)['readback']['available'])
        self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', ('仅阅读第2页，不保存', self.owner))
        with self.assertRaises(PermissionError):
            self.papers.authorized_source(self.owner, self.eid)
