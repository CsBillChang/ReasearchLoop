"""严格等值页正文副本去重：纯数据、三角色准备与旧收据的零模型合同。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

from research_loop.agents.graph import build_graph, initial_state
from research_loop.agents.memory import ContextService
from research_loop.config import Settings
from research_loop.domain.context import deduplicate_read_paper_pages
from research_loop.domain.evidence import Document, PageEvidence
from research_loop.domain.research_progress import ProgressFinding, ResearchProgressInput
from research_loop.domain.tasks import ActionRecord, TaskCreate
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.providers.token_counting import counting_policy, estimate_details
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.services.research_progress import ResearchProgressService
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.storage.memory import digest
from research_loop.storage.token_budget import TokenBudgetStore
from research_loop.tools.registry import ToolRegistry


ROLES = ('supervisor', 'researcher', 'reviewer')


def source_messages(pages, *, name='read_paper', ok=True, extra_data=None):
    """构造配对消息，保留供应商推理和非目标元数据以检查投影边界。"""
    data = {'document_id': 'doc', 'content_sha256': 'a' * 64,
            'total_pages': len(pages), 'returned_pages': [p.get('page_number') for p in pages],
            'pages': deepcopy(pages), 'known_gaps': ['图像未解释']}
    data.update(extra_data or {})
    return [
        {'role': 'system', 'content': '合成角色职责'},
        {'role': 'user', 'content': '核对原文，不保存'},
        {'role': 'assistant', 'content': '', 'reasoning_content': '供应商原始字段',
         'vendor_metadata': {'signature': 'opaque'},
         'tool_calls': [{'id': 'read-one', 'type': 'function',
                         'function': {'name': name, 'arguments': '{"target_ref":"input"}'}}]},
        {'role': 'tool', 'tool_call_id': 'read-one',
         'content': json.dumps({'ok': ok, 'kind': 'success' if ok is True else 'execution_error',
                                'data': data}, ensure_ascii=False)},
    ]


def page(text='Exact source: condition A.\n逐字原文。', **updates):
    """生成带稳定页身份、资产及缺口的合成读页。"""
    value = {'evidence_id': 'page-one', 'document_id': 'doc', 'page_number': 1,
             'text': text, 'parsed_markdown': text,
             'assets': [{'asset_id': 'asset-one', 'kind': 'image', 'page_number': 1}],
             'gaps': ['图片未理解'], 'source_sha256': 'a' * 64,
             'readback': {'target_ref': 'input', 'start_page': 1, 'page_count': 1}}
    value.update(updates)
    return value


def observation(messages):
    """读取本测试唯一原文工具消息，不以私有业务状态替代请求观察。"""
    message = next(m for m in messages if m.get('role') == 'tool' and m.get('tool_call_id') == 'read-one')
    return json.loads(message['content'])


class ExactPageDeduplicationTests(unittest.TestCase):
    """仅删除非空字符串的严格等值副本，不推断相似语义。"""

    def test_exact_duplicate_keeps_full_text_and_every_other_field(self):
        """中文、转义及换行原样保留，整个原始消息对象不被修改。"""
        raw = source_messages([page('条件\\甲\n"逐字引句"\t😀 e\u0301')])
        original = deepcopy(raw)
        result, metrics = deduplicate_read_paper_pages(raw)
        expected = observation(original)
        expected['data']['pages'][0].pop('parsed_markdown')
        self.assertEqual(observation(result), expected)
        self.assertEqual(raw, original)
        self.assertEqual(result[:-1], original[:-1])
        self.assertEqual(result[-1]['tool_call_id'], 'read-one')
        self.assertEqual(metrics['page_count'], 1)
        self.assertEqual(metrics['tool_call_ids'], ['read-one'])
        self.assertEqual(metrics['pages'], [{'tool_call_id': 'read-one', 'page_index': 0}])
        validate_messages(result)
        result[-1]['content'] = 'changed after projection'
        self.assertEqual(raw, original)

    def test_unequal_missing_empty_and_nonstring_fields_are_unchanged(self):
        """不进行空白、Unicode或类型归一化，缺失与空字段保持存在性。"""
        samples = [
            page(text='A', parsed_markdown=' A'),
            page(text='A\nB', parsed_markdown='A B'),
            page(text='é', parsed_markdown='e\u0301'),
            page(text='', parsed_markdown=''),
            page(text=None, parsed_markdown=None),
            page(text=1, parsed_markdown=True),
            page(text=['A'], parsed_markdown=['A']),
        ]
        missing_markdown = page(); missing_markdown.pop('parsed_markdown')
        missing_text = page(); missing_text.pop('text')
        samples += [missing_markdown, missing_text]
        for sample in samples:
            with self.subTest(sample=sample):
                raw = source_messages([sample])
                before = deepcopy(raw)
                result, metrics = deduplicate_read_paper_pages(raw)
                self.assertEqual(result, before)
                self.assertEqual(raw, before)
                self.assertEqual(metrics['page_count'], 0)
                self.assertEqual(metrics['tool_call_ids'], [])

    def test_failed_non_read_and_unbound_messages_are_not_rewritten(self):
        """只有成功read_paper且工具ID有真实绑定才是候选。"""
        samples = [source_messages([page()], ok=False),
                   source_messages([page()], ok=1),
                   source_messages([page()], name='retrieve_evidence'),
                   source_messages([page()], name='read_memory')]
        unbound = source_messages([page()]); unbound[-1]['tool_call_id'] = 'foreign-call'
        samples.append(unbound)
        no_call = source_messages([page()]); no_call.pop(2)
        samples.append(no_call)
        for raw in samples:
            with self.subTest(message=raw[-1]):
                before = deepcopy(raw)
                result, metrics = deduplicate_read_paper_pages(raw)
                self.assertEqual(result, before)
                self.assertEqual(raw, before)
                self.assertEqual(metrics['page_count'], 0)
                self.assertEqual(metrics['tool_call_ids'], [])

    def test_malformed_and_non_page_observations_pass_through(self):
        """历史坏JSON及不属于页合同的结构不由投影器修复。"""
        for content in ('{incomplete', '[]', 'null', '"plain"',
                        '{"ok":true,"data":null}',
                        '{"ok":true,"data":{"pages":"not-list"}}'):
            with self.subTest(content=content):
                raw = source_messages([page()]); raw[-1]['content'] = content
                result, metrics = deduplicate_read_paper_pages(raw)
                self.assertEqual(result, raw)
                self.assertEqual(metrics['page_count'], 0)

    def test_equal_words_do_not_merge_distinct_page_or_document_identities(self):
        """相同正文的多页、不同文档和版本各自保留，第二次投影不再改动。"""
        pages = [page(), page(evidence_id='another-page', page_number=2),
                 page(evidence_id='new-version', document_id='another-doc', source_sha256='b' * 64)]
        raw = source_messages(pages)
        first, metrics = deduplicate_read_paper_pages(raw)
        self.assertEqual(metrics['page_count'], 3)
        actual = observation(first)['data']['pages']
        for before, after in zip(pages, actual):
            restored = {**after, 'parsed_markdown': after['text']}
            self.assertEqual(restored, before)
        second, repeated = deduplicate_read_paper_pages(first)
        self.assertEqual(second, first)
        self.assertEqual(repeated['page_count'], 0)
        self.assertEqual(repeated['tool_call_ids'], [])

    def test_parallel_tool_results_keep_pairing_and_per_call_accounting(self):
        """并行返回仍对应原ID，不合并观察或改写assistant参数。"""
        raw = source_messages([page()])
        second = deepcopy(raw[2]['tool_calls'][0]); second['id'] = 'read-two'
        raw[2]['tool_calls'].append(second)
        result_two = deepcopy(raw[-1]); result_two['tool_call_id'] = 'read-two'
        raw.append(result_two)
        before = deepcopy(raw)
        result, metrics = deduplicate_read_paper_pages(raw)
        self.assertEqual(raw, before)
        self.assertEqual(result[:3], before[:3])
        self.assertEqual(metrics['page_count'], 2)
        self.assertEqual(metrics['tool_call_ids'], ['read-one', 'read-two'])
        self.assertEqual(metrics['pages'], [{'tool_call_id': 'read-one', 'page_index': 0},
                                           {'tool_call_id': 'read-two', 'page_index': 0}])
        self.assertEqual([m['tool_call_id'] for m in result if m['role'] == 'tool'], ['read-one', 'read-two'])
        validate_messages(result)

    def test_reviewer_user_evidence_bundle_is_outside_this_candidate(self):
        """本候选只处理工具观察，不顺手精简Reviewer首轮证据包。"""
        raw = [{'role': 'system', 'content': 'Reviewer'},
               {'role': 'user', 'content': json.dumps({'evidence_bundle': [page()]})}]
        result, metrics = deduplicate_read_paper_pages(raw)
        self.assertEqual(result, raw)
        self.assertEqual(metrics['page_count'], 0)

    def test_reused_tool_call_id_across_rounds_is_not_projected(self):
        """完整历史允许复用ID，但名称映射不唯一时必须保守保留全部观察。"""
        for first_name, second_name in (('read_memory', 'read_paper'),
                                        ('read_paper', 'read_memory'),
                                        ('read_paper', 'read_paper')):
            with self.subTest(first=first_name, second=second_name):
                first = source_messages([page()], name=first_name)
                second = source_messages([page(evidence_id='another-page')], name=second_name)
                raw = first + second[2:]
                original = deepcopy(raw)
                # 每一轮有自己的完整返回；协议合法不代表全局ID唯一。
                validate_messages(raw)
                projected, metrics = deduplicate_read_paper_pages(raw)
                self.assertEqual(projected, original)
                self.assertEqual(raw, original)
                self.assertEqual(metrics['page_count'], 0)
                self.assertEqual(metrics['tool_call_ids'], [])
                self.assertEqual(metrics['pages'], [])


class ContextDeduplicationTests(unittest.TestCase):
    """通过正式ContextService验证三角色、容量和档案合同。"""

    def setUp(self):
        """每例显式临时SQLite与Settings，不读取日常.env或启动服务。"""
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.settings = Settings(project_root=self.root, data_dir=self.root / 'data',
                                 output_dir=self.root / 'notes', database_url='sqlite:///' + str(self.root / 'test.db'),
                                 retrieval_enabled=False, context_deduplicate_read_paper=True)
        self.db = Database(self.settings.database_url); self.db.ensure_schema()
        self.memory = MemoryService(self.db)

    def fixture(self, role, *, conversation=False, settings=None,
                text='Exact source: condition A.\n合成逐字原文。'):
        """注册合成授权页并生成完整观察，不使用本地私人PDF。"""
        settings = settings or self.settings
        cid = None
        if conversation:
            cid = ConversationService(self.db, settings).create('合成会话', role)['conversation_id']
        tid = TaskService(self.db, settings).create(TaskCreate(
            prompt='比较合成原文的条件，不保存', save_policy='deny', conversation_id=cid)).task_id
        doc_id = 'doc-' + tid
        paper_id = 'paper-' + tid
        eid = 'page-' + tid
        doc = Document(document_id=doc_id, paper_id=paper_id, content_sha256='a' * 64,
                       parser_name='fixture', parser_version='1', config_sha256='b' * 64,
                       total_pages=1, cache_path='unused', gaps=['图片未解释'])
        self.db.save_document(doc.model_dump())
        evidence = PageEvidence(evidence_id=eid, document_id=doc_id, page_number=1,
                                text=text, parsed_markdown=text,
                                assets=[{'asset_id': 'fixture-image', 'page_number': 1}],
                                gaps=['图片未解释'])
        self.db.save_page(evidence.model_dump())
        self.db.mark_task_evidence(tid, eid)
        self.db.grant_paper(tid, paper_id, 'fixture')
        messages = source_messages([page(text, evidence_id=eid, document_id=doc_id)],
                                   extra_data={'paper_id': paper_id, 'document_id': doc_id})
        state = {'task_id': tid, 'root_task_id': tid, 'role': role,
                 'prompt': '比较合成原文的条件，不保存', 'model_requests': 0}
        return tid, state, messages

    @staticmethod
    def prepare(context, role, state, messages, tools=None):
        """显式选择生产使用的角色入口，避免只覆盖Supervisor。"""
        return (context.prepare if role == 'supervisor' else context.prepare_role)(state, messages, tools or [])

    def assert_metrics_and_budget(self, tid, result, metrics):
        """容量指标与共享预算按同一实际投影输入估算，不冒称实际Token。"""
        store = TokenBudgetStore(self.db)
        inputs = {'messages': result, 'tools': []}
        details = estimate_details(inputs, store.counting_policy(tid))
        self.assertEqual(metrics['serialized_input_bytes'], details['transport_bytes'])
        self.assertEqual(metrics['estimated_input_tokens'], details['estimated_input_tokens'])
        self.assertFalse(metrics['token_estimate_is_exact'])
        deduplication = metrics['read_paper_deduplication']
        self.assertTrue(deduplication['enabled'])
        self.assertEqual(deduplication['page_count'], 1)
        self.assertEqual(deduplication['tool_call_ids'], ['read-one'])
        self.assertEqual(deduplication['pages'], [{'tool_call_id': 'read-one', 'page_index': 0}])
        store.reserve(tid, tid, 1, {**inputs, 'max_tokens': metrics['output_reserve_tokens']}, 'supervisor')
        self.assertEqual(store.summary(tid)['reserved_tokens'],
                         metrics['estimated_input_tokens'] + metrics['output_reserve_tokens'])

    def test_all_three_roles_project_under_no_capacity_pressure(self):
        """常规短请求也去重，不依赖触发压缩或原文摘录。"""
        for role in ROLES:
            with self.subTest(role=role):
                tid, state, messages = self.fixture(role)
                before = deepcopy(messages)
                result, metrics = self.prepare(ContextService(self.db, self.memory, self.settings), role, state, messages)
                self.assertNotIn('parsed_markdown', observation(result)['data']['pages'][0])
                self.assertEqual(observation(result)['data']['pages'][0]['text'], observation(before)['data']['pages'][0]['text'])
                self.assertEqual(messages, before)
                self.assertEqual(result[2], before[2])
                self.assertFalse(metrics.get('source_projected_calls'))
                self.assertFalse(metrics.get('compressed_groups'))
                validate_messages(result)
                self.assert_metrics_and_budget(tid, result, metrics)
                if role != 'supervisor':
                    archive = self.db.fetchone('SELECT * FROM context_archives WHERE task_id=?', (tid,))
                    self.assertIsNotNone(archive)
                    self.assertEqual(observation(json.loads(archive['original_json'])['messages']), observation(before))
                    self.assertEqual(json.loads(archive['projected_json']), result)

    def test_supervisor_full_conversation_path_preserves_original_archive(self):
        """完整会话路径的新档案保留双字段，待发副本与容量记录一致。"""
        tid, state, messages = self.fixture('supervisor', conversation=True)
        before = deepcopy(messages)
        result, metrics = self.prepare(ContextService(self.db, self.memory, self.settings), 'supervisor', state, messages)
        self.assertNotIn('parsed_markdown', observation(result)['data']['pages'][0])
        self.assertEqual(messages, before)
        archive = self.db.fetchone('SELECT * FROM context_archives WHERE task_id=?', (tid,))
        self.assertIsNotNone(archive)
        raw = json.loads(archive['original_json'])
        self.assertEqual(observation(raw['messages']), observation(before))
        self.assertEqual(json.loads(archive['projected_json']), result)
        self.assertEqual(json.loads(archive['metrics_json'])['read_paper_deduplication'],
                         metrics['read_paper_deduplication'])
        self.assert_metrics_and_budget(tid, result, metrics)

    def test_disabled_mode_does_not_change_short_messages_or_add_metric(self):
        """开关默认关闭，低压力路径保留原请求和旧指标结构。"""
        disabled = self.settings.model_copy(update={'context_deduplicate_read_paper': False})
        for role in ROLES:
            with self.subTest(role=role):
                _, state, messages = self.fixture(role)
                before = deepcopy(messages)
                result, metrics = self.prepare(ContextService(self.db, self.memory, disabled), role, state, messages)
                self.assertEqual(result, before)
                self.assertEqual(messages, before)
                self.assertNotIn('read_paper_deduplication', metrics)

    def test_archive_namespace_does_not_reuse_disabled_projection(self):
        """同一原件开关变化不能命中旧投影，但旧档案内容保持不变。"""
        disabled = self.settings.model_copy(update={'context_deduplicate_read_paper': False})
        for role in ROLES:
            with self.subTest(role=role):
                tid, state, messages = self.fixture(role, conversation=role == 'supervisor')
                self.prepare(ContextService(self.db, self.memory, disabled), role, state, messages)
                old = {r['snapshot_id']: r for r in self.db.fetchall('SELECT * FROM context_archives WHERE task_id=?', (tid,))}
                result, _ = self.prepare(ContextService(self.db, self.memory, self.settings), role, state, messages)
                all_rows = {r['snapshot_id']: r for r in self.db.fetchall('SELECT * FROM context_archives WHERE task_id=?', (tid,))}
                self.assertGreater(len(all_rows), len(old))
                for identity, row in old.items():
                    self.assertEqual(all_rows[identity], row)
                newest = next(r for identity, r in all_rows.items() if identity not in old)
                self.assertEqual(json.loads(newest['projected_json']), result)
                self.assertIn('parsed_markdown', observation(json.loads(newest['original_json'])['messages'])['data']['pages'][0])

    def boundary_fixture(self, role):
        """只由双字段重复形成容量压力；有效正文单份加完整状态仍可容纳。"""
        configured = self.settings.model_copy(update={'context_window_tokens': 32000,
            'context_output_reserve': 1024, 'context_safety_tokens': 500, 'context_compress_ratio': 0.99})
        policy = counting_policy(configured)
        capacity = configured.context_window_tokens - configured.context_output_reserve - configured.context_safety_tokens
        text = None
        for repeats in range(200, 10000, 100):
            candidate = 'Exact source condition A. ' * repeats
            raw = source_messages([page(candidate)])
            projected, _ = deduplicate_read_paper_pages(raw)
            before = estimate_details({'messages': raw, 'tools': []}, policy)['estimated_input_tokens']
            after = estimate_details({'messages': projected, 'tools': []}, policy)['estimated_input_tokens']
            if before > capacity * 1.05 and after < capacity * 0.7:
                text = candidate
                break
        self.assertIsNotNone(text, 'synthetic duplicate must straddle the configured input capacity')
        tid, state, messages = self.fixture(role, settings=configured, text=text)
        return configured, tid, state, messages

    def test_duplicate_only_capacity_pressure_keeps_full_text_for_three_roles(self):
        """去重足以落入窗口时，不再摘录正文或增加读回动作。"""
        for role in ROLES:
            with self.subTest(role=role):
                configured, tid, state, messages = self.boundary_fixture(role)
                original = deepcopy(messages)
                result, metrics = self.prepare(ContextService(self.db, self.memory, configured), role, state, messages)
                self.assertGreater(metrics['before'], metrics['input_capacity'])
                self.assertLessEqual(metrics['after'], metrics['input_capacity'])
                self.assertEqual(observation(result)['data']['pages'][0]['text'],
                                 observation(original)['data']['pages'][0]['text'])
                self.assertFalse(metrics.get('source_projected_calls'))
                self.assertFalse(metrics.get('compressed_groups'))
                self.assertFalse(metrics.get('workspace_externalized'))
                self.assertEqual(metrics['read_paper_deduplication']['page_count'], 1)
                self.assertEqual(messages, original)
                self.assertEqual(self.db.list_model_calls(tid), [])
                validate_messages(result)

    def test_boundary_deduplication_still_rejects_foreign_and_stale_source(self):
        """体积变小不能绕过原页授权、文本和版本检查。"""
        for role in ROLES:
            configured, tid, state, messages = self.boundary_fixture(role)
            for mutation in ('foreign_page', 'stale_text', 'stale_version'):
                with self.subTest(role=role, mutation=mutation):
                    changed = deepcopy(messages)
                    value = observation(changed)
                    if mutation == 'foreign_page':
                        value['data']['pages'][0]['evidence_id'] = 'foreign-page'
                    elif mutation == 'stale_text':
                        value['data']['pages'][0]['text'] += ' text that is not in the authorized page'
                        value['data']['pages'][0]['parsed_markdown'] = value['data']['pages'][0]['text']
                    else:
                        value['data']['content_sha256'] = 'c' * 64
                    changed[-1]['content'] = json.dumps(value, ensure_ascii=False)
                    with self.assertRaisesRegex(ValueError, 'CONTEXT_SOURCE_STALE'):
                        self.prepare(ContextService(self.db, self.memory, configured), role, state, changed)
                    self.assertEqual(self.db.list_model_calls(tid), [])

    def test_switch_rollback_keeps_enabled_archives_and_repeated_projection_stable(self):
        """开关回退只影响新副本，旧档案不改；再启用不制造新身份。"""
        disabled = self.settings.model_copy(update={'context_deduplicate_read_paper': False})
        for role in ROLES:
            with self.subTest(role=role):
                tid, state, messages = self.fixture(role, conversation=role == 'supervisor')
                first, _ = self.prepare(ContextService(self.db, self.memory, self.settings), role, state, messages)
                enabled_rows = {r['snapshot_id']: r for r in self.db.fetchall('SELECT * FROM context_archives WHERE task_id=?', (tid,))}
                rolled_back, metrics = self.prepare(ContextService(self.db, self.memory, disabled), role, state, messages)
                self.assertIn('parsed_markdown', observation(rolled_back)['data']['pages'][0])
                self.assertNotIn('read_paper_deduplication', metrics)
                rows = {r['snapshot_id']: r for r in self.db.fetchall('SELECT * FROM context_archives WHERE task_id=?', (tid,))}
                for identity, value in enabled_rows.items():
                    self.assertEqual(rows[identity], value)
                repeated, _ = self.prepare(ContextService(self.db, self.memory, self.settings), role, state, messages)
                self.assertEqual(repeated, first)
                self.assertEqual({r['snapshot_id']: r for r in self.db.fetchall('SELECT * FROM context_archives WHERE task_id=?', (tid,))}, rows)

    def test_researcher_archive_binds_new_persistent_progress_with_same_messages(self):
        """同一输入消息遇到新持久成果时产生新投影档案，旧原件与关闭hash不变。"""
        tid, state, messages = self.fixture('researcher')
        original = deepcopy(messages)
        context = ContextService(self.db, self.memory, self.settings)
        first, _ = self.prepare(context, 'researcher', state, messages)
        old = {r['snapshot_id']: r for r in self.db.fetchall('SELECT * FROM context_archives WHERE task_id=?', (tid,))}
        self.assertEqual(len(old), 1)
        eid = observation(messages)['data']['pages'][0]['evidence_id']
        payload = ResearchProgressInput(findings=[ProgressFinding(
            text='合成新持久成果：当前条件A已经取得原文。', evidence_ids=[eid])])
        service = ResearchProgressService(self.db)
        prepared = service.prepare(tid, payload)
        action = ActionRecord(task_id=tid, sequence=1, tool_name='record_research_progress',
                              arguments=payload.model_dump())
        self.db.save_action(action.model_dump(mode='json'))
        action.status = 'succeeded'
        action.observation = {'ok': True, 'kind': 'success', 'data': {'research_progress': prepared}}
        service.commit_action(action.model_dump(mode='json'), None)
        self.assertEqual(service.snapshot(tid)['revision'], 1)
        second, _ = self.prepare(context, 'researcher', state, messages)
        self.assertNotEqual(second, first)
        self.assertIn('合成新持久成果', json.dumps(second, ensure_ascii=False))
        updated = {r['snapshot_id']: r for r in self.db.fetchall('SELECT * FROM context_archives WHERE task_id=?', (tid,))}
        self.assertEqual(len(updated), len(old) + 1)
        for identity, row in old.items():
            self.assertEqual(updated[identity], row)
        newest = next(row for identity, row in updated.items() if identity not in old)
        self.assertEqual(json.loads(newest['projected_json']), second)
        self.assertEqual(json.loads(newest['original_json'])['messages'], original)
        self.assertEqual(messages, original)
        disabled = self.settings.model_copy(update={'context_deduplicate_read_paper': False})
        legacy, metrics = self.prepare(ContextService(self.db, self.memory, disabled), 'researcher', state, messages)
        expected_old_hash = digest([tid, legacy, []])
        legacy_archive = self.db.fetchone('SELECT * FROM context_archives WHERE snapshot_id=?', (expected_old_hash,))
        self.assertIsNotNone(legacy_archive)
        self.assertEqual(json.loads(legacy_archive['projected_json']), legacy)
        self.assertNotIn('read_paper_deduplication', metrics)
        self.assertEqual(self.db.list_model_calls(tid), [])

    def test_supervisor_short_noop_does_not_create_projection_archive(self):
        """开启但没有目标页时仅返回零变更指标，不给短任务增加档案写入。"""
        tid, state, _ = self.fixture('supervisor')
        messages = [{'role': 'system', 'content': '合成职责'},
                    {'role': 'user', 'content': '普通短请求，没有读页往返。'}]
        original = deepcopy(messages)
        context = ContextService(self.db, self.memory, self.settings)
        for _ in range(2):
            projected, metrics = self.prepare(context, 'supervisor', state, messages)
            self.assertEqual(projected, original)
            self.assertEqual(messages, original)
            self.assertTrue(metrics['read_paper_deduplication']['enabled'])
            self.assertEqual(metrics['read_paper_deduplication']['page_count'], 0)
            self.assertEqual(self.db.fetchall('SELECT * FROM context_archives WHERE task_id=?', (tid,)), [])
        disabled = self.settings.model_copy(update={'context_deduplicate_read_paper': False})
        projected, metrics = self.prepare(ContextService(self.db, self.memory, disabled), 'supervisor', state, messages)
        self.assertEqual(projected, original)
        self.assertNotIn('read_paper_deduplication', metrics)
        self.assertEqual(self.db.fetchall('SELECT * FROM context_archives WHERE task_id=?', (tid,)), [])

    def test_settings_default_and_explicit_environment_switch(self):
        """只允许实验配置显式开启，普通Settings仍关闭。"""
        self.assertFalse(Settings(project_root=self.root, data_dir=self.root / 'data',
                                  output_dir=self.root / 'notes').context_deduplicate_read_paper)
        with patch.dict('os.environ', {'RESEARCHLOOP_CONTEXT_DEDUPLICATE_READ_PAPER': '1'}, clear=True):
            enabled = Settings.from_env(self.root)
        self.assertTrue(enabled.context_deduplicate_read_paper)

    def test_three_role_new_requests_reach_mock_http_after_persistence(self):
        """正式图、预算与Provider编码共同验证；HTTP/SSE响应仅由本地替身产生。"""
        for role in ROLES:
            with self.subTest(role=role):
                tid, state, messages = self.fixture(role)
                original = deepcopy(messages)
                captured = []
                response_id = 'offline-dedup-' + role
                usage = {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}

                def handler(request):
                    """只接收已去重wire，三个角色均不允许真实HTTP连接。"""
                    self.assertEqual(str(request.url), 'https://fixture.invalid/chat/completions')
                    self.assertEqual(request.headers['Authorization'], 'Bearer fixture-offline-not-a-secret')
                    payload = json.loads(request.content)
                    captured.append(deepcopy(payload))
                    validate_messages(payload['messages'])
                    self.assertNotIn('parsed_markdown', observation(payload['messages'])['data']['pages'][0])
                    self.assertEqual(observation(payload['messages'])['data']['pages'][0]['text'],
                                     observation(original)['data']['pages'][0]['text'])
                    self.assertEqual(payload['messages'][2], original[2])
                    self.assertEqual(payload['max_tokens'], self.settings.context_output_reserve)
                    # 原生请求和预算先于真正HTTP发送落库；wire不能另用未记录投影。
                    receipt = self.db.get_model_call(tid, 1)
                    self.assertIsNotNone(receipt)
                    native = json.loads(receipt['request_json'])
                    self.assertEqual(native['messages'], payload['messages'])
                    self.assertEqual(native['tools'], payload.get('tools', []))
                    self.assertEqual(native['max_tokens'], payload['max_tokens'])
                    self.assertEqual(len(self.db.fetchall('SELECT * FROM token_reservations WHERE task_id=?', (tid,))), 1)
                    if role == 'supervisor':
                        self.assertTrue(payload.get('stream'))
                        self.assertEqual(payload['stream_options'], {'include_usage': True})
                        chunks = [
                            {'id': response_id, 'model': self.settings.deepseek_model,
                             'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'fixture-complete'},
                                          'finish_reason': None}]},
                            {'id': response_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]},
                            {'id': response_id, 'choices': [], 'usage': usage},
                        ]
                        data = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks)
                        return httpx.Response(200, text=data + 'data: [DONE]\n\n',
                                              headers={'content-type': 'text/event-stream'})
                    self.assertNotIn('stream', payload)
                    return httpx.Response(200, json={
                        'id': response_id, 'model': self.settings.deepseek_model, 'usage': usage,
                        'choices': [{'finish_reason': 'stop',
                                     'message': {'role': 'assistant', 'content': 'fixture-complete'}}]})

                context = ContextService(self.db, self.memory, self.settings)
                graph_state = initial_state(tid, state['prompt'], 'deny')
                graph_state.update(role=role, root_task_id=tid, messages=deepcopy(messages))
                with (httpx.Client(transport=httpx.MockTransport(handler), trust_env=False) as client,
                      patch.object(httpx.HTTPTransport, 'handle_request',
                                   side_effect=AssertionError('external network forbidden'))):
                    provider = DeepSeekProvider('fixture-offline-not-a-secret', model=self.settings.deepseek_model,
                                                base_url='https://fixture.invalid', client=client)
                    result = build_graph(provider, ToolRegistry([]),
                                         {'context': context, 'stream_public': True}, self.db).invoke(graph_state)
                self.assertEqual(len(captured), 1, result.get('final_reason'))
                self.assertEqual(result['final_answer'], 'fixture-complete')
                self.assertEqual(messages, original)
                receipt = self.db.get_model_call(tid, 1)
                self.assertEqual(receipt['status'], 'succeeded')
                self.assertEqual(receipt['request_id'], response_id)
                self.assertEqual(json.loads(receipt['usage_json']), usage)
                self.assertEqual(json.loads(receipt['request_json'])['messages'], captured[0]['messages'])
                self.assertEqual(result['context_metrics']['read_paper_deduplication']['page_count'], 1)


class DedupReceiptRecoveryTests(unittest.TestCase):
    """已持久请求不因新开关而重投影，成功与unknown都不能重发。"""

    def test_successful_and_unknown_receipts_bypass_context_for_three_roles(self):
        """旧请求完整保持；若调用投影或供应商，替身立即使本例失败。"""
        for role in ROLES:
            for status in ('succeeded', 'unknown'):
                with self.subTest(role=role, status=status), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    settings = Settings(project_root=root, data_dir=root / 'data', output_dir=root / 'out',
                                        retrieval_enabled=False, context_deduplicate_read_paper=True)
                    db = Database('sqlite:///' + str(root / 'receipt.db')); db.ensure_schema()
                    tid = role + '-' + status
                    db.create_task(tid, None, 'fixture', '合成恢复', 'deny', [], token_settings=settings)
                    request = {'messages': source_messages([page()]), 'tools': [],
                               'force_text': False, 'terminal_only': False, 'max_tokens': 65536}
                    raw = {'id': 'old-receipt', 'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3},
                           'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'done'}}]}
                    db.save_model_call(tid, 1, status, request,
                                       response=raw if status == 'succeeded' else {'error': 'unknown'},
                                       request_id='old-receipt', usage=raw['usage'] if status == 'succeeded' else {},
                                       finish_reason='stop' if status == 'succeeded' else None, sent_state='sent')
                    original = deepcopy(db.get_model_call(tid, 1))
                    context = SimpleNamespace(settings=settings,
                                              prepare=Mock(side_effect=AssertionError('no reprepare')),
                                              prepare_role=Mock(side_effect=AssertionError('no role reprepare')))
                    provider = SimpleNamespace(complete=Mock(side_effect=AssertionError('no resend')))
                    state = initial_state(tid, '合成恢复', 'deny')
                    state['role'] = role
                    result = build_graph(provider, ToolRegistry([]), {'context': context}, db).invoke(state)
                    self.assertEqual(db.get_model_call(tid, 1), original)
                    context.prepare.assert_not_called()
                    context.prepare_role.assert_not_called()
                    provider.complete.assert_not_called()
                    if status == 'succeeded':
                        self.assertEqual(result['final_answer'], 'done')
                    else:
                        self.assertIn('unknown', result['final_reason'])


if __name__ == '__main__':
    unittest.main()
