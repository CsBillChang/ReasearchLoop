"""批量收尾、子任务原因和补读可见性回归；只用替身响应与合成原文。"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from research_loop.agents.graph import build_graph, initial_state
from research_loop.agents.memory import ContextService, estimate
from research_loop.config import Settings
from research_loop.domain.tasks import Observation, TaskCreate
from research_loop.providers.deepseek import validate_messages
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor, research_stop_details
from integration.test_context_continuity import exchange, LongPDF


class ProbeRegistry:
    """工具执行计数替身；终结结果显式为partial，不伪造真实研究通过。"""

    def __init__(self):
        """保留实际被执行的调用，用于区分拒绝回执。"""
        self.executed = []

    def schemas(self, role):
        """只开放本测试相关的角色工具。"""
        return [{'type': 'function', 'function': {'name': name, 'parameters': {'type': 'object'}}}
                for name in ('retrieve_evidence', 'finish_research')]

    def execute(self, name, args, context):
        """只生成合成观察，不访问供应商或保存文件。"""
        self.executed.append(name)
        if name == 'finish_research':
            return Observation(ok=True, kind='success', data={'status': 'partial', 'findings': ['已有候选发现'],
                'paper_ids': [], 'evidence_ids': [], 'gaps': ['合成原文不足以完成全部目标']})
        return Observation(ok=True, kind='success', data={'hits': []})


def pending_batch(state, names):
    """加入合成响应及ProbeRegistry实际范围；观察必须逐一配对。"""
    calls = [{'call_id': uuid4().hex, 'name': name, 'arguments': {}} for name in names]
    state['messages'] = state['messages'] + [{'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': c['call_id'], 'type': 'function', 'function': {'name': c['name'], 'arguments': '{}'}} for c in calls]}]
    state['pending_tool_calls'] = calls
    state['pending_tool_scope'] = ['retrieve_evidence', 'finish_research']
    return state


class BatchTailReadbackTests(unittest.TestCase):
    """默认SQLite，真实数据库仅接受明确隔离库；各例不清理共享旧库。"""

    def setUp(self):
        """创建唯一任务和临时输出目录，权限始终deny。"""
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        url = os.getenv('BATCH_TAIL_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root/'test.db')
        if not url.startswith('sqlite:') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_batch_tail_'):
            raise ValueError('isolated batch tail database required')
        self.settings = Settings(project_root=self.root, data_dir=self.root/'data', output_dir=self.root/'notes',
            database_url=url, max_model_requests=20, max_tool_actions=48, reserve_model_requests=4,
            retrieval_enabled=False, task_token_limit=800000)
        self.db = Database(url); self.db.ensure_schema()
        self.tasks = TaskService(self.db, self.settings)
        self.tid = self.tasks.create(TaskCreate(prompt='只比较已有材料，不保存', idempotency_key=uuid4().hex)).task_id

    def state(self, used=18):
        """复现实际18/20子任务状态，不扩大原局部额度。"""
        state = initial_state(self.tid, '对照原文，不保存', 'deny', max_tool_actions=20)
        state.update(role='researcher', tool_actions=used)
        return state

    def test_batch_drains_and_reserves_structured_handoff(self):
        """18/20时四个研究调用仅执行一个，随后在原额度内交接。"""
        registry = ProbeRegistry(); graph = build_graph(None, registry, {}, self.db)
        state = pending_batch(self.state(), ['retrieve_evidence']*4)
        state.update(graph.nodes['execute_tools'].invoke(state))
        self.assertEqual(registry.executed, ['retrieve_evidence'])
        self.assertEqual((state['tool_actions'], state['action_sequence']), (19, 22))
        self.assertEqual(len(state['actions']), 4)
        self.assertFalse(state['finished']); self.assertFalse(state['pending_tool_calls'])
        validate_messages(state['messages'])
        self.assertEqual(self.db.budget_summary(self.tid)['tool_used'], 1)
        pending_batch(state, ['finish_research'])
        state.update(graph.nodes['execute_tools'].invoke(state))
        self.assertTrue(state['finished'])
        self.assertEqual(state['structured_result']['status'], 'partial')
        self.assertEqual(state['tool_actions'], 20)
        self.assertEqual(self.db.budget_summary(self.tid)['tool_used'], 2)
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))

    def test_action_receipts_recover_after_checkpoint_write_failure(self):
        """整批观察落库但checkpoint写入失败后，重放不执行、不重复扣费。"""
        class FailedCheckpoint:
            """模拟进程在行动落库之后失去checkpoint写入机会。"""
            def save(self, task_id, state):
                """独立恢复边界故障，不修改原观察。"""
                raise RuntimeError('controlled checkpoint failure')
        registry = ProbeRegistry()
        state = pending_batch(self.state(), ['retrieve_evidence']*4)
        original = deepcopy(state)
        graph = build_graph(None, registry, {}, self.db, checkpoint_store=FailedCheckpoint())
        with self.assertRaisesRegex(RuntimeError, 'controlled checkpoint failure'):
            graph.nodes['execute_tools'].invoke(state)
        before = self.db.list_actions(self.tid)
        graph = build_graph(None, registry, {}, self.db)
        recovered = graph.nodes['execute_tools'].invoke(original)
        self.assertEqual(registry.executed, ['retrieve_evidence'])
        self.assertEqual(self.db.list_actions(self.tid), before)
        self.assertEqual(recovered['tool_actions'], 19)
        self.assertEqual(self.db.budget_summary(self.tid)['tool_used'], 1)
        validate_messages(recovered['messages'])

    def test_atomic_shared_tail_survives_parallel_batches(self):
        """竞争者不能越过六槽保护线；子交接和父收尾仍能依次领取。"""
        self.db.execute('UPDATE task_budgets SET tool_used=40 WHERE root_task_id=?', (self.tid,))
        with ThreadPoolExecutor(max_workers=4) as pool:
            slots = list(pool.map(lambda _: self.db.reserve_tool_slot(self.tid, 6), range(12)))
        self.assertEqual(sum(slots), 2)
        self.assertEqual([self.db.reserve_tool_slot(self.tid, 4) for _ in range(3)], [True, True, False])
        self.assertEqual([self.db.reserve_tool_slot(self.tid) for _ in range(5)], [True]*4+[False])
        self.assertEqual(self.db.budget_summary(self.tid)['tool_used'], 48)

    def test_full_graph_only_finishes_and_reuses_successful_model_receipt(self):
        """批量拒绝后实际decide仅开放交接；恢复成功收据不再次发请求。"""
        from research_loop.storage.checkpoints import CheckpointStore
        test = self
        class FinishProvider:
            """一份显式替身响应；不连接DeepSeek。"""
            count = 0
            def send(self, messages, tools, **kwargs):
                """断言生产图所发请求已收缩，工具回复完整配对。"""
                self.count += 1
                test.assertEqual(self.count, 1)
                test.assertEqual({t['function']['name'] for t in tools}, {'finish_research'})
                validate_messages(messages)
                return {'id':'fixture-tail', 'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2},
                    'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':'',
                        'tool_calls':[{'id':'finish-once','type':'function','function':{'name':'finish_research','arguments':'{}'}}]}}]}
        provider = FinishProvider(); registry = ProbeRegistry()
        graph = build_graph(provider, registry, {}, self.db, checkpoint_store=CheckpointStore(self.db))
        original = pending_batch(self.state(), ['retrieve_evidence']*4)
        result = graph.invoke(deepcopy(original))
        self.assertEqual(result['structured_result']['status'], 'partial')
        calls = self.db.list_model_calls(self.tid); budget = self.db.budget_summary(self.tid)
        graph.invoke(deepcopy(original))
        self.assertEqual(provider.count, 1)
        self.assertEqual(registry.executed, ['retrieve_evidence','finish_research'])
        self.assertEqual(self.db.list_model_calls(self.tid), calls)
        self.assertEqual(self.db.budget_summary(self.tid), budget)
        self.assertEqual(calls[0]['status'], 'succeeded')

    def test_shared_tail_rejection_does_not_spend_local_slots(self):
        """共享额度先收尾时，批次全部回执配齐，仍给子任务保留交接。"""
        # 本例没有真实运行子角色，按当前角色自己的一次交接及父交付保留两槽。
        self.db.execute('UPDATE task_budgets SET tool_used=46 WHERE root_task_id=?', (self.tid,))
        registry = ProbeRegistry(); graph = build_graph(None, registry, {}, self.db)
        state = pending_batch(self.state(2), ['retrieve_evidence']*4)
        state.update(graph.nodes['execute_tools'].invoke(state))
        self.assertEqual(state['tool_actions'], 2); self.assertEqual(registry.executed, [])
        self.assertTrue(state['budget_tail']); validate_messages(state['messages'])
        pending_batch(state, ['finish_research'])
        state.update(graph.nodes['execute_tools'].invoke(state))
        self.assertTrue(state['structured_result'])
        self.assertEqual(self.db.budget_summary(self.tid)['tool_used'], 47)

    def test_terminal_in_middle_blocks_later_calls_and_pairs_messages(self):
        """终结在批次中间时，后续调用不执行，仍记录明确拒绝回执。"""
        registry = ProbeRegistry(); graph = build_graph(None, registry, {}, self.db)
        state = pending_batch(self.state(0), ['retrieve_evidence', 'finish_research', 'retrieve_evidence'])
        result = graph.nodes['execute_tools'].invoke(state)
        self.assertEqual(registry.executed, ['retrieve_evidence', 'finish_research'])
        self.assertEqual(result['tool_actions'], 2)
        self.assertEqual(result['actions'][-1]['observation']['data']['error_code'], 'AFTER_TERMINAL')
        validate_messages(result['messages'])

    def test_legacy_exhausted_checkpoint_keeps_reason_and_pairs_rejections(self):
        """旧任务已用尽额度时不虚增配额或冒充结构化交接。"""
        registry = ProbeRegistry(); graph = build_graph(None, registry, {}, self.db)
        state = pending_batch(self.state(20), ['finish_research'])
        result = graph.nodes['execute_tools'].invoke(state)
        self.assertTrue(result['finished']); self.assertEqual(result['tool_actions'], 20)
        self.assertIn('exhausted', result['final_reason'])
        self.assertFalse(registry.executed); validate_messages(result['messages'])

    def test_worker_handoff_persists_stop_reason_and_read_identity(self):
        """真实Worker子任务落库，已有原文与停止原因可读；空发现不伪造覆盖。"""
        from integration.test_collaboration_contract import FixturePDF
        path = self.root/'A.pdf'; path.write_bytes(b'%PDF-fixture')
        tid = self.tasks.create(TaskCreate(prompt='读取附件，不保存', idempotency_key=uuid4().hex,
            attachments=[{'input_ref': 'input-A', 'paper_id': 'paper-A', 'local_path': str(path)}])).task_id
        self.db.claim_task('test-owner', task_id=tid)
        token = self.db.get_task(tid)['lease_token']
        papers = PaperService(self.db, None, FixturePDF())
        class NoProvider:
            """本例没有模型请求。"""
            def close(self):
                """无需释放网络资源。"""
        class StoppedGraph:
            """替身仅复现已经取证后耗尽预算的状态，不替用户填研究发现。"""
            def invoke(self, state, config):
                """实际读服务写入合成页证据，再返回真实失败形状。"""
                data = papers.read(state['task_id'], 'input-A', attachments=state['attachments'])
                from research_loop.domain.tasks import ActionRecord
                action = ActionRecord(task_id=state['task_id'], sequence=1, tool_name='read_paper',
                    arguments={'target_ref':'input-A'}, status='succeeded', observation={'ok':True,'data':data})
                papers.db.save_action(action.model_dump(mode='json'))
                return {**state, 'evidence_ids': [p['evidence_id'] for p in data['pages']],
                        'finished': True, 'final_reason': 'tool action budget exhausted'}
        executor = TaskExecutor(self.settings, self.db, lambda _: NoProvider())
        with patch('research_loop.workers.executor.build_researcher_graph', return_value=StoppedGraph()):
            results = executor._run_research_batch(tid, [{'mode':'investigate', 'objective':'原文条件',
                'input_refs':['input-A']}], 'test-owner', token, self.settings, papers)
        result = results[0]
        self.assertEqual(result['status'], 'partial', result)
        self.assertEqual(result['findings'], []); self.assertEqual(result['paper_ids'], [])
        self.assertEqual(result['read_paper_ids'], ['paper-A'])
        self.assertIn('tool action budget exhausted', result['gaps'][0])
        stored = self.db.list_subtasks(tid)[0]['result']['result']
        self.assertEqual(stored['stop_reason'], result['stop_reason'])
        from research_loop.services.coverage import CoverageService
        parent_context = CoverageService(self.db).current_context(tid, {})
        self.assertEqual(parent_context['subtasks'][0]['stop_reason'], result['stop_reason'])
        self.assertEqual(parent_context['subtasks'][0]['read_paper_ids'], ['paper-A'])
        self.assertFalse(self.db.list_model_calls(tid))

    def test_stop_details_preserve_unknown_and_cancelled(self):
        """不同停止原因不统称原文缺证；结构化成功不增加虚假缺口。"""
        for reason in ('model response unknown: task family is blocked', 'cancelled before tool execution'):
            result = research_stop_details({'final_reason':reason}, 'partial', ['a','a'])
            self.assertIn(reason, result['gaps'][0]); self.assertEqual(result['read_paper_ids'], ['a'])
        self.assertEqual(research_stop_details({'structured_result':{'status':'completed'}}, 'completed', [])['gaps'], [])

    def sources(self):
        """用合成长原文制造历史压力及中间公式，不使用特定论文硬编码。"""
        path = self.root/'A.pdf'; path.write_bytes(b'%PDF-fixture')
        attachments = {'input-A': {'input_ref':'input-A','paper_id':'paper-A','local_path':str(path)}}
        # 本任务显式领取本例附件；不因工具上下文含有路径而绕过授权。
        self.db.grant_input(self.tid, 'input-A', self.tid)
        data = PaperService(self.db, None, LongPDF()).read(self.tid, 'input-A', attachments=attachments)
        messages = [{'role':'system','content':'system'}, {'role':'user','content':'compare'}]
        for n in range(8):
            messages += exchange('read_paper', {'target_ref':'input-A'}, data, f'history-{n}')
        targeted = deepcopy(data); targeted['pages'] = targeted['pages'][:1]
        messages += exchange('read_paper', {'target_ref':'input-A','start_page':1,'page_count':1}, targeted, 'targeted')
        return messages, targeted

    def test_targeted_page_stays_complete_before_old_history(self):
        """新补读页全文能装下时必须连续呈现，老页先摘录，原观察不修改。"""
        messages, data = self.sources(); original = deepcopy(messages)
        context = ContextService(self.db, None, self.settings)
        projected, _ = context._project_sources({'task_id':self.tid,'prompt':'comparison'}, messages, [], 30000)
        target = json.loads(projected[-1]['content'])['data']['pages'][0]
        self.assertFalse(target['context_excerpted'])
        self.assertEqual(target['text'], data['pages'][0]['text'])
        self.assertEqual(len(target['context_spans']), 1)
        self.assertLessEqual(estimate({'messages':projected,'tools':[]}), 30000)
        self.assertEqual(messages, original); validate_messages(projected)
        self.assertNotIn('context_readback', target)

    def test_oversize_target_has_exact_continuous_window_and_gap(self):
        """优先材料本身仍太长时明确展示容量缺口，不虚报全文可见。"""
        messages, data = self.sources()
        context = ContextService(self.db, None, self.settings)
        projected, _ = context._project_sources({'task_id':self.tid,'prompt':'language'}, messages, [], 9000)
        page = json.loads(projected[-1]['content'])['data']['pages'][0]
        self.assertTrue(page['context_excerpted']); self.assertIn('context_capacity_gap', page)
        self.assertEqual(len(page['context_spans']), 1)
        span = page['context_spans'][0]
        self.assertEqual(span['text'], data['pages'][0]['text'][span['start']:span['end']])
        self.assertLessEqual(estimate({'messages':projected,'tools':[]}), 9000)

    def test_targeted_read_precedes_large_retrieval_hits(self):
        """补读页和最新命中竞争容量时，先压命中，仍完整呈现显式补读。"""
        messages, data = self.sources()
        page = data['pages'][0]
        hits = [{**page, 'char_start': start, 'char_end': start+6000, 'text': page['text'][start:start+6000],
                 'source_sha256': data['content_sha256']} for start in (0, 100)]
        messages += exchange('retrieve_evidence', {'query':'language embeddings'}, {'hits': hits}, 'hits')
        context = ContextService(self.db, None, self.settings)
        projected, _ = context._project_sources({'task_id':self.tid,'prompt':'comparison'}, messages, [], 17000)
        target = json.loads(next(m['content'] for m in projected if m.get('tool_call_id')=='targeted'))['data']['pages'][0]
        self.assertEqual(target['text'], page['text']); self.assertFalse(target['context_excerpted'])
        result = json.loads(projected[-1]['content'])['data']
        self.assertTrue(any({**result.get('hits_defaults', {}), **hit}['context_excerpted'] for hit in result['hits']))
        self.assertLessEqual(estimate({'messages':projected,'tools':[]}), 17000)

    def test_necessary_state_over_capacity_stops_without_model(self):
        """连工具身份和必留消息都装不下时明确拒绝，不把截断伪装成功。"""
        tiny = self.settings.model_copy(update={
            'context_window_tokens':4096, 'context_output_reserve':3000, 'context_safety_tokens':500})
        self.tid = TaskService(self.db, tiny).create(TaskCreate(
            prompt='比较已有材料的有限容量案例', idempotency_key=uuid4().hex)).task_id
        messages, _ = self.sources()
        context = ContextService(self.db, None, tiny)
        with self.assertRaisesRegex(ValueError, 'CONTEXT_CAPACITY'):
            context.prepare_researcher({'task_id':self.tid,'prompt':'comparison'}, messages, [])
        self.assertFalse(self.db.list_model_calls(self.tid))

    def test_retrieval_query_selects_window_and_keeps_page_offset(self):
        """检索命中按实际工具query定位，片段的页内偏移不因投影改变。"""
        from research_loop.domain.context import project_source, source_identity
        text = 'plain text. '*600 + 'SPECIAL TARGET equation = x + y.\n' + 'tail text. '*600
        item = {'document_id':'doc','evidence_id':'page','page_number':3,'char_start':120,
                'char_end':120+len(text),'text':text,'source_sha256':'f'*64}
        observation = {'ok':True, 'data':{'hits':[item]}}
        identity = source_identity(item, observation['data'])
        result = project_source(observation, 'SPECIAL TARGET equation', 80, False, None, {identity}, 400)
        hit = result['data']['hits'][0]
        self.assertIn('SPECIAL TARGET', hit['text']); self.assertEqual(hit['char_start'], 120)
        self.assertEqual(hit['source_sha256'], 'f'*64)
        self.assertEqual(len(hit['context_spans']), 1)
