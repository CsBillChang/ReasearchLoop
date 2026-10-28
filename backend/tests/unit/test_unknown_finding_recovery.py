"""未知发现ID的窄错误恢复；只执行工具与真实图节点，不请求模型。"""
from copy import deepcopy
from pathlib import Path
import json
import socket
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx

from research_loop.agents.graph import build_graph, initial_state
from research_loop.agents.researcher import progress_spec
from research_loop.domain.research_progress import ProgressFinding, ResearchProgressInput
from research_loop.domain.tasks import ActionRecord, SavePolicy, TaskCreate
from research_loop.services.papers import PaperService
from research_loop.services.research_progress import (
    ResearchProgressService, UnknownFindingIdError, continuation_fingerprint,
)
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.tools.base import ToolContext
from research_loop.tools.registry import ToolRegistry
from integration.test_collaboration_contract import FixturePDF


class NoModel:
    """意外进入决策节点即失败；不返回合成模型响应。"""

    def __init__(self):
        self.calls = 0

    def send(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError('model execution is outside this tool-node regression')

    send_stream = send


class UnknownFindingRecoveryTests(unittest.TestCase):
    """独立SQLite与合成页验证拒绝、反馈、提交及权限版本边界。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='unknown-finding-recovery-')
        self.root = Path(self.temp.name)
        self.network_calls = 0
        def deny_network(*args, **kwargs):
            self.network_calls += 1
            raise AssertionError('network execution is outside this regression')
        for target in [patch.object(socket.socket, 'connect', deny_network),
                       patch.object(socket, 'create_connection', deny_network),
                       patch.object(httpx.HTTPTransport, 'handle_request', deny_network)]:
            target.start()
            self.addCleanup(target.stop)
        self.db = Database('sqlite:///' + str(self.root / 'isolated.sqlite3'))
        self.db.ensure_schema()
        self.tid, self.eid = self.task_with_evidence()
        self.service = ResearchProgressService(self.db)
        self.registry = ToolRegistry([progress_spec()])
        self.model = NoModel()

    def tearDown(self):
        self.assertEqual(self.network_calls, 0)
        self.assertEqual(self.model.calls, 0)
        self.temp.cleanup()

    def task_with_evidence(self):
        tid = TaskService(self.db).create(TaskCreate(
            prompt='解释合成材料的机制，不保存', idempotency_key=uuid4().hex)).task_id
        paper = 'fixture-' + uuid4().hex
        self.db.grant_paper(tid, paper, tid)
        self.db.save_paper(paper, {'paper_id': paper, 'source': 'arxiv',
            'title': 'synthetic', 'pdf_url': 'https://example.invalid/fixture.pdf'})
        pages = PaperService(self.db, None, FixturePDF()).read(tid, paper)['pages']
        return tid, pages[0]['evidence_id']

    def context(self, tid=None):
        tid = tid or self.tid
        return ToolContext(task_id=tid, save_policy=SavePolicy.deny, role='researcher',
            root_task_id=tid, services={'papers': SimpleNamespace(db=self.db)})

    def arguments(self, *, finding_id=None, evidence_id=None, text='合成页说明语言嵌入'):
        finding = {'text': text, 'evidence_ids': [evidence_id or self.eid]}
        if finding_id is not None:
            finding['finding_id'] = finding_id
        return {'findings': [finding]}

    def execute(self, args, tid=None):
        return self.registry.execute('record_research_progress', args, self.context(tid))

    def commit(self, args=None, tid=None):
        """按intent和服务提交候选，保留生产原子提交校验。"""
        tid = tid or self.tid
        args = args or self.arguments()
        result = self.execute(args, tid)
        self.assertTrue(result.ok, result)
        sequence = max([a['sequence_no'] for a in self.db.list_actions(tid)] or [0]) + 1
        intent = ActionRecord(task_id=tid, sequence=sequence,
            tool_name='record_research_progress', arguments=args)
        self.db.save_action(intent.model_dump(mode='json'))
        intent.status = 'succeeded'
        intent.observation = result.model_dump(mode='json')
        ResearchProgressService(self.db).commit_action(intent.model_dump(mode='json'), None)
        return ResearchProgressService(self.db).snapshot(tid)

    def handoff_state(self, args):
        """显式构造已请求交接状态，不模拟模型为何进入该状态。"""
        state = initial_state(self.tid, '解释合成材料', 'deny')
        state.update(role='researcher', model_requests=1,
            research_handoff={'phase': 'requested',
                'continuation_fingerprint': continuation_fingerprint(self.service.snapshot(self.tid))})
        return self.queue(state, args)

    @staticmethod
    def queue(state, args):
        state = deepcopy(state)
        call_id = 'synthetic-' + uuid4().hex
        state['messages'].append({'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': call_id, 'type': 'function', 'function': {
                'name': 'record_research_progress', 'arguments': json.dumps(args, ensure_ascii=False)}}]})
        state.update(pending_tool_scope=['record_research_progress'],
            pending_tool_calls=[{'call_id': call_id, 'name': 'record_research_progress',
                                 'arguments': deepcopy(args)}])
        return state

    def node(self, state):
        graph = build_graph(self.model, self.registry,
            {'papers': SimpleNamespace(db=self.db)}, db=self.db)
        return graph.nodes['execute_tools'].bound.invoke(state)

    def test_unknown_new_id_is_rejected_without_mutation_and_has_narrow_type(self):
        before = self.service.snapshot(self.tid)
        args = self.arguments(finding_id='model-invented-id')
        with self.assertRaises(UnknownFindingIdError) as caught:
            self.service.prepare(self.tid, ResearchProgressInput.model_validate(args))
        self.assertIsInstance(caught.exception, ValueError)
        result = self.execute(args)
        self.assertFalse(result.ok)
        self.assertEqual(result.kind, 'validation_error')
        self.assertTrue(result.retryable)
        self.assertEqual(result.data['error_code'], 'UNKNOWN_FINDING_ID')
        self.assertEqual(self.service.snapshot(self.tid), before)

    def test_one_feedback_then_explicit_legal_resubmit_commits_unreviewed(self):
        first_state = self.handoff_state(self.arguments(finding_id='invented-id'))
        first = self.node(first_state)
        self.assertIsNot(first.get('finished'), True)
        self.assertEqual(first['research_handoff']['phase'], 'requested')
        self.assertIn('tool_error', first['research_handoff'])
        self.assertEqual(self.service.snapshot(self.tid)['findings'], [])
        corrected_state = {**first_state, **first}
        corrected_state['model_requests'] += 1
        corrected = self.node(self.queue(corrected_state, self.arguments()))
        self.assertIsNot(corrected.get('finished'), True)
        self.assertEqual(corrected['research_handoff']['phase'], 'progress_recorded')
        committed = self.service.snapshot(self.tid)
        self.assertEqual(committed['revision'], 1)
        self.assertEqual(len(committed['findings']), 1)
        self.assertEqual(committed['findings'][0]['verification_status'], 'unreviewed')

    def test_same_invalid_resubmit_still_stops(self):
        args = self.arguments(finding_id='invented-id')
        state = self.handoff_state(args)
        first = self.node(state)
        retry = {**state, **first}
        retry['model_requests'] += 1
        second = self.node(self.queue(retry, args))
        self.assertTrue(second['finished'])
        self.assertIn('no progress', second['final_reason'])
        self.assertEqual(self.service.snapshot(self.tid)['findings'], [])

    def test_changed_invalid_id_does_not_open_another_feedback_slot(self):
        state = self.handoff_state(self.arguments(finding_id='invented-one'))
        first = self.node(state)
        retry = {**state, **first}
        retry['model_requests'] += 1
        second = self.node(self.queue(retry, self.arguments(finding_id='invented-two')))
        self.assertTrue(second['finished'])
        self.assertEqual(second['final_reason'], 'research readback made no progress')
        self.assertEqual(self.service.snapshot(self.tid)['findings'], [])

    def test_valid_existing_id_updates_in_place_and_keeps_evidence(self):
        first = self.commit()
        old = first['findings'][0]
        args = {'base_revision': first['revision'], 'findings': [{
            'finding_id': old['finding_id'], 'text': '合成页进一步说明机制', 'evidence_ids': []}]}
        current = self.commit(args)
        self.assertEqual(current['revision'], 2)
        self.assertEqual(len(current['findings']), 1)
        self.assertEqual(current['findings'][0]['finding_id'], old['finding_id'])
        self.assertEqual(current['findings'][0]['sources'], old['sources'])
        self.assertEqual(current['findings'][0]['verification_status'], 'unreviewed')

    def test_foreign_finding_id_is_not_adopted_by_current_task(self):
        other_tid, other_eid = self.task_with_evidence()
        other = self.commit(self.arguments(evidence_id=other_eid), other_tid)
        before = self.service.snapshot(self.tid)
        result = self.execute(self.arguments(finding_id=other['findings'][0]['finding_id']))
        self.assertFalse(result.ok)
        self.assertEqual(result.data.get('error_code'), 'UNKNOWN_FINDING_ID')
        self.assertEqual(self.service.snapshot(self.tid), before)
        self.assertEqual(ResearchProgressService(self.db).snapshot(other_tid), other)

    def test_foreign_evidence_is_still_nonretryable_permission_error(self):
        _, foreign_eid = self.task_with_evidence()
        before = self.service.snapshot(self.tid)
        result = self.execute(self.arguments(finding_id='invented-id', evidence_id=foreign_eid))
        self.assertFalse(result.ok)
        self.assertEqual(result.kind, 'execution_error')
        self.assertFalse(result.retryable)
        self.assertNotIn('error_code', result.data)
        self.assertEqual(self.service.snapshot(self.tid), before)

    def test_stale_base_revision_does_not_use_unknown_id_recovery(self):
        self.commit()
        before = self.service.snapshot(self.tid)
        args = {**self.arguments(finding_id='invented-id'), 'base_revision': 0}
        result = self.execute(args)
        self.assertFalse(result.ok)
        self.assertEqual(result.kind, 'execution_error')
        self.assertTrue(result.retryable)
        self.assertNotIn('error_code', result.data)
        self.assertIn('revision stale', result.message)
        self.assertEqual(self.service.snapshot(self.tid), before)

    def test_source_version_change_still_rejects_atomic_commit(self):
        args = self.arguments()
        result = self.execute(args)
        self.assertTrue(result.ok)
        action = ActionRecord(task_id=self.tid, sequence=1,
            tool_name='record_research_progress', arguments=args)
        self.db.save_action(action.model_dump(mode='json'))
        action.status = 'succeeded'
        action.observation = result.model_dump(mode='json')
        self.db.execute('UPDATE page_evidence SET text=? WHERE evidence_id=?', ('changed source', self.eid))
        with self.assertRaisesRegex(ValueError, 'source version changed'):
            self.service.commit_action(action.model_dump(mode='json'), None)
        self.assertEqual(self.service.snapshot(self.tid)['findings'], [])

    def test_other_handler_errors_keep_original_classification(self):
        for error, retryable in [(ValueError('other validation domain'), True),
                                 (PermissionError('denied'), False),
                                 (sqlite3.OperationalError('synthetic database failure'), False)]:
            with self.subTest(error=type(error).__name__), patch.object(
                    ResearchProgressService, 'prepare', side_effect=error):
                result = self.execute(self.arguments())
                self.assertFalse(result.ok)
                self.assertEqual(result.kind, 'execution_error')
                self.assertEqual(result.retryable, retryable)
                self.assertNotIn('error_code', result.data)

    def test_other_retryable_execution_error_does_not_gain_feedback_slot(self):
        state = self.handoff_state(self.arguments())
        with patch.object(ResearchProgressService, 'prepare', side_effect=ValueError('other domain error')):
            update = self.node(state)
        self.assertTrue(update['finished'])
        self.assertEqual(update['final_reason'], 'research readback made no progress')
        self.assertNotIn('tool_error', update['research_handoff'])
        self.assertEqual(self.service.snapshot(self.tid)['findings'], [])


if __name__ == '__main__':
    unittest.main()
