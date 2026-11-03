"""程序读取走原生工具链：交付资格、来源权限和核验版本不放宽。"""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx

from experiments.mechanism_pilot.reflection import execute_case, _seed_review
from experiments.mechanism_pilot.resources import FrozenPageResources, file_sha256
from integration import test_mechanism_resources as resource_fixtures
from integration.test_research_reflection import ResearchReflectionScenario, is_reviewer_request
from research_loop.domain.tasks import ActionRecord, FinishTaskInput, ResearchBrief, SubtaskSpec, TaskCreate
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.services.coverage import CoverageService
from research_loop.services.tasks import TaskService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.mysql import Database


class RetainCorrectScenario(ResearchReflectionScenario):
    """替身只返回工具决定；读文、独立审校及结束均由原业务执行。"""

    def handler(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        validate_messages(body['messages'])
        if is_reviewer_request(body):
            target = json.loads(body['messages'][1]['content'].split('\n', 1)[1])['target']
            pages = json.loads(body['messages'][2]['content'].split('\n', 1)[1])
            return self.response('finish_review', {'findings': [{
                'item_id': target['items'][0]['item_id'], 'status': 'supported',
                'basis': 'direct_statement', 'reason': '合成原文范围内支持，非真实语义评分。',
                'evidence_ids': [pages[0]['evidence_id']], 'source_quote': pages[0]['text'],
                'suggested_action': 'keep'}], 'gaps': []})
        done = [c['function']['name'] for m in body['messages'] for c in m.get('tool_calls', [])]
        if 'set_research_plan' not in done:
            return self.response('set_research_plan', {'goal': '核验合成条件', 'task_kind': 'survey',
                'goals': [{'goal_id': 'verify', 'description': '核验合成条件',
                    'min_papers': 1, 'requires_fulltext': True}]})
        observations = [json.loads(m['content']) for m in body['messages'] if m['role'] == 'tool']
        reviewed = next(o['data'] for o in observations if o.get('data', {}).get('review_id'))
        current = reviewed['target']
        return self.response('finish_task', {'reflection_target_id': current['target_id'],
            'reflection_content_hash': current['content_hash'],
            'goals': [{'goal_id': 'verify', 'status': 'satisfied',
                'subtask_ids': [], 'reason': '根任务有实际原文读取及核验结果'}]})

    def factory(self, settings):
        return DeepSeekProvider('fixture', settings.deepseek_model, settings.deepseek_base_url,
            client=httpx.Client(transport=httpx.MockTransport(self.handler)))


class ReadChainTests(unittest.TestCase):
    setUp = resource_fixtures.FrozenResourceTests.setUp
    material = resource_fixtures.FrozenResourceTests.material

    def prepare(self):
        case, entry, _ = self.material()
        runtime = self.root/'runtime'; runtime.mkdir()
        settings = resource_fixtures.settings_for(runtime)
        resources = FrozenPageResources(runtime, settings, case, entry)
        db = Database(settings.database_url); db.ensure_schema()
        with patch('research_loop.config.get_settings', return_value=settings):
            task = TaskService(db, settings).create(TaskCreate(
                prompt='只读取附件 frozen-R01 的物理第2页，核验合成稿，不保存。',
                attachments=resources.attachments, save_policy='deny')).task_id
        return case, settings, resources, db, task

    @staticmethod
    def plan(db, task):
        coverage = CoverageService(db)
        value = coverage.prepare_plan(task, ResearchBrief(goal='核验合成条件', task_kind='survey',
            goals=[{'goal_id': 'verify', 'description': '核验合成条件',
                'min_papers': 1, 'requires_fulltext': True}]))
        sequence = len(db.list_actions(task))+1
        db.save_action(ActionRecord(task_id=task, sequence=sequence, tool_name='set_research_plan',
            arguments={}, status='succeeded', observation={
                'ok': True, 'kind': 'success', 'data': {'research_plan': value}}).model_dump(mode='json'))
        return coverage

    def test_native_read_state_is_preserved_before_programmatic_review(self):
        case, settings, resources, db, task = self.prepare()
        bound = resources.bind(task, db)
        origin = _seed_review(db, task, case, db.get_task(task)['prompt'], resources.attachments,
            bound['evidence_ids'], settings, read_state=bound['programmatic_read_state'])
        state = CheckpointStore(db).load(task)
        self.assertEqual(state['action_sequence'], 1)
        self.assertEqual(state['actions'][0]['tool_name'], 'read_paper')
        self.assertEqual(state['observations'][0]['tool'], 'read_paper')
        calls = [c for m in state['messages'] for c in m.get('tool_calls', [])]
        self.assertEqual([c['function']['name'] for c in calls], ['read_paper', 'review_research'])
        self.assertEqual(calls[-1]['id'], origin['tool_call_id'])
        validate_messages(state['messages'][:-1])
        self.assertFalse(db.fetchall('SELECT * FROM model_calls'))
        self.assertEqual(self.plan(db, task).finish(task, FinishTaskInput(answer_markdown=case['model_input']['draft'],
            goals=[{'goal_id': 'verify', 'status': 'satisfied', 'reason': '真实读取'}]))['coverage']['paper_ids'],
            ['synthetic-paper'])

    def test_only_child_source_requires_binding_and_still_checks_authority(self):
        case, settings, resources, db, task = self.prepare()
        coverage = self.plan(db, task)
        spec = SubtaskSpec(objective='核验合成条件', mode='investigate', goal_ids=['verify'],
            input_refs=['frozen-R01']).model_dump(mode='json')
        cid = db.create_subtask(task, task, 0, spec['objective'], spec, uuid4().hex)['subtask_id']
        db.grant_input(cid, 'frozen-R01', task)
        db.execute('UPDATE tasks SET attachments_json=? WHERE task_id=?',
            (json.dumps(resources.attachments), cid))
        bound = resources.bind(cid, db)
        db.save_subtask_result(cid, 'completed', {'findings': ['合成材料发现。'], 'paper_ids': [],
            'evidence_ids': bound['evidence_ids'], 'gaps': []})
        payload = {'answer_markdown': case['model_input']['draft'], 'goals': [{
            'goal_id': 'verify', 'status': 'satisfied', 'reason': '子成果支持'}]}
        with self.assertRaisesRegex(ValueError, 'missing subtask_ids'):
            coverage.finish(task, FinishTaskInput.model_validate(payload))
        payload['goals'][0]['subtask_ids'] = [cid]
        self.assertEqual(coverage.finish(task, FinishTaskInput.model_validate(payload))['coverage']['paper_ids'],
            ['synthetic-paper'])
        db.execute('DELETE FROM task_evidence WHERE task_id=?', (cid,))
        self.assertEqual(coverage.source_coverage(cid), [])
        with self.assertRaisesRegex(ValueError, 'lacks qualified papers'):
            coverage.finish(task, FinishTaskInput.model_validate(payload))

    def test_source_version_drift_does_not_reuse_programmatic_receipt(self):
        case, settings, resources, db, task = self.prepare()
        bound = resources.bind(task, db)
        coverage = self.plan(db, task)
        db.execute('UPDATE documents SET content_sha256=? WHERE document_id=?',
            ('0'*64, bound['material_identity']['runtime_document_id']))
        self.assertEqual(coverage.source_coverage(task), [])
        with self.assertRaisesRegex(ValueError, 'lacks qualified papers'):
            coverage.finish(task, FinishTaskInput(answer_markdown=case['model_input']['draft'], goals=[{
                'goal_id': 'verify', 'status': 'satisfied', 'reason': '旧来源不能通过'}]))

    def test_worker_delivers_current_reviewed_body_after_native_programmatic_read(self):
        case, entry, _ = self.material()
        path = self.root/'case.json'; path.write_text(json.dumps(case, ensure_ascii=False))
        runtime = self.root/'worker'
        scenario = RetainCorrectScenario()
        result = execute_case(path, runtime, expected_input_sha256=file_sha256(path),
            settings=resource_fixtures.settings_for(runtime), provider_factory=scenario.factory,
            resource_factory=lambda root, settings, frozen: FrozenPageResources(root, settings, frozen, entry))
        self.assertIsNone(result['error'], result)
        self.assertIsNone(result['global_stop'], result.get('global_stop'))
        self.assertEqual(result['view']['status'], 'completed', result)
        self.assertEqual(result['delivery_audit']['answer_body_markdown'], case['model_input']['draft'])
        self.assertTrue(result['delivery_audit']['finish_bound_current'])
        self.assertTrue(result['delivery_audit']['final_body_matches_current'])
        self.assertEqual(result['actions'][0]['tool_name'], 'read_paper')
        db = Database('sqlite:///'+str(runtime/'runtime.sqlite3'))
        self.assertEqual(db.fetchone('SELECT COUNT(*) AS n FROM model_calls')['n'], len(scenario.requests))
        self.assertEqual(result['real_model_calls'], 0)
        args = deepcopy(result['actions'][-1]['arguments'])
        args['reflection_content_hash'] = '0'*64
        db.update_task(result['task_id'], status='running')
        with self.assertRaises(ValueError):
            CoverageService(db).finish(result['task_id'], FinishTaskInput.model_validate(args))
