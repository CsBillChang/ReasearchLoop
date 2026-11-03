"""合成参数错误经过原生Worker反馈后更正，不使用真实模型或原件。"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

from experiments.mechanism_pilot.memory import execute_memory_case
from experiments.mechanism_pilot.recovery import BOUNDED_PARAMETER_RECOVERY, FailedActionGate
from experiments.mechanism_pilot.resources import runtime_namespace
from research_loop.config import Settings
from research_loop.providers.deepseek import DeepSeekProvider


def failed(identity='a1', *, arguments=None, observation=None):
    return {'action_id': identity, 'task_id': 'root', 'status': 'failed',
            'tool_name': 'finish_task', 'arguments': arguments or {'goals': []},
            'observation': observation or {'ok': False, 'kind': 'execution_error',
                'data': {'coverage_blocker': {'code': 'GOAL_SUPPORT_BINDING_MISSING'}},
                'message': 'goal missing subtask_ids', 'retryable': False}}


class FailedActionGateTests(unittest.TestCase):
    def test_legacy_still_stops_at_first_action(self):
        gate = FailedActionGate()
        self.assertEqual(gate.observe([failed()]), 'failed_action:finish_task')
        self.assertIsNotNone(gate.global_stop)
        self.assertIsNone(gate.case_stop)

    def test_allowlisted_error_preserves_first_and_ignores_readback(self):
        gate = FailedActionGate(BOUNDED_PARAMETER_RECOVERY)
        action = failed()
        self.assertIsNone(gate.observe([action]))
        self.assertIsNone(gate.observe([action]))
        self.assertEqual(len(gate.failed_actions), 1)
        self.assertEqual(gate.first_failed_action, action)
        self.assertEqual(gate.failed_actions[0]['classification'], 'recoverable_parameter')

    def test_two_distinct_corrections_then_local_stop(self):
        gate = FailedActionGate(BOUNDED_PARAMETER_RECOVERY)
        self.assertIsNone(gate.observe([failed('a1', arguments={'attempt': 1})]))
        self.assertIsNone(gate.observe([failed('a2', arguments={'attempt': 2})]))
        self.assertEqual(gate.observe([failed('a3', arguments={'attempt': 3})]),
                         'parameter_recovery_exhausted:finish_task')
        self.assertIsNone(gate.global_stop)
        self.assertEqual(len(gate.audit()['failed_action_history']), 3)

    def test_repeated_arguments_stop_only_case(self):
        gate = FailedActionGate(BOUNDED_PARAMETER_RECOVERY)
        gate.observe([failed()])
        self.assertEqual(gate.observe([failed('a2')]), 'repeated_parameter_failure:finish_task')
        self.assertIsNone(gate.global_stop)

    def test_permission_version_and_unknown_errors_remain_global(self):
        for observation in [
            {'kind': 'validation_error', 'retryable': True, 'message': 'source version mismatch'},
            {'kind': 'execution_error', 'message': 'not authorized'},
            {'kind': 'execution_error', 'message': 'unclassified failure'},
        ]:
            with self.subTest(observation=observation):
                gate = FailedActionGate(BOUNDED_PARAMETER_RECOVERY)
                self.assertEqual(gate.observe([failed(observation=observation)]), 'failed_action:finish_task')
                self.assertIsNone(gate.case_stop)

    def test_material_goal_parameter_feedback_can_recover(self):
        action = failed(observation={'ok': False, 'kind': 'execution_error', 'data': {},
            'message': '研究计划至少需要一个active user/legacy且min_papers>0的材料目标；纯历史回顾使用memory',
            'retryable': True})
        action['tool_name'] = 'set_research_plan'
        gate = FailedActionGate(BOUNDED_PARAMETER_RECOVERY)
        self.assertIsNone(gate.observe([action]))
        self.assertEqual(gate.failed_actions[0]['classification'], 'recoverable_parameter')

    def test_missing_hash_schema_feedback_is_not_source_version_failure(self):
        gate = FailedActionGate(BOUNDED_PARAMETER_RECOVERY)
        action = failed(observation={'ok': False, 'kind': 'validation_error', 'data': {},
            'message': '1 validation error for ResearchReviewInput\nValue error, target_id and content_hash must be supplied together',
            'retryable': True})
        action['tool_name'] = 'review_research'
        self.assertIsNone(gate.observe([action]))
        self.assertIsNone(gate.global_stop)

    def test_native_retryable_review_contract_feedback_can_recover(self):
        gate = FailedActionGate(BOUNDED_PARAMETER_RECOVERY)
        action = failed(observation={'ok': False, 'kind': 'execution_error', 'data': {},
            'message': 'existing target uses its stored snapshot; use revise_research for changes',
            'retryable': True})
        action['tool_name'] = 'review_research'
        self.assertIsNone(gate.observe([action]))
        action['action_id'] = 'actual-version-failure'
        action['observation']['message'] = 'source version mismatch'
        self.assertIsNotNone(gate.observe([action]))
        self.assertIsNotNone(gate.global_stop)

    def test_unknown_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'RECOVERY_POLICY'):
            FailedActionGate({**BOUNDED_PARAMETER_RECOVERY, 'max_recoverable_failures': 99})


class RecoveryWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'case'
        self.path = self.base/'input.json'
        self.path.write_text(json.dumps({'case_id': 'M1', 'prompt': '按本条数据说明保留6条。', 'history': []}))
        self.settings = Settings(project_root=self.root, data_dir=self.root/'data', output_dir=self.root/'output',
            database_url='sqlite:///'+str(self.root/'runtime.sqlite3'), deepseek_model='deepseek-flash',
            deepseek_api_key=None, deepseek_base_url='https://fixture.invalid',
            web_search_enabled=False, web_fetch_enabled=False, max_model_requests=8,
            reserve_model_requests=0, context_output_reserve=4096, model_max_output_tokens=4096,
            qdrant_url='http://127.0.0.1:16391', retrieval_namespace=runtime_namespace(self.root))

    def run_calls(self, calls):
        pending = iter(calls)
        self.requests = []

        def handler(request):
            body = json.loads(request.content)
            self.requests.append(body)
            name, arguments = next(pending)
            message = {'role': 'assistant', 'content': '', 'reasoning_content': '合成资格推理',
                'tool_calls': [{'id': 'synthetic-'+str(len(self.requests)), 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
            usage = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
            if body.get('stream'):
                message['tool_calls'][0]['index'] = 0
                event = {'id': 'synthetic-response', 'choices': [{'index': 0, 'delta': message,
                    'finish_reason': 'tool_calls'}], 'usage': usage}
                return httpx.Response(200, content='data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n',
                    headers={'content-type': 'text/event-stream'})
            return httpx.Response(200, json={'id': 'synthetic-response', 'usage': usage,
                'choices': [{'finish_reason': 'tool_calls', 'message': message}]})

        def factory(settings):
            return DeepSeekProvider('synthetic', settings.deepseek_model, settings.deepseek_base_url,
                settings.request_timeout_seconds, client=httpx.Client(transport=httpx.MockTransport(handler)))

        return execute_memory_case(self.path, self.root,
            expected_input_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            settings=self.settings, provider_factory=factory, recovery_policy=BOUNDED_PARAMETER_RECOVERY)

    @staticmethod
    def plan(min_papers=0):
        return {'task_kind': 'memory', 'goal': '说明当前数据', 'goals': [
            {'goal_id': 'derive', 'description': '说明保留条数', 'min_papers': min_papers}]}

    def test_native_feedback_allows_corrected_parameter_and_completion(self):
        result = self.run_calls([
            ('set_research_plan', self.plan(-1)),
            ('get_profile', {}),
            ('set_research_plan', self.plan()),
            ('finish_task', {'answer_markdown': '保留6条。', 'goals': [
                {'goal_id': 'derive', 'status': 'satisfied', 'reason': '按本条数据'}]}),
        ])
        self.assertIsNone(result['error'], result)
        self.assertIsNone(result['global_stop'], result)
        self.assertIsNone(result['case_stop'], result)
        self.assertEqual(result['view']['status'], 'completed')
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(len(result['failed_action_history']), 1)
        self.assertEqual(result['first_failed_action']['tool_name'], 'set_research_plan')
        feedback = [json.loads(m['content']) for m in self.requests[1]['messages'] if m['role'] == 'tool']
        self.assertTrue(any(o.get('kind') == 'validation_error' and not o['ok'] for o in feedback))
        self.assertEqual(result['strict_usage_summary']['total_tokens'], 60)
        self.assertEqual(result['real_model_calls'], 0)

    def test_parameter_exhaustion_retains_native_failure_without_global_stop(self):
        result = self.run_calls([('set_research_plan', self.plan(n)) for n in (-1, -2, -3)])
        self.assertIsNone(result['error'], result)
        self.assertIsNone(result['global_stop'], result)
        self.assertEqual(result['case_stop'], 'parameter_recovery_exhausted:set_research_plan')
        # 尚无可交付正文时业务自然为failed；实验层不能伪造partial成果。
        self.assertEqual(result['view']['status'], 'failed')
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(len(result['failed_action_history']), 3)
        self.assertEqual(result['strict_usage_summary']['total_tokens'], 45)
        self.assertTrue(any(r['sent_state'] == 'not_sent' for r in result['physical_receipts']))

    def test_native_material_goal_error_reaches_next_model_call(self):
        invalid = {**self.plan(), 'task_kind': 'search'}
        result = self.run_calls([
            ('set_research_plan', invalid),
            ('get_profile', {}),
            ('set_research_plan', self.plan()),
            ('finish_task', {'answer_markdown': '保留6条。', 'goals': [
                {'goal_id': 'derive', 'status': 'satisfied', 'reason': '按本条数据'}]}),
        ])
        self.assertEqual(result['view']['status'], 'completed')
        self.assertIsNone(result['global_stop'])
        self.assertEqual(len(self.requests), 4)
        failure = result['first_failed_action']['observation']
        self.assertEqual(failure['kind'], 'execution_error')
        feedback = [json.loads(m['content']) for m in self.requests[1]['messages'] if m['role'] == 'tool']
        self.assertIn(failure, feedback)

    def test_driver_keeps_next_independent_cell_after_local_stop(self):
        from experiments.mechanism_pilot.contract import RunContract
        from experiments.mechanism_pilot.runner import execute_prepared_case, launch_case
        from integration.test_mechanism_driver import DriverTests
        source = Path(__file__).resolve().parents[3]
        baseline_path = os.environ.get('MECHANISM_BASELINE_SOURCE')
        if not baseline_path:
            self.skipTest('explicit isolated baseline required')
        baseline = Path(baseline_path)
        if not (baseline/'backend/research_loop').is_dir():
            self.skipTest('explicit isolated baseline required')
        fixture = DriverTests(methodName='runTest')
        self.addCleanup(fixture.doCleanups)
        with patch.dict(os.environ, {'MECHANISM_BASELINE_SOURCE': str(baseline),
                                     'MECHANISM_CANDIDATE_SOURCE': str(source)}):
            fixture.setUp()
        fixture.manifest['recovery_policy'] = BOUNDED_PARAMETER_RECOVERY
        fixture.manifest_path.write_text(json.dumps(fixture.manifest))
        fixture.freeze = RunContract(fixture.manifest, fixture.auth).freeze_sha256
        payload = fixture.payload('B')
        responses = payload['substitute_responses']
        for index, raw in enumerate(responses):
            call = raw['choices'][0]['message']['tool_calls'][0]['function']
            call.update(name='set_research_plan', arguments=json.dumps(self.plan(-index - 1)))
        result = execute_prepared_case(payload)
        self.assertEqual(result['case_stop'], 'parameter_recovery_exhausted:set_research_plan')
        self.assertIsNone(result['global_stop'])
        self.assertIsNone(result['budget']['halt_reason'])
        next_payload = fixture.payload('A')
        process = launch_case(next_payload, fixture.batch/'A',
            deadline=fixture.deadline, interpreter=sys.executable)
        self.assertEqual(process['returncode'], 0,
            (process, (fixture.batch/'A/process.log').read_text()))
        self.assertTrue(process['cleanup_complete'], process)
        next_result = json.loads((Path(next_payload['settings']['project_root'])/'qualified-result.json').read_text())
        self.assertEqual(next_result['view']['status'], 'completed')
        self.assertIsNone(next_result['budget']['halt_reason'])
        self.assertEqual(next_result['budget']['physical_sent_requests'], 6)


if __name__ == '__main__':
    unittest.main()
