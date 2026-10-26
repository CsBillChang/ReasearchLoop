"""六稿真实入口的合成资格；供应商与资源均显式替换且 socket 全拒绝。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from integration.test_reflection_baseline_runtime import BaselineScenario
from integration.test_research_reflection import ScopedPDF
from research_loop.config import Settings
from research_loop.providers.deepseek import DeepSeekProvider, RequestNotSent
from research_loop.services.papers import PaperService
from research_loop.storage.mysql import Database
from scripts.reflection_baseline_runtime import deny_socket_connections, business_source_hashes
from scripts.reflection_pilot_real import execute_case, run_pilot
from unit.test_reflection_pilot_protocol import validate_thinking_history


class PilotScenario(BaselineScenario):
    """模拟带推理字段的响应，并在HTTP替身边界严格核对下一轮回传。"""

    def __init__(self, mode='revision'):
        self.response_reasoning = {}
        self.programmatic_ids = set()
        super().__init__(mode)

    def response(self, name, arguments):
        response = super().response(name, arguments)
        raw = response.json()
        message = raw['choices'][0]['message']
        identity = message['tool_calls'][0]['id']
        # 明确的合成响应哨兵，验证真实解析/持久化路径，不使用私有模型推理。
        message['reasoning_content'] = '合成响应推理哨兵：' + identity + '\n逐字保留。'
        self.response_reasoning[identity] = message['reasoning_content']
        return httpx.Response(200, json=raw)

    def handler(self, request):
        body = json.loads(request.content)
        validate_thinking_history(body, programmatic_ids=self.programmatic_ids,
            expected_reasoning=self.response_reasoning)
        for message in body['messages']:
            for call in message.get('tool_calls', []):
                if call['id'] not in self.programmatic_ids:
                    assert call['id'] in self.response_reasoning, 'model response fixture provenance missing'
        response = super().handler(request)
        function = response.json()['choices'][0]['message']['tool_calls'][0]['function']
        if function['name'] == 'read_paper':
            args = json.loads(function['arguments'])
            args.update(start_page=2, page_count=1)
            return self.response('read_paper', args)
        return response

    def factory(self, settings):
        # 空字段许可来自本案原生事件，不能只凭模型消息中的ID前缀获得。
        rows = Database(settings.database_url).fetchall(
            'SELECT payload_json FROM task_events WHERE event_type=?',
            ('reflection_pilot_programmatic_review',))
        origins = [json.loads(row['payload_json']) for row in rows]
        self.programmatic_ids.update(origin['tool_call_id'] for origin in origins
            if origin['model_generated'] is False and origin['model_receipt_created'] is False)
        return DeepSeekProvider('fixture', settings.deepseek_model, settings.deepseek_base_url,
            settings.request_timeout_seconds, client=httpx.Client(transport=httpx.MockTransport(self.handler)))


class ToyResources:
    """仅用于合成入口测试；所有资源报告明确说明非真实向量资格。"""

    def __init__(self, root, settings, case):
        self.settings = settings
        source = root/'input.pdf'
        source.write_bytes(b'%PDF-pilot-synthetic')
        self.attachments = [{'input_ref': 'input-A', 'paper_id': 'paper-A', 'local_path': str(source)}]
        self.bound = None

    @staticmethod
    def pdf_factory(settings, files):
        return ScopedPDF()

    def bind(self, task_id, db):
        observed = PaperService(db, None, ScopedPDF()).read(task_id, 'input-A', 2, 1,
            {'input-A': self.attachments[0]})
        self.bound = {'evidence_ids': [page['evidence_id'] for page in observed['pages']],
            'vector_status': 'ready', 'read_observation': observed, 'kind': 'synthetic_resource_substitute'}
        return self.bound

    @staticmethod
    def retrieval_factory(db, settings):
        class NeverRetrieve:
            def search(self, *args, **kwargs):
                raise AssertionError('synthetic scenario does not request vector retrieval')
        return NeverRetrieve()

    def report(self):
        return {'kind': 'synthetic_resource_substitute', 'bound': self.bound}


class StopOnUnknown:
    """模拟共享批次停止门；并非真实预算器或账号调用。"""

    def __init__(self, scenario):
        self.scenario = scenario
        self.reason = None

    def factory(self, settings):
        inner = self.scenario.factory(settings)
        gate = self

        class Guarded:
            def send(self, *args, **kwargs):
                if gate.reason:
                    raise RequestNotSent(gate.reason)
                try:
                    return inner.send(*args, **kwargs)
                except Exception:
                    gate.reason = 'synthetic_unknown_call'
                    raise

            def close(self):
                inner.close()
                inner.client.close()
        return Guarded()


class ReflectionPilotRealEntryTests(unittest.TestCase):
    """验证程序首轮冻结稿、原生修订交付和整批停止；真实发送为零。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = Path(__file__).resolve().parents[3]
        self.before_hashes = business_source_hashes(self.repo)
        self.addCleanup(lambda: self.assertEqual(business_source_hashes(self.repo), self.before_hashes))
        self.approval = {'approved': True, 'approval_id': 'synthetic-pilot-approval',
                         'case_ids': [f'R{number:02}' for number in range(1, 7)]}

    def material(self, case_id='R01', draft=None, question='方法是否使用语言前缀？'):
        source = 'The method uses language prefix embeddings.'
        case = {'case_id': case_id, 'model_input': {'question': question,
            'draft': draft or BaselineScenario().before,
            'documents': [{'document_id': 'synthetic-document', 'paper_id': 'paper-A',
                'allowed_pages': [2], 'initial_pages': [{'physical_page': 2, 'text': source,
                    'page_text_sha256': hashlib.sha256(source.encode()).hexdigest()}]}]}}
        path = self.root/(case_id + '.json')
        path.write_text(json.dumps(case, ensure_ascii=False))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def settings(root, case=None):
        return Settings(project_root=root, data_dir=root/'data', output_dir=root/'output',
            database_url='sqlite:///' + str(root/'runtime.sqlite3'), deepseek_model='deepseek-flash',
            deepseek_base_url='https://api.deepseek.com', request_timeout_seconds=120,
            context_output_reserve=65536, execution_budget_mode='observe',
            retrieval_enabled=True, qdrant_url='http://127.0.0.1:1',
            retrieval_namespace='synthetic_pilot', web_search_enabled=False, web_fetch_enabled=False)

    def run_case(self, mode='revision'):
        scenario = PilotScenario(mode)
        path, digest = self.material(draft=scenario.before)
        root = self.root/'run'
        settings = self.settings(root)
        with deny_socket_connections(), patch('research_loop.config.Settings.from_env',
                side_effect=AssertionError('default settings forbidden')):
            result = execute_case(path, root, expected_input_sha256=digest, settings=settings,
                provider_factory=scenario.factory, resource_factory=ToyResources,
                approval=self.approval, execution_kind='substitute')
        return result, scenario

    def test_first_reviewer_gets_exact_frozen_draft_before_supervisor(self):
        result, scenario = self.run_case()
        self.assertIsNone(result['error'], result['error'])
        self.assertEqual(result['view']['status'], 'completed', result['view'])
        self.assertTrue(result['first_review_original_preserved'])
        self.assertEqual(result['view']['result']['answer_body_markdown'], scenario.after)
        self.assertIn('finish_review', {t['function']['name'] for t in scenario.requests[0]['tools']})
        context = json.loads(scenario.requests[0]['messages'][1]['content'].split('\n', 1)[1])
        self.assertEqual(context['target']['items'][0]['text'], scenario.before)
        self.assertFalse(result['programmatic_review_origin']['model_generated'])
        self.assertFalse(result['programmatic_review_origin']['model_receipt_created'])
        self.assertEqual(result['actions'][0]['tool_name'], 'review_research')
        self.assertEqual(result['actions'][0]['arguments']['answer_markdown'], scenario.before)
        self.assertEqual(len(result['physical_receipts']), len(scenario.requests))
        self.assertEqual(result['real_model_calls'], 0)
        self.assertTrue(all(r['model'] == 'deepseek-flash' and r['max_tokens'] == 65536
                            and 'thinking' not in r for r in scenario.requests))
        patches = [a for a in result['actions'] if a['tool_name'] == 'revise_research']
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0]['status'], 'succeeded')

    def test_thinking_fields_reach_http_unchanged_and_match_native_receipts(self):
        result, scenario = self.run_case()
        self.assertIsNone(result['error'], result['error'])
        self.assertEqual(result['view']['status'], 'completed')
        observed = [json.loads(row['request_json'])['messages'] for row in result['physical_receipts']]
        wire = [body['messages'] for body in scenario.requests]
        canonical = lambda messages: json.dumps(messages, sort_keys=True, ensure_ascii=False)
        self.assertCountEqual([canonical(m) for m in observed], [canonical(m) for m in wire])
        echoes = []
        for body in scenario.requests:
            self.assertNotIn('thinking', body)
            self.assertEqual(body['model'], 'deepseek-flash')
            self.assertEqual(body['max_tokens'], 65536)
            for message in body['messages']:
                for call in message.get('tool_calls', []):
                    if call['id'].startswith('programmatic-review-'):
                        self.assertEqual(message['reasoning_content'], '')
                    else:
                        self.assertEqual(message['reasoning_content'], scenario.response_reasoning[call['id']])
                        echoes.append(call['id'])
        self.assertTrue(echoes, 'must exercise at least one subsequent model-history round trip')
        checkpoint = Database(self.settings(self.root/'run').database_url).get_checkpoint(result['task_id'])
        checkpoint_echoes = []
        for message in checkpoint['messages']:
            for call in message.get('tool_calls', []):
                if call['id'] in scenario.response_reasoning:
                    self.assertEqual(message['reasoning_content'], scenario.response_reasoning[call['id']])
                    checkpoint_echoes.append(call['id'])
        self.assertTrue(checkpoint_echoes, 'must check persisted model reasoning after graph execution')
        self.assertFalse(result['programmatic_review_origin']['model_generated'])
        self.assertFalse(result['programmatic_review_origin']['model_receipt_created'])

    def test_explicit_http400_stops_batch_without_retry_or_zero_filled_usage(self):
        class SupervisorRejected(PilotScenario):
            def handler(self, request):
                response = super().handler(request)
                body = json.loads(request.content)
                if not any(t['function']['name'] == 'finish_review' for t in body['tools']):
                    return httpx.Response(400, json={'error': {'message': 'synthetic protocol rejection'}})
                return response

        scenario = SupervisorRejected()
        first, first_digest = self.material('R01')
        second, second_digest = self.material('R02')
        target = self.root/'rejected-batch'
        with deny_socket_connections():
            report = run_pilot([
                {'case_path': first, 'expected_input_sha256': first_digest},
                {'case_path': second, 'expected_input_sha256': second_digest}], target,
                settings_factory=self.settings, provider_factory=scenario.factory,
                resource_factory=ToyResources, approval=self.approval, execution_kind='substitute')
        self.assertEqual(report['created'], 1)
        self.assertEqual(report['uncreated'], 1)
        self.assertEqual(report['global_stop'], 'provider_response_error')
        self.assertEqual(len(scenario.requests), 2)
        self.assertFalse((target/'cases'/'R02').exists())
        rows = report['results'][0]['physical_receipts']
        self.assertEqual([row['status'] for row in rows], ['succeeded', 'rejected'])
        self.assertEqual(json.loads(rows[-1].get('usage_json') or '{}'), {})
        self.assertNotEqual(json.loads(rows[-1].get('usage_json') or '{}').get('total_tokens'), 0)

    def test_correct_case_remains_unchanged_after_programmatic_review(self):
        result, scenario = self.run_case('correct')
        self.assertIsNone(result['error'])
        self.assertEqual(result['view']['result']['answer_body_markdown'], scenario.before)
        self.assertFalse([a for a in result['actions'] if a['tool_name'] == 'revise_research'])

    def test_missing_evidence_can_finish_partial_without_replacing_draft(self):
        result, scenario = self.run_case('missing')
        self.assertIsNone(result['error'])
        self.assertEqual(result['view']['status'], 'partial')
        self.assertEqual(result['view']['result']['answer_body_markdown'], scenario.before)

    def test_case_directory_cannot_be_reused_for_a_second_attempt(self):
        result, scenario = self.run_case()
        requests = len(scenario.requests)
        path = self.root/'R01.json'
        with self.assertRaises(FileExistsError):
            execute_case(path, self.root/'run', expected_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                settings=self.settings(self.root/'run'), provider_factory=scenario.factory,
                resource_factory=ToyResources, approval=self.approval, execution_kind='substitute')
        self.assertEqual(len(scenario.requests), requests)

    def test_input_drift_and_unapproved_cases_fail_before_resources(self):
        path, digest = self.material()
        for changed_digest, approval, expected in [('0'*64, self.approval, ValueError),
                (digest, {**self.approval, 'approved': False}, PermissionError),
                (digest, {**self.approval, 'case_ids': ['F01']}, PermissionError)]:
            with self.subTest(approval=approval, digest=changed_digest):
                target = self.root/'forbidden'
                with self.assertRaises(expected):
                    execute_case(path, target, expected_input_sha256=changed_digest,
                        settings=self.settings(target), provider_factory=lambda s: self.fail('provider constructed'),
                        resource_factory=lambda *a: self.fail('resources constructed'), approval=approval,
                        execution_kind='substitute')
                self.assertFalse(target.exists())

    def test_labels_never_enter_model_input_or_execution_file(self):
        path, _ = self.material()
        raw = json.loads(path.read_text())
        raw['model_input']['expected'] = 'forbidden label marker'
        path.write_text(json.dumps(raw))
        target = self.root/'forbidden'
        with self.assertRaisesRegex(ValueError, 'LABEL_IN_MODEL_INPUT'):
            execute_case(path, target, expected_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                settings=self.settings(target), provider_factory=lambda s: None, resource_factory=lambda *a: None,
                approval=self.approval, execution_kind='substitute')
        labels = self.root/'labels'
        labels.mkdir()
        wrong = labels/'R01.json'
        wrong.write_text('{}')
        with self.assertRaisesRegex(PermissionError, 'LABEL_FILE_FORBIDDEN'):
            execute_case(wrong, target, expected_input_sha256='0'*64, settings=self.settings(target),
                provider_factory=lambda s: None, resource_factory=lambda *a: None,
                approval=self.approval, execution_kind='substitute')
        self.assertFalse(target.exists())

    def test_nonisolated_database_and_disabled_retrieval_are_rejected(self):
        path, digest = self.material()
        target = self.root/'forbidden'
        for settings in (self.settings(target).model_copy(update={'database_url': 'mysql://fixture/fixture'}),
                         self.settings(target).model_copy(update={'retrieval_enabled': False})):
            with self.subTest(settings=settings.database_url), self.assertRaises(ValueError):
                execute_case(path, target, expected_input_sha256=digest, settings=settings,
                    provider_factory=lambda s: None, resource_factory=lambda *a: None,
                    approval=self.approval, execution_kind='substitute')
            self.assertFalse(target.exists())

    def test_batch_unknown_stops_later_case_without_creating_its_root(self):
        first, first_digest = self.material('R01')
        second, second_digest = self.material('R02')
        scenario = PilotScenario('unknown')
        gate = StopOnUnknown(scenario)
        target = self.root/'batch'
        with deny_socket_connections(), patch('research_loop.config.Settings.from_env',
                side_effect=AssertionError('default settings forbidden')):
            report = run_pilot([
                {'case_path': first, 'expected_input_sha256': first_digest},
                {'case_path': second, 'expected_input_sha256': second_digest}], target,
                settings_factory=self.settings, provider_factory=gate.factory, resource_factory=ToyResources,
                approval=self.approval, execution_kind='substitute', stop_state=lambda: gate.reason)
        self.assertEqual(report['planned'], 2)
        self.assertEqual(report['created'], 1)
        self.assertEqual(report['uncreated'], 1)
        self.assertEqual(report['results'][1]['uncreated_reason'], 'synthetic_unknown_call')
        self.assertFalse((target/'cases'/'R02').exists())
        self.assertEqual(len(scenario.requests), 1)
        self.assertEqual(report['global_stop'], 'synthetic_unknown_call')

    def test_global_stop_before_start_keeps_case_uncreated(self):
        path, digest = self.material()
        target = self.root/'run'
        result = execute_case(path, target, expected_input_sha256=digest, settings=self.settings(target),
            provider_factory=lambda s: self.fail('provider constructed'),
            resource_factory=lambda *a: self.fail('resource constructed'), approval=self.approval,
            execution_kind='substitute', stop_state=lambda: 'http_402')
        self.assertFalse(result['created'])
        self.assertFalse(target.exists())

    def test_quoted_question_scope_does_not_conflict_with_precise_task_permission(self):
        question = '仅根据附带的物理第2页核验方法；没有证据时保留缺证说明。'
        path, digest = self.material('R06', question=question)
        target = self.root/'scope-case'
        scenario = PilotScenario()
        with deny_socket_connections():
            result = execute_case(path, target, expected_input_sha256=digest,
                settings=self.settings(target), provider_factory=scenario.factory, resource_factory=ToyResources,
                approval=self.approval, execution_kind='substitute')
        self.assertIsNone(result['error'], result['error'])
        self.assertTrue(result['first_review_original_preserved'])
        self.assertEqual(result['actions'][0]['arguments']['verification_questions'], [question])
        context = json.loads(scenario.requests[0]['messages'][1]['content'].split('\n', 1)[1])
        self.assertEqual(context['verification_questions'], [question])

    def test_resource_exception_stops_batch_before_next_case(self):
        first, first_digest = self.material('R01')
        second, second_digest = self.material('R02')
        target = self.root/'batch'
        made = []

        def resource_factory(root, settings, case):
            made.append(case['case_id'])
            raise ValueError('synthetic frozen resource mismatch')

        with deny_socket_connections():
            report = run_pilot([
                {'case_path': first, 'expected_input_sha256': first_digest},
                {'case_path': second, 'expected_input_sha256': second_digest}], target,
                settings_factory=self.settings, provider_factory=lambda s: self.fail('provider constructed'),
                resource_factory=resource_factory, approval=self.approval, execution_kind='substitute')
        self.assertEqual(made, ['R01'])
        self.assertEqual(report['created'], 0)
        self.assertEqual(report['uncreated'], 2)
        self.assertEqual(report['global_stop'], 'case_exception:ValueError')
        self.assertFalse((target/'cases'/'R02').exists())


if __name__ == '__main__':
    unittest.main()
