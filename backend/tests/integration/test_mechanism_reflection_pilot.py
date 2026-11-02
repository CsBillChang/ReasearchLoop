"""V1.6.0 六稿适配的替身串联；无真实论文、凭据或网络调用。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import httpx

from experiments.mechanism_pilot.reflection import (
    ScopedRetrievalRuntime, deny_network, execute_case, inspect_delivery, load_case,
    programmatic_review_message, run_batch,
)
from integration.test_research_reflection import ResearchReflectionScenario, ScopedPDF, is_reviewer_request
from research_loop.config import Settings
from research_loop.domain.retrieval import RetrievalQuery
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.services.papers import PaperService
from research_loop.services.retrieval import RetrievalRuntime
from research_loop.storage.mysql import Database


INVALID_USAGES = {
    'null_usage': {'prompt_tokens': None, 'completion_tokens': 1, 'total_tokens': 2},
    'negative_usage': {'prompt_tokens': -1, 'completion_tokens': 1, 'total_tokens': 0},
    'mismatched_usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 3},
    'bool_usage': {'prompt_tokens': True, 'completion_tokens': 1, 'total_tokens': 2},
    'float_usage': {'prompt_tokens': 1.0, 'completion_tokens': 1, 'total_tokens': 2},
    'null_total_usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': None},
    'missing_part_usage': {'completion_tokens': 1, 'total_tokens': 2},
}


class Scenario(ResearchReflectionScenario):
    """合成模型决策经过生产解析与工具合同，不预写成功结果。"""
    def __init__(self, mode='revision'):
        super().__init__('missing' if mode == 'missing' else 'answer')
        self.case_mode, self.origins, self.reasoning = mode, {}, {}
        if mode == 'correct':
            self.before = self.after
        if mode == 'missing':
            self.before = '在所给页范围无法确认该方法设置。'

    def response(self, name, arguments):
        raw = super().response(name, arguments).json()
        message = raw['choices'][0]['message']
        identity = message['tool_calls'][0]['id']
        message['reasoning_content'] = '公开合成推理哨兵:' + identity
        self.reasoning[identity] = message['reasoning_content']
        return httpx.Response(200, json=raw)

    def handler(self, request):
        body = json.loads(request.content)
        validate_messages(body['messages'])
        for message in body['messages']:
            for call in message.get('tool_calls', []):
                identity = call['id']
                if identity in self.origins:
                    assert message == programmatic_review_message(identity, json.loads(call['function']['arguments']))
                    assert self.origins[identity]['model_generated'] is False
                else:
                    assert identity in self.reasoning
                    assert message['reasoning_content'] == self.reasoning[identity]
        if self.case_mode == 'unknown':
            self.requests.append(body)
            raise TimeoutError('synthetic unknown transport')
        if self.case_mode == 'http400':
            self.requests.append(body)
            return httpx.Response(400, json={'error': {'message': 'synthetic rejection'}})
        response = super().handler(request)
        call = response.json()['choices'][0]['message']['tool_calls'][0]['function']
        args = json.loads(call['arguments'])
        if call['name'] == 'read_paper':
            args.update(start_page=2, page_count=1)
        elif self.case_mode == 'correct' and call['name'] == 'revise_research':
            observations = [json.loads(m['content']) for m in body['messages'] if m['role'] == 'tool']
            current = next(o['data']['target'] for o in observations if o.get('data', {}).get('review_id'))
            return self.response('finish_task', {'reflection_target_id': current['target_id'],
                'reflection_content_hash': current['content_hash'],
                'goals': [{'goal_id': 'method', 'status': 'satisfied', 'reason': '原稿有据'}]})
        if call['name'] == 'finish_task':
            args['processing_summary'] = '合成处理说明，不混入待评分正文。'
        if self.case_mode == 'tool_error' and call['name'] == 'finish_review':
            args['findings'][0]['source_quote'] = 'Synthetic quote absent from the source.'
        response = self.response(call['name'], args)
        if self.case_mode == 'missing_usage':
            raw = response.json()
            raw.pop('usage', None)
            return httpx.Response(200, json=raw)
        if self.case_mode in INVALID_USAGES:
            raw = response.json()
            raw['usage'] = INVALID_USAGES[self.case_mode]
            return httpx.Response(200, json=raw)
        return response

    def factory(self, settings):
        db = Database(settings.database_url)
        for row in db.fetchall('SELECT payload_json FROM task_events WHERE event_type=?',
                              ('reflection_pilot_programmatic_review',)):
            origin = json.loads(row['payload_json'])
            if origin['model_generated'] is False and origin['model_receipt_created'] is False:
                self.origins[origin['tool_call_id']] = origin
        return DeepSeekProvider('fixture', settings.deepseek_model, settings.deepseek_base_url,
            settings.request_timeout_seconds, client=httpx.Client(transport=httpx.MockTransport(self.handler)))


class Resources:
    """合成单页通过现有 PaperService 授权；不声称真实 RAG 资格。"""
    def __init__(self, root, settings, case):
        source = root/'synthetic.pdf'
        source.write_bytes(b'%PDF-synthetic')
        self.attachments = [{'input_ref': 'input-A', 'paper_id': 'paper-A', 'local_path': str(source)}]

    def bind(self, task_id, db):
        observed = PaperService(db, None, ScopedPDF()).read(task_id, 'input-A', 2, 1,
            {'input-A': self.attachments[0]})
        return {'evidence_ids': [p['evidence_id'] for p in observed['pages']]}

    @staticmethod
    def pdf_factory(settings, files):
        return ScopedPDF()

    @staticmethod
    def retrieval_factory(db, settings):
        return ScopedRetrievalRuntime(db, settings, lambda *args: None)


class ReflectionPilotTests(unittest.TestCase):
    """每例新 SQLite；主从角色实际走 TaskExecutor 与 Mock HTTP。"""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def material(self, case_id='R01', draft=None):
        pdf = ScopedPDF()
        document = pdf.ingest(str(self.root/'synthetic.pdf'), 'paper-A')
        page = pdf.pages(document, 2, 1)[0]
        value = {'case_id': case_id, 'model_input': {'question': '方法是否使用语言前缀？',
            'draft': draft or Scenario().before, 'documents': [{'paper_id': document.paper_id,
            'document_id': document.document_id, 'source_sha256': document.content_sha256,
            'allowed_pages': [page.page_number], 'initial_pages': [
                {'physical_page': page.page_number, 'text': page.text,
                 'page_text_sha256': hashlib.sha256(page.text.encode()).hexdigest()}]}]}}
        path = self.root/(case_id+'.json')
        path.write_text(json.dumps(value, ensure_ascii=False))
        return {'case_id': case_id, 'case_path': path,
                'expected_input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def settings(self, root):
        return Settings(project_root=root, data_dir=root/'data', output_dir=root/'output',
            database_url='sqlite:///'+str(root/'runtime.sqlite3'), deepseek_model='deepseek-flash',
            deepseek_api_key=None, deepseek_base_url='https://fixture.invalid', retrieval_enabled=True,
            qdrant_url='http://127.0.0.1:1', retrieval_namespace='synthetic_reflection',
            web_search_enabled=False, web_fetch_enabled=False, max_model_requests=12,
            reserve_model_requests=0, context_output_reserve=65536)

    def run_case(self, mode='revision'):
        scenario = Scenario(mode)
        spec = self.material(draft=scenario.before)
        root = self.root/'runtime'
        result = execute_case(spec['case_path'], root, expected_input_sha256=spec['expected_input_sha256'],
            settings=self.settings(root), provider_factory=scenario.factory, resource_factory=Resources)
        return result, scenario

    def test_worker_revision_final_body_and_native_usage(self):
        result, scenario = self.run_case()
        self.assertIsNone(result['error'], result)
        self.assertIsNone(result['global_stop'], result)
        self.assertEqual(result['view']['status'], 'completed', result['view'])
        audit = result['delivery_audit']
        for name in ('first_review_original_preserved', 'current_identity_valid', 'revision_chain_valid',
                     'final_body_matches_current', 'finish_bound_current'):
            self.assertTrue(audit[name], (name, result))
        self.assertEqual(audit['answer_body_markdown'], scenario.after)
        self.assertNotIn('合成处理说明', audit['answer_body_markdown'])
        self.assertEqual(result['real_model_calls'], 0)
        self.assertTrue(any(is_reviewer_request(p) for p in scenario.requests))
        self.assertTrue(any(not is_reviewer_request(p) for p in scenario.requests))
        receipts = result['physical_receipts']
        self.assertEqual(len(receipts), len(scenario.requests))
        expected = sum(json.loads(r['usage_json'])['total_tokens'] for r in receipts)
        self.assertEqual(result['view']['usage']['tokens']['total_tokens'], expected)
        self.assertEqual(result['strict_usage_summary']['total_tokens'], expected)
        self.assertEqual(result['strict_usage_summary']['unknown_receipts'], 0)
        self.assertEqual(result['bound_material_identity']['runtime_document_id'], 'doc-paper-A')
        altered = deepcopy(result['view']['result'])
        altered['answer_body_markdown'] = scenario.before
        self.assertFalse(inspect_delivery(scenario.before, result['reflection_snapshot'], result['actions'], altered)
                         ['final_body_matches_current'])
        snapshot = deepcopy(result['reflection_snapshot'])
        record = next(iter(snapshot['research_targets'].values()))
        record['patches'][0]['patches'][0]['new_text'] = '未经记录的改写'
        self.assertFalse(inspect_delivery(scenario.before, snapshot, result['actions'], result['view']['result'])
                         ['revision_chain_valid'])
        actions = deepcopy(result['actions'])
        next(a for a in actions if a['tool_name'] == 'finish_task')['arguments']['reflection_content_hash'] = '0'*64
        self.assertFalse(inspect_delivery(scenario.before, result['reflection_snapshot'], actions, result['view']['result'])
                         ['finish_bound_current'])

    def test_correct_and_missing_keep_useful_body(self):
        for mode in ('correct', 'missing'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                result, scenario = self.run_case(mode)
                self.assertIsNone(result['error'], result)
                self.assertEqual(result['delivery_audit']['answer_body_markdown'], scenario.before)
                self.assertEqual(result['view']['status'], 'partial' if mode == 'missing' else 'completed')

    def test_real_mode_and_real_configuration_rejected_before_creation(self):
        spec = self.material()
        for mode, settings in [('real_model', self.settings(self.root/'denied')),
                              ('substitute', self.settings(self.root/'denied').model_copy(update={
                                  'deepseek_base_url': 'https://api.deepseek.com', 'deepseek_api_key': 'synthetic-key'}))]:
            with self.subTest(mode=mode), self.assertRaises(PermissionError):
                execute_case(spec['case_path'], self.root/'denied', expected_input_sha256=spec['expected_input_sha256'],
                    settings=settings, provider_factory=lambda s: self.fail('provider constructed'),
                    resource_factory=lambda *a: self.fail('resource constructed'), execution_kind=mode)
            self.assertFalse((self.root/'denied').exists())

    def test_six_frozen_inputs_and_no_empty_collection_success(self):
        for i in range(1, 7):
            spec = self.material(f'R{i:02}')
            self.assertEqual(load_case(spec['case_path'], spec['expected_input_sha256'])['case_id'], spec['case_id'])
        audit = inspect_delivery('synthetic', {}, [], {'answer_body_markdown': 'synthetic'})
        self.assertFalse(audit['first_review_original_preserved'])
        self.assertFalse(audit['finish_bound_current'])

    def test_first_transport_error_stops_next_case_and_retains_missing_usage(self):
        for mode in ('http400', 'unknown', 'missing_usage'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                scenario = Scenario(mode)
                plan = [self.material('R01'), self.material('R02')]
                result = run_batch(plan, self.root/'batch', settings_factory=self.settings,
                    provider_factory=scenario.factory, resource_factory=Resources)
                self.assertEqual(result['created'], 1)
                self.assertTrue(result['global_stop'])
                self.assertFalse((self.root/'batch'/'R02').exists())
                self.assertEqual(len(scenario.requests), 1)
                receipts = result['results'][0]['physical_receipts']
                sent = [r for r in receipts if r['sent_state'] != 'not_sent']
                self.assertEqual(len(sent), 1)
                self.assertIsNone(json.loads(sent[0].get('usage_json') or '{}').get('total_tokens'))
                if mode == 'unknown':
                    self.assertEqual(sent[0]['status'], 'unknown')

    def test_invalid_usage_is_preserved_unknown_and_stops_next_send_and_case(self):
        for mode, usage in INVALID_USAGES.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                scenario = Scenario(mode)
                result = run_batch([self.material('R01'), self.material('R02')], self.root/'batch',
                    settings_factory=self.settings, provider_factory=scenario.factory, resource_factory=Resources)
                self.assertEqual(result['created'], 1)
                self.assertEqual(result['global_stop'], 'successful_call_usage_unknown')
                self.assertFalse((self.root/'batch'/'R02').exists())
                self.assertEqual(len(scenario.requests), 1)
                case = result['results'][0]
                sent = [r for r in case['physical_receipts'] if r['sent_state'] != 'not_sent']
                self.assertEqual(len(sent), 1)
                self.assertEqual(json.loads(sent[0]['response_json'])['usage'], usage)
                strict = next(row for row in case['strict_usage_audit'] if row['call_id'] == sent[0]['call_id'])
                self.assertEqual(strict['usage_status'], 'unknown')
                self.assertIsNone(strict['actual_tokens'])
                self.assertIsNone(case['strict_usage_summary']['total_tokens'])
                self.assertEqual(case['strict_usage_summary']['unknown_receipts'], 1)

    def test_bound_material_drift_rejected_before_seed_and_first_request(self):
        changes = {
            'page': ('UPDATE page_evidence SET page_number=1 WHERE evidence_id=?',),
            'text': ("UPDATE page_evidence SET text='changed source' WHERE evidence_id=?",),
            'document': ("UPDATE page_evidence SET document_id='other-document' WHERE evidence_id=?",),
            'source': ("UPDATE documents SET content_sha256='different-version'",),
            'paper': ("UPDATE documents SET paper_id='different-paper'",),
            'authorization': ('DELETE FROM task_evidence WHERE evidence_id=?',),
        }
        for name, (sql,) in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                scenario = Scenario()
                class DriftingResources(Resources):
                    def bind(self, task_id, db):
                        bound = super().bind(task_id, db)
                        db.execute(sql, (bound['evidence_ids'][0],) if '?' in sql else ())
                        return bound
                result = run_batch([self.material('R01'), self.material('R02')], self.root/'batch',
                    settings_factory=self.settings, provider_factory=scenario.factory, resource_factory=DriftingResources)
                self.assertEqual(result['created'], 1)
                self.assertTrue(result['global_stop'])
                self.assertFalse((self.root/'batch'/'R02').exists())
                self.assertEqual(scenario.requests, [])
                case = result['results'][0]
                self.assertIn('REFLECTION_BOUND_', case['error']['message'])
                self.assertIsNone(case['programmatic_review_origin'])
                self.assertEqual(case['physical_receipts'], [])

    def test_old_document_id_requires_explicit_exact_material_mapping(self):
        spec = self.material()
        case = json.loads(spec['case_path'].read_text())
        doc = case['model_input']['documents'][0]
        doc['document_id'] = 'frozen-old-document-id'
        spec['case_path'].write_text(json.dumps(case))
        spec['expected_input_sha256'] = hashlib.sha256(spec['case_path'].read_bytes()).hexdigest()
        mapping = {'input_document_id': doc['document_id'], 'runtime_document_id': 'doc-paper-A',
            'paper_id': doc['paper_id'], 'source_sha256': doc['source_sha256'],
            'physical_page': 2, 'page_text_sha256': doc['initial_pages'][0]['page_text_sha256']}
        for name, expected in [('absent', None), ('conflicting', {**mapping, 'source_sha256': 'different'}),
                               ('qualified', mapping)]:
            with self.subTest(name=name):
                scenario = Scenario()
                root = self.root/name
                result = execute_case(spec['case_path'], root, expected_input_sha256=spec['expected_input_sha256'],
                    settings=self.settings(root), provider_factory=scenario.factory, resource_factory=Resources,
                    expected_material_identity=expected)
                if name == 'qualified':
                    self.assertIsNone(result['global_stop'], result)
                    self.assertTrue(result['delivery_audit']['finish_bound_current'])
                    self.assertEqual(result['bound_material_identity']['input_document_id'], doc['document_id'])
                else:
                    self.assertTrue(result['global_stop'])
                    self.assertEqual(scenario.requests, [])
                    self.assertIsNone(result['programmatic_review_origin'])

    def test_shared_stop_before_case_and_before_later_send(self):
        spec = self.material()
        root = self.root/'not_created'
        result = execute_case(spec['case_path'], root, expected_input_sha256=spec['expected_input_sha256'],
            settings=self.settings(root), provider_factory=lambda *a: self.fail(), resource_factory=Resources,
            stop_state=lambda: 'synthetic_shared_budget_exhausted')
        self.assertFalse(result['created'])
        self.assertFalse(root.exists())
        scenario = Scenario()
        root = self.root/'one_send'
        result = execute_case(spec['case_path'], root, expected_input_sha256=spec['expected_input_sha256'],
            settings=self.settings(root), provider_factory=scenario.factory, resource_factory=Resources,
            stop_state=lambda: 'synthetic_shared_budget_exhausted' if scenario.requests else None)
        self.assertEqual(len(scenario.requests), 1)
        self.assertEqual(result['global_stop'], 'synthetic_shared_budget_exhausted')

    def test_child_tool_failure_stops_all_followup_sends_and_next_case(self):
        scenario = Scenario('tool_error')
        result = run_batch([self.material('R01'), self.material('R02')], self.root/'batch',
            settings_factory=self.settings, provider_factory=scenario.factory, resource_factory=Resources)
        self.assertEqual(result['created'], 1)
        self.assertEqual(result['global_stop'], 'failed_action:finish_review')
        self.assertEqual(len(scenario.requests), 1)
        self.assertFalse((self.root/'batch'/'R02').exists())
        first = result['results'][0]['first_failed_action']
        self.assertEqual(first['tool_name'], 'finish_review')
        self.assertNotEqual(first['task_id'], result['results'][0]['task_id'])
        self.assertFalse(first['observation']['ok'])
        self.assertIn('quote', first['observation']['message'])
        sent = [r for r in result['results'][0]['physical_receipts'] if r['sent_state'] != 'not_sent']
        self.assertEqual(len(sent), 1)

    def test_scope_wrapper_uses_current_retrieval_without_manual_model_assembly(self):
        settings = self.settings(self.root/'run')
        seen = []
        runtime = ScopedRetrievalRuntime(None, settings, lambda *args: seen.append(args))
        query = RetrievalQuery(query='synthetic', document_ids=['doc-A'])
        with patch.object(RetrievalRuntime, 'search', return_value={'native': True}) as native:
            self.assertEqual(runtime.search('task-A', query), {'native': True})
        self.assertEqual(seen[0], ('task-A', None, ['doc-A']))
        self.assertEqual(native.call_count, 1)
        self.assertEqual(settings.retrieval_route, 'located_fp16')
        self.assertEqual(settings.retrieval_reranker_backend, 'mps_fp16')

    def test_label_drift_and_network_are_rejected(self):
        spec = self.material()
        with self.assertRaisesRegex(ValueError, 'INPUT_DRIFT'):
            load_case(spec['case_path'], '0'*64)
        value = json.loads(spec['case_path'].read_text())
        value['model_input']['expected'] = 'label sentinel'
        spec['case_path'].write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'LABEL_IN_MODEL_INPUT'):
            load_case(spec['case_path'], hashlib.sha256(spec['case_path'].read_bytes()).hexdigest())
        with deny_network(), self.assertRaisesRegex(PermissionError, 'NETWORK_DENIED'):
            socket.create_connection(('127.0.0.1', 1))


if __name__ == '__main__':
    unittest.main()
