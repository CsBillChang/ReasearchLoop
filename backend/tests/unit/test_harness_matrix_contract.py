"""用真实本格收据核验矩阵文本合同，不调用外部模型或部署资源。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from harness_ab_budget import Ledger
from harness_ab_transport import (BudgetTransport, deep_reading_request_contract,
    matrix_request_contract, validate_request_contracts, visual_request_contract)
from run_read_paper_ab import (qualification_configuration, validate_manifest,
    validate_matrix_delivery_witness)
from research_loop.providers.deepseek import DeepSeekProvider, RequestNotSent
from research_loop.storage.mysql import Database


class MatrixContractTests(unittest.TestCase):
    """匹配完整原生请求而非关键词，同时保持预算的发送前拒绝。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = Database('sqlite:///' + str(self.root / 'task.sqlite3'))
        self.db.ensure_schema()
        self.db.create_task('root', 'synthetic-matrix', 'synthetic-hash', '合成矩阵问题', 'deny', [])
        self.state = {'task_id': 'root', 'role': 'supervisor',
            'research_plan': {'evidence_matrix_requested': True}, 'evidence_ids': ['synthetic-evidence']}
        self.db.save_checkpoint('root', self.state)
        self.request = {'messages': [{'role': 'user', 'content': '合成矩阵问题'}],
            'tools': [], 'force_text': True, 'max_tokens': 64, 'thinking': {'type': 'disabled'}}
        self.payload = {'model': 'deepseek-flash', 'messages': self.request['messages'],
            'temperature': 0.1, 'max_tokens': 64, 'thinking': {'type': 'disabled'}}
        self.db.begin_model_call('root', 1, self.request)
        self.db.mark_model_call_sent('root', 1)
        self.ledger = Ledger(self.root / 'budget.sqlite3', allowed_cells=[['synthetic', 'A']])
        self.ledger.begin_root('root', 'synthetic', 'A')
        self.sent = []

    def check(self, payload=None):
        return validate_matrix_delivery_witness(self.db, 'root', payload or self.payload, 'deepseek-flash')

    def test_final_contracts_require_all_components_and_bind_qualification(self):
        contracts = {'visual': visual_request_contract(), 'deep_reading': deep_reading_request_contract(),
                     'matrix_delivery': matrix_request_contract()}
        validate_request_contracts(contracts, final_suite=True)
        for missing in contracts:
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                validate_request_contracts({k: v for k, v in contracts.items() if k != missing}, final_suite=True)
        changed = deepcopy(contracts)
        changed['deep_reading']['max_tokens_by_phase']['continuation'] = 4096
        with self.assertRaises(ValueError):
            validate_request_contracts(changed, final_suite=True)
        self.assertEqual(qualification_configuration({'request_contracts': contracts})['request_contracts'], contracts)
        self.assertNotEqual(qualification_configuration({'request_contracts': contracts}),
                            qualification_configuration({'request_contracts': changed}))
        validate_request_contracts({'visual': visual_request_contract()})

    def test_final_qualification_cannot_omit_or_expand_component_contracts(self):
        visual = {'visual': visual_request_contract()}
        for contracts in (None, visual, {**visual, 'unreviewed': {}}):
            manifest = {'experiment_kind': 'read_paper_exact_duplicate',
                        'final_suite_policy': {}, 'request_contracts': contracts}
            with self.subTest(contracts=contracts), patch('agent_final_runtime.validate_suite_manifest'):
                with self.assertRaisesRegex(PermissionError, 'COMPONENT_REQUEST_CONTRACT_REQUIRED'):
                    validate_manifest(manifest, 'synthetic', 'A', 'qualify')

    def provider(self, *, contract=True, witness=True):
        def handler(request):
            body = json.loads(request.read())
            self.sent.append(body)
            raw = {'id': 'synthetic-matrix-response', 'model': 'deepseek-flash',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '合成回答'},
                             'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
            if body.get('stream'):
                raw['choices'][0]['delta'] = raw['choices'][0].pop('message')
                return httpx.Response(200, content='data: ' + json.dumps(raw) + '\n\ndata: [DONE]\n\n')
            return httpx.Response(200, json=raw)
        transport = BudgetTransport(self.ledger, 'root', self.root / 'requests',
            inner=httpx.MockTransport(handler), matrix_contract=matrix_request_contract() if contract else None,
            matrix_request_check=self.check if witness else None)
        client = httpx.Client(transport=transport, trust_env=False)
        self.addCleanup(client.close)
        return DeepSeekProvider('synthetic-no-secret', model='deepseek-flash', client=client)

    def test_native_nonstream_and_stream_projection(self):
        self.assertTrue(self.check())
        streamed = {**self.payload, 'stream': True, 'stream_options': {'include_usage': True}}
        self.assertFalse(self.check(streamed))
        self.db.execute('INSERT INTO conversation_links(task_id,conversation_id,message_id) VALUES(?,?,?)',
                        ('root', 'synthetic-conversation', 'synthetic-message'))
        self.assertTrue(self.check(streamed))
        self.assertFalse(self.check())
        for altered in ({**self.payload, 'messages': [{'role': 'user', 'content': '篡改正文'}]},
                        {**self.payload, 'max_tokens': 65}, {**streamed, 'stream_options': {}},
                        {**self.payload, 'stage': 'matrix'}, {**self.payload, 'stream': False},
                        {**streamed, 'stream': 1}, {**streamed, 'stream_options': {'include_usage': 1}}):
            with self.subTest(payload=altered):
                self.assertFalse(self.check(altered))

    def test_native_tools_and_force_text_project_exactly(self):
        self.request.update(tools=[{'type': 'function', 'function': {'name': 'synthetic'}}], force_text=False)
        self.db.execute('UPDATE model_calls SET request_json=? WHERE task_id=?',
                        (json.dumps(self.request), 'root'))
        expected = {**self.payload, 'tools': self.request['tools'], 'tool_choice': 'auto'}
        self.assertTrue(self.check(expected))
        self.assertFalse(self.check())

    def test_plan_evidence_and_role_cannot_be_claimed_in_http(self):
        for update in ({'role': 'researcher'}, {'evidence_ids': []},
                       {'research_plan': {'evidence_matrix_requested': False}}, {'task_id': 'other'}):
            self.db.save_checkpoint('root', {**deepcopy(self.state), **update})
            with self.subTest(update=update):
                self.assertFalse(self.check())
        self.db.save_checkpoint('root', self.state)
        self.db.execute('UPDATE tasks SET role=? WHERE task_id=?', ('researcher', 'root'))
        self.assertFalse(self.check())

    def test_no_unique_started_sent_receipt_means_no_allowance(self):
        self.db.execute('UPDATE model_calls SET status=? WHERE task_id=?', ('succeeded', 'root'))
        self.assertFalse(self.check())
        self.db.execute('UPDATE model_calls SET status=?,sent_state=? WHERE task_id=?',
                        ('started', 'not_sent', 'root'))
        self.assertFalse(self.check())
        self.db.mark_model_call_sent('root', 1)
        self.db.begin_model_call('root', 2, self.request)
        self.db.mark_model_call_sent('root', 2)
        self.assertFalse(self.check())

    def test_component_stage_is_not_a_matrix_witness(self):
        self.request['stage'] = 'dpr_deep_reading'
        self.db.execute('UPDATE model_calls SET request_json=? WHERE task_id=?',
                        (json.dumps(self.request), 'root'))
        self.assertFalse(self.check())

    def test_native_matrix_text_reaches_memory_http_and_settles(self):
        provider = self.provider()
        with patch('socket.socket.connect', side_effect=AssertionError('real sockets forbidden')):
            provider.send(self.request['messages'], force_text=True, max_tokens=64, thinking={'type': 'disabled'})
        self.assertEqual(self.sent, [self.payload])
        self.assertEqual(self.ledger.snapshot()['actual_known_tokens'], 15)

    def test_native_matrix_sse_settles_without_changing_wire(self):
        self.db.execute('INSERT INTO conversation_links(task_id,conversation_id,message_id) VALUES(?,?,?)',
                        ('root', 'synthetic-conversation', 'synthetic-message'))
        provider = self.provider()
        with patch('socket.socket.connect', side_effect=AssertionError('real sockets forbidden')):
            provider.send_stream(self.request['messages'], force_text=True, max_tokens=64,
                                 thinking={'type': 'disabled'}, on_public=lambda _: None)
        self.assertEqual(self.sent, [{**self.payload, 'stream': True, 'stream_options': {'include_usage': True}}])
        self.assertEqual(self.ledger.snapshot()['actual_known_tokens'], 15)

    def test_missing_contract_rejects_before_any_physical_reservation(self):
        provider = self.provider(contract=False)
        with self.assertRaises(RequestNotSent):
            provider.send(self.request['messages'], max_tokens=64, thinking={'type': 'disabled'})
        self.assertEqual(self.sent, [])
        self.assertEqual(self.ledger.snapshot()['reservations'], [])
        self.assertTrue(self.ledger.snapshot()['halt_reason'])

    def test_missing_native_witness_rejects_before_http(self):
        provider = self.provider(witness=False)
        with self.assertRaises(RequestNotSent):
            provider.send(self.request['messages'], max_tokens=64, thinking={'type': 'disabled'})
        self.assertEqual(self.sent, [])
        self.assertEqual(self.ledger.snapshot()['physical_sent_requests'], 0)


if __name__ == '__main__':
    unittest.main()
