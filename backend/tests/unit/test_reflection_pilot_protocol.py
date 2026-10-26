"""思考模式消息的严格离线合同；不模拟供应商对空推理槽的实际接受结果。"""
from copy import deepcopy
import json
import unittest

from research_loop.providers.deepseek import InvalidConversation, validate_messages
from scripts.reflection_pilot_real import programmatic_review_message


def validate_thinking_history(payload: dict, *, programmatic_ids=(), expected_reasoning=None) -> None:
    """核对工具配对、必需推理字段及已有响应的逐字回传。

    这是按供应商文档与原始400补充的本地测试合同，不是远端服务的完整实现。
    programmatic_ids 必须由调用方明确传入，不能仅凭ID前缀豁免模型历史。
    """
    messages = payload['messages']
    validate_messages(messages)
    if not payload.get('tools') or payload.get('thinking', {}).get('type') == 'disabled':
        return
    approved = set(programmatic_ids)
    expected = expected_reasoning or {}
    for message in messages:
        if message.get('role') != 'assistant':
            continue
        if not isinstance(message.get('reasoning_content'), str):
            raise InvalidConversation('thinking tool history requires reasoning_content string')
        calls = message.get('tool_calls') or []
        ids = [call['id'] for call in calls]
        is_programmatic = bool(ids) and all(identity in approved for identity in ids)
        if is_programmatic:
            if len(calls) != 1:
                raise InvalidConversation('programmatic review must contain exactly one action')
            call = calls[0]
            original = programmatic_review_message(call['id'], json.loads(call['function']['arguments']))
            if message != original:
                raise InvalidConversation('programmatic review message changed')
        elif not message['reasoning_content']:
            raise InvalidConversation('empty model reasoning has no fixture provenance')
        for identity in ids:
            if identity in expected and message['reasoning_content'] != expected[identity]:
                raise InvalidConversation('model reasoning changed from recorded fixture response')


def protocol_fixture() -> dict:
    """与被拒请求同形的公开合成夹具，不含论文、标签或真实模型响应。"""
    identity = 'programmatic-review-synthetic'
    args = {'target_type': 'answer', 'answer_markdown': '合成原稿。',
            'verification_questions': ['合成问题？'], 'evidence_ids': ['page:synthetic:2']}
    return {'model': 'deepseek-flash', 'max_tokens': 65536, 'temperature': 0.1,
        'tools': [{'type': 'function', 'function': {'name': 'finish_task', 'parameters': {}}}],
        'tool_choice': 'auto', 'messages': [
            {'role': 'system', 'content': '合成系统提示'},
            {'role': 'user', 'content': '合成任务'},
            programmatic_review_message(identity, args),
            {'role': 'tool', 'tool_call_id': identity, 'content': '{"ok":true}'}]}


class ReflectionPilotProtocolTests(unittest.TestCase):
    """严格拒绝原故障及近邻缺陷，保留真实模型推理的来源约束。"""

    def setUp(self):
        self.payload = protocol_fixture()
        self.seed_id = self.payload['messages'][2]['tool_calls'][0]['id']

    def validate(self, payload=None, **kwargs):
        validate_thinking_history(payload or self.payload, programmatic_ids=[self.seed_id], **kwargs)

    def test_program_seed_has_honest_empty_reasoning_and_exact_arguments(self):
        self.validate()
        seed = self.payload['messages'][2]
        self.assertEqual(seed['reasoning_content'], '')
        args = json.loads(seed['tool_calls'][0]['function']['arguments'])
        self.assertEqual(args['answer_markdown'], '合成原稿。')
        self.assertEqual(args['verification_questions'], ['合成问题？'])
        before = deepcopy(args)
        programmatic_review_message(self.seed_id, args)
        self.assertEqual(args, before)

    def test_original_missing_field_is_rejected(self):
        del self.payload['messages'][2]['reasoning_content']
        with self.assertRaisesRegex(InvalidConversation, 'requires reasoning_content'):
            self.validate()

    def test_null_and_nonstring_reasoning_are_rejected(self):
        for value in (None, 0, [], {}):
            with self.subTest(value=value):
                self.payload['messages'][2]['reasoning_content'] = value
                with self.assertRaises(InvalidConversation):
                    self.validate()

    def test_program_seed_must_not_borrow_or_invent_reasoning(self):
        self.payload['messages'][2]['reasoning_content'] = '合成模型推理，不能冒充程序推理'
        with self.assertRaisesRegex(InvalidConversation, 'programmatic review message changed'):
            self.validate()

    def test_prefix_alone_does_not_authorize_empty_reasoning(self):
        with self.assertRaisesRegex(InvalidConversation, 'no fixture provenance'):
            validate_thinking_history(self.payload)

    def test_program_helper_rejects_nonprogram_identity(self):
        for identity in ('model-call', 'programmatic-review-'):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                programmatic_review_message(identity, {})

    def test_orphan_and_missing_tool_replies_are_rejected(self):
        orphan = deepcopy(self.payload)
        orphan['messages'][3]['tool_call_id'] = 'other'
        with self.assertRaisesRegex(InvalidConversation, 'no matching pending call'):
            self.validate(orphan)
        self.payload['messages'].pop()
        with self.assertRaisesRegex(InvalidConversation, 'unanswered tool calls'):
            self.validate()

    def test_duplicate_tool_identity_is_rejected(self):
        self.payload['messages'][2]['tool_calls'] *= 2
        with self.assertRaisesRegex(InvalidConversation, 'duplicate tool call identity'):
            self.validate()

    def test_actual_model_reasoning_must_be_returned_verbatim(self):
        seed = self.payload['messages'][2]
        seed['tool_calls'][0]['id'] = 'model-response-1'
        self.payload['messages'][3]['tool_call_id'] = 'model-response-1'
        seed['reasoning_content'] = '仅供测试的响应推理\n逐字保留。'
        expected = {'model-response-1': seed['reasoning_content']}
        self.validate(expected_reasoning=expected)
        for changed in ('', '仅供测试的响应推理', '改写后的合成推理'):
            with self.subTest(changed=changed):
                seed['reasoning_content'] = changed
                with self.assertRaises(InvalidConversation):
                    self.validate(expected_reasoning=expected)

    def test_tool_request_also_checks_assistant_without_tool_calls(self):
        self.payload['messages'].append({'role': 'assistant', 'content': '合成先前回答'})
        with self.assertRaisesRegex(InvalidConversation, 'requires reasoning_content'):
            self.validate()
        self.payload['messages'][-1]['reasoning_content'] = '合成回答的测试推理'
        self.validate()

    def test_no_tools_or_explicit_disabled_does_not_require_reasoning(self):
        del self.payload['messages'][2]['reasoning_content']
        self.payload['thinking'] = {'type': 'disabled'}
        self.validate()
        del self.payload['thinking']
        self.payload['tools'] = []
        self.validate()


if __name__ == '__main__':
    unittest.main()
