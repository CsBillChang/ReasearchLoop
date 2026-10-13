# 对话协议测试：覆盖工具调用配对、上下文完整性及供应商显式错误分支。
import unittest
import httpx
from research_loop.providers.deepseek import DeepSeekProvider, InvalidConversation, ProviderResponseError, validate_messages


class DeepSeekMessageTests(unittest.TestCase):
    def test_missing_tool_reply_rejected_before_network(self):
        client=httpx.Client(transport=httpx.MockTransport(lambda _:self.fail('must not send')))
        with client:
            provider=DeepSeekProvider('secret',client=client)
            with self.assertRaises(InvalidConversation):
                provider.send([{'role':'assistant','tool_calls':[{'id':'c'}]},{'role':'user','content':'2'}])

    def test_duplicate_and_unmatched_tool_result_rejected(self):
        with self.assertRaises(InvalidConversation):validate_messages([{'role':'tool','tool_call_id':'missing'}])
        with self.assertRaises(InvalidConversation):validate_messages([{'role':'assistant','tool_calls':[{'id':'c'},{'id':'c'}]}])

    def test_parallel_tool_responses_valid(self):
        validate_messages([{'role':'assistant','tool_calls':[{'id':'a'},{'id':'b'}]},
                           {'role':'tool','tool_call_id':'b'},{'role':'tool','tool_call_id':'a'},
                           {'role':'user','content':'continue'}])

    def test_http_error_body_is_captured_and_credential_redacted(self):
        with httpx.Client(transport=httpx.MockTransport(lambda _:httpx.Response(400,text='bad request secret-fixture',headers={'x-request-id':'id'}))) as client:
            with self.assertRaises(ProviderResponseError) as raised:
                DeepSeekProvider('secret-fixture',client=client).send([{'role':'user','content':'hi'}])
        error=raised.exception
        self.assertEqual(error.status_code,400);self.assertEqual(error.request_id,'id')
        self.assertNotIn('secret-fixture',error.body);self.assertFalse(error.as_record()['usage_known'])
