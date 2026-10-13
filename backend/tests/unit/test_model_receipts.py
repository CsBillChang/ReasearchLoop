# 收据恢复测试：成功响应复用，started/unknown 不盲重发；测试替身不产生真实用量。
import tempfile
import unittest
from pathlib import Path

from research_loop.agents.graph import build_graph, initial_state
from research_loop.providers.deepseek import ModelResponse
from research_loop.storage.mysql import Database
from research_loop.tools.registry import ToolRegistry


class Provider:
    def __init__(self): self.calls = 0
    def complete(self, *args, **kwargs):
        self.calls += 1
        return ModelResponse("done", raw={"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]},
                             usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}, request_id="r-1")


class ReceiptsTests(unittest.TestCase):
    def test_succeeded_receipt_is_parsed_without_resend(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Database("sqlite:///" + str(Path(temp) / "researchloop_test_receipt.db")); db.ensure_schema()
            db.create_task("t", None, "h", "prompt", "deny", [])
            provider = Provider()
            first = build_graph(provider, ToolRegistry([]), {}, db).invoke(initial_state("t", "prompt", "deny"))
            second = build_graph(provider, ToolRegistry([]), {}, db).invoke(initial_state("t", "prompt", "deny"))
            self.assertEqual(provider.calls, 1)
            self.assertEqual(first["final_answer"], second["final_answer"])
            self.assertEqual(db.list_model_calls("t")[0]["request_id"], "r-1")

    def test_legacy_started_is_unknown_but_explicit_not_sent_is_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Database("sqlite:///" + str(Path(temp) / "researchloop_test_started.db")); db.ensure_schema()
            db.create_task("t", None, "h", "prompt", "deny", [])
            db.save_model_call("t", 1, "started", {"x": 1})
            self.assertTrue(db.has_unknown_model_call("t"))
            db2 = Database("sqlite:///" + str(Path(temp) / "researchloop_test_unsent.db")); db2.ensure_schema()
            db2.create_task("u", None, "h", "prompt", "deny", [])
            db2.begin_model_call("u", 1, {"x": 1})
            self.assertFalse(db2.has_unknown_model_call("u"))

    def test_raw_response_survives_parser_failure_and_replays_without_send(self):
        from unittest.mock import patch
        class Sender:
            def __init__(self):self.calls=0
            def send(self,*args,**kwargs):
                self.calls+=1
                return {'id':'raw-before-parser','usage':{'total_tokens':7},'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'recovered'}}]}
        with tempfile.TemporaryDirectory() as temp:
            db=Database('sqlite:///'+str(Path(temp)/'researchloop_test_raw.db'));db.ensure_schema()
            db.create_task('raw',None,'h','prompt','deny',[])
            sender=Sender();graph=build_graph(sender,ToolRegistry([]),{},db)
            with patch('research_loop.agents.graph.DeepSeekProvider.parse_response',side_effect=ValueError('controlled parsing failure')):
                first=graph.invoke(initial_state('raw','prompt','deny'))
            self.assertIn('parse failed',first['final_reason'])
            receipt=db.get_model_call('raw',1)
            self.assertEqual(receipt['status'],'succeeded');self.assertEqual(receipt['request_id'],'raw-before-parser')
            second=graph.invoke(initial_state('raw','prompt','deny'))
            self.assertEqual(sender.calls,1);self.assertEqual(second['final_answer'],'recovered')
