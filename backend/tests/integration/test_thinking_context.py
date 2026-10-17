"""思考模式压缩接续：实际HTTP适配器和API/Worker，供应商与论文为隔离替身。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from integration import test_context_continuity as base
from research_loop.agents.graph import initial_state
from research_loop.domain.context import tool_archive_summary
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.memory import dump
from research_loop.workers.executor import TaskExecutor


class ThinkingContextIntegrationTests(unittest.TestCase):
    """压缩不能改写模型身份；兼容旧摘要、成功响应恢复与未知请求阻断。"""
    setUp = base.ContextContinuityTests.setUp
    task = base.ContextContinuityTests.task
    source_messages = base.ContextContinuityTests.source_messages

    def test_legacy_summary_restores_original_exchange_before_projection(self):
        """有/无读回工具均先恢复旧assistant摘要，缺档和失效版本仍拒绝。"""
        tid = self.task()
        messages, _ = self.source_messages(tid)
        projected, _ = self.context.prepare({'task_id':tid, 'prompt':'比较'}, messages, self.tools)
        index = next(i for i, m in enumerate(projected) if tool_archive_summary(m))
        summary = tool_archive_summary(projected[index])
        legacy = {k:v for k,v in summary.items() if k not in {'archived_tool_result','tool_call_id','observation_excerpt'}}
        legacy.update(archived_tool_exchange=True, observations=[summary['observation_excerpt']])
        old = projected[:index-1] + [{'role':'assistant','content':dump(legacy)}] + projected[index+1:]
        originals = [m for m in messages if m.get('role')=='assistant']
        for tools in (self.tools, []):
            restored, metrics = self.context.prepare({'task_id':tid,'prompt':'比较'}, old, tools)
            validate_messages(restored)
            self.assertGreaterEqual(metrics['restored_source_groups'], 1)
            self.assertEqual([m for m in restored if m.get('role')=='assistant'], originals)

    def run_http_chain(self, recovery=False, unknown=False):
        """用HTTP/SSE替身跑五步链，模拟有思考与无思考响应交替及恢复边界。"""
        from research_loop.main import app
        from research_loop.api.dependencies import get_runtime_settings
        scenario = base.ContinuityProvider(readback=True)
        requests, emitted = [], []
        def handler(request):
            """仅接受已实际发出的assistant原件；不把此替身当供应商真实验收。"""
            payload = json.loads(request.content)
            requests.append(deepcopy(payload))
            validate_messages(payload['messages'])
            for message in payload['messages']:
                if message.get('role')=='assistant': self.assertIn(message, emitted)
            self.assertNotIn('thinking', payload)
            if unknown and len(requests)==4:
                raise httpx.ReadTimeout('fixture unknown, never resend', request=request)
            response = scenario.send(payload['messages'], payload.get('tools'))
            message = response['choices'][0]['message']
            if len(requests)>=3: message['reasoning_content'] = '原始推理，不得摘要或伪造：'+str(len(requests))
            emitted.append(deepcopy(message))
            if not payload.get('stream'): return httpx.Response(200, json=response)
            delta = deepcopy(message)
            for i, call in enumerate(delta.get('tool_calls',[])): call['index'] = i
            chunks = [{'id':response['id'],'choices':[{'delta':delta,'finish_reason':'tool_calls'}]},
                      {'choices':[], 'usage':response['usage']}]
            return httpx.Response(200, text=''.join('data: '+dump(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n',
                                  headers={'content-type':'text/event-stream'})
        app.dependency_overrides[get_runtime_settings] = lambda:self.settings
        self.addCleanup(app.dependency_overrides.pop, get_runtime_settings, None)
        allowed={'set_research_plan','read_paper','read_memory','create_note','finish_task'}
        with (TestClient(app) as api, httpx.Client(transport=httpx.MockTransport(handler)) as client,
              patch.object(httpx.HTTPTransport,'handle_request',side_effect=AssertionError('external HTTP forbidden'))):
            tid = self.task(client=api)
            executor = TaskExecutor(self.settings, self.db,
                lambda s:DeepSeekProvider('fixture', model=s.deepseek_model, client=client), lambda *_:base.LongPDF(),
                supervisor_tools=allowed)
            saved = []
            if recovery:
                original_save = CheckpointStore.save
                def interrupt(store, task_id, state):
                    """第四次成功收据入库后、草稿工具执行前中断。"""
                    if len(requests)==4 and state.get('pending_tool_calls'):
                        raise SystemExit('fixture after compacted receipt')
                    return original_save(store, task_id, state)
                with patch.object(CheckpointStore,'save',interrupt), self.assertRaises(SystemExit):
                    executor.execute(tid)
                saved = deepcopy(self.db.list_model_calls(tid))
                self.db.update_task(tid,status='queued',lease_until='2000-01-01T00:00:00+00:00')
            executor.execute(tid)
            view = api.get('/api/tasks/'+tid).json()
            self.assertEqual(view['save_policy'],'deny')
            self.assertTrue(any(tool_archive_summary(m) for p in requests for m in p['messages']))
            self.assertTrue(all(p.get('stream') for p in requests))
            self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
            if unknown:
                self.assertIn(view['status'], ['partial','failed'])
                self.assertEqual(view['usage']['unknown'], 1)
                self.assertGreater(view['usage']['budget']['tokens']['reserved_tokens'], 0)
                self.assertEqual(len(requests), 4)
            else:
                self.assertEqual(view['status'],'completed',view['result'])
                self.assertEqual(len(requests), 5)
                self.assertEqual(view['usage']['unknown'], 0)
                self.assertIn('两份合成材料',view['result']['answer_markdown'])
                self.assertTrue(any(a['tool_name']=='read_memory' and a['status']=='succeeded' for a in self.db.list_actions(tid)))
                rows={r['call_id']:r for r in self.db.list_model_calls(tid)}
                for row in saved: self.assertEqual(rows[row['call_id']],row)
            receipts = deepcopy(self.db.list_model_calls(tid))
            count=len(requests)
            if not unknown:
                state=initial_state(tid,self.db.get_task(tid)['prompt'],'deny',
                    {a['input_ref']:a for a in json.loads(self.db.get_task(tid)['attachments_json'])},5,80,0)
                CheckpointStore(self.db).save(tid,state|{'coverage_required':True})
            self.db.update_task(tid,status='queued',claimed_by=None,lease_token=None,lease_until=None)
            executor.execute(tid)
            self.assertEqual(len(requests),count)
            self.assertEqual(self.db.list_model_calls(tid),receipts)

    def test_api_worker_stream_compaction_and_receipt_replay(self):
        """并行调用、压缩、读回、交付后重放成功收据，网络替身只接收五次。"""
        self.run_http_chain()

    def test_interrupted_successful_receipt_is_reused(self):
        """受控中断之后复用包含推理字段的成功收据，不重发。"""
        self.run_http_chain(recovery=True)

    def test_unknown_after_compaction_blocks_resend(self):
        """发送后超时保留unknown及预占，不能因消息修复而盲重发。"""
        self.run_http_chain(unknown=True)


if __name__=='__main__': unittest.main()
