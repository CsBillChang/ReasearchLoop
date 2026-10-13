# 澄清链路测试：问题、工具回复、checkpoint 与 resume 必须正确衔接。
"""Actual TaskExecutor + API + HTTP adapter; provider network is replaced at HTTP boundary."""
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.domain.tasks import TaskCreate, ResumeInput
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.workers.executor import TaskExecutor


class ClarificationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = Settings(project_root=root, data_dir=root/'data', output_dir=root/'notes',
                                 database_url='sqlite:///'+str(root/'researchloop_test_pause.db'), deepseek_api_key='fixture')
        self.db = Database(self.settings.database_url); self.db.ensure_schema()
        self.service = TaskService(self.db)
        self.task = self.service.create(TaskCreate(prompt='比较论文，先询问数量', idempotency_key='pause'))
        self.requests = []

    def handler(self, request):
        body = json.loads(request.content); self.requests.append(body)
        validate_messages(body['messages'])
        if len(self.requests) == 1:
            message = {'role':'assistant','content':'','reasoning_content':'opaque fixture to preserve',
                       'tool_calls':[{'id':'clarify-call','type':'function','function':{
                           'name':'request_clarification','arguments':json.dumps({'question':'比较几篇？','fields':['paper_count'],'suggested_values':['5']})}}]}
        else:
            self.assertEqual([m['role'] for m in body['messages']], ['system','user','assistant','tool','user','system'])
            self.assertEqual(body['messages'][2]['reasoning_content'], 'opaque fixture to preserve')
            self.assertEqual(body['messages'][3]['tool_call_id'], 'clarify-call')
            self.assertIn('2', body['messages'][4]['content'])
            self.assertIn('用户回答优先', body['messages'][5]['content'])
            message = {'role':'assistant','content':'已收到 2 篇的范围。', 'reasoning_content':'fixture'}
        return httpx.Response(200, json={'id':f'fixture-{len(self.requests)}', 'choices':[{'finish_reason':'tool_calls' if len(self.requests)==1 else 'stop','message':message}], 'usage':{'total_tokens':1}})

    def executor(self):
        client = httpx.Client(transport=httpx.MockTransport(self.handler)); self.addCleanup(client.close)
        return TaskExecutor(self.settings,self.db,lambda _:DeepSeekProvider('fixture', client=client))

    def pause(self):
        result = self.executor().execute(self.task.task_id)
        self.assertEqual(result.reason, 'clarification requested')
        self.assertEqual(self.db.get_task(self.task.task_id)['status'], 'needs_input')
        cp = self.db.get_checkpoint(self.task.task_id)
        self.assertEqual(cp['messages'][-1]['role'], 'tool')
        self.assertFalse(cp['pending_tool_calls'])
        return self.db.get_pending_clarification(self.task.task_id)

    def payload(self,q):
        return ResumeInput(clarification_id=q['clarification_id'],answer='2 篇',idempotency_key='answer')

    def test_api_pause_resume_new_worker_valid_http_messages(self):
        q = self.pause()
        from research_loop.main import app
        with patch('research_loop.api.tasks.get_settings',return_value=self.settings):
            with TestClient(app) as client:
                r=client.post(f'/api/tasks/{self.task.task_id}/resume',json=self.payload(q).model_dump())
                self.assertEqual(r.status_code,200,r.text)
        self.executor().execute(self.task.task_id,'new-worker')
        self.assertEqual(len(self.requests),2)
        self.assertTrue(all(c['status']=='succeeded' for c in self.db.list_model_calls(self.task.task_id)))
        self.assertNotIn('unknown',self.db.get_task(self.task.task_id)['result_json'])

    def test_concurrent_duplicate_resume_single_answer_event(self):
        q=self.pause(); payload=self.payload(q)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _:TaskService(Database(self.settings.database_url)).resume(payload,self.task.task_id),range(2)))
        self.assertEqual(len([e for e in self.db.list_events(self.task.task_id) if e['event_type']=='task_resumed']),1)
        self.assertEqual(len([m for m in self.db.get_checkpoint(self.task.task_id)['messages'] if m['role']=='user']),2)

    def test_cancelled_resume_rejected_without_clearing_cancel(self):
        q=self.pause(); self.service.cancel(self.task.task_id)
        with self.assertRaises(ValueError):self.service.resume(self.payload(q),self.task.task_id)
        self.assertTrue(self.db.get_task(self.task.task_id)['cancel_requested'])
        self.assertEqual(self.db.get_clarification(q['clarification_id'])['status'],'pending')

    def test_legacy_incomplete_checkpoint_rejected_without_resend(self):
        q=self.pause(); cp=self.db.get_checkpoint(self.task.task_id)
        cp['messages'].pop(); cp['pending_tool_calls']=[]
        self.db.save_checkpoint(self.task.task_id,cp)
        with self.assertRaises(ValueError):self.service.resume(self.payload(q),self.task.task_id)
        self.assertEqual(self.db.get_clarification(q['clarification_id'])['status'],'pending')
        self.assertEqual(len(self.requests),1)

    def test_checkpoint_before_pause_publication_recovers_without_new_call(self):
        class ExitBeforePublish(BaseException):pass
        with patch.object(TaskService,'complete',side_effect=ExitBeforePublish):
            with self.assertRaises(ExitBeforePublish):self.executor().execute(self.task.task_id)
        self.assertEqual(self.db.get_task(self.task.task_id)['status'],'running')
        self.assertEqual(self.db.get_checkpoint(self.task.task_id)['messages'][-1]['role'],'tool')
        self.db.update_task(self.task.task_id,lease_until='2000-01-01T00:00:00+00:00')
        self.executor().execute(self.task.task_id,'replacement-worker')
        self.assertEqual(self.db.get_task(self.task.task_id)['status'],'needs_input')
        self.assertEqual(len(self.requests),1)

    def test_lost_worker_still_cannot_write(self):
        self.pause(); cp=self.db.get_checkpoint(self.task.task_id)
        with self.assertRaises(RuntimeError):CheckpointStore(self.db).save(self.task.task_id,cp)

    def test_explicit_provider_400_is_not_unknown_and_not_resent(self):
        def reject(request):
            self.requests.append(request)
            return httpx.Response(400,json={'error':{'message':'missing tool response'}},headers={'x-request-id':'rejected-fixture'})
        client=httpx.Client(transport=httpx.MockTransport(reject));self.addCleanup(client.close)
        factory=lambda _:DeepSeekProvider('fixture',client=client)
        executor=TaskExecutor(self.settings,self.db,factory)
        executor.execute(self.task.task_id)
        call=self.db.list_model_calls(self.task.task_id)[0]
        self.assertEqual(call['status'],'rejected');self.assertFalse(self.db.has_unknown_model_call(self.task.task_id))
        self.assertEqual(json.loads(call['response_json'])['http_status'],400)
        self.assertIn('missing tool response',call['response_json'])
        self.assertEqual(call['request_id'],'rejected-fixture')
        self.db.update_task(self.task.task_id,status='queued',lease_until=None)
        executor.execute(self.task.task_id)
        self.assertEqual(len(self.requests),1)

    def test_real_sigkill_after_tool_checkpoint_before_pause(self):
        import subprocess
        root=Path(self.temp.name)
        env={**os.environ,'PAUSE_TEST_ROOT':str(root),'PAUSE_TEST_URL':self.settings.database_url,
             'PAUSE_TEST_TASK':self.task.task_id,'PYTHONPATH':str(Path(__file__).parents[2])}
        child=subprocess.run([str(Path(__file__).parents[3]/'.venv/bin/python'),str(Path(__file__).with_name('clarification_process.py'))],env=env,timeout=20,capture_output=True)
        self.assertEqual(child.returncode,-9,child.stderr.decode())
        self.assertTrue((root/'ready').exists())
        self.db.update_task(self.task.task_id,lease_until='2000-01-01T00:00:00+00:00')
        self.executor().execute(self.task.task_id,'post-sigkill')
        self.assertEqual(len(self.requests),0)
        self.assertEqual((root/'calls').read_text(),'call\n')
        self.assertEqual(self.db.get_task(self.task.task_id)['status'],'needs_input')
        self.assertEqual(self.db.get_checkpoint(self.task.task_id)['messages'][-1]['role'],'tool')
