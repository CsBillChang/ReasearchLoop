"""AG3 API→Worker→独立 Reviewer→局部修订→保存；供应商为可观察替身。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from research_loop.agents.reflection import ReflectionService
from research_loop.config import Settings
from research_loop.providers.deepseek import DeepSeekProvider, validate_messages
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_adaptive_research import AdaptiveScenario
from integration.test_collaboration_contract import FixturePDF


class ReflectionScenario:
    """仅依据当前 messages 生成工具响应，使恢复也能重放相同路径。"""

    def __init__(self, no_progress=False):
        """选择最终支持或保留缺证，记录实际到达供应商的上下文。"""
        self.requests = []; self.no_progress = no_progress

    def handler(self, request):
        """遍历真实工具观察，不在执行器注入阶段专用动作。"""
        body = json.loads(request.content); self.requests.append(body)
        msgs = body['messages']; validate_messages(msgs)
        names = {t['function']['name'] for t in body.get('tools', [])}
        obs = [json.loads(m['content']) for m in msgs if m['role'] == 'tool']
        done = [c['function']['name'] for m in msgs for c in m.get('tool_calls', [])]
        response = AdaptiveScenario.response
        if 'finish_research' in names:
            if not obs: return response('read_paper', {'target_ref': 'input-A'})
            return response('finish_research', {'findings':['原文使用language prefix embeddings。'], 'evidence_ids':[p['evidence_id'] for p in obs[-1]['data']['pages']], 'paper_ids':['paper-A'], 'gaps':[]})
        if 'finish_review' in names:
            source = json.loads(msgs[-1]['content'].split('\n', 1)[1])
            supported = '修正句：使用语言前缀。' in msgs[1]['content'] and not self.no_progress
            return response('finish_review', {'status':'supported' if supported else 'issues', 'findings':[] if supported else [{
                'claim':'错误句：不使用语言。', 'status':'contradicted', 'basis':'direct_statement', 'source_quote':source[0]['text'], 'reason':'独立来源显示使用语言；fixture_reason_unique', 'evidence_ids':[source[0]['evidence_id']]}], 'gaps':[]})
        if 'set_research_plan' not in done:
            return response('set_research_plan', {'goal':'核对用户草稿', 'requested_count':1, 'count_confirmed':True, 'save_requested':not self.no_progress,
                'input_refs':['input-A'], 'goals':[{'goal_id':'verify','description':'核对语言方法','requires_fulltext':True,'min_papers':1}]})
        if 'delegate_research' not in done:
            return response('delegate_research', {'subtasks':[{'objective':'核对语言方法','input_refs':['input-A'],'goal_ids':['verify'],'mode':'investigate'}]})
        children = next(o['data']['subtasks'] for o in obs if 'subtasks' in o['data'])
        if 'create_note' not in done:
            return response('create_note', {'title':'局部修订案例','markdown':'不变开头。\n错误句：不使用语言。\n不变结尾。','evidence_ids':children[0]['evidence_ids']})
        drafts = [o['data'] for o in obs if o['data'].get('markdown') and o['data'].get('draft_id')]
        current = drafts[-1]
        reviews = [o['data'] for o in obs if 'review_id' in o['data']]
        if not reviews or ('revise_note' in done and len(reviews) == 1):
            return response('review_note', {'draft_id':current['draft_id'],'content_hash':current['content_hash'],'reason':'用户指出语言条件疑点'})
        if 'revise_note' not in done:
            return response('revise_note', {'draft_id':current['draft_id'],'content_hash':current['content_hash'],'review_id':reviews[-1]['review_id'],
                'patches':[{'finding_index':0,'old_text':'错误句：不使用语言。','new_text':'修正句：使用语言前缀。','reason':'来源支持语言前缀','evidence_ids':children[0]['evidence_ids']}]})
        if not self.no_progress and 'save_note' not in done:
            return response('save_note', {'draft_id':current['draft_id'],'content_hash':current['content_hash']})
        return response('finish_task', {'draft_id':current['draft_id'], 'goals':[{'goal_id':'verify','status':'unresolved' if self.no_progress else 'satisfied',
             'reason':'缺口保留，不再循环' if self.no_progress else '有来源并已核验', 'subtask_ids':[children[0]['subtask_id']]}]})

    def factory(self, settings):
        """沿生产 DeepSeek receipt/协议入口测试，不发外网请求。"""
        return DeepSeekProvider('fixture', client=httpx.Client(transport=httpx.MockTransport(self.handler)))


class ReflectionLoopTests(unittest.TestCase):
    """测试数据仅进入显式隔离库及临时输出目录。"""

    def setUp(self):
        """创建独立测试配置及合法附件。"""
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        url = os.getenv('AG3_TEST_DATABASE_URL') or 'sqlite:///' + str(self.root/'db.sqlite')
        if url.startswith('mysql') and not url.rsplit('/',1)[-1].startswith('researchloop_test_'):
            raise ValueError('test DB required')
        self.settings = Settings(project_root=self.root,data_dir=self.root/'data',output_dir=self.root/'notes',database_url=url)
        self.db = Database(url); self.db.ensure_schema(); (self.root/'A.pdf').write_bytes(b'%PDF-test')

    def run_case(self, no_progress=False):
        """以真实 API 入队并执行 Worker，读回所有审校、修订和文件。"""
        from research_loop.main import app
        scenario = ReflectionScenario(no_progress)
        executor = TaskExecutor(self.settings,self.db,scenario.factory,lambda s,f:FixturePDF())
        with patch('research_loop.api.tasks.get_settings', return_value=self.settings), TestClient(app) as client:
            response = client.post('/api/tasks',json={'prompt':'核对语言条件并修订用户草稿。' + ('不要保存' if no_progress else '保存到项目'),
                'save_policy':'deny' if no_progress else 'allow','attachments':[{'input_ref':'input-A','local_path':str(self.root/'A.pdf'),'paper_id':'paper-A'}]})
            self.assertEqual(response.status_code,202); tid = response.json()['task_id']
            executor.execute(tid)
            view = client.get('/api/tasks/'+tid).json()
        self.assertEqual(view['status'], 'partial' if no_progress else 'completed', json.dumps(view,ensure_ascii=False))
        state = ReflectionService(self.db).snapshot(tid)
        self.assertEqual(len(state['attempts']),2); self.assertEqual(len(state['patches']),1)
        self.assertEqual(state['stop_reason'],'no_progress' if no_progress else 'supported')
        reviewer_requests = [r for r in scenario.requests if any(t['function']['name']=='finish_review' for t in r.get('tools', []))]
        self.assertEqual(len(reviewer_requests),2)
        self.assertNotIn('fixture_reason_unique',json.dumps(reviewer_requests[1]['messages']))
        self.assertIn('"non_target_text_preserved": true',reviewer_requests[1]['messages'][1]['content'])
        self.assertEqual(view['result']['answer_markdown'],'不变开头。\n修正句：使用语言前缀。\n不变结尾。')
        if no_progress:
            self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
            self.assertTrue(view['result']['gaps'])
        else:
            self.assertEqual(Path(view['result']['saved_path']).read_text(), view['result']['answer_markdown'])
        return tid, scenario, executor

    def test_success_and_completed_root_recovery_no_resend(self):
        """恢复已完成 checkpoint 不重复请求或产生第二次修订。"""
        tid, scenario, executor = self.run_case(); calls=len(scenario.requests)
        self.db.update_task(tid,status='queued',result_json=None)
        executor.execute(tid,'restored')
        self.assertEqual(len(scenario.requests),calls)

    def test_no_progress_partial_is_readable_and_deny_no_file(self):
        """缺证保留内容；停止不伪造 supported，也不扩大保存权限。"""
        self.run_case(True)

    def test_no_result_failure_has_body_but_remains_failed(self):
        """失败说明不被算作已取得研究成果。"""
        from research_loop.domain.tasks import TaskCreate
        tid=TaskService(self.db).create(TaskCreate(prompt='test failure')).task_id
        executor=TaskExecutor(self.settings,self.db)
        state={'coverage_required':True,'final_reason':'provider unavailable'}
        result=executor._deliverable_result(tid,state)
        self.assertIn('provider unavailable',result.answer_markdown)
        self.assertEqual(executor._delivery_status(tid,state,result).value,'failed')

    def test_chat_explanation_and_saved_note_are_separate_deliverables(self):
        """原文修订已保存，另写总结不丢附件，也不冒称总结正文已保存/核验。"""
        from research_loop.domain.tasks import FinishTaskInput
        from research_loop.services.coverage import CoverageService
        tid, _, executor = self.run_case()
        last = next(a for a in reversed(self.db.list_actions(tid)) if a["tool_name"] == "finish_task")
        args = {**last["arguments"], "draft_id": None, "answer_markdown": "修订及复核已完成，见保存的笔记附件。"}
        delivery = CoverageService(self.db).finish(tid, FinishTaskInput.model_validate(args))
        self.assertEqual(delivery["status"], "completed")
        result = executor._deliverable_result(tid, {"delivery": delivery, "final_reason": "task delivered"})
        self.assertIsNone(result.draft_id); self.assertIsNone(result.saved_path)
        self.assertEqual(result.answer_markdown, args["answer_markdown"])
        self.assertEqual(result.note_artifacts[0]["review_status"], "supported")
        self.assertEqual(result.note_artifacts[0]["integrity"], "verified")
        Path(result.note_artifacts[0]["path"]).write_text("用户修改")
        changed = CoverageService(self.db).finish(tid, FinishTaskInput.model_validate(args))
        self.assertEqual(changed["status"], "partial")
        self.assertEqual(changed["note_artifacts"][0]["integrity"], "missing_or_changed")

    @unittest.skipUnless(os.getenv('AG3_TEST_DATABASE_URL'), 'explicit isolated MySQL required for SIGKILL')
    def test_sigkill_receipt_and_revision_resume_without_replay(self):
        """两处真实 SIGKILL：成功收据不重发，已提交 patch 不重复写入。"""
        import subprocess, sys
        from research_loop.domain.tasks import TaskCreate
        for point in ['receipt','revision']:
            with self.subTest(point=point):
                tid=TaskService(self.db).create(TaskCreate(prompt='核对语言条件并修订后保存',save_policy='allow',
                    attachments=[{'input_ref':'input-A','local_path':str(self.root/'A.pdf'),'paper_id':'paper-A'}])).task_id
                counter=self.root/(point+'.jsonl'); ready=self.root/(point+'.ready')
                env={**os.environ,'AG3_ROOT':str(self.root),'AG3_TASK':tid,'AG3_COUNTER':str(counter),'AG3_READY':str(ready)}
                command=[sys.executable,str(Path(__file__).with_name('reflection_recovery_process.py'))]
                killed=subprocess.run(command+[point],env=env,timeout=40,capture_output=True)
                self.assertEqual(killed.returncode,-9,killed.stderr.decode())
                self.assertTrue(ready.exists())
                before=[json.loads(x) for x in counter.read_text().splitlines()]
                receipts={r['call_id']:r for t in [tid]+[c['subtask_id'] for c in self.db.list_subtasks(tid)] for r in self.db.list_model_calls(t)}
                self.db.update_task(tid,lease_until='2000-01-01T00:00:00+00:00')
                resumed=subprocess.run(command+['resume'],env=env,timeout=40,capture_output=True)
                self.assertEqual(resumed.returncode,0,resumed.stderr.decode())
                view=TaskService(self.db).view(self.db.get_task(tid))
                self.assertEqual(view.status.value,'completed',view.result)
                after=[json.loads(x) for x in counter.read_text().splitlines()]
                self.assertEqual(sum(x['reviewer'] for x in after),2)
                self.assertEqual(sum(x['researcher'] for x in after),sum(x['researcher'] for x in before))
                state=ReflectionService(self.db).snapshot(tid)
                self.assertEqual(len(state['patches']),1);self.assertEqual(len(state['attempts']),2)
                for t in [tid]+[c['subtask_id'] for c in self.db.list_subtasks(tid)]:
                    for r in self.db.list_model_calls(t):
                        if r['call_id'] in receipts:
                            self.assertEqual(r['request_id'],receipts[r['call_id']]['request_id'])
                            self.assertEqual(r['usage_json'],receipts[r['call_id']]['usage_json'])
                self.assertFalse(self.db.has_unknown_family_call(tid))
