"""API→数据库→Worker阶段交接；只有供应商和PDF是显式替身。"""
import json
import os
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from research_loop.api.dependencies import get_runtime_settings
from research_loop.main import app
from research_loop.config import Settings
from research_loop.providers.deepseek import DeepSeekProvider, RequestNotSent, validate_messages
from research_loop.services.conversations import ConversationService
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.workers.executor import TaskExecutor
from integration.test_collaboration_contract import FixturePDF


class ProgressScenario:
    """明确脚本决策，只证明接线及恢复，不代表真实模型会采用策略。"""

    def __init__(self, crash=False):
        """固定两篇材料，必要时在阶段保存后注入供应商前本地异常。"""
        self.crash = crash
        self.requests = []
        self.steps = {}

    @staticmethod
    def response(name, arguments):
        """返回生产解析器可接受的原始响应及已知替身用量。"""
        raw = {'id': uuid4().hex, 'choices': [{'finish_reason': 'tool_calls', 'message': {
            'role': 'assistant', 'content': '', 'tool_calls': [{'id': uuid4().hex, 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        return raw

    def send(self, messages, tools, **kwargs):
        """从实际子目标及观察取证身份，再生成阶段提交或汇总动作。"""
        validate_messages(messages)
        self.requests.append({'messages': messages, 'tools': tools})
        names = {t['function']['name'] for t in tools}
        observations = [json.loads(m['content']) for m in messages if m.get('role') == 'tool']
        done = [c['function']['name'] for m in messages for c in m.get('tool_calls', [])]
        if 'finish_research' in names:
            brief = next(json.loads(m['content'].split('\n', 1)[1]) for m in messages
                         if m.get('content', '').startswith('结构化子任务与已授权定位：'))
            ref = brief['input_refs'][0]
            step = self.steps.get(ref, 0); self.steps[ref] = step + 1
            if step == 0:
                return self.response('read_paper', {'target_ref': ref})
            pages = next(o['data']['pages'] for o in observations if o.get('data', {}).get('pages'))
            finding = ref + '：语言嵌入参与方法，来源为已读合成页。'
            if step == 1:
                return self.response('record_research_progress', {'findings': [{'text': finding, 'evidence_ids': [pages[0]['evidence_id']]}], 'gaps': ['尚未核实实验'] if self.crash else []})
            if self.crash:
                raise RequestNotSent('CONTROLLED_AFTER_PROGRESS')
            return self.response('finish_research', {'findings': [finding], 'evidence_ids': [p['evidence_id'] for p in pages], 'paper_ids': [pages[0]['document_id'].removeprefix('doc-')]})
        if 'set_research_plan' not in done:
            return self.response('set_research_plan', {'goal': '比较附件语言机制', 'task_kind': 'compare',
                'input_refs': ['input-A', 'input-B'], 'requested_count': 2, 'count_confirmed': True,
                'goals': [{'goal_id': 'compare', 'description': '比较语言机制', 'min_papers': 2,
                           'requires_fulltext': True, 'input_refs': ['input-A', 'input-B']}]})
        if 'delegate_research' not in done:
            return self.response('delegate_research', {'subtasks': [{'objective': '调查' + ref, 'mode': 'investigate',
                'goal_ids': ['compare'], 'input_refs': [ref]} for ref in ['input-A', 'input-B']]})
        children = [c for o in observations for c in o.get('data', {}).get('subtasks', [])]
        for child in children:
            if child.get('findings_in'):
                # 模型投影中完整发现只出现一次，替身按明确路径读当前状态。
                from integration.test_context_continuity import memory_payload
                canonical = next(c for c in memory_payload(messages)['current_task']['subtasks']
                                 if c['subtask_id'] == child['subtask_id'])
                child['findings'] = [f['text'] for f in canonical['research_progress']['findings']] + child.get('findings', [])
        if not all(c.get('findings') for c in children):
            raise AssertionError('stage findings disappeared from handoff')
        return self.response('finish_task', {'answer_markdown': '\n\n'.join(c['findings'][0] for c in children),
            'goals': [{'goal_id': 'compare', 'status': 'unresolved' if self.crash else 'satisfied',
                       'subtask_ids': [c['subtask_id'] for c in children],
                       'reason': '实验未完成，保留方法发现' if self.crash else '两份原文均读取并交回方法发现'}]})

    def close(self):
        """替身没有网络连接。"""


class FollowupScenario:
    """只针对历史成果追问，检查实际messages中的阶段记录及定向页读回。"""

    def __init__(self, case):
        """注入断言对象，采用固定三步替身而非真实模型决策。"""
        self.case, self.step = case, 0

    def send(self, messages, tools, **kwargs):
        """要求历史发现与当前禁止保存持续可见。"""
        validate_messages(messages)
        memory = next(m['content'] for m in messages if m.get('role')=='system' and m.get('content','').startswith('[ResearchLoop Memory v1]'))
        payload = json.loads(memory[memory.index('\n{')+1:])
        self.case.assertEqual(payload['current_task']['save_policy'], 'deny')
        history = [f for record in payload['recent_research'] for progress in record.get('stage_findings', []) for f in progress['findings']]
        self.case.assertTrue(any('input-A' in text for text in history))
        step = self.step; self.step += 1
        if step == 0:
            # 使用实际工具结果的合成长目标触发压缩，不依赖删除assistant参数。
            return ProgressScenario.response('set_research_plan', {'goal':'解释上次第一篇的语言机制','task_kind':'search',
                'requested_count':1,'count_confirmed':True,'input_refs':['input-A'],
                'goals':[{'goal_id':'followup','description':'解释第一篇；'+('合成追问限定，用于触发工具结果压缩。'*40),'min_papers':1,'requires_fulltext':True,'input_refs':['input-A']}]})
        if step == 1:
            return ProgressScenario.response('read_paper', {'target_ref':'input-A','start_page':1,'page_count':1})
        if step > 2:
            raise RequestNotSent('synthetic followup expected three steps; inspect prior tool rejection')
        return ProgressScenario.response('finish_task', {'answer_markdown':'根据上次候选发现与本次第1页，input-A采用语言嵌入；其余实验条件未核验。',
            'goals':[{'goal_id':'followup','status':'satisfied','reason':'定向读回当前获授权第一页，解释已记录方法'}]})

    def close(self):
        """没有外部连接需要关闭。"""


class ProgressHandoffTests(unittest.TestCase):
    """覆盖正常、受控失败和快照中断，任何路径均禁止保存业务文件。"""

    def run_case(self, crash=False, interrupt=False, followup=False, child_exception=False):
        """在独立目录与显式隔离库中运行一次API任务。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = os.getenv('CORE_LOOP_TEST_DATABASE_URL') or 'sqlite:///' + str(root / 'test.db')
            if not url.startswith('sqlite:') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_core_loop_'):
                raise ValueError('isolated database required')
            settings = Settings(project_root=root, data_dir=root/'data', output_dir=root/'notes', database_url=url)
            db = Database(url); db.ensure_schema()
            scenario = ProgressScenario(crash)
            app.dependency_overrides[get_runtime_settings] = lambda: settings
            attachments = []
            for name in ('A', 'B'):
                p = root/(name+'.pdf'); p.write_bytes(b'%PDF-synthetic')
                attachments.append({'input_ref': 'input-'+name, 'paper_id': 'paper-'+uuid4().hex, 'local_path': str(p)})
            try:
                with TestClient(app) as client:
                    cid = ConversationService(db, settings).create('接续测试', uuid4().hex)['conversation_id'] if followup else None
                    response = client.post('/api/tasks', json={'conversation_id': cid,'prompt': '比较两份附件语言机制，不保存', 'attachments': attachments,
                        'save_policy': 'deny', 'idempotency_key': uuid4().hex})
                    self.assertEqual(response.status_code, 202, response.text)
                    tid = response.json()['task_id']
                    executor = TaskExecutor(settings, db, provider_factory=lambda _: scenario, pdf_factory=lambda *_: FixturePDF())
                    if interrupt:
                        if interrupt == 'sigkill':
                            config=root/'worker-settings.json';config.write_text(settings.model_dump_json())
                            run=subprocess.run([sys.executable,str(Path(__file__).with_name('progress_recovery_process.py')),str(config),tid],
                                env={**os.environ,'PYTHONPATH':'backend:backend/tests','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,timeout=45)
                            self.assertEqual(run.returncode,-9,run.stderr.decode())
                            children_at_kill = db.list_subtasks(tid)
                            self.assertEqual(len(children_at_kill), 2)
                            for child in children_at_kill:
                                self.assertTrue(any(a['tool_name'] == 'record_research_progress' and a['status'] == 'succeeded'
                                                    for a in db.list_actions(child['subtask_id'])))
                                self.assertTrue(all(c['status'] == 'succeeded' for c in db.list_model_calls(child['subtask_id'])))
                        else:
                            original = CheckpointStore.save
                            def crash_after_action(store, task_id, state):
                                """成功进展已提交后，模拟进程在写下一份Checkpoint前退出。"""
                                if any(a['tool_name']=='record_research_progress' and a['status']=='succeeded' for a in db.list_actions(task_id)):
                                    raise SystemExit('CONTROLLED_CHECKPOINT_EXIT')
                                return original(store, task_id, state)
                            with patch.object(CheckpointStore,'save',crash_after_action):
                                with self.assertRaises(SystemExit): executor.execute(tid)
                        # 只推进当前隔离任务的租约，恢复不会改写历史receipt。
                        from research_loop.services.research_progress import ResearchProgressService
                        for child in db.list_subtasks(tid):
                            if ResearchProgressService(db).snapshot(child['subtask_id'])['revision']:
                                scenario.steps[child['spec']['input_refs'][0]]=2
                        db.execute('UPDATE tasks SET lease_until=? WHERE root_task_id=?',('2000-01-01T00:00:00Z',tid))
                    frozen = {c['call_id']: c for task in db.fetchall('SELECT task_id FROM tasks WHERE root_task_id=?',(tid,)) for c in db.list_model_calls(task['task_id']) if c['status']=='succeeded'} if interrupt else {}
                    if child_exception:
                        original = CheckpointStore.save
                        def child_failure(store, task_id, state):
                            """命中Worker子任务异常分支，而非模型请求异常分类。"""
                            if any(a['tool_name']=='record_research_progress' and a['status']=='succeeded' for a in db.list_actions(task_id)):
                                raise RuntimeError('CONTROLLED_AFTER_PROGRESS')
                            return original(store,task_id,state)
                        with patch.object(CheckpointStore,'save',child_failure):
                            result = executor.execute(tid)
                    else:
                        result = executor.execute(tid)
                    view = client.get('/api/tasks/'+tid).json()
                    self.assertIn('input-A', result.answer_markdown)
                    self.assertIn('input-B', result.answer_markdown)
                    self.assertEqual(view['status'], 'partial' if crash else 'completed', view)
                    self.assertEqual(list(root.rglob('*.md')), [])
                    children = db.list_subtasks(tid)
                    self.assertTrue(all(c['result']['result']['research_progress']['findings'] for c in children))
                    if crash:
                        self.assertTrue(all('CONTROLLED_AFTER_PROGRESS' in c['result']['result']['stop_reason'] for c in children))
                    if followup:
                        # 仅隔离配置触发压缩；不改变模型窗口和产品全局配置。
                        compressed = settings.model_copy(update={'context_compress_ratio': 0.01})
                        provider = FollowupScenario(self)
                        created = client.post('/api/tasks', json={'conversation_id':cid,
                            'prompt':'基于上次记录解释 input-A 的语言机制；只读取物理第1页，不保存。', 'save_policy':'deny','idempotency_key':uuid4().hex})
                        self.assertEqual(created.status_code,202,created.text)
                        next_id=created.json()['task_id']
                        TaskExecutor(compressed,db,provider_factory=lambda _:provider,pdf_factory=lambda *_:FixturePDF()).execute(next_id)
                        view=client.get('/api/tasks/'+next_id).json()
                        self.assertEqual(view['status'],'completed',view)
                        archives=db.fetchall('SELECT metrics_json FROM context_archives WHERE task_id=?',(next_id,))
                        self.assertTrue(any(json.loads(r['metrics_json']).get('compressed_groups') for r in archives))
                        self.assertEqual(list(root.rglob('*.md')), [])
                    if interrupt:
                        # 已成功的阶段动作不会重新发模型请求，观察仍只有一个成功版本。
                        self.assertTrue(all(len(c['result']['result']['research_progress']['findings']) == 1 for c in children))
                        restored = {c['call_id']: c for task in db.fetchall('SELECT task_id FROM tasks WHERE root_task_id=?',(tid,)) for c in db.list_model_calls(task['task_id'])}
                        self.assertTrue(frozen)
                        for call_id, call in frozen.items(): self.assertEqual(restored[call_id], call)

            finally:
                app.dependency_overrides.clear()

    def test_normal_two_paper_handoff(self):
        """两子角色记录阶段成果后正常交接，父任务完成比较。"""
        self.run_case()

    def test_failed_children_keep_submitted_work(self):
        """阶段提交后异常，父任务可交付已有内容和真实未完成项。"""
        self.run_case(crash=True)

    def test_checkpoint_exit_reuses_successful_progress(self):
        """进程退出边界恢复已成功阶段行动，不重新提交。"""
        self.run_case(interrupt=True)

    def test_compressed_followup_retains_stage_findings(self):
        """同会话压缩后可见阶段发现，只定向读第一页且不继承保存权限。"""
        self.run_case(followup=True)

    @unittest.skipUnless(os.name=='posix','SIGKILL requires POSIX')
    def test_worker_sigkill_after_progress_commit(self):
        """实际Worker进程被杀后复用成功阶段行动，双方成果最终交付。"""
        self.run_case(interrupt='sigkill')

    def test_worker_exception_retains_committed_stage(self):
        """子图在成功提交后抛异常，Worker异常分支仍交回已有发现。"""
        self.run_case(crash=True,child_exception=True)
