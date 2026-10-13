"""AG4真实MySQL与SIGKILL恢复；仅在显式专用测试库执行。"""
import json
import os
import subprocess
import sys
from pathlib import Path
import unittest
from integration.test_memory import MemoryTests
from research_loop.domain.memory import NoteUpdate
from research_loop.services.tasks import TaskService


class MemoryRecoveryTests(MemoryTests):
    """复用独立数据准备，只运行下列三处退出窗口。"""
    def setUp(self):
        """无专用MySQL环境时明确跳过，绝不回退生产库。"""
        if not os.getenv('AG4_TEST_DATABASE_URL'):
            self.skipTest('explicit AG4 isolated MySQL required')
        super().setUp()

    def process(self,mode,tid='',payload=None):
        """执行故障子进程，返回真实退出码；不调用业务模型。"""
        root=self.settings.project_root
        path=root/'payload.json'
        if payload:path.write_text(json.dumps(payload))
        env={**os.environ,'AG4_ROOT':str(root),'AG4_TASK':tid,'AG4_COUNTER':str(root/'counter'),'AG4_PAYLOAD':str(path),'PYTHONPATH':'backend:backend/tests'}
        return subprocess.run([sys.executable,'backend/tests/integration/memory_recovery_process.py',mode],env=env,capture_output=True,text=True,timeout=40)

    def test_sigkill_receipt(self):
        """成功receipt后退出，恢复总计三次替身发送而非四次。"""
        tid=self.task('以后默认中文。'+self.prefix)
        first=self.process('receipt',tid)
        self.assertEqual(first.returncode,-9,first.stderr)
        self.db.update_task(tid,lease_until='2000-01-01T00:00:00+00:00')
        second=self.process('resume',tid)
        self.assertEqual(second.returncode,0,second.stderr)
        self.assertEqual((self.settings.project_root/'counter').read_text().splitlines(),['sent']*3)
        self.assertEqual(self.db.get_task(tid)['status'],'completed')
        self.assertEqual(len(self.db.list_model_calls(tid)),3)
        self.assertEqual(self.conv.view(self.cid)['total'],2)

    def test_sigkill_terminal_projection(self):
        """结束后投影前退出，读取会话即可恢复，零新增请求。"""
        tid=self.task('以后默认中文。'+self.prefix)
        first=self.process('projection',tid)
        self.assertEqual(first.returncode,-9,first.stderr)
        before=(self.settings.project_root/'counter').read_text()
        self.assertEqual(self.db.get_task(tid)['status'],'completed')
        self.assertEqual(self.conv.view(self.cid)['total'],2)
        TaskService(self.db).view(self.db.get_task(tid))
        self.assertEqual(self.conv.view(self.cid)['total'],2)
        self.assertEqual((self.settings.project_root/'counter').read_text(),before)

    def test_sigkill_version_file(self):
        """文件写入后退出，恢复复用同一文件并补版本指针。"""
        draft,receipt=self.saved_note()
        args=NoteUpdate(base_hash=draft.content_hash,markdown='# 故障恢复候选',idempotency_key=self.prefix,confirm=True)
        first=self.process('note',payload={'note_id':draft.draft_id,'update':args.model_dump()})
        self.assertEqual(first.returncode,-9,first.stderr)
        files=list(self.settings.output_dir.rglob('*.md'))
        mtimes={str(p):p.stat().st_mtime_ns for p in files}
        self.notes.update(draft.draft_id,args,True)
        self.assertEqual({str(p):p.stat().st_mtime_ns for p in files},mtimes)
        self.assertEqual(len(self.notes.versions(draft.draft_id)),2)
        self.assertEqual(Path(receipt.path).read_text(),draft.markdown)


# 只继承fixture辅助方法，不重复运行MemoryTests的所有用例。
for name in list(MemoryTests.__dict__):
    if name.startswith('test_'):
        setattr(MemoryRecoveryTests,name,None)
del MemoryTests
