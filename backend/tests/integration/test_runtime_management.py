"""本机运维合同：端口、身份、只读、共享锁和配置就绪；无真实模型请求。"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.main import app
from research_loop.services.runtime import application_lock, atomic_json
from research_loop.storage.mysql import Database
from scripts.manage import Manager, ROOT, available, diagnose, owned


class RuntimeManagementTests(unittest.TestCase):
    """对危险操作的边界使用替身；实际启停由独立隔离验收执行。"""

    def setUp(self):
        """每例新建配置与 SQLite，避免碰生产任务。"""
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.settings=Settings(project_root=ROOT,data_dir=self.root/'data',output_dir=self.root/'notes',
                               database_url='sqlite:///'+str(self.root/'db'))
        self.db=Database(self.settings.database_url);self.db.ensure_schema()
        for path in (ROOT/'backend/migrations').glob('*.sql'):
            self.db.execute('INSERT INTO schema_version(version,applied_at) VALUES(?,?)',(path.stem,'test'))

    def test_diagnosis_checks_migrations_without_provider(self):
        """数据库就绪不等于密钥存在或模型可用，缺迁移会单独返回。"""
        with patch('scripts.manage.probe',return_value={'ok':False}):
            report=diagnose(self.settings)
            self.assertTrue(report['database_ready']);self.assertFalse(report['model_key_present'])
            self.assertEqual(report['model_requests'],0)
            self.db.execute("DELETE FROM schema_version WHERE version='009_subscriptions'")
            self.assertFalse(diagnose(self.settings)['database_ready'])

    def test_port_collision_does_not_kill(self):
        """端口被占用时启动失败，原 socket 仍由原持有者使用。"""
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));sock.listen()
            port=sock.getsockname()[1]
            self.assertFalse(available(port))
            with patch('scripts.manage.runtime_root',return_value=self.root),patch('scripts.manage.os.kill') as kill:
                manager=Manager(self.settings,'collision')
                with self.assertRaisesRegex(RuntimeError,'PORT_IN_USE'):
                    manager.start(port,port+1)
                kill.assert_not_called()

    def test_pid_reuse_not_owned(self):
        """相同 PID 不同启动命令不具有停止权限。"""
        with patch('scripts.manage.process_signature',return_value='another-process'):
            self.assertFalse(owned({'pid':123,'signature':'old','token':'random'}))

    def test_missing_key_prevents_worker_start(self):
        """没有密钥时拒绝日常 Worker，不能显示 Agent 已可执行。"""
        with patch('scripts.manage.runtime_root',return_value=self.root),patch('scripts.manage.available',return_value=True),patch('scripts.manage.probe',return_value={'ok':True}):
            with self.assertRaisesRegex(RuntimeError,'MODEL_KEY_MISSING'):
                Manager(self.settings,'missing').start()

    def test_restore_hold(self):
        """恢复目录的执行冻结不会被普通 start 绕过。"""
        self.settings.data_dir.mkdir();(self.settings.data_dir/'restore-hold.json').write_text('{}')
        with patch('scripts.manage.runtime_root',return_value=self.root),patch('scripts.manage.available',return_value=True),patch('scripts.manage.probe',return_value={'ok':True}):
            with self.assertRaisesRegex(RuntimeError,'RESTORE_EXECUTION_HELD'):
                Manager(self.settings.model_copy(update={'deepseek_api_key':'test'}),'restore').start()

    def test_stop_identity_mismatch(self):
        """身份不符只报告，不向该 PID 发送信号。"""
        with patch('scripts.manage.runtime_root',return_value=self.root),patch('scripts.manage.process_signature',return_value='unrelated'),patch('scripts.manage.os.kill') as kill:
            manager=Manager(self.settings,'mismatch')
            atomic_json(manager.state_path,{'services':{'worker':{'pid':1,'signature':'old','token':'old'}}})
            result=manager.stop(.01)
            self.assertFalse(result['stopped']);kill.assert_not_called()

    def test_cold_backup_lock_excludes_application(self):
        """维护锁与应用共享锁互斥；释放后可以重新进入。"""
        with application_lock(self.root,exclusive=True):
            with self.assertRaisesRegex(RuntimeError,'APPLICATION_BUSY'):
                with application_lock(self.root):pass
        with application_lock(self.root):pass

    def test_readonly_blocks_writes(self):
        """演示端口拒绝提交，健康信息明确只读且未探测模型。"""
        with patch.dict(os.environ,{'RESEARCHLOOP_READONLY':'1'}),patch('research_loop.api.health.get_settings',return_value=self.settings):
            with TestClient(app) as client:
                self.assertEqual(client.post('/api/tasks',json={}).status_code,403)
                health=client.get('/api/health').json()
                self.assertEqual(health['mode'],'readonly');self.assertFalse(health['model_probed'])

    def test_root_resolution(self):
        """裸 make 入口读取仓库根配置，不误用 backend 数据目录。"""
        self.assertEqual(Settings.from_env().project_root,ROOT)
