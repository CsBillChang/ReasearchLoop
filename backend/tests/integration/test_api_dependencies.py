"""HTTP 依赖注入验证：共享隔离数据库、请求内配置快照和只读拒绝。"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from research_loop.api.dependencies import get_database, get_runtime_settings
from research_loop.config import Settings
from research_loop.main import app
from research_loop.storage.mysql import Database


class ApiDependencyTests(unittest.TestCase):
    """通过公开依赖替换验证实际 HTTP 链路，不替换路由内部私有函数。"""

    def setUp(self):
        """每例使用独立 SQLite 和输出目录，依赖替换在测试结束后完整恢复。"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.settings = Settings(
            project_root=root, data_dir=root / 'data', output_dir=root / 'notes',
            database_url='sqlite:///' + str(root / 'unused-config.db'),
        )
        self.db = Database('sqlite:///' + str(root / 'injected.db'))
        self.db.ensure_schema()
        self.old_overrides = dict(app.dependency_overrides)
        self.addCleanup(self.restore_overrides)
        app.dependency_overrides[get_runtime_settings] = lambda: self.settings
        app.dependency_overrides[get_database] = lambda: self.db
        readonly = patch.dict(os.environ, {'RESEARCHLOOP_READONLY': '0'})
        readonly.start()
        self.addCleanup(readonly.stop)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def restore_overrides(self):
        """保留调用方原有依赖替换，避免污染其他接口测试。"""
        app.dependency_overrides.clear()
        app.dependency_overrides.update(self.old_overrides)

    def test_resources_share_the_injected_database_without_provider(self):
        """会话、任务、个人面板和笔记查询都使用注入库，配置中的备用库不会被创建。"""
        with patch('research_loop.providers.deepseek.DeepSeekProvider.send', side_effect=AssertionError('no model')):
            conversation = self.client.post('/api/conversations', json={
                'title': '依赖注入', 'idempotency_key': 'conversation',
            })
            self.assertEqual(conversation.status_code, 201, conversation.text)
            response = self.client.post('/api/tasks', json={
                'conversation_id': conversation.json()['conversation_id'],
                'prompt': '只受理，不运行', 'idempotency_key': 'task', 'save_policy': 'deny',
            })
            self.assertEqual(response.status_code, 202, response.text)
            task_id = response.json()['task_id']
            self.assertEqual(self.db.get_task(task_id)['prompt'], '只受理，不运行')
            for path in ('/api/profile', '/api/notes', '/api/reading-list', '/api/subscriptions',
                         '/api/tasks/' + task_id + '/artifacts', '/api/tasks/' + task_id + '/events'):
                self.assertEqual(self.client.get(path).status_code, 200, path)
        self.assertFalse((self.settings.project_root / 'unused-config.db').exists())
        self.assertEqual(self.db.list_model_calls(task_id), [])

    def test_configuration_is_resolved_once_per_request(self):
        """任务服务与数据库依赖共享同一份配置，避免一次请求混用不同数据库或预算。"""
        app.dependency_overrides.pop(get_database)
        calls = []

        def settings():
            """记录依赖解析次数，返回专用测试配置。"""
            calls.append(1)
            return self.settings

        app.dependency_overrides[get_runtime_settings] = settings
        response = self.client.post('/api/tasks', json={'prompt': '配置一致性', 'save_policy': 'deny'})
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(len(calls), 1)
        configured = Database(self.settings.database_url)
        self.assertEqual(configured.get_task(response.json()['task_id'])['prompt'], '配置一致性')

    def test_readonly_mode_rejects_before_creating_a_task(self):
        """依赖可替换也不能绕过应用只读中间件。"""
        with patch.dict(os.environ, {'RESEARCHLOOP_READONLY': '1'}):
            response = self.client.post('/api/tasks', json={'prompt': '不应入队'})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.db.fetchall('SELECT task_id FROM tasks'), [])
