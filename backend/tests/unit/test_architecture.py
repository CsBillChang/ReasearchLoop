"""分层约束：路由不互借私有工厂，业务与领域层不反向依赖 HTTP。"""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2] / 'research_loop'


class ArchitectureTests(unittest.TestCase):
    """约束实际 import 方向，避免资源拆分后重新出现路由耦合。"""

    def test_routes_use_dependencies_instead_of_other_routes(self):
        """只有集中路由表可以导入资源路由；端点的共享依赖来自 dependencies。"""
        for path in (ROOT / 'api').glob('*.py'):
            if path.name == 'router.py':
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('research_loop.api'):
                    with self.subTest(file=path.name, line=node.lineno):
                        self.assertEqual(node.module, 'research_loop.api.dependencies')
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        self.assertFalse(name.name.startswith('research_loop.api.'), str(path))

    def test_business_layers_do_not_import_http(self):
        """服务可由 Worker 和 API 复用，不能要求导入路由或 HTTPException。"""
        for folder in ('services', 'storage', 'domain'):
            for path in (ROOT / folder).glob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    modules = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                               [name.name for name in node.names] if isinstance(node, ast.Import) else [])
                    for module in modules:
                        with self.subTest(file=str(path.relative_to(ROOT)), line=getattr(node, 'lineno', 0)):
                            self.assertFalse(module.startswith(('research_loop.api', 'research_loop.main', 'fastapi')))

    def test_each_http_operation_is_registered_once(self):
        """集中装配不能重复挂载相同方法与路径，防止新旧路由同时生效。"""
        from research_loop.main import app
        from fastapi.routing import APIRoute
        operations = [
            (method, route.path)
            for route in app.routes if isinstance(route, APIRoute)
            for method in route.methods
        ]
        self.assertEqual(len(operations), len(set(operations)))
