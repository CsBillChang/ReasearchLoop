# integration

API→MySQL→Worker→工具及文件/索引，先覆盖通过和拒绝两分支。

当前包含任务、协作、记忆、Reflection、检索、文件保存和恢复等跨模块检查。默认使用合成材料、供应商替身和临时SQLite；真实MySQL、Qdrant、模型与本机材料必须由各测试的显式环境门控启用，跳过项不算真实通过。

HTTP测试优先用`app.dependency_overrides`替换`api.dependencies.get_runtime_settings`或`get_database`，结束后恢复原覆盖。不要导入其他资源路由的私有服务工厂。既有测试中对统一配置入口的patch属于兼容夹具，不能替换真实业务验证。

单独运行引用其他测试夹具的模块时，使用`PYTHONPATH=backend:backend/tests .venv/bin/python -m unittest integration.test_product -v`；仓库根的`make test`会通过discover自动提供该导入根。
