# ResearchLoop 后端

从仓库根操作。入口：`api → services → storage`入队，`Worker → agents/graph → tools → services → providers/storage`执行。API不等待长研究任务；MySQL保存事实，SSE重放事件，Qdrant是可重建派生索引。

## 后端目录与依赖

| 目录/入口 | 负责什么 |
|---|---|
| `main.py` | FastAPI 生命周期、中间件和 `/api` 前缀 |
| `api/router.py` | 集中挂载资源路由，避免重复注册 |
| `api/dependencies.py` | 用 FastAPI Depends 装配请求内配置、数据库及业务服务 |
| `api/tasks.py` / `conversations.py` | 任务与会话、取消、澄清和后续请求 |
| `api/uploads.py` / `events.py` | 有界 PDF 字节流、事件 JSON 和 SSE 重放 |
| `api/artifacts.py` / `feedback.py` | 成果与原文读回、保存确认、版本绑定反馈 |
| `agents` / `tools` | ReAct 角色循环、上下文/Reflection 和工具参数/权限 |
| `domain` | 服务、HTTP 与 Agent 共用的数据合同，不导入 HTTP 实现 |
| `services` | 论文、成果、反馈、记忆、检索和任务等业务规则 |
| `providers` / `storage` | 外部能力适配、数据库与文件实际读写 |
| `workers` | 领取/续租、角色协作、恢复和任务交付 |
| `scripts` / `migrations` / `tests` | 运维与显式评测入口、追加迁移和各层验证 |

资源路由只从 `api/dependencies.py` 取得公共依赖，不导入其他资源路由的私有函数。服务层不依赖 FastAPI；路由负责输入/状态码和响应，服务负责版本、权限与保存等业务判断。测试可用 `app.dependency_overrides` 注入隔离配置或数据库。

模块职责与源码阅读顺序见 [文件地图](../docs/file-map.md)。Supervisor动态委派Researcher并按需调用Reviewer；DPR精读为固定LLM节点，不强制每次问答使用。

```bash
make sync-index
make infra
make migrate
make start
make test
make check
```

来源/模型需要本机配置，见 [运行手册](../docs/delivery.md)。默认测试不调用DeepSeek；环境门控跳过不能报为真实验证通过。历史脚本显式冻结和隔离，不作日常入口。当前完善状态见 [status](../docs/status.md)。
