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

模块职责见上表。Supervisor动态委派Researcher并按需调用Reviewer；精读为固定LLM节点，不强制每次问答使用。

```bash
make sync-gpu  # Apple Silicon；CPU显式回退可使用make sync-index
make infra
make migrate
make start
make test
make check
```

来源/模型需要本机配置，见 [项目说明](../README.md)。默认测试不调用DeepSeek；环境门控跳过不能报为真实验证通过。历史脚本显式冻结和隔离，不作日常入口。本机历史评测材料不随公开仓库分发。


## 1.5.5 检索配置

默认 `RESEARCHLOOP_RETRIEVAL_ROUTE=located_fp16` 与 `RESEARCHLOOP_RERANKER_BACKEND=mps_fp16`。候选采用论文标题/既有章节定位、固定通用概念扩展、18条局部候选与6条原始全授权范围保底，最多24条；重排保持原查询、原文全文、batch4、原序和512 token拒绝边界。没有题录标题时继续正文检索，无物理页码时保留未知值。模型不可用时返回可见降级，不自动转CPU。

模型由用户另行取得和放置：E5与原INT8重排条目沿用模板，新增 `reranker_fp16` 指向同revision的 `model.safetensors` / `config.json`，分词器沿用原 `reranker` 条目。启动前以SHA256核对文件。应用不自动下载权重，不执行远端模型代码。设置 `PYTORCH_ENABLE_MPS_FALLBACK=0`，运行于Apple Silicon；其他设备可显式选择 `RESEARCHLOOP_RETRIEVAL_ROUTE=hybrid_rrf` 与 `RESEARCHLOOP_RERANKER_BACKEND=onnx_cpu`。

历史固定112题离线评测中，定位+FP16相对BM25的Recall@10为59.82%→78.13%、MRR@24为0.2703→0.5400，纯文件检索组件热P50/P95为632/757毫秒，P50增加约592毫秒。该测量不包含在线SQL/来源授权回读、首次建索引或整个Agent交付；不承诺所有数据集相同收益。
