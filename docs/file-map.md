# 文件职责与整理决策

当前保留前后端分层，避免为整理目录修改import、恢复路径和冻结runner。完整逐文件清单见 [code-inventory.md](code-inventory.md)。

| 目录 | 当前职责 | 整理决定 |
|---|---|---|
| backend/research_loop/api | FastAPI与SSE入口 | 保留 |
| backend/research_loop/agents | LangGraph循环/角色/Reflection/Memory | 保留，P0仅定向修改 |
| backend/research_loop/domain | 输入输出合同与共享身份 | 保留，不因简历名词重命名 |
| backend/research_loop/tools | 工具注册、同一schema与执行 | 保留 |
| backend/research_loop/services | 业务规则、授权、覆盖、检索、记忆、简报 | 保留，矩阵标为实验 |
| backend/research_loop/providers | 外部API、DPR、PDF与本机推理 | 保留；citations.py为退役兼容，不在活动工具注册 |
| backend/research_loop/storage | 事务/收据/Checkpoint/文件/派生索引 | 保留，不清库、不删迁移 |
| backend/research_loop/workers | 根任务和简报执行器 | 保留，整理不重启进程 |
| backend/migrations | 001–010追加schema | 保留原顺序与校验 |
| backend/scripts | 日常、评测、历史runner | 按scripts/README分类，保留路径避免断引用 |
| backend/tests | unit/integration/acceptance | 保留；私人PDF和上游源码通过环境变量定位，缺失显式跳过 |
| frontend | 原生JS聊天/个人面板/侧面板 | 保留，矩阵组件标实验 |
| config | 无凭据的来源和模型清单模板 | 新增；不是本机真实配置 |
| docs | 当前计划/状态/架构/手册/证据边界 | 重写活动入口，旧全文归档本机archive |
| data / artifacts | 本机运行数据与原始验证 | 原样保留，除目录说明外不上传 |

## 交付后变更去留

- **保留已交付**：自然语言偏好、对话改名/日期分组、累计Token预算、只读退避、阅读清单。它们没有与新方案冲突。
- **保留待修通用机制**：压缩、档案读回、任务目标和工具范围；在P0修，不重新建设Memory。
- **暂缓实验能力**：domain/research.py、services/evidence_matrix.py、矩阵工具与前端；不删除共享验证/任务合同，不宣称真实矩阵已交付。本轮不改其路由或默认启用状态，正式演示不依赖它。
- **保留未采用实验**：optimize_retrieval.py、verify_reranker_efficiency.py及可选参数；默认排序保持，旧结果保留。
- **归档旧开发说明**：旧AG详细计划、多轮“当前授权”、失败后补丁指令移入本机docs/archive；活动文档仅一个完善计划和一个status。
- **只清除公开包中的个人路径**：5个测试/历史runner改为环境变量＋用户Downloads默认，业务源码不变。没有删除运行文件、原稿或PDF。

## 一条请求怎么找文件

`frontend/src/app.js → api/tasks.py → services/tasks.py → storage/mysql.py → workers/executor.py → agents/graph.py → tools/registry.py → services/papers.py/providers/pdf.py → Observation → agents/graph.py → services/coverage.py → api/product.py/SSE`

并行Researcher、按需Reviewer、持久化响应与授权检查见 [架构](architecture.md)。后续每项改动的输入/输出/完成条件见 [完善计划](improvement-plan.md)。
