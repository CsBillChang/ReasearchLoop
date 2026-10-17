# ResearchLoop

面向个人文献跟踪与课题调研的对话式科研 Agent。本机单用户应用，架构为 **ReAct＋Reflection＋Memory**，使用 **LangGraph** 编排 Supervisor、Researcher、Reviewer。

用户可以用自然语言检索论文、添加链接或 PDF、提出原文问题、生成精读笔记与比较稿。Agent 根据任务选择工具、委派子任务、按需核验，并在无法全部完成时交付已有成果、执行原因和待确认事项。前端提供聊天、个人画像、真实任务进度、原文依据与项目内成果。

## 架构

```mermaid
flowchart TD
    UI[聊天 / 个人面板] --> API[FastAPI + SSE]
    API --> DB[(MySQL 任务 / 记忆 / 收据)]
    DB --> W[独立 Worker]
    W --> S[Supervisor: ReAct 决策与任务覆盖]
    S --> R[Researcher: 检索与阅读]
    S --> V[Reviewer: 按需 Reflection]
    R --> T[结构化工具与业务服务]
    V --> T
    T --> P[DPR 来源 / PDF / 项目笔记]
    T --> Q[BM25 + Qdrant + RRF + Cross-encoder]
    S --> M[历史研究 / 偏好 / 上下文管理]
    M --> DB
```

三个角色使用各自任务、上下文和权限，不表示三套独立训练模型。调用与观察由项目持久化；Checkpoint 是自实现状态快照。仅明确要求精读时调用 DPR 八点总结节点；Reflection 不作为所有回答和保存的统一批准门。

## 状态与边界

- AG0–AG7按约定范围交付，支持本机启动、停止、诊断、冷备与隔离恢复。
- 已补齐：自然语言偏好、对话改名/日期分组、累计 Token 预算、DPR 只读限流退避、聊天阅读清单。
- 上下文工作集、阶段发现和父子交接、同版本正文投影及错误交付分别验证；固定比较/压缩追问与合理部分交付保留实际记录。具体修复是否收束及未覆盖项只见 [当前状态](docs/status.md)，不把任务`partial`自动等同工程未完成，也不承诺任意长任务成功。
- 文字默认，损坏公式或图中关系缺口支持Supervisor按需委派视觉补读；固定复杂页对照及调度证据见 [评测说明](docs/evaluation.md)。视觉输出仍可不完整或出错，源码接入不等于当前演示进程已更新。
- 结构化矩阵保留为实验实现：工程/隔离验证已有，真实模型交付未通过，未作为稳定亮点。
- 检索优化候选出现质量退化，未切换默认路线；原优化评测为6份PDF/24题，后续分层扩库和模拟使用的结果与局限见 [评测说明](docs/evaluation.md)，不能外推为真实用户效果或大规模泛化。
- 不包含多用户鉴权、公网托管、Obsidian 同步、引用次数排序或任意任务零错误保证。

## 开始使用

依赖：Python 3.14、uv、Node.js/npm、Docker。当前锁文件与本机验证环境不等于所有平台兼容；本地重排冻结权重面向 ARM64。

```bash
git clone https://github.com/CsBillChang/ReasearchLoop.git
cd ReasearchLoop
cp .env.example .env
make sync-index
(cd frontend && npm ci)
make infra
make migrate
```

在本机 `.env` 中填写模型连接信息。论文来源配置与本机向量模型**不随仓库分发**，必须按 [配置与运行手册](docs/delivery.md) 配置；缺失时明确不可用/降级，不冒充完整检索。仓库不附带私人研究数据；首次启动是空白应用。

```bash
make start     # API 18082 / 前端 4173 / Worker / Scheduler
make status
make diagnose  # 不调用生成模型
make stop      # 等待在途调用结算
```

打开 http://127.0.0.1:4173/ 。Scheduler 启动不等于订阅已启用；简报在个人面板配置，默认09:00上海时间、最多5篇候选，不自动精读。

## 开发与证据

```bash
make test      # 默认离线；外部依赖/真实调用测试有显式门控
make check     # Python / JavaScript 语法
```

| 入口 | 内容 |
|---|---|
| [开发完善计划](docs/improvement-plan.md) | 固定设计 → 核心闭环 → Reflection范围 → 视觉验证 → 使用评测 → 可复现发布；文件合同与完成条件 |
| [当前状态](docs/status.md) | 唯一当前执行状态 |
| [架构](docs/architecture.md) | 请求链、角色、工具、持久化与授权边界 |
| [文件地图](docs/file-map.md) | 目录职责、后端关键文件与请求调用链 |
| [评测与数据](docs/evaluation.md) | 真实小样本结果、取舍与后续模拟使用方案 |
| [公开证据与演示](docs/evidence/README.md) | 匿名排名/标签/计时离线复算，正常比较、压缩追问与partial记录概述 |
| [参考与归属](docs/references.md) | 实际复用与仅设计参考分别记录 |
| [运行与发布](docs/delivery.md) | 启停、备份恢复、上传清单及数据边界 |

源码、依赖锁与脱敏结果公开；PDF、模型权重、原始运行轨迹和本机历史文档不公开。真实上游访问和生成模型可能产生费用，测试通过不代表所有来源可用。DPR 复用归属见 [第三方说明](THIRD_PARTY_NOTICES.md)。
