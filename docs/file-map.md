# 文件职责与阅读顺序

本页是当前源码地图，不是重新搬迁目录的任务。后端保留八层，已核对最新 HTTP 拆分；未来修改范围见 [improvement-plan.md](improvement-plan.md)；不再维护重复的逐文件清单。以下路径相对于仓库根，后端模块路径相对于 `backend/research_loop/`。

## 仓库顶层

| 路径 | 放什么 / 负责什么 |
|---|---|
| `backend/research_loop/` | 运行中的 Python 业务源码，见下表 |
| `backend/migrations/` | 数据库追加迁移；已有表和迁移不因整理目录而重置 |
| `backend/tests/` | unit 纯函数/合同，integration 服务及持久化，acceptance 显式真实验收；替身、隔离库和真实模型区分 |
| `backend/scripts/` | 启动、诊断、备份、冻结评测和报告；历史 runner 不是默认启动或 CI 入口 |
| `frontend/src/` | 聊天、个人面板、附件/成果侧栏与 API/SSE 客户端；当前不重新设计前端 |
| `backend/pyproject.toml` / `backend/uv.lock` / `.python-version` | Python依赖声明/锁在backend，版本入口在根，环境统一在根`.venv`；不要另建pip/conda说明 |
| `Makefile` / `compose.yaml` / `.env.example` / `config/` | 命令入口、MySQL/Qdrant 基础设施、无凭据配置模板；本机真实 `.env` 不上传 |
| `docs/` | 当前架构、范围、状态、运行手册、参考归属及评测口径 |
| `data/` | 本机 PDF、缓存、模型等运行材料；不上传、不删除 |
| `artifacts/` | 原始验收、失败、收据、截图、代码冻结和文档整理快照；不上传、不改旧结果 |
| 仓库外本机文档备份 | 历史合同与旧 archive 已迁出；位置见 status.md，不参与开发或发布 |
| `THIRD_PARTY_NOTICES.md` | 参考代码归属与完整 DPR 许可证原文；不再单设 licenses 目录 |

## 后端八层

| 层 | 关键文件 | 输入 → 输出 / 边界 |
|---|---|---|
| HTTP `api/` | `router.py`、`dependencies.py`、各资源路由 | HTTP 请求 → 服务调用/HTTP 响应；不安排研究步骤、不持有模型循环 |
| Agent `agents/` | `graph.py`、`state.py`、角色、`memory.py`、`reflection.py` | 目标、状态、工具观察 → 下一行动/角色结果；LangGraph 控制流 |
| 合同 `domain/` | `tasks.py`、`evidence.py`、`reflection.py`、`context.py`、`memory.py`、`reading.py`、`notes.py`、`retrieval.py` | 结构化参数、身份、权限、纯转换；reflection定义通用对象/核验/patch/结束及进展比较 |
| 工具 `tools/` | `base.py`、`registry.py`、`papers.py`、`delegation.py`、`notes.py`、`memory.py` | 模型 tool call → 校验后的服务调用 → Observation；schema/get/execute 共用合同 |
| 业务 `services/` | 任务、覆盖、论文、笔记、记忆、检索、反馈和简报等 | 业务请求 → 经版本、权限、事务规则约束的结果；不反向依赖资源路由 |
| 适配 `providers/` | `deepseek.py`、DPR、`pdf.py`、`paper_media.py`、`embeddings.py`、`reranker.py` | 外部接口/文件/本机推理 → 标准结果和错误；不决定任务是否完成 |
| 存储 `storage/` | `mysql.py`、`files.py`、`checkpoints.py`、`memory.py`、各索引存储 | 业务状态与收据持久化、文件保护、派生索引；数据库是业务事实 |
| 执行 `workers/` | `executor.py`、`scheduler.py` | 已排队任务/到期订阅 → 领取、租约、图执行、恢复、结算和交付 |

## HTTP：本次结构调整已经落在哪里

```text
main.py                        应用生命周期、中间件、挂载 /api
api/router.py                  唯一资源路由登记
api/dependencies.py            Depends：配置、数据库、服务
api/tasks.py                   创建、查询、取消、澄清恢复、接续、上下文诊断
api/conversations.py           会话列表、消息和名称
api/uploads.py                 PDF 上传流及 HTTP 限制
api/events.py                  公开事件 JSON、SSE、游标与终态快照
api/artifacts.py               授权证据/PDF/稿件与保存确认
api/feedback.py                版本绑定反馈入口
api/notes.py                   笔记版本与差异
api/profile.py                 偏好、兴趣及研究记录
api/subscriptions.py           增量简报配置/结果
api/reading_list.py            阅读清单
api/health.py                  健康入口
```

依赖方向：`main → api/router → 资源路由 → dependencies + services + domain`。配置/数据库/服务通过请求内 Depends 装配；不能再导入 `api/tasks.py` 的私有工厂。

- `services/artifacts.py`：成果投影、当前稿保存、上下文诊断；保存规则不留在 HTTP 路由。
- `services/uploads.py`：PDF 验证、上传身份和原子文件存储。
- `services/feedback.py`：原结果/当前稿的版本检查、反馈与授权材料交接。
- `services/continuations.py`：取消后的新要求接续；`storage/continuations.py` 保存相关 schema。
- 旧 `api/product.py`、`services/product.py`、`storage/product.py` 已移除；历史 artifact 中的路径按当时版本理解，不回写历史证据。

这些是已有结构调整，不能计为上下文或研究成果交接已经修好。

## 核心模块怎么读

| 顺序 | 文件与关键入口 | 应理解的职责 |
|---|---|---|
| 1 | `api/tasks.py`、`services/tasks.py`、`services/conversations.py` | 指令/附件/权限/幂等如何入队，澄清为什么沿用原任务，后续任务如何绑定会话 |
| 2 | `storage/mysql.py`、`storage/checkpoints.py` | task、action、receipt、租约和快照怎么存；成功响应和未知请求怎么区分 |
| 3 | `workers/executor.py` 的 `execute` | 从任务/Checkpoint 组装服务和角色图；结束后把什么写回用户可见结果 |
| 4 | `agents/graph.py` 的 `build_graph` | decide→execute_tools→观察→再决策；模型响应先记录后解析，恢复不盲重发 |
| 5 | `agents/supervisor.py`、`tools/delegation.py`、`services/coverage.py` | 用户目标、计划、委派、澄清、覆盖与 finish_task；结构覆盖不等于语义通过 |
| 6 | `agents/researcher.py`、Worker `_run_research_batch` | 独立子目标/工具范围、并行研究、finish_research；阶段进展行动、正常与异常交接；固定真实验证与覆盖局限见status，不把它泛化成所有任务成功 |
| 7 | `tools/papers.py`、`services/papers.py`、`providers/dpr_acquisition.py`、`providers/pdf.py` | 候选、正文缓存、PDF 页、原文证据与授权；有链接不等于读过全文 |
| 8 | `services/indexing.py`、`services/retrieval.py`、`domain/retrieval.py` | 版本化切分、双路召回、融合/重排、原文读回；Qdrant 是派生索引 |
| 9 | `agents/memory.py`、`domain/context.py`、`domain/context_workspace.py` | 原始档案与单次请求工作集分离；完整计入当前成果、历史和混合工具批次；档案截短保留、命中公共字段及同任务同版正文精确去重、相关摘录与位置读回；不合并旧版本，不改原件；byte估计与token配置的局限 |
| 10 | `services/memory.py`、`storage/memory.py`、`tools/memory.py` | 偏好/兴趣/研究历史、来源、撤销与归档读回；不同于单次模型上下文 |
| 11 | `domain/reflection.py`、`agents/reflection.py`、`agents/reviewer.py`、`services/verification.py` | 草稿及候选/阶段发现/回答独立核验；实际元数据/引句/身份检查、逐项进展、局部版本和结束。来源仍用既有evidence合同；字符串匹配不替代语义判断 |
| 12 | `services/notes.py`、`services/note_versions.py`、`storage/files.py` | 创建数据库稿件、用户确认、保存文件是三件事；deny 不写笔记文件 |
| 13 | `services/subscriptions.py`、`services/recommendations.py`、`workers/scheduler.py` | 个性化候选简报、曝光/已读去重、上海 09:00、停机合并补报 |
| 14 | `api/events.py`、`api/artifacts.py`、`frontend/src/api.js` | 持久事件和答案增量如何呈现；断线重放不是重新运行任务 |

`agents/deep_reading.py → providers/dpr_summary.py` 是用户明确要求精读时的专用路径。普通比较不应为了复用该节点而被扩展成整篇精读。`services/evidence_matrix.py` 与 `domain/research.py` 中的矩阵合同为实验能力，不作为核心修复的前置要求。

## 一条普通比较请求的具体路径

```text
frontend/src/app.js 提交 prompt + conversation_id + 附件 + save_policy
  → api/router.py → api/tasks.py + api/dependencies.py
  → services/tasks.py / conversations.py → storage/mysql.py 入队
  → workers/executor.py 领取/恢复
  → agents/graph.py decide
      → agents/memory.py 组装目标、角色上下文及原文投影
      → providers/deepseek.py 返回行动
      → tools/registry.py 校验并执行
          → tools/delegation.py → Worker → Researcher 子图
          → tools/papers.py → services/papers.py / retrieval.py
          → （按需）Reviewer / Reflection
      → action、receipt、Checkpoint → 下一次 decide
  → finish_research 交回子成果 → Supervisor 汇合 → finish_task
  → Worker / TaskService 落终态 → ConversationService 对话读回
  → api/events.py SSE + api/artifacts.py 成果展示
```

阶段发现现由 `domain/research_progress.py`、`services/research_progress.py` 及 `record_research_progress` 工具管理：成功 action 是持久真值，图快照是投影。上下文装配、重复取证观察、Worker 正常/异常交接及会话研究记录已经接入。`domain/context.py` 维护多个互补来源窗口；`services/research_progress.py` 从成功行动识别无进展读回，`agents/graph.py` 使用已有阶段记录/结束工具完成交接或partial结束。代码状态不等于真实语义验收，结果统一见 status.md。

取证可靠性入口为`services/research_progress.py:evidence_spans/visible_evidence/evidence_progress/readback_handoff`：比较精确覆盖、区分当前可见与未知，并累计同批工具观察。`agents/graph.py:execute_tools`使用原request_json装配可见性，request_sequence只标识批次，不是新额度。旧行动标签保持，恢复沿用原观察。

`domain/context_workspace.py` 不生成摘要：先对同版、同值的交接缺口建立引用，再在容量压力下外置诊断详情和旧研究摘录；必要时才对超长候选正文做明确标记的摘录。用户当前要求、澄清、权限以及每条发现的来源/核验状态保持。`agents/memory.py` 先冻结完整档案再计算字符定位，`services/memory.py` 复用现有任务隔离的分页读回；没有 `read_memory` 工具时不外置这些详情。原始成功行动、候选和证据不随投影修改。

通用Reflection复用`reflection_sessions.state_json.research_targets`及已有subtasks/actions/model_calls/checkpoints，无新增表。`services/research_progress.py:reviewable_findings`只读同根已提交发现；后续局部版本保存在Reflection对象中，子任务结果不改写。`tools/delegation.py`注册review_research/revise_research/close_research_issue；`workers/executor.py:_review_research_once`装配当前快照、实际要求和原文，复用成功子结果及收据；`services/coverage.py`允许按当前快照hash直接交付，拒绝额外改文，并展示未解决范围。旧覆盖交付之前的快照保留原工具集合，避免新增schema改变恢复合同或挤占上下文。

`services/verification.py:validate_target_assessments`检查每个对象至少被覆盖一次，允许同对象多条判断并逐条核对来源；`domain/reflection.py:review_progress`按对象保留全部状态，`agents/reflection.py`给每条意见独立展示身份。`workers/executor.py:research_review_failure`将原始工具错误和最后停止原因一起传给父角色。`domain/tasks.py`与`services/coverage.py:finish`分离正文/processing_summary/核验说明，Worker与API读回保持这些字段；不新增全稿批准要求或文件保存权。

`agents/graph.py:handoff_schemas/handoff_navigation`保持交接时实际档案读回可用，按来源版本与字符区间记录导航，不把导航当研究进展。`services/conversations.py:answer_message/referenced_answer`解析同会话且早于当前输入的明确回答身份；`workers/executor.py:_deliverable_result`在缺少当前成果时只保留这一明确来源，API结果以retained_answer_source标明历史身份和未核验状态。没有明确来源的普通任务仍返回自身失败原因，不采用任意最近回复。

当前核验交接修复沿用上述回答解析：`agents/reflection.py:_research_source/_validate_answer_source`在省略`answer_markdown`时加载明确引用的回答，绑定原消息及内容hash，并在当前target复用时检查来源仍有效。新候选正文和旧target/hash保持兼容，局部新版本优先；历史回答不充当原文依据或保存权限。

`agents/memory.py:_current_review_answer`只加载明确来源或唯一当前answer版本；`domain/context.py:project_review_answer`在当前任务中保留完整正文，并对精确相等的重复副本建立本请求内引用。原消息、旧版本和外部会话不被替换；无明确引用时不默认选最近回答。`workers/executor.py:research_review_failure`保留所有错误，同时区分最后结果提交错误与历史首错；`_review_research_once`向独立Reviewer注入带检查范围的实际程序记录，不能把保存/差异约束当论文事实或仅靠补证据ID批准。

`agents/state.py:pending_tool_scope`保存当前响应的实际工具范围，`agents/graph.py:execute_tools`恢复优先读取原收据中的范围，逐调用返回准入结果。范围拒绝不执行或消耗工具槽，独立`read_memory`可继续，后续可能依赖失败前序的状态操作延后到下轮。错误观察不刷新科研进展，失败取证也不能清除交接边界；重复非法操作合理结束。Reviewer组件测试位于`tests/integration/test_review_execution_handoff.py:ReviewerComponentTests`，Supervisor正式API/Worker链由同文件`SupervisorHandoffTests`独立验证；两者使用供应商与PDF替身，不代表真实语义验收。

核验回传的单份投影位于`domain/context.py:project_review_handoffs`：当前状态保留完整findings与target；issues/pending_checks对精确相等的字段建立本请求内路径，只有整份工具结果与当前状态逐字段相等时才引用`current_task.research_reviews`。不同版本、额外诊断或失败观察不据同ID直接合并。`agents/memory.py:prepare`仅在既有压缩阈值触发时应用此投影，并在工作集回退中保持；原始档案先固定，`original_result_readback`用精确JSON路径定位完整原件。没有read_memory权限时只保留本地无损引用，不生成读回入口。业务核验原件、判断、版本、调用参数和供应商推理字段均不变；新合同与分层替身测试见`tests/unit/test_review_handoff_projection.py`、`tests/integration/test_review_handoff_projection.py`。结果和真实验证边界仍以status.md为准。

## 第三项视觉补读组件的位置

以下包含独立组件及Supervisor按需委派接线；范围及调用配置以`improvement-plan.md`顶部方案为准，实际验证层次见status.md。

- `providers/paper_media.py:render`从已校验PDF字节渲染全页/区域PNG，记录物理页、坐标、像素、版本与hash；原`extract`和DPR回退不变。`services/papers.py:prepare_visual`只接受当前任务绑定的本地附件与既有页缓存，复核身份和页权限，不联网取文或回退重解析。
- `domain/visual_reading.py`定义输入与组件配置、A/B请求装配；文字相同，视觉组增加user图片块。`agents/visual_reading.py`负责一次定向调用、原子发送窗口、收据复用和紧凑结果快照；复用`providers/deepseek.py`，模型观察不覆盖原文或自动成为核验事实。
- `storage/token_budget.py:estimate_details`对合法内联PNG单独计量，验证尺寸/字节限制并使用图片Token保守上界；纯文字估算原样保留，原有额度和实际usage结算不变。
- `scripts/evaluate_visual.py`将preflight/freeze/run/report分开；默认零模型，真实执行需显式开关及冻结文件验证。unit/integration的`test_visual_reading.py`覆盖渲染、输入、权限、请求、并发、收据和恢复。原始请求留在隔离收据，图片不进入日常上下文。
- `tools/papers.py:delegate_visual_reading`只向Supervisor开放，输入真实evidence_id、具体问题、原因和两类缺口；`services/papers.py:visual_input_for_evidence`从已授权物理页派生文档/缓存身份，不能由模型指定文件路径或hash。
- `workers/executor.py:_visual_once`复用父子任务，为选中页装配最小授权，执行`agents/visual_reading.py:build_visual_graph`的一次Researcher视觉节点；相同来源/问题/裁区复用已有收据，失败和unknown不重发。父子结果交接与租约检查同事务。
- `domain/visual_reading.py:compact_observation`仅向父角色返回6000字符观察窗口与来源/收据，原回答留在子任务；同一委派带offset继续读回，不新增视觉请求。完整观察供异常保留，base64与页全文不进入父观察。
- 新`test_visual_delegation.py`验证正常API/Worker、两种默认/按需路线、权限、预算、恢复与失败收尾。实际自然语言选择和日常端到端视觉效果仍需单独真实验证；不自动增加Agent、前端入口或强制审校。

## 文档职责与清理规则

`docs/` 仅保留八份文档：

- `architecture.md`：架构、角色、调用链和数据归属；`file-map.md`：当前文件导航。
- `improvement-plan.md`：唯一未来范围、文件输入输出和验收条件；`status.md`：唯一当前结论及证据索引。
- `delivery.md`：启动、停止、诊断、备份、恢复与发布规则。
- `references.md`：参考项目、实际复用和模型归属。
- `evaluation.md`：评测口径、测量值、失败分母与局限。
- `resume-evidence.md`：简历亮点与代码、验证结果、局限的映射。

重复清单已删除；历史导入/采样合同与旧 archive 已迁入仓库外本机备份，定位见 status.md。发布说明合并进 delivery.md，DPR 许可证原文合并进根目录 THIRD_PARTY_NOTICES.md；原独立文件与 licenses 目录已删除。原始运行数据、成果、receipt 与 artifacts 不删除、不修改。

上下文工具历史的思考模式接续位于`domain/context.py:compact_tool_exchange`和`agents/memory.py:prepare/_restore_sources`：优先只缩短工具观察，真实assistant及reasoning_content、tool_calls与对应tool_call_id原样保留。旧assistant档案摘要先按当前任务的原档案还原，不能补造推理字段。长参数仍超限时按既有工作集整组归档，保留最新完整往返；来源读回权限、档案身份与原文版本继续校验。`tests/unit/test_thinking_context.py`和`tests/integration/test_thinking_context.py`分别覆盖纯投影与API→Worker→HTTP/SSE替身、旧摘要、成功收据恢复及unknown阻断，不能替代真实供应商复验。
