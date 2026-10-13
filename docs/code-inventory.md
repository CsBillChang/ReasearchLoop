# 全部发布文件清单

由本轮实际文件生成；包括源码、脚本、测试、迁移、依赖锁、模板与活动文档。未包含运行数据/凭据/权重/缓存/本机历史归档。不是全仓库逐行代码正确性审计。

| 文件 | 分类 | 当前职责/说明 |
|---|---|---|
| `.env.example` | 文档/配置 | 详见所属目录说明 |
| `.gitignore` | 文档/配置 | 详见所属目录说明 |
| `.python-version` | 文档/配置 | 详见所属目录说明 |
| `AGENTS.md` | 文档/配置 | ResearchLoop 协作约定 |
| `Makefile` | 文档/配置 | 详见所属目录说明 |
| `README.md` | 文档/配置 | ResearchLoop |
| `THIRD_PARTY_NOTICES.md` | 文档/配置 | 第三方代码与模型归属 |
| `artifacts/README.md` | 文档/配置 | 本机验证记录（不上传） |
| `backend/README.md` | 文档/配置 | ResearchLoop 后端 |
| `backend/migrations/001_initial.sql` | 迁移 | 详见所属目录说明 |
| `backend/migrations/002_runtime_recovery.sql` | 迁移 | 详见所属目录说明 |
| `backend/migrations/003_multi_agent_tasks.sql` | 迁移 | 详见所属目录说明 |
| `backend/migrations/004_note_exports.sql` | 迁移 | 详见所属目录说明 |
| `backend/migrations/005_reflection.sql` | 迁移 | 详见所属目录说明 |
| `backend/migrations/006_memory.sql` | 迁移 | 详见所属目录说明 |
| `backend/migrations/007_product_continuations.sql` | 迁移 | 详见所属目录说明 |
| `backend/migrations/008_evidence_index.sql` | 迁移 | 详见所属目录说明 |
| `backend/migrations/009_subscriptions.sql` | 迁移 | 详见所属目录说明 |
| `backend/migrations/010_closeout_foundation.sql` | 迁移 | 详见所属目录说明 |
| `backend/migrations/README.md` | 迁移 | 数据库迁移 |
| `backend/pyproject.toml` | 文档/配置 | 详见所属目录说明 |
| `backend/research_loop/__init__.py` | __init__.py | Python包入口 |
| `backend/research_loop/agents/__init__.py` | agents | Python包入口 |
| `backend/research_loop/agents/deep_reading.py` | agents | DPR 精读单节点：上游生成函数不改提示与循环，外层负责已有收据和取消。 |
| `backend/research_loop/agents/graph.py` | agents | 通用行动循环：模型决定下一步，程序校验并执行工具，再把观察交回模型；同时记录恢复状态。 |
| `backend/research_loop/agents/memory.py` | agents | 会话上下文装配与可追溯压缩；不调用LLM，不改变DPR正文输入或历史收据。 |
| `backend/research_loop/agents/reflection.py` | agents | Reflection 持久控制：按需独立核验、精确局部修订、任务级轮次及无进展结束。 |
| `backend/research_loop/agents/researcher.py` | agents | Researcher 子图：只搜索/读取授权论文并提交结构化发现；没有创建笔记或保存文件的权限。 |
| `backend/research_loop/agents/reviewer.py` | agents | Reviewer 子图：独立核对当前草稿及原文，允许补读；每个子图核验一次，不自动改写；修复轮次由根任务持久控制。 |
| `backend/research_loop/agents/state.py` | agents | LangGraph 共享状态：跨节点传递的消息、待执行工具、证据与结束原因都须在此声明。 |
| `backend/research_loop/agents/supervisor.py` | agents | Supervisor 上下文：注入目标、附件和保存权限；工具观察会压缩后进入下一次决策。 |
| `backend/research_loop/api/__init__.py` | api | Python包入口 |
| `backend/research_loop/api/conversations.py` | api | 会话API：创建、列出、分页读回；提问仍走POST /tasks，澄清走原任务resume。 |
| `backend/research_loop/api/health.py` | api | 健康接口：检查服务可响应与配置是否存在，不代表数据库、PDF 或模型请求已通过实测。 |
| `backend/research_loop/api/notes.py` | api | 项目笔记版本API：差异预览与明确确认保存分离，旧hash和用户改动阻断覆盖。 |
| `backend/research_loop/api/product.py` | api | AG5网页适配：有界PDF上传、只读SSE、授权证据、草稿保存及版本绑定反馈。 |
| `backend/research_loop/api/profile.py` | api | 个人画像接口：显式编辑、推断撤销与来源统计，不引入登录/多租户承诺。 |
| `backend/research_loop/api/reading_list.py` | api | 阅读清单HTTP接口；不调用模型、不执行PDF下载或笔记保存。 |
| `backend/research_loop/api/subscriptions.py` | api | 个人面板订阅和聊天简报API；配置操作不会立即发送外部请求。 |
| `backend/research_loop/api/tasks.py` | api | 任务 HTTP 接口：校验提交、查询状态、请求取消和恢复澄清；研究执行留给 Worker。 |
| `backend/research_loop/config.py` | config.py | 运行配置：合并环境变量和 DPR 来源配置；本模块不负责启动服务或验证供应商可用性。 |
| `backend/research_loop/domain/__init__.py` | domain | Python包入口 |
| `backend/research_loop/domain/context.py` | domain | 上下文摘录纯函数：保留原句和省略标识，不生成新事实。 |
| `backend/research_loop/domain/evidence.py` | domain | 原文数据契约：候选元数据、PDF 缓存身份与页级证据分开表达；有链接不等于已读取正文。 |
| `backend/research_loop/domain/memory.py` | domain | 会话、个人记忆和笔记版本参数合同；Pydantic类说明不进入工具提示。 |
| `backend/research_loop/domain/notes.py` | domain | 笔记数据契约：草稿、核验结论和文件保存结果相互独立，均以当前内容 hash 绑定。 |
| `backend/research_loop/domain/reading.py` | domain | 从用户原始指令提取明确的物理页限制，不把 Agent 的阅读批次当成权限。 |
| `backend/research_loop/domain/research.py` | domain | 【实验暂缓】可选研究矩阵参数：表达与证据分离，结构检查不充当科学事实核验。 |
| `backend/research_loop/domain/retrieval.py` | domain | 原文索引契约：身份和定位先于检索评分；不授予论文或页级读取权限。 |
| `backend/research_loop/domain/subscriptions.py` | domain | 订阅输入与日历计算；所有持久时刻为UTC，调度日按订阅时区计算。 |
| `backend/research_loop/domain/tasks.py` | domain | 任务数据契约：区分运行状态、保存权限、工具观察和模型收据，供 API、服务与 Agent 共用。 |
| `backend/research_loop/main.py` | main.py | 应用入口：组装 FastAPI 路由；长任务由独立 Worker 执行，不在 HTTP 请求内调用模型。 |
| `backend/research_loop/providers/__init__.py` | providers | Python包入口 |
| `backend/research_loop/providers/citations.py` | providers | 【退役兼容】历史引用统计适配器：保留诊断/测试代码；当前 Worker 不接入，被引量已退出日常研究能力。 |
| `backend/research_loop/providers/deepseek.py` | providers | 模型适配器：发送兼容 Chat Completions 的请求，保留原始响应；解析与收据持久化分别处理。 |
| `backend/research_loop/providers/discovery.py` | providers | 发现适配器：调用 DPR 配置的逐源 Supabase RPC，统一候选字段并保留每个来源的查询状态。 |
| `backend/research_loop/providers/dpr_acquisition.py` | providers | DPR 论文地址适配；仅定位 PDF，不解析正文或向其他服务搜索替代论文。 |
| `backend/research_loop/providers/dpr_sources.py` | providers | DPR 来源目录：来源别名、RPC 名称与连接配置的统一映射；配置存在不代表来源真实可用。 |
| `backend/research_loop/providers/dpr_summary.py` | providers | 复用 DPR 的八点精读与 Markdown 元数据纯函数。 |
| `backend/research_loop/providers/embeddings.py` | providers | 本机ONNX编码适配：只读固定文件，不下载模型、不执行远端模型代码。 |
| `backend/research_loop/providers/paper_media.py` | providers | DPR 图表提取适配：可选 PaperCropper，失败回退 PyMuPDF 内嵌图片。 |
| `backend/research_loop/providers/pdf.py` | providers | DPR 正文获取：有效缓存→Jina→PDF/PyMuPDF；保留真实页码，不推断远端文本的 PDF 页。 |
| `backend/research_loop/providers/read_retry.py` | providers | DPR只读请求退避策略；不包装模型调用或有副作用的写接口。 |
| `backend/research_loop/providers/reranker.py` | providers | 本机Cross-encoder重排：只输出相关性logit，不生成答案或作事实判决。 |
| `backend/research_loop/providers/streaming.py` | providers | Chat Completions 流装配和公开文本提取；不把内部推理或工具参数整体发给浏览器。 |
| `backend/research_loop/services/__init__.py` | services | Python包入口 |
| `backend/research_loop/services/continuations.py` | services | 取消后继续：持久化后续要求，旧任务结算后幂等入队，不依赖浏览器存活。 |
| `backend/research_loop/services/conversations.py` | services | 会话服务：用户消息和任务原子关联，澄清继续原任务，结束结果可幂等补投影。 |
| `backend/research_loop/services/coverage.py` | services | 研究目标覆盖：复用行动日志保存计划，核对交付事实，不进行科学事实审校。 |
| `backend/research_loop/services/evidence_matrix.py` | services | 【实验暂缓】把已读原文组织成不可变矩阵；复用工具行动存储，无额外模型或文件写入。 |
| `backend/research_loop/services/feedback.py` | services | 当前稿反馈交接：原子入队并复用确切稿件和原文，不复制旧结论或执行状态。 |
| `backend/research_loop/services/indexing.py` | services | 已授权原文→不可变片段→持久词项/向量；索引不扩大任务的读取权限。 |
| `backend/research_loop/services/memory.py` | services | 用户画像和历史召回：保留来源、版本与撤销记录，不把记忆当作论文原文。 |
| `backend/research_loop/services/note_versions.py` | services | 项目笔记的只读查询、差异预览和不可变版本保存；保留原文件与审校历史。 |
| `backend/research_loop/services/notes.py` | services | 笔记服务：保留原始 Markdown，校验身份/版本并按授权保存；审校只作为独立记录。 |
| `backend/research_loop/services/papers.py` | services | 论文服务：串接检索、授权、PDF 解析与页证据持久化；只向任务返回它有权读取的内容。 |
| `backend/research_loop/services/product.py` | services | 产品投影与上传：从权威任务/文件构建界面数据，不触发新的研究或模型请求。 |
| `backend/research_loop/services/reading_list.py` | services | 聊天中的个人阅读清单：只记录用户动作，不自动执行研究或改变核验状态。 |
| `backend/research_loop/services/recommendations.py` | services | 可解释的候选排序与简报渲染；仅使用元数据，不生成全文结论或被引量。 |
| `backend/research_loop/services/retrieval.py` | services | 任务授权范围内的BM25/向量召回、RRF和本机重排；不把检索分数当成事实结论。 |
| `backend/research_loop/services/runtime.py` | services | 本机进程边界：应用共享锁与冷备份独占锁；不负责模型调度或修改业务状态。 |
| `backend/research_loop/services/subscriptions.py` | services | 个性化简报服务：配置、逐源收据、窗口推进和聊天交付共用持久事务。 |
| `backend/research_loop/services/tasks.py` | services | 任务生命周期：幂等创建、只读视图、取消、澄清恢复及最终落库；不在此执行研究模型。 |
| `backend/research_loop/services/verification.py` | services | 定向核验的证据合同检查；检查身份与引句，不冒充语义蕴含判断。 |
| `backend/research_loop/storage/__init__.py` | storage | Python包入口 |
| `backend/research_loop/storage/checkpoints.py` | storage | 图状态存储：保存应用自己的 state JSON；它辅助恢复，不替代业务收据、文件核验或幂等控制。 |
| `backend/research_loop/storage/evidence_index.py` | storage | 原文索引的MySQL事实与构建锁；不改写既有论文、页证据或任务记录。 |
| `backend/research_loop/storage/files.py` | storage | 文件适配器：管理 PDF/解析缓存和项目笔记；限制目录，同内容复用，异内容拒绝覆盖。 |
| `backend/research_loop/storage/indexes.py` | storage | 可重建的词项检索：权威记录在MySQL；此模块只给已过滤的材料计算BM25。 |
| `backend/research_loop/storage/memory.py` | storage | AG4持久层：会话、记忆和版本记录共用事务，不替代任务收据。 |
| `backend/research_loop/storage/mysql.py` | storage | 持久化适配器：生产使用 MySQL，隔离测试可使用 SQLite；保存任务事实、行动、收据、证据与租约。 |
| `backend/research_loop/storage/product.py` | storage | 产品交互持久层：保存取消后继续请求；模型状态仍由任务和收据负责。 |
| `backend/research_loop/storage/reading_list.py` | storage | 最小阅读清单持久结构；论文状态不修改研究/核验/笔记记录。 |
| `backend/research_loop/storage/subscriptions.py` | storage | 简报持久表；与研究任务、模型收据和原文索引分开，不改变它们的事实。 |
| `backend/research_loop/storage/token_budget.py` | storage | 根任务累计Token账本：先预留、后结算；收据是真值，恢复不重复扣款或发送。 |
| `backend/research_loop/storage/vector_index.py` | storage | Qdrant原文索引适配：显式集合身份、确定性point ID和实际读回，不调用LLM。 |
| `backend/research_loop/tools/__init__.py` | tools | Python包入口 |
| `backend/research_loop/tools/base.py` | tools | 工具契约底座：同一个 ToolSpec 生成模型 schema 并绑定执行函数，防止声明与执行入口脱节。 |
| `backend/research_loop/tools/delegation.py` | tools | Supervisor 专属工具：记录能力缺口、请求澄清、分派 Researcher 和调用独立 Reviewer。 |
| `backend/research_loop/tools/memory.py` | tools | Supervisor记忆工具：确定性读写，LLM负责理解意图，数据库负责来源和权限。 |
| `backend/research_loop/tools/notes.py` | tools | 笔记工具：create_note 生成数据库草稿，save_note 才请求项目内文件写入，二者不可混淆。 |
| `backend/research_loop/tools/papers.py` | tools | 论文工具：约束已分派论文身份；正文默认全文，页限制只取自用户要求。 |
| `backend/research_loop/tools/registry.py` | tools | 工具注册表：按角色暴露 schema，调用前再次检查角色权限，再进入参数校验与服务执行。 |
| `backend/research_loop/workers/__init__.py` | workers | Python包入口 |
| `backend/research_loop/workers/executor.py` | workers | 执行编排：领取任务、维护租约、组装各角色工具、运行 LangGraph，再持久化用户可查询的结果。 |
| `backend/research_loop/workers/scheduler.py` | workers | 本机简报调度器：只读DPR元数据，有限请求与租约恢复，不调用研究Agent或模型。 |
| `backend/scripts/README.md` | 脚本 | 脚本分类 |
| `backend/scripts/backup_restore.py` | 脚本 | 本机冷备份与隔离恢复；不覆盖已有数据库、文件，不自动启动任务或重发请求。 |
| `backend/scripts/check_publication.py` | 脚本 | 检查Git暂存内容的公开边界；不联网、不调用模型、不输出敏感值。 |
| `backend/scripts/evaluate_reflection.py` | 脚本 | 固定草稿的Reflection配对评测；零调用原稿基线与既有Agent局部修订链比较。 |
| `backend/scripts/evaluate_reflection_component.py` | 脚本 | AG7.4固定四稿的组件对照；复用产品核验/局部patch，不运行外围研究调度。 |
| `backend/scripts/evaluate_retrieval.py` | 脚本 | 固定六篇本地PDF的四路检索对照；标注先冻结，真实ONNX/Qdrant不调用生成模型。 |
| `backend/scripts/evaluation_common.py` | 脚本 | AG7.4评测共用的隔离、冻结和指标函数；不改变产品提示词或成功判定。 |
| `backend/scripts/import_dpr_sources.py` | 脚本 | Copy only DPR's read-only source config, never execute its scripts or import .env. |
| `backend/scripts/index_evidence.py` | 脚本 | 显式构建授权原文索引：只读既有页，不下载论文、不调用生成模型或启动Agent。 |
| `backend/scripts/manage.py` | 脚本 | 本机单用户启停、状态、诊断；只操作带确切身份记录的本项目进程。 |
| `backend/scripts/migrate.py` | 脚本 | Explicit, idempotent migration entry point. |
| `backend/scripts/optimize_retrieval.py` | 脚本 | 固定旧材料上的一次有界检索优化；仅本机模型和只读Qdrant，禁止外部连接。 |
| `backend/scripts/preflight_retrieval_models.py` | 脚本 | AG7.1本机模型组件预检；不连接数据库、论文源或生成模型。 |
| `backend/scripts/probe_sources.py` | 脚本 | Explicit zero-model DPR source probe; never runs maintenance jobs. |
| `backend/scripts/report_evaluation.py` | 脚本 | 只读重算AG7.4已保存结果与账本；不接数据库、供应商或重新执行Agent。 |
| `backend/scripts/report_reflection_component.py` | 脚本 | 只读重算AG7.4组件对照的账本、版本及固定分母；语义标注由独立原文核对提供。 |
| `backend/scripts/run_ag2.py` | 脚本 | Explicit AG2 acceptance runner. |
| `backend/scripts/run_ag2_adaptive.py` | 脚本 | AG2三个场景的一次有界真实验证：隔离API/Worker/数据库，禁止重放已启动场景。 |
| `backend/scripts/run_ag3.py` | 脚本 | AG3 有界真实验收：固定附件/问题，隔离库与产物，API→Worker，禁止自动语义重试。 |
| `backend/scripts/run_ag4.py` | 脚本 | AG4固定真实会话验收：隔离MySQL，最多四任务/每题四请求，无论文搜索或DPR。 |
| `backend/scripts/run_ag5.py` | 脚本 | AG5一次有界真实产品回归：上传PDF→问答→同会话反馈，只在显式隔离库运行。 |
| `backend/scripts/run_ag6.py` | 脚本 | AG6综合评测：冻结场景、隔离执行和证据导出，不替业务Agent完成研究。 |
| `backend/scripts/run_ag7.py` | 脚本 | AG7.5本机交付验证：先隔离启停，再冻结已有PDF两问与最多12次真实请求。 |
| `backend/scripts/run_dpr_consolidation.py` | 脚本 | 一次固定自然语言验收：真实 API→隔离 MySQL→Worker，不预选论文或替模型排步骤。 |
| `backend/scripts/run_scheduler.py` | 脚本 | 启动本机简报轮询；只有已启用且到期的订阅才查询DPR，不调用生成模型。 |
| `backend/scripts/run_worker.py` | 脚本 | 独立Worker入口：指定任务、单次领取或持续轮询；保留权限、收据和优雅停机边界。 |
| `backend/scripts/verify_ag71.py` | 脚本 | AG7.1隔离真实验收：已有PDF/缓存、本地编码、MySQL/Qdrant与SIGKILL，不调用LLM。 |
| `backend/scripts/verify_ag72.py` | 脚本 | AG7.2固定组件与Agent接线验收：已有两篇PDF，独立MySQL/Qdrant，拒绝覆盖批次。 |
| `backend/scripts/verify_ag73.py` | 脚本 | AG7.3冻结验收：隔离库、只读DPR、有界两窗、真实进程退出与资产读回。 |
| `backend/scripts/verify_ag75_followup.py` | 脚本 | AG7.5最小补验：沿用真实首问和PDF，只追加一次同会话追问。 |
| `backend/scripts/verify_dpr_followup.py` | 脚本 | 复用已下载正文验证精读配置，再独立执行用户的 CVPR 检索输入。 |
| `backend/scripts/verify_reranker_efficiency.py` | 脚本 | 只验证一次固定推理优化：旧排序、24候选、旧材料，禁止业务模型与重复实验。 |
| `backend/tests/acceptance/README.md` | 测试/夹具 | acceptance |
| `backend/tests/acceptance/__init__.py` | 测试/夹具 | Python包入口 |
| `backend/tests/acceptance/ag6_cases.json` | 测试/夹具 | 详见所属目录说明 |
| `backend/tests/acceptance/ag74_cases.json` | 测试/夹具 | 详见所属目录说明 |
| `backend/tests/acceptance/run_controlled_reroute.py` | 测试/夹具 | Run the AG1 controlled-PDF-failure case with the real DeepSeek provider. |
| `backend/tests/acceptance/test_ag1.py` | 测试/夹具 | Evidence gate for the explicitly-run AG1 real acceptance. |
| `backend/tests/acceptance/test_ag2.py` | 测试/夹具 | AG2 验收入口测试：核对固定场景与证据合同，真实调用须显式授权开启。 |
| `backend/tests/acceptance/test_ag6_contract.py` | 测试/夹具 | AG6评测器合同：预算、固定分母、隔离和真实API/Worker接线，默认不联网。 |
| `backend/tests/acceptance/test_ag7_contract.py` | 测试/夹具 | 最终交付冻结合同；默认不运行真实模型或旧评测。 |
| `backend/tests/integration/README.md` | 测试/夹具 | integration |
| `backend/tests/integration/__init__.py` | 测试/夹具 | Python包入口 |
| `backend/tests/integration/clarification_process.py` | 测试/夹具 | SIGKILL exactly after durable tool checkpoint and before publishing needs_input. |
| `backend/tests/integration/collaboration_recovery_process.py` | 测试/夹具 | Isolated SIGKILL fixture; no real provider requests. |
| `backend/tests/integration/dpr_recovery_process.py` | 测试/夹具 | 精读节点的隔离 SIGKILL 子进程；供应商替身，绝不发真实模型请求。 |
| `backend/tests/integration/memory_recovery_process.py` | 测试/夹具 | AG4隔离故障注入进程；只有替身模型，数据库必须为专用测试库。 |
| `backend/tests/integration/recovery_process.py` | 测试/夹具 | Child process used by the explicit SIGKILL recovery test. |
| `backend/tests/integration/reflection_recovery_process.py` | 测试/夹具 | 隔离 SIGKILL 子进程；只调用供应商替身，保留数据库和请求计数。 |
| `backend/tests/integration/test_adaptive_research.py` | 测试/夹具 | 三个可观察场景的API→Worker回归；模型/论文边界为替身，不冒充真实语义验收。 |
| `backend/tests/integration/test_ag6_repairs.py` | 测试/夹具 | AG6暴露的跨层缺陷回归；真实数据库服务、合成论文/模型材料，不冒充语义验收。 |
| `backend/tests/integration/test_backup_restore.py` | 测试/夹具 | 冷备份恢复合同：完整性、路径重绑定、原始收据与禁止覆盖；零模型。 |
| `backend/tests/integration/test_clarification_mysql.py` | 测试/夹具 | MySQL 澄清事务测试：恢复回答与重入队保持一致，拒绝过期或冲突的幂等请求。 |
| `backend/tests/integration/test_clarification_pipeline.py` | 测试/夹具 | Actual TaskExecutor + API + HTTP adapter; provider network is replaced at HTTP boundary. |
| `backend/tests/integration/test_clarified_plan.py` | 测试/夹具 | 澄清计划与精读交付路由：实际API/Worker，模型与PDF使用可控替身。 |
| `backend/tests/integration/test_closeout_foundation.py` | 测试/夹具 | 交付后补齐接线验收：真实API/数据库/Worker，供应商明确替身且禁止网络。 |
| `backend/tests/integration/test_collaboration_contract.py` | 测试/夹具 | API -> actual role graphs/services/storage, synthetic model/PDF boundaries only. |
| `backend/tests/integration/test_conversation_attachment_delivery.py` | 测试/夹具 | AG7.5暴露的同会话附件接续合同；真实API/Worker，供应商替身，零模型费用。 |
| `backend/tests/integration/test_discovery_citations.py` | 测试/夹具 | 历史发现/引用接口集成测试；引用统计退出活动工具后仍保留兼容测试证据。 |
| `backend/tests/integration/test_dpr_consolidation.py` | 测试/夹具 | DPR 集中适配回归：真实图/服务/收据，供应商与 HTTP 均为明确替身。 |
| `backend/tests/integration/test_dpr_exact_alignment.py` | 测试/夹具 | DPR 行为对照：以本地上游函数为预期，比较实际请求序列、返回文本和发布边界。 |
| `backend/tests/integration/test_evaluation_seed.py` | 测试/夹具 | 评测草稿必须能走到真实收尾服务，不能只验证API创建成功。 |
| `backend/tests/integration/test_evidence_matrix.py` | 测试/夹具 | 【实验暂缓】矩阵API到Worker集成；供应商和论文内容明确合成，支持专用MySQL库。 |
| `backend/tests/integration/test_fulltext_and_five_paper_resume.py` | 测试/夹具 | 验证全文传递及五篇澄清恢复；外部模型/PDF 用替身，业务服务与图实际运行。 |
| `backend/tests/integration/test_indexing.py` | 测试/夹具 | AG7.1真实隔离索引读回；需要显式库和已完成报告，默认不访问生产数据。 |
| `backend/tests/integration/test_memory.py` | 测试/夹具 | AG4真实接口/Worker合同；默认SQLite，显式配置后运行隔离MySQL，模型用可控替身。 |
| `backend/tests/integration/test_memory_recovery.py` | 测试/夹具 | AG4真实MySQL与SIGKILL恢复；仅在显式专用测试库执行。 |
| `backend/tests/integration/test_multi_agent_api.py` | 测试/夹具 | API 多角色测试：提交与查询接口需返回持久化的子任务、澄清及结果状态。 |
| `backend/tests/integration/test_multi_agent_recovery.py` | 测试/夹具 | 协作恢复测试：父子任务、模型收据与租约应一起恢复，不重复执行已完成副作用。 |
| `backend/tests/integration/test_paper_tools.py` | 测试/夹具 | 论文工具集成测试：检索授权、PDF 读取与页级证据应通过真实服务边界衔接。 |
| `backend/tests/integration/test_product.py` | 测试/夹具 | AG5产品接口、流装配与恢复测试：本地真实API/数据库，供应商使用明确替身。 |
| `backend/tests/integration/test_profile_navigation.py` | 测试/夹具 | 自然语言偏好与会话导航：真实HTTP合同，隔离SQLite，不发送模型请求。 |
| `backend/tests/integration/test_reflection_loop.py` | 测试/夹具 | AG3 API→Worker→独立 Reviewer→局部修订→保存；供应商为可观察替身。 |
| `backend/tests/integration/test_retrieval_agent.py` | 测试/夹具 | API→数据库→Worker→检索→模型messages合同；模型/向量为明确替身。 |
| `backend/tests/integration/test_runtime_management.py` | 测试/夹具 | 本机运维合同：端口、身份、只读、共享锁和配置就绪；无真实模型请求。 |
| `backend/tests/integration/test_subscriptions.py` | 测试/夹具 | 订阅、并发调度、逐源窗口和聊天交付合同；默认SQLite，显式隔离MySQL复验。 |
| `backend/tests/integration/test_subtasks_mysql.py` | 测试/夹具 | MySQL 子任务测试：验证数据库持久身份和恢复；无显式测试环境时跳过。 |
| `backend/tests/integration/test_verification_handoff.py` | 测试/夹具 | 实际 API/Worker 交接缺证判断、解析范围和独立核验；模型为显式供应商替身。 |
| `backend/tests/integration/test_worker_claim_mysql.py` | 测试/夹具 | 真实 MySQL 并发领取测试：使用显式测试环境，不允许多个 Worker 同时持有同一有效租约。 |
| `backend/tests/integration/test_worker_recovery.py` | 测试/夹具 | Worker 恢复测试：按持久收据恢复任务，保护已发生的模型/文件副作用。 |
| `backend/tests/test_bootstrap.py` | 测试/夹具 | HTTP contract tests use an isolated SQLite database and never start a worker. |
| `backend/tests/unit/README.md` | 测试/夹具 | unit |
| `backend/tests/unit/__init__.py` | 测试/夹具 | Python包入口 |
| `backend/tests/unit/test_agent_loop.py` | 测试/夹具 | 行动循环测试：用可控模型验证决策、工具观察、取消和有界结束。 |
| `backend/tests/unit/test_citation_matching.py` | 测试/夹具 | 历史引用统计匹配测试；当前日常工具链不启用被引量功能。 |
| `backend/tests/unit/test_deepseek_messages.py` | 测试/夹具 | 对话协议测试：覆盖工具调用配对、上下文完整性及供应商显式错误分支。 |
| `backend/tests/unit/test_discovery_sources.py` | 测试/夹具 | 来源目录测试：17 源状态与默认范围独立，不能将未查询来源记成成功。 |
| `backend/tests/unit/test_dpr_acquisition.py` | 测试/夹具 | 获取层固定响应回归；检查 DPR 地址、错误分类与真实本地解析，不调用模型。 |
| `backend/tests/unit/test_dpr_discovery.py` | 测试/夹具 | DPR 适配测试：覆盖逐源 RPC、候选映射与日期条件；替身响应不代表线上来源可用。 |
| `backend/tests/unit/test_evaluation.py` | 测试/夹具 | 评测口径不受输出成绩影响；覆盖分母、越界、重复、缺证与冻结失败。 |
| `backend/tests/unit/test_evidence_matrix.py` | 测试/夹具 | 【实验暂缓】矩阵结构与出处验证：合成页明确分离于真实论文语义验收。 |
| `backend/tests/unit/test_hybrid_retrieval.py` | 测试/夹具 | 混合检索离线合同：权限/降级/版本/排序，不把替身分数当真实模型质量。 |
| `backend/tests/unit/test_matrix_inference.py` | 测试/夹具 | 矩阵请求模式边界、实际 HTTP 参数和旧收据恢复；全部使用本地替身。 |
| `backend/tests/unit/test_matrix_page_scope.py` | 测试/夹具 | 矩阵真实批次暴露的页范围入口缺口；只扩展明确语法，不猜歧义范围。 |
| `backend/tests/unit/test_model_receipts.py` | 测试/夹具 | 收据恢复测试：成功响应复用，started/unknown 不盲重发；测试替身不产生真实用量。 |
| `backend/tests/unit/test_multi_agent_graph.py` | 测试/夹具 | 多角色图测试：用可控依赖检查子任务执行及父任务汇总，不冒充真实多 Agent 语义验收。 |
| `backend/tests/unit/test_note_files.py` | 测试/夹具 | 笔记文件测试：权限、版本身份、重复写入和异内容冲突分别验证。 |
| `backend/tests/unit/test_paper_cache.py` | 测试/夹具 | PDF 缓存测试：内容、解析器和配置身份决定复用，读取页范围单独验证。 |
| `backend/tests/unit/test_paper_media.py` | 测试/夹具 | 真实 PDF 组件配合提取器替身，验证降级、授权范围与定位真实性。 |
| `backend/tests/unit/test_publication.py` | 测试/夹具 | 验证公开边界扫描不会放行私有配置，也不会回显发现的密钥。 |
| `backend/tests/unit/test_read_retry.py` | 测试/夹具 | DPR限流退避合同：用HTTP替身固定状态和时钟，不访问来源。 |
| `backend/tests/unit/test_recommendations.py` | 测试/夹具 | 候选身份、画像优先级和时区边界的固定组件例。 |
| `backend/tests/unit/test_reflection.py` | 测试/夹具 | 核验轮次、原子局部修改及恢复边界；这里的结论是测试替身，不是模型语义成绩。 |
| `backend/tests/unit/test_reflection_component.py` | 测试/夹具 | Reflection组件评测的真实服务接线与有界失败测试；模型替身不代表语义通过。 |
| `backend/tests/unit/test_reranker_efficiency.py` | 测试/夹具 | 推理独立对照的零模型合同：原排序、标注分母与采用门。 |
| `backend/tests/unit/test_retrieval_contracts.py` | 测试/夹具 | 原文索引确定性测试：替身仅测合同，真实模型/Qdrant另由隔离验收执行。 |
| `backend/tests/unit/test_retrieval_models.py` | 测试/夹具 | 本机模型接口合同；显式提供冻结模型时执行，不自动下载依赖或模型。 |
| `backend/tests/unit/test_retrieval_optimization.py` | 测试/夹具 | 检索优化的零模型合同：排名、组批身份、容量及评测分母。 |
| `backend/tests/unit/test_review_results.py` | 测试/夹具 | 审校合同测试：核验当前稿 hash、证据授权和结果一致性；不评估模型真实理解能力。 |
| `backend/tests/unit/test_role_tools.py` | 测试/夹具 | 角色工具测试：Supervisor、Researcher、Reviewer 各自的可见/可执行权限应一致。 |
| `backend/tests/unit/test_subtask_contracts.py` | 测试/夹具 | 子任务数据契约测试：验证分派、范围及结构化返回。 |
| `backend/tests/unit/test_token_budget.py` | 测试/夹具 | 累计预算的可重放、并发和保守结算测试；不调用真实模型。 |
| `backend/tests/unit/test_tool_contracts.py` | 测试/夹具 | 工具合同测试：模型 schema、参数验证和 handler 共用同一注册入口。 |
| `backend/tests/unit/test_tool_scope.py` | 测试/夹具 | 冻结工具范围的schema与执行保持一致，日常缺省不收缩。 |
| `backend/tests/unit/test_verification.py` | 测试/夹具 | 缺证、明确反证与读取范围回归；使用合成来源，不宣称模型语义正确率。 |
| `backend/uv.lock` | 文档/配置 | 详见所属目录说明 |
| `compose.yaml` | 文档/配置 | 详见所属目录说明 |
| `config/dpr-sources.example.json` | 文档/配置 | 详见所属目录说明 |
| `config/model-manifest.example.json` | 文档/配置 | 详见所属目录说明 |
| `data/README.md` | 文档/配置 | 本机运行数据（不上传） |
| `docs/architecture.md` | 文档/配置 | 架构与真实调用链 |
| `docs/delivery.md` | 文档/配置 | 本机运行、备份与恢复 |
| `docs/evaluation.md` | 文档/配置 | 评测、语料与模拟使用边界 |
| `docs/file-map.md` | 文档/配置 | 文件职责与整理决策 |
| `docs/improvement-plan.md` | 文档/配置 | ResearchLoop 交付后完善计划 |
| `docs/licenses/daily-paper-reader.txt` | 文档/配置 | 详见所属目录说明 |
| `docs/plan.md` | 文档/配置 | 计划入口 |
| `docs/publication.md` | 文档/配置 | 公开仓库边界 |
| `docs/references.md` | 文档/配置 | 参考项目与实际复用 |
| `docs/reset.md` | 文档/配置 | 历史重置记录入口 |
| `docs/resume-closeout.md` | 文档/配置 | 旧补齐合同的后继入口 |
| `docs/resume-evidence.md` | 文档/配置 | 简历声明与证据边界 |
| `docs/retrieval-optimization-proposal.md` | 文档/配置 | 检索优化状态 |
| `docs/status.md` | 文档/配置 | 当前状态 |
| `frontend/README.md` | 前端 | ResearchLoop 前端 |
| `frontend/dev-server.mjs` | 前端 | 详见所属目录说明 |
| `frontend/index.html` | 前端 | 详见所属目录说明 |
| `frontend/package-lock.json` | 前端 | 详见所属目录说明 |
| `frontend/package.json` | 前端 | 详见所属目录说明 |
| `frontend/src/api.js` | 前端 | 详见所属目录说明 |
| `frontend/src/app.js` | 前端 | 详见所属目录说明 |
| `frontend/src/components/artifacts.js` | 前端 | 详见所属目录说明 |
| `frontend/src/components/chat.js` | 前端 | 详见所属目录说明 |
| `frontend/src/components/conversations.js` | 前端 | 详见所属目录说明 |
| `frontend/src/components/evidence-matrix.js` | 前端 | 【实验暂缓】详见所属目录说明 |
| `frontend/src/components/profile.js` | 前端 | 详见所属目录说明 |
| `frontend/src/components/reading-list.js` | 前端 | 详见所属目录说明 |
| `frontend/src/components/subscriptions.js` | 前端 | 详见所属目录说明 |
| `frontend/src/markdown.js` | 前端 | 详见所属目录说明 |
| `frontend/src/styles.css` | 前端 | 详见所属目录说明 |
| `frontend/tests/conversations.test.mjs` | 测试/夹具 | 详见所属目录说明 |
| `frontend/tests/evidence-matrix.test.mjs` | 测试/夹具 | 【实验暂缓】详见所属目录说明 |
| `frontend/tests/server.test.mjs` | 测试/夹具 | 详见所属目录说明 |

实际列出 259 个文件；本清单自身另计。分类数量：{"文档/配置": 29, "迁移": 11, "__init__.py": 1, "agents": 9, "api": 9, "config.py": 1, "domain": 10, "main.py": 1, "providers": 13, "services": 19, "storage": 12, "tools": 7, "workers": 3, "脚本": 32, "测试/夹具": 86, "前端": 16}。
