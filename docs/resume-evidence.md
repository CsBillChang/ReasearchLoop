# 简历亮点 → 代码 → 验证 → 局限

公开展示入口为[匿名证据与记录型演示](evidence/README.md)：包括完整检索排名/计时的数值重算、Reflection固定标签的配对计数，以及正常比较、压缩追问、核验失败后partial交付的脱敏概述。不同批次和代码身份单独记录；原文、私人对话、模型响应与收据仍仅在本机保存。版本固定与部署结果见status，公开Git发布仍需另行核对。

本表于2026-10-17对照用户最后明确指定的`Resume-latex/resume-zh_CN.tex`更新，采用“多Agent工作流设计／工具集成与证据检索／记忆层设计／持久执行与故障恢复／量化成果”这一认可版，不使用`resume-zh_CN.before.tex`或参考图片中他人的功能作为开发要求。本次只读简历，未编辑原文件。ResearchLoop段落（至下一项目之前，原文件90–121行）的SHA-256为`e3fce06a4bd957f82f56bdac672e5fcb20e7c7e7af3ae5994d2d6aeb085a5cfa`；不复制其他个人信息。AG阶段不重开，活动状态只见[status.md](status.md)，后续范围只见[improvement-plan.md](improvement-plan.md)。

认可版的主要架构与功能已有对应实现；剩余必做工作集中于版本/部署对应、可展示证据和公开发布。下表中的模型可靠性、样本大小与分支覆盖不足是应披露的局限，不要求靠反复真实调用补成“所有任务成功”。

“源码存在”“替身/隔离测试”“真实模型任务”“原文语义核对”“用户体验”“文件保存”分别列证据，不能互相代替。下表源码是当前工作区；真实证据属于各批次冻结版本，不表示代码变化后都重新跑过。目录名均相对于本机`artifacts/`，不随公开仓库分发。

| 可表述的亮点 | 当前代码入口 / 关键函数 | 已有验证与本机证据 | 局限与不能宣称的内容 |
|---|---|---|---|
| ReAct＋Reflection＋Memory，LangGraph编排Supervisor、Researcher、Reviewer | [graph.py](../backend/research_loop/agents/graph.py) `build_graph`；[executor.py](../backend/research_loop/workers/executor.py) `_run_research_batch/_review_once`；[registry.py](../backend/research_loop/tools/registry.py) | `ag7-5-followup-closeout-20261011/`真实父子任务、角色messages和工具记录；AG6反馈另有Reviewer轨迹 | 三角色分别实例化共享图构造器、配置独立上下文/工具；Researcher最多两线程并行。不是三种独立训练模型，也不是示例的六Agent流水线 |
| 目标覆盖、澄清恢复、动态委派、有界退出 | [coverage.py](../backend/research_loop/services/coverage.py) `prepare_plan/validate_delegation/finish`；[tasks.py](../backend/research_loop/services/tasks.py)；[executor.py](../backend/research_loop/workers/executor.py) `_deliverable_result` | AG6固定12场景采用真实、修复接续和零模型边界的分层证据；`ag6-current-draft-delivery-20261011/README.md`逐场景交接。P0旧失败保留完整partial答复 | 不写“解决所有长任务偏题和不收敛”。保留成果并报告原因不等于任务全部完成；阶段关闭不等于首轮12/12成功 |
| 独立核验、原文补读、局部修订和进展控制 | [reflection.py](../backend/research_loop/agents/reflection.py) `review/revise`；[reviewer.py](../backend/research_loop/agents/reviewer.py) `research_reviewer_state`；[verification.py](../backend/research_loop/services/verification.py)；[executor.py](../backend/research_loop/workers/executor.py) `_review_research_once` | 原4固定稿/8标注组件结果保持。通用入口覆盖candidates/findings/answer，核验适用性、支持性及问题范围；`retrieval-review-reliability-real-20261017/`真实接收10项意见并应用3处局部修改，但后续容量失败；`revision-answer-projection-repair-20261017/`修补及隔离继续交付通过；`revision-answer-projection-real-20261017/`真实核验执行失败后Supervisor主动交付原正文与独立说明 | 不再仅审核笔记，也不强制全输出审校。各批次不可拼成一次完整成功；最新真实未发生修订，修订后继续仅有工程/隔离证据。引句匹配错误可导致整批未接收；执行失败不等于反证/缺证或批准。默认两轮/Reviewer4次是原配置，不代表全局成功保证；组件非独立双人金标或端到端因果消融 |
| 原文版本化与BM25＋向量＋RRF＋Cross-encoder | [retrieval.py](../backend/research_loop/domain/retrieval.py) `split_page`；[indexing.py](../backend/research_loop/services/indexing.py)；[retrieval.py](../backend/research_loop/services/retrieval.py) `search`；[embeddings.py](../backend/research_loop/providers/embeddings.py)、[reranker.py](../backend/research_loop/providers/reranker.py) | AG7.2真实授权范围检索/降级；AG7.4六PDF/63页/282片段/24题原始排名可重算。本机E5-small 384维、Qdrant过滤、RRF k=60、默认24条纯CE重排 | 页内段落/token切分，保留字符偏移和可识别章节，不代表所有PDF结构完整恢复。旧24题是开发回归，非未见确认集；Recall收益和MRR/时延代价须一起报告 |
| DPR发现/精读复用，结构化工具及权限边界 | [discovery.py](../backend/research_loop/providers/discovery.py)、[pdf.py](../backend/research_loop/providers/pdf.py)、[deep_reading.py](../backend/research_loop/agents/deep_reading.py)、[base.py](../backend/research_loop/tools/base.py) | 17源目录和逐源状态；AG6真实获取/读取/精读记录及成品体验。缓存→Jina→PDF/PyMuPDF，附件/限定页直接PDF | 17源配置不等于实时全可用；没有被引量服务。明确精读才用DPR八点节点，普通比较不强制精读；图片提取位置与另行视觉理解分开，复用代码归属保留在第三方说明 |
| 按需视觉补读（认可版之外已有的增强） | [visual_reading.py](../backend/research_loop/agents/visual_reading.py)、[delegation.py](../backend/research_loop/tools/delegation.py) `delegate_visual_reading`、[paper_media.py](../backend/research_loop/providers/paper_media.py) | `visual-fixed-real-20261017/`两PDF/4页/5题、10次对照，字段20/29→26/29；按需视觉V1与`visual-text-real-recheck-20261017/`普通文字T1分别有真实路由/交付证据，后者零视觉请求 | 文字为默认；视觉是具体公式/图中关系缺口的一次Researcher委派，不等于理解全部图表。小样本代价增加，收益范围见evaluation；生产部署另核对，不强制写进简历 |
| 累计Token预算、只读退避、最小阅读清单 | [token_budget.py](../backend/research_loop/storage/token_budget.py) `reserve/settle`；[read_retry.py](../backend/research_loop/providers/read_retry.py)；[reading_list.py](../backend/research_loop/services/reading_list.py) | `resume-closeout-foundation-20261012/`：24个不同单元/集成用例，10项隔离MySQL通过，含SIGKILL后结算；原网页/部署读回保留 | 模型侧使用供应商替身验证故障，不证明长期费用下降；只读HTTP重试不用于盲重发未知模型请求，清单操作不自动授权精读/保存 |
| 共享研究状态、分层Memory、可撤销画像、可追溯压缩 | [state.py](../backend/research_loop/agents/state.py) `AgentState`；[research_progress.py](../backend/research_loop/services/research_progress.py)；[memory.py](../backend/research_loop/services/memory.py) `remember/search/read`；[memory.py](../backend/research_loop/agents/memory.py) `history/prepare`；[context.py](../backend/research_loop/domain/context.py)、[context_workspace.py](../backend/research_loop/domain/context_workspace.py) | AG4/AG6偏好召回与撤销；原P0短材料证据保留；`core-loop-working-context-real-20261016/`原两篇比较及压缩后同会话追问completed，28请求/712241tokens，阶段发现/父子成果保持。最新同版正文投影原失败输入重放与20项不同MySQL用例通过，旧版和原文/意见可读回 | 当前要求优先，历史不等于原文或保存授权；保守UTF-8字节上界不是准确Token计数，抽取式压缩不是无损语义摘要。必要读回允许，仍无普遍减少重复读取/Token/耗时的量化证据；固定任务及隔离测试不外推任意长上下文。当前部署身份另核对 |
| 个性化增量简报与曝光去重 | [subscriptions.py](../backend/research_loop/services/subscriptions.py) `claim/publish`；[recommendations.py](../backend/research_loop/services/recommendations.py) `rank_candidates`；[scheduler.py](../backend/research_loop/workers/scheduler.py) | `ag7-3-incremental-brief-20261011/`：日历、唯一领取、去重、撤销/恢复隔离验证；真实DPR候选简报及页面。上海09:00、最多5篇、停机合并补一份 | 元数据规则排序，明确偏好/推断/已读/交付分别处理；不自动精读。专用正负反馈未完成，不声称训练推荐模型或相关性提升 |
| SSE接续、持久执行、幂等保存、本机运维 | [events.py](../backend/research_loop/api/events.py)、[artifacts.py](../backend/research_loop/api/artifacts.py)、[tasks.py](../backend/research_loop/api/tasks.py)；[mysql.py](../backend/research_loop/storage/mysql.py) `claim_task/begin_model_call`；[checkpoints.py](../backend/research_loop/storage/checkpoints.py)；[files.py](../backend/research_loop/storage/files.py)；[manage.py](../backend/scripts/manage.py)、[backup_restore.py](../backend/scripts/backup_restore.py) | API→MySQL→Worker→收据/文件，幂等、租约、SIGKILL/取消、SSE序号恢复有隔离记录；AG6 E04文件hash本轮再读回一致；AG7.5备份恢复/本机交付证据 | 自实现state JSON Checkpoint，未使用LangGraph原生Checkpointer；unknown保守结束，不保证远端exactly-once。本机单用户，无多租户/大并发/SLA证据 |
| 可复算评测与失败归因 | [report_evidence.py](../backend/scripts/report_evidence.py)、[evaluation_common.py](../backend/scripts/evaluation_common.py)、[evaluate_retrieval.py](../backend/scripts/evaluate_retrieval.py)；[验收目录](../backend/tests/acceptance/README.md) | P1独立重算原四路及两次候选Recall/MRR/P50/P95，逐题与标签相符；Reflection收据/账本一致，旧失败保留 | 没有大规模推荐相关性、全任务引用正确率、总体成本收益或用户满意度指标。历史组件32处源码hash已变，结果可重算不等于历史环境完整重建 |

## 认可版逐部分的交付判断

| 认可版内容 | 对齐判断 | 是否必须继续补业务功能 |
|---|---|---|
| 多Agent工作流：ReAct／角色协作／反馈路由 | 有实际图、角色权限、动态委派、通用核验、局部修改和合理结束证据 | 无新增必做功能；核验提交可靠性和行动效率可后续优化，不宣称各动态路径均一次成功 |
| 工具与混合RAG | 获取、PDF读取、精读、检索/索引、保存及故障处理均有实现 | 无需重写DPR或增加任意SQL工具；上游可用性、复杂PDF与CE性能继续披露 |
| 记忆层：共享状态／分层Memory／压缩／增量简报 | 阶段进展、上下文投影、原件读回、偏好/兴趣及规则简报均有实现与分层证据 | 无需为了该表述新增推荐学习或无限上下文；不能承诺完全消除重读 |
| 持久执行与SSE | 原子领取、租约、项目快照、收据、幂等/取消/恢复、事件补发已有验证 | 仅发布时确认当前演示版本；专业技能中的Checkpoint需明确自实现，不暗示原生组件 |
| 量化成果 | 认可版数字与现有evaluation重算结果一致：RRF相对向量的21题质量指标、24题组件耗时；Reflection为4稿8标注 | 保留范围就可使用，不要求扩库或强求均衡提升；公开证据和发布身份需收尾，后续采样另定 |

**尚需完成的是交付对应关系，不是再加一轮架构或修复阶段。** 已有本机部署读回和本页公开证据入口；执行状态只见status。最终发布仍需核对干净环境配置/启动、显式提交清单和GitHub版本；精确完成条件只在[完善计划顶部](improvement-plan.md#收束后的交付清单与停止条件2026-10-17)维护。未核验远端最新提交，不宣称公开仓库已包含最新修补。

## 历史旧稿表述的取舍（不是认可版新增待办）

| 原描述 | 当前可支持的表达 |
|---|---|
| ResearchState汇总任务 | 实际类型`AgentState`配合MySQL的目标、稿件与证据；解释机制，不强行更名 |
| 按章节和段落切分 | 页内段落/token切分，保留原文偏移和可靠时的章节标签 |
| 六维结构化证据矩阵已交付 | 暂缓，真实矩阵输出未通过；可写已有带来源的普通方法比较，八点精读不等于矩阵 |
| 正负推荐反馈沉淀画像 | 已有明确偏好、可撤销兴趣、已读/曝光规则；专用推荐反馈未纳入本次核心修复，不作为交付前置 |
| 解决偏题/重复检索/不收敛 | 用权限、目标覆盖、预算及退出机制约束；仍保留事实错误和部分交付 |
| 全面消融、推荐/引用正确率、Token下降 | 现有小样本四路检索对照、Reflection固定稿组件核验、任务分层验证；未测收益不写 |

Python/FastAPI/LangGraph/MySQL/Qdrant/BM25/Cross-encoder/SSE/Docker均有代码或交付配置；Docker用于本机基础设施，不等于云端集群部署。技术名对应实现，不代表每项已有独立收益。

## 两条记录型演示

1. **正常交付：同稿定向复审并保存。** `ag6-current-draft-delivery-20261011/real/E04-current.json`保留输入、source_task/source_draft/hash、Supervisor/Reviewer及保存记录。既有三篇比较稿仅核对GenerateU修改相关部分，正文保持；5请求/123023tokens、completed，文件hash `ae3b4f2a…4ee5f`。前轮补读/局部修订与本轮复审属于不同批次，不能展示成一次从搜索开始就成功。历史三视口截图与两份成品的用户认可佐证可用性，不等于全部陈述人工核验。
2. **受限交付：读取失败后如实结束。** `p0-context-continuity-20261013/real-result.json`原首问，6请求后partial；8次读工具动作被页范围解析拒绝，输出“未获得指定页正文，不能比较”，没有稿件文件/保存动作。此为后来已修复的产品缺陷，不说成论文没有相关内容，也不伪装自然上游故障。修复后的同题成功和追问在`p0-real-followup-20261013/`，旧失败保留。

P1本机`artifacts/p1-evidence-audit-20261013/demo-evidence.json`导出两条演示的准确输入/原输出/工具序列/账本及文件校验；没有新浏览器点击或模型答复。面试可先看轨迹/成品，再沿代码解释决策、失败、保存授权，不依赖现场反复付费跑到满意。

另可展示`core-loop-working-context-real-20261016/`的正常比较/压缩追问和`revision-answer-projection-real-20261017/`的主动partial收尾。两者来自不同批次，后者没有局部修订；不能剪成同一次“核验→修订→交付”成功录像。展示原正文、处理说明与未解决目标即可，不为演示新增真实调用。

修复收束状态见status；后续按完善计划完成发布对应。扩展评测属于后续优化，仍须论文分组、独立标签、完整分母与同口径成本；模拟用户明确标注，不冒称真人规模。已有指标与失败详见[评测说明](evaluation.md)。
