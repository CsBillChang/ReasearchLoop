# ResearchLoop 协作约定

## 当前入口与范围

- 先读 README.md；本机存在 docs/ 时再读 docs/status.md、docs/improvement-plan.md、docs/file-map.md。当前计划以用户最新指令优先，其次为 improvement-plan 与 status；旧 AG/补齐合同仅为历史资料。
- AG0–AG7按原约定完成，不重开。本轮是代码/文档盘点、完善准备与首次 GitHub 发布；没有授权自动执行 P0–P3、扩大语料、调用 DeepSeek 或修改外部简历。
- 活动状态只写 docs/status.md，后续任务范围只写 docs/improvement-plan.md；不要再向 README/AGENTS 叠加每批次的“当前授权”。旧文档保存在仓库外本机备份（位置见 docs/status.md），原始证据保存在 artifacts/，均不上传。
- 架构为 ReAct＋Reflection＋Memory，LangGraph 编排 Supervisor / Researcher / Reviewer。Checkpoint 为项目自实现快照，不冒称原生 Checkpointer。
- 当前业务源码、依赖、配置入口与迁移保留；未交付矩阵标记为实验能力，不能阻挡无技术依赖的其他任务。不得为了整理目录删除运行数据或反复搬迁模块。

## 产品与实现边界

1. 只设聊天和个人面板两个主入口；附件、笔记、证据、阅读清单作为聊天产物。单用户本地应用，不包含 Obsidian 同步或公网多租户部署。
2. 模型负责自然语言理解、选择工具及委派；参数、版本、权限、事务、预算由代码验证。工具 schema/get/execute 共用合同。
3. 论文获取/精读沿用现有适配。普通全文问答、比较、综述不隐含八点精读；只有明确精读请求才调用精读单节点。三轮重生成合同保持，不再叠加普遍引用格式或强制 Reviewer 门。
4. Reflection 由 Supervisor 按需选择；当前稿独立核验，支持/反证/缺证分开，局部修订与有界终止。不能把结构合法当作语义正确，也不能承诺所有任务成功。
5. 当前用户要求优先于长期偏好，显式偏好优先于可撤销推断；历史记忆不继承保存权限，不把旧笔记当原文。
6. 原子领取、租约、取消、成功收据复用、unknown 不盲重发、版本/hash 和文件保护不得绕过。预算余量不授权追加真实实验。
7. 新实现核对具体源码和 NOTICE 许可证；本机有 docs/references.md 时一并参考；不整批复制参考项目，不恢复旧废弃实现。

## 修改、验证与发布

- 先定位文件、输入输出、完成与停止条件，再实施。先零模型/隔离验证，真实调用另冻结材料、代码与预算；本轮真实业务调用预算为 0。
- 测试区分替身、隔离 MySQL、真实模型、浏览器和用户体验；partial/失败如实报告，不改旧账本、不把未知用量填零。
- 代码使用 Google 风格中文 docstring；关键原因用注释说明，前端中文 JSDoc。Pydantic 参数模型优先使用 # 注释，避免注释改变工具 schema。
- 本次已授权上传 https://github.com/CsBillChang/ReasearchLoop 的经过检查源码。以后其他外部发布按用户授权执行，不 force push、不替换无关历史。
- GitHub 只上传源码、锁文件、迁移、合成/可公开测试、必要 README、配置模板和 NOTICE。docs/、artifacts/、data/ 及 THIRD_PARTY_NOTICES.md 留在本机，不再上传。不得上传 .env、来源凭据、PDF、权重、个人对话、原始模型响应、数据库/备份、截图或含个人信息的历史文档。
- 每次提交使用显式清单，并运行 backend/scripts/check_publication.py --staged。历史验收脚本依赖本机材料，不能作为默认启动/持续集成入口自动执行。
