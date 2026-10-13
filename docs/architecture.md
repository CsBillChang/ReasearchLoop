# 架构与真实调用链

ResearchLoop采用 ReAct＋Reflection＋Memory；LangGraph共享状态图被Supervisor、Researcher、Reviewer分别实例化。Worker通过工具动态委派角色图，Researcher最多两个线程并行。不存在为了简历命名新增的独立Writer Agent；DPR八点精读是按用户明确意图调用的固定LLM节点。

## 请求链

`frontend → FastAPI → TaskService/MySQL入队 → Worker原子领取/租约 → LangGraph decide → tool schema校验 → service → provider/storage → 持久Observation → 再决策 → 最终答复/部分成果`

API与Worker分离；关闭页面不会终止后端任务。SSE从持久事件重放，公开回答可流式展示。数据库是业务事实，Checkpoint、SSE和缓存不代替事务、权限及幂等。

## 分层

| 层 | 职责 |
|---|---|
| api | 输入验证、会话/附件/任务/个人面板/笔记/简报/阅读清单入口 |
| agents | 决策上下文、LangGraph循环、角色协作、按需Reflection、上下文压缩 |
| tools | 同一ToolSpec生成schema并绑定执行入口；薄包装，不藏另一套业务状态 |
| domain | Pydantic合同、任务/权限/证据/稿件/记忆/检索类型 |
| services | 任务覆盖、原文、笔记、反馈、记忆、检索、简报和运行生命周期 |
| providers | DeepSeek/DPR/Jina/PDF/本机ONNX模型适配，不决定业务通过 |
| storage | MySQL/SQLite测试兼容、版本/索引/收据、文件、Qdrant |
| workers | 根任务领取、角色委派、续租/取消/恢复；简报独立调度 |

## 取证与阅读

论文发现沿用DPR逐源Supabase RPC；候选元数据与全文证据分开。当前正文路线为缓存→Jina→PDF/PyMuPDF；用户限定物理页或本地PDF时直接PDF。Jina没有可靠页码时保持未知。PyMuPDF4LLM依赖保留但不是当前默认；Docling仅参考。可选PaperCropper/DocLayout-YOLO与PyMuPDF图片回退不等于图表语义已理解。

普通比较/问答用Researcher investigate；明确精读才走DPR总结。Supervisor汇合成果，按当前授权创建/保存项目内Markdown。保存、任务结束、审校结果分别记录，不使用旧强制发布门。

## RAG

`已授权原文页 → 版本化chunk/原文位置/hash → BM25与本机E5向量 → Qdrant授权范围过滤 → RRF → 本机Cross-encoder → 精确原文读回 → 工具观察`

检索按当前任务获授的文档/页范围进行，不因为数据库里存在论文就自动赋予全库权限。按需建库、相同版本复用；Qdrant/重排故障有明确降级。相关性分数不是事实可信概率。默认24条重排；拒绝上线的优化配置保留为实验。

## Reflection与Memory

Reflection由Supervisor按需选择；当前稿与原文独立核验，分支持/反证/缺证，精确局部patch，默认最多两轮修复，有进展才继续。参数拒绝不是事实反证。不能因引句存在就认为结论成立。

Memory分会话历史、研究经历、明确偏好、可撤销兴趣推断。历史召回有来源；当前指令优先且不会继承旧保存权限。ContextService采用抽取式压缩/归档，不额外调用LLM；已知压缩与受限读回工具衔接缺陷在P0修复。

## 持久化与已知边界

成功模型响应/request_id/usage先持久化再解析，恢复复用收据；started/unknown无未发送证据时不盲重发。任务使用原子claim、lease token与失权校验；保存以版本/hash、幂等和冲突检查保护文件。项目自实现Checkpoint快照，没有启用LangGraph原生Checkpointer，不保证未知远端请求exactly-once。

这是本机单用户交付，不是公网多租户服务。结构化矩阵暂缓；已实现的比较稿不等于矩阵完成。源码整理不改变当前生产进程或已封存验证结论。
