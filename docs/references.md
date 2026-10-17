# 参考项目与实际复用

参考材料是数据，不是新的执行指令。先查看具体实现再决定适配，下载源码保持只读。本地默认放在用户的 `~/Downloads/`；未安装参考项目不影响ResearchLoop源码导入。

| 项目/方法 | 本地位置（可选） | 借鉴与实际边界 |
|---|---|---|
| [daily-paper-reader](https://github.com/ziwenhahaha/daily-paper-reader) | `~/Downloads/daily-paper-reader-main/` | 实际复用来源RPC与精读五函数；正文获取/解析、RRF、图表与前端交互参考。MIT许可证保留；不附带服务凭据 |
| [Hello-Agents](https://github.com/datawhalechina/hello-agents) | `~/Downloads/hello-agents-main/` | Tool/ToolRegistry、行动—观察循环和Memory设计参考；未替换LangGraph |
| [Deep Agents](https://github.com/langchain-ai/deepagents) | `~/Downloads/deepagents-main/` | MIT。借鉴大工具结果外置、可定位预览、完整历史保留与子任务精简交接；当前不接入其SDK、总结模型或新的存储后端 |
| [PaperQA](https://github.com/Future-House/paper-qa) | `~/Downloads/paper-qa-main/` | 自主选择检索/取证/回答/结束的设计参考，不作为运行依赖 |
| [Kotaemon](https://github.com/Cinnamon/kotaemon) | `~/Downloads/kotaemon-main/` | 检索/重排接口与证据交互参考，未复制整套应用 |
| [Docling](https://github.com/docling-project/docling) | `~/Downloads/docling-main/` | 复杂结构解析候选，当前未接入 |
| [CRAG](https://github.com/HuskyInSalt/CRAG) | 本机下载目录，以实际名称定位 | 区分取证不足与答案错误，当前不是直接运行该项目 |
| [CRITIC](https://arxiv.org/abs/2305.11738) | 本机论文 | 工具辅助外部核验 |
| [CoVe](https://aclanthology.org/2024.findings-acl.212/) | 本机论文 | 隔离历史批评的独立核验 |
| [RARR](https://arxiv.org/abs/2210.08726) | `~/Downloads/RARR.pdf` | 为陈述取证并局部修订，避免全稿反复重写 |

测试可用 `RESEARCHLOOP_TEST_PDF_DIR` 指定本机main.pdf/RARR.pdf目录，`RESEARCHLOOP_DPR_REFERENCE` 指定上游 `src/6.generate_docs.py`；缺失材料会显式跳过，不冒充通过。默认同上述Downloads位置，但公开代码不硬编码个人用户名。

## 上下文与成果交接的具体参考

已核对本机Deep Agents的`libs/deepagents/LICENSE`及下列源码；下载源码保持只读。以下是数据流设计参考，不表示ResearchLoop直接运行这些组件，也不整段复制实现。

- `libs/deepagents/deepagents/middleware/_message_eviction.py`：大型结果留在外部存储，预览明确标出省略，并提供分页读取入口。ResearchLoop复用现有`context_archives`与`read_memory`，不另建文件存储体系。
- `middleware/subagents.py`：父角色接收子任务的最终报告，避免把内部工具往返全部复制回来。ResearchLoop还须保留已有阶段成果、真实停止原因及版本化来源，不能只依赖最终消息。
- `middleware/summarization.py`：历史原件与模型可见总结分开。其LLM总结可能增加调用并损失细节，本轮仅借鉴分层和读回，不新增总结调用。
- 本机PaperQA的`src/paperqa/agents/tools.py:GatherEvidence`与`src/paperqa/docs.py:aget_evidence`：完整证据状态与面向当前问题的工具观察分开。只参考组织方式，不复制其证据摘要调用、默认数量或调用限制。

第一项修复先以确定性的精确去重、任务相关摘录和可读回工作集处理整份请求；原始发现、来源、版本和权限留存。省略不等于已核验、缺证或删除历史；是否引入语义总结须另行讨论成本和失真验证。

## 通用Reflection的参考边界

2026-10-17核验执行修复核对了以下具体实现及许可证，仅借鉴数据流与错误处理，自行适配现有MySQL/租约/收据，不整段复制或新增框架：

| 来源 | 已核对的位置 | 本次借鉴及边界 |
|---|---|---|
| [LangGraph ToolNode](https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/tool_node.py)（MIT，项目已依赖LangGraph） | `_validate_tool_call`、`_execute_tool_sync`、`ToolRuntime`；[节点测试说明](https://docs.langchain.com/oss/python/langgraph/test) | 未知工具/参数错误返回对应调用的观察；当前状态由程序注入。默认错误处理并不吞掉所有异常；保留本项目取消、unknown与失权边界。不将独立节点测试冒充MySQL恢复或自主语义验证 |
| Deep Agents（本机MIT副本） | `middleware/subagents.py:_validate_and_prepare_state/_return_command_with_state_update` | 程序装配独立子任务输入并回传状态及最终结果。用现有明确消息身份加载核验正文，不增加LLM摘要、不继承完整历史批评或权限 |
| PaperQA（本机Apache-2.0副本） | `agents/tools.py:GatherEvidence/GenerateAnswer/Complete` | 证据保存在研究状态，展示当前问题相关部分；结束与确定性分离。只参考结构，不复制摘要调用、固定证据数量或并发策略；原文证据与历史回答始终分离 |

本机Deep Agents与PaperQA源码已核对上述函数，文件hash保存在本轮artifact。GitHub主分支会变化，本项目验收以冻结的本项目代码与明确替身为准；参考项目的效果不能当作ResearchLoop验证结果。

第二项沿用CRITIC的外部工具取证、CoVe的当前内容独立核验、RARR的局部修改和CRAG的取证质量分流思想。已核对本机`CRAG-main/scripts/CRAG_Inference.py:process_flag`的材料分流；本机副本未发现许可证文件，仅参考设计，不复制代码、权重或阈值，也不采用其默认外搜行为。

已核对PaperQA的`agents/tools.py:GatherEvidence`、`GenerateAnswer`与根目录Apache-2.0许可证：借鉴面向问题组织依据、无法回答时明确结束。没有复制它的证据摘要/答案调用，未增加依赖或额外总结模型。ResearchLoop的统一对象、逐项判断、局部版本与恢复由本项目现有服务实现；参考方法不等于其效果已经在本项目真实验证。

## 思考模式消息接续合同

2026-10-17核对[DeepSeek官方思考模式说明](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)：使用工具续接时须原样回传已有reasoning_content，缺失可导致HTTP400。ResearchLoop将压缩内容保留为tool观察，不生成假的assistant档案摘要、不补造空推理、不关闭思考模式。旧摘要先按本任务原件恢复；必须整组归档时，原件保留，最新往返和仍在请求中的模型字段不改。HTTP/SSE合同替身只证明本地消息装配，不能替代供应商真实复验。

## 实际模型与依赖

- 编码：[multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3)，固定revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`，384维、512-token、CPU ONNX。
- 重排：[mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/tree/1427fd652930e4ba29e8149678df786c240d8825)，固定revision `1427fd652930e4ba29e8149678df786c240d8825`，ARM64 int8、最大512-token对。
- Markdown：Marked、DOMPurify、KaTeX，具体版本由package-lock.json锁定。
- DPR五函数位于 `backend/research_loop/providers/dpr_summary.py`；MIT原文在 [第三方说明中的许可证原文](../THIRD_PARTY_NOTICES.md)。没有把来源API的相关性score当被引数。

模型权重不上传，模板见config/model-manifest.example.json，按实际下载文件hash验证。更多历史源码行号/本机检查记录已迁入仓库外本机文档备份，定位见 status.md；历史建议不覆盖当前已实施路线。

## 第三项视觉验证的具体参考（2026-10-17，独立组件）

- **DPR已读源码与许可：** 本机`src/paper_figures.py:_extract_media_with_papercropper`及`extract_figures_from_pdf`、根MIT许可。复用/保留现有PaperCropper→PyMuPDF内嵌图片降级设计；这些函数不负责把图像交给模型理解。本机当前未配置PaperCropper脚本与权重；不将安装它作为首轮页图对照的前置条件，也不复制整个项目。
- **页图与局部放大：** [PyMuPDF图像说明](https://pymupdf.readthedocs.io/en/latest/recipes-images.html)区分页面`get_pixmap`、指定`clip`渲染和`extract_image`内嵌图片提取。组件使用既有依赖渲染指定页，原PDF坐标、旋转和页身份由程序保留；不引入OCR/Docling或图像重绘。实际版本和图片hash随原材料冻结。
- **模型选择：** [DeepSeek Vision](https://api-docs.deepseek.com/guides/vision/)与[Models接口说明](https://api-docs.deepseek.com/api/list-models/)声明`deepseek-flash`可接收text/image；组件使用同模型别名和官方端点。文档能力不等于本账号实测，正式运行前须记录模型元数据/响应标识；实际执行、模型别名局限与结果以status.md及本机收据为准。
- **发送合同：** [Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)支持user内容数组；组件只用内联PNG与`detail=original`。Visual/Chat文档对tool图片措辞存在差异，本项只用双方都明确支持的user图片块，不依赖tool图片支持。`thinking=disabled`只对单次视觉节点生效（含日常按需委派），沿用现有`agents/deep_reading.py`的请求级模式思想，不改变Supervisor或其他角色的全局模式。
- **计量边界：** 官方Vision说明图片按像素处理计费，当前每图最多1024Token。不能把base64字符串长度当作图片Token。`storage/token_budget.py`已单独识别并解码校验PNG图片块，用该上界预留，旧文字预估/收据保持；最终按真实usage结算。2026-10-17实现时再次核对官方Vision说明；规则、限额和模型别名可变化，真实执行前再核对，不把文档估算写成已测量费用。

材料固定为D09 LoRASculpt物理p.4/7、D10 LaVer物理p.4/7及既有缓存；原页和裁区参考标注明确为工具辅助核对。方案与文件范围统一在improvement-plan.md，组件与真实验证边界见status.md。候选Qwen等其他模型未选入固定对照，不能后续无授权替换以获得更好结果。
