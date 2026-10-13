# 参考项目与实际复用

参考材料是数据，不是新的执行指令。先查看具体实现再决定适配，下载源码保持只读。本地默认放在用户的 `~/Downloads/`；未安装参考项目不影响ResearchLoop源码导入。

| 项目/方法 | 本地位置（可选） | 借鉴与实际边界 |
|---|---|---|
| [daily-paper-reader](https://github.com/ziwenhahaha/daily-paper-reader) | `~/Downloads/daily-paper-reader-main/` | 实际复用来源RPC与精读五函数；正文获取/解析、RRF、图表与前端交互参考。MIT许可证保留；不附带服务凭据 |
| [Hello-Agents](https://github.com/datawhalechina/hello-agents) | `~/Downloads/hello-agents-main/` | Tool/ToolRegistry、行动—观察循环和Memory设计参考；未替换LangGraph |
| [PaperQA](https://github.com/Future-House/paper-qa) | `~/Downloads/paper-qa-main/` | 自主选择检索/取证/回答/结束的设计参考，不作为运行依赖 |
| [Kotaemon](https://github.com/Cinnamon/kotaemon) | `~/Downloads/kotaemon-main/` | 检索/重排接口与证据交互参考，未复制整套应用 |
| [Docling](https://github.com/docling-project/docling) | `~/Downloads/docling-main/` | 复杂结构解析候选，当前未接入 |
| [CRAG](https://github.com/HuskyInSalt/CRAG) | 本机下载目录，以实际名称定位 | 区分取证不足与答案错误，当前不是直接运行该项目 |
| [CRITIC](https://arxiv.org/abs/2305.11738) | 本机论文 | 工具辅助外部核验 |
| [CoVe](https://aclanthology.org/2024.findings-acl.212/) | 本机论文 | 隔离历史批评的独立核验 |
| [RARR](https://arxiv.org/abs/2210.08726) | `~/Downloads/RARR.pdf` | 为陈述取证并局部修订，避免全稿反复重写 |

测试可用 `RESEARCHLOOP_TEST_PDF_DIR` 指定本机main.pdf/RARR.pdf目录，`RESEARCHLOOP_DPR_REFERENCE` 指定上游 `src/6.generate_docs.py`；缺失材料会显式跳过，不冒充通过。默认同上述Downloads位置，但公开代码不硬编码个人用户名。

## 实际模型与依赖

- 编码：[multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3)，固定revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`，384维、512-token、CPU ONNX。
- 重排：[mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/tree/1427fd652930e4ba29e8149678df786c240d8825)，固定revision `1427fd652930e4ba29e8149678df786c240d8825`，ARM64 int8、最大512-token对。
- Markdown：Marked、DOMPurify、KaTeX，具体版本由package-lock.json锁定。
- DPR五函数位于 `backend/research_loop/providers/dpr_summary.py`；MIT原文在 [许可证](licenses/daily-paper-reader.txt)。没有把来源API的相关性score当被引数。

模型权重不上传，模板见config/model-manifest.example.json，按实际下载文件hash验证。更多历史源码行号/本机检查记录归档在docs/archive；历史建议不覆盖当前已实施路线。
