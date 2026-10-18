# acceptance

当前公开功能见 [项目说明](../../../README.md)；下列AG用例保持历史合同，不自动执行真实验收。真实首轮、失败、partial及未执行记录仅保留在本机。

已有AG1/AG2门控验收与AG6评测合同测试。`ag6_cases.json`固定12场景，`test_ag6_contract.py`默认只用替身检查评测器；真实批次必须显式执行`backend/scripts/run_ag6.py --real`并指定隔离库/新证据目录。单测通过不代表真实论文语义评测通过。

原6PDF/24题属于开发回归集，原件与评测记录不随源码发布。独立指标边界测试在 [test_evidence_report.py](../unit/test_evidence_report.py)，不含模型调用；不能把这些单测当作旧语义标签或真实任务的复验。

公共克隆不包含旧AG1证据或AG7 PDF，相关原件检查会明确跳过；范围、预算及隔离配置变换仍使用合成配置离线验证，拒绝远程/非MySQL验收地址。默认测试不要求读取发布者的.env，也不会为补齐本机材料自动下载论文或执行真实runner。
