# acceptance

当前完善范围以 [docs/improvement-plan.md](../../../docs/improvement-plan.md) 为准；下列AG用例保持历史合同，不重新开启阶段。保存真实首轮、失败、partial及未执行项。

已有AG1/AG2门控验收与AG6评测合同测试。`ag6_cases.json`固定12场景，`test_ag6_contract.py`默认只用替身检查评测器；真实批次必须显式执行`backend/scripts/run_ag6.py --real`并指定隔离库/新证据目录。单测通过不代表真实论文语义评测通过。

P1不运行这些真实runner。原6PDF/24题属于开发回归集；指标定义、只读重算命令、标签局限和未来论文分组采样见 [docs/evaluation.md](../../../docs/evaluation.md)。新报表的独立指标边界测试在 `../unit/test_evidence_report.py`，不含模型调用。

公共克隆不包含旧AG1证据或AG7 PDF，相关原件检查会明确跳过；范围、预算及隔离配置变换仍使用合成配置离线验证，拒绝远程/非MySQL验收地址。默认测试不要求读取发布者的.env，也不会为补齐本机材料自动下载论文或执行真实runner。
