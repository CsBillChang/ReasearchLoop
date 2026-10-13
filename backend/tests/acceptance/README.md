# acceptance

按 docs/plan.md 冻结任务与判断标准，保存真实首轮和失败证据。

已有AG1/AG2门控验收与AG6评测合同测试。`ag6_cases.json`固定12场景，`test_ag6_contract.py`默认只用替身检查评测器；真实批次必须显式执行`backend/scripts/run_ag6.py --real`并指定隔离库/新证据目录。单测通过不代表真实论文语义评测通过。
