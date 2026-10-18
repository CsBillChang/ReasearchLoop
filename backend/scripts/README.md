# 脚本分类

业务脚本采用职责命名，旧冻结记录留在本机，不自动执行。业务逻辑放 research_loop；运行输出放不上传的 artifacts/data。

| 分类 | 文件 | 使用条件 |
|---|---|---|
| 日常运行 | manage.py / migrate.py / run_worker.py / run_scheduler.py | 从仓库根通过Makefile运行 |
| 备份恢复 | backup_restore.py | 停写冷备、新空隔离库恢复，默认只读 |
| 配置与索引 | import_paper_sources.py / preflight_retrieval_models.py / index_evidence.py | 显式配置/材料/授权，无隐式生成调用 |
| 评测及报表 | evaluation_common.py / evaluate_*.py / report_*.py | 准备/冻结/run分开；旧批次不能重跑 |
| P1只读重算 | report_evidence.py | 仅标准库读现存排名/标签/耗时/账本，输出到stdout；无模型/网络/数据库。源码漂移与原材料变化分别报告，仅适用于本机材料齐备时 |
| 公开匿名包重算 | report_public_evidence.py | 标准库只读显式指定的匿名包；docs/evidence现仅本机保留，公开克隆不附带包，不重新证明语义或执行旧评测 |
| 未采用的优化实验 | optimize_retrieval.py / verify_reranker_efficiency.py | 不属于默认检索；不得为追分自动执行 |
| 历史阶段runner | run_ag*.py / verify_ag*.py / run_reading_regression.py / verify_reading_followup.py / probe_sources.py | 保留复现入口与门控，依赖未公开本机材料，不作为日常或CI入口 |
| 发布检查 | check_publication.py | 只检查显式暂存清单与敏感模式，无网络/模型调用 |

运行和配置见 [项目说明](../../README.md)。不会因为发布仓库就自动运行任何历史runner。
