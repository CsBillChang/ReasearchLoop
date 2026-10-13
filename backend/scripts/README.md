# 脚本分类

脚本保持原路径，避免破坏测试导入和已冻结记录。业务逻辑放 research_loop；运行输出放不上传的 artifacts/data。

| 分类 | 文件 | 使用条件 |
|---|---|---|
| 日常运行 | manage.py / migrate.py / run_worker.py / run_scheduler.py | 从仓库根通过Makefile运行 |
| 备份恢复 | backup_restore.py | 停写冷备、新空隔离库恢复，默认只读 |
| 配置与索引 | import_dpr_sources.py / preflight_retrieval_models.py / index_evidence.py | 显式配置/材料/授权，无隐式生成调用 |
| 评测及报表 | evaluation_common.py / evaluate_*.py / report_*.py | 准备/冻结/run分开；旧批次不能重跑 |
| 未采用的优化实验 | optimize_retrieval.py / verify_reranker_efficiency.py | 不属于默认检索；不得为追分自动执行 |
| 历史阶段runner | run_ag*.py / verify_ag*.py / run_dpr_consolidation.py / verify_dpr_followup.py / probe_sources.py | 保留复现入口与门控，依赖未公开本机材料，不作为日常或CI入口 |
| 发布检查 | check_publication.py | 只检查显式暂存清单与敏感模式，无网络/模型调用 |

当前任务见 [完善计划](../../docs/improvement-plan.md)，运行见 [手册](../../docs/delivery.md)。不会因为发布仓库就自动运行任何历史runner。
