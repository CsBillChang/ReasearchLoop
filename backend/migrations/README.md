# 数据库迁移

旧基线的 SQL 已退出活动项目。当前目录是 2026-10-06 重建后已实现的版本迁移，不代表当前数据库仍为空：

| 文件 | 内容 |
|---|---|
| 001_initial.sql | 任务、事件、行动、模型收据、论文正文、草稿、保存及快照 |
| 002_runtime_recovery.sql | Worker 租约 token、模型发送/请求身份与恢复字段 |
| 003_multi_agent_tasks.sql | 父子任务、角色、授权、澄清、核验、共享额度；含已退役被引统计表 |
| 004_note_exports.sql | 被审稿内容 hash 与带审校提示的导出文件 hash 分开记录 |

追加版本：005 Reflection、006 Memory、007 产品接续、008 版本化证据索引、009 订阅简报、010 交付后累计Token与阅读清单。010只新增四表，不更新既有任务、预算或论文；历史无Token策略的任务保持原恢复行为。

`make migrate` 调 backend/scripts/migrate.py，根据 schema_version 顺序执行未应用版本。已经部署的迁移不可原地改写；需要改变表结构时新增迁移。测试使用独立库，不重置生产库。storage/mysql.py 另有 ensure_schema 的创建缺表代码；它不替代版本迁移，两个定义存在同步维护责任。
