# 简历声明与证据边界

外部简历未修改。用户认可的候选文案以当前架构为基础，压缩接续增强须P0完成后才能正式成立；可选提速和推荐反馈不能预写为成果。

| 声明 | 当前证据 / 边界 |
|---|---|
| ReAct＋Reflection＋Memory、LangGraph三角色 | Supervisor/Researcher/Reviewer与父子任务真实存在；角色不是三套训练模型；Researcher并行使用ThreadPoolExecutor |
| 动态行动/取证/局部修订/有限结束 | 已有AG2/AG3/AG6场景与Reflection组件验证；不保证任何论文或任务零错误 |
| 混合RAG | 版本化原文、BM25/Qdrant/RRF/Cross-encoder已接入；检索限制在授权材料；效果见evaluation |
| 会话与长期Memory | 显式偏好、可撤销推断、历史召回、抽取式压缩已实现；特定压缩接续缺陷尚待P0 |
| 个性化增量简报 | 每日09:00上海时间/最多5候选/曝光去重/停机合并补报；元数据规则排序，不是训练的推荐器；专用正负反馈未完成 |
| 持久执行与SSE | 原子claim/租约、收据恢复、Last-Event-ID、幂等与安全保存已验证；Checkpoint为项目自实现，unknown不保证exactly-once |
| 结构化矩阵 | 工程/隔离验证已有，但真实交付失败，暂不写成已完成亮点 |
| 效果与规模 | 固定小样本组件数据；无大规模用户效果、全库引用准确率或总体成本降低证明 |

[已有量化结果与分母](evaluation.md) · [完善范围](improvement-plan.md)。原来的详细本机证据链接与历史目标稿路径保存在docs/archive中，不作为公开仓库的运行依赖。
