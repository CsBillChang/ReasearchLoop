# read_paper 严格等值正文副本去重

候选只影响新模型请求的观察副本：成功 `read_paper` 返回的页在非空字符串 `text == parsed_markdown` 时保留完整 text，移除该页 parsed_markdown。不同、缺失、空或非字符串值不处理；跨轮复用或有歧义的工具调用 ID 保守保持。不合并页，不改证据身份、页序、原文、资产、范围和缺口。

默认关闭。仅在隔离实验中用显式 `Settings(context_deduplicate_read_paper=True)` 或 `RESEARCHLOOP_CONTEXT_DEDUPLICATE_READ_PAPER=1` 开启；三角色都经过各自的新请求准备入口。原收据、完整档案与证据存储保持，启用后的新档案有独立投影版本身份。恢复已持久请求仍使用原 request_json，成功/unknown 不重新投影或盲重发。

`read_paper_deduplication` 记录准备期间处理过的工具调用和页下标并集，不能据它推算最终 Token 或时延；真正发送内容的容量估算仍由原计量策略计算。原始归档与 projected_json 分别保存原件及实际待发内容。启用时无目标材料的短请求不额外写入档案。

合成零模型验证入口（先准备隔离依赖、临时目录与禁止网络的运行环境）：

```sh
PYTHONPATH=backend:backend/tests python -B -m unittest unit.test_read_paper_dedup
```

测试覆盖严格等值边界、三角色与短请求、原件和档案、容量/预算、真实 Provider 到 HTTPX MockTransport 的 JSON/SSE 协议，以及旧收据恢复。Mock响应中的 usage 是合成数据，不能作为真实模型的性能结果。私人历史请求与实际用量不随代码上传。

部署采用和真实模型 A/B 需要独立授权与冻结：两版用相同源码基线、RAG、模型、材料、历史与完成要求，仅切换本开关；核对完整任务族实际 Token 与端到端墙钟，保留失败与缺失用量。代码回退用实验开关或共享提交的 revert，Git 不恢复数据库或向量状态。
