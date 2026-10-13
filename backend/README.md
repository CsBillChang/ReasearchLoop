# ResearchLoop 后端

从仓库根操作。入口：`api → services → storage`入队，`Worker → agents/graph → tools → services → providers/storage`执行。API不等待长研究任务；MySQL保存事实，SSE重放事件，Qdrant是可重建派生索引。

模块职责见 [文件地图](../docs/file-map.md) 和 [逐文件清单](../docs/code-inventory.md)。Supervisor动态委派Researcher并按需调用Reviewer；DPR精读为固定LLM节点，不强制每次问答使用。

```bash
make sync-index
make infra
make migrate
make start
make test
make check
```

来源/模型需要本机配置，见 [运行手册](../docs/delivery.md)。默认测试不调用DeepSeek；环境门控跳过不能报为真实验证通过。历史脚本显式冻结和隔离，不作日常入口。当前完善状态见 [status](../docs/status.md)。
