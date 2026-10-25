# 固定输入的重排线程对照准备

本实验准备工具只创建可核对输入与计划，不构造ONNX Session或运行评分；没有真实执行入口，不接入默认API/Worker或业务配置。

## 固定比较

复用词法路线比较中located_bm25_rerank的原问题、授权scope、候选ID/原序和全文。只比较CPU intra_op线程2与4；同权重/tokenizer/运行时、inter_op=1、batch4、sort_by_length=False、512-token拒绝边界保持。

准备器先重新选择候选并与源first-hot逐项核对，再按既有分词与padding配方生成每题/每批输入投影hash。输入不含答案标签、旧排名或旧分数。源运行没有保存实际feed，因此新投影不冒称与源实际张量逐值相同；真正执行前仍需检查Session输入名、可选token_type_ids及最终feed。

## 准备入口

从仓库根目录设置PYTHONDONTWRITEBYTECODE=1、PYTHONPATH=backend后，使用既有只读Python解释器执行：

```sh
python -m scripts.prepare_reranker_thread_comparison \
  --source-freeze /absolute/input/freeze.json \
  --source-results /absolute/input/execution \
  --source-manifest /absolute/input/output-manifest.json \
  --expected-source-freeze-sha256 <previously-frozen-sha256> \
  --expected-source-manifest-sha256 <previously-frozen-sha256> \
  --output /absolute/new-empty-output
```

两个expected SHA必须来自事前独立保存的原件记录，不能现场对待验证文件自算后回填；准备器在解析冻结与建立读取白名单之前验证锚点。历史raw会整体解析，但只使用身份字段，旧分数/排名不参与输入准备。

输出目录必须预先创建、为空且是绝对真实路径；不可覆盖或复用。准备器使用进程内守卫拒绝网络、SQL、子进程和输出外写入；这不是OS级沙箱。不得读取.env、启动业务服务、写默认依赖环境或自动下载模型。

## 后续评价

本轮只准备114题×2配置×3热轮=684格，加2初始化，全部not_executed；后续上限686评分、16134个query-passage对、60分钟，不自动重试或扩网格。真正执行使用当时重新核验的冻结与授权，本准备工具不提供执行门。

主时延是同期两配置的重排组件墙钟，包含分词、组批、评分和输出校验，另报模型推理时间。候选重选/读写/hash与初始化单列，不能与包含候选选择的历史整链路时间直接相减。质量用原可回答题及原非穷尽标签，报Recall@5/@8/@10、MRR@24、候选覆盖、逐题退步、失败和重复稳定性。

4线程应保持全部质量主指标；是否值得采用以预先约定的时延收益与资源要求审核，不预设线程更多就更快。CPU结果之后再独立评估一个Apple后端；若兼容/质量/效率不满足可保留CPU。完成对照后结束本阶段优化，不自动修改检索算法或切部署。

## 版本和环境

在隔离worktree/分支开发，保留源比较分支。只把公开代码、合成测试与本说明保存到GitHub草稿PR；材料、标签、模型、运行记录及私人开发文档留在本机。撤销共享代码使用git revert对应实现提交，不强推、不对部署目录执行reset。Git不回滚环境与数据；部署通过不写入和前后核对保持。
