# 同版本浮点重排器的MPS实验

独立组件实验：固定既有问题/授权候选/顺序/全文及tokenizer/batch4/512边界，先用同一训练revision的浮点权重检查Apple MPS，再与既有INT8 ONNX CPU4方案比较。此入口不接API、Worker或默认检索配置。

小样本由输入长度分位位置事前确定，不看模型分数或标签；CPU4/MPS FP32/MPS FP16各一轮和独立warmup。FP16必须通过完整执行、实际张量/设备、有限输出及预定数值/已标质量门槛。只有独立reporter产出且完整绑定freeze和pilot原件的通过收据，performance入口才放行。全量固定原114题、两个变体、3热轮，顺序轮换，失败/unknown/未执行保留。

GPU前后同步，主墙钟含同样分词/padding、输入核验、推理、输出读回校验与排序；加载和首次编译单列。独立报告Recall@5/8/10、MRR@24、候选覆盖、分数/排名漂移与重复稳定性；非穷尽历史标签不代表独立泛化或答案正确率。比较后端与精度同时变化，结论只归于两个完整方案，不当作纯GPU硬件因果。

解释器和所有新依赖在独立实验venv安装，固定版本，不改日常依赖或业务代码；权重同revision、显式hash、安全safetensors、本地只读加载，不启remote code、自动下载或MPS CPU fallback。离线评分禁止网络/SQL/子进程/.env，只允许新证据独占新增和显式实验cache/tmp可变。Python边界不等于对全部Metal/driver缓存的OS沙箱；另核实际部署文件/依赖/资产/旧证据保护。

公开仓库仅保存代码、合成测试、实验依赖和本说明；模型、私有输入/标签、环境配置和原始结果留本机。使用独立分支和git revert回退，完成后交用户审核，不自动合并、采用或部署。

参考：[PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)、[MPS同步](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html)、[MPS回退控制](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)、[CrossEncoder效率](https://www.sbert.net/docs/cross_encoder/usage/efficiency.html)。
