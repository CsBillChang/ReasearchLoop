# 原量化重排模型的Apple后端兼容性检查

以既有四线程CPU对照结果为基线，先检查当前模型及运行时，只有资格通过才考虑同输入性能比较。此工具仅做metadata/static模式，没有模型Session、编译、评分、模型转换或性能执行入口。

## 检查与边界

- 显式冻结原权重/模型清单、已安装包版本、CPU基线报告及同版本官方CoreML注册/支持/编译缓存源码。先验证独立SHA锚点，再只读处理；官方源码仅作为文本证据，不导入或执行。
- 以protobuf线格式读取原ONNX图的节点/op元数据，跳过权重tensor内容，不安装onnx/protobuf或转换原权重。静态op计数与CoreML注册名称做精确匹配；已注册仍不保证dtype/shape被支持。
- 对原INT8关键DynamicQuantizeLinear/MatMulInteger缺注册的情况，输出not_qualified_for_direct_gpu_comparison。可能存在外围CoreML浮点分区，不能把静态结论夸大为所有图节点都不能执行；实际分区、设备分配和速度仍未测。
- 即使静态必要条件通过，也只输出runtime_partition_qualification_required，不能将provider枚举或初始化成功当GPU可用。get_available_providers可能触发原生设备枚举，结果不是GPU实际执行证据。
- ModelCacheDirectory只指定ORT的模型缓存目的地；Apple编译可先产生系统临时产物，失败还可能回退临时目录。Python进程守卫不覆盖所有原生/XPC写入，未确认隔离前不构造CoreML Session。

## 准备入口

使用既有只读解释器，设置PYTHONDONTWRITEBYTECODE=1和PYTHONPATH=backend后运行：

```sh
python -m scripts.inspect_apple_reranker_compatibility \
  --freeze /absolute/approved/freeze.json \
  --expected-freeze-sha256 REPLACE_WITH_PREVIOUSLY_FIXED_SHA256 \
  --output /absolute/new-empty-output
```

输出必须是预先创建的空绝对真实目录，只独占新增。进程守卫拒绝网络、SQL、子进程和输出外写入，不读取.env、不写部署.venv/源码/配置。原生设备枚举不构造模型，会如实记录这一边界，不冒称整个操作系统毫无副作用。

## 结果与收尾

缺关键算子支持或编译缓存隔离资格不足时，不运行GPU性能全量测试，GPU时延与质量保持N/A，保留四线程CPU既有结果。若未来需要换精度、改图、下载权重或安装依赖，属于另外的模型/部署方案，本阶段不自动扩张。

检查代码在独立分支保存到GitHub并记录git revert回退点；私人材料、权重、原始结果、官方源码证据和本机开发文档不上传。结果交用户审核后结束本阶段RAG优化，不自动合并或部署。

资料：[CoreML EP文档](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)、[v1.31.0算子注册表](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.31.0/onnxruntime/core/providers/coreml/builders/op_builder_factory.cc)、[编译缓存实现](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.31.0/onnxruntime/core/providers/coreml/model/model.mm)。
