# 同checkpoint Apple GPU 8位权重试验

本实验通过独立MLX适配复用原XLM-RoBERTa单logit分类重排模型。先验证未量化FP32前向，再比较74个Linear的8bit affine/group64权重量化（W8A32）与四线程ORT INT8 CPU；Embedding、激活、LayerNorm和bias保持FP32。不能称整模型纯INT8或假定与原动态INT8数值等价。

只读原模型/问题/候选/授权范围，保留原tokenizer、batch4、512上限、候选原序与raw-logit排序。固定12题资格/试点通过绑定原件的门后，才执行114题CPU4/MLX8同期三热轮。资格/试点/正式上限分别13/13/686个score，合计712；失败/未执行保留，不按结果修改量化配置或样本。

MLX的惰性执行须在显式GPU stream内eval并同步后计时；记录真实输入回读、量化层/调用证明。统一内存设备证明不冒充Torch逐tensor.device核验，未跟踪全部Metal kernel时不宣称全程整数算术。质量与时延分开报告，不使用旧FP16时延计算同期收益。

所有依赖、缓存、权重转化及私有原件应位于独立实验目录，执行过程离线并受Python守卫限制；原生OS/Metal缓存不能由Python守卫完整约束。业务源码/默认/部署配置不修改。公开仓库只含源码、合成测试和说明，真实输入与模型不上传；以独立分支和提交保留回退点，结果审核后再决定是否采用。

数学结构参考Hugging Face Transformers的Apache-2.0 XLM-R实现；MLX API来自Apple的MIT实现，不引入第三方GPL encoder实现。原模型版本、资产hash、全部解析包RECORD、执行预算与原件引用在本机显式冻结。
