# 严格等值去重的真实对照入口

共同实验层在同一冻结业务源码上仅切换 `context_deduplicate_read_paper`，不使用旧H1的两份业务版本。`run_read_paper_ab.py`负责一个带轮次身份的格，`run_read_paper_batch.py`按冻结矩阵只执行一次，凭据仅经标准输入传入；私有授权、材料、账本和轨迹不随源码发布。

运行前以独立目录、venv、SQLite、Qdrant端口/集合及解析缓存完成两版材料资格。检索显式装配当前 `located_fp16 + mps_fp16`，记录实际组件身份；不把Settings名称当作实际路线证明。已有合法索引收据复制后逐点读回，任务权限仍由原生读取入口授予。

预算、动态官方DNS与压缩传输守卫来自已验证的实验层，独立参数完整覆盖总Token、请求、金额与执行绝对截止。SSE和JSON均须完整解码压缩EOF及合法usage后结算；原生收据与HTTP账本每格对账，任一unknown/缺usage/身份矛盾停止，无自动重发。父驱动计时包括初始化与预热，子进程硬截止；保留收尾时间。

视觉请求必须在共同manifest中显式冻结`request_contracts.visual`，并通过材料资格收据绑定。它仅放行现有业务节点的官方Flash、关闭思考、8192输出、非流式无工具合同，以及user消息中的1—2张内联PNG（整页或整页加裁区）。图片尺寸和字节上限在发送前核对；不放行外部图片URL、Files API或额外模型参数。验证复用已有PyMuPDF，不改变部署依赖。默认未提供视觉合同仍拒绝图片。

文字和视觉共用一个根任务族/整批账本；视觉按完整输入容量上界加8192输出预留，不按图片张数重复增加实际usage。确定`not_sent`单独列示原始空usage，仍触发停止；成功原生记录只与实际settled物理记录逐响应ID和usage核对，不能把本地拒绝误报为远端丢失收据。unknown、拒绝响应或对账矛盾保持停止。

续跑使用新一次性账本并显式传入`ledger_limits.carried_usage`，包含历史tokens、requests、nano_yuan和向上取整的seconds。批次原始上限保持不变；历史消耗计入每次预留和整个进程组绝对截止，新根任务限额单独计算。资格收据与批准文件绑定同一carry合同，不能重开旧真实账本或把旧控制下的单格拼入新配对。

合成资格入口：

```sh
PYTHONPATH=backend:backend/tests python -B -m unittest unit.test_harness_ab_budget unit.test_harness_ab_transport unit.test_harness_ab_network unit.test_dedup_ab_transport unit.test_dedup_ab_runner unit.test_visual_ab_contract
```

上述替身验证不表示远端或语义评分通过。正式运行必须有新冻结、批准的金额与样本合同、公开源码回退点和一次性空账本，不能直接复用旧授权或默认预算。真实主指标仍为完整任务族实际input+output Token与根任务交付墙钟；失败、partial、未创建和质量差异均保留。
