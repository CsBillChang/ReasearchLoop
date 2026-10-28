# 阶段发现ID错误的单变量实验合同

本实验从已保留的Supervisor范围优化版与持久首错实验入口开始。A使用原交接处理，B仅增加窄的未知发现ID异常，并在`record_research_progress`处理器中把该类型转入现有一次参数反馈通道。原错误仍拒绝、阶段状态不变；合法补交后才形成未核验候选。其他执行错误、权限、版本与重复失败规则不放宽。

## 精确源码与身份

独立类型为`research_progress_id_recovery_v1`，使用`handoff_source_contract`，不借用旧Supervisor提示词合同。`SUPERVISOR_KINDS`仍只包含原两种提示词实验，新增`PAIRED_KINDS`统一选择共用配对控制。

静态合同只接受两份业务文件的批准变换：`services/research_progress.py`增加`UnknownFindingIdError(ValueError)`及对应分支；`agents/researcher.py`只捕获该类型并返回`validation_error`与`UNKNOWN_FINDING_ID`。宽化为捕获全部ValueError、增加其他语义或改动工具schema均拒绝。

Supervisor整个文件及系统提示词、图、schema、其他业务、共用脚本、迁移与配置在两侧完全一致。提示词基线须保留原第1条和既有第4、7条；双方正文去重均开启。共同Ledger的`first_error_policy_version=1`进入独立合同。两份源码目录必须独立，源码清单与全部摘要被资格和运行配置绑定。

启动与每次发送前重新检查完整源码合同，并核对实际已加载Supervisor、Researcher及阶段进展模块的源路径。两版不能共享已加载的另一源码包，也不能把旧的目标解释候选混入本轮。

## 沿用运行门

新类型继续要求相邻配对顺序、外部语义审核目录、一次性父调度授权、此前配对审核、历史累计预算与绝对截止时间。资格收据及授权grant绑定对应的交接源码合同；子任务不能通过生成或改写审核结果获得下一对权限。两格结束且运行收据合格后才进入原有语义门；已关闭批次不能恢复。

此入口适配仅为离线准备，不授予真实模型调用或部署权限。真实材料、任务ID、标签、回放数据库及轨迹均留在私有工件目录。合成合同测试只能证明范围、身份和门控；真实模型会否利用纠正机会、最终答案质量及端到端时延/Token收益需独立授权验证。
