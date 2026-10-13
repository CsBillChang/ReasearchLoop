# 当前状态

更新：2026-10-13。唯一当前进度入口；完整旧阶段记录已归档在本机 `docs/archive/20261013-before-readiness/`，原始证据继续留在 `artifacts/`。

## 当前结论

- AG0–AG7按原约定范围完成，本机单用户应用已交付；不重新开启阶段。
- 已交付维护：自然语言偏好、最近对话改名/日期分组、累计Token预算、只读退避、阅读清单。
- 已知通用缺陷：在特定长任务/受限工具范围下，压缩后目标ID和证据读回指引接续不足。P0待实施，不宣称任意长任务可靠。
- 实验矩阵：工程及隔离验证完成，真实交付未通过；最后批次8请求/166853tokens、unknown0，未部署。暂缓，不再作为其他独立功能前置条件。
- 检索默认保持原RRF＋纯CE、24候选、batch4/2线程。旧候选存在质量退化，未上线。
- P1证据整理与后续模拟采样待执行；P2推理效率与P3推荐反馈可选。具体合同见 [完善计划](improvement-plan.md)。

## 本轮整理与首次发布

本轮范围为文档重整、源码清单、公共配置说明、发布排除规则和GitHub初次上传准备。未执行P0–P3、没有DeepSeek/论文检索/精读调用，没有修改外部简历、生产配置、数据库或重启服务。

旧文档逐字存入本机archive；代码保持原目录，实验与历史工具在文件地图中明确分类，避免搬迁破坏import/恢复/冻结脚本。本轮检查：后端392项运行、373通过、19项环境门控跳过；前端9项通过；make check通过。暂存树260文件，发布模式扫描和本机真实凭据值对照均无命中，活动Markdown相对链接无缺失。业务源码hash全部保持。原有2222份材料/记录中仅data/artifacts的目录说明有预期更新，其余hash一致；不将此文件核对称为数据库快照核验。

GitHub上传使用用户指定仓库的授权连接；目标公开范围见publication.md，实际提交以仓库历史为准。未上传.env、data/artifacts原始内容或docs/archive。本机未安装GitHub CLI/HTTPS凭据，采用已有GitHub连接提交，不要求在聊天中提供密钥。

## 历史证据索引（本机，不随公开仓库分发）

| 范围 | 本机目录 | 边界 |
|---|---|---|
| AG6交付 | `artifacts/ag6-current-draft-delivery-20261011/` | 用户认可两份成果；不等于任意论文正确 |
| AG7检索 | `artifacts/ag7-4-evaluation-20261011/` | 6 PDF/24题组件对照 |
| Reflection | `artifacts/ag7-4-reflection-component-20261011/` | 4稿/8项标注组件验证 |
| 本机交付 | `artifacts/ag7-5-followup-closeout-20261011/` | 固定PDF同会话追问完成 |
| Token/退避/清单 | `artifacts/resume-closeout-foundation-20261012/` | 零模型及隔离/浏览器验证 |
| 矩阵失败 | `artifacts/resume-evidence-matrix-locator-20261013/` | 同文档错页及压缩接续缺口，0矩阵 |

旧失败、收据、未知用量和原成果不改写；公开汇总见 [评测说明](evaluation.md)。
