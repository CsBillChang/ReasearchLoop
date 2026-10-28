# 受控真实验证入口

仅在获得真实调用授权后使用 `run_real.py`。默认入口不发请求，准备与逐轮执行分开；本脚本不启动部署服务，不自动合并或部署候选。

## 范围

场景为 `real_scenarios.json` 的三轮合成研究记录：记录V1定义、明确更新为V2、新会话继续。只把 `prompt` 交模型，预期语义与机制检查保留给离线审阅。它不证明自然压缩收益、一般Memory成功率或独立成本提升。

`prepare` 仅在内存读取凭据、当前模型和官方端点，重新构造独立SQLite及runtime目录；不继承生产数据库、目录或工具配置。`execute` 只运行指定一轮；上一轮的机制和离线语义审阅均通过才允许下一轮。`semantic-review.json` 必须基于已经保存的实际回答，不能预填或由预期答案生成。

凭据只来自显式指定的配置文件，环境变量不能覆盖账户、模型或端点。官方域名解析出的IPv4/IPv6地址也必须为公网单播，拒绝回环、私网和多播。

```sh
.venv/bin/python -B backend/experiments/memory_continuation/run_real.py prepare --output artifacts/real-review/pilot-001 --credentials <credential-config>
.venv/bin/python -B backend/experiments/memory_continuation/run_real.py execute --output artifacts/real-review/pilot-001 --credentials <credential-config> --turn T1
```

输出须为当前worktree内的新目录。准备冻结源码、材料、模型和停止条件；重复轮次不恢复发送，变更源码或配置后须建立新批次。

## 逐轮语义审阅

每轮审阅本轮回答和实际保存的摘要，覆盖场景的全部 `required_semantics`；`applied` 不替代语义判断，空结论摘要不满足机制覆盖。审阅应先准备好字段结构，在真实输出产生后填写，600秒窗口不会因审阅暂停。

将本轮 `audit.json` 中 `semantic_review_binding` 的三个字段原样放入本轮 `semantic-review.json`，再填写实际审阅结果。`passed` 和各项通过值必须是JSON布尔值，字符串或整数不通过。示例中的hash必须用实际工件hash，不能复制其他轮次或批次：

```json
{
  "turn_id": "T1",
  "task_id": "实际根任务身份",
  "artifact_sha256": {
    "answer.md": "实际字节sha256",
    "memory.json": "实际字节sha256",
    "trace.json": "实际字节sha256"
  },
  "passed": true,
  "checks": {
    "answer_semantics": true,
    "summary_semantics": true
  }
}
```

T2还须增加 `correction_target_reviews`，每条实际关系对应一条明确语义判断。复制该关系的以下五个身份字段，再填写 `passed` 和说明 `reason`；审阅须确认目标是V1固定分母结论，新结论采用V2保留集分母，不能仅因为关系类型为 `updates` 就判通过。关系遗漏、无关更新、错目标、空理由或工件hash变化时停止后续轮次。

```json
{
  "correction_target_reviews": [{
    "previous_task_id": "实际T1根身份",
    "previous_key": "模型实际生成的旧结论key",
    "previous_content_hash": "实际旧结论hash",
    "replacement_key": "模型实际生成的新结论key",
    "relation": "updates",
    "passed": true,
    "reason": "基于真实新旧结论填写的目标语义审核说明"
  }]
}
```

上述关系身份只用于输出后的审阅，不进入模型prompt，不替模型生成摘要或更新关系。T3也保存同样绑定工件的回答与摘要语义审阅，便于最终交付核对。

## 发送门

- 三轮共享最多12次物理发送尝试、160000 Token的保守发送门与600秒绝对发送窗口，单请求输出冻结8192。
- Token预留按UTF-8输入JSON字节、格式余量和输出上限计算，不是供应商账单的数学上界。实际usage才是结算依据；未知预留不清零。
- 仅允许官方HTTPS模型端点；关闭连接重试、环境代理和重定向，拒绝本机生产网络及其他外部服务。
- 流迭代前后检查截止；阻塞读取仍受请求read timeout约束。截止或断流按未知结果处理，不重新发送。
- 首个工具错误、HTTP拒绝、未知结果、缺usage、机制失败或语义审阅失败即停止。

实际原件保存在本机SQLite、请求收据、响应原始字节与私有工件中，不发布到GitHub。HTTP请求头和凭据不写入这些记录。

## 结果解释

必须区分真实发送与本地 `not_sent` 收据；后者没有实际usage，不应当作未知收费。摘要 `applied` 仅表示索引通过结构与来源检查，仍需审阅语义。若在规划阶段失败，记为目标Memory路径未覆盖，不能据此评价摘要质量。
