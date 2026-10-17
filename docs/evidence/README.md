# 可公开的简历证据与记录型演示

本包整理已有实验和任务记录，不产生新回答。目标是让读者沿着“亮点 → 代码 → 验证范围 → 实际结果 → 局限”检查项目。PDF、个人问题、原始回答、模型请求/响应、收据、数据库和截图均不包含在本包中。

## 阅读与复算入口

| 文件 | 内容与可核查范围 |
|---|---|
| [retrieval.json](retrieval.json) | 6份材料的匿名片段目录、24题的相关标签、全部816份排名及保存耗时；包含两次未采用候选 |
| [reflection.json](reflection.json) | 4份固定稿、8个槽位的前后配对标签、正确原句保留检查和分阶段聚合投入；原整链失败另列 |
| [demonstrations.json](demonstrations.json) | 正常比较、压缩后同会话追问、核验失败后的partial交付；均为脱敏概述，没有复制私人对话或原始模型响应 |
| [provenance.json](provenance.json) | 包身份、数据文件hash、历史冻结身份、测量环境及本机部署身份；明确不同版本的区别 |
| [release.json](release.json) | 最终业务实现提交、110份运行源码hash、干净安装/启动/零模型测试范围；不包含私人配置或研究数据 |
| [简历证据映射](../resume-evidence.md) | 更完整的亮点、源码入口、分层验证及限制 |

从仓库根运行，Python标准库即可；无需密钥、数据库、向量服务、模型权重或第三方Python包：

```bash
PYTHONPATH=backend python3 -S backend/scripts/report_public_evidence.py
```

已有项目环境也可使用：

```bash
PYTHONPATH=backend .venv/bin/python -S backend/scripts/report_public_evidence.py
```

脚本只读JSON、默认输出到stdout。文件hash不符、缺题/重复测量、越出本题材料范围的排名、分母或Token合计不一致都会失败。它重算已保存数字，不执行检索、测量新的耗时、调用生成模型或重新判断标签的语义正确性。包内hash用于发现内容不一致，不是第三方签名认证。

## 简历亮点对应到哪里

| 亮点 | 代码入口 | 可展示证据 | 需要说明的限制 |
|---|---|---|---|
| ReAct与多Agent调度 | [graph.py](../../backend/research_loop/agents/graph.py)、[executor.py](../../backend/research_loop/workers/executor.py)、[coverage.py](../../backend/research_loop/services/coverage.py) | D1/D2的Supervisor与Researcher阶段及动作分账 | 三个角色的上下文/权限分离；不代表三套训练模型或每次都调用Reviewer |
| 工具辅助Reflection与局部修订 | [reflection.py](../../backend/research_loop/agents/reflection.py)、[verification.py](../../backend/research_loop/services/verification.py) | 固定4稿8槽组件；D3显示具体核验错误后的交付 | 组件与自主调度分开；D3实际没有局部修订，不能把不同批次拼成一条成功链 |
| 版本化混合RAG | [retrieval.py](../../backend/research_loop/services/retrieval.py)、[indexing.py](../../backend/research_loop/services/indexing.py) | 匿名排名可复算全部四路与未采用候选 | 质量与时延取舍保留；没有大规模泛化或生产SLA结论 |
| 分层Memory与上下文接续 | [memory.py](../../backend/research_loop/services/memory.py)、[context_workspace.py](../../backend/research_loop/domain/context_workspace.py)、[research_progress.py](../../backend/research_loop/services/research_progress.py) | D1/D2同一会话、同材料、显式压缩后的追问 | 两轮固定开发任务，不是无损压缩或任意长任务成功率 |
| 持久执行、SSE与文件保护 | [mysql.py](../../backend/research_loop/storage/mysql.py)、[checkpoints.py](../../backend/research_loop/storage/checkpoints.py)、[events.py](../../backend/research_loop/api/events.py) | 既有隔离领取/恢复/收据测试，部署读回见运行手册 | Checkpoint为项目自实现快照；匿名演示不是新的SIGKILL或SSE故障实验 |

## 检索：样本、公式与计时边界

历史批次为2026-10-11：6份PDF、63物理页、282片段，24题，其中21题可回答，3题缺证。公开材料编号P01–P06、片段编号C001–C282在三批中一致；原问题、原文引句和数据库身份不公开。相关标签由开发Agent在实验前按原文位置建立，非穷举相关性，也不是独立双人金标准。题目后来已用于开发，不能再称未见确认集。

- Recall@10：单题前10名中的已标注相关片段数 / 本题全部已标注相关片段数，然后对21题宏平均。不能换成“命中一条即满分”。
- MRR@24：前24条中首个相关片段名次的倒数；未命中为0，然后对21题取均值。
- Q04、Q20、Q24没有相关标注，仍留在执行与计时的24题中，质量均值排除这3题。空结果不证明客观上没有答案。
- 原批每题单次热运行，耗时为共享实测组件求和，P50跨24题取中位数，P95用nearest-rank：排序后第ceil(0.95×n)项。不是包含生成的回答耗时，也不是独立端到端检索测量。
- 两次优化各题各配置5次测量：先取每题中位数，再跨24题求P50/P95；质量按每题重复均值后对21题宏平均。重复计时不增加语义样本数，不混合不同日期的基线。

| 原路线 | Recall@10，21题 | MRR@24，21题 | P50，24题 | P95，24题 |
|---|---:|---:|---:|---:|
| BM25 | 66.67% | 0.3127 | 1.22ms | 1.75ms |
| 向量 | 76.19% | 0.3052 | 42.98ms | 70.44ms |
| RRF | 85.71% | 0.4167 | 43.97ms | 72.04ms |
| RRF＋Cross-encoder | 90.48% | 0.3767 | 1938.22ms | 2139.77ms |

简历中的均衡对照是**RRF相对纯向量**，不包含Cross-encoder。默认产品路线包含CE，不能套用43.97ms作为默认检索或回答耗时。两次未采用优化的所有候选也保留在JSON中：减少重排数量出现Recall退化，单独推理效率候选出现MRR退化，均未切换默认配置。完整候选表见[评测说明](../evaluation.md)。

历史环境为Apple M2/8GB、ARM64、Python3.14.6、本机推理2线程；模型为multilingual-e5-small和mmarco-mMiniLMv2-L12-H384-v1，RRF k=60、默认24候选重排。模型revision与6个文件hash见[模型模板](../../config/model-manifest.example.json)。冷初始化、建库成本、网络、数据库、生成请求和Qdrant服务内存不包含在上表的同一计时范围。

匿名包能够验证算术与排名身份，不能独立验证每条相关标签是否正确、原文解析是否完整或原始计时是否可再次达到。重新执行检索需另备合法PDF、固定解析/切分、索引、模型与环境；不承诺公开克隆后自动重现旧私有实验。

## Reflection：保留改对、未改与失败

2026-10-11在同4份受控稿、8个事前槽位上，保留已有开发Agent原文核对标签：

| 固定标签转移 | 数量 | 含义 |
|---|---:|---|
| supported → supported | 5 | 原正确句逐字保留 |
| contradicted → supported | 2 | 与原文矛盾的陈述已作局部修正 |
| insufficient → bounded_disclosure | 1 | 删除无据猜测并限定已读范围，没有假装核实未知细节 |

后续核验—局部修改组件共17次历史请求、681190 tokens：初审7次、修订3次、复核7次，unknown/供应商失败均0。公开包只包含阶段聚合投入；逐条收据和正文保留检查仍需本机私有原件。独立原文核对来自开发Agent，不是本次新真人评分。

此前完整调度链的4个任务均partial、没有patch，32次/611115 tokens；失败原位保留。两批执行设计不同，不能合并计算“成功率”或比较成普遍成本下降。组件由固定实验调用方进入复核，不是完整自主Agent的因果消融；4稿8槽也不能证明普遍纠错率或消除幻觉。

## 三条记录型演示

这些是已有轨迹的概述，非现场重新生成、原始对话导出或新浏览器录像。建议按D1→D2→D3讲解，展示不同能力与失败边界；前两条属于同一会话，第三条是另一批次。

| 记录 | 原任务摘要 | 实际阶段与交付 | 历史投入 |
|---|---|---|---|
| D1：正常比较，2026-10-16 | 基于两份已授权PDF比较方法，禁止保存 | Supervisor定目标→委派两名Researcher读取/查询→保留阶段发现→汇合→finish_task；根completed，子任务partial/unreviewed状态仍保留；交付1929字符正文，无稿件文件 | 16次，426511 tokens，181.72秒 |
| D2：压缩追问，同日同会话 | 不重新上传材料，追问两种方法的区别 | 沿用目标、附件及禁止保存权限→显式触发压缩/归档→子任务交接→交付1302字符回答；completed | 12次，285730 tokens，81.56秒 |
| D3：核验失败后交付，2026-10-17 | 核验已有比较回答，禁止保存 | Reviewer提交引句未满足位置合同→Supervisor收到具体错误→主动finish_task→保留1929字符原正文与独立1665字符处理说明；partial | 14次，693065 tokens；unknown/供应商失败均0 |

D1/D2共28次/712241 tokens，只执行两轮固定任务；包括模型自己的参数纠正，原失败行动保留。首问存在组合引用页码定位不精确的细节，短稿未核对全部图表；不包装成全部陈述正确。压缩过程中记录的是UTF-8字节上界，不能以投影大小变化声称实际Token同比下降。

D3没有局部修订，不能演示为“修订后继续”通过。核验执行失败不等于原回答被证实错误，也不等于原文缺证；处理说明中的科学线索仍未获独立核验。成果保留和准确报告失败符合合理partial交付，研究目标本身仍未完成。Token与耗时为各历史批次的实际记录，货币费用未测量，不表示未来单次任务的固定代价。

## 版本与本机原件复核

本包关联的本机部署读回身份为`local-8f5181d5f9750e51`，当时冻结303份入口文件并完成既有成果读回；这只是内容指纹，不是新Git提交或公开发布标签。新增公开报表/测试和文档须在最终发布时另纳清单，本轮未重启或改变运行中的业务源码。D1/D2、D3、原检索和Reflection各自使用不同历史冻结，见provenance；不能把历史结果说成当前代码全部重新实测。各批次的指纹只覆盖其原本记录的文件集合，覆盖文件数不同，不据此直接比较性能。

最终公开业务实现为[提交c01a4a4](https://github.com/CsBillChang/ReasearchLoop/commit/c01a4a414ff2f3e5f96436fc8319dfc818580970)，后续文档收口不改变运行源码。`release.json`提供逐文件身份；在公共克隆根目录可仅用标准库核对：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
record = json.loads(Path('docs/evidence/release.json').read_text())['runtime_source']
actual = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
          for name in record['files']}
assert actual == record['files']
assert hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() == record['sha256']
print('runtime source hashes match')
PY
```

这验证公共源码与发布记录的字节对应，不读取或证明用户本机运行进程、账户配置、原文内容或模型质量。最终干净环境检查没有新增生成请求，也没有重跑上述历史评测。

本机的[report_evidence.py](../../backend/scripts/report_evidence.py)额外核对原文字符区间、标签绑定、收据与原文件身份：

```bash
PYTHONPATH=backend .venv/bin/python backend/scripts/report_evidence.py
```

该命令依赖未公开的本机artifacts，公共克隆缺材料时会失败，不能当作默认安装/CI命令。历史归档可供原检索记录核对；Reflection组件的当前源码已有漂移，数值重算不能代替完整历史环境重建。原件中涉及凭据、私人路径、原始响应等信息，不应为解决缺文件而直接上传。

## 保留的负结果与总体边界

除上述候选和完整Reflection失败，还保留30篇采样批次：计划30篇、导入29篇，459页/1806片段；11个实际模拟会话中9 failed、2 partial、0 completed。该批次使用较严格累计配置，含程序与上下文问题，不代表当前日常默认成功率；后续修复/固定任务成功也不回写旧分母。详见[评测说明](../evaluation.md)。

本项目为本机单用户应用；三个Agent为角色而非三套基础模型，Checkpoint是自实现状态快照。未交付的矩阵不列为稳定亮点；没有学习型推荐收益、大规模用户效果、全任务引用正确率、任意长任务稳定性或总体Token下降证据。公开材料支持可检查的实现与限定范围结果，不承担这些额外结论。
