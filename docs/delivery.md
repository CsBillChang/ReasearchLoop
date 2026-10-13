# 本机运行、备份与恢复

本说明针对当前 Mac 的单用户本地应用。AG7.5及AG0–AG7约定范围已完成；修复后的同会话追问真实完成，复用原首问与同一PDF，现可以最终交付。权威进度见 [status.md](status.md)，最终复验报告（本机历史记录）保留旧失败和实际边界；运维/恢复证据见原AG7.5报告（本机历史记录）。

## 日常启动

首次安装或锁文件变化时：

```bash
cd ReasearchLoop
make sync-index
(cd frontend && npm ci)
make infra
make migrate
```

确认项目 `.env` 已配置数据库、DeepSeek；不要将密钥粘贴进聊天、文档或命令参数。基础服务为本项目 Compose 的 MySQL/Qdrant；模型为项目固定清单指定的本机 ONNX 文件，启动时不重新下载或换模型。保留 `data/ag71-preflight/models/manifest.json` 及其权重；权重缺失会明确降级，不代表检索实测已通过。

重启电脑后打开 Docker Desktop，待 Docker 可用，运行：

```bash
cd ReasearchLoop
make infra
make start
make status
```

访问 [科研对话](http://127.0.0.1:4173/)。API 为 `http://127.0.0.1:18082`。`make start` 管理 API、前端、研究 Worker、简报 Scheduler 四个进程；重复相同配置启动复用当前进程。启动调度器不等于启用订阅，订阅是否启用以本机个人面板实际配置为准。启用后按个人面板配置执行，默认上海时间09:00、最多5篇候选，不自动精读；休眠/停机期间不能执行，恢复后合并补一份。

无需再次执行任何 `run_ag*` / `evaluate_*` 验收脚本。旧批次及旧 unknown 不重跑。普通任务使用独立日常配置，不能挪用封存验收余额。

## 状态、停止和只读演示

```bash
make diagnose
make stop
make demo
```

- `diagnose`：分别报告密钥存在性、数据库迁移、Qdrant、进程/Worker就绪和未结算请求，**不会调用模型**。
- `/api/health`：进程及配置存在性；`/api/ready`：数据库迁移；均不证明上游模型可用。
- `stop`：只给启动记录、命令、启动时间、随机标识全部匹配的本项目进程发 TERM，不按进程名清扫、不强杀其他项目。Worker完成当前任务执行并结算在途请求后退出；15秒内尚未结束会返回非零并保留进程，稍后 `make status` / `make stop`。不能用强杀来让备份通过。
- `demo`：`http://127.0.0.1:4174/`，只读展示已有成果，写API拒绝403，不启动Worker/Scheduler。停止演示：`PYTHONPATH=backend .venv/bin/python backend/scripts/manage.py stop --profile demo`。
- 端口冲突时拒绝启动；旧手工 `make api/worker/frontend/scheduler` 不会被自动接管，须在其原终端停止。正常 TIME_WAIT 不误报为监听进程。
- 进程登记、心跳和日志位于 `data/runtime/<profile>/`。更换配置先停止再启动。启动后不是操作系统自启动服务，电脑关机后仍需上述命令。

## 冷备份

先停止所有本项目日常、演示和隔离实例，等待运行任务结算。备份使用应用级独占锁与一致性数据库事务；API/Worker/Scheduler持有共享锁，不能与备份同时运行。不防护绕过应用的手工SQL/外部文件编辑，操作期间也须停止这些写入。

```bash
make stop
# 目标必须不存在，且在应用 data 目录之外；自行选唯一名称。
make backup ARGS="--path backups/local-20261011-01"
PYTHONPATH=backend .venv/bin/python backend/scripts/backup_restore.py verify \
  --path backups/local-20261011-01
make start
```

备份包括全库业务记录、正文原件/解析缓存、项目笔记、相关附件、来源配置的脱敏副本及模型/索引重建清单；数据库关联的历史目录原件也会收集。已有文件做SHA-256验证，成功保存笔记或必要PDF缺失则拒绝。`.env`、SQLite二进制库文件及模型权重不复制；数据库按表导出。**备份仍包含私人研究、原始messages和路径，未加密，不是公开演示包**；目录权限700，不对外上传。机器迁移需另行妥善保留密钥、锁文件、来源凭据和冻结模型权重。

## 隔离恢复

恢复目标必须是**新的空本机 MySQL 库** `researchloop_restore_*` 和不存在的新文件目录。用现有数据库管理方式创建空库并给项目账号授权；不要使用生产库名，不删除生产表。将该空库连接串放入本机环境变量 `RESEARCHLOOP_RESTORE_DATABASE_URL`，勿输出它。

```bash
make stop
make restore ARGS="--path backups/local-20261011-01 --target-dir data-restores/restore-20261011-01"
```

工具在写目标库前校验清单、内容hash、路径越界和重复归档，拒绝非空库/已有目录。失败保留目标以供诊断，不清空或覆盖后重试；使用新的空目标重做。恢复保留 raw response、request_id、usage 和 unknown 状态；不自动重发。只重定位结构化文件字段，普通正文和原始messages保持原样。

`restore-report.json` 返回新的 `data_dir`、`output_dir`、独立 `retrieval_namespace`，以及文件映射和收据一致性结果。配置 `DATABASE_URL` 为恢复库、`RESEARCHLOOP_DATA_DIR` / `RESEARCHLOOP_OUTPUT_DIR` / `RESEARCHLOOP_RETRIEVAL_NAMESPACE` 为报告返回值，再以独立profile只读启动：

```bash
PYTHONPATH=backend .venv/bin/python backend/scripts/manage.py start \
  --profile restored --readonly --api-port 18084 --frontend-port 4178
```

恢复目录中的 `data/restore-hold.json` 阻止普通执行；本版默认只读核对，不提供无审核的生产切换/自动恢复旧任务按钮。先核对文件、任务、未结算收据，再决定后续启用，不删除hold来规避检查。

BM25词项保留，Qdrant向量派生状态置待重建。使用恢复报告的独立namespace和已有 `index_evidence.py --help`，对**该恢复库实际授权页**显式重建；不要写回原集合。AG7.5已在真实隔离MySQL中恢复43表/22文件，验证原始收据不变，并在独立Qdrant集合重建3个片段。这是恢复演练，不是重新评测检索效果。

## 当前边界与故障定位

| 现象 | 处理 |
|---|---|
| DATABASE_NOT_READY | 确认 Docker、项目端口/连接串及 `make migrate`，不清空数据库 |
| MODEL_KEY_MISSING | 本机配置 `.env`；只展示成果可使用 `make demo` |
| PORT_IN_USE_OR_INVALID | 查看原进程/终端；不自动杀掉未知服务 |
| stop 尚未退出 | 等待在途调用结算并查询状态；备份暂不执行 |
| started/unknown | 保留收据并诊断远端执行情况，禁止盲重发或补用量为0 |
| Qdrant或本机重排不可用 | 返回降级原因，可补读正文；健康检查不冒充检索成功 |
| 追问附件不可读 | 新版同会话自动绑定最近附件集合，仍由当前任务重新读取授予页证据；跨会话不继承，当前显式附件优先；旧失败任务不改写为成功 |

单用户本地应用，无多用户鉴权、远程公网部署、Obsidian同步或任意论文零错误保证。简历技术实现及未量化部分见 [resume-evidence.md](resume-evidence.md)。

## 交付后工具治理与阅读清单

`RESEARCHLOOP_TASK_TOKEN_LIMIT=1000000`、`RESEARCHLOOP_TASK_TOKEN_TAIL=32768` 控制新根任务及子角色的共享Token准入；修改.env后安全重启，仅新任务生效。使用量可在聊天回答的“记忆与执行详情”查看。估算、真实结算、未知占用分开；不足时交付已有成果和原因。迁移010只追加4表，正常 `make migrate` 即可，不需要重建数据。

聊天顶栏“阅读清单”查看待读/已读，先从结果的“论文与笔记”来源卡加入。操作不会开始模型或保存Markdown；清单状态与审校/精读完成不同。本地附件尚无论文元数据时入口禁用并说明原因。

## 公共克隆配置准备

1. 复制.env.example为.env，仅在本机填写DeepSeek连接和数据库信息；不要把真实密钥提交到仓库。Compose的密码仅为本机开发默认值，服务只绑定127.0.0.1。
2. 来源有两种明确配置方式：复制config/dpr-sources.example.json到data/config/dpr-sources.json后填写有权限的只读连接并启用所需来源；或从自己下载的DPR配置导入：

```bash
PYTHONPATH=backend .venv/bin/python backend/scripts/import_dpr_sources.py /path/to/dpr-config.yaml
```

该工具不读取上游.env、不运行采集脚本；目标已存在时拒绝覆盖。不得在终端输出来源凭据。
3. 本机RAG需另外获取references.md列出的两个固定revision文件（权重、tokenizer.json、config.json）。将文件放到data/local-models/embedding和reranker；按config/model-manifest.example.json的文件SHA-256核对，复制模板到data/local-models/manifest.json，并在.env设置 `RESEARCHLOOP_RETRIEVAL_MODEL_MANIFEST` 为该文件绝对路径。模板里的directory相对于仓库根，手工从其他工作目录启动时应改为自己的绝对路径。不要修改hash以绕过校验。
4. 本机ARM64权重未在其他架构验证；缺模型时应显示unavailable/降级，不能报完整RAG就绪。启动不自动下载或替换模型。既有本机使用者无需改动原data配置。
5. 新克隆没有论文和个人数据。可由聊天上传合法PDF、配置来源后检索；不要下载私人artifact冒充初始化步骤。`make test`缺本地PDF/上游对照源码或隔离服务时会跳过对应门控测试。
