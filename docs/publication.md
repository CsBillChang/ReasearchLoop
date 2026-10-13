# 公开仓库边界

目标仓库： https://github.com/CsBillChang/ReasearchLoop （保留用户建立的拼写）。

## 上传

- backend/research_loop 业务源码；backend/scripts运维/显式评测工具；backend/migrations追加迁移；后端测试与合成夹具。
- frontend静态源码、代理、测试、package.json与锁文件。
- 当前docs、AGENTS、README、Makefile、compose.yaml、依赖锁、空凭据配置模板、第三方许可证。
- 精简脱敏的评测汇总；本机历史验收脚本明确不是日常入口。

## 不上传

.env及任何真实连接/凭据；data与artifacts原始内容（仅目录说明可公开）；数据库/备份/恢复副本；PDF/全文解析缓存/个人笔记/聊天/画像；模型权重；原始messages/response/usage与截图；本机历史文档archive；.venv/node_modules/cache/IDE本机配置。

.gitignore防止意外加入；提交前运行 `PYTHONPATH=backend .venv/bin/python backend/scripts/check_publication.py --staged` 检查显式暂存列表、敏感模式、个人路径、类型与体积。扫描只输出位置/规则，不输出凭据值；扫描不是密码学安全保证，需要检查实际清单。

本机数据不会因为未上传而删除。不能用公共仓库同步数据库或模型权重。上传不代表功能修复、生产重启或当前实验矩阵获准发布。

## 许可证与运行前提

复制的DPR代码附带其MIT许可证和归属，见 THIRD_PARTY_NOTICES.md。项目整体暂未指定再分发许可证，不把第三方MIT自动当作所有依赖或整个项目的许可证。

外部服务账号与可访问数据由使用者自行配置；不会发布作者的Supabase连接配置或假定其服务永远可用。模型权重按官方固定版本获取，清单模板不含本机路径。公共仓库为源码与文档交付，不是携带私人数据的一键运行镜像。
