"""本机冷备份与隔离恢复；不覆盖已有数据库、文件，不自动启动任务或重发请求。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlparse
from uuid import uuid4

from research_loop.config import Settings, paper_sources_path
from research_loop.services.runtime import application_lock, atomic_json, identity
from research_loop.storage.mysql import Database

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    """按文件字节计算身份，既用于原件也用于恢复后核对。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tables(db: Database, conn=None) -> list[str]:
    """枚举当前库表名并检查 SQL 标识符；不接收任意插值表名。"""
    query = 'SHOW TABLES' if db.kind == 'mysql' else "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    rows = db.fetchall(query) if conn is None else list_rows(db, conn, query)
    names = sorted(next(iter(row.values())) for row in rows)
    if any(not re.fullmatch(r'[a-z_][a-z_0-9]*', name) for name in names):
        raise ValueError('INVALID_TABLE_NAME')
    return names


def list_rows(db: Database, conn, query: str) -> list[dict]:
    """在同一快照连接读回，避免各表来自不同事务时刻。"""
    cursor = conn.cursor()
    cursor.execute(query)
    return [dict(row) for row in cursor.fetchall()]


def snapshot(db: Database) -> dict:
    """应用停写后读取一致性事务；只读取业务表，不创建或升级 schema。"""
    with db.connection() as conn:
        cur = conn.cursor()
        if db.kind == 'mysql':
            cur.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            cur.execute('START TRANSACTION WITH CONSISTENT SNAPSHOT')
        else:
            cur.execute('BEGIN')
        return {name: list_rows(db, conn, 'SELECT * FROM `'+name+'`') for name in tables(db, conn)}


def path_values(value):
    """遍历结构化 JSON 中的绝对文件引用；不把普通论文句子解释为路径。"""
    if isinstance(value, dict):
        for item in value.values():
            yield from path_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from path_values(item)
    elif isinstance(value, str):
        if value.startswith('/') and '\n' not in value and len(value) < 4096:
            yield Path(value)
        elif value.startswith(('{','[')):
            try:
                yield from path_values(json.loads(value))
            except (ValueError, TypeError):
                pass


def assert_no_legacy_writers() -> None:
    """旧手工入口可能没有共享锁；发现本项目进程则拒绝冷备份，不代为杀掉。"""
    output = subprocess.run(['ps','-axo','pid=,command='],capture_output=True,text=True,check=True).stdout
    for line in output.splitlines():
        pid, _, command = line.strip().partition(' ')
        if pid == str(os.getpid()):
            continue
        if str(ROOT) in command and any(marker in command for marker in (
                'manage.py serve', 'frontend/dev-server.mjs', 'scripts/run_worker.py', 'scripts/run_scheduler.py')):
            raise RuntimeError('LEGACY_OR_MANAGED_PROCESS_ACTIVE: 请先停止本项目应用')


def backup(settings: Settings, destination: Path) -> dict:
    """导出全库、数据目录和 DB 关联原件/笔记，额外文件引用跨目录也纳入。

    Args:
        settings: 当前待备份实例配置；密钥不写入备份配置。
        destination: 必须不存在的新目录，不覆盖任何旧备份。

    Returns:
        已校验的清单。备份包含私人研究内容，不能作为公开演示包。
    """
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(settings.data_dir):
        raise ValueError('BACKUP_DESTINATION_MUST_BE_NEW_OUTSIDE_DATA')
    with application_lock(ROOT, exclusive=True):
        assert_no_legacy_writers()
        rows = snapshot(Database(settings.database_url))
        running = [r['task_id'] for r in rows['tasks'] if r['status'] == 'running']
        if running:
            raise RuntimeError('RUNNING_TASKS_REQUIRE_SETTLEMENT: 不强杀或改写任务来备份')
        files = {p.resolve() for folder in {settings.data_dir, settings.output_dir} if folder.exists()
                 for p in folder.rglob('*') if p.is_file() and not p.is_symlink()}
        missing = []
        for path in path_values(rows):
            if path.is_file():
                files.add(path.resolve())
            elif path.suffix.lower() in {'.pdf','.md','.json','.png','.jpg','.jpeg'}:
                missing.append(str(path))
        # 解析缓存的 PDF 为派生邻接路径，不一定直接存于数据库字段。
        for doc in rows['documents']:
            cache = Path(doc['cache_path'])
            if not cache.is_file():
                raise ValueError('MISSING_DOCUMENT_CACHE: '+doc['document_id'])
            if doc['parser_name'] != 'jina':
                pdf = cache.parent.parent/'pdf'/(doc['content_sha256']+'.pdf')
                if not pdf.is_file() or digest(pdf) != doc['content_sha256']:
                    raise ValueError('MISSING_OR_CHANGED_ORIGINAL: '+doc['document_id'])
                files.add(pdf.resolve())
        for receipt in rows['save_receipts']:
            if not Path(receipt['path']).is_file():
                raise ValueError('MISSING_SAVED_NOTE: '+receipt['receipt_id'])
        # 配置仅记录无秘密可重建字段；认证信息由原 .env 或重新配置提供。
        config = {'data_dir': str(settings.data_dir), 'output_dir': str(settings.output_dir),
                  'model': settings.deepseek_model, 'retrieval_namespace': settings.retrieval_namespace,
                  'qdrant_url': settings.qdrant_url, 'parser_name': settings.parser_name,
                  'parser_version': settings.parser_version, 'parser_config': settings.parser_config,
                  'model_manifest': str(settings.retrieval_model_manifest or ROOT/'data/ag71-preflight/models/manifest.json'),
                  'secrets_included': False}
        source_config = paper_sources_path(ROOT)
        if source_config.exists():
            def without_secrets(value):
                """保留来源/RPC配置，认证字段只记录需要重新配置。"""
                if isinstance(value,dict):
                    return {k:('[configure locally]' if any(part in k.lower() for part in ('key','token','password','secret')) else without_secrets(v)) for k,v in value.items()}
                if isinstance(value,list):
                    return [without_secrets(v) for v in value]
                return value
            config['paper_sources_without_credentials'] = without_secrets(json.loads(source_config.read_text()))
        model_path = Path(config['model_manifest'])
        if model_path.exists():
            files.add(model_path)
        destination.mkdir(parents=True, mode=0o700)
        manifest = {'schema':'researchloop-cold-backup-v1','configuration_id':identity(settings),
            'files':[], 'historical_missing_paths':sorted(set(missing)), 'tables':{k:len(v) for k,v in rows.items()},
            'index_rebuilds':rows.get('evidence_index_builds',[]), 'config':config}
        for source in sorted(files):
            if source.name == '.env' or source.suffix in {'.sqlite','.db'} or source.name.endswith(('-wal','-shm')):
                continue
            # 保存树结构，保证 cache/parsed 与 cache/pdf 的邻接关系。
            archive = Path('files')/source.relative_to('/')
            target = destination/archive
            target.parent.mkdir(parents=True, exist_ok=True)
            before = digest(source)
            shutil.copyfile(source, target)
            if digest(target) != before or digest(source) != before:
                raise ValueError('FILE_CHANGED_DURING_BACKUP')
            manifest['files'].append({'original':str(source),'archive':str(archive),'sha256':before,'bytes':target.stat().st_size})
        atomic_json(destination/'database.json', rows)
        manifest['database_sha256'] = digest(destination/'database.json')
        atomic_json(destination/'manifest.json', manifest)
        return manifest


def verify_backup(source: Path) -> tuple[dict, dict]:
    """恢复前验证全部 hash 和归档路径；失败时尚未连接目标数据库。"""
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest.get('schema') != 'researchloop-cold-backup-v1' or digest(source/'database.json') != manifest['database_sha256']:
        raise ValueError('BACKUP_DATABASE_HASH_MISMATCH')
    seen = set()
    for item in manifest['files']:
        original = Path(item['original'])
        # original 也会参与恢复目标计算，不能只校验 archive 一侧的路径。
        if not original.is_absolute() or '..' in original.parts:
            raise ValueError('UNSAFE_ORIGINAL_PATH')
        relative = Path(item['archive'])
        path = source/relative
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'files' or item['archive'] in seen:
            raise ValueError('UNSAFE_ARCHIVE_PATH')
        seen.add(item['archive'])
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()) or digest(path) != item['sha256']:
            raise ValueError('BACKUP_FILE_HASH_MISMATCH')
    rows = json.loads((source/'database.json').read_text())
    if {k:len(v) for k,v in rows.items()} != manifest['tables']:
        raise ValueError('BACKUP_TABLE_COUNTS_MISMATCH')
    return manifest, rows


def remap(value, mapping: dict):
    """只替换确切结构化文件字段；普通原文、提示词和 Markdown 内容保持原样。"""
    if isinstance(value, dict):
        return {key:remap(item,mapping) for key,item in value.items()}
    if isinstance(value, list):
        return [remap(item,mapping) for item in value]
    if isinstance(value, str):
        if value in mapping:
            return mapping[value]
        if value.startswith(('{','[')):
            try:
                parsed = json.loads(value)
                changed = remap(parsed,mapping)
                return json.dumps(changed,ensure_ascii=False) if changed != parsed else value
            except ValueError:
                pass
    return value


def restore(source: Path, target_url: str, directory: Path) -> dict:
    """恢复到新空库和新目录；冻结执行，保留收据原文，索引暂置待重建。

    Args:
        source: 已有冷备份目录。
        target_url: 预先创建的空 researchloop_restore_* 本机 MySQL 库，或隔离 SQLite。
        directory: 不存在的新文件根目录。
    """
    source, directory = source.resolve(), directory.resolve()
    manifest, rows = verify_backup(source)
    parsed = urlparse(target_url)
    if parsed.scheme == 'mysql':
        if parsed.hostname not in {'127.0.0.1','localhost'} or not re.fullmatch(r'researchloop_restore_[a-z0-9_]+',parsed.path.lstrip('/')):
            raise ValueError('ISOLATED_RESTORE_DATABASE_REQUIRED')
    elif parsed.scheme not in {'sqlite','sqlite3'} or Path(parsed.path).exists():
        raise ValueError('NEW_RESTORE_DATABASE_REQUIRED')
    if directory.exists() or directory.is_relative_to(source):
        raise ValueError('NEW_RESTORE_DIRECTORY_REQUIRED')
    db = Database(target_url)
    with application_lock(ROOT, exclusive=True):
        assert_no_legacy_writers()
        if tables(db):
            raise ValueError('RESTORE_DATABASE_NOT_EMPTY')
        db.ensure_schema()
        if set(tables(db)) != set(rows):
            raise ValueError('RESTORE_SCHEMA_MISMATCH: 需要对应版本代码，目标库已创建空表')
        directory.mkdir(parents=True,mode=0o700)
        def target_path(item):
            """恢复 data/uploads 等运行时约定，同时保留外部历史缓存的树结构。"""
            original=Path(item['original'])
            for key,folder in [('output_dir','notes'),('data_dir','data')]:
                base=Path(manifest['config'][key]).resolve()
                if original.is_relative_to(base):
                    return directory/folder/original.relative_to(base)
            return directory/item['archive']
        mapping = {item['original']:str(target_path(item)) for item in manifest['files']}
        if any(not Path(target).resolve().is_relative_to(directory) for target in mapping.values()):
            raise ValueError('RESTORE_PATH_ESCAPES_TARGET')
        if len(set(mapping.values())) != len(mapping):
            raise ValueError('RESTORE_PATH_COLLISION')
        for item in manifest['files']:
            target = Path(mapping[item['original']])
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/item['archive'],target)
            if digest(target) != item['sha256']:
                raise ValueError('RESTORED_FILE_HASH_MISMATCH')
        # macOS /var 与 /private/var 是同一原件；数据库可能保留未规范化别名。
        for path in path_values(rows):
            if str(path.resolve()) in mapping:
                mapping[str(path)]=mapping[str(path.resolve())]
        rewritten_caches=[]
        for doc in rows['documents']:
            path=Path(mapping[doc['cache_path']])
            cache=json.loads(path.read_text())
            changed=remap(cache,mapping)
            if changed != cache:
                atomic_json(path,changed)
                rewritten_caches.append({'path':str(path),'sha256':digest(path),'reason':'relocate asset paths only'})
        # 原始模型 messages/response/usage 是审计事实，不因本机路径变更改写。
        immutable = {'model_calls','task_events','review_results'}
        transformed = {table: records if table in immutable else remap(records,mapping) for table,records in rows.items()}
        for record in transformed.get('evidence_index_builds',[]):
            record['vector_status'] = 'pending'
            record['last_error'] = 'RESTORED_REBUILD_REQUIRED'
        with db.connection() as conn:
            cur = conn.cursor()
            for table, records in transformed.items():
                for record in records:
                    columns = list(record)
                    if any(not re.fullmatch(r'[a-z_][a-z_0-9]*',key) for key in columns):
                        raise ValueError('INVALID_COLUMN')
                    marks = ','.join(['%s' if db.kind == 'mysql' else '?']*len(columns))
                    cur.execute(f"INSERT INTO `{table}` ({','.join('`'+k+'`' for k in columns)}) VALUES ({marks})",tuple(record[k] for k in columns))
        namespace = 'researchloop_restore_'+uuid4().hex[:12]
        report = {'schema':'researchloop-restore-v1','file_count':len(mapping),
            'tables':{k:len(v) for k,v in transformed.items()},'path_mapping':mapping,
            'retrieval_namespace':namespace,'execution_enabled':False,
            'data_dir':str(directory/'data'),'output_dir':str(directory/'notes'),
            'rewritten_cache_paths':rewritten_caches,
            'model_receipts_unchanged':sorted(db.fetchall('SELECT * FROM model_calls'),key=lambda r:r['call_id']) == sorted(rows['model_calls'],key=lambda r:r['call_id']),
            'index_policy':'BM25 retained; vectors pending; explicitly rebuild in isolated namespace',
            'source_manifest_sha256':digest(source/'manifest.json')}
        (directory/'data').mkdir(exist_ok=True)
        (directory/'notes').mkdir(exist_ok=True)
        atomic_json(directory/'data/restore-hold.json',{'reason':'恢复实例默认只读；需核对未完成任务和 unknown 后另行启用执行'})
        atomic_json(directory/'restore-report.json',report)
        return report


def main() -> None:
    """命令行只接收目标库环境变量名，不把口令写进命令参数或报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['backup','verify','restore'])
    parser.add_argument('--path',type=Path,required=True)
    parser.add_argument('--target-dir',type=Path)
    parser.add_argument('--target-url-env',default='RESEARCHLOOP_RESTORE_DATABASE_URL')
    args = parser.parse_args()
    try:
        if args.command == 'backup':
            value = backup(Settings.from_env(ROOT),args.path)
            result = {'path':str(args.path),'tables':value['tables'],'files':len(value['files']),'secrets_included':False}
        elif args.command == 'verify':
            manifest,_ = verify_backup(args.path)
            result = {'verified':True,'files':len(manifest['files'])}
        else:
            if not args.target_dir or not os.getenv(args.target_url_env):
                raise ValueError('RESTORE_TARGET_REQUIRED')
            result = restore(args.path,os.environ[args.target_url_env],args.target_dir)
            result.pop('path_mapping',None)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except Exception as exc:
        # 不输出数据库连接异常原文，可能包含用户名或连接参数。
        print(json.dumps({'error_type':type(exc).__name__,'detail':str(exc) if isinstance(exc,(ValueError,RuntimeError)) else '维护失败；原数据未覆盖，检查配置与空目标库'},ensure_ascii=False))
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()
