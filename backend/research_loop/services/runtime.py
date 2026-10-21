"""本机进程边界：应用共享锁与冷备份独占锁；不负责模型调度或修改业务状态。"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def runtime_root(root: Path) -> Path:
    """返回项目唯一运行目录，避免隔离实例绕过冷备份锁。"""
    path = root / 'data/runtime'
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


@contextmanager
def application_lock(root: Path, *, exclusive: bool = False):
    """持有应用共享锁或维护独占锁；冲突立即失败，不等待未知远端请求。

    Args:
        root: 当前项目根目录。
        exclusive: 冷备份/恢复使用 True，API、Worker、Scheduler 使用 False。
    """
    with (runtime_root(root) / 'application.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('APPLICATION_BUSY: 请先停止本项目应用进程，再维护数据') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path: Path, value: dict) -> None:
    """原子发布本机状态，避免读到半份记录；文件只允许本机用户访问。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temp.open('w') as handle:
        os.chmod(temp, 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def identity(settings) -> str:
    """生成配置身份指纹，不输出数据库口令或供应商密钥。"""
    return hashlib.sha256(json.dumps([settings.database_url, str(settings.data_dir),
        str(settings.output_dir), settings.retrieval_namespace], ensure_ascii=False).encode()).hexdigest()


def heartbeat(state: str) -> None:
    """受管进程报告已完成初始化或正在执行；未受管命令不产生伪就绪文件。"""
    target = os.getenv('RESEARCHLOOP_HEARTBEAT')
    if target:
        if _PROCESS_FINGERPRINT is None:
            from research_loop.config import get_settings
            configure_runtime(get_settings())
        atomic_json(Path(target), {'pid': os.getpid(), 'token': os.getenv('RESEARCHLOOP_PROCESS_TOKEN'),
            'state': state, 'time': time.time(),
            'runtime_fingerprint_id': _PROCESS_FINGERPRINT['fingerprint_id']})


def _digest(value) -> str:
    """对规范JSON计算确定性摘要，不把配置原文写入运行记录。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), default=str).encode()).hexdigest()


def _file_digest(path: Path) -> dict:
    """流式核对文件，不加载模型、不下载缺失文件。"""
    try:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return {'sha256': digest.hexdigest()}
    except OSError as exc:
        return {'state': 'unavailable', 'error_type': type(exc).__name__}


def runtime_fingerprint(settings, *, root: Path | None = None) -> dict:
    """冻结解释器、源码与有效配置身份；与停止/备份的实例身份分离。

    Args:
        settings: 进程实际加载的配置；密钥与含凭据的地址不直接输出。
        root: 源码根目录，默认使用配置根目录；测试可注入临时目录。

    Returns:
        可比较的脱敏清单及摘要；文件快照不声称是环境变量值的来源证明。
    """
    import importlib.metadata
    import sys
    from research_loop.config import paper_sources_path

    root = Path(root or settings.project_root).resolve()
    paths = set()
    for folder, suffixes in (('backend/research_loop', {'.py'}), ('backend/scripts', {'.py'}),
                             ('frontend', {'.js', '.mjs', '.css', '.html'})):
        paths.update(p for p in (root / folder).rglob('*') if p.is_file() and p.suffix in suffixes
                     and not {'node_modules', '__pycache__'}.intersection(p.relative_to(root).parts))
    paths.update(p for p in (root / 'backend/migrations').glob('*.sql') if p.is_file())
    source = {str(p.relative_to(root)): _file_digest(p) for p in sorted(paths)}
    files = {name: _file_digest(root / name) for name in
             ('backend/pyproject.toml', 'backend/uv.lock', 'frontend/package.json',
              'frontend/package-lock.json', '.env')}
    manifest = settings.retrieval_model_manifest or root / 'data/ag71-preflight/models/manifest.json'
    files['paper_sources'] = {'path': str(paper_sources_path(root)), **_file_digest(paper_sources_path(root))}
    files['retrieval_manifest'] = {'path': str(manifest), **_file_digest(Path(manifest))}
    files['tokenizer'] = ({'path': str(settings.token_counting_tokenizer_path),
                           **_file_digest(settings.token_counting_tokenizer_path)}
                          if settings.token_counting_tokenizer_path else {'state': 'not_configured'})
    # 白名单之外的复合配置只写摘要；数据库/服务URL可能包含口令或认证query。
    hidden = {'database_url', 'deepseek_api_key', 'supabase_key', 'openalex_api_key',
              'discovery_backends', 'parser_config', 'deepseek_base_url', 'supabase_url', 'qdrant_url'}
    effective = {key: value for key, value in settings.model_dump(mode='json').items() if key not in hidden}
    for key in ('database_url', 'discovery_backends', 'parser_config', 'deepseek_base_url', 'supabase_url', 'qdrant_url'):
        effective[key + '_sha256'] = _digest(getattr(settings, key))
    for key in ('deepseek_api_key', 'supabase_key', 'openalex_api_key'):
        effective[key + '_present'] = bool(getattr(settings, key))
    versions = {}
    for package in ('pydantic', 'httpx', 'langgraph', 'pymupdf', 'onnxruntime', 'numpy', 'tokenizers'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    detail = {'schema': 'runtime-fingerprint-v1',
              'interpreter': {'executable': os.path.abspath(sys.executable),
                              'resolved_executable': str(Path(sys.executable).resolve()),
                              'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
                              'version': sys.version, 'runtime_module': str(Path(__file__).resolve())},
              'source_manifest_sha256': _digest(source), 'source_files': source,
              'configuration_files': files, 'effective_settings': effective, 'dependencies': versions,
              'explicit_setting_fields': sorted(settings.model_fields_set),
              'credential_configuration_sha256': _digest({key: getattr(settings, key) for key in
                  ('deepseek_api_key', 'supabase_key', 'openalex_api_key')}),
              'configuration_provenance': 'effective values; file snapshots do not prove env value origins'}
    return {'fingerprint_id': _digest(detail), **detail}


def local_component_readiness(settings) -> dict:
    """只检查本机导入及模型文件hash；不推理、不联网、不下载。

    Args:
        settings: 有效组件开关和现有模型清单位置。

    Returns:
        每个本机RAG组件的就绪与缺项；缺项不阻断无关的只读路径。
    """
    import importlib
    if not settings.retrieval_enabled:
        return {name: {'enabled': False, 'status': 'disabled'} for name in ('embedding', 'reranker')}
    imports = {}
    for name in ('numpy', 'onnxruntime', 'tokenizers'):
        try:
            importlib.import_module(name)
            imports[name] = {'status': 'available'}
        except Exception as exc:
            imports[name] = {'status': 'unavailable', 'error_type': type(exc).__name__}
    manifest = settings.retrieval_model_manifest or settings.project_root / 'data/ag71-preflight/models/manifest.json'
    try:
        payload = json.loads(Path(manifest).read_text())
    except (OSError, ValueError) as exc:
        return {name: {'enabled': True, 'status': 'unavailable', 'imports': imports,
                       'manifest': _file_digest(Path(manifest)), 'error_type': type(exc).__name__}
                for name in ('embedding', 'reranker')}
    results = {}
    for name, repo in (('embedding', 'intfloat/multilingual-e5-small'),
                       ('reranker', 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')):
        files = {}
        try:
            spec = payload[name]
            if spec['repo'] != repo:
                raise ValueError('unsupported model identity')
            # 与现有provider一致，清单相对路径基于当前工作目录解释。
            directory = Path(spec['directory'])
            for filename in (spec['weights'], 'tokenizer.json', 'config.json'):
                actual = _file_digest(directory / filename)
                files[filename] = {**actual, 'matches': isinstance(actual.get('sha256'), str) and
                                   actual['sha256'] == spec['files'][filename]['sha256']}
            ready = all(item['matches'] for item in files.values()) and all(
                item['status'] == 'available' for item in imports.values())
            results[name] = {'enabled': True, 'status': 'ready' if ready else 'unavailable',
                             'imports': imports, 'files': files, 'inference_probed': False}
        except (KeyError, TypeError, ValueError) as exc:
            results[name] = {'enabled': True, 'status': 'unavailable', 'imports': imports,
                             'error_type': type(exc).__name__, 'inference_probed': False}
    return results


_PROCESS_FINGERPRINT: dict | None = None


def configure_runtime(settings) -> dict:
    """启动时冻结一次指纹，拒绝受管启动配置漂移；心跳不反复扫描源码。"""
    global _PROCESS_FINGERPRINT
    report = runtime_fingerprint(settings)
    expected = os.getenv('RESEARCHLOOP_EXPECTED_RUNTIME_FINGERPRINT')
    if expected and expected != report['fingerprint_id']:
        raise RuntimeError('RUNTIME_FINGERPRINT_MISMATCH: 进程实际配置/解释器/源码与启动预期不同')
    _PROCESS_FINGERPRINT = report
    return report
