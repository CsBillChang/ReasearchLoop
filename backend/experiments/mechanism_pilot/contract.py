"""小样本发送资格：冻结文件、资源、预算与后续明确授权共同校验。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit


def digest(value):
    """仅对规范JSON计算身份，不将密钥或环境内容放入合同。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def settings_profile(settings):
    """冻结因果相关设置；仅排除凭据、替身入口和由本格派生的运行路径。"""
    return digest(settings.model_dump(mode='json', exclude={
        'project_root', 'data_dir', 'output_dir', 'database_url', 'deepseek_api_key',
        'deepseek_base_url', 'retrieval_namespace'}))


def authorize_execution(settings, execution_kind, authorize=None, freeze_sha256=None):
    """真实模式在创建目录、资源或传输之前必须取得完整匹配资格。"""
    if execution_kind == 'substitute':
        if settings.deepseek_api_key or settings.deepseek_base_url != 'https://fixture.invalid':
            raise PermissionError('REFLECTION_SYNTHETIC_SETTINGS_REQUIRED')
        if authorize is not None:
            return authorize()
        return {'allow_real_model': False}
    if execution_kind != 'real' or not callable(authorize) or not freeze_sha256:
        raise PermissionError('REFLECTION_REAL_EXECUTION_NOT_QUALIFIED')
    grant = authorize()
    if (grant.get('allow_real_model') is not True
            or grant.get('freeze_sha256') != freeze_sha256
            or grant.get('model') != settings.deepseek_model
            or grant.get('base_url') != settings.deepseek_base_url
            or settings.deepseek_base_url.rstrip('/') != 'https://api.deepseek.com'
            or not settings.deepseek_api_key):
        raise PermissionError('MECHANISM_EXECUTION_GRANT_MISMATCH')
    return grant


class RunContract:
    """启动与每次发送均重验；本轮只生成草案授权，不开放真实调用。"""

    def __init__(self, manifest, authorization_path, *, clock=time.time):
        self.manifest = json.loads(json.dumps(manifest))
        self.freeze_sha256 = digest(self.manifest)
        self.authorization_path = Path(authorization_path).resolve()
        self.clock = clock

    def check_files(self):
        """所有文件必须保持冻结字节；源文件可只读位于运行根之外。"""
        for item in self.manifest['files']:
            path = Path(item['path']).resolve()
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise PermissionError('MECHANISM_FROZEN_FILE_DRIFT')

    def validate(self, settings, runtime_dir, *, phase, real=False):
        """检查隔离配置；真实资格还需读取匹配的新授权及固定截止。"""
        self.check_files()
        manifest = self.manifest
        if manifest['limits'] != {'tokens': 3000000, 'requests': 96, 'nano_yuan': 12000000000,
                                 'admission_seconds': 2400, 'total_seconds': 2700}:
            raise PermissionError('MECHANISM_LEDGER_LIMIT_CONTRACT_MISMATCH')
        root = Path(manifest['runtime_root']).resolve()
        case_root = Path(runtime_dir).resolve()
        if case_root == root or not case_root.is_relative_to(root):
            raise PermissionError('MECHANISM_CASE_OUTSIDE_RUNTIME')
        if Path(settings.project_root).resolve() != case_root:
            raise PermissionError('MECHANISM_SETTINGS_ROOT_DRIFT')
        from experiments.mechanism_pilot.resources import runtime_namespace
        if settings.retrieval_namespace != runtime_namespace(case_root):
            raise PermissionError('MECHANISM_NAMESPACE_DRIFT')
        database = urlsplit(settings.database_url)
        if (database.scheme != 'sqlite' or database.netloc
                or Path(database.path).resolve() != case_root / 'runtime.sqlite3'):
            raise PermissionError('MECHANISM_DATABASE_NOT_ISOLATED')
        for path in (settings.data_dir, settings.output_dir):
            path = Path(path).resolve()
            if path == case_root or not path.is_relative_to(case_root):
                raise PermissionError('MECHANISM_WRITABLE_PATH_OUTSIDE_CASE')
        if (settings.web_search_enabled or settings.web_fetch_enabled
                or settings.retrieval_route != 'located_fp16'
                or settings.retrieval_reranker_backend != 'mps_fp16'):
            raise PermissionError('MECHANISM_TOOL_OR_RAG_POLICY_DRIFT')
        if phase not in manifest['phases']:
            raise PermissionError('MECHANISM_PHASE_NOT_FROZEN')
        if (settings.discovery_backends or settings_profile(settings)
                != manifest.get('settings_profiles', {}).get(phase)):
            raise PermissionError('MECHANISM_SETTINGS_PROFILE_DRIFT')
        if settings.deepseek_model != manifest['model']:
            raise PermissionError('MECHANISM_MODEL_DRIFT')
        if settings.context_window_tokens != manifest['context_window_tokens']:
            raise PermissionError('MECHANISM_CONTEXT_POLICY_DRIFT')
        phase_config = manifest['phases'][phase]
        if settings.model_max_output_tokens != phase_config['max_output_tokens']:
            raise PermissionError('MECHANISM_OUTPUT_BOUND_DRIFT')
        if not real:
            if settings.deepseek_api_key or settings.deepseek_base_url != 'https://fixture.invalid':
                raise PermissionError('MECHANISM_SUBSTITUTE_CONFIG_REQUIRED')
            return {'allow_real_model': False, 'freeze_sha256': self.freeze_sha256,
                    'model': settings.deepseek_model, 'base_url': settings.deepseek_base_url}
        if settings.deepseek_base_url.rstrip('/') != 'https://api.deepseek.com' or not settings.deepseek_api_key:
            raise PermissionError('MECHANISM_REAL_PROVIDER_CONFIG_REQUIRED')
        if not self.authorization_path.is_file():
            raise PermissionError('MECHANISM_FRESH_AUTHORIZATION_REQUIRED')
        authorization = json.loads(self.authorization_path.read_text())
        now = self.clock()
        if (authorization.get('status') != 'approved'
                or authorization.get('scope') != 'mechanism-14'
                or not isinstance(authorization.get('approval_reference'), str)
                or not authorization['approval_reference'].strip()
                or authorization.get('freeze_sha256') != self.freeze_sha256
                or authorization.get('limits') != manifest['limits']
                or authorization.get('egress') != manifest['egress']):
            raise PermissionError('MECHANISM_AUTHORIZATION_NOT_MATCHED')
        started = authorization.get('started_at')
        if type(started) not in (int, float) or not 0 <= now - started < manifest['limits']['admission_seconds']:
            raise PermissionError('MECHANISM_AUTHORIZATION_TIME_BOUND')
        return {'allow_real_model': True, 'freeze_sha256': self.freeze_sha256,
                'model': settings.deepseek_model, 'base_url': settings.deepseek_base_url,
                'input_token_limit': phase_config['input_token_limit'],
                'max_output_tokens': phase_config['max_output_tokens'],
                'started_at': started,
                'hard_deadline': started + manifest['limits']['total_seconds']}

    def authorizer(self, settings, runtime_dir, *, phase, real=False):
        """返回每次调用都重新检查文件、配置与授权的回调。"""
        return lambda: self.validate(settings, runtime_dir, phase=phase, real=real)


def cancel_case(settings, db, root_task_id):
    """取消和收尾也固定本格settings，防止全局配置回读部署环境。"""
    from unittest.mock import patch
    from research_loop.services.tasks import TaskService
    with patch('research_loop.config.get_settings', return_value=settings), \
            patch('research_loop.config.Settings.from_env', side_effect=PermissionError('MECHANISM_DEFAULT_ENV_DENIED')):
        return TaskService(db, settings).cancel(root_task_id)


def finalize_reservations(ledger, process_record):
    """仅确认所属进程组已退出后处理悬挂资格，保留发送后的未知占用。"""
    if not process_record.get('process_group_stopped') or process_record.get('callback_threads_alive'):
        raise PermissionError('MECHANISM_PROCESS_STILL_UNCONFIRMED')
    owner = process_record.get('owned_root_id')
    if not owner:
        return
    for row in ledger.snapshot()['reservations']:
        if row['root_id'] != owner:
            continue
        if row['status'] == 'reserved':
            ledger.not_sent(row['reservation_id'], 'owned_process_stopped_before_send')
        elif row['status'] == 'sent':
            ledger.unknown(row['reservation_id'], 'owned_process_stopped_after_send')
