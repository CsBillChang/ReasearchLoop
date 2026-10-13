"""健康接口：检查服务可响应与配置是否存在，不代表数据库、PDF 或模型请求已通过实测。"""
from fastapi import APIRouter
from research_loop.config import get_settings
import os
from research_loop.storage.mysql import Database

router = APIRouter()

@router.get("/health")
def health() -> dict:
    settings = get_settings()
    # agent_ready 是旧前端兼容字段，只说明配置存在，不代表模型调用成功。
    return {"service": "ResearchLoop", "status": "ok", "stage": "AG7.5",
            "agent_ready": bool(settings.deepseek_api_key),
            "model_key_present": bool(settings.deepseek_api_key),
            "mode": 'readonly' if os.getenv('RESEARCHLOOP_READONLY') == '1' else 'daily',
            "worker_required": True, 'model_probed': False}


@router.get('/ready')
def ready():
    """只读验证数据库和迁移；不创建表、不调用供应商、不代替 Worker 状态检查。"""
    from fastapi.responses import JSONResponse
    settings = get_settings()
    try:
        versions = {row['version'] for row in Database(settings.database_url).fetchall('SELECT version FROM schema_version')}
        expected = {p.stem for p in (settings.project_root/'backend/migrations').glob('*.sql')}
        ok = bool(expected) and expected <= versions
        return JSONResponse({'database_ready': ok, 'missing_migrations': sorted(expected-versions),
                             'worker_checked': False}, status_code=200 if ok else 503)
    except Exception as exc:
        return JSONResponse({'database_ready': False, 'error_type': type(exc).__name__}, status_code=503)
