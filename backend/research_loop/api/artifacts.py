"""研究成果入口：授权原文、PDF、当前草稿和指定版本保存。"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from research_loop.api.dependencies import get_artifact_service
from research_loop.domain.notes import SaveDraft
from research_loop.services.artifacts import ArtifactService

router = APIRouter()


@router.get('/tasks/{task_id}/artifacts')
def task_artifacts(task_id: str, service: ArtifactService = Depends(get_artifact_service)):
    """返回任务家族实际取得的成果，不伪造未取得的来源。"""
    try:
        return service.artifacts(task_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get('/tasks/{task_id}/evidence/{evidence_id}')
def evidence(task_id: str, evidence_id: str, service: ArtifactService = Depends(get_artifact_service)):
    """按任务权限返回原文摘录与可信物理页码。"""
    try:
        return service.evidence(task_id, evidence_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get('/tasks/{task_id}/evidence/{evidence_id}/pdf')
def evidence_pdf(task_id: str, evidence_id: str, service: ArtifactService = Depends(get_artifact_service)):
    """返回授权 PDF 缓存，不接受客户端文件路径。"""
    try:
        return FileResponse(service.pdf(task_id, evidence_id), media_type='application/pdf')
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get('/tasks/{task_id}/drafts/{draft_id}')
def draft(task_id: str, draft_id: str, service: ArtifactService = Depends(get_artifact_service)):
    """读取具体稿件，分别展示核验状态和保存状态。"""
    try:
        return service.draft(task_id, draft_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post('/tasks/{task_id}/drafts/{draft_id}/save')
def save_draft(
    task_id: str, draft_id: str, payload: SaveDraft,
    service: ArtifactService = Depends(get_artifact_service),
):
    """接收本次明确保存确认；版本、冲突及幂等规则由服务层验证。"""
    if not payload.confirm:
        raise HTTPException(422, '需要明确保存确认')
    try:
        return service.save_draft(task_id, draft_id, payload.content_hash)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc
