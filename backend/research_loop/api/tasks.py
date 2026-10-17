"""任务 HTTP 接口：校验提交、查询状态、请求取消和恢复澄清；研究执行留给 Worker。"""
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status

from research_loop.api.dependencies import get_artifact_service, get_task_service
from research_loop.domain.tasks import ResumeInput, TaskCreate
from research_loop.services.tasks import IdempotencyConflict, TaskService
from research_loop.services.artifacts import ArtifactService

router = APIRouter()




@router.post("/tasks", status_code=status.HTTP_202_ACCEPTED)
def create_task(payload: TaskCreate, service: TaskService = Depends(get_task_service)) -> dict:
    """输入 TaskCreate；检查本地附件并创建 queued 任务。重复键冲突返回 409，受理成功不等于执行完成。"""
    from research_loop.services.uploads import resolve_upload
    try:
        payload = payload.model_copy(update={"attachments":[resolve_upload(service.settings,a) for a in payload.attachments]})
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    for attachment in payload.attachments:
        if attachment.local_path:
            path = Path(attachment.local_path).expanduser().resolve()
            if not path.exists() or path.suffix.lower() != ".pdf":
                raise HTTPException(status_code=422, detail="local_path must point to an existing PDF")
    try:
        view = service.create(payload)
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return view.model_dump(mode="json")


@router.get("/tasks/{task_id}")
def get_task(task_id: str, service: TaskService = Depends(get_task_service)) -> dict:
    try:
        return service.view(service.db.get_task(task_id)).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc


@router.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str, service: TaskService = Depends(get_task_service)) -> dict:
    """记录取消意愿并返回当前视图；已发送的远端请求不会因此被同步撤销。"""
    try:
        return service.cancel(task_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc


@router.post("/tasks/{task_id}/resume")
def resume_task(task_id: str, payload: ResumeInput, service: TaskService = Depends(get_task_service)) -> dict:
    """输入澄清 ID、回答和幂等键；由服务校验暂停点并重新入队，不直接补发模型请求。"""
    try:
        return service.resume(payload, task_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc




@router.get("/tasks/{task_id}/subtasks")
def get_subtasks(task_id: str, service: TaskService = Depends(get_task_service)) -> dict:
    if not service.db.get_task(task_id):
        raise HTTPException(status_code=404, detail="task not found")
    return {"task_id": task_id, "subtasks": service.db.list_subtasks(task_id)}


@router.get('/tasks/{task_id}/context')
def task_context(task_id: str, service: ArtifactService = Depends(get_artifact_service)):
    """读取实际记忆使用和压缩诊断；不公开原始模型 messages。"""
    try:
        return service.context(task_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0])) from exc


@router.post('/tasks/{task_id}/continue', status_code=202)
def continue_task(
    task_id: str, payload: TaskCreate,
    service: TaskService = Depends(get_task_service),
):
    """持久化停止后续接请求；页面关闭或刷新不丢失新要求。"""
    from research_loop.services.continuations import ContinuationService
    from research_loop.services.uploads import resolve_upload
    if not payload.conversation_id or not payload.idempotency_key:
        raise HTTPException(422, '会话和幂等键必需')
    try:
        payload = payload.model_copy(update={
            'attachments': [resolve_upload(service.settings, a) for a in payload.attachments],
        })
        return ContinuationService(service.db).request(task_id, payload)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
