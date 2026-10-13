"""任务 HTTP 接口：校验提交、查询状态、请求取消和恢复澄清；研究执行留给 Worker。"""
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Response, status

from research_loop.config import get_settings
from research_loop.domain.tasks import ResumeInput, TaskCreate
from research_loop.services.tasks import IdempotencyConflict, TaskService
from research_loop.storage.mysql import Database

router = APIRouter()


def _service() -> TaskService:
    db = Database(get_settings().database_url)
    db.ensure_schema()
    return TaskService(db, get_settings())


@router.post("/tasks", status_code=status.HTTP_202_ACCEPTED)
def create_task(payload: TaskCreate) -> dict:
    """输入 TaskCreate；检查本地附件并创建 queued 任务。重复键冲突返回 409，受理成功不等于执行完成。"""
    from research_loop.services.product import resolve_upload
    try:
        payload = payload.model_copy(update={"attachments":[resolve_upload(get_settings(),a) for a in payload.attachments]})
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    for attachment in payload.attachments:
        if attachment.local_path:
            path = Path(attachment.local_path).expanduser().resolve()
            if not path.exists() or path.suffix.lower() != ".pdf":
                raise HTTPException(status_code=422, detail="local_path must point to an existing PDF")
    try:
        view = _service().create(payload)
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return view.model_dump(mode="json")


@router.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    service = _service()
    try:
        return service.view(service.db.get_task(task_id)).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc


@router.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str) -> dict:
    """记录取消意愿并返回当前视图；已发送的远端请求不会因此被同步撤销。"""
    try:
        return _service().cancel(task_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc


@router.post("/tasks/{task_id}/resume")
def resume_task(task_id: str, payload: ResumeInput) -> dict:
    """输入澄清 ID、回答和幂等键；由服务校验暂停点并重新入队，不直接补发模型请求。"""
    try:
        return _service().resume(payload, task_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/tasks/{task_id}/events")
def get_events(task_id: str, after_id: str | None = Query(default=None)) -> dict:
    """按 after_id 返回持久事件列表；这是 JSON 查询接口，不是实时 SSE 流。"""
    service = _service()
    if not service.db.get_task(task_id):
        raise HTTPException(status_code=404, detail="task not found")
    return {"task_id": task_id, "events": service.db.list_events(task_id, after_id)}


@router.get("/tasks/{task_id}/subtasks")
def get_subtasks(task_id: str) -> dict:
    service = _service()
    if not service.db.get_task(task_id):
        raise HTTPException(status_code=404, detail="task not found")
    return {"task_id": task_id, "subtasks": service.db.list_subtasks(task_id)}
