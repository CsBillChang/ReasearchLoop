"""任务事件入口：公开事件投影、SSE 游标重放和终态快照，不执行研究任务。"""
import asyncio
import json

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from research_loop.api.dependencies import get_task_service
from research_loop.services.tasks import TaskService

router = APIRouter()
TERMINAL = {'completed', 'partial', 'failed', 'cancelled', 'needs_input'}


def public_event(event):
    """只公开行动摘要和回答增量，原始请求、工具观察及模型推理不进入事件流。"""
    payload = event['payload']
    if event['event_type'] == 'answer_stream':
        data = {k: payload[k] for k in ('stream_key', 'request_number', 'text', 'provisional') if k in payload}
    elif event['event_type'] == 'observation':
        obs = payload.get('observation') or {}
        data = {'tool': payload.get('tool'), 'ok': obs.get('ok'), 'message': obs.get('message', '')[:300]}
    elif event['event_type'] == 'task_finished':
        data = {'status': payload.get('status')}
    else:
        data = {'status': payload.get('status')} if payload.get('status') else {}
    return {'id': event['event_id'], 'type': event['event_type'], 'data': data}


@router.get('/tasks/{task_id}/events')
def get_events(
    task_id: str,
    after_id: str | None = Query(default=None),
    service: TaskService = Depends(get_task_service),
) -> dict:
    """按 after_id 返回持久事件 JSON 列表；实时订阅使用 stream 接口。"""
    if not service.db.get_task(task_id):
        raise HTTPException(404, 'task not found')
    return {'task_id': task_id, 'events': service.db.list_events(task_id, after_id)}


@router.get('/tasks/{task_id}/stream')
async def task_stream(
    task_id: str,
    request: Request,
    after_id: str | None = None,
    last_event_id: str | None = Header(None),
    service: TaskService = Depends(get_task_service),
):
    """重放任务公开事件；断开浏览器不取消任务、不创建模型请求。"""
    if not service.db.get_task(task_id):
        raise HTTPException(404, '任务不存在')

    async def events():
        """以本任务游标补发事件，结算后发最终快照并结束连接。"""
        cursor = last_event_id or after_id
        while not await request.is_disconnected():
            rows = await asyncio.to_thread(service.db.list_events, task_id, cursor)
            for row in rows:
                event = public_event(row)
                cursor = event['id']
                yield 'id: ' + cursor + '\nevent: progress\ndata: ' + json.dumps(event, ensure_ascii=False) + '\n\n'
            task = await asyncio.to_thread(service.db.get_task, task_id)
            if task['status'] in TERMINAL:
                view = await asyncio.to_thread(service.view, task)
                yield 'event: settled\ndata: ' + view.model_dump_json() + '\n\n'
                return
            yield ': heartbeat\n\n'
            await asyncio.sleep(.5)

    return StreamingResponse(
        events(), media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )
