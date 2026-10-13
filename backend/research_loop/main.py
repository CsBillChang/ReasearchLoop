"""应用入口：组装 FastAPI 路由；长任务由独立 Worker 执行，不在 HTTP 请求内调用模型。"""
from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from research_loop.config import get_settings
from research_loop.services.runtime import application_lock, heartbeat
from research_loop.api.health import router as health_router
from research_loop.api.tasks import router as tasks_router
from research_loop.api.conversations import router as conversations_router
from research_loop.api.profile import router as profile_router
from research_loop.api.notes import router as notes_router

@asynccontextmanager
async def lifespan(app):
    """应用存活期间阻止冷备份，启动成功才报告进程就绪。"""
    with application_lock(get_settings().project_root):
        heartbeat('ready')
        yield


app = FastAPI(title="ResearchLoop", version="0.3.0", lifespan=lifespan)


@app.middleware('http')
async def readonly_demo(request: Request, call_next):
    """只读演示拒绝业务写入；不把演示模式当作可执行 Agent。"""
    if os.getenv('RESEARCHLOOP_READONLY') == '1' and request.method not in {'GET','HEAD','OPTIONS'}:
        return JSONResponse({'detail': 'READONLY_DEMO: 当前入口只读，请使用日常入口'}, status_code=403)
    return await call_next(request)
app.include_router(health_router, prefix="/api")
app.include_router(tasks_router, prefix="/api")
app.include_router(conversations_router, prefix="/api")
app.include_router(profile_router, prefix="/api")
app.include_router(notes_router, prefix="/api")

from research_loop.api.product import router as product_router
app.include_router(product_router, prefix="/api")

from research_loop.api.subscriptions import router as subscriptions_router
app.include_router(subscriptions_router, prefix="/api")

from research_loop.api.reading_list import router as reading_list_router
app.include_router(reading_list_router, prefix='/api')
