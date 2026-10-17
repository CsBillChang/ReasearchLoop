"""集中登记 HTTP 资源路由；主应用只负责生命周期、中间件和挂载前缀。"""
from fastapi import APIRouter

from research_loop.api import (
    artifacts, conversations, events, feedback, health, notes, profile,
    reading_list, subscriptions, tasks, uploads,
)

router = APIRouter()
for resource in (
    health, tasks, conversations, profile, notes, uploads, events, artifacts,
    feedback, subscriptions, reading_list,
):
    router.include_router(resource.router)
