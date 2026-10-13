"""个人面板订阅和聊天简报API；配置操作不会立即发送外部请求。"""
from fastapi import APIRouter, HTTPException
from research_loop.api.tasks import _service
from research_loop.domain.subscriptions import SubscriptionInput, CandidateReadInput
from research_loop.services.subscriptions import SubscriptionService
from research_loop.providers.dpr_sources import SUPPORTED_SOURCES, DEFAULT_SEARCH_SOURCES

router=APIRouter()


def service():
    """复用本机配置数据库，不构建模型或调度器。"""
    return SubscriptionService(_service().db)


@router.get('/subscriptions')
def list_subscriptions():
    """返回现有配置及17个可选来源；配置存在不表示上游可用。"""
    return {'subscriptions':service().list(),'sources':SUPPORTED_SOURCES,'default_sources':DEFAULT_SEARCH_SOURCES}


@router.post('/subscriptions')
def create_subscription(payload: SubscriptionInput):
    """新增默认关闭的订阅；启用需用户显式提交enabled。"""
    try:
        return service().configure(payload)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.post('/subscriptions/{subscription_id}')
def update_subscription(subscription_id: str, payload: SubscriptionInput):
    """编辑时校验版本，防止另一页面的旧配置覆盖新偏好。"""
    try:
        return service().configure(payload,subscription_id)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.get('/briefs/{run_id}')
def get_brief(run_id: str):
    """查看简报不会触发检索或自动标为已读。"""
    try:
        return service().brief(run_id)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc


@router.post('/briefs/{run_id}/displayed')
def display_brief(run_id: str):
    """页面真实渲染后记录首次展示，不代表用户读完论文。"""
    try:
        return service().mark_displayed(run_id)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc


@router.post('/brief-read-marks')
def read_mark(payload: CandidateReadInput):
    """用户主动标记/撤销已读，保留精读与审校独立状态。"""
    try:
        return service().mark_read(payload.identity,payload.read)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc
