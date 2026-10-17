"""个人面板订阅和聊天简报API；配置操作不会立即发送外部请求。"""
from fastapi import APIRouter, Depends, HTTPException
from research_loop.api.dependencies import get_database
from research_loop.storage.mysql import Database
from research_loop.domain.subscriptions import SubscriptionInput, CandidateReadInput
from research_loop.services.subscriptions import SubscriptionService
from research_loop.providers.dpr_sources import SUPPORTED_SOURCES, DEFAULT_SEARCH_SOURCES

router=APIRouter()




@router.get('/subscriptions')
def list_subscriptions(db: Database = Depends(get_database)):
    """返回现有配置及17个可选来源；配置存在不表示上游可用。"""
    return {'subscriptions':SubscriptionService(db).list(),'sources':SUPPORTED_SOURCES,'default_sources':DEFAULT_SEARCH_SOURCES}


@router.post('/subscriptions')
def create_subscription(payload: SubscriptionInput, db: Database = Depends(get_database)):
    """新增默认关闭的订阅；启用需用户显式提交enabled。"""
    try:
        return SubscriptionService(db).configure(payload)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.post('/subscriptions/{subscription_id}')
def update_subscription(subscription_id: str, payload: SubscriptionInput, db: Database = Depends(get_database)):
    """编辑时校验版本，防止另一页面的旧配置覆盖新偏好。"""
    try:
        return SubscriptionService(db).configure(payload,subscription_id)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.get('/briefs/{run_id}')
def get_brief(run_id: str, db: Database = Depends(get_database)):
    """查看简报不会触发检索或自动标为已读。"""
    try:
        return SubscriptionService(db).brief(run_id)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc


@router.post('/briefs/{run_id}/displayed')
def display_brief(run_id: str, db: Database = Depends(get_database)):
    """页面真实渲染后记录首次展示，不代表用户读完论文。"""
    try:
        return SubscriptionService(db).mark_displayed(run_id)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc


@router.post('/brief-read-marks')
def read_mark(payload: CandidateReadInput, db: Database = Depends(get_database)):
    """用户主动标记/撤销已读，保留精读与审校独立状态。"""
    try:
        return SubscriptionService(db).mark_read(payload.identity,payload.read)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc
