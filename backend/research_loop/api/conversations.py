"""会话API：创建、列出、分页读回；提问仍走POST /tasks，澄清走原任务resume。"""
from fastapi import APIRouter, HTTPException, Query
from research_loop.api.tasks import _service
from research_loop.services.conversations import ConversationService
from research_loop.domain.memory import ConversationCreate, ConversationRename

router=APIRouter()


@router.post('/conversations',status_code=201)
def create_conversation(payload: ConversationCreate):
    """幂等创建本地用户会话，冲突返回409。"""
    try:
        return ConversationService(_service().db).create(payload.title,payload.idempotency_key)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.get('/conversations')
def list_conversations():
    """列出会话；不触发模型请求。"""
    return {'conversations':ConversationService(_service().db).list()}


@router.get('/conversations/{conversation_id}')
def get_conversation(conversation_id: str, offset: int=Query(0,ge=0),limit: int=Query(100,ge=1,le=200)):
    """按稳定顺序分页读回，任务终态后的投影中断可恢复。"""
    try:
        return ConversationService(_service().db).view(conversation_id,offset,limit)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc


@router.post('/conversations/{conversation_id}/rename')
def rename_conversation(conversation_id: str, payload: ConversationRename):
    """仅修改标题；过期并发写返回409，缺失会话返回404。"""
    try:
        return ConversationService(_service().db).rename(conversation_id, payload)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
