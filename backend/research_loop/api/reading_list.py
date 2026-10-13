"""阅读清单HTTP接口；不调用模型、不执行PDF下载或笔记保存。"""
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from research_loop.api.tasks import _service
from research_loop.services.reading_list import ReadingListService

router = APIRouter()


# 只接收服务器论文身份和用户操作，不接收自定义标题、链接或文件路径。
class ReadingChange(BaseModel):
    paper_id: str = Field(min_length=1, max_length=200)
    task_id: str | None = Field(default=None, max_length=64)
    action: Literal['add', 'read', 'unread', 'remove']
    expected_version: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1, max_length=200)


@router.get('/reading-list')
def reading_list():
    """读取清单与移除版本，不把该操作计为阅读或自动精读。"""
    return {'items': ReadingListService(_service().db).list()}


@router.post('/reading-list')
def change_reading_list(payload: ReadingChange):
    """处理用户明确清单操作，版本或幂等冲突返回409。"""
    try:
        return ReadingListService(_service().db).change(payload)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
