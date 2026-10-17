"""个人画像接口：显式编辑、推断撤销与来源统计，不引入登录/多租户承诺。"""
from fastapi import APIRouter, Depends, HTTPException
from research_loop.api.dependencies import get_database, get_note_version_service
from research_loop.services.note_versions import NoteVersionService
from research_loop.storage.mysql import Database
from research_loop.services.memory import MemoryService
from research_loop.domain.memory import ProfileSet, ForgetInput, RecallInput, PreferenceStatement

router=APIRouter()


@router.get('/profile')
def get_profile(db: Database = Depends(get_database)):
    """返回有效偏好、独立推断与精读统计。"""
    return MemoryService(db).profile()


@router.post('/profile/preferences')
def set_preference(payload: ProfileSet, db: Database = Depends(get_database)):
    """按当前版本更新偏好；请求重放不重复写入。"""
    try:
        return MemoryService(db).set_profile(payload)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.post('/profile/forget')
def forget_preference(payload: ForgetInput, db: Database = Depends(get_database)):
    """撤销但保留来源审计；过期版本返回409。"""
    try:
        return MemoryService(db).forget(payload.memory_id,payload.expected_version)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.post('/profile/preferences/statement')
def add_preference_statement(payload: PreferenceStatement, db: Database = Depends(get_database)):
    """接收自然语言明确偏好；不启动研究任务或模型请求。"""
    try:
        return MemoryService(db).add_statement(payload)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post('/memory/search')
def search_memory(payload: RecallInput, notes: NoteVersionService = Depends(get_note_version_service)):
    """按来源搜索历史及笔记，不把旧笔记当论文正文。"""
    return {'matches':MemoryService(notes.db,notes).search(payload.query,payload.source_type,payload.limit)}
