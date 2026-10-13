"""项目笔记版本API：差异预览与明确确认保存分离，旧hash和用户改动阻断覆盖。"""
from fastapi import APIRouter, HTTPException
from research_loop.config import get_settings
from research_loop.storage.mysql import Database
from research_loop.storage.files import FileStore
from research_loop.services.note_versions import NoteVersionService
from research_loop.domain.memory import NoteUpdate

router=APIRouter()


def _notes():
    """创建项目内服务；文件目录来自配置，不接受调用方路径。"""
    settings=get_settings()
    db=Database(settings.database_url)
    db.ensure_schema()
    return NoteVersionService(db,FileStore(settings.data_dir,settings.output_dir,settings.project_root))


@router.get('/notes')
def list_notes():
    """返回已保存笔记的当前版本。"""
    return {'notes':_notes().list_current()}


@router.get('/notes/{note_id}')
def get_note(note_id: str):
    """返回内容、版本与完整性，缺失返回404。"""
    try:
        return _notes().get(note_id)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc


@router.get('/notes/{note_id}/versions')
def note_versions(note_id: str):
    """读回不可变版本历史。"""
    try:
        return {'versions':_notes().versions(note_id)}
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc


@router.post('/notes/{note_id}/versions')
def update_note(note_id: str,payload: NoteUpdate):
    """confirm=false只预览；明确confirm=true是当前人工HTTP保存授权。"""
    try:
        return _notes().update(note_id,payload,save_allowed=payload.confirm)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc
