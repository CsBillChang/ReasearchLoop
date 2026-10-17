"""项目笔记版本API：差异预览与明确确认保存分离，旧hash和用户改动阻断覆盖。"""
from fastapi import APIRouter, Depends, HTTPException
from research_loop.api.dependencies import get_note_version_service
from research_loop.services.note_versions import NoteVersionService
from research_loop.domain.memory import NoteUpdate

router=APIRouter()




@router.get('/notes')
def list_notes(notes: NoteVersionService = Depends(get_note_version_service)):
    """返回已保存笔记的当前版本。"""
    return {'notes':notes.list_current()}


@router.get('/notes/{note_id}')
def get_note(note_id: str, notes: NoteVersionService = Depends(get_note_version_service)):
    """返回内容、版本与完整性，缺失返回404。"""
    try:
        return notes.get(note_id)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc


@router.get('/notes/{note_id}/versions')
def note_versions(note_id: str, notes: NoteVersionService = Depends(get_note_version_service)):
    """读回不可变版本历史。"""
    try:
        return {'versions':notes.versions(note_id)}
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc


@router.post('/notes/{note_id}/versions')
def update_note(note_id: str,payload: NoteUpdate, notes: NoteVersionService = Depends(get_note_version_service)):
    """confirm=false只预览；明确confirm=true是当前人工HTTP保存授权。"""
    try:
        return notes.update(note_id,payload,save_allowed=payload.confirm)
    except KeyError as exc:
        raise HTTPException(404,str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc
