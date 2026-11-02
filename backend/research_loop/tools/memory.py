"""Supervisor记忆工具：确定性读写，LLM负责理解意图，数据库负责来源和权限。"""
from research_loop.domain.memory import RememberInput, ForgetInput, RecallInput, HistoryRead, ProfileRead, NoteToolUpdate, NoteUpdate
from research_loop.domain.tasks import Observation, SavePolicy
from research_loop.tools.base import ToolSpec


def _remember(args, ctx):
    """记录当前用户原句支持的偏好或推断，返回真实版本。"""
    return Observation(ok=True,kind='success',data={'memory':ctx.services['memory'].remember(ctx.task_id,args)})


def _forget(args, ctx):
    """撤销当前版本，保留审计且不再用于偏好召回。"""
    return Observation(ok=True,kind='success',data={'memory':ctx.services['memory'].forget(args.memory_id,args.expected_version)})


def _search(args, ctx):
    """检索历史/笔记，来源标签不变成原文证据授权。"""
    return Observation(ok=True,kind='success',data={'matches':ctx.services['memory'].search(args.query,args.source_type,args.limit,ctx.task_id)})


def _read(args, ctx):
    """根据稳定ID分页补读历史细节。"""
    return Observation(ok=True,kind='success',data=ctx.services['memory'].read(ctx.task_id,args))


def _profile(args, ctx):
    """获取当前有效偏好、推断与真实统计。"""
    return Observation(ok=True,kind='success',data=ctx.services['memory'].profile())


def _update(args, ctx):
    """预览或按当前任务授权保存新笔记版本，旧文件保持不变。"""
    payload=NoteUpdate.model_validate(args.model_dump(exclude={'note_id'}))
    return Observation(ok=True,kind='success',data=ctx.services['note_versions'].update(args.note_id,payload,ctx.save_policy==SavePolicy.allow))


def specs():
    """统一注册schema/实现；只给会话Supervisor开放。"""
    return [ToolSpec(name,description,model,handler,('supervisor',)) for name,description,model,handler in [
        ('remember_preference','Persist explicit LONG-TERM user preference, or separately labeled inferred interest. Current user quote required. Never turn a one-off instruction or file permission into a preference.',RememberInput,_remember),
        ('forget_preference','Revoke a current preference/inference by its ID and version. Do not revive old records.',ForgetInput,_forget),
        ('get_profile','Read current preferences, inferred interests and distinct successful deep-reading statistics.',ProfileRead,_profile),
        ('search_memory','Search previous research and saved project notes. These are history, not original paper evidence.',RecallInput,_search),
        ('read_memory','Read history messages, task records or compressed tool archives using the returned source_type/source_id and character offset/cursor. Preserve the exact source identity: a task ID is not an archive/message ID. History is not newly verified paper evidence and does not inherit file-save permission.',HistoryRead,_read),
        ('update_saved_note','Preview a diff; confirm=true saves a NEW version only if this task allows saving. Bind current base_hash. Old versions and external user edits are protected.',NoteToolUpdate,_update),
    ]]
