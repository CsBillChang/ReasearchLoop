"""笔记工具：create_note 生成数据库草稿，save_note 才请求项目内文件写入，二者不可混淆。"""
from __future__ import annotations
from research_loop.domain.notes import CreateNoteInput, SaveNoteInput, RevisionInput
from research_loop.domain.tasks import Observation
from research_loop.tools.base import ToolContext, ToolSpec


def _create(args: CreateNoteInput, ctx: ToolContext) -> Observation:
    draft = ctx.services["notes"].create(ctx.task_id, args)
    return Observation(ok=True, kind="success", data=draft.model_dump(mode="json"), message="draft persisted; not independently reviewed")


def _save(args: SaveNoteInput, ctx: ToolContext) -> Observation:
    saved = ctx.services["notes"].save(ctx.task_id, ctx.save_policy, args.draft_id, args.content_hash)
    return Observation(ok=True, kind="success", data=saved.model_dump(mode="json"), message="project-local save receipt created")


def _revise(args: RevisionInput, ctx: ToolContext) -> Observation:
    """应用 Supervisor 提供的局部修改；返回新版本，不自动调用模型或保存。"""
    result = ctx.services["reflection"].revise(ctx.task_id, args)
    return Observation(ok=True, kind="success", data=result, message="local revision persisted; independently review the new hash next")


def specs() -> list[ToolSpec]:
    return [
        ToolSpec("revise_note", "Apply exact local patches bound to the latest unresolved review. Preserve all other text; then independently review the new draft. No model call inside this tool.", RevisionInput, _revise, ("supervisor",)),
        ToolSpec("create_note", "Persist requested Markdown as-is. Evidence IDs are internal optional metadata, not required text citations.", CreateNoteInput, _create),
        ToolSpec("save_note", "Save the current draft only within the backend-authorized project output directory.", SaveNoteInput, _save),
    ]
