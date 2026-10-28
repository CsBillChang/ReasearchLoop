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
        ToolSpec("revise_note", 'Apply exact local patches to the current draft_id/content_hash and latest unresolved review_id. finding_index is zero-based in that review; old_text must occur uniquely. Preserve unrelated text and evidence. Success creates an unreviewed draft version, without a model call or file save; independently review its new hash with the original neutral questions.', RevisionInput, _revise, ("supervisor",)),
        ToolSpec("create_note", "Create a database draft when the task needs a note, using retained Markdown as-is. Evidence IDs are optional internal metadata; no mandatory Sources template or evidence_id strings in the prose. Creating a draft neither writes a file nor grants independent approval. Keep a complete deep-reading note's existing end marker and disclose incomplete results.", CreateNoteInput, _create),
        ToolSpec("save_note", 'Write the current draft_id/content_hash only when the current user requests file saving and save_policy=allow, within the backend-authorized output directory. Historical permissions do not apply. deny permits task/answer snapshots but no file save. A successful save receipt establishes the file operation, not semantic correctness.', SaveNoteInput, _save),
    ]
