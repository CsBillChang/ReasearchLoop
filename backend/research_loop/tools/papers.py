"""论文工具：约束已分派论文身份；正文默认全文，页限制只取自用户要求。"""
from __future__ import annotations
from pydantic import BaseModel, Field
from research_loop.domain.tasks import Observation
from research_loop.domain.retrieval import RetrievalQuery
from research_loop.domain.visual_reading import VisualDelegationInput
from research_loop.providers.dpr_acquisition import PaperAcquisitionError
from research_loop.tools.base import ToolContext, ToolSpec


# 论文检索参数；日期是独立筛选字段，不要拼进关键词冒充过滤。
class SearchPapersArgs(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    date_from: str | None = None
    date_to: str | None = None
    sources: list[str] | None = None
    limit: int = Field(default=5, ge=1, le=20)


# 原文读取参数；start_page 从 1 起算，page_count 是本次调用返回页数。
class ReadPaperArgs(BaseModel):
    target_ref: str = Field(min_length=1, max_length=255)
    start_page: int = Field(default=1, ge=1)
    page_count: int | None = Field(default=None, ge=1)


def _search(args: SearchPapersArgs, ctx: ToolContext) -> Observation:
    """固定论文/附件子任务不得搜索替换材料；其余请求结合已分派来源和日期执行检索。"""
    if ctx.cancelled():
        return Observation(ok=False, kind="cancelled", message="cancelled")
    scope = ctx.services.get("scope") or {}
    if scope.get("paper_ids") or scope.get("input_refs"):
        raise PermissionError("fixed references: search cannot replace assigned papers")
    if scope.get("sources") and args.sources and not set(x.lower() for x in args.sources).issubset(set(x.lower() for x in scope["sources"])):
        raise PermissionError("search sources exceed assigned scope")
    result = ctx.services["papers"].search(args.query, scope.get("date_from") or args.date_from,
        scope.get("date_to") or args.date_to, scope.get("sources") or args.sources, args.limit, ctx.task_id)
    kind = "success" if result.candidates else ("empty" if all(item.status not in {"failed", "partial_results"} for item in result.source_results) else "execution_error")
    return Observation(ok=bool(result.candidates), kind=kind, data=result.model_dump(mode="json"),
                       message="search completed" if result.candidates else "no candidate; source statuses are included",
                       retryable=kind == "execution_error")


def _read(args: ReadPaperArgs, ctx: ToolContext) -> Observation:
    """返回授权论文正文；默认全文，只在用户明确要求时限制范围。

    Args:
        args: 论文身份与可选的本次读取片段。
        ctx: 角色权限、附件和业务服务。

    Returns:
        含页级原文、覆盖范围和解析缺口的工具观察。

    Raises:
        PermissionError: 子任务读取了未获分派的论文。
    """
    if ctx.cancelled():
        return Observation(ok=False, kind="cancelled", message="cancelled")
    scope = ctx.services.get("scope") or {}
    refs = (scope.get("paper_ids") or []) + (scope.get("input_refs") or [])
    if refs and args.target_ref not in refs:
        raise PermissionError("target_ref exceeds assigned references; use: " + ", ".join(refs))
    try:
        data = ctx.services["papers"].read(ctx.task_id, args.target_ref, args.start_page, args.page_count, ctx.attachments)
    except PaperAcquisitionError as exc:
        return Observation(ok=False, kind="execution_error", message=str(exc), retryable=exc.retryable,
                           data={"target_ref": args.target_ref, "error_code": exc.code, "acquisition": exc.acquisition})
    return Observation(ok=True, kind="success", data=data, message="page evidence returned")


def _media(args: ReadPaperArgs, ctx: ToolContext) -> Observation:
    """限制论文身份，返回媒体文件和能力缺口，不悄悄切换到图像模型。"""
    scope = ctx.services.get("scope") or {}
    refs = (scope.get("paper_ids") or []) + (scope.get("input_refs") or [])
    if refs and args.target_ref not in refs:
        raise PermissionError("media target exceeds assigned references")
    result = ctx.services["papers"].inspect_media(ctx.task_id, args.target_ref, args.start_page, args.page_count, ctx.attachments)
    return Observation(ok=True, kind="success" if result["assets"] else "empty", data=result,
                       message="media extraction only; image understanding is unavailable")


def _retrieve(args: RetrievalQuery, ctx: ToolContext) -> Observation:
    """按当前任务已获证据的确切文档版本检索，不授予新权限或触发精读。"""
    if ctx.cancelled():
        return Observation(ok=False, kind="cancelled", message="cancelled")
    runtime = ctx.services.get("retrieval")
    if runtime is None:
        return Observation(ok=False, kind="execution_error", message="retrieval service unavailable; use read_paper", retryable=False)
    data = runtime.search(ctx.task_id, args, ctx.cancelled)
    return Observation(ok=data["status"] != "unavailable", kind="success" if data.get("hits") else "empty",
        data=data, message="authorized source chunks; relevance is not factual verification", retryable=False)


def _visual(args: VisualDelegationInput, ctx: ToolContext) -> Observation:
    """委派单次Researcher节点；本处理函数不直接发送LLM请求或生成原文。"""
    dispatch = ctx.services.get('visual_reading')
    if not callable(dispatch):
        return Observation(ok=False, kind='execution_error', retryable=False,
            message='visual node unavailable; retain the textual result and disclose the specific gap')
    result = dispatch(ctx.task_id, args)
    success = result['status'] in {'complete', 'partial'}
    return Observation(ok=success, kind='success' if success else 'execution_error', retryable=False,
        data={'visual_observation': result},
        message='visual model observation, not original text or semantic approval' if success
                else 'visual execution failed: ' + (result.get('error') or 'unknown failure'))


def specs() -> list[ToolSpec]:
    """三类Agent共用参数/执行契约；Reviewer另限制只读工具集合。"""
    return [
        ToolSpec('delegate_visual_reading', 'Optional single Researcher visual node for an already-read physical PDF page, only when a damaged formula or missing spatial relation needs images. Give its actual evidence_id, specific question, reason and trigger; no paths, hashes or new search. Plain text stays the default. Same source/question/bbox reuses the result; use returned next_offset to read more without another visual request. Model observations are not exact source quotes or Reviewer approval. Failure or uncertainty may be disclosed and delivered.', VisualDelegationInput, _visual, ('supervisor',)),
        ToolSpec("retrieve_evidence", "Search already-read authorized original evidence using BM25 + vector + RRF + local Cross-encoder. Pass exact document_ids returned by read_paper; cannot search ungranted versions. For targeted questions/follow-ups, inspect ranked chunks and explicit degradations; scores are relevance, not factual proof or full-paper coverage. Use read_paper first if no page evidence, and for missing context. This does not generate deep-reading notes.", RetrievalQuery, _retrieve),
        ToolSpec("inspect_paper_media", "Extract authorized PDF figures/tables via optional PaperCropper, fallback embedded images. No image understanding; unknown page/bbox remain null.", ReadPaperArgs, _media),
        ToolSpec("search_papers", "Search configured paper sources. Abstracts and links are not full text.", SearchPapersArgs, _search),
        ToolSpec("read_paper", "Read the full paper by default. Omit page_count for all pages; use page ranges only for user restrictions or targeted follow-up. Returns page evidence and coverage.", ReadPaperArgs, _read),
    ]
