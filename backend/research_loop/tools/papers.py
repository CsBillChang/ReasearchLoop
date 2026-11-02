"""论文工具：约束已分派论文身份；正文默认全文，页限制只取自用户要求。"""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from research_loop.domain.tasks import Observation
from research_loop.domain.source_access import WebSource, normalize_web_sources, source_error_diagnostic
from research_loop.domain.retrieval import RetrievalQuery
from research_loop.domain.visual_reading import VisualDelegationInput
from research_loop.providers.paper_acquisition import PaperAcquisitionError
from research_loop.providers.pdf import PaperReadCancelled
from research_loop.tools.base import ToolContext, ToolSpec


# 论文检索参数；日期是独立筛选字段，不要拼进关键词冒充过滤。
class SearchPapersArgs(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    date_from: str | None = None
    date_to: str | None = None
    sources: list[str] | None = None
    limit: int = Field(default=5, ge=1, le=20)


# 网页Search的官方参数边界独立于原Supabase查询。
class SearchWebPapersArgs(BaseModel):
    query: str = Field(min_length=1, max_length=128)
    count: int = Field(default=5, ge=1, le=50)
    date_from: str | None = None
    date_to: str | None = None
    sources: list[WebSource] | None = None

    @field_validator('sources', mode='before')
    @classmethod
    def normalize_sources(cls, value):
        """schema声明的来源和执行使用同一归一化合同，非法值不发送请求。"""
        return normalize_web_sources(value)

    @classmethod
    def validation_error_data(cls, error, arguments):
        """只为来源参数提供无请求诊断；其他字段沿用通用参数校验。"""
        if any(item['loc'] and item['loc'][0] == 'sources' for item in error.errors()):
            return {**source_error_diagnostic(error), 'tool_executed':False}
        return {}


# 原文读取参数；start_page 从 1 起算，page_count 是本次调用返回页数。
class ReadPaperArgs(BaseModel):
    target_ref: str = Field(min_length=1, max_length=255)
    start_page: int = Field(default=1, ge=1)
    page_count: int | None = Field(default=None, ge=1)


# 备用获取保留显式身份线索；缺省可由当前用户明确且唯一的题录自动衔接。
class AcquirePaperPdfArgs(ReadPaperArgs):
    metadata_ref: str | None = Field(default=None, min_length=1, max_length=255)


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
        scope.get("date_to") or args.date_to, args.sources if args.sources is not None else scope.get("sources"), args.limit, ctx.task_id)
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
        data = ctx.services["papers"].read(ctx.task_id, args.target_ref, args.start_page, args.page_count,
                                           ctx.attachments, _cancelled=ctx.cancelled)
    except PaperReadCancelled as exc:
        return Observation(ok=False, kind="cancelled", message=str(exc), retryable=False,
                           data={"target_ref": args.target_ref, "error_code": "PAPER_READ_CANCELLED"})
    except PaperAcquisitionError as exc:
        return Observation(ok=False, kind="execution_error", message=str(exc), retryable=exc.retryable,
                           data={"target_ref": args.target_ref, "error_code": exc.code, "acquisition": exc.acquisition})
    return Observation(ok=True, kind="success", data=data, message="page evidence returned")


def _search_web(args: SearchWebPapersArgs, ctx: ToolContext) -> Observation:
    """显式备用检索；固定论文子目标不能搜索替换，其余权限从原任务读取。"""
    scope = ctx.services.get('scope') or {}
    if scope.get('paper_ids') or scope.get('input_refs'):
        raise PermissionError('fixed references: topic search cannot replace assigned papers')
    try:
        data = ctx.services['papers'].search_web(ctx.task_id, args.query, args.count,
            date_from=scope.get('date_from') or args.date_from, date_to=scope.get('date_to') or args.date_to,
            sources=args.sources, assigned_sources=scope.get('sources'), cancelled=ctx.cancelled)
    except PaperAcquisitionError as exc:
        return Observation(ok=False, kind='execution_error', message=str(exc), retryable=False,
                           data={'error_code': exc.code, 'acquisition': exc.acquisition})
    return Observation(ok=bool(data['results']), kind='success' if data['results'] else 'empty',
                       data=data, message='backup candidates only; publication dates and original evidence are not verified')


def _acquire_pdf(args: AcquirePaperPdfArgs, ctx: ToolContext) -> Observation:
    """仅为已授权论文选择同版本PDF备用链，保留原获取失败与页权限。"""
    scope = ctx.services.get('scope') or {}
    refs = (scope.get('paper_ids') or []) + (scope.get('input_refs') or [])
    try:
        data = ctx.services['papers'].acquire_pdf(ctx.task_id, args.target_ref, args.start_page, args.page_count,
            ctx.attachments, cancelled=ctx.cancelled, sources=scope.get('sources'),
            metadata_ref=args.metadata_ref, allowed_refs=refs)
    except PaperReadCancelled as exc:
        return Observation(ok=False, kind='cancelled', message=str(exc), retryable=False,
                           data={'target_ref': args.target_ref, 'error_code': 'PAPER_READ_CANCELLED'})
    except PaperAcquisitionError as exc:
        return Observation(ok=False, kind='execution_error', message=str(exc), retryable=exc.retryable if exc.code == 'PDF_METADATA_VERSION_MISMATCH' else False,
                           data={'target_ref': args.target_ref, 'error_code': exc.code, 'acquisition': exc.acquisition})
    if data.get('status') == 'metadata_reference_required':
        return Observation(ok=False, kind='validation_error', data=data, retryable=True,
            message='已实际尝试原URL但取文失败；当前用户身份线索不足、冲突或不唯一，未自动绑定题录。按diagnostic明确选择授权metadata_ref，不猜同论文关联。')
    success = bool(data.get('pages'))
    locator = (data.get('read_coverage') or {}).get('locator_kind')
    success_message = ('physical PDF evidence returned' if locator == 'physical_page' else
                       'document text evidence returned; physical PDF pages are unavailable' if locator == 'document_text' else
                       'source evidence returned; physical PDF coverage is unconfirmed')
    return Observation(ok=success, kind='success' if success else 'empty', data=data, retryable=False,
                       message=success_message if success else 'same-paper/version PDF remains unconfirmed; original failure retained')


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


def specs(*, web_search_enabled: bool = False, web_fetch_enabled: bool = False) -> list[ToolSpec]:
    """三类Agent共用参数/执行契约；Reviewer另限制只读工具集合。"""
    optional = []
    if web_search_enabled:
        optional.append(ToolSpec('search_web_papers', 'Optional backup after configured paper sources are empty or fail. Search official paper metadata with the ORIGINAL query and date/source constraints; no more than 128 Unicode characters. Canonical paper candidates can be selected by paper_id and read_paper; webpage dates are not publication dates, and publication eligibility remains unverified; arbitrary non-paper URLs remain ungranted clues. Preserve prior RPC failures. Search summaries are not original evidence. User instructions forbidding external search always win.', SearchWebPapersArgs, _search_web, ('supervisor',)))
    if web_search_enabled and web_fetch_enabled:
        optional.append(ToolSpec('acquire_paper_pdf', 'Optional backup for an already-authorized paper_id or a canonical paper URL explicitly supplied by the current user. A user URL is tried as its own identity first; it is never an alias for a different paper. If original metadata is unavailable, the service automatically continues using a uniquely matching authorized reference only when the current user explicitly provided its full title and complete authors or DOI. Otherwise supply metadata_ref from the returned authorized candidates; never infer identity from a lone candidate. Fetch must still verify title/authors or DOI. Source preference is not a hard restriction: the internal backup route may select other legitimate sources unless the current user or assigned scope forbids them. An unread original URL association stays original_locator_identity_unverified; this does not prove it equals the reference. Preserve assigned materials. When the existing PDF route fails. Search the exact paper identity, Fetch a matching detail page ONLY to locate its PDF, then use the original physical-page parser. Never silently substitute versions; explicit user version/page restrictions block variants. Without that restriction, a verified same-paper alternate is returned with a separate identity and version warning. Do not bypass paywalls/authentication/challenges or treat Fetch Markdown as full text. Retains original failure; may return unconfirmed candidates. Existing page and material restrictions remain.', AcquirePaperPdfArgs, _acquire_pdf, ('supervisor',)))
    return optional + [
        ToolSpec('delegate_visual_reading', 'Use one Researcher visual node only for an already-read physical PDF page whose damaged formula or missing spatial relation needs images. Supply actual evidence_id, specific question/reason and trigger=damaged_formula or missing_spatial_relation; bbox is optional. Do not supply paths/hashes or launch a new search. Reuse the same source/question/bbox; follow returned next_offset without regenerating. The result is an unverified model observation, not an exact source quote, whole-paper understanding or Reviewer approval. On error/uncertainty retain existing results and disclose the gap.', VisualDelegationInput, _visual, ('supervisor',)),
        ToolSpec("retrieve_evidence", 'Retrieve relevant chunks from already-read authorized original evidence. Use exact document_ids returned by read_paper; no ungranted versions. Inspect returned scope and degradation notices: relevance scores do not prove a claim or full-paper coverage. Read missing original context with read_paper. Retrieval does not create a deep-reading note or grant independent approval.', RetrievalQuery, _retrieve),
        ToolSpec("inspect_paper_media", "Extract authorized PDF figures/tables via optional PaperCropper, fallback embedded images. No image understanding; unknown page/bbox remain null.", ReadPaperArgs, _media),
        ToolSpec("search_papers", "Search configured paper sources. Abstracts and links are not full text.", SearchPapersArgs, _search),
        ToolSpec("read_paper", 'Read authorized paper_ids/input_refs exactly as returned; do not guess IDs or replace assigned attachments with a search. Full paper is the default: omit page_count; page ranges are for an explicit user restriction or a targeted follow-up. Assigned start_page/page_count are suggestions, not permanent permission. Check returned read_coverage/user_page_range and gaps. Page evidence is extracted text, not proof of understanding images/formulas; page_number=null means no reliable physical page number, so cite evidence_id without inventing one.', ReadPaperArgs, _read),
    ]
