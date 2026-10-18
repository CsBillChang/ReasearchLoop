"""发现适配器：调用逐源 Supabase RPC，统一候选字段并记录查询状态。

涉及第三方适配的版权和许可声明见仓库根 NOTICE。"""
from __future__ import annotations
from datetime import date, datetime, timedelta, timezone
import time
from typing import Any
import httpx
from research_loop.domain.evidence import PaperCandidate, SearchResult, SourceResult
from research_loop.providers.paper_sources import SUPPORTED_SOURCES, DEFAULT_SEARCH_SOURCES, RPC_NAMES, normalize_source, resolve_backends

SOURCE_ROUTES = {s: f"dpr_supabase:{rpc}" for s, rpc in RPC_NAMES.items()}


class DiscoveryProvider:
    def __init__(self, supabase_url: str | None = None, supabase_key: str | None = None,
                 bm25_rpc: str = "match_arxiv_papers_bm25", timeout: float | None = None,
                 client: httpx.Client | None = None, *, backends: dict[str, dict[str, Any]] | None = None,
                 retry_wait_seconds: float = 1.0, max_date_windows: int = 8, max_attempts: int = 4):
        self.backends = backends if backends is not None else resolve_backends({
            "supabase": {"url": supabase_url, "anon_key": supabase_key, "bm25_rpc": bm25_rpc}})
        self.client = client or httpx.Client(follow_redirects=False)
        self._owns_client = client is None
        self.timeout = timeout
        self.retry_wait_seconds = retry_wait_seconds
        self.max_date_windows = max(1, max_date_windows)
        self.max_attempts = max(1, min(4, max_attempts))
        self.last_requests: list[dict[str, Any]] = []

    @classmethod
    def from_settings(cls, settings, **kwargs):
        """从配置解析逐源后端；不把缺失配置的来源标为已查询成功。"""
        # 论文源 RPC 超时独立于模型 HTTP 超时：arXiv BM25 45 秒、exact 75 秒，其余 20 秒。
        return cls(backends=settings.discovery_backends, **kwargs)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def search(self, query: str, date_from: str | None = None, date_to: str | None = None,
               sources: list[str] | None = None, limit: int = 5) -> SearchResult:
        """返回有界候选和全部来源状态；未查、空结果、失败相互独立，不能宣称三年全量覆盖。"""
        selected = list(dict.fromkeys(normalize_source(s) for s in (sources or DEFAULT_SEARCH_SOURCES)))
        unknown = set(selected) - set(SUPPORTED_SOURCES)
        if unknown:
            raise ValueError("unsupported sources: " + ",".join(sorted(unknown)))
        if date_from: date.fromisoformat(date_from)
        if date_to: date.fromisoformat(date_to)
        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from must not be after date_to")
        if not query.strip():
            raise ValueError("query must not be empty")
        cap = max(1, min(limit, 20))
        results, groups = [], []
        self.last_requests = []
        saturated = False
        for source in SUPPORTED_SOURCES:
            cfg = self.backends.get(source) or {}
            if source not in selected:
                results.append(SourceResult(source=source, status="not_queried", message="source not selected"))
                continue
            if not cfg.get("enabled", True) or not cfg.get("use_bm25_rpc", True) or not cfg.get("url") or not cfg.get("anon_key"):
                results.append(SourceResult(source=source, status="not_configured", message="论文 Supabase read endpoint is missing or disabled"))
                continue
            try:
                found, count, request = self._search_windows(query, date_from, date_to, cap, source)
                saturated |= count >= cap or bool(request.get("unresolved_windows"))
                groups.append(found)
                results.append(SourceResult(source=source, status=("partial_results" if request.get("unresolved_windows") else ("queried_success" if found else "queried_empty")),
                                            message=f"论文 Supabase RPC {cfg['bm25_rpc']}; bounded top-k, not full coverage",
                                            candidate_count=len(found), request=request))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}".replace(str(cfg.get("anon_key") or "[none]"), "[redacted]")
                request = self.last_requests[-1] if self.last_requests and self.last_requests[-1].get("source") == source else {}
                results.append(SourceResult(source=source, status="failed", message=message, request=request))
        # 不同来源的 RPC 分数不可直接比较；在总候选上限内轮流取样，保留来源多样性。
        candidates, seen = [], set()
        for index in range(cap):
            for group in groups:
                if index < len(group) and group[index].paper_id not in seen:
                    seen.add(group[index].paper_id)
                    candidates.append(group[index])
        return SearchResult(candidates=candidates[:cap], source_results=results, date_from=date_from, date_to=date_to,
                            truncated=saturated or len(candidates) > cap,
                            query_scope=f"sources={','.join(selected)}; per-source top-k={cap}; global limit={cap}")

    def _search_windows(self, query: str, date_from: str | None, date_to: str | None,
                        limit: int, source: str) -> tuple[list[PaperCandidate], int, dict[str, Any]]:
        """论文源数据库 57014 细分日期窗口；有界保留成功结果及未完成区间。

        Args:
            query: 原始基础关键词，分片时不改写。
            date_from: 包含的起始日；无完整日期时不能猜测分片范围。
            date_to: 包含的结束日。
            limit: 每个窗口与返回候选上限。
            source: 配置的来源身份。

        Returns:
            去重候选、最大窗口饱和值、每个实际窗口诊断；不宣称分页或全量覆盖。
        """
        pending = [(date_from, date_to)]
        windows, unresolved, found, seen = [], [], [], set()
        max_count = 0
        while pending and len(windows) < self.max_date_windows:
            lower, upper = pending.pop(0)
            try:
                rows, count, request = self._search_supabase(query, lower, upper, limit, source)
                windows.append(request)
                max_count = max(max_count, count)
                for row in rows:
                    if row.paper_id not in seen:
                        seen.add(row.paper_id)
                        found.append(row)
            except Exception as exc:
                request = self.last_requests[-1]
                windows.append(request)
                can_split = (request.get("retry_stopped_reason") == "database_statement_timeout"
                             and lower and upper and lower < upper)
                if can_split:
                    start, end = date.fromisoformat(lower), date.fromisoformat(upper)
                    middle = start + (end - start) // 2
                    pending.extend([(lower, middle.isoformat()), ((middle + timedelta(days=1)).isoformat(), upper)])
                else:
                    unresolved.append({"date_from": lower, "date_to": upper, "reason": str(exc)[:500]})
                if len(windows) == 1 and not can_split:
                    raise  # 普通 HTTP/单日失败继续由原逐源错误分支记录。
        unresolved.extend({"date_from": lo, "date_to": hi, "reason": "date window budget exhausted"} for lo, hi in pending)
        if len(windows) == 1 and not unresolved:
            return found[:limit], max_count, windows[0]
        aggregate = {"source": source, "windows": windows, "max_date_windows": self.max_date_windows,
                     "unresolved_windows": unresolved, "deduplicated_candidates": len(found),
                     "coverage": "bounded_date_windows"}
        self.last_requests.append(aggregate)
        return found[:limit], max(max_count, len(found)), aggregate

    def _search_supabase(self, query: str, date_from: str | None, date_to: str | None, limit: int,
                         source: str) -> tuple[list[PaperCandidate], int, dict[str, Any]]:
        """调用来源专属 RPC 并保存请求诊断；上游 HTTP 失败会进入对应来源的失败状态。"""
        cfg = self.backends[source]
        base = str(cfg["url"]).rstrip("/")
        if not base.endswith("/rest/v1"):
            base += "/rest/v1"
        rpc = cfg.get("bm25_rpc") or RPC_NAMES[source]
        payload: dict[str, Any] = {"query_text": query.strip(), "match_count": limit}
        if date_from:
            payload["filter_published_start"] = datetime.combine(date.fromisoformat(date_from), datetime.min.time(), timezone.utc).isoformat()
        if date_to:
            # 工具日期包含结束当天，上游 SQL 结束边界不包含当天，因此传入次日零点。
            payload["filter_published_end"] = datetime.combine(date.fromisoformat(date_to) + timedelta(days=1), datetime.min.time(), timezone.utc).isoformat()
        headers = {"apikey": cfg["anon_key"], "Accept": "application/json", "Content-Type": "application/json",
                   "Accept-Profile": cfg.get("schema", "public"), "Content-Profile": cfg.get("schema", "public")}
        if not cfg["anon_key"].startswith("sb_publishable_"):
            headers["Authorization"] = f"Bearer {cfg['anon_key']}"
        endpoint = f"{base}/rpc/{rpc}"
        request = {"source": source, "endpoint": endpoint, "rpc": rpc, "method": "POST",
                   "payload": payload, "schema": cfg.get("schema", "public")}
        self.last_requests.append(request)
        timeout = max(1.0, self.timeout if self.timeout is not None else {
            "match_arxiv_papers_exact": 75.0, "match_arxiv_papers_bm25": 45.0}.get(rpc, 20.0))
        response = self._request_with_retries(endpoint, payload, headers, timeout, request)
        if response.status_code >= 300:
            raise ValueError(f"论文 RPC HTTP {response.status_code}: {response.text[:500]}")
        rows = response.json()
        if not isinstance(rows, list):
            raise ValueError("论文 RPC returned non-list")
        candidates = [self._row_to_candidate(row, source, rpc) for row in rows
                      if isinstance(row, dict) and row.get("id") and self._in_window(row.get("published"), date_from, date_to)]
        request["response_rows"] = len(rows)
        request["candidate_rows_in_window"] = len(candidates)
        return candidates[:limit], len(rows), request

    def _request_with_retries(self, endpoint: str, payload: dict, headers: dict,
                              timeout: float, request: dict) -> httpx.Response:
        """适配只读 RPC 重试，默认四次；简报由持久调度层限制重试，仅429/临时5xx/传输失败有限退避，数据库语句超时不重试。

        Args:
            endpoint: 已配置来源的 RPC 地址。
            payload: 关键词、数量及独立日期参数。
            headers: 只读认证头；不写入诊断。
            timeout: 当前 RPC 的单次 HTTP 超时秒数。
            request: 收集逐次状态/网络错误的诊断对象，异常也保留。

        Returns:
            最终 HTTP 响应；调用方区分失败、空结果和候选。

        Raises:
            httpx.RequestError: 四次传输尝试后仍未取得响应。
        """
        from research_loop.providers.read_retry import read_with_backoff
        request["timeout_seconds"] = timeout
        return read_with_backoff(
            lambda: self.client.post(endpoint, json=payload, headers=headers, timeout=timeout),
            request, attempts=self.max_attempts, base=self.retry_wait_seconds, sleep=time.sleep)

    @staticmethod
    def _in_window(value: str | None, date_from: str | None, date_to: str | None) -> bool:
        """根据日期精度判断是否与窗口相交；年份日期不能伪装为精确发表日。"""
        if not value: return True
        value = str(value)[:10]
        try:
            if len(value) == 4:
                lower, upper = value + "-01-01", value + "-12-31"
            elif len(value) == 7:
                import calendar
                year, month = map(int, value.split("-"))
                lower, upper = value + "-01", value + f"-{calendar.monthrange(year, month)[1]}"
            else:
                lower = upper = date.fromisoformat(value).isoformat()
            return not ((date_from and upper < date_from) or (date_to and lower > date_to))
        except ValueError:
            return True

    @staticmethod
    def _row_to_candidate(row: dict[str, Any], source: str = "arxiv", rpc: str = "") -> PaperCandidate:
        """将上游字段映射为统一候选；PDF 链接仍需后续下载验证，不能当作正文证据。"""
        import re
        published = str(row.get("published") or "")[:10] or None
        precision = {4: "year", 7: "month", 10: "day"}.get(len(published or ""), "unknown")
        pdf_url = row.get("pdf_url")
        # arXiv RPC 可能缺少 pdf_url；只从合法 arXiv ID 推导 PDF 地址，不能把详情页当 PDF。
        identifier = str(row["id"]).removeprefix("arxiv:")
        if not pdf_url and source == "arxiv" and re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", identifier):
            pdf_url = "https://arxiv.org/pdf/" + identifier
        return PaperCandidate(
            paper_id=str(row["id"]), title=str(row.get("title") or "").strip(),
            authors=row.get("authors") if isinstance(row.get("authors"), list) else [],
            abstract=str(row.get("abstract") or "").strip() or None, source=source,
            published=published, date_precision=precision, paper_url=row.get("link"), pdf_url=pdf_url,
            score=row.get("score"), doi=row.get("doi"), source_id=str(row["id"]), version=row.get("version"),
            metadata={"discovery_backend": "dpr_supabase_rpc", "rpc": rpc, "source_raw": row.get("source"),
                      "published_raw": row.get("published"), "categories": row.get("categories") or [],
                      "primary_category": row.get("primary_category"), "similarity": row.get("similarity"),
                      "date_provenance": "upstream metadata; conference dates may be nominal"},
        )
