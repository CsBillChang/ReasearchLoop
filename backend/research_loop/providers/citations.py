"""历史引用统计适配器：保留诊断/测试代码；当前 Worker 不接入，被引量已退出日常研究能力。"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import httpx

from research_loop.domain.evidence import CitationStats, PaperCandidate


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


class OpenAlexCitationProvider:
    def __init__(self, api_key: str | None = None, timeout: float = 20.0,
                 client: httpx.Client | None = None):
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def lookup(self, candidate: PaperCandidate) -> CitationStats:
        queried_at = datetime.now(timezone.utc).isoformat()
        access_mode = "api_key" if self.api_key else "anonymous"
        try:
            params = {"api_key": self.api_key} if self.api_key else {}
            data: dict[str, Any] | None = None
            matched_by = "none"
            if candidate.doi:
                doi = candidate.doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
                response = self.client.get("https://api.openalex.org/works/https://doi.org/" + doi, params=params)
                if response.status_code == 200:
                    data = response.json()
                    matched_by = "doi"
                elif response.status_code != 404:
                    return self._http_unavailable(candidate, queried_at, access_mode, response)
            if data is None:
                response = self.client.get("https://api.openalex.org/works",
                                           params={**params, "search": candidate.title, "per-page": 5})
                if response.status_code != 200:
                    return self._http_unavailable(candidate, queried_at, access_mode, response)
                rows = (response.json() or {}).get("results") or []
                matched = [row for row in rows if self._matches(candidate, row)]
                if len(matched) != 1:
                    return CitationStats(paper_id=candidate.paper_id,
                                         status="ambiguous" if matched else "unmatched",
                                         queried_at=queried_at,
                                         message="title/author/year match is not unique",
                                         access_mode=access_mode)
                data = matched[0]
                matched_by = "title_authors_year"
            count = data.get("cited_by_count")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                return CitationStats(paper_id=candidate.paper_id, status="unavailable", queried_at=queried_at,
                                     message="matched record has no valid cited_by_count; not assumed zero",
                                     access_mode=access_mode)
            return CitationStats(paper_id=candidate.paper_id, status="matched",
                                 count=count, provider_id=data.get("id"),
                                 matched_by=matched_by,
                                 matched_title=data.get("title"), queried_at=queried_at,
                                 access_mode=access_mode)
        except Exception as exc:
            message = type(exc).__name__ + ": " + str(exc)
            if self.api_key:
                message = message.replace(self.api_key, "[redacted]")
            return CitationStats(paper_id=candidate.paper_id, status="unavailable", queried_at=queried_at,
                                 message=message, access_mode=access_mode)

    @staticmethod
    def _header_int(response: httpx.Response, name: str) -> int | None:
        value = response.headers.get(name)
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None

    def _http_unavailable(self, candidate: PaperCandidate, queried_at: str,
                          access_mode: str, response: httpx.Response) -> CitationStats:
        # Preserve a bounded diagnostic without exposing an API key in a URL/body.
        body = response.text[:1000]
        if self.api_key:
            body = body.replace(self.api_key, "[redacted]")
        return CitationStats(
            paper_id=candidate.paper_id, status="unavailable", queried_at=queried_at,
            message=f"HTTP {response.status_code}: {body}", http_status=response.status_code,
            retry_after_seconds=self._header_int(response, "retry-after") or self._header_int(response, "x-ratelimit-reset"),
            rate_limit_remaining=self._header_int(response, "x-ratelimit-remaining"),
            access_mode=access_mode,
        )

    @staticmethod
    def _matches(candidate: PaperCandidate, row: dict[str, Any]) -> bool:
        if _norm(candidate.title) != _norm(str(row.get("title") or "")):
            return False
        year = str(candidate.published or "")[:4]
        publication_date = str(row.get("publication_date") or "")[:4]
        if year and publication_date != year:
            return False
        wanted = {_norm(author).split()[-1] for author in candidate.authors if author}
        actual = {_norm((item.get("author") or {}).get("display_name", "")).split()[-1]
                  for item in (row.get("authorships") or [])}
        return not wanted or bool(wanted & actual)
