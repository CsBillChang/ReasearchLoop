# 来源诊断入口：记录各源查询结果；不把元数据查询成功当作 PDF 已下载或理解。
"""Explicit zero-model 论文 source probe; never runs maintenance jobs."""
from __future__ import annotations

import argparse
import json
from research_loop.config import get_settings
from research_loop.providers.discovery import DiscoveryProvider, SUPPORTED_SOURCES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", default="open vocabulary object detection", nargs="?")
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--skip-citations", action="store_true", help="Probe only 论文; no OpenAlex calls")
    args = parser.parse_args()
    settings = get_settings()
    discovery = DiscoveryProvider.from_settings(settings)
    try:
        result = discovery.search(args.query, args.date_from, args.date_to, sources=args.sources, limit=args.limit)
        print(json.dumps({"query": args.query, "supported_sources": list(SUPPORTED_SOURCES),
                          "search": result.model_dump(mode="json"), "citation_capability": "unsupported_by_user_decision"}, ensure_ascii=False, indent=2))
    finally:
        discovery.close()


if __name__ == "__main__":
    main()
