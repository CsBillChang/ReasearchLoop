"""论文来源目录：来源别名、RPC 名称与连接配置的统一映射；配置存在不代表来源真实可用。

以下声明适用于本模块改编的第三方部分，不改变其他代码或依赖的许可：
Portions adapted from daily-paper-reader by ziwenhahaha.
Source: https://github.com/ziwenhahaha/daily-paper-reader

MIT License

Copyright (c) 2026 ziwenhahaha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations
from typing import Any

SUPPORTED_SOURCES = (
    "arxiv", "biorxiv", "medrxiv", "chemrxiv", "neurips", "iclr", "icml",
    "acl", "emnlp", "aaai", "cvpr", "eccv", "ijcai", "osdi", "sosp", "ieee_sp", "ndss",
)
DEFAULT_SEARCH_SOURCES = ("arxiv", "cvpr", "eccv", "iclr", "neurips", "icml", "aaai", "ijcai", "acl", "emnlp")
RPC_NAMES = {s: f"match_{s}{'_openreview' if s in {'iclr', 'neurips', 'icml'} else ''}_papers_bm25" for s in SUPPORTED_SOURCES}


def normalize_source(value: str) -> str:
    """将调用方来源别名转换为目录中的稳定来源名，供检索与配置查找共用。"""
    key = value.strip().lower().replace("-", "_")
    return {"s&p": "ieee_sp", "sp": "ieee_sp", "ieeesp": "ieee_sp", "ieee s&p": "ieee_sp", "nips": "neurips"}.get(key, key)


def resolve_backends(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """合并共享连接与逐源覆盖，返回可交给检索适配器的配置；不在此请求任何来源。"""
    legacy = config.get("supabase") or {}
    shared = config.get("supabase_shared") or legacy
    specific = config.get("source_backends") or {}
    result = {}
    for source in SUPPORTED_SOURCES:
        default = {"enabled": True, "schema": "public", "bm25_rpc": RPC_NAMES[source], "use_bm25_rpc": True}
        # 只共享连接字段，不能把 arXiv 的 RPC 复制给其他来源。
        default.update({k: shared[k] for k in ("url", "anon_key", "schema") if k in shared})
        if source == "arxiv":
            default.update(legacy)
        default.update(specific.get(source) or {})
        result[source] = default
    return result
