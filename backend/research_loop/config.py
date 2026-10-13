"""运行配置：合并环境变量和 DPR 来源配置；本模块不负责启动服务或验证供应商可用性。"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from research_loop.providers.dpr_sources import resolve_backends


def _load_dotenv(path: Path) -> None:
    """读取本地 .env，但不覆盖进程中已有变量；调用方不得把密钥配置打印到日志。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


class Settings(BaseModel):
    project_root: Path
    data_dir: Path
    output_dir: Path
    database_url: str = "mysql://researchloop:researchloop_local@127.0.0.1:33316/researchloop"
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    supabase_url: str | None = None
    supabase_key: str | None = None
    supabase_bm25_rpc: str = "match_arxiv_papers_bm25"
    discovery_backends: dict[str, dict[str, Any]] = Field(default_factory=dict, repr=False)
    openalex_api_key: str | None = None
    openalex_timeout_seconds: float = 20.0
    request_timeout_seconds: float = 120.0
    max_pdf_bytes: int = 80 * 1024 * 1024
    max_reflection_rounds: int = Field(default=2, ge=0, le=5)
    task_token_limit: int = Field(default=1000000, ge=1)
    task_token_tail: int = Field(default=32768, ge=0)
    max_model_requests: int = 48
    max_tool_actions: int = 80
    reserve_model_requests: int = 10
    memory_history_tokens: int = Field(default=12000, ge=256)
    memory_recent_messages: int = Field(default=4, ge=1)
    context_window_tokens: int = Field(default=128000, ge=4096)
    context_output_reserve: int = Field(default=8192, ge=256)
    context_safety_tokens: int = Field(default=4096, ge=256)
    context_compress_ratio: float = Field(default=0.8, gt=0, lt=1)
    retrieval_enabled: bool = True
    retrieval_model_manifest: Path | None = None
    retrieval_namespace: str = "researchloop_evidence_v1"
    qdrant_url: str = "http://127.0.0.1:6339"
    parser_name: str = "dpr"
    parser_version: str = "1"
    parser_config: dict[str, Any] = Field(default_factory=lambda: {"text_route": "cache-jina-pymupdf", "min_text_chars": 1200})

    @classmethod
    def from_env(cls, project_root: str | Path | None = None) -> "Settings":
        """输入可选项目根目录，输出合并后的 Settings；只读取配置，不探测外部接口。"""
        # 源码在 backend/research_loop；统一从仓库根读取 .env，手工入口与受管入口一致。
        root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        _load_dotenv(root / ".env")
        data_dir = Path(os.getenv("RESEARCHLOOP_DATA_DIR", str(root / "data" / "ag1"))).resolve()
        output_dir = Path(os.getenv("RESEARCHLOOP_OUTPUT_DIR", str(data_dir / "notes"))).resolve()
        config_path = Path(os.getenv("RESEARCHLOOP_DPR_CONFIG") or root / "data" / "config" / "dpr-sources.json")
        dpr_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        override = {k: os.environ[e] for k, e in (("url", "SUPABASE_URL"), ("anon_key", "SUPABASE_ANON_KEY"), ("schema", "SUPABASE_SCHEMA")) if os.getenv(e)}
        if override:
            dpr_config["supabase_shared"] = {**(dpr_config.get("supabase_shared") or dpr_config.get("supabase") or {}), **override}
            dpr_config["supabase"] = {**(dpr_config.get("supabase") or {}), **override}
            for backend in (dpr_config.get("source_backends") or {}).values():
                backend.update(override)
        if os.getenv("SUPABASE_BM25_RPC"):
            dpr_config.setdefault("source_backends", {}).setdefault("arxiv", {})["bm25_rpc"] = os.environ["SUPABASE_BM25_RPC"]
        return cls(
            project_root=root,
            data_dir=data_dir,
            output_dir=output_dir,
            database_url=os.getenv("DATABASE_URL", cls.model_fields["database_url"].default),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            deepseek_model=os.getenv("DEEPSEEK_MODEL") or "deepseek-chat",
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            supabase_url=os.getenv("SUPABASE_URL") or None,
            supabase_key=os.getenv("SUPABASE_ANON_KEY") or None,
            supabase_bm25_rpc=os.getenv("SUPABASE_BM25_RPC") or "match_arxiv_papers_bm25",
            discovery_backends=resolve_backends(dpr_config),
            task_token_limit=int(os.getenv("RESEARCHLOOP_TASK_TOKEN_LIMIT", "1000000")),
            task_token_tail=int(os.getenv("RESEARCHLOOP_TASK_TOKEN_TAIL", "32768")),
            max_reflection_rounds=int(os.getenv("RESEARCHLOOP_REFLECTION_ROUNDS", "2")),
            context_window_tokens=int(os.getenv("RESEARCHLOOP_CONTEXT_WINDOW", "128000")),
            memory_history_tokens=int(os.getenv("RESEARCHLOOP_HISTORY_TOKENS", "12000")),
            retrieval_enabled=os.getenv("RESEARCHLOOP_RETRIEVAL_ENABLED", "1") == "1",
            retrieval_model_manifest=Path(os.environ["RESEARCHLOOP_RETRIEVAL_MODEL_MANIFEST"]) if os.getenv("RESEARCHLOOP_RETRIEVAL_MODEL_MANIFEST") else None,
            retrieval_namespace=os.getenv("RESEARCHLOOP_RETRIEVAL_NAMESPACE", "researchloop_evidence_v1"),
            qdrant_url=os.getenv("RESEARCHLOOP_QDRANT_URL", "http://127.0.0.1:6339"),
            openalex_api_key=os.getenv("OPENALEX_API_KEY") or None,
        )


def get_settings() -> Settings:
    return Settings.from_env()
