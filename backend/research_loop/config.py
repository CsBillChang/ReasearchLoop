"""运行配置：合并环境变量和论文来源配置；不负责启动服务或验证供应商可用性。"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from research_loop.providers.paper_sources import resolve_backends


def paper_sources_path(root: Path) -> Path:
    """定位论文源配置，优先新入口，同时兼容已有本机配置。

    Args:
        root: 项目根目录；默认配置位于其 data/config 下。

    Returns:
        显式指定或默认配置路径；不存在时由调用方按未配置处理。
    """
    explicit = os.getenv("RESEARCHLOOP_PAPER_SOURCES_CONFIG") or os.getenv("RESEARCHLOOP_DPR_CONFIG")
    if explicit:
        return Path(explicit).expanduser().resolve()
    preferred = root / "data" / "config" / "paper-sources.json"
    legacy = root / "data" / "config" / "dpr-sources.json"
    return preferred if preferred.exists() or not legacy.exists() else legacy


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
    researcher_concurrency: int = Field(default=2, ge=1)
    execution_budget_mode: Literal["enforce", "observe"] = "enforce"
    web_search_enabled: bool = False
    web_fetch_enabled: bool = False
    web_tools_timeout_seconds: float = Field(default=30.0, gt=0)
    web_tools_max_response_bytes: int = Field(default=2*1024*1024, gt=0)
    max_model_requests: int = 48
    max_tool_actions: int = 80
    reserve_model_requests: int = 10
    memory_history_tokens: int = Field(default=12000, ge=256)
    memory_recent_messages: int = Field(default=4, ge=1)
    # 0 表示按模型能力解析；显式环境值（包括旧128000）始终优先。
    context_window_tokens: int = Field(default=0, ge=0)
    model_max_output_tokens: int = Field(default=0, ge=0)
    token_counting_tokenizer_path: Path | None = None
    token_counting_tokenizer_model: str | None = None
    token_counting_tokenizer_sha256: str | None = None
    # None 表示未显式配置；解析后按数值冻结，旧根与显式小值不重新推断。
    context_output_reserve: int | None = Field(default=None, ge=256)
    context_safety_tokens: int = Field(default=4096, ge=256)
    context_compress_ratio: float = Field(default=0.8, gt=0, lt=1)
    # 实验开关仅影响新模型请求的副本；默认沿用既有正文呈现。
    context_deduplicate_read_paper: bool = False
    retrieval_enabled: bool = True
    retrieval_route: Literal["located_fp16", "hybrid_rrf"] = "located_fp16"
    retrieval_reranker_backend: Literal["mps_fp16", "onnx_cpu"] = "mps_fp16"
    retrieval_model_manifest: Path | None = None
    retrieval_namespace: str = "researchloop_evidence_v1"
    qdrant_url: str = "http://127.0.0.1:6339"
    parser_name: str = "dpr"
    parser_version: str = "1"
    parser_config: dict[str, Any] = Field(default_factory=lambda: {"text_route": "cache-jina-pymupdf", "min_text_chars": 1200})

    @model_validator(mode="after")
    def resolve_model_capacity(self):
        """按已知模型补容量与普通角色输出默认，不覆盖显式数值。"""
        from research_loop.providers.token_counting import model_capabilities
        capability = model_capabilities(self.deepseek_model)
        if self.context_output_reserve is None:
            # 普通角色沿用供应商默认思考模式；矩阵例外仍共享根预留，
            # 精读与视觉节点的独立请求合同不在此推断模式或修改。
            value = 8192 if capability['profile'] == 'legacy_or_unknown' else 65536
            # 保留未显式设置的来源；图仅为无冻结策略的更老根恢复旧默认。
            # model_copy显式更新仍会进入fields_set，序列化的根数值也视为显式。
            object.__setattr__(self, 'context_output_reserve', value)
            self.__pydantic_fields_set__.discard('context_output_reserve')
        if not self.context_window_tokens:
            self.context_window_tokens = capability['context_window_tokens']
        if not self.model_max_output_tokens:
            self.model_max_output_tokens = capability['max_output_tokens']
        if capability['profile'] != 'legacy_or_unknown' and self.context_window_tokens > capability['context_window_tokens']:
            raise ValueError('context_window_tokens exceeds known model capability')
        if self.context_window_tokens < 4096:
            raise ValueError('context_window_tokens must be >=4096')
        if capability['profile'] != 'legacy_or_unknown' and self.model_max_output_tokens > capability['max_output_tokens']:
            raise ValueError('model_max_output_tokens exceeds known model capability')
        if self.context_output_reserve+self.context_safety_tokens >= self.context_window_tokens:
            raise ValueError('context output and safety reserves consume the model window')
        if self.context_output_reserve > self.model_max_output_tokens:
            raise ValueError('context_output_reserve exceeds model capability')
        return self

    @classmethod
    def from_env(cls, project_root: str | Path | None = None) -> "Settings":
        """输入可选项目根目录，输出合并后的 Settings；只读取配置，不探测外部接口。"""
        # 源码在 backend/research_loop；统一从仓库根读取 .env，手工入口与受管入口一致。
        root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        _load_dotenv(root / ".env")
        data_dir = Path(os.getenv("RESEARCHLOOP_DATA_DIR") or root / "data" / "ag1").resolve()
        output_dir = Path(os.getenv("RESEARCHLOOP_OUTPUT_DIR") or data_dir / "notes").resolve()
        config_path = paper_sources_path(root)
        sources_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        override = {k: os.environ[e] for k, e in (("url", "SUPABASE_URL"), ("anon_key", "SUPABASE_ANON_KEY"), ("schema", "SUPABASE_SCHEMA")) if os.getenv(e)}
        if override:
            sources_config["supabase_shared"] = {**(sources_config.get("supabase_shared") or sources_config.get("supabase") or {}), **override}
            sources_config["supabase"] = {**(sources_config.get("supabase") or {}), **override}
            for backend in (sources_config.get("source_backends") or {}).values():
                backend.update(override)
        if os.getenv("SUPABASE_BM25_RPC"):
            sources_config.setdefault("source_backends", {}).setdefault("arxiv", {})["bm25_rpc"] = os.environ["SUPABASE_BM25_RPC"]
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
            discovery_backends=resolve_backends(sources_config),
            task_token_limit=int(os.getenv("RESEARCHLOOP_TASK_TOKEN_LIMIT", "1000000")),
            task_token_tail=int(os.getenv("RESEARCHLOOP_TASK_TOKEN_TAIL", "32768")),
            max_reflection_rounds=int(os.getenv("RESEARCHLOOP_REFLECTION_ROUNDS", "2")),
            context_window_tokens=int(os.getenv("RESEARCHLOOP_CONTEXT_WINDOW", "0")),
            context_output_reserve=int(os.environ["RESEARCHLOOP_CONTEXT_OUTPUT_RESERVE"]) if "RESEARCHLOOP_CONTEXT_OUTPUT_RESERVE" in os.environ else None,
            context_safety_tokens=int(os.getenv("RESEARCHLOOP_CONTEXT_SAFETY_TOKENS", "4096")),
            context_deduplicate_read_paper=os.getenv("RESEARCHLOOP_CONTEXT_DEDUPLICATE_READ_PAPER", "0") == "1",
            token_counting_tokenizer_path=Path(os.environ["RESEARCHLOOP_TOKENIZER_PATH"]).expanduser() if os.getenv("RESEARCHLOOP_TOKENIZER_PATH") else None,
            token_counting_tokenizer_model=os.getenv("RESEARCHLOOP_TOKENIZER_MODEL") or None,
            token_counting_tokenizer_sha256=os.getenv("RESEARCHLOOP_TOKENIZER_SHA256") or None,
            researcher_concurrency=int(os.getenv("RESEARCHLOOP_RESEARCHER_CONCURRENCY", "2")),
            execution_budget_mode=os.getenv("RESEARCHLOOP_EXECUTION_BUDGET_MODE", "enforce"),
            max_model_requests=int(os.getenv("RESEARCHLOOP_MAX_MODEL_REQUESTS", "48")),
            max_tool_actions=int(os.getenv("RESEARCHLOOP_MAX_TOOL_ACTIONS", "80")),
            reserve_model_requests=int(os.getenv("RESEARCHLOOP_RESERVE_MODEL_REQUESTS", "10")),
            web_search_enabled=os.getenv("RESEARCHLOOP_WEB_SEARCH_ENABLED", "0") == "1",
            web_fetch_enabled=os.getenv("RESEARCHLOOP_WEB_FETCH_ENABLED", "0") == "1",
            web_tools_timeout_seconds=float(os.getenv("RESEARCHLOOP_WEB_TOOLS_TIMEOUT_SECONDS", "30")),
            web_tools_max_response_bytes=int(os.getenv("RESEARCHLOOP_WEB_TOOLS_MAX_RESPONSE_BYTES", "2097152")),
            memory_history_tokens=int(os.getenv("RESEARCHLOOP_HISTORY_TOKENS", "12000")),
            retrieval_enabled=os.getenv("RESEARCHLOOP_RETRIEVAL_ENABLED", "1") == "1",
            retrieval_route=os.getenv("RESEARCHLOOP_RETRIEVAL_ROUTE", "located_fp16"),
            retrieval_reranker_backend=os.getenv("RESEARCHLOOP_RERANKER_BACKEND", "mps_fp16"),
            retrieval_model_manifest=Path(os.environ["RESEARCHLOOP_RETRIEVAL_MODEL_MANIFEST"]) if os.getenv("RESEARCHLOOP_RETRIEVAL_MODEL_MANIFEST") else None,
            retrieval_namespace=os.getenv("RESEARCHLOOP_RETRIEVAL_NAMESPACE", "researchloop_evidence_v1"),
            qdrant_url=os.getenv("RESEARCHLOOP_QDRANT_URL", "http://127.0.0.1:6339"),
            openalex_api_key=os.getenv("OPENALEX_API_KEY") or None,
        )


def get_settings() -> Settings:
    return Settings.from_env()
