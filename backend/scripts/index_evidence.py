"""显式构建授权原文索引：只读既有页，不下载论文、不调用生成模型或启动Agent。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_loop.config import Settings
from research_loop.providers.embeddings import LocalEmbedding
from research_loop.services.indexing import IndexingService
from research_loop.storage.mysql import Database
from research_loop.storage.vector_index import VectorIndex


def main() -> None:
    """命令行参数→已迁移数据库/本机模型→构建状态；partial返回退出码2。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--evidence-id", action="append", required=True)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6339")
    parser.add_argument("--namespace", default="researchloop_original_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env(Path(__file__).resolve().parents[2])
    vector = VectorIndex(args.qdrant_url, args.namespace)
    try:
        result = IndexingService(Database(settings.database_url), LocalEmbedding(args.model_manifest), vector).build(
            args.task_id, args.document_id, args.evidence_id)
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # 证据输出也不静默覆盖调用者已有报告。
            with args.output.open("x") as handle:
                handle.write(payload + "\n")
        print(payload)
        raise SystemExit(0 if result["vector_status"] == "ready" else 2)
    finally:
        vector.close()


if __name__ == "__main__":
    main()
