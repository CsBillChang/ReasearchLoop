"""显式冻结、纯文件与本地重排的两路线对照；不加载业务配置。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import sys
import time

from research_loop.domain.lexical_retrieval import LexicalRetriever
from research_loop.providers.reranker import LocalReranker
from scripts.retrieval_experiment_guard import (
    BoundaryViolation, install_guards, write_json_new,
)


ROUTES = ("bm25_rerank", "located_bm25_rerank")


def digest(path: Path) -> str:
    """只读核对输入或源码字节。"""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_reference(ref: dict):
    """读取必须通过冻结hash的JSON文件。"""
    path = Path(ref["path"])
    if not path.is_absolute() or digest(path) != ref["sha256"]:
        raise BoundaryViolation("FROZEN_INPUT_CHANGED")
    return json.loads(path.read_text())


def load_frozen(path: Path) -> tuple[dict, list, list, list]:
    """校验执行许可、程序、材料及运行时，标签仅核hash而不交给检索器。"""
    frozen = json.loads(path.read_text())
    expected = {"routes": list(ROUTES), "candidate_limit": 24, "hot_repeats": 3,
                "threads": 2, "batch_size": 4, "sort_by_length": False,
                "max_score_attempts": 686, "max_pairs": 16464, "max_seconds": 3600}
    if frozen.get("settings") != expected or frozen.get("authorization") != {
        "local_reranker": True, "external_api": False, "embedding": False,
        "deployment": False, "new_assets": False,
    }:
        raise BoundaryViolation("EXPERIMENT_CONTRACT_CHANGED")
    for ref in frozen["source_files"] + frozen["protected_input_files"]:
        if digest(Path(ref["path"])) != ref["sha256"]:
            raise BoundaryViolation("FROZEN_FILE_CHANGED")
    for package, version in frozen["packages"].items():
        if importlib.metadata.version(package) != version:
            raise BoundaryViolation("RUNTIME_VERSION_CHANGED")
    inputs = frozen["inputs"]
    chunks, catalog, questions = [read_reference(inputs[name]) for name in
                                   ("chunks", "catalog", "queries")]
    if (len(chunks), len(catalog), len(questions)) != (3821, 54, 114):
        raise BoundaryViolation("MATERIAL_COUNTS_CHANGED")
    if len({q["case_id"] for q in questions}) != 114:
        raise BoundaryViolation("DUPLICATE_QUERY_ID")
    for q in questions:
        if set(q) != {"case_id", "query", "scope_document_ids"}:
            raise BoundaryViolation("QUERY_PROJECTION_CONTAINS_EXTRA_FIELDS")
    return frozen, chunks, catalog, questions


def run_one(retriever, reranker, question: dict, route: str, counters: dict) -> dict:
    """计量一次同范围候选选择与真实评分，失败时保留词法候选与原因。"""
    started = time.perf_counter()
    if route not in ROUTES:
        raise BoundaryViolation("EXPERIMENT_ROUTE_INVALID")
    selected = retriever.select(question["query"], question["scope_document_ids"], route)
    candidates = selected["candidates"]
    before_model = time.perf_counter()
    if (not isinstance(candidates, list) or len(candidates) > 24
            or any(not isinstance(r, dict) or not isinstance(r.get("chunk_id"), str)
                   or not r["chunk_id"] for r in candidates)
            or len({r["chunk_id"] for r in candidates}) != len(candidates)):
        raise BoundaryViolation("CANDIDATE_CONTRACT_INVALID")
    for row in candidates:
        if (not isinstance(row.get("chunk_id"), str) or not row["chunk_id"]
                or row.get("document_id") not in question["scope_document_ids"]
                or not isinstance(row.get("text"), str)
                or hashlib.sha256(row["text"].encode()).hexdigest() != row.get("text_sha256")):
            raise BoundaryViolation("CANDIDATE_IDENTITY_OR_SCOPE_INVALID")
    status, error, scores, stats = "empty" if not candidates else "executed", None, [], {}
    if candidates:
        counters["score_attempts"] += 1
        counters["pair_attempts"] += len(candidates)
        if counters["score_attempts"] > 686 or counters["pair_attempts"] > 16464:
            raise BoundaryViolation("EXPERIMENT_BUDGET_EXCEEDED")
        # 清空本实例的上次诊断，容量失败不能沿用上一题的成功统计。
        reranker.last_stats = {}
        try:
            scores = reranker.score(question["query"], [r["text"] for r in candidates])
            if (len(scores) != len(candidates) or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                        not math.isfinite(v) for v in scores)):
                raise ValueError("RERANK_OUTPUT_INVALID")
            stats = deepcopy(reranker.last_stats)
            counters["successful_score_calls"] += 1
            counters["successful_pairs"] += len(scores)
        except (MemoryError, SystemError, OSError):
            raise
        except Exception as exc:
            if str(exc).startswith(("RERANK_MODEL_", "RERANK_CONFIG_")):
                raise BoundaryViolation("MODEL_IDENTITY_OR_CONFIGURATION_CHANGED") from exc
            status, error, scores = "degraded", {"type": type(exc).__name__, "reason": str(exc)}, []
            counters["degraded"] += 1
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i)) if scores else list(range(len(candidates)))
    hits = [{"chunk_id": candidates[i]["chunk_id"], "document_id": candidates[i]["document_id"],
             "text_sha256": candidates[i]["text_sha256"], "rank": rank,
             "candidate_rank": i + 1, "reranker_score": scores[i] if scores else None}
            for rank, i in enumerate(order, 1)]
    finished = time.perf_counter()
    return {"requested_route": route, "effective_path": route if scores else "lexical_fallback",
            "status": status, "error": error, "query": question["query"],
            "scope_document_ids": question["scope_document_ids"],
            "candidate_ids": [r["chunk_id"] for r in candidates],
            "ranked_ids": [r["chunk_id"] for r in hits], "hits": hits,
            "selection_trace": {k: v for k, v in selected.items() if k != "candidates"},
            "local_reranker_stats": stats, "model_identity": reranker.identity,
            "timing": {"selection_seconds": before_model - started,
                       "reranking_seconds": finished - before_model,
                       "wall_seconds": finished - started},
            "usage": None, "measurement": "local_file_component_not_agent_end_to_end"}


def execute(freeze_path: Path, output: Path, *, prepare_only=False) -> dict:
    """仅在新目录运行一次冻结计划，不恢复或重跑旧的结果目录。"""
    freeze_path, output = freeze_path.resolve(), output.resolve()
    frozen = json.loads(freeze_path.read_text())
    roots = [Path(ref["path"]).parent for ref in
             list(frozen["inputs"].values()) + frozen["source_files"] + frozen["protected_input_files"]]
    roots.extend([Path(__file__).resolve().parents[1], freeze_path.parent])
    guard_counts = install_guards(output, roots)
    started = time.perf_counter()
    counters = {"score_attempts": 0, "pair_attempts": 0, "successful_score_calls": 0,
                "successful_pairs": 0, "degraded": 0, "hot_rows": 0, "warmup_rows": 0}
    error, final_status = None, "complete"
    try:
        frozen, chunks, catalog, questions = load_frozen(freeze_path)
        ready = time.perf_counter()
        retriever = LexicalRetriever(chunks, catalog)
        preparation_seconds = time.perf_counter() - ready
        engines = {route: LocalReranker(frozen["inputs"]["model_manifest"]["path"],
                                       threads=2, batch_size=4, sort_by_length=False) for route in ROUTES}
        write_json_new(output / "execution-plan.json", {
            "freeze_sha256": digest(freeze_path), "settings": frozen["settings"],
            "queries": questions, "preparation_seconds": preparation_seconds,
            "boundary": "All candidate selection uses query/scope/catalog only; qrels are not loaded.",
        })
        if prepare_only:
            counts = {}
            for q in questions:
                for route in ROUTES:
                    selected = retriever.select(q["query"], q["scope_document_ids"], route)
                    counts[q["case_id"] + ":" + route] = len(selected["candidates"])
            write_json_new(output / "preflight.json", {"candidate_counts": counts,
                "score_attempts": 0, "network_allowed": False, "retrieval_selection_checked": 228})
        else:
            for route in ROUTES:
                result = run_one(retriever, engines[route], questions[0], route, counters)
                write_json_new(output / ("warmup-" + route + ".json"), result)
                counters["warmup_rows"] += 1
            for repetition in range(3):
                for ordinal, q in enumerate(questions):
                    order = ROUTES if (repetition + ordinal) % 2 == 0 else tuple(reversed(ROUTES))
                    for route in order:
                        if time.perf_counter() - started > 3600:
                            raise BoundaryViolation("EXPERIMENT_TIME_LIMIT")
                        result = run_one(retriever, engines[route], q, route, counters)
                        wrapper = {"phase": "hot", "repetition": repetition,
                                   "case_id": q["case_id"], "route": route, "result": result}
                        name = f"hot-{repetition}-{ordinal:03d}-{route}.json"
                        write_json_new(output / name, wrapper)
                        counters["hot_rows"] += 1
                    if (ordinal + 1) % 10 == 0:
                        print(json.dumps({"repetition": repetition, "queries_done": ordinal + 1,
                                          "hot_rows": counters["hot_rows"],
                                          "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    except BaseException as exc:
        final_status, error = "stopped", {"type": type(exc).__name__, "reason": str(exc)}
    terminal = {"status": final_status, "prepare_only": prepare_only,
                "finished_at": datetime.now(timezone.utc).isoformat(), "error": error,
                "counters": counters, "guard_counts": guard_counts,
                "elapsed_seconds": time.perf_counter() - started,
                "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json_new(output / "terminal.json", terminal)
    print(json.dumps(terminal), flush=True)
    return terminal


def main() -> int:
    """CLI显式要求冻结合同和独立输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    return 0 if execute(args.freeze, args.output, prepare_only=args.prepare_only)["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
