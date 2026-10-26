"""在冻结纯文件边界内同期比较BM25、真实查询向量与定位FP16完整检索。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

from research_loop.domain.lexical_retrieval import LexicalRetriever
from research_loop.providers.embeddings import LocalEmbedding
from scripts.compare_reranker_mps import (
    MODEL_ID, MODEL_REVISION, build_engine, new_counters, validate_runtime,
)
from scripts.compare_reranker_threads import (
    VerifiedSession, check_reference, validate_case, validate_feed_projection,
)
from scripts.prepare_reranker_thread_comparison import SOURCE_FIELDS, canonical_digest
from scripts.retrieval_experiment_guard import BoundaryViolation, write_json_new


ROUTES = ("bm25", "vector", "located_fp16")
MEASUREMENT = "same_run_full_file_retrieval_with_actual_feed_verification"
SETTINGS = {"routes": list(ROUTES), "hot_repeats": 3, "candidate_limit": 24,
            "embedding_threads": 2, "reranker_threads": 4, "batch_size": 4,
            "sort_by_length": False, "max_pair_tokens": 512, "max_seconds": 3600}
AUTHORIZATION = {"local_reranker": True, "local_embedding_query": True,
                 "document_embedding": False, "external_api": False,
                 "database": False, "network": False, "deployment": False,
                 "new_assets": False, "mps": True}


def references(frozen):
    """标签仅核字节，查询执行器不解析报告材料。"""
    return (list(frozen["inputs"].values()) + list(frozen["report_inputs"].values())
            + frozen["source_files"] + frozen["protected_files"] + frozen["model_files"]
            + frozen.get("historical_reference_files", []))


def build_plan(queries, feeds):
    """固定全部计划格与本地模型上限，查询顺序不按已有结果筛选。"""
    ids = [q["case_id"] for q in queries]
    if len(ids) != 114 or len(set(ids)) != 114 or set(ids) != set(feeds):
        raise BoundaryViolation("THREEWAY_CASE_SET_CHANGED")
    cells = [{"phase": "warmup", "case_id": ids[0], "route": route,
              "state": "not_executed"} for route in ROUTES]
    for rep in range(3):
        for ordinal, case_id in enumerate(ids):
            offset = (rep + ordinal) % 3
            order = ROUTES[offset:] + ROUTES[:offset]
            cells.extend({"phase": "hot", "repetition": rep, "ordinal": ordinal,
                          "case_id": case_id, "route": route, "state": "not_executed"}
                         for route in order)
    model_cells = [c for c in cells if c["route"] == "located_fp16"]
    budget = {"max_query_calls": 1029, "max_score_attempts": 343,
              "max_pairs": sum(feeds[c["case_id"]]["pairs"] for c in model_cells),
              "max_native_batches": sum(len(feeds[c["case_id"]]["batches"]) for c in model_cells),
              "max_embedding_queries": 343, "max_embedding_native_batches": 343,
              "max_seconds": 3600}
    if budget["max_pairs"] != 8067 or budget["max_native_batches"] != 2025:
        raise BoundaryViolation("THREEWAY_NATIVE_BUDGET_CHANGED")
    return {"routes": list(ROUTES), "case_ids": ids, "hot_repeats": 3,
            "budget": budget, "cells": cells}


def project_selection(question, selected):
    """把实时定位结果投影为原候选合同，防止偷偷复用预存检索结果。"""
    result = {"case_id": question["case_id"], "query": question["query"],
              "scope_document_ids": list(question["scope_document_ids"]),
              "candidates": [{k: row[k] for k in SOURCE_FIELDS}
                             for row in selected["candidates"]]}
    return {**result, "input_sha256": canonical_digest(result)}


class EmbeddingSession(VerifiedSession):
    """复用实际整数feed核验，单独限制查询编码次数，不重复编码文档。"""

    def __init__(self, native, counters, budget, started):
        super().__init__(native, expected_threads=2)
        self.counters, self.budget, self.started = counters, budget, started

    def get_providers(self):
        return self.native.get_providers()

    def get_session_options(self):
        return self.native.get_session_options()

    def run(self, output_names, feed):
        if (self.counters["embedding_native_reservations"] >= self.budget["max_embedding_native_batches"]
                or time.perf_counter() - self.started > self.budget["max_seconds"]):
            raise BoundaryViolation("THREEWAY_EMBEDDING_BUDGET_EXCEEDED")
        self.counters["embedding_native_reservations"] += 1
        attempts, successes = self.run_attempts, self.run_successes
        try:
            return super().run(output_names, feed)
        finally:
            self.counters["embedding_native_attempts"] += self.run_attempts - attempts
            self.counters["embedding_native_successes"] += self.run_successes - successes


class CellStopped(BoundaryViolation):
    """携带已经开始的格，失败时先留原件再终止剩余计划。"""

    def __init__(self, record, cause):
        self.record, self.cause = record, cause
        super().__init__(str(cause))


def run_one(question, cell, lexical, vector, engine, case, feed, embedding_feed,
            counters, budget, started):
    """主墙钟覆盖实时召回、编码/重排、校验与统一全文结果装配。"""
    row = {**cell, "status": "stopped", "error": None, "query": question["query"],
           "scope_document_ids": question["scope_document_ids"], "candidate_ids": [],
           "ranked_ids": [], "hits": [], "scores": None, "selection_trace": {},
           "measurement": MEASUREMENT, "host_load_average": list(os.getloadavg())}
    route, fatal, score_attempted = cell["route"], None, False
    selection_seconds = reranking_seconds = embedding_seconds = vector_seconds = 0.0
    materialization_seconds = 0.0
    counters["query_calls"] += 1
    beginning = time.perf_counter()
    try:
        if route == "located_fp16":
            engine.session.reset(feed)
            engine.last_stats = {}
        elif route == "vector":
            vector.embedding.session.reset(embedding_feed)
            vector.last_stats = {}
        if (route not in ROUTES or counters["query_calls"] > budget["max_query_calls"]
                or time.perf_counter() - started > budget["max_seconds"]):
            raise BoundaryViolation("THREEWAY_QUERY_BUDGET_EXCEEDED")
        if set(question) != {"case_id", "query", "scope_document_ids"}:
            raise BoundaryViolation("THREEWAY_QUERY_PROJECTION_CHANGED")
        if route == "vector":
            if counters["embedding_queries"] >= budget["max_embedding_queries"]:
                raise BoundaryViolation("THREEWAY_EMBEDDING_QUERY_LIMIT")
            counters["embedding_queries"] += 1
            selected = vector.select(question["query"], question["scope_document_ids"])
            stats = selected["trace"]
            embedding_seconds = stats["embedding_seconds"]
            vector_seconds = stats["vector_search_seconds"]
            if not vector.embedding.session.snapshot()["complete"]:
                raise BoundaryViolation("THREEWAY_EMBEDDING_ACTUAL_FEED_INCOMPLETE")
            candidates = selected["candidates"]
            scores = [r["vector_score"] for r in candidates]
            order = list(range(len(candidates)))
        else:
            selected = lexical.select(question["query"], question["scope_document_ids"],
                                      "bm25_rerank" if route == "bm25" else "located_bm25_rerank")
            candidates = selected["candidates"]
            scores = [float(r["score"]) for r in candidates] if route == "bm25" else []
            order = list(range(len(candidates)))
        selected_at = time.perf_counter()
        selection_seconds = selected_at - beginning
        if route == "located_fp16":
            actual = project_selection(question, selected)
            validate_case(actual, allow_empty=True)
            if actual != case:
                raise BoundaryViolation("THREEWAY_LIVE_CANDIDATES_CHANGED")
            validate_feed_projection(case, feed)
            if (counters["score_attempts"] >= budget["max_score_attempts"]
                    or counters["pair_attempts"] + len(candidates) > budget["max_pairs"]):
                raise BoundaryViolation("THREEWAY_RERANK_BUDGET_EXCEEDED")
            if candidates:
                counters["score_attempts"] += 1
                counters["pair_attempts"] += len(candidates)
                score_attempted = True
                scores = engine.score(question["query"], [r["text"] for r in candidates])
                if (len(scores) != len(candidates) or any(type(s) not in (int, float)
                                                        or not math.isfinite(s) for s in scores)
                        or not engine.session.snapshot()["complete"]):
                    raise BoundaryViolation("THREEWAY_MPS_RESULT_INVALID")
                order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                counters["successful_score_calls"] += 1
                counters["successful_pairs"] += len(scores)
            reranking_seconds = time.perf_counter() - selected_at
        output_at = time.perf_counter()
        if (len(candidates) > 24 or len({r["chunk_id"] for r in candidates}) != len(candidates)
                or any(r["document_id"] not in question["scope_document_ids"] for r in candidates)):
            raise BoundaryViolation("THREEWAY_OUTPUT_SCOPE_OR_IDS_CHANGED")
        hits = []
        for rank, index in enumerate(order, 1):
            candidate = candidates[index]
            if hashlib.sha256(candidate["text"].encode()).hexdigest() != candidate["text_sha256"]:
                raise BoundaryViolation("THREEWAY_OUTPUT_TEXT_CHANGED")
            hits.append({k: candidate[k] for k in ("chunk_id", "document_id", "text_sha256", "text")}
                        | {"rank": rank, "candidate_rank": index + 1,
                           "score": float(scores[index]) if scores is not None else None})
        row.update(candidate_ids=[r["chunk_id"] for r in candidates],
                   ranked_ids=[r["chunk_id"] for r in hits], hits=hits,
                   scores=[float(s) for s in scores] if scores is not None else None,
                   status="executed" if candidates else "empty")
        materialization_seconds = time.perf_counter() - output_at
    except BaseException as exc:
        row["error"] = {"type": type(exc).__name__, "reason": str(exc)}
        if route == "vector":
            stats = getattr(vector, "last_stats", {})
            embedding_seconds = stats.get("embedding_seconds", 0.0)
            vector_seconds = stats.get("vector_search_seconds", 0.0)
        if score_attempted:
            counters["failed_score_calls"] += 1
        fatal = exc
    finished = time.perf_counter()
    row["timing"] = {"wall_seconds": finished - beginning,
                     "selection_seconds": selection_seconds, "reranking_seconds": reranking_seconds,
                     "embedding_seconds": embedding_seconds, "vector_search_seconds": vector_seconds,
                     "materialization_seconds": materialization_seconds}
    if route == "located_fp16":
        row["feed_verification"] = engine.session.snapshot()
        for key in ("session_run_attempts", "session_run_successes", "verified_batches"):
            counters[key] += row["feed_verification"][key]
        row["input_sha256"] = case["input_sha256"]
        row["model_identity"] = deepcopy(engine.identity)
        row["local_reranker_stats"] = deepcopy(engine.last_stats)
    elif route == "vector":
        row["vector_proof"] = vector.embedding.session.snapshot()
        row["model_identity"] = deepcopy(vector.embedding.identity)
        row["vector_adapter_stats"] = deepcopy(vector.last_stats)
    if "selected" in locals():
        row["selection_trace"] = {k: v for k, v in selected.items() if k != "candidates"}
    row["state"], row["counters_after"] = row["status"], deepcopy(counters)
    if fatal is not None:
        raise CellStopped(row, fatal)
    return row


def load_frozen(frozen):
    """模型导入前核对合同、材料、依赖与原候选张量。"""
    if (frozen.get("schema") != "threeway-retrieval-execution-v1"
            or frozen.get("settings") != SETTINGS or frozen.get("authorization") != AUTHORIZATION):
        raise BoundaryViolation("THREEWAY_FROZEN_CONTRACT_CHANGED")
    for ref in references(frozen):
        check_reference(ref)
    validate_runtime(frozen["runtime"])
    for name, version in frozen["packages"].items():
        if importlib.metadata.version(name) != version:
            raise BoundaryViolation("THREEWAY_PACKAGE_VERSION_CHANGED")
    inputs = {k: json.loads(Path(v["path"]).read_text()) for k, v in frozen["inputs"].items()}
    model_refs = {r["path"]: r["sha256"] for r in frozen["model_files"]}
    reranker, floating = inputs["model_manifest"]["reranker"], inputs["float_manifest"]
    if any(spec.get("repo") != MODEL_ID or spec.get("revision") != MODEL_REVISION
           for spec in (reranker, floating)):
        raise BoundaryViolation("THREEWAY_RERANKER_CHECKPOINT_CHANGED")
    for spec, names in ((reranker, (reranker["weights"], "tokenizer.json", "config.json")),
                        (inputs["model_manifest"]["embedding"],
                         (inputs["model_manifest"]["embedding"]["weights"], "tokenizer.json", "config.json")),
                        (floating, ("model.safetensors", "config.json"))):
        for name in names:
            path = str(Path(spec["directory"]) / name)
            if model_refs.get(path) != spec["files"][name]["sha256"]:
                raise BoundaryViolation("THREEWAY_MODEL_ASSET_COVERAGE_CHANGED")
    if floating["files"]["config.json"]["sha256"] != reranker["files"]["config.json"]["sha256"]:
        raise BoundaryViolation("THREEWAY_FLOAT_CONFIGURATION_CHANGED")
    if (len(inputs["chunks"]) != 3821 or len(inputs["catalog"]) != 54
            or len(inputs["cases"]) != 114 or len(inputs["queries"]) != 114):
        raise BoundaryViolation("THREEWAY_CORPUS_COUNTS_CHANGED")
    cases = {c["case_id"]: c for c in inputs["cases"]}
    feeds = {f["case_id"]: f for f in inputs["feed_manifest"]["cases"]}
    if len(cases) != 114 or len(feeds) != 114:
        raise BoundaryViolation("THREEWAY_DUPLICATED_CASES")
    for q in inputs["queries"]:
        c = cases[q["case_id"]]
        validate_case(c, allow_empty=True)
        validate_feed_projection(c, feeds[q["case_id"]])
        if c["query"] != q["query"] or c["scope_document_ids"] != q["scope_document_ids"]:
            raise BoundaryViolation("THREEWAY_QUERY_OR_SCOPE_CHANGED")
    plan = build_plan(inputs["queries"], feeds)
    if frozen["budget"] != plan["budget"]:
        raise BoundaryViolation("THREEWAY_FROZEN_BUDGET_CHANGED")
    return inputs, cases, feeds, plan


def execute(freeze_path, expected_sha256, output, *, prepare_only=False):
    """独占输出、关闭外部连接，并用完整计划记录失败和未执行格。"""
    freeze_path, output = freeze_path.resolve(), output.resolve()
    if hashlib.sha256(freeze_path.read_bytes()).hexdigest() != expected_sha256:
        raise BoundaryViolation("THREEWAY_FREEZE_ANCHOR_CHANGED")
    frozen = json.loads(freeze_path.read_text())
    artifact_root = Path(frozen["runtime"]["venv_prefix"]).parent.parent
    if (not output.is_relative_to(artifact_root) or output == artifact_root
            or any(not Path(p).is_relative_to(artifact_root / "runtime")
                   for p in frozen["runtime"]["mutable_roots"])):
        raise BoundaryViolation("THREEWAY_OUTPUT_OR_MUTABLE_ROOT_OUTSIDE_RUN")
    from scripts.mps_experiment_guard import install_guards
    reads = [Path(r["path"]) for r in references(frozen)]
    reads.extend([freeze_path, Path(__file__).resolve().parents[1]])
    counts = install_guards(output, reads, [Path(p) for p in frozen["runtime"]["mutable_roots"]])
    started, counters, executed = time.perf_counter(), new_counters(), []
    counters.update(dict.fromkeys(("query_calls", "embedding_queries", "embedding_native_attempts",
                                   "embedding_native_successes", "embedding_native_reservations"), 0))
    status, error, plan, initializations = "complete", None, None, []
    load_begin = list(os.getloadavg())
    try:
        inputs, cases, feeds, plan = load_frozen(frozen)
        lexical = LexicalRetriever(inputs["chunks"], inputs["catalog"])
        for q in inputs["queries"]:
            actual = project_selection(q, lexical.select(q["query"], q["scope_document_ids"],
                                                        "located_bm25_rerank"))
            if actual != cases[q["case_id"]]:
                raise BoundaryViolation("THREEWAY_PREFLIGHT_CANDIDATES_CHANGED")
        write_json_new(output / "execution-plan.json", {"freeze_sha256": expected_sha256,
            "measurement": MEASUREMENT, "plan": plan, "labels_parsed": False,
            "timing_boundary": "request query/scope through live selection/encoding/reranking, actual-feed checks and full-text final hit materialization; model/index load and evidence writes excluded"})
        if prepare_only:
            if "torch" in sys.modules or "onnxruntime" in sys.modules:
                raise BoundaryViolation("THREEWAY_PREFLIGHT_IMPORTED_MODEL_RUNTIME")
            write_json_new(output / "preflight.json", {"candidate_cases_checked": 114,
                "model_loaded": False, "query_calls": 0, "labels_parsed": False})
        else:
            import torch
            import onnxruntime as ort
            from scripts.threeway_vector_adapter import FrozenVectorRetriever
            if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
                raise BoundaryViolation("THREEWAY_MPS_UNAVAILABLE")
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
                raise BoundaryViolation("THREEWAY_TORCH_ACTUAL_THREADS_CHANGED")
            at = time.perf_counter()
            engine = build_engine("mps_fp16", frozen, inputs, counters, plan["budget"], started, torch)
            initializations.append({"route": "located_fp16", "seconds": time.perf_counter() - at,
                                    "model_identity": engine.identity, "session_identity": engine.session.identity})
            at = time.perf_counter()
            embedding = LocalEmbedding(frozen["inputs"]["model_manifest"]["path"], threads=2)
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
            native = ort.InferenceSession(str(embedding.directory / embedding.spec["weights"]),
                                          sess_options=options, providers=["CPUExecutionProvider"])
            embedding.session = EmbeddingSession(native, counters, plan["budget"], started)
            vector = FrozenVectorRetriever(inputs["chunks"], inputs["vectors"], embedding)
            initializations.append({"route": "vector", "seconds": time.perf_counter() - at,
                                    "model_identity": embedding.identity,
                                    "session_identity": embedding.session.identity,
                                    "vector_search_identity": vector.identity})
            write_json_new(output / "initializations.json", initializations)
            by_id = {q["case_id"]: q for q in inputs["queries"]}
            for index, cell in enumerate(plan["cells"]):
                cid, fatal = cell["case_id"], None
                try:
                    row = run_one(by_id[cid], cell, lexical, vector, engine, cases[cid], feeds[cid],
                                  inputs["embedding_feeds"][cid], counters, plan["budget"], started)
                except CellStopped as exc:
                    row, fatal = exc.record, exc.cause
                write_json_new(output / f"cell-{index:04d}-{cell['phase']}-{cell['route']}.json", row)
                executed.append(index)
                counters["warmup_rows" if cell["phase"] == "warmup" else "hot_rows"] += 1
                if fatal is not None:
                    raise fatal
                if time.perf_counter() - started > plan["budget"]["max_seconds"]:
                    raise BoundaryViolation("THREEWAY_TIME_LIMIT_AFTER_CALL")
                if counters["hot_rows"] and counters["hot_rows"] % 30 == 0:
                    print(json.dumps({"hot_rows": counters["hot_rows"], "query_calls": counters["query_calls"],
                                      "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
            expected = {"query_calls": 1029, "embedding_queries": 343,
                        "embedding_native_attempts": 343, "embedding_native_successes": 343,
                        "embedding_native_reservations": 343,
                        "score_attempts": 343, "successful_score_calls": 343,
                        "pair_attempts": 8067, "successful_pairs": 8067,
                        "native_batch_reservations": 2025, "session_run_attempts": 2025,
                        "session_run_successes": 2025, "verified_batches": 2025,
                        "failed_score_calls": 0}
            if any(counters[k] != v for k, v in expected.items()):
                raise BoundaryViolation("THREEWAY_FINAL_BUDGET_COUNTS_CHANGED")
    except BaseException as exc:
        status, error = "stopped", {"type": type(exc).__name__, "reason": str(exc)}
    terminal = {"status": status, "prepare_only": prepare_only, "error": error,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "freeze_sha256": expected_sha256, "counters": counters,
                "guard_counts": counts, "initializations": initializations,
                "unexecuted": [{**cell, "index": i} for i, cell in enumerate(plan["cells"])
                               if i not in set(executed)] if plan else None,
                "elapsed_seconds": time.perf_counter() - started,
                "host_load_average_begin": load_begin, "host_load_average_end": list(os.getloadavg()),
                "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "measurement": MEASUREMENT, "labels_parsed": False}
    write_json_new(output / "terminal.json", terminal)
    print(json.dumps({k: v for k, v in terminal.items() if k != "unexecuted"}), flush=True)
    return terminal


def main():
    """CLI必须显式传入事前冻结hash及全新输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    result = execute(args.freeze, args.expected_freeze_sha256, args.output, prepare_only=args.prepare_only)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
