"""从冻结原件复算三路完整检索的质量、同期墙钟及逐题取舍。

本模块只使用标准库，不导入检索器、模型运行时或业务配置。成功空结果
在原正例分母中记0；失败、未知和未执行保持未知，不用筛选后的均值代替全量。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


ROUTES = ("bm25", "vector", "located_fp16")
SUCCESS = {"executed", "empty"}
FAILURE = {"stopped", "failed", "unknown", "not_executed"}
METRICS = ("recall5", "recall8", "recall10", "mrr24", "candidate_recall24")
TIMINGS = ("wall_seconds", "selection_seconds", "reranking_seconds", "embedding_seconds",
           "vector_search_seconds", "materialization_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read(path):
    return json.loads(Path(path).read_text())


def frozen_references(frozen):
    """递归收集显式文件引用，不将目录、运行环境字符串扩大为读取授权。"""
    refs = []
    def visit(value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= set(value):
                refs.append(value)
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for key, value in frozen.items():
        if key in {"inputs", "report_inputs", "package_records"} or key.endswith("_files"):
            visit(value)
    return refs


def check_reference(ref):
    path = Path(ref["path"])
    require(path.is_absolute() and path == path.resolve() and path.is_file(), "REFERENCE_PATH_CHANGED")
    require(is_hash(ref.get("sha256")) and sha(path) == ref["sha256"], "REFERENCE_HASH_CHANGED:" + str(path))
    if "bytes" in ref:
        require(path.stat().st_size == ref["bytes"], "REFERENCE_SIZE_CHANGED:" + str(path))
    return path


def load_context(freeze, expected_sha):
    require(is_hash(expected_sha) and sha(freeze) == expected_sha, "THREEWAY_FREEZE_ANCHOR_CHANGED")
    frozen = read(freeze)
    require(frozen.get("schema") == "threeway-retrieval-execution-v1", "THREEWAY_FREEZE_SCHEMA_CHANGED")
    refs = frozen_references(frozen)
    seen = {}
    for ref in refs:
        check_reference(ref)
        require(ref["path"] not in seen or seen[ref["path"]] == ref["sha256"], "FROZEN_REFERENCE_CONFLICT")
        seen[ref["path"]] = ref["sha256"]
    inputs = frozen["inputs"]
    queries = read(inputs["queries"]["path"])
    chunks = read(inputs["chunks"]["path"])
    questions = read(frozen["report_inputs"]["original_questions"]["path"])
    labels = read(frozen["report_inputs"]["labels"]["path"])
    by_id = {q["case_id"]: q for q in queries}
    metadata = {q["case_id"]: q for q in questions}
    corpus = {c["chunk_id"]: c for c in chunks}
    gold = defaultdict(set)
    for row in labels["positive_pairs"]:
        gold[row["case_id"]].add(row["chunk_id"])
    require(len(queries) == len(by_id) == len(metadata) == len(questions) == 114
            and set(by_id) == set(metadata) and not set(gold) - set(by_id), "THREEWAY_CASE_SET_CHANGED")
    require(len(corpus) == len(chunks) and corpus, "THREEWAY_CORPUS_IDS_CHANGED")
    for chunk in chunks:
        require(isinstance(chunk["text"], str) and hashlib.sha256(chunk["text"].encode()).hexdigest()
                == chunk["text_sha256"], "THREEWAY_CORPUS_TEXT_CHANGED")
    for cid, query in by_id.items():
        require(query["query"] == metadata[cid]["query"]
                and query["scope_document_ids"] == metadata[cid]["scope_document_ids"]
                and metadata[cid]["answerable"] == bool(gold[cid]), "THREEWAY_LABEL_QUERY_IDENTITY_CHANGED")
        require(query["scope_document_ids"] and len(set(query["scope_document_ids"]))
                == len(query["scope_document_ids"]), "THREEWAY_QUERY_SCOPE_INVALID")
        for chunk_id in gold[cid]:
            require(chunk_id in corpus and corpus[chunk_id]["document_id"] in query["scope_document_ids"],
                    "THREEWAY_GOLD_OUTSIDE_SCOPE")
    positive = [cid for cid in by_id if gold[cid]]
    require(len(positive) == 112 and Counter(metadata[cid]["split"] for cid in positive)
            == {"regression": 58, "pending": 54}, "THREEWAY_LABEL_DENOMINATOR_CHANGED")
    cases = read(inputs["cases"]["path"]) if "cases" in inputs else []
    case_map = {c["case_id"]: c for c in cases}
    feeds = read(inputs["feed_manifest"]["path"])["cases"] if "feed_manifest" in inputs else []
    embedding_feeds = read(inputs["embedding_feeds"]["path"])
    require(isinstance(embedding_feeds, dict) and set(embedding_feeds) == set(by_id),
            "THREEWAY_EMBEDDING_FEED_CASE_SET_CHANGED")
    return {"frozen": frozen, "queries": queries, "by_id": by_id, "metadata": metadata,
            "gold": gold, "corpus": corpus, "cases": case_map,
            "feeds": {row["case_id"]: row for row in feeds}, "references": refs,
            "embedding_feeds": embedding_feeds, "unique_reference_count": len(seen)}


def expected_plan(context):
    frozen, queries = context["frozen"], context["queries"]
    cells = [{"phase": "warmup", "case_id": queries[0]["case_id"], "route": route,
              "state": "not_executed"} for route in ROUTES]
    for repetition in range(3):
        for ordinal, query in enumerate(queries):
            offset = (ordinal + repetition) % 3
            for route in ROUTES[offset:] + ROUTES[:offset]:
                cells.append({"phase": "hot", "case_id": query["case_id"], "route": route,
                              "repetition": repetition, "ordinal": ordinal, "state": "not_executed"})
    require(len(cells) == 1029, "THREEWAY_PLANNED_CELL_COUNT_CHANGED")
    return {"routes": list(ROUTES), "case_ids": [q["case_id"] for q in queries],
            "hot_repeats": 3, "budget": frozen["budget"], "cells": cells}


def key(row):
    return row["phase"], row.get("repetition"), row["case_id"], row["route"]


def validate_mps_snapshot(row, context):
    """核保存的实际输入和MPS完成证明，历史feed只在候选原序相同时对照。"""
    snapshot = row.get("feed_verification")
    require(isinstance(snapshot, dict) and snapshot.get("complete") is True, "THREEWAY_MPS_FEED_MISSING")
    batches = snapshot["batches"]
    count = (len(row["candidate_ids"]) + 3) // 4
    require(len(batches) == snapshot["expected_batches"] == snapshot["verified_batches"]
            == snapshot["session_run_attempts"] == snapshot["session_run_successes"] == count,
            "THREEWAY_MPS_BATCH_COUNT_CHANGED")
    cid = row["case_id"]
    old_case, old_feed = context["cases"].get(cid), context["feeds"].get(cid)
    historical_match = old_case is not None and [c["chunk_id"] for c in old_case["candidates"]] == row["candidate_ids"]
    for ordinal, batch in enumerate(batches):
        indices = list(range(ordinal * 4, min((ordinal + 1) * 4, len(row["candidate_ids"]))))
        require(batch["batch_index"] == ordinal and batch["candidate_indices"] == indices
                and batch.get("verification_passed") is True and batch.get("native_attempted") is True
                and batch.get("native_succeeded") is True, "THREEWAY_MPS_BATCH_PROOF_CHANGED")
        tensors = batch["actual_tensors"]
        require(set(tensors) == {"input_ids", "attention_mask"}, "THREEWAY_MPS_INPUT_NAMES_CHANGED")
        for ref in tensors.values():
            require(ref["dtype"] == "int64" and ref["byte_order"] == "little"
                    and ref["shape"][0] == len(indices) and 1 <= ref["shape"][1] <= 512
                    and is_hash(ref["sha256"]), "THREEWAY_MPS_INPUT_TENSOR_INVALID")
        if historical_match and old_feed is not None:
            require(tensors == {name: old_feed["batches"][ordinal]["tensors"][name] for name in tensors},
                    "THREEWAY_MPS_HISTORICAL_FEED_CHANGED")
        proof = batch["device_proof"]
        require(proof.get("sync_before") is True and proof.get("sync_after") is True
                and proof.get("transferred_tensor_hashes") == tensors
                and proof.get("output_dtype") == "torch.float16"
                and str(proof.get("output_device", "")).startswith("mps")
                and set(proof.get("input_dtypes", {}).values()) == {"torch.int64"}
                and len(proof.get("input_devices", {})) == 2
                and all(str(v).startswith("mps") for v in proof["input_devices"].values())
                and proof.get("model_parameter_dtypes") == ["torch.float16"]
                and proof.get("model_parameter_devices") and all(str(v).startswith("mps")
                    for v in proof["model_parameter_devices"]), "THREEWAY_MPS_DEVICE_OR_SYNC_CHANGED")
    return {"batches": count, "historical_candidate_input_match": historical_match}


def validate_row(row, cell, context):
    require(all(row.get(k) == value for k, value in cell.items() if k != "state"), "THREEWAY_RAW_PLAN_IDENTITY_CHANGED")
    query = context["by_id"][row["case_id"]]
    require(row.get("query") == query["query"] and row.get("scope_document_ids") == query["scope_document_ids"],
            "THREEWAY_RAW_QUERY_OR_SCOPE_CHANGED")
    require(row.get("status") in SUCCESS | FAILURE, "THREEWAY_RAW_STATUS_INVALID")
    candidates = row.get("candidate_ids")
    require(isinstance(candidates, list) and len(candidates) <= 24 and len(candidates) == len(set(candidates)),
            "THREEWAY_RAW_CANDIDATES_INVALID")
    for cid in candidates:
        require(cid in context["corpus"] and context["corpus"][cid]["document_id"] in query["scope_document_ids"],
                "THREEWAY_RAW_CANDIDATE_OUTSIDE_SCOPE")
    if row["status"] not in SUCCESS:
        return {"successful": False, "mps_batches": 0}
    require(row.get("error") is None, "THREEWAY_SUCCESS_WITH_ERROR")
    ranked, scores, hits = row.get("ranked_ids"), row.get("scores"), row.get("hits")
    unscored = scores is None and (row["route"] == "bm25" or not candidates)
    require(isinstance(ranked, list) and isinstance(hits, list)
            and len(ranked) == len(hits) == len(candidates)
            and (unscored or isinstance(scores, list) and len(scores) == len(candidates)
                 and all(finite(s) for s in scores)), "THREEWAY_RAW_OUTPUT_SHAPE_INVALID")
    order = list(range(len(candidates))) if unscored else sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    require(ranked == [candidates[i] for i in order], "THREEWAY_RAW_RANKING_CHANGED")
    if row["status"] == "empty":
        require(not candidates, "THREEWAY_EMPTY_WITH_RESULTS")
    for rank, (index, hit) in enumerate(zip(order, hits), 1):
        chunk = context["corpus"][candidates[index]]
        require(hit.get("chunk_id") == candidates[index] and hit.get("rank") == rank
                and hit.get("candidate_rank") == index + 1 and hit.get("document_id") == chunk["document_id"]
                and hit.get("text_sha256") == chunk["text_sha256"] and hit.get("text") == chunk["text"]
                and (hit.get("score") is None if unscored else finite(hit.get("score"))
                     and hit["score"] == scores[index]), "THREEWAY_FINAL_HIT_IDENTITY_CHANGED")
    timing = row.get("timing", {})
    require(all(finite(timing.get(name)) and timing[name] >= 0 for name in TIMINGS)
            and timing["wall_seconds"] > 0, "THREEWAY_SUCCESS_TIMING_INVALID")
    mps = validate_mps_snapshot(row, context) if row["route"] == "located_fp16" and candidates else None
    if row["route"] == "vector":
        snapshot = row.get("vector_proof", {})
        wanted = context["embedding_feeds"][row["case_id"]]["batches"]
        require(snapshot.get("complete") is True and len(wanted) == 1
                and snapshot.get("expected_batches") == snapshot.get("verified_batches")
                == snapshot.get("session_run_attempts") == snapshot.get("session_run_successes") == 1
                and len(snapshot.get("batches", [])) == 1, "THREEWAY_VECTOR_ENCODING_PROOF_INCOMPLETE")
        actual = snapshot["batches"][0]
        tensors = actual["actual_tensors"]
        require(set(tensors) in ({"input_ids", "attention_mask"}, {"input_ids", "attention_mask", "token_type_ids"})
                and actual["candidate_indices"] == [0] and actual["verification_passed"] is True
                and actual["native_attempted"] is True and actual["native_succeeded"] is True,
                "THREEWAY_VECTOR_ENCODING_CALL_CHANGED")
        for name, ref in tensors.items():
            expected_name = "token_type_ids_if_required" if name == "token_type_ids" else name
            require(ref == wanted[0]["tensors"][expected_name], "THREEWAY_VECTOR_ACTUAL_INPUT_CHANGED")
        trace = row.get("selection_trace", {}).get("trace", {})
        require(trace.get("query_encode_calls") == trace.get("query_encode_successes") == 1
                and trace.get("query_vector_cache") is False and trace.get("query_result_cache") is False
                and trace.get("embedding_seconds") == row["timing"]["embedding_seconds"]
                and trace.get("vector_search_seconds") == row["timing"]["vector_search_seconds"],
                "THREEWAY_VECTOR_QUERY_CACHE_OR_CALL_CHANGED")
        encoder = trace.get("encoder_session", {})
        require(encoder.get("providers") == ["CPUExecutionProvider"] and encoder.get("intra_op_num_threads") == 2
                and encoder.get("inter_op_num_threads") == 1, "THREEWAY_VECTOR_ENCODER_SESSION_CHANGED")
        for name, shape in (("matrix_reference", [len(context["corpus"]), 384]), ("query_vector_reference", [384])):
            ref = trace.get(name, {})
            require(ref.get("dtype") == "float32" and ref.get("byte_order") == "little"
                    and ref.get("shape") == shape and ref.get("bytes") == math.prod(shape) * 4
                    and is_hash(ref.get("sha256")), "THREEWAY_VECTOR_FLOAT_ARRAY_REFERENCE_INVALID")
        matrix = trace["matrix_reference"]
        require(context.setdefault("vector_matrix_reference", matrix) == matrix, "THREEWAY_VECTOR_MATRIX_CHANGED_BETWEEN_QUERIES")
    return {"successful": True, "mps_batches": mps["batches"] if mps else 0,
            "historical_candidate_input_match": mps["historical_candidate_input_match"] if mps else None,
            "vector_proof_recorded": isinstance(row.get("vector_proof"), dict) if row["route"] == "vector" else None}


def read_execution(execution, freeze_sha, context):
    wrapper = read(execution / "execution-plan.json")
    plan = expected_plan(context)
    require(wrapper.get("freeze_sha256") == freeze_sha, "THREEWAY_PLAN_FREEZE_CHANGED")
    for name, value in plan.items():
        require(wrapper["plan"].get(name) == value, "THREEWAY_EXECUTION_PLAN_CHANGED:" + name)
    terminal = read(execution / "terminal.json")
    if "freeze_sha256" in terminal:
        require(terminal["freeze_sha256"] == freeze_sha, "THREEWAY_TERMINAL_FREEZE_CHANGED")
    require(terminal.get("labels_parsed") is False, "THREEWAY_EXECUTOR_LABEL_BOUNDARY_CHANGED")
    filenames = {f"cell-{i:04d}-{cell['phase']}-{cell['route']}.json": (i, cell)
                 for i, cell in enumerate(plan["cells"])}
    rows, refs, proofs = {}, [], []
    for path in sorted(execution.glob("cell-*.json")):
        require(path.name in filenames, "THREEWAY_UNPLANNED_RAW_FILENAME")
        index, cell = filenames[path.name]
        row = read(path)
        require(key(row) not in rows, "THREEWAY_DUPLICATE_RAW_CELL")
        proof = validate_row(row, cell, context)
        rows[key(row)] = row
        refs.append(file_reference(path))
        proofs.append({"index": index, **proof})
    missing = {key(c) for c in plan["cells"]} - set(rows)
    require({p["index"] for p in proofs} == set(range(len(rows))), "THREEWAY_EXECUTION_NOT_A_PREFIX")
    declared = terminal.get("unexecuted")
    require(isinstance(declared, list) and len(declared) == len({key(c) for c in declared})
            and {key(c) for c in declared} == missing, "THREEWAY_UNEXECUTED_CELLS_CHANGED")
    for item in declared:
        require(isinstance(item.get("index"), int) and 0 <= item["index"] < len(plan["cells"])
                and all(item.get(k) == v for k, v in plan["cells"][item["index"]].items()),
                "THREEWAY_UNEXECUTED_INDEX_CHANGED")
    complete = (terminal.get("status") == "complete" and not terminal.get("prepare_only")
                and not missing and all(row["status"] in SUCCESS for row in rows.values()))
    if terminal.get("status") == "complete" and not terminal.get("prepare_only"):
        require(complete, "THREEWAY_COMPLETE_TERMINAL_CONTRADICTS_RAW")
    counts = terminal.get("counters", {})
    for name, phase in (("hot_rows", "hot"), ("warmup_rows", "warmup")):
        if name in counts:
            require(counts[name] == sum(r["phase"] == phase for r in rows.values()), "THREEWAY_PHASE_COUNTER_CHANGED")
    refs.extend(file_reference(execution / name) for name in ("execution-plan.json", "terminal.json"))
    return plan, terminal, rows, refs, proofs, complete, wrapper.get("measurement")


def quality(ranked, candidates, gold):
    if not gold:
        return {**dict.fromkeys(METRICS), "first_gold_rank": None}
    first = next((rank for rank, cid in enumerate(ranked[:24], 1) if cid in gold), None)
    return {**{f"recall{k}": len(set(ranked[:k]) & gold) / len(gold) for k in (5, 8, 10)},
            "mrr24": 1 / first if first else 0., "candidate_recall24": len(set(candidates) & gold) / len(gold),
            "first_gold_rank": first}


def aggregate(rows):
    known = [r for r in rows if r["mrr24"] is not None]
    complete = bool(rows) and len(known) == len(rows)
    return {"planned_n": len(rows), "known_n": len(known), "unknown_n": len(rows) - len(known), "complete": complete,
            **{m: statistics.mean(r[m] for r in known) if complete else None for m in METRICS},
            "observed_mean": {m: statistics.mean(r[m] for r in known) if known else None for m in METRICS},
            "first_hot_statuses": dict(Counter(r["first_hot_status"] for r in rows))}


def distribution(values):
    ordered = sorted(values)
    def percentile(p):
        if not ordered:
            return None
        position = (len(ordered) - 1) * p
        a, b = math.floor(position), math.ceil(position)
        return ordered[a] + (ordered[b] - ordered[a]) * (position - a)
    return {"n": len(values), "p50": percentile(.5), "p95": percentile(.95),
            "mean": statistics.mean(values) if values else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None}


def timing_distribution(rows, *, planned_n):
    values = [r["median_wall_seconds"] for r in rows if r["all_hot_successful"]]
    observed = distribution(values)
    complete = len(values) == planned_n
    return {"planned_n": planned_n, "known_n": len(values), "unknown_n": planned_n - len(values),
            "complete": complete, "question_repeats": 3,
            "primary": observed if complete else None, "observed": observed}


def per_case_rows(context, plan, raw):
    results = []
    for route in ROUTES:
        for query in context["queries"]:
            cid = query["case_id"]
            runs = [raw.get(("hot", repetition, cid, route)) for repetition in range(3)]
            first = runs[0]
            successful = first is not None and first["status"] in SUCCESS
            known = all(row is not None and row["status"] in SUCCESS for row in runs)
            q = quality(first["ranked_ids"], first["candidate_ids"], context["gold"][cid]) if successful else {
                **dict.fromkeys(METRICS), "first_gold_rank": None}
            results.append({"case_id": cid, "route": route, "split": context["metadata"][cid]["split"],
                "answerable": bool(context["gold"][cid]), "first_hot_status": first["status"] if first else "not_executed",
                "first_hot_empty": successful and not first["ranked_ids"], **q,
                "hot_statuses": [row["status"] if row else "not_executed" for row in runs],
                "all_hot_successful": known,
                "median_wall_seconds": statistics.median(row["timing"]["wall_seconds"] for row in runs) if known else None,
                "median_components_seconds": {name: statistics.median(row["timing"][name] for row in runs)
                                              for name in TIMINGS} if known else None,
                "ranking_stable": all(row["ranked_ids"] == runs[0]["ranked_ids"] for row in runs) if known else None,
                "scores_stable": all(row["scores"] == runs[0]["scores"] for row in runs)
                    if known and all(isinstance(row["scores"], list) for row in runs) else None})
    return results


def paired_changes(per_case, baseline, *, include_gaps=False):
    rows = {(r["case_id"], r["route"]): r for r in per_case}
    ids = [r["case_id"] for r in per_case if r["route"] == baseline and (include_gaps or r["answerable"])]
    paired = []
    for cid in ids:
        a, b = rows[(cid, baseline)], rows[(cid, "located_fp16")]
        timed = a["all_hot_successful"] and b["all_hot_successful"]
        known = a["mrr24"] is not None and b["mrr24"] is not None
        av, bv = a["median_wall_seconds"], b["median_wall_seconds"]
        paired.append({"case_id": cid, "baseline": baseline, "candidate": "located_fp16", "answerable": a["answerable"],
            "timing_available": timed, "baseline_wall_seconds": av, "candidate_wall_seconds": bv,
            "delta_ms": (bv - av) * 1000 if timed else None,
            "absolute_reduction_ms": (av - bv) * 1000 if timed else None,
            "relative_reduction_percent": 100 * (1 - bv / av) if timed else None,
            "quality_available": known,
            "quality_deltas": {m: b[m] - a[m] if known else None for m in METRICS}})
    fair = [r for r in paired if r["timing_available"]]
    complete = len(fair) == len(ids)
    a = distribution([r["baseline_wall_seconds"] for r in fair])
    b = distribution([r["candidate_wall_seconds"] for r in fair])
    total = {"baseline": baseline, "candidate": "located_fp16", "planned_n": len(ids), "known_n": len(fair),
        "unknown_n": len(ids) - len(fair), "complete": complete,
        "primary_baseline_seconds": a if complete else None, "primary_candidate_seconds": b if complete else None,
        "observed_baseline_seconds": a, "observed_candidate_seconds": b,
        "observed_paired_delta_ms": distribution([r["delta_ms"] for r in fair]),
        "observed_paired_relative_reduction_percent": distribution([r["relative_reduction_percent"] for r in fair]),
        "faster_cases": sum(r["delta_ms"] < 0 for r in fair), "equal_cases": sum(r["delta_ms"] == 0 for r in fair),
        "slower_cases": sum(r["delta_ms"] > 0 for r in fair),
        "distribution_changes": {stat: {"absolute_reduction_ms": (a[stat] - b[stat]) * 1000,
            "relative_reduction_percent": 100 * (1 - b[stat] / a[stat])} if complete else None
            for stat in ("p50", "p95", "mean")}}
    for metric in METRICS:
        known_q = [r["quality_deltas"][metric] for r in paired if r["quality_available"]]
        total.setdefault("quality_changes", {})[metric] = {
            "planned_n": sum(r["answerable"] for r in paired), "known_n": len(known_q),
            "mean_delta": statistics.mean(known_q) if len(known_q) == sum(r["answerable"] for r in paired) else None,
            "observed_mean_delta": statistics.mean(known_q) if known_q else None,
            "improved_cases": sum(x > 0 for x in known_q), "equal_cases": sum(x == 0 for x in known_q),
            "worse_cases": sum(x < 0 for x in known_q)}
    return paired, total


def historical_drift(context, raw, per_case):
    refs = context["frozen"].get("historical_reference_files", [])
    if not refs:
        return {"available": False, "reason": "No historical scoring references frozen; historical latency never participates."}, []
    history = {}
    for ref in refs:
        row = read(check_reference(ref))
        require(row.get("phase") == "hot" and row.get("variant") == "mps_fp16"
                and row.get("status") == "executed" and row["case_id"] in context["by_id"], "HISTORICAL_ROW_CONTRACT_CHANGED")
        k = row["case_id"], row["repetition"]
        require(k not in history, "HISTORICAL_ROW_DUPLICATED")
        query = context["by_id"][row["case_id"]]
        require(row["query"] == query["query"] and row["scope_document_ids"] == query["scope_document_ids"],
                "HISTORICAL_QUERY_SCOPE_CHANGED")
        candidates, scores = row["candidate_ids"], row["scores"]
        require(len(candidates) == len(set(candidates)) == len(scores) and len(candidates) <= 24
                and all(finite(value) for value in scores)
                and row["ranked_ids"] == [candidates[i] for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i))],
                "HISTORICAL_SCORE_OR_RANK_CHANGED")
        require(all(cid in context["corpus"] and context["corpus"][cid]["document_id"] in query["scope_document_ids"]
                    for cid in candidates), "HISTORICAL_CANDIDATE_SCOPE_CHANGED")
        history[k] = row
    require({cid for cid, repeat in history if repeat == 0} == set(context["by_id"]), "HISTORICAL_FIRST_HOT_COVERAGE_CHANGED")
    changes, old_quality = [], []
    current_quality = [r for r in per_case if r["route"] == "located_fp16" and r["answerable"]]
    for cid in context["by_id"]:
        old, current = history[(cid, 0)], raw.get(("hot", 0, cid, "located_fp16"))
        q = quality(old["ranked_ids"], old["candidate_ids"], context["gold"][cid])
        if context["gold"][cid]:
            old_quality.append({**q, "first_hot_status": "executed"})
        known = current is not None and current["status"] in SUCCESS
        same_inputs = current["candidate_ids"] == old["candidate_ids"] if known else None
        changes.append({"case_id": cid, "current_successful": known, "candidate_ids_match": same_inputs,
            "ranking_changed": current["ranked_ids"] != old["ranked_ids"] if known else None,
            "scores_changed": current["scores"] != old["scores"] if same_inputs else None,
            "max_abs_score_delta": max((abs(a - b) for a, b in zip(current["scores"], old["scores"])), default=0.) if same_inputs else None,
            "historical_quality": q})
    old_agg, new_agg = aggregate(old_quality), aggregate(current_quality)
    return {"available": True, "planned_cases": 114, "known_current_cases": sum(r["current_successful"] for r in changes),
        "historical_reference_rows": len(history), "candidate_changed_cases": sum(r["candidate_ids_match"] is False for r in changes),
        "ranking_changed_cases": sum(r["ranking_changed"] is True for r in changes),
        "scores_changed_cases": sum(r["scores_changed"] is True for r in changes),
        "historical_quality": old_agg, "current_quality": new_agg,
        "quality_deltas": {m: new_agg[m] - old_agg[m] if new_agg["complete"] else None for m in METRICS},
        "meaning": "Historical score/rank/label comparison only; no historical timing is used."}, changes


def report(freeze, expected_freeze_sha256, execution, output):
    require(output.is_dir() and not any(output.iterdir()), "THREEWAY_REPORT_OUTPUT_MUST_BE_EMPTY")
    context = load_context(freeze, expected_freeze_sha256)
    plan, terminal, raw, refs, proofs, complete, measurement = read_execution(execution, expected_freeze_sha256, context)
    per_case = per_case_rows(context, plan, raw)
    aggregates, times, stability = [], [], {}
    for route in ROUTES:
        rows = [r for r in per_case if r["route"] == route]
        for split in ("all", "regression", "pending"):
            positive = [r for r in rows if r["answerable"] and (split == "all" or r["split"] == split)]
            aggregates.append({"route": route, "split": split, **aggregate(positive)})
        times.append({"route": route, "population": "positive112", **timing_distribution([r for r in rows if r["answerable"]], planned_n=112)})
        times.append({"route": route, "population": "all114_including_gaps", **timing_distribution(rows, planned_n=114)})
        stability[route] = {"planned_cases": 114, "known_cases": sum(r["all_hot_successful"] for r in rows),
            "score_comparison_known_cases": sum(r["scores_stable"] is not None for r in rows),
            "ranking_changed_cases": sum(r["ranking_stable"] is False for r in rows),
            "scores_changed_cases": sum(r["scores_stable"] is False for r in rows)}
    paired, comparisons = [], []
    for baseline in ("bm25", "vector"):
        rows, comparison = paired_changes(per_case, baseline)
        paired.extend(rows)
        comparisons.append(comparison)
    historical, historical_cases = historical_drift(context, raw, per_case)
    states = Counter(row["status"] for row in raw.values())
    states["not_executed"] += len(plan["cells"]) - len(raw)
    warmups = [{"cell_index": index, "route": cell["route"], "case_id": cell["case_id"],
                "status": raw[key(cell)]["status"] if key(cell) in raw else "not_executed",
                "timing": raw[key(cell)].get("timing") if key(cell) in raw else None}
               for index, cell in enumerate(plan["cells"]) if cell["phase"] == "warmup"]
    summary = {"schema": "threeway-retrieval-report-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": expected_freeze_sha256, "measurement": measurement,
        "planned_cells": len(plan["cells"]), "observed_cells": len(raw), "states": dict(states),
        "all_planned_cells_successful": complete, "answerable_cases": 112, "gap_cases": 2,
        "independently_certified": 0, "corpus_chunks": len(context["corpus"]),
        "frozen_reference_entries_rechecked": len(context["references"]),
        "unique_frozen_paths_rechecked": context["unique_reference_count"],
        "quality_aggregates": aggregates, "timing_distributions": times, "comparisons_vs_same_run_baselines": comparisons,
        "stability": stability, "warmups_excluded_from_hot_metrics": warmups, "historical_drift": historical,
        "proof_summary": {"successful_located_mps_batches": sum(p["mps_batches"] for p in proofs),
                          "vector_proof_records": sum(p.get("vector_proof_recorded") is True for p in proofs)},
        "terminal": terminal, "quality_regression_gate": None, "automatic_adoption": False,
        "limits": ["All 112 planned positives remain in first-hot quality; successful empty results are zero, failures/unknown/unexecuted remain unknown.",
                   "Primary wall distribution uses each question median of three successful hot repeats; 114-question timing including gaps is secondary.",
                   "P50/P95 distribution reductions differ from the distribution of per-question percentage changes; both are named separately.",
                   "Same-run baseline comparisons only; model/index/corpus preparation and result file writing are outside the declared common wall boundary.",
                   "Stored MPS input/device/sync proof is not a complete Metal trace. Vector query encoding call/feed/cache/CPU session and float-array identities are checked; this is in-memory exact cosine, not Qdrant transport or bitwise parity.",
                   "Historical exposure and non-exhaustive labels remain; independent certification zero; not full Agent or production capacity acceptance.",
                   "No zero-regression quality gate; measured quality and latency tradeoffs are reported separately for user review."]}
    artifacts = {"summary.json": summary, "per-case.json": per_case, "paired-cases.json": paired,
                 "quality-changes.json": [{k: v for k, v in r.items() if k in {"case_id", "baseline", "candidate", "quality_available", "quality_deltas"}} for r in paired],
                 "historical-drift-cases.json": historical_cases,
                 "planned-cells.json": [{"index": i, **cell, "observed_status": raw[key(cell)]["status"] if key(cell) in raw else "not_executed"}
                                        for i, cell in enumerate(plan["cells"])],
                 "raw-references.json": [file_reference(freeze), *refs]}
    for name, value in artifacts.items():
        with (output / name).open("x") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    lines = ["# 三路完整检索同期复算", "", f"计划1029格，保存{len(raw)}格；完整成功：{complete}。", "",
             "质量为112道正例首热宏平均；成功空结果为0，失败/未知保持N/A。", "",
             "|路线|Recall@5|Recall@8|Recall@10|MRR@24|候选覆盖@24|", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        if row["split"] == "all":
            lines.append("|" + "|".join([row["route"], *["N/A" if row[m] is None else f"{row[m]:.6f}" for m in METRICS]]) + "|")
    lines += ["", "时间为每题三轮墙钟中位数后计算112题分布，暖机单列。", "",
              "|路线|配对可用n/N|P50秒|P95秒|均值秒|", "|---|---:|---:|---:|---:|"]
    for row in times:
        if row["population"] == "positive112":
            d = row["primary"]
            lines.append("|" + "|".join([row["route"], f"{row['known_n']}/{row['planned_n']}",
                *["N/A" if d is None else f"{d[k]:.6f}" for k in ("p50", "p95", "mean")]]) + "|")
    lines += ["", "全部同批差异、逐题质量/耗时取舍、失败格和历史漂移见JSON。绝对减少ms、相对减少百分比为正表示候选更快；不扣仪器开销、不与历史耗时相减。", "",
              "本次不设质量零退步门，不自动采用、合并或部署。"]
    with (output / "report.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.freeze, args.expected_freeze_sha256, args.execution, args.output)
    print(json.dumps({"planned_cells": result["planned_cells"], "observed_cells": result["observed_cells"],
                      "complete": result["all_planned_cells_successful"], "model_imports_or_calls": 0}))


if __name__ == "__main__":
    main()
