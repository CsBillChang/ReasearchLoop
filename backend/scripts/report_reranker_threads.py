"""只读复算同输入CPU线程对照；失败保留，未执行不补零。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


METRICS = ("recall5", "recall8", "recall10", "mrr24", "candidate_recall24")
TIMINGS = ("reranker_wall_seconds", "provider_inference_seconds",
           "feed_verification_seconds", "native_session_run_seconds")
STATUSES = {"executed", "failed", "empty", "stopped", "unknown"}


def canonical_sha(value: object) -> str:
    """按准备器公开的JSON配方独立计算投影身份。"""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    """返回原件字节身份，不改写来源。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_reference(ref: dict):
    """读取冻结引用；缺失或换版直接拒绝报告资格。"""
    path = Path(ref["path"])
    if not path.is_absolute() or file_sha(path) != ref["sha256"]:
        raise ValueError("REPORT_FROZEN_INPUT_CHANGED")
    return json.loads(path.read_text())


def percentile(values: list[float], p: float) -> float | None:
    """按(n−1)p线性插值，不把重复轮次当作独立题。"""
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    low = math.floor(position)
    return values[low] + (values[math.ceil(position)] - values[low]) * (position - low)


def distribution(values: list[float]) -> dict:
    """保留样本数及描述分布，不宣称统计显著性。"""
    return {"n": len(values), "p50": percentile(values, .5), "p95": percentile(values, .95),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "mean": statistics.mean(values) if values else None}


def quality(ranked: list[str], candidates: list[str], gold: set[str]) -> dict:
    """计算非穷尽正例指标；缺证题保持N/A。"""
    if not gold:
        return {**dict.fromkeys(METRICS), "first_gold_rank": None}
    first = next((i for i, chunk in enumerate(ranked[:24], 1) if chunk in gold), None)
    return {**{f"recall{k}": len(set(ranked[:k]) & gold) / len(gold) for k in (5, 8, 10)},
            "mrr24": 1 / first if first else 0.0,
            "candidate_recall24": len(set(candidates) & gold) / len(gold),
            "first_gold_rank": first}


def cell_key(row: dict) -> tuple:
    """使用语义格身份，不能靠文件名吞掉重复格。"""
    return row["phase"], row.get("repetition"), row["case_id"], row["threads"]


def finite_number(value) -> bool:
    """布尔值与NaN均不是合法耗时或分数。"""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_row(row: dict, case: dict, feed: dict, expected_model: dict | None) -> dict | None:
    """验证实际输入、分数身份与成功feed核对，不以状态字符串替代证据。"""
    if row.get("status") not in STATUSES:
        raise ValueError("REPORT_UNKNOWN_STATUS")
    candidate_ids = [c["chunk_id"] for c in case["candidates"]]
    if (row.get("query") != case["query"]
            or row.get("scope_document_ids") != case["scope_document_ids"]
            or row.get("candidate_ids") != candidate_ids
            or row.get("input_sha256") != case["input_sha256"]
            or row.get("feed_projection_sha256") != canonical_sha(feed)):
        raise ValueError("REPORT_RAW_INPUT_IDENTITY_CHANGED")
    model = row.get("model_identity")
    if model is not None:
        if model.get("threads") != row["threads"]:
            raise ValueError("REPORT_MODEL_THREADS_CHANGED")
        normalized = {k: v for k, v in model.items() if k != "threads"}
        if expected_model is not None and normalized != expected_model:
            raise ValueError("REPORT_MODEL_IDENTITY_CHANGED")
        expected_model = normalized
    if row["status"] != "executed":
        return expected_model
    if not model:
        raise ValueError("REPORT_SUCCESS_WITHOUT_MODEL_IDENTITY")
    session = row.get("session_identity") or {}
    if (session.get("providers") != ["CPUExecutionProvider"]
            or session.get("intra_op_num_threads") != row["threads"]
            or session.get("inter_op_num_threads") != 1
            or not {"input_ids", "attention_mask"}.issubset(session.get("input_names", []))
            or not set(session.get("input_names", [])).issubset(
                {"input_ids", "attention_mask", "token_type_ids"})):
        raise ValueError("REPORT_SESSION_IDENTITY_CHANGED")
    verification = row.get("feed_verification") or {}
    batches = len(feed["batches"])
    if (verification.get("complete") is not True
            or any(verification.get(k) != batches for k in
                   ("expected_batches", "verified_batches", "session_run_attempts", "session_run_successes"))):
        raise ValueError("REPORT_ACTUAL_FEED_NOT_VERIFIED")
    actual_batches = verification.get("batches")
    if not isinstance(actual_batches, list) or len(actual_batches) != batches:
        raise ValueError("REPORT_ACTUAL_BATCH_RECORDS_INCOMPLETE")
    for index, (actual, expected) in enumerate(zip(actual_batches, feed["batches"])):
        if (actual.get("batch_index") != index
                or actual.get("candidate_indices") != expected["candidate_indices"]
                or actual.get("verification_passed") is not True
                or actual.get("native_attempted") is not True
                or actual.get("native_succeeded") is not True or actual.get("error") is not None
                or set(actual.get("actual_tensors", {})) != set(session["input_names"])):
            raise ValueError("REPORT_ACTUAL_BATCH_IDENTITY_CHANGED")
        for name, ref in actual["actual_tensors"].items():
            expected_name = "token_type_ids_if_required" if name == "token_type_ids" else name
            if ref != expected["tensors"][expected_name]:
                raise ValueError("REPORT_ACTUAL_TENSOR_HASH_CHANGED")
    scores = row.get("scores")
    if not isinstance(scores, list) or len(scores) != len(candidate_ids) or not all(map(finite_number, scores)):
        raise ValueError("REPORT_SCORES_INVALID")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    if row.get("ranked_ids") != [candidate_ids[i] for i in order]:
        raise ValueError("REPORT_RANK_SCORE_IDENTITY_CHANGED")
    expected_hits = [{"chunk_id": case["candidates"][i]["chunk_id"],
                      "document_id": case["candidates"][i]["document_id"],
                      "text_sha256": case["candidates"][i]["text_sha256"],
                      "rank": rank, "candidate_rank": i + 1, "reranker_score": scores[i]}
                     for rank, i in enumerate(order, 1)]
    if row.get("hits") != expected_hits:
        raise ValueError("REPORT_HIT_SOURCE_IDENTITY_CHANGED")
    timing = row.get("timing") or {}
    if any(not finite_number(timing.get(k)) or timing[k] < 0 for k in TIMINGS):
        raise ValueError("REPORT_TIMING_INVALID")
    return expected_model


def score_differences(a: dict, b: dict) -> dict:
    """按候选身份比较已成功评分，浮点差异与排名变化分开。"""
    if a.get("status") != "executed" or b.get("status") != "executed":
        return {"available": False}
    if a["candidate_ids"] != b["candidate_ids"]:
        raise ValueError("REPORT_COMPARISON_CANDIDATES_CHANGED")
    a_ranks = {cid: i for i, cid in enumerate(a["ranked_ids"], 1)}
    b_ranks = {cid: i for i, cid in enumerate(b["ranked_ids"], 1)}
    entries = [{"chunk_id": cid, "a_score": x, "b_score": y, "delta": y - x,
                "a_rank": a_ranks[cid], "b_rank": b_ranks[cid]}
               for cid, x, y in zip(a["candidate_ids"], a["scores"], b["scores"])]
    return {"available": True, "ranking_changed": a["ranked_ids"] != b["ranked_ids"],
            "scores_changed": a["scores"] != b["scores"],
            "score_changes": sum(e["delta"] != 0 for e in entries),
            "max_abs_logit_delta": max((abs(e["delta"]) for e in entries), default=0),
            "entries": entries}


def load_historical(frozen: dict, cases: list[dict]) -> dict:
    """只在报告阶段读取371历史排名与分数，并核对原manifest身份。"""
    refs = frozen["report_inputs"]
    old_freeze = read_reference(refs["historical_freeze"])
    old_manifest = read_reference(refs["historical_manifest"])
    for name in ("labels", "original_questions"):
        old_ref = old_freeze["inputs"][name]
        if (refs[name]["sha256"] != old_ref["sha256"]
                or Path(refs[name]["path"]).resolve() != Path(old_ref["path"]).resolve()):
            raise ValueError("REPORT_HISTORICAL_LABEL_REFERENCE_CHANGED")
    root = Path(refs["historical_manifest"]["path"]).parent.parent
    case_ids = {case["case_id"] for case in cases}
    result = {}
    for ref in old_manifest:
        relative = Path(ref["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("REPORT_HISTORICAL_PATH_INVALID")
        if relative.name.startswith("hot-0-") and relative.name.endswith("-located_bm25_rerank.json"):
            path = (root / relative).resolve()
            if path.parent != (root / "execution").resolve():
                raise ValueError("REPORT_HISTORICAL_PATH_INVALID")
            raw = read_reference({"path": str(path), "sha256": ref["sha256"]})
            cid = raw["case_id"]
            if cid not in case_ids or cid in result or raw.get("repetition") != 0:
                raise ValueError("REPORT_HISTORICAL_CELL_CHANGED")
            row = raw["result"]
            if row.get("status") != "executed":
                raise ValueError("REPORT_HISTORICAL_B_NOT_SUCCESSFUL")
            scores_by_id = {hit["chunk_id"]: hit["reranker_score"] for hit in row["hits"]}
            result[cid] = {**row, "scores": [scores_by_id[x] for x in row["candidate_ids"]],
                           "source_reference": {"path": str(path), "sha256": ref["sha256"]}}
    if set(result) != case_ids:
        raise ValueError("REPORT_HISTORICAL_CASES_INCOMPLETE")
    return result


def summarize_quality(rows: list[dict]) -> dict:
    """完整聚合保留计划分母；缺格不伪造零分或悄悄缩小分母。"""
    known = [row for row in rows if row["mrr24"] is not None]
    complete = len(known) == len(rows) and bool(rows)
    return {"planned_n": len(rows), "known_n": len(known), "complete": complete,
            **{name: statistics.mean(row[name] for row in known) if complete else None for name in METRICS},
            "observed_mean": {name: statistics.mean(row[name] for row in known) if known else None
                              for name in METRICS},
            "first_hot_statuses": dict(Counter(row["status"] for row in rows))}


def paired_timing(pairs: list[dict], planned_n: int) -> dict:
    """公平时延只取两配置全部三轮成功的同题，保留排除分母。"""
    good = [row for row in pairs if row["fair_timing_eligible"]]
    result = {"n": len(good), "planned_n": planned_n, "excluded_n": planned_n - len(good)}
    for key in TIMINGS:
        a = [row["timing_2"][key] for row in good]
        b = [row["timing_4"][key] for row in good]
        result[key] = {"threads_2": distribution(a), "threads_4": distribution(b),
                       "paired_delta_seconds": distribution([y - x for x, y in zip(a, b)]),
                       "paired_ratio_4_over_2": distribution([y / x for x, y in zip(a, b) if x > 0]),
                       "faster_cases": sum(y < x for x, y in zip(a, b)),
                       "equal_cases": sum(y == x for x, y in zip(a, b)),
                       "slower_cases": sum(y > x for x, y in zip(a, b))}
    return result


def write_new(path: Path, value, *, raw=False) -> None:
    """只独占创建报告，防止覆盖历史失败或结果。"""
    data = value if raw else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(data)


def build_report(freeze_path: Path, execution: Path, output: Path) -> dict:
    """校验完整计划、实际原件和历史基线，再一次性发布可复算报告。"""
    if not output.is_absolute() or output != output.resolve() or not output.is_dir() or any(output.iterdir()):
        raise ValueError("REPORT_OUTPUT_MUST_BE_NEW_EMPTY_REAL_DIRECTORY")
    frozen = json.loads(freeze_path.read_text())
    inputs, report_inputs = frozen["inputs"], frozen["report_inputs"]
    cases = read_reference(inputs["cases"])
    feed_file = read_reference(inputs["feed_manifest"])
    feeds = {row["case_id"]: row for row in feed_file["cases"]}
    plan = read_reference(inputs["execution_plan"])
    model_manifest = read_reference(inputs["model_manifest"])
    metadata_rows = read_reference(report_inputs["original_questions"])
    metadata = {row["case_id"]: row for row in metadata_rows}
    gold = defaultdict(set)
    for pair in read_reference(report_inputs["labels"])["positive_pairs"]:
        gold[pair["case_id"]].add(pair["chunk_id"])
    case_map = {case["case_id"]: case for case in cases}
    if (len(case_map) != len(cases) or len(feeds) != len(feed_file["cases"])
            or len(metadata) != len(metadata_rows)
            or set(case_map) != set(feeds) or set(case_map) != set(metadata)
            or set(gold) - set(case_map)):
        raise ValueError("REPORT_CASE_SET_CHANGED")
    for case in cases:
        cid = case["case_id"]
        body = {k: v for k, v in case.items() if k != "input_sha256"}
        if (canonical_sha(body) != case["input_sha256"] or feeds[cid]["input_sha256"] != case["input_sha256"]
                or metadata[cid]["query"] != case["query"]
                or metadata[cid]["scope_document_ids"] != case["scope_document_ids"]
                or metadata[cid]["answerable"] != bool(gold[cid])):
            raise ValueError("REPORT_CASE_METADATA_IDENTITY_CHANGED")
        ids = [c["chunk_id"] for c in case["candidates"]]
        if len(set(ids)) != len(ids) or not 1 <= len(ids) <= 24:
            raise ValueError("REPORT_CANDIDATE_SET_CHANGED")
        for candidate in case["candidates"]:
            if (candidate["document_id"] not in case["scope_document_ids"]
                    or hashlib.sha256(candidate["text"].encode()).hexdigest() != candidate["text_sha256"]):
                raise ValueError("REPORT_FROZEN_TEXT_OR_SCOPE_CHANGED")
    historical = load_historical(frozen, cases)
    for cid, old in historical.items():
        case = case_map[cid]
        if (old["query"] != case["query"] or old["scope_document_ids"] != case["scope_document_ids"]
                or old["candidate_ids"] != [c["chunk_id"] for c in case["candidates"]]
                or {h["chunk_id"]: (h["document_id"], h["text_sha256"]) for h in old["hits"]}
                    != {c["chunk_id"]: (c["document_id"], c["text_sha256"]) for c in case["candidates"]}):
            raise ValueError("REPORT_HISTORICAL_INPUT_CHANGED")
    expected_cells = plan["cells"]
    expected_keys = {cell_key(row) for row in expected_cells}
    if len(expected_keys) != len(expected_cells) or any(key[3] not in (2, 4) for key in expected_keys):
        raise ValueError("REPORT_PLAN_DUPLICATE_OR_THREAD_CHANGED")
    if {key for key in expected_keys if key[0] == "hot"} != {
        ("hot", repetition, case["case_id"], threads) for repetition in range(3)
        for case in cases for threads in (2, 4)}:
        raise ValueError("REPORT_HOT_PLAN_CHANGED")
    if {key for key in expected_keys if key[0] == "warmup"} != {
        ("warmup", None, cases[0]["case_id"], threads) for threads in (2, 4)}:
        raise ValueError("REPORT_WARMUP_PLAN_CHANGED")
    planned_pairs = sum(len(case_map[cell["case_id"]]["candidates"]) for cell in expected_cells)
    if (plan["budget"]["exact_planned_score_attempts"] != len(expected_cells)
            or plan["budget"]["exact_planned_pairs"] != planned_pairs
            or plan["budget"]["max_score_attempts"] < len(expected_cells)
            or plan["budget"]["max_pairs"] < planned_pairs):
        raise ValueError("REPORT_PLAN_BUDGET_CHANGED")
    raw, references, model_identity = {}, [], None
    for path in sorted(execution.glob("*.json")):
        row = json.loads(path.read_text())
        if not isinstance(row, dict) or row.get("phase") not in ("warmup", "hot"):
            continue
        key = cell_key(row)
        if key in raw or key not in expected_keys:
            raise ValueError("REPORT_DUPLICATE_OR_UNPLANNED_CELL")
        model_identity = validate_row(row, case_map[row["case_id"]], feeds[row["case_id"]], model_identity)
        raw[key] = row
        references.append({"path": str(path.resolve()), "sha256": file_sha(path)})
    if model_identity is not None:
        spec = model_manifest["reranker"]
        if (model_identity.get("model_id") != spec["repo"]
                or model_identity.get("revision") != spec["revision"]
                or model_identity.get("weights_sha256") != spec["files"][spec["weights"]]["sha256"]
                or model_identity.get("tokenizer_sha256") != spec["files"]["tokenizer.json"]["sha256"]
                or model_identity.get("batch_size") != 4 or model_identity.get("sort_by_length") is not False
                or model_identity.get("max_pair_tokens") != 512):
            raise ValueError("REPORT_FROZEN_MODEL_IDENTITY_CHANGED")
    terminal_path = execution / "terminal.json"
    terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
    if terminal is not None:
        references.append({"path": str(terminal_path.resolve()), "sha256": file_sha(terminal_path)})
    cells = [{**row, "state": raw[cell_key(row)]["status"] if cell_key(row) in raw else "unexecuted",
              "observed": cell_key(row) in raw} for row in expected_cells]
    per_case, drift, score_rows = [], [], []
    for case in cases:
        cid = case["case_id"]
        for threads in (2, 4):
            repeats = [raw.get(("hot", repetition, cid, threads)) for repetition in range(3)]
            first = repeats[0]
            status = first["status"] if first else "unexecuted"
            candidate_ids = [c["chunk_id"] for c in case["candidates"]]
            if first and status in {"executed", "failed", "empty"}:
                measured = quality(first["ranked_ids"] if status == "executed" else [], candidate_ids, gold[cid])
            else:
                measured = {**dict.fromkeys(METRICS), "first_gold_rank": None}
            successful = [row for row in repeats if row and row["status"] == "executed"]
            all_success = len(successful) == 3
            medians = {k: statistics.median(row["timing"][k] for row in successful) for k in TIMINGS} if all_success else None
            stable_ranks = all(row["ranked_ids"] == successful[0]["ranked_ids"] for row in successful) if all_success else None
            stable_scores = all(row["scores"] == successful[0]["scores"] for row in successful) if all_success else None
            row = {"case_id": cid, "threads": threads, "answerable": bool(gold[cid]),
                   "split": metadata[cid]["split"], "question_type": metadata[cid].get("question_type"),
                   "paper_ids": metadata[cid].get("paper_ids", []), "status": status, **measured,
                   "observed_repeats": sum(r is not None for r in repeats),
                   "repeat_statuses": [r["status"] if r else "unexecuted" for r in repeats],
                   "all_repeats_executed": all_success, "timing_median": medians,
                   "ranking_stable": stable_ranks, "scores_stable": stable_scores,
                   "candidate_count": len(candidate_ids), "candidate_ids": candidate_ids,
                   "ranked_ids": first["ranked_ids"] if first else None}
            per_case.append(row)
            if threads == 2:
                if first and first["status"] == "executed" and {
                    k: v for k, v in first["model_identity"].items() if k != "threads"
                } != {k: v for k, v in historical[cid]["model_identity"].items() if k != "threads"}:
                    raise ValueError("REPORT_HISTORICAL_MODEL_IDENTITY_CHANGED")
                comparison = score_differences(historical[cid], first) if first else {"available": False}
                drift.append({"case_id": cid, "first_hot_status": status,
                              "historical_reference": historical[cid]["source_reference"], **comparison})
        a, b = [raw.get(("hot", 0, cid, t)) for t in (2, 4)]
        score_rows.append({"case_id": cid, **(score_differences(a, b) if a and b else {"available": False})})
    lookup = {(row["case_id"], row["threads"]): row for row in per_case}
    pairs = []
    for case in cases:
        cid = case["case_id"]
        a, b = [lookup[cid, t] for t in (2, 4)]
        pairs.append({"case_id": cid, "answerable": bool(gold[cid]), "split": metadata[cid]["split"],
                      **{f"{name}_delta": b[name] - a[name] if a[name] is not None and b[name] is not None else None for name in METRICS},
                      "first_gold_rank_2": a["first_gold_rank"], "first_gold_rank_4": b["first_gold_rank"],
                      "fair_timing_eligible": a["all_repeats_executed"] and b["all_repeats_executed"],
                      "timing_2": a["timing_median"], "timing_4": b["timing_median"]})
    positives = [row for row in pairs if row["answerable"]]
    aggregates = [{"threads": threads, "split": split,
                   **summarize_quality([row for row in per_case if row["threads"] == threads and row["answerable"]
                        and (split == "all" or row["split"] == split)])}
                  for threads in (2, 4) for split in ("all", "regression", "pending")]
    states = Counter(cell["state"] for cell in cells)
    expected_runs = sum(len(feeds[cell["case_id"]]["batches"]) for cell in expected_cells)
    actual_runs = sum((row.get("feed_verification") or {}).get("session_run_attempts", 0) for row in raw.values())
    summary = {"schema": "reranker-thread-report-v1", "freeze_sha256": file_sha(freeze_path),
        "terminal": terminal, "planned_cells": len(cells), "observed_cells": len(raw),
        "planned_hot_cells": sum(c["phase"] == "hot" for c in cells),
        "observed_hot_cells": sum(k[0] == "hot" for k in raw), "states": dict(states),
        "unknown_or_stopped_cells": states["unknown"] + states["stopped"],
        "unexecuted_cells": states["unexecuted"], "all_planned_cells_observed": len(raw) == len(cells),
        "all_cells_executed": len(raw) == len(cells) and states["executed"] == len(cells),
        "states_by_threads": {str(t): dict(Counter(c["state"] for c in cells if c["threads"] == t)) for t in (2, 4)},
        "hot_states_by_threads_and_repetition": {
            f"{t}:{rep}": dict(Counter(c["state"] for c in cells
                if c["threads"] == t and c["phase"] == "hot" and c["repetition"] == rep))
            for t in (2, 4) for rep in range(3)},
        "planned_budget": plan["budget"], "planned_session_run_calls": expected_runs,
        "observed_session_run_attempts": actual_runs,
        "terminal_counters": terminal.get("counters") if terminal else None,
        "latest_recorded_cumulative_counters": max(raw.values(), key=lambda r: (r.get("counters_after") or {}).get("score_attempts", -1)).get("counters_after") if raw else None,
        "aggregates": aggregates, "paired_timing": paired_timing(positives, len(positives)),
        "paired_timing_by_split": {split: paired_timing([row for row in positives if row["split"] == split],
              sum(row["split"] == split for row in positives)) for split in ("regression", "pending")},
        "first_hot_quality_changes": {name: dict(Counter("unavailable" if row[f"{name}_delta"] is None else
            "increase" if row[f"{name}_delta"] > 0 else "decrease" if row[f"{name}_delta"] < 0 else "equal" for row in positives)) for name in METRICS},
        "historical_drift": {"compared_cases": sum(r["available"] for r in drift),
            "ranking_changed_cases": sum(r.get("ranking_changed", False) for r in drift),
            "scores_changed_cases": sum(r.get("scores_changed", False) for r in drift),
            "max_abs_logit_delta": max((r["max_abs_logit_delta"] for r in drift if r["available"]), default=None)},
        "cross_thread_drift": {"compared_cases": sum(r["available"] for r in score_rows),
            "ranking_changed_cases": sum(r.get("ranking_changed", False) for r in score_rows),
            "scores_changed_cases": sum(r.get("scores_changed", False) for r in score_rows),
            "max_abs_logit_delta": max((r["max_abs_logit_delta"] for r in score_rows if r["available"]), default=None)},
        "stability": {str(t): {"complete_three_repeat_cases": sum(r["threads"] == t and r["all_repeats_executed"] for r in per_case),
            "ranking_unstable_cases": sum(r["threads"] == t and r["ranking_stable"] is False for r in per_case),
            "scores_unstable_cases": sum(r["threads"] == t and r["scores_stable"] is False for r in per_case)} for t in (2, 4)},
        "candidate_and_text_identity_verified": True, "successful_actual_feeds_verified": True,
        "model_identity_except_threads": model_identity, "answerable_cases": len(positives),
        "gap_cases": len(cases) - len(positives), "independently_certified": 0,
        "limits": ["Original labels are non-exhaustive; historical exposure, independently certified cases 0.",
            "Failed/empty first-hot cells retain zero ranking quality in their planned denominator; stopped/unknown/unexecuted remain N/A.",
            "Primary wall includes actual feed verification; overhead is reported without subtraction.",
            "Same-run reranker component time cannot be subtracted from historical 371 whole retrieval time.",
            "Missing terminal or cells do not establish absent physical inference; counters and unresolved work need separate audit."]}
    if summary["all_cells_executed"] and summary["historical_drift"]["ranking_changed_cases"] == 0:
        all_a, all_b = [next(r for r in aggregates if r["threads"] == t and r["split"] == "all") for t in (2, 4)]
        summary["quality_gate"] = {"available": True,
            "all_macro_metrics_non_decreasing": all(all_b[m] >= all_a[m] for m in METRICS),
            "individual_regressions_still_require_review": True}
    else:
        summary["quality_gate"] = {"available": False, "reason": "incomplete execution or 2-thread historical ranking drift"}
    wall = summary["paired_timing"]["reranker_wall_seconds"]
    a50, b50 = wall["threads_2"]["p50"], wall["threads_4"]["p50"]
    a95, b95 = wall["threads_2"]["p95"], wall["threads_4"]["p95"]
    speed_available = summary["all_cells_executed"] and a50 is not None and a50 > 0
    summary["efficiency_gate"] = {"available": speed_available,
        "operation_threshold": "P50 at least 10% lower and P95 non-increasing; not statistical significance",
        "p50_at_least_10_percent_lower": b50 <= .9 * a50 if speed_available else None,
        "p95_non_increasing": b95 <= a95 if speed_available else None,
        "automatic_adoption": False}
    markdown = render_markdown(summary)
    for name, value in (("summary.json", summary), ("per-case.json", per_case),
                        ("paired-cases.json", pairs), ("planned-cells.json", cells),
                        ("historical-drift.json", drift), ("score-changes.json", score_rows),
                        ("gap-diagnostics.json", [r for r in per_case if not r["answerable"]]),
                        ("raw-references.json", references)):
        write_new(output / name, value)
    stream = io.StringIO()
    flat = [{k: v for k, v in row.items() if k not in {"timing_2", "timing_4"}} |
             {f"wall_seconds_{t}": row[f"timing_{t}"]["reranker_wall_seconds"] if row[f"timing_{t}"] else None for t in (2, 4)} for row in pairs]
    writer = csv.DictWriter(stream, fieldnames=list(flat[0]) if flat else ["case_id"])
    writer.writeheader(); writer.writerows(flat)
    write_new(output / "paired-cases.csv", "\ufeff" + stream.getvalue(), raw=True)
    write_new(output / "report.md", markdown, raw=True)
    return summary


def render_markdown(summary: dict) -> str:
    """用明确分母呈现质量、配对时延与执行边界。"""
    text = ["# 同输入CPU线程对照", "",
            f"已观察 {summary['observed_cells']}/{summary['planned_cells']} 格；状态 {summary['states']}。",
            "首热轮质量，三轮逐题中位数计时；原标签非穷尽，独立认证0。", "",
            "| 线程/分组 | 已知/计划正例 | Recall@5 | Recall@8 | Recall@10 | MRR@24 | 候选覆盖@24 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["aggregates"]:
        values = ["N/A" if row[m] is None else f"{row[m]:.4f}" if m == "mrr24" else f"{row[m]*100:.2f}%" for m in METRICS]
        text.append(f"| {row['threads']}/{row['split']} | {row['known_n']}/{row['planned_n']} | " + " | ".join(values) + " |")
    paired = summary["paired_timing"]
    text += ["", f"公平时延共同正例 {paired['n']}/{paired['planned_n']}；两配置各三轮均成功。", "",
             "| 同期重排组件 | P50秒 | P95秒 |", "|---|---:|---:|"]
    for threads in (2, 4):
        row = paired["reranker_wall_seconds"][f"threads_{threads}"]
        text.append(f"| {threads}线程 | " + " | ".join("N/A" if row[k] is None else f"{row[k]:.6f}" for k in ("p50", "p95")) + " |")
    text += ["", "墙钟包含实际feed核对，不扣减开销；不能与371约2.15秒完整检索时间直接相减。",
             f"底层session.run尝试 {summary['observed_session_run_attempts']}，计划 {summary['planned_session_run_calls']}；评分格与底层批次调用分开计量。",
             f"2线程相对371历史漂移：{summary['historical_drift']}。",
             f"两线程配置之间漂移：{summary['cross_thread_drift']}。",
             "失败保留原分母；未执行、stopped/unknown保持N/A。缺证题只作诊断，不进入Recall/MRR。",
             "", "[逐题](per-case.json) · [配对](paired-cases.csv) · [完整格](planned-cells.json) · [分数变化](score-changes.json) · [汇总](summary.json)"]
    return "\n".join(text) + "\n"


def main() -> None:
    """显式读取冻结与执行原件，不导入模型或业务服务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.freeze.resolve(), args.execution.resolve(), args.output.resolve())
    print(json.dumps({"observed": result["observed_cells"], "planned": result["planned_cells"],
                      "unexecuted": result["unexecuted_cells"]}))


if __name__ == "__main__":
    main()
