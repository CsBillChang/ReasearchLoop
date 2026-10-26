"""独立复算MPS兼容、FP16质量与CPU对照；不导入模型或执行推理。"""
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
GATE_METRICS = METRICS[:4]
TIMINGS = ("reranker_wall_seconds", "provider_inference_seconds",
           "feed_verification_seconds", "native_session_run_seconds")
VARIANTS = {"pilot": ("ort_cpu4", "mps_fp32", "mps_fp16"),
            "performance": ("ort_cpu4", "mps_fp16")}
THRESHOLDS = {"mae_max": 0.02, "max_abs_max": 0.1, "metric_tolerance": 1e-12}
STATUSES = {"executed", "failed", "empty", "stopped", "unknown"}
MEASUREMENT = "same_run_reranker_component_with_feed_verification"


def canonical_sha(value: object) -> str:
    """对JSON语义投影独立散列，不调用执行器的身份函数。"""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    """分块散列大权重，避免在报告进程复制整份模型。"""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite(value) -> bool:
    """布尔、NaN和无穷不充当分数或耗时。"""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def reference(path: Path) -> dict:
    """给原件生成可复验身份，不覆盖来源。"""
    return {"path": str(path.resolve()), "sha256": file_sha(path)}


def read_reference(ref: dict):
    """只接受真实绝对文件及已冻结的字节身份。"""
    path = Path(ref["path"])
    if (not path.is_absolute() or path != path.resolve() or not path.is_file()
            or file_sha(path) != ref.get("sha256")):
        raise ValueError("MPS_REPORT_REFERENCE_CHANGED")
    return json.loads(path.read_text())


def preparation_identity(frozen: dict) -> str:
    """与运行合同同配方独立计算，门收据绑定实际准备身份。"""
    return canonical_sha({key: frozen[key] for key in (
        "inputs", "source_files", "protected_files", "model_files", "packages", "settings", "runtime")})


def percentile(values: list[float], fraction: float) -> float | None:
    """对逐题中位数采用(n−1)p线性插值，不扩大独立样本量。"""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def distribution(values: list[float]) -> dict:
    """保留描述性分布和分母，不声称统计显著性。"""
    return {"n": len(values), "p50": percentile(values, .5), "p95": percentile(values, .95),
            "minimum": min(values) if values else None, "maximum": max(values) if values else None,
            "mean": statistics.mean(values) if values else None}


def quality(ranked: list[str], candidates: list[str], gold: set[str]) -> dict:
    """非穷尽正例按原分母计算，缺证题保留N/A。"""
    if not gold:
        return {**dict.fromkeys(METRICS), "first_gold_rank": None}
    first = next((i for i, cid in enumerate(ranked[:24], 1) if cid in gold), None)
    return {**{f"recall{k}": len(set(ranked[:k]) & gold) / len(gold) for k in (5, 8, 10)},
            "mrr24": 1 / first if first else 0.,
            "candidate_recall24": len(set(candidates) & gold) / len(gold),
            "first_gold_rank": first}


def cell_key(row: dict) -> tuple:
    """用格身份识别重复记录，不依赖文件名。"""
    return row["phase"], row.get("repetition"), row["case_id"], row["variant"]


def mps_device(value) -> bool:
    """允许设备展示的规范mps及mps:0两种形式。"""
    return value in {"mps", "mps:0"}


def validate_device_proof(proof: dict, tensors: dict, variant: str) -> None:
    """逐批核设备、dtype、回读张量和同步；输入迁移本身不作为执行完成。"""
    dtype = "torch.float16" if variant == "mps_fp16" else "torch.float32"
    names = set(tensors)
    devices, dtypes = proof.get("input_devices", {}), proof.get("input_dtypes", {})
    parameter_devices, parameter_dtypes = proof.get("model_parameter_devices"), proof.get("model_parameter_dtypes")
    if (set(devices) != names or set(dtypes) != names
            or not all(mps_device(value) for value in devices.values())
            or not all(value == "torch.int64" for value in dtypes.values())
            or not isinstance(parameter_devices, list) or not parameter_devices
            or not all(mps_device(value) for value in parameter_devices)
            or not isinstance(parameter_dtypes, list) or not parameter_dtypes
            or any(value != dtype for value in parameter_dtypes)
            or proof.get("transferred_tensor_hashes") != tensors
            or proof.get("sync_before") is not True or proof.get("sync_after") is not True
            or not mps_device(proof.get("output_device")) or proof.get("output_dtype") != dtype):
        raise ValueError("MPS_REPORT_DEVICE_OR_SYNCHRONIZATION_NOT_VERIFIED")


def validate_row(row: dict, case: dict, feed: dict, spec: dict, float_spec: dict) -> None:
    """从原始输入与实际证明重核成功格，拒绝仅凭status声明质量。"""
    if row.get("status") not in STATUSES:
        raise ValueError("MPS_REPORT_UNKNOWN_STATUS")
    ids = [c["chunk_id"] for c in case["candidates"]]
    if (row.get("query") != case["query"] or row.get("scope_document_ids") != case["scope_document_ids"]
            or row.get("candidate_ids") != ids or row.get("input_sha256") != case["input_sha256"]
            or row.get("feed_projection_sha256") != canonical_sha(feed)):
        raise ValueError("MPS_REPORT_RAW_INPUT_CHANGED")
    if row["status"] != "executed":
        return
    if row.get("error") is not None or row.get("measurement") != MEASUREMENT:
        raise ValueError("MPS_REPORT_SUCCESS_OR_MEASUREMENT_CHANGED")
    variant, model = row["variant"], row.get("model_identity") or {}
    model_spec = spec if variant == "ort_cpu4" else float_spec
    if (model.get("model_id") != spec["repo"] or model.get("revision") != spec["revision"]
            or model.get("weights_sha256") != model_spec["files"][model_spec["weights"]]["sha256"]
            or model.get("tokenizer_sha256") != spec["files"]["tokenizer.json"]["sha256"]
            or model.get("batch_size") != 4 or model.get("sort_by_length") is not False
            or model.get("max_pair_tokens") != 512 or model.get("threads") != 4
            or model.get("score_kind") != "raw_relevance_logit"):
        raise ValueError("MPS_REPORT_MODEL_IDENTITY_CHANGED")
    session = row.get("session_identity") or {}
    if variant == "ort_cpu4":
        if (session.get("providers") != ["CPUExecutionProvider"]
                or session.get("intra_op_num_threads") != 4 or session.get("inter_op_num_threads") != 1):
            raise ValueError("MPS_REPORT_CPU_SESSION_CHANGED")
    else:
        dtype = "torch.float16" if variant == "mps_fp16" else "torch.float32"
        if (session.get("backend") != "pytorch_mps" or session.get("cpu_fallback_enabled") is not False
                or not mps_device(session.get("device")) or session.get("dtype") != dtype
                or model.get("dtype") != dtype or not mps_device(model.get("device"))
                or not model.get("parameter_devices") or not model.get("parameter_dtypes")
                or not all(mps_device(v) for v in model["parameter_devices"])
                or not all(v == dtype for v in model["parameter_dtypes"])
                or session.get("intra_op_num_threads") != 4 or session.get("inter_op_num_threads") != 1):
            raise ValueError("MPS_REPORT_MPS_SESSION_CHANGED")
    names = session.get("input_names")
    if (not isinstance(names, list) or len(names) != len(set(names))
            or not {"input_ids", "attention_mask"}.issubset(names)
            or not set(names).issubset({"input_ids", "attention_mask", "token_type_ids"})
            or (variant != "ort_cpu4" and set(names) != {"input_ids", "attention_mask"})):
        raise ValueError("MPS_REPORT_INPUT_NAMES_CHANGED")
    verified = row.get("feed_verification") or {}
    count = len(feed["batches"])
    if (verified.get("complete") is not True or any(verified.get(key) != count for key in (
            "expected_batches", "verified_batches", "session_run_attempts", "session_run_successes"))
            or not isinstance(verified.get("batches"), list) or len(verified["batches"]) != count):
        raise ValueError("MPS_REPORT_ACTUAL_FEEDS_INCOMPLETE")
    for index, (actual, expected) in enumerate(zip(verified["batches"], feed["batches"])):
        tensors = actual.get("actual_tensors", {})
        if (actual.get("batch_index") != index or actual.get("candidate_indices") != expected["candidate_indices"]
                or actual.get("verification_passed") is not True or actual.get("native_attempted") is not True
                or actual.get("native_succeeded") is not True or actual.get("error") is not None
                or set(tensors) != set(names)):
            raise ValueError("MPS_REPORT_ACTUAL_BATCH_CHANGED")
        for name, ref in tensors.items():
            expected_name = "token_type_ids_if_required" if name == "token_type_ids" else name
            if ref != expected["tensors"][expected_name]:
                raise ValueError("MPS_REPORT_ACTUAL_TENSOR_CHANGED")
        if variant != "ort_cpu4":
            validate_device_proof(actual.get("device_proof", {}), tensors, variant)
    scores = row.get("scores")
    if not isinstance(scores, list) or len(scores) != len(ids) or not all(map(finite, scores)):
        raise ValueError("MPS_REPORT_SCORES_INVALID")
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
    if row.get("ranked_ids") != [ids[i] for i in order]:
        raise ValueError("MPS_REPORT_RANKS_CHANGED")
    hits = [{"chunk_id": case["candidates"][i]["chunk_id"],
             "document_id": case["candidates"][i]["document_id"],
             "text_sha256": case["candidates"][i]["text_sha256"],
             "rank": rank, "candidate_rank": i + 1, "reranker_score": scores[i]}
            for rank, i in enumerate(order, 1)]
    if row.get("hits") != hits:
        raise ValueError("MPS_REPORT_HIT_IDENTITY_CHANGED")
    timing = row.get("timing") or {}
    if any(not finite(timing.get(key)) or timing[key] < 0 for key in TIMINGS):
        raise ValueError("MPS_REPORT_TIMING_INVALID")


def load_context(freeze_path: Path, expected_hash: str, phase: str) -> dict:
    """先核事前锚点，再读取同输入和独立标签；模型仅按字节散列。"""
    if file_sha(freeze_path) != expected_hash:
        raise ValueError("MPS_REPORT_FREEZE_ANCHOR_CHANGED")
    frozen = json.loads(freeze_path.read_text())
    if frozen.get("schema") != "reranker-mps-execution-v1" or phase not in VARIANTS:
        raise ValueError("MPS_REPORT_SCHEMA_OR_PHASE_CHANGED")
    refs = (list(frozen["inputs"].values()) + frozen["source_files"]
            + frozen["protected_files"] + frozen["model_files"])
    for ref in refs:
        path = Path(ref["path"])
        if not path.is_absolute() or path != path.resolve() or file_sha(path) != ref["sha256"]:
            raise ValueError("MPS_REPORT_FROZEN_FILE_CHANGED")
    inputs = {name: read_reference(ref) for name, ref in frozen["inputs"].items()}
    cases = inputs["cases"]
    case_map = {case["case_id"]: case for case in cases}
    feed_rows = inputs["feed_manifest"]["cases"]
    feeds = {row["case_id"]: row for row in feed_rows}
    metadata_rows = read_reference(frozen["report_inputs"]["original_questions"])
    metadata = {row["case_id"]: row for row in metadata_rows}
    gold = defaultdict(set)
    for pair in read_reference(frozen["report_inputs"]["labels"])["positive_pairs"]:
        gold[pair["case_id"]].add(pair["chunk_id"])
    if (len(case_map) != len(cases) or len(feeds) != len(feed_rows) or len(metadata) != len(metadata_rows)
            or set(case_map) != set(feeds) or set(case_map) != set(metadata) or set(gold) - set(case_map)):
        raise ValueError("MPS_REPORT_CASE_SET_CHANGED")
    for case in cases:
        cid = case["case_id"]
        if (canonical_sha({k: v for k, v in case.items() if k != "input_sha256"}) != case["input_sha256"]
                or feeds[cid]["input_sha256"] != case["input_sha256"]
                or metadata[cid]["query"] != case["query"]
                or metadata[cid]["scope_document_ids"] != case["scope_document_ids"]
                or metadata[cid]["answerable"] != bool(gold[cid])):
            raise ValueError("MPS_REPORT_CASE_METADATA_CHANGED")
        ids = [c["chunk_id"] for c in case["candidates"]]
        if len(ids) != len(set(ids)) or not 1 <= len(ids) <= 24:
            raise ValueError("MPS_REPORT_CANDIDATES_CHANGED")
        for candidate in case["candidates"]:
            if (candidate["document_id"] not in case["scope_document_ids"]
                    or hashlib.sha256(candidate["text"].encode()).hexdigest() != candidate["text_sha256"]):
                raise ValueError("MPS_REPORT_TEXT_OR_SCOPE_CHANGED")
    pilot_ids = inputs["pilot_selection"]["case_ids"]
    if not pilot_ids or len(set(pilot_ids)) != len(pilot_ids) or not set(pilot_ids).issubset(case_map):
        raise ValueError("MPS_REPORT_PILOT_SELECTION_CHANGED")
    selected = [case_map[cid] for cid in pilot_ids] if phase == "pilot" else cases
    spec, float_spec = inputs["model_manifest"]["reranker"], inputs["float_manifest"]
    if (float_spec["repo"] != spec["repo"] or float_spec["revision"] != spec["revision"]):
        raise ValueError("MPS_REPORT_FLOAT_CHECKPOINT_CHANGED")
    return {"frozen": frozen, "cases": selected, "all_cases": case_map, "feeds": feeds,
            "metadata": metadata, "gold": gold, "spec": spec, "float_spec": float_spec}


def aggregate(rows: list[dict]) -> dict:
    """不筛掉失败正例，未知和未执行使正式均值保持N/A。"""
    known = [row for row in rows if row["mrr24"] is not None]
    complete = bool(rows) and len(known) == len(rows)
    return {"planned_n": len(rows), "known_n": len(known), "complete": complete,
            **{name: statistics.mean(row[name] for row in known) if complete else None for name in METRICS},
            "observed_mean": {name: statistics.mean(row[name] for row in known) if known else None for name in METRICS},
            "first_hot_statuses": dict(Counter(row["status"] for row in rows))}


def score_difference(a: dict | None, b: dict | None) -> dict:
    """逐候选呈现数值和排名变化，缺格不构造零差。"""
    if not a or not b or a["status"] != "executed" or b["status"] != "executed":
        return {"available": False}
    if a["candidate_ids"] != b["candidate_ids"]:
        raise ValueError("MPS_REPORT_PAIR_IDENTITY_CHANGED")
    ar = {cid: i for i, cid in enumerate(a["ranked_ids"], 1)}
    br = {cid: i for i, cid in enumerate(b["ranked_ids"], 1)}
    return {"available": True, "ranking_changed": a["ranked_ids"] != b["ranked_ids"],
            "scores_changed": a["scores"] != b["scores"],
            "entries": [{"chunk_id": cid, "reference_score": x, "candidate_score": y,
                         "delta": y - x, "reference_rank": ar[cid], "candidate_rank": br[cid]}
                        for cid, x, y in zip(a["candidate_ids"], a["scores"], b["scores"])]}


def quality_comparison(per_case: list[dict], reference_variant: str) -> dict:
    """按全部计划正例比较两变体，不用共同成功子集掩盖失败。"""
    candidate = [row for row in per_case if row["answerable"] and row["variant"] == "mps_fp16"]
    other = {row["case_id"]: row for row in per_case if row["variant"] == reference_variant}
    pairs = [(other[row["case_id"]], row) for row in candidate]
    known = [(a, b) for a, b in pairs if a["status"] == b["status"] == "executed"]
    complete = bool(pairs) and len(known) == len(pairs)
    ref = {m: statistics.mean(a[m] for a, _ in known) if complete else None for m in GATE_METRICS}
    cand = {m: statistics.mean(b[m] for _, b in known) if complete else None for m in GATE_METRICS}
    deltas = {m: cand[m] - ref[m] if complete else None for m in GATE_METRICS}
    return {"complete": complete, "planned_n": len(pairs), "known_n": len(known),
            "reference": ref, "candidate": cand, "deltas": deltas,
            "non_decreasing": complete and all(value >= -THRESHOLDS["metric_tolerance"] for value in deltas.values())}


def write_new(path: Path, value, *, raw: bool = False) -> None:
    """独占写报告，所有旧实验原件保持只读。"""
    text = value if raw else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def build_report(freeze_path: Path, execution: Path, output: Path, *,
                 expected_freeze_sha256: str, phase: str) -> dict:
    """独立验证原件、设备与同步证明，再生成质量门和完整报告。"""
    if not output.is_absolute() or output != output.resolve() or not output.is_dir() or any(output.iterdir()):
        raise ValueError("MPS_REPORT_OUTPUT_MUST_BE_NEW_EMPTY_REAL_DIRECTORY")
    context = load_context(freeze_path, expected_freeze_sha256, phase)
    frozen, cases = context["frozen"], context["cases"]
    variants, repetitions = VARIANTS[phase], 1 if phase == "pilot" else 3
    plan_path = execution / "execution-plan.json"
    plan_wrapper = json.loads(plan_path.read_text())
    if (plan_wrapper.get("freeze_sha256") != expected_freeze_sha256
            or plan_wrapper.get("preparation_identity_sha256") != preparation_identity(frozen)
            or plan_wrapper.get("measurement") != MEASUREMENT or plan_wrapper.get("labels_parsed") is not False):
        raise ValueError("MPS_REPORT_PLAN_IDENTITY_CHANGED")
    plan = plan_wrapper["plan"]
    cells = plan["cells"]
    expected = {cell_key(row) for row in cells}
    intended = {(stage, repeat, case["case_id"], variant)
                for stage, repeat in [("hot", r) for r in range(repetitions)]
                for case in cases for variant in variants}
    intended |= {("warmup", None, cases[0]["case_id"], variant) for variant in variants}
    planned_pairs = sum(len(context["all_cases"][row["case_id"]]["candidates"]) for row in cells)
    planned_batches = sum(len(context["feeds"][row["case_id"]]["batches"]) for row in cells)
    if (len(expected) != len(cells) or expected != intended or plan.get("phase") != phase
            or plan.get("variants") != list(variants) or plan.get("hot_repeats") != repetitions
            or plan.get("case_ids") != [case["case_id"] for case in cases]
            or plan.get("budget", {}).get("max_score_attempts") != len(cells)
            or plan["budget"].get("max_pairs") != planned_pairs
            or plan["budget"].get("max_native_batches") != planned_batches):
        raise ValueError("MPS_REPORT_EXECUTION_PLAN_CHANGED")
    raw, refs = {}, [reference(freeze_path), reference(plan_path)]
    for path in sorted(execution.glob("*.json")):
        if path.name in {"execution-plan.json", "terminal.json", "initialization.json"}:
            continue
        row = json.loads(path.read_text())
        if not isinstance(row, dict) or row.get("phase") not in {"warmup", "hot"}:
            continue
        key = cell_key(row)
        if key not in expected or key in raw:
            raise ValueError("MPS_REPORT_DUPLICATE_OR_UNPLANNED_CELL")
        case = context["all_cases"][row["case_id"]]
        validate_row(row, case, context["feeds"][row["case_id"]], context["spec"], context["float_spec"])
        raw[key] = row
        refs.append(reference(path))
    terminal_path = execution / "terminal.json"
    terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
    if terminal is not None:
        refs.append(reference(terminal_path))
        if (terminal.get("freeze_sha256") != expected_freeze_sha256
                or terminal.get("preparation_identity_sha256") != preparation_identity(frozen)
                or terminal.get("phase") != phase or terminal.get("labels_parsed") is not False
                or terminal.get("measurement") != MEASUREMENT):
            raise ValueError("MPS_REPORT_TERMINAL_IDENTITY_CHANGED")
    observed = [{**row, "observed": cell_key(row) in raw,
                 "state": raw[cell_key(row)]["status"] if cell_key(row) in raw else "unexecuted"}
                for row in cells]
    all_executed = (bool(terminal) and terminal.get("status") == "complete"
                    and len(raw) == len(cells) and all(row["status"] == "executed" for row in raw.values()))
    if all_executed:
        expected_counts = {"score_attempts": len(cells), "successful_score_calls": len(cells),
            "pair_attempts": planned_pairs, "successful_pairs": planned_pairs, "failed_score_calls": 0,
            "session_run_attempts": planned_batches, "session_run_successes": planned_batches,
            "verified_batches": planned_batches, "native_batch_reservations": planned_batches,
            "hot_rows": sum(cell["phase"] == "hot" for cell in cells),
            "warmup_rows": sum(cell["phase"] == "warmup" for cell in cells)}
        if (terminal.get("unexecuted") != [] or terminal.get("error") is not None
                or terminal.get("prepare_only") is not False
                or any(terminal.get("counters", {}).get(k) != v for k, v in expected_counts.items())):
            raise ValueError("MPS_REPORT_COMPLETE_COUNTERS_CHANGED")
    per_case, comparisons, timing_pairs = [], [], []
    for case in cases:
        cid, ids = case["case_id"], [c["chunk_id"] for c in case["candidates"]]
        for variant in variants:
            repeated = [raw.get(("hot", r, cid, variant)) for r in range(repetitions)]
            first = repeated[0]
            status = first["status"] if first else "unexecuted"
            if first and status in {"executed", "failed", "empty"}:
                measured = quality(first["ranked_ids"] if status == "executed" else [], ids, context["gold"][cid])
            else:
                measured = {**dict.fromkeys(METRICS), "first_gold_rank": None}
            stable_eligible = all(row and row["status"] == "executed" for row in repeated)
            timings = {key: statistics.median(row["timing"][key] for row in repeated) for key in TIMINGS} if stable_eligible else None
            per_case.append({"case_id": cid, "variant": variant, "status": status,
                "split": context["metadata"][cid].get("split", "unknown"), "answerable": bool(context["gold"][cid]),
                "candidate_count": len(ids), **measured, "all_repeats_executed": stable_eligible,
                "ranking_stable": all(row["ranked_ids"] == first["ranked_ids"] for row in repeated) if stable_eligible and repetitions > 1 else None,
                "scores_stable": all(row["scores"] == first["scores"] for row in repeated) if stable_eligible and repetitions > 1 else None,
                "median_timing": timings})
        for ref_variant in ("ort_cpu4", "mps_fp32") if phase == "pilot" else ("ort_cpu4",):
            comparison = score_difference(raw.get(("hot", 0, cid, ref_variant)), raw.get(("hot", 0, cid, "mps_fp16")))
            comparisons.append({"case_id": cid, "reference_variant": ref_variant, "candidate_variant": "mps_fp16", **comparison})
        a = next(row for row in per_case if row["case_id"] == cid and row["variant"] == "ort_cpu4")
        b = next(row for row in per_case if row["case_id"] == cid and row["variant"] == "mps_fp16")
        timing_pairs.append({"case_id": cid, "answerable": a["answerable"], "split": a["split"],
            "fair_timing_eligible": a["all_repeats_executed"] and b["all_repeats_executed"],
            "reference_timing": a["median_timing"], "candidate_timing": b["median_timing"]})
    positives = [case for case in cases if context["gold"][case["case_id"]]]
    aggregates = []
    splits = ["all", *sorted({row["split"] for row in per_case if row["answerable"]})]
    for variant in variants:
        for split in splits:
            selected = [row for row in per_case if row["variant"] == variant and row["answerable"]
                        and (split == "all" or row["split"] == split)]
            aggregates.append({"variant": variant, "split": split, **aggregate(selected)})
    paired = {"planned_n": len(positives), "n": 0, "excluded_n": len(positives)}
    good = [row for row in timing_pairs if row["answerable"] and row["fair_timing_eligible"]]
    paired.update(n=len(good), excluded_n=len(positives) - len(good))
    for key in TIMINGS:
        a, b = [row["reference_timing"][key] for row in good], [row["candidate_timing"][key] for row in good]
        paired[key] = {"ort_cpu4": distribution(a), "mps_fp16": distribution(b),
            "paired_delta_seconds": distribution([y - x for x, y in zip(a, b)]),
            "paired_ratio": distribution([y / x for x, y in zip(a, b) if x > 0]),
            "faster_cases": sum(y < x for x, y in zip(a, b)), "equal_cases": sum(y == x for x, y in zip(a, b)),
            "slower_cases": sum(y > x for x, y in zip(a, b))}
    quality_comparisons = {variant: quality_comparison(per_case, variant)
                           for variant in (("ort_cpu4", "mps_fp32") if phase == "pilot" else ("ort_cpu4",))}
    numeric_comparisons = [row for row in comparisons if row["reference_variant"] == "mps_fp32"]
    numeric_deltas = [abs(e["delta"]) for row in numeric_comparisons if row["available"] for e in row["entries"]]
    numeric_complete = phase == "pilot" and len(numeric_comparisons) == len(cases) and all(row["available"] for row in numeric_comparisons)
    numeric = {"complete": numeric_complete, "pairs": len(numeric_deltas),
               "mae": statistics.mean(numeric_deltas) if numeric_complete and numeric_deltas else None,
               "max_abs": max(numeric_deltas) if numeric_complete and numeric_deltas else None}
    summary = {"schema": "reranker-mps-report-v1", "phase": phase,
        "freeze_sha256": expected_freeze_sha256, "preparation_identity_sha256": preparation_identity(frozen),
        "planned_cells": len(cells), "observed_cells": len(raw), "states": dict(Counter(row["state"] for row in observed)),
        "unexecuted_cells": len(cells) - len(raw), "all_cells_executed": all_executed,
        "answerable_cases": len(positives), "gap_cases": len(cases) - len(positives), "independently_certified": 0,
        "aggregates": aggregates, "paired_timing": paired, "fp16_vs_fp32_numeric": numeric,
        "quality_comparisons": quality_comparisons,
        "stability": {variant: {"planned_cases": len(cases), "repetitions": repetitions,
            "assessment_available": repetitions > 1,
            "known_cases": sum(row["variant"] == variant and row["all_repeats_executed"] for row in per_case),
            "ranking_changed_cases": sum(row["variant"] == variant and row["ranking_stable"] is False for row in per_case),
            "scores_changed_cases": sum(row["variant"] == variant and row["scores_stable"] is False for row in per_case)} for variant in variants},
        "planned_pairs": planned_pairs, "planned_native_batches": planned_batches,
        "observed_native_attempts": sum(batch.get("native_attempted") is True for row in raw.values()
            for batch in (row.get("feed_verification") or {}).get("batches", [])),
        "observed_native_successes": sum(batch.get("native_succeeded") is True for row in raw.values()
            for batch in (row.get("feed_verification") or {}).get("batches", [])),
        "terminal": terminal, "automatic_adoption": False,
        "limits": ["Original labels are non-exhaustive and historically exposed; independently certified cases 0.",
            "Pilot selection is fixed before scoring, based on input metadata rather than labels or results.",
            "All successful cells have independently checked input, device, dtype and synchronization proofs.",
            "Explicit MPS CPU fallback is disabled; these proofs do not substitute for a full Metal kernel trace.",
            "ORT INT8 CPU4 versus PyTorch FP16 MPS changes both backend and precision; not pure hardware causality.",
            "Wall time includes tokenization, feed verification, synchronization, scoring and sorting; no overhead subtraction.",
            "Failed/empty first-hot cells retain zero ranking quality; unknown/stopped/unexecuted remain N/A.",
            "A small pilot gate and synthetic tests do not establish full-set quality or production capacity."]}
    if phase == "pilot":
        checks = {"complete_execution": all_executed, "verified_actual_feeds": all_executed,
            "mps_device_and_dtype_verified": all_executed,
            "fp16_numeric_within_limits": bool(numeric_complete and numeric["mae"] is not None
                and numeric["mae"] <= THRESHOLDS["mae_max"] and numeric["max_abs"] <= THRESHOLDS["max_abs_max"]),
            "fp16_quality_non_decreasing_vs_ort_cpu4": quality_comparisons["ort_cpu4"]["non_decreasing"],
            "fp16_quality_non_decreasing_vs_mps_fp32": quality_comparisons["mps_fp32"]["non_decreasing"]}
        gate = {"schema": "reranker-mps-pilot-gate-v1", "phase": "pilot", "pass": all(checks.values()),
            "preparation_identity_sha256": preparation_identity(frozen), "freeze_sha256": expected_freeze_sha256,
            "thresholds": THRESHOLDS, "checks": checks, "numeric": numeric,
            "quality_comparisons": quality_comparisons, "reference_files": refs,
            "automatic_adoption": False,
            "meaning": "Operational gate for this frozen experiment only; numeric thresholds are not industry standards."}
        summary["pilot_gate"] = {key: value for key, value in gate.items() if key != "reference_files"}
        write_new(output / "pilot-gate.json", gate)
    for name, value in (("summary.json", summary), ("per-case.json", per_case),
                        ("paired-cases.json", timing_pairs), ("planned-cells.json", observed),
                        ("score-changes.json", comparisons), ("raw-references.json", refs)):
        write_new(output / name, value)
    flat = [{"case_id": row["case_id"], "answerable": row["answerable"], "split": row["split"],
             "fair_timing_eligible": row["fair_timing_eligible"],
             "ort_cpu4_wall_seconds": row["reference_timing"][TIMINGS[0]] if row["reference_timing"] else None,
             "mps_fp16_wall_seconds": row["candidate_timing"][TIMINGS[0]] if row["candidate_timing"] else None}
            for row in timing_pairs]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(flat[0]) if flat else ["case_id"])
    writer.writeheader()
    writer.writerows(flat)
    write_new(output / "comparison.csv", "\ufeff" + stream.getvalue(), raw=True)
    write_new(output / "report.md", render_markdown(summary), raw=True)
    return summary


def render_markdown(summary: dict) -> str:
    """呈现原分母、N/A和精度变化，避免把局部门说成部署验收。"""
    lines = ["# 同版本浮点重排器MPS实验", "", f"阶段：{summary['phase']}；已观察{summary['observed_cells']}/{summary['planned_cells']}格。",
        f"状态：{summary['states']}；缺证题{summary['gap_cases']}，独立认证题0。", "",
        "| 变体/分组 | 已知/计划正例 | Recall@5 | Recall@8 | Recall@10 | MRR@24 | 候选覆盖@24 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["aggregates"]:
        values = ["N/A" if row[m] is None else f"{row[m]:.4f}" if m == "mrr24" else f"{100 * row[m]:.2f}%" for m in METRICS]
        lines.append(f"| {row['variant']}/{row['split']} | {row['known_n']}/{row['planned_n']} | " + " | ".join(values) + " |")
    lines += ["", "| 同期重排组件墙钟 | P50秒 | P95秒 |", "|---|---:|---:|"]
    for variant in ("ort_cpu4", "mps_fp16"):
        row = summary["paired_timing"][TIMINGS[0]][variant]
        lines.append(f"| {variant} | " + " | ".join("N/A" if row[k] is None else f"{row[k]:.6f}" for k in ("p50", "p95")) + " |")
    if "pilot_gate" in summary:
        lines += ["", f"小样本门通过：{summary['pilot_gate']['pass']}；FP16数值检查：{summary['fp16_vs_fp32_numeric']}。",
            "该阈值是本次事前工程条件，不是行业标准；小样本通过不代表全量质量已验证。"]
    lines += ["", "ORT INT8 CPU4与PyTorch FP16 MPS同时改变后端和精度，不把全部差异归因于GPU。",
        "主耗时含分词、实际feed核验、同步、评分与排序；不扣除校验开销，不等于完整RAG热查询时间。",
        "失败保留原分母，未知/未执行保持N/A；缺证题仅作诊断。模型参数与输入设备证明不替代完整Metal逐kernel追踪。",
        "本报告不自动采用、合并或部署。", "",
        "[汇总](summary.json) · [逐题](per-case.json) · [配对](comparison.csv) · [分数变化](score-changes.json) · [原件](raw-references.json)"]
    return "\n".join(lines) + "\n"


def main() -> None:
    """显式读取冻结与已有结果，只向新空目录写报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.freeze.resolve(), args.execution.resolve(), args.output.resolve(),
                          expected_freeze_sha256=args.expected_freeze_sha256, phase=args.phase)
    print(json.dumps({"phase": args.phase, "observed": result["observed_cells"],
                      "planned": result["planned_cells"], "all_executed": result["all_cells_executed"],
                      "pilot_gate": result.get("pilot_gate", {}).get("pass")}))


if __name__ == "__main__":
    main()
