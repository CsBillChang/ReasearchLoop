"""仅从冻结材料和原始格复算MLX前向资格、W8质量与同期耗时。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from scripts.compare_lexical_retrieval import digest
from scripts.compare_reranker_mlx_int8 import (
    PHASES, SCHEMA, THRESHOLDS, VARIANTS, all_references, cell_key, check_reference,
    file_reference, finite_number, historical_rows, preparation_identity, read_execution,
    score_errors, validate_gate,
)
from scripts.report_reranker_mps import aggregate, distribution, quality
from scripts.retrieval_experiment_guard import BoundaryViolation, write_json_new


METRICS = ("recall5", "recall8", "recall10", "mrr24", "candidate_recall24")


def comparison(reference: list[dict], candidate: list[dict]) -> dict:
    """只在原分母完整时比较，不以缺失格制造零或筛选后均值。"""
    a, b = aggregate(reference), aggregate(candidate)
    complete = a["complete"] and b["complete"] and a["planned_n"] == b["planned_n"]
    values = {"complete": complete, "planned_n": b["planned_n"], "known_n": b["known_n"],
              "reference": {k: a[k] for k in METRICS}, "candidate": {k: b[k] for k in METRICS},
              "deltas": {k: b[k] - a[k] if complete else None for k in METRICS}}
    values["non_decreasing"] = (all(values["deltas"][k] >= -THRESHOLDS["metric_tolerance"]
                                    for k in ("recall5", "recall8", "recall10", "mrr24"))
                                if complete else None)
    return values


def load_labels(frozen: dict, cases: list) -> tuple:
    """标签由独立报告器读取，严格保持问题、scope和正例分母。"""
    labels = json.loads(check_reference(frozen["report_inputs"]["labels"]).read_text())
    questions = json.loads(check_reference(frozen["report_inputs"]["original_questions"]).read_text())
    metadata = {row["case_id"]: row for row in questions}
    gold = defaultdict(set)
    for row in labels["positive_pairs"]:
        gold[row["case_id"]].add(row["chunk_id"])
    if (len(metadata) != len(questions) or set(metadata) != {c["case_id"] for c in cases}
            or set(gold) - set(metadata)):
        raise BoundaryViolation("MLX_REPORT_LABEL_CASE_SET_CHANGED")
    for case in cases:
        item = metadata[case["case_id"]]
        if (item["query"] != case["query"] or item["scope_document_ids"] != case["scope_document_ids"]
                or item["answerable"] != bool(gold[case["case_id"]])):
            raise BoundaryViolation("MLX_REPORT_LABEL_IDENTITY_CHANGED")
    if len([c for c in cases if gold[c["case_id"]]]) != 112:
        raise BoundaryViolation("MLX_REPORT_POSITIVE_DENOMINATOR_CHANGED")
    return metadata, gold


def measure_case(case: dict, row: dict | None, gold: set, variant: str, split: str) -> dict:
    status = row["status"] if row else "not_executed"
    scores = quality(row["ranked_ids"], row["candidate_ids"], gold) if status == "executed" else {
        **dict.fromkeys(METRICS), "first_gold_rank": None}
    return {"case_id": case["case_id"], "variant": variant, "status": status,
            "split": split, "answerable": bool(gold), "candidate_count": len(case["candidates"]), **scores}


def checked_timing(row: dict) -> dict:
    values = row.get("timing", {})
    names = ("reranker_wall_seconds", "provider_inference_seconds", "feed_verification_seconds", "native_session_run_seconds")
    if any(not finite_number(values.get(k)) or values[k] < 0 for k in names):
        raise BoundaryViolation("MLX_REPORT_TIMING_INVALID")
    if values["reranker_wall_seconds"] <= 0:
        raise BoundaryViolation("MLX_REPORT_TIMING_NONPOSITIVE")
    return values


def unique_references(refs: list[dict]) -> list[dict]:
    by_path = {}
    for ref in refs:
        if ref["path"] in by_path and by_path[ref["path"]] != ref:
            raise BoundaryViolation("MLX_REPORT_REFERENCE_COLLISION")
        by_path[ref["path"]] = ref
    return sorted(by_path.values(), key=lambda r: r["path"])


def report(freeze_path: Path, expected_freeze_sha256: str, execution: Path, output: Path, *,
           previous_gate: Path | None = None, expected_previous_gate_sha256: str | None = None) -> dict:
    """阶段完成或停止均可报告，只有完整合格阶段才能生成可放行门票。"""
    if digest(freeze_path) != expected_freeze_sha256:
        raise BoundaryViolation("MLX_REPORT_FREEZE_ANCHOR_CHANGED")
    frozen = json.loads(freeze_path.read_text())
    if frozen.get("schema") != SCHEMA:
        raise BoundaryViolation("MLX_REPORT_SCHEMA_CHANGED")
    terminal = json.loads((execution / "terminal.json").read_text())
    phase = terminal["phase"]
    if phase not in PHASES:
        raise BoundaryViolation("MLX_REPORT_PHASE_CHANGED")
    plan, terminal, rows, refs, cases, feeds, pilot_ids, inputs = read_execution(
        execution, frozen, expected_freeze_sha256, phase)
    metadata, gold = load_labels(frozen, cases)
    selected = plan["case_ids"]
    by_id = {c["case_id"]: c for c in cases}
    previous = None
    if phase in {"pilot", "performance"}:
        if previous_gate is None or expected_previous_gate_sha256 is None:
            raise BoundaryViolation("MLX_REPORT_PREVIOUS_GATE_REQUIRED")
        expected_phase = "float_qualification" if phase == "pilot" else "pilot"
        previous = validate_gate(previous_gate, expected_previous_gate_sha256, frozen,
                                 expected_freeze_sha256, expected_phase)
        if terminal.get("previous_gate_sha256") != expected_previous_gate_sha256:
            raise BoundaryViolation("MLX_REPORT_PREVIOUS_GATE_BINDING_CHANGED")
    complete = (terminal.get("status") == "complete" and terminal.get("prepare_only") is False
                and terminal.get("unexecuted") == [] and len(rows) == len(plan["cells"])
                and all(row["status"] == "executed" for row in rows.values()))
    if complete:
        for actual, expected in (("score_attempts", "max_score_attempts"), ("pair_attempts", "max_pairs"),
                                 ("native_batch_reservations", "max_native_batches"),
                                 ("session_run_attempts", "max_native_batches"),
                                 ("session_run_successes", "max_native_batches")):
            if terminal["counters"].get(actual) != plan["budget"][expected]:
                raise BoundaryViolation("MLX_REPORT_COUNTERS_CHANGED")
    per_case, aggregates, stability = [], [], {}
    for variant in plan["variants"]:
        same_variant = []
        for cid in selected:
            expected_repeats = plan["hot_repeats"]
            runs = [rows.get(("hot", i, cid, variant)) for i in range(expected_repeats)]
            measured = measure_case(by_id[cid], runs[0], gold[cid], variant, metadata[cid]["split"])
            good = all(row is not None and row["status"] == "executed" for row in runs)
            measured["all_repeats_executed"] = good
            measured["ranking_stable"] = all(r["ranked_ids"] == runs[0]["ranked_ids"] for r in runs) if good else None
            measured["scores_stable"] = all(r["scores"] == runs[0]["scores"] for r in runs) if good else None
            measured["median_timing"] = {k: statistics.median(checked_timing(r)[k] for r in runs)
                                         for k in checked_timing(runs[0])} if good else None
            same_variant.append(measured)
            per_case.append(measured)
        for split in ("all", "regression", "pending"):
            positives = [r for r in same_variant if r["answerable"] and (split == "all" or r["split"] == split)]
            if positives:
                aggregates.append({"variant": variant, "split": split, **aggregate(positives)})
        stability[variant] = {"planned_cases": len(selected), "repetitions": plan["hot_repeats"],
            "known_cases": sum(r["all_repeats_executed"] for r in same_variant),
            "ranking_changed_cases": sum(r["ranking_stable"] is False for r in same_variant),
            "scores_changed_cases": sum(r["scores_stable"] is False for r in same_variant)}
    references = refs + [file_reference(freeze_path), file_reference(execution / "terminal.json"),
                         file_reference(execution / "execution-plan.json")]
    if previous is not None:
        references += [file_reference(previous_gate), *previous["reference_files"]]
    numeric = {"complete": False, "pairs": 0, "mae": None, "max_abs": None}
    comparisons = {}
    candidate_variant = plan["variants"][-1]
    candidate_quality = [r for r in per_case if r["variant"] == candidate_variant and r["answerable"]]
    if phase in {"float_qualification", "pilot"}:
        history = historical_rows(frozen, cases, feeds, pilot_ids)
        references += frozen["historical_pilot_files"]
        cpu_quality = [measure_case(by_id[cid], history[(cid, "ort_cpu4")], gold[cid],
                                   "ort_cpu4_historical", metadata[cid]["split"])
                       for cid in pilot_ids if gold[cid]]
        comparisons["ort_cpu4"] = comparison(cpu_quality, candidate_quality)
        if phase == "float_qualification":
            other = {cid: history[(cid, "mps_fp32")] for cid in pilot_ids}
        else:
            _, _, float_rows, _, _, _, _, _ = read_execution(Path(previous["execution_directory"]), frozen,
                                                             expected_freeze_sha256, "float_qualification")
            other = {cid: float_rows[("hot", 0, cid, "mlx_fp32")] for cid in pilot_ids}
        if complete:
            numeric = score_errors([s for cid in pilot_ids for s in rows[("hot", 0, cid, candidate_variant)]["scores"]],
                                   [s for cid in pilot_ids for s in other[cid]["scores"]])
        reference_quality = [measure_case(by_id[cid], other[cid], gold[cid], "float_reference",
                                         metadata[cid]["split"]) for cid in pilot_ids if gold[cid]]
        comparisons["float_reference"] = comparison(reference_quality, candidate_quality)
    else:
        comparisons["ort_cpu4"] = comparison([r for r in per_case if r["variant"] == "ort_cpu4" and r["answerable"]],
                                                candidate_quality)
    paired, paired_timing = [], None
    if phase == "performance":
        mapped = {(r["case_id"], r["variant"]): r for r in per_case}
        for cid in selected:
            a, b = mapped[(cid, "ort_cpu4")], mapped[(cid, "mlx_w8a32")]
            paired.append({"case_id": cid, "answerable": bool(gold[cid]), "split": metadata[cid]["split"],
                "fair_timing_eligible": bool(gold[cid]) and a["all_repeats_executed"] and b["all_repeats_executed"],
                "reference_timing": a["median_timing"], "candidate_timing": b["median_timing"]})
        fair = [r for r in paired if r["fair_timing_eligible"]]
        seconds = {v: [r[k]["reranker_wall_seconds"] for r in fair]
                   for v, k in (("ort_cpu4", "reference_timing"), ("mlx_w8a32", "candidate_timing"))}
        deltas = [b - a for a, b in zip(seconds["ort_cpu4"], seconds["mlx_w8a32"])]
        paired_timing = {"planned_n": 112, "n": len(fair), "excluded_n": 112 - len(fair),
            "complete": len(fair) == 112, "reranker_wall_seconds": {
                **{v: distribution(s) for v, s in seconds.items()}, "paired_delta_seconds": distribution(deltas),
                "faster_cases": sum(x < 0 for x in deltas), "equal_cases": sum(x == 0 for x in deltas),
                "slower_cases": sum(x > 0 for x in deltas)}}
    gate_pass = complete
    if phase == "float_qualification":
        gate_pass = gate_pass and numeric["mae"] <= THRESHOLDS["float_mae_max"] and numeric["max_abs"] <= THRESHOLDS["float_max_abs_max"]
    elif phase == "pilot":
        gate_pass = gate_pass and comparisons["ort_cpu4"]["non_decreasing"] is True
    summary = {"schema": "reranker-mlx-int8-report-v1", "phase": phase,
        "freeze_sha256": expected_freeze_sha256, "preparation_identity_sha256": preparation_identity(frozen),
        "planned_cells": len(plan["cells"]), "observed_cells": len(rows),
        "states": dict(Counter(r["status"] for r in rows.values())), "all_cells_executed": complete,
        "unexecuted_cells": len(plan["cells"]) - len(rows),
        "answerable_cases": sum(bool(gold[cid]) for cid in selected),
        "gap_cases": sum(not gold[cid] for cid in selected), "independently_certified": 0,
        "aggregates": aggregates, "quality_comparisons": comparisons, "numeric": numeric,
        "numeric_reference": "375_mps_fp32" if phase == "float_qualification" else "current_mlx_fp32" if phase == "pilot" else None,
        "stability": stability, "paired_timing": paired_timing, "budget": plan["budget"],
        "terminal": terminal, "gate_pass": gate_pass if phase != "performance" else None,
        "automatic_adoption": False,
        "limits": ["Original non-exhaustive historically exposed labels; independent certification remains zero.",
                   "W8A32 means only 74 Linear weights use 8-bit affine quantization; embedding/activation/norm/bias stay FP32.",
                   "One warmup is not a complete cold-start/JIT distribution; wall includes synchronization and verification.",
                   "CPU INT8 vs MLX W8A32 changes quantization scheme and backend; no pure hardware causality claim.",
                   "Historical MPS FP16 latency is not part of this same-run performance comparison.",
                   "Quality is computed by this frozen independent reporter from protected labels; the scorer binds its evidence and rechecks arithmetic/label-free numeric differences, without parsing labels."]}
    if not output.is_dir() or any(output.iterdir()):
        raise BoundaryViolation("MLX_REPORT_OUTPUT_MUST_BE_EMPTY")
    write_json_new(output / "summary.json", summary)
    write_json_new(output / "per-case.json", per_case)
    write_json_new(output / "paired-cases.json", paired)
    summary_ref, quality_ref, reporter_ref = (file_reference(path) for path in (
        output / "summary.json", output / "per-case.json", Path(__file__).resolve()))
    references += [summary_ref, quality_ref, reporter_ref]
    write_json_new(output / "raw-references.json", unique_references(references))
    if phase != "performance":
        gate = {"schema": "reranker-mlx-int8-gate-v1", "phase": phase, "pass": bool(gate_pass),
            "freeze_sha256": expected_freeze_sha256, "preparation_identity_sha256": preparation_identity(frozen),
            "thresholds": THRESHOLDS, "execution_directory": str(execution.resolve()),
            "reference_files": unique_references(references), "numeric": numeric,
            "quality_vs_cpu4": comparisons["ort_cpu4"], "automatic_adoption": False,
            "report_summary_ref": summary_ref, "quality_rows_ref": quality_ref,
            "reporter_source_ref": reporter_ref}
        if previous is not None:
            gate["previous_gate"] = file_reference(previous_gate)
        write_json_new(output / "gate.json", gate)
    lines = ["# MLX W8A32阶段复算", "", f"阶段：{phase}；完整执行：{complete}；后续资格：{summary['gate_pass']}。", "",
             f"原正例{summary['answerable_cases']}，缺证{summary['gap_cases']}；独立认证0。", "",
             "|变体/分组|Recall5|Recall8|Recall10|MRR24|候选覆盖24|", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        values = ["N/A" if row[k] is None else f"{row[k]:.6f}" for k in METRICS]
        lines.append("|" + "|".join([f"{row['variant']}/{row['split']}", *values]) + "|")
    lines += ["", "数值误差：" + json.dumps(numeric, ensure_ascii=False), "",
              "质量与完整墙钟分开呈现；不自动采用、合并或部署。详细原件、逐题和限制见summary.json。"]
    with (output / "report.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-gate", type=Path)
    parser.add_argument("--expected-previous-gate-sha256")
    args = parser.parse_args()
    result = report(args.freeze, args.expected_freeze_sha256, args.execution, args.output,
                    previous_gate=args.previous_gate,
                    expected_previous_gate_sha256=args.expected_previous_gate_sha256)
    print(json.dumps({"phase": result["phase"], "complete": result["all_cells_executed"],
                      "gate_pass": result["gate_pass"]}))


if __name__ == "__main__":
    main()
