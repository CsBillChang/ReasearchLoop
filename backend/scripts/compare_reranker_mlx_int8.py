"""以冻结输入验证MLX浮点前向，再比较单一W8A32候选与CPU4。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

from research_loop.providers.reranker import LocalReranker
from scripts.compare_lexical_retrieval import digest
from scripts.compare_reranker_mps import (
    BudgetedCpuSession, CellStopped, MODEL_ID, MODEL_REVISION,
    finite_number, new_counters, run_one,
)
from scripts.compare_reranker_threads import (
    array_reference, check_reference, is_hash, validate_case, validate_feed_projection,
)
from scripts.prepare_reranker_thread_comparison import canonical_digest
from scripts.retrieval_experiment_guard import BoundaryViolation, write_json_new


SCHEMA = "reranker-mlx-int8-execution-v1"
PHASES = ("float_qualification", "pilot", "performance")
SETTINGS = {"batch_size": 4, "max_pair_tokens": 512, "padding_id": 1,
            "sort_by_length": False, "cpu_intra_op_threads": 4, "cpu_inter_op_threads": 1,
            "hot_repeats": 3, "pilot_case_count": 12, "bits": 8, "group_size": 64,
            "quantized_linear_count": 74, "activation_dtype": "float32",
            "float_qualification_max_seconds": 600, "pilot_max_seconds": 600,
            "performance_max_seconds": 3600}
AUTHORIZATION = {"local_reranker": True, "gpu_int8": True, "external_api": False,
                 "embedding": False, "deployment": False,
                 "model_download_during_execution": False,
                 "dependency_install_during_execution": False}
THRESHOLDS = {"float_mae_max": .001, "float_max_abs_max": .01, "metric_tolerance": 1e-12}
VARIANTS = {"float_qualification": ["mlx_fp32"], "pilot": ["mlx_w8a32"],
            "performance": ["ort_cpu4", "mlx_w8a32"]}
MEASUREMENT = "same_run_reranker_component_with_feed_verification"


def all_references(frozen: dict) -> list[dict]:
    """标签仅绑定字节身份，评分器不解析其内容。"""
    return (list(frozen["inputs"].values()) + list(frozen["report_inputs"].values())
            + frozen["source_files"] + frozen["protected_files"] + frozen["model_files"]
            + frozen["historical_pilot_files"] + frozen.get("package_records", []))


def preparation_identity(frozen: dict) -> str:
    """后续门票绑定同一源码、运行包、资产、材料和阶段设置。"""
    names = ("inputs", "report_inputs", "source_files", "protected_files", "model_files",
             "historical_pilot_files", "packages", "settings", "runtime")
    return canonical_digest({name: frozen[name] for name in names}
                            | {"package_records": frozen.get("package_records", [])})


def build_execution_plan(cases: list[dict], feeds: dict, pilot_ids: list[str], phase: str) -> dict:
    """固定全部格及预算，正式阶段轮换两个方案先后。"""
    ids = [case["case_id"] for case in cases]
    if (phase not in PHASES or len(ids) != len(set(ids)) or not ids
            or len(pilot_ids) != 12 or len(set(pilot_ids)) != 12 or not set(pilot_ids) <= set(ids)):
        raise BoundaryViolation("MLX_PLAN_INPUT_INVALID")
    selected = ids if phase == "performance" else pilot_ids
    variants, repeats = VARIANTS[phase], 3 if phase == "performance" else 1
    cells = [{"phase": "warmup", "case_id": selected[0], "variant": variant,
              "state": "not_executed"} for variant in variants]
    for repetition in range(repeats):
        for ordinal, case_id in enumerate(selected):
            offset = (ordinal + repetition) % len(variants)
            for variant in variants[offset:] + variants[:offset]:
                cells.append({"phase": "hot", "repetition": repetition, "ordinal": ordinal,
                              "case_id": case_id, "variant": variant, "state": "not_executed"})
    budget = {"max_score_attempts": len(cells),
              "max_pairs": sum(feeds[c["case_id"]]["pairs"] for c in cells),
              "max_native_batches": sum(len(feeds[c["case_id"]]["batches"]) for c in cells),
              "max_seconds": SETTINGS[f"{phase}_max_seconds"]}
    if len(cells) > (686 if phase == "performance" else 13):
        raise BoundaryViolation("MLX_PLAN_BUDGET_CHANGED")
    return {"schema": "reranker-mlx-int8-plan-v1", "phase": phase, "variants": variants,
            "case_ids": selected, "hot_repeats": repeats, "settings": SETTINGS,
            "budget": budget, "cells": cells}


def validate_runtime(runtime: dict) -> None:
    """专属解释器前缀和离线缓存身份必须与事前冻结一致。"""
    if (Path(sys.executable).resolve() != Path(runtime["interpreter"]).resolve()
            or Path(sys.prefix).resolve() != Path(runtime["venv_prefix"]).resolve()
            or sys.prefix == sys.base_prefix):
        raise BoundaryViolation("MLX_ISOLATED_INTERPRETER_CHANGED")
    env = runtime["environment"]
    required = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
                "TOKENIZERS_PARALLELISM": "false", "MLX_ENABLE_TF32": "0"}
    if (any(env.get(k) != v for k, v in required.items())
            or any(os.environ.get(k) != v for k, v in env.items())):
        raise BoundaryViolation("MLX_FROZEN_OFFLINE_ENV_CHANGED")
    roots = [Path(p) for p in runtime["mutable_roots"]]
    if not roots or any(not p.is_absolute() or p != p.resolve() or not p.is_dir() for p in roots):
        raise BoundaryViolation("MLX_MUTABLE_ROOT_INVALID")
    for key in ("TMPDIR", "HF_HOME", "XDG_CACHE_HOME"):
        value = Path(env.get(key, ""))
        if not value.is_absolute() or not any(value == p or value.is_relative_to(p) for p in roots):
            raise BoundaryViolation("MLX_CACHE_OUTSIDE_MUTABLE_ROOT")


def load_inputs(frozen: dict, *, runtime: bool = True) -> tuple:
    """校验固定114题和同revision资产；只有实际入口检查当前运行环境。"""
    required = {"mlx", "numpy", "onnxruntime", "tokenizers", "safetensors"}
    if (frozen.get("schema") != SCHEMA or frozen.get("settings") != SETTINGS
            or frozen.get("authorization") != AUTHORIZATION
            or set(frozen["inputs"]) != {"cases", "feed_manifest", "model_manifest", "float_manifest", "pilot_selection"}
            or set(frozen["report_inputs"]) != {"labels", "original_questions"}
            or not required <= set(frozen["packages"])):
        raise BoundaryViolation("MLX_EXECUTION_CONTRACT_CHANGED")
    for ref in all_references(frozen):
        path = check_reference(ref)
        if "bytes" in ref and path.stat().st_size != ref["bytes"]:
            raise BoundaryViolation("MLX_REFERENCE_SIZE_CHANGED")
    if runtime:
        validate_runtime(frozen["runtime"])
        for package, version in frozen["packages"].items():
            if importlib.metadata.version(package) != version:
                raise BoundaryViolation("MLX_RUNTIME_PACKAGE_VERSION_CHANGED")
    inputs = {name: json.loads(Path(ref["path"]).read_text()) for name, ref in frozen["inputs"].items()}
    cases = inputs["cases"]
    feeds = {row["case_id"]: row for row in inputs["feed_manifest"]["cases"]}
    if (len(cases) != 114 or len(feeds) != 114 or len({c["case_id"] for c in cases}) != 114
            or set(feeds) != {c["case_id"] for c in cases}
            or sum(len(c["candidates"]) for c in cases) != 2681):
        raise BoundaryViolation("MLX_FIXED_CASES_CHANGED")
    for case in cases:
        validate_case(case, allow_empty=True)
        validate_feed_projection(case, feeds[case["case_id"]])
    spec, floating = inputs["model_manifest"]["reranker"], inputs["float_manifest"]
    refs = {r["path"]: r["sha256"] for r in frozen["model_files"]}
    for model, names in ((spec, (spec["weights"], "tokenizer.json", "config.json")),
                         (floating, ("model.safetensors", "config.json"))):
        if model["repo"] != MODEL_ID or model["revision"] != MODEL_REVISION:
            raise BoundaryViolation("MLX_MODEL_REVISION_CHANGED")
        for name in names:
            if refs.get(str(Path(model["directory"]) / name)) != model["files"][name]["sha256"]:
                raise BoundaryViolation("MLX_MODEL_FILE_COVERAGE_CHANGED")
    if floating["files"]["config.json"]["sha256"] != spec["files"]["config.json"]["sha256"]:
        raise BoundaryViolation("MLX_CONFIG_CHANGED")
    selection = inputs["pilot_selection"]
    if selection.get("schema") != "reranker-mps-pilot-selection-v1":
        raise BoundaryViolation("MLX_PILOT_SELECTION_CHANGED")
    pilot_ids = selection["case_ids"]
    build_execution_plan(cases, feeds, pilot_ids, "float_qualification")
    historical_rows(frozen, cases, feeds, pilot_ids)
    return cases, feeds, pilot_ids, inputs


def cell_key(row: dict) -> tuple:
    return row["phase"], row.get("repetition"), row["case_id"], row["variant"]


def validate_row_identity(row: dict, case: dict, feed: dict) -> None:
    """历史参照与当前输出均须逐题、逐候选按原序匹配。"""
    ids = [c["chunk_id"] for c in case["candidates"]]
    if (row.get("query") != case["query"] or row.get("scope_document_ids") != case["scope_document_ids"]
            or row.get("candidate_ids") != ids or row.get("input_sha256") != case["input_sha256"]
            or row.get("feed_projection_sha256") != canonical_digest(feed)):
        raise BoundaryViolation("MLX_RAW_INPUT_IDENTITY_CHANGED")
    if row.get("status") == "executed":
        scores = row.get("scores")
        if not isinstance(scores, list) or len(scores) != len(ids) or not all(finite_number(x) for x in scores):
            raise BoundaryViolation("MLX_RAW_SCORES_INVALID")
        expected = [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-scores[i], i))]
        if row.get("ranked_ids") != expected:
            raise BoundaryViolation("MLX_RAW_RANKING_CHANGED")


def historical_rows(frozen: dict, cases: list, feeds: dict, pilot_ids: list) -> dict:
    """只读取375两种既有原分数，不把历史时延放进同期对照。"""
    by_id = {c["case_id"]: c for c in cases}
    rows = {}
    for ref in frozen["historical_pilot_files"]:
        row = json.loads(check_reference(ref).read_text())
        key = (row["case_id"], row["variant"])
        if (row.get("phase") != "hot" or row.get("repetition") != 0
                or row.get("status") != "executed" or key in rows
                or row["case_id"] not in pilot_ids or row["variant"] not in {"ort_cpu4", "mps_fp32"}):
            raise BoundaryViolation("MLX_HISTORICAL_REFERENCE_CHANGED")
        validate_row_identity(row, by_id[key[0]], feeds[key[0]])
        identity = row["model_identity"]
        if identity.get("model_id") != MODEL_ID or identity.get("revision") != MODEL_REVISION:
            raise BoundaryViolation("MLX_HISTORICAL_MODEL_CHANGED")
        rows[key] = row
    if set(rows) != {(cid, v) for cid in pilot_ids for v in ("ort_cpu4", "mps_fp32")}:
        raise BoundaryViolation("MLX_HISTORICAL_COVERAGE_CHANGED")
    return rows


def reserve_native(counters: dict, budget: dict, started_at: float) -> None:
    if (counters["native_batch_reservations"] >= budget["max_native_batches"]
            or time.perf_counter() - started_at > budget["max_seconds"]):
        raise BoundaryViolation("MLX_NATIVE_BUDGET_EXCEEDED")
    counters["native_batch_reservations"] += 1


def validate_mlx_proof(proof: dict, actual: dict, variant: str, *, parameter_digest=None) -> None:
    """验证MLX统一内存/显式GPU流证明，不伪造Torch式array.device。"""
    quantized = variant == "mlx_w8a32"
    count = 74 if quantized else 0
    if (proof.get("input_tensor_references") != actual
            or proof.get("transferred_tensor_references") != actual
            or any(proof.get(k) is not True for k in (
                "gpu_stream_verified", "gpu_eval_complete", "sync_before", "sync_after", "quantization_verified"))
            or proof.get("output_dtype") != "float32" or proof.get("activation_dtype") != "float32"
            or proof.get("linear_call_count") != 74 or proof.get("quantized_linear_count") != count
            or proof.get("quantized_matmul_calls") != count
            or len(proof.get("linear_call_names", [])) != 74
            or len(set(proof["linear_call_names"])) != 74
            or not is_hash(proof.get("quantized_parameters_sha256"))
            or parameter_digest is not None and proof["quantized_parameters_sha256"] != parameter_digest):
        raise BoundaryViolation("MLX_GPU_OR_QUANTIZATION_PROOF_CHANGED")
    if quantized and (proof.get("bits") != 8 or proof.get("group_size") != 64
                      or proof.get("weight_packing") != "uint32"
                      or set(proof.get("quantized_layer_names", [])) != set(proof["linear_call_names"])):
        raise BoundaryViolation("MLX_W8_COVERAGE_CHANGED")


class MlxVerifiedSession:
    """在底层adapter之外核实际feed、预算和每批GPU执行证明。"""

    def __init__(self, native, variant, counters, budget, started_at):
        self.native, self.variant = native, variant
        self.counters, self.budget, self.started_at = counters, budget, started_at
        self.identity = deepcopy(native.identity)
        if (self.identity.get("backend") != "mlx" or self.identity.get("variant") != variant
                or self.identity.get("device") != "gpu"):
            raise BoundaryViolation("MLX_SESSION_IDENTITY_CHANGED")
        self.reset(None)

    def get_inputs(self):
        return self.native.get_inputs()

    def reset(self, expected):
        self.expected, self.batch_index = expected, 0
        self.batches = []
        self.verification_seconds = self.native_seconds = 0.0
        self.run_attempts = self.run_successes = 0

    def run(self, output_names, feed):
        batch = {"batch_index": self.batch_index, "candidate_indices": None,
                 "actual_tensors": {}, "verification_passed": False,
                 "native_attempted": False, "native_succeeded": False, "error": None}
        self.batches.append(batch)
        at = time.perf_counter()
        try:
            if (self.expected is None or self.batch_index >= len(self.expected["batches"])
                    or output_names is not None or set(feed) != {"input_ids", "attention_mask"}):
                raise BoundaryViolation("MLX_ACTUAL_FEED_NAMES_CHANGED")
            expected = self.expected["batches"][self.batch_index]
            batch["candidate_indices"] = expected["candidate_indices"]
            batch["actual_tensors"] = {name: array_reference(value) for name, value in feed.items()}
            if batch["actual_tensors"] != {name: expected["tensors"][name] for name in feed}:
                raise BoundaryViolation("MLX_ACTUAL_FEED_HASH_CHANGED")
            batch["verification_passed"] = True
            self.batch_index += 1
        except BaseException as exc:
            batch["error"] = {"type": type(exc).__name__, "reason": str(exc)}
            raise
        finally:
            self.verification_seconds += time.perf_counter() - at
        reserve_native(self.counters, self.budget, self.started_at)
        self.run_attempts += 1
        batch["native_attempted"] = True
        at = time.perf_counter()
        before = len(self.native.batch_proofs)
        try:
            result = self.native.run(output_names, feed)
            if len(self.native.batch_proofs) != before + 1:
                raise BoundaryViolation("MLX_NATIVE_PROOF_COUNT_CHANGED")
            proof = deepcopy(self.native.batch_proofs[-1])
            batch["device_proof"] = proof
            validate_mlx_proof(proof, batch["actual_tensors"], self.variant,
                               parameter_digest=self.identity.get("quantized_parameters_sha256"))
            self.run_successes += 1
            batch["native_succeeded"] = True
            return result
        except BaseException as exc:
            batch["error"] = {"type": type(exc).__name__, "reason": str(exc)}
            raise
        finally:
            # 异常也保留adapter已进入的线性层/量化算子计数，不将部分执行填0。
            batch["native_proof_append_count"] = len(self.native.batch_proofs) - before
            if len(self.native.batch_proofs) > before:
                batch["device_proof"] = deepcopy(self.native.batch_proofs[-1])
            self.native_seconds += time.perf_counter() - at

    def snapshot(self):
        expected = len(self.expected["batches"]) if self.expected else 0
        return {"expected_batches": expected, "verified_batches": self.batch_index,
                "session_run_attempts": self.run_attempts, "session_run_successes": self.run_successes,
                "complete": self.batch_index == expected and self.run_successes == expected
                and len(self.batches) == expected and all(b["native_succeeded"] for b in self.batches),
                "batches": deepcopy(self.batches)}


def build_engine(variant, frozen, inputs, counters, budget, started_at):
    engine = LocalReranker(frozen["inputs"]["model_manifest"]["path"], threads=4,
                           batch_size=4, sort_by_length=False)
    if variant == "ort_cpu4":
        engine._load()
        engine.session = BudgetedCpuSession(engine.session, counters, budget, started_at)
        engine.identity = {**engine.identity, "backend": "onnxruntime", "variant": variant,
                           "device": "cpu", "dtype": "int8_dynamic"}
        return engine
    if variant not in {"mlx_fp32", "mlx_w8a32"}:
        raise BoundaryViolation("MLX_VARIANT_CHANGED")
    from tokenizers import Tokenizer
    from scripts.mlx_reranker_adapter import MlxRerankerSession
    spec, floating = inputs["model_manifest"]["reranker"], inputs["float_manifest"]
    native = MlxRerankerSession(spec, floating, variant=variant)
    engine.tokenizer = Tokenizer.from_file(str(Path(spec["directory"]) / "tokenizer.json"))
    engine.tokenizer.no_truncation()
    engine.tokenizer.no_padding()
    engine.session = MlxVerifiedSession(native, variant, counters, budget, started_at)
    engine.identity = {**native.identity, "model_id": MODEL_ID, "revision": MODEL_REVISION,
                       "weights_sha256": floating["files"]["model.safetensors"]["sha256"],
                       "tokenizer_sha256": spec["files"]["tokenizer.json"]["sha256"],
                       "max_pair_tokens": 512, "batch_size": 4, "sort_by_length": False,
                       "threads": 4, "score_kind": "raw_relevance_logit"}
    return engine


def read_execution(output: Path, frozen: dict, freeze_sha: str, phase: str) -> tuple:
    """无模型地核计划格覆盖、原始分数和实际张量，供报告与门票重核。"""
    cases, feeds, pilot_ids, inputs = load_inputs(frozen, runtime=False)
    plan = build_execution_plan(cases, feeds, pilot_ids, phase)
    wrapper = json.loads((output / "execution-plan.json").read_text())
    if (wrapper.get("plan") != plan or wrapper.get("freeze_sha256") != freeze_sha
            or wrapper.get("preparation_identity_sha256") != preparation_identity(frozen)):
        raise BoundaryViolation("MLX_EXECUTION_PLAN_CHANGED")
    terminal = json.loads((output / "terminal.json").read_text())
    if (terminal.get("phase") != phase or terminal.get("freeze_sha256") != freeze_sha
            or terminal.get("preparation_identity_sha256") != preparation_identity(frozen)):
        raise BoundaryViolation("MLX_TERMINAL_IDENTITY_CHANGED")
    by_id, rows, refs = {c["case_id"]: c for c in cases}, {}, []
    expected = {cell_key(c) for c in plan["cells"]}
    for path in sorted(output.glob("cell-*.json")):
        row = json.loads(path.read_text())
        key = cell_key(row)
        if key not in expected or key in rows:
            raise BoundaryViolation("MLX_EXECUTION_COVERAGE_CHANGED")
        validate_row_identity(row, by_id[row["case_id"]], feeds[row["case_id"]])
        if row.get("status") == "executed":
            identity = row["model_identity"]
            spec, floating = inputs["model_manifest"]["reranker"], inputs["float_manifest"]
            weights_hash = (spec["files"][spec["weights"]]["sha256"] if row["variant"] == "ort_cpu4"
                            else floating["files"]["model.safetensors"]["sha256"])
            if (identity.get("model_id") != MODEL_ID or identity.get("revision") != MODEL_REVISION
                    or identity.get("weights_sha256") != weights_hash
                    or identity.get("tokenizer_sha256") != spec["files"]["tokenizer.json"]["sha256"]
                    or identity.get("variant") != row["variant"] or identity.get("threads") != 4
                    or identity.get("batch_size") != 4 or identity.get("max_pair_tokens") != 512
                    or identity.get("sort_by_length") is not False):
                raise BoundaryViolation("MLX_RAW_MODEL_IDENTITY_CHANGED")
            proof = row["feed_verification"]
            batches = feeds[row["case_id"]]["batches"]
            if (proof.get("complete") is not True or len(proof["batches"]) != len(batches)
                    or proof.get("session_run_successes") != len(batches)):
                raise BoundaryViolation("MLX_RAW_FEED_INCOMPLETE")
            for actual, wanted in zip(proof["batches"], batches):
                values = {name: wanted["tensors"][name] for name in ("input_ids", "attention_mask")}
                if (actual.get("actual_tensors") != values
                        or actual.get("candidate_indices") != wanted["candidate_indices"]
                        or actual.get("verification_passed") is not True
                        or actual.get("native_succeeded") is not True):
                    raise BoundaryViolation("MLX_RAW_TENSOR_PROOF_CHANGED")
                if row["variant"].startswith("mlx_"):
                    identity = row["model_identity"]
                    if (not is_hash(identity.get("quantized_parameters_sha256"))
                            or row["session_identity"].get("quantized_parameters_sha256") != identity["quantized_parameters_sha256"]):
                        raise BoundaryViolation("MLX_RAW_QUANTIZED_PARAMETER_IDENTITY_CHANGED")
                    validate_mlx_proof(actual["device_proof"], values, row["variant"],
                                       parameter_digest=identity["quantized_parameters_sha256"])
            if row["model_identity"].get("variant") != row["variant"]:
                raise BoundaryViolation("MLX_RAW_VARIANT_CHANGED")
        rows[key] = row
        refs.append(file_reference(path))
    return plan, terminal, rows, refs, cases, feeds, pilot_ids, inputs


def file_reference(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}


def score_errors(a: list[float], b: list[float]) -> dict:
    if len(a) != len(b) or not a or not all(finite_number(v) for v in a + b):
        raise BoundaryViolation("MLX_NUMERIC_PAIRING_CHANGED")
    errors = [abs(x - y) for x, y in zip(a, b)]
    return {"complete": True, "pairs": len(errors), "mae": sum(errors) / len(errors), "max_abs": max(errors)}


def validate_gate(path: Path, expected_sha: str, frozen: dict, freeze_sha: str,
                  expected_phase: str) -> dict:
    """门票重核原件与浮点误差/质量数值，不能只信pass字段。"""
    if not is_hash(expected_sha) or digest(path) != expected_sha:
        raise BoundaryViolation("MLX_GATE_ANCHOR_CHANGED")
    gate = json.loads(path.read_text())
    if (gate.get("schema") != "reranker-mlx-int8-gate-v1" or gate.get("phase") != expected_phase
            or gate.get("freeze_sha256") != freeze_sha or gate.get("thresholds") != THRESHOLDS
            or gate.get("preparation_identity_sha256") != preparation_identity(frozen)
            or gate.get("pass") is not True or gate.get("automatic_adoption") is not False):
        raise BoundaryViolation("MLX_GATE_NOT_QUALIFIED")
    for ref in gate["reference_files"]:
        check_reference(ref)
    summary_ref, quality_ref, reporter_ref = (gate[k] for k in (
        "report_summary_ref", "quality_rows_ref", "reporter_source_ref"))
    bound = {(r["path"], r["sha256"]) for r in gate["reference_files"]}
    for ref in (summary_ref, quality_ref, reporter_ref):
        check_reference(ref)
        if (ref["path"], ref["sha256"]) not in bound:
            raise BoundaryViolation("MLX_GATE_REPORTER_EVIDENCE_NOT_BOUND")
    reporter_path = str(Path(__file__).with_name("report_reranker_mlx_int8.py").resolve())
    if (reporter_ref["path"] != reporter_path
            or not any(r["path"] == reporter_path and r["sha256"] == reporter_ref["sha256"]
                       for r in frozen["source_files"])):
        raise BoundaryViolation("MLX_GATE_REPORTER_SOURCE_CHANGED")
    summary = json.loads(Path(summary_ref["path"]).read_text())
    if (summary.get("phase") != expected_phase or summary.get("freeze_sha256") != freeze_sha
            or summary.get("preparation_identity_sha256") != preparation_identity(frozen)
            or summary.get("quality_comparisons", {}).get("ort_cpu4") != gate.get("quality_vs_cpu4")
            or summary.get("numeric") != gate.get("numeric") or summary.get("gate_pass") is not True):
        raise BoundaryViolation("MLX_GATE_REPORT_SUMMARY_CHANGED")
    execution = Path(gate["execution_directory"])
    plan, terminal, rows, refs, cases, feeds, pilot_ids, _ = read_execution(execution, frozen, freeze_sha, expected_phase)
    if (terminal.get("status") != "complete" or terminal.get("prepare_only") is not False
            or terminal.get("unexecuted") != [] or len(rows) != len(plan["cells"])
            or any(r.get("status") != "executed" for r in rows.values())):
        raise BoundaryViolation("MLX_GATE_EXECUTION_INCOMPLETE")
    required = refs + [file_reference(execution / n) for n in ("terminal.json", "execution-plan.json")]
    if not all((r["path"], r["sha256"]) in bound for r in required):
        raise BoundaryViolation("MLX_GATE_EVIDENCE_COVERAGE_CHANGED")
    counters, budget = terminal["counters"], plan["budget"]
    for actual, expected in (("score_attempts", "max_score_attempts"), ("pair_attempts", "max_pairs"),
                             ("native_batch_reservations", "max_native_batches"),
                             ("session_run_attempts", "max_native_batches"),
                             ("session_run_successes", "max_native_batches")):
        if counters.get(actual) != budget[expected]:
            raise BoundaryViolation("MLX_GATE_COUNTERS_CHANGED")
    if expected_phase == "float_qualification":
        historical = historical_rows(frozen, cases, feeds, pilot_ids)
        errors = score_errors([s for cid in pilot_ids for s in rows[("hot", 0, cid, "mlx_fp32")]["scores"]],
                              [s for cid in pilot_ids for s in historical[(cid, "mps_fp32")]["scores"]])
        if (gate.get("numeric") != errors or errors["mae"] > THRESHOLDS["float_mae_max"]
                or errors["max_abs"] > THRESHOLDS["float_max_abs_max"]):
            raise BoundaryViolation("MLX_FLOAT_FORWARD_GATE_FAILED")
    elif expected_phase == "pilot":
        previous = gate["previous_gate"]
        validate_gate(Path(previous["path"]), previous["sha256"], frozen, freeze_sha, "float_qualification")
        previous_data = json.loads(Path(previous["path"]).read_text())
        _, _, float_rows, _, _, _, _, _ = read_execution(Path(previous_data["execution_directory"]),
                                                        frozen, freeze_sha, "float_qualification")
        errors = score_errors([s for cid in pilot_ids for s in rows[("hot", 0, cid, "mlx_w8a32")]["scores"]],
                              [s for cid in pilot_ids for s in float_rows[("hot", 0, cid, "mlx_fp32")]["scores"]])
        if gate.get("numeric") != errors:
            raise BoundaryViolation("MLX_PILOT_NUMERIC_RECORD_CHANGED")
        # 人工相关性由受保护reporter读取标签计算。执行器绑定其原件与源码，
        # 只重核算术和无标签分数误差，不在此读取人工标签或宣称独立质量裁决。
        comparison = gate["quality_vs_cpu4"]
        if (comparison.get("planned_n") != 11 or comparison.get("known_n") != 11
                or comparison.get("complete") is not True or comparison.get("non_decreasing") is not True):
            raise BoundaryViolation("MLX_PILOT_QUALITY_INCOMPLETE")
        for metric in ("recall5", "recall8", "recall10", "mrr24"):
            ref, candidate, delta = (comparison[k][metric] for k in ("reference", "candidate", "deltas"))
            if (not all(finite_number(v) for v in (ref, candidate, delta)) or not 0 <= ref <= 1
                    or not 0 <= candidate <= 1 or abs(candidate - ref - delta) > 1e-12
                    or delta < -THRESHOLDS["metric_tolerance"]):
                raise BoundaryViolation("MLX_PILOT_QUALITY_GATE_FAILED")
    else:
        raise BoundaryViolation("MLX_GATE_PHASE_INVALID")
    return gate


def execute(freeze_path: Path, expected_freeze_sha256: str, output: Path, phase: str, *,
            prepare_only=False, gate=None, expected_gate_sha256=None):
    """只在新输出目录执行一次固定阶段，失败保留已发生及未运行格。"""
    if phase not in PHASES or not is_hash(expected_freeze_sha256) or digest(freeze_path) != expected_freeze_sha256:
        raise BoundaryViolation("MLX_FREEZE_ANCHOR_OR_PHASE_CHANGED")
    frozen = json.loads(freeze_path.read_text())
    reads = [Path(r["path"]).resolve() for r in all_references(frozen)]
    reads += [freeze_path.resolve(), Path(__file__).resolve().parents[1]]
    gate_data = None
    if phase != "float_qualification":
        if gate is None or expected_gate_sha256 is None:
            raise BoundaryViolation("MLX_PREVIOUS_GATE_REQUIRED")
        previous_phase = "float_qualification" if phase == "pilot" else "pilot"
        gate_data = validate_gate(gate, expected_gate_sha256, frozen, expected_freeze_sha256, previous_phase)
        reads += [gate.resolve(), *[Path(r["path"]).resolve() for r in gate_data["reference_files"]]]
    from scripts.mps_experiment_guard import install_guards
    guard_counts = install_guards(output, reads, mutable_roots=[Path(p) for p in frozen["runtime"]["mutable_roots"]])
    started, counters = time.perf_counter(), new_counters()
    status, error, plan = "complete", None, None
    executed, initializations = [], []
    host_begin = list(os.getloadavg())
    try:
        cases, feeds, pilot_ids, inputs = load_inputs(frozen)
        plan = build_execution_plan(cases, feeds, pilot_ids, phase)
        write_json_new(output / "execution-plan.json", {"freeze_sha256": expected_freeze_sha256,
            "preparation_identity_sha256": preparation_identity(frozen), "plan": plan,
            "measurement": MEASUREMENT, "labels_parsed": False,
            "timing_boundary": "Tokenization/padding, actual-feed checks, evaluated synchronized GPU scoring, output readback and sorting; no subtraction."})
        if prepare_only:
            if any(n in sys.modules for n in ("mlx.core", "onnxruntime", "torch")):
                raise BoundaryViolation("MLX_MODEL_RUNTIME_IMPORTED_IN_PREFLIGHT")
            write_json_new(output / "preflight.json", {"status": "prepared_not_executed", "phase": phase,
                           "planned_cells": len(plan["cells"]), "model_loaded": False, "score_attempts": 0})
        else:
            engines = {}
            for variant in plan["variants"]:
                at = time.perf_counter()
                item = {"variant": variant, "status": "stopped", "error": None}
                try:
                    engine = build_engine(variant, frozen, inputs, counters, plan["budget"], started)
                    engines[variant] = engine
                    item.update(status="loaded", model_identity=engine.identity, session_identity=engine.session.identity)
                except BaseException as exc:
                    item["error"] = {"type": type(exc).__name__, "reason": str(exc)}
                    raise
                finally:
                    item["load_seconds"] = time.perf_counter() - at
                    initializations.append(item)
                    write_json_new(output / f"initialization-{variant}.json", item)
            by_id = {c["case_id"]: c for c in cases}
            for index, cell in enumerate(plan["cells"]):
                cause = None
                try:
                    row = run_one(by_id[cell["case_id"]], feeds[cell["case_id"]], cell,
                                  engines[cell["variant"]], counters, plan["budget"], started)
                except CellStopped as exc:
                    row, cause = exc.record, exc.cause
                write_json_new(output / f"cell-{index:04d}-{cell['phase']}-{cell['variant']}.json", row)
                executed.append(index)
                counters["warmup_rows" if cell["phase"] == "warmup" else "hot_rows"] += 1
                if cause is not None:
                    raise cause
                if time.perf_counter() - started > plan["budget"]["max_seconds"]:
                    raise BoundaryViolation("MLX_TIME_LIMIT_AFTER_CALL")
                if cell["phase"] == "hot" and counters["hot_rows"] % 12 == 0:
                    print(json.dumps({"phase": phase, "hot_rows": counters["hot_rows"],
                                      "score_attempts": counters["score_attempts"],
                                      "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
            budget = plan["budget"]
            for actual, expected in (("score_attempts", "max_score_attempts"), ("pair_attempts", "max_pairs"),
                                     ("native_batch_reservations", "max_native_batches"),
                                     ("session_run_attempts", "max_native_batches"),
                                     ("session_run_successes", "max_native_batches")):
                if counters[actual] != budget[expected]:
                    raise BoundaryViolation("MLX_FINAL_COUNTERS_CHANGED")
    except BaseException as exc:
        status, error = "stopped", {"type": type(exc).__name__, "reason": str(exc)}
    terminal = {"status": status, "phase": phase, "prepare_only": prepare_only,
        "finished_at": datetime.now(timezone.utc).isoformat(), "error": error,
        "freeze_sha256": expected_freeze_sha256, "preparation_identity_sha256": preparation_identity(frozen),
        "counters": counters, "guard_counts": guard_counts, "initializations": initializations,
        "unexecuted": [{**cell, "index": i, "reason": "prepare_only" if prepare_only else "execution_stopped"}
                       for i, cell in enumerate(plan["cells"]) if i not in executed] if plan else None,
        "elapsed_seconds": time.perf_counter() - started,
        "host_load_average_begin": host_begin, "host_load_average_end": list(os.getloadavg()),
        "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "measurement": MEASUREMENT, "labels_parsed": False,
        "previous_gate_sha256": expected_gate_sha256 if gate_data else None}
    write_json_new(output / "terminal.json", terminal)
    print(json.dumps({k: v for k, v in terminal.items() if k != "unexecuted"}), flush=True)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--expected-gate-sha256")
    args = parser.parse_args()
    result = execute(args.freeze, args.expected_freeze_sha256, args.output, args.phase,
                     prepare_only=args.prepare_only, gate=args.gate, expected_gate_sha256=args.expected_gate_sha256)
    raise SystemExit(0 if result["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
