"""冻结原输入后检查MPS浮点质量，并在先验门槛通过时进行CPU4同输入对照。"""
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
import struct
import sys
import time

from research_loop.providers.reranker import LocalReranker
from scripts.compare_lexical_retrieval import digest
from scripts.compare_reranker_threads import (
    VerifiedSession, array_reference, check_reference, is_hash, validate_case,
    validate_feed_projection,
)
from scripts.prepare_reranker_thread_comparison import canonical_digest
from scripts.retrieval_experiment_guard import BoundaryViolation, write_json_new


MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
SETTINGS = {"batch_size": 4, "max_pair_tokens": 512, "padding_id": 1,
            "sort_by_length": False, "attention_implementation": "eager",
            "cpu_intra_op_threads": 4, "cpu_inter_op_threads": 1,
            "hot_repeats": 3, "pilot_case_count": 12,
            "pilot_max_seconds": 600, "performance_max_seconds": 3600}
AUTHORIZATION = {"local_reranker": True, "mps": True, "external_api": False,
                 "embedding": False, "deployment": False,
                 "model_download_during_execution": False,
                 "dependency_install_during_execution": False}
PACKAGES = {"onnxruntime", "tokenizers", "numpy", "torch", "transformers", "safetensors"}
THRESHOLDS = {"mae_max": 0.02, "max_abs_max": 0.1, "metric_tolerance": 1e-12}
GATE_CHECKS = {"complete_execution", "verified_actual_feeds", "mps_device_and_dtype_verified",
               "fp16_numeric_within_limits", "fp16_quality_non_decreasing_vs_ort_cpu4",
               "fp16_quality_non_decreasing_vs_mps_fp32"}
MEASUREMENT = "same_run_reranker_component_with_feed_verification"


class CellStopped(BoundaryViolation):
    """携带已经发生的格，先落失败证据再停止剩余格。"""

    def __init__(self, record, cause):
        self.record, self.cause = record, cause
        super().__init__(str(cause))


def preparation_identity(frozen: dict) -> str:
    """绑定两阶段共同材料、源码、包及运行边界，不把后产生的门票纳入自身。"""
    keys = ("inputs", "source_files", "protected_files", "model_files", "packages", "settings", "runtime")
    return canonical_digest({key: frozen[key] for key in keys})


def all_references(frozen: dict) -> list[dict]:
    """报告标签只散列不解析，执行器只消费无标签输入。"""
    return (list(frozen["inputs"].values()) + list(frozen.get("report_inputs", {}).values())
            + frozen["source_files"] + frozen["protected_files"] + frozen["model_files"])


def build_execution_plan(cases: list[dict], feeds: dict, pilot_ids: list[str], phase: str) -> dict:
    """轮换配置顺序，明确记录初始化及全部计划格，不筛选历史快题。"""
    ids = [case["case_id"] for case in cases]
    if (phase not in {"pilot", "performance"} or len(ids) != len(set(ids)) or not ids
            or len(pilot_ids) != 12 or len(set(pilot_ids)) != 12 or not set(pilot_ids) <= set(ids)):
        raise BoundaryViolation("MPS_PLAN_INPUT_INVALID")
    selected = pilot_ids if phase == "pilot" else ids
    variants = ["ort_cpu4", "mps_fp32", "mps_fp16"] if phase == "pilot" else ["ort_cpu4", "mps_fp16"]
    repeats = 1 if phase == "pilot" else 3
    cells = [{"phase": "warmup", "case_id": selected[0], "variant": variant,
              "state": "not_executed"} for variant in variants]
    for repetition in range(repeats):
        for ordinal, case_id in enumerate(selected):
            offset = (repetition + ordinal) % len(variants)
            order = variants[offset:] + variants[:offset]
            for variant in order:
                cells.append({"phase": "hot", "repetition": repetition, "ordinal": ordinal,
                              "case_id": case_id, "variant": variant, "state": "not_executed"})
    budget = {"max_score_attempts": len(cells),
              "max_pairs": sum(feeds[cell["case_id"]]["pairs"] for cell in cells),
              "max_native_batches": sum(len(feeds[cell["case_id"]]["batches"]) for cell in cells),
              "max_seconds": SETTINGS[f"{phase}_max_seconds"]}
    if len(cells) > (39 if phase == "pilot" else 686):
        raise BoundaryViolation("MPS_PLAN_SCORE_LIMIT_CHANGED")
    return {"schema": "reranker-mps-plan-v1", "phase": phase, "variants": variants,
            "case_ids": selected, "hot_repeats": repeats, "settings": SETTINGS,
            "budget": budget, "cells": cells}


def validate_runtime(runtime: dict) -> None:
    """验证专属解释器前缀；解释器软链目标本身不能证明依赖已经隔离。"""
    if (Path(sys.executable).resolve() != Path(runtime["interpreter"]).resolve()
            or Path(sys.prefix).resolve() != Path(runtime["venv_prefix"]).resolve()
            or sys.prefix == sys.base_prefix):
        raise BoundaryViolation("MPS_ISOLATED_INTERPRETER_CHANGED")
    required = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                "TOKENIZERS_PARALLELISM": "false"}
    env = runtime["environment"]
    if any(env.get(key) != value for key, value in required.items()):
        raise BoundaryViolation("MPS_REQUIRED_OFFLINE_ENV_CHANGED")
    if any(os.environ.get(key) != value for key, value in env.items()):
        raise BoundaryViolation("MPS_FROZEN_ENV_CHANGED")
    roots = [Path(value) for value in runtime["mutable_roots"]]
    if not roots or any(not p.is_absolute() or p != p.resolve() or not p.is_dir() for p in roots):
        raise BoundaryViolation("MPS_MUTABLE_ROOT_INVALID")
    for key in ("TMPDIR", "TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME"):
        value = Path(env.get(key, ""))
        if not value.is_absolute() or not any(value == root or value.is_relative_to(root) for root in roots):
            raise BoundaryViolation("MPS_CACHE_OUTSIDE_MUTABLE_ROOT")


def load_inputs(frozen: dict) -> tuple[list[dict], dict, list[str], dict]:
    """校验先验材料及同revision浮点资产，标签始终只作字节保护。"""
    if (frozen.get("schema") != "reranker-mps-execution-v1" or frozen.get("settings") != SETTINGS
            or frozen.get("authorization") != AUTHORIZATION
            or set(frozen["inputs"]) != {"cases", "feed_manifest", "model_manifest", "float_manifest", "pilot_selection"}
            or set(frozen["packages"]) != PACKAGES):
        raise BoundaryViolation("MPS_EXECUTION_CONTRACT_CHANGED")
    for ref in all_references(frozen):
        check_reference(ref)
    validate_runtime(frozen["runtime"])
    for package, version in frozen["packages"].items():
        if importlib.metadata.version(package) != version:
            raise BoundaryViolation("MPS_RUNTIME_PACKAGE_VERSION_CHANGED")
    inputs = {key: json.loads(Path(ref["path"]).read_text()) for key, ref in frozen["inputs"].items()}
    cases = inputs["cases"]
    feeds = {row["case_id"]: row for row in inputs["feed_manifest"]["cases"]}
    if (len(cases) != 114 or len(feeds) != 114 or len({c["case_id"] for c in cases}) != 114
            or set(feeds) != {c["case_id"] for c in cases}
            or sum(len(c["candidates"]) for c in cases) != 2681):
        raise BoundaryViolation("MPS_FIXED_CASES_CHANGED")
    for case in cases:
        validate_case(case, allow_empty=True)
        validate_feed_projection(case, feeds[case["case_id"]])
    spec, floating = inputs["model_manifest"]["reranker"], inputs["float_manifest"]
    for model in (spec, floating):
        if model["repo"] != MODEL_ID or model["revision"] != MODEL_REVISION:
            raise BoundaryViolation("MPS_MODEL_REVISION_CHANGED")
    refs = {ref["path"]: ref["sha256"] for ref in frozen["model_files"]}
    for model, names in ((spec, (spec["weights"], "tokenizer.json", "config.json")),
                         (floating, ("model.safetensors", "config.json"))):
        for name in names:
            if refs.get(str(Path(model["directory"]) / name)) != model["files"][name]["sha256"]:
                raise BoundaryViolation("MPS_MODEL_FILE_COVERAGE_CHANGED")
    if floating["files"]["config.json"]["sha256"] != spec["files"]["config.json"]["sha256"]:
        raise BoundaryViolation("MPS_FLOAT_MODEL_CONFIG_CHANGED")
    if floating.get("weights", "model.safetensors") != "model.safetensors":
        raise BoundaryViolation("MPS_FLOAT_WEIGHTS_FORMAT_CHANGED")
    selection = inputs["pilot_selection"]
    if selection.get("schema") != "reranker-mps-pilot-selection-v1":
        raise BoundaryViolation("MPS_PILOT_SELECTION_CHANGED")
    pilot_ids = selection["case_ids"]
    build_execution_plan(cases, feeds, pilot_ids, "pilot")
    return cases, feeds, pilot_ids, inputs


def finite_number(value) -> bool:
    """拒绝布尔和非有限数，避免门票用NaN绕过质量阈值。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_pilot_gate(path: Path, expected_sha256: str, frozen: dict, freeze_sha256: str) -> dict:
    """后续阶段重核独立门票、全部原件和数值，不能只信一个pass字符串。"""
    if not is_hash(expected_sha256) or digest(path) != expected_sha256:
        raise BoundaryViolation("MPS_PILOT_GATE_ANCHOR_CHANGED")
    gate = json.loads(path.read_text())
    if (gate.get("schema") != "reranker-mps-pilot-gate-v1" or gate.get("phase") != "pilot"
            or gate.get("pass") is not True or gate.get("freeze_sha256") != freeze_sha256
            or gate.get("preparation_identity_sha256") != preparation_identity(frozen)
            or gate.get("thresholds") != THRESHOLDS or set(gate.get("checks", {})) != GATE_CHECKS
            or not all(value is True for value in gate["checks"].values())):
        raise BoundaryViolation("MPS_PILOT_GATE_NOT_QUALIFIED")
    numeric = gate["numeric"]
    if (numeric.get("complete") is not True or not isinstance(numeric.get("pairs"), int)
            or isinstance(numeric["pairs"], bool) or numeric["pairs"] <= 0
            or not all(finite_number(numeric.get(key)) for key in ("mae", "max_abs"))
            or not 0 <= numeric["mae"] <= THRESHOLDS["mae_max"]
            or not numeric["mae"] <= numeric["max_abs"] <= THRESHOLDS["max_abs_max"]):
        raise BoundaryViolation("MPS_PILOT_NUMERIC_GATE_FAILED")
    comparisons = gate["quality_comparisons"]
    if set(comparisons) != {"ort_cpu4", "mps_fp32"}:
        raise BoundaryViolation("MPS_PILOT_QUALITY_REFERENCES_CHANGED")
    for value in comparisons.values():
        if (value.get("complete") is not True or value.get("non_decreasing") is not True
                or not isinstance(value.get("planned_n"), int) or isinstance(value["planned_n"], bool)
                or value["planned_n"] <= 0 or value.get("known_n") != value["planned_n"]):
            raise BoundaryViolation("MPS_PILOT_QUALITY_GATE_FAILED")
        for metric in ("recall5", "recall8", "recall10", "mrr24"):
            ref, candidate, delta = (value[key].get(metric) for key in ("reference", "candidate", "deltas"))
            if (not all(finite_number(v) for v in (ref, candidate, delta)) or not 0 <= ref <= 1
                    or not 0 <= candidate <= 1 or abs(candidate - ref - delta) > 1e-12
                    or delta < -THRESHOLDS["metric_tolerance"]):
                raise BoundaryViolation("MPS_PILOT_QUALITY_VALUE_FAILED")
    refs = gate["reference_files"]
    if len(refs) < 42 or len({ref["path"] for ref in refs}) != len(refs):
        raise BoundaryViolation("MPS_PILOT_GATE_EVIDENCE_INCOMPLETE")
    for ref in refs:
        check_reference(ref)
    freeze_refs = [ref for ref in refs if ref["sha256"] == freeze_sha256]
    terminal_refs = [ref for ref in refs if Path(ref["path"]).name == "terminal.json"]
    plan_refs = [ref for ref in refs if Path(ref["path"]).name == "execution-plan.json"]
    if len(freeze_refs) != 1 or len(terminal_refs) != 1 or len(plan_refs) != 1:
        raise BoundaryViolation("MPS_PILOT_GATE_EVIDENCE_IDENTITY_CHANGED")
    terminal = json.loads(Path(terminal_refs[0]["path"]).read_text())
    if (terminal.get("status") != "complete" or terminal.get("phase") != "pilot"
            or terminal.get("freeze_sha256") != freeze_sha256 or terminal.get("unexecuted") != []
            or terminal.get("preparation_identity_sha256") != preparation_identity(frozen)
            or terminal["counters"].get("failed_score_calls") != 0):
        raise BoundaryViolation("MPS_PILOT_TERMINAL_NOT_COMPLETE")
    validate_gate_raw_files(gate, frozen, Path(plan_refs[0]["path"]))
    return gate


def validate_gate_raw_files(gate: dict, frozen: dict, plan_path: Path) -> None:
    """门票必须引用完整先验计划格，FP16误差从原始分数独立重算。"""
    payloads = {}
    for key in ("cases", "feed_manifest", "pilot_selection"):
        payloads[key] = json.loads(check_reference(frozen["inputs"][key]).read_text())
    cases = payloads["cases"]
    by_id = {case["case_id"]: case for case in cases}
    feeds = {row["case_id"]: row for row in payloads["feed_manifest"]["cases"]}
    pilot_ids = payloads["pilot_selection"]["case_ids"]
    expected_plan = build_execution_plan(cases, feeds, pilot_ids, "pilot")
    wrapper = json.loads(plan_path.read_text())
    if (wrapper.get("plan") != expected_plan
            or wrapper.get("freeze_sha256") != gate["freeze_sha256"]
            or wrapper.get("preparation_identity_sha256") != preparation_identity(frozen)):
        raise BoundaryViolation("MPS_PILOT_GATE_PLAN_CHANGED")
    def key(row):
        return row["phase"], row.get("repetition"), row["case_id"], row["variant"]
    expected = {key(cell) for cell in expected_plan["cells"]}
    raw = {}
    for ref in gate["reference_files"]:
        if not Path(ref["path"]).name.startswith("cell-"):
            continue
        row = json.loads(Path(ref["path"]).read_text())
        identity = key(row)
        if identity not in expected or identity in raw or row.get("status") != "executed":
            raise BoundaryViolation("MPS_PILOT_GATE_RAW_COVERAGE_CHANGED")
        case = by_id[row["case_id"]]
        ids = [candidate["chunk_id"] for candidate in case["candidates"]]
        if (row.get("candidate_ids") != ids or row.get("input_sha256") != case["input_sha256"]
                or row.get("feed_projection_sha256") != canonical_digest(feeds[row["case_id"]])
                or not isinstance(row.get("scores"), list) or len(row["scores"]) != len(ids)
                or not all(finite_number(value) for value in row["scores"])):
            raise BoundaryViolation("MPS_PILOT_GATE_RAW_IDENTITY_CHANGED")
        raw[identity] = row
    if len(raw) != 39 or set(raw) != expected:
        raise BoundaryViolation("MPS_PILOT_GATE_RAW_COVERAGE_CHANGED")
    deltas = [abs(a - b) for case_id in pilot_ids
              for a, b in zip(raw[("hot", 0, case_id, "mps_fp32")]["scores"],
                              raw[("hot", 0, case_id, "mps_fp16")]["scores"])]
    numeric = gate["numeric"]
    if (not deltas or numeric["pairs"] != len(deltas)
            or abs(numeric["mae"] - sum(deltas) / len(deltas)) > 1e-12
            or abs(numeric["max_abs"] - max(deltas)) > 1e-12):
        raise BoundaryViolation("MPS_PILOT_GATE_RAW_NUMERIC_CHANGED")


def reserve_native(counters: dict, budget: dict, started_at: float) -> None:
    """在底层调用前预留格，失败不退还本阶段预算。"""
    if time.perf_counter() - started_at > budget["max_seconds"]:
        raise BoundaryViolation("MPS_EXPERIMENT_TIME_LIMIT")
    if counters["native_batch_reservations"] >= budget["max_native_batches"]:
        raise BoundaryViolation("MPS_NATIVE_BUDGET_EXCEEDED")
    counters["native_batch_reservations"] += 1


class BudgetedCpuSession(VerifiedSession):
    """复用原CPU实际feed核验，并在每次底层调用前预留预算。"""

    def __init__(self, native, counters, budget, started_at):
        super().__init__(native, expected_threads=4)
        self.identity = {**self.identity, "backend": "onnxruntime", "device": "cpu"}
        self.counters, self.budget, self.started_at = counters, budget, started_at

    def run(self, output_names, feed):
        reserve_native(self.counters, self.budget, self.started_at)
        return super().run(output_names, feed)


class MpsVerifiedSession:
    """适配原分词/组批过程；核传入GPU的实际整数张量及参数设备。"""

    def __init__(self, model, torch, dtype, counters, budget, started_at):
        self.model, self.torch, self.dtype = model, torch, dtype
        self.counters, self.budget, self.started_at = counters, budget, started_at
        self.parameter_devices = sorted({str(p.device) for p in model.parameters()})
        self.parameter_dtypes = sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
        if (not self.parameter_devices or any(not d.startswith("mps") for d in self.parameter_devices)
                or self.parameter_dtypes != [str(dtype)] or any(module.training for module in model.modules())):
            raise BoundaryViolation("MPS_MODEL_DEVICE_OR_DTYPE_CHANGED")
        self.identity = {"backend": "pytorch_mps", "cpu_fallback_enabled": False,
                         "input_names": ["input_ids", "attention_mask"], "device": self.parameter_devices[0],
                         "dtype": str(dtype), "intra_op_num_threads": 4, "inter_op_num_threads": 1}
        self.reset(None)

    def get_inputs(self):
        return [type("Input", (), {"name": name})() for name in self.identity["input_names"]]

    def reset(self, expected):
        self.expected, self.batch_index = expected, 0
        self.verification_seconds = self.native_seconds = 0.0
        self.run_attempts = self.run_successes = 0
        self.batches = []

    def run(self, output_names, feed):
        import numpy as np
        started = time.perf_counter()
        record = {"batch_index": self.batch_index, "candidate_indices": None, "actual_tensors": {},
                  "verification_passed": False, "native_attempted": False, "native_succeeded": False,
                  "error": None, "device_proof": None}
        self.batches.append(record)
        try:
            if (self.expected is None or self.batch_index >= len(self.expected["batches"])
                    or output_names is not None or set(feed) != {"input_ids", "attention_mask"}):
                raise BoundaryViolation("MPS_ACTUAL_FEED_NAMES_CHANGED")
            expected = self.expected["batches"][self.batch_index]
            record["candidate_indices"] = expected["candidate_indices"]
            for name, array in feed.items():
                actual = array_reference(array)
                record["actual_tensors"][name] = actual
                if actual != expected["tensors"][name]:
                    raise BoundaryViolation("MPS_ACTUAL_FEED_HASH_CHANGED")
            tensors = {name: self.torch.as_tensor(value, dtype=self.torch.int64, device="mps")
                       for name, value in feed.items()}
            proof = {"input_devices": {name: str(value.device) for name, value in tensors.items()},
                     "input_dtypes": {name: str(value.dtype) for name, value in tensors.items()},
                     "model_parameter_devices": sorted({str(p.device) for p in self.model.parameters()}),
                     "model_parameter_dtypes": sorted({str(p.dtype) for p in self.model.parameters()
                                                       if p.is_floating_point()}),
                     "transferred_tensor_hashes": {}, "sync_before": False, "sync_after": False,
                     "output_device": None, "output_dtype": None}
            record["device_proof"] = proof
            if (any(not device.startswith("mps") for device in proof["input_devices"].values())
                    or set(proof["input_dtypes"].values()) != {"torch.int64"}
                    or proof["model_parameter_devices"] != self.parameter_devices
                    or proof["model_parameter_dtypes"] != self.parameter_dtypes):
                raise BoundaryViolation("MPS_ACTUAL_DEVICE_OR_DTYPE_CHANGED")
            for name, value in tensors.items():
                actual = array_reference(value.detach().cpu().numpy())
                proof["transferred_tensor_hashes"][name] = actual
                if actual != record["actual_tensors"][name]:
                    raise BoundaryViolation("MPS_TRANSFERRED_TENSOR_CHANGED")
            self.batch_index += 1
            record["verification_passed"] = True
        except BaseException as exc:
            record["error"] = {"type": type(exc).__name__, "reason": str(exc)}
            raise
        finally:
            self.verification_seconds += time.perf_counter() - started
        reserve_native(self.counters, self.budget, self.started_at)
        self.run_attempts += 1
        record["native_attempted"] = True
        self.torch.mps.synchronize()
        proof["sync_before"] = True
        native_at = time.perf_counter()
        try:
            with self.torch.inference_mode():
                output = self.model(**tensors).logits
            self.torch.mps.synchronize()
            proof["sync_after"] = True
            proof["output_device"], proof["output_dtype"] = str(output.device), str(output.dtype)
            if not str(output.device).startswith("mps") or output.dtype != self.dtype:
                raise BoundaryViolation("MPS_OUTPUT_DEVICE_OR_DTYPE_CHANGED")
            result = output.detach().float().cpu().numpy()
            if result.shape != (len(feed["input_ids"]), 1) or not np.isfinite(result).all():
                raise BoundaryViolation("MPS_OUTPUT_INVALID")
            self.run_successes += 1
            record["native_succeeded"] = True
            return [result]
        except BaseException as exc:
            record["error"] = {"type": type(exc).__name__, "reason": str(exc)}
            raise
        finally:
            self.native_seconds += time.perf_counter() - native_at

    def snapshot(self):
        expected = len(self.expected["batches"]) if self.expected is not None else 0
        return {"expected_batches": expected, "verified_batches": self.batch_index,
                "session_run_attempts": self.run_attempts, "session_run_successes": self.run_successes,
                "complete": (self.batch_index == expected and self.run_successes == expected
                             and len(self.batches) == expected
                             and all(b["verification_passed"] and b["native_succeeded"] for b in self.batches)),
                "batches": deepcopy(self.batches)}


def validate_loading_info(info: dict, weights: Path) -> dict:
    """只容许旧checkpoint中可逐值证明的非参数position_ids被新版库忽略。"""
    ignored = info.get("unexpected_keys", [])
    if (any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs"))
            or not isinstance(ignored, list)
            or any(key != "roberta.embeddings.position_ids" for key in ignored)
            or len(ignored) != len(set(ignored))):
        raise BoundaryViolation("MPS_MODEL_LOADING_KEYS_CHANGED")
    if ignored:
        with weights.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise BoundaryViolation("MPS_POSITION_BUFFER_HEADER_INVALID")
            length = struct.unpack("<Q", prefix)[0]
            if not 1 <= length <= 16 * 1024 * 1024:
                raise BoundaryViolation("MPS_POSITION_BUFFER_HEADER_INVALID")
            header = json.loads(stream.read(length))
            buffer = header.get("roberta.embeddings.position_ids", {})
            if buffer.get("dtype") != "I64" or buffer.get("shape") != [1, 514]:
                raise BoundaryViolation("MPS_POSITION_BUFFER_SCHEMA_CHANGED")
            offsets = buffer.get("data_offsets", [])
            if (len(offsets) != 2 or any(type(n) is not int or n < 0 for n in offsets)
                    or offsets[1] - offsets[0] != 514 * 8):
                raise BoundaryViolation("MPS_POSITION_BUFFER_OFFSETS_CHANGED")
            stream.seek(8 + length + offsets[0])
            values = stream.read(514 * 8)
            if len(values) != 514 * 8 or struct.unpack("<514q", values) != tuple(range(514)):
                raise BoundaryViolation("MPS_POSITION_BUFFER_VALUES_CHANGED")
    return {"ignored_nonparameter_buffers": ignored,
            "ignored_buffer_values_verified": bool(ignored),
            "reason": "Only the legacy nonparameter 0..513 position ID buffer may be regenerated; no missing model parameters accepted."}


def build_engine(variant, frozen, inputs, counters, budget, started_at, torch=None):
    """原CPU实现和MPS适配器共用同一个分词/组批/排序入口。"""
    engine = LocalReranker(frozen["inputs"]["model_manifest"]["path"], threads=4,
                           batch_size=4, sort_by_length=False)
    if variant == "ort_cpu4":
        engine._load()
        engine.session = BudgetedCpuSession(engine.session, counters, budget, started_at)
        engine.identity = {**engine.identity, "backend": "onnxruntime", "variant": variant,
                           "device": "cpu", "dtype": "int8_dynamic"}
        return engine
    if variant not in {"mps_fp32", "mps_fp16"} or torch is None:
        raise BoundaryViolation("MPS_VARIANT_CHANGED")
    from tokenizers import Tokenizer
    from transformers import AutoModelForSequenceClassification
    spec, floating = inputs["model_manifest"]["reranker"], inputs["float_manifest"]
    dtype = torch.float32 if variant == "mps_fp32" else torch.float16
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        floating["directory"], local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation="eager", dtype=dtype, output_loading_info=True)
    loading_compatibility = validate_loading_info(loading_info, Path(floating["directory"]) / "model.safetensors")
    if (type(model).__name__ != "XLMRobertaForSequenceClassification"
            or model.config.model_type != "xlm-roberta" or model.config.num_labels != 1
            or model.config.hidden_size != 384 or model.config.num_hidden_layers != 12):
        raise BoundaryViolation("MPS_MODEL_LOADING_OR_ARCHITECTURE_CHANGED")
    model.to("mps")
    model.eval()
    torch.mps.synchronize()
    engine.tokenizer = Tokenizer.from_file(str(Path(spec["directory"]) / "tokenizer.json"))
    engine.tokenizer.no_truncation()
    engine.tokenizer.no_padding()
    engine.session = MpsVerifiedSession(model, torch, dtype, counters, budget, started_at)
    engine.session.identity["loading_info"] = loading_info
    engine.session.identity["loading_compatibility"] = loading_compatibility
    engine.identity = {"backend": "pytorch_mps", "variant": variant, "model_id": MODEL_ID,
                       "revision": MODEL_REVISION,
                       "weights_sha256": floating["files"]["model.safetensors"]["sha256"],
                       "tokenizer_sha256": spec["files"]["tokenizer.json"]["sha256"],
                       "device": engine.session.identity["device"], "dtype": str(dtype),
                       "parameter_devices": engine.session.parameter_devices,
                       "parameter_dtypes": engine.session.parameter_dtypes,
                       "max_pair_tokens": 512, "score_kind": "raw_relevance_logit",
                       "batch_size": 4, "sort_by_length": False, "threads": 4}
    return engine


def new_counters():
    return dict.fromkeys(("score_attempts", "pair_attempts", "successful_score_calls", "successful_pairs",
                          "failed_score_calls", "session_run_attempts", "session_run_successes",
                          "verified_batches", "native_batch_reservations", "hot_rows", "warmup_rows"), 0)


def run_one(case, feed, cell, engine, counters, budget, started_at):
    """保留每题原文身份与全分数，任何不合格立即停止，不重试挑选成绩。"""
    record = {**cell, "status": "stopped", "error": None, "query": case["query"],
              "scope_document_ids": case["scope_document_ids"],
              "candidate_ids": [row["chunk_id"] for row in case["candidates"]],
              "input_sha256": case["input_sha256"], "feed_projection_sha256": canonical_digest(feed),
              "scores": None, "ranked_ids": [], "hits": [], "model_identity": deepcopy(engine.identity),
              "session_identity": deepcopy(engine.session.identity), "measurement": MEASUREMENT,
              "host_load_average": list(os.getloadavg()), "local_reranker_stats": {}}
    engine.session.reset(feed)
    beginning, attempted, fatal = time.perf_counter(), False, None
    try:
        validate_case(case, allow_empty=True)
        validate_feed_projection(case, feed)
        if (engine.identity.get("variant") != cell["variant"] or engine.identity.get("model_id") != MODEL_ID
                or engine.identity.get("revision") != MODEL_REVISION or engine.identity.get("threads") != 4):
            raise BoundaryViolation("MPS_ENGINE_IDENTITY_CHANGED")
        if time.perf_counter() - started_at > budget["max_seconds"]:
            raise BoundaryViolation("MPS_EXPERIMENT_TIME_LIMIT")
        if not case["candidates"]:
            record["status"] = "empty"
        else:
            if (counters["score_attempts"] >= budget["max_score_attempts"]
                    or counters["pair_attempts"] + len(case["candidates"]) > budget["max_pairs"]):
                raise BoundaryViolation("MPS_SCORE_BUDGET_EXCEEDED")
            counters["score_attempts"] += 1
            counters["pair_attempts"] += len(case["candidates"])
            attempted = True
            engine.last_stats = {}
            scores = engine.score(case["query"], [row["text"] for row in case["candidates"]])
            if (not isinstance(scores, list) or len(scores) != len(case["candidates"])
                    or not all(finite_number(value) for value in scores)):
                raise BoundaryViolation("MPS_SCORE_OUTPUT_INVALID")
            if not engine.session.snapshot()["complete"]:
                raise BoundaryViolation("MPS_ACTUAL_FEED_INCOMPLETE")
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            record["scores"] = [float(value) for value in scores]
            record["ranked_ids"] = [case["candidates"][i]["chunk_id"] for i in order]
            record["hits"] = [{"chunk_id": case["candidates"][i]["chunk_id"],
                               "document_id": case["candidates"][i]["document_id"],
                               "text_sha256": case["candidates"][i]["text_sha256"],
                               "rank": rank, "candidate_rank": i + 1,
                               "reranker_score": float(scores[i])} for rank, i in enumerate(order, 1)]
            record["local_reranker_stats"] = deepcopy(engine.last_stats)
            record["status"] = "executed"
            counters["successful_score_calls"] += 1
            counters["successful_pairs"] += len(scores)
    except BaseException as exc:
        record["status"] = "stopped"
        record["error"] = {"type": type(exc).__name__, "reason": str(exc)}
        if attempted:
            counters["failed_score_calls"] += 1
        fatal = exc
    finished = time.perf_counter()
    verification = engine.session.snapshot()
    for key in ("session_run_attempts", "session_run_successes", "verified_batches"):
        counters[key] += verification[key]
    record["feed_verification"] = verification
    record["timing"] = {"reranker_wall_seconds": finished - beginning,
                        "provider_inference_seconds": record["local_reranker_stats"].get("inference_seconds"),
                        "feed_verification_seconds": engine.session.verification_seconds,
                        "native_session_run_seconds": engine.session.native_seconds}
    record["state"], record["counters_after"] = record["status"], deepcopy(counters)
    if fatal is not None:
        raise CellStopped(record, fatal)
    return record


def execute(freeze_path, expected_freeze_sha256, output, phase, *, prepare_only=False,
            pilot_gate=None, expected_pilot_gate_sha256=None):
    """唯一真实入口；必须新目录，模型前通过输入、环境、预算和门票验证。"""
    if not is_hash(expected_freeze_sha256) or digest(freeze_path) != expected_freeze_sha256:
        raise BoundaryViolation("MPS_FREEZE_ANCHOR_CHANGED")
    frozen = json.loads(freeze_path.read_text())
    refs = all_references(frozen)
    read_roots = [Path(ref["path"]).resolve() for ref in refs]
    read_roots.extend([freeze_path.resolve(), Path(__file__).resolve().parents[1]])
    gate_for_reads = None
    if phase == "performance":
        if pilot_gate is None or expected_pilot_gate_sha256 is None:
            raise BoundaryViolation("MPS_PILOT_GATE_REQUIRED")
        gate_for_reads = validate_pilot_gate(pilot_gate, expected_pilot_gate_sha256, frozen, expected_freeze_sha256)
        read_roots.extend([pilot_gate.resolve(), *[Path(r["path"]).resolve() for r in gate_for_reads["reference_files"]]])
    from scripts.mps_experiment_guard import install_guards
    guard_counts = install_guards(output, read_roots, mutable_roots=[Path(p) for p in frozen["runtime"]["mutable_roots"]])
    started, counters = time.perf_counter(), new_counters()
    status, error, plan = "complete", None, None
    executed, initializations = [], []
    host_begin = list(os.getloadavg())
    try:
        cases, feeds, pilot_ids, inputs = load_inputs(frozen)
        plan = build_execution_plan(cases, feeds, pilot_ids, phase)
        if phase == "performance":
            validate_pilot_gate(pilot_gate, expected_pilot_gate_sha256, frozen, expected_freeze_sha256)
        write_json_new(output / "execution-plan.json", {"freeze_sha256": expected_freeze_sha256,
            "preparation_identity_sha256": preparation_identity(frozen), "plan": plan,
            "measurement": MEASUREMENT, "labels_parsed": False,
            "timing_boundary": "Common tokenization, batching, actual-feed checks, raw scoring, output validation, sorting; no subtraction.",
            "time_limit_boundary": "Per-call checks plus external hard supervisor; native calls are not realtime interruptible."})
        if prepare_only:
            if "torch" in sys.modules or "onnxruntime" in sys.modules:
                raise BoundaryViolation("MPS_RUNTIME_IMPORTED_IN_PREFLIGHT")
            write_json_new(output / "preflight.json", {"status": "prepared_not_executed", "phase": phase,
                "planned_cells": len(plan["cells"]), "model_loaded": False, "score_attempts": 0,
                "labels_parsed": False})
        else:
            import torch
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or not torch.backends.mps.is_available():
                raise BoundaryViolation("MPS_NOT_AVAILABLE_OR_FALLBACK_ENABLED")
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
                raise BoundaryViolation("MPS_TORCH_THREADS_CHANGED")
            engines = {}
            for variant in plan["variants"]:
                at = time.perf_counter()
                initialized = {"variant": variant, "status": "stopped", "error": None}
                try:
                    engine = build_engine(variant, frozen, inputs, counters, plan["budget"], started, torch)
                    engines[variant] = engine
                    initialized.update(status="loaded", model_identity=engine.identity,
                                       session_identity=engine.session.identity)
                except BaseException as exc:
                    initialized["error"] = {"type": type(exc).__name__, "reason": str(exc)}
                    raise
                finally:
                    initialized["load_seconds"] = time.perf_counter() - at
                    initializations.append(initialized)
                    write_json_new(output / f"initialization-{variant}.json", initialized)
            by_id = {case["case_id"]: case for case in cases}
            for index, cell in enumerate(plan["cells"]):
                stopped = None
                try:
                    row = run_one(by_id[cell["case_id"]], feeds[cell["case_id"]], cell,
                                  engines[cell["variant"]], counters, plan["budget"], started)
                except CellStopped as exc:
                    row, stopped = exc.record, exc.cause
                write_json_new(output / f"cell-{index:04d}-{cell['phase']}-{cell['variant']}.json", row)
                executed.append(index)
                counters["warmup_rows" if cell["phase"] == "warmup" else "hot_rows"] += 1
                if stopped is not None:
                    raise stopped
                if time.perf_counter() - started > plan["budget"]["max_seconds"]:
                    raise BoundaryViolation("MPS_EXPERIMENT_TIME_LIMIT_AFTER_CALL")
                if cell["phase"] == "hot" and counters["hot_rows"] % 12 == 0:
                    print(json.dumps({"phase": phase, "hot_rows": counters["hot_rows"],
                        "score_attempts": counters["score_attempts"], "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
            expected_budget = plan["budget"]
            if (counters["score_attempts"] != expected_budget["max_score_attempts"]
                    or counters["pair_attempts"] != expected_budget["max_pairs"]
                    or counters["session_run_attempts"] != expected_budget["max_native_batches"]
                    or counters["native_batch_reservations"] != expected_budget["max_native_batches"]
                    or counters["session_run_successes"] != expected_budget["max_native_batches"]):
                raise BoundaryViolation("MPS_FINAL_EXECUTION_COUNTS_CHANGED")
    except BaseException as exc:
        status, error = "stopped", {"type": type(exc).__name__, "reason": str(exc)}
    terminal = {"status": status, "phase": phase, "prepare_only": prepare_only,
                "finished_at": datetime.now(timezone.utc).isoformat(), "error": error,
                "freeze_sha256": expected_freeze_sha256,
                "preparation_identity_sha256": preparation_identity(frozen), "counters": counters,
                "guard_counts": guard_counts, "initializations": initializations,
                "unexecuted": ([{**cell, "index": i, "reason": "prepare_only" if prepare_only else "execution_stopped"}
                                for i, cell in enumerate(plan["cells"]) if i not in set(executed)] if plan else None),
                "elapsed_seconds": time.perf_counter() - started,
                "host_load_average_begin": host_begin, "host_load_average_end": list(os.getloadavg()),
                "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "measurement": MEASUREMENT, "labels_parsed": False,
                "pilot_gate_sha256": expected_pilot_gate_sha256 if gate_for_reads is not None else None}
    write_json_new(output / "terminal.json", terminal)
    print(json.dumps({key: value for key, value in terminal.items() if key != "unexecuted"}), flush=True)
    return terminal


def main():
    """只接受固定阶段和事前hash锚点，输出不可复用或覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "performance"), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pilot-gate", type=Path)
    parser.add_argument("--expected-pilot-gate-sha256")
    args = parser.parse_args()
    terminal = execute(args.freeze, args.expected_freeze_sha256, args.output, args.phase,
        prepare_only=args.prepare_only, pilot_gate=args.pilot_gate,
        expected_pilot_gate_sha256=args.expected_pilot_gate_sha256)
    return 0 if terminal["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
