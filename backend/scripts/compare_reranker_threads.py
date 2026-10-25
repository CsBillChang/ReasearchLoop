"""按事前冻结重放同一输入的CPU线程对照，逐批核实际张量并保留全部结果。"""
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
import re
import resource
import sys
import time

from research_loop.providers.reranker import LocalReranker
from scripts.compare_lexical_retrieval import digest
from scripts.prepare_reranker_thread_comparison import (
    SOURCE_FIELDS, build_execution_plan, canonical_digest,
)
from scripts.retrieval_experiment_guard import BoundaryViolation, install_guards, write_json_new


SETTINGS = {"threads": [2, 4], "provider": "CPUExecutionProvider", "batch_size": 4,
            "sort_by_length": False, "max_pair_tokens": 512, "hot_repeats": 3,
            "max_score_attempts": 686, "max_pairs": 16134, "max_seconds": 3600}
AUTHORIZATION = {"local_reranker": True, "external_api": False, "embedding": False,
                 "deployment": False, "new_assets": False}
COUNTS = {"cases": 114, "pairs_per_repeat": 2681, "planned_cells": 686,
          "session_run_calls": 4050}
MEASUREMENT = "same_run_reranker_component_with_feed_verification"


class CellStopped(BoundaryViolation):
    """携带已发生格的失败记录，供外层先留证再终止余下计划。"""

    def __init__(self, record: dict, cause: BaseException):
        self.record, self.cause = record, cause
        super().__init__(str(cause))


def is_hash(value: object) -> bool:
    """只接受完整小写SHA-256，避免把未知身份当作可执行冻结。"""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def check_reference(ref: dict) -> Path:
    """核对显式真实绝对路径和事前文件hash，不扩大到父目录读取。"""
    path = Path(ref["path"])
    if (not path.is_absolute() or path != path.resolve() or not path.is_file()
            or not is_hash(ref.get("sha256")) or digest(path) != ref["sha256"]):
        raise BoundaryViolation("FROZEN_REFERENCE_CHANGED")
    return path


def all_references(frozen: dict) -> list[dict]:
    """评分输入、报告材料和源码都校验字节；报告材料只散列不解析。"""
    return (list(frozen["inputs"].values()) + list(frozen.get("report_inputs", {}).values())
            + frozen["source_files"] + frozen["protected_files"] + frozen["model_files"])


def validate_case(case: dict, *, allow_empty: bool = False) -> None:
    """每次调用前核原文、授权、白名单及候选顺序身份。"""
    if set(case) != {"case_id", "query", "scope_document_ids", "candidates", "input_sha256"}:
        raise BoundaryViolation("CASE_PROJECTION_CHANGED")
    if (not isinstance(case["case_id"], str) or not case["case_id"]
            or not isinstance(case["query"], str) or not case["query"]):
        raise BoundaryViolation("CASE_QUERY_INVALID")
    scope, candidates = case["scope_document_ids"], case["candidates"]
    if (not isinstance(scope, list) or not scope or any(not isinstance(v, str) or not v for v in scope)
            or len(scope) != len(set(scope)) or not isinstance(candidates, list)
            or not (0 if allow_empty else 1) <= len(candidates) <= 24):
        raise BoundaryViolation("CASE_SCOPE_OR_COUNT_INVALID")
    ids = []
    for row in candidates:
        if not isinstance(row, dict) or set(row) != set(SOURCE_FIELDS):
            raise BoundaryViolation("CANDIDATE_PROJECTION_CHANGED")
        ids.append(row["chunk_id"])
        if (not isinstance(row["chunk_id"], str) or not row["chunk_id"]
                or row["document_id"] not in scope or not isinstance(row["text"], str)
                or not is_hash(row["source_sha256"]) or not is_hash(row["text_sha256"])
                or hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]):
            raise BoundaryViolation("CANDIDATE_IDENTITY_CHANGED")
        if (any(isinstance(row[k], bool) or not isinstance(row[k], int)
                for k in ("page_number", "char_start", "char_end"))
                or row["page_number"] < 1 or row["char_start"] < 0
                or row["char_end"] - row["char_start"] != len(row["text"])):
            raise BoundaryViolation("CANDIDATE_OFFSET_CHANGED")
    if len(ids) != len(set(ids)):
        raise BoundaryViolation("DUPLICATE_CANDIDATE")
    body = {k: v for k, v in case.items() if k != "input_sha256"}
    if not is_hash(case["input_sha256"]) or canonical_digest(body) != case["input_sha256"]:
        raise BoundaryViolation("CASE_INPUT_HASH_CHANGED")


def validate_feed_projection(case: dict, feed: dict) -> None:
    """核准备清单的分组、形状和身份，不使用任何已保存重排分数。"""
    if (feed["case_id"] != case["case_id"] or feed["input_sha256"] != case["input_sha256"]
            or feed["pairs"] != len(case["candidates"]) or feed["batch_size"] != 4
            or feed["sort_by_length"] is not False or feed["max_pair_tokens"] != 512
            or feed["padding_id"] != 1):
        raise BoundaryViolation("FEED_CASE_IDENTITY_CHANGED")
    indices, input_tokens, padded_tokens = [], 0, 0
    for ordinal, batch in enumerate(feed["batches"]):
        expected_indices = list(range(ordinal * 4, min((ordinal + 1) * 4, feed["pairs"])))
        lengths = batch["unpadded_lengths"]
        if (batch["candidate_indices"] != expected_indices or not expected_indices
                or len(lengths) != len(expected_indices)
                or any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 512 for n in lengths)):
            raise BoundaryViolation("FEED_BATCH_ORDER_CHANGED")
        tensors = batch["tensors"]
        if set(tensors) != {"input_ids", "attention_mask", "token_type_ids_if_required"}:
            raise BoundaryViolation("FEED_TENSOR_NAMES_CHANGED")
        shape = [len(expected_indices), max(lengths)]
        for ref in tensors.values():
            if (set(ref) != {"dtype", "byte_order", "shape", "sha256"} or ref["dtype"] != "int64"
                    or ref["byte_order"] != "little" or ref["shape"] != shape
                    or not is_hash(ref["sha256"])):
                raise BoundaryViolation("FEED_TENSOR_PROJECTION_CHANGED")
        if canonical_digest(tensors) != batch["projection_sha256"]:
            raise BoundaryViolation("FEED_PROJECTION_HASH_CHANGED")
        indices.extend(expected_indices)
        input_tokens += sum(lengths)
        padded_tokens += shape[0] * shape[1]
    if (indices != list(range(feed["pairs"])) or input_tokens != feed["input_tokens"]
            or padded_tokens != feed["padded_tokens"]):
        raise BoundaryViolation("FEED_BATCH_COVERAGE_CHANGED")


def load_inputs(frozen: dict) -> tuple[list[dict], dict[str, dict], dict]:
    """严格校验准备格与模型文件，只解析执行所需的四份JSON。"""
    if (frozen.get("schema") != "reranker-thread-execution-v1"
            or frozen.get("settings") != SETTINGS or frozen.get("authorization") != AUTHORIZATION
            or frozen.get("expected_counts") != COUNTS
            or set(frozen["inputs"]) != {"cases", "feed_manifest", "execution_plan", "model_manifest"}):
        raise BoundaryViolation("EXECUTION_CONTRACT_CHANGED")
    for ref in all_references(frozen):
        check_reference(ref)
    if set(frozen["packages"]) != {"onnxruntime", "tokenizers", "numpy"}:
        raise BoundaryViolation("RUNTIME_PACKAGE_SET_CHANGED")
    for package, version in frozen["packages"].items():
        if importlib.metadata.version(package) != version:
            raise BoundaryViolation("RUNTIME_PACKAGE_VERSION_CHANGED")
    inputs = {key: json.loads(Path(ref["path"]).read_text()) for key, ref in frozen["inputs"].items()}
    cases, feed_rows, plan = inputs["cases"], inputs["feed_manifest"]["cases"], inputs["execution_plan"]
    if len(cases) != 114 or len({c["case_id"] for c in cases}) != 114 or len(feed_rows) != 114:
        raise BoundaryViolation("PREPARED_CASE_COUNTS_CHANGED")
    feeds = {feed["case_id"]: feed for feed in feed_rows}
    if len(feeds) != len(feed_rows) or set(feeds) != {case["case_id"] for case in cases}:
        raise BoundaryViolation("PREPARED_FEED_SET_CHANGED")
    for case in cases:
        validate_case(case)
        validate_feed_projection(case, feeds[case["case_id"]])
    if plan != build_execution_plan(cases):
        raise BoundaryViolation("PREPARED_PLAN_CHANGED")
    if (sum(len(c["candidates"]) for c in cases) != 2681
            or len(plan["cells"]) != 686
            or sum(len(feeds[cell["case_id"]]["batches"]) for cell in plan["cells"]) != 4050):
        raise BoundaryViolation("PREPARED_EXACT_BUDGET_CHANGED")
    spec = inputs["model_manifest"]["reranker"]
    expected = frozen["expected_model_identity"]
    fixed = {"model_id": spec["repo"], "revision": spec["revision"],
             "weights_sha256": spec["files"][spec["weights"]]["sha256"],
             "tokenizer_sha256": spec["files"]["tokenizer.json"]["sha256"],
             "onnxruntime": frozen["packages"]["onnxruntime"], "max_pair_tokens": 512,
             "score_kind": "raw_relevance_logit", "batch_size": 4, "sort_by_length": False}
    if expected != fixed or spec["repo"] != "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1":
        raise BoundaryViolation("MODEL_IDENTITY_CHANGED")
    model_refs = {str(Path(r["path"])): r["sha256"] for r in frozen["model_files"]}
    for name in (spec["weights"], "tokenizer.json", "config.json"):
        if model_refs.get(str(Path(spec["directory"]) / name)) != spec["files"][name]["sha256"]:
            raise BoundaryViolation("MODEL_FILE_COVERAGE_CHANGED")
    return cases, feeds, plan


def array_reference(array) -> dict:
    """散列实际传给Session的连续little-endian int64张量，不改动输入。"""
    import numpy as np
    if (not isinstance(array, np.ndarray) or array.ndim != 2
            or array.dtype != np.dtype("<i8") or not array.flags.c_contiguous):
        raise BoundaryViolation("ACTUAL_TENSOR_DTYPE_OR_LAYOUT_CHANGED")
    return {"dtype": "int64", "byte_order": "little", "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


class VerifiedSession:
    """代理原Session，逐次送入推理前核实际feed，并单列仪器耗时。"""

    def __init__(self, native_session, *, expected_threads: int):
        self.native = native_session
        names = [value.name for value in native_session.get_inputs()]
        if (len(names) != len(set(names)) or set(names) not in (
                {"input_ids", "attention_mask"}, {"input_ids", "attention_mask", "token_type_ids"})):
            raise BoundaryViolation("SESSION_INPUT_NAMES_CHANGED")
        if native_session.get_providers() != ["CPUExecutionProvider"]:
            raise BoundaryViolation("SESSION_PROVIDER_CHANGED")
        if any(value.type != "tensor(int64)" for value in native_session.get_inputs()):
            raise BoundaryViolation("SESSION_INPUT_TYPE_CHANGED")
        options = native_session.get_session_options()
        if (expected_threads not in (2, 4) or options.intra_op_num_threads != expected_threads
                or options.inter_op_num_threads != 1):
            raise BoundaryViolation("SESSION_ACTUAL_THREADS_CHANGED")
        self.identity = {"providers": native_session.get_providers(), "input_names": names,
                         "provider_options": native_session.get_provider_options(),
                         "intra_op_num_threads": options.intra_op_num_threads,
                         "inter_op_num_threads": options.inter_op_num_threads}
        self.input_names = set(names)
        self.expected = None
        self.reset(None)

    def get_inputs(self):
        """委托原对象，保持业务provider的输入探测行为。"""
        return self.native.get_inputs()

    def reset(self, expected: dict | None) -> None:
        """每个score调用重新绑定题目和批次，禁止沿用上一题状态。"""
        self.expected, self.batch_index = expected, 0
        self.verification_seconds = self.native_seconds = 0.0
        self.run_attempts = self.run_successes = 0
        self.batches = []

    def run(self, output_names, feed):
        """验证通过后才调用原Session；模型失败仍保留尝试次数。"""
        verified_at = time.perf_counter()
        batch_record = {"batch_index": self.batch_index, "candidate_indices": None,
                        "actual_tensors": {}, "verification_passed": False,
                        "native_attempted": False, "native_succeeded": False, "error": None}
        self.batches.append(batch_record)
        try:
            if (self.expected is None or self.batch_index >= len(self.expected["batches"])
                    or output_names is not None or set(feed) != self.input_names):
                raise BoundaryViolation("ACTUAL_FEED_NAMES_OR_BATCH_COUNT_CHANGED")
            expected_batch = self.expected["batches"][self.batch_index]
            expected = expected_batch["tensors"]
            batch_record["candidate_indices"] = list(expected_batch["candidate_indices"])
            for name, value in feed.items():
                ref_name = "token_type_ids_if_required" if name == "token_type_ids" else name
                actual = array_reference(value)
                batch_record["actual_tensors"][name] = actual
                if actual != expected[ref_name]:
                    raise BoundaryViolation("ACTUAL_FEED_TENSOR_HASH_CHANGED")
            self.batch_index += 1
            batch_record["verification_passed"] = True
        except BaseException as exc:
            batch_record["error"] = {"type": type(exc).__name__, "reason": str(exc)}
            raise
        finally:
            self.verification_seconds += time.perf_counter() - verified_at
        self.run_attempts += 1
        batch_record["native_attempted"] = True
        started = time.perf_counter()
        try:
            result = self.native.run(output_names, feed)
            self.run_successes += 1
            batch_record["native_succeeded"] = True
            return result
        except BaseException as exc:
            batch_record["error"] = {"type": type(exc).__name__, "reason": str(exc)}
            raise
        finally:
            self.native_seconds += time.perf_counter() - started

    def snapshot(self) -> dict:
        """返回本次格的验证与底层执行计数。"""
        expected = len(self.expected["batches"]) if self.expected is not None else 0
        return {"expected_batches": expected, "verified_batches": self.batch_index,
                "session_run_attempts": self.run_attempts, "session_run_successes": self.run_successes,
                "complete": (self.batch_index == expected and self.run_successes == expected
                             and len(self.batches) == expected
                             and all(b["verification_passed"] and b["native_succeeded"] for b in self.batches)),
                "batches": deepcopy(self.batches)}


def new_counters() -> dict:
    """评分调用和底层Session调用分别计量，避免把组批数当题数。"""
    return dict.fromkeys(("score_attempts", "pair_attempts", "successful_score_calls", "successful_pairs",
                          "failed_score_calls", "session_run_attempts", "session_run_successes",
                          "verified_batches", "hot_rows", "warmup_rows"), 0)


def check_engine_identity(engine, threads: int, expected: dict) -> None:
    """两配置只能改变threads，权重、版本、分词器及组批均相同。"""
    if engine.identity != {**expected, "threads": threads}:
        raise BoundaryViolation("RERANKER_RUNTIME_IDENTITY_CHANGED")


def run_one(case: dict, feed: dict, cell: dict, engine, counters: dict, *,
            expected_identity: dict, started_at: float, settings: dict = SETTINGS) -> dict:
    """执行单格原query原文，普通失败留记录，边界及资源错误携记录停止。"""
    record = {**cell, "status": "stopped", "error": None, "query": case["query"],
              "scope_document_ids": case["scope_document_ids"],
              "candidate_ids": [r["chunk_id"] for r in case["candidates"]],
              "input_sha256": case["input_sha256"], "feed_projection_sha256": canonical_digest(feed),
              "scores": None, "ranked_ids": [], "hits": [], "model_identity": deepcopy(engine.identity),
              "session_identity": deepcopy(engine.session.identity), "local_reranker_stats": {},
              "measurement": MEASUREMENT, "host_load_average": list(os.getloadavg())}
    # 对完整的单题feed投影做canonical_digest，与372记录逐字段一致。
    session = engine.session
    session.reset(feed)
    beginning = time.perf_counter()
    attempted, fatal, scores = False, None, None
    try:
        if beginning - started_at > settings["max_seconds"]:
            raise BoundaryViolation("EXPERIMENT_TIME_LIMIT")
        validate_case(case, allow_empty=True)
        validate_feed_projection(case, feed)
        if cell["threads"] not in (2, 4):
            raise BoundaryViolation("THREAD_CONFIGURATION_CHANGED")
        check_engine_identity(engine, cell["threads"], expected_identity)
        if not case["candidates"]:
            record["status"] = "empty"
        else:
            if (counters["score_attempts"] + 1 > settings["max_score_attempts"]
                    or counters["pair_attempts"] + len(case["candidates"]) > settings["max_pairs"]):
                raise BoundaryViolation("EXPERIMENT_BUDGET_EXCEEDED")
            counters["score_attempts"] += 1
            counters["pair_attempts"] += len(case["candidates"])
            attempted = True
            engine.last_stats = {}
            scores = engine.score(case["query"], [row["text"] for row in case["candidates"]])
            if (not isinstance(scores, list) or len(scores) != len(case["candidates"])
                    or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                           for v in scores)):
                raise ValueError("RERANK_OUTPUT_INVALID")
            if not session.snapshot()["complete"]:
                raise BoundaryViolation("ACTUAL_FEED_BATCHES_INCOMPLETE")
            record["local_reranker_stats"] = deepcopy(engine.last_stats)
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            record["scores"] = [float(value) for value in scores]
            record["ranked_ids"] = [case["candidates"][i]["chunk_id"] for i in order]
            record["hits"] = [{"chunk_id": case["candidates"][i]["chunk_id"],
                               "document_id": case["candidates"][i]["document_id"],
                               "text_sha256": case["candidates"][i]["text_sha256"],
                               "rank": rank, "candidate_rank": i + 1,
                               "reranker_score": float(scores[i])} for rank, i in enumerate(order, 1)]
            record["status"] = "executed"
            counters["successful_score_calls"] += 1
            counters["successful_pairs"] += len(scores)
    except BaseException as exc:
        resource_message = str(exc).lower()
        native_resource_error = any(term in resource_message for term in (
            "out of memory", "bad_alloc", "failed to allocate memory", "memory allocation failed"))
        ordinary = (isinstance(exc, Exception) and not isinstance(exc, (MemoryError, SystemError, OSError))
                    and not native_resource_error
                    and not str(exc).startswith(("RERANK_MODEL_", "RERANK_CONFIG_")))
        record["status"] = "failed" if ordinary else "stopped"
        record["error"] = {"type": type(exc).__name__, "reason": str(exc)}
        if not ordinary:
            fatal = exc
        if attempted:
            counters["failed_score_calls"] += 1
    finished = time.perf_counter()
    verification = session.snapshot()
    counters["session_run_attempts"] += verification["session_run_attempts"]
    counters["session_run_successes"] += verification["session_run_successes"]
    counters["verified_batches"] += verification["verified_batches"]
    record["feed_verification"] = verification
    record["timing"] = {"reranker_wall_seconds": finished - beginning,
                        "provider_inference_seconds": record["local_reranker_stats"].get("inference_seconds"),
                        "feed_verification_seconds": session.verification_seconds,
                        "native_session_run_seconds": session.native_seconds}
    record["counters_after"] = deepcopy(counters)
    record["state"] = record["status"]
    if fatal is not None:
        raise CellStopped(record, fatal)
    return record


def execute(freeze_path: Path, expected_freeze_sha256: str, output: Path, *, prepare_only=False) -> dict:
    """显式入口锚点核对后装守卫；不恢复、不重试、不覆盖既有输出。"""
    freeze_path = freeze_path.resolve()
    if not is_hash(expected_freeze_sha256) or digest(freeze_path) != expected_freeze_sha256:
        raise BoundaryViolation("EXECUTION_FREEZE_ANCHOR_CHANGED")
    frozen = json.loads(freeze_path.read_text())
    references = all_references(frozen)
    roots = [Path(ref["path"]).resolve() for ref in references]
    roots.extend([freeze_path, Path(__file__).resolve().parents[1]])
    guard_counts = install_guards(output, roots)
    started = time.perf_counter()
    host_load_begin = list(os.getloadavg())
    counters = new_counters()
    status, error, plan = "complete", None, None
    executed_indices, initializations = [], []
    try:
        cases, feeds, plan = load_inputs(frozen)
        write_json_new(output / "execution-plan.json", {"freeze_sha256": expected_freeze_sha256,
            "plan": plan, "measurement": MEASUREMENT,
            "timing_boundary": "Observed wall includes input/feed/output validation and final sorting; no subtraction.",
            "time_limit_boundary": "Checked before each call and after it returns; not a hard realtime interrupt."})
        if prepare_only:
            if "onnxruntime" in sys.modules:
                raise BoundaryViolation("ORT_IMPORTED_DURING_PREFLIGHT")
            write_json_new(output / "preflight.json", {"status": "prepared_not_executed", "cases": len(cases),
                "planned_cells": len(plan["cells"]), "score_attempts": 0, "session_run_attempts": 0,
                "ort_imported": False, "model_loaded": False, "labels_parsed": False,
                "expected_session_run_calls": 4050})
        else:
            engines = {}
            for threads in (2, 4):
                if time.perf_counter() - started > 3600:
                    raise BoundaryViolation("EXPERIMENT_TIME_LIMIT")
                loading_at = time.perf_counter()
                engine = LocalReranker(frozen["inputs"]["model_manifest"]["path"], threads=threads,
                                       batch_size=4, sort_by_length=False)
                try:
                    engine._load()
                    check_engine_identity(engine, threads, frozen["expected_model_identity"])
                    engine.session = VerifiedSession(engine.session, expected_threads=threads)
                except BaseException as exc:
                    initialized = {"threads": threads, "status": "stopped",
                                   "load_seconds": time.perf_counter() - loading_at,
                                   "error": {"type": type(exc).__name__, "reason": str(exc)},
                                   "score_attempts": 0, "session_run_attempts": 0}
                    initializations.append(initialized)
                    write_json_new(output / f"initialization-threads-{threads}.json", initialized)
                    raise
                engines[threads] = engine
                initialized = {"threads": threads, "status": "loaded",
                               "load_seconds": time.perf_counter() - loading_at,
                               "model_identity": engine.identity, "session_identity": engine.session.identity,
                               "score_attempts": 0, "session_run_attempts": 0}
                initializations.append(initialized)
                write_json_new(output / f"initialization-threads-{threads}.json", initialized)
            by_id = {case["case_id"]: case for case in cases}
            for index, cell in enumerate(plan["cells"]):
                stopped = None
                try:
                    record = run_one(by_id[cell["case_id"]], feeds[cell["case_id"]], cell,
                                     engines[cell["threads"]], counters,
                                     expected_identity=frozen["expected_model_identity"], started_at=started)
                except CellStopped as exc:
                    record, stopped = exc.record, exc.cause
                filename = (f"cell-{index:04d}-{cell['phase']}-threads-{cell['threads']}.json")
                write_json_new(output / filename, record)
                executed_indices.append(index)
                counters["warmup_rows" if cell["phase"] == "warmup" else "hot_rows"] += 1
                if stopped is not None:
                    raise stopped
                if time.perf_counter() - started > 3600:
                    raise BoundaryViolation("EXPERIMENT_TIME_LIMIT_AFTER_CALL")
                if cell["phase"] == "hot" and counters["hot_rows"] % 20 == 0:
                    print(json.dumps({"hot_rows": counters["hot_rows"], "score_attempts": counters["score_attempts"],
                                      "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
            if counters["failed_score_calls"] == 0 and (counters["score_attempts"] != 686
                    or counters["pair_attempts"] != 16134 or counters["session_run_attempts"] != 4050
                    or counters["session_run_successes"] != 4050):
                raise BoundaryViolation("FINAL_EXECUTION_COUNTS_CHANGED")
    except BaseException as exc:
        status, error = "stopped", {"type": type(exc).__name__, "reason": str(exc)}
    unexecuted = ([{**cell, "index": i, "reason": "prepare_only" if prepare_only else "execution_stopped"}
                   for i, cell in enumerate(plan["cells"]) if i not in set(executed_indices)]
                  if plan is not None else None)
    terminal = {"status": status, "prepare_only": prepare_only,
                "finished_at": datetime.now(timezone.utc).isoformat(), "error": error,
                "counters": counters, "guard_counts": guard_counts,
                "initializations": initializations, "unexecuted": unexecuted,
                "elapsed_seconds": time.perf_counter() - started,
                "host_load_average_begin": host_load_begin, "host_load_average_end": list(os.getloadavg()),
                "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "measurement": MEASUREMENT, "labels_parsed": False}
    write_json_new(output / "terminal.json", terminal)
    print(json.dumps({k: v for k, v in terminal.items() if k != "unexecuted"}), flush=True)
    return terminal


def main() -> int:
    """仅显式冻结与全新目录可执行；prepare-only不导入模型运行时。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    return 0 if execute(args.freeze, args.expected_freeze_sha256, args.output,
                        prepare_only=args.prepare_only)["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
