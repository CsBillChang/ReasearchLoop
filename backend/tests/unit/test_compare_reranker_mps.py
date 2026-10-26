"""以合成张量核对MPS执行门、设备证明与预算，不导入真实Torch或模型。"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import struct
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import compare_reranker_mps as comparison
from scripts.prepare_reranker_thread_comparison import build_feed_projection, canonical_digest
from scripts.retrieval_experiment_guard import BoundaryViolation


def make_case(count=5, case_id="synthetic"):
    rows = []
    for i in range(count):
        text = f"synthetic passage {i}"
        rows.append({"chunk_id": f"chunk-{i}", "document_id": "authorized", "evidence_id": "e",
                     "index_id": "i", "source_sha256": "a" * 64, "page_number": 1,
                     "char_start": 0, "char_end": len(text), "text": text,
                     "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    case = {"case_id": case_id, "query": "original query", "scope_document_ids": ["authorized"], "candidates": rows}
    return {**case, "input_sha256": canonical_digest(case)}


class Encoder:
    def encode(self, query, text):
        return SimpleNamespace(ids=list(range(10, 12 + int(text.rsplit(" ", 1)[1]) % 3)))


def actual_feeds(case):
    encoded = [Encoder().encode(case["query"], c["text"]).ids for c in case["candidates"]]
    result = []
    for start in range(0, len(encoded), 4):
        group = encoded[start:start + 4]
        size = max(map(len, group))
        ids = np.ones((len(group), size), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, row in enumerate(group):
            ids[i, :len(row)], mask[i, :len(row)] = row, 1
        result.append({"input_ids": ids, "attention_mask": mask})
    return result


class Tensor:
    def __init__(self, value, dtype="torch.int64", device="mps:0"):
        self.value, self.dtype, self.device = np.array(value), dtype, device

    def detach(self): return self
    def cpu(self): return Tensor(self.value, self.dtype, "cpu")
    def float(self): return Tensor(self.value.astype(np.float32), "torch.float32", self.device)
    def numpy(self): return self.value


class Parameter:
    def __init__(self, dtype="torch.float16", device="mps:0"):
        self.dtype, self.device = dtype, device

    def is_floating_point(self): return True


class Torch:
    int64, float16, float32 = "torch.int64", "torch.float16", "torch.float32"

    def __init__(self):
        self.sync_calls, self.corrupt_transfer = 0, False
        self.mps = SimpleNamespace(synchronize=self.synchronize)

    def synchronize(self): self.sync_calls += 1
    def inference_mode(self): return nullcontext()

    def as_tensor(self, value, *, dtype, device):
        return Tensor(value + 1 if self.corrupt_transfer else value, dtype, "mps:0")


class Model:
    training = False

    def __init__(self, dtype="torch.float16", device="mps:0"):
        self.parameter = Parameter(dtype, device)
        self.calls, self.output_dtype, self.output_device, self.nan = 0, dtype, device, False

    def parameters(self): return iter([self.parameter])
    def modules(self): return iter([self])

    def __call__(self, **feed):
        self.calls += 1
        values = np.full((len(feed["input_ids"].value), 1), np.nan if self.nan else -2.5)
        return SimpleNamespace(logits=Tensor(values, self.output_dtype, self.output_device))


class Engine:
    def __init__(self, case, model, torch, counters, budget, started):
        self.case = case
        self.identity = {"model_id": comparison.MODEL_ID, "revision": comparison.MODEL_REVISION,
                         "threads": 4, "variant": "mps_fp16"}
        self.session = comparison.MpsVerifiedSession(model, torch, torch.float16, counters, budget, started)
        self.last_stats, self.values = {}, [-2.0] * len(case["candidates"])

    def score(self, query, texts):
        for feed in actual_feeds(self.case): self.session.run(None, feed)
        self.last_stats = {"inference_seconds": 0.1}
        return self.values


class MpsComparisonTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.feed = build_feed_projection(self.case, Encoder())
        self.counters = comparison.new_counters()
        self.budget = {"max_score_attempts": 1, "max_pairs": 5, "max_native_batches": 2, "max_seconds": 600}
        self.started, self.torch, self.model = time.perf_counter(), Torch(), Model()

    def session(self):
        session = comparison.MpsVerifiedSession(self.model, self.torch, self.torch.float16,
                                                self.counters, self.budget, self.started)
        session.reset(self.feed)
        return session

    def test_plan_has_fixed_pilot_and_full_budgets(self):
        cases = [make_case(24, f"case-{i}") for i in range(114)]
        feeds = {c["case_id"]: build_feed_projection(c, Encoder()) for c in cases}
        ids = [c["case_id"] for c in cases[:12]]
        pilot = comparison.build_execution_plan(cases, feeds, ids, "pilot")
        full = comparison.build_execution_plan(cases, feeds, ids, "performance")
        self.assertEqual((len(pilot["cells"]), len(full["cells"])), (39, 686))
        self.assertEqual(pilot["budget"]["max_pairs"], 936)
        self.assertEqual(full["budget"]["max_pairs"], 16464)
        self.assertEqual(pilot["variants"], ["ort_cpu4", "mps_fp32", "mps_fp16"])
        self.assertNotEqual([c["variant"] for c in pilot["cells"][3:6]],
                            [c["variant"] for c in pilot["cells"][6:9]])

    def test_duplicate_or_unknown_pilot_ids_rejected(self):
        cases = [make_case(1, str(i)) for i in range(12)]
        feeds = {c["case_id"]: build_feed_projection(c, Encoder()) for c in cases}
        for ids in [["0"] * 12, [str(i) for i in range(11)] + ["outside"]]:
            with self.assertRaises(BoundaryViolation):
                comparison.build_execution_plan(cases, feeds, ids, "pilot")

    def test_transferred_feed_device_dtype_and_synchronization_proven(self):
        session = self.session()
        for feed in actual_feeds(self.case):
            result = session.run(None, feed)
            self.assertTrue(np.isfinite(result[0]).all())
        proof = session.snapshot()
        self.assertTrue(proof["complete"])
        self.assertEqual((self.model.calls, self.torch.sync_calls), (2, 4))
        self.assertEqual(self.counters["native_batch_reservations"], 2)
        for batch in proof["batches"]:
            device = batch["device_proof"]
            self.assertEqual(device["transferred_tensor_hashes"], batch["actual_tensors"])
            self.assertTrue(device["sync_before"] and device["sync_after"])
            self.assertEqual(device["output_dtype"], "torch.float16")
            self.assertEqual(set(device["input_dtypes"].values()), {"torch.int64"})

    def test_host_feed_change_stops_before_model(self):
        session = self.session()
        feed = actual_feeds(self.case)[0]
        feed["input_ids"][0, 0] += 1
        with self.assertRaises(BoundaryViolation): session.run(None, feed)
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(self.counters["native_batch_reservations"], 0)

    def test_corrupt_gpu_transfer_stops_before_model(self):
        session = self.session(); self.torch.corrupt_transfer = True
        with self.assertRaises(BoundaryViolation): session.run(None, actual_feeds(self.case)[0])
        self.assertEqual(self.model.calls, 0)
        self.assertFalse(session.snapshot()["complete"])

    def test_cpu_parameter_or_wrong_precision_is_rejected(self):
        for model in (Model(device="cpu"), Model(dtype="torch.float32")):
            with self.assertRaises(BoundaryViolation):
                comparison.MpsVerifiedSession(model, self.torch, self.torch.float16,
                                               self.counters, self.budget, self.started)

    def test_training_mode_is_rejected(self):
        self.model.training = True
        with self.assertRaises(BoundaryViolation): self.session()

    def test_output_device_dtype_and_nonfinite_are_fatal(self):
        for attr, value in (("output_device", "cpu"), ("output_dtype", "torch.float32"), ("nan", True)):
            with self.subTest(attr=attr):
                self.setUp(); session = self.session(); setattr(self.model, attr, value)
                with self.assertRaises(BoundaryViolation): session.run(None, actual_feeds(self.case)[0])
                self.assertEqual(self.model.calls, 1)
                self.assertEqual(session.run_successes, 0)

    def test_native_budget_is_reserved_before_call(self):
        self.budget["max_native_batches"] = 0
        session = self.session()
        with self.assertRaises(BoundaryViolation): session.run(None, actual_feeds(self.case)[0])
        self.assertEqual(self.model.calls, 0)

    def test_time_limit_stops_before_native_call(self):
        with self.assertRaises(BoundaryViolation):
            comparison.reserve_native(self.counters, self.budget, time.perf_counter() - 700)
        self.assertEqual(self.counters["native_batch_reservations"], 0)

    def test_success_cell_preserves_negative_logits_and_stable_ties(self):
        engine = Engine(self.case, self.model, self.torch, self.counters, self.budget, self.started)
        row = comparison.run_one(self.case, self.feed,
            {"phase": "hot", "repetition": 0, "case_id": self.case["case_id"], "variant": "mps_fp16"},
            engine, self.counters, self.budget, self.started)
        self.assertEqual(row["status"], "executed")
        self.assertEqual(row["ranked_ids"], [r["chunk_id"] for r in self.case["candidates"]])
        self.assertEqual(row["scores"], [-2.0] * 5)
        self.assertEqual(self.counters["score_attempts"], 1)
        self.assertEqual(self.counters["session_run_successes"], 2)

    def test_score_budget_and_variant_identity_stop_without_scoring(self):
        for change in ("budget", "identity"):
            self.setUp()
            engine = Engine(self.case, self.model, self.torch, self.counters, self.budget, self.started)
            if change == "budget": self.budget["max_score_attempts"] = 0
            else: engine.identity["variant"] = "mps_fp32"
            with self.assertRaises(comparison.CellStopped) as caught:
                comparison.run_one(self.case, self.feed, {"phase": "hot", "case_id": "synthetic", "variant": "mps_fp16"},
                                   engine, self.counters, self.budget, self.started)
            self.assertEqual(caught.exception.record["status"], "stopped")
            self.assertEqual(self.model.calls, 0)

    def test_nonfinite_scores_keep_failed_cell(self):
        engine = Engine(self.case, self.model, self.torch, self.counters, self.budget, self.started)
        engine.values[0] = float("nan")
        with self.assertRaises(comparison.CellStopped) as caught:
            comparison.run_one(self.case, self.feed, {"phase": "hot", "case_id": "synthetic", "variant": "mps_fp16"},
                               engine, self.counters, self.budget, self.started)
        self.assertIsNone(caught.exception.record["scores"])
        self.assertEqual(self.counters["failed_score_calls"], 1)

    def test_preparation_identity_binds_runtime_packages_sources_and_assets(self):
        frozen = {key: {} for key in ("inputs", "source_files", "protected_files", "model_files", "packages", "settings", "runtime")}
        original = comparison.preparation_identity(frozen)
        for key in frozen:
            changed = deepcopy(frozen); changed[key] = {"changed": True}
            self.assertNotEqual(original, comparison.preparation_identity(changed))

    def test_runtime_requires_venv_prefix_not_only_symlink_target(self):
        runtime = {"interpreter": str(Path(comparison.sys.executable).resolve()),
                   "venv_prefix": "/not/the/current/prefix", "environment": {}, "mutable_roots": []}
        with self.assertRaisesRegex(BoundaryViolation, "ISOLATED_INTERPRETER"):
            comparison.validate_runtime(runtime)

    def test_performance_refuses_missing_pilot_gate_before_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); freeze = p / "freeze.json"
            frozen = {key: {} for key in ("inputs", "source_files", "protected_files", "model_files")}
            frozen["source_files"] = frozen["protected_files"] = frozen["model_files"] = []
            freeze.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(BoundaryViolation, "PILOT_GATE_REQUIRED"):
                comparison.execute(freeze, comparison.digest(freeze), p, "performance")

    def test_old_position_buffer_is_accepted_only_after_value_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "synthetic.safetensors"
            header = json.dumps({"roberta.embeddings.position_ids": {
                "dtype": "I64", "shape": [1, 514], "data_offsets": [0, 4112]}}).encode()
            p.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<514q", *range(514)))
            info = {"missing_keys": [], "unexpected_keys": ["roberta.embeddings.position_ids"],
                    "mismatched_keys": [], "error_msgs": []}
            proof = comparison.validate_loading_info(info, p)
            self.assertTrue(proof["ignored_buffer_values_verified"])
            p.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<514q", *([1] * 514)))
            with self.assertRaisesRegex(BoundaryViolation, "BUFFER_VALUES_CHANGED"):
                comparison.validate_loading_info(info, p)

    def test_loading_does_not_allow_random_classifier_or_other_unused_parameters(self):
        for info in ({"missing_keys": ["classifier.out_proj.weight"]},
                     {"unexpected_keys": ["other.weight"]}, {"mismatched_keys": ["classifier"]},
                     {"error_msgs": ["load failed"]}):
            with self.assertRaisesRegex(BoundaryViolation, "LOADING_KEYS_CHANGED"):
                comparison.validate_loading_info(info, Path("not-read"))

    def test_pilot_gate_rejects_boolean_pass_with_bad_values_or_stale_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory).resolve()
            cases = [make_case(1, f"case-{i}") for i in range(12)]
            feeds = {row["case_id"]: build_feed_projection(row, Encoder()) for row in cases}
            ids = list(feeds)
            frozen = {key: {} for key in ("inputs", "source_files", "protected_files", "model_files", "packages", "settings", "runtime")}
            for name, payload in (("cases", cases), ("feed_manifest", {"cases": list(feeds.values())}),
                                  ("pilot_selection", {"case_ids": ids})):
                q = folder / (name + ".json"); q.write_text(json.dumps(payload))
                frozen["inputs"][name] = {"path": str(q), "sha256": comparison.digest(q)}
            freeze = folder / "freeze.json"; freeze.write_text(json.dumps(frozen))
            freeze_sha = comparison.digest(freeze)
            plan = comparison.build_execution_plan(cases, feeds, ids, "pilot")
            refs = [{"path": str(freeze), "sha256": freeze_sha}]
            for name, body in [("execution-plan.json", {"plan": plan, "freeze_sha256": freeze_sha,
                "preparation_identity_sha256": comparison.preparation_identity(frozen)}), ("terminal.json", {
                "status": "complete", "phase": "pilot", "freeze_sha256": freeze_sha, "unexecuted": [],
                "preparation_identity_sha256": comparison.preparation_identity(frozen),
                "counters": {"failed_score_calls": 0}})]:
                q = folder / name; q.write_text(json.dumps(body))
                refs.append({"path": str(q), "sha256": comparison.digest(q)})
            by_id = {case["case_id"]: case for case in cases}
            for i, cell in enumerate(plan["cells"]):
                case = by_id[cell["case_id"]]
                row = {**cell, "status": "executed", "candidate_ids": [c["chunk_id"] for c in case["candidates"]],
                       "input_sha256": case["input_sha256"], "feed_projection_sha256": canonical_digest(feeds[cell["case_id"]]),
                       "scores": [0.0]}
                q = folder / f"cell-{i:04d}.json"; q.write_text(json.dumps(row))
                refs.append({"path": str(q), "sha256": comparison.digest(q)})
            metrics = {key: 0.5 for key in ("recall5", "recall8", "recall10", "mrr24")}
            quality = {"complete": True, "planned_n": 10, "known_n": 10,
                       "reference": metrics, "candidate": metrics,
                       "deltas": {key: 0.0 for key in metrics}, "non_decreasing": True}
            gate = {"schema": "reranker-mps-pilot-gate-v1", "phase": "pilot", "pass": True,
                    "freeze_sha256": freeze_sha, "preparation_identity_sha256": comparison.preparation_identity(frozen),
                    "thresholds": comparison.THRESHOLDS, "checks": {key: True for key in comparison.GATE_CHECKS},
                    "numeric": {"complete": True, "pairs": 12, "mae": 0.0, "max_abs": 0.0},
                    "quality_comparisons": {key: deepcopy(quality) for key in ("ort_cpu4", "mps_fp32")},
                    "reference_files": refs}
            q = folder / "gate.json"
            def validate(value):
                q.write_text(json.dumps(value)); return comparison.validate_pilot_gate(q, comparison.digest(q), frozen, freeze_sha)
            self.assertTrue(validate(gate)["pass"])
            for key, value in (("mae", float("nan")), ("max_abs", 0.2), ("pairs", True), ("pairs", 13)):
                changed = deepcopy(gate); changed["numeric"][key] = value
                with self.assertRaises(BoundaryViolation): validate(changed)
            changed = deepcopy(gate); changed["quality_comparisons"]["ort_cpu4"]["deltas"]["mrr24"] = -0.1
            with self.assertRaises(BoundaryViolation): validate(changed)
            changed = deepcopy(gate)
            extra = folder / "extra.json"; extra.write_text("{}")
            changed["reference_files"][-1] = {"path": str(extra), "sha256": comparison.digest(extra)}
            with self.assertRaisesRegex(BoundaryViolation, "RAW_COVERAGE_CHANGED"): validate(changed)
            (folder / "cell-0000.json").write_text("changed")
            with self.assertRaisesRegex(BoundaryViolation, "FROZEN_REFERENCE_CHANGED"): validate(gate)

    def test_pilot_gate_anchor_cannot_be_self_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "gate.json"; gate.write_text('{"pass": true}')
            with self.assertRaisesRegex(BoundaryViolation, "ANCHOR_CHANGED"):
                comparison.validate_pilot_gate(gate, "a" * 64, {}, "b" * 64)


if __name__ == "__main__":
    unittest.main()
