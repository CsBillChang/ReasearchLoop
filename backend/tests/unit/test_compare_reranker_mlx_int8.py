"""用替身验证新阶段预算、前向门与MLX实际输入证明的拒绝边界。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import compare_reranker_mlx_int8 as target


def population():
    """合成相同规模与预算，不引用任何本机私有材料。"""
    sizes = [24] * 114
    for index, size in ((12, 1), (13, 13), (14, 13), (15, 14)):
        sizes[index] = size
    cases = [{"case_id": f"c{i}"} for i in range(114)]
    feeds = {f"c{i}": {"pairs": n, "batches": [{} for _ in range((n + 3) // 4)]}
             for i, n in enumerate(sizes)}
    ids = ["c12", *[f"c{i}" for i in range(10)], "c112"]
    return cases, feeds, ids


def proof(variant, actual):
    quantized = variant == "mlx_w8a32"
    return {"input_tensor_references": actual, "transferred_tensor_references": actual,
            "gpu_stream_verified": True, "gpu_eval_complete": True, "sync_before": True,
            "sync_after": True, "quantization_verified": True, "output_dtype": "float32",
            "activation_dtype": "float32", "linear_call_count": 74,
            "linear_call_names": [f"layer{i}" for i in range(74)],
            "quantized_parameters_sha256": "d" * 64,
            "quantized_linear_count": 74 if quantized else 0,
            "quantized_matmul_calls": 74 if quantized else 0,
            "bits": 8, "group_size": 64, "weight_packing": "uint32",
            "quantized_layer_names": [f"layer{i}" for i in range(74)] if quantized else []}


class PlanTests(unittest.TestCase):
    def test_three_stage_budget_is_finite_and_exact(self):
        cases, feeds, ids = population()
        plans = [target.build_execution_plan(cases, feeds, ids, phase) for phase in target.PHASES]
        self.assertEqual([p["budget"]["max_score_attempts"] for p in plans], [13, 13, 686])
        self.assertEqual([p["budget"]["max_pairs"] for p in plans], [266, 266, 16134])
        self.assertEqual([p["budget"]["max_native_batches"] for p in plans], [68, 68, 4050])
        self.assertEqual(sum(p["budget"]["max_native_batches"] for p in plans), 4186)
        self.assertEqual(sum(p["budget"]["max_score_attempts"] for p in plans), 712)

    def test_performance_alternates_first_variant_and_keeps_candidates(self):
        cases, feeds, ids = population()
        p = target.build_execution_plan(cases, feeds, ids, "performance")
        hot = [c for c in p["cells"] if c["phase"] == "hot"]
        self.assertEqual([c["variant"] for c in hot[:4]], ["ort_cpu4", "mlx_w8a32", "mlx_w8a32", "ort_cpu4"])
        self.assertEqual(len({target.cell_key(c) for c in p["cells"]}), 686)

    def test_no_alternative_precision_or_replacement_pilot(self):
        cases, feeds, ids = population()
        for phase, selected in (("fp16", ids), ("pilot", ids[:-1]), ("pilot", ids[:-1] + [ids[0]])):
            with self.subTest(phase=phase, ids=selected):
                with self.assertRaises(target.BoundaryViolation):
                    target.build_execution_plan(cases, feeds, selected, phase)

    def test_native_reservation_rejects_before_increment(self):
        counts = target.new_counters()
        counts["native_batch_reservations"] = 1
        with self.assertRaises(target.BoundaryViolation):
            target.reserve_native(counts, {"max_native_batches": 1, "max_seconds": 100}, time.perf_counter())
        self.assertEqual(counts["native_batch_reservations"], 1)

    def test_isolation_checks_prefix_not_only_interpreter_symlink(self):
        with patch.object(target.sys, "prefix", "/unexpected"), patch.object(target.sys, "base_prefix", "/base"):
            with self.assertRaisesRegex(target.BoundaryViolation, "ISOLATED_INTERPRETER"):
                target.validate_runtime({"interpreter": target.sys.executable, "venv_prefix": "/expected"})


class ProofTests(unittest.TestCase):
    def setUp(self):
        self.actual = {name: {"dtype": "int64", "shape": [1, 3], "byte_order": "little", "sha256": char * 64}
                       for name, char in (("input_ids", "a"), ("attention_mask", "b"))}

    def test_both_expected_execution_modes(self):
        for variant in ("mlx_fp32", "mlx_w8a32"):
            target.validate_mlx_proof(proof(variant, self.actual), self.actual, variant)

    def test_missing_sync_or_wrong_precision_cannot_be_success(self):
        for key, value in (("sync_after", False), ("gpu_eval_complete", False),
                           ("activation_dtype", "float16"), ("quantized_matmul_calls", 73),
                           ("weight_packing", "int8"), ("bits", 4), ("group_size", 128)):
            altered = proof("mlx_w8a32", self.actual)
            altered[key] = value
            with self.subTest(key=key):
                with self.assertRaises(target.BoundaryViolation):
                    target.validate_mlx_proof(altered, self.actual, "mlx_w8a32")

    def test_duplicate_layer_or_changed_input_is_rejected(self):
        altered = proof("mlx_w8a32", self.actual)
        altered["quantized_layer_names"][-1] = altered["quantized_layer_names"][0]
        with self.assertRaises(target.BoundaryViolation):
            target.validate_mlx_proof(altered, self.actual, "mlx_w8a32")
        altered = proof("mlx_w8a32", self.actual)
        altered["transferred_tensor_references"] = {}
        with self.assertRaises(target.BoundaryViolation):
            target.validate_mlx_proof(altered, self.actual, "mlx_w8a32")

    def test_float_mode_must_not_claim_quantized_kernel(self):
        with self.assertRaises(target.BoundaryViolation):
            target.validate_mlx_proof(proof("mlx_w8a32", self.actual), self.actual, "mlx_fp32")

    def test_quantized_parameter_digest_cannot_change_between_batches(self):
        with self.assertRaises(target.BoundaryViolation):
            target.validate_mlx_proof(proof("mlx_w8a32", self.actual), self.actual, "mlx_w8a32",
                                      parameter_digest="e" * 64)

    def test_wrapper_reserves_once_and_keeps_actual_proof(self):
        native = type("Native", (), {})()
        native.identity = {"backend": "mlx", "variant": "mlx_w8a32", "device": "gpu"}
        native.batch_proofs = []
        def run(names, feed):
            native.batch_proofs.append(proof("mlx_w8a32", feed))
            return [[[1.]]]
        native.run = run
        counts = target.new_counters()
        session = target.MlxVerifiedSession(native, "mlx_w8a32", counts,
            {"max_native_batches": 1, "max_seconds": 100}, time.perf_counter())
        session.reset({"batches": [{"candidate_indices": [0], "tensors": self.actual}]})
        with patch.object(target, "array_reference", side_effect=lambda value: deepcopy(value)):
            session.run(None, self.actual)
        self.assertTrue(session.snapshot()["complete"])
        self.assertEqual(counts["native_batch_reservations"], 1)
        self.assertEqual(session.snapshot()["batches"][0]["device_proof"]["quantized_matmul_calls"], 74)

    def test_native_failure_keeps_partial_quantized_operation_count(self):
        native = type("Native", (), {})()
        native.identity = {"backend": "mlx", "variant": "mlx_w8a32", "device": "gpu"}
        native.batch_proofs = []
        def run(*_):
            native.batch_proofs.append({"linear_call_count": 7, "quantized_matmul_calls": 7,
                                        "gpu_eval_complete": False})
            raise RuntimeError("synthetic native failure")
        native.run = run
        counts = target.new_counters()
        session = target.MlxVerifiedSession(native, "mlx_w8a32", counts,
            {"max_native_batches": 1, "max_seconds": 100}, time.perf_counter())
        session.reset({"batches": [{"candidate_indices": [0], "tensors": self.actual}]})
        with patch.object(target, "array_reference", side_effect=lambda value: deepcopy(value)):
            with self.assertRaisesRegex(RuntimeError, "synthetic native failure"):
                session.run(None, self.actual)
        saved = session.snapshot()
        self.assertFalse(saved["complete"])
        self.assertEqual(saved["session_run_attempts"], 1)
        self.assertEqual(saved["session_run_successes"], 0)
        self.assertEqual(saved["batches"][0]["device_proof"]["quantized_matmul_calls"], 7)

    def test_changed_feed_prevents_native_call_and_budget_use(self):
        native = type("Native", (), {})()
        native.identity = {"backend": "mlx", "variant": "mlx_w8a32", "device": "gpu"}
        native.batch_proofs = []
        native.run = lambda *_: self.fail("Wrong feed reached native kernel")
        counts = target.new_counters()
        session = target.MlxVerifiedSession(native, "mlx_w8a32", counts,
            {"max_native_batches": 1, "max_seconds": 100}, time.perf_counter())
        session.reset({"batches": [{"candidate_indices": [0], "tensors": self.actual}]})
        changed = deepcopy(self.actual)
        changed["input_ids"]["sha256"] = "c" * 64
        with patch.object(target, "array_reference", side_effect=lambda value: deepcopy(value)):
            with self.assertRaises(target.BoundaryViolation):
                session.run(None, changed)
        self.assertEqual(counts["native_batch_reservations"], 0)
        self.assertEqual(session.snapshot()["session_run_attempts"], 0)


class GateTests(unittest.TestCase):
    def test_nan_or_unpaired_numeric_cannot_qualify(self):
        for a, b in (([float("nan")], [0.]), ([1.], []), ([True], [1.])):
            with self.assertRaises(target.BoundaryViolation):
                target.score_errors(a, b)

    def test_self_reported_pass_without_expected_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "gate.json"
            p.write_text(json.dumps({"pass": True}))
            with self.assertRaisesRegex(target.BoundaryViolation, "ANCHOR_CHANGED"):
                target.validate_gate(p, "0" * 64, {}, "f" * 64, "float_qualification")

    def test_scores_cannot_change_order_without_rank_record_changing(self):
        case = {"case_id": "c", "query": "q", "scope_document_ids": ["d"], "input_sha256": "a" * 64,
                "candidates": [{"chunk_id": "a"}, {"chunk_id": "b"}]}
        feed = {"case_id": "c"}
        row = {"query": "q", "scope_document_ids": ["d"], "input_sha256": "a" * 64,
               "feed_projection_sha256": target.canonical_digest(feed), "candidate_ids": ["a", "b"],
               "status": "executed", "scores": [1., 2.], "ranked_ids": ["a", "b"]}
        with self.assertRaisesRegex(target.BoundaryViolation, "RANKING_CHANGED"):
            target.validate_row_identity(row, case, feed)
        row["scores"] = [1., 1.]
        target.validate_row_identity(row, case, feed)


if __name__ == "__main__":
    unittest.main()
