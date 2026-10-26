"""用合成原件检验MPS报告与阶段门，不导入或执行真实模型。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import report_reranker_mps as report


def save(path: Path, value) -> dict:
    """保存可修改的合成夹具，返回实际原件引用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return report.reference(path)


def fixture(root: Path, phase: str = "pilot") -> dict:
    """构造两道有金标题和一缺证题，保留所有热格与初始化。"""
    root = root.resolve()
    cases, feeds, questions, positives = [], [], [], []
    for index, count in enumerate((6, 2, 1)):
        cid = f"synthetic-{index}"
        candidates = []
        for position in range(count):
            text = f"Original synthetic passage {index}-{position}."
            candidates.append({"chunk_id": f"chunk-{index}-{position}", "document_id": "document",
                "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "evidence_id": "evidence", "index_id": "index", "source_sha256": "a" * 64,
                "page_number": 1, "char_start": 0, "char_end": len(text)})
        case = {"case_id": cid, "query": f"Original query {index}",
                "scope_document_ids": ["document"], "candidates": candidates}
        case["input_sha256"] = report.canonical_sha(case)
        cases.append(case)
        batches = []
        for offset in range(0, count, 4):
            indices = list(range(offset, min(offset + 4, count)))
            tensors = {name: {"dtype": "int64", "byte_order": "little", "shape": [len(indices), 3],
                             "sha256": hashlib.sha256(f"{cid}:{offset}:{name}".encode()).hexdigest()}
                       for name in ("input_ids", "attention_mask", "token_type_ids_if_required")}
            batches.append({"candidate_indices": indices, "tensors": tensors})
        feeds.append({"case_id": cid, "input_sha256": case["input_sha256"], "batches": batches})
        questions.append({"case_id": cid, "query": case["query"], "scope_document_ids": ["document"],
                          "answerable": index < 2, "split": ("regression", "pending", "diagnostic")[index]})
        if index < 2:
            positives.append({"case_id": cid, "chunk_id": candidates[0]["chunk_id"]})
    spec = {"repo": "synthetic-model", "revision": "fixed-revision", "weights": "model.onnx",
            "files": {"model.onnx": {"sha256": "b" * 64}, "tokenizer.json": {"sha256": "c" * 64}}}
    float_spec = {"repo": spec["repo"], "revision": spec["revision"], "weights": "model.safetensors",
                  "files": {"model.safetensors": {"sha256": "d" * 64}}}
    frozen = {"schema": "reranker-mps-execution-v1", "inputs": {
        "cases": save(root / "cases.json", cases), "feed_manifest": save(root / "feeds.json", {"cases": feeds}),
        "model_manifest": save(root / "model-manifest.json", {"reranker": spec}),
        "float_manifest": save(root / "float-manifest.json", float_spec),
        "pilot_selection": save(root / "pilot-selection.json", {"case_ids": [case["case_id"] for case in cases]})},
        "source_files": [], "protected_files": [], "model_files": [], "packages": {},
        "settings": {"batch_size": 4}, "runtime": {"mutable_roots": []},
        "report_inputs": {"labels": save(root / "labels.json", {"positive_pairs": positives}),
                          "original_questions": save(root / "questions.json", questions)}}
    freeze = root / "freeze.json"
    save(freeze, frozen)
    execution = root / "execution"
    execution.mkdir()
    variants = report.VARIANTS[phase]
    repetitions = 1 if phase == "pilot" else 3
    cells = [{"phase": "warmup", "case_id": cases[0]["case_id"], "variant": variant,
              "state": "not_executed"} for variant in variants]
    cells += [{"phase": "hot", "repetition": repeat, "case_id": case["case_id"], "variant": variant,
               "state": "not_executed"} for repeat in range(repetitions) for case in cases for variant in variants]
    planned_pairs = sum(len(next(c for c in cases if c["case_id"] == row["case_id"])["candidates"]) for row in cells)
    planned_batches = sum(len(next(f for f in feeds if f["case_id"] == row["case_id"])["batches"]) for row in cells)
    save(execution / "execution-plan.json", {"freeze_sha256": report.file_sha(freeze),
        "preparation_identity_sha256": report.preparation_identity(frozen),
        "measurement": report.MEASUREMENT, "labels_parsed": False,
        "plan": {"cells": cells, "phase": phase, "variants": list(variants), "hot_repeats": repetitions,
                 "case_ids": [case["case_id"] for case in cases],
                 "budget": {"max_score_attempts": len(cells), "max_pairs": planned_pairs,
                            "max_native_batches": planned_batches}}})
    rows = []
    for sequence, cell in enumerate(cells):
        case = next(case for case in cases if case["case_id"] == cell["case_id"])
        feed = next(feed for feed in feeds if feed["case_id"] == cell["case_id"])
        variant = cell["variant"]
        ids = [c["chunk_id"] for c in case["candidates"]]
        scores = [float(len(ids) - i) + (.001 if variant == "mps_fp16" else 0.) for i in range(len(ids))]
        dtype = "torch.float16" if variant == "mps_fp16" else "torch.float32"
        is_mps = variant != "ort_cpu4"
        model = {"model_id": spec["repo"], "revision": spec["revision"],
                 "weights_sha256": "d" * 64 if is_mps else "b" * 64, "tokenizer_sha256": "c" * 64,
                 "batch_size": 4, "max_pair_tokens": 512, "sort_by_length": False,
                 "score_kind": "raw_relevance_logit", "threads": 4}
        if is_mps:
            model.update(device="mps:0", dtype=dtype, parameter_devices=["mps:0"], parameter_dtypes=[dtype])
        session = {"input_names": ["input_ids", "attention_mask"], "intra_op_num_threads": 4,
                   "inter_op_num_threads": 1}
        if is_mps:
            session.update(backend="pytorch_mps", cpu_fallback_enabled=False, device="mps:0", dtype=dtype)
        else:
            session["providers"] = ["CPUExecutionProvider"]
        proofs = []
        for batch_index, batch in enumerate(feed["batches"]):
            tensors = {name: batch["tensors"][name] for name in session["input_names"]}
            proof = {"batch_index": batch_index, "candidate_indices": batch["candidate_indices"],
                     "actual_tensors": tensors, "verification_passed": True,
                     "native_attempted": True, "native_succeeded": True, "error": None}
            if is_mps:
                proof["device_proof"] = {"input_devices": {name: "mps:0" for name in tensors},
                    "input_dtypes": {name: "torch.int64" for name in tensors},
                    "model_parameter_devices": ["mps:0"], "model_parameter_dtypes": [dtype],
                    "transferred_tensor_hashes": tensors, "sync_before": True, "sync_after": True,
                    "output_device": "mps:0", "output_dtype": dtype}
            proofs.append(proof)
        row = {key: value for key, value in cell.items() if key != "state"}
        row.update(status="executed", error=None, query=case["query"], scope_document_ids=["document"],
            candidate_ids=ids, input_sha256=case["input_sha256"], feed_projection_sha256=report.canonical_sha(feed),
            scores=scores, ranked_ids=ids, model_identity=model, session_identity=session,
            measurement=report.MEASUREMENT,
            hits=[{"chunk_id": c["chunk_id"], "document_id": c["document_id"], "text_sha256": c["text_sha256"],
                   "rank": i + 1, "candidate_rank": i + 1, "reranker_score": scores[i]}
                  for i, c in enumerate(case["candidates"])],
            timing={"reranker_wall_seconds": 1. if is_mps else 2.,
                    "provider_inference_seconds": .9 if is_mps else 1.9,
                    "native_session_run_seconds": .8 if is_mps else 1.8, "feed_verification_seconds": .1},
            feed_verification={"complete": True, "expected_batches": len(proofs), "verified_batches": len(proofs),
                "session_run_attempts": len(proofs), "session_run_successes": len(proofs), "batches": proofs})
        path = execution / f"cell-{sequence:03d}.json"
        save(path, row)
        rows.append((path, row))
    save(execution / "terminal.json", {"status": "complete", "freeze_sha256": report.file_sha(freeze),
        "preparation_identity_sha256": report.preparation_identity(frozen), "labels_parsed": False,
        "phase": phase, "measurement": report.MEASUREMENT, "unexecuted": [], "prepare_only": False,
        "error": None, "counters": {"score_attempts": len(cells), "successful_score_calls": len(cells),
            "pair_attempts": planned_pairs, "successful_pairs": planned_pairs, "failed_score_calls": 0,
            "session_run_attempts": planned_batches, "session_run_successes": planned_batches,
            "verified_batches": planned_batches, "native_batch_reservations": planned_batches,
            "hot_rows": len(cases) * len(variants) * repetitions, "warmup_rows": len(variants)}})
    output = root / "analysis"
    output.mkdir()
    return {"root": root, "freeze": freeze, "frozen": frozen, "execution": execution, "output": output,
            "rows": rows, "cases": cases, "feeds": feeds, "phase": phase}


def run(f: dict) -> dict:
    """以合成冻结锚点运行只读报告器。"""
    return report.build_report(f["freeze"], f["execution"], f["output"],
        expected_freeze_sha256=report.file_sha(f["freeze"]), phase=f["phase"])


def hot(f: dict, variant: str, cid: str = "synthetic-0", repeat: int = 0) -> tuple:
    """查找单个热格，允许测试直接修改其原件。"""
    return next((path, row) for path, row in f["rows"] if row["phase"] == "hot"
                and row["variant"] == variant and row["case_id"] == cid and row["repetition"] == repeat)


def rescore(row: dict, scores: list[float]) -> None:
    """同时修正合成分数、排序与来源，避免测试只打破冗余字段。"""
    row["scores"] = scores
    ids = row["candidate_ids"]
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
    old = {hit["chunk_id"]: hit for hit in row["hits"]}
    row["ranked_ids"] = [ids[i] for i in order]
    row["hits"] = [{**old[ids[i]], "rank": rank, "candidate_rank": i + 1, "reranker_score": scores[i]}
                   for rank, i in enumerate(order, 1)]


class MPSReportTests(unittest.TestCase):
    """验证门禁、失败分母、数值、实际设备与原件身份。"""

    def test_complete_pilot_gate_and_bound_references(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            result = run(f)
            self.assertTrue(result["pilot_gate"]["pass"])
            self.assertEqual((result["planned_cells"], result["observed_cells"]), (12, 12))
            self.assertEqual((result["answerable_cases"], result["gap_cases"]), (2, 1))
            self.assertAlmostEqual(result["fp16_vs_fp32_numeric"]["mae"], .001)
            self.assertEqual(result["fp16_vs_fp32_numeric"]["pairs"], 9)
            receipt = json.loads((f["output"] / "pilot-gate.json").read_text())
            self.assertEqual(receipt["thresholds"], report.THRESHOLDS)
            self.assertEqual(len(receipt["reference_files"]), 15)
            for ref in receipt["reference_files"]:
                self.assertEqual(report.file_sha(Path(ref["path"])), ref["sha256"])
            self.assertFalse(receipt["automatic_adoption"])

    def test_fp16_numeric_limit_fails_even_when_ranking_is_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path, row = hot(f, "mps_fp16")
            rescore(row, [score + .2 for score in row["scores"]])
            save(path, row)
            result = run(f)
            self.assertFalse(result["pilot_gate"]["pass"])
            self.assertFalse(result["pilot_gate"]["checks"]["fp16_numeric_within_limits"])
            self.assertTrue(result["quality_comparisons"]["mps_fp32"]["non_decreasing"])

    def test_quality_regression_is_not_hidden_by_numeric_success(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            for variant in ("ort_cpu4", "mps_fp32", "mps_fp16"):
                path, row = hot(f, variant)
                scores = [0.006 - i * .001 for i in range(6)]
                if variant == "mps_fp16":
                    scores[0] = 0.
                rescore(row, scores)
                save(path, row)
            result = run(f)
            self.assertTrue(result["pilot_gate"]["checks"]["fp16_numeric_within_limits"])
            self.assertFalse(result["quality_comparisons"]["ort_cpu4"]["non_decreasing"])
            self.assertLess(result["quality_comparisons"]["ort_cpu4"]["deltas"]["mrr24"], 0)
            self.assertFalse(result["pilot_gate"]["pass"])

    def test_missing_cell_remains_na_with_original_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path, _ = hot(f, "mps_fp16")
            path.unlink()
            result = run(f)
            aggregate = next(row for row in result["aggregates"] if row["variant"] == "mps_fp16" and row["split"] == "all")
            self.assertEqual((aggregate["known_n"], aggregate["planned_n"]), (1, 2))
            self.assertIsNone(aggregate["mrr24"])
            self.assertEqual(result["unexecuted_cells"], 1)
            self.assertEqual(result["paired_timing"]["n"], 1)
            self.assertFalse(result["pilot_gate"]["pass"])

    def test_failed_cell_retains_zero_quality_and_fails_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path, row = hot(f, "mps_fp16")
            row.update(status="failed", scores=None, ranked_ids=[], hits=[], error={"type": "Synthetic"})
            save(path, row)
            result = run(f)
            aggregate = next(row for row in result["aggregates"] if row["variant"] == "mps_fp16" and row["split"] == "all")
            self.assertEqual((aggregate["known_n"], aggregate["planned_n"]), (2, 2))
            self.assertEqual(aggregate["mrr24"], .5)
            self.assertFalse(result["pilot_gate"]["pass"])

    def test_missing_terminal_cannot_pass_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            (f["execution"] / "terminal.json").unlink()
            result = run(f)
            self.assertFalse(result["all_cells_executed"])
            self.assertFalse(result["pilot_gate"]["pass"])

    def test_complete_terminal_counter_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path = f["execution"] / "terminal.json"
            terminal = json.loads(path.read_text())
            terminal["counters"]["session_run_attempts"] -= 1
            save(path, terminal)
            with self.assertRaisesRegex(ValueError, "COMPLETE_COUNTERS"):
                run(f)

    def test_single_pilot_observation_does_not_prove_stability(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            result = run(f)
            self.assertFalse(result["stability"]["mps_fp16"]["assessment_available"])
            rows = json.loads((f["output"] / "per-case.json").read_text())
            self.assertTrue(all(row["ranking_stable"] is None and row["scores_stable"] is None for row in rows))

    def test_actual_feed_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path, row = hot(f, "mps_fp16")
            row["feed_verification"]["batches"][0]["actual_tensors"]["input_ids"]["sha256"] = "e" * 64
            save(path, row)
            with self.assertRaisesRegex(ValueError, "ACTUAL_TENSOR"):
                run(f)

    def test_unsynchronized_mps_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path, row = hot(f, "mps_fp16")
            row["feed_verification"]["batches"][0]["device_proof"]["sync_after"] = False
            save(path, row)
            with self.assertRaisesRegex(ValueError, "SYNCHRONIZATION"):
                run(f)

    def test_output_on_cpu_cannot_claim_mps_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path, row = hot(f, "mps_fp16")
            row["feed_verification"]["batches"][0]["device_proof"]["output_device"] = "cpu"
            save(path, row)
            with self.assertRaisesRegex(ValueError, "DEVICE_OR_SYNCHRONIZATION"):
                run(f)

    def test_wrong_parameter_precision_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path, row = hot(f, "mps_fp16")
            row["model_identity"]["parameter_dtypes"] = ["torch.float32"]
            save(path, row)
            with self.assertRaisesRegex(ValueError, "MPS_SESSION"):
                run(f)

    def test_cpu_fallback_cannot_be_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path, row = hot(f, "mps_fp16")
            row["session_identity"]["cpu_fallback_enabled"] = True
            save(path, row)
            with self.assertRaisesRegex(ValueError, "MPS_SESSION"):
                run(f)

    def test_duplicate_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            _, row = hot(f, "mps_fp16")
            save(f["execution"] / "duplicate.json", row)
            with self.assertRaisesRegex(ValueError, "DUPLICATE"):
                run(f)

    def test_changed_freeze_anchor_fails_before_reading_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, "FREEZE_ANCHOR"):
                report.build_report(f["freeze"], f["execution"], f["output"],
                    expected_freeze_sha256="f" * 64, phase="pilot")

    def test_changed_label_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            Path(f["frozen"]["report_inputs"]["labels"]["path"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "REFERENCE_CHANGED"):
                run(f)

    def test_performance_uses_per_case_three_repeat_medians(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder), phase="performance")
            path, row = hot(f, "mps_fp16", repeat=2)
            row["timing"]["reranker_wall_seconds"] = 100.
            save(path, row)
            result = run(f)
            self.assertEqual((result["planned_cells"], result["observed_cells"]), (20, 20))
            self.assertNotIn("pilot_gate", result)
            self.assertFalse((f["output"] / "pilot-gate.json").exists())
            self.assertEqual(result["paired_timing"]["reranker_wall_seconds"]["mps_fp16"]["p50"], 1.)
            self.assertEqual(result["paired_timing"]["reranker_wall_seconds"]["ort_cpu4"]["p50"], 2.)
            self.assertEqual(result["paired_timing"]["n"], 2)
            self.assertIn("后端和精度", (f["output"] / "report.md").read_text())

    def test_performance_preserves_ranking_and_score_instability(self):
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder), phase="performance")
            path, row = hot(f, "mps_fp16", repeat=2)
            rescore(row, list(reversed(row["scores"])))
            save(path, row)
            result = run(f)
            self.assertEqual(result["stability"]["mps_fp16"]["ranking_changed_cases"], 1)
            self.assertEqual(result["stability"]["mps_fp16"]["scores_changed_cases"], 1)


if __name__ == "__main__":
    unittest.main()
