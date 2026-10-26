"""以合成原件检查报告分母、partial、前向数值门与引用锚点。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import compare_reranker_mlx_int8 as runner
from scripts import report_reranker_mlx_int8 as target


def metric_row(value=1., *, status="executed"):
    return {"status": status, "answerable": True,
            **{k: value if status == "executed" else None for k in target.METRICS}}


class MetricsTests(unittest.TestCase):
    def test_mrr_loss_is_not_hidden_by_recall_gain(self):
        a, b = metric_row(), metric_row()
        b["mrr24"] = .5
        result = target.comparison([a], [b])
        self.assertEqual(result["deltas"]["recall5"], 0)
        self.assertEqual(result["deltas"]["mrr24"], -.5)
        self.assertFalse(result["non_decreasing"])

    def test_partial_denominator_is_unknown_instead_of_observed_only_mean(self):
        result = target.comparison([metric_row(), metric_row()], [metric_row(), metric_row(status="not_executed")])
        self.assertEqual(result["planned_n"], 2)
        self.assertEqual(result["known_n"], 1)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["candidate"]["mrr24"])
        self.assertIsNone(result["non_decreasing"])

    def test_multi_positive_recall_and_missing_gold_have_different_denominators(self):
        values = target.quality(["x", "a", "b"], ["x", "a", "b"], {"a", "b", "c", "d"})
        self.assertEqual(values["recall5"], .5)
        self.assertEqual(values["mrr24"], .5)
        self.assertIsNone(target.quality(["x"], ["x"], set())["mrr24"])

    def test_three_repetitions_do_not_triple_independent_question_count(self):
        values = [1., 2., 3.]
        self.assertEqual(target.distribution(values)["n"], 3)
        self.assertEqual(target.distribution(values)["p95"], 2.9)

    def test_nonfinite_or_negative_wall_is_invalid(self):
        row = {"timing": {k: 1. for k in ("reranker_wall_seconds", "provider_inference_seconds",
                                         "feed_verification_seconds", "native_session_run_seconds")}}
        for value in (float("nan"), -1., 0.):
            row["timing"]["reranker_wall_seconds"] = value
            with self.assertRaises(runner.BoundaryViolation):
                target.checked_timing(row)

    def test_same_path_cannot_bind_two_different_hashes(self):
        with self.assertRaises(runner.BoundaryViolation):
            target.unique_references([{"path": "/a", "sha256": "a"}, {"path": "/a", "sha256": "b"}])


class SyntheticReportTests(unittest.TestCase):
    def make_fixture(self, root, *, stopped=False, error=0.):
        cases = [{"case_id": f"c{i}", "query": f"q{i}", "scope_document_ids": ["doc"],
                  "candidates": [{"chunk_id": f"g{i}"}], "input_sha256": "a" * 64} for i in range(114)]
        pilot = [*[f"c{i}" for i in range(11)], "c112"]
        metadata = [{"case_id": c["case_id"], "query": c["query"], "scope_document_ids": ["doc"],
                     "answerable": i < 112, "split": "regression" if i < 58 else "pending"}
                    for i, c in enumerate(cases)]
        labels = {"positive_pairs": [{"case_id": f"c{i}", "chunk_id": f"g{i}"} for i in range(112)]}
        questions_path, labels_path = root / "questions.json", root / "labels.json"
        questions_path.write_text(json.dumps(metadata))
        labels_path.write_text(json.dumps(labels))
        frozen = {"schema": runner.SCHEMA, "inputs": {},
                  "report_inputs": {"labels": runner.file_reference(labels_path),
                                    "original_questions": runner.file_reference(questions_path)},
                  "source_files": [runner.file_reference(Path(target.__file__).resolve())],
                  "protected_files": [], "model_files": [], "historical_pilot_files": [],
                  "packages": {}, "settings": {}, "runtime": {}}
        freeze = root / "freeze.json"
        freeze.write_text(json.dumps(frozen))
        freeze_hash = runner.digest(freeze)
        cells = [{"phase": "warmup", "case_id": pilot[0], "variant": "mlx_fp32"}]
        cells += [{"phase": "hot", "repetition": 0, "ordinal": i, "case_id": cid, "variant": "mlx_fp32"}
                  for i, cid in enumerate(pilot)]
        plan = {"cells": cells, "phase": "float_qualification", "case_ids": pilot,
                "variants": ["mlx_fp32"], "hot_repeats": 1,
                "budget": {"max_score_attempts": 13, "max_pairs": 13,
                           "max_native_batches": 13, "max_seconds": 600}}
        terminal = {"phase": "float_qualification", "status": "stopped" if stopped else "complete",
                    "prepare_only": False, "unexecuted": cells[1:] if stopped else [],
                    "counters": {"score_attempts": 1 if stopped else 13, "pair_attempts": 1 if stopped else 13,
                                 "native_batch_reservations": 1 if stopped else 13,
                                 "session_run_attempts": 1 if stopped else 13,
                                 "session_run_successes": 1 if stopped else 13}}
        execution = root / "execution"
        execution.mkdir()
        (execution / "terminal.json").write_text(json.dumps(terminal))
        (execution / "execution-plan.json").write_text(json.dumps(plan))
        rows, references = {}, []
        for index, cell in enumerate(cells[:1] if stopped else cells):
            cid = cell["case_id"]
            row = {**cell, "status": "executed", "scores": [1. + error], "candidate_ids": ["g" + cid[1:]],
                   "ranked_ids": ["g" + cid[1:]], "timing": {k: 1. for k in (
                       "reranker_wall_seconds", "provider_inference_seconds", "feed_verification_seconds", "native_session_run_seconds")}}
            rows[runner.cell_key(row)] = row
            path = execution / f"cell-{index:04d}.json"
            path.write_text(json.dumps(row))
            references.append(runner.file_reference(path))
        history = {(cid, variant): {"status": "executed", "scores": [1.],
                    "candidate_ids": ["g" + cid[1:]], "ranked_ids": ["g" + cid[1:]]}
                   for cid in pilot for variant in ("ort_cpu4", "mps_fp32")}
        context = (plan, terminal, rows, references, cases, {}, pilot, {})
        return freeze, freeze_hash, frozen, execution, context, history

    def test_complete_report_binds_source_summary_raw_and_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze, sha, frozen, execution, context, history = self.make_fixture(root)
            output = root / "report"
            output.mkdir()
            with patch.object(target, "read_execution", return_value=context), patch.object(target, "historical_rows", return_value=history):
                result = target.report(freeze, sha, execution, output)
            self.assertTrue(result["all_cells_executed"])
            self.assertTrue(result["gate_pass"])
            self.assertEqual((result["answerable_cases"], result["gap_cases"]), (11, 1))
            gate = json.loads((output / "gate.json").read_text())
            self.assertEqual(gate["quality_vs_cpu4"]["planned_n"], 11)
            self.assertEqual(gate["report_summary_ref"]["sha256"], runner.digest(output / "summary.json"))
            with patch.object(runner, "read_execution", return_value=context), patch.object(runner, "historical_rows", return_value=history):
                runner.validate_gate(output / "gate.json", runner.digest(output / "gate.json"), frozen, sha, "float_qualification")
            # 没有修改门票，篡改其引用的报告原件仍会被拒绝。
            (output / "summary.json").write_text("{}")
            with self.assertRaises(runner.BoundaryViolation):
                runner.validate_gate(output / "gate.json", runner.digest(output / "gate.json"), frozen, sha, "float_qualification")

    def test_partial_keeps_all_eleven_planned_positives_and_cannot_open_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze, sha, _, execution, context, history = self.make_fixture(root, stopped=True)
            output = root / "report"
            output.mkdir()
            with patch.object(target, "read_execution", return_value=context), patch.object(target, "historical_rows", return_value=history):
                result = target.report(freeze, sha, execution, output)
            self.assertFalse(result["gate_pass"])
            aggregate = next(r for r in result["aggregates"] if r["split"] == "all")
            self.assertEqual((aggregate["planned_n"], aggregate["known_n"]), (11, 0))
            self.assertIsNone(aggregate["mrr24"])
            self.assertEqual(result["unexecuted_cells"], 12)

    def test_forward_difference_cannot_be_laundered_by_equal_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze, sha, _, execution, context, history = self.make_fixture(root, error=.02)
            output = root / "report"
            output.mkdir()
            with patch.object(target, "read_execution", return_value=context), patch.object(target, "historical_rows", return_value=history):
                result = target.report(freeze, sha, execution, output)
            self.assertFalse(result["gate_pass"])
            self.assertTrue(result["quality_comparisons"]["ort_cpu4"]["non_decreasing"])
            self.assertGreater(result["numeric"]["max_abs"], .01)

    def test_changed_label_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, frozen, _, context, _ = self.make_fixture(root)
            questions = Path(frozen["report_inputs"]["original_questions"]["path"])
            data = json.loads(questions.read_text())
            data[0]["scope_document_ids"] = ["foreign"]
            questions.write_text(json.dumps(data))
            frozen["report_inputs"]["original_questions"] = runner.file_reference(questions)
            with self.assertRaisesRegex(runner.BoundaryViolation, "LABEL_IDENTITY"):
                target.load_labels(frozen, context[4])


if __name__ == "__main__":
    unittest.main()
