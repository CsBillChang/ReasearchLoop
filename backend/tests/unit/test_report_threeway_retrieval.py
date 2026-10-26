"""以合成三路查询原件验证报告分母、时间边界、失败与身份拒绝。"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

from scripts import report_threeway_retrieval as target


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False))


def mps_snapshot():
    tensors = {name: {"dtype": "int64", "byte_order": "little", "shape": [2, 3],
                      "sha256": char * 64} for name, char in (("input_ids", "a"), ("attention_mask", "b"))}
    return {"complete": True, "expected_batches": 1, "verified_batches": 1,
            "session_run_attempts": 1, "session_run_successes": 1, "batches": [{
                "batch_index": 0, "candidate_indices": [0, 1], "actual_tensors": tensors,
                "verification_passed": True, "native_attempted": True, "native_succeeded": True,
                "device_proof": {"sync_before": True, "sync_after": True,
                    "transferred_tensor_hashes": tensors, "output_dtype": "torch.float16", "output_device": "mps:0",
                    "input_dtypes": {"input_ids": "torch.int64", "attention_mask": "torch.int64"},
                    "input_devices": {"input_ids": "mps:0", "attention_mask": "mps:0"},
                    "model_parameter_dtypes": ["torch.float16"], "model_parameter_devices": ["mps:0"]}}]}


def vector_snapshot():
    value = mps_snapshot()
    value["batches"][0]["candidate_indices"] = [0]
    value["batches"][0].pop("device_proof")
    for ref in value["batches"][0]["actual_tensors"].values():
        ref["shape"] = [1, 4]
    return value


def fixture(root, *, historical=False):
    corpus, queries, metadata, positives = [], [], [], []
    for i in range(114):
        query = {"case_id": f"q{i}", "query": f"question {i}", "scope_document_ids": [f"d{i}"]}
        queries.append(query)
        metadata.append({**query, "answerable": i < 112, "split": "regression" if i < 58 else "pending"})
        if i < 112:
            positives.append({"case_id": f"q{i}", "chunk_id": f"a{i}"})
        for prefix in ("a", "b"):
            text = f"synthetic evidence {prefix} {i}"
            corpus.append({"chunk_id": f"{prefix}{i}", "document_id": f"d{i}", "text": text,
                           "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    embedding_feeds = {q["case_id"]: {"pairs": 1, "batches": [{"candidate_indices": [0],
        "tensors": deepcopy(vector_snapshot()["batches"][0]["actual_tensors"])}]} for q in queries}
    data = {}
    for name, value in (("chunks", corpus), ("queries", queries), ("labels", {"positive_pairs": positives}),
                        ("original_questions", metadata), ("embedding_feeds", embedding_feeds)):
        path = root / (name + ".json")
        dump(path, value)
        data[name] = target.file_reference(path)
    frozen = {"schema": "threeway-retrieval-execution-v1",
              "inputs": {name: data[name] for name in ("chunks", "queries", "embedding_feeds")},
              "report_inputs": {name: data[name] for name in ("labels", "original_questions")},
              "source_files": [], "model_files": [], "protected_files": [],
              "budget": {"max_query_calls": 1029, "max_seconds": 3600}}
    cells = [{"phase": "warmup", "case_id": "q0", "route": route, "state": "not_executed"} for route in target.ROUTES]
    for repetition in range(3):
        for ordinal, query in enumerate(queries):
            order = list(target.ROUTES)
            rotation = (repetition + ordinal) % 3
            order = order[rotation:] + order[:rotation]
            cells += [{"phase": "hot", "case_id": query["case_id"], "route": route,
                       "state": "not_executed", "repetition": repetition, "ordinal": ordinal} for route in order]
    plan = {"routes": list(target.ROUTES), "case_ids": [q["case_id"] for q in queries], "hot_repeats": 3,
            "budget": frozen["budget"], "cells": cells}
    lookup = {c["chunk_id"]: c for c in corpus}
    execution = root / "execution"
    execution.mkdir()
    rows = []
    for index, cell in enumerate(cells):
        i = int(cell["case_id"][1:]); route = cell["route"]
        empty = cell["phase"] == "warmup" or route == "located_fp16" and i == 0 or route == "vector" and i == 1
        ids = [] if empty else [f"b{i}", f"a{i}"] if route == "vector" else [f"a{i}", f"b{i}"]
        scores = [] if empty else [.9, .8] if route == "vector" else [2., 1.]
        wall = 999. if cell["phase"] == "warmup" else {"bm25": .01, "vector": .02, "located_fp16": .03}[route] + (cell["repetition"] - 1) * .001
        row = {k: v for k, v in cell.items() if k != "state"}
        row.update({"query": queries[i]["query"], "scope_document_ids": [f"d{i}"], "status": "empty" if empty else "executed",
                    "error": None, "candidate_ids": ids, "ranked_ids": ids, "scores": scores,
                    "hits": [{**lookup[cid], "rank": rank, "candidate_rank": rank, "score": scores[rank - 1]}
                             for rank, cid in enumerate(ids, 1)],
                    "timing": {name: wall if name == "wall_seconds" else 0. for name in target.TIMINGS},
                    "selection_trace": {}, "vector_proof": vector_snapshot() if route == "vector" else None})
        if route == "vector":
            row["selection_trace"] = {"trace": {"query_encode_calls": 1, "query_encode_successes": 1,
                "query_vector_cache": False, "query_result_cache": False, "embedding_seconds": 0.,
                "vector_search_seconds": 0., "encoder_session": {"providers": ["CPUExecutionProvider"],
                    "intra_op_num_threads": 2, "inter_op_num_threads": 1},
                "matrix_reference": {"dtype": "float32", "byte_order": "little", "shape": [len(corpus), 384],
                    "bytes": len(corpus) * 384 * 4, "sha256": "d" * 64},
                "query_vector_reference": {"dtype": "float32", "byte_order": "little", "shape": [384],
                    "bytes": 384 * 4, "sha256": "e" * 64}}}
        if route == "located_fp16" and ids:
            row["feed_verification"] = mps_snapshot()
        path = execution / f"cell-{index:04d}-{cell['phase']}-{route}.json"
        dump(path, row)
        rows.append((path, row))
    if historical:
        history = []
        for i, query in enumerate(queries):
            path = root / f"historical-{i:03d}.json"
            dump(path, {"phase": "hot", "repetition": 0, "variant": "mps_fp16", "status": "executed", **query,
                        "candidate_ids": [f"a{i}", f"b{i}"], "ranked_ids": [f"a{i}", f"b{i}"], "scores": [2., 1.],
                        "timing": {"reranker_wall_seconds": 999999.}})
            history.append(target.file_reference(path))
        frozen["historical_reference_files"] = history
    freeze = root / "freeze.json"
    dump(freeze, frozen)
    freeze_sha = target.sha(freeze)
    dump(execution / "execution-plan.json", {"freeze_sha256": freeze_sha, "plan": plan,
                                             "measurement": "same_run_full_retrieval_wall"})
    dump(execution / "terminal.json", {"status": "complete", "prepare_only": False, "labels_parsed": False,
        "freeze_sha256": freeze_sha, "counters": {"hot_rows": 1026, "warmup_rows": 3}, "unexecuted": []})
    output = root / "analysis"
    output.mkdir()
    return freeze, freeze_sha, execution, output, plan, rows


class FormulaTests(unittest.TestCase):
    def test_success_empty_is_zero_but_gap_is_not_a_positive_zero(self):
        values = target.quality([], [], {"gold"})
        self.assertTrue(all(values[m] == 0 for m in target.METRICS))
        self.assertTrue(all(target.quality([], [], set())[m] is None for m in target.METRICS))

    def test_recall_is_fraction_of_all_marked_positives(self):
        values = target.quality(["a", "x", "b"], ["a", "x", "b"], {"a", "b", "c", "d"})
        self.assertEqual(values["recall5"], .5)
        self.assertEqual(values["mrr24"], 1.)

    def test_missing_positive_keeps_formal_mean_unknown(self):
        good = {"first_hot_status": "empty", **target.quality([], [], {"a"})}
        missing = {"first_hot_status": "unknown", **dict.fromkeys(target.METRICS)}
        result = target.aggregate([good, missing])
        self.assertEqual((result["planned_n"], result["known_n"], result["unknown_n"]), (2, 1, 1))
        self.assertIsNone(result["mrr24"])
        self.assertEqual(result["observed_mean"]["mrr24"], 0.)

    def test_partial_timing_has_observed_distribution_without_primary_claim(self):
        result = target.timing_distribution([{"all_hot_successful": True, "median_wall_seconds": 2.},
                                             {"all_hot_successful": False, "median_wall_seconds": None}], planned_n=2)
        self.assertIsNone(result["primary"])
        self.assertEqual(result["observed"]["n"], 1)
        self.assertEqual(result["known_n"], 1)

    def test_report_module_imports_only_standard_library(self):
        tree = ast.parse(Path(target.__file__).read_text())
        imports = [n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertTrue(set(imports) <= sys.stdlib_module_names)


class FullReportTests(unittest.TestCase):
    def test_complete_all_routes_and_tradeoffs_without_zero_regression_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            freeze, sha, execution, output, plan, _ = fixture(Path(tmp))
            result = target.report(freeze, sha, execution, output)
            self.assertEqual(len(plan["cells"]), 1029)
            self.assertTrue(result["all_planned_cells_successful"])
            self.assertEqual((result["answerable_cases"], result["gap_cases"]), (112, 2))
            located = next(r for r in result["quality_aggregates"] if r["route"] == "located_fp16" and r["split"] == "all")
            self.assertEqual(located["planned_n"], 112)
            self.assertAlmostEqual(located["mrr24"], 111 / 112)
            self.assertIsNone(result["quality_regression_gate"])
            self.assertFalse(result["automatic_adoption"])
            for item in result["comparisons_vs_same_run_baselines"]:
                self.assertEqual(item["known_n"], 112)
                self.assertLess(item["distribution_changes"]["p50"]["relative_reduction_percent"], 0)
            self.assertEqual(result["comparisons_vs_same_run_baselines"][0]["quality_changes"]["mrr24"]["worse_cases"], 1)
            for item in result["timing_distributions"]:
                self.assertLess(item["primary"]["p95"], .1)
            self.assertEqual(len(result["warmups_excluded_from_hot_metrics"]), 3)
            self.assertTrue(all(r["timing"]["wall_seconds"] == 999 for r in result["warmups_excluded_from_hot_metrics"]))
            self.assertEqual(len(json.loads((output / "planned-cells.json").read_text())), 1029)
            self.assertEqual(len(json.loads((output / "quality-changes.json").read_text())), 224)

    def test_historical_timing_does_not_enter_primary_and_real_empty_counts_as_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            freeze, sha, execution, output, _, _ = fixture(Path(tmp), historical=True)
            result = target.report(freeze, sha, execution, output)
            historical = result["historical_drift"]
            self.assertTrue(historical["available"])
            self.assertEqual(historical["candidate_changed_cases"], 1)
            self.assertEqual(historical["ranking_changed_cases"], 1)
            self.assertAlmostEqual(historical["quality_deltas"]["mrr24"], -1 / 112)
            self.assertTrue(all(r["observed"]["maximum"] < 1 for r in result["timing_distributions"]))

    def test_stop_retains_planned_unknown_and_successful_empty_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            freeze, sha, execution, output, plan, rows = fixture(Path(tmp))
            cutoff = 10
            for path, _ in rows[cutoff:]:
                path.unlink()
            path, row = rows[cutoff - 1]
            row.update(status="stopped", error={"type": "SyntheticFailure", "reason": "planned failure"})
            dump(path, row)
            terminal = json.loads((execution / "terminal.json").read_text())
            terminal.update(status="stopped", counters={"hot_rows": cutoff - 3, "warmup_rows": 3},
                unexecuted=[{**c, "index": i, "reason": "execution_stopped"} for i, c in enumerate(plan["cells"]) if i >= cutoff])
            dump(execution / "terminal.json", terminal)
            result = target.report(freeze, sha, execution, output)
            self.assertFalse(result["all_planned_cells_successful"])
            self.assertEqual(result["states"]["stopped"], 1)
            self.assertEqual(result["states"]["not_executed"], 1019)
            located = next(r for r in result["quality_aggregates"] if r["route"] == "located_fp16" and r["split"] == "all")
            self.assertEqual((located["planned_n"], located["known_n"]), (112, 2))
            self.assertIsNone(located["mrr24"])
            self.assertEqual(located["observed_mean"]["mrr24"], .5)
            self.assertTrue(all(r["primary"] is None for r in result["timing_distributions"]))

    def test_corrupt_frozen_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze, sha, execution, output, _, _ = fixture(root)
            (root / "chunks.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "REFERENCE_HASH_CHANGED"):
                target.report(freeze, sha, execution, output)

    def test_wrong_cell_filename_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            freeze, sha, execution, output, _, rows = fixture(Path(tmp))
            rows[0][0].rename(execution / "cell-9999-warmup-bm25.json")
            with self.assertRaisesRegex(ValueError, "UNPLANNED_RAW_FILENAME"):
                target.report(freeze, sha, execution, output)

    def test_scope_or_hit_text_mutation_is_rejected(self):
        for field in ("scope_document_ids", "text"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                freeze, sha, execution, output, _, rows = fixture(Path(tmp))
                path, row = rows[3]
                if field == "scope_document_ids":
                    row[field] = ["foreign"]
                else:
                    row["hits"][0]["text"] = "changed"
                dump(path, row)
                with self.assertRaises(ValueError):
                    target.report(freeze, sha, execution, output)

    def test_async_submission_without_completed_mps_proof_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            freeze, sha, execution, output, _, rows = fixture(Path(tmp))
            path, row = next((p, r) for p, r in rows if r["route"] == "located_fp16" and r["candidate_ids"])
            row["feed_verification"]["batches"][0]["device_proof"]["sync_after"] = False
            dump(path, row)
            with self.assertRaisesRegex(ValueError, "MPS_DEVICE_OR_SYNC_CHANGED"):
                target.report(freeze, sha, execution, output)

    def test_query_embedding_cache_or_changed_actual_feed_is_rejected(self):
        for changed in ("query_vector_cache", "input_hash"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                freeze, sha, execution, output, _, rows = fixture(Path(tmp))
                path, row = next((p, r) for p, r in rows if r["route"] == "vector")
                if changed == "query_vector_cache":
                    row["selection_trace"]["trace"][changed] = True
                else:
                    row["vector_proof"]["batches"][0]["actual_tensors"]["input_ids"]["sha256"] = "f" * 64
                dump(path, row)
                with self.assertRaises(ValueError):
                    target.report(freeze, sha, execution, output)

    def test_nonfinite_wall_and_existing_output_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            freeze, sha, execution, output, _, rows = fixture(Path(tmp))
            path, row = rows[3]
            row["timing"]["wall_seconds"] = float("nan")
            dump(path, row)
            with self.assertRaisesRegex(ValueError, "SUCCESS_TIMING_INVALID"):
                target.report(freeze, sha, execution, output)
            (output / "old.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "OUTPUT_MUST_BE_EMPTY"):
                target.report(freeze, sha, execution, output)


if __name__ == "__main__":
    unittest.main()
