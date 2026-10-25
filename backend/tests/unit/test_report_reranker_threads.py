"""用合成固定输入与排名验证线程报告，不导入或执行模型。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import report_reranker_threads as report


def save(path: Path, value) -> dict:
    """保存合成夹具并返回冻结身份。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return {"path": str(path), "sha256": report.file_sha(path)}


def fixture(root: Path) -> dict:
    """构造两道正例、一缺证题与全部二十个计划格。"""
    root = root.resolve()
    cases, feeds, metadata, positives = [], [], [], []
    for ordinal, count in enumerate((6, 2, 1)):
        cid = f"synthetic-{ordinal}"
        candidates = []
        for number in range(count):
            text = f"synthetic body {ordinal}-{number}"
            candidates.append({"chunk_id": f"chunk-{ordinal}-{number}", "document_id": "document",
                "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "evidence_id": "synthetic-evidence", "index_id": "synthetic-index",
                "source_sha256": "c" * 64, "page_number": 1, "char_start": 0, "char_end": len(text)})
        case = {"case_id": cid, "query": f"original query {ordinal}",
                "scope_document_ids": ["document"], "candidates": candidates}
        case["input_sha256"] = report.canonical_sha(case)
        cases.append(case)
        batches = []
        for start in range(0, count, 4):
            indices = list(range(start, min(start + 4, count)))
            tensors = {name: {"dtype": "int64", "byte_order": "little", "shape": [len(indices), 3],
                             "sha256": hashlib.sha256(f"{cid}:{start}:{name}".encode()).hexdigest()}
                       for name in ("input_ids", "attention_mask", "token_type_ids_if_required")}
            batches.append({"candidate_indices": indices, "tensors": tensors})
        feeds.append({"case_id": cid, "input_sha256": case["input_sha256"], "batches": batches})
        metadata.append({"case_id": cid, "query": case["query"],
                         "scope_document_ids": case["scope_document_ids"], "answerable": ordinal < 2,
                         "split": ("regression", "pending", "diagnostic")[ordinal],
                         "question_type": "synthetic", "paper_ids": ["synthetic-paper"]})
        if ordinal < 2:
            positives.append({"case_id": cid, "chunk_id": candidates[0]["chunk_id"]})
    spec = {"repo": "synthetic-model", "revision": "synthetic-revision", "weights": "model.onnx",
            "files": {"model.onnx": {"sha256": "a" * 64}, "tokenizer.json": {"sha256": "b" * 64}}}
    model = {"model_id": spec["repo"], "revision": spec["revision"],
             "weights_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
             "batch_size": 4, "sort_by_length": False, "max_pair_tokens": 512,
             "onnxruntime": "synthetic", "score_kind": "raw_relevance_logit"}
    cells = [{"phase": "warmup", "case_id": cases[0]["case_id"], "threads": t,
              "state": "not_executed"} for t in (2, 4)]
    for repetition in range(3):
        for ordinal, case in enumerate(cases):
            order = (2, 4) if (repetition + ordinal) % 2 == 0 else (4, 2)
            cells.extend({"phase": "hot", "repetition": repetition, "ordinal": ordinal,
                "case_id": case["case_id"], "threads": t, "state": "not_executed"} for t in order)
    plan = {"cells": cells, "budget": {"max_score_attempts": 20, "max_pairs": 66,
            "max_seconds": 3600, "exact_planned_score_attempts": 20, "exact_planned_pairs": 66}}
    execution = root / "execution"; execution.mkdir()
    rows = []
    for sequence, cell in enumerate(cells):
        case = next(c for c in cases if c["case_id"] == cell["case_id"])
        feed = next(f for f in feeds if f["case_id"] == cell["case_id"])
        candidate_ids = [c["chunk_id"] for c in case["candidates"]]
        scores = [float(len(candidate_ids) - i) for i in range(len(candidate_ids))]
        if cell["threads"] == 4 and case["case_id"] == "synthetic-0":
            scores.reverse()
        if cell["threads"] == 4 and case["case_id"] == "synthetic-1":
            scores[0] += .1
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        batches = len(feed["batches"])
        row = {k: v for k, v in cell.items() if k != "state"}
        row.update({"status": "executed", "error": None, "query": case["query"],
            "scope_document_ids": case["scope_document_ids"], "candidate_ids": candidate_ids,
            "input_sha256": case["input_sha256"], "feed_projection_sha256": report.canonical_sha(feed),
            "scores": scores, "ranked_ids": [candidate_ids[i] for i in order],
            "hits": [{"chunk_id": case["candidates"][i]["chunk_id"],
                      "document_id": case["candidates"][i]["document_id"],
                      "text_sha256": case["candidates"][i]["text_sha256"],
                      "rank": rank, "candidate_rank": i + 1, "reranker_score": scores[i]}
                     for rank, i in enumerate(order, 1)],
            "model_identity": {**model, "threads": cell["threads"]},
            "session_identity": {"providers": ["CPUExecutionProvider"],
                                 "input_names": ["input_ids", "attention_mask"], "provider_options": {},
                                 "intra_op_num_threads": cell["threads"], "inter_op_num_threads": 1},
            "timing": {"reranker_wall_seconds": 2.0 if cell["threads"] == 2 else 1.0,
                       "provider_inference_seconds": 1.9 if cell["threads"] == 2 else .9,
                       "feed_verification_seconds": .05, "native_session_run_seconds": 1.85 if cell["threads"] == 2 else .85},
            "feed_verification": {"expected_batches": batches, "verified_batches": batches,
                "session_run_attempts": batches, "session_run_successes": batches, "complete": True,
                "batches": [{"batch_index": i, "candidate_indices": b["candidate_indices"],
                    "actual_tensors": {name: b["tensors"][name] for name in ("input_ids", "attention_mask")},
                    "verification_passed": True, "native_attempted": True, "native_succeeded": True,
                    "error": None} for i, b in enumerate(feed["batches"])]},
            "counters_after": {"score_attempts": sequence + 1},
            "measurement": "same_run_reranker_component_with_feed_verification"})
        path = execution / f"{cell['phase']}-{sequence:03d}.json"
        save(path, row)
        rows.append((path, row))
    save(execution / "terminal.json", {"status": "complete", "counters": {"score_attempts": 20}})
    historical = root / "historical"
    historical_entries = []
    for ordinal, case in enumerate(cases):
        row = next(row for _, row in rows if row["phase"] == "hot" and row["repetition"] == 0
                   and row["threads"] == 2 and row["case_id"] == case["case_id"])
        filename = f"execution/hot-0-{ordinal:03d}-located_bm25_rerank.json"
        ref = save(historical / filename, {"phase": "hot", "repetition": 0,
            "case_id": case["case_id"], "route": "located_bm25_rerank", "result": row})
        historical_entries.append({"path": filename, "sha256": ref["sha256"]})
    labels = save(root / "labels.json", {"positive_pairs": positives, "non_exhaustive": True})
    questions = save(root / "metadata.json", metadata)
    old_freeze = save(historical / "preparation/freeze.json", {"inputs": {"labels": labels, "original_questions": questions}})
    old_manifest = save(historical / "audit/output-manifest.json", historical_entries)
    frozen = {"inputs": {"cases": save(root / "cases.json", cases),
                         "feed_manifest": save(root / "feeds.json", {"cases": feeds}),
                         "execution_plan": save(root / "plan.json", plan),
                         "model_manifest": save(root / "model-manifest.json", {"reranker": spec})},
              "report_inputs": {"labels": labels, "original_questions": questions,
                                "historical_freeze": old_freeze, "historical_manifest": old_manifest}}
    freeze = root / "freeze.json"; save(freeze, frozen)
    output = root / "analysis"; output.mkdir()
    return {"freeze": freeze, "execution": execution, "output": output, "rows": rows,
            "cases": cases, "feeds": feeds, "frozen": frozen}


class RerankerThreadReportTests(unittest.TestCase):
    """验证完整分母、失败缺格、输入身份和时间边界。"""

    def test_complete_metrics_drift_and_paired_time(self):
        """首热质量与三轮配对分别计算，缺证不进入质量分母。"""
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            result = report.build_report(f["freeze"], f["execution"], f["output"])
            self.assertEqual(result["planned_cells"], 20)
            self.assertEqual(result["observed_cells"], 20)
            self.assertEqual(result["answerable_cases"], 2)
            self.assertEqual(result["gap_cases"], 1)
            self.assertTrue(result["all_cells_executed"])
            a, b = [next(r for r in result["aggregates"] if r["threads"] == t and r["split"] == "all") for t in (2, 4)]
            self.assertEqual(a["recall5"], 1)
            self.assertEqual(b["recall5"], .5)
            self.assertAlmostEqual(b["mrr24"], 7 / 12)
            self.assertEqual(a["candidate_recall24"], b["candidate_recall24"])
            paired = result["paired_timing"]
            self.assertEqual((paired["n"], paired["planned_n"]), (2, 2))
            self.assertEqual(paired["reranker_wall_seconds"]["threads_2"]["p50"], 2)
            self.assertEqual(paired["reranker_wall_seconds"]["threads_4"]["p50"], 1)
            self.assertEqual(paired["reranker_wall_seconds"]["paired_ratio_4_over_2"]["p50"], .5)
            self.assertEqual(paired["feed_verification_seconds"]["threads_4"]["p50"], .05)
            self.assertFalse(result["quality_gate"]["all_macro_metrics_non_decreasing"])
            self.assertEqual(result["historical_drift"]["ranking_changed_cases"], 0)
            self.assertEqual(result["cross_thread_drift"]["ranking_changed_cases"], 1)
            self.assertEqual(result["cross_thread_drift"]["scores_changed_cases"], 2)
            self.assertEqual(result["planned_session_run_calls"], 28)
            self.assertEqual(result["observed_session_run_attempts"], 28)
            self.assertIn("完整检索时间", (f["output"] / "report.md").read_text())

    def test_failed_first_hot_retains_zero_in_original_denominator(self):
        """失败不是删题；公平时间排除任一配置失败的题。"""
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path, row = next((p, r) for p, r in f["rows"] if r["phase"] == "hot" and r["repetition"] == 0
                             and r["threads"] == 4 and r["case_id"] == "synthetic-1")
            row.update({"status": "failed", "scores": None, "ranked_ids": [], "hits": [],
                        "error": {"type": "SyntheticError", "reason": "synthetic"}})
            save(path, row)
            result = report.build_report(f["freeze"], f["execution"], f["output"])
            b = next(r for r in result["aggregates"] if r["threads"] == 4 and r["split"] == "all")
            self.assertEqual((b["known_n"], b["planned_n"]), (2, 2))
            self.assertTrue(b["complete"])
            self.assertAlmostEqual(b["mrr24"], 1 / 12)
            self.assertEqual(result["states"]["failed"], 1)
            self.assertEqual(result["paired_timing"]["n"], 1)

    def test_missing_first_hot_is_na_and_not_a_zero_or_changed_denominator(self):
        """未执行的主指标保持N/A，已知均值明确单列。"""
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path, _ = next((p, r) for p, r in f["rows"] if r["phase"] == "hot" and r["repetition"] == 0
                           and r["threads"] == 4 and r["case_id"] == "synthetic-1")
            path.unlink()
            result = report.build_report(f["freeze"], f["execution"], f["output"])
            b = next(r for r in result["aggregates"] if r["threads"] == 4 and r["split"] == "all")
            self.assertEqual((b["known_n"], b["planned_n"]), (1, 2))
            self.assertIsNone(b["mrr24"])
            self.assertAlmostEqual(b["observed_mean"]["mrr24"], 1 / 6)
            self.assertEqual(result["unexecuted_cells"], 1)
            self.assertFalse(result["quality_gate"]["available"])
            cells = json.loads((f["output"] / "planned-cells.json").read_text())
            self.assertEqual(sum(c["state"] == "unexecuted" for c in cells), 1)

    def test_interrupted_cell_remains_unknown_quality(self):
        """stopped原件保留，不猜测未交付分数或填零。"""
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path, row = next((p, r) for p, r in f["rows"] if r["phase"] == "hot" and r["repetition"] == 0
                             and r["threads"] == 4 and r["case_id"] == "synthetic-1")
            row.update({"status": "stopped", "scores": None, "ranked_ids": [], "hits": []})
            save(path, row)
            result = report.build_report(f["freeze"], f["execution"], f["output"])
            self.assertEqual(result["unknown_or_stopped_cells"], 1)
            self.assertIsNone(next(r for r in result["aggregates"] if r["threads"] == 4 and r["split"] == "all")["mrr24"])

    def test_three_repeat_median_includes_verification_without_subtraction(self):
        """使用三轮中位数，不能挑最快轮或扣掉feed核对。"""
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            for path, row in f["rows"]:
                if row["phase"] == "hot" and row["threads"] == 4:
                    row["timing"]["reranker_wall_seconds"] = (100, 1, 3)[row["repetition"]]
                    row["timing"]["feed_verification_seconds"] = .5
                    save(path, row)
            result = report.build_report(f["freeze"], f["execution"], f["output"])
            self.assertEqual(result["paired_timing"]["reranker_wall_seconds"]["threads_4"]["p50"], 3)
            self.assertEqual(result["paired_timing"]["feed_verification_seconds"]["threads_4"]["p50"], .5)

    def test_later_better_ranking_does_not_replace_first_hot_quality(self):
        """后两轮改善只体现稳定性，不能替换预定首热质量。"""
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            for path, row in f["rows"]:
                if (row["phase"] == "hot" and row["threads"] == 4
                        and row["case_id"] == "synthetic-0" and row["repetition"] > 0):
                    by_id = {h["chunk_id"]: h for h in row["hits"]}
                    row["scores"] = list(reversed(row["scores"]))
                    row["ranked_ids"] = list(row["candidate_ids"])
                    row["hits"] = [{**by_id[cid], "rank": i + 1, "reranker_score": row["scores"][i]}
                                   for i, cid in enumerate(row["candidate_ids"])]
                    save(path, row)
            result = report.build_report(f["freeze"], f["execution"], f["output"])
            b = next(r for r in result["aggregates"] if r["threads"] == 4 and r["split"] == "all")
            self.assertAlmostEqual(b["mrr24"], 7 / 12)
            self.assertEqual(result["stability"]["4"]["ranking_unstable_cases"], 1)
            self.assertEqual(result["stability"]["4"]["scores_unstable_cases"], 1)

    def test_historical_rank_drift_is_visible_and_blocks_automatic_quality_gate(self):
        """同身份历史排名变化不能被沿用原指标掩盖。"""
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            manifest_ref = f["frozen"]["report_inputs"]["historical_manifest"]
            manifest_path = Path(manifest_ref["path"])
            manifest = json.loads(manifest_path.read_text())
            history_root = manifest_path.parent.parent
            raw_path = history_root / manifest[0]["path"]
            wrapper = json.loads(raw_path.read_text()); old = wrapper["result"]
            old["ranked_ids"].reverse(); old["hits"].reverse()
            for rank, hit in enumerate(old["hits"], 1):
                hit["rank"] = rank
                hit["reranker_score"] = float(len(old["hits"]) - rank + 1)
            save(raw_path, wrapper)
            manifest[0]["sha256"] = report.file_sha(raw_path)
            updated_ref = save(manifest_path, manifest)
            f["frozen"]["report_inputs"]["historical_manifest"] = updated_ref
            save(f["freeze"], f["frozen"])
            result = report.build_report(f["freeze"], f["execution"], f["output"])
            self.assertEqual(result["historical_drift"]["ranking_changed_cases"], 1)
            self.assertEqual(result["historical_drift"]["scores_changed_cases"], 1)
            self.assertFalse(result["quality_gate"]["available"])

    def test_auxiliary_reference_bytes_field_does_not_change_identity(self):
        """历史引用的额外文件大小字段不改变相同path/hash身份。"""
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            path = Path(f["frozen"]["report_inputs"]["historical_freeze"]["path"])
            old = json.loads(path.read_text())
            old["inputs"]["labels"]["bytes"] = Path(old["inputs"]["labels"]["path"]).stat().st_size
            f["frozen"]["report_inputs"]["historical_freeze"] = save(path, old)
            save(f["freeze"], f["frozen"])
            result = report.build_report(f["freeze"], f["execution"], f["output"])
            self.assertTrue(result["all_cells_executed"])

    def test_duplicate_or_unplanned_raw_cell_is_rejected(self):
        """不能以额外执行覆盖原格或只选择有利结果。"""
        for mode in ("duplicate", "unplanned"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                f = fixture(Path(folder)); row = deepcopy(f["rows"][0][1])
                if mode == "unplanned": row["threads"] = 3
                save(f["execution"] / "extra.json", row)
                with self.assertRaisesRegex(ValueError, "DUPLICATE_OR_UNPLANNED"):
                    report.build_report(f["freeze"], f["execution"], f["output"])

    def test_input_order_body_scope_and_feed_hash_changes_rejected(self):
        """原文/候选/input/feed任何身份变化均不能仍报告质量通过。"""
        for field in ("query", "scope_document_ids", "candidate_ids", "input_sha256", "feed_projection_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                f = fixture(Path(folder)); path, row = f["rows"][0]
                row[field] = [] if isinstance(row[field], list) else "changed"
                save(path, row)
                with self.assertRaisesRegex(ValueError, "INPUT_IDENTITY"):
                    report.build_report(f["freeze"], f["execution"], f["output"])

    def test_success_requires_actual_complete_feed_verification(self):
        """状态executed不替代真实feed核对记录。"""
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder)); path, row = f["rows"][0]
            row["feed_verification"]["verified_batches"] = 0
            save(path, row)
            with self.assertRaisesRegex(ValueError, "FEED_NOT_VERIFIED"):
                report.build_report(f["freeze"], f["execution"], f["output"])

    def test_actual_threads_and_each_tensor_are_verified(self):
        """核对实际native线程读回与每批tensor，不能只信汇总complete。"""
        for mutation in ("intra", "inter", "hash", "batch_index", "candidate_indices", "native_failed", "missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                f = fixture(Path(folder)); path, row = f["rows"][0]
                if mutation == "intra": row["session_identity"]["intra_op_num_threads"] = 1
                elif mutation == "inter": row["session_identity"]["inter_op_num_threads"] = 2
                elif mutation == "missing": row["feed_verification"]["batches"] = []
                else:
                    batch = row["feed_verification"]["batches"][0]
                    if mutation == "hash": batch["actual_tensors"]["input_ids"]["sha256"] = "changed"
                    elif mutation == "batch_index": batch["batch_index"] = 99
                    elif mutation == "candidate_indices": batch["candidate_indices"].reverse()
                    else: batch["native_succeeded"] = False
                save(path, row)
                with self.assertRaises(ValueError):
                    report.build_report(f["freeze"], f["execution"], f["output"])

    def test_wrong_score_rank_hit_or_nonfinite_values_rejected(self):
        """按分数独立重建排名和引用身份，拒绝NaN。"""
        for mutation in ("rank", "hit", "score_nan", "time_nan", "model", "provider"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                f = fixture(Path(folder)); path, row = f["rows"][0]
                if mutation == "rank": row["ranked_ids"].reverse()
                elif mutation == "hit": row["hits"][0]["text_sha256"] = "changed"
                elif mutation == "score_nan": row["scores"][0] = float("nan")
                elif mutation == "time_nan": row["timing"]["reranker_wall_seconds"] = float("nan")
                elif mutation == "model": row["model_identity"]["weights_sha256"] = "changed"
                else: row["session_identity"]["providers"] = ["CoreMLExecutionProvider"]
                save(path, row)
                with self.assertRaises(ValueError): report.build_report(f["freeze"], f["execution"], f["output"])

    def test_frozen_labels_hash_and_output_overwrite_rejected(self):
        """标签换版和覆盖旧报告均拒绝。"""
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder))
            Path(f["frozen"]["report_inputs"]["labels"]["path"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "FROZEN_INPUT_CHANGED"):
                report.build_report(f["freeze"], f["execution"], f["output"])
        with tempfile.TemporaryDirectory() as folder:
            f = fixture(Path(folder)); save(f["output"] / "old.json", {"preserve": True})
            with self.assertRaisesRegex(ValueError, "OUTPUT_MUST_BE_NEW_EMPTY"):
                report.build_report(f["freeze"], f["execution"], f["output"])

    def test_percentile_interpolates_and_unknowns_stay_none(self):
        """分位定义与历史合同一致，空集合不填零。"""
        self.assertEqual(report.percentile([1, 2, 3, 4], .5), 2.5)
        self.assertAlmostEqual(report.percentile([1, 2, 3, 4], .95), 3.85)
        self.assertIsNone(report.percentile([], .5))
        self.assertIsNone(report.quality([], [], set())["mrr24"])


if __name__ == "__main__":
    unittest.main()
