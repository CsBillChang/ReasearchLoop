"""用合成原件和编码器核对CPU线程实验准备，不构造推理Session。"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import prepare_reranker_thread_comparison as preparation
from scripts.retrieval_experiment_guard import BoundaryViolation, write_json_new


def candidate(number: int) -> dict:
    """制造完整来源字段，并附加不能流入输入包的诱导字段。"""
    text = f"synthetic passage {number}"
    return {"chunk_id": f"chunk-{number}", "document_id": "allowed-document",
            "evidence_id": f"evidence-{number}", "index_id": "synthetic-index",
            "source_sha256": "a" * 64, "page_number": 1,
            "char_start": 0, "char_end": len(text), "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "gold": True, "canonical_paper_id": "not-an-input", "score": 999}


def fixture(count=5):
    """返回query、候选与旧身份投影，保留不同顺序的旧hits。"""
    question = {"case_id": "synthetic-case", "query": "original query",
                "scope_document_ids": ["allowed-document"]}
    rows = [candidate(i) for i in range(count)]
    old = {"query": question["query"], "scope_document_ids": list(question["scope_document_ids"]),
           "candidate_ids": [r["chunk_id"] for r in rows],
           "hits": [{k: r[k] for k in ("chunk_id", "document_id", "text_sha256")}
                    for r in reversed(rows)]}
    return question, {"candidates": rows}, old


class FakeEncoder:
    """只返回约定token序列，没有模型或网络依赖。"""

    def __init__(self, lengths=(2, 4, 3, 5, 1)):
        self.lengths, self.calls = lengths, []

    def encode(self, query, text):
        self.calls.append((query, text))
        number = int(text.rsplit(" ", 1)[1])
        return SimpleNamespace(ids=list(range(10, 10 + self.lengths[number])))


class ThreadPreparationTests(unittest.TestCase):
    """验证输入白名单、历史身份、张量投影与未执行预算。"""

    def test_projection_preserves_original_values_without_mutation_or_labels(self):
        """旧score和gold字段不参与准备，原始对象也不改写。"""
        question, selected, old = fixture()
        before = deepcopy((question, selected, old))
        case = preparation.project_case(question, selected, old)
        self.assertEqual(set(case), {"case_id", "query", "scope_document_ids", "candidates", "input_sha256"})
        self.assertTrue(all(set(row) == set(preparation.SOURCE_FIELDS) for row in case["candidates"]))
        self.assertEqual([r["chunk_id"] for r in case["candidates"]], old["candidate_ids"])
        self.assertEqual((question, selected, old), before)
        body = {k: v for k, v in case.items() if k != "input_sha256"}
        self.assertEqual(case["input_sha256"], preparation.canonical_digest(body))

    def test_changed_query_scope_order_duplicate_and_text_are_rejected(self):
        """输入有任何可影响排序的变动时，准备不产生通过资格。"""
        for change in ("query", "scope", "order", "duplicate", "text", "extra_query", "outside", "old_hits"):
            with self.subTest(change=change):
                q, selected, old = fixture()
                if change == "query": q["query"] = "changed"
                elif change == "scope": q["scope_document_ids"].append("another")
                elif change == "order": selected["candidates"].reverse()
                elif change == "duplicate": selected["candidates"][1] = selected["candidates"][0]
                elif change == "text": selected["candidates"][0]["text"] = "tampered"
                elif change == "extra_query": q["gold"] = "not allowed"
                elif change == "outside": selected["candidates"][0]["document_id"] = "outside"
                elif change == "old_hits": old["hits"].append(old["hits"][0])
                with self.assertRaises(BoundaryViolation):
                    preparation.project_case(q, selected, old)

    def test_feed_projection_has_fixed_companions_padding_and_int64_hashes(self):
        """按原序4+1组批，ids补1、mask补0、类型全0，哈希对应真实字节。"""
        q, selected, old = fixture()
        case = preparation.project_case(q, selected, old)
        encoder = FakeEncoder()
        result = preparation.build_feed_projection(case, encoder)
        self.assertEqual(result["pairs"], 5)
        self.assertEqual(result["input_tokens"], 15)
        self.assertEqual(result["padded_tokens"], 21)
        self.assertEqual([b["candidate_indices"] for b in result["batches"]], [[0, 1, 2, 3], [4]])
        self.assertEqual(result["batches"][0]["tensors"]["input_ids"]["shape"], [4, 5])
        expected = [[10, 11, 1, 1, 1], [10, 11, 12, 13, 1], [10, 11, 12, 1, 1], [10, 11, 12, 13, 14]]
        data = struct.pack("<" + "q" * 20, *(v for row in expected for v in row))
        self.assertEqual(result["batches"][0]["tensors"]["input_ids"]["sha256"], hashlib.sha256(data).hexdigest())
        masks = [[1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        self.assertEqual(result["batches"][0]["tensors"]["attention_mask"], preparation.tensor_reference(masks))
        self.assertEqual(encoder.calls, [(q["query"], r["text"]) for r in case["candidates"]])

    def test_equal_inputs_have_same_feed_identity_for_both_thread_configs(self):
        """线程参数不进入输入张量生成，生成结果完全相同。"""
        q, selected, old = fixture()
        case = preparation.project_case(q, selected, old)
        self.assertEqual(preparation.build_feed_projection(case, FakeEncoder()),
                         preparation.build_feed_projection(deepcopy(case), FakeEncoder()))

    def test_capacity_boundary_rejects_overlong_and_empty_without_truncation(self):
        """512可准备，513与空编码拒绝，不通过裁剪换取资格。"""
        q, selected, old = fixture(1)
        case = preparation.project_case(q, selected, old)
        self.assertEqual(preparation.build_feed_projection(case, FakeEncoder((512,)))["input_tokens"], 512)
        for size in (0, 513):
            with self.assertRaisesRegex(BoundaryViolation, "TOO_LONG_OR_EMPTY"):
                preparation.build_feed_projection(case, FakeEncoder((size,)))

    def test_invalid_token_values_are_rejected(self):
        """无效token不能因Python整数兼容而通过。"""
        q, selected, old = fixture(1)
        case = preparation.project_case(q, selected, old)
        for value in (-1, True, 1.5):
            class InvalidEncoder:
                def encode(self, query, text): return SimpleNamespace(ids=[value])
            with self.assertRaisesRegex(BoundaryViolation, "TOKEN_IDS_INVALID"):
                preparation.build_feed_projection(case, InvalidEncoder())

    def test_plan_has_exact_budget_alternating_order_and_no_execution(self):
        """两个初始化与三热轮都仅计划，线程轮换且候选相同。"""
        q, selected, old = fixture(3)
        first = preparation.project_case(q, selected, old)
        second = {**deepcopy(first), "case_id": "second-case", "candidates": first["candidates"][:1]}
        result = preparation.build_execution_plan([first, second])
        self.assertEqual(result["budget"]["exact_planned_score_attempts"], 14)
        self.assertEqual(result["budget"]["exact_planned_pairs"], 30)
        self.assertEqual(result["budget"]["max_pairs"], 16134)
        self.assertEqual([c["threads"] for c in result["cells"][:6]], [2, 4, 2, 4, 4, 2])
        self.assertTrue(all(c["state"] == "not_executed" for c in result["cells"]))
        self.assertFalse(result["execution_authorized_by_preparation"])
        self.assertEqual(result["score_attempts"], 0)
        self.assertEqual(result["timing"]["primary"], "reranker_wall_seconds")
        self.assertEqual(result["settings"]["batch_size"], 4)

    def test_plan_rejects_duplicate_queries_and_excess_budget(self):
        """不靠截掉末尾计划格满足预算。"""
        q, selected, old = fixture()
        case = preparation.project_case(q, selected, old)
        for cases in ([], [case, case], [{**case, "case_id": str(i), "candidates": [candidate(j) for j in range(24)]} for i in range(114)]):
            with self.assertRaises(BoundaryViolation):
                preparation.build_execution_plan(cases)

    def test_labels_are_hashed_without_json_parse(self):
        """标签即使不是合法JSON也可核字节，证明准备不解析其内容。"""
        with tempfile.TemporaryDirectory() as folder:
            label = Path(folder) / "labels.json"
            label.write_text("not JSON; labels must never be parsed")
            frozen = {"inputs": {"labels": {"path": str(label), "sha256": preparation.digest(label)}}}
            with patch.object(preparation.json, "loads", side_effect=AssertionError("labels parsed")):
                preparation.verify_input_hashes(frozen)
            label.write_text("changed")
            with self.assertRaisesRegex(BoundaryViolation, "FROZEN_INPUT_CHANGED"):
                preparation.verify_input_hashes(frozen)

    def test_source_anchor_rejects_self_consistent_replacement_freeze_before_parsing(self):
        """替换后的内层hash即使自洽，也必须被事前外部锚点拒绝。"""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve(); output = base / "output"; output.mkdir()
            source = base / "source.txt"; source.write_text("original input")
            frozen = base / "freeze.json"; manifest = base / "manifest.json"
            frozen.write_text(json.dumps({"inputs": {"query": {"path": str(source),
                                                                  "sha256": preparation.digest(source)}}}))
            manifest.write_text("[]")
            anchor = preparation.digest(frozen); manifest_anchor = preparation.digest(manifest)
            source.write_text("replaced but internally consistent")
            frozen.write_text(json.dumps({"inputs": {"query": {"path": str(source),
                                                                  "sha256": preparation.digest(source)}}}))
            with patch.object(preparation.json, "loads", side_effect=AssertionError("freeze parsed too soon")), \
                    patch.object(preparation, "install_guards", side_effect=AssertionError("roots expanded too soon")):
                with self.assertRaisesRegex(BoundaryViolation, "SOURCE_ANCHOR_HASH_CHANGED"):
                    preparation.prepare(frozen, base, manifest, output,
                                        expected_source_freeze_sha256=anchor,
                                        expected_source_manifest_sha256=manifest_anchor)
            self.assertEqual(list(output.iterdir()), [])

    def test_source_anchor_rejects_replaced_manifest_and_malformed_hashes(self):
        """历史清单也绑定事前锚点，不采信与新原件一同替换的hash。"""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); frozen = base / "freeze.json"; manifest = base / "manifest.json"
            frozen.write_text("{}"); manifest.write_text("[]")
            frozen_anchor, manifest_anchor = preparation.digest(frozen), preparation.digest(manifest)
            preparation.verify_source_anchors(frozen, manifest, frozen_anchor, manifest_anchor)
            manifest.write_text(json.dumps([{"path": "execution/replacement.json", "sha256": "f" * 64}]))
            with self.assertRaisesRegex(BoundaryViolation, "SOURCE_ANCHOR_HASH_CHANGED"):
                preparation.verify_source_anchors(frozen, manifest, frozen_anchor, manifest_anchor)
            for invalid in (None, "bad", "g" * 64, "f" * 63):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(BoundaryViolation, "SOURCE_ANCHOR_HASH_INVALID"):
                        preparation.verify_source_anchors(frozen, manifest, invalid, manifest_anchor)

    def test_historical_raw_hash_and_projection(self):
        """只接受清单内完整原件，历史分数不进入返回值。"""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); results = base / "execution"; results.mkdir()
            q, selected, old = fixture(1)
            raw = results / "hot-0-000-located_bm25_rerank.json"
            result = {**old, "status": "executed", "effective_path": preparation.ROUTE,
                      "model_identity": {"synthetic": True}, "ranked_ids": ["do-not-copy"], "score": 999}
            raw.write_text(json.dumps({"phase": "hot", "repetition": 0, "case_id": q["case_id"],
                                       "route": preparation.ROUTE, "result": result}))
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps([{"path": "execution/" + raw.name, "sha256": preparation.digest(raw)}]))
            rows = preparation.load_source_rows(results, manifest, [q])
            self.assertNotIn("score", rows[q["case_id"]])
            self.assertNotIn("ranked_ids", rows[q["case_id"]])
            raw.write_text(raw.read_text() + " ")
            with self.assertRaisesRegex(BoundaryViolation, "HISTORICAL_RAW_HASH_CHANGED"):
                preparation.load_source_rows(results, manifest, [q])

    def test_manifest_path_escape_is_rejected(self):
        """清单路径不可跨出历史结果目录。"""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); results = base / "execution"; results.mkdir()
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps([{"path": "../hot-0-000-located_bm25_rerank.json", "sha256": "a" * 64}]))
            with self.assertRaisesRegex(BoundaryViolation, "PATH_INVALID"):
                preparation.load_source_rows(results, manifest, [fixture()[0]])

    def test_model_identity_allows_no_change_except_future_threads(self):
        """旧基线必须是同权重、同分词器及同组批设置。"""
        spec = {"repo": "synthetic", "revision": "revision", "weights": "model.onnx",
                "files": {"model.onnx": {"sha256": "weights"}}}
        identity = {"model_id": "synthetic", "revision": "revision", "weights_sha256": "weights",
                    "tokenizer_sha256": "tokenizer", "max_pair_tokens": 512,
                    "batch_size": 4, "threads": 2, "sort_by_length": False,
                    "onnxruntime": "synthetic-version", "score_kind": "raw_relevance_logit"}
        preparation.check_model_identity({"model_identity": identity}, spec, "tokenizer", "synthetic-version")
        for field in identity:
            changed = {**identity, field: "changed"}
            with self.assertRaisesRegex(BoundaryViolation, "MODEL_IDENTITY_CHANGED"):
                preparation.check_model_identity({"model_identity": changed}, spec, "tokenizer", "synthetic-version")

    def test_module_contains_no_model_execution_or_import(self):
        """准备模块自身不导入ORT或提供任何score/Session调用。"""
        tree = ast.parse(Path(preparation.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(name.name.startswith("onnxruntime") for name in node.names))
            if isinstance(node, ast.ImportFrom): self.assertNotEqual(node.module, "onnxruntime")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"score", "InferenceSession", "_load"})

    def test_output_writer_refuses_existing_file(self):
        """准备工件只能独占新增，不能覆盖旧失败记录。"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            write_json_new(path, {"first": True})
            with self.assertRaises(FileExistsError): write_json_new(path, {"overwritten": True})
            self.assertEqual(json.loads(path.read_text()), {"first": True})


if __name__ == "__main__":
    unittest.main()
