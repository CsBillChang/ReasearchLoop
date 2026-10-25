"""使用合成张量和替身Session核对线程运行器，绝不调用真实模型。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import compare_reranker_threads as comparison
from scripts.prepare_reranker_thread_comparison import build_feed_projection, canonical_digest
from scripts.retrieval_experiment_guard import BoundaryViolation, write_json_new


IDENTITY = {"model_id": "synthetic", "revision": "synthetic-revision",
            "weights_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
            "onnxruntime": "synthetic-version", "max_pair_tokens": 512,
            "score_kind": "raw_relevance_logit", "batch_size": 4, "sort_by_length": False}


def make_case(count=5, case_id="synthetic-case"):
    """构造原文与来源身份齐全的无标签输入。"""
    rows = []
    for number in range(count):
        text = f"synthetic passage {number}"
        rows.append({"chunk_id": f"chunk-{number}", "document_id": "authorized-document",
                     "evidence_id": "synthetic-evidence", "index_id": "synthetic-index",
                     "source_sha256": "a" * 64, "page_number": 1,
                     "char_start": 0, "char_end": len(text), "text": text,
                     "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    case = {"case_id": case_id, "query": "original query", "scope_document_ids": ["authorized-document"],
            "candidates": rows}
    return {**case, "input_sha256": canonical_digest(case)}


class FakeEncoder:
    """只返回长度可控的合成整数编码。"""

    def encode(self, query, text):
        count = int(text.rsplit(" ", 1)[1]) % 4 + 2
        return SimpleNamespace(ids=list(range(10, 10 + count)))


def actual_feeds(case, *, include_types=True):
    """按原provider配方制作实际NumPy数组，用于代理校验。"""
    rows = [FakeEncoder().encode(case["query"], row["text"]).ids for row in case["candidates"]]
    result = []
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]; size = max(map(len, batch))
        ids = np.ones((len(batch), size), dtype=np.int64); mask = np.zeros_like(ids)
        for ordinal, row in enumerate(batch):
            ids[ordinal, :len(row)] = row; mask[ordinal, :len(row)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if include_types: feed["token_type_ids"] = np.zeros_like(ids)
        result.append(feed)
    return result


class FakeNativeSession:
    """观察代理是否真正调用底层，不创建ONNX对象。"""

    def __init__(self, include_types=True, threads=2):
        self.names = ["input_ids", "attention_mask"] + (["token_type_ids"] if include_types else [])
        self.providers, self.calls, self.error = ["CPUExecutionProvider"], [], None
        self.options = SimpleNamespace(intra_op_num_threads=threads, inter_op_num_threads=1)

    def get_inputs(self): return [SimpleNamespace(name=name, type="tensor(int64)") for name in self.names]
    def get_providers(self): return self.providers
    def get_provider_options(self): return {"CPUExecutionProvider": {}}
    def get_session_options(self): return self.options

    def run(self, names, feed):
        self.calls.append((names, {k: v.copy() for k, v in feed.items()}))
        if self.error is not None: raise self.error
        return [np.zeros((len(feed["input_ids"]), 1), dtype=np.float32)]


class FakeReranker:
    """模拟provider组批；输出仍须经过真实feed代理验证。"""

    def __init__(self, case, values=None, *, threads=2, error=None, skip_last=False):
        self.case = case
        self.identity = {**IDENTITY, "threads": threads}
        self.session = comparison.VerifiedSession(FakeNativeSession(threads=threads), expected_threads=threads)
        self.last_stats = {"stale": "must be cleared"}
        self.values = list(range(len(case["candidates"]))) if values is None else values
        self.calls, self.error, self.skip_last = [], error, skip_last

    def score(self, query, texts):
        self.calls.append((query, texts))
        if self.error is not None: raise self.error
        feeds = actual_feeds(self.case)
        for feed in feeds[:-1] if self.skip_last else feeds:
            self.session.run(None, feed)
        self.last_stats = {"inference_seconds": .1, "pairs": len(texts)}
        return self.values


def full_synthetic_freeze(base):
    """产生完整114题合成冻结，使独立子进程可验证prepare-only没有推理。"""
    base = Path(base).resolve()
    counts = [24] * 107 + [1, 20, 18, 18, 18, 16, 22]
    cases = [make_case(count, f"synthetic-{index}") for index, count in enumerate(counts)]
    feeds = {"warning": "synthetic only", "cases": [build_feed_projection(c, FakeEncoder()) for c in cases]}
    plan = comparison.build_execution_plan(cases)
    model_dir = base / "model"; model_dir.mkdir()
    files = {}
    for name in ("model.onnx", "tokenizer.json", "config.json"):
        path = model_dir / name; path.write_text("synthetic file; never execute")
        files[name] = {"sha256": comparison.digest(path)}
    packages = {name: importlib.metadata.version(name) for name in ("onnxruntime", "tokenizers", "numpy")}
    spec = {"repo": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "revision": "synthetic-revision",
            "weights": "model.onnx", "directory": str(model_dir), "files": files}
    payloads = {"cases": cases, "feed_manifest": feeds, "execution_plan": plan,
                "model_manifest": {"reranker": spec}}
    inputs = {}
    for name, value in payloads.items():
        path = base / (name + ".json"); path.write_text(json.dumps(value))
        inputs[name] = {"path": str(path), "sha256": comparison.digest(path)}
    label = base / "labels.json"; label.write_text("intentionally invalid JSON; hash only")
    expected = {**IDENTITY, "model_id": spec["repo"], "weights_sha256": files["model.onnx"]["sha256"],
                "tokenizer_sha256": files["tokenizer.json"]["sha256"], "onnxruntime": packages["onnxruntime"]}
    frozen = {"schema": "reranker-thread-execution-v1", "settings": comparison.SETTINGS,
              "authorization": comparison.AUTHORIZATION, "expected_counts": comparison.COUNTS,
              "inputs": inputs, "source_files": [], "protected_files": [],
              "model_files": [{"path": str(model_dir / name), "sha256": ref["sha256"]} for name, ref in files.items()],
              "report_inputs": {"labels": {"path": str(label), "sha256": comparison.digest(label)}},
              "packages": packages, "expected_model_identity": expected}
    path = base / "freeze.json"; path.write_text(json.dumps(frozen))
    return path, comparison.digest(path)


class CompareRerankerThreadsTests(unittest.TestCase):
    """测试输入身份、有限输出、计量与失败留证，不测试模型效果。"""

    def setUp(self):
        self.case = make_case()
        self.feed = build_feed_projection(self.case, FakeEncoder())
        self.cell = {"phase": "hot", "repetition": 0, "ordinal": 0,
                     "case_id": self.case["case_id"], "threads": 2, "state": "not_executed"}

    def run_cell(self, engine=None, counters=None, *, started_at=None):
        if engine is None: engine = FakeReranker(self.case)
        return comparison.run_one(self.case, self.feed, self.cell, engine,
                                  counters if counters is not None else comparison.new_counters(),
                                  expected_identity=IDENTITY,
                                  started_at=comparison.time.perf_counter() if started_at is None else started_at)

    def test_success_maps_scores_full_ranks_and_original_input(self):
        """实际调用使用原query原文，分数、来源与完整名次按身份对应。"""
        engine = FakeReranker(self.case, [2., 2., -1., 0., 5.]); counts = comparison.new_counters()
        result = self.run_cell(engine, counts)
        self.assertEqual(engine.calls, [(self.case["query"], [r["text"] for r in self.case["candidates"]])])
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["ranked_ids"], ["chunk-4", "chunk-0", "chunk-1", "chunk-3", "chunk-2"])
        self.assertEqual([h["candidate_rank"] for h in result["hits"]], [5, 1, 2, 4, 3])
        self.assertEqual(result["feed_projection_sha256"], canonical_digest(self.feed))
        self.assertEqual(result["input_sha256"], self.case["input_sha256"])
        self.assertEqual(counts["score_attempts"], 1); self.assertEqual(counts["pair_attempts"], 5)
        self.assertEqual(counts["session_run_successes"], 2)
        self.assertTrue(result["feed_verification"]["complete"])
        self.assertEqual(result["session_identity"]["intra_op_num_threads"], 2)
        self.assertEqual(result["session_identity"]["inter_op_num_threads"], 1)
        observed_batches = result["feed_verification"]["batches"]
        self.assertEqual(len(observed_batches), 2)
        for index, observed in enumerate(observed_batches):
            expected = self.feed["batches"][index]
            self.assertEqual(observed["batch_index"], index)
            self.assertEqual(observed["candidate_indices"], expected["candidate_indices"])
            self.assertTrue(observed["verification_passed"])
            self.assertTrue(observed["native_attempted"])
            self.assertTrue(observed["native_succeeded"])
            self.assertEqual(observed["actual_tensors"]["input_ids"], expected["tensors"]["input_ids"])
            self.assertEqual(observed["actual_tensors"]["token_type_ids"],
                             expected["tensors"]["token_type_ids_if_required"])
        self.assertEqual(len(result["host_load_average"]), 3)
        self.assertGreaterEqual(result["timing"]["reranker_wall_seconds"], 0)
        self.assertEqual(result["state"], "executed")

    def test_optional_token_types_and_input_delegation(self):
        """允许模型只声明两项输入，get_inputs保持原Session委托行为。"""
        native = FakeNativeSession(include_types=False); proxy = comparison.VerifiedSession(native, expected_threads=2)
        proxy.reset(self.feed)
        for feed in actual_feeds(self.case, include_types=False): proxy.run(None, feed)
        self.assertEqual(proxy.get_inputs()[0].name, native.get_inputs()[0].name)
        self.assertEqual(proxy.snapshot()["session_run_successes"], 2)

    def test_hash_shape_dtype_mask_and_type_tampering_stop_before_native(self):
        """实际feed任一内容不符时，不进行底层推理。"""
        for change in ("ids", "mask", "types", "shape", "dtype", "extra", "missing"):
            with self.subTest(change=change):
                native = FakeNativeSession(); proxy = comparison.VerifiedSession(native, expected_threads=2); proxy.reset(self.feed)
                feed = actual_feeds(self.case)[0]
                if change == "ids": feed["input_ids"][0, 0] += 1
                elif change == "mask": feed["attention_mask"][0, 0] = 0
                elif change == "types": feed["token_type_ids"][0, 0] = 1
                elif change == "shape": feed["input_ids"] = feed["input_ids"][:, :-1].copy()
                elif change == "dtype": feed["input_ids"] = feed["input_ids"].astype(np.int32)
                elif change == "extra": feed["unexpected"] = feed["input_ids"]
                elif change == "missing": del feed["attention_mask"]
                with self.assertRaises(BoundaryViolation): proxy.run(None, feed)
                self.assertEqual(native.calls, [])
                observed = proxy.snapshot()["batches"][0]
                self.assertFalse(observed["verification_passed"])
                self.assertFalse(observed["native_attempted"])
                self.assertIsNotNone(observed["error"])

    def test_extra_batch_and_unexpected_session_contract_are_rejected(self):
        """额外批、意外输入或非CPU后端均不能悄悄执行。"""
        proxy = comparison.VerifiedSession(FakeNativeSession(), expected_threads=2); proxy.reset(self.feed)
        for feed in actual_feeds(self.case): proxy.run(None, feed)
        with self.assertRaisesRegex(BoundaryViolation, "BATCH_COUNT_CHANGED"):
            proxy.run(None, actual_feeds(self.case)[0])
        for change in ("provider", "names", "duplicate"):
            native = FakeNativeSession()
            if change == "provider": native.providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            elif change == "names": native.names.append("unexpected")
            elif change == "duplicate": native.names.append("input_ids")
            with self.assertRaises(BoundaryViolation): comparison.VerifiedSession(native, expected_threads=2)

    def test_actual_thread_options_are_read_back_and_enforced(self):
        """不只采信外层identity，必须读回原Session的真实线程参数。"""
        native = FakeNativeSession(threads=4)
        proxy = comparison.VerifiedSession(native, expected_threads=4)
        self.assertEqual(proxy.identity["intra_op_num_threads"], 4)
        for intra, inter in ((2, 1), (4, 2), (0, 1)):
            native = FakeNativeSession(threads=intra); native.options.inter_op_num_threads = inter
            with self.assertRaisesRegex(BoundaryViolation, "ACTUAL_THREADS_CHANGED"):
                comparison.VerifiedSession(native, expected_threads=4)

    def test_proxy_native_failure_keeps_attempt_and_verification_counts(self):
        """底层异常保留已验证与已尝试数量，不计成功。"""
        native = FakeNativeSession(); native.error = RuntimeError("synthetic native failure")
        proxy = comparison.VerifiedSession(native, expected_threads=2); proxy.reset(self.feed)
        with self.assertRaises(RuntimeError): proxy.run(None, actual_feeds(self.case)[0])
        snapshot = proxy.snapshot()
        self.assertEqual({k: v for k, v in snapshot.items() if k != "batches"},
                         {"expected_batches": 2, "verified_batches": 1,
                          "session_run_attempts": 1, "session_run_successes": 0, "complete": False})
        self.assertTrue(snapshot["batches"][0]["verification_passed"])
        self.assertTrue(snapshot["batches"][0]["native_attempted"])
        self.assertFalse(snapshot["batches"][0]["native_succeeded"])

    def test_ordinary_failure_has_null_scores_no_ranks_and_no_stale_stats(self):
        """失败保留候选和错误，不能沿用上一题成功stats或伪造fallback。"""
        engine = FakeReranker(self.case, error=ValueError("synthetic failure")); counts = comparison.new_counters()
        result = self.run_cell(engine, counts)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["scores"]); self.assertEqual(result["ranked_ids"], [])
        self.assertEqual(result["local_reranker_stats"], {}); self.assertEqual(engine.last_stats, {})
        self.assertEqual(result["candidate_ids"], [r["chunk_id"] for r in self.case["candidates"]])
        self.assertEqual(counts["failed_score_calls"], 1)

    def test_nonfinite_boolean_wrong_count_and_nonlist_outputs_fail(self):
        """输出校验不会把NaN、无穷或错误长度写成成功排名。"""
        for values in ([0., 1.], [1.] * 4 + [float("nan")], [1.] * 4 + [float("inf")],
                       [1.] * 4 + [True], [1.] * 4 + ["bad"], "not a list"):
            with self.subTest(values=values):
                result = self.run_cell(FakeReranker(self.case, values))
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["scores"]); self.assertEqual(result["hits"], [])

    def test_budget_time_and_identity_stop_before_scoring(self):
        """超预算、超时与线程身份变化都在score前拒绝并携单格记录。"""
        for condition in ("calls", "pairs", "time", "identity", "threads"):
            with self.subTest(condition=condition):
                engine = FakeReranker(self.case); counts = comparison.new_counters()
                start = comparison.time.perf_counter()
                if condition == "calls": counts["score_attempts"] = 686
                elif condition == "pairs": counts["pair_attempts"] = 16134
                elif condition == "time": start -= 3601
                elif condition == "identity": engine.identity["revision"] = "changed"
                elif condition == "threads": engine.identity["threads"] = 4
                with self.assertRaises(comparison.CellStopped) as stopped:
                    self.run_cell(engine, counts, started_at=start)
                self.assertEqual(engine.calls, [])
                self.assertEqual(stopped.exception.record["status"], "stopped")

    def test_resource_error_stops_with_record(self):
        """内存和系统错误保持原错误原因及评分尝试，不继续余格。"""
        for error in (MemoryError("synthetic"), OSError("synthetic"), BoundaryViolation("synthetic"),
                      RuntimeError("native out of memory")):
            with self.subTest(error=type(error).__name__):
                counts = comparison.new_counters()
                with self.assertRaises(comparison.CellStopped) as stopped:
                    self.run_cell(FakeReranker(self.case, error=error), counts)
                self.assertEqual(stopped.exception.record["error"]["type"], type(error).__name__)
                self.assertEqual(counts["score_attempts"], 1)

    def test_incomplete_actual_batches_cannot_be_success(self):
        """替身即使给出正确数量分数，也必须完成实际feed的所有批。"""
        with self.assertRaises(comparison.CellStopped) as stopped:
            self.run_cell(FakeReranker(self.case, skip_last=True))
        self.assertIn("BATCHES_INCOMPLETE", stopped.exception.record["error"]["reason"])
        self.assertEqual(stopped.exception.record["ranked_ids"], [])

    def test_empty_case_does_not_score(self):
        """空候选单列empty，评分预算不增加。"""
        case = make_case(0); feed = {"case_id": case["case_id"], "input_sha256": case["input_sha256"],
            "pairs": 0, "batch_size": 4, "sort_by_length": False, "max_pair_tokens": 512,
            "padding_id": 1, "batches": [], "input_tokens": 0, "padded_tokens": 0}
        engine = FakeReranker(case); counts = comparison.new_counters()
        result = comparison.run_one(case, feed, {**self.cell, "case_id": case["case_id"]}, engine, counts,
                                    expected_identity=IDENTITY, started_at=comparison.time.perf_counter())
        self.assertEqual(result["status"], "empty"); self.assertEqual(engine.calls, [])
        self.assertEqual(counts["score_attempts"], 0)

    def test_source_hash_anchor_checked_before_json_or_guard(self):
        """整份freeze被替换时，内层自洽也不能绕过事前锚点。"""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve(); frozen = base / "freeze.json"; frozen.write_text("{}")
            anchor = comparison.digest(frozen); frozen.write_text('{"self_consistent": true}')
            with patch.object(comparison.json, "loads", side_effect=AssertionError("parsed too early")), \
                    patch.object(comparison, "install_guards", side_effect=AssertionError("guard roots too early")):
                with self.assertRaisesRegex(BoundaryViolation, "FREEZE_ANCHOR_CHANGED"):
                    comparison.execute(frozen, anchor, base / "unused")

    def test_prepare_only_full_synthetic_fixture_never_loads_or_scores(self):
        """完整计划只预检，非法JSON标签也不被解析，ORT未导入。"""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve(); frozen, anchor = full_synthetic_freeze(base)
            output = base / "output"; output.mkdir()
            program = '''
from pathlib import Path
import sys
from scripts.compare_reranker_threads import execute, LocalReranker
def forbidden(*args, **kwargs): raise AssertionError("model was used")
LocalReranker._load = forbidden
LocalReranker.score = forbidden
result = execute(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), prepare_only=True)
assert result['status'] == 'complete'
assert result['counters']['score_attempts'] == 0
assert len(result['unexecuted']) == 686
assert 'onnxruntime' not in sys.modules
'''
            result = subprocess.run([sys.executable, "-c", program, str(frozen), anchor, str(output)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            preflight = json.loads((output / "preflight.json").read_text())
            self.assertFalse(preflight["model_loaded"]); self.assertFalse(preflight["labels_parsed"])
            self.assertEqual(len(list(output.glob("cell-*"))), 0)

    def test_process_guard_blocks_network_database_and_outside_writes(self):
        """独立子进程装守卫，被拒绝的操作不会污染测试主进程或外部服务。"""
        with tempfile.TemporaryDirectory() as folder:
            program = '''
from pathlib import Path
import socket, sqlite3, sys
from scripts.retrieval_experiment_guard import install_guards, BoundaryViolation
base = Path(sys.argv[1]).resolve(); output = base / 'output'; output.mkdir()
counts = install_guards(output, [])
for operation in (lambda: socket.create_connection(('127.0.0.1', 6339)),
                  lambda: sqlite3.connect(':memory:'), lambda: (base / 'outside').write_text('bad')):
    try: operation()
    except BoundaryViolation: pass
    else: raise AssertionError('guard missed operation')
assert counts['blocked_network'] >= 1 and counts['blocked_database'] == 1 and counts['blocked_write'] == 1
'''
            result = subprocess.run([sys.executable, "-c", program, folder], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_fatal_cell_is_saved_before_terminal_with_remaining_cells_unexecuted(self):
        """真实执行支架面对替身资源错误时先保存失败格，再终止而不重试。"""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve(); frozen, anchor = full_synthetic_freeze(base)
            output = base / "output"; output.mkdir()
            program = '''
from pathlib import Path
import json, sys
from scripts import compare_reranker_threads as comparison
from backend.tests.unit.test_compare_reranker_threads import FakeNativeSession
expected = json.loads(Path(sys.argv[1]).read_text())['expected_model_identity']
class StoppingReranker:
    def __init__(self, manifest, threads, batch_size, sort_by_length):
        self.threads = threads; self.last_stats = {}
    def _load(self):
        self.identity = {**expected, 'threads': self.threads}; self.session = FakeNativeSession(threads=self.threads)
    def score(self, query, texts): raise MemoryError('synthetic memory failure')
comparison.LocalReranker = StoppingReranker
result = comparison.execute(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
assert result['status'] == 'stopped'
assert result['counters']['score_attempts'] == 1
assert result['counters']['warmup_rows'] == 1 and result['counters']['hot_rows'] == 0
assert len(result['unexecuted']) == 685
assert 'onnxruntime' not in sys.modules
'''
            result = subprocess.run([sys.executable, "-c", program, str(frozen), anchor, str(output)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            failed = json.loads((output / "cell-0000-warmup-threads-2.json").read_text())
            self.assertEqual(failed["status"], "stopped")
            self.assertEqual(failed["error"]["type"], "MemoryError")
            self.assertIsNone(failed["scores"])

    def test_write_never_overwrites_existing_evidence(self):
        """同名格文件存在时保留原件，不把重跑覆盖成新成绩。"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cell-0000.json"; write_json_new(path, {"first": True})
            with self.assertRaises(FileExistsError): write_json_new(path, {"overwritten": True})
            self.assertEqual(json.loads(path.read_text()), {"first": True})


if __name__ == "__main__":
    unittest.main()
