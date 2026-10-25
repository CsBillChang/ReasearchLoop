"""评测入口与隔离存储验证；所有编码/生成均为替身，不是收益实验。"""
from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
from uuid import uuid4

from scripts.component_eval_contracts import freeze_manifest, validate_manifest
from scripts.evaluate_components import _case_isolation, main, report, run_variant


PROJECT = Path(__file__).resolve().parents[3]


def file_ref(path: Path) -> dict:
    """返回实际文件身份；不以文件名代替内容版本。"""
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def chunks_fixture() -> list[dict]:
    """生成完整来源身份的合成片段；没有读取真实论文。"""
    rows = []
    for number, text in enumerate(("language prefix detection", "image classification", "open vocabulary detection")):
        row = {"chunk_id": f"chunk-{number}", "document_id": "synthetic-document-v1",
               "index_id": "synthetic-index-v1", "evidence_id": f"page-{number + 1}",
               "source_sha256": "a" * 64, "text_sha256": sha256(text.encode()).hexdigest(),
               "page_number": number + 1, "char_start": 0, "char_end": len(text), "text": text}
        rows.append(row)
    return rows


class FakeRuntime:
    """每题每组新建真实SQLite与文件目录，检索仅运行现有纯BM25。"""

    execution_kind = "substitute"

    def __init__(self, *, manifest, case, variant, isolation, output_dir):
        """接收入口派生的独立存储，绝不读取.env或默认生产连接。"""
        from research_loop.config import Settings
        from research_loop.storage.mysql import Database
        self.directory = output_dir
        self.settings = Settings(project_root=output_dir, database_url=isolation["database_url"],
                                 data_dir=output_dir / "data", output_dir=output_dir / "notes")
        self.db = Database(isolation["database_url"])
        self.db.ensure_schema()
        self.task_id = case["case_id"] + "-" + variant
        self.db.create_task(self.task_id, self.task_id, "synthetic-input", "synthetic query", "deny", [],
                            token_settings=self.settings)
        (output_dir / "synthetic-only.txt").write_text("评测替身；不含真实论文或用户数据。")

    def run_case(self, case, input_data, labels, variant):
        """调用真正BM25算法，另一路故障为明确注入，不伪称向量效果。"""
        from scripts.component_eval_adapters import RetrievalAdapter
        self.db.append_event(self.task_id, "eval_substitute", {"variant": variant})
        if variant == "bm25":
            raw = RetrievalAdapter().run(input_data["query"], input_data["chunks"], variant, case["scope"])
        else:
            raw = {"status": "failed", "effective_path": "unavailable", "ranked_ids": [], "timing": {},
                   "degradations": ["controlled_vector_failure"], "trace": [], "usage": None}
        return {"status": raw["status"], "requested_variant": variant, "effective_path": raw["effective_path"],
                "ranked_ids": raw["ranked_ids"], "timing": raw["timing"], "degradations": raw["degradations"],
                "trace": raw["trace"], "usage": raw["usage"], "output": {"substitute": True,
                    "task_id": self.task_id, "root_rows": len(self.db.fetchall("SELECT * FROM tasks")),
                    "save_policy": self.db.get_task(self.task_id)["save_policy"],
                    "policy": self.db.execution_policy(self.task_id)}}

    def close(self):
        """SQLite连接由Database逐次管理；没有后台线程或模型客户端。"""


def fake_factory(**kwargs):
    """已冻结的替身装配函数，供CLI同等入口验证。"""
    return FakeRuntime(**kwargs)


def unknown_factory(**kwargs):
    """明确注入结果未知，验证不自动重发且保留剩余未执行格。"""
    class UnknownRuntime(FakeRuntime):
        """只在替身执行边界制造未知结果。"""

        def run_case(self, *args):
            """抛出边界异常，不发送任何真实请求。"""
            raise RuntimeError("controlled unsettled result")
    return UnknownRuntime(**kwargs)


def malformed_after_usage_factory(**kwargs):
    """结果字段错误发生在计费观测后，已知用量不能随异常丢失。"""
    class MalformedRuntime(FakeRuntime):
        """模拟已观察用量但输出字段不合法的组件。"""

        def run_case(self, *args):
            """只返回合成用量，格式错误不代表可重发请求。"""
            return {"status": "executed", "effective_path": "bm25", "output": "invalid output type",
                    "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}}
    return MalformedRuntime(**kwargs)


def malformed_extra_after_usage_factory(**kwargs):
    """注入未知顶层字段，验证坏格式不能漏掉已经进入执行的格。"""
    class MalformedExtraRuntime(FakeRuntime):
        """模拟额外字段与错误计时，保留原始诊断输入。"""

        def run_case(self, *args):
            """返回非法trace/timing及字段，但用量本身有效。"""
            return {"status": "executed", "effective_path": "bm25", "output": {},
                    "unexpected": True, "timing": {"total_seconds": -1}, "trace": "invalid",
                    "usage": {"total_tokens": 10}}
    return MalformedExtraRuntime(**kwargs)


class ComponentEvalIsolationTests(unittest.TestCase):
    """默认本地无网络：校验/冻结/单路线/报告及文件、状态隔离。"""

    def setUp(self):
        """创建两题两路线的合成计划，不引用任何旧实验配置。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.isolation = self.root / "isolated"
        self.output = self.isolation / "batch"
        input_file = self.root / "input.json"
        input_file.write_text(json.dumps({"query": "language detection", "chunks": chunks_fixture()}))
        label_file = self.root / "labels.json"
        label_file.write_text(json.dumps({"answerable": True, "relevant": ["chunk-0"]}))
        config = self.root / "config.json"
        config.write_text(json.dumps({"origin": "explicit synthetic fixture", "budget_changes": False}))
        self.manifest = {
            "experiment_id": "component-integration", "component": "retrieval", "measurement_level": "substitute",
            "variants": ["bm25", "vector"],
            "cases": [{"case_id": f"case-{n}", "pair_id": f"pair-{n}",
                       "split": "regression" if n == 0 else "pending", "input": file_ref(input_file),
                       "labels": file_ref(label_file), "scope": [r["chunk_id"] for r in chunks_fixture()]} for n in range(2)],
            "identities": {"code": [file_ref(Path(__file__).resolve()),
                                      file_ref(PROJECT / "backend/scripts/evaluate_components.py"),
                                      file_ref(PROJECT / "backend/scripts/component_eval_contracts.py"),
                                      file_ref(PROJECT / "backend/scripts/component_eval_adapters.py")],
                           "dependencies": [file_ref(PROJECT / "backend/uv.lock")],
                           "configuration": [file_ref(config)], "weights": []},
            "isolation": {"root": str(self.isolation), "database_url": "sqlite:///" + str(self.isolation / "seed.sqlite3"),
                          "vector_url": None, "collection_prefix": "researchloop_eval_component_fixture"},
            "execution": {"kind": "substitute", "candidate_limit": 24},
        }
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))

    def freeze(self):
        """按正式纯合同冻结，不使用历史runner。"""
        return freeze_manifest(self.manifest, self.root, self.output)

    def test_validate_is_standard_library_only_and_does_not_write(self):
        """-S环境中导入/validate无需Settings、数据库、HTTP或模型依赖。"""
        code = """import json,pathlib,sys,socket
sys.path.insert(0,sys.argv[1])
def deny(*args,**kwargs): raise AssertionError('unexpected network')
socket.socket.connect=deny
from scripts.component_eval_contracts import validate_manifest
from scripts.evaluate_components import main
assert 'research_loop.config' not in sys.modules
assert main(['validate','--manifest',sys.argv[2]])==0
assert 'research_loop.config' not in sys.modules
"""
        before = sorted(str(p) for p in self.root.rglob("*"))
        result = subprocess.run([sys.executable, "-S", "-c", code, str(PROJECT / "backend"), str(self.manifest_path)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob("*")))

    def test_run_preserves_full_matrix_and_isolates_case_writes(self):
        """已执行故障为0，未运行格不补0；两组同源但存储独立。"""
        self.freeze()
        with patch("socket.socket.connect", side_effect=AssertionError("unexpected network")):
            run_variant(self.output, "bm25", runtime_factory=fake_factory)
            partial = report(self.output)
            self.assertEqual(partial["totals"]["planned"], 4)
            self.assertEqual(partial["totals"]["status_counts"]["unexecuted"], 2)
            self.assertEqual(partial["paired_complete"], 0)
            run_variant(self.output, "vector", runtime_factory=fake_factory)
            complete = report(self.output)
        self.assertEqual(complete["paired_complete"], 2)
        self.assertEqual(complete["by_variant"]["vector"]["retrieval_metrics"]["recall_at_10"]["value"], 0)
        self.assertEqual(complete["totals"]["usage"]["total_tokens"]["known_sum"], None)
        databases = list(self.output.glob("runs/*/*/runtime.sqlite3"))
        self.assertEqual(len(databases), 4)
        for row in complete["rows"]:
            self.assertEqual(row["output"]["root_rows"], 1)
            self.assertEqual(row["output"]["save_policy"], "deny")
        self.assertEqual(len(list(self.output.glob("runs/*/*/synthetic-only.txt"))), 4)
        self.assertFalse((self.isolation / "seed.sqlite3").exists())

    def test_repeated_run_is_refused_without_second_factory(self):
        """成功或失败均不可覆盖重跑，避免实验刷分和请求重发。"""
        self.freeze()
        run_variant(self.output, "bm25", runtime_factory=fake_factory)
        with self.assertRaises(FileExistsError):
            run_variant(self.output, "bm25", runtime_factory=fake_factory)

    def test_unknown_stops_without_replay_and_reports_remaining_cases(self):
        """未知结果保持unknown，后续题和另组仍是未执行。"""
        self.freeze()
        result = run_variant(self.output, "bm25", runtime_factory=unknown_factory)
        self.assertTrue(result["stopped_on_unknown"])
        value = report(self.output)
        self.assertEqual(value["totals"]["status_counts"]["unknown"], 1)
        self.assertEqual(value["totals"]["status_counts"]["unexecuted"], 3)

    def test_drift_and_real_execution_are_rejected_before_runtime(self):
        """材料变更或真实执行类型不会隐式落到真实组件工厂。"""
        self.freeze()
        Path(self.manifest["cases"][0]["input"]["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "FROZEN_FILE_CHANGED"):
            run_variant(self.output, "bm25", runtime_factory=fake_factory)
        self.assertFalse((self.output / "runs").exists())

    def test_real_execution_is_not_enabled_by_manifest_only(self):
        """合法的未来执行清单不自动授权当前runner加载真实组件。"""
        self.manifest["execution"]["kind"] = "local_components"
        self.freeze()
        with self.assertRaisesRegex(ValueError, "REAL_EXECUTION_NOT_AUTHORIZED"):
            run_variant(self.output, "bm25", runtime_factory=fake_factory)
        self.assertFalse((self.output / "runs").exists())

    def test_known_usage_survives_result_normalization_error(self):
        """无效输出保留原始结果和合法已知用量，未知状态不自动补零。"""
        self.freeze()
        run_variant(self.output, "bm25", runtime_factory=malformed_after_usage_factory)
        value = report(self.output)
        self.assertEqual(value["totals"]["status_counts"]["unknown"], 1)
        self.assertEqual(value["totals"]["usage"]["total_tokens"]["known_sum"], 10)
        raw = list(self.output.glob("runs/*/*/raw-result.json"))
        self.assertEqual(len(raw), 1)
        self.assertEqual(json.loads(raw[0].read_text())["usage"]["total_tokens"], 10)

    def test_remote_namespaces_include_batch_identity(self):
        """同题同路线在不同冻结批次不能共用远端schema或集合前缀。"""
        first = validate_manifest(self.manifest, self.root)
        second = json.loads(json.dumps(first))
        second["experiment_id"] = "different-batch"
        first["isolation"]["database_url"] = "mysql://127.0.0.1:33328/researchloop_test_component_eval_batch"
        second["isolation"]["database_url"] = first["isolation"]["database_url"]
        a = _case_isolation(first, first["cases"][0], "bm25", self.root / "a")
        b = _case_isolation(second, second["cases"][0], "bm25", self.root / "b")
        self.assertNotEqual(a["database_url"], b["database_url"])
        self.assertNotEqual(a["collection_prefix"], b["collection_prefix"])

    def test_bad_extra_fields_do_not_erase_executed_cell_and_cost(self):
        """错误字段只留原件，标准未知记录仍落盘并计入已知成本。"""
        self.freeze()
        run_variant(self.output, "bm25", runtime_factory=malformed_extra_after_usage_factory)
        value = report(self.output)
        self.assertEqual(value["totals"]["status_counts"]["unknown"], 1)
        self.assertEqual(value["totals"]["usage"]["total_tokens"]["known_sum"], 10)
        raw = json.loads(next(self.output.glob("runs/*/*/raw-result.json")).read_text())
        self.assertTrue(raw["unexpected"])

    def test_report_path_must_stay_inside_batch(self):
        """汇总不能写回旧成果或任意路径。"""
        self.freeze()
        with redirect_stdout(io.StringIO()):
            code = main(["report", "--output", str(self.output), "--write", str(self.root / "outside.json")])
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "outside.json").exists())
        with redirect_stdout(io.StringIO()):
            code = main(["report", "--output", str(self.output), "--write", str(self.output / ".." / "outside.json")])
        self.assertEqual(code, 2)
        self.assertFalse((self.output.parent / "outside.json").exists())


@unittest.skipUnless(os.getenv("COMPONENT_EVAL_TEST_DATABASE_URL") and os.getenv("COMPONENT_EVAL_TEST_VECTOR_URL"),
                     "需要显式独立MySQL和Qdrant地址；不使用生产默认")
class ComponentEvalExternalStorageTests(unittest.TestCase):
    """真实隔离存储与合成向量；不运行模型或接触生产服务器。"""

    def test_mysql_qdrant_namespaces_and_receipts(self):
        """两组相同片段写不同namespace，收据/取消/失权保护仍生效。"""
        from research_loop.agents.graph import build_graph, initial_state
        from research_loop.providers.deepseek import ModelResponse
        from research_loop.storage.mysql import Database
        from research_loop.storage.vector_index import VectorIndex, point_id
        from research_loop.tools.registry import ToolRegistry
        import httpx
        url = os.environ["COMPONENT_EVAL_TEST_DATABASE_URL"]
        vector_url = os.environ["COMPONENT_EVAL_TEST_VECTOR_URL"]
        parsed = urlsplit(url)
        self.assertEqual(parsed.hostname, "127.0.0.1")
        self.assertEqual(parsed.port, 33328)
        self.assertTrue(parsed.path.startswith("/researchloop_test_component_eval_"))
        self.assertEqual(vector_url, "http://127.0.0.1:6349")
        db = Database(url)
        db.ensure_schema()
        self.assertEqual(db.fetchone("SELECT DATABASE() AS name")["name"], parsed.path[1:])
        token = uuid4().hex[:10]
        scopes = []
        for group in ("a", "b"):
            # 显式隔离客户端不继承机器代理，防止本地夹具绕行未知网络端点。
            client = httpx.Client(base_url=vector_url, trust_env=False)
            self.addCleanup(client.close)
            vectors = VectorIndex(vector_url, "researchloop_eval_" + token + "_" + group, client=client)
            self.addCleanup(vectors.close)
            collection = vectors.collection("f" * 64)
            vectors.ensure_collection(collection, 2)
            point = {"id": point_id("same-index", "same-chunk"), "vector": [1.0, 0.0],
                     "payload": {"index_id": "same-index", "chunk_id": "same-chunk", "group": group}}
            vectors.upsert(collection, [point])
            vectors.upsert(collection, [point])
            self.assertEqual(vectors.count(collection, "same-index"), 1)
            self.assertEqual(vectors.retrieve(collection, [point["id"]])[point["id"]]["payload"]["group"], group)
            scopes.append(collection)
        self.assertNotEqual(*scopes)

        class Provider:
            """合成生成器，统计调用以检验收据恢复是否重发。"""

            def __init__(self):
                """从零记录替身调用数，不初始化供应商客户端。"""
                self.calls = 0

            def complete(self, *args, **kwargs):
                """返回固定合成回答及已披露的测试用量。"""
                self.calls += 1
                return ModelResponse("合成回答", raw={"choices": [{"message": {"content": "合成回答"}, "finish_reason": "stop"}]},
                                     usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}, request_id="fake-" + token)

        tid = "eval-" + token
        db.create_task(tid, tid, "synthetic-hash", "合成指令", "deny", [])
        provider = Provider()
        graph = build_graph(provider, ToolRegistry([]), {}, db)
        graph.invoke(initial_state(tid, "合成指令", "deny"))
        graph.invoke(initial_state(tid, "合成指令", "deny"))
        self.assertEqual(provider.calls, 1)
        self.assertEqual(db.list_model_calls(tid)[0]["request_id"], "fake-" + token)
        unknown = tid + "-unknown"
        db.create_task(unknown, unknown, "unknown-hash", "合成未知", "deny", [])
        db.save_model_call(unknown, 1, "unknown", {"explicit_fixture": True})
        build_graph(provider, ToolRegistry([]), {}, db).invoke(initial_state(unknown, "合成未知", "deny"))
        self.assertEqual(provider.calls, 1)
        self.assertTrue(db.has_unknown_model_call(unknown))
        claimed = tid + "-claimed"
        db.create_task(claimed, claimed, "claim-hash", "合成领取", "deny", [])
        owner = db.claim_task("eval-owner", task_id=claimed)
        self.assertIsNotNone(owner)
        self.assertFalse(db.is_owner(claimed, "wrong-owner", owner["lease_token"]))
        db.update_task(claimed, cancel_requested=1)
        self.assertEqual(db.get_task(claimed)["cancel_requested"], 1)
        # 取消/失权必须由真实服务拒写，而非仅检查任务标记。
        from research_loop.agents.reflection import ReflectionService
        from research_loop.domain.notes import CreateNoteInput
        from research_loop.services.notes import NoteService
        from research_loop.storage.files import FileStore
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory).resolve()
            draft = NoteService(db, FileStore(local / "data", local / "notes", local)).create(
                tid, CreateNoteInput(title="synthetic", markdown="合成原稿。"))
            with self.assertRaises(PermissionError):
                ReflectionService(db, owner=("wrong-owner", "wrong-lease")).review(
                    tid, draft.draft_id, draft.content_hash, "合成核验", [], [], lambda *a: self.fail("must not invoke"))
            db.update_task(tid, cancel_requested=1)
            with self.assertRaises(PermissionError):
                ReflectionService(db).review(tid, draft.draft_id, draft.content_hash,
                                             "合成核验", [], [], lambda *a: self.fail("must not invoke"))
            self.assertEqual(db.get_draft(draft.draft_id)["markdown"], "合成原稿。")
            self.assertFalse(list((local / "notes").rglob("*.md")))


if __name__ == "__main__":
    unittest.main()
