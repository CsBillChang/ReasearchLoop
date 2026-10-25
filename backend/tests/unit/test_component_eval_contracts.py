"""组件合同的合成材料测试；不调用模型、网络或数据库。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.component_eval_contracts import (
    freeze_manifest, manifest_digest, summarize_results, validate_manifest,
    validate_result, verify_freeze,
)


class ComponentEvalContractTests(unittest.TestCase):
    """用独立临时文件验证计划格、身份和隔离边界。"""

    def setUp(self):
        """准备不含真实材料或服务地址的合成manifest。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "isolated"
        self.payload = {"query": "合成问题", "chunks": [
            {"chunk_id": "a", "text": "甲", "page": 1},
            {"chunk_id": "b", "text": "乙", "page": 2}]}
        self.labels = {"answerable": True, "relevant": ["a"], "corpus": ["a", "b"]}
        self.manifest = {
            "experiment_id": "synthetic-contract", "component": "retrieval",
            "measurement_level": "local_retrieval_component", "variants": ["bm25", "vector"],
            "cases": [{"case_id": "q1", "pair_id": "p1", "split": "regression",
                       "input": self.write_json("input.json", self.payload),
                       "labels": self.write_json("labels.json", self.labels), "scope": ["a", "b"]}],
            "identities": {name: [self.write_json(name + ".json", {"synthetic": name})]
                           for name in ("code", "dependencies", "configuration")},
            "isolation": {"root": str(self.root), "database_url": "sqlite:///" + str(self.root / "eval.db"),
                          "vector_url": None, "collection_prefix": "researchloop_eval_synthetic"},
            "execution": {"kind": "substitute", "candidate_limit": 24},
        }
        self.manifest["identities"]["weights"] = []

    def write_json(self, name, value):
        """写入合成文件并返回显式hash引用。"""
        path = self.base / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def normalized(self):
        """返回已校验的规范manifest。"""
        return validate_manifest(self.manifest, self.base)

    def result(self, variant="bm25", **changes):
        """生成带可见路线的一个合成执行结果。"""
        return {"case_id": "q1", "pair_id": "p1", "variant": variant,
                "requested_variant": variant, "effective_path": variant,
                "status": "executed", "ranked_ids": ["b", "a"], **changes}

    def test_validation_is_read_only_without_historical_import(self):
        """合同校验不加载旧配置模块，不写隔离目录或修改调用者对象。"""
        before = deepcopy(self.manifest)
        with patch.dict("sys.modules", {"scripts.evaluation_common": None}):
            normalized = self.normalized()
        self.assertEqual(self.manifest, before)
        self.assertFalse(self.root.exists())
        self.assertTrue(Path(normalized["cases"][0]["input"]["path"]).is_absolute())
        self.assertEqual(normalized, validate_manifest(normalized, self.base))

    def test_digest_is_stable_but_preserves_array_order(self):
        """键顺序不改变身份，路线顺序改变时必须形成新身份。"""
        normalized = self.normalized()
        self.assertEqual(manifest_digest(normalized), manifest_digest(dict(reversed(list(normalized.items())))))
        changed = deepcopy(normalized)
        changed["variants"].reverse()
        self.assertNotEqual(manifest_digest(normalized), manifest_digest(changed))

    def test_strict_fields_pairing_and_variants(self):
        """未知字段、候选规模变化和多Memory策略不能静默接受。"""
        changes = [lambda m: m.update(typo=True),
                   lambda m: m.update(experiment_id="../history"),
                   lambda m: m["execution"].update(candidate_limit=8),
                   lambda m: m["cases"].append(deepcopy(m["cases"][0])),
                   lambda m: m["cases"][0].update(split="new"),
                   lambda m: m["identities"].update(code=[]),
                   lambda m: m.update(variants=["bm25", "bm25"]),
                   lambda m: m.update(variants=["baseline"]),
                   lambda m: m.update(component="memory", variants=["baseline", "history_recall", "preference_context"])]
        for change in changes:
            with self.subTest(change=change):
                manifest = deepcopy(self.manifest)
                change(manifest)
                with self.assertRaises(ValueError):
                    validate_manifest(manifest, self.base)

    def test_reflection_and_each_memory_strategy_accept_one_pair(self):
        """非检索组件一次只接受baseline和声明的单个策略。"""
        for component, strategy in [("reflection", "reflection"), ("memory", "history_recall"),
                                    ("memory", "preference_context"), ("memory", "context_projection")]:
            with self.subTest(component=component, strategy=strategy):
                manifest = deepcopy(self.manifest)
                manifest.update(component=component, variants=["baseline", strategy])
                self.assertEqual(validate_manifest(manifest, self.base)["variants"], ["baseline", strategy])

    def test_file_hash_qrels_and_scope_are_checked(self):
        """身份漂移、相关性矛盾和越界标签均在运行前拒绝。"""
        for labels in ({"answerable": True, "relevant": []},
                       {"answerable": False, "relevant": ["a"]},
                       {"answerable": True, "relevant": ["outside"]},
                       {"answerable": 1, "relevant": ["a"]},
                       {"answerable": True, "relevant": ["a", "a"]}):
            with self.subTest(labels=labels):
                manifest = deepcopy(self.manifest)
                manifest["cases"][0]["labels"] = self.write_json("bad-labels.json", labels)
                with self.assertRaises(ValueError):
                    validate_manifest(manifest, self.base)
        self.manifest["cases"][0]["input"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "FROZEN_FILE_CHANGED"):
            self.normalized()

    def test_retrieval_query_must_exist_and_be_nonempty_text(self):
        """问题缺失、非字符串或仅空白时在校验和冻结前拒绝。"""
        for query in (None, 1, [], {}, "", "  "):
            with self.subTest(query=query):
                payload = deepcopy(self.payload)
                if query is None:
                    payload.pop("query")
                else:
                    payload["query"] = query
                manifest = deepcopy(self.manifest)
                manifest["cases"][0]["input"] = self.write_json("invalid-query.json", payload)
                with self.assertRaisesRegex(ValueError, "INVALID_STRING: query"):
                    validate_manifest(manifest, self.base)
                with self.assertRaisesRegex(ValueError, "INVALID_STRING: query"):
                    freeze_manifest(manifest, self.base, self.root / "batch")
        self.assertFalse(self.root.exists())

    def test_path_traversal_and_symlink_are_rejected(self):
        """合法hash也不能绕过路径穿越或符号链接保护。"""
        manifest = deepcopy(self.manifest)
        manifest["cases"][0]["input"]["path"] = "nested/../input.json"
        with self.assertRaisesRegex(ValueError, "PATH_TRAVERSAL"):
            validate_manifest(manifest, self.base)
        (self.base / "alias").symlink_to(self.base / "input.json")
        manifest["cases"][0]["input"]["path"] = "alias"
        with self.assertRaisesRegex(ValueError, "SYMLINK"):
            validate_manifest(manifest, self.base)

    def test_credentials_and_nonisolated_endpoints_are_rejected(self):
        """明文凭据、生产schema、远程服务和错误namespace不进入冻结。"""
        changes = [lambda m: m["isolation"].update(database_url="mysql://localhost/researchloop"),
                   lambda m: m["isolation"].update(database_url="mysql://user:secret@localhost/researchloop_test_component_eval_x"),
                   lambda m: m["isolation"].update(database_url="mysql://remote/researchloop_test_component_eval_x"),
                   lambda m: m["isolation"].update(database_url="sqlite:///relative.db"),
                   lambda m: m["isolation"].update(vector_url="https://example.com"),
                   lambda m: m["isolation"].update(collection_prefix="production"),
                   lambda m: m.update(annotation={"api_key": "do-not-save"})]
        for change in changes:
            with self.subTest(change=change):
                manifest = deepcopy(self.manifest)
                change(manifest)
                with self.assertRaises(ValueError):
                    validate_manifest(manifest, self.base)

    def test_explicit_local_databases_are_accepted(self):
        """SQLite文件必须位于隔离根内，MySQL必须显式测试schema。"""
        for url in ("sqlite:///" + str(self.root / "eval.db"),
                    "mysql://localhost:33326/researchloop_test_component_eval_synthetic"):
            with self.subTest(url=url):
                manifest = deepcopy(self.manifest)
                manifest["isolation"]["database_url"] = url
                validate_manifest(manifest, self.base)

    def test_strict_namespace_and_supported_database_schemes(self):
        """集合名与底层向量合同一致，仅接受实际支持的两种数据库scheme。"""
        for prefix in ("researchloop_eval_UPPER", "researchloop_eval_with-hyphen",
                       "researchloop_eval_" + "a" * 60, "researchloop_eval_"):
            with self.subTest(prefix=prefix):
                manifest = deepcopy(self.manifest)
                manifest["isolation"]["collection_prefix"] = prefix
                with self.assertRaisesRegex(ValueError, "ISOLATED_COLLECTION_REQUIRED"):
                    validate_manifest(manifest, self.base)
        for scheme in ("sqlite3", "sqlite+pysqlite", "mysql+pymysql"):
            with self.subTest(scheme=scheme):
                manifest = deepcopy(self.manifest)
                manifest["isolation"]["database_url"] = scheme + ":///" + str(self.root / "eval.db")
                with self.assertRaisesRegex(ValueError, "ISOLATED_DATABASE_REQUIRED"):
                    validate_manifest(manifest, self.base)
        manifest = deepcopy(self.manifest)
        manifest["isolation"]["database_url"] = "sqlite:///:memory:"
        with self.assertRaisesRegex(ValueError, "DATABASE_OUTSIDE_ISOLATION"):
            validate_manifest(manifest, self.base)
        for root in (Path("/"), *Path(__file__).resolve().parents[4].parents):
            with self.subTest(root=root):
                manifest = deepcopy(self.manifest)
                manifest["isolation"]["root"] = str(root)
                with self.assertRaisesRegex(ValueError, "PROTECTED_ISOLATION_ROOT"):
                    validate_manifest(manifest, self.base)

    def test_explicit_nonproduction_loopback_ports_are_required(self):
        """生产端口、默认端口和向量URL附加路径在运行前拒绝。"""
        for host in ("localhost", "localhost:3306", "127.0.0.1:33316", "localhost:0"):
            with self.subTest(database_host=host):
                manifest = deepcopy(self.manifest)
                manifest["isolation"]["database_url"] = "mysql://" + host + "/researchloop_test_component_eval_x"
                with self.assertRaisesRegex(ValueError, "ISOLATED_DATABASE_PORT_REQUIRED"):
                    validate_manifest(manifest, self.base)
        for vector in ("http://localhost", "http://localhost:6339", "http://localhost:0",
                       "http://localhost:6340/", "http://localhost:6340/private",
                       "http://localhost:6340?collection=production"):
            with self.subTest(vector=vector):
                manifest = deepcopy(self.manifest)
                manifest["isolation"]["vector_url"] = vector
                with self.assertRaisesRegex(ValueError, "LOCAL_VECTOR_URL_REQUIRED"):
                    validate_manifest(manifest, self.base)
        manifest = deepcopy(self.manifest)
        manifest["isolation"]["vector_url"] = "http://127.0.0.1:6340"
        validate_manifest(manifest, self.base)

    def test_freeze_is_exclusive_and_detects_identity_drift(self):
        """新目录可冻结与读回，旧目录不能覆盖，源文件漂移不能通过。"""
        output = self.root / "batch"
        frozen = freeze_manifest(self.manifest, self.base, output)
        self.assertEqual(verify_freeze(output), frozen)
        with self.assertRaises(FileExistsError):
            freeze_manifest(self.manifest, self.base, output)
        (self.base / "code.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "FROZEN_FILE_CHANGED"):
            verify_freeze(output)

    def test_freeze_rejects_external_or_preexisting_output(self):
        """冻结不写授权根外，不借已有空目录复用历史身份。"""
        with self.assertRaisesRegex(ValueError, "OUTPUT_OUTSIDE_ISOLATION"):
            freeze_manifest(self.manifest, self.base, self.base / "outside")
        output = self.root / "already"
        output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            freeze_manifest(self.manifest, self.base, output)

    def test_digest_tampering_is_rejected(self):
        """修改冻结manifest不能借原文件hash掩盖合同改变。"""
        output = self.root / "batch"
        freeze_manifest(self.manifest, self.base, output)
        path = output / "freeze.json"
        record = json.loads(path.read_text())
        record["manifest"]["variants"] = ["bm25"]
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "FREEZE_DIGEST_MISMATCH"):
            verify_freeze(output)

    def test_nullable_usage_unjudged_and_route_degradation(self):
        """缺usage不补零，路径降级必须有原因且不冒称请求路线成功。"""
        manifest = self.normalized()
        row = validate_result(manifest, self.result())
        self.assertEqual(row["usage"], dict.fromkeys(("prompt_tokens", "completion_tokens", "total_tokens")))
        self.assertEqual(row["quality"], "unjudged")
        with self.assertRaisesRegex(ValueError, "REQUIRES_REASON"):
            validate_result(manifest, self.result(effective_path="fallback"))
        row = validate_result(manifest, self.result(effective_path="fallback", degradations=["合成降级"]))
        self.assertEqual(row["requested_variant"], "bm25")
        self.assertEqual(row["effective_path"], "fallback")

    def test_bad_results_are_rejected(self):
        """越界排名、重复、错配键、虚假成功和无效计量均拒绝。"""
        manifest = self.normalized()
        for change in ({"ranked_ids": ["a", "a"]}, {"ranked_ids": ["outside"]},
                       {"pair_id": "wrong"}, {"requested_variant": "vector"},
                       {"usage": {"total_tokens": -1}}, {"usage": {"total_tokens": True}},
                       {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 8}},
                       {"timing": {"elapsed": -1}}, {"status": "failed", "quality": "full"},
                       {"status": "unexecuted"}, {"status": []}, {"status": {}},
                       {"extra": "unknown"}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    validate_result(manifest, self.result(**change))

    def test_summary_retains_full_grid_failure_zero_and_unknown(self):
        """已执行失败算零，unknown不补零，缺格和完整计划分母保留。"""
        self.manifest["variants"] = ["bm25", "vector", "rrf", "rrf_rerank"]
        manifest = self.normalized()
        summary = summarize_results(manifest, [self.result(quality="partial_usable",
            usage={"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}),
            self.result("vector", status="failed", ranked_ids=[]),
            self.result("rrf", status="unknown", ranked_ids=[])])
        self.assertEqual(len(summary["rows"]), 4)
        self.assertEqual(summary["totals"]["planned"], 4)
        self.assertEqual(summary["totals"]["entered_execution"], 3)
        self.assertEqual(summary["rows"][1]["retrieval_metrics"]["recall_at_10"], 0)
        self.assertIsNone(summary["rows"][2]["retrieval_metrics"])
        self.assertEqual(summary["rows"][3]["status"], "unexecuted")
        self.assertEqual(summary["totals"]["retrieval_metrics"]["recall_at_10"], {"value": .5, "denominator": 2})
        self.assertEqual(summary["totals"]["task_usable"]["value"], 1 / 3)
        self.assertEqual(summary["totals"]["usage"]["total_tokens"]["known_sum"], 10)
        self.assertEqual(summary["totals"]["usage"]["total_tokens"]["missing_count"], 2)
        self.assertFalse(summary["totals"]["cost_complete"])
        self.assertEqual(summary["pairs"][0]["missing_variants"], ["rrf_rerank"])
        self.assertEqual(summary["pairs"][0]["unresolved_variants"], ["rrf"])
        self.assertEqual(summary["paired_complete"], 0)

    def test_unanswerable_zero_denominator_and_qualification(self):
        """缺证题不计检索分，空执行N/A，独立组不自动获得认证。"""
        self.manifest["cases"][0]["labels"] = self.write_json("missing.json", {"answerable": False, "relevant": []})
        self.manifest["cases"][0]["split"] = "independent"
        manifest = self.normalized()
        summary = summarize_results(manifest, [])
        self.assertIsNone(summary["totals"]["task_usable"]["value"])
        self.assertIsNone(summary["totals"]["usage"]["total_tokens"]["known_sum"])
        self.assertEqual(summary["by_split"]["independent"]["planned"], 2)
        self.assertEqual(summary["independent_qualification"], "not_certified")
        summary = summarize_results(manifest, [self.result(), self.result("vector")])
        self.assertEqual(summary["paired_complete"], 1)
        self.assertIsNone(summary["totals"]["retrieval_metrics"]["recall_at_10"]["value"])

    def test_duplicate_results_and_changed_labels_do_not_summarize(self):
        """重复计划格与标签漂移不通过汇总，不覆盖原标签或筛选成绩。"""
        manifest = self.normalized()
        with self.assertRaisesRegex(ValueError, "DUPLICATE_RESULT_CELL"):
            summarize_results(manifest, [self.result(), self.result()])
        (self.base / "labels.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "FROZEN_FILE_CHANGED"):
            summarize_results(manifest, [self.result()])

    def test_pair_differences_and_timing_keep_raw_values(self):
        """配对保存原值与差，零基线相对差为N/A，计时不与其他阶段相加。"""
        summary = summarize_results(self.normalized(), [
            self.result(ranked_ids=["a"], quality="full", usage={"total_tokens": 0},
                        timing={"component_seconds": 2, "queue_seconds": 3}),
            self.result("vector", quality="partial_usable", usage={"total_tokens": 10},
                        timing={"component_seconds": 4, "queue_seconds": 1})])
        comparison = summary["pairs"][0]["comparisons"]["vector"]
        self.assertEqual(comparison["retrieval_metrics"]["reciprocal_rank_at_24"]["delta"], -.5)
        self.assertEqual(comparison["usage"]["total_tokens"]["delta"], 10)
        self.assertIsNone(comparison["usage"]["total_tokens"]["relative_change"])
        self.assertEqual(comparison["timing"]["component_seconds"]["delta"], 2)
        self.assertEqual(summary["totals"]["timing"]["component_seconds"]["p95"], 4)
        self.assertTrue(summary["totals"]["timing"]["component_seconds"]["small_sample"])

    def test_failed_partial_ranking_is_retained_but_scores_zero(self):
        """故障中观察到的排名保留诊断，但已执行失败不获检索正分。"""
        summary = summarize_results(self.normalized(), [self.result(status="failed", ranked_ids=["a"])])
        self.assertEqual(summary["rows"][0]["ranked_ids"], ["a"])
        self.assertEqual(summary["rows"][0]["retrieval_metrics"]["recall_at_10"], 0)


if __name__ == "__main__":
    unittest.main()
