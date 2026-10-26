"""合成protobuf与假运行时覆盖Apple兼容性检查，全程不导入ORT或创建Session。"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import inspect_apple_reranker_compatibility as inspection
from scripts.retrieval_experiment_guard import BoundaryViolation, write_json_new


def varint(value):
    """只为合成夹具编码无符号varint。"""
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128); value >>= 7
    result.append(value)
    return bytes(result)


def field(number, value):
    """构造长度字段，不调用任何protobuf依赖。"""
    if isinstance(value, str): value = value.encode()
    return varint((number << 3) | 2) + varint(len(value)) + value


def node(op_type, domain=""):
    """构造具有输入、输出与算子身份的合成NodeProto。"""
    return field(1, "input") + field(2, "output") + field(3, "synthetic-node") + field(4, op_type) + field(7, domain)


def model(operators):
    """只包含顶层图；与真实权重文件无关联。"""
    graph = b"".join(field(1, node(op)) for op in operators) + field(2, "synthetic-graph")
    return field(7, graph)


def fake_runtime(*, coreml=True):
    """返回仅提供元数据的替身，Session入口应先被检查器封住。"""
    calls = {"available": 0, "all": 0, "build": 0}
    def available():
        calls["available"] += 1
        return (["CoreMLExecutionProvider"] if coreml else []) + ["CPUExecutionProvider"]
    def all_providers(): calls["all"] += 1; return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    def build(): calls["build"] += 1; return "synthetic runtime build"
    def session(*args, **kwargs): raise AssertionError("unpatched session entry")
    runtime = SimpleNamespace(__version__="1.31.0", get_available_providers=available,
                              get_all_providers=all_providers, get_build_info=build,
                              InferenceSession=session, TrainingSession=session)
    return runtime, calls


class AppleRerankerCompatibilityTests(unittest.TestCase):
    """检查拒绝边界与有限结论，不把静态兼容性当作运行或速度证据。"""

    def test_static_graph_counts_and_domains_without_tensor_parsing(self):
        """大initializer内容可为任意字节，静态统计不解析或复制其payload。"""
        graph = (field(1, node("DynamicQuantizeLinear")) + field(1, node("MatMulInteger"))
                 + field(1, node("CustomOp", "synthetic.domain"))
                 + field(5, b"\xff" * 10000) + field(2, "synthetic"))
        result = inspection.inspect_graph(field(7, graph))
        self.assertEqual(result["node_count"], 3)
        self.assertEqual(result["operator_counts"], {"CustomOp": 1, "DynamicQuantizeLinear": 1, "MatMulInteger": 1})
        self.assertIn({"domain": "synthetic.domain", "op_type": "CustomOp", "count": 1}, result["operators_by_domain"])
        buffer = memoryview(field(5, b"large-payload"))
        payload = next(inspection.protobuf_fields(buffer))[2]
        self.assertIsInstance(payload, memoryview); self.assertIs(payload.obj, buffer.obj)

    def test_varint_accepts_uint64_and_rejects_truncation_overflow_noncanonical(self):
        """严格区分64位上界、未结束编码和多余前导组。"""
        encoded = varint((1 << 64) - 1)
        self.assertEqual(inspection.read_varint(memoryview(encoded), 0), ((1 << 64) - 1, 10))
        for value in (b"", b"\x80", b"\xff" * 10 + b"\x00", b"\x80\x00", b"\xff" * 9 + b"\x02"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): inspection.read_varint(memoryview(value), 0)

    def test_protobuf_lengths_field_numbers_and_wire_types_are_bounded(self):
        """长度越界、零字段号、截断fixed字段和groups都拒绝。"""
        for data in (b"\x3a\x10short", b"\x00", b"\x09short", b"\x0b", b"\x0e"):
            with self.subTest(data=data):
                with self.assertRaises(ValueError): list(inspection.protobuf_fields(memoryview(data)))

    def test_missing_duplicate_graph_and_wrong_node_wire_are_rejected(self):
        """ModelProto只允许一个长度型graph，graph节点也必须为消息。"""
        for data in (b"", field(7, b"") + field(7, b""), b"\x38\x01", field(7, b"\x08\x01")):
            with self.subTest(data=data):
                with self.assertRaises(ValueError): inspection.inspect_graph(data)

    def test_invalid_utf8_and_duplicate_singular_node_fields_are_rejected(self):
        """不能用损坏字符串或重复op_type隐蔽更换算子身份。"""
        bad_nodes = (field(4, b"\xff"), field(1, "x"), node("MatMul") + field(4, "Add"), b"\x20\x01")
        for bad in bad_nodes:
            with self.subTest(data=bad):
                with self.assertRaises(ValueError): inspection.inspect_graph(field(7, field(1, bad)))
        with self.assertRaisesRegex(ValueError, "GRAPH_UTF8_INVALID"):
            inspection.inspect_graph(field(7, field(2, b"\xff")))

    def test_registry_matches_literal_calls_and_ignores_comments_and_string_decoys(self):
        """只提取调用token，注释、示例字符串、变量参数不成为支持证据。"""
        source = r'''
// CreateFakeOpBuilder("DynamicQuantizeLinear", op_registrations);
/* CreateFakeOpBuilder("MatMulInteger", op_registrations); */
const char* example = "CreateFakeOpBuilder(\"DequantizeLinear\", op_registrations)";
const char* raw = R"tag(CreateFakeOpBuilder("Fake", op_registrations))tag";
CreateUnaryOpBuilder("Sqrt", op_registrations);
CreateGemmOpBuilder /* comment */ ("MatMul", op_registrations);
CreateOtherOpBuilder(dynamic_op_name, op_registrations);
'''
        result = inspection.parse_registry(source)
        self.assertEqual(result["registered_op_types"], ["MatMul", "Sqrt"])
        self.assertEqual(len(result["registrations"]), 2)
        self.assertTrue(all(r["line"] > 0 for r in result["registrations"]))

    def test_registry_unterminated_sources_or_no_calls_are_rejected(self):
        """不从截断源码或空匹配得出所有算子均不支持的假结论。"""
        for text in ("/* missing end", 'const char* x = "missing end', 'R"tag(missing end', "// no calls"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError): inspection.parse_registry(text)

    def test_missing_critical_quantized_ops_stops_direct_comparison_qualification(self):
        """浮点MatMul注册不能替代真实模型中的MatMulInteger。"""
        graph = {"operator_counts": {"DynamicQuantizeLinear": 98, "MatMulInteger": 98, "DequantizeLinear": 3}}
        registry = {"registered_op_types": ["MatMul", "Add"]}
        result = inspection.qualify_compatibility(graph, registry, {"available_providers": ["CoreMLExecutionProvider"]})
        self.assertEqual(result["status"], "not_qualified_for_direct_gpu_comparison")
        self.assertEqual(result["critical_unregistered_int8_ops"], ["DynamicQuantizeLinear", "MatMulInteger"])
        self.assertEqual(result["actual_gpu_execution"], "not_measured")
        self.assertEqual(result["performance_comparison"], "not_performed")

    def test_registered_ops_are_not_sufficient_for_actual_gpu_qualification(self):
        """所有名称注册后仍需实际类型、分区和隔离资格，不自动执行。"""
        graph = {"operator_counts": {"MatMul": 1}}
        result = inspection.qualify_compatibility(graph, {"registered_op_types": ["MatMul"]},
                                                 {"available_providers": ["CoreMLExecutionProvider"]})
        self.assertEqual(result["status"], "runtime_partition_qualification_required")
        self.assertEqual(result["native_compile_cache_isolation"], "not_qualified")
        self.assertEqual(result["session_calls"], 0)
        unavailable = inspection.qualify_compatibility(graph, {"registered_op_types": ["MatMul"]},
                                                       {"available_providers": ["CPUExecutionProvider"]})
        self.assertEqual(unavailable["status"], "not_qualified_for_direct_gpu_comparison")

    def test_fake_inventory_enumerates_each_api_once_and_blocks_sessions(self):
        """只有三项元数据API调用；任何后续Session尝试会立即被封住。"""
        runtime, calls = fake_runtime()
        result = inspection.collect_runtime_inventory(runtime)
        self.assertEqual(calls, {"available": 1, "all": 1, "build": 1})
        self.assertEqual(result["counters"]["session_attempts"], 0)
        self.assertEqual(result["counters"]["score_attempts"], 0)
        self.assertIn("CoreMLExecutionProvider", result["available_providers"])
        with self.assertRaisesRegex(BoundaryViolation, "SESSION_FORBIDDEN"):
            runtime.InferenceSession("not-a-model")
        self.assertEqual(result["counters"]["session_attempts"], 1)

    def test_inventory_session_attempt_during_metadata_is_rejected_and_audited(self):
        """元数据替身若试图创建Session，失败也保留尝试计数而不重试。"""
        runtime, _calls = fake_runtime()
        runtime.get_available_providers = lambda: runtime.InferenceSession("forbidden")
        with self.assertRaises(BoundaryViolation): inspection.collect_runtime_inventory(runtime)
        self.assertEqual(inspection.RUNTIME_AUDIT["session_attempts"], 1)
        self.assertEqual(inspection.RUNTIME_AUDIT["available_providers_calls"], 1)
        self.assertEqual(inspection.RUNTIME_AUDIT["all_providers_calls"], 0)

    def test_invalid_runtime_metadata_is_rejected(self):
        """未知provider形态或无build字符串都不伪造兼容性通过。"""
        runtime, _calls = fake_runtime(); runtime.get_available_providers = lambda: [None]
        with self.assertRaisesRegex(BoundaryViolation, "METADATA_RESULT_INVALID"):
            inspection.collect_runtime_inventory(runtime)

    def test_freeze_anchor_replacement_stops_before_parsing_or_guard(self):
        """自洽替换后的冻结不能绕过事前独立hash。"""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve(); frozen = base / "freeze.json"; frozen.write_text("{}")
            anchor = inspection.digest(frozen); frozen.write_text('{"self_consistent":true}')
            with patch.object(inspection.json, "loads", side_effect=AssertionError("parsed too early")), \
                    patch.object(inspection, "install_guards", side_effect=AssertionError("roots too early")):
                with self.assertRaisesRegex(BoundaryViolation, "FREEZE_ANCHOR_CHANGED"):
                    inspection.inspect(frozen, anchor, base / "output")

    def test_output_creation_is_exclusive(self):
        """兼容性工件与旧失败记录同名时拒绝覆盖。"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "compatibility.json"; write_json_new(path, {"original": True})
            with self.assertRaises(FileExistsError): write_json_new(path, {"overwritten": True})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_guard_rejects_network_database_outside_write_and_nonempty_output(self):
        """独立子进程检查不可逆守卫，不发送请求或污染主进程。"""
        with tempfile.TemporaryDirectory() as folder:
            program = '''
from pathlib import Path
import socket, sqlite3, sys
from scripts.retrieval_experiment_guard import install_guards, BoundaryViolation, write_json_new
base = Path(sys.argv[1]).resolve(); output = base / 'output'; output.mkdir()
(output / 'old').write_text('retained')
try: install_guards(output, [])
except ValueError: pass
else: raise AssertionError('nonempty output accepted')
(output / 'old').unlink()
counts = install_guards(output, [])
for operation in (lambda: socket.create_connection(('127.0.0.1', 6339)),
                  lambda: sqlite3.connect(':memory:'), lambda: (base / 'outside').write_text('bad')):
    try: operation()
    except BoundaryViolation: pass
    else: raise AssertionError('operation was not blocked')
assert counts['blocked_network'] >= 1 and counts['blocked_database'] == 1 and counts['blocked_write'] == 1
'''
            result = subprocess.run([sys.executable, "-c", program, folder], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_script_exposes_no_model_execution_or_source_execution_path(self):
        """AST中没有Session构造、score、转换、compile或exec调用。"""
        tree = ast.parse(Path(inspection.__file__).read_text())
        forbidden = {"InferenceSession", "TrainingSession", "score", "compile", "exec", "eval", "_load"}
        for token in ast.walk(tree):
            if isinstance(token, ast.Call):
                name = token.func.attr if isinstance(token.func, ast.Attribute) else (
                    token.func.id if isinstance(token.func, ast.Name) else "")
                self.assertNotIn(name, forbidden)


if __name__ == "__main__":
    unittest.main()
