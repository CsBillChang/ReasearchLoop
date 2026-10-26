"""只枚举运行时元数据并静态核对CoreML算子注册，不创建模型Session。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

from scripts.compare_lexical_retrieval import digest
from scripts.retrieval_experiment_guard import BoundaryViolation, install_guards, write_json_new


INPUT_NAMES = {"model_manifest", "weights", "coreml_factory", "coreml_helper", "coreml_model",
               "coreml_host_utils", "coreml_model_builder", "cpu_baseline_summary"}
AUTHORIZATION = {"runtime_inventory": True, "model_sessions": False, "model_scoring": False,
                 "model_conversion": False, "new_model_assets": False,
                 "dependency_install": False, "performance_comparison": False}
EXPECTED_OPERATORS = {"DynamicQuantizeLinear": 98, "MatMulInteger": 98,
                      "DequantizeLinear": 3, "MatMul": 0}
RUNTIME_AUDIT = {}


def is_hash(value: object) -> bool:
    """只接受完整小写SHA-256锚点。"""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def anchored_freeze(path: Path, expected: str) -> dict:
    """解析任何路径清单前，先与事前独立锚点核对。"""
    if not is_hash(expected) or digest(path) != expected:
        raise BoundaryViolation("COMPATIBILITY_FREEZE_ANCHOR_CHANGED")
    return json.loads(path.read_text())


def read_varint(buffer: memoryview, position: int) -> tuple[int, int]:
    """严格读取至多64位varint，拒绝截断、溢出与非规范长编码。"""
    if position < 0 or position >= len(buffer):
        raise ValueError("PROTOBUF_VARINT_TRUNCATED")
    value = 0
    for ordinal in range(10):
        if position >= len(buffer):
            raise ValueError("PROTOBUF_VARINT_TRUNCATED")
        byte = buffer[position]
        position += 1
        if ordinal == 9 and byte > 1:
            raise ValueError("PROTOBUF_VARINT_OVERFLOW")
        value |= (byte & 127) << (ordinal * 7)
        if byte < 128:
            if ordinal and byte == 0:
                raise ValueError("PROTOBUF_VARINT_NONCANONICAL")
            return value, position
    raise ValueError("PROTOBUF_VARINT_OVERFLOW")


def protobuf_fields(buffer: memoryview):
    """以memoryview跳过权重张量，仅展开调用方需要的消息字段。"""
    position = 0
    while position < len(buffer):
        key, position = read_varint(buffer, position)
        number, wire = key >> 3, key & 7
        if not 1 <= number <= (1 << 29) - 1:
            raise ValueError("PROTOBUF_FIELD_NUMBER_INVALID")
        if wire == 0:
            value, position = read_varint(buffer, position)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, position = read_varint(buffer, position)
            else:
                size = 8 if wire == 1 else 4
            if size > len(buffer) - position:
                raise ValueError("PROTOBUF_LENGTH_OUT_OF_BOUNDS")
            value = buffer[position:position + size]
            position += size
        else:
            raise ValueError("PROTOBUF_WIRE_TYPE_UNSUPPORTED")
        yield number, wire, value


def parse_node(buffer: memoryview) -> dict:
    """只读NodeProto输入、输出、名称、算子类型和domain。"""
    result = {"inputs": [], "outputs": [], "name": "", "domain": ""}
    seen = set()
    for number, wire, value in protobuf_fields(buffer):
        if number not in (1, 2, 3, 4, 7):
            continue
        if wire != 2:
            raise ValueError("ONNX_NODE_STRING_WIRE_INVALID")
        try:
            text = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("ONNX_NODE_UTF8_INVALID") from exc
        if number in (1, 2):
            result["inputs" if number == 1 else "outputs"].append(text)
        else:
            if number in seen:
                raise ValueError("ONNX_NODE_SINGULAR_FIELD_DUPLICATED")
            seen.add(number)
            result[{3: "name", 4: "op_type", 7: "domain"}[number]] = text
    if not result.get("op_type"):
        raise ValueError("ONNX_NODE_OP_TYPE_MISSING")
    return result


def inspect_graph(data: bytes | memoryview) -> dict:
    """静态统计顶层GraphProto，不验证完整ONNX语义或创建运行时对象。"""
    buffer = memoryview(data)
    graph = None
    for number, wire, value in protobuf_fields(buffer):
        if number == 7:
            if wire != 2:
                raise ValueError("ONNX_GRAPH_WIRE_INVALID")
            if graph is not None:
                raise ValueError("ONNX_GRAPH_DUPLICATED")
            graph = value
    if graph is None:
        raise ValueError("ONNX_GRAPH_MISSING")
    nodes, graph_name = [], None
    for number, wire, value in protobuf_fields(graph):
        if number == 1:
            if wire != 2:
                raise ValueError("ONNX_GRAPH_NODE_WIRE_INVALID")
            nodes.append(parse_node(value))
        elif number == 2:
            if wire != 2 or graph_name is not None:
                raise ValueError("ONNX_GRAPH_NAME_INVALID")
            try:
                graph_name = bytes(value).decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ValueError("ONNX_GRAPH_UTF8_INVALID") from exc
    counts = Counter(node["op_type"] for node in nodes)
    domains = Counter((node["domain"], node["op_type"]) for node in nodes)
    return {"node_count": len(nodes), "graph_name": graph_name,
            "operator_counts": dict(sorted(counts.items())),
            "operators_by_domain": [{"domain": domain, "op_type": op, "count": count}
                                    for (domain, op), count in sorted(domains.items())],
            "critical_node_examples": [node for node in nodes
                                       if node["op_type"] in EXPECTED_OPERATORS][:6],
            "scope": "top-level graph only; initializer payloads skipped without copying",
            "limitations": ["Not a complete ONNX schema or type/shape validator.",
                            "No graph optimization, partitioning, compilation, or execution was performed."]}


def cpp_tokens(text: str):
    """跳过C++注释并将字符串视为单个token，不把示例字符串当注册调用。"""
    position, size, line = 0, len(text), 1
    while position < size:
        if text[position].isspace():
            line += text[position] == "\n"; position += 1; continue
        if text.startswith("//", position):
            end = text.find("\n", position)
            position = size if end < 0 else end
            continue
        if text.startswith("/*", position):
            end = text.find("*/", position + 2)
            if end < 0:
                raise ValueError("CPP_COMMENT_UNTERMINATED")
            line += text[position:end + 2].count("\n"); position = end + 2; continue
        if text.startswith('R"', position):
            opening = text.find("(", position + 2)
            if opening < 0 or opening - position > 18:
                raise ValueError("CPP_RAW_STRING_INVALID")
            delimiter = text[position + 2:opening]
            closing = text.find(")" + delimiter + '"', opening + 1)
            if closing < 0:
                raise ValueError("CPP_RAW_STRING_UNTERMINATED")
            ending = closing + len(delimiter) + 2
            yield "string", text[opening + 1:closing], line
            line += text[position:ending].count("\n"); position = ending; continue
        character = text[position]
        if character in ('"', "'"):
            start, token_line = position + 1, line
            position += 1
            while position < size and text[position] != character:
                if text[position] == "\\":
                    position += 2
                else:
                    position += 1
            if position >= size:
                raise ValueError("CPP_STRING_UNTERMINATED")
            value = text[start:position]
            yield "string" if character == '"' else "character", value, token_line
            line += text[start:position].count("\n"); position += 1; continue
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[position:])
        if match:
            value = match.group(0); yield "identifier", value, line
            position += len(value); continue
        yield "punctuation", character, line
        position += 1


def parse_registry(text: str) -> dict:
    """提取字面量Create…OpBuilder调用，仅证明名称在该源注册表出现。"""
    tokens = list(cpp_tokens(text))
    registrations = []
    for index in range(len(tokens) - 3):
        kind, builder, line = tokens[index]
        if (kind == "identifier" and re.fullmatch(r"Create[A-Za-z0-9_]+OpBuilder", builder)
                and tokens[index + 1][:2] == ("punctuation", "(")
                and tokens[index + 2][0] == "string"
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tokens[index + 2][1])
                and tokens[index + 3][:2] == ("punctuation", ",")):
            registrations.append({"op_type": tokens[index + 2][1], "builder": builder, "line": line})
    if not registrations:
        raise ValueError("COREML_REGISTRY_NO_LITERAL_REGISTRATIONS")
    return {"registered_op_types": sorted({r["op_type"] for r in registrations}),
            "registrations": registrations,
            "limitations": ["Registration is a name-level necessary condition, not dtype/shape/opset qualification.",
                            "Source registration does not establish the installed binary's exact build or runtime partition.",
                            "This parser does not evaluate C++ preprocessor conditions or execute official source."]}


def collect_runtime_inventory(runtime=None) -> dict:
    """仅调用三项元数据API各一次，先封住已加载模块暴露的Session入口。"""
    global RUNTIME_AUDIT
    counters = {"available_providers_calls": 0, "all_providers_calls": 0, "build_info_calls": 0,
                "session_attempts": 0, "score_attempts": 0, "model_compile_requests": 0}
    RUNTIME_AUDIT = counters
    if runtime is None:
        runtime = importlib.import_module("onnxruntime")

    def denied_session(*args, **kwargs):
        counters["session_attempts"] += 1
        raise BoundaryViolation("MODEL_SESSION_FORBIDDEN_IN_METADATA_INSPECTION")

    targets = [("runtime", runtime)] + [(name, module) for name, module in list(sys.modules.items())
                                       if name.startswith("onnxruntime.") and module is not None]
    blocked = []
    for module_name, module in targets:
        for entry in ("InferenceSession", "TrainingSession"):
            if entry in vars(module):
                setattr(module, entry, denied_session)
                blocked.append(module_name + "." + entry)
    results = {}
    for api, key in (("get_available_providers", "available_providers"),
                     ("get_all_providers", "all_providers"), ("get_build_info", "build_info")):
        counters[{"available_providers": "available_providers_calls", "all_providers": "all_providers_calls",
                  "build_info": "build_info_calls"}[key]] += 1
        results[key] = getattr(runtime, api)()
    if (not isinstance(results["available_providers"], list)
            or not isinstance(results["all_providers"], list)
            or any(not isinstance(v, str) for k in ("available_providers", "all_providers") for v in results[k])
            or not isinstance(results["build_info"], str)):
        raise BoundaryViolation("RUNTIME_METADATA_RESULT_INVALID")
    if counters["session_attempts"]:
        raise BoundaryViolation("MODEL_SESSION_ATTEMPTED_DURING_METADATA_INSPECTION")
    return {**results, "runtime_version": str(runtime.__version__), "platform": sys.platform,
            "machine": os.uname().machine, "blocked_session_entrypoints": blocked, "counters": counters,
            "limitations": ["ORT native runtime was imported for provider/build metadata; this is not pure text inspection.",
                            "No model path was passed to ORT, no Session was created, and no model inference/compile API was called.",
                            "Python process guards do not prove absence of all native-library OS side effects."]}


def qualify_compatibility(graph: dict, registry: dict, inventory: dict) -> dict:
    """缺关键INT8注册即止于未具备直接对照资格，不外推GPU性能。"""
    supported = set(registry["registered_op_types"])
    counts = graph["operator_counts"]
    coverage = [{"op_type": op, "node_count": count, "registered": op in supported}
                for op, count in sorted(counts.items())]
    critical_missing = [op for op in ("DynamicQuantizeLinear", "MatMulInteger")
                        if counts.get(op, 0) and op not in supported]
    coreml_available = "CoreMLExecutionProvider" in inventory["available_providers"]
    status = ("not_qualified_for_direct_gpu_comparison" if critical_missing or not coreml_available
              else "runtime_partition_qualification_required")
    return {"status": status, "coreml_runtime_provider_available": coreml_available,
            "critical_unregistered_int8_ops": critical_missing,
            "operator_registration_coverage": coverage,
            "unregistered_operator_types": [row["op_type"] for row in coverage if not row["registered"]],
            "actual_coreml_partition": "not_measured", "actual_gpu_execution": "not_measured",
            "native_compile_cache_isolation": "not_qualified", "performance_comparison": "not_performed",
            "session_calls": 0, "model_score_calls": 0, "model_compile_requests": 0,
            "interpretation": ["Partial floating-point subgraphs may still be eligible for CoreML.",
                               "Missing direct registration does not prove the entire model cannot run with CPU fallback.",
                               "No conclusion about GPU speed or end-to-end latency follows from this inspection.",
                               "All registered names would still require dtype/shape/opset and actual partition qualification."]}


def check_reference(ref: dict) -> Path:
    """只读冻结的真实绝对文件，任何身份变化都停止。"""
    path = Path(ref["path"])
    if (not path.is_absolute() or path != path.resolve() or not path.is_file()
            or not is_hash(ref.get("sha256")) or digest(path) != ref["sha256"]):
        raise BoundaryViolation("COMPATIBILITY_REFERENCE_CHANGED")
    return path


def references(frozen: dict) -> list[dict]:
    """仅从已锚定合同获得读取白名单。"""
    return list(frozen["inputs"].values()) + frozen["source_files"] + frozen["protected_files"]


def inspect(freeze: Path, expected_sha256: str, output: Path) -> dict:
    """在新输出目录完成一次元数据/静态核验，无可切换的推理模式。"""
    freeze = freeze.resolve()
    frozen = anchored_freeze(freeze, expected_sha256)
    refs = references(frozen)
    roots = [Path(ref["path"]).resolve() for ref in refs]
    roots.extend([freeze, Path(__file__).resolve().parents[1]])
    guards = install_guards(output, roots)
    started = time.perf_counter()
    status, error, inventory, compatibility = "complete", None, None, None
    try:
        if (frozen.get("schema") != "apple-reranker-compatibility-v1"
                or frozen.get("mode") != "metadata_static_only" or frozen.get("authorization") != AUTHORIZATION
                or set(frozen["inputs"]) != INPUT_NAMES or frozen.get("baseline") != {"threads": 4}
                or frozen.get("expected_operator_counts") != EXPECTED_OPERATORS
                or frozen.get("packages") != {"onnxruntime": "1.31.0", "tokenizers": "0.23.3", "numpy": "2.5.4"}):
            raise BoundaryViolation("COMPATIBILITY_CONTRACT_CHANGED")
        for ref in refs:
            check_reference(ref)
        for name, version in frozen["packages"].items():
            if importlib.metadata.version(name) != version:
                raise BoundaryViolation("COMPATIBILITY_PACKAGE_VERSION_CHANGED")
        inputs = frozen["inputs"]
        model = json.loads(Path(inputs["model_manifest"]["path"]).read_text())["reranker"]
        weight_path = Path(inputs["weights"]["path"])
        if (weight_path != Path(model["directory"]) / model["weights"]
                or inputs["weights"]["sha256"] != model["files"][model["weights"]]["sha256"]):
            raise BoundaryViolation("COMPATIBILITY_MODEL_IDENTITY_CHANGED")
        graph = inspect_graph(weight_path.read_bytes())
        if any(graph["operator_counts"].get(op, 0) != count for op, count in EXPECTED_OPERATORS.items()):
            raise BoundaryViolation("COMPATIBILITY_OPERATOR_COUNTS_CHANGED")
        registry = parse_registry(Path(inputs["coreml_factory"]["path"]).read_text())
        inventory = collect_runtime_inventory()
        if inventory["runtime_version"] != frozen["packages"]["onnxruntime"]:
            raise BoundaryViolation("COMPATIBILITY_RUNTIME_VERSION_CHANGED")
        compatibility = qualify_compatibility(graph, registry, inventory)
        compatibility["registry"] = registry
        compatibility["cpu_baseline_reference"] = inputs["cpu_baseline_summary"]
        compatibility["cpu_baseline_usage"] = "historical reference only; not a same-run Apple performance comparison"
        compatibility["freeze_sha256"] = expected_sha256
        write_json_new(output / "inventory.json", inventory)
        write_json_new(output / "graph-static.json", graph)
        write_json_new(output / "compatibility.json", compatibility)
    except BaseException as exc:
        status, error = "stopped", {"type": type(exc).__name__, "reason": str(exc)}
    terminal = {"status": status, "finished_at": datetime.now(timezone.utc).isoformat(), "error": error,
                "mode": "metadata_static_only", "session_calls": 0, "model_score_calls": 0,
                "model_compile_requests": 0, "guard_counts": guards,
                "runtime_inventory_counters": dict(RUNTIME_AUDIT) if RUNTIME_AUDIT else None,
                "compatibility_status": compatibility["status"] if compatibility else None,
                "elapsed_seconds": time.perf_counter() - started,
                "boundary": "No Session, inference, conversion, source execution, or model compilation path is provided."}
    write_json_new(output / "terminal.json", terminal)
    print(json.dumps(terminal, ensure_ascii=False), flush=True)
    return terminal


def main() -> int:
    """CLI只接受元数据静态检查，未提供GPU运行或模型转换开关。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if inspect(args.freeze, args.expected_freeze_sha256, args.output)["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
