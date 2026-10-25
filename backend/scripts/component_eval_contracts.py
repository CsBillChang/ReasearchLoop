"""组件评测的显式合同、独占冻结和完整分母汇总。

导入与校验只使用标准库，不读取环境配置、不连接服务。排名汇总才惰性
复用历史模块的纯指标函数，不调用其配置、冻结或运行入口。
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlparse


VARIANTS = {
    "retrieval": {"vector", "bm25", "rrf", "rrf_rerank"},
    "reflection": {"baseline", "reflection"},
    "memory": {"baseline", "history_recall", "preference_context", "context_projection"},
}
MEASUREMENT_LEVELS = {
    "local_retrieval_component", "fixed_draft_reflection",
    "context_projection_replay", "full_agent_task", "substitute",
}
STATUSES = {"executed", "failed", "unknown", "cancelled", "unexecuted"}
QUALITIES = {"unjudged", "full", "partial_usable", "correct_unanswerable", "wrong_or_missing"}
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")
_SECRET = re.compile(r"(?:password|passwd|api[_-]?key|access[_-]?token|secret|authorization)", re.I)


def _object(value: object, name: str) -> dict:
    """要求JSON对象，避免列表或隐式类型转换掩盖输入错误。"""
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"INVALID_OBJECT: {name}")
    return value


def _keys(value: dict, required: set[str], optional: set[str], name: str) -> None:
    """拒绝缺项和未知字段，防止拼写错误静默改变实验。"""
    if required - value.keys() or value.keys() - required - optional:
        raise ValueError(f"INVALID_FIELDS: {name}")


def _text(value: object, name: str) -> str:
    """要求非空字符串身份。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"INVALID_STRING: {name}")
    return value


def _ids(value: object, name: str) -> list[str]:
    """要求有序且唯一的字符串身份集合。"""
    if not isinstance(value, list):
        raise ValueError(f"INVALID_LIST: {name}")
    for item in value:
        _text(item, name)
    if len(value) != len(set(value)):
        raise ValueError(f"DUPLICATE_ID: {name}")
    return value


def _json_copy(value: object) -> object:
    """深拷贝有限JSON值，同时拒绝NaN、非JSON类型及凭据。"""
    def inspect(item: object) -> None:
        """递归拒绝明文凭据字段及带认证信息的URL。"""
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("NON_STRING_JSON_KEY")
                if _SECRET.search(key) and child not in (None, "", False):
                    raise ValueError("PLAINTEXT_CREDENTIAL_REJECTED")
                inspect(child)
        elif isinstance(item, list):
            for child in item:
                inspect(child)
        elif isinstance(item, str) and "://" in item:
            parsed = urlparse(item)
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("URL_CREDENTIAL_REJECTED")
    inspect(value)
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("INVALID_JSON_VALUE") from error


def _path(value: object, base: Path | None = None) -> Path:
    """解析显式路径并拒绝穿越、符号链接和非规范绝对入口。"""
    raw = Path(_text(value, "path"))
    if ".." in raw.parts:
        raise ValueError("PATH_TRAVERSAL_REJECTED")
    if not raw.is_absolute():
        if base is None:
            raise ValueError("ABSOLUTE_PATH_REQUIRED")
        raw = base / raw
    for part in (raw, *raw.parents):
        if part.is_symlink():
            raise ValueError("SYMLINK_REJECTED")
    return raw.resolve(strict=False)


def _within(path: Path, root: Path) -> bool:
    """判断路径是否严格位于隔离根之下。"""
    return path != root and path.is_relative_to(root)


def _file_ref(value: object, base: Path) -> dict:
    """校验显式文件身份并返回绝对路径，不修复或重写源文件。"""
    ref = _object(value, "file_ref")
    _keys(ref, {"path", "sha256"}, set(), "file_ref")
    expected = ref["sha256"]
    if not isinstance(expected, str) or not _SHA.fullmatch(expected):
        raise ValueError("INVALID_SHA256")
    path = _path(ref["path"], base)
    if not path.is_file():
        raise ValueError("FILE_NOT_FOUND")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected.lower():
        raise ValueError("FROZEN_FILE_CHANGED")
    return {"path": str(path), "sha256": expected.lower()}


def _read_ref(ref: dict) -> dict:
    """重新核验文件hash后读取JSON对象，避免汇总使用已漂移标签。"""
    checked = _file_ref(ref, Path.cwd())
    try:
        return _object(json.loads(Path(checked["path"]).read_text()), "file_content")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("INVALID_JSON_FILE") from error


def _retrieval_labels(case: dict) -> dict:
    """固定可回答性、相关性及授权范围，拒绝相关ID越界。"""
    labels = _read_ref(case["labels"])
    if "answerable" not in labels or "relevant" not in labels:
        raise ValueError("RETRIEVAL_LABELS_REQUIRED")
    if type(labels["answerable"]) is not bool:
        raise ValueError("INVALID_ANSWERABILITY")
    relevant = _ids(labels["relevant"], "relevant")
    payload = _read_ref(case["input"])
    _text(payload.get("query"), "query")
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("RETRIEVAL_CHUNKS_REQUIRED")
    chunk_ids = []
    for chunk in chunks:
        chunk = _object(chunk, "chunk")
        chunk_ids.append(_text(chunk.get("chunk_id"), "chunk_id"))
        _text(chunk.get("text"), "chunk.text")
    _ids(chunk_ids, "chunk_ids")
    corpus = _ids(labels.get("corpus", chunk_ids), "corpus")
    if (not set(case["scope"]).issubset(chunk_ids) or not set(corpus).issubset(chunk_ids)
            or not set(case["scope"]).issubset(corpus)):
        raise ValueError("OUT_OF_SCOPE_CORPUS")
    if not set(relevant).issubset(case["scope"]) or not set(relevant).issubset(corpus):
        raise ValueError("OUT_OF_SCOPE_QRELS")
    if bool(relevant) != labels["answerable"]:
        raise ValueError("ANSWERABILITY_QRELS_MISMATCH")
    return {"answerable": labels["answerable"], "relevant": relevant,
            "corpus": list(set(corpus) & set(case["scope"]))}


def _isolation(value: object, base: Path) -> dict:
    """校验独立文件根、数据库和索引命名，不访问这些服务。"""
    isolation = _object(value, "isolation")
    _keys(isolation, {"root", "database_url", "vector_url", "collection_prefix"}, set(), "isolation")
    root = _path(isolation["root"])
    project = Path(__file__).absolute().parents[2]
    # 不能把项目材料树当作可写的隔离工作区；历史目录也不能直接复用。
    artifacts = project / "artifacts"
    if root in project.parents or (root.is_relative_to(project) and not _within(root, artifacts)):
        raise ValueError("PROTECTED_ISOLATION_ROOT")
    if root == artifacts:
        raise ValueError("EXISTING_ARTIFACT_ROOT_REJECTED")
    database_url = _text(isolation["database_url"], "database_url")
    parsed = urlparse(database_url)
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("DATABASE_URL_PARAMETERS_REJECTED")
    if parsed.scheme == "sqlite":
        if parsed.netloc or not database_url.startswith(parsed.scheme + ":///"):
            raise ValueError("EXPLICIT_SQLITE_REQUIRED")
        database = _path(parsed.path)
        if not _within(database, root):
            raise ValueError("DATABASE_OUTSIDE_ISOLATION")
    elif parsed.scheme == "mysql":
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("LOCAL_DATABASE_REQUIRED")
        if parsed.port is None or parsed.port in {0, 3306, 33316}:
            raise ValueError("ISOLATED_DATABASE_PORT_REQUIRED")
        if not re.fullmatch(r"/researchloop_test_component_eval_[A-Za-z0-9_]+", parsed.path):
            raise ValueError("ISOLATED_DATABASE_REQUIRED")
    else:
        raise ValueError("ISOLATED_DATABASE_REQUIRED")
    prefix = _text(isolation["collection_prefix"], "collection_prefix")
    if (not re.fullmatch(r"researchloop_eval_[a-z0-9_]+", prefix)
            or not re.fullmatch(r"researchloop_[a-z0-9_]{1,64}", prefix)):
        raise ValueError("ISOLATED_COLLECTION_REQUIRED")
    if isolation["vector_url"] is not None:
        vector = urlparse(_text(isolation["vector_url"], "vector_url"))
        if (vector.scheme not in {"http", "https"} or vector.hostname not in
                {"localhost", "127.0.0.1", "::1"} or vector.query or vector.fragment
                or vector.path or vector.username is not None or vector.password is not None
                or vector.port is None or vector.port in {0, 6339}):
            raise ValueError("LOCAL_VECTOR_URL_REQUIRED")
    return {**isolation, "root": str(root)}


def validate_manifest(manifest: dict, base_dir: Path) -> dict:
    """校验组件合同及所有文件身份，返回路径绝对化的独立副本。

    Args:
        manifest: 不含环境继承和凭据的显式JSON合同。
        base_dir: 相对文件引用的绝对基目录。

    Returns:
        标准化合同；不写目录、不创建数据库、不认证独立样本资格。

    Raises:
        ValueError: 字段、开关、隔离路径或材料身份无效。
    """
    base = _path(str(base_dir))
    if not base.is_dir():
        raise ValueError("BASE_DIRECTORY_REQUIRED")
    result = _object(_json_copy(manifest), "manifest")
    _keys(result, {"experiment_id", "component", "measurement_level", "variants", "cases",
                   "identities", "isolation", "execution"},
          {"annotation", "lineage", "timing"}, "manifest")
    if not _SLUG.fullmatch(_text(result["experiment_id"], "experiment_id")):
        raise ValueError("UNSAFE_EXPERIMENT_ID")
    component = result["component"]
    if not isinstance(component, str) or component not in VARIANTS:
        raise ValueError("INVALID_COMPONENT")
    if not isinstance(result["measurement_level"], str) or result["measurement_level"] not in MEASUREMENT_LEVELS:
        raise ValueError("INVALID_MEASUREMENT_LEVEL")
    variants = _ids(result["variants"], "variants")
    if not variants or not set(variants).issubset(VARIANTS[component]):
        raise ValueError("INVALID_COMPONENT_VARIANTS")
    if component in {"reflection", "memory"} and (
            len(variants) != 2 or "baseline" not in variants):
        raise ValueError("ONE_PAIRED_STRATEGY_REQUIRED")
    result["isolation"] = _isolation(result["isolation"], base)
    execution = _object(result["execution"], "execution")
    _keys(execution, {"kind", "candidate_limit"}, set(), "execution")
    if not isinstance(execution["kind"], str) or execution["kind"] not in {"substitute", "local_components", "real_model"}:
        raise ValueError("INVALID_EXECUTION_KIND")
    if type(execution["candidate_limit"]) is not int or execution["candidate_limit"] != 24:
        raise ValueError("CANDIDATE_LIMIT_MUST_BE_24")
    identities = _object(result["identities"], "identities")
    _keys(identities, {"code", "dependencies", "configuration", "weights"}, set(), "identities")
    for category, refs in identities.items():
        if not isinstance(refs, list) or (category != "weights" and not refs):
            raise ValueError("EXPLICIT_IDENTITIES_REQUIRED")
        identities[category] = [_file_ref(ref, base) for ref in refs]
        if len({ref["path"] for ref in identities[category]}) != len(refs):
            raise ValueError("DUPLICATE_FILE_IDENTITY")
    cases = result["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("CASES_REQUIRED")
    case_ids, pair_ids = set(), set()
    for case in cases:
        case = _object(case, "case")
        _keys(case, {"case_id", "pair_id", "split", "input", "labels", "scope"}, set(), "case")
        for name, seen in (("case_id", case_ids), ("pair_id", pair_ids)):
            identity = _text(case[name], name)
            if not _SLUG.fullmatch(identity) or identity in seen:
                raise ValueError("DUPLICATE_OR_UNSAFE_PAIR_ID")
            seen.add(identity)
        if not isinstance(case["split"], str) or case["split"] not in {"regression", "independent", "pending"}:
            raise ValueError("INVALID_SPLIT")
        _ids(case["scope"], "scope")
        case["input"] = _file_ref(case["input"], base)
        case["labels"] = _file_ref(case["labels"], base)
        if component == "retrieval":
            _retrieval_labels(case)
        else:
            _read_ref(case["input"])
            _read_ref(case["labels"])
    for name in ("annotation", "lineage", "timing"):
        if name in result:
            _object(result[name], name)
    return result


def manifest_digest(manifest: dict) -> str:
    """计算JSON规范序列化hash；列表顺序属于冻结实验身份。"""
    value = _json_copy(manifest)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def freeze_manifest(manifest: dict, base_dir: Path, output_dir: Path) -> dict:
    """在隔离根内的新目录独占冻结，不覆盖任何已有目录或成绩。

    Args:
        manifest: 显式实验合同。
        base_dir: 材料相对路径的绝对基目录。
        output_dir: 隔离根下尚不存在的输出目录。

    Returns:
        已写入freeze.json的记录，不表示运行或语义验收通过。
    """
    normalized = validate_manifest(manifest, base_dir)
    output = _path(str(output_dir))
    root = Path(normalized["isolation"]["root"])
    artifacts = Path(__file__).absolute().parents[2] / "artifacts"
    if root.is_relative_to(artifacts) and root.exists():
        raise ValueError("EXISTING_ARTIFACT_ROOT_REJECTED")
    if not _within(output, root):
        raise ValueError("OUTPUT_OUTSIDE_ISOLATION")
    if output.exists():
        raise FileExistsError("OUTPUT_ALREADY_EXISTS")
    frozen = {"format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "base_dir": str(_path(str(base_dir))), "manifest": normalized,
              "manifest_digest": manifest_digest(normalized)}
    output.parent.mkdir(parents=True, exist_ok=True)
    _path(str(output))
    output.mkdir(exist_ok=False)
    with (output / "freeze.json").open("x", encoding="utf-8") as handle:
        json.dump(frozen, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return frozen


def verify_freeze(output_dir: Path) -> dict:
    """复核冻结合同和材料身份；不恢复、重发或修改旧结果。"""
    output = _path(str(output_dir))
    freeze = _path(str(output / "freeze.json"))
    try:
        record = _object(json.loads(freeze.read_text()), "freeze")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("INVALID_FREEZE_FILE") from error
    _keys(record, {"format_version", "created_at", "base_dir", "manifest", "manifest_digest"}, set(), "freeze")
    if record["format_version"] != 1 or manifest_digest(record["manifest"]) != record["manifest_digest"]:
        raise ValueError("FREEZE_DIGEST_MISMATCH")
    normalized = validate_manifest(record["manifest"], Path(record["base_dir"]))
    if not _within(output, Path(normalized["isolation"]["root"])):
        raise ValueError("OUTPUT_OUTSIDE_ISOLATION")
    return {**record, "manifest": normalized}


def validate_result(manifest: dict, result: dict) -> dict:
    """校验一个计划格的结果，保留未知用量和实际有效路线。

    Args:
        manifest: 已通过validate_manifest的合同。
        result: 一次case与variant的执行结果。

    Returns:
        显式填充空审阅/空计时及nullable用量后的独立副本。
    """
    normalized = _object(_json_copy(result), "result")
    _keys(normalized, {"case_id", "pair_id", "variant", "requested_variant", "effective_path", "status"},
          {"ranked_ids", "usage", "quality", "timing", "degradations", "trace", "output"}, "result")
    case = next((item for item in manifest["cases"] if item["case_id"] == normalized["case_id"]), None)
    if case is None or normalized["pair_id"] != case["pair_id"]:
        raise ValueError("UNKNOWN_CASE_OR_PAIR")
    variant = normalized["variant"]
    if not isinstance(variant, str) or variant not in manifest["variants"] or normalized["requested_variant"] != variant:
        raise ValueError("UNPLANNED_OR_MISMATCHED_VARIANT")
    status = normalized["status"]
    if not isinstance(status, str) or status not in STATUSES:
        raise ValueError("INVALID_RESULT_STATUS")
    path = normalized["effective_path"]
    if path is not None:
        _text(path, "effective_path")
    if status == "executed" and path is None:
        raise ValueError("EXECUTED_PATH_REQUIRED")
    ranked = _ids(normalized.setdefault("ranked_ids", []), "ranked_ids")
    if len(ranked) > manifest["execution"]["candidate_limit"] or not set(ranked).issubset(case["scope"]):
        raise ValueError("INVALID_OR_OUT_OF_SCOPE_RANKING")
    if status == "unexecuted" and ranked:
        raise ValueError("NONEXECUTED_RANKING_REJECTED")
    if status == "unexecuted" and path is not None:
        raise ValueError("UNEXECUTED_PATH_REJECTED")
    usage = normalized.get("usage")
    if usage is None:
        usage = {}
    usage = _object(usage, "usage")
    _keys(usage, set(), set(TOKEN_FIELDS), "usage")
    for field in TOKEN_FIELDS:
        token = usage.setdefault(field, None)
        if token is not None and (type(token) is not int or token < 0):
            raise ValueError("INVALID_TOKEN_USAGE")
    if all(usage[field] is not None for field in TOKEN_FIELDS):
        if usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]:
            raise ValueError("TOKEN_USAGE_MISMATCH")
    if status == "unexecuted" and any(value is not None for value in usage.values()):
        raise ValueError("UNEXECUTED_USAGE_REJECTED")
    normalized["usage"] = usage
    quality = normalized.setdefault("quality", "unjudged")
    if not isinstance(quality, str) or quality not in QUALITIES or (status != "executed" and quality in
                                    {"full", "partial_usable", "correct_unanswerable"}):
        raise ValueError("INVALID_QUALITY_OR_STATUS")
    timing = _object(normalized.setdefault("timing", {}), "timing")
    for name, seconds in timing.items():
        if seconds is not None and (type(seconds) not in {int, float} or
                                    not math.isfinite(seconds) or seconds < 0):
            raise ValueError("INVALID_TIMING")
    for name in ("degradations", "trace"):
        if not isinstance(normalized.setdefault(name, []), list):
            raise ValueError("INVALID_RESULT_LIST")
    _object(normalized.setdefault("output", {}), "output")
    if path is not None and path != variant and not normalized["degradations"]:
        raise ValueError("EFFECTIVE_PATH_CHANGE_REQUIRES_REASON")
    return normalized


def _mean(values: list[float]) -> float | None:
    """计算非空观测均值；零分母保留N/A。"""
    return sum(values) / len(values) if values else None


def _aggregate(rows: list[dict]) -> dict:
    """计算完整计划格的覆盖、质量和已知成本，不用未知补零。"""
    entered = [row for row in rows if row["status"] != "unexecuted"]
    usable = sum(row["quality"] in {"full", "partial_usable"} for row in entered)
    usage = {}
    for field in TOKEN_FIELDS:
        known = [row["usage"][field] for row in entered if row["usage"][field] is not None]
        usage[field] = {"known_sum": sum(known) if known else None,
                        "known_count": len(known), "missing_count": len(entered) - len(known),
                        "complete": bool(entered) and len(known) == len(entered)}
    tokens = usage["total_tokens"]["known_sum"]
    metrics = {}
    for field in ("recall_at_10", "reciprocal_rank_at_24"):
        values = [row["retrieval_metrics"][field] for row in rows
                  if row.get("retrieval_metrics") is not None and row["retrieval_metrics"][field] is not None]
        metrics[field] = {"value": _mean(values), "denominator": len(values)}
    timing = {}
    for field in sorted({field for row in rows for field in row["timing"]}):
        values = sorted(row["timing"][field] for row in entered if row["timing"].get(field) is not None)
        timing[field] = {"count": len(values), "missing_count": len(entered) - len(values),
                         "p50": values[math.ceil(len(values) * .5) - 1] if values else None,
                         "p95": values[math.ceil(len(values) * .95) - 1] if values else None,
                         "method": "nearest_rank", "small_sample": len(values) < 20}
    return {"planned": len(rows), "entered_execution": len(entered),
            "status_counts": {status: sum(row["status"] == status for row in rows) for status in sorted(STATUSES)},
            "quality_counts": {quality: sum(row["quality"] == quality for row in rows) for quality in sorted(QUALITIES)},
            "task_usable": {"numerator": usable, "denominator": len(entered),
                            "value": usable / len(entered) if entered else None},
            "usage": usage, "tokens_per_usable_known": tokens / usable if usable and tokens is not None else None,
            "cost_complete": usage["total_tokens"]["complete"], "retrieval_metrics": metrics,
            "timing": timing}


def _difference(baseline: float | None, treatment: float | None) -> dict:
    """保留配对原值、绝对差及相对差，零基线不计算比例。"""
    available = baseline is not None and treatment is not None
    delta = treatment - baseline if available else None
    return {"baseline": baseline, "treatment": treatment, "delta": delta,
            "relative_change": delta / baseline if available and baseline else None}


def _paired_comparison(baseline: dict, treatment: dict) -> dict:
    """仅对两边均执行且结算明确的观测计算差值。"""
    available = all(row["status"] in {"executed", "failed"} for row in (baseline, treatment))
    def value(row: dict, section: str, field: str):
        """读取可配对的原始观测，未结算时保持未知。"""
        if not available:
            return None
        content = row[section]
        return content.get(field) if content is not None else None
    result = {"baseline_variant": baseline["variant"], "available": available,
              "retrieval_metrics": {field: _difference(value(baseline, "retrieval_metrics", field),
                                                       value(treatment, "retrieval_metrics", field))
                                    for field in ("recall_at_10", "reciprocal_rank_at_24")},
              "usage": {field: _difference(value(baseline, "usage", field), value(treatment, "usage", field))
                        for field in TOKEN_FIELDS},
              "timing": {field: _difference(value(baseline, "timing", field), value(treatment, "timing", field))
                         for field in sorted(baseline["timing"].keys() | treatment["timing"].keys())}}
    # 未审阅不是语义不正确，不能把它直接当作质量0计算收益。
    judged = available and all(row["quality"] != "unjudged" for row in (baseline, treatment))
    result["task_usable"] = _difference(
        int(baseline["quality"] in {"full", "partial_usable"}) if judged else None,
        int(treatment["quality"] in {"full", "partial_usable"}) if judged else None)
    return result


def summarize_results(manifest: dict, results: list[dict]) -> dict:
    """保留全部题×路线、配对缺项和资格分组，汇总已知观测。

    Args:
        manifest: 已校验且文件引用绝对化的合同。
        results: 每个计划格至多一条结果，不自动重试未知状态。

    Returns:
        包含完整计划表、逐路线/分组统计和完整配对的可JSON序列化报告。
        独立标签只作为申报分组，绝不自动认证独立资格。
    """
    if not isinstance(results, list):
        raise ValueError("RESULTS_LIST_REQUIRED")
    observed = {}
    for result in results:
        row = validate_result(manifest, result)
        key = row["case_id"], row["variant"]
        if key in observed:
            raise ValueError("DUPLICATE_RESULT_CELL")
        observed[key] = row
    rows = []
    pairs = []
    for case in manifest["cases"]:
        case_rows = []
        labels = _retrieval_labels(case) if manifest["component"] == "retrieval" else None
        for variant in manifest["variants"]:
            row = observed.get((case["case_id"], variant))
            if row is None:
                row = validate_result(manifest, {"case_id": case["case_id"], "pair_id": case["pair_id"],
                    "variant": variant, "requested_variant": variant, "effective_path": None,
                    "status": "unexecuted"})
            row["split"] = case["split"]
            row["retrieval_metrics"] = None
            if labels is not None and row["status"] in {"executed", "failed"}:
                from scripts.evaluation_common import ranked_metrics
                row["retrieval_metrics"] = ranked_metrics([] if row["status"] == "failed" else row["ranked_ids"], labels["relevant"],
                    answerable=labels["answerable"], corpus=set(labels["corpus"]))
            case_rows.append(row)
            rows.append(row)
        completed = all(row["status"] in {"executed", "failed"} for row in case_rows)
        baseline = next((row for row in case_rows if row["variant"] == "baseline"), case_rows[0])
        pairs.append({"case_id": case["case_id"], "pair_id": case["pair_id"], "split": case["split"],
                      "complete": completed,
                      "comparisons": {row["variant"]: _paired_comparison(baseline, row)
                                      for row in case_rows if row is not baseline},
                      "missing_variants": [row["variant"] for row in case_rows if row["status"] == "unexecuted"],
                      "unresolved_variants": [row["variant"] for row in case_rows
                                               if row["status"] in {"unknown", "cancelled"}]})
    return {"experiment_id": manifest["experiment_id"], "component": manifest["component"],
            "measurement_level": manifest["measurement_level"], "execution_kind": manifest["execution"]["kind"],
            "manifest_digest": manifest_digest(manifest), "independent_qualification": "not_certified",
            "rows": rows, "totals": _aggregate(rows),
            "by_variant": {variant: _aggregate([row for row in rows if row["variant"] == variant])
                           for variant in manifest["variants"]},
            "by_split": {split: _aggregate([row for row in rows if row["split"] == split])
                         for split in ("regression", "independent", "pending")},
            "pairs": pairs, "paired_complete": sum(pair["complete"] for pair in pairs),
            "paired_planned": len(pairs),
            "limitations": (["substitute结果仅证明装配，不是组件收益"]
                            if manifest["execution"]["kind"] == "substitute" else []) +
                           ["独立资格未自动认证", "unknown/取消/未执行不补排名零分",
                            "缺失usage只报告已知成本", "物理收据去重由执行适配器负责",
                            "计时仅按原始字段分组，小样本P95不支持SLA结论"]}
