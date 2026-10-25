"""隔离组件评测入口；校验与冻结不连接数据库，也不加载模型。

当前仅开放显式替身装配。真实本地推理及生成实验须先冻结材料、标签、
调用配置并另行授权；这里不会调用历史runner的config、prepare或run。
"""
from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import importlib
import inspect
import json
from pathlib import Path
import re
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

from scripts.component_eval_contracts import (
    freeze_manifest, manifest_digest, summarize_results, validate_manifest, validate_result, verify_freeze,
)


def _load(path: Path) -> dict:
    """读取显式JSON对象，不从环境推导配置。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("EVAL_JSON_OBJECT_REQUIRED")
    return value


def _write_new(path: Path, value: dict) -> None:
    """独占创建工件；已有结果和中断标记不被覆盖。"""
    if ".." in path.parts:
        raise ValueError("EVAL_PATH_TRAVERSAL")
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("EVAL_SYMLINK_OUTPUT")
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized + "\n")


def _factory(spec: str | None, injected: Callable | None, manifest: dict) -> Callable:
    """仅run阶段载入明确且已冻结的装配函数，不提供真实供应商默认值。"""
    if injected is None:
        if not spec or not re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", spec):
            raise ValueError("EVAL_EXPLICIT_FACTORY_REQUIRED")
        module, name = spec.split(":", 1)
        injected = getattr(importlib.import_module(module), name)
    if not callable(injected):
        raise ValueError("EVAL_FACTORY_NOT_CALLABLE")
    source = inspect.getsourcefile(injected)
    if source is None:
        raise ValueError("EVAL_FACTORY_SOURCE_REQUIRED")
    source = Path(source).resolve()
    code_refs = manifest["identities"]["code"]
    if not any(Path(item["path"]) == source and
               item["sha256"] == sha256(source.read_bytes()).hexdigest() for item in code_refs):
        raise ValueError("EVAL_FACTORY_NOT_FROZEN")
    return injected


def _case_isolation(manifest: dict, case: dict, variant: str, directory: Path) -> dict:
    """从冻结隔离范围派生每题每组独立存储，避免Memory等写入跨组泄漏。"""
    isolation = copy.deepcopy(manifest["isolation"])
    suffix = sha256((manifest_digest(manifest) + "\0" + case["case_id"] + "\0" + variant).encode()).hexdigest()[:12]
    parsed = urlsplit(isolation["database_url"])
    if parsed.scheme in {"sqlite", "sqlite3"}:
        isolation["database_url"] = "sqlite:///" + str(directory / "runtime.sqlite3")
    else:
        # 原URL只提供获准的专用服务器；每组新schema仍须由装配器用单库账号明确开通。
        schema = parsed.path.strip("/")
        isolation["database_url"] = urlunsplit(parsed._replace(path="/" + schema[:49] + "_" + suffix))
    isolation["root"] = str(directory)
    isolation["collection_prefix"] = isolation["collection_prefix"][:44] + "_" + suffix
    return isolation


def _known_usage(value: dict) -> dict | None:
    """外围归一化失败仍保留合法已知usage；不推算缺失字段或纠正矛盾。"""
    usage = value.get("usage")
    if not isinstance(usage, dict):
        return None
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    result = {field: usage.get(field) for field in fields}
    if any(token is not None and (type(token) is not int or token < 0) for token in result.values()):
        return None
    if all(result[field] is not None for field in fields) and (
            result["prompt_tokens"] + result["completion_tokens"] != result["total_tokens"]):
        return None
    return result


def run_variant(output_dir: Path, variant: str, *, factory_spec: str | None = None,
                runtime_factory: Callable | None = None) -> dict:
    """执行冻结清单中的一组；成功、失败和unknown均不自动重发。

    Args:
        output_dir: 已独占冻结的新批次目录。
        variant: 清单中已声明的一条路线。
        factory_spec: 显式module:callable装配入口，仅run时导入。
        runtime_factory: 测试可注入同样经过源码hash校验的装配函数。

    Returns:
        本次实际结果及未执行原因；完整计划分母由report统一补齐。
    """
    frozen = verify_freeze(output_dir)
    manifest = frozen["manifest"]
    if variant not in manifest["variants"]:
        raise ValueError("EVAL_VARIANT_NOT_PLANNED")
    if manifest["execution"]["kind"] != "substitute":
        raise ValueError("EVAL_REAL_EXECUTION_NOT_AUTHORIZED")
    from scripts.component_eval_adapters import IsolationGuard
    # Guard先于工厂和任何连接执行；未选中的真实组件没有构造入口。
    IsolationGuard(manifest["isolation"], project_root=Path(__file__).resolve().parents[2])
    factory = _factory(factory_spec, runtime_factory, manifest)
    run_dir = output_dir / "runs" / variant
    _write_new(run_dir / "started.json", {
        "variant": variant, "manifest_digest": frozen["manifest_digest"],
        "execution_kind": "substitute", "retry_policy": "no_automatic_replay",
    })
    results = []
    for position, case in enumerate(manifest["cases"]):
        # 每次执行前重新核对输入，避免首次校验之后变更问题或标签。
        verify_freeze(output_dir)
        case_dir = run_dir / (f"{position:04d}_" + sha256(case["case_id"].encode()).hexdigest()[:12])
        case_dir.mkdir(exist_ok=False)
        isolation = _case_isolation(manifest, case, variant, case_dir)
        IsolationGuard(isolation, project_root=Path(__file__).resolve().parents[2])
        _write_new(case_dir / "isolation.json", isolation)
        result = {
            "case_id": case["case_id"], "pair_id": case["pair_id"], "variant": variant,
            "requested_variant": variant, "effective_path": None, "status": "unexecuted",
            "ranked_ids": [], "usage": None, "quality": "unjudged", "timing": {},
            "degradations": [], "trace": [], "output": {},
        }
        runtime = None
        try:
            runtime = factory(manifest=copy.deepcopy(manifest), case=copy.deepcopy(case),
                              variant=variant, isolation=isolation, output_dir=case_dir)
            if getattr(runtime, "execution_kind", None) != "substitute":
                raise ValueError("EVAL_SUBSTITUTE_RUNTIME_REQUIRED")
            observed = runtime.run_case(
                copy.deepcopy(case), _load(Path(case["input"]["path"])),
                _load(Path(case["labels"]["path"])), variant)
            if not isinstance(observed, dict):
                raise ValueError("EVAL_RUNTIME_RESULT_OBJECT_REQUIRED")
            result.update(observed)
            _write_new(case_dir / "raw-result.json", observed)
            if any(result.get(key) != expected for key, expected in (
                ("case_id", case["case_id"]), ("pair_id", case["pair_id"]),
                ("variant", variant), ("requested_variant", variant))):
                raise ValueError("EVAL_RUNTIME_RESULT_IDENTITY_MISMATCH")
            result = validate_result(manifest, result)
        except Exception as error:
            # 外围不能推断供应商是否已经发送；异常保守记unknown，不重试或伪填usage。
            result = {
                "case_id": case["case_id"], "pair_id": case["pair_id"],
                "variant": variant, "requested_variant": variant, "status": "unknown",
                "usage": _known_usage(result), "effective_path": "runtime_error",
                "ranked_ids": [], "quality": "unjudged",
                "timing": {}, "trace": [],
                "degradations": [type(error).__name__],
                "output": {"error_type": type(error).__name__, "reason": str(error)},
            }
        finally:
            if runtime is not None and callable(getattr(runtime, "close", None)):
                try:
                    runtime.close()
                except Exception as error:
                    result["degradations"].append("close:" + type(error).__name__)
        result = validate_result(manifest, result)
        _write_new(case_dir / "result.json", result)
        results.append(result)
        if result["status"] == "unknown":
            break
    summary = {"variant": variant, "results": results,
               "stopped_on_unknown": any(r["status"] == "unknown" for r in results)}
    _write_new(run_dir / "finished.json", summary)
    return summary


def report(output_dir: Path) -> dict:
    """汇总所有计划格；缺失、失败和中断保持可见，不补跑任务。"""
    frozen = verify_freeze(output_dir)
    results = []
    for variant in frozen["manifest"]["variants"]:
        directory = output_dir / "runs" / variant
        if directory.is_symlink():
            raise ValueError("EVAL_SYMLINK_RESULTS")
        for path in sorted(directory.glob("*/result.json")):
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError("EVAL_SYMLINK_RESULTS")
            results.append(_load(path))
    return summarize_results(frozen["manifest"], results)


def main(argv: list[str] | None = None, *, runtime_factory: Callable | None = None) -> int:
    """执行一个显式阶段，错误只回报诊断，不自动修订或追加实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("validate", "freeze"):
        command = sub.add_parser(stage)
        command.add_argument("--manifest", type=Path, required=True)
        if stage == "freeze":
            command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("run")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--variant", required=True)
    command.add_argument("--factory", help="已冻结的替身装配函数module:callable")
    command = sub.add_parser("report")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--write", type=Path, help="可选独占报告路径，限本批目录内")
    args = parser.parse_args(argv)
    try:
        if args.stage in {"validate", "freeze"}:
            if args.manifest.is_symlink():
                raise ValueError("EVAL_SYMLINK_MANIFEST")
            manifest = validate_manifest(_load(args.manifest), args.manifest.resolve().parent)
            result = ({"valid": True, "experiment_id": manifest["experiment_id"]}
                      if args.stage == "validate" else
                      freeze_manifest(manifest, args.manifest.resolve().parent, args.output))
        elif args.stage == "run":
            result = run_variant(args.output, args.variant, factory_spec=args.factory,
                                 runtime_factory=runtime_factory)
        else:
            result = report(args.output)
            if args.write:
                if ".." in args.write.parts or args.write.is_symlink() or any(p.is_symlink() for p in args.write.parents):
                    raise ValueError("EVAL_REPORT_PATH_ALIAS")
                target = args.write.resolve()
                if target == args.output.resolve() or not target.is_relative_to(args.output.resolve()):
                    raise ValueError("EVAL_REPORT_OUTSIDE_BATCH")
                _write_new(target, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"error": type(error).__name__, "reason": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
