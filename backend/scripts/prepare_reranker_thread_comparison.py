"""准备固定输入的CPU线程对照；只选择、编码和冻结，不提供模型执行入口。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

from research_loop.domain.lexical_retrieval import LexicalRetriever
from scripts.compare_lexical_retrieval import digest, load_frozen
from scripts.retrieval_experiment_guard import BoundaryViolation, install_guards, write_json_new


ROUTE = "located_bm25_rerank"
SOURCE_FIELDS = ("chunk_id", "document_id", "evidence_id", "index_id", "source_sha256",
                 "page_number", "char_start", "char_end", "text_sha256", "text")
PROJECTION_WARNING = (
    "Prepared tensors are a new common projection for both thread configurations. "
    "The historical run did not save feed tensors; historical byte equality is not claimed. "
    "A future executor must validate actual Session input names before inference."
)


def canonical_digest(value: object) -> str:
    """对可移植JSON投影计算身份，不依赖字典插入顺序。"""
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def verify_input_hashes(frozen: dict) -> None:
    """逐个核输入字节身份；标签也只散列，绝不解析标签JSON。"""
    for ref in frozen["inputs"].values():
        if digest(Path(ref["path"])) != ref["sha256"]:
            raise BoundaryViolation("FROZEN_INPUT_CHANGED")


def verify_source_anchors(source_freeze: Path, source_manifest: Path,
                          expected_source_freeze_sha256: str,
                          expected_source_manifest_sha256: str) -> None:
    """先核独立事前锚点，阻止整份自洽冻结或清单被替换后扩大读取范围。"""
    for path, expected in ((source_freeze, expected_source_freeze_sha256),
                           (source_manifest, expected_source_manifest_sha256)):
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise BoundaryViolation("SOURCE_ANCHOR_HASH_INVALID")
        if digest(path) != expected:
            raise BoundaryViolation("SOURCE_ANCHOR_HASH_CHANGED")


def load_source_rows(source_results: Path, source_manifest: Path, questions: list[dict]) -> dict:
    """按旧工件清单核对首热B原件，只提取后续核对需要的身份字段。"""
    results = source_results.resolve()
    entries = json.loads(source_manifest.read_text())
    expected_ids = [q["case_id"] for q in questions]
    if len(set(expected_ids)) != len(expected_ids):
        raise BoundaryViolation("DUPLICATE_QUERY_ID")
    references = []
    seen_paths = set()
    for ref in entries:
        relative = Path(ref["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise BoundaryViolation("SOURCE_MANIFEST_PATH_INVALID")
        if relative.name.startswith("hot-0-") and relative.name.endswith("-" + ROUTE + ".json"):
            candidate = (results.parent / relative).resolve()
            if candidate.parent != results or candidate in seen_paths:
                raise BoundaryViolation("SOURCE_MANIFEST_PATH_INVALID")
            seen_paths.add(candidate)
            references.append((candidate, ref["sha256"]))
    if len(references) != len(expected_ids):
        raise BoundaryViolation("HISTORICAL_CASE_COUNT_CHANGED")
    rows = {}
    for path, expected_hash in references:
        if digest(path) != expected_hash:
            raise BoundaryViolation("HISTORICAL_RAW_HASH_CHANGED")
        wrapper = json.loads(path.read_text())
        case_id = wrapper.get("case_id")
        if (wrapper.get("phase") != "hot" or wrapper.get("repetition") != 0
                or wrapper.get("route") != ROUTE or case_id not in expected_ids
                or case_id in rows):
            raise BoundaryViolation("HISTORICAL_CELL_IDENTITY_CHANGED")
        result = wrapper["result"]
        if result.get("status") != "executed" or result.get("effective_path") != ROUTE:
            raise BoundaryViolation("HISTORICAL_B_NOT_SUCCESSFUL")
        # 排名和分数不进入准备投影；此处只从旧hits取片段身份。
        hits = [{k: h[k] for k in ("chunk_id", "document_id", "text_sha256")}
                for h in result["hits"]]
        rows[case_id] = {"query": result["query"],
                         "scope_document_ids": result["scope_document_ids"],
                         "candidate_ids": result["candidate_ids"], "hits": hits,
                         "model_identity": result["model_identity"],
                         "source_reference": {"path": str(path), "sha256": expected_hash}}
    if set(rows) != set(expected_ids):
        raise BoundaryViolation("HISTORICAL_CASE_SET_CHANGED")
    return rows


def project_case(question: dict, selected: dict, historical: dict) -> dict:
    """核对原问题、授权范围、候选原序与原文，输出无标签的白名单输入。"""
    if set(question) != {"case_id", "query", "scope_document_ids"}:
        raise BoundaryViolation("QUERY_PROJECTION_CONTAINS_EXTRA_FIELDS")
    scope = question["scope_document_ids"]
    if (not scope or len(scope) != len(set(scope)) or not question["query"]
            or historical["query"] != question["query"]
            or historical["scope_document_ids"] != scope):
        raise BoundaryViolation("QUERY_OR_SCOPE_CHANGED")
    candidates = selected["candidates"]
    ids = [row["chunk_id"] for row in candidates]
    if not 1 <= len(ids) <= 24 or len(set(ids)) != len(ids):
        raise BoundaryViolation("CANDIDATE_CONTRACT_INVALID")
    if ids != historical["candidate_ids"]:
        raise BoundaryViolation("CANDIDATE_ORDER_CHANGED")
    old_hits = {h["chunk_id"]: h for h in historical["hits"]}
    if len(old_hits) != len(historical["hits"]) or set(old_hits) != set(ids):
        raise BoundaryViolation("HISTORICAL_CANDIDATE_IDENTITY_CHANGED")
    projected = []
    for row in candidates:
        if any(key not in row for key in SOURCE_FIELDS):
            raise BoundaryViolation("CANDIDATE_SOURCE_FIELDS_MISSING")
        old = old_hits[row["chunk_id"]]
        if (row["document_id"] not in scope or row["document_id"] != old["document_id"]
                or not isinstance(row["text"], str)
                or hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]
                or row["text_sha256"] != old["text_sha256"]):
            raise BoundaryViolation("CANDIDATE_TEXT_OR_SCOPE_CHANGED")
        projected.append({k: row[k] for k in SOURCE_FIELDS})
    result = {"case_id": question["case_id"], "query": question["query"],
              "scope_document_ids": list(scope), "candidates": projected}
    return {**result, "input_sha256": canonical_digest(result)}


def tensor_reference(rows: list[list[int]]) -> dict:
    """固定为little-endian int64字节，避免依赖NumPy或模型运行时。"""
    flat = [value for row in rows for value in row]
    return {"dtype": "int64", "byte_order": "little", "shape": [len(rows), len(rows[0])],
            "sha256": hashlib.sha256(struct.pack("<" + "q" * len(flat), *flat)).hexdigest()}


def build_feed_projection(case: dict, encoder) -> dict:
    """沿用原顺序batch4配方，冻结编码、补齐和掩码而不执行模型。"""
    encoded = [list(encoder.encode(case["query"], row["text"]).ids)
               for row in case["candidates"]]
    if any(not ids or len(ids) > 512 for ids in encoded):
        raise BoundaryViolation("RERANK_INPUT_TOO_LONG_OR_EMPTY")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for ids in encoded for value in ids):
        raise BoundaryViolation("TOKEN_IDS_INVALID")
    batches = []
    for start in range(0, len(encoded), 4):
        group = encoded[start:start + 4]
        size = max(map(len, group))
        ids = [row + [1] * (size - len(row)) for row in group]
        masks = [[1] * len(row) + [0] * (size - len(row)) for row in group]
        types = [[0] * size for _ in group]
        tensors = {"input_ids": tensor_reference(ids), "attention_mask": tensor_reference(masks),
                   "token_type_ids_if_required": tensor_reference(types)}
        batches.append({"candidate_indices": list(range(start, start + len(group))),
                        "unpadded_lengths": list(map(len, group)), "tensors": tensors,
                        "projection_sha256": canonical_digest(tensors)})
    return {"case_id": case["case_id"], "input_sha256": case["input_sha256"],
            "pairs": len(encoded), "input_tokens": sum(map(len, encoded)),
            "padded_tokens": sum(b["tensors"]["input_ids"]["shape"][0]
                                 * b["tensors"]["input_ids"]["shape"][1] for b in batches),
            "batch_size": 4, "sort_by_length": False, "max_pair_tokens": 512,
            "padding_id": 1, "batches": batches}


def build_execution_plan(cases: list[dict]) -> dict:
    """只列出尚未执行的固定格和预算，不授予真实推理权限。"""
    ids = [case["case_id"] for case in cases]
    if not ids or len(ids) != len(set(ids)):
        raise BoundaryViolation("PLAN_CASES_INVALID")
    cells = [{"phase": "warmup", "case_id": ids[0], "threads": threads,
              "state": "not_executed"} for threads in (2, 4)]
    for repetition in range(3):
        for ordinal, case_id in enumerate(ids):
            order = (2, 4) if (repetition + ordinal) % 2 == 0 else (4, 2)
            cells.extend({"phase": "hot", "repetition": repetition, "ordinal": ordinal,
                          "case_id": case_id, "threads": threads, "state": "not_executed"}
                         for threads in order)
    pairs = 6 * sum(len(case["candidates"]) for case in cases) + 2 * len(cases[0]["candidates"])
    if pairs > 16134 or len(cells) > 686:
        raise BoundaryViolation("PLAN_BUDGET_EXCEEDED")
    return {"schema": "reranker-thread-plan-v1", "execution_authorized_by_preparation": False,
            "settings": {"threads": [2, 4], "provider": "CPUExecutionProvider", "batch_size": 4,
                         "sort_by_length": False, "hot_repeats": 3, "max_pair_tokens": 512},
            "budget": {"max_score_attempts": 686, "max_pairs": 16134, "max_seconds": 3600,
                       "exact_planned_score_attempts": len(cells), "exact_planned_pairs": pairs},
            "timing": {"primary": "reranker_wall_seconds", "secondary": "inference_seconds",
                       "included": ["tokenization", "batching", "model_scoring", "output_validation"],
                       "excluded": ["candidate_selection", "file_reads", "file_writes", "model_load"],
                       "comparison": "same-run 2-thread versus 4-thread reranker component only"},
            "quality": {"first_hot_only": True, "labels_loaded_by_executor": False,
                        "metrics": ["Recall@5", "Recall@8", "Recall@10", "MRR@24"],
                        "score_and_rank_changes_required": True},
            "cells": cells, "score_attempts": 0, "pair_attempts": 0}


def check_model_identity(historical: dict, spec: dict, tokenizer_hash: str,
                         onnxruntime_version: str) -> None:
    """准备前核对旧记录所用模型，不能混入另一权重或组批基线。"""
    identity = historical["model_identity"]
    expected = {"model_id": spec["repo"], "revision": spec["revision"],
                "weights_sha256": spec["files"][spec["weights"]]["sha256"],
                "tokenizer_sha256": tokenizer_hash, "max_pair_tokens": 512,
                "batch_size": 4, "threads": 2, "sort_by_length": False,
                "onnxruntime": onnxruntime_version, "score_kind": "raw_relevance_logit"}
    if any(identity.get(k) != v for k, v in expected.items()):
        raise BoundaryViolation("HISTORICAL_MODEL_IDENTITY_CHANGED")


def prepare(source_freeze: Path, source_results: Path, source_manifest: Path, output: Path, *,
            expected_source_freeze_sha256: str, expected_source_manifest_sha256: str) -> dict:
    """在进程守卫下完成准备；无推理Session、评分或业务服务路径。"""
    source_freeze, source_results, source_manifest = (
        p.resolve() for p in (source_freeze, source_results, source_manifest))
    if not output.is_absolute() or output != output.resolve():
        raise ValueError("OUTPUT_MUST_BE_ABSOLUTE_REAL_DIRECTORY")
    verify_source_anchors(source_freeze, source_manifest, expected_source_freeze_sha256,
                          expected_source_manifest_sha256)
    preliminary = json.loads(source_freeze.read_text())
    references = (list(preliminary["inputs"].values()) + preliminary["source_files"]
                  + preliminary["protected_input_files"])
    roots = [Path(ref["path"]).resolve() for ref in references]
    roots.extend([source_freeze, source_results, source_manifest, Path(__file__).resolve().parents[1]])
    guards = install_guards(output, roots)
    error, status = None, "prepared"
    try:
        frozen, chunks, catalog, questions = load_frozen(source_freeze)
        verify_input_hashes(frozen)
        old_rows = load_source_rows(source_results, source_manifest, questions)
        model_manifest = json.loads(Path(frozen["inputs"]["model_manifest"]["path"]).read_text())
        spec = model_manifest["reranker"]
        tokenizer_path = Path(spec["directory"]) / "tokenizer.json"
        tokenizer_hash = digest(tokenizer_path)
        if tokenizer_hash != spec["files"]["tokenizer.json"]["sha256"]:
            raise BoundaryViolation("TOKENIZER_HASH_CHANGED")
        from tokenizers import Tokenizer
        encoder = Tokenizer.from_file(str(tokenizer_path))
        encoder.no_truncation()
        encoder.no_padding()
        selector = LexicalRetriever(chunks, catalog)
        cases, feeds = [], []
        for question in questions:
            historical = old_rows[question["case_id"]]
            check_model_identity(historical, spec, tokenizer_hash, frozen["packages"]["onnxruntime"])
            selected = selector.select(question["query"], question["scope_document_ids"], ROUTE)
            case = project_case(question, selected, historical)
            cases.append(case)
            feeds.append(build_feed_projection(case, encoder))
        plan = build_execution_plan(cases)
        if (len(cases), sum(len(c["candidates"]) for c in cases),
                plan["budget"]["exact_planned_pairs"]) != (114, 2681, 16134):
            raise BoundaryViolation("FIXED_B_COUNTS_CHANGED")
        qualification = {
            "status": "prepared_not_executed", "cases": len(cases), "pairs_per_route_repeat": 2681,
            "score_attempts": 0, "pair_attempts": 0, "model_loaded": False,
            "labels_parsed": False, "ort_imported": "onnxruntime" in sys.modules,
            "cases_sha256": canonical_digest(cases), "feeds_sha256": canonical_digest(feeds),
            "source_freeze": {"path": str(source_freeze), "sha256": digest(source_freeze)},
            "source_manifest": {"path": str(source_manifest), "sha256": digest(source_manifest)},
            "independent_source_anchors": {
                "source_freeze_sha256": expected_source_freeze_sha256,
                "source_manifest_sha256": expected_source_manifest_sha256},
            "historical_raw_references": [old_rows[q["case_id"]]["source_reference"] for q in questions],
            "warning": PROJECTION_WARNING,
            "model": {"repo": spec["repo"], "revision": spec["revision"],
                      "weights_sha256": spec["files"][spec["weights"]]["sha256"],
                      "tokenizer_sha256": tokenizer_hash},
        }
        if qualification["ort_imported"]:
            raise BoundaryViolation("ORT_IMPORTED_DURING_PREPARATION")
        # 校验全部完成后才发布输入包，资格失败只保存终态，不留下半份可执行计划。
        write_json_new(output / "cases.json", cases)
        write_json_new(output / "feed-manifest.json", {"warning": PROJECTION_WARNING, "cases": feeds})
        write_json_new(output / "execution-plan.json", plan)
        write_json_new(output / "qualification.json", qualification)
    except BaseException as exc:
        status, error = "stopped", {"type": type(exc).__name__, "reason": str(exc)}
    terminal = {"status": status, "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": error, "score_attempts": 0, "pair_attempts": 0,
                "model_loaded": False, "guard_counts": guards}
    write_json_new(output / "terminal.json", terminal)
    return terminal


def main() -> int:
    """CLI只提供准备操作，无法通过开关启动真实推理。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-freeze-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source_freeze, args.source_results, args.source_manifest, args.output,
                     expected_source_freeze_sha256=args.expected_source_freeze_sha256,
                     expected_source_manifest_sha256=args.expected_source_manifest_sha256)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "prepared" else 1


if __name__ == "__main__":
    sys.exit(main())
