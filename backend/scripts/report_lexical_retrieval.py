"""独立读取冻结标签与实验原件，复算两条路线的质量、时延和失败。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

from scripts.retrieval_experiment_guard import write_bytes_new, write_json_new


ROUTES = ("bm25_rerank", "located_bm25_rerank")


def read_frozen(ref: dict):
    """标签仅在独立报告阶段读取，必须与执行前冻结身份一致。"""
    data = Path(ref["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("REPORT_INPUT_HASH_CHANGED")
    return json.loads(data)


def percentile(values: list[float], p: float) -> float | None:
    """按(n-1)p线性插值，与历史描述统计定义保持一致。"""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lower = math.floor(pos)
    return ordered[lower] + (ordered[math.ceil(pos)] - ordered[lower]) * (pos - lower)


def quality(ids: list[str], candidates: list[str], gold: set[str]) -> dict:
    """对单题计算已标正例覆盖，未标片段不自动作为负例。"""
    if not gold:
        return {"recall5": None, "recall8": None, "recall10": None,
                "candidate_recall24": None, "mrr24": None, "first_gold_rank": None}
    first = next((i for i, cid in enumerate(ids, 1) if cid in gold), None)
    return {**{f"recall{k}": len(set(ids[:k]) & gold)/len(gold) for k in (5, 8, 10)},
            "candidate_recall24": len(set(candidates) & gold)/len(gold),
            "mrr24": 1/first if first else 0, "first_gold_rank": first}


def build_report(freeze_path: Path, execution: Path, output: Path) -> dict:
    """保留全部计划格及缺项；不把未执行补零或筛选最好一次。"""
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("REPORT_OUTPUT_MUST_BE_EMPTY")
    frozen = json.loads(freeze_path.read_text())
    questions = read_frozen(frozen["inputs"]["original_questions"])
    by_question = {q["case_id"]: q for q in questions}
    gold = defaultdict(set)
    for pair in read_frozen(frozen["inputs"]["labels"])["positive_pairs"]:
        gold[pair["case_id"]].add(pair["chunk_id"])
    terminal = json.loads((execution/"terminal.json").read_text())
    raw, references = {}, []
    for path in sorted(execution.glob("hot-*.json")):
        data = path.read_bytes(); row = json.loads(data)
        key = (row["case_id"], row["route"], row["repetition"])
        if key in raw:
            raise ValueError("DUPLICATE_EXPERIMENT_CELL")
        result = row["result"]
        if (result["query"] != by_question[key[0]]["query"]
                or result["scope_document_ids"] != by_question[key[0]]["scope_document_ids"]
                or result["ranked_ids"] != [h["chunk_id"] for h in result["hits"]]
                or set(result["ranked_ids"]) != set(result["candidate_ids"])):
            raise ValueError("RAW_QUERY_SCOPE_OR_RANK_CHANGED")
        raw[key] = row
        references.append({"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()})
    per_case, plan = [], []
    for q in questions:
        cid = q["case_id"]
        for route in ROUTES:
            repeats = [raw.get((cid, route, rep)) for rep in range(3)]
            for rep, row in enumerate(repeats):
                plan.append({"case_id": cid, "route": route, "repetition": rep,
                             "state": row["result"]["status"] if row else "unexecuted"})
            first = repeats[0]
            if first is None:
                continue
            result = first["result"]
            metrics = quality(result["ranked_ids"], result["candidate_ids"], gold[cid])
            pre = quality(result["candidate_ids"], result["candidate_ids"], gold[cid])
            available = [r["result"] for r in repeats if r]
            timing = {name: statistics.median(r["timing"][name] for r in available)
                      for name in ("wall_seconds", "selection_seconds", "reranking_seconds")}
            per_case.append({"case_id": cid, "route": route, "answerable": bool(gold[cid]),
                "split": q["split"], "question_type": q.get("question_type"),
                "paper_ids": q["paper_ids"], "query": q["query"], "status": result["status"],
                **metrics, "pre_rerank_quality": pre, "timing_median": timing,
                "hot_observed": len(available), "ranking_stable": all(
                    r["ranked_ids"] == result["ranked_ids"] for r in available),
                "all_repetitions_ce_succeeded": len(available) == 3 and all(
                    r["status"] == "executed" for r in available),
                "candidate_count": len(result["candidate_ids"]),
                "local_document_ids": result["selection_trace"]["local_document_ids"],
                "routing": result["selection_trace"]["trace"]["routing"],
                "candidate_ids": result["candidate_ids"], "ranked_ids": result["ranked_ids"]})
    fields = ("recall5", "recall8", "recall10", "mrr24", "candidate_recall24")
    aggregates = []
    for route in ROUTES:
        for split in ("all", "regression", "pending"):
            rows = [r for r in per_case if r["route"] == route and r["answerable"]
                    and (split == "all" or r["split"] == split)]
            aggregates.append({"route": route, "split": split, "n": len(rows),
                **{field: statistics.mean(r[field] for r in rows) if rows else None for field in fields},
                "pre_rerank": {field: statistics.mean(r["pre_rerank_quality"][field] for r in rows)
                               if rows else None for field in fields},
                "p50_seconds": percentile([r["timing_median"]["wall_seconds"] for r in rows], .5),
                "p95_seconds": percentile([r["timing_median"]["wall_seconds"] for r in rows], .95),
                "unstable_cases": sum(not r["ranking_stable"] for r in rows),
                "statuses": dict(Counter(r["status"] for r in rows))})
    lookup = {(r["case_id"], r["route"]): r for r in per_case}
    pairs = []
    for q in questions:
        cid = q["case_id"]
        if not gold[cid] or any((cid, route) not in lookup for route in ROUTES):
            continue
        a, b = [lookup[cid, route] for route in ROUTES]
        pairs.append({"case_id": cid, "split": q["split"],
                      **{field + "_delta": b[field]-a[field] for field in fields},
                      "a_first_gold_rank": a["first_gold_rank"], "b_first_gold_rank": b["first_gold_rank"],
                      "a_seconds": a["timing_median"]["wall_seconds"],
                      "b_seconds": b["timing_median"]["wall_seconds"],
                      "both_ce_succeeded": a["status"] == b["status"] == "executed",
                      "all_repetitions_ce_succeeded": a["all_repetitions_ce_succeeded"]
                          and b["all_repetitions_ce_succeeded"]})
    speed_pairs = [r for r in pairs if r["all_repetitions_ce_succeeded"]]
    summary = {"terminal": terminal, "planned_hot_rows": 684, "observed_hot_rows": len(raw),
               "unexecuted": sum(r["state"] == "unexecuted" for r in plan),
               "independently_certified": 0, "aggregates": aggregates,
               "paired_positive_cases": len(pairs),
               "mrr_changes": dict(Counter("increase" if r["mrr24_delta"] > 0 else
                                            "decrease" if r["mrr24_delta"] < 0 else "equal" for r in pairs)),
               "first_hot_statuses_all_queries": {route: dict(Counter(r["status"] for r in per_case
                    if r["route"] == route)) for route in ROUTES},
               "all_hot_statuses_by_route": {route: dict(Counter(r["result"]["status"]
                    for r in raw.values() if r["route"] == route)) for route in ROUTES},
               "all_hot_statuses_by_route_and_repetition": {
                   f"{route}:{rep}": dict(Counter(r["result"]["status"] for r in raw.values()
                       if r["route"] == route and r["repetition"] == rep))
                   for route in ROUTES for rep in range(3)},
               "paired_complete_ce_timing": {"n": len(speed_pairs), "planned_positive_pairs": 112,
                   "a_p50": percentile([r["a_seconds"] for r in speed_pairs], .5),
                   "a_p95": percentile([r["a_seconds"] for r in speed_pairs], .95),
                   "b_p50": percentile([r["b_seconds"] for r in speed_pairs], .5),
                   "b_p95": percentile([r["b_seconds"] for r in speed_pairs], .95)}}
    write_json_new(output/"summary.json", summary)
    write_json_new(output/"per-case.json", per_case)
    write_json_new(output/"paired-cases.json", pairs)
    write_json_new(output/"planned-cells.json", plan)
    write_json_new(output/"raw-references.json", references)
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(pairs[0]) if pairs else ["case_id"])
    writer.writeheader(); writer.writerows(pairs)
    write_bytes_new(output/"paired-cases.csv", b"\xef\xbb\xbf" + stream.getvalue().encode())
    text = ["# 两条词法检索路线同批比较", "", f"终态：{terminal['status']}；热运行 {len(raw)}/684。", "",
            "质量按原非穷尽标签、首热轮计算。题目均有历史暴露，认证独立题0。", "",
            "| 路线/分组 | n | Recall@5 | Recall@8 | Recall@10 | MRR@24 | 候选覆盖@24 | 热P50秒 | 热P95秒 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        values = ["N/A" if row[k] is None else f"{row[k]*100:.2f}%" for k in ("recall5", "recall8", "recall10")]
        values += ["N/A" if row["mrr24"] is None else f"{row['mrr24']:.4f}",
                   "N/A" if row["candidate_recall24"] is None else f"{row['candidate_recall24']*100:.2f}%"]
        values += ["N/A" if row[k] is None else f"{row[k]:.4f}" for k in ("p50_seconds", "p95_seconds")]
        text.append(f"| {row['route']}/{row['split']} | {row['n']} | " + " | ".join(values) + " |")
    text += ["", "该时间为本轮纯文件组件边界，不能与旧SQL/Qdrant装配时间直接相减称提速。",
             "失败、空候选与降级保留；未执行不补零。缺证题不进入正例分母。",
             "", "[逐题结果](per-case.json) · [配对差值](paired-cases.csv) · [汇总与终态](summary.json)"]
    write_bytes_new(output/"report.md", ("\n".join(text) + "\n").encode())
    return summary


def main():
    """显式读取执行目录，不装配任何模型或服务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.freeze.resolve(), args.execution.resolve(), args.output.resolve())
    print(json.dumps({"observed": result["observed_hot_rows"], "unexecuted": result["unexecuted"]}))


if __name__ == "__main__":
    main()
