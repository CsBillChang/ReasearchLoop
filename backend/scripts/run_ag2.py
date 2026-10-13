# AG2 有界验收脚本：分开记录预检、真实任务与最终读回；隔离验证不替代普通用户输入测试。
"""Explicit AG2 acceptance runner.

R0 is zero-model provider probing. R1/R2 call the running production API and
depend on a separately running Worker; this runner never implements another
agent loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from research_loop.config import get_settings
from research_loop.providers.discovery import DiscoveryProvider
from research_loop.providers.pdf import PDFProvider
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database


def run_r0(sources: list[str], output_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    discovery = DiscoveryProvider.from_settings(settings)
    try:
        end = date.today().isoformat()
        today = date.today()
        try:
            start = today.replace(year=today.year - 3).isoformat()
        except ValueError:
            start = today.replace(year=today.year - 3, day=28).isoformat()
        result = discovery.search("open vocabulary object detection", start, end, sources, 5)
        files = FileStore(output_dir.resolve() / "source-data", output_dir.resolve() / "unused-notes", settings.project_root)
        pdf = PDFProvider(files, timeout=60)
        checks = []
        seen = set()
        try:
            for candidate in result.candidates:
                if candidate.source in seen or not candidate.pdf_url:
                    continue
                seen.add(candidate.source)
                try:
                    document = pdf.ingest(candidate.pdf_url, candidate.paper_id)
                    pages = pdf.pages(document)
                    checks.append({"paper_id": candidate.paper_id, "source": candidate.source,
                                   "status": "parsed" if pages and pages[0].text.strip() else "no_text",
                                   "document": document.model_dump(mode="json"),
                                   "pdf_path": str(files.cache_pdf_path(document.content_sha256)) if document.total_pages is not None else None,
                                   "fetch": getattr(pdf, "last_fetch", {})})
                except Exception as exc:
                    checks.append({"paper_id": candidate.paper_id, "source": candidate.source,
                                   "status": "failed", "error": str(exc),
                                   "fetch": getattr(pdf, "last_fetch", {})})
                if len(checks) == 2:
                    break
        finally:
            pdf.close()
        pdf_ok = len(checks) == 2 and all(check["status"] == "parsed" for check in checks)
        return {"case": "R0", "model_requests": 0, "date_from": start, "date_to": end,
                "config": {"discovery_sources": sources, "citation_provider": None},
                "search": result.model_dump(mode="json"), "citation_capability": "unsupported_by_user_decision",
                "request_records": [item.request for item in result.source_results if item.request],
                "non_arxiv_candidates": sum(1 for item in result.candidates if item.source != "arxiv"),
                "status": "pass" if (len({item.source for item in result.candidates}) >= 2
                    and any(item.source != "arxiv" for item in result.candidates)
                    and pdf_ok) else "partial",
                "pdf_preflight": "passed" if pdf_ok else "partial", "pdf_checks": checks}
    finally:
        discovery.close()


def wait_task(client: httpx.Client, task_id: str, timeout: int = 900) -> dict[str, Any]:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        response = client.get(f"/api/tasks/{task_id}")
        response.raise_for_status()
        last = response.json()
        if last.get("status") in {"completed", "partial", "failed", "cancelled", "needs_input"}:
            return last
        time.sleep(2)
    raise TimeoutError(f"task did not finish: {task_id}; last={last.get('status')}")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_task(db: Database, task_id: str) -> dict[str, Any]:
    row = db.get_task(task_id)
    if not row:
        return {"task_id": task_id, "missing": True}
    result = json.loads(row["result_json"]) if row.get("result_json") else None
    draft = db.get_draft(result.get("draft_id")) if result and result.get("draft_id") else None
    receipt = db.get_receipt(draft["draft_id"], draft["content_hash"]) if draft else None
    file_hash = _sha256(Path(receipt["path"])) if receipt and receipt.get("path") else None
    calls = []
    for call in db.list_model_calls(task_id):
        calls.append({"sequence": call["sequence_no"], "status": call["status"],
                      "request_id": call.get("request_id"), "request_hash": call.get("request_hash"),
                      "sent_state": call.get("sent_state"), "finish_reason": call.get("finish_reason"),
                      "usage": json.loads(call.get("usage_json") or "{}")})
    return {"task": {"task_id": row["task_id"], "status": row["status"], "save_policy": row["save_policy"],
                      "error": row.get("error"), "parent_task_id": row.get("parent_task_id"),
                      "root_task_id": row.get("root_task_id")},
            "events": db.list_events(task_id), "actions": db.list_actions(task_id),
            "subtasks": db.list_subtasks(task_id), "task_papers": db.list_task_papers(task_id),
            "task_evidence": db.list_task_evidence(task_id), "model_calls": calls,
            "checkpoint": {"message_roles": [item.get("role") for item in (db.get_checkpoint(task_id) or {}).get("messages", [])],
                           "message_count": len((db.get_checkpoint(task_id) or {}).get("messages", []))},
            "draft": ({"draft_id": draft["draft_id"], "content_hash": draft["content_hash"],
                       "status": draft["status"], "evidence_ids": draft["evidence_ids"]} if draft else None),
            "children": [inspect_task(db, child["subtask_id"]) for child in db.list_subtasks(task_id)],
            "save_receipt": ({"receipt_id": receipt["receipt_id"], "path": receipt["path"],
                              "status": receipt["status"], "file_exists": file_hash is not None,
                              "file_sha256": file_hash} if receipt else None)}


def run_api_case(case: str, client: httpx.Client, db: Database, prompt: str, save_policy: str,
                 idempotency_key: str, clarification_answer: str | None = None,
                 attachments: list[dict[str, Any]] | None = None,
                 expected_input_refs: list[str] | None = None) -> dict[str, Any]:
    payload = {"prompt": prompt, "save_policy": save_policy, "idempotency_key": idempotency_key,
               "attachments": attachments or []}
    created = client.post("/api/tasks", json=payload)
    created.raise_for_status()
    task_id = created.json()["task_id"]
    initial = wait_task(client, task_id)
    clarification = initial.get("clarification")
    resume_response = None
    task = initial
    if task.get("status") == "needs_input" and clarification and clarification_answer is not None:
        resume = client.post(f"/api/tasks/{task_id}/resume", json={
            "clarification_id": clarification["clarification_id"],
            "answer": clarification_answer, "idempotency_key": idempotency_key + "-resume"})
        resume.raise_for_status()
        resume_response = resume.json()
        task = wait_task(client, task_id)
    evidence = inspect_task(db, task_id)
    checks = {"task_id_stable": task.get("task_id") == task_id,
              "actions_recorded": bool(evidence["actions"]),
              "model_receipts_recorded": bool(evidence["model_calls"]),
              "subtasks_recorded": bool(evidence["subtasks"]),
              "expected_input_refs_authorized": all(db.task_has_input(task_id, ref) for ref in (expected_input_refs or []))}
    return {"case": case, "task": task, "clarification_before_resume": clarification,
            "resume_response": resume_response, "evidence": evidence, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["r0", "r1", "r2", "all"], default="all")
    parser.add_argument("--api-url", default="http://127.0.0.1:18082")
    parser.add_argument("--paper-id-a")
    parser.add_argument("--paper-id-b")
    parser.add_argument("--pdf-a")
    parser.add_argument("--pdf-b")
    parser.add_argument("--sources", nargs="+", default=["arxiv", "eccv", "ijcai"])
    parser.add_argument("--output-dir", type=Path, required=True, help="new evidence directory; old evidence is never overwritten")
    parser.add_argument("--preflight", type=Path, help="successful R0 report including verified PDF availability for R1")
    args = parser.parse_args()
    if os.getenv("RESEARCHLOOP_RUN_AG2_REAL") != "1":
        raise SystemExit("set RESEARCHLOOP_RUN_AG2_REAL=1 for explicit AG2 real acceptance")
    target = args.output_dir / "run-ag2.json"
    if target.exists():
        raise SystemExit("output already exists; preserve old evidence and choose a new directory")
    if args.case in {"r1", "all"}:
        if not args.preflight:
            raise SystemExit("R1 requires --preflight: source/PDF checks must pass before model calls")
        preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
        if "cases" in preflight:
            preflight = next((case for case in preflight["cases"] if case.get("case") == "R0"), {})
        if preflight.get("status") != "pass" or preflight.get("pdf_preflight") != "passed":
            raise SystemExit("R0 source/PDF preflight is not passed; no model calls")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from uuid import uuid4
    batch_key = "ag2-" + uuid4().hex
    settings = get_settings()
    db = Database(settings.database_url)
    db.ensure_schema()
    report: dict[str, Any] = {"date": date.today().isoformat(), "config": {"model": settings.deepseek_model,
             "database": settings.database_url.rsplit("@", 1)[-1], "output": str(args.output_dir),
             "max_model_requests": settings.max_model_requests, "max_tool_actions": settings.max_tool_actions}, "cases": []}
    if args.case in {"r0", "all"}:
        report["cases"].append(run_r0(args.sources, args.output_dir))
    if args.case in {"r1", "all"}:
        with httpx.Client(base_url=args.api_url, timeout=30.0) as client:
            report["cases"].append(run_api_case(
                "R1", client, db,
                "找一下近三年关于开放词汇目标检测的高被引文献，比较它们如何利用语言信息，并保存一份简短比较笔记。",
                "allow", batch_key + "-r1", "本次先比较2篇"))
    if args.case in {"r2", "all"}:
        if not args.paper_id_a or not args.paper_id_b:
            raise SystemExit("R2 requires --paper-id-a and --paper-id-b")
        if not args.pdf_a or not args.pdf_b:
            raise SystemExit("R2 requires --pdf-a and --pdf-b as authorized same-paper inputs")
        for path in (args.pdf_a, args.pdf_b):
            if not Path(path).is_file() or Path(path).suffix.lower() != ".pdf":
                raise SystemExit("R2 PDF input must be an existing local PDF")
        attachments = [{"input_ref": "r2-paper-a", "local_path": str(Path(args.pdf_a).resolve()),
                        "filename": Path(args.pdf_a).name, "paper_id": args.paper_id_a},
                       {"input_ref": "r2-paper-b", "local_path": str(Path(args.pdf_b).resolve()),
                        "filename": Path(args.pdf_b).name, "paper_id": args.paper_id_b}]
        with httpx.Client(base_url=args.api_url, timeout=30.0) as client:
            report["cases"].append(run_api_case(
                "R2", client, db,
                "只读取已授权输入 r2-paper-a 和 r2-paper-b 对应的固定论文物理第1页，分别说明实验使用的 GPU 驱动版本。禁止保存；证据不足时交付明确标注的 partial。",
                "deny", batch_key + "-r2", attachments=attachments,
                expected_input_refs=["r2-paper-a", "r2-paper-b"]))
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
