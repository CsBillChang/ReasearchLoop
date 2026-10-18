"""AG2三个场景的一次有界真实验证：隔离API/Worker/数据库，禁止重放已启动场景。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit

import httpx

from research_loop.config import Settings
from research_loop.providers.paper_acquisition import PaperAcquisitionError
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from scripts.run_reading_regression import snapshot

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/ag2-adaptive-scenarios-20261009"
DATABASE = "researchloop_test_ag2_adaptive_live_20261009"


def write(path: Path, value: object) -> None:
    """只写验收目录的非凭据数据，不导出Settings或环境变量。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def sha(path: Path) -> str:
    """读取内容身份，不更改旧PDF或旧证据。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config() -> tuple[Settings, Settings]:
    """真实供应商配置保持原样，数据库和产物目录强制隔离。"""
    production = Settings.from_env(ROOT)
    url = urlsplit(production.database_url)
    isolated = production.model_copy(update={
        "database_url": urlunsplit((url.scheme, url.netloc, "/" + DATABASE, "", "")),
        "data_dir": OUT / "runtime", "output_dir": OUT / "notes",
        "max_model_requests": 16, "max_tool_actions": 40, "reserve_model_requests": 4})
    return production, isolated


def prepare() -> None:
    """冻结输入、业务源码和现有PDF身份；本步骤不发模型请求。"""
    if (OUT / "freeze.json").exists():
        raise RuntimeError("already frozen; inspect existing evidence")
    production, settings = config()
    db = Database(settings.database_url)
    db.ensure_schema()
    assert db.fetchone("SELECT COUNT(*) AS n FROM tasks")["n"] == 0
    previous = ROOT / "artifacts/ag2-current-e2e-20261009"
    pdfs = json.loads((previous / "preflight.json").read_text())["papers"]
    attachments = json.loads((previous / "input.json").read_text())["attachments"]
    for entry in pdfs:
        assert sha(Path(entry["pdf_path"])) == entry["sha256"]
        db.save_paper(entry["paper_id"], entry["original_search_metadata"])
    inputs = {
        "S": {"prompt": "找3篇CVPR 2024关于开放词汇目标检测的论文，给出标题、链接和简短相关理由。只需要候选推荐，不要精读、审校或保存。", "save_policy": "deny"},
        "C": {"prompt": "基于附件两篇论文，写一份简短综述，比较语言信息如何表示、如何进入检测器，以及各自的适用边界。请自主分派定向研究，遇到缺口判断能否补足，最后说明抽样范围并保存到项目。不要八点逐篇精读，不要额外审校。", "save_policy": "allow", "attachments": attachments},
        "P": {"prompt": "仅基于附件这篇论文，说明它如何使用语言信息，并查明实验机器使用的GPU驱动精确版本。找不到就保留已有发现，明确说明未完成的部分，不要联网找其他资料，也不要保存。", "save_policy": "deny", "attachments": attachments[:1]},
    }
    for name, payload in inputs.items():
        write(OUT / name / "input.json", {**payload, "idempotency_key": "adaptive-20261009-" + name})
    write(OUT / "production-before.json", snapshot(Database(production.database_url)))
    write(OUT / "freeze.json", {"source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "backend/research_loop").rglob("*.py")},
        "runner_sha256": sha(Path(__file__)), "model": settings.deepseek_model, "database": DATABASE,
        "max_total_requests": 48, "max_requests_per_case": 16, "tool_limit_per_case": 40,
        "cases": ["S", "C", "P"], "pdfs": pdfs,
        "controlled_injection": "C only: first Researcher accessing GUIDED gets a transient acquisition failure for that child; a newly delegated child can read the same real PDF. This is not a real upstream outage.",
        "boundaries": "S uses live 论文; C/P use unchanged downloaded real PDFs. Real model decisions, no model replies or summaries replayed."})


def prepare_correction() -> None:
    """纠正首个C的矛盾故障合同；保留原批次，冻结唯一替代场景，不扩容总账本。"""
    if (OUT / "freeze-correction.json").exists():
        raise RuntimeError("correction already frozen")
    _, settings = config()
    db = Database(settings.database_url)
    count = db.fetchone("SELECT COUNT(*) AS n FROM model_calls")["n"]
    assert count <= 32 and not db.fetchone("SELECT 1 FROM model_calls WHERE status IN ('started','unknown') LIMIT 1")
    prior = json.loads((OUT / "C/input.json").read_text())
    backup = {**prior["attachments"][1], "input_ref": "guided-backup", "filename": "guided-same-version-backup.pdf"}
    payload = {**prior, "idempotency_key": "adaptive-20261009-C-corrected",
        "attachments": prior["attachments"] + [backup],
        "prompt": "基于主附件yolo-world和guided两篇论文，写一份简短综述，比较语言信息如何表示、如何进入检测器，以及各自的适用边界。先只委派主附件；若某篇实际获取失败，再将根任务已授权的同版本备用附件guided-backup按需分派给新子任务补足，不把备用文件算成第三篇。最终说明抽样范围并保存到项目。其他步骤由你自主决定；不做八点精读，不额外审校。"}
    write(OUT / "C2/input.json", payload)
    write(OUT / "freeze-correction.json", {"source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "backend/research_loop").rglob("*.py")},
        "runner_sha256": sha(Path(__file__)), "model": settings.deepseek_model, "database": DATABASE,
        "requests_already_spent": count, "max_total_requests": 48, "max_requests_this_case": 16,
        "reason": "C initial test incorrectly made the same source nonretryable while expecting its retry. C2 provides an explicitly authorized alternate input_ref; primary remains nonretryable.",
        "controlled": True, "real_pdf_and_model": True, "repeat_of_initial_task": False,
        "post_batch_code_fixes": ["screen evidence_id clarification", "retain findings after premature short stop", "honor selected final answer rather than old draft"]})


def worker(case: str, task_id: str) -> None:
    """定向执行唯一新任务；C故障注入只位于验收脚本，不改生产工具。"""
    _, settings = config()
    executor = TaskExecutor(settings)
    if case not in {"C", "C2"}:
        executor.execute(task_id, "adaptive-" + case)
        return
    blocked_child = None
    original = PaperService.read
    guided = json.loads((OUT / "C/input.json").read_text())["attachments"][1]

    def controlled_read(service, task_id, target_ref, *args, **kwargs):
        nonlocal blocked_child
        if case == "C2" and target_ref in {guided["input_ref"], guided["paper_id"]}:
            write(OUT / "C2/injection.json", {"blocked_child": task_id, "target": target_ref,
                  "controlled": True, "alternate_authorized_ref": "guided-backup"})
            raise PaperAcquisitionError("CONTROLLED_PRIMARY_UNAVAILABLE", "主附件guided获取失败且不可重试。根任务已授权同版本备用input_ref=guided-backup；当前子任务不可越过分派范围，请将缺口和备用定位交回Supervisor，由其决定是否重新分派备用附件。", retryable=False)
        if case == "C2":
            return original(service, task_id, target_ref, *args, **kwargs)
        if target_ref in {guided["input_ref"], guided["paper_id"]}:
            task = service.db.get_task(task_id)
            if task.get("parent_task_id") and blocked_child is None:
                blocked_child = task_id
                write(OUT / "C/injection.json", {"blocked_child": task_id, "target": target_ref,
                      "controlled": True, "kind": "temporary acquisition failure"})
            if task_id == blocked_child:
                raise PaperAcquisitionError("CONTROLLED_TRANSIENT_SOURCE", "本次子任务未取得正文：临时获取失败。保留已知定位及缺口，不在本子任务重复同一下载。", retryable=False)
        return original(service, task_id, target_ref, *args, **kwargs)

    with patch.object(PaperService, "read", controlled_read):
        executor.execute(task_id, "adaptive-C")


def run(case: str) -> None:
    """启动隔离HTTP服务和定向Worker；每个场景只能启动一次，结束后不自动重试。"""
    production, settings = config()
    frozen = json.loads((OUT / ("freeze-correction.json" if case == "C2" else "freeze.json")).read_text())
    assert all(sha(ROOT / p) == h for p, h in frozen["source_hashes"].items())
    assert sha(Path(__file__)) == frozen["runner_sha256"]
    if (OUT / "STOP.json").exists():
        raise RuntimeError("batch stopped; no further model calls")
    directory = OUT / case
    with (directory / "started.json").open("x") as f:
        json.dump({"time": time.time(), "case": case}, f)
    db = Database(settings.database_url)
    env = {**os.environ, "DATABASE_URL": settings.database_url,
           "RESEARCHLOOP_DATA_DIR": str(settings.data_dir), "RESEARCHLOOP_OUTPUT_DIR": str(settings.output_dir),
           "PYTHONPATH": str(ROOT / "backend")}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    api_log = (directory / "api.log").open("w")
    worker_log = (directory / "worker.log").open("w")
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "research_loop.main:app", "--host", "127.0.0.1", "--port", str(port)],
                           cwd=ROOT, env=env, stdout=api_log, stderr=subprocess.STDOUT)
    process = None
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False) as client:
            for _ in range(80):
                try:
                    if client.get("/api/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.2)
            else:
                raise RuntimeError("isolated API not ready")
            payload = json.loads((directory / "input.json").read_text())
            response = client.post("/api/tasks", json=payload)
            response.raise_for_status()
            task_id = response.json()["task_id"]
            write(directory / "created.json", response.json())
            db.execute("UPDATE task_budgets SET model_limit=16,tool_limit=40,reserve_tail=4 WHERE root_task_id=? AND model_used=0", (task_id,))
            process = subprocess.Popen([sys.executable, "-m", "scripts.run_ag2_adaptive", "--worker", case, "--task-id", task_id],
                                       cwd=ROOT, env=env, stdout=worker_log, stderr=subprocess.STDOUT)
            last = None
            while process.poll() is None:
                row = db.get_task(task_id)
                budget = db.budget_summary(task_id)
                state = {"case": case, "status": row["status"], "model_used": budget["model_used"], "tool_used": budget["tool_used"],
                         "children": [c["status"] for c in db.list_subtasks(task_id)]}
                if state != last:
                    print(json.dumps(state, ensure_ascii=False), flush=True)
                    last = state
                time.sleep(2)
            response = client.get("/api/tasks/" + task_id)
            response.raise_for_status()
            result = response.json()
            write(directory / "result.json", result)
            for identity in [task_id] + [s["subtask_id"] for s in db.list_subtasks(task_id)]:
                write(directory / (identity + ".json"), {"task": db.get_task(identity), "actions": db.list_actions(identity),
                    "events": db.list_events(identity), "receipts": db.list_model_calls(identity), "checkpoint": db.get_checkpoint(identity)})
            answer = (result.get("result") or {}).get("answer_markdown")
            if answer:
                (directory / "answer.md").write_text(answer)
            duplicate = client.post("/api/tasks", json=payload).json()
            assert duplicate["task_id"] == task_id and duplicate["usage"] == result["usage"]
            total = db.fetchone("SELECT COUNT(*) AS n FROM model_calls")["n"]
            after = snapshot(Database(production.database_url))
            write(directory / "production-after.json", after)
            assets_ok = after == json.loads((OUT / "production-before.json").read_text())
            assert result["usage"]["requests"] <= 16 and total <= 48
            write(directory / "boundary.json", {"production_unchanged": assets_ok, "batch_requests": total,
                 "worker_exit": process.returncode, "duplicate_post_no_new_call": True})
            if not assets_ok or result["usage"]["unknown"] or process.returncode:
                write(OUT / "STOP.json", {"case": case, "reason": "asset/unknown/process boundary failure"})
            print(json.dumps({"case": case, "status": result["status"], "usage": result["usage"]}, ensure_ascii=False), flush=True)
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        api.terminate()
        api.wait(timeout=15)
        api_log.close()
        worker_log.close()


def main() -> None:
    """显式prepare/run/worker，重复执行不会恢复未知任务或消耗新一轮额度。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepare-correction", action="store_true")
    parser.add_argument("--run", choices=["S", "C", "P", "C2"])
    parser.add_argument("--worker", choices=["S", "C", "P", "C2"])
    parser.add_argument("--task-id")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.prepare_correction:
        prepare_correction()
    elif args.worker and args.task_id:
        worker(args.worker, args.task_id)
    elif args.run:
        run(args.run)
    else:
        parser.error("select prepare/run or worker with task-id")


if __name__ == "__main__":
    main()
