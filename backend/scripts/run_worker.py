"""独立Worker入口：指定任务、单次领取或持续轮询；保留权限、收据和优雅停机边界。"""
from __future__ import annotations
import argparse
import signal
import threading
from research_loop.config import get_settings
from research_loop.workers.executor import TaskExecutor
from research_loop.services.runtime import application_lock, heartbeat


def main() -> None:
    """启动前校验数据库；SIGTERM 后完成当前任务再退出，不丢弃远端收据。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--worker-id", default="researchloop-worker")
    args = parser.parse_args()
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    settings = get_settings()
    with application_lock(settings.project_root):
        worker = TaskExecutor(settings)
        worker.db.fetchone('SELECT 1 AS ok')
        heartbeat('ready')
        if args.task_id:
            heartbeat('busy')
            worker.execute(args.task_id, args.worker_id)
        elif args.once:
            worker.run_once(args.worker_id)
        else:
            while not stopping.is_set():
                heartbeat('polling_or_executing')
                worker.run_once(args.worker_id)
                heartbeat('ready')
                stopping.wait(1)
        heartbeat('stopped')


if __name__ == "__main__":
    main()
