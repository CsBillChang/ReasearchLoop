"""启动本机简报轮询；只有已启用且到期的订阅才查询论文，不调用生成模型。"""
import argparse
import signal
import threading
from research_loop.config import get_settings
from research_loop.workers.scheduler import BriefScheduler
from research_loop.services.runtime import application_lock, heartbeat


def main():
    """提供独立持续进程和一次到期检查，便于关闭及诊断。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set())
    signal.signal(signal.SIGINT,lambda *_:stop.set())
    settings=get_settings()
    with application_lock(settings.project_root):
        worker=BriefScheduler(settings)
        worker.db.fetchone('SELECT 1 AS ok')
        heartbeat('ready')
        if args.once:
            import json
            print(json.dumps(worker.run_once(),ensure_ascii=False))
        else:
            while not stop.is_set():
                heartbeat('polling_or_executing')
                worker.run_once()
                heartbeat('ready')
                stop.wait(30)
        heartbeat('stopped')


if __name__=='__main__':
    main()
