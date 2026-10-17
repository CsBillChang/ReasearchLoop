"""隔离Worker故障注入：阶段行动提交后在Checkpoint写入前SIGKILL自身。"""
import json
import os
import signal
import sys
import threading
from pathlib import Path

from research_loop.config import Settings
from research_loop.storage.mysql import Database
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.workers.executor import TaskExecutor
from integration.test_collaboration_contract import FixturePDF
from integration.test_research_progress_handoff import ProgressScenario


def main():
    """仅运行显式临时配置任务，不读取生产环境或创建真实供应商。"""
    settings = Settings.model_validate(json.loads(Path(sys.argv[1]).read_text()))
    url = settings.database_url
    if not url.startswith('sqlite:') and not url.rsplit('/', 1)[-1].startswith('researchloop_test_core_loop_'):
        raise ValueError('isolated database required')
    db = Database(url)
    original = CheckpointStore.save
    # 此用例验证成功收据后的恢复。两名Researcher都到达同一提交边界再杀进程，
    # 避免随机打断另一名在途请求，把应保守阻断的unknown误当恢复缺陷。
    committed = threading.Barrier(2, timeout=10)
    def stop_after_committed_progress(store, task_id, state):
        """持久行动可见后真实结束整个Worker进程，留下旧快照供恢复。"""
        if any(a['tool_name']=='record_research_progress' and a['status']=='succeeded' for a in db.list_actions(task_id)):
            committed.wait()
            os.kill(os.getpid(), signal.SIGKILL)
        return original(store, task_id, state)
    CheckpointStore.save = stop_after_committed_progress
    TaskExecutor(settings, db, provider_factory=lambda _: ProgressScenario(), pdf_factory=lambda *_: FixturePDF()).execute(sys.argv[2])


if __name__ == '__main__': main()
