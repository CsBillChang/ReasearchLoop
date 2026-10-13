"""Child process used by the explicit SIGKILL recovery test."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from research_loop.config import Settings
from research_loop.providers.deepseek import ModelResponse
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.mysql import Database
import research_loop.workers.executor as executor_module
from research_loop.workers.executor import TaskExecutor


class Provider:
    def __init__(self, counter: Path):
        self.counter = counter

    def complete(self, *args, **kwargs):
        with self.counter.open("a", encoding="utf-8") as handle:
            handle.write("call\n")
        return ModelResponse("recovered", raw={"choices": [{"message": {"content": "recovered"}, "finish_reason": "stop"}]},
                             usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}, request_id="child-request-1")

    def close(self):
        pass


class BlockingCheckpoint(CheckpointStore):
    def save(self, task_id, state):
        Path(os.environ["AG2_RECOVERY_READY"]).write_text("receipt-written\n", encoding="utf-8")
        time.sleep(60)


def main():
    db_url = os.environ["DATABASE_URL"]
    root = Path(os.environ["AG2_RECOVERY_ROOT"])
    settings = Settings(project_root=root, data_dir=root / "data", output_dir=root / "out",
                        database_url=db_url, deepseek_api_key="synthetic", max_model_requests=6,
                        max_tool_actions=10)
    if sys.argv[1] == "crash":
        executor_module.CheckpointStore = BlockingCheckpoint
    provider_factory = lambda _: Provider(Path(os.environ["AG2_RECOVERY_COUNTER"]))
    TaskExecutor(settings, provider_factory=provider_factory).execute(os.environ["AG2_RECOVERY_TASK"], sys.argv[1])


if __name__ == "__main__":
    main()
