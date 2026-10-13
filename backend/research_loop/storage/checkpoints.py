"""图状态存储：保存应用自己的 state JSON；它辅助恢复，不替代业务收据、文件核验或幂等控制。"""
from __future__ import annotations
from typing import Any
from research_loop.storage.mysql import Database


class CheckpointStore:
    def __init__(self, db: Database):
        self.db = db

    def save(self, task_id: str, state: dict[str, Any]) -> None:
        """序列化当前图状态供恢复；模型与工具副作用仍须依赖各自持久收据去重。"""
        serializable = {key: value for key, value in state.items() if not key.startswith("_")}
        owner = serializable.get("worker_id")
        token = serializable.get("lease_token")
        if owner and not self.db.is_owner(task_id, owner, token):
            raise RuntimeError("worker lease lost; checkpoint write rejected")
        self.db.save_checkpoint(task_id, serializable)

    def load(self, task_id: str) -> dict[str, Any] | None:
        """返回保存的图状态或空值；调用方决定恢复还是创建初始状态。"""
        return self.db.get_checkpoint(task_id)
