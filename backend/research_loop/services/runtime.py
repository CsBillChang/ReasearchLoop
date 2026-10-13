"""本机进程边界：应用共享锁与冷备份独占锁；不负责模型调度或修改业务状态。"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def runtime_root(root: Path) -> Path:
    """返回项目唯一运行目录，避免隔离实例绕过冷备份锁。"""
    path = root / 'data/runtime'
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


@contextmanager
def application_lock(root: Path, *, exclusive: bool = False):
    """持有应用共享锁或维护独占锁；冲突立即失败，不等待未知远端请求。

    Args:
        root: 当前项目根目录。
        exclusive: 冷备份/恢复使用 True，API、Worker、Scheduler 使用 False。
    """
    with (runtime_root(root) / 'application.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('APPLICATION_BUSY: 请先停止本项目应用进程，再维护数据') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path: Path, value: dict) -> None:
    """原子发布本机状态，避免读到半份记录；文件只允许本机用户访问。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temp.open('w') as handle:
        os.chmod(temp, 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def identity(settings) -> str:
    """生成配置身份指纹，不输出数据库口令或供应商密钥。"""
    return hashlib.sha256(json.dumps([settings.database_url, str(settings.data_dir),
        str(settings.output_dir), settings.retrieval_namespace], ensure_ascii=False).encode()).hexdigest()


def heartbeat(state: str) -> None:
    """受管进程报告已完成初始化或正在执行；未受管命令不产生伪就绪文件。"""
    target = os.getenv('RESEARCHLOOP_HEARTBEAT')
    if target:
        atomic_json(Path(target), {'pid': os.getpid(), 'token': os.getenv('RESEARCHLOOP_PROCESS_TOKEN'),
            'state': state, 'time': time.time()})
