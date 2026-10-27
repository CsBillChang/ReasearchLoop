"""实验文件、数据库与网络的审计边界，不导入或启动业务服务。"""
from __future__ import annotations
import os
from pathlib import Path
import sys
import traceback

class IsolationError(RuntimeError):
    """实验尝试越过独立存储、配置或网络边界。"""

class Boundary:
    """限制本进程所有线程的 Python 可审计写入和网络操作。"""

    def __init__(self, output: Path):
        """只允许向本次新建的输出目录写入，不授予业务资源访问权。"""
        self.output = output
        self.enabled = True
        self.phase = 'qualification'
        self.denials = []
        self.sqlite_paths = set()

    def reject(self, operation: str):
        """记录阻断类别，避免输出可能含凭据的调用参数。"""
        origin = [{'file': frame.filename, 'line': frame.lineno, 'function': frame.name}
                  for frame in traceback.extract_stack(limit=8)[:-1]]
        self.denials.append({'phase': self.phase, 'operation': operation, 'origin': origin})
        raise IsolationError('isolated qualification rejected: ' + operation)

    @staticmethod
    def fd_root(fd: int) -> Path:
        """还原相对 dir_fd 的实际目录，支持 Linux 与 macOS。"""
        if sys.platform == 'darwin':
            import fcntl
            raw = fcntl.fcntl(fd, 50, bytes(1024))  # macOS F_GETPATH。
            return Path(os.fsdecode(raw.split(b'\0', 1)[0])).resolve()
        return Path('/proc/self/fd/' + str(fd)).resolve()

    def allowed_path(self, value, fd: int = -1) -> bool:
        """路径先解析符号链接，防止用相对目录或链接写入日常实例。"""
        if isinstance(value, int):
            return True
        path = Path(os.fsdecode(value))
        if not path.is_absolute() and fd not in {-1, None}:
            path = self.fd_root(fd) / path
        return path.resolve().is_relative_to(self.output)

    def write_path(self, value, fd: int = -1):
        """拒绝输出根以外的文件与目录变更。"""
        if not self.allowed_path(value, fd):
            self.reject('write_outside_output')

    def audit(self, event: str, args):
        """覆盖文件变更、SQLite 连接、进程启动和套接字连接边界。"""
        if not self.enabled:
            return
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.bind', 'subprocess.Popen'}:
            self.reject(event)
        if event == 'open':
            path, mode, flags = args
            if not isinstance(path, int) and Path(os.fsdecode(path)).name == '.env':
                self.reject('dotenv_read')
            writes = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if writes:
                self.write_path(path)
        elif event == 'sqlite3.connect':
            path = args[0]
            if path != ':memory:' and not self.allowed_path(path):
                self.reject('sqlite_outside_output')
            self.sqlite_paths.add(str(path))
        elif event in {'os.remove', 'os.rmdir'}:
            self.write_path(args[0], args[1])
        elif event in {'os.mkdir', 'os.chmod'}:
            self.write_path(args[0], args[2])
        elif event in {'os.rename', 'os.link'}:
            self.write_path(args[0], args[2])
            self.write_path(args[1], args[3])
        elif event in {'os.truncate', 'os.chown', 'os.chflags'}:
            self.write_path(args[0])
        elif event == 'os.utime':
            self.write_path(args[0], args[3])
        elif event == 'os.symlink':
            self.reject('symlink_creation')
