"""文件型检索实验的进程内边界：拒绝联网、数据库和输出目录外写入。

审计钩子不可卸载，调用方应在独立实验进程中安装。本模块约束 Python
入口，不替代操作系统沙箱；已打开文件对象和原生扩展的直接系统调用不在
完整隔离保证内，因此安装前不得创建业务客户端或打开可写业务文件。
"""
from __future__ import annotations

import builtins
from collections.abc import Sequence
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import sysconfig


class BoundaryViolation(BaseException):
    """越界立即终止实验，不作为普通检索失败或降级吞掉。"""


_INSTALLED = False


def write_bytes_new(path: Path, data: bytes) -> None:
    """独占写入新证据；拒绝覆盖已有文件，父目录由调用方显式创建。"""
    with Path(path).open('xb') as stream:
        stream.write(data)


def write_json_new(path: Path, value: object) -> None:
    """以UTF-8独占保存JSON，不修改旧证据。"""
    write_bytes_new(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def install_guards(output_root: Path, read_roots: Sequence[Path]) -> dict[str, int]:
    """安装本进程不可逆的文件、网络和执行边界。

    Args:
        output_root: 预先创建且为空的绝对真实目录，不接受符号链接别名。
        read_roots: 允许读取的冻结文件或目录；另允许Python包和运行时读取。

    Returns:
        可直接写入终态的实时计数字典。拒绝操作也计数，不代表请求已发送。

    Raises:
        ValueError: 路径不合格、输出非空或读取根不存在。
        RuntimeError: 当前进程已安装本模块守卫。
        BoundaryViolation: 安装后发生任何受限操作。
    """
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError('EXPERIMENT_GUARDS_ALREADY_INSTALLED')
    root = Path(output_root)
    if not root.is_absolute() or root != root.resolve() or not root.is_dir() or any(root.iterdir()):
        raise ValueError('EXPERIMENT_OUTPUT_MUST_BE_EXISTING_EMPTY_REAL_DIRECTORY')
    roots = []
    for supplied in read_roots:
        value = Path(supplied)
        if not value.is_absolute() or value != value.resolve() or not value.exists():
            raise ValueError('EXPERIMENT_READ_ROOT_INVALID')
        roots.append(value)
    # 包、标准库和解释器配置只读可用；禁止导入时写共享__pycache__。
    sys.dont_write_bytecode = True
    # 不采纳任意sys.path/PYTHONPATH，避免调用环境扩大到部署源码或数据目录。
    runtime = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    runtime.extend(Path(v).resolve() for k, v in sysconfig.get_paths().items()
                   if k in {'stdlib', 'platstdlib', 'purelib', 'platlib'} and Path(v).exists())
    allowed_reads = tuple(dict.fromkeys([root, *roots, *runtime]))
    counters = {name: 0 for name in ('blocked_network', 'blocked_database', 'blocked_subprocess',
                                    'blocked_write', 'blocked_read', 'exclusive_files_created',
                                    'output_directories_created')}
    writable_fds: dict[int, tuple[int, int]] = {}

    def refuse(kind, code):
        counters[kind] += 1
        raise BoundaryViolation(code)

    def resolve(value):
        if isinstance(value, int):
            return None
        try:
            return Path(os.fsdecode(value)).resolve()
        except (TypeError, ValueError, OSError):
            refuse('blocked_read', 'EXPERIMENT_PATH_NOT_RESOLVABLE')

    def writable_fd(fd):
        try:
            info = os.fstat(fd)
            return writable_fds.get(fd) == (info.st_dev, info.st_ino)
        except OSError:
            return False

    def check_read(value):
        if isinstance(value, int):
            return
        path = resolve(value)
        if path.name == '.env' or not any(path == p or (p.is_dir() and path.is_relative_to(p)) for p in allowed_reads):
            refuse('blocked_read', 'EXPERIMENT_READ_OUTSIDE_ALLOWLIST')

    def check_write(value, flags):
        if isinstance(value, int):
            if not writable_fd(value):
                refuse('blocked_write', 'EXPERIMENT_FOREIGN_WRITABLE_FD')
            return
        path = resolve(value)
        if path == root or not path.is_relative_to(root):
            refuse('blocked_write', 'EXPERIMENT_WRITE_OUTSIDE_OUTPUT')
        if flags & (os.O_APPEND | os.O_TRUNC) or flags & (os.O_CREAT | os.O_EXCL) != (os.O_CREAT | os.O_EXCL):
            refuse('blocked_write', 'EXPERIMENT_EXCLUSIVE_CREATE_REQUIRED')
        # O_EXCL仍由内核最终实施；已有输出保持FileExistsError且不被改动。

    def writing(flags):
        return flags & os.O_ACCMODE != os.O_RDONLY or bool(flags & (os.O_CREAT | os.O_TRUNC | os.O_APPEND))

    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            if writing(flags):
                check_write(path, flags)
            else:
                check_read(path)
        elif event == 'os.mkdir':
            path, _mode, dir_fd = args
            target = resolve(path)
            if dir_fd not in (None, -1) or target == root or not target.is_relative_to(root):
                refuse('blocked_write', 'EXPERIMENT_MKDIR_OUTSIDE_OUTPUT')
        elif event in {'os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink',
                       'os.truncate', 'os.chmod', 'os.chown', 'os.utime'}:
            refuse('blocked_write', 'EXPERIMENT_MUTATION_FORBIDDEN')
        elif event.startswith('socket.') and event not in {'socket.__new__'}:
            refuse('blocked_network', 'EXPERIMENT_NETWORK_FORBIDDEN')
        elif event.startswith('sqlite3.connect'):
            refuse('blocked_database', 'EXPERIMENT_DATABASE_FORBIDDEN')
        elif event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.fork',
                       'os.forkpty', 'os.exec', 'os.spawn'}:
            refuse('blocked_subprocess', 'EXPERIMENT_SUBPROCESS_FORBIDDEN')

    def remember(stream):
        fd = stream if isinstance(stream, int) else stream.fileno()
        info = os.fstat(fd)
        writable_fds[fd] = (info.st_dev, info.st_ino)
        counters['exclusive_files_created'] += 1
        return stream

    original_builtin_open, original_io_open, original_os_open = builtins.open, io.open, os.open
    original_write, original_mkdir = os.write, os.mkdir

    def wrapped_open(original):
        def guarded(file, mode='r', *args, **kwargs):
            stream = original(file, mode, *args, **kwargs)
            return remember(stream) if any(c in mode for c in 'wax+') else stream
        return guarded

    def guarded_os_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is not None:
            refuse('blocked_write' if writing(flags) else 'blocked_read', 'EXPERIMENT_DIR_FD_FORBIDDEN')
        fd = original_os_open(path, flags, mode)
        return remember(fd) if writing(flags) else fd

    def guarded_mkdir(path, mode=0o777, *, dir_fd=None):
        result = original_mkdir(path, mode, dir_fd=dir_fd)
        counters['output_directories_created'] += 1
        return result

    def guarded_write(fd, data):
        if not writable_fd(fd):
            refuse('blocked_write', 'EXPERIMENT_FOREIGN_WRITABLE_FD')
        return original_write(fd, data)

    def deny(kind, code):
        def blocked(*args, **kwargs):
            refuse(kind, code)
        return blocked

    async def blocked_async_send(*args, **kwargs):
        refuse('blocked_network', 'EXPERIMENT_HTTP_FORBIDDEN')

    _INSTALLED = True
    sys.addaudithook(audit)
    builtins.open, io.open = wrapped_open(original_builtin_open), wrapped_open(original_io_open)
    os.open, os.write, os.mkdir = guarded_os_open, guarded_write, guarded_mkdir
    for name in ('connect', 'connect_ex', 'send', 'sendall', 'sendto', 'sendmsg', 'bind', 'listen'):
        if hasattr(socket.socket, name):
            setattr(socket.socket, name, deny('blocked_network', 'EXPERIMENT_NETWORK_FORBIDDEN'))
    for name in ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr', 'getnameinfo', 'create_connection'):
        setattr(socket, name, deny('blocked_network', 'EXPERIMENT_NETWORK_FORBIDDEN'))
    sqlite3.connect = deny('blocked_database', 'EXPERIMENT_DATABASE_FORBIDDEN')
    subprocess.Popen = deny('blocked_subprocess', 'EXPERIMENT_SUBPROCESS_FORBIDDEN')
    # 可选包不构成依赖；已有或后续导入均由socket审计兜底。
    for module_name in ('httpx', 'pymysql'):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        if module_name == 'httpx':
            module.Client.send = deny('blocked_network', 'EXPERIMENT_HTTP_FORBIDDEN')
            module.AsyncClient.send = blocked_async_send
        else:
            module.connect = deny('blocked_database', 'EXPERIMENT_DATABASE_FORBIDDEN')
            module.Connection = deny('blocked_database', 'EXPERIMENT_DATABASE_FORBIDDEN')
    return counters
