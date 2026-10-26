"""为离线MPS实验分离只增证据、可变临时缓存和只读模型/依赖。

本模块约束Python入口，不是操作系统沙箱。原生Metal、动态库或系统服务的
直接系统调用不受完整约束；调用者另须事前/事后核验受保护部署路径。
"""
from __future__ import annotations

import builtins
from collections.abc import Sequence
import io
import os
from pathlib import Path
import socket
import sqlite3
import stat
import subprocess
import sys
import sysconfig
import threading

from scripts.retrieval_experiment_guard import BoundaryViolation, write_json_new


_INSTALLED = False


def install_guards(output_root: Path, read_roots: Sequence[Path],
                   mutable_roots: Sequence[Path] = ()) -> dict[str, object]:
    """安装不可卸载的离线边界，允许专属缓存清理但保护原始证据。

    Args:
        output_root: 已存在且为空的真实绝对目录，只允许独占新增证据。
        read_roots: 显式只读输入和源码，标准库与当前解释器依赖自动只读。
        mutable_roots: 仅本次实验的真实缓存/临时目录，可覆盖、重命名和删除。

    Returns:
        守卫实时计数；原生库绕开Python入口的操作不在这些计数中。

    Raises:
        ValueError: 路径或证据目录不符合要求。
        RuntimeError: 当前进程已安装本守卫。
        BoundaryViolation: 安装后尝试越界读写、联网、SQL或启动子进程。
    """
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError('MPS_GUARDS_ALREADY_INSTALLED')

    def supplied_path(value, *, directory=False):
        path = Path(value)
        if (not path.is_absolute() or path != path.resolve() or not path.exists()
                or directory and not path.is_dir()):
            raise ValueError('MPS_GUARD_PATH_INVALID')
        return path

    output = supplied_path(output_root, directory=True)
    if any(output.iterdir()):
        raise ValueError('MPS_OUTPUT_MUST_BE_EMPTY')
    mutable = tuple(supplied_path(p, directory=True) for p in mutable_roots)
    if len(set(mutable)) != len(mutable):
        raise ValueError('MPS_MUTABLE_ROOT_DUPLICATED')
    for ordinal, path in enumerate((output, *mutable)):
        for other in (output, *mutable)[ordinal + 1:]:
            if path == other or path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError('MPS_WRITE_ROOTS_OVERLAP')
    reads = [supplied_path(p) for p in read_roots]
    runtime = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    runtime.extend(Path(v).resolve() for k, v in sysconfig.get_paths().items()
                   if k in {'stdlib', 'platstdlib', 'purelib', 'platlib'} and Path(v).exists())
    # 可变目录不得覆盖只读输入、源码或解释器。明确只读父目录仍可含专属缓存。
    for path in mutable:
        if any(path == other or other.is_relative_to(path) for other in [*reads, *runtime]):
            raise ValueError('MPS_MUTABLE_ROOT_COVERS_PROTECTED_INPUT')
    allowed_reads = tuple(dict.fromkeys([output, *mutable, *runtime, *reads]))
    counts = {name: 0 for name in (
        'blocked_network', 'blocked_database', 'blocked_subprocess', 'blocked_write',
        'blocked_read', 'exclusive_files_created', 'mutable_file_opens',
        'mutable_mutations', 'output_directories_created', 'absent_platform_probes')}
    counts['ipv6_import_probe_disabled'] = True
    counts['blocked_read_paths'] = []
    known_fds: dict[int, tuple[int, int, Path, bool, bool]] = {}
    open_context = threading.local()

    def refuse(kind, code):
        counts[kind] += 1
        raise BoundaryViolation(code)

    def fd_reference(fd):
        try:
            info = os.fstat(fd)
            ref = known_fds.get(fd)
            if ref is not None and ref[:2] == (info.st_dev, info.st_ino):
                return ref
        except OSError:
            pass
        return None

    def resolve(value, dir_fd=None):
        if isinstance(value, int):
            ref = fd_reference(value)
            if ref is None:
                refuse('blocked_read', 'MPS_FOREIGN_FILE_DESCRIPTOR')
            return ref[2]
        try:
            path = Path(os.fsdecode(value))
            if dir_fd not in (None, -1):
                ref = fd_reference(dir_fd)
                if ref is None or not ref[3] or path.is_absolute():
                    refuse('blocked_read', 'MPS_DIR_FD_UNVERIFIED')
                path = ref[2] / path
            return path.resolve()
        except (TypeError, ValueError, OSError):
            refuse('blocked_read', 'MPS_PATH_NOT_RESOLVABLE')

    def within(path, roots, *, allow_root=True):
        return any((allow_root and path == root) or path != root and path.is_relative_to(root)
                   for root in roots)

    def reject_read(path, code):
        text = str(path)
        if len(counts['blocked_read_paths']) < 8 and text not in counts['blocked_read_paths']:
            counts['blocked_read_paths'].append(text)
        refuse('blocked_read', code)

    def check_read(value, dir_fd=None, *, count_probe=True):
        # Torch在加载全局依赖时尝试读取Linux maps；Darwin上须保留可捕获的ENOENT。
        # 只认可字面路径和实际不存在状态，不能借这个探测读取真实文件或符号链接。
        if (not isinstance(value, int) and os.fsdecode(value) == '/proc/self/maps'
                and dir_fd in (None, -1)):
            probe = Path('/proc/self/maps')
            try:
                os.lstat(probe)
                absent = False
            except FileNotFoundError:
                absent = True
            except OSError:
                absent = False
            if (sys.platform == 'darwin' and absent and not os.path.lexists(probe)
                    and not any(part.is_symlink() for part in (Path('/proc'), Path('/proc/self'), probe))
                    and probe.resolve() == probe):
                if count_probe:
                    counts['absent_platform_probes'] += 1
                return probe
            reject_read(probe, 'MPS_PLATFORM_PROBE_IS_NOT_ABSENT')
        path = resolve(value, dir_fd)
        if path.name == '.env' or not within(path, allowed_reads):
            reject_read(path, 'MPS_READ_OUTSIDE_ALLOWLIST')
        return path

    def writing(flags):
        return flags & os.O_ACCMODE != os.O_RDONLY or bool(flags & (os.O_CREAT | os.O_TRUNC | os.O_APPEND))

    def check_write(value, flags, dir_fd=None):
        path = resolve(value, dir_fd)
        if isinstance(value, int):
            ref = fd_reference(value)
            if ref is None or not ref[4]:
                refuse('blocked_write', 'MPS_FOREIGN_WRITABLE_FD')
            # fdopen只复用本守卫已准入的写句柄，不是重新打开或覆盖旧证据。
            if within(path, (output,), allow_root=False):
                return path
        if within(path, mutable, allow_root=False):
            return path
        if not within(path, (output,), allow_root=False):
            refuse('blocked_write', 'MPS_WRITE_OUTSIDE_EXPERIMENT')
        if flags & (os.O_APPEND | os.O_TRUNC) or flags & (os.O_CREAT | os.O_EXCL) != (os.O_CREAT | os.O_EXCL):
            refuse('blocked_write', 'MPS_EVIDENCE_EXCLUSIVE_CREATE_REQUIRED')
        return path

    def check_mutation(value, dir_fd=None):
        path = resolve(value, dir_fd)
        if not within(path, mutable, allow_root=False):
            refuse('blocked_write', 'MPS_MUTATION_OUTSIDE_MUTABLE_CACHE')
        return path

    def audit(event, args):
        if event == 'open':
            path, _mode, flags = args
            overrides = getattr(open_context, 'overrides', ())
            if overrides and overrides[-1][0] == path and overrides[-1][1] == flags:
                path = overrides[-1][2]
            if writing(flags):
                check_write(path, flags)
            else:
                check_read(path)
        elif event == 'os.mkdir':
            path = resolve(args[0], args[2])
            if not within(path, (*mutable, output), allow_root=False):
                refuse('blocked_write', 'MPS_MKDIR_OUTSIDE_EXPERIMENT')
        elif event in {'os.remove', 'os.rmdir'}:
            check_mutation(args[0], args[1])
            counts['mutable_mutations'] += 1
        elif event == 'os.rename':
            check_mutation(args[0], args[2])
            check_mutation(args[1], args[3])
            counts['mutable_mutations'] += 1
        elif event in {'os.link', 'os.symlink'}:
            refuse('blocked_write', 'MPS_LINK_CREATION_FORBIDDEN')
        elif event in {'os.truncate', 'os.chmod', 'os.chown', 'os.utime'}:
            dir_fd = args[-1] if event in {'os.chmod', 'os.chown', 'os.utime'} else None
            check_mutation(args[0], dir_fd)
            counts['mutable_mutations'] += 1
        elif event.startswith('socket.') and event != 'socket.__new__':
            refuse('blocked_network', 'MPS_NETWORK_FORBIDDEN')
        elif event.startswith('sqlite3.connect'):
            refuse('blocked_database', 'MPS_DATABASE_FORBIDDEN')
        elif event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.fork',
                       'os.forkpty', 'os.exec', 'os.spawn'}:
            refuse('blocked_subprocess', 'MPS_SUBPROCESS_FORBIDDEN')

    def remember(stream, path, flags):
        fd = stream if isinstance(stream, int) else stream.fileno()
        info = os.fstat(fd)
        is_writable = writing(flags)
        known_fds[fd] = (info.st_dev, info.st_ino, path, stat.S_ISDIR(info.st_mode), is_writable)
        if is_writable:
            counts['mutable_file_opens' if within(path, mutable) else 'exclusive_files_created'] += 1
        return stream

    original_builtin_open, original_io_open, original_os_open = builtins.open, io.open, os.open
    original_write, original_mkdir = os.write, os.mkdir

    def wrapped_open(original):
        def guarded(file, mode='r', *args, **kwargs):
            path = resolve(file)
            stream = original(file, mode, *args, **kwargs)
            flags = os.O_RDWR if '+' in mode else os.O_WRONLY if any(c in mode for c in 'wax') else os.O_RDONLY
            return remember(stream, path, flags)
        return guarded

    def guarded_os_open(path, flags, mode=0o777, *, dir_fd=None):
        resolved = (check_write(path, flags, dir_fd) if writing(flags)
                    else check_read(path, dir_fd, count_probe=False))
        if not hasattr(open_context, 'overrides'):
            open_context.overrides = []
        open_context.overrides.append((path, flags, resolved))
        try:
            fd = original_os_open(path, flags, mode, dir_fd=dir_fd)
        finally:
            open_context.overrides.pop()
        return remember(fd, resolved, flags)

    def guarded_write(fd, data):
        ref = fd_reference(fd)
        if ref is None or not ref[4]:
            refuse('blocked_write', 'MPS_FOREIGN_WRITABLE_FD')
        return original_write(fd, data)

    def guarded_mkdir(path, mode=0o777, *, dir_fd=None):
        result = original_mkdir(path, mode, dir_fd=dir_fd)
        counts['output_directories_created'] += 1
        return result

    def denied(kind, code):
        def reject(*args, **kwargs):
            refuse(kind, code)
        return reject

    sys.dont_write_bytecode = True
    _INSTALLED = True
    sys.addaudithook(audit)
    builtins.open, io.open = wrapped_open(original_builtin_open), wrapped_open(original_io_open)
    os.open, os.write, os.mkdir = guarded_os_open, guarded_write, guarded_mkdir
    # urllib3在导入时会探测IPv6 bind。离线实验直接禁用该探测，不放行任何bind。
    socket.has_ipv6 = False
    for name in ('connect', 'connect_ex', 'send', 'sendall', 'sendto', 'sendmsg', 'bind', 'listen'):
        if hasattr(socket.socket, name):
            setattr(socket.socket, name, denied('blocked_network', 'MPS_NETWORK_FORBIDDEN'))
    for name in ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr', 'getnameinfo', 'create_connection'):
        setattr(socket, name, denied('blocked_network', 'MPS_NETWORK_FORBIDDEN'))
    sqlite3.connect = denied('blocked_database', 'MPS_DATABASE_FORBIDDEN')
    subprocess.Popen = denied('blocked_subprocess', 'MPS_SUBPROCESS_FORBIDDEN')
    counts['low_level_spawn_entrypoints_blocked'] = []
    # multiprocessing.spawnv_passfds直接调用C扩展，不经过Popen/os.fork审计入口。
    try:
        import _posixsubprocess
    except ImportError:
        pass
    else:
        _posixsubprocess.fork_exec = denied('blocked_subprocess', 'MPS_SUBPROCESS_FORBIDDEN')
        counts['low_level_spawn_entrypoints_blocked'].append('_posixsubprocess.fork_exec')
        if hasattr(subprocess, '_fork_exec'):
            subprocess._fork_exec = denied('blocked_subprocess', 'MPS_SUBPROCESS_FORBIDDEN')
            counts['low_level_spawn_entrypoints_blocked'].append('subprocess._fork_exec')
    return counts
