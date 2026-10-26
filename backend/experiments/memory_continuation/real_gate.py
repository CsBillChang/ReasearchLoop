"""真实小试验的共享发送门；跨进程保留预算，不重试或改写供应商响应。"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

import httpx

from research_loop.providers.deepseek import RequestNotSent
from research_loop.storage.token_budget import actual_usage


OUTPUT_LIMIT = 8192
FORMAT_ALLOWANCE = 1024
SCHEMA_VERSION = 1
NOTICE = ('预留按messages/tools的UTF-8 JSON字节数、1024格式余量及8192输出上限计算；'
          '这是保守发送门，不是供应商账单Token的数学上界保证。实际用量以原始usage为准。')


def _json_bytes(value) -> bytes:
    """使用稳定UTF-8序列化计算身份和本地预留。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


class SharedGate:
    """持久化跨三轮根任务的请求、Token及发送时间窗口。"""

    def __init__(self, output: Path, max_requests: int = 12,
                 token_limit: int = 160000, seconds: float = 600):
        """创建或读回相同上限的账本，重入不重置已用次数或截止时间。

        Args:
            output: 本次隔离实验目录，只在目录内写账本和响应原件。
            max_requests: 全局最多准入的物理发送尝试数。
            token_limit: 全局已知用量与未结算预留的发送门限。
            seconds: 从首次建立账本起计算的绝对发送窗口。
        """
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError('max_requests must be a positive integer')
        if type(token_limit) is not int or token_limit < 1 or seconds <= 0:
            raise ValueError('token_limit and seconds must be positive')
        self.output = Path(output).resolve()
        self.output.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = self.output / 'ledger.json'
        self._lock_path = self.output / '.gate.lock'
        self._lock = threading.RLock()
        self._limits = {'max_requests': max_requests, 'token_limit': token_limit,
                        'seconds': float(seconds)}
        with self._locked():
            if self.path.exists():
                self._state = self._read()
                if self._state.get('schema_version') != SCHEMA_VERSION:
                    raise ValueError('GATE_LEDGER_VERSION_MISMATCH')
                if any(self._state.get(key) != value for key, value in self._limits.items()):
                    raise ValueError('GATE_FROZEN_LIMITS_MISMATCH')
                if self._state.get('output_limit') != OUTPUT_LIMIT:
                    raise ValueError('GATE_FROZEN_OUTPUT_MISMATCH')
            else:
                self._state = {'schema_version': SCHEMA_VERSION, **self._limits,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'created_monotonic': time.monotonic(), 'output_limit': OUTPUT_LIMIT,
                    'format_allowance': FORMAT_ALLOWANCE, 'notice': NOTICE,
                    'stop_reason': None, 'requests': []}
                self._write()

    @contextmanager
    def _locked(self):
        """用线程锁和文件锁串行准入；等待网络时不持账本锁。"""
        with self._lock:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def _read(self) -> dict:
        """每次操作重新读盘，防止另一CLI已更新计数却被旧实例覆盖。"""
        return json.loads(self.path.read_text(encoding='utf-8'))

    def _write(self) -> None:
        """以原子替换保存私有账本，避免中途退出留下半份JSON。"""
        fd, name = tempfile.mkstemp(prefix='.ledger-', dir=self.output)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(_json_bytes(self._state))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @staticmethod
    def _totals(state: dict) -> tuple[int, int]:
        """已知用量与未知预留分开累计，不把缺usage当零。"""
        used = sum(row['actual_tokens'] for row in state['requests']
                   if row.get('actual_tokens') is not None)
        reserved = sum(row['reserved_tokens'] for row in state['requests']
                       if row.get('actual_tokens') is None)
        return used, reserved

    def _remaining(self) -> float:
        """跨CLI沿用首次monotonic起点；时钟起点倒退时拒绝发送。"""
        elapsed = time.monotonic() - self._state['created_monotonic']
        return 0.0 if elapsed < 0 else max(0.0, self._state['seconds'] - elapsed)

    @property
    def active_index(self) -> int | None:
        """返回唯一未结算发送身份；未知结果仍保留该身份。"""
        with self._locked():
            self._state = self._read()
            pending = [row['index'] for row in self._state['requests']
                       if row.get('actual_tokens') is None]
            return pending[0] if pending else None

    def before(self, payload: dict) -> int:
        """在真正进入网络前预留一次发送，拒绝时抛RequestNotSent。

        Args:
            payload: 供应商将实际发送的JSON对象，输出上限必须为8192。

        Returns:
            本次全局发送号，从1开始。
        """
        with self._locked():
            self._state = self._read()
            reason = None
            if self._state.get('stop_reason'):
                reason = 'GATE_ALREADY_STOPPED'
            elif any(row.get('actual_tokens') is None for row in self._state['requests']):
                reason = 'GATE_UNSETTLED_REQUEST'
            elif self._remaining() <= 0:
                reason = 'GATE_TIME_LIMIT'
            elif len(self._state['requests']) >= self._state['max_requests']:
                reason = 'GATE_REQUEST_LIMIT'
            elif (not isinstance(payload, dict) or type(payload.get('max_tokens')) is not int
                    or payload['max_tokens'] != OUTPUT_LIMIT):
                reason = 'GATE_OUTPUT_LIMIT_MISMATCH'
            if reason:
                self._state['stop_reason'] = self._state.get('stop_reason') or reason
                self._write()
                raise RequestNotSent(reason)
            try:
                inputs = _json_bytes({'messages': payload['messages'], 'tools': payload.get('tools', [])})
                request_hash = hashlib.sha256(_json_bytes(payload)).hexdigest()
            except (KeyError, TypeError, ValueError) as exc:
                self._state['stop_reason'] = 'GATE_INVALID_PAYLOAD'
                self._write()
                raise RequestNotSent('GATE_INVALID_PAYLOAD') from exc
            amount = len(inputs) + FORMAT_ALLOWANCE + OUTPUT_LIMIT
            used, reserved = self._totals(self._state)
            if used + reserved + amount > self._state['token_limit']:
                self._state['stop_reason'] = 'GATE_TOKEN_RESERVATION_LIMIT'
                self._write()
                raise RequestNotSent('GATE_TOKEN_RESERVATION_LIMIT')
            index = len(self._state['requests']) + 1
            self._state['requests'].append({'index': index, 'status': 'reserved',
                'admitted_at': datetime.now(timezone.utc).isoformat(),
                'admitted_monotonic': time.monotonic(), 'input_json_bytes': len(inputs),
                'reserved_tokens': amount, 'actual_tokens': None,
                'payload_sha256': request_hash, 'usage': None, 'usage_unknown': True})
            self._write()
            return index

    def settle(self, index: int, usage: dict | None) -> None:
        """按原始usage结算；缺失或矛盾用量保留预留并停止后续发送。"""
        amount = actual_usage(usage) if isinstance(usage, dict) else None
        with self._locked():
            self._state = self._read()
            row = next((item for item in self._state['requests'] if item['index'] == index), None)
            if row is None:
                raise ValueError('GATE_UNKNOWN_REQUEST_INDEX')
            if row.get('actual_tokens') is not None:
                if amount != row['actual_tokens']:
                    raise ValueError('GATE_SETTLEMENT_CONFLICT')
                return
            row['usage'] = deepcopy(usage)
            if amount is None:
                row['status'] = 'unknown'
                row['usage_unknown'] = True
                self._state['stop_reason'] = self._state.get('stop_reason') or 'GATE_USAGE_UNKNOWN'
            else:
                row.update(status='settled', actual_tokens=amount, usage_unknown=False)
                used, reserved = self._totals(self._state)
                if used + reserved > self._state['token_limit']:
                    self._state['stop_reason'] = self._state.get('stop_reason') or 'GATE_ACTUAL_TOKEN_OVERRUN'
            self._write()

    def stop(self, reason: str) -> None:
        """保留首次停止原因与在途预留；调用方不得传入凭据或请求头。"""
        with self._locked():
            self._state = self._read()
            self._state['stop_reason'] = self._state.get('stop_reason') or str(reason)[:500]
            for row in self._state['requests']:
                if row.get('actual_tokens') is None:
                    row['status'] = ('rejected' if row.get('http_status', 0) >= 300 else 'unknown')
                    row['usage_unknown'] = True
            self._write()

    def summary(self) -> dict:
        """返回可审查账本与剩余额度，未知部分继续显示为预留。"""
        with self._locked():
            self._state = self._read()
            used, reserved = self._totals(self._state)
            return {**deepcopy(self._state), 'used_tokens': used, 'reserved_tokens': reserved,
                    'remaining_tokens': max(0, self._state['token_limit'] - used - reserved),
                    'admitted_requests': len(self._state['requests']),
                    'remaining_seconds': self._remaining()}

    def _transport_metadata(self, index: int, **metadata) -> None:
        """补记实际线上的哈希和响应原件身份，不保存请求头。"""
        with self._locked():
            self._state = self._read()
            row = next(item for item in self._state['requests'] if item['index'] == index)
            row.update(metadata)
            self._write()

    def _check_stream_deadline(self) -> None:
        """发送后的超时仍属远端未知，不能冒充RequestNotSent。"""
        with self._locked():
            self._state = self._read()
            expired = self._remaining() <= 0
        if expired:
            self.stop('GATE_STREAM_DEADLINE')
            raise TimeoutError('GATE_STREAM_DEADLINE')


class _RecordedStream(httpx.SyncByteStream):
    """逐块保存实际响应，异常时保留已收到的部分。"""

    def __init__(self, inner, gate: SharedGate, index: int, path: Path):
        self.inner, self.gate, self.index = inner, gate, index
        self.stream = path.open('xb')
        os.chmod(path, 0o600)
        self.sha = hashlib.sha256()
        self.size = 0
        self.exhausted = False
        self.closed = False

    def __iter__(self):
        """读取前后检查绝对窗口；只转交原始字节，不解析或修补SSE。"""
        try:
            iterator = iter(self.inner)
            while True:
                self.gate._check_stream_deadline()
                try:
                    chunk = next(iterator)
                except StopIteration:
                    self.exhausted = True
                    break
                self.stream.write(chunk)
                self.stream.flush()
                self.sha.update(chunk)
                self.size += len(chunk)
                self.gate._check_stream_deadline()
                yield chunk
        except Exception:
            self.gate.stop('GATE_RESPONSE_STREAM_FAILED')
            raise
        finally:
            self.close()

    def close(self):
        """关闭连接并固定部分响应的字节数与hash，不编造流完成状态。"""
        if self.closed:
            return
        self.closed = True
        try:
            self.inner.close()
        finally:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            self.gate._transport_metadata(self.index, response_bytes=self.size,
                response_sha256=self.sha.hexdigest(), response_stream_exhausted=self.exhausted)


class GuardedTransport(httpx.BaseTransport):
    """只允许官方模型端点，将每次真实发送纳入同一持久发送门。"""

    def __init__(self, gate: SharedGate, inner: httpx.BaseTransport | None = None):
        """显式关闭连接重试；离线自检可传入MockTransport。"""
        self.gate = gate
        self.inner = inner if inner is not None else httpx.HTTPTransport(retries=0, trust_env=False)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """校验端点后预留发送，响应原件中不记录Authorization请求头。"""
        url = request.url
        if (request.method != 'POST' or url.scheme != 'https' or url.host != 'api.deepseek.com'
                or url.port not in {None, 443} or url.username or url.password or url.query or url.fragment
                or url.raw_path not in {b'/chat/completions', b'/v1/chat/completions'}):
            self.gate.stop('GATE_FORBIDDEN_ENDPOINT')
            raise RequestNotSent('GATE_FORBIDDEN_ENDPOINT')
        try:
            raw = request.content
            payload = json.loads(raw)
        except (ValueError, TypeError, httpx.RequestNotRead) as exc:
            self.gate.stop('GATE_INVALID_REQUEST_BODY')
            raise RequestNotSent('GATE_INVALID_REQUEST_BODY') from exc
        index = self.gate.before(payload)
        directory = self.gate.output / 'raw-responses'
        directory.mkdir(mode=0o700, exist_ok=True)
        path = directory / f'{index:03d}.bin'
        self.gate._transport_metadata(index, request_sha256=hashlib.sha256(raw).hexdigest(),
            endpoint=str(url), response_path=str(path.relative_to(self.gate.output)))
        remaining = self.gate.summary()['remaining_seconds']
        timeout = dict(request.extensions.get('timeout') or {})
        for key in ('connect', 'read', 'write', 'pool'):
            previous = timeout.get(key)
            timeout[key] = min(float(previous), remaining) if previous is not None else remaining
        request.extensions['timeout'] = timeout
        try:
            response = self.inner.handle_request(request)
            self.gate._transport_metadata(index, http_status=response.status_code)
            if response.status_code >= 300:
                self.gate.stop('GATE_HTTP_STATUS_' + str(response.status_code))
            stream = _RecordedStream(response.stream, self.gate, index, path)
            return httpx.Response(response.status_code, headers=response.headers,
                                  stream=stream, extensions=response.extensions)
        except BaseException:
            self.gate.stop('GATE_TRANSPORT_FAILED')
            raise

    def close(self) -> None:
        """关闭底层连接池，不重试未结算请求。"""
        self.inner.close()
