"""在实际 HTTP 边界预留预算，完整收据到达后结算；不重试未知请求。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import httpx

from research_loop.providers.deepseek import RequestNotSent


class BudgetTransport(httpx.BaseTransport):
    """约束一个根任务族的所有原生生成请求，不改变消息或工具内容。"""

    def __init__(self, ledger, root_id, directory: Path, *, inner=None,
                 expected_model='deepseek-flash', deadline=None):
        """绑定共同账本、根身份和私有请求记录目录。

        Args:
            ledger: 持久且线程安全的共同预算账本。
            root_id: 真实根任务 ID，子角色均归入此身份。
            directory: 本格独立输出目录，不保存认证请求头。
            inner: 测试可注入内存传输，真实模式默认零重试传输。
            expected_model: 冻结的模型别名。
            deadline: 可选绝对墙钟截止时间。
        """
        self.ledger, self.root_id = ledger, root_id
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.inner = inner or httpx.HTTPTransport(retries=0, trust_env=False)
        self.expected_model, self.deadline = expected_model, deadline

    def _unknown(self, identity, reason):
        """保留完整预留并冻结后续真实请求。"""
        self.ledger.unknown(identity, reason)

    def _write(self, path, value):
        """私有记录只写请求正文和响应元数据，不记录凭据。"""
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def handle_request(self, request):
        """仅向冻结生成入口发送，预留失败转换为确定未发送。"""
        if (request.method != 'POST' or request.url.scheme != 'https'
                or request.url.host != 'api.deepseek.com'
                or request.url.port not in {None, 443}
                or request.url.path != '/chat/completions'
                or request.url.query):
            raise RequestNotSent('H1 endpoint outside frozen generation scope')
        payload = json.loads(request.read())
        for message in payload.get('messages', []):
            content = message.get('content')
            if isinstance(content, list) and any(isinstance(item, dict) and item.get('type') in
                    {'image_url', 'input_image', 'image'} for item in content):
                raise RequestNotSent('H1 visual input has no approved allowance')
        if (payload.get('model') != self.expected_model
                or payload.get('temperature') != 0.1 or payload.get('thinking') is not None):
            raise RequestNotSent('H1 model parameters differ from freeze')
        maximum = payload.get('max_tokens')
        if not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= 65536:
            raise RequestNotSent('H1 explicit bounded max_tokens required')
        if self.deadline is not None:
            remaining = self.deadline - time.time()
            if remaining <= 1:
                raise RequestNotSent('H1 wall deadline reached')
            current = request.extensions.get('timeout', {})
            request.extensions['timeout'] = {key: min(current.get(key) or remaining, remaining)
                for key in ('connect', 'read', 'write', 'pool')}
        try:
            reservation = self.ledger.reserve(self.root_id, max_output_tokens=maximum,
                input_upper_bound=1048576,
                request_metadata={'model': payload['model'], 'stream': bool(payload.get('stream')),
                    'body_sha256': hashlib.sha256(request.content).hexdigest()})
        except Exception as exc:
            raise RequestNotSent('H1 budget denied before transport: ' + type(exc).__name__) from exc
        identity = reservation['reservation_id']
        try:
            self._write(self.directory / (identity + '.request.json'), payload)
        except BaseException:
            self.ledger.not_sent(identity, 'request_journal_failed_before_transport')
            raise
        try:
            self.ledger.mark_sent(identity)
        except Exception as exc:
            self.ledger.not_sent(identity, 'mark_sent_denied_before_transport')
            raise RequestNotSent('H1 send denied before transport: ' + type(exc).__name__) from exc
        try:
            response = self.inner.handle_request(request)
            if response.is_error:
                response.read()
                (self.directory / (identity + '.error-body')).write_bytes(response.content)
                self._unknown(identity, 'provider_http_' + str(response.status_code))
                return response
            if payload.get('stream'):
                response.stream = _UsageStream(response.stream, self, identity)
                return response
            response.read()
            (self.directory / (identity + '.response.json')).write_bytes(response.content)
            raw = response.json()
            self.ledger.settle(identity, raw.get('usage'), response_metadata={
                'response_id': raw.get('id'), 'model': raw.get('model'), 'stream': False})
            return response
        except BaseException as exc:
            self._unknown(identity, 'transport_or_receipt_' + type(exc).__name__)
            raise

    def close(self):
        """关闭本传输持有的连接池。"""
        self.inner.close()


class _UsageStream(httpx.SyncByteStream):
    """按原字节转发 SSE，并要求 DONE 和完整 usage 同时到达。"""

    def __init__(self, inner, owner, identity):
        self.inner, self.owner, self.identity = inner, owner, identity
        self.buffer, self.usage, self.metadata = b'', None, {'stream': True}
        self.done = self.finished = False

    def _line(self, line):
        """处理任意网络分块边界，拒绝变化的重复 usage。"""
        if not line.startswith(b'data:'):
            return
        data = line[5:].strip()
        if data == b'[DONE]':
            self.done = True
            return
        if not data:
            return
        raw = json.loads(data)
        if raw.get('id'):
            self.metadata['response_id'] = raw['id']
        if raw.get('model'):
            previous = self.metadata.get('model')
            if previous is not None and previous != raw['model']:
                raise ValueError('response model changed inside stream')
            self.metadata['model'] = raw['model']
        if raw.get('usage') is not None:
            if self.usage is not None and self.usage != raw['usage']:
                raise ValueError('conflicting stream usage')
            self.usage = raw['usage']

    def __iter__(self):
        path = self.owner.directory / (self.identity + '.response.sse')
        try:
            with path.open('xb') as output:
                for chunk in self.inner:
                    if self.owner.deadline is not None and time.time() >= self.owner.deadline:
                        raise TimeoutError('H1 stream wall deadline')
                    output.write(chunk)
                    output.flush()
                    self.buffer += chunk
                    while b'\n' in self.buffer:
                        line, self.buffer = self.buffer.split(b'\n', 1)
                        self._line(line.rstrip(b'\r'))
                    if self.done:
                        self.owner.ledger.settle(self.identity, self.usage, response_metadata=self.metadata)
                        self.finished = True
                    yield chunk
                    if self.done:
                        return
            if self.buffer:
                self._line(self.buffer.rstrip(b'\r'))
            if not self.done:
                raise RuntimeError('stream ended without DONE')
            self.owner.ledger.settle(self.identity, self.usage, response_metadata=self.metadata)
            self.finished = True
        except BaseException as exc:
            if not self.finished:
                self.owner._unknown(self.identity, 'stream_' + type(exc).__name__)
            raise

    def close(self):
        """调用方中途结束也保持未知预留，不能当作免费取消。"""
        try:
            self.inner.close()
        finally:
            if not self.finished:
                self.owner._unknown(self.identity, 'stream_closed_before_complete_receipt')
