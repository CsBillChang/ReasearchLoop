"""机制实验的物理发送前预算门；默认替身，真实模式须每次通过外部冻结授权。

复用本项目 299b881 的 harness_ab_transport.py 的压缩流与SSE完整性处理。
本轮测试只构造 MockTransport，不提供真实授权，不自动安装依赖或读取.env。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
import zlib

import httpx

from experiments.mechanism_pilot.budget import BudgetHalt
from research_loop.providers.deepseek import RequestNotSent


class QualificationTransport(httpx.BaseTransport):
    """将同一根任务及其子角色请求计入共享账本，保持请求正文原字节。"""

    def __init__(self, inner_factory, ledger, root_id, input_token_limit,
                 max_output_tokens, *, directory=None, mode='substitute',
                 authorize=None, freeze_sha256=None, cancelled=lambda: False,
                 expected_model='deepseek-flash'):
        if mode not in {'substitute', 'real'}:
            raise ValueError('unknown qualification transport mode')
        if (type(input_token_limit) is not int or input_token_limit <= 0
                or type(max_output_tokens) is not int or max_output_tokens <= 0):
            raise ValueError('positive frozen capacity/output bounds required')
        self.ledger, self.root_id = ledger, root_id
        self.input_token_limit, self.max_output_tokens = input_token_limit, max_output_tokens
        self.mode, self.authorize, self.freeze_sha256 = mode, authorize, freeze_sha256
        self._cancelled, self._cancel_check, self._closed = False, cancelled, False
        self.expected_model = expected_model
        self.directory = Path(directory) if directory else ledger.path.parent / ('http-' + root_id + '-' + uuid4().hex)
        self._active = set()
        self._grant = self._check_authorization()
        # 真实传输池工厂只会在显式授权及冻结身份检查通过后调用。
        self.inner = inner_factory()
        if mode == 'substitute' and not isinstance(self.inner, httpx.MockTransport):
            self.inner.close()
            raise PermissionError('QUALIFICATION_MOCK_TRANSPORT_REQUIRED')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.deadline = ledger.snapshot()['batch_absolute_deadline']

    def _check_authorization(self):
        if self.mode == 'substitute' and self.authorize is None:
            return {'allow_real_model': False, 'base_url': 'https://fixture.invalid',
                    'model': self.expected_model}
        if (not callable(self.authorize) or not isinstance(self.freeze_sha256, str)
                or len(self.freeze_sha256) != 64):
            raise PermissionError('QUALIFICATION_FRESH_AUTHORIZATION_REQUIRED')
        grant = self.authorize()
        if (not isinstance(grant, dict) or grant.get('allow_real_model') is not (self.mode == 'real')
                or grant.get('freeze_sha256') != self.freeze_sha256
                or grant.get('model') != self.expected_model
                or grant.get('base_url') != ('https://api.deepseek.com' if self.mode == 'real' else 'https://fixture.invalid')):
            raise PermissionError('QUALIFICATION_AUTHORIZATION_OR_FREEZE_MISMATCH')
        if self.mode == 'real' and self.input_token_limit != 1_048_576:
            raise PermissionError('QUALIFICATION_PROVIDER_CONTEXT_CEILING_REQUIRED')
        for key, expected in (('input_token_limit', self.input_token_limit),
                              ('max_output_tokens', self.max_output_tokens)):
            if key in grant and (type(grant[key]) is not int or grant[key] != expected):
                raise PermissionError('QUALIFICATION_REQUEST_CAPACITY_GRANT_MISMATCH')
        if ('budget_mode' in grant and grant['budget_mode']
                != self.ledger.config.get('budget_mode', 'bounded')):
            raise PermissionError('QUALIFICATION_LEDGER_MODE_GRANT_MISMATCH')
        if grant.get('budget_mode') == 'unbounded' and grant.get('hard_deadline') is not None:
            raise PermissionError('QUALIFICATION_UNBOUNDED_DEADLINE_MISMATCH')
        if hasattr(self, '_grant') and grant != self._grant:
            raise PermissionError('QUALIFICATION_AUTHORIZATION_CHANGED')
        return grant

    def cancelled(self):
        return self._cancelled or self._closed or bool(self._cancel_check())

    def cancel(self, reason='qualification cancelled'):
        """阻止新请求，已发送请求的完整迟到收据仍可正常结算。"""
        self._cancelled = True
        self.ledger.halt(reason, operation='cancel', root_id=self.root_id)

    def _unknown(self, identity, reason):
        row = next(r for r in self.ledger.snapshot()['reservations'] if r['reservation_id'] == identity)
        if row['status'] == 'settled':
            self.ledger.halt(reason, operation='after_settlement_error', root_id=self.root_id)
        else:
            self.ledger.unknown(identity, reason)
        self._active.discard(identity)

    def _write(self, path, value):
        """私有日志不记录请求认证头。"""
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')

    def _before_send(self, request):
        self._check_authorization()
        if self.cancelled():
            raise RequestNotSent('QUALIFICATION_CANCELLED_BEFORE_SEND')
        target = urlsplit(self._grant['base_url'])
        if (request.method != 'POST' or request.url.scheme != 'https'
                or request.url.host != target.hostname or request.url.port not in {None, 443}
                or request.url.path != '/chat/completions' or request.url.query):
            raise RequestNotSent('QUALIFICATION_ENDPOINT_OUTSIDE_FREEZE')
        raw = request.read()
        value = json.loads(raw)
        if (not isinstance(value, dict) or value.get('model') != self.expected_model
                or not isinstance(value.get('messages'), list) or not value['messages']
                or any(not isinstance(m, dict) or not isinstance(m.get('content'), (str, type(None)))
                       for m in value['messages'])
                or type(value.get('max_tokens')) is not int
                or not 0 < value['max_tokens'] <= self.max_output_tokens
                or type(value.get('stream', False)) is not bool):
            raise RequestNotSent('QUALIFICATION_TEXT_REQUEST_CONTRACT')
        if self.deadline is not None:
            remaining = self.deadline - self.ledger.clock()
            if remaining <= 0:
                raise RequestNotSent('QUALIFICATION_HARD_DEADLINE')
            existing = request.extensions.get('timeout') or {}
            request.extensions['timeout'] = {key: min(existing.get(key) or remaining, remaining)
                for key in ('connect', 'read', 'write', 'pool')}
        return value, raw

    def handle_request(self, request):
        """预留失败是确定未发送；进入inner后的未知与错误不可释放预留。"""
        identity, sent = None, False
        try:
            value, raw = self._before_send(request)
            reservation = self.ledger.reserve(self.root_id,
                input_upper_bound=self.input_token_limit, max_output_tokens=value['max_tokens'],
                request_metadata={'body_sha256': hashlib.sha256(raw).hexdigest(),
                    'model': value['model'], 'stream': value.get('stream', False),
                    'input_bound_basis': 'frozen_provider_context_hard_limit',
                    'mode': self.mode, 'freeze_sha256': self.freeze_sha256})
            identity = reservation['reservation_id']
            self._write(self.directory/(identity+'.request.json'), value)
            # 冻结漂移、取消及时间均在真正放行前再次校验。
            self._check_authorization()
            if self.cancelled():
                raise RequestNotSent('QUALIFICATION_CANCELLED_AFTER_RESERVE')
            self.ledger.mark_sent(identity)
            sent = True
            self._active.add(identity)
        except Exception as exc:
            if identity is not None and not sent:
                self.ledger.not_sent(identity, 'before_inner:' + type(exc).__name__)
            self.ledger.halt(type(exc).__name__ + ':' + str(exc),
                operation='before_inner', root_id=self.root_id)
            raise RequestNotSent('QUALIFICATION_PRE_SEND_DENIED:' + str(exc)) from exc
        try:
            response = self.inner.handle_request(request)
            encoding = response.headers.get('content-encoding')
            if encoding and encoding.lower() in {'gzip', 'deflate'}:
                response.stream = _CompressionCheckedStream(response.stream, encoding.lower())
            if response.is_error:
                response.read()
                (self.directory/(identity+'.error-body')).write_bytes(response.content)
                self._unknown(identity, 'provider_http_' + str(response.status_code))
                return response
            if value.get('stream', False):
                self._write(self.directory/(identity+'.response-meta.json'), {
                    'status_code': response.status_code, 'source_content_encoding': encoding})
                headers = httpx.Headers(response.headers)
                headers.pop('content-encoding', None)
                headers.pop('content-length', None)
                return httpx.Response(response.status_code, headers=headers,
                    stream=_UsageStream(response, self, identity),
                    extensions=response.extensions, request=request)
            response.read()
            (self.directory/(identity+'.response.json')).write_bytes(response.content)
            result = response.json()
            self.ledger.settle(identity, result.get('usage'), response_metadata={
                'response_id': result.get('id'), 'model': result.get('model'), 'stream': False})
            self._active.discard(identity)
            if self.cancelled():
                self.ledger.halt('cancelled after complete receipt', operation='late_receipt', root_id=self.root_id)
            return response
        except BaseException as exc:
            self._unknown(identity, 'inner_or_receipt:' + type(exc).__name__)
            raise

    def close(self):
        """收尾冻结未决在途请求；关闭不把unknown变成未发送。"""
        self._closed = True
        if self._active:
            self.ledger.halt('transport closed with pending responses', operation='cleanup', root_id=self.root_id)
        for identity in list(self._active):
            self._unknown(identity, 'transport_closed_before_complete_receipt')
        self.inner.close()


class _CompressionCheckedStream(httpx.SyncByteStream):
    """增量校验gzip/deflate帧完整性，原字节继续交由HTTPX公开API解码。"""

    def __init__(self, inner, encoding):
        self.inner = inner
        windows = [16 + zlib.MAX_WBITS] if encoding == 'gzip' else [zlib.MAX_WBITS, -zlib.MAX_WBITS]
        self.validators = [zlib.decompressobj(window) for window in windows]

    def __iter__(self):
        for chunk in self.inner:
            valid = []
            for validator in self.validators:
                try:
                    validator.decompress(chunk)
                except zlib.error:
                    continue
                valid.append(validator)
            self.validators = valid
            if not valid:
                raise httpx.DecodingError('mechanism compressed stream integrity failure')
            yield chunk
        for validator in self.validators:
            try:
                validator.flush()
            except zlib.error:
                continue
            if validator.eof and not validator.unused_data:
                return
        raise httpx.DecodingError('mechanism compressed stream ended before a complete frame')

    def close(self):
        self.inner.close()


class _UsageStream(httpx.SyncByteStream):
    """归档并转发解码后的SSE，完整解码结束且收到DONE/usage后结算。"""

    def __init__(self, inner, owner, identity):
        self.inner, self.owner, self.identity = inner, owner, identity
        self.buffer, self.usage, self.metadata = b'', None, {
            'stream': True, 'archive_body_encoding': 'httpx_decoded_bytes',
            'source_content_encoding': inner.headers.get('content-encoding')}
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
        terminal_chunks = []
        try:
            with path.open('xb') as output:
                for chunk in self.inner.iter_bytes():
                    if self.owner.cancelled():
                        raise RuntimeError('qualification stream cancelled before complete receipt')
                    if self.owner.deadline is not None and self.owner.ledger.clock() >= self.owner.deadline:
                        raise TimeoutError('mechanism stream wall deadline')
                    output.write(chunk)
                    output.flush()
                    self.buffer += chunk
                    while b'\n' in self.buffer:
                        line, self.buffer = self.buffer.split(b'\n', 1)
                        self._line(line.rstrip(b'\r'))
                    if self.done:
                        # 供应商适配器见DONE即停止迭代；先等解码器EOF校验，避免漏记或接受截断压缩流。
                        terminal_chunks.append(chunk)
                    else:
                        yield chunk
            if self.buffer:
                self._line(self.buffer.rstrip(b'\r'))
            if not self.done:
                raise RuntimeError('stream ended without DONE')
            self.owner.ledger.settle(self.identity, self.usage, response_metadata=self.metadata)
            self.finished = True
            self.owner._active.discard(self.identity)
            if self.owner.cancelled():
                self.owner.ledger.halt('cancelled after complete stream receipt', operation='late_receipt', root_id=self.owner.root_id)
            yield from terminal_chunks
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
