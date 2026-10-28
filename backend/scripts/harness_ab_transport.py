"""在实际 HTTP 边界预留预算，完整收据到达后结算；不重试未知请求。"""
from __future__ import annotations

import hashlib
import base64
import json
from pathlib import Path
import struct
import time
import zlib

import httpx

from research_loop.providers.deepseek import RequestNotSent


def require_generation_identity(ledger, root_id, check, *, operation):
    """身份门在HTTP传输之前执行，也必须通过共同账本冻结首错。"""
    try:
        return check()
    except Exception as exc:
        reason = type(exc).__name__ + ': ' + str(exc)
        ledger.halt(reason, operation=operation, root_id=root_id)
        raise RequestNotSent('H1 generation identity rejected: ' + reason) from exc


def visual_request_contract():
    """返回当前业务视觉节点的窄合同；不默认为实验开启图片能力。"""
    return {'version': 1, 'model': 'deepseek-flash', 'max_tokens': 8192,
        'temperature': 0.1, 'thinking': {'type': 'disabled'}, 'stream': False,
        'max_images': 2, 'mime': 'image/png', 'detail': 'original',
        'max_image_bytes': 32 * 1024 * 1024, 'max_image_side': 8192,
        'max_request_bytes': 48 * 1024 * 1024, 'input_upper_bound': 1_048_576}


def validate_visual_contract(contract):
    """仅接受显式冻结的同一合同，禁止通过配置扩大模型或图片范围。"""
    if contract is not None and contract != visual_request_contract():
        raise ValueError('visual contract differs from current business component')
    return visual_request_contract() if contract is not None else None


def validate_visual_payload(payload, body_size, contract):
    """校验原生节点的图片形状，保持原始请求字节与来源授权路径不变。"""
    if contract is None:
        raise RequestNotSent('visual input has no frozen experiment allowance')
    allowed = {'model', 'messages', 'temperature', 'max_tokens', 'thinking', 'stream'}
    if (set(payload) - allowed or body_size > contract['max_request_bytes']
            or any(payload.get(k) != contract[k] for k in
                ('model', 'temperature', 'max_tokens', 'thinking'))
            or payload.get('stream', False) is not False):
        raise RequestNotSent('visual parameters differ from frozen component contract')
    messages = payload.get('messages')
    if (not isinstance(messages, list) or len(messages) != 2
            or any(not isinstance(message, dict) for message in messages)
            or messages[0].get('role') != 'system' or not isinstance(messages[0].get('content'), str)
            or messages[1].get('role') != 'user'):
        raise RequestNotSent('visual messages differ from frozen component contract')
    blocks = messages[1].get('content')
    if (not isinstance(blocks, list) or not 2 <= len(blocks) <= 1 + contract['max_images']
            or any(not isinstance(block, dict) for block in blocks)
            or blocks[0].get('type') != 'text' or not isinstance(blocks[0].get('text'), str)
            or not blocks[0]['text']):
        raise RequestNotSent('visual text or image count outside frozen scope')
    import fitz
    for block in blocks[1:]:
        value = block.get('image_url') if block.get('type') == 'image_url' else None
        if (not isinstance(value, dict) or set(value) != {'url', 'detail'}
                or value['detail'] != contract['detail'] or not isinstance(value['url'], str)
                or not value['url'].startswith('data:image/png;base64,')):
            raise RequestNotSent('only current user inline PNG original images are allowed')
        encoded = value['url'].partition(',')[2]
        if len(encoded) > 4 * ((contract['max_image_bytes'] + 2) // 3):
            raise RequestNotSent('visual image exceeds frozen byte limit')
        try:
            raw = base64.b64decode(encoded, validate=True)
            if (len(raw) < 33 or len(raw) > contract['max_image_bytes']
                    or raw[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'):
                raise ValueError('image bytes or PNG header')
            width, height = struct.unpack('>II', raw[16:24])
            if min(width, height) <= 0 or max(width, height) > contract['max_image_side']:
                raise ValueError('image dimensions')
            # 先校验IHDR上界再解码；复用现有渲染依赖，不安装或改变部署依赖。
            picture = fitz.Pixmap(raw)
            if (picture.width, picture.height) != (width, height):
                raise ValueError('decoded image dimensions differ')
        except Exception as exc:
            # PyMuPDF各版本的解码异常类型不同；这里只包住发送前的本地图片检查。
            raise RequestNotSent('invalid or out-of-scope visual PNG') from exc
    return len(blocks) - 1


class BudgetTransport(httpx.BaseTransport):
    """约束一个根任务族的所有原生生成请求，不改变消息或工具内容。"""

    def __init__(self, ledger, root_id, directory: Path, *, inner=None,
                 expected_model='deepseek-flash', deadline=None, visual_contract=None):
        """绑定共同账本、根身份和私有请求记录目录。

        Args:
            ledger: 持久且线程安全的共同预算账本。
            root_id: 真实根任务 ID，子角色均归入此身份。
            directory: 本格独立输出目录，不保存认证请求头。
            inner: 测试可注入内存传输，真实模式默认零重试传输。
            expected_model: 冻结的模型别名。
            deadline: 可选绝对墙钟截止时间。
            visual_contract: 显式冻结的视觉合同；未提供时保持禁止图片。
        """
        self.ledger, self.root_id = ledger, root_id
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.inner = inner or httpx.HTTPTransport(retries=0, trust_env=False)
        self.expected_model, self.deadline = expected_model, deadline
        self.visual_contract = validate_visual_contract(visual_contract)

    def _unknown(self, identity, reason):
        """保留完整预留并冻结后续真实请求。"""
        self.ledger.unknown(identity, reason)

    def _write(self, path, value):
        """私有记录只写请求正文和响应元数据，不记录凭据。"""
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def handle_request(self, request):
        """任何本地拒绝都持久冻结整批；此前已放行的在途请求仍可结算。"""
        try:
            return self._handle_request(request)
        except Exception as exc:
            self.ledger.halt(type(exc).__name__ + ': ' + str(exc),
                operation='request_rejected', root_id=self.root_id)
            raise

    def _handle_request(self, request):
        """仅向冻结生成入口发送，预留失败转换为确定未发送。"""
        if (request.method != 'POST' or request.url.scheme != 'https'
                or request.url.host != 'api.deepseek.com'
                or request.url.port not in {None, 443}
                or request.url.path != '/chat/completions'
                or request.url.query):
            raise RequestNotSent('H1 endpoint outside frozen generation scope')
        body = request.read()
        payload = json.loads(body)
        if (not isinstance(payload, dict) or not isinstance(payload.get('messages', []), list)
                or any(not isinstance(message, dict) for message in payload.get('messages', []))):
            raise RequestNotSent('invalid experiment message envelope')
        image_blocks = []
        for message in payload.get('messages', []):
            content = message.get('content')
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict) or item.get('type') not in {'text', 'image_url'}:
                        raise RequestNotSent('multimodal block outside frozen inline PNG scope')
                    if item['type'] == 'image_url':
                        image_blocks.append(item)
        image_count = 0
        if image_blocks:
            image_count = validate_visual_payload(payload, len(body), self.visual_contract)
        elif (payload.get('temperature') != 0.1 or payload.get('thinking') is not None):
            raise RequestNotSent('H1 model parameters differ from freeze')
        if payload.get('model') != self.expected_model:
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
                    'body_sha256': hashlib.sha256(body).hexdigest(),
                    'request_kind': 'visual' if image_count else 'text', 'image_count': image_count})
        except Exception as exc:
            raise RequestNotSent('H1 budget denied before transport: '
                + type(exc).__name__ + ': ' + str(exc)) from exc
        identity = reservation['reservation_id']
        try:
            self._write(self.directory / (identity + '.request.json'), payload)
        except BaseException as exc:
            self.ledger.not_sent(identity, 'request_journal_failed_before_transport: '
                + type(exc).__name__ + ': ' + str(exc))
            raise
        try:
            self.ledger.mark_sent(identity)
        except Exception as exc:
            self.ledger.not_sent(identity, 'mark_sent_denied_before_transport: '
                + type(exc).__name__ + ': ' + str(exc))
            raise RequestNotSent('H1 send denied before transport: '
                + type(exc).__name__ + ': ' + str(exc)) from exc
        try:
            response = self.inner.handle_request(request)
            encoding = response.headers.get('content-encoding')
            if encoding and encoding.lower() in {'gzip', 'deflate'}:
                # 两种角色响应均先核对压缩帧EOF，JSON解出完整正文也不能替代缺尾检查。
                response.stream = _CompressionCheckedStream(response.stream, encoding.lower())
            if response.is_error:
                response.read()
                (self.directory / (identity + '.error-body')).write_bytes(response.content)
                self._unknown(identity, 'provider_http_' + str(response.status_code))
                return response
            if payload.get('stream'):
                self._write(self.directory / (identity + '.response-meta.json'), {
                    'status_code': response.status_code,
                    'archive_body_encoding': 'httpx_decoded_bytes',
                    'source_content_encoding': encoding})
                headers = httpx.Headers(response.headers)
                headers.pop('content-encoding', None)
                headers.pop('content-length', None)
                # 内层以公开API完成解压；外层收到解码字节，不能再次按原编码解压。
                return httpx.Response(response.status_code, headers=headers,
                    stream=_UsageStream(response, self, identity),
                    extensions=response.extensions, request=request)
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
                raise httpx.DecodingError('H1 compressed stream integrity failure')
            yield chunk
        for validator in self.validators:
            try:
                validator.flush()
            except zlib.error:
                continue
            if validator.eof and not validator.unused_data:
                return
        raise httpx.DecodingError('H1 compressed stream ended before a complete frame')

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
                    if self.owner.deadline is not None and time.time() >= self.owner.deadline:
                        raise TimeoutError('H1 stream wall deadline')
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
