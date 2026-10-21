"""DeepSeek独立Search适配；仅返回题录线索，不产生原文证据。"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit
import httpx

from research_loop.providers.paper_acquisition import PaperAcquisitionError


def public_http_url(value: str) -> str:
    """验证HTTP地址和凭据边界；拒绝本机/私网字面地址。"""
    import ipaddress
    if not isinstance(value, str) or not value.strip() or len(value) > 8192:
        raise ValueError('web URL must contain 1..8192 characters')
    parsed = urlsplit(value)
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or any(ord(char) < 33 for char in value)):
        raise ValueError('web URL requires HTTP(S) host without embedded credentials')
    if parsed.hostname.lower() in {'localhost', 'localhost.localdomain'} or parsed.hostname.lower().endswith('.local'):
        raise ValueError('web URL must not target local services')
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError('web URL must not target private addresses')
    return value


class WebSearchProvider:
    """发送一次Search/Fetch HTTP请求；付费请求不沿用免费只读重试。"""

    operation = 'search'

    def __init__(self, api_key: str | None, base_url: str = 'https://api.deepseek.com',
                 timeout: float = 30.0, client: httpx.Client | None = None,
                 max_response_bytes: int = 2 * 1024 * 1024):
        """注入共享密钥与HTTP客户端；密钥不写诊断或持久收据。"""
        self.api_key = api_key
        base = base_url.rstrip('/')
        self.url = base + ('/tools/' if base.endswith('/v1') else '/v1/tools/') + self.operation
        self.timeout = timeout
        if max_response_bytes < 1:
            raise ValueError('web response byte resource guard must be positive')
        self.max_response_bytes = max_response_bytes
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=False)
        self._owns_client = client is None

    def close(self):
        """关闭本组件创建的HTTP连接。"""
        if self._owns_client:
            self.client.close()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发送有界单次请求，错误正文及认证头不向模型或日志泄漏。"""
        if not self.api_key:
            raise PaperAcquisitionError('WEB_NOT_CONFIGURED', '网页工具未配置认证信息')
        try:
            with self.client.stream('POST', self.url, json=payload,
                                    headers={'Authorization': 'Bearer ' + self.api_key}, timeout=self.timeout) as response:
                if response.status_code != 200:
                    raise PaperAcquisitionError('WEB_HTTP_ERROR', f'网页工具返回HTTP {response.status_code}；不自动重发',
                                                acquisition={'http_status': response.status_code, 'operation': self.operation})
                chunks, size = [], 0
                for part in response.iter_bytes():
                    size += len(part)
                    if size > self.max_response_bytes:
                        raise PaperAcquisitionError('WEB_RESPONSE_TOO_LARGE', '网页工具响应超过配置的网络字节保护；不自动重发',
                                                    acquisition={'max_response_bytes': self.max_response_bytes})
                    chunks.append(part)
                import json
                body = json.loads(b''.join(chunks))
                self.validate_response(body)
                return body
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            if isinstance(exc, PaperAcquisitionError):
                raise
            raise PaperAcquisitionError('WEB_REQUEST_UNKNOWN', '网页请求未确认完整结果；不自动重发',
                                        acquisition={'error_type': type(exc).__name__}) from exc

    @staticmethod
    def payload(query: str, count: int = 10) -> dict[str, Any]:
        """按官方128个Unicode字符和50条上限校验，不静默截断原查询。"""
        if not isinstance(query, str) or not query.strip() or not 1 <= len(query) <= 128:
            raise ValueError('search query must contain 1..128 Unicode characters')
        if type(count) is not int or not 1 <= count <= 50:
            raise ValueError('search count must be an integer in 1..50')
        return {'query': query, 'count': count, 'need_content': False}

    @staticmethod
    def validate_response(body):
        """拒绝错误JSON形状，不把摘要字段解释成原文。"""
        if not isinstance(body, dict) or not isinstance(body.get('results'), list):
            raise ValueError('invalid search response')
        if any(not isinstance(item, dict) or not isinstance(item.get('url'), str)
               or not isinstance(item.get('title'), str) for item in body['results']):
            raise ValueError('invalid search result')
