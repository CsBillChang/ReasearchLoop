"""DeepSeek Fetch适配：Markdown仅供查找PDF链接，不导入原文证据。"""
from research_loop.providers.web_search import WebSearchProvider, public_http_url


class WebFetchProvider(WebSearchProvider):
    """沿用单次请求的连接与安全错误，不重试独立计费请求。"""

    operation = 'fetch'

    @staticmethod
    def payload(url: str) -> dict:
        """只接受带主机且不含内嵌凭据的HTTP地址。"""
        return {'url': public_http_url(url)}

    @staticmethod
    def validate_response(body):
        """校验官方标题/Markdown字段，不检查或声明全文覆盖。"""
        if not isinstance(body, dict) or not isinstance(body.get('title'), str) or not isinstance(body.get('content'), str):
            raise ValueError('invalid fetch response')
