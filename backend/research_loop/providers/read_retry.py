"""论文只读请求退避策略；不包装模型调用或有副作用的写接口。"""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import random
import time
import httpx


def retry_after(value: str | None, now: datetime) -> float | None:
    """解析HTTP秒数或日期；非法值回退指数退避，已过去的日期允许立即尝试。"""
    if not value:
        return None
    try:
        if value.strip().isdigit():
            return float(int(value))
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0, (date - now).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return None


def read_with_backoff(send, request: dict, *, attempts: int = 4, base: float = 1,
                      max_wait: float = 60, sleep=None, jitter=None, now=None):
    """执行显式只读回调，记录尝试与等待；超出等待预算不提前重试。

    Args:
        send: 无参数的只读HTTP回调；单次网络超时由调用方限定。
        request: 逐次诊断输出，不含认证头或请求正文。
        attempts: 最大物理尝试数，含首次，最多4次。
        base: 指数退避初始秒数，0用于测试或上层单次调度。
        max_wait: 整次请求的累计等待上限，不包含网络处理时间。
        sleep: 可注入等待函数。
        jitter: 可注入[0,1]随机值。
        now: 可注入UTC时钟，便于日期边界测试。

    Returns:
        最后一份HTTP响应；调用方仍需按HTTP状态报告失败。

    Raises:
        httpx.RequestError: 最后一次传输失败，无响应可返回。
    """
    sleep, jitter = sleep or time.sleep, jitter or random.random
    now = now or (lambda: datetime.now(timezone.utc))
    request['attempts'] = []
    waited = 0.0
    cap = max(1, min(4, attempts))
    for number in range(1, cap + 1):
        record = {'attempt': number}
        request['attempts'].append(record)
        request.pop('http_status', None)
        request.pop('response_content_type', None)
        response, failure = None, None
        try:
            response = send()
            code = response.status_code
            record['http_status'] = code
            request.update(http_status=code, response_content_type=response.headers.get('content-type'))
            try:
                body = response.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and str(body.get('code')) == '57014':
                request['retry_stopped_reason'] = 'database_statement_timeout'
                return response
            if code != 429 and not 500 <= code < 600:
                request['retry_stopped_reason'] = 'success' if code < 400 else 'non_retryable_http'
                return response
        except httpx.RequestError as exc:
            failure = exc
            record['error'] = type(exc).__name__
        if number == cap:
            request['retry_stopped_reason'] = 'attempt_limit'
        else:
            header = response.headers.get('retry-after') if response is not None else None
            specified = retry_after(header, now())
            delay = specified if specified is not None else max(0, base) * 2 ** (number - 1) * (0.5 + 0.5 * jitter())
            record.update(wait_seconds=delay, wait_source='retry_after' if specified is not None else 'exponential_jitter')
            if waited + delay <= max_wait:
                sleep(delay)
                waited += delay
                request['total_wait_seconds'] = waited
                continue
            request['retry_stopped_reason'] = 'retry_after_exceeds_wait_budget' if specified is not None else 'wait_budget'
        if failure:
            raise failure
        return response
