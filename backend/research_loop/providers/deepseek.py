"""模型适配器：发送兼容 Chat Completions 的请求，保留原始响应；解析与收据持久化分别处理。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx


class InvalidConversation(ValueError):
    """Locally invalid tool history; no HTTP request was sent."""


class RequestNotSent(RuntimeError):
    """发送前的确定性拒绝；仅适配器能保证尚未进入网络发送时使用。

    超时、断线及已取得错误响应不能使用本类型，仍须保留远端未知或拒绝记录。
    """


class ProviderResponseError(RuntimeError):
    """The server replied explicitly; this is not a lost remote response."""

    def __init__(self, status_code: int, body: str, request_id: str | None = None):
        super().__init__(f"provider HTTP {status_code}: {body[:1000]}")
        self.status_code = status_code
        self.body = body[:4096]
        self.request_id = request_id

    def as_record(self) -> dict[str, Any]:
        return {"http_status": self.status_code, "body": self.body,
                "request_id": self.request_id, "usage_known": False}


def validate_messages(messages: list[dict[str, Any]]) -> None:
    """在发 HTTP 前检查 assistant tool_calls 与 tool 回复逐一配对；不完整对话直接拒绝。"""
    pending: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise InvalidConversation("tool response has no matching pending call")
            pending.remove(call_id)
            continue
        if pending:
            raise InvalidConversation("missing tool responses before the next message")
        calls = message.get("tool_calls") or []
        ids = [call.get("id") for call in calls]
        if calls and (role != "assistant" or any(not value for value in ids) or len(set(ids)) != len(ids)):
            raise InvalidConversation("invalid or duplicate tool call identity")
        pending.update(ids)
    if pending:
        raise InvalidConversation("conversation ends with unanswered tool calls")


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    reasoning_content: str | None = None


class DeepSeekProvider:
    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com", timeout: float = 90.0,
                 client: httpx.Client | None = None):
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not configured")
        self.api_key = api_key
        self.model = model
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    @staticmethod
    def parse_response(raw: dict[str, Any], request_id: str | None = None,
                       usage: dict[str, Any] | None = None,
                       finish_reason: str | None = None) -> ModelResponse:
        """从已取得的原始响应解析内容和工具参数；不发网络请求，因此可用于成功收据恢复。"""
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                parsed = {"_invalid_json": arguments}
            calls.append(ToolCall(call_id=raw_call.get("id") or str(uuid4()),
                                  name=function.get("name") or "", arguments=parsed))
        return ModelResponse(content=message.get("content"), tool_calls=calls,
                             finish_reason=finish_reason if finish_reason is not None else choice.get("finish_reason"),
                             raw=raw, usage=usage if usage is not None else (raw.get("usage") or {}),
                             request_id=request_id if request_id is not None else raw.get("id"),
                             reasoning_content=message.get("reasoning_content"))

    def send(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             force_text: bool = False, *, max_tokens: int | None = None,
             temperature: float = 0.1, thinking: dict[str, str] | None = None) -> dict[str, Any]:
        """发送一次请求并返回原始 JSON；显式 HTTP 错误与响应丢失分开处理，不在适配器中自动重试。"""
        validate_messages(messages)
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if thinking is not None:
            payload["thinking"] = thinking
        if tools and not force_text:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        response = self.client.post(self.endpoint, headers={
            "Authorization": "Bearer " + self.api_key, "Content-Type": "application/json",
        }, json=payload)
        request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        if response.is_error:
            raise ProviderResponseError(response.status_code, response.text.replace(self.api_key, "[redacted]"), request_id)
        try:
            raw = response.json()
            if not isinstance(raw, dict):
                raise ValueError("response must be an object")
            return raw
        except ValueError as exc:
            raise ProviderResponseError(response.status_code, response.text.replace(self.api_key, "[redacted]"), request_id) from exc

    def send_stream(self, messages, tools=None, force_text=False, *, on_public, max_tokens=None, thinking=None):
        """发送一次真实SSE请求；断流/缺usage抛错，由现有收据层记录unknown。

        Args:
            messages: 冻结后的模型消息。
            tools: 当前角色的工具schema。
            force_text: 是否强制文字收尾。
            on_public: 公开正文快照回调，不包含内部推理。
            max_tokens: 当前请求的输出预算。
            thinking: 当前请求显式推理配置；None 保持原供应商默认，不改变全局模式。

        Returns:
            与非流式send相同结构的完整响应，调用方先持久化再解析。
        """
        from research_loop.providers.streaming import StreamAssembly
        validate_messages(messages)
        payload = {"model":self.model,"messages":messages,"temperature":0.1,
                   "stream":True,"stream_options":{"include_usage":True}}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if thinking is not None:
            payload["thinking"] = thinking
        if tools and not force_text:
            payload.update(tools=tools,tool_choice="auto")
        assembly = StreamAssembly(on_public)
        done = False
        with self.client.stream("POST",self.endpoint,headers={"Authorization":"Bearer "+self.api_key},json=payload) as response:
            if response.is_error:
                response.read()
                raise ProviderResponseError(response.status_code,response.text.replace(self.api_key,"[redacted]"),response.headers.get("x-request-id"))
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data=line[5:].strip()
                if data == "[DONE]":
                    done=True
                    break
                if data:
                    assembly.feed(json.loads(data))
        if not done:
            raise RuntimeError("model stream disconnected before DONE; no automatic resend")
        return assembly.response()

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                 force_text: bool = False) -> ModelResponse:
        """直接调用的便捷入口；持久化图使用 send 后先落收据，再调用 parse_response。"""
        raw = self.send(messages, tools, force_text)
        return self.parse_response(raw)
