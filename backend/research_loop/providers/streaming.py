"""Chat Completions 流装配和公开文本提取；不把内部推理或工具参数整体发给浏览器。"""
from __future__ import annotations
import json
import re
from typing import Callable


def string_prefix(arguments: str, field: str) -> str:
    """读取尚未闭合JSON中的字符串前缀，截断转义等待后续delta补齐。

    Args:
        arguments: 工具调用累计JSON参数。
        field: 白名单中的正文属性。

    Returns:
        可完整解码的正文前缀；不执行或修复工具调用。
    """
    match = re.search(r'(?:^\s*\{|,)\s*"'+field+r'"\s*:\s*"((?:[^"\\]|\\.)*)', arguments, re.S)
    if not match:
        return ''
    value = match[1]
    for drop in range(min(7, len(value)+1)):
        try:
            return json.loads('"'+(value[:-drop] if drop else value)+'"')
        except (ValueError, json.JSONDecodeError):
            continue
    return ''


class StreamAssembly:
    """将真实delta还原为普通响应，公开正文以快照方式持久化，重连不会重复拼接。"""

    def __init__(self, publish: Callable[[dict], None]):
        self.publish = publish
        self.content = ''
        self.reasoning = ''
        self.calls = {}
        self.raw = {}
        self.finish = None
        self.last = {}

    def feed(self, raw: dict) -> None:
        """装配供应商字段；仅投影最终回答/草稿正文，不公开reasoning_content。"""
        for key in ('id', 'model', 'created', 'usage'):
            if raw.get(key) is not None:
                self.raw[key] = raw[key]
        for choice in raw.get('choices', []):
            if choice.get('index', 0) != 0:
                continue
            delta = choice.get('delta') or {}
            self.content += delta.get('content') or ''
            self.reasoning += delta.get('reasoning_content') or ''
            self.finish = choice.get('finish_reason') or self.finish
            for call in delta.get('tool_calls') or []:
                index = call['index']
                item = self.calls.setdefault(index, {'id':'', 'type':'function', 'function':{'name':'','arguments':''}})
                if call.get('id'):
                    item['id'] = call['id']
                for key in ('name','arguments'):
                    item['function'][key] += (call.get('function') or {}).get(key) or ''
            self.flush(final=bool(self.finish))

    def flush(self, final=False) -> None:
        """只在增加足够文本或收尾时写入；内容为生成中预览，终态以业务结果覆盖。"""
        targets = {}
        # 混合工具决策时的assistant旁白不是最终答案，因此只公布无工具的content。
        if not self.calls and self.content:
            targets['answer'] = self.content
        for index, call in self.calls.items():
            name = call['function']['name']
            field = {'finish_task':'answer_markdown', 'create_note':'markdown'}.get(name)
            if field:
                targets[str(index)+':'+name] = string_prefix(call['function']['arguments'],field)
        for key, value in targets.items():
            previous = self.last.get(key, '')
            if value and value != previous and (final or not previous or len(value)-len(previous)>=128):
                self.publish({'stream_key':key, 'text':value, 'provisional':True})
                self.last[key] = value

    def response(self) -> dict:
        """流结束后还原完整收据；usage必须存在，否则不能假装用量已知。"""
        if not self.finish or not self.raw.get('usage'):
            raise RuntimeError('incomplete model stream or missing usage; no automatic resend')
        self.flush(final=True)
        message = {'role':'assistant','content':self.content}
        if self.reasoning:
            message['reasoning_content'] = self.reasoning
        if self.calls:
            message['tool_calls'] = [self.calls[i] for i in sorted(self.calls)]
        return {**self.raw,'choices':[{'index':0,'finish_reason':self.finish,'message':message}]}
