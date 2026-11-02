"""Supervisor 上下文：注入目标、附件和保存权限；工具观察会压缩后进入下一次决策。"""
from __future__ import annotations
import json
from datetime import date
from typing import Any


from research_loop.agents.prompt_parts import COMMON_PRINCIPLES, SUPERVISOR_ROLE, compose_prompt


SYSTEM_PROMPT = compose_prompt(COMMON_PRINCIPLES, SUPERVISOR_ROLE)

def initial_messages(prompt: str, save_policy: str, attachments: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """把自然语言目标、保存权限与附件定位组装为初始消息；不在程序中替模型预选论文。"""
    refs = ", ".join(sorted((attachments or {}).keys())) or "无"
    today = date.today()
    try:
        three_years_ago = today.replace(year=today.year - 3)
    except ValueError:
        three_years_ago = today.replace(year=today.year - 3, day=28)
    return [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n运行日期={today.isoformat()}；近三年滚动窗口={three_years_ago.isoformat()} 至 {today.isoformat()}（不是三年前的1月1日）。"},
        {"role": "user", "content": "任务：" + chr(10) + prompt + "；后端保存权限：" + save_policy
         + "；当前已授权附件 input_ref：" + refs
         + "。若任务指向附件，必须使用该 input_ref，可通过 delegate_research 的 input_refs 交给子 Agent 读取，不能改为搜索替换附件。"},
    ]


def compact_for_model(value: Any, limit: int = 18000) -> Any:
    """压缩普通工具元数据，同时完整保留权威页正文。

    Args:
        value: 待写入模型消息的工具观察。
        limit: 普通长字符串的字符阈值，不作用于页证据。

    Returns:
        可 JSON 序列化的观察；页证据中的原文、Markdown、定位和缺口保持原样。
    """
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + chr(10) + "[observation truncated; page range remains authoritative]"
    if isinstance(value, list):
        return [compact_for_model(item, max(2000, limit // max(len(value), 1))) for item in value]
    if isinstance(value, dict):
        if "subtask_id" in value and "findings" in value and "evidence_ids" in value:
            return value  # 论文 产物必须完整交给汇总者，不能再按元数据字符串截断。
        if "evidence_id" in value and "page_number" in value and ("text" in value or "parsed_markdown" in value):
            return value
        return {key: compact_for_model(item, limit) for key, item in value.items()}
    return value


def append_tool_observation(messages: list[dict[str, Any]], call_id: str, observation: dict[str, Any]) -> None:
    """使用原 tool_call_id 配对工具回复，后续模型调用依赖这份完整的调用/观察序列。"""
    messages.append({"role": "tool", "tool_call_id": call_id,
                     "content": json.dumps(compact_for_model(observation), ensure_ascii=False)})
