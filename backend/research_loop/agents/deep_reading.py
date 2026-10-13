"""DPR 精读单节点：上游生成函数不改提示与循环，外层负责已有收据和取消。

精读/发布行为对齐 DPR long-range 严格阅读入口；未知请求与取消不会被当作
普通生成异常重试。原文身份保留在数据库，不要求模型把长 ID 写进总结。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from research_loop.providers import dpr_summary
from research_loop.providers.deepseek import DeepSeekProvider, ProviderResponseError

SYSTEM_PROMPT = "你是一名资深学术论文分析助手，请使用中文、以 Markdown 形式，对给定论文做结构化、深入、客观的总结。"


class _RunInterrupted(BaseException):
    """中断整个持久任务；绕过上游对普通 Exception 的三轮重试。"""


def paper_inputs(bundle: list[dict], metadata: list[dict]) -> tuple[str, str]:
    """将已有正文和元数据映射到 DPR 的 txt/md 输入，不注入任务或引用约束。

    Args:
        bundle: 已授权、按阅读顺序提供的原文片段。
        metadata: 相应论文元数据；缺少字段不推测或另调模型补齐。

    Returns:
        原文文本与上游格式 Markdown；证据 ID/解析警告只留在任务记录。
    """
    text = "\n\n".join(item.get("text") or "" for item in bundle)
    markdown = "\n\n".join(dpr_summary.build_markdown_content(
        paper, "deep", "", "", []) for paper in metadata)
    return text, markdown


class _ReceiptClient:
    """把 DPR client.chat 接到单次供应商调用；重入时按固定序号本地回放。

    每次重入从该节点的第一条收据开始回放，使上游整个三轮循环可以原样复用。
    已成功响应不重发，显式失败重现异常，远端结果未知则中断，不伪造 usage。
    """

    def __init__(self, state: dict, stage: dict, provider: Any, db: Any, commit: Any):
        self.state, self.stage = state, stage
        self.provider, self.db, self.commit = provider, db, commit
        self.kwargs = {"thinking": {"type": "disabled"}}
        self.offset = 0
        self.last_initial = ""
        self.finish_reason = None

    def chat(self, messages: list[dict], response_format: dict | None = None) -> dict:
        """执行或回放一次 DPR 请求，并先落盘响应再让上游判断结束标记。

        Args:
            messages: 上游函数构造的原始 messages，不在这里追加约束。
            response_format: 上游文本路径为 None；保留接口形状。

        Returns:
            上游 client 所需的 content/finish_reason 字段。

        Raises:
            _RunInterrupted: 取消、额度不足、旧输入不匹配或未结算请求。
            ProviderResponseError: 已明确收到的 HTTP 错误，交由上游进入下一轮。
        """
        task_id = self.state["task_id"]
        task = self.db.get_task(task_id)
        if task and task.get("cancel_requested"):
            self.commit({"cancelled": True})
            raise _RunInterrupted("cancelled before deep reading request")
        self.offset += 1
        sequence = self.stage["base_sequence"] + self.offset
        request = {"stage": "dpr_deep_reading", "messages": messages, "tools": [], "force_text": True,
                   "max_tokens": self.kwargs["max_tokens"], "temperature": self.kwargs["temperature"],
                   "thinking": self.kwargs["thinking"]}
        receipt = self.db.get_model_call(task_id, sequence)
        if receipt and json.loads(receipt["request_json"]) != request:
            raise _RunInterrupted("deep reading receipt request identity mismatch; history unchanged")
        from research_loop.storage.token_budget import TokenBudgetStore
        tokens = TokenBudgetStore(self.db)
        if receipt:
            tokens.settle(task_id, sequence)
        if receipt and receipt["status"] == "succeeded":
            raw = json.loads(receipt["response_json"])
        elif receipt and receipt["status"] in {"rejected", "response_error"}:
            error = json.loads(receipt["response_json"])
            self.commit({"model_requests": sequence})
            raise ProviderResponseError(error["http_status"], error.get("body", ""), receipt.get("request_id"))
        else:
            if receipt and not (receipt["status"] == "started" and receipt.get("sent_state") == "not_sent"):
                raise _RunInterrupted("existing deep reading receipt blocks resend: " + receipt["status"])
            root = self.state.get("root_task_id") or task_id
            if self.db.has_unknown_family_call(root):
                raise _RunInterrupted("task family contains unknown request; no resend")
            if sequence > self.state.get("max_model_requests", 12):
                raise _RunInterrupted("deep reading model request budget exhausted")
            try:
                tokens.reserve(root, task_id, sequence, request, "researcher")
            except ValueError as exc:
                raise _RunInterrupted(str(exc)) from exc
            if not receipt:
                if not self.db.reserve_model_slot(root, "researcher"):
                    self.db.save_model_call(task_id, sequence, "not_sent", request,
                                            response={"error": "shared model budget exhausted"}, sent_state="not_sent")
                    raise _RunInterrupted("shared model budget exhausted")
                self.db.begin_model_call(task_id, sequence, request)
            self.db.mark_model_call_sent(task_id, sequence)
            try:
                raw = self.provider.send(messages, [], force_text=True,
                                         max_tokens=request["max_tokens"], temperature=request["temperature"],
                                         thinking=request["thinking"])
            except ProviderResponseError as exc:
                self.db.save_model_call(task_id, sequence, "rejected" if 400 <= exc.status_code < 500 else "response_error",
                                        request, response=exc.as_record(), request_id=exc.request_id, sent_state="sent")
                self.commit({"model_requests": sequence})
                raise
            except Exception as exc:
                self.db.save_model_call(task_id, sequence, "unknown", request,
                                        response={"error": type(exc).__name__ + ": " + str(exc)}, sent_state="sent")
                self.commit({"model_requests": sequence})
                raise _RunInterrupted("unknown response: " + str(exc)) from exc
            self.db.save_model_call(task_id, sequence, "succeeded", request, response=raw,
                                    usage=raw.get("usage"), request_id=raw.get("id"),
                                    finish_reason=(raw.get("choices") or [{}])[0].get("finish_reason"), sent_state="sent")
        response = DeepSeekProvider.parse_response(raw)
        content = (response.content or "").strip()
        if request["max_tokens"] == 4096 and content:
            self.last_initial = content
        self.finish_reason = response.finish_reason
        self.commit({"model_requests": sequence})
        return {"content": content, "finish_reason": response.finish_reason}


def deep_reading_node(state: dict, provider: Any, papers: Any, db: Any, record: Any) -> dict:
    """调用 DPR 原函数；成功按结束标记发布，不引入引用或 Reviewer 完成门。

    Args:
        state: Researcher 已有正文交接状态。
        provider: 供应商单次请求接口。
        papers: 授权原文服务。
        db: 权威请求收据和任务库。
        record: 租约检查后的 checkpoint 写入函数。

    Returns:
        八点总结、完整性状态与真实请求计数；不完整文本可查看但不标记严格发布成功。
    """
    working = dict(state)
    stage = dict(state.get("deep_reading") or {})

    def commit(update: dict) -> None:
        """先提交 checkpoint 再推进内存，发生进程退出时由固定收据序号恢复。"""
        record(working, update)
        working.update(update)

    if stage.get("done"):
        return {}
    if stage and "base_sequence" not in stage:
        # 旧节点请求形状不同，不能以新提示词覆盖或重放其已发送请求。
        raise RuntimeError("unfinished pre-alignment deep reading requires its original runner; no resend")
    if not stage:
        refs = state.get("evidence_ids") or []
        if not refs:
            return record(state, {"deep_reading": {"done": True, "status": "skipped", "reason": "no evidence"}})
        bundle = papers.evidence_bundle(state["task_id"], refs)
        metadata = [(db.get_paper(pid) or {}).get("metadata", {"paper_id": pid})
                    for pid in dict.fromkeys(p["paper_id"] for p in bundle)]
        text, markdown = paper_inputs(bundle, metadata)
        stage = {"done": False, "status": "running", "markdown": "", "evidence_ids": refs,
                 "base_sequence": int(state.get("model_requests", 0)), "text": text, "metadata_markdown": markdown}
        commit({"deep_reading": stage})
    client = _ReceiptClient(working, stage, provider, db, commit)
    error = None
    try:
        # 文件仅映射上游函数的 md/txt 参数，模型消息与控制流由复制的函数决定。
        with tempfile.TemporaryDirectory(prefix="researchloop-dpr-") as directory:
            md_path, txt_path = Path(directory) / "paper.md", Path(directory) / "paper.txt"
            md_path.write_text(stage["metadata_markdown"], encoding="utf-8")
            txt_path.write_text(stage["text"], encoding="utf-8")
            markdown = dpr_summary.generate_deep_summary(str(md_path), str(txt_path), client=client) or ""
    except _RunInterrupted as exc:
        error = str(exc)
        markdown = client.last_initial
    # 与 ensure_reading_content(require_complete=True) 相同，不再附加 stop/行尾条件。
    complete = bool(markdown) and "（完）" in markdown and not error
    stage = {**stage, "done": True, "status": "complete" if complete else "partial", "markdown": markdown,
             "finish_reason": client.finish_reason, "request_count": client.offset}
    if error:
        stage["error"] = error
    structured = dict(working.get("structured_result") or {})
    structured.update(findings=[markdown] if markdown else [], evidence_ids=stage["evidence_ids"])
    gaps = list(working.get("gaps") or [])
    if not complete:
        gaps.append(error or "论文精读总结未完整生成")
    structured["gaps"] = gaps
    return record(working, {"deep_reading": stage, "structured_result": structured, "gaps": gaps,
                            "final_answer": markdown, "model_requests": working.get("model_requests", 0),
                            "cancelled": working.get("cancelled", False)})
