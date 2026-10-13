"""复用 DPR 的八点精读与 Markdown 元数据纯函数。

来源：daily-paper-reader/src/6.generate_docs.py，MIT 归属见项目 docs/licenses。
仅提供可调用函数，不导入或启动上游全局客户端、采集器或发布程序。
三轮生成、每轮一次续写、结束标记与最终回退均保留上游行为。
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List

DeepSeekClient = Any
LLM_CLIENT = None


def log(message: str) -> None:
    """使用项目日志记录上游诊断，不改变生成控制流。"""
    logging.getLogger(__name__).warning(message)


def call_llm_text(
    client: DeepSeekClient,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    response_format: Dict[str, Any] | None = None,
) -> str:
    """发送上游文本请求并只返回去除边缘空白的 content。"""
    client.kwargs.update(
        {
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
    )
    resp = client.chat(messages=messages, response_format=response_format)
    return (resp.get("content") or "").strip()


def strip_auto_sections(md_text: str) -> str:
    """移除旧自动总结和速览，避免旧产物进入本次总结输入。"""
    if not md_text:
        return ""
    markers = [
        "\n\n---\n\n## 论文详细总结（自动生成）",
        "\n\n---\n\n## 速览摘要（自动生成）",
    ]
    cut_points = [md_text.find(m) for m in markers if md_text.find(m) != -1]
    if not cut_points:
        return md_text
    cut = min(cut_points)
    return md_text[:cut].rstrip()


def yaml_escape_value(s: str) -> str:
    """按上游规则转义 Markdown front matter 中的字符串。"""
    if not s:
        return '""'
    if any(c in s for c in [':', '#', '"', "'", '\n', '[', ']', '{', '}', ',', '&', '*', '!', '|', '>', '%', '@', '`']):
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'
    return s


def build_markdown_content(
    paper: Dict[str, Any],
    section: str,
    zh_title: str,
    zh_abstract: str,
    tags_list: List[str],
) -> str:
    """按上游格式构造论文元数据和摘要，不额外调用模型。"""
    title = (paper.get("title") or "").strip()
    authors = paper.get("authors") or []
    published = str(paper.get("published") or "").strip()
    if published:
        published = published[:10]
    pdf_url = str(paper.get("pdf_url") or paper.get("link") or "").strip()
    score = paper.get("llm_score")
    evidence = str(paper.get("canonical_evidence") or "").strip()
    tldr = (
        paper.get("llm_tldr_cn")
        or paper.get("llm_tldr")
        or paper.get("llm_tldr_en")
        or ""
    ).strip()
    abstract_en = (paper.get("abstract") or "").strip()
    if not abstract_en:
        abstract_en = "arXiv did not provide an abstract for this paper."
    paper_source = str(paper.get("source") or "").strip()
    selection_source = str(paper.get("selection_source") or "").strip()
    figure_assets = paper.get("_figure_assets") if isinstance(paper.get("_figure_assets"), list) else []
    table_assets = paper.get("_table_assets") if isinstance(paper.get("_table_assets"), list) else []

    # 解析速览内容
    glance = paper.get("_glance_overview", "").strip()
    glance_tldr = ""
    glance_motivation = ""
    glance_method = ""
    glance_result = ""
    glance_conclusion = ""

    if glance:
        for line in glance.split("\n"):
            line = line.strip().rstrip("\\").strip()
            if line.startswith("**TLDR**：") or line.startswith("**TLDR**:"):
                glance_tldr = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("**Motivation**：") or line.startswith("**Motivation**:"):
                glance_motivation = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("**Method**：") or line.startswith("**Method**:"):
                glance_method = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("**Result**：") or line.startswith("**Result**:"):
                glance_result = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("**Conclusion**：") or line.startswith("**Conclusion**:"):
                glance_conclusion = line.split("：", 1)[-1].split(":", 1)[-1].strip()

    # 优先使用速览生成的 TLDR（100字左右），否则使用原来的 TLDR
    display_tldr = glance_tldr if glance_tldr else tldr

    # 辅助函数：转义 YAML 字符串中的特殊字符
    # 构建 YAML front matter
    lines = ["---"]
    lines.append(f"title: {yaml_escape_value(title)}")
    if zh_title:
        lines.append(f"title_zh: {yaml_escape_value(zh_title)}")
    lines.append(f"authors: {yaml_escape_value(', '.join(authors) if authors else 'Unknown')}")
    lines.append(f"date: {yaml_escape_value(published or 'Unknown')}")
    if pdf_url:
        lines.append(f"pdf: {yaml_escape_value(pdf_url)}")
    if tags_list:
        # 保留完整的 kind:label 格式，前端渲染时再处理
        lines.append(f"tags: [{', '.join(yaml_escape_value(t) for t in tags_list)}]")
    if score is not None:
        lines.append(f"score: {score}")
    if evidence:
        lines.append(f"evidence: {yaml_escape_value(evidence)}")
    if display_tldr:
        lines.append(f"tldr: {yaml_escape_value(display_tldr)}")
    if paper_source:
        lines.append(f"source: {yaml_escape_value(paper_source)}")
    if selection_source:
        lines.append(f"selection_source: {yaml_escape_value(selection_source)}")
    if figure_assets:
        lines.append(f"figures_json: {yaml_escape_value(json.dumps(figure_assets, ensure_ascii=False))}")
    if table_assets:
        lines.append(f"tables_json: {yaml_escape_value(json.dumps(table_assets, ensure_ascii=False))}")

    # 速览字段
    if glance_motivation:
        lines.append(f"motivation: {yaml_escape_value(glance_motivation)}")
    if glance_method:
        lines.append(f"method: {yaml_escape_value(glance_method)}")
    if glance_result:
        lines.append(f"result: {yaml_escape_value(glance_result)}")
    if glance_conclusion:
        lines.append(f"conclusion: {yaml_escape_value(glance_conclusion)}")

    lines.append("---")
    lines.append("")

    # 正文部分：摘要
    if zh_abstract:
        lines.append("## 摘要")
        lines.append(zh_abstract)
        lines.append("")

    lines.append("## Abstract")
    lines.append(abstract_en)

    return "\n".join(lines)


def generate_deep_summary(
    md_file_path: str,
    txt_file_path: str,
    max_retries: int = 3,
    client: DeepSeekClient | None = None,
) -> str | None:
    """按 DPR 原行为运行最多三轮总结，每轮首次生成后最多续写一次。"""
    active_client = client or LLM_CLIENT
    if active_client is None:
        log("[WARN] 未配置 DEEPSEEK_API_KEY 或 SUMMARY_API_KEY，跳过精读总结。")
        return None
    if not os.path.exists(md_file_path):
        return None

    with open(md_file_path, "r", encoding="utf-8") as f:
        paper_md_content = strip_auto_sections(f.read())

    paper_txt_content = ""
    if os.path.exists(txt_file_path):
        with open(txt_file_path, "r", encoding="utf-8") as f:
            paper_txt_content = f.read()

    system_prompt = (
        "你是一名资深学术论文分析助手，请使用中文、以 Markdown 形式，"
        "对给定论文做结构化、深入、客观的总结。"
    )
    user_prompt = (
        "请基于下面提供的论文内容，生成一段详细的中文总结，要求按照如下要点依次展开：\n"
        "1. 论文的核心问题与整体含义（研究动机和背景）。\n"
        "2. 论文提出的方法论：核心思想、关键技术细节、公式或算法流程（用文字说明即可）。\n"
        "3. 实验设计：使用了哪些数据集 / 场景，它的 benchmark 是什么，对比了哪些方法。\n"
        "4. 资源与算力：如果文中有提到，请总结使用了多少算力（GPU 型号、数量、训练时长等）。若未明确说明，也请指出这一点。\n"
        "5. 实验数量与充分性：大概做了多少组实验（如不同数据集、消融实验等），这些实验是否充分、是否客观、公平。\n"
        "6. 论文的主要结论与发现。\n"
        "7. 优点：方法或实验设计上有哪些亮点。\n"
        "8. 不足与局限：包括实验覆盖、偏差风险、应用限制等。\n\n"
        "请用分层标题和项目符号（Markdown 格式）组织上述内容，语言尽量简洁但信息要尽量完整。\n"
        "要求：最后单独输出一行“（完）”作为结束标记。"
    )

    messages = [{"role": "system", "content": system_prompt}]
    if paper_txt_content:
        messages.append({"role": "user", "content": f"### 论文 PDF 提取文本 ###\n{paper_txt_content}"})
    messages.append({"role": "user", "content": f"### 论文 Markdown 元数据 ###\n{paper_md_content}"})
    messages.append({"role": "user", "content": user_prompt})

    last = ""
    for attempt in range(1, max_retries + 1):
        try:
            summary = call_llm_text(active_client, messages, temperature=0.3, max_tokens=4096)
            summary = (summary or "").strip()
            if not summary:
                continue
            last = summary
            if os.getenv("DPR_DEBUG_STEP6") == "1":
                log(f"[DEBUG][STEP6] deep_summary attempt={attempt} len={len(summary)} tail={summary[-20:]!r}")
            if "（完）" in summary:
                return summary
            # 续写一次：避免输出被截断
            cont_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "你上一次的总结可能被截断了，请从中断处继续补全，不要重复已输出内容。"},
                {"role": "user", "content": f"上一次输出如下：\n\n{summary}\n\n请继续补全，最后以一行“（完）”结束。"},
            ]
            cont = call_llm_text(active_client, cont_messages, temperature=0.3, max_tokens=2048)
            cont = (cont or "").strip()
            merged = f"{summary}\n\n{cont}".strip()
            if os.getenv("DPR_DEBUG_STEP6") == "1":
                log(f"[DEBUG][STEP6] deep_summary_cont attempt={attempt} len={len(cont)} merged_tail={merged[-20:]!r}")
            if "（完）" in merged:
                return merged
        except Exception as e:
            log(f"[WARN] 精读总结失败（第 {attempt} 次）：{e}")
            time.sleep(2 * attempt)
    return last or None
