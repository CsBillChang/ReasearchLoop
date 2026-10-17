"""定向核验的证据合同检查；检查身份与引句，不冒充语义蕴含判断。"""
from __future__ import annotations
import re

from research_loop.domain.evidence import ClaimAssessment


# 只识别原文中真实的英文字母跨行断词；数字和普通连字符不参与归一。
_SPLIT_WORD = r"(?<=[A-Za-z])-\r?\n\s*(?=[A-Za-z])"


def _quote_projection(text: str, marker: str | None = None) -> tuple[str, list[tuple[int, int]]]:
    """生成匹配视图和原文字符区间，不修改持久化的来源文本。

    Args:
        text: 原始页文字。
        marker: 非空时将真实跨行断词压成此标记；None只折叠空白。

    Returns:
        归一文本及每个字符对应的原文[start, end)区间。
    """
    pattern = ((r"(?P<wrap>" + _SPLIT_WORD + r")|") if marker else "") + r"(?P<space>\s+)|(?P<char>.)"
    chars, offsets = [], []
    for match in re.finditer(pattern, text, re.S):
        char = marker if match.lastgroup == 'wrap' else (' ' if match.lastgroup == 'space' else match[0])
        chars.append(char)
        offsets.append(match.span())
    # 与旧的" ".join(text.split())一致，只去掉边界空白。
    left = 1 if chars and chars[0] == ' ' else 0
    right = len(chars) - (1 if chars and chars[-1] == ' ' else 0)
    return ''.join(chars[left:right]), offsets[left:right]


def quote_spans(quote: str, text: str) -> list[tuple[int, int]]:
    """复用受控断词规则，查找全部连续引句并映射回原文。

    Args:
        quote: 模型摘录的引句，允许排版空白及真实字母断词差异。
        text: 未改写的来源页文字。

    Returns:
        去重后的原文[start, end)区间。无匹配返回空列表；多个位置均保留，
        由要求唯一定位的调用方拒绝。仅是字串匹配，不代表语义支持。
    """
    if not quote.strip():
        return []
    # 标记不与实际文本冲突；不把来源中的普通字符误当可省略位置。
    marker = '\ue000'
    while marker in text or marker in quote:
        marker = chr(ord(marker) + 1)
    normalize = lambda value: ' '.join(value.split())
    wanted = normalize(re.sub(_SPLIT_WORD, '', quote))
    parts = []
    for index, char in enumerate(wanted):
        parts.append(r'(?:-|' + re.escape(marker) + ')' if char == '-' else re.escape(char))
        if char.isascii() and char.isalpha() and index + 1 < len(wanted) and wanted[index + 1].isascii() and wanted[index + 1].isalpha():
            parts.append(re.escape(marker) + '?')
    spans = set()
    for pattern, (source, offsets) in (
            (re.escape(normalize(quote)), _quote_projection(text)),
            (''.join(parts), _quote_projection(text, marker))):
        # lookahead保留重叠匹配；一个精确位置和一个断词位置也算歧义。
        for match in re.finditer('(?=(' + pattern + '))', source):
            start, end = match.span(1)
            if end > start:
                spans.add((offsets[start][0], offsets[end - 1][1]))
    return sorted(spans)


def quote_in_text(quote: str, text: str) -> bool:
    """沿用Reflection的存在性接口；矩阵另要求唯一位置，不改变核验语义。"""
    return bool(quote_spans(quote, text))


def validate_assessments(task_id: str, assessments: list[ClaimAssessment], papers) -> None:
    """确认引句属于当前任务实际读过的来源，仅规范化 PDF 空白。

    Args:
        task_id: 当前执行核验的任务身份。
        assessments: 模型对具体主张的结构化判断。
        papers: 负责读取授权证据的服务。

    Raises:
        ValueError: 引句不属于所声明的任何证据，或缺少绑定。
        PermissionError: 证据不属于当前任务。
    """
    errors = []
    for index, item in enumerate(assessments):
        bundle = papers.evidence_bundle(task_id, item.evidence_ids)
        if item.source_quote:
            if not any(quote_in_text(item.source_quote, page["text"]) for page in bundle):
                errors.append(f"assessment[{index}] {item.claim[:120]}: source_quote must occur in one cited evidence text; use one short exact excerpt, not stitched or paraphrased text")
    # 一次返回本批所有引句错误，避免逐个报错耗尽有界核验额度；不修写模型引句。
    if errors:
        raise ValueError("; ".join(errors))


def validate_target_assessments(task_id: str, target: dict, assessments: list, papers) -> None:
    """核对每个选定项、实际元数据字段和页级引句；不判断语义真伪。

    Args:
        task_id: 当前独立Reviewer任务。
        target: 服务加载的权威对象快照。
        assessments: 独立角色提交的逐项判断。
        papers: 已授权原文服务。

    Raises:
        ValueError: 遗漏项目、错用判断类型或引句不属于实际依据。
        PermissionError: 引用越过任务授权。
    """
    items = {item['item_id']: item for item in target['items']}
    ids = {f.item_id for f in assessments}
    # item是审核对象，不是意见的唯一键；一个回答可以包含多个独立问题。
    if ids != set(items):
        raise ValueError(f'review must cover selected item_ids; missing={sorted(set(items) - ids)}, '
                         f'unknown={sorted(ids - set(items))}; multiple findings per item are allowed')
    errors = []
    for index, finding in enumerate(assessments):
        item = items[finding.item_id]
        if target['target_type'] == 'candidates':
            if finding.suitability is None or finding.status is not None:
                raise ValueError('candidate review requires suitability, not a factual verdict')
            fields = finding.metadata_fields
            if not fields or any(field not in item['metadata'] or item['metadata'][field] in (None, '') for field in fields):
                raise ValueError('candidate review must identify available metadata_fields')
            if finding.source_quote and not any(quote_in_text(finding.source_quote, str(item['metadata'][field])) for field in fields):
                raise ValueError('candidate quote does not occur in the declared metadata fields')
        else:
            if finding.status is None or finding.suitability is not None:
                raise ValueError('research findings/answers require factual status')
            if finding.basis == 'scope_disclosure':
                # 查不到支持与准确披露查证范围不同；范围本身是执行事实，不要求论文原句证明。
                # 这里仍只验证范围真实存在，说明是否准确由独立角色判断。
                papers.evidence_bundle(task_id, finding.evidence_ids)
                continue
            fact = ClaimAssessment(claim=item['text'], status=finding.status, basis=finding.basis,
                reason=finding.reason, evidence_ids=finding.evidence_ids, source_quote=finding.source_quote)
            try:
                validate_assessments(task_id, [fact], papers)
            except ValueError as exc:
                # 保留原findings序号；scope_disclosure等分支不能使错误定位错位。
                errors.append(f'findings[{index}] item_id={finding.item_id}: {exc}')
    if errors:
        raise ValueError('; '.join(errors))
