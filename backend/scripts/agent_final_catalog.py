"""固定验收集的题录词项排序；不读取答案、调用模型或访问外部资源。"""
from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata


SUITE_KIND = 'agent_final_v1'
CATALOG_LIMIT = 54


def _terms(text: str) -> list[str]:
    """统一大小写和全角字符，保留英文词项与中文单字、相邻双字。"""
    normalized = unicodedata.normalize('NFKC', text).casefold()
    terms = re.findall(r'[a-z0-9]+', normalized)
    for span in re.findall(r'[\u3400-\u9fff]+', normalized):
        terms.extend(span)
        terms.extend(span[i:i + 2] for i in range(len(span) - 1))
    return terms


def ranked_indices(rows: list[tuple[str, str]], query: str) -> list[int]:
    """用冻结题名和摘要对完整候选池作确定性BM25排序。

    Args:
        rows: 按冻结中性顺序排列的题名、摘要二元组，不接受评分或标签。
        query: 原发现工具给出的查询文本，不注入按题答案词。

    Returns:
        原列表索引，相关性相同时沿用冻结中性顺序。英文和中文仅按词项
        匹配，不声称支持语义检索、跨语言翻译或线上召回质量。

    Raises:
        ValueError: 查询或题录形状不合法。
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError('AGENT_FINAL_CATALOG_QUERY_INVALID')
    if (not isinstance(rows, list) or len(rows) > CATALOG_LIMIT
            or any(not isinstance(row, tuple) or len(row) != 2
                   or any(not isinstance(value, str) for value in row) for row in rows)):
        raise ValueError('AGENT_FINAL_CATALOG_ROWS_INVALID')
    if not rows:
        return []
    # 题名统一加权，不按case或预期相关论文调整排序参数。
    documents = [Counter(_terms(title) * 2 + _terms(abstract)) for title, abstract in rows]
    lengths = [sum(doc.values()) for doc in documents]
    average_length = sum(lengths) / len(documents) or 1.0
    queries = set(_terms(query))
    frequencies = Counter(term for doc in documents for term in doc if term in queries)
    scores = []
    for doc, length in zip(documents, lengths):
        score = 0.0
        for term in sorted(queries):
            frequency = doc.get(term, 0)
            if not frequency:
                continue
            inverse = math.log1p((len(documents) - frequencies[term] + 0.5)
                                / (frequencies[term] + 0.5))
            score += inverse * frequency * 2.2 / (
                frequency + 1.2 * (0.25 + 0.75 * length / average_length))
        scores.append(score)
    return sorted(range(len(rows)), key=lambda index: (-scores[index], index))
