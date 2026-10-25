"""可重建的词项检索：权威记录在MySQL；此模块只给已过滤的材料计算BM25。"""
from __future__ import annotations
import math
import re
from collections import Counter

from research_loop.domain.retrieval_strategy import validate_fusion_weights


def terms(text: str) -> list[str]:
    """将英文词和中文相邻双字转为词项；不声称支持语义向量召回。"""
    result = re.findall(r"[a-z0-9_]+", text.lower())
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        result.extend(chunk[i:i+2] for i in range(max(1, len(chunk)-1)))
    return result


def rank(query: str, records: list[dict], limit: int = 5) -> list[dict]:
    """对调用方已做有效状态和来源过滤的记录排序，零匹配不冒充相关。"""
    if not records:
        return []
    return rank_terms(query, records, [Counter(terms(r['search_text'])) for r in records], limit)


def rank_terms(query: str, records: list[dict], bags: list[dict], limit: int = 24) -> list[dict]:
    """对同一授权语料的持久词项计算BM25，复用Memory原有参数与排序。

    Args:
        query: 未扩写的用户查询。
        records: 已核验身份的候选记录。
        bags: 与records同序的词频，不以查询命中子集计算IDF。
        limit: 返回正分候选上限。
    """
    if len(records) != len(bags):
        raise ValueError("BM25_INPUT_MISMATCH")
    if not records:
        return []
    sizes = [sum(b.values()) for b in bags]
    avg = max(sum(sizes) / len(sizes), 1)
    scores = [0.0] * len(records)
    for term in set(terms(query)):
        df = sum(term in b for b in bags)
        idf = math.log(1 + (len(records) - df + .5) / (df + .5))
        for i, bag in enumerate(bags):
            tf = bag.get(term, 0)
            scores[i] += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * sizes[i] / avg))
    return [{**records[i], 'score': scores[i]} for i in sorted(range(len(records)), key=lambda i: (-scores[i], i)) if scores[i] > 0][:limit]


def reciprocal_rank_fusion(channels: dict[str, list[dict]], k: int = 60, *, weights=None) -> list[dict]:
    """参考论文的1/(k+rank)融合思想；只合并同一版本片段，不相加异量纲分数。

    同通道重复候选只算一次；并列以完整候选身份稳定排序，保留原通道排名/分数。
    显式权重采用weight/(k+rank)，缺通道不补分；省略时保持原分数运算。
    """
    if k < 1:
        raise ValueError("RRF_K_INVALID")
    if weights is not None:
        weights = validate_fusion_weights(weights)
        if set(channels) - set(weights):
            raise ValueError('RETRIEVAL_FUSION_CHANNEL_UNKNOWN')
    result = {}
    for name, rows in channels.items():
        seen = set()
        for row in rows:
            key = row["key"]
            if key in seen:
                continue
            seen.add(key)
            rank_no = len(seen)
            target = result.setdefault(key, {"key": key, "rrf_score": 0.0, "channels": {}})
            target["rrf_score"] += (1 if weights is None else weights[name]) / (k + rank_no)
            target["channels"][name] = {"rank": rank_no, "score": row["score"]}
    return sorted(result.values(), key=lambda row: (-row["rrf_score"], row["key"]))


def fuse_reranker_ranks(rows: list[dict], scores: list[float], *, weight: float = 1.0,
                       k: int = 60) -> list[dict]:
    """融合原RRF名次和Cross-encoder名次，未评分尾部保持原序。

    Args:
        rows: 已融合、去重且按RRF排序的同一授权候选集。
        scores: 前N条候选的有限相关性logit；不可包含其他候选。
        weight: 重排名次权重；1保持旧纯重排，0保持原RRF序。
        k: 平滑常数，不直接相加异量纲模型分数。

    Returns:
        新列表，带原名次、重排分数和融合分数；不修改输入或来源字段。

    Raises:
        ValueError: 身份重复、分数越界或参数非法。
    """
    if (not math.isfinite(weight) or not 0 <= weight <= 1 or k < 1
        or len(scores) > len(rows) or len({r['key'] for r in rows}) != len(rows)
        or not all(math.isfinite(s) for s in scores)):
        raise ValueError('RERANK_FUSION_INVALID')
    output = [{**r, 'rrf_rank': i, 'reranker_score': None, 'reranker_rank': None,
               'ranking_score': None} for i, r in enumerate(rows, 1)]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    for position, index in enumerate(order, 1):
        output[index].update(reranker_score=float(scores[index]), reranker_rank=position,
            ranking_score=(1-weight)/(k+index+1)+weight/(k+position))
    prefix = sorted(output[:len(scores)], key=lambda r: (-r['ranking_score'], r['rrf_rank']))
    return prefix + output[len(scores):]
