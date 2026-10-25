"""显式检索候选的不可变合同；不读取环境、不装配模型或改变生产默认。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


_CONFIGS = {
    'baseline': (1, 1, False),
    'fusion_bm25_2_1': (2, 1, False),
    'efficiency_length_sort': (1, 1, True),
}


def validate_fusion_weights(weights) -> dict:
    """校验两路权重并返回副本，不以缺通道为由补造得分。

    Args:
        weights: 必须完整声明bm25和vector的有限非负实数映射，至少一路为正。

    Raises:
        ValueError: 通道缺失、多余或权重不是有限非负实数。
    """
    if not isinstance(weights, dict) or set(weights) != {'bm25', 'vector'}:
        raise ValueError('RETRIEVAL_FUSION_WEIGHTS_INVALID')
    try:
        invalid = any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0
                      for v in weights.values())
    except (OverflowError, TypeError, ValueError):
        invalid = True
    if invalid:
        raise ValueError('RETRIEVAL_FUSION_WEIGHTS_INVALID')
    if not any(v > 0 for v in weights.values()):
        raise ValueError('RETRIEVAL_FUSION_WEIGHTS_INVALID')
    return dict(weights)


@dataclass(frozen=True)
class RetrievalStrategy:
    """由唯一名称确定参数，避免生成联合候选或修改已冻结配置。"""

    name: str

    def __post_init__(self):
        """构造时拒绝未冻结的名称，不进行外部访问。"""
        if not isinstance(self.name, str) or self.name not in _CONFIGS:
            raise ValueError('RETRIEVAL_STRATEGY_UNKNOWN')

    @property
    def fusion_weights(self) -> dict:
        """返回新的权重字典，调用方修改副本不能改变策略。"""
        bm25, vector, _ = _CONFIGS[self.name]
        return {'bm25': bm25, 'vector': vector}

    @property
    def reranker_kwargs(self) -> dict:
        """返回新的重排构造参数，保持固定线程和候选组批大小。"""
        return {'threads': 2, 'batch_size': 4, 'sort_by_length': _CONFIGS[self.name][2]}

    @property
    def candidate_limit(self) -> int:
        """返回显式候选的固定规模，不修改生产请求合同。"""
        return 24

    @property
    def rrf_k(self) -> int:
        """返回与旧等权算法相同的平滑常数。"""
        return 60

    def as_dict(self) -> dict:
        """完整序列化候选参数，嵌套字典均为独立副本。"""
        return {'name': self.name, 'fusion_weights': self.fusion_weights,
                'reranker_kwargs': self.reranker_kwargs,
                'candidate_limit': self.candidate_limit, 'rrf_k': self.rrf_k}

    def inspect_reranker(self, reranker) -> dict:
        """核对注入组件实际公开参数，未知或不匹配时拒绝推理。

        Args:
            reranker: 已构造的组件；检查属性不加载模型。

        Returns:
            与候选一致的实际组批配置副本。

        Raises:
            ValueError: 组件未公开参数或其类型和值与候选不同。
        """
        actual = self.observe_reranker(reranker)
        if any(type(actual[k]) is not type(v) or actual[k] != v
               for k, v in self.reranker_kwargs.items()):
            raise ValueError('RETRIEVAL_RERANKER_CONFIG_MISMATCH')
        return actual

    def observe_reranker(self, reranker) -> dict:
        """只记录安全标量或类型标签，坏配置也必须能序列化交付。

        Args:
            reranker: 待检查的注入组件，不读取模型文件。

        Returns:
            参数观察；未知对象不使用repr，非有限数值不进入JSON。
        """
        actual = {}
        for key in self.reranker_kwargs:
            try:
                value = getattr(reranker, key, None)
            except Exception:
                actual[key] = {'status': 'unknown', 'type': 'unreadable_attribute'}
                continue
            actual[key] = (value if value is None or type(value) in (int, bool)
                           or type(value) is float and math.isfinite(value)
                           else {'status': 'unknown', 'type': type(value).__name__})
        return actual


def get_retrieval_strategy(name: str) -> RetrievalStrategy:
    """按精确名称取得冻结策略，不接受额外调参或组合配置。"""
    return RetrievalStrategy(name)
