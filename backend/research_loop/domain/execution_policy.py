"""执行策略纯投影：根任务冻结额度、运行中交接与图调度资源分离。"""
from __future__ import annotations

from copy import deepcopy
import sys


def inherit_execution_policy(state: dict, policy: dict | None, budget: dict | None = None) -> dict:
    """只初始化新状态；调用方必须先排除已存在的快照。

    Args:
        state: 新创建的角色状态。
        policy: 根任务创建时冻结的统一执行策略，历史任务可能没有。
        budget: 历史根任务已持久化的共享额度。

    Returns:
        继承根额度的新状态；不会回写策略或历史请求。
    """
    limits = policy or budget or {}
    for target, keys in {
        'max_model_requests': ('max_model_requests', 'model_limit'),
        'max_tool_actions': ('max_tool_actions', 'tool_limit'),
        'reserve_model_requests': ('reserve_model_requests', 'reserve_tail'),
    }.items():
        for key in keys:
            if key in limits:
                state[target] = int(limits[key])
                break
    if policy:
        state['execution_policy'] = deepcopy(policy)
    return state


def enforcing(state: dict, budget: dict | None = None) -> bool:
    """优先采用冻结状态；没有新策略的历史任务继续强制执行。"""
    policy = state.get('execution_policy') or (budget or {}).get('execution_policy') or {}
    return policy.get('mode', 'enforce') != 'observe'


def graph_run_config(state: dict) -> dict:
    """图递归仅作为框架容量，业务终止由取消、无进展与冻结策略负责。"""
    if not enforcing(state):
        # Python整数上界使监测模式不被LangGraph默认25/旧264次提前截断。
        return {'recursion_limit': sys.maxsize}
    return {'recursion_limit': 2 * (int(state['max_model_requests']) + int(state['max_tool_actions'])) + 8}


def handoff_reservations(role: str, terminal: bool, active_children: int, parent_delivery: int = 1) -> int:
    """按实际运行角色保护每个子任务交接和父任务一次最终交付。"""
    active_children = max(0, active_children)
    if role == 'supervisor':
        return 0 if terminal else active_children + parent_delivery
    # 自己的结束行动可领取自己的保留槽，其他角色及父交付仍受保护。
    return max(0, active_children - int(terminal)) + parent_delivery
