"""工具注册表：按角色暴露 schema，调用前再次检查角色权限，再进入参数校验与服务执行。"""
from __future__ import annotations
from typing import Any
from research_loop.domain.tasks import Observation
from research_loop.tools.base import ToolContext, ToolSpec, execute_spec


def schema_names(schemas: list[dict[str, Any]]) -> set[str]:
    """返回本次请求真正可见的工具名，包括收尾时收缩后的范围。"""
    return {item["function"]["name"] for item in schemas}


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None, *, allowed_names: set[str] | None = None):
        """以同一可选范围限制schema与执行；None保留日常角色全部工具。"""
        self._specs: dict[str, ToolSpec] = {}
        self._allowed_names = frozenset(allowed_names) if allowed_names is not None else None
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        """同名工具拒绝覆盖，避免模型看到的声明与实际执行函数不一致。"""
        if spec.name in self._specs:
            raise ValueError("duplicate tool: " + spec.name)
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """范围外工具即使被手工或模型构造调用也不得进入handler。"""
        if self._allowed_names is not None and name not in self._allowed_names:
            raise KeyError('tool outside active scope: ' + name)
        if name not in self._specs:
            raise KeyError("unknown tool: " + name)
        return self._specs[name]

    def schemas(self, role: str | None = None) -> list[dict[str, Any]]:
        """按角色过滤模型可见工具；隐藏 schema 之外，execute 仍进行后端权限检查。"""
        return [spec.schema() for spec in self._specs.values()
                if (self._allowed_names is None or spec.name in self._allowed_names)
                and (role is None or role in spec.roles)]

    def execute(self, name: str, arguments: dict[str, Any], context: ToolContext) -> Observation:
        """输入工具名、参数和任务上下文，输出统一观察；未知工具或越权角色不进入业务服务。"""
        try:
            spec = self.get(name)
        except KeyError as exc:
            return Observation(ok=False, kind="validation_error", message=str(exc), retryable=False)
        if context.role not in spec.roles:
            return Observation(ok=False, kind="validation_error", message=f"role {context.role} cannot use {name}", retryable=False)
        return execute_spec(spec, arguments, context)

    def names(self) -> list[str]:
        """列出与当前执行范围一致的名称，不改变角色的独立检查。"""
        return [name for name in self._specs if self._allowed_names is None or name in self._allowed_names]
