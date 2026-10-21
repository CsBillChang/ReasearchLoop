"""工具契约底座：同一个 ToolSpec 生成模型 schema 并绑定执行函数，防止声明与执行入口脱节。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError

from research_loop.domain.tasks import Observation, SavePolicy


# 执行时由后端注入的权限与服务，不接受模型任意构造这些上下文。
@dataclass
class ToolContext:
    task_id: str
    save_policy: SavePolicy
    attachments: dict[str, dict[str, Any]] = field(default_factory=dict)
    cancelled: Callable[[], bool] = lambda: False
    services: dict[str, Any] = field(default_factory=dict)
    role: str = "supervisor"
    root_task_id: str | None = None
    parent_task_id: str | None = None


# 工具名称、说明、参数模型、处理函数和角色权限的单一注册记录。
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: Type[BaseModel]
    handler: Callable[[BaseModel, ToolContext], Observation]
    roles: tuple[str, ...] = ("supervisor", "researcher", "reviewer")

    def schema(self) -> dict[str, Any]:
        """由参数模型生成 JSON Schema；修改参数类的 docstring 也可能改变模型输入，补注释使用 #。"""
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                "parameters": self.args_model.model_json_schema()}}


def execute_spec(spec: ToolSpec, arguments: dict[str, Any], context: ToolContext) -> Observation:
    """校验并调用工具，将已知异常转换成统一观察。

    Args:
        spec: 绑定 schema、处理函数和角色的工具声明。
        arguments: 模型提供的参数对象，必须先经参数模型校验。
        context: 后端注入的任务权限、附件和服务。

    Returns:
        工具观察，明确区分取消、参数错误、权限拒绝与实际执行结果。
    """
    if context.cancelled():
        return Observation(ok=False, kind="cancelled", message="task cancellation requested")
    if isinstance(arguments, dict) and '_invalid_json' in arguments:
        # 截断响应不可让全默认参数模型当作空的合法提交，也不能猜测补全。
        return Observation(ok=False, kind='validation_error', retryable=True,
            message='工具参数不是完整JSON，未执行；请根据已有成果重新提交合法参数或合理结束。',
            data={'error_code': 'INVALID_TOOL_JSON', 'tool_executed': False, 'budget_counted': False})
    try:
        parsed = spec.args_model.model_validate(arguments)
    except ValidationError as exc:
        describe = getattr(spec.args_model, 'describe_validation_error', None)
        message = describe(exc, arguments) if callable(describe) else str(exc)
        # 个别外部工具需要标明“参数未通过、尚未发送”，避免被当成供应商失败。
        # 未提供诊断扩展的参数模型保持原来的空data合同。
        diagnostic = getattr(spec.args_model, 'validation_error_data', None)
        data = diagnostic(exc, arguments) if callable(diagnostic) else {}
        return Observation(ok=False, kind="validation_error", message=message, data=data, retryable=True)
    try:
        return spec.handler(parsed, context)
    except PermissionError as exc:
        return Observation(ok=False, kind="execution_error", message=str(exc), retryable=False)
    except (ValueError, FileNotFoundError) as exc:
        return Observation(ok=False, kind="execution_error", message=str(exc), retryable=True)
    except Exception as exc:
        return Observation(ok=False, kind="execution_error", message=type(exc).__name__ + ": " + str(exc), retryable=False)
