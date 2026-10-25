"""Reflection 基线的零模型资格支架；真实模型执行仍明确拒绝。

本模块只验证当前业务链路与隔离边界。模拟供应商和 PDF 工厂由调用方
显式注入，不接收评分标签或参考修订。语义效果必须另经真实输出评分。
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Callable
from unittest.mock import patch
from uuid import uuid4


class BudgetRefused(RuntimeError):
    """批次预算不足或已有未结算调用，发送前拒绝。"""


def _count(value: int, name: str, *, positive: bool = False) -> int:
    """拒绝负数、布尔值和非整数，避免预算单位被静默转换。"""
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


@dataclass
class BatchBudget:
    """独立实验的发送前预留门，不修改产品任务预算。

    金额按冻结的输入未缓存/输出费率保守估算，不称实际账单。unknown 或
    缺 usage 保留完整预留并停止后续发送，不能填零后继续使用余额。
    """

    max_requests: int = 48
    max_tokens: int = 2_000_000
    max_cost_cny: Decimal = Decimal('16')
    input_per_million_cny: Decimal = Decimal('2')
    output_per_million_cny: Decimal = Decimal('8')
    requests_reserved: int = field(default=0, init=False)
    not_sent_calls: int = field(default=0, init=False)
    known_tokens: int = field(default=0, init=False)
    known_cost_upper_cny: Decimal = field(default=Decimal('0'), init=False)
    stop_reason: str | None = field(default=None, init=False)
    _pending: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _settled: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self):
        """预算和费率必须显式有效，不接受无限、负数或浮点价格。"""
        _count(self.max_requests, 'max_requests', positive=True)
        _count(self.max_tokens, 'max_tokens', positive=True)
        for name in ('max_cost_cny', 'input_per_million_cny', 'output_per_million_cny'):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ValueError(f'{name} must be a finite positive Decimal')

    def _cost(self, prompt: int, completion: int) -> Decimal:
        """输入全部按未缓存费率计算，缓存命中不重复相加。"""
        return (Decimal(prompt) * self.input_per_million_cny +
                Decimal(completion) * self.output_per_million_cny) / Decimal(1_000_000)

    def reserve(self, input_upper: int, max_output: int) -> str:
        """为一次物理发送预留输入上界和原配置最大输出。"""
        _count(input_upper, 'input_upper')
        _count(max_output, 'max_output', positive=True)
        with self._lock:
            if self.stop_reason:
                raise BudgetRefused(self.stop_reason)
            pending_tokens = sum(item['tokens'] for item in self._pending.values())
            pending_cost = sum((item['cost'] for item in self._pending.values()), Decimal('0'))
            tokens, cost = input_upper + max_output, self._cost(input_upper, max_output)
            if self.requests_reserved >= self.max_requests:
                raise BudgetRefused('request_limit')
            if self.known_tokens + pending_tokens + tokens > self.max_tokens:
                raise BudgetRefused('token_reservation_limit')
            if self.known_cost_upper_cny + pending_cost + cost > self.max_cost_cny:
                raise BudgetRefused('money_reservation_limit')
            identity = uuid4().hex
            self._pending[identity] = {'prompt': input_upper, 'completion': max_output,
                                       'tokens': tokens, 'cost': cost}
            self.requests_reserved += 1
            return identity

    def settle(self, identity: str, usage: dict | None, *, status: str = 'succeeded') -> None:
        """保存已知用量；失败或未知停止，保留缺失用量的未结算预留。"""
        if status not in {'succeeded', 'failed', 'rejected', 'unknown'}:
            raise ValueError('invalid settlement status')
        with self._lock:
            if identity in self._settled:
                if self._settled[identity] != {'usage': usage, 'status': status}:
                    raise ValueError('settlement identity changed')
                return
            reservation = self._pending.get(identity)
            if reservation is None:
                raise ValueError('unknown reservation')
            if status == 'unknown' or usage is None:
                self.stop_reason = 'unknown_call' if status == 'unknown' else 'usage_missing'
                return
            try:
                prompt = _count(usage['prompt_tokens'], 'prompt_tokens')
                completion = _count(usage['completion_tokens'], 'completion_tokens')
                total = _count(usage['total_tokens'], 'total_tokens')
                if total != prompt + completion:
                    raise ValueError('usage total mismatch')
            except (KeyError, TypeError, ValueError):
                self.stop_reason = 'usage_invalid'
                raise
            self.known_tokens += total
            self.known_cost_upper_cny += self._cost(prompt, completion)
            self._pending.pop(identity)
            self._settled[identity] = {'usage': dict(usage), 'status': status}
            if prompt > reservation['prompt'] or completion > reservation['completion']:
                self.stop_reason = 'reservation_upper_bound_breached'
                raise BudgetRefused(self.stop_reason)
            if status != 'succeeded':
                self.stop_reason = 'provider_' + status

    def settle_not_sent(self, identity: str, reason: str) -> None:
        """适配器明确未发送时释放预留，单列原因并停止本批。

        未发送不是有 usage=0 的物理请求；不将其写入已知用量账本。
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('not_sent reason must be nonempty')
        value = {'usage': None, 'status': 'not_sent', 'reason': reason}
        with self._lock:
            if identity in self._settled:
                if self._settled[identity] != value:
                    raise ValueError('settlement identity changed')
                return
            if identity not in self._pending:
                raise ValueError('unknown reservation')
            self._pending.pop(identity)
            self._settled[identity] = value
            self.requests_reserved -= 1
            self.not_sent_calls += 1
            self.stop_reason = self.stop_reason or 'provider_not_sent'

    def snapshot(self) -> dict:
        """导出保守计量与未结算数量；不把未知费用转成零。"""
        with self._lock:
            return {'requests_reserved': self.requests_reserved, 'known_tokens': self.known_tokens,
                    'not_sent_calls': self.not_sent_calls,
                    'not_sent_reasons': [item['reason'] for item in self._settled.values()
                                         if item['status'] == 'not_sent'],
                    'known_cost_upper_cny': str(self.known_cost_upper_cny),
                    'pending_reservations': len(self._pending), 'stop_reason': self.stop_reason,
                    'limits': {'requests': self.max_requests, 'tokens': self.max_tokens,
                               'cost_cny': str(self.max_cost_cny)},
                    'billing_status': 'conservative_rate_estimate_not_account_bill'}


@contextmanager
def deny_socket_connections():
    """零模型资格进程禁止网络，连错误装配的供应商也不能发出请求。"""
    def blocked(*args, **kwargs):
        raise PermissionError('REFLECTION_ZERO_MODEL_NETWORK_DENIED')

    with patch('socket.socket.connect', blocked), patch('socket.socket.connect_ex', blocked), \
            patch('socket.create_connection', blocked), patch('socket.getaddrinfo', blocked):
        yield


def baseline_result(draft: str) -> dict:
    """原样基线无需构造数据库、工厂或供应商。"""
    if not isinstance(draft, str) or not draft.strip():
        raise ValueError('draft must be nonempty')
    return {'execution_kind': 'no_action_fixed_draft', 'answer_body_markdown': draft,
            'physical_requests': 0, 'total_tokens': 0, 'semantic_quality': 'unreviewed'}


class _BudgetedSubstitute:
    """仅包装零模型 send 边界，角色参数原样传给显式替身。"""

    def __init__(self, provider, budget: BatchBudget):
        self.provider, self.budget = provider, budget

    def send(self, messages, tools=None, force_text=False, **kwargs):
        """以 UTF8 字节保守界和原 max_tokens 预留，随后结算模拟收据。"""
        from research_loop.providers.deepseek import RequestNotSent, ProviderResponseError
        payload = json.dumps({'messages': messages, 'tools': tools}, ensure_ascii=False)
        # 这里只用于合成资格；不能用它宣布真实 tokenizer 或费用门已经合格。
        input_upper = len(payload.encode('utf-8')) + 4096
        try:
            reservation = self.budget.reserve(input_upper, kwargs.get('max_tokens') or 8192)
        except BudgetRefused as exc:
            raise RequestNotSent(str(exc)) from exc
        try:
            raw = self.provider.send(messages, tools, force_text=force_text, **kwargs)
        except RequestNotSent as exc:
            self.budget.settle_not_sent(reservation, str(exc))
            raise
        except ProviderResponseError:
            self.budget.settle(reservation, None, status='rejected')
            raise
        except Exception:
            self.budget.settle(reservation, None, status='unknown')
            raise
        self.budget.settle(reservation, raw.get('usage'))
        return raw

    def close(self):
        """关闭本工厂新建的替身资源，不触碰其他任务的连接。"""
        self.provider.close()
        client = getattr(self.provider, 'client', None)
        if client is not None:
            client.close()


@dataclass
class SubstituteRun:
    """保留独立库、执行器和持久读回，便于恢复及拒绝路径检查。"""

    root: Path
    db: Any
    executor: Any
    task_id: str
    budget: BatchBudget
    resource_boundary: dict
    settings: Any

    def view(self) -> dict:
        """从正式 TaskService 重新聚合结果，不返回适配器内存稿。"""
        from research_loop.services.tasks import TaskService
        with patch('research_loop.config.get_settings', return_value=self.settings):
            return TaskService(self.db, self.settings).view(self.db.get_task(self.task_id)).model_dump(mode='json')

    def resume(self) -> dict:
        """恢复同一任务，网络仍被禁止；由原成功/unknown收据决定是否执行。"""
        with deny_socket_connections(), patch('research_loop.config.get_settings', return_value=self.settings):
            self.executor.execute(self.task_id, 'reflection-zero-resume')
        return self.view()

    def report(self) -> dict:
        """仅导出明确为替身的零模型证据，不报告真实质量或付费节省。"""
        rows = self.db.fetchall('SELECT m.* FROM model_calls m JOIN tasks t ON t.task_id=m.task_id '
                               'WHERE t.task_id=? OR t.root_task_id=? ORDER BY m.created_at',
                               (self.task_id, self.task_id))
        return {'execution_kind': 'substitute', 'real_model_calls': 0,
                'semantic_quality': 'not_evaluated', 'task_id': self.task_id,
                'view': self.view(), 'physical_receipts': rows,
                'budget': self.budget.snapshot(), 'resources': self.resource_boundary}


def run_substitute(root: Path, *, prompt: str, provider_factory: Callable,
                   pdf_factory: Callable, execution_kind: str = 'substitute',
                   budget: BatchBudget | None = None) -> SubstituteRun:
    """在全新本机目录运行合成 TaskService→Worker→Reviewer→交付链路。

    Args:
        root: 此前不存在的独占实验根目录。
        prompt: 仅含合成任务和稿件，不含评分标签。
        provider_factory: 显式模拟供应商工厂；所有 socket 始终被禁止。
        pdf_factory: 显式合成 PDF 工厂。
        execution_kind: 只接受 substitute，真实执行入口尚未开放。
        budget: 本批独立预留门；不改产品预算配置。

    Returns:
        可从持久库重新读回和恢复的替身运行记录。
    """
    if execution_kind != 'substitute':
        raise PermissionError('REFLECTION_REAL_EXECUTION_NOT_AUTHORIZED')
    if not callable(provider_factory) or not callable(pdf_factory):
        raise ValueError('explicit substitute factories required')
    root = Path(root).resolve()
    # 普通仓库使用.git目录，worktree使用.git文件；两种祖先均须在mkdir前拒绝。
    for ancestor in (root, *root.parents):
        marker = ancestor/'.git'
        if marker.exists() or marker.is_symlink():
            raise ValueError('runtime must be outside every Git checkout')
    root.mkdir(parents=True, exist_ok=False)
    from research_loop.config import Settings
    from research_loop.domain.tasks import TaskCreate
    from research_loop.services.tasks import TaskService
    from research_loop.storage.mysql import Database
    from research_loop.workers.executor import TaskExecutor

    # 仅零模型资格关闭神经检索；不据此生成真实 RAG/Reflection 效果成绩。
    settings = Settings(project_root=root, data_dir=root/'data', output_dir=root/'output',
        database_url='sqlite:///' + str(root/'runtime.sqlite3'),
        deepseek_model='deepseek-flash', deepseek_base_url='https://fixture.invalid',
        deepseek_api_key=None, discovery_backends={}, web_search_enabled=False,
        web_fetch_enabled=False, retrieval_enabled=False,
        qdrant_url='http://127.0.0.1:1', retrieval_namespace='reflection_zero_' + uuid4().hex)
    budget = budget or BatchBudget()
    db = Database(settings.database_url)
    db.ensure_schema()
    pdf = root/'input.pdf'
    pdf.write_bytes(b'%PDF-reflection-zero-model-fixture')

    def guarded_factory(current):
        return _BudgetedSubstitute(provider_factory(current), budget)

    # Worker 内部服务使用 get_settings 时也必须得到本批显式配置；不访问默认.env。
    with deny_socket_connections(), patch('research_loop.config.get_settings', return_value=settings):
        executor = TaskExecutor(settings, db, guarded_factory, pdf_factory)
        task = TaskService(db, settings).create(TaskCreate(prompt=prompt, save_policy='deny',
            attachments=[{'input_ref': 'input-A', 'paper_id': 'paper-A', 'local_path': str(pdf)}]))
        executor.execute(task.task_id, 'reflection-zero-model')
    boundary = {'database_kind': 'sqlite', 'database_path': str(root/'runtime.sqlite3'),
                'data_root': str(settings.data_dir), 'output_root': str(settings.output_dir),
                'network': 'socket_connections_denied', 'dotenv_loaded': False,
                'retrieval': 'explicitly_disabled_for_substitute_only',
                'qdrant_url': settings.qdrant_url, 'namespace': settings.retrieval_namespace,
                'factories': 'explicit_substitutes', 'product_configuration_changed': False}
    return SubstituteRun(root, db, executor, task.task_id, budget, boundary, settings)


def business_source_hashes(source_root: Path) -> dict[str, str]:
    """只读哈希业务源码，供资格运行前后保护核对。"""
    root = Path(source_root).resolve()/'backend/research_loop'
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*.py'))}
