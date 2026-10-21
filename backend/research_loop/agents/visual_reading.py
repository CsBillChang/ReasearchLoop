"""单次视觉补读节点：供独立组件或Supervisor委派的专用Researcher任务复用。"""
from __future__ import annotations

import json
import time
from uuid import uuid4

from research_loop.domain.visual_reading import VisualConfig, VisualInput, build_request
from research_loop.providers.deepseek import DeepSeekProvider, ProviderResponseError, RequestNotSent
from research_loop.storage.memory import Transaction, check_owner, digest, dump
from research_loop.storage.mysql import _now
from research_loop.storage.token_budget import TokenBudgetStore, estimate_details
from research_loop.providers.token_counting import validate_frozen_request


def build_visual_graph(node, task_id: str, item: VisualInput, *, worker_id: str, lease_token: str):
    """以单节点LangGraph运行原组件，节点内部仍使用独立子任务收据。

    Args:
        node: 已注入存储、读取服务和供应商的VisualReadingNode。
        task_id: 已领取/授权的专用子任务。
        item: 服务端派生并校验的页身份与当前问题。
        worker_id: 当前Worker身份。
        lease_token: 当前父子链租约。

    Returns:
        不含自主循环或额外审校的编译图；结果仍可能partial或执行失败。
    """
    from langgraph.graph import START, END, StateGraph
    graph = StateGraph(dict)
    graph.add_node('visual_reading', lambda state: node.run(
        task_id, item, worker_id=worker_id, lease_token=lease_token))
    graph.add_edge(START, 'visual_reading')
    graph.add_edge('visual_reading', END)
    return graph.compile()


class VisualReadingNode:
    """在独占组件任务内运行，不生成笔记、证据事实或额外Reviewer请求。"""

    def __init__(self, db, papers, provider, config: VisualConfig | None = None):
        """注入现有服务与单次供应商适配器；不隐式建立客户端或发送请求。"""
        self.db, self.papers, self.provider = db, papers, provider
        self.config = config or VisualConfig()

    def _guard(self, task_id: str, worker: str, token: str) -> dict:
        """使用现有租约与取消规则，拒绝失权执行。"""
        task = self.db.get_task(task_id)
        if not task or task['cancel_requested'] or not self.db.is_owner(task_id, worker, token):
            raise PermissionError('visual task cancelled or worker lease lost')
        return task

    def _intent(self, task_id, worker, token, request):
        """任务行锁内原子建立发送意图和请求额度；同任务并发只能发送一次。"""
        with self.db.connection() as conn:
            tx = Transaction(self.db, conn)
            if self.db.kind == 'sqlite':
                tx.execute('BEGIN IMMEDIATE')
            check_owner(tx, (task_id, worker, token))
            old = tx.one('SELECT * FROM model_calls WHERE task_id=? AND sequence_no=1', (task_id,))
            if old:
                if json.loads(old['request_json']) != request:
                    raise ValueError('visual receipt request identity mismatch')
                return old
            task = tx.one('SELECT * FROM tasks WHERE task_id=?', (task_id,))
            root = task['root_task_id'] or task_id
            frozen = tx.one('SELECT policy_json FROM token_counting_policies WHERE root_task_id=?', (root,))
            policy = json.loads(frozen['policy_json']) if frozen else {}
            condition = ('1=1' if policy.get('execution', {}).get('mode') == 'observe'
                         else 'model_used < model_limit-reserve_tail')
            used = tx.execute('UPDATE task_budgets SET model_used=model_used+1 WHERE root_task_id=? AND '+condition, (root,)).rowcount
            if used != 1:
                raise ValueError('shared model budget unavailable')
            tx.execute('INSERT INTO model_calls(call_id,task_id,sequence_no,status,request_json,usage_json,request_hash,sent_state,attempt_no,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (str(uuid4()), task_id, 1, 'started', dump(request), '{}', digest(request), 'not_sent', 1, _now()))
        return self.db.get_model_call(task_id, 1)

    def _mark_sent(self, task_id, worker, token):
        """租约围栏下CAS取得发送窗口，不持数据库锁等待网络。"""
        with self.db.connection() as conn:
            tx = Transaction(self.db, conn)
            if self.db.kind == 'sqlite':
                tx.execute('BEGIN IMMEDIATE')
            check_owner(tx, (task_id, worker, token))
            changed = tx.execute("UPDATE model_calls SET sent_state='sent' WHERE task_id=? AND sequence_no=1 AND status='started' AND sent_state='not_sent'", (task_id,)).rowcount
            if changed != 1:
                raise RuntimeError('visual request already entered send window; no resend')

    def _checkpoint(self, task_id, worker, token, result):
        """只存紧凑观察；事务内检查取消/租约，避免失权覆盖新执行者。"""
        with self.db.connection() as conn:
            tx = Transaction(self.db, conn)
            if self.db.kind == 'sqlite':
                tx.execute('BEGIN IMMEDIATE')
            check_owner(tx, (task_id, worker, token))
            old = tx.one('SELECT state_json FROM checkpoints WHERE task_id=?', (task_id,))
            state = json.loads(old['state_json']) if old else {}
            state['visual_reading'] = result
            if old:
                tx.execute('UPDATE checkpoints SET state_json=?,updated_at=? WHERE task_id=?', (dump(state), _now(), task_id))
            else:
                tx.execute('INSERT INTO checkpoints(task_id,state_json,updated_at) VALUES(?,?,?)', (task_id, dump(state), _now()))

    def run(self, task_id: str, item: VisualInput, *, worker_id: str, lease_token: str) -> dict:
        """执行或回放一条独立组件请求；每个组件任务只占一条收据。

        Args:
            task_id: 专用组件任务，不能复用已有日常图任务。
            item: 冻结问题与授权页身份。
            worker_id: 已领取本任务的执行者。
            lease_token: 本次有效租约token。

        Returns:
            complete/partial及原回答、来源、真实收据；状态不代表语义正确。

        Raises:
            RuntimeError: unknown、旧请求失败或不同执行链，保留历史不重发。
        """
        task = self._guard(task_id, worker_id, lease_token)
        root = task['root_task_id'] or task_id
        checkpoint = self.db.get_checkpoint(task_id) or {}
        if set(checkpoint) - {'visual_reading'} or any(c['sequence_no'] != 1 for c in self.db.list_model_calls(task_id)):
            raise RuntimeError('visual component requires a dedicated task')
        prepared = self.papers.prepare_visual(task_id, item)
        request = build_request(prepared, self.config, self.papers.pdf.file_store.cache_dir)
        estimate = estimate_details(request)
        receipt = self.db.get_model_call(task_id, 1)
        if receipt and json.loads(receipt['request_json']) != request:
            raise ValueError('visual receipt request identity mismatch; no resend')
        tokens = TokenBudgetStore(self.db)
        if receipt:
            tokens.settle(task_id, 1)
        elapsed = None
        replayed = bool(receipt and receipt['status'] == 'succeeded')
        if replayed:
            raw = json.loads(receipt['response_json'])
        else:
            if receipt and not (receipt['status'] == 'started' and receipt['sent_state'] == 'not_sent'):
                raise RuntimeError('visual receipt blocks resend: ' + receipt['status'])
            if self.db.has_unknown_family_call(root):
                raise RuntimeError('task family contains unknown request')
            if self.provider.model != self.config.model or self.provider.endpoint != self.config.base_url + '/chat/completions':
                raise ValueError('visual provider differs from frozen config')
            if self.provider.client.timeout.read != self.config.timeout_seconds:
                raise ValueError('visual provider timeout differs from frozen config')
            # 固定视觉输出合同保持8192，但仍须进入根任务单请求容量与计量准入。
            # 成功/未知旧收据已在上方处理；不能因后来容量变化重发旧请求。
            frozen_estimate = validate_frozen_request(request, tokens.counting_policy(root),
                self.provider.model, allow_model_override=True)
            if frozen_estimate:
                estimate = frozen_estimate
            self._guard(task_id, worker_id, lease_token)
            self._intent(task_id, worker_id, lease_token, request)
            try:
                tokens.reserve(root, task_id, 1, request, 'researcher')
            except ValueError as exc:
                # 请求槽已固定但Token准入失败时留下not_sent；不遗留无收据预留。
                # 此分支不重试，实际物理请求数仍为0。
                current = self.db.get_model_call(task_id, 1)
                if current['status'] == 'started' and current['sent_state'] == 'not_sent':
                    self.db.save_model_call(task_id, 1, 'not_sent', request,
                        response={'error': str(exc)}, sent_state='not_sent')
                raise
            self._mark_sent(task_id, worker_id, lease_token)
            started = time.perf_counter()
            try:
                raw = self.provider.send(request['messages'], [], force_text=True,
                    max_tokens=self.config.max_tokens, temperature=self.config.temperature, thinking={'type': 'disabled'})
            except RequestNotSent as exc:
                self.db.save_model_call(task_id, 1, 'not_sent', request, response={'error': str(exc)}, sent_state='not_sent')
                raise
            except ProviderResponseError as exc:
                self.db.save_model_call(task_id, 1, 'rejected' if 400 <= exc.status_code < 500 else 'response_error',
                    request, response=exc.as_record(), request_id=exc.request_id, sent_state='sent')
                raise
            except Exception as exc:
                self.db.save_model_call(task_id, 1, 'unknown', request,
                    response={'error': type(exc).__name__ + ': ' + str(exc)}, sent_state='sent')
                raise RuntimeError('visual response unknown; no automatic retry') from exc
            elapsed = time.perf_counter() - started
            # 响应先落收据；即便随后取消/失权，也不能丢失已付费响应并导致恢复重发。
            self.db.save_model_call(task_id, 1, 'succeeded', request, response=raw, usage=raw.get('usage'),
                request_id=raw.get('id'), finish_reason=(raw.get('choices') or [{}])[0].get('finish_reason'), sent_state='sent')
        parsed = DeepSeekProvider.parse_response(raw)
        text = parsed.content if isinstance(parsed.content, str) else ''
        complete = bool(text.strip()) and parsed.finish_reason == 'stop'
        result = {'status': 'complete' if complete else 'partial', 'answer': text,
                  'processing_summary': '模型观察尚未语义核验；不覆盖原文。' if complete else '输出为空或未正常结束，保留原响应，不自动续写。',
                  'source': prepared['manifest'], 'request_id': parsed.request_id, 'usage': parsed.usage,
                  'finish_reason': parsed.finish_reason, 'response_model': raw.get('model'),
                  'model_seconds': elapsed, 'render_seconds': prepared['render_seconds'],
                  'estimate': estimate, 'replayed': replayed, 'request_hash': digest(request)}
        self._checkpoint(task_id, worker_id, lease_token, result)
        return result
