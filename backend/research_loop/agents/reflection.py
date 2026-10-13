"""Reflection 持久控制：按需独立核验、精确局部修订、任务级轮次及无进展结束。"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
from typing import Any
from uuid import uuid4

from research_loop.domain.notes import RevisionInput
from research_loop.storage.mysql import Database, _now


def digest(value: Any) -> str:
    """计算稳定输入指纹，用于提交后工具重放；不代表事实正确。"""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def unresolved(result: dict) -> int:
    """统计独立核验中的未解决主张；不能把这个数量视作准确率。"""
    count = sum(f.get('status') != 'supported' for f in result.get('findings', []))
    # supported附带的范围说明/可选润色不算阻断；执行错误也不是语义问题已减少。
    return max(count, 1) if result.get('status') in {'issues', 'insufficient'} else count


class ReflectionService:
    """使用根任务行锁保护轮次和 patch，真实 LLM 请求在锁外执行。"""

    def __init__(self, db: Database, max_rounds: int = 2, owner: tuple[str, str] | None = None):
        """注入存储、任务级轮次上限及可选 Worker 租约。"""
        self.db, self.max_rounds, self.owner = db, max_rounds, owner

    def revision_checks(self, task_id: str, draft_id: str) -> dict:
        """只读验证局部修改不变量；给 Reviewer 事实性程序观察，不传历史判错意见。"""
        record = next((p for p in reversed(self.snapshot(task_id).get('patches', [])) if p['draft_id'] == draft_id), None)
        if not record:
            return {'revision_present': False}
        before = self.db.get_draft(record['parent_draft_id'])
        after = self.db.get_draft(draft_id)
        if not before or not after or before['task_id'] != task_id or after['task_id'] != task_id:
            return {'revision_present': True, 'non_target_text_preserved': False}
        text = before['markdown']; spans = []
        for patch in record['patches']:
            if text.count(patch['old_text']) != 1:
                return {'revision_present': True, 'non_target_text_preserved': False}
            start = text.index(patch['old_text'])
            spans.append((start, start + len(patch['old_text']), patch['new_text']))
        expected = text
        for start, end, new in sorted(spans, reverse=True):
            expected = expected[:start] + new + expected[end:]
        valid = (expected == after['markdown'] and sha256(text.encode()).hexdigest() == record['before_hash']
                 and sha256(after['markdown'].encode()).hexdigest() == record['after_hash'])
        return {'revision_present': True, 'draft_id': draft_id, 'content_hash': after['content_hash'],
                'non_target_text_preserved': valid, 'exact_patch_count': len(spans),
                'scope': '仅证明精确文字差异，不证明新句的科学事实正确。'}

    @contextmanager
    def _locked(self, task_id: str):
        """锁住根任务后读写会话；失权或取消时拒绝新增审校和修改。"""
        with self.db.connection() as conn:
            cur = conn.cursor()
            def sql(query, params=()):
                cur.execute(query if self.db.kind == 'sqlite' else query.replace('?', '%s'), params)
                return cur
            if self.db.kind == 'sqlite':
                sql('BEGIN IMMEDIATE')
            row = sql('SELECT * FROM tasks WHERE task_id=?' + (' FOR UPDATE' if self.db.kind == 'mysql' else ''), (task_id,)).fetchone()
            task = dict(row) if row else {}
            if not task or task.get('parent_task_id'):
                raise ValueError('Reflection requires a root task')
            if task.get('cancel_requested'):
                raise PermissionError('task cancelled')
            if self.owner and (task.get('claimed_by'), task.get('lease_token')) != self.owner:
                raise PermissionError('reflection worker lease lost')
            row = sql('SELECT state_json FROM reflection_sessions WHERE task_id=?', (task_id,)).fetchone()
            state = json.loads(dict(row)['state_json']) if row else {'limit': self.max_rounds, 'attempts': [], 'patches': [], 'stop_reason': None}
            yield state, sql
            raw = json.dumps(state, ensure_ascii=False)
            if row:
                sql('UPDATE reflection_sessions SET state_json=?,updated_at=? WHERE task_id=?', (raw, _now(), task_id))
            else:
                sql('INSERT INTO reflection_sessions(task_id,state_json,updated_at) VALUES(?,?,?)', (task_id, raw, _now()))

    def snapshot(self, task_id: str) -> dict:
        """返回已提交的会话，供 API 和失败收尾使用；不修改计数。"""
        row = self.db.fetchone('SELECT state_json FROM reflection_sessions WHERE task_id=?', (task_id,))
        return json.loads(row['state_json']) if row else {}

    def review(self, task_id: str, draft_id: str, content_hash: str, reason: str,
               evidence_ids: list[str], questions: list[str], execute) -> dict:
        """预留稳定审校身份，再调用独立子图；恢复复用相同身份和成功 receipt。

        Args:
            task_id: 根任务身份。
            draft_id: 当前数据库草稿。
            content_hash: 精确内容版本。
            reason: Supervisor 的可读触发理由。
            evidence_ids: 已读取的补充证据，不授权新来源。
            questions: 中性核验问题，不携带历史判错理由。
            execute: 接受轮次和证据的独立 Reviewer 回调。

        Returns:
            原始审校结果及任务级进展/停止元数据；停止不自动批准。
        """
        draft = self.db.get_draft(draft_id)
        if not draft or draft['task_id'] != task_id or draft['content_hash'] != content_hash:
            raise ValueError('review draft identity/hash mismatch')
        # 显式选择的是本轮定向证据包，不再无条件拼入整稿所有页。
        # 其余获授权页仍可由Reviewer补读；空选择沿用草稿证据，历史成功收据不重发。
        refs = sorted(set(evidence_ids or draft['evidence_ids']))
        if any(not self.db.task_has_evidence(task_id, ref) for ref in refs):
            raise PermissionError('review evidence is not authorized')
        key = digest([draft_id, content_hash, refs])
        with self._locked(task_id) as (state, sql):
            attempts = state['attempts']
            prior = next((a for a in attempts if a['key'] == key), None)
            if prior and prior.get('result'):
                return self._projection(state, prior)
            if prior:
                attempt = prior
            else:
                if state['stop_reason'] or len(attempts) > state['limit']:
                    return {'status': 'insufficient', 'reflection': {'stop_reason': state['stop_reason'] or 'round_limit', 'can_continue': False}, 'gaps': ['核验修复已结束，保留已有成果与未解决问题。']}
                if attempts:
                    previous = attempts[-1]
                    if not previous.get('result'):
                        raise ValueError('previous review is not settled; resume it first')
                    if draft_id != previous['draft_id'] and not any(p['draft_id'] == draft_id and p['review_id'] == previous['result']['review_id'] for p in state['patches']):
                        raise ValueError('use revise_note for a reviewed draft, not a whole-draft replacement')
                    if draft_id == previous['draft_id'] and not set(refs).difference(previous['evidence_ids']):
                        raise ValueError('no new draft or evidence; reuse the prior review')
                    if draft_id == previous['draft_id'] and not questions:
                        raise ValueError('new evidence requires a specific neutral verification question')
                attempt = {'key': key, 'round': len(attempts), 'draft_id': draft_id, 'content_hash': content_hash,
                           'evidence_ids': refs, 'reason': reason, 'questions': questions, 'status': 'started'}
                attempts.append(attempt)
        # 子图有持久 request_id/receipt 和 checkpoint；不在数据库事务中等待模型。
        result = execute(attempt['round'], refs, questions)
        with self._locked(task_id) as (state, sql):
            current = state['attempts'][attempt['round']]
            current['result'] = result
            current['status'] = 'completed' if result.get('status') != 'execution_failed' else 'failed'
            old = state['attempts'][attempt['round'] - 1]['result'] if attempt['round'] else None
            progress = bool(result.get('status') != 'execution_failed' and (result.get('status') == 'supported' or (old and unresolved(result) < unresolved(old))))
            current['progress'] = {'unresolved_before': unresolved(old) if old else None, 'unresolved_after': unresolved(result),
                                   'reduced_unresolved': progress, 'semantic_guarantee': False}
            if result.get('status') == 'execution_failed':
                state['stop_reason'] = 'review_execution_failed'
            elif result.get('status') == 'supported':
                state['stop_reason'] = 'supported'
            elif attempt['round'] and not progress:
                state['stop_reason'] = 'no_progress'
            elif attempt['round'] >= state['limit']:
                state['stop_reason'] = 'round_limit'
            return self._projection(state, current)

    @staticmethod
    def _projection(state: dict, attempt: dict) -> dict:
        """将审校和控制层状态分开，原始语义判断不被轮次逻辑覆盖。"""
        return {**attempt['result'], 'reflection': {'round': attempt['round'], 'max_rounds': state['limit'],
                'stop_reason': state['stop_reason'], 'can_continue': not state['stop_reason'],
                'progress': attempt.get('progress'), 'trigger_reason': attempt['reason']}}

    def revise(self, task_id: str, payload: RevisionInput) -> dict:
        """原子应用绑定当前审校的局部 patch；非目标字节保持，重复操作复用版本。

        Raises:
            ValueError: 版本/审校过期、原句不唯一、重叠修改或没有实际变更。
            PermissionError: 证据越权、任务取消或 Worker 失权。
        """
        key = digest(payload.model_dump())
        with self._locked(task_id) as (state, sql):
            replay = next((p for p in state['patches'] if p['key'] == key), None)
            if replay:
                row = dict(sql('SELECT * FROM drafts WHERE draft_id=?', (replay['draft_id'],)).fetchone())
                row['evidence_ids'] = json.loads(row.pop('evidence_ids_json'))
                return {**row, 'revision': replay}
            if state['stop_reason'] or not state['attempts'] or len(state['attempts']) > state['limit']:
                raise ValueError('reflection closed or repair limit reached; deliver preserved work')
            current = state['attempts'][-1]
            review = current.get('result') or {}
            if (payload.review_id, payload.draft_id, payload.content_hash) != (review.get('review_id'), current['draft_id'], current['content_hash']):
                raise ValueError('revision must bind the latest settled review and draft hash')
            if any(p['review_id'] == payload.review_id for p in state['patches']):
                raise ValueError('one atomic patch batch per review; review the new draft next')
            row = dict(sql('SELECT * FROM drafts WHERE draft_id=? AND task_id=?', (payload.draft_id, task_id)).fetchone())
            original = row['markdown']; edits = []; evidence = set(json.loads(row['evidence_ids_json']))
            for patch in payload.patches:
                if original.count(patch.old_text) != 1 or patch.old_text == patch.new_text:
                    raise ValueError('old_text must occur exactly once and replacement must change it')
                if patch.finding_index >= len(review.get('findings', [])) or review['findings'][patch.finding_index]['status'] == 'supported':
                    raise ValueError('patch must target an unresolved finding index')
                if not patch.reason.strip():
                    raise ValueError('patch reason is required')
                for ref in patch.evidence_ids:
                    found = sql('SELECT 1 FROM task_evidence WHERE task_id=? AND evidence_id=?', (task_id, ref)).fetchone()
                    if not found:
                        raise PermissionError('patch evidence is not authorized')
                evidence.update(patch.evidence_ids)
                start = original.index(patch.old_text); edits.append((start, start + len(patch.old_text), patch.new_text))
            edits.sort()
            if any(a[1] > b[0] for a, b in zip(edits, edits[1:])):
                raise ValueError('patch ranges overlap')
            revised = original
            for start, end, replacement in reversed(edits):
                revised = revised[:start] + replacement + revised[end:]
            content_hash = sha256(revised.encode()).hexdigest()
            if sql('SELECT 1 FROM drafts WHERE task_id=? AND content_hash=?', (task_id, content_hash)).fetchone():
                raise ValueError('revision returns to an existing draft; no progress')
            draft_id = str(uuid4())
            sql('INSERT INTO drafts(draft_id,task_id,title,markdown,content_hash,evidence_ids_json,status,created_at) VALUES(?,?,?,?,?,?,?,?)',
                (draft_id, task_id, row['title'], revised, content_hash, json.dumps(sorted(evidence)), 'pending_review', _now()))
            record = {'key': key, 'review_id': payload.review_id, 'parent_draft_id': payload.draft_id, 'draft_id': draft_id,
                      'before_hash': payload.content_hash, 'after_hash': content_hash, 'patches': [p.model_dump() for p in payload.patches]}
            state['patches'].append(record)
            sql('UPDATE tasks SET quality_status=?,save_status=? WHERE task_id=?', ('unreviewed', 'draft_created', task_id))
            return {'draft_id': draft_id, 'task_id': task_id, 'title': row['title'], 'markdown': revised,
                    'content_hash': content_hash, 'evidence_ids': sorted(evidence), 'status': 'pending_review', 'revision': record}
