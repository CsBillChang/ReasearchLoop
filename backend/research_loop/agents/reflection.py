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
            from research_loop.domain.reflection import review_progress
            def claim_items(value):
                """仅对同一主张比较状态；减少意见条数不等于修复。"""
                return {'findings': [{'item_id': digest(' '.join(f['claim'].split())), 'status': f['status']}
                                     for f in (value or {}).get('findings', [])]}
            changes = review_progress(claim_items(old), claim_items(result))
            progress = bool(result.get('status') != 'execution_failed' and
                            (result.get('status') == 'supported' or changes['effective_change']))
            current['progress'] = {'unresolved_before': unresolved(old) if old else None, 'unresolved_after': unresolved(result),
                                   'reduced_unresolved': progress, **changes}
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

    def _research_source(self, task_id: str, args) -> dict:
        """从成功搜索/阶段行动构造对象，不信任模型自报的版本和来源。"""
        from research_loop.services.research_progress import reviewable_findings
        source_task = args.source_task_id or task_id
        progress = reviewable_findings(self.db, task_id, source_task)
        if args.target_type == 'findings':
            items = [{'item_id': f['finding_id'], 'text': f['text'], 'sources': f['sources']}
                     for f in progress['findings']]
            origin = {'source_revision': progress['revision'], 'read_ranges': progress['read_ranges']}
        elif args.target_type == 'candidates':
            selected = {}
            for action in self.db.list_actions(source_task):
                if action['tool_name'] == 'search_papers' and action['status'] == 'succeeded':
                    for candidate in action['observation'].get('data', {}).get('candidates', []):
                        paper_id = candidate['paper_id']
                        if not self.db.task_has_paper(source_task, paper_id):
                            raise PermissionError('candidate source permission expired')
                        selected[paper_id] = {'item_id': paper_id, 'metadata': candidate,
                                              'source_action_id': action['action_id']}
            items, origin = list(selected.values()), {}
        else:
            answer, origin = args.answer_markdown, {}
            if answer is None:
                from research_loop.services.conversations import ConversationService
                # 仅加载当前任务明确引用的回答；数据库有历史不等于授权任选最近回复。
                # 原回答是核验对象，不是原文证据，也不继承其保存或审校权限。
                if source_task != task_id:
                    raise ValueError('referenced answer must belong to the current task conversation')
                referenced = ConversationService(self.db).referenced_answer(task_id)
                if referenced is None:
                    raise ValueError('answer source missing or ambiguous; reference an exact prior assistant answer '
                                     'in this conversation or provide a new answer_markdown')
                answer = referenced['content']
                origin['answer_source'] = {k: v for k, v in referenced.items() if k != 'content'}
            if not answer or not answer.strip():
                raise ValueError('new answer target requires nonblank answer_markdown or an explicit answer source')
            items = [{'item_id': 'answer', 'text': answer, 'sources': []}]
        if args.item_ids:
            ids = set(args.item_ids)
            if ids.difference(i['item_id'] for i in items):
                raise ValueError('unknown selected item_ids; use successful action output')
            items = [i for i in items if i['item_id'] in ids]
        if not items or len(items) > 30:
            raise ValueError('select between 1 and 30 actual items for one review')
        from research_loop.services.coverage import CoverageService
        return {'target_type': args.target_type, 'source_task_id': source_task,
                'items': items, 'origin': origin,
                'requirements': {'user_request': self.db.get_task(task_id)['prompt'],
                                 'current_plan': CoverageService(self.db).plan(task_id)}}

    def _research_refs(self, task_id: str, refs: list[str]) -> list[dict]:
        """按真实版本/位置/文字去重补证；新ID和重复页不算证据增量。"""
        from research_loop.services.research_progress import ResearchProgressService
        sources = [ResearchProgressService(self.db).source(task_id, ref) for ref in sorted(set(refs))]
        unique = {}
        for source in sources:
            key = digest({k: source[k] for k in ('source_sha256', 'page_number', 'text_sha256')})
            unique[key] = {**source, 'location_key': key}
        return [unique[k] for k in sorted(unique)]

    def research_target(self, task_id: str, target_id: str, content_hash: str | None = None) -> dict:
        """读取本根任务当前核验快照；旧hash或失效来源不能被当前操作复用。"""
        target = self.snapshot(task_id).get('research_targets', {}).get(target_id)
        if not target or (content_hash and target['current']['content_hash'] != content_hash):
            raise ValueError('research target missing or stale; use current target_id/content_hash')
        self._validate_answer_source(task_id, target['current'])
        self._research_refs(task_id, target['current'].get('evidence_ids', []))
        for item in target['current']['items']:
            for ref in item.get('sources', []):
                from research_loop.services.research_progress import ResearchProgressService
                if not ResearchProgressService(self.db).source_version_matches(task_id, ref):
                    raise ValueError('research source version changed; review current evidence again')
        if target['attempts']:
            last = target['attempts'][-1]
            expected = last['sources'] + last.get('observed_sources', [])
            actual = self._research_refs(task_id, [s['evidence_id'] for s in expected])
            if {s['location_key'] for s in actual} != {s['location_key'] for s in expected}:
                raise ValueError('review source version changed; prior result cannot approve current evidence')
        # 当前原始发现可继续增加；旧核验只说明保存快照，不自动核验新增发现。
        return json.loads(json.dumps(target))

    def _validate_answer_source(self, task_id: str, target: dict) -> None:
        """确认历史回答身份仍有效；局部新版本不要求等于旧正文，但保留原始来源链。"""
        source = target.get('origin', {}).get('answer_source')
        if not source:
            return
        from research_loop.services.conversations import ConversationService
        row = ConversationService(self.db).answer_message(task_id, source['source_id'])
        if row is None or sha256(row['content'].encode()).hexdigest() != source['content_sha256']:
            raise ValueError('referenced answer source missing or stale; current review cannot reuse it')

    def review_research(self, task_id: str, args, execute) -> dict:
        """持久独立核验通用对象；重复结果复用，执行失败保留为失败。

        Args:
            task_id: 根任务。
            args: 通用核验参数；已有对象必须带当前hash。
            execute: 接受稳定attempt ID、当前快照、依据和中性问题的回调。

        Returns:
            原始逐项判断与可读的版本/进展/结束观察。
        """
        from research_loop.domain.reflection import review_progress, target_fingerprint
        questions = [' '.join(q.split()) for q in args.verification_questions]
        source = None if args.target_id else self._research_source(task_id, args)
        target_id = args.target_id or digest([task_id, args.target_type, source['source_task_id'],
            sorted(i['item_id'] for i in source['items'])])[:24]
        with self._locked(task_id) as (state, sql):
            targets = state.setdefault('research_targets', {})
            record = targets.get(target_id)
            if args.target_id:
                if not record or record['current']['content_hash'] != args.content_hash:
                    raise ValueError('research target missing or stale')
                if record['current']['target_type'] != args.target_type:
                    raise ValueError('target_type cannot change for an existing target')
                if args.answer_markdown is not None or args.item_ids or args.source_task_id:
                    raise ValueError('existing target uses its stored snapshot; use revise_research for changes')
                if args.refresh_source:
                    if args.target_type == 'answer' or record['decisions'] or any(not a.get('result') for a in record['attempts']):
                        raise ValueError('refresh requires settled candidate/finding review')
                    # 明确刷新保留所有历史与计数，不能以新来源版本重置已处理的问题。
                    refresh = args.model_copy(update={'target_id': None, 'content_hash': None,
                        'source_task_id': record['current']['source_task_id'],
                        'item_ids': [i['item_id'] for i in record['current']['items']]})
                    fresh = self._research_source(task_id, refresh)
                    fresh_hash = target_fingerprint(fresh)
                    if fresh_hash != record['base_hash']:
                        record['base_hash'] = fresh_hash
                        record['current'] = {**fresh, 'target_id': target_id, 'content_hash': fresh_hash,
                            'goal_ids': record['current'].get('goal_ids', []), 'evidence_ids': []}
                        record['available_evidence_ids'] = []
                        record['versions'].append(record['current'].copy())
            elif record:
                if record['base_hash'] != target_fingerprint(source):
                    raise ValueError('research source changed; review the stored target or explicitly refresh its source snapshot')
            else:
                record = {'base_hash': target_fingerprint(source), 'versions': [], 'attempts': [], 'patches': [], 'decisions': []}
                source['content_hash'] = target_fingerprint(source)
                record['current'] = {**source, 'target_id': target_id, 'evidence_ids': [], 'goal_ids': args.goal_ids}
                record['versions'].append(record['current'].copy())
                targets[target_id] = record
            current = record['current']
            self._validate_answer_source(task_id, current)
            from research_loop.services.coverage import CoverageService
            plan = CoverageService(self.db).plan(task_id)
            if any(g not in {g['goal_id'] for g in (plan or {}).get('goals', [])} for g in args.goal_ids):
                raise ValueError('review goal_ids must belong to current plan')
            if args.goal_ids and sorted(args.goal_ids) != sorted(current.get('goal_ids', [])):
                raise ValueError('goal_ids cannot change for an existing review target')
            refs = sorted(set(args.evidence_ids or record.get('available_evidence_ids', []) or current.get('evidence_ids', []) or
                              [r['evidence_id'] for i in current['items'] for r in i.get('sources', [])]))
            sources = self._research_refs(task_id, refs)
            key = digest([current['content_hash'], questions, [s['location_key'] for s in sources]])
            prior = next((a for a in record['attempts'] if a['key'] == key), None)
            if prior and prior.get('result'):
                return self._research_projection(record, prior)
            if prior:
                attempt = prior
            else:
                previous = next((a for a in reversed(record['attempts']) if a['questions'] == questions), None)
                if any(not a.get('result') for a in record['attempts']):
                    raise ValueError('unsettled research review must be resumed with identical inputs')
                if previous and previous['target']['content_hash'] == current['content_hash']:
                    known = {s['location_key'] for s in previous['sources'] + previous.get('observed_sources', [])}
                    if {s['location_key'] for s in sources}.issubset(known):
                        # Reviewer已自行补读过的文字也是已处理证据；缩小证据包不产生新事实。
                        return self._research_projection(record, previous)
                stop = (previous or {}).get('stop_reason')
                if stop == 'supported' and previous['target']['content_hash'] != current['content_hash']:
                    stop = None
                if stop or len(record['attempts']) > state['limit'] or record['decisions']:
                    return {**self._research_projection(record, previous or record['attempts'][-1]),
                            'reflection': {'stop_reason': stop or 'round_limit' if not record['decisions'] else 'explicit_end',
                                           'can_continue': False, 'target_id': target_id},
                            'message': '核验已结束；保留成果和未解决原因，不重复调用。'}
                attempt = {'attempt_id': str(uuid4()), 'key': key, 'round': len(record['attempts']),
                    'questions': questions, 'reason': args.reason, 'target': json.loads(json.dumps(current)),
                    'sources': sources, 'evidence_ids': refs, 'status': 'started'}
                record['attempts'].append(attempt)
        result = execute(attempt['attempt_id'], attempt['target'], attempt['evidence_ids'], attempt['questions'])
        with self._locked(task_id) as (state, sql):
            record = state['research_targets'][target_id]
            current_attempt = next(a for a in record['attempts'] if a['attempt_id'] == attempt['attempt_id'])
            old = next((a for a in reversed(record['attempts'][:attempt['round']]) if a['questions'] == questions), None)
            if result.get('target_id') != target_id or result.get('content_hash') != attempt['target']['content_hash']:
                raise ValueError('review result must bind reserved target snapshot')
            current_attempt['result'] = result
            current_attempt['status'] = 'failed' if result['status'] == 'execution_failed' else 'completed'
            progress = review_progress((old or {}).get('result'), result)
            # 实际补读由成功Reviewer行动授权回根任务；不只比较入包证据。
            observed = self._research_refs(task_id, [ref for f in result.get('findings', []) for ref in f.get('evidence_ids', [])])
            previous_sources = (old or {}).get('sources', []) + (old or {}).get('observed_sources', [])
            progress['evidence_changed'] = bool(old and {s['location_key'] for s in sources + observed}.difference(s['location_key'] for s in previous_sources))
            current_attempt['progress'] = progress
            # Reviewer自行补读也属于实际证据范围，后续恢复不遗漏该部分身份。
            current_attempt['observed_sources'] = observed
            record['available_evidence_ids'] = sorted(set(attempt['evidence_ids'] + [s['evidence_id'] for s in observed]))
            if result['status'] == 'execution_failed':
                stop = 'review_execution_failed'
            elif result['status'] == 'supported':
                stop = 'supported'
            elif old and not progress['effective_change']:
                stop = 'no_progress'
            elif attempt['round'] >= state['limit']:
                stop = 'round_limit'
            else:
                stop = None
            current_attempt['stop_reason'] = stop
            return self._research_projection(record, current_attempt)

    @staticmethod
    def _research_projection(record: dict, attempt: dict) -> dict:
        """返回当前快照与核验所属版本，防止旧判断被展示为新版通过。"""
        from research_loop.domain.reflection import assessment_state
        current = record['current']
        result = attempt['result']
        same = result['content_hash'] == current['content_hash']
        # 新问题的通过不能覆盖同一对象上其他尚未解决的问题。
        latest_checks = {}
        for check in record['attempts']:
            if check.get('result'):
                latest_checks[digest(check['questions'])] = check
        pending = [c for c in latest_checks.values() if c['result']['status'] != 'supported']
        quality = result['status'] if same else 'unreviewed'
        if pending and quality == 'supported':
            quality = 'issues'
        return {**result, 'target': current,
                'current_version_reviewed': same,
                'quality_status': quality,
                'pending_checks': [{'questions': c['questions'], 'review_id': c['result'].get('review_id'),
                    'content_hash': c['result']['content_hash'], 'status': c['result']['status'],
                    'findings': c['result'].get('findings', []),
                    'execution_error': c['result'].get('execution_error')} for c in pending],
                # 结果内序号属于持久核验记录，同一对象多条意见不会再共用一个ID。
                'issues': [{**f, 'issue_id': digest([result['review_id'], f['item_id'], index])[:24],
                            'state': assessment_state(f)} for index, f in enumerate(result.get('findings', []))],
                'reflection': {'target_id': current['target_id'], 'round': attempt['round'],
                    'stop_reason': attempt.get('stop_reason'), 'can_continue': not attempt.get('stop_reason') and not record['decisions'],
                    'progress': attempt.get('progress'), 'trigger_reason': attempt['reason']},
                'decisions': record['decisions']}

    def revise_research(self, task_id: str, payload) -> dict:
        """原子追加核验对象局部版本，既有子任务和成功行动保持不可变。"""
        key = digest(payload.model_dump())
        with self._locked(task_id) as (state, sql):
            record = state.get('research_targets', {}).get(payload.target_id)
            if not record:
                raise ValueError('research target missing')
            replay = next((p for p in record['patches'] if p['key'] == key), None)
            if replay:
                return {'target': next(v for v in record['versions'] if v['content_hash'] == replay['after_hash']), 'revision': replay}
            current = record['current']
            latest = next((a for a in reversed(record['attempts']) if (a.get('result') or {}).get('review_id') == payload.review_id), None)
            if not latest or any(a['questions'] == latest['questions'] for a in record['attempts'][latest['round'] + 1:]):
                raise ValueError('revision review is missing or superseded for the same question')
            review = latest.get('result') or {}
            if payload.content_hash != current['content_hash'] or payload.review_id != review.get('review_id') or review.get('content_hash') != current['content_hash']:
                raise ValueError('revision must bind current target hash and latest settled review')
            if latest.get('stop_reason') or record['decisions']:
                raise ValueError('reflection closed; deliver retained work')
            if current['target_type'] == 'candidates':
                raise ValueError('candidate metadata is immutable; use exclude disposition')
            self._research_refs(task_id, current.get('evidence_ids', []))
            expected = latest['sources'] + latest.get('observed_sources', [])
            if {s['location_key'] for s in self._research_refs(task_id, [s['evidence_id'] for s in expected])} != {s['location_key'] for s in expected}:
                raise ValueError('review source version changed before revision')
            items = {i['item_id']: json.loads(json.dumps(i)) for i in current['items']}
            unresolved_ids = {f['item_id'] for f in review['findings'] if f.get('status') != 'supported'}
            edits, refs = {}, set(current.get('evidence_ids', []))
            for patch in payload.patches:
                if patch.item_id not in items or patch.item_id not in unresolved_ids:
                    raise ValueError('patch must target an unresolved selected item')
                text = items[patch.item_id]['text']
                if text.count(patch.old_text) != 1 or patch.old_text == patch.new_text:
                    raise ValueError('old_text must occur exactly once and replacement must change it')
                self._research_refs(task_id, patch.evidence_ids)
                refs.update(patch.evidence_ids)
                from research_loop.services.research_progress import ResearchProgressService
                item = items[patch.item_id]
                source_map = {r['evidence_id']: r for r in item.get('sources', [])}
                source_map.update({ref: ResearchProgressService(self.db).source(task_id, ref) for ref in patch.evidence_ids})
                item['sources'] = list(source_map.values())
                start = text.index(patch.old_text)
                edits.setdefault(patch.item_id, []).append((start, start + len(patch.old_text), patch.new_text))
            for item_id, spans in edits.items():
                spans.sort()
                if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                    raise ValueError('patch ranges overlap')
                for start, end, replacement in reversed(spans):
                    items[item_id]['text'] = items[item_id]['text'][:start] + replacement + items[item_id]['text'][end:]
            revised = {**current, 'items': list(items.values()), 'evidence_ids': sorted(refs)}
            # 内容身份只看对象原文，不靠新增证据或问题措辞绕过A→B→A检测。
            from research_loop.domain.reflection import target_fingerprint
            revised['content_hash'] = target_fingerprint(revised)
            text_identity = lambda v: sorted((i['item_id'], i['text']) for i in v['items'])
            if any(text_identity(v) == text_identity(revised) for v in record['versions']):
                raise ValueError('revision returns to an existing version; no progress')
            patch_record = {'key': key, 'review_id': payload.review_id, 'before_hash': current['content_hash'],
                            'after_hash': revised['content_hash'], 'patches': [p.model_dump() for p in payload.patches],
                            'non_target_text_preserved': True}
            record['patches'].append(patch_record)
            record['current'] = revised
            record['versions'].append(revised)
            return {'target': revised, 'quality_status': 'unreviewed', 'revision': patch_record}

    def close_research(self, task_id: str, payload) -> dict:
        """记录明确排除/披露/结束，原始核验意见与内容均保留。"""
        key = digest(payload.model_dump())
        with self._locked(task_id) as (state, sql):
            record = state.get('research_targets', {}).get(payload.target_id)
            if not record or record['current']['content_hash'] != payload.content_hash:
                raise ValueError('research target missing or stale')
            if record['decisions']:
                if record['decisions'][-1]['key'] != key:
                    raise ValueError('research disposition already recorded')
                return record['decisions'][-1]
            attempt = record['attempts'][-1]
            review = attempt.get('result') or {}
            derived = any(p['review_id'] == payload.review_id and p['after_hash'] == payload.content_hash for p in record['patches'])
            if review.get('review_id') != payload.review_id or (review.get('content_hash') != payload.content_hash and not derived):
                raise ValueError('close must bind current settled review')
            if payload.action == 'exclude' and record['current']['target_type'] != 'candidates':
                raise ValueError('exclude is for candidate suitability only')
            decision = {**payload.model_dump(), 'key': key,
                        'review_status': review['status'] if review['content_hash'] == payload.content_hash else 'unreviewed'}
            record['decisions'].append(decision)
            return decision

    def research_context(self, task_id: str) -> list[dict]:
        """构造当前核验对象投影；原件保留，过期判断不继承到新文字。"""
        output = []
        for record in self.snapshot(task_id).get('research_targets', {}).values():
            if record['attempts'] and record['attempts'][-1].get('result'):
                output.append(self._research_projection(record, record['attempts'][-1]))
        return output
