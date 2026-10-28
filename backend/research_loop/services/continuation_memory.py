"""实验接续索引：终态后校验模型候选，保留原始历史且不发送额外请求。"""
from __future__ import annotations

import json

from pydantic import ValidationError

from research_loop.domain.continuation_memory import ContinuationCandidate
from research_loop.storage.memory import MemoryStore, check_owner, digest, dump, record_operation


POLICY = 'continuation-policy:'
PROJECTION = 'continuation-projection:'
VERSION = 1
NOTICE = ('模型生成的历史接续索引，未经独立科学核验；当前要求优先，'
          '不继承保存权限，原文细节须按当前权限读取。')
INSTRUCTION = ('\n本新任务启用了接续记忆实验。请在同一次finish_task响应中附continuation_memory，'
    '按现有research_plan/research_progress整理目标、结论及条件、未解决问题和下一步。'
    '结论sources使用current_user/current_answer的逐字引句，或visible_history目录中的research_history任务身份与逐字引句。'
    '用户纠错只表示更正来源，不自动成为核验通过的科学事实。只有明确更正、补充条件或更新时才填写corrections，'
    '准确引用旧结论key/content_hash并指向本轮replacement_key。没有关系返回[]。'
    '不要把保存权限或临时要求写成长期偏好，不为摘要另行检索或调用模型；无法整理可以省略此字段。')


def record_identity(record: dict) -> str:
    """绑定原始研究内容，不把可恢复派生字段纳入来源身份。"""
    return digest({key: record.get(key) for key in
                   ('task_id', 'prompt', 'answer', 'research_progress', 'paper_ids', 'evidence_ids')})


class ContinuationMemoryService:
    """仅处理显式标记的新根任务；默认流程没有模型摘要副作用。"""

    def __init__(self, db):
        """复用调用方数据库，不迁移、不构造供应商。"""
        self.db = db
        self.store = MemoryStore(db)

    def enabled(self, task_id: str) -> bool:
        """读取冻结到任务的实验策略；恢复不从全局环境重新推断。"""
        row = self.db.fetchone('SELECT result_json FROM memory_operations WHERE request_key=?', (POLICY + task_id,))
        return bool(row and json.loads(row['result_json']).get('version') == VERSION)

    def enable_new_task(self, task_id: str, *, owner: tuple[str, str] | None = None) -> bool:
        """只标记尚未执行的新会话根；已领取时必须校验当前Worker租约。"""
        with self.store.transaction() as tx:
            if tx.one('SELECT request_key FROM memory_operations WHERE request_key=?', (POLICY + task_id,)):
                return True
            task = tx.one('SELECT * FROM tasks WHERE task_id=?', (task_id,))
            if task and task['status'] == 'running' and owner is not None:
                check_owner(tx, (task_id, *owner))
            fresh_status = task and (task['status'] == 'queued' or (task['status'] == 'running' and owner is not None))
            if (not fresh_status or task.get('parent_task_id')
                    or task.get('role') != 'supervisor'
                    or not tx.one('SELECT task_id FROM conversation_links WHERE task_id=?', (task_id,))
                    or tx.one('SELECT task_id FROM model_calls WHERE task_id=?', (task_id,))
                    or tx.one('SELECT task_id FROM actions WHERE task_id=?', (task_id,))
                    or tx.one('SELECT task_id FROM checkpoints WHERE task_id=?', (task_id,))):
                return False
            record_operation(tx, POLICY + task_id, VERSION, {'version': VERSION})
            return True

    def _earlier(self, current: dict, previous: dict) -> bool:
        """同会话用消息顺序，跨会话用严格早于当前任务的创建时间。"""
        rows = self.db.fetchall('SELECT l.task_id,l.conversation_id,m.sequence_no FROM conversation_links l '
            'JOIN conversation_messages m ON m.message_id=l.message_id WHERE l.task_id IN (?,?)',
            (current['task_id'], previous['task_id']))
        mapping = {row['task_id']: row for row in rows}
        left, right = mapping.get(current['task_id']), mapping.get(previous['task_id'])
        if left and right and left['conversation_id'] == right['conversation_id']:
            return right['sequence_no'] < left['sequence_no']
        return previous['created_at'] < current['created_at']

    def _record(self, task_id: str) -> dict:
        """读取既存研究历史；不存在的模型引用不得补造。"""
        row = self.db.fetchone('SELECT record_json FROM research_memories WHERE task_id=?', (task_id,))
        if not row:
            raise ValueError('CONTINUATION_HISTORY_NOT_FOUND')
        return json.loads(row['record_json'])

    def context(self, task_id: str, seed_ids: list[str], *, max_tokens: int = 4000) -> dict:
        """提供相关历史及显式更正闭包，索引不授予论文或保存权限。"""
        if not self.enabled(task_id):
            return {'enabled': False}
        current = self.db.get_task(task_id)
        from research_loop.services.continuation_scope import revoked_history_tasks
        blocked = revoked_history_tasks(self.db)
        records = {}
        for row in self.db.fetchall('SELECT record_json FROM research_memories ORDER BY created_at,task_id'):
            record = json.loads(row['record_json'])
            if record['task_id'] not in blocked and self._earlier(current, record):
                records[record['task_id']] = record
        selected = set(seed_ids) & records.keys()
        # 每轮只扩展与已有集合相连的更正；严格时间序保证无环。
        changed = True
        while changed:
            changed = False
            for identity, record in records.items():
                note = record.get('continuation_memory') or {}
                if note.get('status') != 'applied' or note.get('source_record_hash') != record_identity(record):
                    continue
                targets = {item['previous_task_id'] for item in note.get('corrections', [])}
                if identity in selected or targets & selected:
                    additions = (({identity} | targets) & records.keys()) - selected
                    if additions:
                        selected.update(additions)
                        changed = True
        notes, visible = [], []
        for identity in sorted(selected, key=lambda key: (records[key]['created_at'], key)):
            record = records[identity]
            visible.append({'task_id': identity, 'record_hash': record_identity(record)})
            note = record.get('continuation_memory') or {}
            if note.get('status') == 'applied' and note.get('source_record_hash') == record_identity(record):
                notes.append({'task_id': identity, 'research_goal': note['research_goal'],
                    'conclusions': note['conclusions'], 'corrections': note['corrections'],
                    'unresolved': note['unresolved'], 'next_steps': note['next_steps'],
                    'readback': {'source_type': 'research_history', 'source_id': identity}})
        payload = {'enabled': True, 'version': VERSION, 'visible_history': visible,
                   'notes': notes, 'notice': NOTICE, 'omitted_notes': 0, 'omitted_history': 0}
        from research_loop.providers.token_counting import estimate
        # 只卸载完整摘要，不裁断结论的条件或更正句；原件仍通过历史身份读回。
        while notes and estimate(payload) > max_tokens:
            notes.pop(0)
            payload['omitted_notes'] += 1
        while visible and estimate(payload) > max_tokens:
            visible.pop(0)
            payload['omitted_history'] += 1
        if estimate(payload) > max_tokens:
            return {'enabled': True, 'notes': [], 'visible_history': [], 'omitted_history': len(selected),
                    'notice': '接续目录超出额度；沿用原始历史读回，当前要求与权限保持。'}
        return payload

    def _visible_history(self, task_id: str) -> dict[str, dict]:
        """复用实际收据和观察的可见片段，不从数据库全文补造模型输入。"""
        from research_loop.services.continuation_scope import visible_history
        return visible_history(self.db, task_id, record_identity)

    def _validate(self, task: dict, record: dict, candidate: ContinuationCandidate) -> dict:
        """校验可核实的来源和关系；不把结构检查冒充语义核验。"""
        if len(dump(candidate.model_dump()).encode('utf-8')) > 24000:
            raise ValueError('CONTINUATION_TOO_LARGE')
        from research_loop.services.continuation_scope import quote_visible, revoked_history_tasks
        visible = self._visible_history(task['task_id'])
        blocked = revoked_history_tasks(self.db)
        conclusions, keys = [], set()
        for value in candidate.conclusions:
            if value.key in keys:
                raise ValueError('CONTINUATION_DUPLICATE_KEY')
            keys.add(value.key)
            if any(not item or len(item) > 500 for item in value.conditions):
                raise ValueError('CONTINUATION_INVALID_CONDITION')
            sources = []
            for source in value.sources:
                if source.source_type in {'current_user', 'current_answer'}:
                    if source.task_id not in {None, task['task_id']}:
                        raise ValueError('CONTINUATION_CURRENT_SOURCE_MISMATCH')
                    content = record['answer'] if source.source_type == 'current_answer' else task['prompt']
                    if source.source_type == 'current_user':
                        content += '\n' + '\n'.join(row['answer'] for row in self.db.fetchall(
                            "SELECT answer FROM clarifications WHERE task_id=? AND status='answered'", (task['task_id'],)))
                    source_id = task['task_id']
                else:
                    previous = self._record(source.task_id or '')
                    if (source.task_id in blocked
                            or visible.get(source.task_id, {}).get('record_hash') != record_identity(previous)
                            or not self._earlier(task, previous)):
                        raise ValueError('CONTINUATION_HISTORY_OUTSIDE_VISIBLE_SCOPE')
                    content = previous['prompt'] + '\n' + previous['answer']
                    if not quote_visible(source.quote, visible[source.task_id]['texts']):
                        raise ValueError('CONTINUATION_QUOTE_OUTSIDE_VISIBLE_WINDOW')
                    source_id = source.task_id
                if source.quote not in content:
                    raise ValueError('CONTINUATION_QUOTE_MISMATCH')
                sources.append({'source_type': source.source_type, 'task_id': source_id,
                                'quote': source.quote, 'content_hash': digest(content)})
            item = {'key': value.key, 'text': value.text, 'conditions': value.conditions, 'sources': sources,
                    'verification_status': 'historical_unverified'}
            item['content_hash'] = digest(item)
            conclusions.append(item)
        corrections, targets = [], set()
        for relation in candidate.corrections:
            previous = self._record(relation.previous_task_id)
            note = previous.get('continuation_memory') or {}
            old = next((item for item in note.get('conclusions', []) if item['key'] == relation.previous_key), None)
            replacement = next((item for item in conclusions if item['key'] == relation.replacement_key), None)
            target = (relation.previous_task_id, relation.previous_key)
            if (relation.previous_task_id in blocked
                    or visible.get(relation.previous_task_id, {}).get('record_hash') != record_identity(previous)
                    or not self._earlier(task, previous) or note.get('status') != 'applied'
                    or note.get('source_record_hash') != record_identity(previous)
                    or not old or old['content_hash'] != relation.previous_content_hash
                    or not quote_visible(relation.previous_content_hash,
                        visible.get(relation.previous_task_id, {}).get('texts', []))):
                raise ValueError('CONTINUATION_STALE_OR_INVISIBLE_TARGET')
            if target in targets or replacement is None:
                raise ValueError('CONTINUATION_INVALID_REPLACEMENT')
            if not any(source['source_type'] in {'current_user', 'current_answer'} for source in replacement['sources']):
                raise ValueError('CONTINUATION_CORRECTION_NEEDS_CURRENT_SOURCE')
            targets.add(target)
            corrections.append(relation.model_dump())
        for text in [*candidate.unresolved, *candidate.next_steps]:
            if not text or len(text) > 1500:
                raise ValueError('CONTINUATION_INVALID_NEXT_STEP')
        return {'research_goal': candidate.research_goal, 'conclusions': conclusions,
                'corrections': corrections, 'unresolved': candidate.unresolved, 'next_steps': candidate.next_steps,
                'history_dependencies': sorted(visible)}

    def apply(self, task_id: str) -> dict:
        """回答终态后独立投影；事务重放只读回原结果，不再调用模型。"""
        if not self.enabled(task_id):
            return {'status': 'disabled'}
        task = self.db.get_task(task_id)
        if task['status'] not in {'completed', 'partial', 'failed', 'cancelled'}:
            return {'status': 'not_ready'}
        from research_loop.services.conversations import ConversationService
        ConversationService(self.db).reconcile(task_id)
        with self.store.transaction() as tx:
            old = tx.one('SELECT result_json FROM memory_operations WHERE request_key=?', (PROJECTION + task_id,))
            if old:
                return json.loads(old['result_json'])
            record = self._record(task_id)
            action = tx.one("SELECT * FROM actions WHERE task_id=? AND tool_name='finish_task' AND status='succeeded' ORDER BY sequence_no DESC LIMIT 1", (task_id,))
            result = {'version': VERSION, 'status': 'not_proposed', 'notice': NOTICE,
                      'source_record_hash': record_identity(record)}
            if task['status'] in {'completed', 'partial'} and action:
                args = json.loads(action['arguments_json'])
                result['finish_action_id'] = action['action_id']
                result['finish_arguments_hash'] = digest(args)
                raw = args.get('continuation_memory')
                if raw is not None:
                    try:
                        candidate = ContinuationCandidate.model_validate(raw)
                        result.update(self._validate(task, record, candidate), status='applied')
                    except (ValueError, ValidationError) as exc:
                        result.update(status='rejected', reason=str(exc).split('\n', 1)[0])
            record['continuation_memory'] = result
            search = record['prompt'] + '\n' + record['answer'] + '\n' + dump(record.get('papers', []))
            if result['status'] == 'applied':
                search += '\n' + dump({key: result[key] for key in ('research_goal', 'conclusions', 'unresolved', 'next_steps')})
            tx.execute('UPDATE research_memories SET record_json=?,search_text=? WHERE task_id=?',
                       (dump(record), search, task_id))
            record_operation(tx, PROJECTION + task_id, result.get('finish_arguments_hash'), result)
            return result

    def recover_pending(self) -> list[dict]:
        """仅补带实验策略且已经终态的投影，不领取或发送研究任务。"""
        rows = self.db.fetchall('SELECT request_key FROM memory_operations WHERE request_key LIKE ?', (POLICY + '%',))
        return [self.apply(row['request_key'][len(POLICY):]) for row in rows]
