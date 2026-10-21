"""将同一次研究结束响应中的兴趣候选投影为可撤销画像，不调用模型。"""
from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from pydantic import ValidationError

from research_loop.domain.interests import InterestCandidate, normalize_topic_key
from research_loop.storage.memory import MemoryStore, dump
from research_loop.storage.mysql import _now


POLICY_VERSION = 1
TERMINAL = {'completed', 'partial', 'failed', 'cancelled'}


class InterestService:
    """只处理新任务的成功结束行动；画像错误与研究交付分别持久化。"""

    def __init__(self, db):
        """复用数据库，不创建任务、迁移或模型请求。"""
        self.db = db
        self.store = MemoryStore(db)

    def enable_tx(self, tx, task_id: str) -> None:
        """在新根任务提交事务内启用；已有任务不自动补标记。"""
        if tx.one('SELECT task_id FROM interest_projections WHERE task_id=?', (task_id,)):
            return
        now = _now()
        tx.execute('INSERT INTO interest_projections(task_id,policy_version,status,result_json,created_at,updated_at) VALUES(?,?,?,?,?,?)',
                   (task_id, POLICY_VERSION, 'pending', None, now, now))

    def context(self, task_id: str) -> dict:
        """返回轻量主题身份与墓碑，不重复注入历史引句目录。"""
        marker = self.db.fetchone('SELECT policy_version FROM interest_projections WHERE task_id=?', (task_id,))
        if not marker or marker['policy_version'] != POLICY_VERSION:
            return {'enabled': False}
        latest = {}
        for row in self.db.fetchall('SELECT memory_id,slot,kind,content,source_json,status,version FROM memory_entries ORDER BY version'):
            latest[row['slot']] = row
        identities = []
        for row in latest.values():
            if row['kind'] != 'inferred':
                continue
            source = json.loads(row['source_json'])
            identities.append({'memory_id': row['memory_id'], 'version': row['version'],
                'topic_key': source.get('topic_key', row['slot']), 'content': row['content'],
                'status': row['status'], 'distinct_task_count': source.get('distinct_task_count', 1)})
        return {'enabled': True, 'policy_version': POLICY_VERSION, 'known_interests': identities,
                'source_scope': '仅当前根用户问题与已回答澄清的逐字片段；历史依据只能通过已有兴趣ID/版本合并'}

    def apply(self, task_id: str) -> dict:
        """研究终态后原子应用旁路；重放返回原处理结果，无外部请求。

        Args:
            task_id: 要投影的新会话根任务。

        Returns:
            独立处理状态与逐候选结果。旧任务和未终态任务保持原数据。
        """
        with self.store.transaction() as tx:
            marker = tx.one('SELECT * FROM interest_projections WHERE task_id=?', (task_id,))
            if not marker:
                return {'task_id': task_id, 'status': 'legacy', 'outcomes': []}
            if marker['status'] != 'pending':
                return json.loads(marker['result_json'])
            task = tx.one('SELECT * FROM tasks WHERE task_id=?', (task_id,))
            if not task or task['status'] not in TERMINAL:
                return {'task_id': task_id, 'status': 'not_ready', 'outcomes': []}
            if marker['policy_version'] != POLICY_VERSION:
                return self._complete(tx, task_id, 'rejected', [], reason='unsupported_policy')
            # 终态身份来自数据库。此路径不是人工remember，不跳过其运行中租约合同。
            if task.get('parent_task_id') or task.get('role') != 'supervisor':
                return self._complete(tx, task_id, 'rejected', [], reason='root_supervisor_required')
            action = tx.one("SELECT * FROM actions WHERE task_id=? AND tool_name='finish_task' AND status='succeeded' ORDER BY sequence_no DESC LIMIT 1", (task_id,))
            if task['status'] not in {'completed', 'partial'} or not action:
                return self._complete(tx, task_id, 'no_candidates', [], reason='no_successful_delivery')
            args = json.loads(action['arguments_json'])
            if isinstance(args, dict) and 'interest_candidates' not in args:
                return self._complete(tx, task_id, 'not_proposed', [], action_id=action['action_id'], reason='field_omitted')
            raw = args.get('interest_candidates', []) if isinstance(args, dict) else None
            if not isinstance(raw, list):
                return self._complete(tx, task_id, 'rejected', [], action_id=action['action_id'], reason='candidates_must_be_list')
            outcomes = []
            for index, value in enumerate(raw):
                try:
                    candidate = InterestCandidate.model_validate(value)
                    evidence = self._source(tx, task, candidate.source_quote)
                    result = self._put(tx, task, candidate, evidence, action['action_id'])
                    outcomes.append({'index': index, **result})
                except (ValidationError, ValueError) as exc:
                    # 格式或业务拒绝单独记账；其余候选及已交付回答继续保留。
                    outcomes.append({'index': index, 'status': 'rejected', 'reason': str(exc)})
            accepted = any(row['status'] in {'applied', 'reused'} for row in outcomes)
            status = 'applied' if accepted else ('rejected' if outcomes else 'no_candidates')
            return self._complete(tx, task_id, status, outcomes, action_id=action['action_id'])

    def recover_pending(self) -> list[dict]:
        """只恢复带新策略标记的终态投影，不领取或重跑研究任务。"""
        rows = self.db.fetchall("SELECT p.task_id FROM interest_projections p JOIN tasks t ON t.task_id=p.task_id WHERE p.status='pending' AND t.status IN ('completed','partial','failed','cancelled') ORDER BY p.created_at")
        return [self.apply(row['task_id']) for row in rows]

    def sources(self, memory_id: str) -> list[dict]:
        """按画像身份读回独立来源；画像列表和模型上下文不复制全部历史。"""
        row = self.db.fetchone("SELECT slot,version FROM memory_entries WHERE memory_id=? AND kind='inferred'", (memory_id,))
        if not row:
            raise KeyError('inferred interest not found')
        return [json.loads(item['source_json']) for item in self.db.fetchall(
            'SELECT source_json FROM interest_sources WHERE interest_slot=? AND added_version<=? ORDER BY created_at,task_id', (row['slot'], row['version']))]

    @staticmethod
    def _complete(tx, task_id: str, status: str, outcomes: list[dict], **details) -> dict:
        """与画像版本同事务保存处理结果，避免写入后崩溃重复计数。"""
        result = {'task_id': task_id, 'status': status, 'outcomes': outcomes, **details}
        tx.execute('UPDATE interest_projections SET status=?,result_json=?,updated_at=? WHERE task_id=?',
                   (status, dump(result), _now(), task_id))
        return result

    @staticmethod
    def _source(tx, task: dict, quote: str) -> dict:
        """仅在当前真实用户问题/已回答澄清中绑定引句与原文字节hash。"""
        task_id = task['task_id']
        link = tx.one('SELECT message_id FROM conversation_links WHERE task_id=?', (task_id,))
        if not link:
            raise ValueError('INTEREST_SOURCE: conversation root user message required')
        message = tx.one("SELECT * FROM conversation_messages WHERE message_id=? AND task_id=? AND role='user'", (link['message_id'], task_id))
        if message and quote in message['content'] and quote in task['prompt']:
            return {'task_id': task_id, 'source_type': 'message', 'source_id': message['message_id'],
                    'message_id': message['message_id'], 'content_hash': hashlib.sha256(message['content'].encode()).hexdigest(), 'quote': quote}
        for clarification in tx.all("SELECT c.clarification_id,c.answer,r.answer AS resumed_answer FROM clarifications c JOIN resume_requests r ON r.task_id=c.task_id AND r.clarification_id=c.clarification_id WHERE c.task_id=? AND c.status='answered'", (task_id,)):
            # Python逐字比对，不能借MySQL大小写不敏感排序伪造恢复来源。
            if clarification['answer'] == clarification['resumed_answer'] and quote in (clarification['answer'] or ''):
                return {'task_id': task_id, 'source_type': 'clarification', 'source_id': clarification['clarification_id'],
                        'content_hash': hashlib.sha256(clarification['answer'].encode()).hexdigest(), 'quote': quote}
        raise ValueError('INTEREST_SOURCE: 引句必须来自当前根用户问题或已回答澄清')

    @staticmethod
    def _put(tx, task: dict, candidate: InterestCandidate, evidence: dict, action_id: str) -> dict:
        """串行合并推断依据，保留显式优先与撤销墓碑；不按模型调用数计轮次。"""
        key = normalize_topic_key(candidate.topic_key)
        # MySQL默认排序可能忽略重音；ASCII hash确保不同规范主题不会误合并。
        slot = 'interest:auto:' + hashlib.sha256(key.encode()).hexdigest()
        target = None
        if candidate.merge_memory_id:
            target = tx.one('SELECT * FROM memory_entries WHERE memory_id=?', (candidate.merge_memory_id,))
            if not target or target['kind'] != 'inferred':
                raise ValueError('INTEREST_MERGE: only an existing inferred entry can be merged')
            if target['status'] != 'active':
                raise ValueError('INTEREST_REVOKED: inactive or revoked interests cannot be restored')
            if target['version'] != candidate.expected_version:
                raise ValueError('INTEREST_STALE_VERSION: merge version changed')
            slot = target['slot']
        latest = tx.one('SELECT * FROM memory_entries WHERE slot=? ORDER BY version DESC LIMIT 1', (slot,))
        # 同主题即使曾换槽，墓碑中的稳定主题键仍然有效；语义同义词不由字串规则假装解决。
        for revoked in tx.all("SELECT slot,source_json FROM memory_entries WHERE status='revoked'"):
            source = json.loads(revoked['source_json'])
            if revoked['slot'] == slot or normalize_topic_key(str(source.get('topic_key', ''))) == key:
                raise ValueError('INTEREST_REVOKED: topic was revoked; do not revive it by inference')
        if latest and latest['status'] == 'revoked':
            raise ValueError('INTEREST_REVOKED: topic was revoked')
        if latest and latest['kind'] == 'explicit':
            raise ValueError('INTEREST_PRIORITY: inferred interest cannot overwrite explicit preference')
        if target and latest and latest['memory_id'] != target['memory_id']:
            raise ValueError('INTEREST_STALE_VERSION: merge head changed')
        source = json.loads(latest['source_json']) if latest else {}
        version = latest['version'] + 1 if latest else 1
        if latest and source.get('type') != 'behavioral_interest' and source.get('task_id') and source.get('quote'):
            # 只延用已有画像实际保留的来源，不重扫或抽取生产历史。
            if not tx.one('SELECT task_id FROM interest_sources WHERE interest_slot=? AND task_id=?', (slot, source['task_id'])):
                prior_task = tx.one('SELECT * FROM tasks WHERE task_id=?', (source['task_id'],))
                if not prior_task:
                    raise ValueError('INTEREST_SOURCE: existing interest source task is missing')
                previous = InterestService._source(tx, prior_task, source['quote'])
                tx.execute('INSERT INTO interest_sources(interest_slot,task_id,added_version,source_json,created_at) VALUES(?,?,?,?,?)',
                           (slot, source['task_id'], latest['version'], dump(previous), _now()))
        if tx.one('SELECT task_id FROM interest_sources WHERE interest_slot=? AND task_id=?', (slot, task['task_id'])):
            return {'status': 'reused', 'memory_id': latest['memory_id'], 'version': latest['version']}
        tx.execute('INSERT INTO interest_sources(interest_slot,task_id,added_version,source_json,created_at) VALUES(?,?,?,?,?)',
                   (slot, task['task_id'], version, dump(evidence), _now()))
        distinct = tx.one('SELECT COUNT(*) AS n FROM interest_sources WHERE interest_slot=?', (slot,))['n']
        next_source = {'type': 'behavioral_interest', 'task_id': task['task_id'], 'quote': candidate.source_quote,
            'topic_key': source.get('topic_key', key), 'reason': candidate.reason,
            'source_ref': {'source_type': 'interest_sources', 'slot': slot},
            'strength': 'tentative' if distinct == 1 else 'repeated', 'distinct_task_count': distinct,
            'finish_action_id': action_id, 'policy_version': POLICY_VERSION}
        identity = str(uuid4())
        if latest:
            tx.execute("UPDATE memory_entries SET status='superseded' WHERE slot=? AND status='active'", (slot,))
        tx.execute('INSERT INTO memory_entries(memory_id,slot,kind,content,source_json,status,version,created_at) VALUES(?,?,?,?,?,?,?,?)',
                   (identity, slot, 'inferred', candidate.content, dump(next_source), 'active', version, _now()))
        return {'status': 'applied', 'memory_id': identity, 'version': version,
                'distinct_task_count': distinct, 'strength': next_source['strength']}
