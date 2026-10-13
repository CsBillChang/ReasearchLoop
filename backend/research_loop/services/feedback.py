"""当前稿反馈交接：原子入队并复用确切稿件和原文，不复制旧结论或执行状态。"""
from __future__ import annotations

import json
import hashlib
from uuid import uuid4, uuid5, NAMESPACE_URL

from research_loop.domain.tasks import TaskCreate
from research_loop.domain.reading import user_page_range
from research_loop.services.conversations import ConversationService, TERMINAL
from research_loop.services.papers import PaperService
from research_loop.storage.memory import MemoryStore, dump
from research_loop.storage.mysql import _now


def submit_draft_feedback(db, payload, draft_id: str) -> str:
    """将同会话当前草稿原样交接给新任务；事务提交前Worker看不到半成品。

    Args:
        db: 当前业务数据库。
        payload: 已验证的反馈输入，保存权限只取本次字段。
        draft_id: 被反馈的源任务当前草稿。

    Returns:
        幂等的新任务身份；旧任务、审校、收据和文件不变。

    Raises:
        ValueError: 源稿过期、身份/hash不符或会话忙。
        KeyError: 反馈来源不在当前会话。
    """
    old_scope = PaperService(db, None, None).user_scope(payload.source_task_id)
    scope = user_page_range([payload.prompt]) or old_scope
    target_id = str(uuid5(NAMESPACE_URL, 'researchloop-feedback:' + payload.conversation_id + ':' + payload.idempotency_key))
    with MemoryStore(db).transaction() as tx:
        if not tx.one('SELECT * FROM conversation_links WHERE task_id=? AND conversation_id=?',
                      (payload.source_task_id, payload.conversation_id)):
            raise KeyError('反馈对象不在当前对话')
        source = tx.one('SELECT * FROM tasks WHERE task_id=?', (payload.source_task_id,))
        draft = tx.one('SELECT * FROM drafts WHERE draft_id=? AND task_id=?', (draft_id, payload.source_task_id))
        latest = tx.one('SELECT draft_id FROM drafts WHERE task_id=? ORDER BY created_at DESC,draft_id DESC LIMIT 1', (payload.source_task_id,))
        if (not draft or source['status'] not in TERMINAL or latest['draft_id'] != draft_id
                or draft['content_hash'] != payload.content_hash
                or hashlib.sha256(draft['markdown'].encode()).hexdigest() != payload.content_hash):
            raise ValueError('反馈草稿身份或版本已变化，请刷新')
        prompt = payload.prompt + '\n\n本次接续已持久化的当前候选稿，不需create_note重建。新任务稿件ID=' + target_id + '，hash=' + draft['content_hash']
        prompt += '\n正文在Worker的候选稿数据中提供；可用review_note核验，finish_task用此draft_id交付。已有原文将以实际证据身份继承，不继承旧核验意见或保存权限。'
        if scope:
            prompt += f'\n本次仍只读物理第{scope.start}至{scope.end}页。'
        task_input = TaskCreate(conversation_id=payload.conversation_id, prompt=prompt,
            attachments=json.loads(source['attachments_json']), save_policy=payload.save_policy,
            idempotency_key=payload.idempotency_key)
        tid = ConversationService(db).submit_tx(tx, task_input)
        if tx.one('SELECT draft_id FROM drafts WHERE draft_id=? AND task_id=?', (target_id, tid)):
            return tid
        refs = []
        # 只继承真实原文记录；历史摘要、Reviewer意见和unknown收据不会成为新任务证据。
        for row in tx.all('SELECT e.evidence_id,e.page_number FROM page_evidence e JOIN task_evidence t ON e.evidence_id=t.evidence_id WHERE t.task_id=?', (payload.source_task_id,)):
            if scope and (not row['page_number'] or not scope.start <= row['page_number'] <= scope.end):
                continue
            refs.append(row['evidence_id'])
            tx.execute('INSERT INTO task_evidence(task_id,evidence_id,returned_at) VALUES(?,?,?)', (tid,row['evidence_id'],_now()))
        for row in tx.all('SELECT * FROM task_papers WHERE task_id=?', (payload.source_task_id,)):
            tx.execute('INSERT INTO task_papers(task_id,paper_id,permission,granted_by,created_at) VALUES(?,?,?,?,?)',
                       (tid,row['paper_id'],'read',payload.source_task_id,_now()))
        tx.execute('INSERT INTO drafts(draft_id,task_id,title,markdown,content_hash,evidence_ids_json,status,created_at) VALUES(?,?,?,?,?,?,?,?)',
                   (target_id,tid,draft['title'],draft['markdown'],draft['content_hash'],dump(refs),'draft',_now()))
        tx.execute("UPDATE tasks SET save_status='draft_created',quality_status='unreviewed' WHERE task_id=?", (tid,))
        data = {'source_task_id':payload.source_task_id,'source_draft_id':draft_id,
                'draft_id':target_id,'content_hash':draft['content_hash'],'evidence_ids':refs,
                'origin':'explicit_current_draft_feedback','review_inherited':False}
        tx.execute('INSERT INTO task_events(event_id,task_id,event_type,payload_json,created_at) VALUES(?,?,?,?,?)',
                   (str(uuid4()),tid,'feedback_draft_bound',dump(data),_now()))
        return tid
