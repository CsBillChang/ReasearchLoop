"""接续索引的历史可见范围；只使用已留证片段，不把未读全文补成已见。"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

from research_loop.storage.memory import digest


def _object(value: Any) -> dict:
    """解析持久JSON对象；无效旁路数据不扩大可见范围。"""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _strings(value: Any) -> Iterator[str]:
    """遍历实际呈现字段中的字符串，不拼接本来不相邻的内容。"""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def quote_visible(quote: str, texts: list[str]) -> bool:
    """判断引句是否真正出现在已呈现片段中，兼容JSON字符串转义。

    Args:
        quote: 待核对的原始引句。
        texts: 已记录的上下文、索引卡或分页窗口字符串。

    Returns:
        原始引句或其JSON转义形式完整命中一个片段时返回True。
    """
    if not isinstance(quote, str) or not quote:
        return False
    escaped = json.dumps(quote, ensure_ascii=False)[1:-1]
    return any(isinstance(text, str) and (quote in text or escaped in text) for text in texts)


def visible_history(db, task_id: str, identity_of: Callable[[dict], str]) -> dict[str, dict]:
    """从成功模型请求与工具观察恢复历史身份和实际可见片段。

    Args:
        db: 当前隔离数据库，提供模型收据和行动读接口。
        task_id: 当前根任务身份。
        identity_of: 对原始研究记录计算稳定身份的函数。

    Returns:
        任务身份到record_hash及texts的映射。来源改变或只有未留证全文时不收录。
    """
    visible: dict[str, dict] = {}
    observations = []

    def load(identity):
        """读取当前记录仅用于核身份；返回全文不会自动加入可见片段。"""
        if not isinstance(identity, str) or not identity:
            return None
        row = db.fetchone('SELECT record_json FROM research_memories WHERE task_id=?', (identity,))
        if not row:
            return None
        record = _object(row['record_json'])
        if record.get('task_id') != identity:
            return None
        return record, row['record_json']

    def add(identity, expected_hash, texts):
        """先验证观察时身份，再累积对应来源实际呈现的字符串。"""
        loaded = load(identity)
        if loaded is None or not isinstance(expected_hash, str):
            return
        record, _ = loaded
        if identity_of(record) != expected_hash:
            return
        entry = visible.setdefault(identity, {'record_hash': expected_hash, 'texts': []})
        if entry['record_hash'] != expected_hash:
            return
        for text in texts:
            if isinstance(text, str) and text and text not in entry['texts']:
                entry['texts'].append(text)

    for call in db.list_model_calls(task_id):
        if call.get('status') != 'succeeded':
            continue
        request = _object(call.get('request_json'))
        names = {item.get('id'): item.get('function', {}).get('name')
                 for message in request.get('messages', []) for item in message.get('tool_calls', [])}
        for message in request.get('messages', []):
            name = names.get(message.get('tool_call_id'))
            if message.get('role') == 'tool' and name in {'search_memory', 'read_memory'}:
                observation = _object(message.get('content'))
                if observation.get('ok'):
                    observations.append((name, observation.get('data') or {}))
            content = message.get('content')
            if (message.get('role') != 'system' or not isinstance(content, str)
                    or not content.startswith('[ResearchLoop Memory v1]\n')):
                continue
            context = _object(content.rsplit('\n', 1)[-1])
            continuation = context.get('continuation_memory') or {}
            declared = {item.get('task_id'): item.get('record_hash')
                        for item in continuation.get('visible_history', []) if isinstance(item, dict)}
            for identity, expected_hash in declared.items():
                add(identity, expected_hash, [])
            for item in context.get('related_history', []):
                identity = item.get('source_id')
                if item.get('source_type') == 'research_history' and identity in declared:
                    add(identity, declared[identity], _strings(item.get('record', {})))
            for item in context.get('recent_research', []):
                identity = item.get('task_id')
                if identity in declared:
                    add(identity, declared[identity], _strings(item))
            for item in context.get('messages', []):
                identity = item.get('task_id')
                if identity in declared:
                    add(identity, declared[identity], _strings(item.get('content')))
            for item in continuation.get('notes', []):
                identity = item.get('task_id')
                if identity in declared:
                    add(identity, declared[identity], _strings(item))

    # 同一响应批次的read+finish尚未向模型呈现观察，不能事后补成已见来源。
    for name, data in observations:
        if name == 'search_memory':
            for item in data.get('matches', []):
                if item.get('source_type') == 'research_history':
                    add(item.get('source_id'), item.get('record_hash'), _strings(item.get('record', {})))
        elif name == 'read_memory' and data.get('source_type') == 'research_history':
            identity = data.get('source_id')
            loaded = load(identity)
            if loaded is None:
                continue
            record, raw = loaded
            content, offset = data.get('content'), data.get('offset')
            # 页观察的hash绑定完整原件，而可引用范围始终只限实际返回的窗口。
            if (not isinstance(content, str) or not isinstance(offset, int) or offset < 0
                    or data.get('content_hash') != digest(raw)
                    or raw[offset:offset + len(content)] != content):
                continue
            add(identity, identity_of(record), [content])
    return visible


def revoked_history_tasks(db) -> set[str]:
    """保守排除撤销来源及引用这些来源的派生接续摘要。

    Args:
        db: 当前数据库，不修改偏好或研究历史原件。

    Returns:
        应从摘要目录排除的任务身份，包括明确来源和传递依赖。
    """
    revoked = set()
    for row in db.fetchall("SELECT source_json FROM memory_entries WHERE status='revoked'"):
        source = _object(row.get('source_json'))
        if isinstance(source.get('task_id'), str):
            revoked.add(source['task_id'])
    dependencies = {}
    for row in db.fetchall('SELECT record_json FROM research_memories'):
        record = _object(row.get('record_json'))
        note = record.get('continuation_memory') or {}
        if not record.get('task_id') or note.get('status') != 'applied':
            continue
        parents = {source.get('task_id') for item in note.get('conclusions', [])
                   for source in item.get('sources', []) if isinstance(source.get('task_id'), str)}
        parents.update(item.get('previous_task_id') for item in note.get('corrections', [])
                       if isinstance(item.get('previous_task_id'), str))
        parents.update(identity for identity in note.get('history_dependencies', []) if isinstance(identity, str))
        dependencies[record['task_id']] = parents
    while True:
        derived = {identity for identity, parents in dependencies.items() if parents & revoked}
        if derived <= revoked:
            return revoked
        revoked.update(derived)
