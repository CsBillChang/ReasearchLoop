"""整份请求的确定性工作集投影；原件外置，摘录不增加模型或改变核验结论。"""
from __future__ import annotations

import copy
import json

from research_loop.domain.context import excerpt
from research_loop.providers.token_counting import estimate
from research_loop.storage.memory import dump


def archive_location(archive: dict, path: list) -> tuple[int, int]:
    """计算稳定 JSON 中某个值的精确字符区间，供现有 read_memory 分页读取。

    Args:
        archive: 将用 dump 原样持久化的档案，不能在定位后继续修改。
        path: 字典键和列表下标组成的路径，避免相同文本的首次匹配歧义。

    Returns:
        原值序列化后的起点和字符数；不是字节数或模型 Token 数。
    """
    offset, node = 0, archive
    for key in path:
        if isinstance(node, dict):
            if key not in node:
                raise KeyError(key)
            offset += 1
            for sibling in sorted(node):
                offset += len(dump(sibling)) + 2
                if sibling == key:
                    break
                offset += len(dump(node[sibling])) + 2
        elif isinstance(node, list) and isinstance(key, int) and 0 <= key < len(node):
            offset += 1 + sum(len(dump(item)) + 2 for item in node[:key])
        else:
            raise ValueError('CONTEXT_ARCHIVE_RANGE: 工作集原件路径失效')
        node = node[key]
    return offset, len(dump(node))


def project_workspace(messages, context, tools, archive, archive_id, query, capacity, header):
    """按整份请求容量外置详情，保留目标、权限、发现身份和真实停止状态。

    Args:
        messages: 不含动态 Memory 头的消息，调用参数与供应商推理字段不改。
        context: 当前任务与历史的待发副本。
        tools: 当前实际可用工具 schema。
        archive: 完整原始 context 与已解码工具结果，已固定且将原样入库。
        archive_id: 当前任务档案身份，不跨任务读取。
        query: 当前问题，仅用于选择原文摘录，不做语义重写。
        capacity: 既有单请求容量，函数不新增累计限制。
        header: 将 context 包装为动态系统消息的函数。

    Returns:
        消息副本、上下文副本与投影记录。必要内容仍超限时由调用方明确结束。
    """
    output, current = copy.deepcopy(messages), copy.deepcopy(context)
    metrics = {'workspace_externalized': [], 'workspace_exact_links': []}

    def size():
        """统一计入工具、系统头和所有消息，不分别用各模块的局部长度判断。"""
        whole = output[:1] + [header(current)] + output[1:]
        return estimate({'messages': whole, 'tools': tools})

    def pointer(path):
        """读回参数只包含现有 HistoryRead 合同字段；长度信息单独存放。"""
        start, length = archive_location(archive, path)
        return {'source_type': 'context_archive', 'source_id': archive_id,
                'offset': start, 'limit': min(6000, max(100, length))}, length

    candidates = []
    children = {c['subtask_id']: (i, c) for i, c in enumerate(current.get('current_task', {}).get('subtasks', []))}
    calls = {c['id']: c['function']['name'] for m in output for c in m.get('tool_calls', [])}
    decoded = []
    for i, message in enumerate(output):
        call_id = message.get('tool_call_id')
        tool = calls.get(call_id)
        if message.get('role') != 'tool' or tool not in {'delegate_research', 'search_papers'}:
            continue
        try:
            value = json.loads(message['content'])
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict) or not value.get('ok'):
            continue
        if tool == 'search_papers':
            if call_id in archive['tool_results']:
                base = ['tool_results', call_id, 'data']
                # 身份、标题、日期和链接始终可见；先外置RPC诊断，再摘录长摘要。
                for j, source in enumerate(value.get('data', {}).get('source_results', [])):
                    if source.get('request'):
                        candidates.append((source, 'request', base + ['source_results', j, 'request']))
                for j, paper in enumerate(value.get('data', {}).get('candidates', [])):
                    if paper.get('metadata'):
                        candidates.append((paper, 'metadata', base + ['candidates', j, 'metadata']))
                for j, paper in enumerate(value.get('data', {}).get('candidates', [])):
                    if paper.get('abstract'):
                        candidates.append((paper, 'abstract', base + ['candidates', j, 'abstract']))
            decoded.append((i, value))
            continue
        for j, child in enumerate(value.get('data', {}).get('subtasks', [])):
            pair = children.get(child.get('subtask_id'))
            progress = child.get('research_progress') or {}
            # 仅同一交接版本且逐字段相等才引用已有缺口，不把相近表述合并。
            if pair and progress.get('context_path') == f'current_task.subtasks[{pair[0]}].research_progress':
                target = pair[1].get('research_progress') or {}
                if child.get('gaps') and child['gaps'] == target.get('gaps'):
                    child.pop('gaps')
                    child['gaps_in'] = progress['context_path'] + '.gaps'
                    metrics['workspace_exact_links'].append({'tool_call_id': call_id, 'subtask_id': child['subtask_id'], 'field': 'gaps'})
            if child.get('assessments') and call_id in archive['tool_results']:
                candidates.append((child, 'assessments', ['tool_results', call_id, 'data', 'subtasks', j, 'assessments']))
        decoded.append((i, value))

    def sync():
        """刷新已解码副本；原工具消息和原始 archive 不可变。"""
        for index, value in decoded:
            output[index]['content'] = dump(value)

    sync()
    if not any(t.get('function', {}).get('name') == 'read_memory' for t in tools):
        return output, current, metrics
    # 先卸载旧研究与诊断详情；只有容量仍不够才摘录当前发现的长正文。
    for i, item in enumerate(current.get('recent_research', [])):
        if item.get('stage_findings'):
            candidates.append((item, 'stage_findings', ['workspace_context', 'recent_research', i, 'stage_findings']))
    for i, item in enumerate(current.get('messages', [])):
        if item.get('role') == 'assistant' and isinstance(item.get('content'), str):
            candidates.append((item, 'content', ['workspace_context', 'messages', i, 'content']))
    for i, item in enumerate(current.get('related_history', [])):
        if item.get('record'):
            candidates.append((item, 'record', ['workspace_context', 'related_history', i, 'record']))
    progresses = [(current.get('current_task', {}).get('research_progress'), ['workspace_context', 'current_task', 'research_progress'])]
    progresses += [(c.get('research_progress'), ['workspace_context', 'current_task', 'subtasks', i, 'research_progress'])
                   for i, c in enumerate(current.get('current_task', {}).get('subtasks', []))]
    for progress, path in progresses:
        for i, finding in enumerate((progress or {}).get('findings', [])):
            if isinstance(finding.get('text'), str) and len(finding['text']) > 1200:
                candidates.append((finding, 'text', path + ['findings', i, 'text']))
    for owner, key, path in candidates:
        if size() <= capacity:
            break
        value = owner[key]
        preview = excerpt(value if isinstance(value, str) else dump(value), query, 500)
        readback, chars = pointer(path)
        detail = {'readback': readback, 'value_chars': chars, 'original_path': path,
                  'notice': '仅保留相关摘录；按 readback 分页读取累计 value_chars 字符即为完整原值。摘录不是核验通过或原文替代。'}
        replacement = preview if isinstance(value, str) else {'preview': preview, 'total_items': len(value), 'context_excerpted': True}
        if estimate(replacement) + estimate(detail) >= estimate(value):
            continue
        owner[key] = replacement
        owner[key + '_context_archive'] = detail
        if key == 'content':
            owner['excerpted'] = True
        metrics['workspace_externalized'].append({'path': path, 'value_chars': chars})
        sync()
    return output, current, metrics
