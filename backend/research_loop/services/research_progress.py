"""从成功行动重放阶段成果；快照可落后，发现不从内部推理日志猜测。"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy

from research_loop.domain.research_progress import ResearchProgressInput
from research_loop.storage.memory import MemoryStore, check_owner, digest, dump


class UnknownFindingIdError(ValueError):
    """发现ID不属于当前任务，拒绝本次提交并允许有界参数纠正。"""


class ResearchProgressService:
    """复用行动日志保存候选发现、缺口及版本，无额外模型或文件写入。"""

    def __init__(self, db):
        """注入权威数据库，构造时不迁移或写入数据。"""
        self.db = db

    def snapshot(self, task_id: str) -> dict:
        """按成功行动重放最新进展，不依赖可能滞后的图快照。

        Args:
            task_id: 当前角色任务，不能通过根任务身份绕过子任务授权。

        Returns:
            已提交的进展；没有记录时返回空发现，不合成研究结论。
        """
        task = self.db.get_task(task_id) or {}
        brief = json.loads(task.get('brief_json') or '{}')
        current = {'task_id': task_id, 'goal_ids': brief.get('goal_ids', []), 'objective': task.get('prompt', ''),
                   'revision': 0, 'findings': [], 'gaps': [], 'next_question': None}
        ranges = {}
        for action in self.db.list_actions(task_id):
            if action['status'] == 'succeeded' and action['tool_name'] in {'read_paper', 'acquire_paper_pdf'}:
                data = (action.get('observation') or {}).get('data', {})
                if data.get('document_id'):
                    key = (data['document_id'], data.get('content_sha256'))
                    ranges.setdefault(key, set()).update(p.get('page_number') for p in data.get('pages', []))
            if action['status'] == 'succeeded' and action['tool_name'] == 'set_research_plan':
                plan = action['observation'].get('data', {}).get('research_plan', {})
                current['goal_ids'] = [g['goal_id'] for g in plan.get('goals', [])]
            value = (action.get('observation') or {}).get('data', {}).get('research_progress')
            if action['status'] == 'succeeded' and action['tool_name'] in {'record_research_progress', 'finish_research'} and value:
                current = deepcopy(value)
                current['action_id'] = action['action_id']
        current['read_ranges'] = [{'document_id': key[0], 'source_sha256': key[1],
                                   'page_numbers': sorted(pages, key=lambda n: -1 if n is None else n)} for key, pages in ranges.items()]
        return current

    def source(self, task_id: str, evidence_id: str, tx=None) -> dict:
        """从已授权实际页派生版本和定位，不接受模型自报 hash。"""
        from research_loop.services.papers import PaperService
        return PaperService(self.db, None, None).authorized_source(
            task_id, evidence_id, tx, with_readback=False)['reference']

    @staticmethod
    def _source_shape(value: dict) -> bool:
        """复用共同完整引用合同，不接受展示入口等额外字段。"""
        from research_loop.domain.evidence import valid_source_reference
        return valid_source_reference(value)

    def _versioned_paper_locator(self, paper_id: str, tx=None) -> tuple | None:
        """仅从已存题录确认明确同版arXiv地址，不按标题或正文hash猜别名。"""
        from research_loop.domain.evidence import explicit_versioned_paper_locator
        one = tx.one if tx else self.db.fetchone
        row = one('SELECT metadata_json FROM papers WHERE paper_id=?', (paper_id,))
        try:
            metadata = json.loads(row['metadata_json']) if row else None
        except (ValueError, TypeError):
            return None
        return explicit_versioned_paper_locator(paper_id, metadata)

    def source_version_matches(self, task_id: str, stored: dict, tx=None) -> bool:
        """比较授权原文版本，仅兼容可证明同一明确版本的论文显示别名。

        Args:
            task_id: 持有该页证据权限的当前角色，沿用source的授权JOIN。
            stored: 已提交的来源引用；不会改写旧引用或旧核验结果。
            tx: 可选当前事务，提交时题录与页版本在同一事务读回。

        Returns:
            全部不可变字段相同，且paper_id相同或双方题录明确同版时返回True。
            文档身份包含解析版本/配置；不同文档、内容、页位置或页文本不能兼容。

        Raises:
            PermissionError: 当前角色没有该evidence权限，不因别名取得新材料授权。
        """
        if not self._source_shape(stored):
            return False
        try:
            current = self.source(task_id, stored['evidence_id'], tx)
        except (ValueError, KeyError, TypeError):
            return False
        if not self._source_shape(current):
            return False
        if any(current[key] != stored[key] for key in current if key != 'paper_id'):
            return False
        if current['paper_id'] == stored['paper_id']:
            return True
        # 共享内容缓存可能由另一合法身份重读；这不是授予新paper或新页的入口。
        original = self._versioned_paper_locator(stored['paper_id'], tx)
        return original is not None and original == self._versioned_paper_locator(current['paper_id'], tx)

    def prepare(self, task_id: str, args: ResearchProgressInput) -> dict:
        """校验并合并阶段发现，仅返回候选；成功行动提交后才成为有效版本。"""
        old = self.snapshot(task_id)
        if args.base_revision is not None and args.base_revision != old['revision']:
            raise ValueError(f"progress revision stale; current={old['revision']}")
        findings = {f['finding_id']: deepcopy(f) for f in old['findings']}
        for item in args.findings:
            refs = [self.source(task_id, eid) for eid in dict.fromkeys(item.evidence_ids)]
            if item.finding_id and item.finding_id not in findings:
                raise UnknownFindingIdError('unknown finding_id; omit it for a new finding')
            key = item.finding_id or digest([task_id, ' '.join(item.text.split()).casefold()])[:24]
            if key in findings and not item.finding_id:
                refs = list({ref['evidence_id']: ref for ref in findings[key]['sources'] + refs}.values())
            previous = findings.get(key)
            # 现有发现可基于已取得原文继续分析；省略证据不能抹掉其来源身份。
            if previous and item.finding_id and not item.evidence_ids:
                refs = deepcopy(previous['sources'])
            candidate = {'finding_id': key, 'text': item.text, 'sources': refs,
                         'verification_status': 'unreviewed'}
            changed = previous is None or any(candidate[k] != previous.get(k) for k in ('text', 'sources'))
            if previous and not changed:
                findings[key] = previous
            else:
                candidate['revision'] = (previous.get('revision', 1) if previous else 0) + 1
                findings[key] = candidate
        value = {**old, 'findings': list(findings.values()),
                 'gaps': old['gaps'] if args.gaps is None else list(dict.fromkeys(args.gaps)),
                 'next_question': old.get('next_question') if args.next_question is None else args.next_question}
        value.pop('action_id', None)
        changed = any(value.get(k) != old.get(k) for k in ('findings', 'gaps', 'next_question'))
        value['revision'] = old['revision'] + int(changed)
        value['base_revision'] = old['revision']
        findings_changed = continuation_fingerprint(value) != continuation_fingerprint(old)
        plan_changed = any(value.get(k) != old.get(k) for k in ('gaps', 'next_question'))
        value['progress_change'] = {'findings_changed': findings_changed, 'plan_changed': plan_changed,
                                    'kind': 'finding_revision' if findings_changed else 'plan_only' if plan_changed else 'unchanged'}
        return value

    def commit_action(self, action: dict, owner: tuple | None) -> None:
        """同事务提交成功进展、核对前版和租约，崩溃重放不覆盖旧成功行动。"""
        task_id = action['task_id']
        progress = action['observation']['data']['research_progress']
        with MemoryStore(self.db).transaction() as tx:
            check_owner(tx, owner)
            task = tx.one('SELECT cancel_requested FROM tasks WHERE task_id=?', (task_id,))
            if not task or task['cancel_requested']:
                raise PermissionError('progress task missing or cancelled')
            prior = tx.one('SELECT * FROM actions WHERE task_id=? AND sequence_no=?', (task_id, action['sequence']))
            if not prior or prior['tool_name'] != action['tool_name'] or json.loads(prior['arguments_json']) != action['arguments']:
                raise ValueError('progress action intent mismatch')
            if prior['status'] == 'succeeded':
                if json.loads(prior['observation_json']) != action['observation']:
                    raise ValueError('successful progress action is immutable')
                return
            latest = tx.one("SELECT observation_json FROM actions WHERE task_id=? AND status='succeeded' "
                            "AND tool_name IN ('record_research_progress','finish_research') ORDER BY sequence_no DESC LIMIT 1", (task_id,))
            previous = (json.loads(latest['observation_json']).get('data', {}).get('research_progress') or {}) if latest else {}
            # 历史 finish_research 没有进展字段时，保持零版本兼容。
            if progress['base_revision'] != previous.get('revision', 0):
                raise ValueError('progress changed before commit')
            for finding in progress['findings']:
                for ref in finding['sources']:
                    if not self.source_version_matches(task_id, ref, tx):
                        raise ValueError('progress source version changed before commit')
            tx.execute('UPDATE actions SET status=?,observation_json=? WHERE task_id=? AND sequence_no=?',
                       ('succeeded', dump(action['observation']), task_id, action['sequence']))

    def cached_read(self, task_id: str, arguments: dict, attachments: dict, *, provider=None) -> dict | None:
        """同任务同参数且本地原文件和页版本未变时复用成功读取。"""
        from pathlib import Path
        if not self.db.task_has_input(task_id, arguments.get('target_ref')):
            return None
        from research_loop.services.papers import PaperService
        try:
            scope = PaperService(self.db, None, None).user_scope(task_id, target_ref=arguments.get('target_ref'))
        except (ValueError, PermissionError):
            return None
        target = attachments.get(arguments.get('target_ref'), {})
        path = Path(target['local_path']) if target.get('local_path') else None
        if not path or not path.is_file():
            return None
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for action in reversed(self.db.list_actions(task_id)):
            if action['tool_name'] != 'read_paper' or action['status'] != 'succeeded' or action['arguments'] != arguments:
                continue
            observation = deepcopy(action['observation'])
            pages = observation.get('data', {}).get('pages', [])
            if not pages or (scope and any(p.get('page_number') is None or not scope.start <= p['page_number'] <= scope.end for p in pages)):
                continue
            try:
                contract = getattr(provider, 'parse_contract_matches', None)
                if callable(contract):
                    from research_loop.domain.evidence import Document
                    document = self.db.fetchone('SELECT * FROM documents WHERE document_id=?', (pages[0]['document_id'],))
                    if not document:
                        continue
                    document = {**document, 'total_pages': document['total_pages'] or None,
                                'gaps': json.loads(document.get('gaps_json') or '[]')}
                    if not contract(Document.model_validate(document)):
                        continue
                if any(self.source(task_id, page['evidence_id'])['source_sha256'] != source_hash or
                       self.db.get_page(page['evidence_id'])['text'] != page['text'] for page in pages):
                    continue
            except (PermissionError, ValueError):
                continue
            observation['data']['reused_action_id'] = action['action_id']
            return observation
        return None

    def retained(self, task_id: str) -> dict:
        """返回保留成果和来源完整性；历史内容保留，失效来源不冒充当前可用。"""
        result = self.snapshot(task_id)
        for finding in result['findings']:
            finding.setdefault('revision', 1)
            finding['source_integrity'] = 'intact'
            for ref in finding['sources']:
                try:
                    if not self.source_version_matches(task_id, ref):
                        raise ValueError('source changed')
                except (ValueError, PermissionError):
                    finding['source_integrity'] = 'stale'
        return result


def evidence_locations(data: dict) -> set[tuple]:
    """按版本/位置/正文识别材料；查询词、chunk ID和排序不算材料变化。"""
    return {(r.get('document_id'),r.get('source_sha256') or data.get('content_sha256'),
             r.get('page_number'),r.get('char_start',0),hashlib.sha256(r.get('text','').encode()).hexdigest())
            for r in data.get('pages',[])+data.get('hits',[])}


def evidence_spans(data: dict) -> list[tuple]:
    """展开实际展示窗口，保留文档/版本/页/字符位置和原字词。

    Args:
        data: 原始观察或已压缩观察；继承公共字段并展开context_spans。

    Returns:
        五元组列表；未知版本只能精确判等，不据此合并包含区间。
    """
    rows = data.get('pages', []) + [{**data.get('hits_defaults', {}), **r} for r in data.get('hits', [])]
    output = []
    for row in rows:
        text = row.get('text')
        if not row.get('document_id') or not isinstance(text, str) or not text:
            continue
        key = (row['document_id'], row.get('source_sha256') or data.get('content_sha256'), row.get('page_number'))
        start = row.get('char_start', 0)
        for span in row.get('context_spans') or [{'start': 0, 'text': text}]:
            if isinstance(start, int) and isinstance(span.get('start', 0), int) and span.get('text'):
                output.append((*key, start + span.get('start', 0), span['text']))
    return output


def _span_covered(span: tuple, previous: list[tuple]) -> bool:
    """只以同版本同位置的相同原字词覆盖；版本缺失时仅精确判等。"""
    if span in previous:
        return True
    if span[1] is None:
        return False
    cursor, end = span[3], span[3] + len(span[4])
    for row in sorted((r for r in previous if r[:3] == span[:3]), key=lambda r: r[3]):
        left, right = max(span[3], row[3]), min(end, row[3] + len(row[4]))
        if left > cursor or right <= cursor:
            continue
        if span[4][left-span[3]:right-span[3]] == row[4][left-row[3]:right-row[3]]:
            cursor = right
        if cursor == end:
            return True
    return False


def visible_evidence(messages: list[dict]) -> list[tuple]:
    """只读取实际请求中的原文工具观察/工作集，历史ID与摘要不算可见正文。"""
    output = []
    names = {c['id']: c['function']['name'] for m in messages for c in m.get('tool_calls', [])}
    for message in messages:
        tool = message.get('role') == 'tool' and names.get(message.get('tool_call_id')) in {'read_paper', 'acquire_paper_pdf', 'retrieve_evidence'}
        workspace = message.get('role') == 'user' and str(message.get('content', '')).startswith('{"archived_tool_groups"')
        if not (tool or workspace):
            continue
        try:
            value = json.loads(message['content'])
        except (ValueError, TypeError):
            continue
        if tool and value.get('ok'):
            output.extend(evidence_spans(value.get('data') or {}))
        elif workspace:
            output.extend(evidence_spans({'pages': value.get('sources', [])}))
    return output


def evidence_progress(previous: list[dict], name: str, arguments: dict, observation: dict,
                      visible_messages: list[dict] | None = None) -> dict:
    """分别报告取得材料增量和本次可见性，不用查询/切片变化冒充研究结论。

    Args:
        previous: 已持久化行动，不回写旧标签。
        name: 本次只读工具名。
        arguments: 实际工具参数。
        observation: 本次原始观察。
        visible_messages: 产生本批工具调用的实际请求；未知时不推断上下文读回。

    Returns:
        材料覆盖、可见性和参数身份；semantic_verdict始终为False。
    """
    if not observation.get('ok'):
        return {'kind': 'action_failed', 'semantic_verdict': False}
    rows = evidence_spans(observation.get('data') or {})
    if not rows:
        return {'kind': 'no_evidence_returned', 'semantic_verdict': False}
    old, same_request = [], False
    for action in previous:
        if action.get('status') == 'succeeded' and action['tool_name'] in {'read_paper', 'acquire_paper_pdf', 'retrieve_evidence'}:
            old.extend(evidence_spans((action.get('observation') or {}).get('data', {})))
            same_request |= action['tool_name'] == name and action['arguments'] == arguments
    new = sum(not _span_covered(row, old) for row in rows)
    visible = visible_evidence(visible_messages) if visible_messages is not None else None
    shown = sum(_span_covered(row, visible) for row in rows) if visible is not None else None
    kind = ('new_evidence' if new else 'existing_evidence' if shown is None else
            'same_evidence' if shown == len(rows) else 'context_readback')
    return {'kind': kind, 'new_source_count': new, 'same_request': same_request,
            'already_visible_count': shown, 'returned_source_count': len(rows),
            'visibility_known': visible is not None, 'semantic_verdict': False,
            'notice': '取得材料、当前可见和形成发现分别判断；缺失窗口可以定向读回，已可见材料应交接或明确新缺口。'}


def progress_fingerprint(value: dict) -> str:
    """对实际候选发现/缺口/下一问题取指纹，不把修订号变化算作新进展。"""
    return digest({k:value.get(k) for k in ('findings','gaps','next_question')})


def continuation_fingerprint(value: dict) -> str:
    """仅对实际发现内容与来源取指纹；计划文字和自增版本不构成研究增量。

    Args:
        value: 当前或历史进展，旧行动无需回写新标签。

    Returns:
        稳定内容指纹；既有证据上的有效分析修订也会改变指纹。
    """
    return digest([{'finding_id': finding.get('finding_id'), 'text': finding.get('text'),
                    'sources': [{key: val for key, val in source.items() if key != 'paper_id'}
                                for source in finding.get('sources', [])]} for finding in value.get('findings', [])])


def readback_handoff(actions: list[dict]) -> dict | None:
    """重复读回既有材料且没有新阶段成果时，请求一次现有交接步骤。

    Args:
        actions: 本角色有序持久行动，不依赖Checkpoint中的临时计数。

    Returns:
        重复动作和阶段身份，或None。首次压缩补读仍可执行；不判断科学结论真伪，
        不新增请求数/Token/页数额度。新材料或真实阶段变化会重新开放研究。
    """
    stage = None
    seen_stages = {continuation_fingerprint({})}
    since_stage = []
    for action in actions:
        if action.get('status') != 'succeeded':
            if action['tool_name'] in {'read_paper','acquire_paper_pdf','retrieve_evidence'}:
                since_stage.append(action)
            continue
        data = (action.get('observation') or {}).get('data', {})
        value = data.get('research_progress')
        if action['tool_name'] == 'record_research_progress' and value:
            signature = continuation_fingerprint(value)
            if signature not in seen_stages and value.get('revision',0) > value.get('base_revision',0):
                since_stage = []
                stage = action.get('action_id')
            seen_stages.add(signature)
        elif action['tool_name'] in {'read_paper','acquire_paper_pdf','retrieve_evidence'}:
            since_stage.append(action)
    if not since_stage:
        return None
    latest = since_stage[-1]
    batch_id = (latest.get('observation') or {}).get('data', {}).get('research_increment', {}).get('request_sequence')
    batch_start = len(since_stage) - 1
    if batch_id is not None:
        while batch_start and (since_stage[batch_start-1].get('observation') or {}).get('data', {}).get('research_increment', {}).get('request_sequence') == batch_id:
            batch_start -= 1
    batch, prior = since_stage[batch_start:], since_stage[:batch_start]
    # 同一响应有真实新增材料或可恢复取证错误时，先让模型看到整批结果。
    if any(a.get('status') != 'succeeded' or (a.get('observation') or {}).get('data', {}).get('research_increment', {}).get('kind') in {'new_evidence', 'action_failed'} for a in batch):
        return None
    # 新行动已按实际冻结请求记录缺失窗口，不能再用历史原始整页推翻它。
    # 先让恢复的互补窗口进入下一次决策；只有已可见的重复才请求交接。
    # 旧记录没有visibility_known时仍按原合同恢复，不回写历史标签。
    if any((a.get('observation') or {}).get('data', {}).get('research_increment', {}).get('visibility_known')
           and (a.get('observation') or {}).get('data', {}).get('research_increment', {}).get('kind') == 'context_readback'
           for a in batch):
        return None
    for current in reversed(batch):
        data = (current.get('observation') or {}).get('data', {})
        increment = data.get('research_increment', {})
        if increment.get('kind') not in {'context_readback', 'same_evidence', 'existing_evidence'}:
            continue
        spans = evidence_spans(data)
        eligible = []
        for action in reversed(prior):
            if action.get('status') != 'succeeded':
                continue
            source = (action.get('observation') or {}).get('data', {})
            exact = action['tool_name'] == current['tool_name'] and action['arguments'] == current['arguments']
            # 首次恢复缺失窗口可以继续；窗口已可见或曾实际恢复，才转交已有成果。
            recovered = source.get('research_increment', {}).get('kind') in {'context_readback', 'same_evidence', 'existing_evidence'}
            covered = bool(spans) and all(_span_covered(row, evidence_spans(source)) for row in spans)
            if exact and (not spans or covered):
                eligible = [action]
                break
            if spans and (recovered or increment['kind'] == 'same_evidence'):
                eligible.append(action)
        # 旧查询分别返回A/B，新查询同时返回A+B，也只能算既有材料。
        returned = [s for a in eligible for s in evidence_spans((a.get('observation') or {}).get('data', {}))]
        covered = bool(spans) and all(_span_covered(row, returned) for row in spans)
        if eligible and (covered or not spans):
            return {'trigger_action_id': current.get('action_id'), 'previous_action_id': eligible[0].get('action_id'),
                    'previous_action_ids': [a.get('action_id') for a in eligible],
                    'stage_action_id': stage, 'tool': current['tool_name'],
                    'reason': 'same_material_without_research_progress'}
    return None


def reviewable_findings(db, root_task_id: str, source_task_id: str) -> dict:
    """读回同根成功进展供核验，保持原始子任务结果与核验状态不变。

    Args:
        db: 当前业务数据库。
        root_task_id: 发起Reflection的根任务。
        source_task_id: 根任务或其Researcher子任务。

    Returns:
        带原版本、来源和稳定finding_id的只读副本。

    Raises:
        PermissionError: 试图读取外部任务或把Reviewer结果当研究发现。
    """
    task = db.get_task(source_task_id)
    if not task or (source_task_id != root_task_id and task.get('root_task_id') != root_task_id):
        raise PermissionError('review source_task_id outside current root')
    if source_task_id != root_task_id and task.get('role') != 'researcher':
        raise PermissionError('review source must be the root or a researcher')
    return ResearchProgressService(db).retained(source_task_id)
