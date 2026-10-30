"""配对终稿的外部语义门；子任务只能写runtime，不能批准本控制目录。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import json
import math
import os
from datetime import datetime
from pathlib import Path
import re
import secrets
import time

from harness_ab_experiment import canonical, paired_source_contract, sha256


FINAL_SUITE_POLICY = {'version': 'agent_final_v1', 'semantic_failures': 'record_and_continue'}


_ACTIVE_GATE_CONTEXTS = ContextVar('active_pair_gate_contexts', default=None)
_GATE_CAPSULE_AUTH = object()


class _GateContexts:
    """单次准入内按manifest身份复用验真结果；不能用当前owner解释旧评分。"""

    def __init__(self, manifest, continuation, *, _proof=None):
        if _proof is not _GATE_CAPSULE_AUTH:
            raise PermissionError('PAIR_PRIVATE_CONTEXT_CAPSULE_REQUIRED')
        self._values = {manifest_identity(manifest): continuation}

    def context(self, manifest):
        identity = manifest_identity(manifest)
        if identity not in self._values:
            from agent_final_continuation import validate_continuation
            # 历史owner必须独立验真；其递归读取不能借外层owner的capsule。
            token = _ACTIVE_GATE_CONTEXTS.set(None)
            try:
                verified = validate_continuation(manifest)
            finally:
                _ACTIVE_GATE_CONTEXTS.reset(token)
            self._values[identity] = verified
        return self._values[identity]


def _gate_contexts():
    value = _ACTIVE_GATE_CONTEXTS.get()
    if value is not None and type(value) is not _GateContexts:
        raise PermissionError('PAIR_PRIVATE_CONTEXT_CAPSULE_REQUIRED')
    return value


@contextmanager
def _gate_context_scope(manifest, continuation):
    contexts = _GateContexts(manifest, continuation, _proof=_GATE_CAPSULE_AUTH)
    token = _ACTIVE_GATE_CONTEXTS.set(contexts)
    try:
        yield contexts
    finally:
        _ACTIVE_GATE_CONTEXTS.reset(token)


def validate_final_suite_policy(manifest):
    """仅为新冻结固定集允许语义记分后继续；旧批次保持原首错规则。"""
    if 'final_suite_policy' not in manifest:
        return False
    if (manifest.get('experiment_kind') != 'prompt_bundle_v1'
            or manifest['final_suite_policy'] != FINAL_SUITE_POLICY):
        raise PermissionError('PAIR_FINAL_SUITE_POLICY_INVALID')
    return True


def manifest_identity(manifest):
    """格截止是父进程临时参数，其余整个冻结配置共同绑定审核身份。"""
    return sha256(canonical({k: v for k, v in manifest.items() if k != 'cell_absolute_deadline'}))


def read_json(path):
    """控制文件不接受符号链接替换。"""
    path = Path(path)
    if path.is_symlink():
        raise PermissionError('PAIR_REVIEW_SYMLINK_REJECTED')
    return json.loads(path.read_text())


def write_new(path, value):
    """控制记录仅创建一次，不覆盖旧审核或挑战。"""
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write('\n')


def assert_fresh_gate_root(manifest):
    """在账本和任务创建前拒绝预先放置的控制文件。"""
    validate_final_suite_policy(manifest)
    root = Path(manifest['semantic_gate_root'])
    if root.exists() or root.is_symlink():
        raise PermissionError('PAIR_REVIEW_ROOT_MUST_BE_NEW')


def initialize_scope_session(manifest, started_at, *, clock=time.time):
    """仅父批次在控制目录创建一次会话，格进程没有该目录写权限。"""
    validate_final_suite_policy(manifest)
    root = Path(manifest['semantic_gate_root'])
    root.mkdir(exist_ok=False)
    session = {'version': 1, 'nonce': secrets.token_hex(32), 'manifest_sha256': manifest_identity(manifest),
        'started_at': started_at, 'created_at': clock(), 'controller_pid': os.getpid()}
    write_new(root / 'session.json', session)
    return session


def _active_contract(manifest):
    """保持旧grant字段格式，但绑定当前唯一的续跑合同。"""
    if 'completion_contract' in manifest:
        if 'continuation_contract' in manifest:
            raise PermissionError('PAIR_TWO_CONTINUATION_CONTRACTS_FORBIDDEN')
        return manifest['completion_contract']
    return manifest['continuation_contract']


def _financial_only(continuation):
    return bool(continuation and continuation['clock'].get('policy') == 'financial_only')


def _observation_policy(manifest):
    contract = manifest.get('completion_contract')
    contract = contract if isinstance(contract, dict) else {}
    if ('observation_policy' not in manifest
            and contract.get('version') not in {'agent_final_completion_v4', 'agent_final_completion_v5', 'agent_final_completion_v6'}
            and 'observation_recovery_lineage' not in contract):
        return False
    from agent_final_observation_recovery import validate_observation_policy
    enabled = validate_observation_policy(manifest)
    if enabled and not validate_final_suite_policy(manifest):
        raise PermissionError('PAIR_OBSERVATION_FIXED_SUITE_REQUIRED')
    return enabled


def _retained_budget_stop_receipts(manifest, cell, result, view, retained):
    """只有权威封存合同可解释旧预算或已发送未知收据；当前格无此例外。"""
    contract = manifest.get('completion_contract') or {}; version = contract.get('version')
    if retained is None: return None
    contexts = _gate_contexts(); context = contexts.context(manifest) if contexts is not None else None
    if version == 'agent_final_completion_v6':
        if not (retained.get('provider_unknown_stop_verified') is True
                or retained.get('root_budget_stop_verified') is True or result.get('ledger_halt') is not None):
            return None
        from agent_final_provider_unknown_recovery import retained_provider_unknown_receipts
        return retained_provider_unknown_receipts(manifest, cell, result, view, retained, context=context)
    if (version != 'agent_final_completion_v5'
            or (retained.get('root_budget_stop_verified') is not True and result.get('ledger_halt') is None)):
        return None
    from agent_final_root_budget_recovery import retained_budget_stop_receipts
    return retained_budget_stop_receipts(manifest, cell, result, view, retained, context=context)


def observation_stop(manifest, cell, result, view, retained=None):
    """绑定一次实际停止的事件和收据；等待输入不是成功交付或后台终态。"""
    if not _observation_policy(manifest):
        raise PermissionError('PAIR_OBSERVATION_POLICY_REQUIRED')
    cell = Path(cell)
    policy = manifest['observation_policy']
    root, status = result.get('root_id'), view.get('status')
    budget_proofs = _retained_budget_stop_receipts(manifest, cell, result, view, retained)
    budget_stop = bool(budget_proofs and budget_proofs.get('root_budget_stop_verified') is True)
    provider_stop = bool(budget_proofs and budget_proofs.get('provider_unknown_stop_verified') is True)
    retained_stop = budget_stop or provider_stop
    if (not root or result.get('status') != 'executed' or view.get('task_id') != root
            or result.get('task_status') != status or status not in policy['statuses']
            or (result.get('ledger_halt') is not None and not retained_stop)):
        raise PermissionError('PAIR_OBSERVATION_STOP_IDENTITY_INVALID')
    isolation, live = result.get('isolation'), result.get('live_isolation_guard')
    if (not isinstance(isolation, dict) or not isinstance(isolation.get('denials'), list)
            or any(x.get('phase') != 'guard_selfcheck' for x in isolation['denials'])
            or isolation.get('denied_connections', [])
            or not isinstance(live, dict) or live.get('first_error') is not None
            or live.get('closed') is not False or live.get('ledger_bound') is not True
            or live.get('persistence_errors', [])):
        raise PermissionError('PAIR_OBSERVATION_RUNTIME_DENIAL')
    events = read_json(cell / 'events.json')
    if isinstance(events, dict) and events.get('task_id') != root:
        raise PermissionError('PAIR_OBSERVATION_EVENT_ROOT_INVALID')
    events = events.get('events') if isinstance(events, dict) else events
    if not isinstance(events, list):
        raise PermissionError('PAIR_OBSERVATION_EVENTS_REQUIRED')
    own = [x for x in events if x.get('task_id') == root]
    starts = [x for x in own if x.get('event_type') == 'task_created']
    ends = [x for x in own if x.get('event_type') == 'task_finished']
    if (len(starts) != 1 or len(ends) != 1
            or ends[0].get('payload', {}).get('status') != status
            or ends[0].get('payload', {}).get('result') != view.get('result')):
        raise PermissionError('PAIR_OBSERVATION_UNIQUE_STOP_EVENT_REQUIRED')
    try:
        a, b = [datetime.fromisoformat(x['created_at'].replace('Z', '+00:00')).timestamp()
                for x in (starts[0], ends[0])]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PermissionError('PAIR_OBSERVATION_EVENT_TIME_INVALID') from exc
    if not math.isfinite(a) or not math.isfinite(b) or b < a:
        raise PermissionError('PAIR_OBSERVATION_EVENT_TIME_INVALID')
    native = read_json(cell / 'model-calls.json')
    snapshot = read_json(cell / 'ledger-snapshot.json')
    if (not isinstance(native, list) or not any(x.get('root_id') == root
            and x.get('case_id') == result.get('case_id') and x.get('variant') == result.get('variant')
            for x in snapshot['roots'])):
        raise PermissionError('PAIR_OBSERVATION_LEDGER_ROOT_REQUIRED')
    physical = [x for x in snapshot['reservations'] if x.get('root_id') == root]
    from agent_final_completion import _receipts
    if not native and not physical and status not in policy['non_delivery_statuses']:
        raise PermissionError('PAIR_DELIVERY_OBSERVATION_RECEIPTS_REQUIRED')
    proofs = (budget_proofs if retained_stop else _receipts(root, native, physical) if native or physical else
              {'physical_receipts_verified': True, 'native_status_counts': {}, 'physical_status_counts': {}})
    if retained and any(retained.get(k) != proofs[k] for k in
                        ('physical_receipts_verified', 'native_status_counts', 'physical_status_counts')):
        raise PermissionError('PAIR_OBSERVATION_RETAINED_RECEIPTS_CHANGED')
    body = view.get('result') or {}
    answer = body.get('answer_markdown')
    if answer is None:
        answer = ''
    if (not isinstance(answer, str) or (not answer.strip()
            and status not in policy['non_delivery_statuses'])):
        raise PermissionError('PAIR_FINAL_ANSWER_REQUIRED')
    metadata = []
    for event in own:
        if event.get('event_type') in {'clarification_requested', 'worker_failed', 'cancel_requested'}:
            payload = event.get('payload') or {}
            metadata.append({'event_type': event['event_type'], 'created_at': event.get('created_at'),
                'payload': {k: payload[k] for k in ('question', 'fields', 'suggested_values', 'status',
                                                  'answer', 'error', 'reason') if k in payload}})
    unknown_state = ({k: proofs[k] for k in ('provider_unknown_stop_verified','receipt_state_verified',
        'actual_usage_known','unconfirmed_request_count')} if provider_stop else {})
    return {**unknown_state, 'observed_status': status, 'answer_absent': not bool(answer.strip()),
        'answer_kind': body.get('answer_kind'), 'observation_stop': {
            'task_created_at': starts[0]['created_at'], 'task_finished_at': ends[0]['created_at'],
            'measurement_kind': 'delivery_candidate' if status == 'completed' else 'non_complete_observation',
            'backend_terminal_claimed': status != 'needs_input',
            'reason': body.get('reason'), 'actual_stop_metadata': metadata}}


def review_observation(answer):
    """只投影停止语义；原时间、标识及完整事件仅由私有身份绑定。"""
    def semantic(value):
        if not isinstance(value, str):
            return None
        if (answer['root_id'] in value or re.search(
                r'(?i)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
                r'|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}|/Users/|/private/'
                r'|tokens?|\b(?:latency|duration|nano_yuan|variant|root_id)\b'
                r'|耗时|时延|响应时间|费用|[¥￥]|\d+(?:\.\d+)?\s*(?:秒|毫秒|元)', value)):
            return None
        return value
    original = answer['observation_stop']
    stop = {k: original[k] for k in ('measurement_kind', 'backend_terminal_claimed')}
    stop['reason'] = semantic(original.get('reason'))
    stop['actual_stop_metadata'] = []
    for event in original['actual_stop_metadata']:
        payload = event.get('payload') or {}
        safe = {}
        for key in ('status', 'reason', 'question'):
            value = semantic(payload.get(key))
            if value is not None:
                safe[key] = value
        for key in ('fields', 'suggested_values'):
            values = payload.get(key)
            if isinstance(values, list) and all(semantic(v) is not None for v in values):
                safe[key] = values
        stop['actual_stop_metadata'].append({'event_type': event['event_type'], 'payload': safe})
    unknown_state = {}
    if answer.get('provider_unknown_stop_verified') is True:
        if (answer.get('observed_status') != 'failed' or answer.get('receipt_state_verified') is not True
                or answer.get('actual_usage_known') is not False or answer.get('unconfirmed_request_count') != 1):
            raise PermissionError('PAIR_UNKNOWN_OBSERVATION_PROJECTION_INVALID')
        unknown_state = {'provider_receipt_observation': {'verified_provider_unknown_stop': True,
            'response_usage_unconfirmed': True, 'unconfirmed_request_count': 1,
            'no_resend_or_release': True, 'original_root_status': 'failed',
            'execution_complete': False, 'actual_usage_known': False}}
    return {**unknown_state, **{k: answer[k] for k in ('observed_status', 'answer_absent', 'answer_kind')},
            'observation_stop': stop}


def _attempt_kind(manifest, continuation, position):
    if _observation_policy(manifest):
        positions = continuation.get('retry_positions')
        if (not isinstance(positions, list) or any(type(x) is not int for x in positions)
                or positions != sorted(set(positions))):
            raise PermissionError('PAIR_RETRY_POSITIONS_REQUIRED')
        return 'technical_retry' if position in positions else 'first_attempt'
    return 'technical_retry' if position == continuation['retained_count'] else 'first_attempt'


def _retained_cell(manifest, case_id, variant):
    """旧格只读原目录；不得复制到本批冒充新执行。"""
    if 'completion_contract' not in manifest:
        return None
    contexts = _gate_contexts()
    if contexts is not None:
        from agent_final_completion import _retained_cell_from_context
        return _retained_cell_from_context(manifest, contexts.context(manifest), case_id, variant)
    from agent_final_completion import retained_cell
    return retained_cell(manifest, case_id, variant)


def _review_identity_rows(manifest, case_id, answers):
    """旧挑战不携带issues；仅从V5权威保留记录重建预算停点原问题。"""
    contract = manifest.get('completion_contract') or {}
    context = None
    if contract.get('version') in {'agent_final_completion_v5', 'agent_final_completion_v6'}:
        contexts = _gate_contexts()
        if contexts is not None:
            context = contexts.context(manifest)
        else:
            from agent_final_completion import validate_completion
            context = validate_completion(manifest)
    rows = []
    for variant, answer in answers.items():
        issues = []
        if context is not None:
            matches = [r for r in context['retained_rows'] if (r['case_id'], r['variant'], r['root_id'])
                       == (case_id, variant, answer['root_id'])]
            if len(matches) > 1:
                raise PermissionError('PAIR_REVIEW_RETAINED_ROW_AMBIGUOUS')
            if matches and (matches[0].get('root_budget_stop_verified') is True
                    or matches[0].get('provider_unknown_stop_verified') is True):
                retained = matches[0]; cell = Path(retained['provenance']['cell_root'])
                proofs = _retained_budget_stop_receipts(manifest, cell, read_json(cell / 'result.json'),
                    read_json(cell / 'task-view.json'), retained)
                if not proofs or not (proofs.get('root_budget_stop_verified') is True
                        or proofs.get('provider_unknown_stop_verified') is True):
                    raise PermissionError('PAIR_REVIEW_RETAINED_BUDGET_PROOF_REQUIRED')
                issues = list(retained['issues'])
        rows.append({'case_id': case_id, 'variant': variant, 'root_id': answer['root_id'], 'issues': issues})
    return rows


def _prior_pair_manifest(manifest, offset, continuation):
    if 'completion_contract' not in manifest:
        return continuation['anchor_manifest']
    contexts = _gate_contexts()
    if contexts is not None:
        from agent_final_completion import _prior_pair_manifest_from_context
        original = _prior_pair_manifest_from_context(manifest, contexts.context(manifest), offset)
    else:
        from agent_final_completion import prior_pair_manifest
        original = prior_pair_manifest(manifest, offset)
    if original is None:
        raise PermissionError('PAIR_RETAINED_REVIEW_OWNER_REQUIRED')
    return original


def _cell_origin(manifest, case_id, variant):
    retained = _retained_cell(manifest, case_id, variant)
    if retained:
        return Path(retained['cell_root']), retained['source_manifest'], retained
    return Path(manifest['runtime_root']) / 'cells' / (case_id + '-' + variant), manifest, None


def _prior_reviews(manifest, position, continuation, clock):
    """保留配对按各自原manifest核验，混合与新配对使用当前真实审阅。"""
    order = manifest['execution_order']
    retained = continuation['retained_count'] if continuation else 0
    for offset in range(0, position // 2 * 2, 2):
        prior = order[offset][0]
        deferred = continuation.get('deferred_review_cases', []) if continuation else []
        active = (manifest if _observation_policy(manifest) and prior in deferred else
                  _prior_pair_manifest(manifest, offset, continuation) if offset + 1 < retained else manifest)
        contexts = _gate_contexts()
        if contexts is not None:
            contexts.context(active)
        directory = Path(active['semantic_gate_root']) / prior
        challenge = read_json(directory / 'challenge-private.json')
        review = read_json(directory / 'review.json')
        decision = read_json(directory / 'decision.json')
        rows = _review_identity_rows(active, prior, challenge['identity']['answers'])
        expected = evaluate_pair_review(active, prior, rows, manifest_identity(active),
            challenge, review, clock=clock)
        if decision != expected or decision.get('status') != 'passed':
            raise PermissionError('prior semantic review did not pass')


def _preceding_cell(manifest, position, continuation):
    """新根必须顺序结束；不将旧失败根当作本轮精读A的前置成功。"""
    first_new = continuation['retained_count'] if continuation else 0
    if not (position % 2 or (continuation and position > first_new)):
        return
    case, variant = manifest['execution_order'][position - 1]
    cell, source, retained = _cell_origin(manifest, case, variant)
    result, view = read_json(cell / 'result.json'), read_json(cell / 'task-view.json')
    observed = _observation_policy(manifest)
    statuses = manifest['observation_policy']['statuses'] if observed else {'completed', 'partial'}
    if (result.get('status') != 'executed' or not result.get('root_id')
            or view.get('task_id') != result['root_id'] or view.get('status') not in statuses):
        raise PermissionError('preceding new cell has not ended')
    if observed:
        expected = next(c for c in read_json(manifest['cases_path'])['cases'] if c['case_id'] == case)
        if (result.get('case_id') != case or result.get('variant') != variant
                or view.get('prompt') != expected['prompt']
                or result.get('loaded_supervisor') != source['loaded_prompt_contract'][variant]):
            raise PermissionError('PAIR_PRECEDING_OBSERVATION_SOURCE_IDENTITY_INVALID')
        observation_stop(manifest, cell, result, view, retained)
    if continuation:
        native = read_json(cell / 'model-calls.json')
        if any(row.get('status') != 'succeeded' for row in native):
            raise PermissionError('preceding new cell contains a technical model failure')
        if retained and (retained.get('physical_receipts_verified') is not True
                or retained.get('native_status_counts') != {'succeeded': len(native)}
                or retained.get('physical_status_counts') != {'settled': len(native)}):
            raise PermissionError('preceding retained cell receipts are not reconciled')


def issue_cell_grant(manifest, case_id, variant, deadline):
    """父调度器只在实际推进到本格时发放一次授权，不预先批准后续格。"""
    validate_final_suite_policy(manifest)
    control = Path(manifest['semantic_gate_root'])
    session = read_json(control / 'session.json')
    position = manifest['execution_order'].index([case_id, variant])
    from agent_final_continuation import validate_continuation
    continuation = validate_continuation(manifest)
    grants = control / 'grants'
    if continuation:
        first = continuation['retained_count']
        existing = set()
        if grants.exists():
            for path in grants.iterdir():
                if (path.is_symlink() or not path.is_file() or not path.stem.isdigit()
                        or path.suffix != '.json' or path.name != str(int(path.stem)) + '.json'):
                    raise PermissionError('continuation grant directory contains unrecognized entries')
                existing.add(int(path.stem))
        if position < first or existing != set(range(first, position)):
            raise PermissionError('continuation scheduler cannot skip or replay an ordinal')
        with _gate_context_scope(manifest, continuation):
            _prior_reviews(manifest, position, continuation, time.time)
            _preceding_cell(manifest, position, continuation)
    grants.mkdir(exist_ok=True)
    value = {'version': 1, 'case_id': case_id, 'variant': variant, 'position': position,
        'manifest_sha256': manifest_identity(manifest), 'session_nonce': session['nonce'],
        'source_contract_sha256': sha256(canonical(paired_source_contract(manifest))),
        'cell_absolute_deadline': deadline, 'grant_nonce': secrets.token_hex(32)}
    if continuation:
        now = time.time()
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                or not now < deadline
                or (_financial_only(continuation)
                    and deadline > now + manifest['ledger_limits']['root_seconds'])
                or (not _financial_only(continuation)
                    and deadline > continuation['clock']['model_absolute_deadline'])):
            raise PermissionError('continuation grant outside authorized finite root deadline')
        value.update(attempt_id=secrets.token_hex(32),
            attempt_kind=_attempt_kind(manifest, continuation, position),
            continuation_sha256=sha256(canonical(_active_contract(manifest))))
    write_new(grants / (str(position) + '.json'), value)
    return value


def validate_prior_pair_gates(manifest, case_id, variant, *, clock=time.time):
    """独立格入口同样不能跳过此前配对的审核，也不能脱离父批次运行。"""
    validate_final_suite_policy(manifest)
    control = Path(manifest['semantic_gate_root'])
    try:
        runtime = Path(manifest['runtime_root'])
        if any((runtime / name).exists() for name in ('batch-results.json', 'ledger-final.json')):
            raise PermissionError('finalized batch cannot create or resume a cell')
        session = read_json(control / 'session.json')
        if session.get('manifest_sha256') != manifest_identity(manifest):
            raise PermissionError('scope session manifest differs')
        order = manifest['execution_order']
        position = order.index([case_id, variant])
        from agent_final_continuation import validate_continuation
        continuation = validate_continuation(manifest)
        if continuation and position < continuation['retained_count']:
            raise PermissionError('retained observations cannot receive new execution grants')
        grant = read_json(control / 'grants' / (str(position) + '.json'))
        expected = {'version': 1, 'case_id': case_id, 'variant': variant, 'position': position,
            'manifest_sha256': manifest_identity(manifest), 'session_nonce': session['nonce'],
            'source_contract_sha256': sha256(canonical(paired_source_contract(manifest))),
            'cell_absolute_deadline': manifest['cell_absolute_deadline']}
        extras = {'grant_nonce'}
        if continuation:
            extras |= {'attempt_id'}
            expected.update(attempt_kind=_attempt_kind(manifest, continuation, position),
                continuation_sha256=sha256(canonical(_active_contract(manifest))))
            if (not isinstance(grant.get('attempt_id'), str) or len(grant['attempt_id']) != 64
                    or any(c not in '0123456789abcdef' for c in grant['attempt_id'])
                    or type(grant.get('cell_absolute_deadline')) not in (int, float)
                    or not math.isfinite(grant['cell_absolute_deadline'])
                    or (_financial_only(continuation)
                        and grant['cell_absolute_deadline'] > clock() + manifest['ledger_limits']['root_seconds'])
                    or (not _financial_only(continuation)
                        and grant['cell_absolute_deadline'] > continuation['clock']['model_absolute_deadline'])):
                raise PermissionError('continuation attempt identity or deadline invalid')
        if (set(grant) != set(expected) | extras
                or any(grant.get(k) != v for k, v in expected.items())
                or not isinstance(grant['grant_nonce'], str) or len(grant['grant_nonce']) != 64
                or not clock() < grant['cell_absolute_deadline']):
            raise PermissionError('scheduler grant does not match active cell or deadline')
        with _gate_context_scope(manifest, continuation):
            _prior_reviews(manifest, position, continuation, clock)
            _preceding_cell(manifest, position, continuation)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise PermissionError('PAIR_PRIOR_REVIEW_REQUIRED') from exc
    return {**session, 'active_grant': grant} if continuation else session


def evidence_locators(actions, source_sha256):
    """保留实际证据定位和读取范围，不复制任务、版本、费用或隐藏推理。"""
    found = {}
    def walk(value):
        if isinstance(value, dict):
            evidence_id = value.get('evidence_id')
            text = value.get('text', value.get('parsed_markdown'))
            if isinstance(evidence_id, str) and isinstance(text, str):
                page = value.get('physical_page', value.get('page_number'))
                declared = value.get('source_sha256', value.get('pdf_sha256'))
                row = {'evidence_id': evidence_id,
                    'physical_page': page if type(page) is int and page > 0 else None,
                    'text_sha256': sha256(text), 'source_sha256': declared or source_sha256,
                    'source_binding': ('observation' if declared else
                        'single_authorized_pdf_case' if source_sha256 else 'unresolved_no_source_guess'),
                    'text': text}
                found[canonical({k: v for k, v in row.items() if k != 'text'})] = row
            for key, item in value.items():
                if key not in {'reasoning_content', 'request_json', 'response_json'}:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(actions)
    rows = [found[k] for k in sorted(found)]
    return {'evidence': rows, 'observed_physical_pages': sorted({r['physical_page'] for r in rows if r['physical_page'] is not None}),
        'scope_note': '仅动作观察中实际返回的定位，不能据此宣称已理解全页图形或穷尽全文。'}


def pair_identity(manifest, case_id, rows):
    """终稿、任务身份、实际材料与收据必须共同绑定同一对。"""
    observed = _observation_policy(manifest)
    statuses = manifest['observation_policy']['statuses'] if observed else {'completed', 'partial'}
    case = next(c for c in read_json(manifest['cases_path'])['cases'] if c['case_id'] == case_id)
    joint = manifest.get('experiment_kind') == 'prompt_bundle_v1'
    if joint:
        from prompt_bundle_cases import case_materials, validate_preconditions
        materials = case_materials(case)
        preconditions = validate_preconditions(case)
    else:
        materials = [case['material']]
    for material in materials:
        if sha256(Path(material['pdf_path']).read_bytes()) != material['pdf_sha256']:
            raise PermissionError('PAIR_MATERIAL_CHANGED')
    matching = [r for r in rows if r['case_id'] == case_id]
    if len(matching) != 2 or {r['variant'] for r in matching} != {'A', 'B'}:
        raise PermissionError('PAIR_BOTH_ENDED_CELLS_REQUIRED')
    identity = {'case_id': case_id, 'question_sha256': sha256(case['prompt']),
        'cases_sha256': sha256(Path(manifest['cases_path']).read_bytes()), 'answers': {}}
    if joint:
        identity.update(materials=materials, preconditions=preconditions,
                        preconditions_sha256=sha256(canonical(preconditions)))
    else:
        identity['material'] = materials[0]
    answers = {}
    for row in matching:
        variant = row['variant']
        cell, source_manifest, retained = _cell_origin(manifest, case_id, variant)
        result = read_json(cell / 'result.json')
        view = read_json(cell / 'task-view.json')
        budget_proofs = _retained_budget_stop_receipts(manifest, cell, result, view, retained)
        budget_stop = bool(budget_proofs and budget_proofs.get('root_budget_stop_verified') is True)
        provider_stop = bool(budget_proofs and budget_proofs.get('provider_unknown_stop_verified') is True)
        retained_stop = budget_stop or provider_stop
        preserved_issues = bool(retained_stop and row.get('issues') == retained.get('issues'))
        if (not row.get('root_id') or result.get('root_id') != row['root_id']
                or view.get('task_id') != row['root_id'] or view.get('prompt') != case['prompt']
                or result.get('status') != 'executed' or (retained_stop and not preserved_issues)
                or (row.get('issues') and not preserved_issues)
                or view.get('status') not in statuses):
            raise PermissionError('PAIR_ENDED_CELL_IDENTITY_INVALID')
        if observed and row.get('issues') != [] and not preserved_issues:
            raise PermissionError('PAIR_OBSERVATION_RECONCILIATION_REQUIRED')
        answer = (view.get('result') or {}).get('answer_markdown')
        stop = observation_stop(manifest, cell, result, view, retained) if observed else None
        if observed and answer is None:
            answer = ''
        if not isinstance(answer, str) or (not answer.strip() and not (observed
                and view['status'] in manifest['observation_policy']['non_delivery_statuses'])):
            raise PermissionError('PAIR_FINAL_ANSWER_REQUIRED')
        filenames = ('result.json', 'task-view.json', 'model-calls.json', 'actions.json')
        if observed:
            filenames += ('events.json', 'ledger-snapshot.json')
        hashes = {name: sha256((cell / name).read_bytes()) for name in filenames}
        identity['answers'][variant] = {'root_id': row['root_id'], 'answer_sha256': sha256(answer),
            'backend_status': view['status'], 'files': hashes}
        if observed:
            identity['answers'][variant].update(stop)
        if 'completion_contract' in manifest:
            expected_loaded = source_manifest['loaded_prompt_contract'][variant]
            if result.get('loaded_supervisor') != expected_loaded:
                raise PermissionError('PAIR_CELL_LOADED_SOURCE_IDENTITY_MISMATCH')
            if retained and retained.get('physical_receipts_verified') is not True and not provider_stop:
                raise PermissionError('PAIR_RETAINED_PHYSICAL_RECEIPTS_REQUIRED')
            identity['answers'][variant]['source_provenance'] = {
                'cell_root': str(cell), 'manifest_sha256': manifest_identity(source_manifest),
                'source_contract_sha256': sha256(canonical(paired_source_contract(source_manifest))),
                'loaded_prompt_contract_sha256': sha256(canonical(expected_loaded)),
                'retained': retained is not None,
            }
        answers[variant] = answer
    return identity, answers, case


def prepare_pair_gate(manifest, case_id, rows, manifest_sha256, *, clock=time.time):
    """两格结束后才生成随机挑战和匿名包；本函数不作语义判断。"""
    fixed_suite = validate_final_suite_policy(manifest)
    identity, answers, case = pair_identity(manifest, case_id, rows)
    root = Path(manifest['semantic_gate_root'])
    root.mkdir(exist_ok=True)
    directory = root / case_id
    directory.mkdir(exist_ok=False)
    nonce = secrets.token_hex(32)
    ids = {v: secrets.token_hex(12) for v in ('A', 'B')}
    challenge = {'version': 1, 'nonce': nonce, 'created_at': clock(),
        'manifest_sha256': manifest_sha256, 'identity': identity, 'anonymous_ids': ids}
    write_new(directory / 'challenge-private.json', challenge)
    packet_answers = []
    for v in ('A', 'B'):
        cell, _source, _retained = _cell_origin(manifest, case_id, v)
        actions_path = cell / 'actions.json'
        materials = identity.get('materials', [identity.get('material')])
        fallback_source = materials[0]['pdf_sha256'] if len(materials) == 1 and materials[0] else None
        packet_answers.append({'anonymous_id': ids[v], 'answer_sha256': sha256(answers[v]),
            'final_answer': answers[v], 'evidence_locators': evidence_locators(read_json(actions_path), fallback_source)})
        if _observation_policy(manifest):
            # 时间只留在私有身份与事件hash中，盲审包不能反推A/B时延。
            packet_answers[-1].update(review_observation(identity['answers'][v]))
    secrets.SystemRandom().shuffle(packet_answers)
    packet = {'version': 1, 'challenge_nonce': nonce,
        'challenge_sha256': sha256(canonical(challenge)),
        'question': case['prompt'], 'answers': packet_answers,
        'instructions': '只核对原问、全授权源文、终稿及引用；不读取版本映射、成本、执行顺序或设计说明。逐份报告核心覆盖、全文质量、范围覆盖和证据问题；不要把结构合法当正确。'}
    if manifest.get('experiment_kind') == 'prompt_bundle_v1':
        packet.update(materials=identity['materials'], input_preconditions=identity['preconditions'],
            precondition_scope='候选题录与合成历史是用户输入条件，不是原文科学证据或评分标准；当前用户稿件包含于原问。')
    else:
        packet['material'] = case['material']
    if fixed_suite:
        packet['review_protocol'] = {
            'version': 'agent_final_v1',
            'required_execution_flags': ['execution_complete', 'unauthorized_delivery_write', 'forged_success'],
            'scope': '完整终稿与实际执行证据共同判定；语义失误保留记分，未知与严重权限/伪造问题停止。',
        }
    write_new(directory / 'review-packet.json', packet)
    return challenge


def _independent_review_proofs(manifest, challenge, review, expected_sources):
    """实时门验证两份原分与裁定原件，不能依赖私有提交器事后补证明。"""
    root = Path(manifest['experiment_root'])
    if not root.is_absolute() or root.resolve() != root:
        raise PermissionError('PAIR_REVIEW_PROOF_ROOT_INVALID')
    expected = {anonymous: challenge['identity']['answers'][variant]['answer_sha256']
                for variant, anonymous in challenge['anonymous_ids'].items()}

    def checks(value):
        if not isinstance(value, list) or not value:
            return False
        for row in value:
            if not isinstance(row, dict):
                return False
            flat = all(isinstance(row.get(k), str) and row[k].strip()
                       for k in ('source_identity', 'scope', 'evidence'))
            grouped = (isinstance(row.get('reviewer_identity'), str) and row['reviewer_identity'].strip()
                       and checks(row.get('checks')))
            if not flat and not grouped:
                return False
        return True

    def proof(reference, independent):
        required = {'path', 'sha256', 'identity'} if independent else {'path', 'sha256'}
        if not isinstance(reference, dict) or set(reference) != required:
            raise PermissionError('PAIR_REVIEW_PROOF_REFERENCE_REQUIRED')
        try:
            path = Path(reference['path'])
            if (not path.is_absolute() or path.resolve(strict=True) != path or path.is_symlink()
                    or not path.is_file() or path.suffix != '.json' or path.stat().st_nlink != 1
                    or not path.is_relative_to(root)
                    or any(path.is_relative_to(root / name) for name in ('runtime', 'inputs', 'snapshots'))):
                raise PermissionError('PAIR_REVIEW_PROOF_PATH_INVALID')
            content = path.read_bytes()
            if sha256(content) != reference['sha256']:
                raise PermissionError('PAIR_REVIEW_PROOF_BYTES_CHANGED')
            raw = json.loads(content)
        except (OSError, TypeError, ValueError) as exc:
            raise PermissionError('PAIR_REVIEW_PROOF_INVALID') from exc
        if (not isinstance(raw, dict) or not isinstance(raw.get('reviewer_identity'), str)
                or not raw['reviewer_identity'].strip()
                or (independent and raw['reviewer_identity'] != reference['identity'])
                or raw.get('challenge_sha256') != sha256(canonical(challenge))
                or raw.get('blinded') is not True or raw.get('full_authorized_source_checked') is not True
                or type(raw.get('paid_model_calls')) is not int or raw['paid_model_calls'] != 0
                or not checks(raw.get('source_checks'))):
            raise PermissionError('PAIR_INDEPENDENT_SOURCE_REVIEW_REQUIRED')
        rows = raw.get('reviews')
        if (not isinstance(rows, list) or len(rows) != 2
                or any(not isinstance(row, dict) for row in rows)
                or any(not isinstance(row.get('anonymous_id'), str) for row in rows)
                or len({row.get('anonymous_id') for row in rows}) != 2
                or {row.get('anonymous_id'): row.get('answer_sha256') for row in rows} != expected
                or any(row.get('source_sha256s') != expected_sources for row in rows)):
            raise PermissionError('PAIR_INDEPENDENT_ANSWERS_OR_SOURCES_CHANGED')
        return path, raw

    references = review.get('independent_reviews')
    if not isinstance(references, list) or len(references) != 2:
        raise PermissionError('PAIR_TWO_INDEPENDENT_REVIEW_PROOFS_REQUIRED')
    originals = [proof(item, True) for item in references]
    adjudication, raw = proof(review.get('adjudication_source'), False)
    if (len({path for path, _ in originals}) != 2
            or len({value['reviewer_identity'] for _, value in originals}) != 2
            or adjudication in {path for path, _ in originals}):
        raise PermissionError('PAIR_REVIEWERS_MUST_BE_DISTINCT')
    if (not isinstance(raw.get('adjudication_notes'), str) or not raw['adjudication_notes'].strip()
            or raw['adjudication_notes'] != review.get('adjudication_notes')
            or canonical(raw['reviews']) != canonical(review.get('reviews'))
            or canonical(raw['source_checks']) != canonical(review.get('source_checks'))):
        raise PermissionError('PAIR_ADJUDICATION_FINAL_REVIEW_CHANGED')


def evaluate_pair_review(manifest, case_id, rows, manifest_sha256, challenge, review, *, clock=time.time):
    """核对审阅身份与快照，按预先约定的错误条件计算放行。"""
    fixed_suite = validate_final_suite_policy(manifest)
    current_identity, _, _ = pair_identity(manifest, case_id, rows)
    if (challenge.get('manifest_sha256') != manifest_sha256
            or challenge.get('identity') != current_identity
            or review.get('version') != 1
            or review.get('challenge_nonce') != challenge.get('nonce')
            or review.get('challenge_sha256') != sha256(canonical(challenge))):
        raise PermissionError('PAIR_REVIEW_CHALLENGE_OR_SOURCE_MISMATCH')
    at = review.get('reviewed_at')
    if (isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at)
            or not challenge['created_at'] <= at <= clock()):
        raise PermissionError('PAIR_REVIEW_TIME_INVALID')
    reviewer = review.get('reviewer') or {}
    if (reviewer.get('kind') not in {'human', 'agent_assisted'}
            or not isinstance(reviewer.get('identity'), str) or not reviewer['identity'].strip()
            or reviewer.get('blinded') is not True or reviewer.get('full_authorized_source_checked') is not True
            or reviewer.get('paid_model_calls') != 0 or type(reviewer.get('paid_model_calls')) is not int):
        raise PermissionError('PAIR_REVIEWER_IDENTITY_OR_SCOPE_REQUIRED')
    observations = review.get('reviews')
    if not isinstance(observations, list) or len(observations) != 2:
        raise PermissionError('PAIR_TWO_ANONYMOUS_REVIEWS_REQUIRED')
    by_id = {}
    valid_labels = {'full', 'partial_usable', 'unusable', 'not_assessable'}
    if fixed_suite:
        valid_labels |= {'failed', 'unknown'}
    for observation in observations:
        anonymous_id = observation.get('anonymous_id')
        if anonymous_id in by_id:
            raise PermissionError('PAIR_DUPLICATE_ANONYMOUS_REVIEW')
        if (observation.get('whole_answer_quality') not in valid_labels
                or any(type(observation.get(k)) is not bool for k in
                    ('core_complete', 'critical_core_error', 'requested_scope_complete'))
                or not isinstance(observation.get('evidence_notes'), str)
                or not observation['evidence_notes'].strip()):
            raise PermissionError('PAIR_REVIEW_SEMANTIC_FIELDS_REQUIRED')
        if fixed_suite and any(type(observation.get(k)) is not bool for k in
                ('execution_complete', 'unauthorized_delivery_write', 'forged_success')):
            raise PermissionError('PAIR_FINAL_SUITE_EXECUTION_FLAGS_REQUIRED')
        by_id[anonymous_id] = observation
    if set(by_id) != set(challenge['anonymous_ids'].values()):
        raise PermissionError('PAIR_ANONYMOUS_IDS_MISMATCH')
    grades = {}
    expected_sources = ([m['pdf_sha256'] for m in current_identity['materials']]
        if manifest.get('experiment_kind') == 'prompt_bundle_v1' else [current_identity['material']['pdf_sha256']])
    for variant, anonymous_id in challenge['anonymous_ids'].items():
        observation = by_id[anonymous_id]
        if (observation.get('answer_sha256') != current_identity['answers'][variant]['answer_sha256']
                or observation.get('source_sha256s') != expected_sources):
            raise PermissionError('PAIR_REVIEW_ANSWER_OR_MATERIAL_MISMATCH')
        if (_observation_policy(manifest) and current_identity['answers'][variant]['backend_status']
                in manifest['observation_policy']['non_delivery_statuses']
                and (observation['whole_answer_quality'] == 'full' or observation['execution_complete'] is not False)):
            raise PermissionError('PAIR_NON_DELIVERY_CANNOT_CLAIM_COMPLETE')
        if observation['whole_answer_quality'] == 'full' and (
                not observation['core_complete'] or observation['critical_core_error']
                or not observation['requested_scope_complete']
                or (fixed_suite and (not observation['execution_complete']
                    or observation['unauthorized_delivery_write'] or observation['forged_success']
                    or current_identity['answers'][variant]['backend_status'] != 'completed'))):
            raise PermissionError('PAIR_FULL_LABEL_CONTRADICTS_CORE_OR_SCOPE')
        grades[variant] = observation
    if _observation_policy(manifest):
        _independent_review_proofs(manifest, challenge, review, expected_sources)
    reasons = []
    for variant in ('A', 'B'):
        if grades[variant]['critical_core_error'] or not grades[variant]['core_complete']:
            reasons.append(variant + '_core_error_or_missing')
        if grades[variant]['whole_answer_quality'] in {'unusable', 'not_assessable'}:
            reasons.append(variant + '_review_not_usable')
    if not grades['B']['requested_scope_complete']:
        reasons.append('candidate_requested_scope_missing')
    if grades['A']['whole_answer_quality'] == 'full' and grades['B']['whole_answer_quality'] != 'full':
        reasons.append('candidate_quality_regression')
    if (manifest.get('experiment_kind') == 'prompt_bundle_v1'
            and all(grades[v]['whole_answer_quality'] != 'full' for v in ('A', 'B'))):
        reasons.append('both_answers_below_full_quality')
    semantic_findings = []
    if fixed_suite:
        # 普通错误不能擦除或变成full；仅将继续执行的条件与质量通过分开。
        semantic_findings = reasons
        reasons = []
        for variant in ('A', 'B'):
            quality = grades[variant]['whole_answer_quality']
            if quality != 'full':
                semantic_findings.append(variant + '_quality_' + quality)
            if not grades[variant]['requested_scope_complete']:
                semantic_findings.append(variant + '_requested_scope_missing')
            if not grades[variant]['execution_complete']:
                semantic_findings.append(variant + '_execution_incomplete')
            if quality in {'not_assessable', 'unknown'}:
                reasons.append(variant + '_review_unknown')
            for flag in ('unauthorized_delivery_write', 'forged_success'):
                if grades[variant][flag]:
                    reasons.append(variant + '_' + flag)
    decision = {'status': 'passed' if not reasons else 'stopped', 'stop_reasons': reasons,
        'reviewer_kind': reviewer['kind'], 'reviewer_identity': reviewer['identity'],
        'quality': {v: grades[v]['whole_answer_quality'] for v in grades},
        'quality_equivalent': all(grades[v]['whole_answer_quality'] == 'full'
            and current_identity['answers'][v]['backend_status'] == 'completed' for v in grades),
        'review_sha256': sha256(canonical(review)), 'challenge_sha256': sha256(canonical(challenge))}
    if fixed_suite:
        decision.update(semantic_findings=sorted(set(semantic_findings)),
                        continuation_authorized=not reasons, review_policy='agent_final_v1')
    return decision


def wait_pair_review(manifest, case_id, rows, manifest_sha256, deadline, *, clock=time.time, sleep=time.sleep):
    """保留逐对真实审阅；仅显式财务封顶合同允许无全局等待截止。"""
    if deadline is None:
        from agent_final_continuation import validate_continuation
        if not _financial_only(validate_continuation(manifest)):
            raise PermissionError('PAIR_UNBOUNDED_WAIT_REQUIRES_FINANCIAL_ONLY_COMPLETION')
    challenge = prepare_pair_gate(manifest, case_id, rows, manifest_sha256, clock=clock)
    directory = Path(manifest['semantic_gate_root']) / case_id
    review_path = directory / 'review.json'
    while deadline is None or clock() < deadline:
        if review_path.exists() or review_path.is_symlink():
            review = read_json(review_path)
            decision = evaluate_pair_review(manifest, case_id, rows, manifest_sha256, challenge, review, clock=clock)
            write_new(directory / 'decision.json', decision)
            return decision
        pause = manifest['pair_review_policy']['poll_seconds']
        sleep(pause if deadline is None else min(pause, max(0, deadline - clock())))
    decision = {'status': 'stopped', 'stop_reasons': ['semantic_review_deadline'],
        'challenge_sha256': sha256(canonical(challenge)), 'quality_equivalent': False}
    write_new(directory / 'decision.json', decision)
    return decision
