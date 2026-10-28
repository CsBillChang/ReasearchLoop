"""配对终稿的外部语义门；子任务只能写runtime，不能批准本控制目录。"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import secrets
import time

from harness_ab_experiment import canonical, paired_source_contract, sha256


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
    root = Path(manifest['semantic_gate_root'])
    if root.exists() or root.is_symlink():
        raise PermissionError('PAIR_REVIEW_ROOT_MUST_BE_NEW')


def initialize_scope_session(manifest, started_at, *, clock=time.time):
    """仅父批次在控制目录创建一次会话，格进程没有该目录写权限。"""
    root = Path(manifest['semantic_gate_root'])
    root.mkdir(exist_ok=False)
    session = {'version': 1, 'nonce': secrets.token_hex(32), 'manifest_sha256': manifest_identity(manifest),
        'started_at': started_at, 'created_at': clock(), 'controller_pid': os.getpid()}
    write_new(root / 'session.json', session)
    return session


def issue_cell_grant(manifest, case_id, variant, deadline):
    """父调度器只在实际推进到本格时发放一次授权，不预先批准后续格。"""
    control = Path(manifest['semantic_gate_root'])
    session = read_json(control / 'session.json')
    position = manifest['execution_order'].index([case_id, variant])
    grants = control / 'grants'
    grants.mkdir(exist_ok=True)
    value = {'version': 1, 'case_id': case_id, 'variant': variant, 'position': position,
        'manifest_sha256': manifest_identity(manifest), 'session_nonce': session['nonce'],
        'source_contract_sha256': sha256(canonical(paired_source_contract(manifest))),
        'cell_absolute_deadline': deadline, 'grant_nonce': secrets.token_hex(32)}
    write_new(grants / (str(position) + '.json'), value)
    return value


def validate_prior_pair_gates(manifest, case_id, variant, *, clock=time.time):
    """独立格入口同样不能跳过此前配对的审核，也不能脱离父批次运行。"""
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
        grant = read_json(control / 'grants' / (str(position) + '.json'))
        expected = {'version': 1, 'case_id': case_id, 'variant': variant, 'position': position,
            'manifest_sha256': manifest_identity(manifest), 'session_nonce': session['nonce'],
            'source_contract_sha256': sha256(canonical(paired_source_contract(manifest))),
            'cell_absolute_deadline': manifest['cell_absolute_deadline']}
        if (set(grant) != set(expected) | {'grant_nonce'}
                or any(grant.get(k) != v for k, v in expected.items())
                or not isinstance(grant['grant_nonce'], str) or len(grant['grant_nonce']) != 64
                or not clock() < grant['cell_absolute_deadline']):
            raise PermissionError('scheduler grant does not match active cell or deadline')
        for offset in range(0, position // 2 * 2, 2):
            prior = order[offset][0]
            directory = control / prior
            challenge = read_json(directory / 'challenge-private.json')
            review = read_json(directory / 'review.json')
            decision = read_json(directory / 'decision.json')
            rows = [{'case_id': prior, 'variant': v, 'root_id': a['root_id'], 'issues': []}
                for v, a in challenge['identity']['answers'].items()]
            expected = evaluate_pair_review(manifest, prior, rows, manifest_identity(manifest), challenge, review, clock=clock)
            if decision != expected or decision.get('status') != 'passed':
                raise PermissionError('prior semantic review did not pass')
        if position % 2:
            preceding_case, preceding_variant = order[position - 1]
            cell = Path(manifest['runtime_root']) / 'cells' / (preceding_case + '-' + preceding_variant)
            result, view = read_json(cell / 'result.json'), read_json(cell / 'task-view.json')
            if (result.get('status') != 'executed' or not result.get('root_id')
                    or view.get('task_id') != result['root_id'] or view.get('status') not in {'completed', 'partial'}):
                raise PermissionError('first cell of this pair has not ended')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise PermissionError('PAIR_PRIOR_REVIEW_REQUIRED') from exc
    return session


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
                    'source_binding': 'observation' if declared else 'single_authorized_pdf_case',
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
    case = next(c for c in read_json(manifest['cases_path'])['cases'] if c['case_id'] == case_id)
    material = case['material']
    if sha256(Path(material['pdf_path']).read_bytes()) != material['pdf_sha256']:
        raise PermissionError('PAIR_MATERIAL_CHANGED')
    matching = [r for r in rows if r['case_id'] == case_id]
    if len(matching) != 2 or {r['variant'] for r in matching} != {'A', 'B'}:
        raise PermissionError('PAIR_BOTH_ENDED_CELLS_REQUIRED')
    identity = {'case_id': case_id, 'question_sha256': sha256(case['prompt']),
        'material': material, 'cases_sha256': sha256(Path(manifest['cases_path']).read_bytes()), 'answers': {}}
    answers = {}
    runtime = Path(manifest['runtime_root'])
    for row in matching:
        variant = row['variant']
        cell = runtime / 'cells' / (case_id + '-' + variant)
        result = read_json(cell / 'result.json')
        view = read_json(cell / 'task-view.json')
        if (not row.get('root_id') or result.get('root_id') != row['root_id']
                or view.get('task_id') != row['root_id'] or view.get('prompt') != case['prompt']
                or result.get('status') != 'executed' or row.get('issues')
                or view.get('status') not in {'completed', 'partial'}):
            raise PermissionError('PAIR_ENDED_CELL_IDENTITY_INVALID')
        answer = (view.get('result') or {}).get('answer_markdown')
        if not isinstance(answer, str) or not answer.strip():
            raise PermissionError('PAIR_FINAL_ANSWER_REQUIRED')
        filenames = ('result.json', 'task-view.json', 'model-calls.json', 'actions.json')
        hashes = {name: sha256((cell / name).read_bytes()) for name in filenames}
        identity['answers'][variant] = {'root_id': row['root_id'], 'answer_sha256': sha256(answer),
            'backend_status': view['status'], 'files': hashes}
        answers[variant] = answer
    return identity, answers, case


def prepare_pair_gate(manifest, case_id, rows, manifest_sha256, *, clock=time.time):
    """两格结束后才生成随机挑战和匿名包；本函数不作语义判断。"""
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
        actions_path = Path(manifest['runtime_root']) / 'cells' / (case_id + '-' + v) / 'actions.json'
        packet_answers.append({'anonymous_id': ids[v], 'answer_sha256': sha256(answers[v]),
            'final_answer': answers[v], 'evidence_locators': evidence_locators(read_json(actions_path), case['material']['pdf_sha256'])})
    secrets.SystemRandom().shuffle(packet_answers)
    packet = {'version': 1, 'challenge_nonce': nonce,
        'challenge_sha256': sha256(canonical(challenge)),
        'question': case['prompt'], 'material': case['material'], 'answers': packet_answers,
        'instructions': '只核对原问、全授权源文、终稿及引用；不读取版本映射、成本、执行顺序或设计说明。逐份报告核心覆盖、全文质量、范围覆盖和证据问题；不要把结构合法当正确。'}
    write_new(directory / 'review-packet.json', packet)
    return challenge


def evaluate_pair_review(manifest, case_id, rows, manifest_sha256, challenge, review, *, clock=time.time):
    """核对审阅身份与快照，按预先约定的错误条件计算放行。"""
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
        by_id[anonymous_id] = observation
    if set(by_id) != set(challenge['anonymous_ids'].values()):
        raise PermissionError('PAIR_ANONYMOUS_IDS_MISMATCH')
    grades = {}
    expected_sources = [current_identity['material']['pdf_sha256']]
    for variant, anonymous_id in challenge['anonymous_ids'].items():
        observation = by_id[anonymous_id]
        if (observation.get('answer_sha256') != current_identity['answers'][variant]['answer_sha256']
                or observation.get('source_sha256s') != expected_sources):
            raise PermissionError('PAIR_REVIEW_ANSWER_OR_MATERIAL_MISMATCH')
        if observation['whole_answer_quality'] == 'full' and (
                not observation['core_complete'] or observation['critical_core_error']
                or not observation['requested_scope_complete']):
            raise PermissionError('PAIR_FULL_LABEL_CONTRADICTS_CORE_OR_SCOPE')
        grades[variant] = observation
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
    return {'status': 'passed' if not reasons else 'stopped', 'stop_reasons': reasons,
        'reviewer_kind': reviewer['kind'], 'reviewer_identity': reviewer['identity'],
        'quality': {v: grades[v]['whole_answer_quality'] for v in grades},
        'quality_equivalent': all(grades[v]['whole_answer_quality'] == 'full'
            and current_identity['answers'][v]['backend_status'] == 'completed' for v in grades),
        'review_sha256': sha256(canonical(review)), 'challenge_sha256': sha256(canonical(challenge))}


def wait_pair_review(manifest, case_id, rows, manifest_sha256, deadline, *, clock=time.time, sleep=time.sleep):
    """计时不暂停；缺审阅到截止即停，不重新创建任务或账本。"""
    challenge = prepare_pair_gate(manifest, case_id, rows, manifest_sha256, clock=clock)
    directory = Path(manifest['semantic_gate_root']) / case_id
    review_path = directory / 'review.json'
    while clock() < deadline:
        if review_path.exists() or review_path.is_symlink():
            review = read_json(review_path)
            decision = evaluate_pair_review(manifest, case_id, rows, manifest_sha256, challenge, review, clock=clock)
            write_new(directory / 'decision.json', decision)
            return decision
        sleep(min(manifest['pair_review_policy']['poll_seconds'], max(0, deadline - clock())))
    decision = {'status': 'stopped', 'stop_reasons': ['semantic_review_deadline'],
        'challenge_sha256': sha256(canonical(challenge)), 'quality_equivalent': False}
    write_new(directory / 'decision.json', decision)
    return decision
