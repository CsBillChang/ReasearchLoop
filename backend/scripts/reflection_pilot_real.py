"""六稿 Reflection 专属执行入口；依赖显式授权、资源和已包装的供应商工厂。

首轮普通回答核验由评测程序提交，原稿字节不经 Supervisor 预先改写。
后续修订和结束由当前业务图中的模型决定；本模块不读取评分标签或密钥。
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch
from urllib.parse import urlparse
from uuid import uuid4

from scripts.prepare_reflection_baseline import check_model_payload


PILOT_IDS = frozenset(f'R{number:02}' for number in range(1, 7))
POLICY_FIELDS = ('deepseek_model', 'deepseek_base_url', 'request_timeout_seconds',
    'max_model_requests', 'max_tool_actions', 'reserve_model_requests', 'max_reflection_rounds',
    'execution_budget_mode', 'task_token_limit', 'task_token_tail', 'context_window_tokens',
    'context_output_reserve', 'context_safety_tokens', 'context_compress_ratio',
    'model_max_output_tokens', 'retrieval_enabled', 'web_search_enabled', 'web_fetch_enabled')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_new(path: Path, value: dict) -> None:
    """独占写新记录，禁止用重跑覆盖首次输出。"""
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def _outside_checkout(path: Path) -> Path:
    """所有运行状态必须在任意 Git checkout 之外，且先检查再创建目录。"""
    result = Path(path).resolve()
    for ancestor in (result, *result.parents):
        marker = ancestor/'.git'
        if marker.exists() or marker.is_symlink():
            raise ValueError('REFLECTION_PILOT_RUNTIME_INSIDE_CHECKOUT')
    return result


def _load_case(path: Path, expected_sha256: str) -> tuple[dict, str]:
    """只读已冻结的模型输入文件；不能把标签文件传作执行材料。"""
    if 'labels' in Path(path).parts:
        raise PermissionError('REFLECTION_PILOT_LABEL_FILE_FORBIDDEN')
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError('REFLECTION_PILOT_INPUT_DRIFT')
    case = json.loads(raw)
    if case.get('case_id') not in PILOT_IDS:
        raise ValueError('REFLECTION_PILOT_CASE_NOT_APPROVED')
    model_input = case.get('model_input')
    if not isinstance(model_input, dict):
        raise ValueError('REFLECTION_PILOT_MODEL_INPUT_REQUIRED')
    check_model_payload(model_input)
    for first, second in [('question', 'original_question'), ('draft', 'draft_markdown')]:
        if first in model_input and second in model_input and model_input[first] != model_input[second]:
            raise ValueError('REFLECTION_PILOT_INPUT_ALIAS_CONFLICT')
    if not (model_input.get('question') or model_input.get('original_question')):
        raise ValueError('REFLECTION_PILOT_QUESTION_REQUIRED')
    if not (model_input.get('draft') or model_input.get('draft_markdown')):
        raise ValueError('REFLECTION_PILOT_DRAFT_REQUIRED')
    documents = model_input.get('documents') or []
    if len(documents) != 1 or len(documents[0].get('allowed_pages', [])) != 1:
        raise ValueError('REFLECTION_PILOT_SINGLE_APPROVED_PAGE_REQUIRED')
    for page in documents[0].get('initial_pages', []):
        if page.get('physical_page') not in documents[0]['allowed_pages']:
            raise ValueError('REFLECTION_PILOT_PAGE_OUTSIDE_SCOPE')
        if hashlib.sha256(page['text'].encode()).hexdigest() != page.get('page_text_sha256'):
            raise ValueError('REFLECTION_PILOT_PAGE_TEXT_DRIFT')
    return case, actual


def _approval(case_id: str, approval: dict, actual_sha256: str) -> None:
    """每案必须在六稿真实授权内；计划或其他批次授权不能代替它。"""
    if approval.get('approved') is not True or not approval.get('approval_id'):
        raise PermissionError('REFLECTION_PILOT_APPROVAL_REQUIRED')
    approved_ids = set(approval.get('case_ids', []))
    if not approved_ids or not approved_ids.issubset(PILOT_IDS) or case_id not in approved_ids:
        raise PermissionError('REFLECTION_PILOT_SCOPE_NOT_APPROVED')
    frozen = approval.get('case_input_sha256s')
    if frozen is not None and frozen.get(case_id) != actual_sha256:
        raise PermissionError('REFLECTION_PILOT_APPROVAL_INPUT_MISMATCH')


def _settings(settings: Any, root: Path) -> dict:
    """只接受调用方构造的隔离 Settings，保留模型与业务策略原数值。"""
    if Path(settings.project_root).resolve() != root:
        raise ValueError('REFLECTION_PILOT_PROJECT_ROOT_MISMATCH')
    for value in (settings.data_dir, settings.output_dir):
        selected = Path(value).resolve()
        if selected == root or not selected.is_relative_to(root):
            raise ValueError('REFLECTION_PILOT_DATA_OUTPUT_NOT_ISOLATED')
    parsed = urlparse(settings.database_url)
    if parsed.scheme != 'sqlite' or parsed.netloc:
        raise ValueError('REFLECTION_PILOT_SQLITE_REQUIRED')
    database = Path(parsed.path).resolve()
    if database == root or not database.is_relative_to(root):
        raise ValueError('REFLECTION_PILOT_DATABASE_NOT_ISOLATED')
    if not settings.retrieval_enabled:
        raise ValueError('REFLECTION_PILOT_RETRIEVAL_MUST_REMAIN_ENABLED')
    if settings.web_search_enabled or settings.web_fetch_enabled:
        raise ValueError('REFLECTION_PILOT_NO_WEB_AUTHORIZATION')
    return {name: getattr(settings, name) for name in POLICY_FIELDS}


def _prompt(case: dict, attachments: list[dict]) -> str:
    """将泛指所附页展开成精确用户范围，避免默认全文读取。"""
    value = case['model_input']
    if len(attachments) != 1:
        raise ValueError('REFLECTION_PILOT_ONE_ATTACHMENT_REQUIRED')
    page = value['documents'][0]['allowed_pages'][0]
    return ('本次任务是核验并处理一份已给定的回答。首轮核验由评测程序提交当前原稿。\n'
        '原始问题逐字保存在首轮核验动作的 verification_questions，原稿在 answer_markdown。\n'
        f"仅允许读取附件 {attachments[0]['input_ref']} 的物理第{page}页。\n"
        '核验后依据实际结果决定保留、局部修订、披露缺证或结束；保留其他正确内容。'
        '禁止保存文件或联网扩展。最终交付当前正文及必要的处理说明。')


def _seed_review(db, task_id: str, case: dict, prompt: str, attachments: list[dict],
                 evidence_ids: list[str], settings) -> dict:
    """仅种入程序首轮核验动作；不伪造模型响应或模型收据。"""
    from research_loop.agents.graph import initial_state
    from research_loop.domain.execution_policy import inherit_execution_policy
    from research_loop.storage.checkpoints import CheckpointStore
    value = case['model_input']
    draft = value.get('draft') or value['draft_markdown']
    question = value.get('question') or value['original_question']
    args = {'target_type': 'answer', 'answer_markdown': draft,
            'reason': '用户批准对这份冻结原稿进行当前基线核验。',
            'verification_questions': [question], 'evidence_ids': evidence_ids}
    identity = 'programmatic-review-' + uuid4().hex
    state = initial_state(task_id, prompt, 'deny', {a['input_ref']: a for a in attachments},
        settings.max_model_requests, settings.max_tool_actions, settings.reserve_model_requests)
    inherit_execution_policy(state, db.execution_policy(task_id), db.budget_summary(task_id))
    state.update(coverage_required=True, evidence_ids=evidence_ids,
        pending_tool_scope=['review_research'],
        pending_tool_calls=[{'call_id': identity, 'name': 'review_research', 'arguments': args}])
    state['messages'].append({'role': 'assistant',
        'content': '评测程序按已批准固定稿入口发起首轮核验；此动作不是模型生成。',
        'tool_calls': [{'id': identity, 'type': 'function', 'function': {
            'name': 'review_research', 'arguments': json.dumps(args, ensure_ascii=False)}}]})
    origin = {'origin': 'evaluation_program_explicit_fixed_draft_review', 'model_generated': False,
              'model_receipt_created': False, 'tool_call_id': identity,
              'draft_sha256': hashlib.sha256(draft.encode()).hexdigest()}
    db.append_event(task_id, 'reflection_pilot_programmatic_review', origin)
    CheckpointStore(db).save(task_id, state)
    return origin


def _receipts(db, task_id: str) -> list[dict]:
    return db.fetchall('SELECT m.* FROM model_calls m JOIN tasks t ON t.task_id=m.task_id '
        'WHERE t.task_id=? OR t.root_task_id=? ORDER BY m.created_at, m.call_id', (task_id, task_id))


def _external_stop(stop_state: Callable | None) -> str | None:
    if stop_state is None:
        return None
    value = stop_state()
    if isinstance(value, dict):
        return value.get('global_stop') or value.get('stop_reason')
    return str(value) if value else None


def _receipt_stop(rows: list[dict]) -> str | None:
    """业务图保留失败后仍须停止整批，不能自动送下一份稿件。"""
    for row in rows:
        if row['status'] == 'unknown':
            return 'unknown_model_call'
        if row['status'] == 'succeeded':
            usage = json.loads(row.get('usage_json') or '{}')
            if not all(key in usage for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
                return 'successful_call_usage_missing'
        if row['status'] in {'rejected', 'response_error'}:
            raw = json.loads(row.get('response_json') or '{}')
            if raw.get('status_code') == 402:
                return 'http_402'
            return 'provider_response_error'
    return None


def execute_case(case_path: Path, case_dir: Path, *, expected_input_sha256: str,
                 settings: Any, provider_factory: Callable, resource_factory: Callable,
                 approval: dict, execution_kind: str = 'real_model',
                 stop_state: Callable | None = None) -> dict:
    """对一份明确批准的冻结稿运行一次；不读密钥、标签或默认.env。

    resource_factory(case_dir, settings, case) 必须返回具有 attachments、bind、
    pdf_factory、retrieval_factory、report 的专属资源对象。供应商工厂由调用方
    注入共享的持久预算包装器；本入口不会替换它或另建无预算供应商。
    """
    if execution_kind not in {'real_model', 'substitute'}:
        raise ValueError('REFLECTION_PILOT_EXECUTION_KIND')
    case, source_hash = _load_case(case_path, expected_input_sha256)
    _approval(case['case_id'], approval, source_hash)
    root = _outside_checkout(case_dir)
    policy = _settings(settings, root)
    if not callable(provider_factory) or not callable(resource_factory):
        raise ValueError('REFLECTION_PILOT_EXPLICIT_FACTORIES_REQUIRED')
    stopped = _external_stop(stop_state)
    if stopped:
        return {'case_id': case['case_id'], 'created': False, 'uncreated_reason': stopped,
                'global_stop': stopped, 'execution_kind': execution_kind}
    root.mkdir(parents=True, exist_ok=False)
    _write_new(root/'started.json', {'case_id': case['case_id'], 'at': _now(),
        'input_sha256': source_hash, 'approval_id': approval['approval_id'],
        'execution_kind': execution_kind, 'retry_policy': 'once_no_automatic_replay'})
    from research_loop.services.tasks import TaskService
    from research_loop.domain.tasks import TaskCreate
    from research_loop.storage.mysql import Database
    from research_loop.workers.executor import TaskExecutor
    from research_loop.agents.reflection import ReflectionService

    db, resources, task_id, origin = None, None, None, None
    error = None
    try:
        resources = resource_factory(root, settings, case)
        db = Database(settings.database_url)
        db.ensure_schema()
        prompt = _prompt(case, resources.attachments)
        with patch('research_loop.config.get_settings', return_value=settings):
            task = TaskService(db, settings).create(TaskCreate(prompt=prompt, save_policy='deny',
                attachments=resources.attachments))
            task_id = task.task_id
            bound = resources.bind(task_id, db)
            if bound.get('vector_status') != 'ready' or not bound.get('evidence_ids'):
                raise ValueError('REFLECTION_PILOT_RESOURCES_NOT_READY')
            origin = _seed_review(db, task_id, case, prompt, resources.attachments,
                                  bound['evidence_ids'], settings)
            executor = TaskExecutor(settings, db, provider_factory, resources.pdf_factory)
            executor.retrieval = resources.retrieval_factory(db, settings)
            executor.execute(task_id, 'reflection-pilot-' + case['case_id'])
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    rows = _receipts(db, task_id) if db is not None and task_id else []
    view, actions, events, reviews, snapshot = None, [], [], [], None
    if task_id:
        with patch('research_loop.config.get_settings', return_value=settings):
            view = TaskService(db, settings).view(db.get_task(task_id)).model_dump(mode='json')
        actions, events = db.list_actions(task_id), db.list_events(task_id)
        reviews = ReflectionService(db).research_context(task_id)
        snapshot = ReflectionService(db).snapshot(task_id)
    value = case['model_input']
    draft = value.get('draft') or value['draft_markdown']
    first_targets = [record['attempts'][0]['target'] for record in
        (snapshot or {}).get('research_targets', {}).values() if record.get('attempts')]
    original_preserved = bool(first_targets) and all(
        target['items'][0]['text'] == draft for target in first_targets if target['target_type'] == 'answer')
    global_stop = _external_stop(stop_state) or _receipt_stop(rows)
    if error and not global_stop:
        global_stop = 'case_exception:' + error['type']
    if first_targets and not original_preserved:
        global_stop = 'frozen_first_draft_not_preserved'
    report = {'case_id': case['case_id'], 'created': bool(task_id), 'task_id': task_id,
        'execution_kind': execution_kind, 'input_sha256': source_hash,
        'original_draft': draft, 'original_draft_sha256': hashlib.sha256(draft.encode()).hexdigest(),
        'first_review_original_preserved': original_preserved, 'programmatic_review_origin': origin,
        'view': view, 'actions': actions, 'events': events, 'research_reviews': reviews,
        'reflection_snapshot': snapshot, 'physical_receipts': rows,
        'sent_receipt_count': sum(row.get('sent_state') == 'sent' for row in rows),
        'real_model_calls': sum(row.get('sent_state') == 'sent' for row in rows)
                            if execution_kind == 'real_model' else 0,
        'actual_request_parameters': [{key: json.loads(row['request_json'])[key]
            for key in ('model', 'max_tokens', 'thinking') if key in json.loads(row['request_json'])}
            for row in rows], 'policy_values': policy,
        'resources': resources.report() if resources is not None else None,
        'error': error, 'global_stop': global_stop,
        'semantic_quality': 'unreviewed' if execution_kind == 'real_model' else 'not_evaluated_substitute',
        'finished_at': _now()}
    if not task_id:
        report['uncreated_reason'] = (error or {}).get('message') or 'preparation_failed'
    _write_new(root/'result.json', report)
    return report


def run_pilot(case_specs: list[dict], run_root: Path, *, settings_factory: Callable,
              provider_factory: Callable, resource_factory: Callable, approval: dict,
              execution_kind: str = 'real_model', stop_state: Callable | None = None) -> dict:
    """顺序运行已批准六稿，明确保留停止后的未创建项，不自动恢复或重跑。"""
    ids = []
    for spec in case_specs:
        case, digest = _load_case(spec['case_path'], spec['expected_input_sha256'])
        _approval(case['case_id'], approval, digest)
        ids.append(case['case_id'])
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('REFLECTION_PILOT_DUPLICATE_OR_EMPTY_PLAN')
    root = _outside_checkout(run_root)
    root.mkdir(parents=True, exist_ok=True)
    _write_new(root/'batch-started.json', {'case_ids': ids, 'approval_id': approval['approval_id'],
        'execution_kind': execution_kind, 'at': _now(), 'retry_policy': 'once_no_automatic_replay'})
    results, stopped = [], _external_stop(stop_state)
    for spec, case_id in zip(case_specs, ids):
        stopped = stopped or _external_stop(stop_state)
        if stopped:
            results.append({'case_id': case_id, 'created': False, 'uncreated_reason': stopped})
            continue
        case, _ = _load_case(spec['case_path'], spec['expected_input_sha256'])
        case_dir = Path(spec.get('case_dir') or root/'cases'/case_id).resolve()
        if not case_dir.is_relative_to(root):
            raise ValueError('REFLECTION_PILOT_CASE_OUTSIDE_BATCH_ROOT')
        try:
            result = execute_case(spec['case_path'], case_dir,
                expected_input_sha256=spec['expected_input_sha256'],
                settings=settings_factory(case_dir, case), provider_factory=provider_factory,
                resource_factory=resource_factory, approval=approval,
                execution_kind=execution_kind, stop_state=stop_state)
            results.append(result)
            stopped = result.get('global_stop')
        except Exception as exc:
            stopped = type(exc).__name__ + ': ' + str(exc)
            results.append({'case_id': case_id, 'created': False, 'uncreated_reason': stopped,
                            'execution_kind': execution_kind})
    report = {'case_ids': ids, 'planned': len(ids), 'created': sum(r['created'] for r in results),
        'uncreated': sum(not r['created'] for r in results), 'results': results,
        'global_stop': stopped, 'execution_kind': execution_kind, 'finished_at': _now()}
    _write_new(root/'batch-results.json', report)
    return report
