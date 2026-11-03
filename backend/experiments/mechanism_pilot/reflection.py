"""六稿 Reflection 的隔离替身入口；真实执行尚未开放。

仅装配当前业务 TaskExecutor，不提供新 Reflection 策略或物理财务账本。
后续真实运行必须另行接入经资格验证的共享预算，不因传入批准标志而开放。
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from unittest.mock import patch

import httpx

from research_loop.services.retrieval import RetrievalRuntime
from research_loop.storage.token_budget import actual_usage


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@contextmanager
def deny_network():
    """替身资格期间连错误配置的工厂也不能连接网络。"""
    def denied(*args, **kwargs):
        raise PermissionError('MECHANISM_ZERO_MODEL_NETWORK_DENIED')
    with patch('socket.socket.connect', denied), patch('socket.socket.connect_ex', denied), \
            patch('socket.create_connection', denied), patch('socket.getaddrinfo', denied):
        yield


def load_case(path: Path, expected_sha256: str) -> dict:
    """校验六稿输入、单页身份和标签隔离，不加载评分文件。"""
    path = Path(path).resolve()
    if 'labels' in path.parts:
        raise PermissionError('REFLECTION_LABEL_FILE_FORBIDDEN')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('REFLECTION_INPUT_DRIFT')
    case = json.loads(raw)
    if case.get('case_id') not in {f'R{i:02}' for i in range(1, 7)}:
        raise ValueError('REFLECTION_SIX_CASE_SCOPE')
    value = case.get('model_input')
    def check(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {'gold', 'labels', 'expected', 'expected_text', 'expected_handling',
                           'category', 'claims', 'before_status'}:
                    raise ValueError('REFLECTION_LABEL_IN_MODEL_INPUT')
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
    check(value)
    if not isinstance(value, dict):
        raise ValueError('REFLECTION_MODEL_INPUT_REQUIRED')
    for key in ('question', 'draft'):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError('REFLECTION_QUESTION_AND_DRAFT_REQUIRED')
    docs = value.get('documents', [])
    if len(docs) != 1 or len(docs[0].get('allowed_pages', [])) != 1:
        raise ValueError('REFLECTION_SINGLE_PAGE_REQUIRED')
    if any(not isinstance(docs[0].get(key), str) or not docs[0][key].strip()
           for key in ('document_id', 'paper_id', 'source_sha256')):
        raise ValueError('REFLECTION_MATERIAL_IDENTITY_REQUIRED')
    pages = docs[0].get('initial_pages', [])
    if len(pages) != 1 or pages[0]['physical_page'] != docs[0]['allowed_pages'][0]:
        raise ValueError('REFLECTION_INITIAL_PAGE_SCOPE')
    if _hash(pages[0]['text']) != pages[0]['page_text_sha256']:
        raise ValueError('REFLECTION_PAGE_TEXT_DRIFT')
    return case


def _validate_bound_material(db, task_id, case, attachments, bound, expected_identity=None):
    """在程序首轮前核对冻结原文与任务实际获授权的页，不自动猜测旧 ID。"""
    document = case['model_input']['documents'][0]
    page = document['initial_pages'][0]
    identity = {'input_document_id': document['document_id'], 'runtime_document_id': document['document_id'],
        'paper_id': document['paper_id'], 'source_sha256': document['source_sha256'],
        'physical_page': page['physical_page'], 'page_text_sha256': page['page_text_sha256']}
    if expected_identity is not None:
        # 映射须由冻结资格清单显式注入；除运行时 ID 外不能改写输入身份。
        if (set(expected_identity) != set(identity)
                or any(expected_identity[key] != value for key, value in identity.items()
                       if key != 'runtime_document_id')
                or not isinstance(expected_identity['runtime_document_id'], str)
                or not expected_identity['runtime_document_id'].strip()):
            raise ValueError('REFLECTION_MATERIAL_MAPPING_CONFLICT')
        identity = dict(expected_identity)
    evidence_ids = bound.get('evidence_ids')
    if (not isinstance(evidence_ids, list) or len(evidence_ids) != 1
            or not isinstance(evidence_ids[0], str) or not db.task_has_evidence(task_id, evidence_ids[0])):
        raise ValueError('REFLECTION_BOUND_SINGLE_AUTHORIZED_EVIDENCE_REQUIRED')
    actual_page = db.get_page(evidence_ids[0])
    actual_document = db.fetchone('SELECT * FROM documents WHERE document_id=?',
        (actual_page['document_id'],)) if actual_page else None
    if (not actual_page or not actual_document
            or attachments[0].get('paper_id') != identity['paper_id']
            or actual_document['document_id'] != identity['runtime_document_id']
            or actual_document['paper_id'] != identity['paper_id']
            or actual_document['content_sha256'] != identity['source_sha256']
            or actual_page['page_number'] != identity['physical_page']
            or _hash(actual_page['text']) != identity['page_text_sha256']
            or actual_page['text'] != page['text']):
        raise ValueError('REFLECTION_BOUND_MATERIAL_MISMATCH')
    return {**identity, 'evidence_id': evidence_ids[0]}


def _strict_actual_usage(usage):
    """六稿入口要求三个完整且一致的整数；未知不能使用业务兼容推算。"""
    keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    if not isinstance(usage, dict) or not all(type(usage.get(k)) is int and usage[k] >= 0 for k in keys):
        return None
    return actual_usage(usage)


class ScopedRetrievalRuntime(RetrievalRuntime):
    """只增加任务范围检查；原生装配保留 V1.6.0 检索路线与后端。"""
    def __init__(self, db, settings, validate_scope):
        super().__init__(db, settings)
        self.validate_scope = validate_scope

    def search(self, task_id, request, cancelled=lambda: False):
        self.validate_scope(task_id, self.db, request.document_ids)
        return super().search(task_id, request, cancelled)


def programmatic_review_message(identity: str, arguments: dict) -> dict:
    """仅新建程序首轮动作的空推理槽；不补写任何真实模型历史。"""
    if not identity.startswith('programmatic-review-') or identity == 'programmatic-review-':
        raise ValueError('REFLECTION_PROGRAM_ID_REQUIRED')
    return {'role': 'assistant', 'content': '评测程序提交冻结原稿；此动作不是模型生成。',
            'reasoning_content': '', 'tool_calls': [{'id': identity, 'type': 'function',
            'function': {'name': 'review_research', 'arguments': json.dumps(arguments, ensure_ascii=False)}}]}


def _seed_review(db, task_id, case, prompt, attachments, evidence_ids, settings, read_state=None):
    """记录显式程序来源和原稿 hash 后保存首轮待执行动作。"""
    from research_loop.agents.graph import initial_state
    from research_loop.domain.execution_policy import inherit_execution_policy
    from research_loop.storage.checkpoints import CheckpointStore
    value = case['model_input']
    args = {'target_type': 'answer', 'answer_markdown': value['draft'],
            'reason': '核验冻结原稿。', 'verification_questions': [value['question']],
            'evidence_ids': evidence_ids}
    identity = 'programmatic-review-' + uuid4().hex
    state = read_state if read_state is not None else initial_state(
        task_id, prompt, 'deny', {a['input_ref']: a for a in attachments},
        settings.max_model_requests, settings.max_tool_actions, settings.reserve_model_requests)
    if (state['task_id'] != task_id or state.get('model_requests') != 0
            or state.get('pending_tool_calls') or state.get('finished')):
        raise ValueError('REFLECTION_PROGRAMMATIC_READ_STATE_INVALID')
    inherit_execution_policy(state, db.execution_policy(task_id), db.budget_summary(task_id))
    state.update(coverage_required=True, evidence_ids=evidence_ids,
        pending_tool_scope=['review_research'],
        pending_tool_calls=[{'call_id': identity, 'name': 'review_research', 'arguments': args}])
    state['messages'].append(programmatic_review_message(identity, args))
    origin = {'origin': 'evaluation_program_explicit_fixed_draft_review', 'model_generated': False,
        'model_receipt_created': False, 'tool_call_id': identity, 'draft_sha256': _hash(value['draft'])}
    db.append_event(task_id, 'reflection_pilot_programmatic_review', origin)
    CheckpointStore(db).save(task_id, state)
    return origin


def inspect_delivery(draft: str, snapshot: dict, actions: list, result: dict | None) -> dict:
    """检查原稿、修订链与实际交付；无对象绝不因空集合而通过。"""
    from research_loop.domain.reflection import target_fingerprint
    records = [r for r in snapshot.get('research_targets', {}).values()
        if r.get('attempts') and r['attempts'][0]['target'].get('target_type') == 'answer']
    matching = [r for r in records if r['attempts'][0]['target'].get('items')
        and r['attempts'][0]['target']['items'][0].get('text') == draft]
    checks = {'first_review_original_preserved': len(records) == len(matching) == 1,
              'current_identity_valid': False, 'revision_chain_valid': False,
              'final_body_matches_current': False, 'finish_bound_current': False}
    result = result or {}
    body = result.get('answer_body_markdown')
    if len(matching) == 1:
        record = matching[0]
        current = record['current']
        versions = {v['content_hash']: v for v in record['versions']}
        checks['current_identity_valid'] = current['content_hash'] == target_fingerprint(current)
        chain_valid = all(v['content_hash'] == target_fingerprint(v) for v in versions.values())
        for change in record.get('patches', []):
            before, after = versions.get(change['before_hash']), versions.get(change['after_hash'])
            if not before or not after or not any(
                    (a.get('result') or {}).get('review_id') == change['review_id']
                    and a['target']['content_hash'] == change['before_hash'] for a in record['attempts']):
                chain_valid = False
                continue
            texts = {item['item_id']: item['text'] for item in before['items']}
            edits = {}
            for edit in change['patches']:
                text = texts.get(edit['item_id'], '')
                old = edit['old_text']
                if not old or text.count(old) != 1:
                    chain_valid = False
                    continue
                start = text.index(old)
                edits.setdefault(edit['item_id'], []).append((start, start + len(old), edit['new_text']))
            for identity, spans in edits.items():
                spans.sort()
                if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
                    chain_valid = False
                for start, end, new in reversed(spans):
                    texts[identity] = texts[identity][:start] + new + texts[identity][end:]
            chain_valid = chain_valid and texts == {item['item_id']: item['text'] for item in after['items']}
        checks['revision_chain_valid'] = chain_valid
        checks['final_body_matches_current'] = isinstance(body, str) and body == current['items'][0]['text']
        finishes = [a for a in actions if a['tool_name'] == 'finish_task' and a['status'] == 'succeeded']
        if finishes:
            args = finishes[-1]['arguments']
            checks['finish_bound_current'] = (args.get('reflection_target_id') == current['target_id']
                and args.get('reflection_content_hash') == current['content_hash'])
    return {**checks, 'answer_body_markdown': body, 'processing_summary': result.get('processing_summary'),
            'answer_body_sha256': _hash(body) if isinstance(body, str) else None,
            'delivery_path': 'bound_finish' if checks['finish_bound_current'] else 'fallback_or_missing',
            'semantic_quality': 'not_evaluated_substitute'}


def _validate_runtime(root, settings, execution_kind='substitute'):
    root = Path(root).resolve()
    if any((p/'.git').exists() for p in (root, *root.parents)):
        raise ValueError('REFLECTION_RUNTIME_INSIDE_CHECKOUT')
    if Path(settings.project_root).resolve() != root:
        raise ValueError('REFLECTION_SETTINGS_ROOT_MISMATCH')
    url = urlsplit(settings.database_url)
    if url.scheme != 'sqlite' or url.netloc or not Path(url.path).resolve().is_relative_to(root):
        raise ValueError('REFLECTION_ISOLATED_SQLITE_REQUIRED')
    if any(Path(p).resolve() == root or not Path(p).resolve().is_relative_to(root)
           for p in (settings.data_dir, settings.output_dir)):
        raise ValueError('REFLECTION_ISOLATED_DATA_OUTPUT_REQUIRED')
    if settings.web_search_enabled or settings.web_fetch_enabled:
        raise ValueError('REFLECTION_WEB_DISABLED_REQUIRED')
    if execution_kind == 'substitute' and (settings.deepseek_api_key or settings.deepseek_base_url != 'https://fixture.invalid'):
        raise PermissionError('REFLECTION_SYNTHETIC_SETTINGS_REQUIRED')
    return root


def execute_case(case_path, runtime_dir, *, expected_input_sha256, settings,
                 provider_factory, resource_factory, stop_state=lambda: None,
                 execution_kind='substitute', expected_material_identity=None,
                 authorize=None, freeze_sha256=None, on_task_created=lambda task_id, db: None,
                 recovery_policy=None) -> dict:
    """经当前Worker执行冻结稿；真实模式在创建目录前校验完整新授权。

    资格回归使用MockTransport或包裹它的共享预算传输。真实模式必须使用
    QualificationTransport且逐次重验冻结合同；本模块不自行创建HTTP连接。
    """
    from experiments.mechanism_pilot.recovery import FailedActionGate
    from experiments.mechanism_pilot.contract import authorize_execution
    authorize_execution(settings, execution_kind, authorize, freeze_sha256)
    failures = FailedActionGate(recovery_policy)
    case = load_case(case_path, expected_input_sha256)
    root = _validate_runtime(runtime_dir, settings, execution_kind)
    stopped = stop_state()
    if stopped:
        return {'created': False, 'case_id': case['case_id'], 'global_stop': stopped}
    root.mkdir(parents=True, exist_ok=False)
    from research_loop.agents.reflection import ReflectionService
    from research_loop.domain.tasks import TaskCreate
    from research_loop.providers.deepseek import RequestNotSent
    from research_loop.services.tasks import TaskService
    from research_loop.storage.mysql import Database
    from research_loop.workers.executor import TaskExecutor
    gate = {'stop': None}

    def failed_action_stop():
        """参数更正只停止本格，资格/计费错误仍停止整批。"""
        return failures.poll(db, task_id)

    class GuardedProvider:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(self.original, name)

        def _invoke(self, method, *args, **kwargs):
            reason = gate['stop'] or failed_action_stop() or stop_state()
            if reason:
                raise RequestNotSent(str(reason))
            try:
                raw = method(*args, **kwargs)
            except BaseException as exc:
                gate['stop'] = gate['stop'] or 'provider_error:' + type(exc).__name__
                raise
            usage = raw.get('usage')
            if _strict_actual_usage(usage) is None:
                gate['stop'] = gate['stop'] or 'successful_call_usage_unknown'
            return raw

        def send(self, *args, **kwargs):
            return self._invoke(self.original.send, *args, **kwargs)

        def send_stream(self, *args, **kwargs):
            return self._invoke(self.original.send_stream, *args, **kwargs)

        def complete(self, *args, **kwargs):
            return self.original.parse_response(self.send(*args, **kwargs))

        def close(self):
            self.original.close()
            self.original.client.close()

    def factory(current):
        from experiments.mechanism_pilot.transport import QualificationTransport
        provider = provider_factory(current)
        transport = getattr(getattr(provider, 'client', None), '_transport', None)
        qualified = isinstance(transport, QualificationTransport) and transport.mode == execution_kind
        if not qualified and not (execution_kind == 'substitute' and isinstance(transport, httpx.MockTransport)):
            provider.close()
            raise PermissionError('REFLECTION_QUALIFIED_TRANSPORT_REQUIRED')
        return GuardedProvider(provider)

    db, task_id, origin, error, material_identity = None, None, None, None, None
    with (deny_network() if execution_kind == 'substitute' else nullcontext()), patch('research_loop.config.get_settings', return_value=settings), \
            patch('research_loop.config.Settings.from_env', side_effect=PermissionError('REFLECTION_DEFAULT_ENV_DENIED')):
        try:
            resources = resource_factory(root, settings, case)
            db = Database(settings.database_url)
            db.ensure_schema()
            attachments = resources.attachments
            if len(attachments) != 1:
                raise ValueError('REFLECTION_ONE_ATTACHMENT_REQUIRED')
            page = case['model_input']['documents'][0]['allowed_pages'][0]
            prompt = (f'核验并处理冻结回答；首轮由程序提交。仅允许读取附件 {attachments[0]["input_ref"]} '
                f'的物理第{page}页。核验后保留正确内容，按需局部修订或披露缺证，最终交付当前正文。禁止保存文件或联网扩展。')
            task_id = TaskService(db, settings).create(TaskCreate(prompt=prompt,
                save_policy='deny', attachments=attachments)).task_id
            on_task_created(task_id, db)
            bound = resources.bind(task_id, db)
            material_identity = _validate_bound_material(db, task_id, case, attachments, bound,
                expected_material_identity)
            origin = _seed_review(db, task_id, case, prompt, attachments, bound['evidence_ids'], settings,
                read_state=bound.get('programmatic_read_state'))
            executor = TaskExecutor(settings, db, factory, resources.pdf_factory)
            executor.retrieval = resources.retrieval_factory(db, settings)
            executor.execute(task_id, 'reflection-' + execution_kind)
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        # 最后一个动作失败后可能直接 fallback，没有下一次 send；收尾也须停批。
        failed_action_stop()
        view = TaskService(db, settings).view(db.get_task(task_id)).model_dump(mode='json') if task_id else None
        actions = db.list_actions(task_id) if task_id else []
        snapshot = ReflectionService(db).snapshot(task_id) if task_id else {}
        rows = db.fetchall('SELECT m.* FROM model_calls m JOIN tasks t ON t.task_id=m.task_id '
            'WHERE t.task_id=? OR t.root_task_id=? ORDER BY m.created_at,m.call_id', (task_id, task_id)) if task_id else []
    audit = inspect_delivery(case['model_input']['draft'], snapshot, actions, (view or {}).get('result'))
    usage_audit = []
    for receipt in rows:
        raw = json.loads(receipt.get('response_json') or '{}')
        amount = _strict_actual_usage(raw.get('usage'))
        usage_audit.append({'call_id': receipt['call_id'], 'task_id': receipt['task_id'],
            'sent_state': receipt['sent_state'],
            'usage_status': 'not_sent' if receipt['sent_state'] == 'not_sent' else 'known' if amount is not None else 'unknown',
            'actual_tokens': amount})
    unknown_count = sum(item['usage_status'] == 'unknown' for item in usage_audit)
    known_tokens = sum(item['actual_tokens'] for item in usage_audit if item['usage_status'] == 'known')
    strict_usage_summary = {'known_tokens_subtotal': known_tokens,
        'total_tokens': None if unknown_count else known_tokens, 'unknown_receipts': unknown_count,
        'aggregation_source': 'strict_usage_audit_only',
        'raw_view_usage_is_not_qualified_for_billing': True}
    report = {'execution_kind': execution_kind,
        'real_model_calls': sum(r['sent_state'] != 'not_sent' for r in rows) if execution_kind == 'real' else 0,
        'case_id': case['case_id'],
        'created': bool(task_id), 'task_id': task_id, 'view': view, 'actions': actions,
        'reflection_snapshot': snapshot, 'programmatic_review_origin': origin,
        'physical_receipts': rows, 'strict_usage_audit': usage_audit, 'strict_usage_summary': strict_usage_summary,
        'bound_material_identity': material_identity, 'delivery_audit': audit, 'error': error,
        'first_failed_action': failures.first_failed_action,
        **failures.audit(),
        'global_stop': gate['stop'] or failures.global_stop or stop_state() or ('case_exception:' + error['type'] if error else None),
        'qualification_scope': ('synthetic_worker_chain_only_not_real_semantic_validation'
                                if execution_kind == 'substitute' else 'frozen_authorized_execution_not_semantic_score')}
    with (root/'result.json').open('x') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    return report


def run_batch(case_specs, run_root, *, settings_factory, provider_factory, resource_factory,
              stop_state=lambda: None, execution_kind='substitute'):
    """顺序执行固定案例；首错后的案例不创建任务或目录。"""
    if execution_kind != 'substitute':
        raise PermissionError('REFLECTION_REAL_EXECUTION_NOT_QUALIFIED')
    identities = []
    for spec in case_specs:
        case = load_case(spec['case_path'], spec['expected_input_sha256'])
        if spec['case_id'] != case['case_id'] or case['case_id'] in identities:
            raise ValueError('REFLECTION_BATCH_CASE_ID_MISMATCH_OR_DUPLICATE')
        identities.append(case['case_id'])
    if not identities:
        raise ValueError('REFLECTION_BATCH_EMPTY')
    results, stopped = [], stop_state()
    for spec in case_specs:
        if stopped:
            results.append({'created': False, 'case_id': spec['case_id'], 'uncreated_reason': stopped})
            continue
        root = Path(run_root)/spec['case_id']
        result = execute_case(spec['case_path'], root, expected_input_sha256=spec['expected_input_sha256'],
            settings=settings_factory(root), provider_factory=provider_factory,
            resource_factory=resource_factory, stop_state=stop_state,
            expected_material_identity=spec.get('expected_material_identity'))
        results.append(result)
        stopped = result['global_stop']
    return {'execution_kind': 'substitute', 'real_model_calls': 0, 'results': results,
            'created': sum(r['created'] for r in results), 'global_stop': stopped}
