"""枚举合同下的单格入口：去重、Supervisor提示词或阶段交接对照。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import traceback
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_ab_experiment import PAIRED_KINDS


def read(path):
    """读取已明确指定的私有冻结文件。"""
    return json.loads(Path(path).read_text())


def write(path, value):
    """仅写本格目录，不把配置凭据输出。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def qualification_configuration(manifest):
    """资格收据绑定本批源码、材料入口、资源、参数与允许矩阵。"""
    from harness_ab_experiment import PAIRED_KINDS, source_contract_key
    result = {key: manifest.get(key) for key in ('experiment_kind', 'runtime_root', 'source_roots',
        'cases_path', 'material_project_root', 'vector_manifest_path', 'vector_endpoint', 'vector_port',
        'model_manifest_path', 'settings', 'official_ip_addresses', 'variant_settings', 'github_commit',
        'ledger_limits', 'carried_budget_source', 'request_contracts', 'qualification_cells')}
    if manifest.get('experiment_kind') in PAIRED_KINDS:
        result.update({key: manifest.get(key) for key in (
            source_contract_key(manifest['experiment_kind']), 'semantic_gate_root', 'pair_review_policy', 'execution_order')})
    if manifest.get('experiment_kind') == 'prompt_bundle_v1':
        result['loaded_prompt_contract'] = manifest.get('loaded_prompt_contract')
        result['batch_id'] = manifest.get('batch_id')
    if 'final_suite_policy' in manifest:
        result.update({key:manifest.get(key) for key in ('final_suite_policy','experiment_root',
            'prepared_input_root','material_snapshot_sha256','final_cases_sha256','final_plan_sha256')})
    return result

def select_variant_source(manifest, variant):
    """先选择目标包再导入传输，防止控制器的research_loop污染另一版。"""
    source = Path(manifest['source_roots'][variant]).resolve(strict=True)
    backend = source / 'backend'
    for name, module in list(sys.modules.items()):
        if name == 'research_loop' or name.startswith('research_loop.'):
            filename = getattr(module, '__file__', None)
            if filename and not Path(filename).resolve().is_relative_to(backend):
                raise PermissionError('EXPERIMENT_BUSINESS_PACKAGE_ALREADY_LOADED_FROM_OTHER_SOURCE')
    sys.path[:0] = [str(backend), str(backend / 'tests')]
    return source


def verify_loaded_supervisor(manifest, variant):
    """对照磁盘摘要检查实际已加载的角色常量，而非只检查sys.path。"""
    from harness_ab_experiment import HANDOFF_KIND, HANDOFF_FILES, PAIRED_KINDS, paired_source_contract, sha256
    if manifest.get('experiment_kind') not in PAIRED_KINDS:
        return None
    if manifest.get('experiment_kind') == 'prompt_bundle_v1':
        from prompt_bundle_ab import verify_loaded
        return verify_loaded(manifest, variant)
    from research_loop.agents import supervisor
    root = Path(manifest['source_roots'][variant]).resolve()
    expected = paired_source_contract(manifest)['variants'][variant]
    actual_file = Path(supervisor.__file__).resolve()
    if (actual_file != root / 'backend/research_loop/agents/supervisor.py'
            or sha256(supervisor.SYSTEM_PROMPT) != expected['system_prompt_sha256']):
        raise PermissionError('SUPERVISOR_LOADED_SOURCE_OR_PROMPT_MISMATCH')
    result = {'file': str(actual_file), 'system_prompt_sha256': expected['system_prompt_sha256']}
    if manifest.get('experiment_kind') == HANDOFF_KIND:
        from research_loop.agents import researcher
        from research_loop.services import research_progress
        loaded = {name: module for name, module in (
            ('backend/research_loop/agents/researcher.py', researcher),
            ('backend/research_loop/services/research_progress.py', research_progress))}
        if set(loaded) != HANDOFF_FILES or any(Path(module.__file__).resolve() != root / name
                for name, module in loaded.items()):
            raise PermissionError('HANDOFF_LOADED_BUSINESS_SOURCE_MISMATCH')
        result['business_module_files'] = {name: str(Path(module.__file__).resolve()) for name, module in loaded.items()}
    return result


def validate_active_variant(manifest, variant):
    """每次预留/发送前检查磁盘与内存角色身份，漂移在远端调用前拒绝。"""
    from harness_ab_experiment import PAIRED_KINDS, validate_paired_sources
    if manifest.get('experiment_kind') in PAIRED_KINDS:
        if manifest.get('experiment_kind') == 'prompt_bundle_v1':
            from prompt_bundle_ab import validate_activation
            validate_activation(manifest, 'execute')
        validate_paired_sources(manifest)
        return verify_loaded_supervisor(manifest, variant)
    return None


def validate_carried_budget_source(manifest):
    """从不可变的旧批次收据计算carry，防止初始冻结时漏填或归零。"""
    source = manifest.get('carried_budget_source')
    if not isinstance(source, dict) or set(source) != {
            'batch_results_path', 'batch_results_sha256', 'ledger_final_path', 'ledger_final_sha256'}:
        raise PermissionError('DEDUP_CARRIED_BUDGET_SOURCE_REQUIRED')
    try:
        values = {}
        for name in ('batch_results', 'ledger_final'):
            path = Path(source[name + '_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != source[name + '_sha256']:
                raise ValueError('old receipt hash changed')
            values[name] = read(path)
        batch, ledger = values['batch_results'], values['ledger_final']
        if any(row['status'] not in {'settled', 'not_sent'} for row in ledger['reservations']):
            raise ValueError('old batch has unresolved requests')
        settled = [row for row in ledger['reservations'] if row['status'] == 'settled']
        if any(type(row[key]) is not int or row[key] < 0 for row in settled
                for key in ('actual_tokens', 'actual_nano')):
            raise ValueError('invalid historical settled amounts')
        if (len(settled) != ledger['physical_sent_requests']
                or sum(row['actual_tokens'] for row in settled) != ledger['actual_known_tokens']
                or sum(row['actual_nano'] for row in settled) != ledger['actual_known_estimated_nano_yuan']):
            raise ValueError('historical details disagree with totals')
        for key in ('actual_known_tokens', 'physical_requests', 'actual_known_estimated_nano_yuan'):
            other = 'physical_sent_requests' if key == 'physical_requests' else key
            if batch[key] != ledger[other]:
                raise ValueError('old batch and ledger disagree')
        totals = ledger['charged_or_reserved']
        prior = ledger.get('carried_usage') or {}
        if any(type(value) is not int or value < 0 for value in prior.values()):
            raise ValueError('invalid earlier carry')
        cumulative = {'tokens': ledger['actual_known_tokens'] + prior.get('tokens', 0),
            'requests': ledger['physical_sent_requests'] + prior.get('requests', 0),
            'nano': ledger['actual_known_estimated_nano_yuan'] + prior.get('nano_yuan', 0)}
        if totals != cumulative:
            raise ValueError('historical cumulative usage disagrees')
        seconds = Decimal(str(batch['execution_seconds']))
        if not seconds.is_finite() or seconds < 0:
            raise ValueError('invalid old elapsed seconds')
        expected = {'tokens': totals['tokens'], 'requests': totals['requests'],
            'nano_yuan': totals['nano'],
            'seconds': int(seconds.to_integral_value(rounding=ROUND_CEILING)) + prior.get('seconds', 0)}
        actual = manifest['ledger_limits'].get('carried_usage')
        if (not isinstance(actual, dict) or any(type(v) is not int or v < 0 for v in actual.values())
                or actual != expected):
            raise ValueError('carry missing or differs from historical receipts')
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise PermissionError('DEDUP_CARRIED_BUDGET_SOURCE_INVALID') from exc
    return expected


def validate_matrix_delivery_witness(db, root_id, payload, model):
    """用本格根Supervisor的原生计划和唯一在途收据证明矩阵请求身份。"""
    from research_loop.agents.graph import matrix_thinking
    task = db.get_task(root_id) or {}
    state = db.get_checkpoint(root_id) or {}
    if (task.get('role') != 'supervisor' or task.get('parent_task_id')
            or task.get('root_task_id') != root_id or state.get('task_id') != root_id
            or matrix_thinking(state) != {'type': 'disabled'}):
        return False
    streaming = bool(db.fetchone('SELECT conversation_id FROM conversation_links WHERE task_id=?', (root_id,)))
    matched = 0
    for row in db.fetchall("SELECT request_json FROM model_calls WHERE task_id=? "
            "AND status='started' AND sent_state='sent'", (root_id,)):
        request = json.loads(row['request_json'])
        if request.get('stage') or request.get('thinking') != {'type': 'disabled'}:
            continue
        expected = {'model': model, 'messages': request['messages'], 'temperature': 0.1}
        for key in ('max_tokens', 'thinking'):
            if key in request:
                expected[key] = request[key]
        if streaming:
            expected.update(stream=True, stream_options={'include_usage': True})
        if request.get('tools') and not request.get('force_text'):
            expected.update(tools=request['tools'], tool_choice='auto')
        if json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
                expected, sort_keys=True, separators=(',', ':'), allow_nan=False):
            matched += 1
    return matched == 1


def validate_manifest(manifest, case_id, variant, mode):
    """在创建目录或任务前检查冻结范围，拒绝借配置更换第二个实验变量。"""
    from harness_ab_transport import validate_request_contracts
    if 'final_suite_policy' in manifest:
        from agent_final_runtime import validate_suite_manifest
        validate_suite_manifest(manifest)
    if manifest.get('experiment_kind') == 'prompt_bundle_v1':
        from prompt_bundle_ab import validate_activation
        validate_activation(manifest, mode)
    contracts = manifest.get('request_contracts')
    if contracts is not None or mode == 'execute' or 'final_suite_policy' in manifest:
        try:
            validate_request_contracts(contracts, final_suite='final_suite_policy' in manifest)
        except ValueError as exc:
            raise PermissionError('DEDUP_COMPONENT_REQUEST_CONTRACT_REQUIRED') from exc
    if mode == 'execute' or manifest.get('carried_budget_source') is not None:
        validate_carried_budget_source(manifest)
    from harness_ab_experiment import PAIRED_KINDS, validate_paired_manifest
    kind = manifest.get('experiment_kind')
    if kind not in {'read_paper_exact_duplicate', *PAIRED_KINDS}:
        raise PermissionError('DEDUP_EXPERIMENT_KIND_REQUIRED')
    if variant not in {'A', 'B'} or [case_id, variant] not in manifest['ledger_limits']['allowed_cells']:
        raise PermissionError('DEDUP_CELL_OUTSIDE_MATRIX')
    if kind in PAIRED_KINDS:
        validate_paired_manifest(manifest)
    else:
        if Path(manifest['source_roots']['A']).resolve() != Path(manifest['source_roots']['B']).resolve():
            raise PermissionError('DEDUP_SHARED_SOURCE_REQUIRED')
        if manifest.get('variant_settings') != {'A': {'context_deduplicate_read_paper': False},
                'B': {'context_deduplicate_read_paper': True}}:
            raise PermissionError('DEDUP_ONLY_SWITCH_MAY_DIFFER')
    settings = manifest['settings']
    if contracts is not None and (settings.get('deepseek_model') != 'deepseek-flash'
            or settings.get('deepseek_base_url') != 'https://api.deepseek.com'
            or settings.get('request_timeout_seconds') != 120):
        raise PermissionError('DEDUP_VISUAL_PROVIDER_SETTINGS_REQUIRED')
    if settings.get('retrieval_route') != 'located_fp16' or settings.get('retrieval_reranker_backend') != 'mps_fp16':
        raise PermissionError('DEDUP_CURRENT_RAG_REQUIRED')
    if mode == 'execute' and (manifest.get('real_execution_released') is not True
            or not manifest.get('frozen_files') or not manifest.get('github_commit')
            or not manifest.get('qualification_receipt') or not manifest.get('cell_absolute_deadline')):
        raise PermissionError('DEDUP_EXECUTION_QUALIFICATION_REQUIRED')
    if mode == 'execute':
        receipt = read(manifest['qualification_receipt'])
        expected = {tuple(cell) for cell in manifest['qualification_cells']}
        actual = {(row['case_id'], row['variant']) for row in receipt.get('rows', [])
                  if row.get('status') == 'passed' and row.get('real_model_calls') == 0}
        if (receipt.get('status') != 'passed' or receipt.get('real_model_calls') != 0
                or receipt.get('configuration') != qualification_configuration(manifest)
                or not expected or actual != expected):
            raise PermissionError('DEDUP_QUALIFICATION_RECEIPT_MISMATCH')


def create_bundle_input_task(client, db, case, variant, mode, bundles):
    """通过真实后端API建立本格输入；种子与权限不由评分标签推导。"""
    from prompt_bundle_cases import case_save_policy, seed_history, upload_materials, validate_preconditions
    case_id = case['case_id']
    preconditions = validate_preconditions(case)
    report = {}
    upload_case = case
    if mode == 'qualify' and preconditions.get('catalog'):
        upload_case = {**case, 'preconditions': {**preconditions, 'catalog': []}}
        report['qualification_only_attachment_override'] = True
    attachments = upload_materials(client, upload_case, bundles)
    conversation = client.post('/api/conversations', json={'title': 'Prompt bundle independent task',
        'idempotency_key': case_id + '-' + variant + '-' + mode + '-conversation'})
    assert conversation.status_code == 201, conversation.text
    conversation_id = conversation.json()['conversation_id']
    empty = client.get('/api/conversations/' + conversation_id).json()
    assert not empty.get('messages'), empty
    seed_history(db, conversation_id, preconditions, suite_kind=case.get('suite_kind'))
    created = client.post('/api/tasks', json={'prompt': case['prompt'],
        'save_policy': case_save_policy(case), 'attachments': attachments, 'conversation_id': conversation_id,
        'idempotency_key': case_id + '-' + variant + '-' + mode})
    assert created.status_code == 202, created.text
    report.update(root_id=created.json()['task_id'], conversation_id=conversation_id,
        synthetic_history_messages=len(preconditions['history_messages']),
        initial_messages=len(preconditions['history_messages']),
        http_upload=201 if attachments else None, upload_count=len(attachments), http_create=202)
    return report, attachments


def run_cell(manifest, case_id, variant, mode, credential=None):
    """资格通过后才允许在官方传输边界构造真实请求。

    Args:
        manifest: 已核对的实验配置与来源清单。
        case_id: 四个冻结问题之一。
        variant: A 或 B，选择冻结业务源码。
        mode: qualify 不调用生成模型；execute 只执行一次本格。
        credential: 真实模式由父进程内存管道注入，不写入工件。
    """
    source = select_variant_source(manifest, variant)
    validate_manifest(manifest, case_id, variant, mode)
    if mode == 'execute' and manifest.get('experiment_kind') in PAIRED_KINDS:
        from harness_ab_pair_gate import validate_prior_pair_gates
        validate_prior_pair_gates(manifest, case_id, variant)
    allowed_cells = manifest.get('ledger_limits', {}).get('allowed_cells')
    if allowed_cells is not None and [case_id, variant] not in allowed_cells:
        raise PermissionError('H1 cell outside remaining authorized matrix')
    if mode == 'execute' and (manifest.get('real_execution_released') is not True
            or not manifest.get('frozen_files') or not manifest.get('github_commit')
            or not manifest.get('qualification_receipt')):
        raise PermissionError('H1 real execution freeze and qualification absent')
    runtime = Path(manifest['runtime_root']).resolve()
    case = next(c for c in read(manifest['cases_path'])['cases'] if c['case_id'] == case_id)
    joint = manifest.get('experiment_kind') == 'prompt_bundle_v1'
    if joint:
        from prompt_bundle_cases import case_materials, case_save_policy, validate_preconditions
        materials = case_materials(case)
        preconditions = validate_preconditions(case)
    else:
        materials, preconditions = [case['material']], {}
    stage = manifest.get('qualification_run', 'qualification') if mode == 'qualify' else 'cells'
    cell = runtime / stage / (case_id + '-' + variant)
    if 'final_suite_policy' in manifest:
        from agent_final_runtime import cell_directory
        cell = cell_directory(manifest,case_id,variant,mode)
    cell.mkdir(parents=True, exist_ok=False)
    temp = cell / 'tmp'
    temp.mkdir()
    tempfile.tempdir = str(temp)
    os.chdir(cell)
    sys.path[:0] = [str(source / 'backend'), str(source / 'backend/tests')]
    # 在被测试版本导入前确认只存在明确的实验路径。
    for entry in manifest['frozen_files']:
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('H1 frozen source or material changed: ' + entry['path'])
    import httpx
    from harness_ab_boundary import Boundary, IsolationError
    from harness_ab_budget import Ledger
    from harness_ab_resources import (load_source_bundles, rebind_bundle, prepare_parsed_cache,
        install_bundle, HarnessRetrievalRuntime)
    from harness_ab_transport import (BudgetTransport, bind_deep_reading_client,
        require_generation_identity)
    from harness_ab_network import OfficialNetworkPolicy
    network_policy = OfficialNetworkPolicy(mode, manifest['vector_port'], manifest['official_ip_addresses'])
    denied_connections = []

    class ExperimentBoundary(Boundary):
        """文件写入限定本次运行根，网络仅许可冻结官方 IP 与专属向量端口。"""
        def audit(self, event, values):
            if not self.enabled:
                return
            if event == 'socket.getaddrinfo':
                host, port = values[:2]
                if not network_policy.allow_dns(host, port):
                    self.reject('network_name_outside_experiment')
                return
            if event == 'socket.connect':
                address = values[1]
                if not network_policy.allow_connect(address):
                    denied_connections.append({'address': address, 'phase': self.phase,
                        'at': datetime.now(timezone.utc).isoformat(), 'audit_before_connect': True})
                    self.reject('network_destination_outside_experiment')
                return
            super().audit(event, values)

    if 'final_suite_policy' in manifest:
        from agent_final_runtime import CellBoundary
        class FinalExperimentBoundary(ExperimentBoundary, CellBoundary):
            """保持共同网络策略，并将写入边界收紧为当前格。"""
            pass
        boundary = FinalExperimentBoundary(cell, runtime)
    else:
        boundary = ExperimentBoundary(runtime)
    sys.addaudithook(boundary.audit)
    clients = []
    ledger = db = root_id = None
    report = {'case_id': case_id, 'variant': variant, 'mode': mode, 'status': 'starting',
        'started_at': datetime.now(timezone.utc).isoformat(), 'real_model_calls': 0}
    try:
        with ExitStack() as stack:
            original_resolver = socket.getaddrinfo
            dns_log_lock = threading.RLock()

            def guarded_resolver(host, port, *args, **kwargs):
                """只按当次官方DNS回答授予连接IP；HTTP端点及TLS校验仍由原传输保持。"""
                if not network_policy.allow_dns(host, port):
                    boundary.reject('network_name_outside_experiment')
                try:
                    answers = network_policy.resolve(original_resolver, host, port, *args, **kwargs)
                except PermissionError:
                    boundary.reject('official_dns_answer_outside_scope')
                with dns_log_lock:
                    write(cell / 'network-dns.json', network_policy.snapshot())
                return answers

            stack.enter_context(patch.object(socket, 'getaddrinfo', guarded_resolver))
            stack.enter_context(patch.dict(os.environ, {'PYTHONDONTWRITEBYTECODE': '1',
                'RESEARCHLOOP_READONLY': '0', 'TMPDIR': str(temp), 'HOME': str(temp / 'home'),
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'HF_HOME': str(temp / 'huggingface'), 'XDG_CACHE_HOME': str(temp / 'cache'),
                'TOKENIZERS_PARALLELISM': 'false', 'PYTORCH_ENABLE_MPS_FALLBACK': '0'}, clear=True))
            stack.enter_context(patch.object(socket, 'has_ipv6', False))
            report['loaded_supervisor'] = verify_loaded_supervisor(manifest, variant)
            from research_loop.config import Settings
            from research_loop import config
            values = dict(manifest['settings'])
            variant_values = manifest['variant_settings'][variant]
            values.update(variant_values)
            values.update(project_root=cell, data_dir=cell / 'data', output_dir=cell / 'output',
                database_url='sqlite:///' + str(cell / 'task.sqlite3'), deepseek_api_key=credential,
                qdrant_url=manifest['vector_endpoint'],
                retrieval_namespace='researchloop_h1_' + stage.replace('-', '_') + '_' + case_id.lower().replace('-', '_') + '_' + variant.lower(),
                retrieval_model_manifest=Path(manifest['model_manifest_path']))
            settings = Settings(**values)
            original_init = Settings.__init__

            def guarded_init(self, **kwargs):
                """隐式配置也必须落在本格，不接受部署数据库或向量服务。"""
                original_init(self, **kwargs)
                if any(not getattr(self, key).resolve().is_relative_to(cell) for key in ('project_root', 'data_dir', 'output_dir')):
                    raise IsolationError('H1 implicit settings path outside cell')
                if self.database_url != settings.database_url or self.qdrant_url != settings.qdrant_url:
                    raise IsolationError('H1 implicit database or vector endpoint differs')

            stack.enter_context(patch.object(Settings, '__init__', guarded_init))
            stack.enter_context(patch.object(Settings, 'from_env', side_effect=IsolationError('H1 default settings denied')))
            stack.enter_context(patch.object(config, '_load_dotenv', side_effect=IsolationError('H1 dotenv denied')))
            stack.enter_context(patch.object(config, 'get_settings', return_value=settings))
            boundary.phase = 'guard_selfcheck'
            checks = []
            probes = [('dotenv', lambda: (source / '.env').read_text()),
                ('outside_write', lambda: (source / 'h1-forbidden.tmp').write_text('blocked')),
                ('default_settings', lambda: Settings.from_env()),
                ('deployment_network', lambda: socket.create_connection(('127.0.0.1', 6339)))]
            for name, probe in probes:
                try:
                    probe()
                except IsolationError:
                    checks.append({'name': name, 'blocked_before_operation': True})
                else:
                    raise AssertionError('H1 isolation selfcheck failed: ' + name)
            report['guard_selfchecks'] = checks
            boundary.phase = 'qualification' if mode == 'qualify' else 'execution'
            from research_loop import main
            from research_loop.api import dependencies, health
            for module in (main, dependencies, health):
                stack.enter_context(patch.object(module, 'get_settings', return_value=settings))
            from research_loop.providers.discovery import DiscoveryProvider
            if not joint:
                stack.enter_context(patch.object(DiscoveryProvider, 'search', side_effect=PermissionError('H1 new corpus search outside frozen task')))
            from research_loop.storage.mysql import Database
            from research_loop.storage.files import FileStore
            from research_loop.providers.pdf import PDFProvider
            from research_loop.services.papers import PaperService
            from research_loop.services.tasks import TaskService
            from research_loop.workers.executor import TaskExecutor
            from research_loop.providers.deepseek import DeepSeekProvider
            from fastapi.testclient import TestClient

            db = Database(settings.database_url)
            db.ensure_schema()
            files = FileStore(settings.data_dir, settings.output_dir, settings.project_root)
            bundles = (load_source_bundles(manifest['material_project_root'], manifest['vector_manifest_path'],
                source_sha256s={item['pdf_sha256'] for item in materials}) if materials else {})
            originals = [bundles[item['pdf_sha256']] for item in materials]
            rebound = [rebind_bundle(item, settings.retrieval_namespace) for item in originals]
            report['indexes'] = [install_bundle(db, settings, item) for item in rebound]
            report['copied_caches'] = [prepare_parsed_cache(files, item['document_ref']) for item in originals]
            if not joint:
                report['index'] = report['indexes'][0]
                report['copied_cache'] = report['copied_caches'][0]
            pdf = PDFProvider(files, parser_name=settings.parser_name, parser_version=settings.parser_version,
                parser_config=settings.parser_config)
            stack.callback(pdf.close)
            if joint:
                from prompt_bundle_cases import install_catalog, seed_history, upload_materials
                install_catalog(stack, DiscoveryProvider, case, bundles)
            report['settings'] = {k: v for k, v in settings.model_dump(mode='json').items() if not any(
                token in k for token in ('api_key', 'supabase_key'))}
            if materials:
                retrieval = HarnessRetrievalRuntime(db, settings, rebound)
                from research_loop.providers.embeddings import LocalEmbedding
                from research_loop.providers.mps_reranker import MpsReranker
                from research_loop.services.retrieval import RetrievalService
                from research_loop.storage.vector_index import VectorIndex
                embedding = LocalEmbedding(settings.retrieval_model_manifest)
                reranker = MpsReranker(settings.retrieval_model_manifest)
                warm_started = time.monotonic()
                embedding.encode(['research method'], query=True)
                reranker.score('research method', ['A scientific method is described.'])
                vector_client = httpx.Client(base_url=settings.qdrant_url, timeout=30, trust_env=False)
                stack.callback(vector_client.close)
                retrieval.service = RetrievalService(db, embedding,
                    VectorIndex(settings.qdrant_url, settings.retrieval_namespace, vector_client), reranker,
                    route=settings.retrieval_route)
                report['pre_task_warmup_seconds'] = time.monotonic() - warm_started
                report['warmup'] = {'query_encodes': 1, 'rerank_scores': 1,
                    'text': 'fixed neutral fixture; not a test question'}
                report['rag_identity'] = {'route': retrieval.service.route,
                    'embedding': embedding.identity, 'reranker': reranker.identity}
            else:
                class NoPaperRuntime:
                    """零论文Memory格禁止隐式新建语料或调用检索组件。"""
                    def ensure_task_resources(self, *args, **kwargs):
                        raise PermissionError('PROMPT_MEMORY_CASE_HAS_NO_PAPER_RESOURCES')
                    def search(self, *args, **kwargs):
                        raise PermissionError('PROMPT_MEMORY_CASE_HAS_NO_PAPER_RESOURCES')
                    def close(self):
                        pass
                retrieval = NoPaperRuntime()
                report.update(warmup={'query_encodes': 0, 'rerank_scores': 0},
                    rag_identity={'status': 'not_applicable_zero_paper_memory'})

            overrides = dict(main.app.dependency_overrides)
            main.app.dependency_overrides[dependencies.get_runtime_settings] = lambda: settings
            main.app.dependency_overrides[dependencies.get_database] = lambda: db
            stack.callback(lambda: (main.app.dependency_overrides.clear(), main.app.dependency_overrides.update(overrides)))
            with TestClient(main.app) as client:
                if joint:
                    input_report, attachments_for_task = create_bundle_input_task(client, db, case, variant, mode, bundles)
                    report.update(input_report)
                    root_id = report['root_id']
                else:
                    uploaded = client.post('/api/uploads', params={'filename': 'source.pdf'},
                        content=Path(case['material']['pdf_path']).read_bytes())
                    assert uploaded.status_code == 201, uploaded.text
                    attachments_for_task = [{'input_ref': uploaded.json()['input_ref'], 'filename': 'source.pdf',
                        'paper_id': originals[0]['document_ref']['paper_id']}]
                    conversation = client.post('/api/conversations', json={'title': 'H1 independent task',
                        'idempotency_key': case_id + '-' + variant + '-' + mode + '-conversation'})
                    assert conversation.status_code == 201, conversation.text
                    conversation_id = conversation.json()['conversation_id']
                    empty = client.get('/api/conversations/' + conversation_id).json()
                    assert not empty.get('messages'), empty
                    created = client.post('/api/tasks', json={'prompt': case['prompt'], 'save_policy': 'deny',
                        'attachments': attachments_for_task, 'conversation_id': conversation_id,
                        'idempotency_key': case_id + '-' + variant + '-' + mode})
                    assert created.status_code == 202, created.text
                    root_id = created.json()['task_id']
                    report.update(root_id=root_id, conversation_id=conversation_id, initial_messages=0,
                        http_upload=201, upload_count=1, http_create=202)
                if mode == 'qualify':
                    attachments = {a['input_ref']: a for a in json.loads(db.get_task(root_id)['attachments_json'])}
                    qualified = []
                    for material, attachment, bundle in zip(materials, attachments_for_task, rebound, strict=True):
                        data = PaperService(db, None, pdf).read(root_id, attachment['input_ref'], attachments=attachments)
                        assert len(data['pages']) == material['physical_pages']
                        resources = retrieval.ensure_task_resources(root_id, [bundle['manifest']['source']['document_id']])
                        from research_loop.domain.retrieval import RetrievalQuery
                        found = retrieval.search(root_id, RetrievalQuery(query='research method',
                            document_ids=[bundle['manifest']['source']['document_id']]))
                        if (found.get('status') != 'ready' or not found.get('hits')
                                or found.get('strategy', {}).get('name') != 'located_fp16' or found.get('degradations')
                                or found.get('effective_path') != 'located_bm25_rerank'
                                or found.get('reranker', {}).get('variant') != 'mps_fp16'):
                            raise ValueError('DEDUP_NATIVE_FP16_QUALIFICATION_FAILED')
                        qualified.append({'pdf_sha256': material['pdf_sha256'], 'pages': len(data['pages']),
                            'authorized_resources': resources, 'retrieval_status': found['status'], 'hits': len(found['hits']),
                            'retrieval': {key: found.get(key) for key in ('strategy', 'effective_path', 'reranker',
                                'embedding', 'selection_trace', 'degradations')}})
                    report.update(status='passed', real_model_calls=0, qualification_materials=qualified,
                        qualification={'materials': len(qualified), 'rag_required': bool(materials)})
                    if not joint:
                        report['qualification_retrieval'] = qualified[0]['retrieval']
                        report['qualification'] = {key: qualified[0][key] for key in
                            ('pages', 'authorized_resources', 'retrieval_status', 'hits')}
                else:
                    if not credential or read(manifest['authorization_path'])['approved'] is not True:
                        raise PermissionError('H1 execution authorization or credential absent')
                    ledger = Ledger(runtime / 'budget.sqlite3', **manifest.get('ledger_limits', {}))
                    begun = ledger.begin_root(root_id, case_id, variant)
                    snap = ledger.snapshot()
                    deadline = min(begun['started_at'] + ledger.config['root_seconds'],
                        snap['batch_absolute_deadline'], manifest['cell_absolute_deadline'])

                    class SourceBoundTransport(BudgetTransport):
                        """共享传输保留原预算合同，增加新提示词实验的发送前身份门。"""

                        def handle_request(self, request):
                            require_generation_identity(ledger, root_id,
                                lambda: validate_active_variant(manifest, variant),
                                operation='source_identity')
                            return super().handle_request(request)

                    def factory(_settings):
                        """所有角色共用根账本；原生适配器仅替换 HTTP 传输。"""
                        def build():
                            if (_settings.deepseek_model != settings.deepseek_model
                                    or _settings.deepseek_base_url != settings.deepseek_base_url):
                                from research_loop.providers.deepseek import RequestNotSent
                                raise RequestNotSent('H1 model or visual provider outside frozen scope')
                            transport = SourceBoundTransport(ledger, root_id, cell / 'requests', deadline=deadline,
                                visual_contract=manifest['request_contracts']['visual'],
                                deep_reading_contract=manifest['request_contracts'].get('deep_reading'),
                                matrix_contract=manifest['request_contracts'].get('matrix_delivery'),
                                matrix_request_check=lambda payload: validate_matrix_delivery_witness(
                                    db, root_id, payload, settings.deepseek_model))
                            native_client = httpx.Client(transport=transport, timeout=settings.request_timeout_seconds,
                                trust_env=False, follow_redirects=False)
                            clients.append(native_client)
                            return DeepSeekProvider(credential, settings.deepseek_model, settings.deepseek_base_url,
                                timeout=settings.request_timeout_seconds, client=native_client)
                        return require_generation_identity(ledger, root_id, build, operation='provider_identity')

                    executor = TaskExecutor(settings, db, provider_factory=factory, pdf_factory=lambda *_: pdf)
                    executor.retrieval = retrieval
                    stop_watchdog = threading.Event()

                    def deadline_watchdog():
                        """到期只取消本格任务；不启动额外收尾模型请求。"""
                        if not stop_watchdog.wait(max(0, deadline - time.time())):
                            TaskService(db, settings).cancel(root_id)

                    watchdog = threading.Thread(target=deadline_watchdog, daemon=True)
                    watchdog.start()
                    try:
                        from research_loop.agents.deep_reading import _ReceiptClient
                        with bind_deep_reading_client(_ReceiptClient):
                            executor.execute(root_id, 'harness-h1-' + case_id + '-' + variant)
                    finally:
                        stop_watchdog.set()
                        watchdog.join(timeout=1)
                    view = client.get('/api/tasks/' + root_id)
                    events = client.get('/api/tasks/' + root_id + '/events')
                    subtasks = client.get('/api/tasks/' + root_id + '/subtasks')
                    assert view.status_code == events.status_code == subtasks.status_code == 200
                    write(cell / 'task-view.json', view.json())
                    write(cell / 'events.json', events.json())
                    write(cell / 'subtasks.json', subtasks.json())
                    task_ids = [root_id] + [r['task_id'] for r in db.fetchall('SELECT task_id FROM tasks WHERE root_task_id=? AND task_id<>?', (root_id, root_id))]
                    write(cell / 'actions.json', {tid: db.list_actions(tid) for tid in task_ids})
                    receipts = db.fetchall('SELECT * FROM model_calls')
                    write(cell / 'model-calls.json', receipts)
                    write(cell / 'ledger-snapshot.json', ledger.snapshot())
                    report.update(status='executed', task_status=view.json()['status'],
                        task_ids=task_ids,
                        ledger_halt=ledger.snapshot()['halt_reason'])
    except BaseException as exc:
        report.update(status='failed', exception_type=type(exc).__name__)
        (cell / 'failure.log').write_text(traceback.format_exc())
    finally:
        close_errors = []
        for client in clients:
            try:
                client.close()
            except BaseException as exc:
                close_errors.append(type(exc).__name__)
        if close_errors:
            report['client_close_errors'] = close_errors
        if ledger is not None:
            snapshot = ledger.snapshot()
            write(cell / 'ledger-snapshot.json', snapshot)
            report['real_model_calls'] = sum(r['root_id'] == root_id and r['status'] in ('sent', 'settled', 'unknown')
                for r in snapshot['reservations'])
            report['ledger_halt'] = snapshot['halt_reason']
        if db is not None:
            write(cell / 'model-calls.json', db.fetchall('SELECT * FROM model_calls'))
            if root_id is not None:
                task_ids = [r['task_id'] for r in db.fetchall('SELECT task_id FROM tasks WHERE root_task_id=?', (root_id,))]
                write(cell / 'actions.json', {tid: db.list_actions(tid) for tid in task_ids})
                write(cell / 'root-row.json', db.get_task(root_id))
        report['isolation'] = {'write_root': str(runtime), 'denials': boundary.denials,
            'sqlite_paths': sorted(boundary.sqlite_paths), 'implicit_settings_and_dotenv_denied': True,
            'network_policy': network_policy.snapshot(), 'denied_connections': denied_connections}
        if any(row['phase'] != 'guard_selfcheck' for row in boundary.denials):
            report['status'] = 'failed'
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write(cell / 'result.json', report)
        boundary.enabled = False
    return report


def main():
    """一次只运行已冻结的一个格；正式授权从内存标准输入读取。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--variant', required=True, choices=['A', 'B'])
    parser.add_argument('--mode', required=True, choices=['qualify', 'execute'])
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--qualification-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = read(args.manifest)
    if args.deadline is not None:
        manifest['cell_absolute_deadline'] = args.deadline
    if manifest.get('experiment_kind') == 'prompt_bundle_v1' and args.mode == 'qualify' and not args.qualification_child:
        # 资格格同样由所属进程组硬截止，不依赖模型账本才计算运行时间。
        from prompt_bundle_ab import validate_activation
        from launch_read_paper_ab import run_guarded
        auth = validate_activation(manifest, 'qualify')
        validate_manifest(manifest, args.case, args.variant, 'qualify')
        deadline = min(auth['round_absolute_deadline'], time.time() + manifest['ledger_limits']['root_seconds'])
        runtime = Path(manifest['runtime_root'])
        runtime.mkdir(parents=True, exist_ok=True)
        log_path = runtime / ('qualification-' + args.case + '-' + args.variant + '.log')
        with log_path.open('x') as log:
            result = run_guarded([sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                '--manifest', str(args.manifest), '--case', args.case, '--variant', args.variant,
                '--mode', 'qualify', '--qualification-child'], '', deadline, log)
        write(runtime / ('qualification-' + args.case + '-' + args.variant + '-launcher.json'), result)
        raise SystemExit(0 if result['status'] == 'exited' and result['returncode'] == 0 else 1)
    credential = json.loads(sys.stdin.read())['api_key'] if args.mode == 'execute' else None
    report = run_cell(manifest, args.case, args.variant, args.mode, credential)
    print(json.dumps({key: report.get(key) for key in ('case_id', 'variant', 'mode', 'status',
        'task_status', 'root_id', 'exception_type')}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report['status'] in {'passed', 'executed'} else 1)


if __name__ == '__main__':
    main()
