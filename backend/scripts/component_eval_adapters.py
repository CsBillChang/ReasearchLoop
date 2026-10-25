"""组件评测薄适配器：只接受显式隔离配置，复用现有排序、核验和上下文服务。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

from research_loop.storage.indexes import rank_terms, reciprocal_rank_fusion, terms
from research_loop.domain.retrieval_strategy import RetrievalStrategy


def validate_isolation(config: dict, *, project_root=None, protected_roots=()) -> dict:
    """在工厂初始化网络或数据库前纯校验显式隔离身份。

    Args:
        config: root（兼容 artifact_root）、database_url、collection_prefix 和可选 vector_url。
        project_root: 当前业务仓库，用于拒绝 data 和其他运行目录。
        protected_roots: 调用方明确保护的数据库、缓存或成果目录。

    Returns:
        规范化配置；本函数不建目录、不连接数据库、不读取环境变量。

    Raises:
        ValueError: 隔离身份缺失、路径重叠、远端地址或生产命名空间。
    """
    value = config.get('root') or config.get('artifact_root')
    if not value or not Path(value).is_absolute():
        raise ValueError('EVAL_ISOLATION_ROOT_REQUIRED')
    supplied_root = Path(value)
    root = supplied_root.resolve()
    if '..' in supplied_root.parts or supplied_root != root:
        raise ValueError('EVAL_ISOLATION_PATH_ALIAS')
    project_root = project_root or Path(__file__).resolve().parents[2]
    if root == Path(root.anchor):
        raise ValueError('EVAL_ISOLATION_ROOT_UNSAFE')
    protected = [Path(p).resolve() for p in protected_roots]
    if project_root:
        project = Path(project_root).resolve()
        protected.extend([project / 'data', project / '.env'])
        if root == project or project.is_relative_to(root):
            raise ValueError('EVAL_ISOLATION_ROOT_OVERLAP')
    if any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError('EVAL_ISOLATION_ROOT_OVERLAP')
    database_url = config.get('database_url', '')
    parsed = urlsplit(database_url)
    if parsed.scheme == 'sqlite':
        if (not database_url.startswith('sqlite:////') or parsed.netloc
                or parsed.query or parsed.fragment):
            raise ValueError('EVAL_SQLITE_ABSOLUTE_REQUIRED')
        supplied_database = Path(unquote(database_url[len('sqlite:///'):]))
        database = supplied_database.resolve()
        if '..' in supplied_database.parts or supplied_database != database:
            raise ValueError('EVAL_DATABASE_PATH_ALIAS')
        if database == root or not database.is_relative_to(root):
            raise ValueError('EVAL_DATABASE_OUTSIDE_ROOT')
    elif parsed.scheme == 'mysql':
        if (parsed.hostname not in {'localhost', '127.0.0.1', '::1'} or parsed.port in {None, 3306, 33316}
                or parsed.query or parsed.fragment or len(parsed.path.lstrip('/')) > 64
                or not re.fullmatch(r'/researchloop_test_component_eval_[a-z0-9_]+', parsed.path)):
            raise ValueError('EVAL_DATABASE_NOT_ISOLATED')
    else:
        raise ValueError('EVAL_DATABASE_REQUIRED')
    prefix = config.get('collection_prefix', '')
    if not re.fullmatch(r'researchloop_eval_[a-z0-9_]+', prefix) or len(prefix.removeprefix('researchloop_')) > 64:
        raise ValueError('EVAL_COLLECTION_NOT_ISOLATED')
    if config.get('vector_url'):
        vector = urlsplit(config['vector_url'])
        if (vector.scheme not in {'http', 'https'} or vector.hostname not in {'localhost', '127.0.0.1', '::1'}
                or vector.port in {None, 6339} or vector.path or vector.username or vector.password
                or vector.query or vector.fragment):
            raise ValueError('EVAL_VECTOR_NOT_LOCAL')
    return {**config, 'root': str(root), 'database_url': database_url, 'collection_prefix': prefix}


class IsolationGuard:
    """保存已校验的隔离配置，供外部显式工厂复用。"""

    def __init__(self, config: dict, *, project_root=None, protected_roots=()):
        """只校验配置，不初始化任何运行组件。"""
        self.config = validate_isolation(config, project_root=project_root, protected_roots=protected_roots)

    def check_collection(self, collection: str) -> None:
        """逐集合拒绝与其他实验或生产共享写入范围。"""
        if not collection.startswith(self.config['collection_prefix'] + '_'):
            raise ValueError('EVAL_COLLECTION_OUTSIDE_NAMESPACE')


def _result(variant: str, path: str, **values) -> dict:
    """保留缺失用量，统一适配器结果外壳而不填造零成本。"""
    return {'requested_variant': variant, 'effective_path': path, 'status': 'executed',
            'usage': None, 'timing': {}, 'degradations': [], 'trace': [], **values}


_IDENTITY = ('chunk_id', 'document_id', 'index_id', 'evidence_id', 'source_sha256',
             'text_sha256', 'page_number', 'char_start', 'char_end')


def _authorized_chunks(chunks: list[dict], scope, frozen_identity=None) -> list[dict]:
    """仅选择冻结授权ID，严格拒绝片段hash、版本和身份冲突。

    scope 为 chunk_id 列表或以 chunk_id 为键的完整身份映射；前者依赖上游
    已校验的输入文件hash，后者额外逐字段比较。无 scope 不退化成全库。
    """
    if not isinstance(scope, (list, tuple, dict)) or len(scope) != len(set(scope)):
        raise ValueError('EVAL_SCOPE_REQUIRED')
    allowed = set(scope)
    expected = frozen_identity if frozen_identity is not None else scope if isinstance(scope, dict) else None
    records, seen = [], set()
    for raw in chunks:
        cid = raw.get('chunk_id')
        if cid in seen:
            raise ValueError('EVAL_DUPLICATE_CHUNK_ID')
        seen.add(cid)
        if cid not in allowed:
            continue
        if any(field not in raw for field in _IDENTITY) or not isinstance(raw.get('text'), str):
            raise ValueError('EVAL_CHUNK_IDENTITY_REQUIRED')
        if (not all(isinstance(raw[field], str) and raw[field] for field in
                    ('chunk_id', 'document_id', 'index_id', 'evidence_id'))
                or any(not re.fullmatch(r'[a-f0-9]{64}', str(raw[field])) for field in ('source_sha256', 'text_sha256'))
                or raw['text_sha256'] != sha256(raw['text'].encode()).hexdigest()
                or type(raw['char_start']) is not int or type(raw['char_end']) is not int
                or raw['char_start'] < 0 or raw['char_end'] - raw['char_start'] != len(raw['text'])):
            raise ValueError('EVAL_CHUNK_HASH_OR_RANGE_INVALID')
        if expected is not None and (cid not in expected or any(raw[k] != expected[cid].get(k) for k in _IDENTITY)):
            raise ValueError('EVAL_FROZEN_IDENTITY_MISMATCH')
        records.append({**deepcopy(raw), 'key': raw.get('key', cid)})
    if allowed != {row['chunk_id'] for row in records} or len({r['key'] for r in records}) != len(records):
        raise ValueError('EVAL_SCOPE_INCOMPLETE_OR_KEY_COLLISION')
    return records


class RetrievalAdapter:
    """仅按指定路线懒装配组件，所有通道使用同一份冻结授权片段。"""

    def __init__(self, embedding_factory=None, vector_factory=None, reranker_factory=None, *, guard=None,
                 strategy=None):
        """注入零参数懒工厂；纯BM25不构造任何编码、向量或重排器。"""
        self.embedding_factory = embedding_factory
        self.vector_factory = vector_factory
        self.reranker_factory = reranker_factory
        self.guard = guard
        if strategy is not None and type(strategy) is not RetrievalStrategy:
            raise ValueError('RETRIEVAL_STRATEGY_INVALID')
        self.strategy = strategy

    def run(self, query: str, chunks: list[dict], variant: str, scope, *, candidate_limit=24,
            frozen_identity=None) -> dict:
        """返回完整最多24名评测排名，不采用生产工具的5/8条截断。

        Args:
            query: 固定原查询。
            chunks: 冻结片段；向量通道另需 collection 和 vector_payload。
            variant: vector、bm25、rrf 或 rrf_rerank。
            scope: 明确授权 chunk_id 列表或完整身份映射。
            candidate_limit: 本轮固定24；不得单路线改变候选量。
            frozen_identity: 可选独立冻结的完整片段身份映射。
        """
        if variant not in {'vector', 'bm25', 'rrf', 'rrf_rerank'} or candidate_limit != 24 or not query.strip():
            raise ValueError('EVAL_RETRIEVAL_VARIANT_OR_LIMIT_INVALID')
        started = perf_counter()
        rows = _authorized_chunks(chunks, scope, frozen_identity)
        by_key = {r['key']: r for r in rows}
        result = _result(variant, variant, ranked_ids=[], scope=list(scope), corpus=[r['chunk_id'] for r in rows])
        if self.strategy is not None:
            result['strategy'] = self.strategy.as_dict()
            result['reranker_configuration'] = {'requested': self.strategy.reranker_kwargs,
                                               'actual': None, 'verified': False}
        timing, channels = result['timing'], {}
        timing['scope_seconds'] = perf_counter() - started
        if variant != 'vector':
            before = perf_counter()
            channels['bm25'] = rank_terms(query, rows, [Counter(terms(r['text'])) for r in rows], 24)
            timing['bm25_seconds'] = perf_counter() - before
            result['trace'].append({'component': 'bm25', 'candidate_ids': [r['chunk_id'] for r in channels['bm25']]})
        if variant != 'bm25' and rows:
            phase = 'initialization'
            before = perf_counter()
            try:
                if not self.embedding_factory or not self.vector_factory:
                    raise ValueError('EVAL_VECTOR_FACTORY_REQUIRED')
                for row in rows:
                    if not row.get('collection') or 'vector_payload' not in row:
                        raise ValueError('EVAL_VECTOR_FROZEN_PAYLOAD_REQUIRED')
                    if self.guard:
                        self.guard.check_collection(row['collection'])
                embedding, vectors = self.embedding_factory(), self.vector_factory()
                timing['vector_initialization_seconds'] = perf_counter() - before
                phase, before = 'encoding', perf_counter()
                encoded = embedding.encode([query], query=True)[0]
                timing['encoding_seconds'] = perf_counter() - before
                phase, before = 'vector', perf_counter()
                hits = []
                for collection in sorted({r['collection'] for r in rows}):
                    ids = [r['key'] for r in rows if r['collection'] == collection]
                    observed = vectors.query(collection, encoded, ids, 24)
                    if len(observed) != min(len(ids), 24) or len({str(h['id']) for h in observed}) != len(observed):
                        raise ValueError('EVAL_VECTOR_COVERAGE_MISMATCH')
                    for hit in observed:
                        key = str(hit['id'])
                        if (key not in ids or hit.get('payload') != by_key[key]['vector_payload']
                                or isinstance(hit.get('score'), bool) or not isinstance(hit.get('score'), (int, float))
                                or not math.isfinite(hit['score'])):
                            raise ValueError('EVAL_VECTOR_IDENTITY_MISMATCH')
                        hits.append({'key': key, 'score': hit['score']})
                channels['vector'] = sorted(hits, key=lambda r: (-r['score'], r['key']))[:24]
                timing['vector_seconds'] = perf_counter() - before
                result['trace'].append({'component': 'vector', 'candidate_ids': [by_key[r['key']]['chunk_id'] for r in channels['vector']]})
            except Exception as exc:
                timing.setdefault(phase + '_seconds', perf_counter() - before)
                code = (str(exc) if self.strategy is not None and isinstance(exc, ValueError)
                        and str(exc).startswith(('RETRIEVAL_', 'EMBEDDING_', 'EVAL_')) else type(exc).__name__)
                result['degradations'].append({'component': 'vector', 'code': code, 'phase': phase})
                channels['vector'] = []
                result['effective_path'] = 'bm25' if variant != 'vector' else 'unavailable'
                if variant == 'vector':
                    result['status'] = 'failed'
        before = perf_counter()
        if variant in {'rrf', 'rrf_rerank'}:
            ranked = reciprocal_rank_fusion(channels,
                **({'k': self.strategy.rrf_k, 'weights': self.strategy.fusion_weights}
                   if self.strategy is not None else {}))[:24]
            timing['fusion_seconds'] = perf_counter() - before
        else:
            ranked = channels.get(variant, [])
        if variant == 'rrf_rerank' and ranked:
            before = perf_counter()
            try:
                if not self.reranker_factory:
                    raise ValueError('EVAL_RERANKER_FACTORY_REQUIRED')
                reranker = self.reranker_factory()
                if self.strategy is not None:
                    result['reranker_configuration']['actual'] = self.strategy.observe_reranker(reranker)
                    self.strategy.inspect_reranker(reranker)
                timing['reranker_initialization_seconds'] = perf_counter() - before
                before = perf_counter()
                scores = reranker.score(query, [by_key[r['key']]['text'] for r in ranked])
                if self.strategy is not None:
                    result['reranker_configuration']['actual'] = self.strategy.observe_reranker(reranker)
                    self.strategy.inspect_reranker(reranker)
                if len(scores) != len(ranked) or any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) for s in scores):
                    raise ValueError('EVAL_RERANK_OUTPUT_INVALID')
                # 诊断读回也可能失败，先完整收集再替换排序，异常仍保留原融合结果。
                reranker_stats = (deepcopy(getattr(reranker, 'last_stats', {}))
                                  if self.strategy is not None else None)
                ranked = [{**r, 'reranker_score': s, 'rrf_rank': n} for n, (r, s) in enumerate(zip(ranked, scores), 1)]
                ranked.sort(key=lambda r: (-r['reranker_score'], r['rrf_rank']))
                result['trace'].append({'component': 'reranker', 'candidate_count': len(ranked)})
                if self.strategy is not None:
                    result['reranker_configuration']['verified'] = True
                    result['trace'][-1]['stats'] = reranker_stats
                result['effective_path'] = ('bm25_rerank' if result['degradations'] else 'rrf_rerank')
            except Exception as exc:
                code = (str(exc) if self.strategy is not None and isinstance(exc, ValueError)
                        and str(exc).startswith(('RETRIEVAL_', 'RERANK_', 'EVAL_')) else type(exc).__name__)
                result['degradations'].append({'component': 'reranker', 'code': code})
                result['effective_path'] = 'bm25' if 'vector' not in channels or not channels['vector'] else 'rrf'
            timing['rerank_seconds'] = perf_counter() - before
        result['hits'] = [{**by_key[r['key']], **r, 'rank': n} for n, r in enumerate(ranked, 1)]
        result['ranked_ids'] = [r['chunk_id'] for r in result['hits']]
        result['execution_status'] = 'degraded' if result['degradations'] else 'ready' if ranked else 'empty'
        timing['total_seconds'] = perf_counter() - started
        return result


class ReflectionAdapter:
    """同稿基线和现有ReflectionService的单次核验/局部修订接点。"""

    def __init__(self, service=None, execute=None):
        """注入真实服务和独立Reviewer回调，构造时不调用。"""
        self.service, self.execute = service, execute
        self._unknown_tasks = set()

    def run(self, task_id: str, draft: str, variant: str, *, questions=None, evidence_ids=None,
            patches=None, reason='组件评测：核验冻结当前稿') -> dict:
        """基线保留原稿；开启组核验后允许交付未解决项和处理说明。

        patches 为可选局部patch列表，或接受核验结果并返回列表的回调；修订仍由
        真实服务逐项校验。不设置或改变服务轮次，max_rounds=0仍进行首次核验。
        回调开始后服务未结算的异常全部记unknown，该实例不再执行此任务。
        恢复必须经原收据层显式处理后新建适配器；持久未结算attempt仍会被拒绝。
        """
        if variant not in {'baseline', 'reflection'} or not isinstance(draft, str) or not draft.strip():
            raise ValueError('EVAL_REFLECTION_INPUT_INVALID')
        result = _result(variant, variant, output=draft, processing_summary='未经新增核验。', review=None)
        if variant == 'baseline':
            return result
        if self.service is None or self.execute is None:
            return {**result, 'status': 'unexecuted', 'effective_path': None,
                    'reason': 'reflection_runtime_not_supplied',
                    'degradations': [{'component': 'reflection', 'code': 'runtime_not_supplied'}]}
        from research_loop.domain.reflection import ResearchReviewInput, ResearchRevisionInput
        started = perf_counter()
        callback_started, service_review_finished = False, False

        def wrapped_execute(*args):
            """以真实回调边界识别发送后未知，不按异常类猜测是否计费。"""
            nonlocal callback_started
            callback_started = True
            return self.execute(*args)

        try:
            snapshot = self.service.snapshot(task_id)
            pending = any(not attempt.get('result')
                for target in snapshot.get('research_targets', {}).values()
                for attempt in target.get('attempts', []))
            if task_id in self._unknown_tasks or pending:
                return {**result, 'status': 'unknown', 'usage': None,
                    'processing_summary': '存在未结算核验；先经原收据恢复路径明确处理，本适配器不重发。',
                    'degradations': [{'component': 'reflection', 'code': 'unsettled_review_requires_receipt_recovery'}],
                    'trace': [{'component': 'reflection_execution_boundary', 'callback_started': False,
                               'service_review_finished': False, 'blocked_unsettled': True}],
                    'timing': {'total_seconds': perf_counter() - started}}
            args = ResearchReviewInput(target_type='answer', answer_markdown=draft, reason=reason,
                verification_questions=questions or ['核对当前稿关键结论是否有原文支持。'], evidence_ids=evidence_ids or [])
            review = self.service.review_research(task_id, args, wrapped_execute)
            service_review_finished = True
            result['review'] = review
            result['timing']['verification_seconds'] = perf_counter() - started
            result['trace'].append({'component': 'reflection', 'target_id': review.get('target_id'),
                                    'content_hash': review.get('content_hash'), 'review_status': review.get('status')})
            result['processing_summary'] = '已调用独立核验；判断与未解决项见review，不等于人工金标。'
            if review.get('status') == 'execution_failed':
                result.update(status='failed', processing_summary='核验执行失败，保留原稿及失败说明。')
            chosen = patches(review) if callable(patches) else patches
            if chosen:
                before = perf_counter()
                updated = self.service.revise_research(task_id, ResearchRevisionInput(
                    target_id=review['target_id'], content_hash=review['content_hash'],
                    review_id=review['review_id'], patches=chosen))
                result['output'] = updated['target']['items'][0]['text']
                result['revision'] = updated
                result['timing']['revision_seconds'] = perf_counter() - before
                result['processing_summary'] = '已局部修订，修订稿尚未经新增复审；其余文字由服务保持。'
        except Exception as exc:
            unknown = callback_started and not service_review_finished
            if unknown:
                self._unknown_tasks.add(task_id)
            result.update(status='unknown' if unknown else 'failed',
                          processing_summary='核验或局部修订未完成；保留当前正文，未解决项不得当作通过。')
            result['degradations'].append({'component': 'reflection', 'code': type(exc).__name__})
        result['trace'].append({'component': 'reflection_execution_boundary',
                               'callback_started': callback_started,
                               'service_review_finished': service_review_finished})
        result['timing']['total_seconds'] = perf_counter() - started
        return result


class _ControlledMemory:
    """按单一子类型过滤自动曝光和显式读回，其余委托真实MemoryService。"""

    def __init__(self, service, boundary, enabled, allow_explicit_history):
        """记录本轮边界及实际曝光，不修改持久偏好或历史。"""
        self.service, self.boundary, self.enabled = service, boundary, enabled
        self.allow_explicit_history = allow_explicit_history
        self.exposure = []

    def active(self):
        """只控制偏好自动注入；关闭时不读取并伪称已曝光。"""
        if self.boundary == 'preference_context' and not self.enabled:
            return []
        rows = self.service.active()
        self.exposure.extend({'kind': 'preference', 'source_id': r['memory_id']} for r in rows)
        return rows

    def search(self, query, source_type='all', limit=5, task_id=None):
        """关闭相关历史自动召回时仍允许独立user_note子类型。"""
        if self.boundary == 'history_recall' and not self.enabled:
            if source_type == 'research_history':
                return []
            source_type = 'user_note' if source_type == 'all' else source_type
        rows = self.service.search(query, source_type, limit, task_id=task_id)
        self.exposure.extend({'kind': 'automatic_' + r['source_type'], 'source_id': r['source_id']} for r in rows)
        return rows

    def read(self, task_id, args):
        """当前档案/会话/消息读回始终保留；研究历史读回由manifest明确约束。"""
        if (self.boundary == 'history_recall' and not self.enabled and not self.allow_explicit_history
                and args.source_type == 'research_history'):
            raise PermissionError('EVAL_HISTORY_READ_DISABLED')
        value = self.service.read(task_id, args)
        self.exposure.append({'kind': 'explicit_' + args.source_type, 'source_id': args.source_id})
        return value

    def profile(self):
        """画像工具与自动偏好注入共用边界，统计与其余业务权限保持。"""
        value = self.service.profile()
        if self.boundary == 'preference_context' and not self.enabled:
            return {**value, 'explicit_preferences': [], 'inferred_interests': [], 'revoked': []}
        self.exposure.extend({'kind': 'explicit_preference', 'source_id': row['memory_id']}
            for name in ('explicit_preferences', 'inferred_interests') for row in value.get(name, []))
        return value


class MemoryAdapter:
    """复用真实ContextService投影与容量检查，冻结基本会话只改变一个边界。"""

    def __init__(self, db, settings, *, memory_service=None, basic_history=None, context_factory=None):
        """注入独立库及显式Settings，绝不通过环境变量装配生产配置。

        basic_history 为已冻结、两组共用的基本会话快照；不接受 preferences 或
        related_history 预注入，以免绕过开关。当前目标、权限及同轮成果仍由真实
        ContextService从当前任务装配。此快照投影与生产自动history装配分别记录。
        context_factory 接受 ContextService 子类、db、受控 memory、settings；
        当前服务内部仍有隐式配置接点，未提供经隔离核验的工厂时明确不执行。
        单测在测试作用域显式替换 get_settings，适配器自身不全局修改配置函数。
        """
        from research_loop.services.memory import MemoryService
        self.db, self.settings = db, settings
        self.memory = memory_service or MemoryService(db)
        self.context_factory = context_factory
        self.basic_history = deepcopy(basic_history or {})
        if any(k in self.basic_history for k in ('preferences', 'related_history')):
            raise ValueError('EVAL_HISTORY_SNAPSHOT_CONTAINS_CONTROLLED_FIELDS')

    def bind_memory(self, boundary: str, enabled: bool, *, allow_explicit_history=False):
        """为隔离工具runtime提供同边界search/read/profile，不回退原始服务。"""
        if boundary not in {'history_recall', 'preference_context'}:
            raise ValueError('EVAL_MEMORY_BOUNDARY_UNSUPPORTED')
        return _ControlledMemory(self.memory, boundary, enabled, allow_explicit_history)

    def run(self, state: dict, messages: list[dict], tools: list[dict], variant: str, *, boundary: str,
            explicit_reads=None, allow_explicit_history=False) -> dict:
        """单次准备上下文，返回输入曝光和估算；不发送模型请求。

        context_projection 尚无合法原始基线接点，两组均明确未执行，不能把删除
        整个ContextService或改变压缩阈值当作消融。history/preference组始终复用
        同一投影、当前状态、档案读回和容量检查，不改变设置或工具schema。
        """
        if boundary not in {'history_recall', 'preference_context', 'context_projection'} or variant not in {'baseline', boundary}:
            raise ValueError('EVAL_MEMORY_VARIANT_INVALID')
        result = _result(variant, boundary + ('_enabled' if variant != 'baseline' else '_baseline'), output=None)
        if boundary == 'context_projection':
            return {**result, 'status': 'unexecuted', 'effective_path': None,
                    'reason': 'context_projection_baseline_strategy_pending',
                    'output': {'reason': 'context_projection_baseline_strategy_pending'}}
        from research_loop.tools.registry import schema_names
        if boundary == 'preference_context' and 'finish_task' in schema_names(tools):
            return {**result, 'status': 'unexecuted', 'effective_path': None,
                    'reason': 'interest_inference_finish_context_control_pending',
                    'output': {'reason': 'interest_inference_finish_context_control_pending'}}
        if self.context_factory is None:
            return {**result, 'status': 'unexecuted', 'effective_path': None,
                    'reason': 'explicit_context_service_settings_injection_pending',
                    'output': {'reason': 'explicit_context_service_settings_injection_pending'}}
        from research_loop.agents.memory import ContextService
        controlled = self.bind_memory(boundary, variant != 'baseline', allow_explicit_history=allow_explicit_history)
        frozen = deepcopy(self.basic_history)

        class FrozenHistoryContext(ContextService):
            """只替换历史装配接点，其余使用现有上下文算法。"""

            def history(inner, task_id):
                """保持冻结基本会话，分别通过真实服务取得相关历史与偏好。"""
                context = deepcopy(frozen)
                context['preferences'] = controlled.active()
                context['related_history'] = controlled.search(inner.db.get_task(task_id)['prompt'], 'research_history', 3)
                return context

        context = self.context_factory(FrozenHistoryContext, self.db, controlled, self.settings)
        started = perf_counter()
        try:
            readbacks = []
            for request in explicit_reads or []:
                before = perf_counter()
                readbacks.append(controlled.read(state['task_id'], SimpleNamespace(**request)))
                result['timing']['readback_seconds'] = result['timing'].get('readback_seconds', 0.0) + perf_counter() - before
            before = perf_counter()
            projected, metrics = context.prepare(deepcopy(state), deepcopy(messages), deepcopy(tools))
            result['timing']['projection_seconds'] = perf_counter() - before
            result.update(output=projected, context_metrics=metrics, readbacks=readbacks,
                          projection_origin='frozen_basic_history_with_existing_context_service',
                          token_estimate_is_exact=False)
        except ValueError as exc:
            result.update(status='failed', reason='context_capacity_or_integrity_rejected',
                          error_code=str(exc).split(':', 1)[0])
        except PermissionError:
            result.update(status='failed', reason='history_read_not_allowed')
        result['trace'] = controlled.exposure
        result['timing']['total_seconds'] = perf_counter() - started
        return result
