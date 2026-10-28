"""联合提示词实验的冻结输入与无模型前置；评分资料不参与消息或候选装配。"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch


_PRECONDITION_KEYS = {'catalog', 'history_messages', 'save_policy'}
_CATALOG_KEYS = {'material_index', 'title', 'abstract', 'authors', 'source_url',
                 'source', 'published_at', 'metadata_scope'}


def case_materials(case: dict) -> list[dict]:
    """取得显式材料，兼容旧单篇输入；不把其它case字段当作模型材料。"""
    values = case.get('materials')
    if values is None:
        values = [] if case.get('material') is None else [case['material']]
    if not isinstance(values, list):
        raise ValueError('PROMPT_BUNDLE_MATERIALS_INVALID')
    seen = set()
    for value in values:
        if (not isinstance(value, dict) or not isinstance(value.get('pdf_path'), str)
                or not isinstance(value.get('pdf_sha256'), str)
                or len(value['pdf_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in value['pdf_sha256'])
                or type(value.get('physical_pages')) is not int or value['physical_pages'] < 1
                or value['pdf_sha256'] in seen):
            raise ValueError('PROMPT_BUNDLE_MATERIAL_INVALID')
        seen.add(value['pdf_sha256'])
    if case.get('material') is not None:
        if not values or any(case['material'].get(k) != values[0].get(k) for k in ('pdf_path', 'pdf_sha256')):
            raise ValueError('PROMPT_BUNDLE_MATERIAL_ALIAS_CHANGED')
    return deepcopy(values)


def validate_preconditions(case: dict) -> dict:
    """校验并复制三种显式前置，拒绝把标签或工具历史夹入种子。"""
    pre = case.get('preconditions', {})
    if pre is None:
        pre = {}
    if not isinstance(pre, dict) or set(pre) - _PRECONDITION_KEYS:
        raise ValueError('PROMPT_BUNDLE_PRECONDITION_INVALID')
    result = {'catalog': pre.get('catalog', []), 'history_messages': pre.get('history_messages', []),
              'save_policy': pre.get('save_policy', 'deny')}
    if result['save_policy'] not in {'allow', 'deny'}:
        raise ValueError('PROMPT_BUNDLE_SAVE_POLICY_INVALID')
    if not isinstance(result['catalog'], list) or not isinstance(result['history_messages'], list):
        raise ValueError('PROMPT_BUNDLE_PRECONDITION_INVALID')
    if len(result['catalog']) > 20 or len(result['history_messages']) > 16:
        raise ValueError('PROMPT_BUNDLE_PRECONDITION_TOO_LARGE')
    for row in result['catalog']:
        if (not isinstance(row, dict) or set(row) - _CATALOG_KEYS
                or type(row.get('material_index')) is not int or row['material_index'] < 0
                or not isinstance(row.get('title'), str) or not row['title'].strip()
                or not isinstance(row.get('abstract'), str) or not row['abstract'].strip()
                or not isinstance(row.get('source_url'), str)
                or not isinstance(row.get('authors', []), list)
                or any(not isinstance(author, str) for author in row.get('authors', []))):
            raise ValueError('PROMPT_BUNDLE_CATALOG_INVALID')
    for row in result['history_messages']:
        if (not isinstance(row, dict) or set(row) != {'role', 'content'}
                or row['role'] not in {'user', 'assistant'}
                or not isinstance(row['content'], str) or not row['content'].strip()
                or len(row['content']) > 20000):
            raise ValueError('PROMPT_BUNDLE_HISTORY_INVALID')
    if result['catalog'] and result['history_messages']:
        raise ValueError('PROMPT_BUNDLE_MIXED_PRECONDITIONS')
    return deepcopy(result)


def case_save_policy(case: dict) -> str:
    """返回本格显式保存约束，不从历史assistant消息推导权限。"""
    return validate_preconditions(case)['save_policy']


def install_catalog(stack, discovery_type, case: dict, bundles: dict) -> dict:
    """给本格安装固定题录供应源，仍由原PaperService赋予候选权限。

    Args:
        stack: 本格ExitStack，退出时恢复原Discovery方法。
        discovery_type: 原生DiscoveryProvider类型，延迟导入由共同入口负责。
        case: 仅取显式catalog及材料字段，不读取评分标签。
        bundles: 按PDF内容hash映射的已核验来源收据。
    """
    catalog = validate_preconditions(case)['catalog']
    if not catalog:
        stack.enter_context(patch.object(discovery_type, 'search',
            side_effect=PermissionError('PROMPT_BUNDLE_NEW_CORPUS_SEARCH_DENIED')))
        return {'kind': 'search_denied', 'candidate_count': 0}
    from research_loop.domain.evidence import PaperCandidate, SearchResult, SourceResult
    materials = case_materials(case)
    candidates = []
    for row in catalog:
        if row['material_index'] >= len(materials):
            raise ValueError('PROMPT_BUNDLE_CATALOG_MATERIAL_UNKNOWN')
        material = materials[row['material_index']]
        bundle = bundles.get(material['pdf_sha256'])
        if not isinstance(bundle, dict) or not bundle.get('document_ref', {}).get('paper_id'):
            raise ValueError('PROMPT_BUNDLE_CATALOG_SOURCE_MISSING')
        ref = bundle['document_ref']
        metadata = ref.get('metadata') or {}
        candidates.append(PaperCandidate(paper_id=ref['paper_id'], title=row['title'],
            authors=row.get('authors', []), abstract=row['abstract'], paper_url=row['source_url'],
            source=metadata.get('source') or 'arxiv', published=None, date_precision='unknown',
            metadata={'controlled_catalog': True, 'source_pdf_sha256': material['pdf_sha256'],
                      'basis': 'actual_title_authors_and_abstract_excerpt', 'web_verified': False}))
    if len({c.paper_id for c in candidates}) != len(candidates):
        raise ValueError('PROMPT_BUNDLE_CATALOG_DUPLICATE_ID')

    def search(_provider, query, date_from=None, date_to=None, sources=None, limit=5):
        """返回冻结候选池及覆盖说明，不冒称在线召回或未知日期资格。"""
        if not isinstance(query, str) or not query.strip() or type(limit) is not int or limit < 1:
            raise ValueError('PROMPT_BUNDLE_CATALOG_QUERY_INVALID')
        selected = [c for c in candidates if not sources or c.source.lower() in {s.lower() for s in sources}]
        # 题录未核实日期，不允许静默忽略调用者的日期条件。
        unknown_dates = bool(date_from or date_to)
        eligible = [] if unknown_dates else selected
        cap = min(limit, 20)
        returned = [c.model_copy(deep=True) for c in eligible[:cap]]
        return SearchResult(candidates=returned, date_from=date_from, date_to=date_to,
            truncated=len(eligible) > cap,
            source_results=[SourceResult(source='controlled_local_catalog',
                status='partial_results' if unknown_dates else ('queried_success' if returned else 'queried_empty'),
                candidate_count=len(returned), message=('publication dates unverified in frozen pool' if unknown_dates
                    else 'fixed local metadata pool; not live search or relevance-ranking evaluation'))],
            query_scope='frozen_local_candidate_pool_unranked_not_web_search')
    stack.enter_context(patch.object(discovery_type, 'search', new=search))
    return {'kind': 'controlled_local_catalog', 'candidate_count': len(candidates),
            'live_search': False, 'original_evidence_grants_created': False}


def seed_history(db, conversation_id: str, preconditions: dict) -> dict:
    """在独立空会话事务中写合成消息，不创建虚假任务或模型收据。"""
    from research_loop.services.conversations import ConversationService
    from research_loop.storage.memory import MemoryStore
    rows = validate_preconditions({'preconditions': preconditions})['history_messages']
    if not rows:
        return {'kind': 'none', 'message_count': 0, 'message_ids': [], 'real_model_calls': 0}
    with MemoryStore(db).transaction() as tx:
        if not tx.one('SELECT conversation_id FROM conversations WHERE conversation_id=?', (conversation_id,)):
            raise ValueError('PROMPT_BUNDLE_HISTORY_CONVERSATION_MISSING')
        existing = tx.all('SELECT message_key,role,content,task_id FROM conversation_messages '
                          'WHERE conversation_id=? ORDER BY sequence_no', (conversation_id,))
        expected = [{'message_key': f'prompt-bundle-seed:{conversation_id}:{i}', **row, 'task_id': ''}
                    for i, row in enumerate(rows)]
        if existing and existing != expected:
            raise ValueError('PROMPT_BUNDLE_HISTORY_NOT_EMPTY_OR_CHANGED')
        # task_id列为NOT NULL；空值明确表示人工种子，不冒充某次模型任务。
        mids = [ConversationService.add_message(tx, conversation_id, '', row['message_key'],
                    row['role'], row['content']) for row in expected]
    return {'kind': 'synthetic_history_not_natural_model_run', 'message_count': len(rows),
            'message_ids': mids, 'real_model_calls': 0, 'evaluation_roots_created': 0}


def upload_materials(client, case: dict, bundles: dict) -> list[dict]:
    """经原HTTP入口上传附件；题录池不成为根任务必须覆盖的附件。"""
    if validate_preconditions(case)['catalog']:
        return []
    attachments = []
    for i, material in enumerate(case_materials(case)):
        content = Path(material['pdf_path']).read_bytes()
        if sha256(content).hexdigest() != material['pdf_sha256']:
            raise ValueError('PROMPT_BUNDLE_UPLOAD_MATERIAL_CHANGED')
        bundle = bundles.get(material['pdf_sha256'])
        if not isinstance(bundle, dict) or not bundle.get('document_ref', {}).get('paper_id'):
            raise ValueError('PROMPT_BUNDLE_UPLOAD_SOURCE_MISSING')
        filename = f'source-{i+1}.pdf'
        response = client.post('/api/uploads', params={'filename': filename}, content=content)
        if response.status_code != 201:
            raise ValueError('PROMPT_BUNDLE_UPLOAD_HTTP_FAILED')
        ref = response.json().get('input_ref')
        if not isinstance(ref, str) or not ref:
            raise ValueError('PROMPT_BUNDLE_UPLOAD_ID_MISSING')
        attachments.append({'input_ref': ref, 'filename': filename,
                            'paper_id': bundle['document_ref']['paper_id']})
    return attachments
