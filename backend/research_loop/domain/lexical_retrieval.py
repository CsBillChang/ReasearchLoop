"""从已授权原文选择词法候选，不读取标签、环境或装配模型。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import re
import unicodedata

from research_loop.storage.indexes import rank_terms, terms


ROUTES = ('bm25_rerank', 'located_bm25_rerank')
CONCEPT_LEXICON = (
    ('方法', ('method', 'approach')), ('机制', ('mechanism', 'method')),
    ('结构', ('architecture', 'structure')), ('模块', ('module', 'component')),
    ('训练', ('training', 'train')), ('推理', ('inference',)),
    ('实验', ('experiment', 'experimental')), ('数据集', ('dataset', 'data')),
    ('指标', ('metric', 'evaluation')), ('局限', ('limitation', 'limitations')),
    ('限制', ('limitation', 'constraint')), ('消融', ('ablation',)),
    ('生成', ('generation', 'generate')), ('构造', ('construction', 'construct')),
    ('设计', ('design',)), ('实现', ('implementation',)),
    ('步骤', ('step', 'procedure')), ('优化', ('optimization',)),
    ('对齐', ('alignment',)), ('特征', ('feature',)), ('损失', ('loss',)),
    ('分类', ('classification',)), ('计数', ('count', 'counting')),
    ('检测', ('detection',)), ('评估', ('evaluation',)),
    ('比较', ('comparison',)), ('区别', ('difference',)),
    ('条件', ('condition',)), ('阶段', ('stage',)), ('压缩', ('compression',)),
    ('验证', ('validation',)), ('更新', ('update',)),
)
_GENERIC_IDENTIFIERS = frozenset({
    'ai', 'api', 'cnn', 'cv', 'clip', 'detr', 'gpt', 'llm', 'llms', 'ml',
    'nlp', 'ovd', 'pdf', 'rag', 'rgb', 'svd', 'vit', 'vlm', 'vlms', 'vqa',
    'dnn', 'gpu', 'cpu', 'bert', 'relu', 'lora', 'sam', 'vq',
})
_GENERIC_NAME_WORDS = frozenset({
    'a', 'an', 'the', 'analysis', 'approach', 'benchmark', 'benchmarks',
    'evaluation', 'experiment', 'experiments', 'framework', 'introduction',
    'learning', 'method', 'methods', 'model', 'models', 'overview', 'study',
    'survey', 'system', 'systems', 'training', 'vision', 'visual',
})
_IDENTITY_STRINGS = ('chunk_id', 'document_id', 'index_id', 'evidence_id')
_HASH_FIELDS = ('source_sha256', 'text_sha256')


def _normalize(value: str) -> str:
    """只规范Unicode、连字符和空白，不把近似名称当成同一论文。"""
    value = unicodedata.normalize('NFKC', value)
    value = value.translate(str.maketrans({'–': '-', '—': '-', '−': '-', '’': "'", '‘': "'"}))
    return ' '.join(value.split()).casefold()


def _specific_prefix(prefix: str) -> bool:
    """拒绝问题句和泛词前缀，保留可解释的短论文名称。"""
    tokens = re.findall(r'[A-Za-z0-9][A-Za-z0-9_-]*', prefix)
    if not tokens or len(tokens) > 6 or '?' in prefix:
        return False
    if tokens[0].casefold() in {'a', 'an', 'the', 'can', 'what', 'why', 'how', 'when', 'where', 'are', 'is', 'does', 'do', 'towards', 'toward', 'on'}:
        return False
    return any(t.casefold() not in _GENERIC_NAME_WORDS | _GENERIC_IDENTIFIERS for t in tokens)


def _title_aliases(title: str) -> set[str]:
    """仅从公开标题生成全名、短名称与显式模型标识，不消费问题。"""
    canonical = _normalize(title)
    aliases = {canonical}
    if ':' in canonical:
        prefix = canonical.split(':', 1)[0].strip()
        if _specific_prefix(prefix):
            aliases.add(prefix)
    for token in re.findall(r'(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*', title):
        if len(token) >= 3 and (re.search(r'[a-z][A-Z]', token) or token.isupper()):
            aliases.add(_normalize(token))
    return {a for a in aliases if len(a) >= 3 and a not in _GENERIC_IDENTIFIERS
            and any(t not in _GENERIC_NAME_WORDS for t in terms(a))}


def _expanded_query(query: str) -> tuple[str, list[str]]:
    """固定通用词典只扩展已出现的中文概念，不补造论文特有术语。"""
    added = []
    present = set(terms(query))
    for chinese, english in CONCEPT_LEXICON:
        if chinese in query:
            for word in english:
                if word not in present:
                    added.append(word)
                    present.add(word)
    return ' '.join([query.strip(), *added]).strip(), added


class LexicalRetriever:
    """一次冻结完整身份和词频，在每次选择时仅接受查询与明确授权范围。"""

    def __init__(self, chunks: list[dict], catalog: list[dict]):
        """核对原文身份并构建正文/既有章节词频，不初始化任何外部组件。

        Args:
            chunks: 完整原文片段；key可省略，此时使用chunk_id。
            catalog: 仅含document_id和title的公开论文目录。

        Raises:
            ValueError: 身份、hash、范围、重复项或目录字段非法。
        """
        if not isinstance(chunks, list) or not chunks:
            raise ValueError('LEXICAL_CHUNKS_REQUIRED')
        rows, seen_ids, seen_keys = [], set(), set()
        document_versions, page_evidence, evidence_pages = {}, {}, {}
        for raw in chunks:
            if not isinstance(raw, dict) or any(not isinstance(raw.get(k), str) or not raw[k] for k in _IDENTITY_STRINGS):
                raise ValueError('LEXICAL_CHUNK_IDENTITY_REQUIRED')
            if any(not isinstance(raw.get(k), str) or not re.fullmatch(r'[a-f0-9]{64}', raw[k]) for k in _HASH_FIELDS):
                raise ValueError('LEXICAL_CHUNK_HASH_INVALID')
            text = raw.get('text')
            if not isinstance(text, str) or sha256(text.encode()).hexdigest() != raw['text_sha256']:
                raise ValueError('LEXICAL_CHUNK_TEXT_HASH_MISMATCH')
            if (type(raw.get('char_start')) is not int or type(raw.get('char_end')) is not int
                    or raw['char_start'] < 0 or raw['char_end'] - raw['char_start'] != len(text)
                    or 'page_number' not in raw
                    or (raw.get('page_number') is not None
                        and (type(raw['page_number']) is not int or raw['page_number'] < 1))):
                raise ValueError('LEXICAL_CHUNK_RANGE_INVALID')
            key = raw.get('key', raw['chunk_id'])
            if not isinstance(key, str) or not key or raw['chunk_id'] in seen_ids or key in seen_keys:
                raise ValueError('LEXICAL_DUPLICATE_OR_INVALID_ID')
            if raw.get('section_origin') == 'explicit_heading' and (
                    not isinstance(raw.get('section'), str) or not raw['section'].strip()):
                raise ValueError('LEXICAL_EXPLICIT_SECTION_INVALID')
            document_id = raw['document_id']
            version = (raw['source_sha256'], raw['index_id'])
            # HTML等来源没有物理页码；此时按确切证据页区分，不能捏造页号。
            page = ((document_id, None, raw['evidence_id']) if raw['page_number'] is None
                    else (document_id, raw['page_number']))
            if (document_id in document_versions and document_versions[document_id] != version
                    or page in page_evidence and page_evidence[page] != raw['evidence_id']
                    or raw['evidence_id'] in evidence_pages and evidence_pages[raw['evidence_id']] != page):
                raise ValueError('LEXICAL_SOURCE_VERSION_OR_PAGE_IDENTITY_CONFLICT')
            document_versions[document_id] = version
            page_evidence[page] = raw['evidence_id']
            evidence_pages[raw['evidence_id']] = page
            seen_ids.add(raw['chunk_id']); seen_keys.add(key)
            rows.append({**deepcopy(raw), 'key': key})
        documents = {row['document_id'] for row in rows}
        if not isinstance(catalog, list):
            raise ValueError('LEXICAL_PUBLIC_CATALOG_REQUIRED')
        titles = {}
        for entry in catalog:
            if (not isinstance(entry, dict) or set(entry) != {'document_id', 'title'}
                    or not isinstance(entry['document_id'], str) or entry['document_id'] not in documents
                    or not isinstance(entry['title'], str)
                    or entry['document_id'] in titles):
                raise ValueError('LEXICAL_PUBLIC_CATALOG_INVALID')
            titles[entry['document_id']] = entry['title']
        if set(titles) != documents:
            raise ValueError('LEXICAL_CATALOG_INCOMPLETE')
        self._rows = tuple(rows)
        self._documents = frozenset(documents)
        self._body_bags = {row['chunk_id']: Counter(terms(row['text'])) for row in rows}
        self._located_bags = {
            row['chunk_id']: Counter(terms(row['text'] + ('\n' + row['section']
                if row.get('section_origin') == 'explicit_heading' else ''))) for row in rows
        }
        owners = {}
        for document_id, title in titles.items():
            # 未知标题只关闭别名定位，仍保留该文档的正文和既有章节召回。
            for alias in _title_aliases(title):
                owners.setdefault(alias, set()).add(document_id)
        self._aliases = {alias: frozenset(ids) for alias, ids in owners.items()}

    def _matches(self, query: str, allowed: set[str]) -> list[dict]:
        """最长精确别名优先，避免完整标题内的较短歧义别名覆盖它。"""
        candidates = []
        for alias, owners in self._aliases.items():
            ids = sorted(owners & allowed)
            if not ids:
                continue
            pattern = r'(?<![A-Za-z0-9_])' + re.escape(alias) + r'(?![A-Za-z0-9_])'
            for found in re.finditer(pattern, query):
                candidates.append({'alias': alias, 'document_ids': ids,
                    'start': found.start(), 'end': found.end(), 'ambiguous': len(ids) > 1})
        selected = []
        for match in sorted(candidates, key=lambda r: (-(r['end'] - r['start']), r['start'], r['alias'])):
            if not any(match['start'] < other['end'] and other['start'] < match['end'] for other in selected):
                selected.append(match)
        return sorted(selected, key=lambda r: (r['start'], r['end'], r['alias']))

    def select(self, query: str, scope_document_ids: list[str], route: str) -> dict:
        """返回最多24条正分候选及可审阅的路由轨迹，保留原文供外部CE使用。

        Args:
            query: 用户原始查询；原值保留在返回轨迹。
            scope_document_ids: 非空、唯一且属于冻结语料的完整授权文档ID。
            route: bm25_rerank或located_bm25_rerank。

        Returns:
            candidates、search_query、local_document_ids和trace；候选均为独立副本。

        Raises:
            ValueError: 查询、路线或授权范围缺失、重复或越界。
        """
        if not isinstance(query, str) or not query.strip() or route not in ROUTES:
            raise ValueError('LEXICAL_QUERY_OR_ROUTE_INVALID')
        if (not isinstance(scope_document_ids, (list, tuple)) or not scope_document_ids
                or any(not isinstance(d, str) or not d for d in scope_document_ids)
                or len(set(scope_document_ids)) != len(scope_document_ids)
                or not set(scope_document_ids) <= self._documents):
            raise ValueError('LEXICAL_SCOPE_INVALID')
        allowed = set(scope_document_ids)
        rows = [row for row in self._rows if row['document_id'] in allowed]
        # 并列沿用冻结片段输入序，与原rank_terms合同一致，不依赖set迭代顺序。
        original = rank_terms(query, rows, [self._body_bags[r['chunk_id']] for r in rows], 24)
        matches, local_ids, added, selection = [], [], [], []
        search_query, routing = query, 'baseline'
        if route == 'bm25_rerank':
            selected = original
            selection = [{'chunk_id': r['chunk_id'], 'source': 'original_global'} for r in selected]
        else:
            normalized = _normalize(query)
            matches = self._matches(normalized, allowed)
            ambiguous = any(match['ambiguous'] for match in matches)
            if not ambiguous:
                local_ids = sorted({d for match in matches for d in match['document_ids']})
            # 任一未消歧名称使本次定位整体回退，必须保留其他已识别论文名。
            stripped = normalized
            if not ambiguous:
                for match in reversed(matches):
                    stripped = stripped[:match['start']] + ' ' + stripped[match['end']:]
            search_query, added = _expanded_query(' '.join(stripped.split()))
            if not local_ids:
                routing = 'ambiguous_fallback' if ambiguous else 'no_paper_fallback'
                selected = rank_terms(search_query, rows, [self._located_bags[r['chunk_id']] for r in rows], 24)
                selection = [{'chunk_id': r['chunk_id'], 'source': 'expanded_global'} for r in selected]
            else:
                routing = 'explicit_paper'
                local_lists = []
                for document_id in local_ids:
                    local_rows = [r for r in rows if r['document_id'] == document_id]
                    local_lists.append(rank_terms(search_query, local_rows,
                        [self._located_bags[r['chunk_id']] for r in local_rows], 24))
                # 每篇独立计算IDF，跨篇只轮询名次，不直接相加不同池的BM25分数。
                local_round_robin = [ranked[i] for i in range(24) for ranked in local_lists if i < len(ranked)]
                selected, seen = [], set()

                def append_from(pool: list[dict], count: int, source: str) -> None:
                    """配额只计新片段，避免重复候选浪费保底槽。"""
                    appended = 0
                    for candidate in pool:
                        if len(selected) == 24 or appended == count:
                            break
                        if candidate['chunk_id'] not in seen:
                            selected.append(candidate)
                            seen.add(candidate['chunk_id'])
                            selection.append({'chunk_id': candidate['chunk_id'], 'source': source})
                            appended += 1

                append_from(local_round_robin, 18, 'local_quota')
                append_from(original, 6, 'original_global_reserve')
                append_from(local_round_robin, 24 - len(selected), 'local_fill')
                append_from(original, 24 - len(selected), 'original_global_fill')
        return {'route': route, 'candidates': deepcopy(selected), 'search_query': search_query,
            'local_document_ids': local_ids, 'trace': {
                'original_query': query, 'routing': routing, 'alias_matches': matches,
                'expanded_terms': added, 'scope_document_ids': list(scope_document_ids),
                'authorized_chunk_count': len(rows), 'candidate_limit': 24,
                'local_quota': 18, 'original_global_reserve': 6,
                'selection': selection, 'tie_break': 'frozen_chunk_input_order',
                'heading_policy': 'existing_explicit_heading_only',
                'reranker_input_policy': 'original_query_and_original_text',
            }}
