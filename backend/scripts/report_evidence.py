"""只读重算现存评测，不导入运行服务、不联网、不加载模型，也不覆盖历史产物。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def quality(keys: list[str], relevant: list[str], answerable: bool, corpus: set[str]) -> dict:
    """独立计算单题指标，避免复用被核对脚本的计算函数。

    Args:
        keys: 最多24个、按排名排列的唯一片段身份。
        relevant: 事前标注的相关片段，不代表穷举相关性。
        answerable: 缺证题必须没有相关标注，不计入质量分母。
        corpus: 本题可检索的片段集合。

    Returns:
        Recall@10、MRR@24及首个相关片段的排名。

    Raises:
        ValueError: 身份重复、越界或可回答标记与标注冲突。
    """
    if (len(keys) > 24 or len(set(keys)) != len(keys)
            or len(set(relevant)) != len(relevant)
            or not set(keys + relevant) <= corpus or bool(relevant) != answerable):
        raise ValueError('INVALID_RANKING_OR_LABELS')
    first = next((i for i, key in enumerate(keys, 1) if key in relevant), None)
    return {'recall_at_10': sum(key in relevant for key in keys[:10]) / len(relevant) if answerable else None,
            'mrr_at_24': (1 / first if first else 0.0) if answerable else None,
            'first_relevant_rank': first}


def latency(samples: list[list[float]]) -> dict:
    """先取每题重复测量的中位数，再算跨题P50/P95；不混成端到端耗时。

    Args:
        samples: 每题的组件总耗时，单位为秒；至少每题有一次测量。

    Returns:
        题数、实际测量数及采用nearest-rank定义的P95。
    """
    if not samples or any(not row for row in samples):
        raise ValueError('EMPTY_TIMINGS')
    if any(not math.isfinite(x) or x < 0 for row in samples for x in row):
        raise ValueError('INVALID_TIMING')
    medians = sorted(statistics.median(row) for row in samples)
    return {'timed_questions': len(samples), 'measurements': sum(map(len, samples)),
            'median_seconds': statistics.median(medians),
            'p95_seconds': medians[math.ceil(.95 * len(medians)) - 1]}


def accounting(rows: list[dict]) -> dict:
    """按物理请求重算已知用量；缺失用量和未知状态分别保留。

    Args:
        rows: 带request_id、status和usage的原始物理请求账本。

    Returns:
        全部分母、状态分布、已知token小计和缺失request_id。
    """
    ids = [r['request_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('DUPLICATE_PHYSICAL_REQUEST')
    names = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    known, missing = [], []
    for row in rows:
        usage = row.get('usage')
        if not isinstance(usage, dict) or any(usage.get(k) is None for k in names):
            missing.append(row['request_id'])
            continue
        if any(not isinstance(usage[k], int) or usage[k] < 0 for k in names):
            raise ValueError('INVALID_USAGE')
        if usage['prompt_tokens'] + usage['completion_tokens'] != usage['total_tokens']:
            raise ValueError('TOKEN_TOTAL_MISMATCH')
        known.append(usage)
    return {'physical_requests': len(rows), 'status_counts': dict(Counter(r['status'] for r in rows)),
            'known_tokens': {k: sum(u[k] for u in known) for k in names},
            'missing_usage_request_ids': missing, 'all_usage_known': not missing}


class Reader:
    """读取指定仓库内证据，并记录本次实际消费的文件hash。"""

    def __init__(self, root: Path):
        """初始化只读根目录；不连接数据库或创建输出文件。"""
        self.root = root.resolve()
        self.inputs: dict[str, str] = {}

    def read(self, relative: str):
        """读取JSON并纳入追溯清单；拒绝指向仓库外的路径。"""
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('OUTSIDE_REPOSITORY')
        raw = path.read_bytes()
        self.inputs[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def freeze_status(self, directory: str) -> dict:
        """分别记录当前源码漂移与材料完整性，不把旧hash改写成当前hash。"""
        frozen = self.read(directory + '/freeze.json')
        mapping_path = self.root / directory / 'frozen-source-map.json'
        mapping = self.read(directory + '/frozen-source-map.json') if mapping_path.exists() else {}
        mismatch, verified = [], 0
        for name, digest in frozen['files'].items():
            path = (self.root / mapping.get(name, name)).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('OUTSIDE_REPOSITORY')
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if actual != digest:
                mismatch.append({'path': name, 'expected': digest, 'actual': actual,
                                 'kind': 'source_drift' if name.startswith('backend/') else 'material_drift'})
            else:
                verified += 1
        if any(r['kind'] == 'material_drift' for r in mismatch):
            raise ValueError('FROZEN_MATERIAL_CHANGED')
        return {'verified': verified, 'mismatches': mismatch,
                'historical_file_set_available': not mismatch,
                'scope': 'file hashes only; not a new execution or full environment reconstruction'}


def mean_quality(rows: list[dict]) -> dict:
    """对可回答题宏平均；空集合没有质量分数，不补满分或零分。"""
    valid = [r for r in rows if r['recall_at_10'] is not None]
    return {'answerable': len(valid), 'excluded_missing': len(rows) - len(valid),
            **{k: statistics.mean(r[k] for r in valid) if valid else None
               for k in ('recall_at_10', 'mrr_at_24')}}


def assert_close(actual: float | None, expected: float | None) -> None:
    """检查计算结果与历史报表一致，允许浮点求和舍入差异。"""
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError('NULL_METRIC_MISMATCH')
    elif not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'METRIC_MISMATCH: {actual} != {expected}')


def retrieval_audit(reader: Reader) -> dict:
    """重算原始四路与两次未采用候选，使用所有固定题及所有重复测量。"""
    base = 'artifacts/ag7-4-evaluation-20261011'
    corpus = reader.read(base + '/corpus.json')
    cases = reader.read(base + '/retrieval-cases.json')
    by_key = {r['key']: r for r in corpus['chunks']}
    # 核查标签实际指向原页字符区间，不用模型答案生成标签。
    for case in cases:
        for anchor in case['source_annotations']:
            doc = next(d for d in corpus['documents'] if d['key'] == anchor['material'])
            page = next(p for p in doc['pages'] if p['page_number'] == anchor['page'])
            if page['text'][anchor['char_start']:anchor['char_end']] != anchor['quote']:
                raise ValueError('SOURCE_SPAN_MISMATCH')
            for key in anchor['chunk_keys']:
                chunk = by_key[key]
                if (chunk['evidence_id'] != anchor['evidence_id'] or chunk['material'] != anchor['material']
                        or chunk['char_start'] >= anchor['char_end'] or chunk['char_end'] <= anchor['char_start']):
                    raise ValueError('LABEL_CHUNK_MISMATCH')
        if set(case['relevant_chunks']) != {k for a in case['source_annotations'] for k in a['chunk_keys']}:
            raise ValueError('LABEL_UNION_MISMATCH')
    saved = reader.read(base + '/retrieval/result.json')
    original = {}
    for mode, expected in saved['summary'].items():
        scores, times = [], []
        for case in cases:
            row = reader.read(base + f'/retrieval/{case["id"]}.json')['results'][mode]
            allowed = {k for k, v in by_key.items() if v['material'] in case['materials']}
            metric = quality([r['key'] for r in row['ranking']], case['relevant_chunks'], case['answerable'], allowed)
            assert_close(metric['recall_at_10'], row['metrics']['recall_at_10'])
            assert_close(metric['mrr_at_24'], row['metrics']['reciprocal_rank_at_24'])
            scores.append(metric); times.append([row['seconds']])
        calculated = {**mean_quality(scores), **latency(times)}
        for key, old_key in [('recall_at_10', 'macro_recall_at_10'), ('mrr_at_24', 'mrr_at_24'),
                             ('median_seconds', 'median_seconds'), ('p95_seconds', 'p95_seconds')]:
            assert_close(calculated[key], expected[old_key])
        original[mode] = calculated
    repeats = {}
    for directory, folder, names, metric_key in [
        ('retrieval-optimization-20261011', 'quantity', ['legacy', 'candidate24', 'candidate16', 'candidate12'], 'all_answerable'),
        ('reranker-efficiency-20261012', 'measurements', ['baseline', 'candidate'], 'quality')]:
        path = 'artifacts/' + directory
        frozen, report = reader.read(path + '/freeze.json'), reader.read(path + '/result.json')
        n = frozen['quantity']['repeats'] if folder == 'quantity' else frozen['repeats']
        batch = {}
        for name in names:
            scores, times, drift = [], [], []
            for case in cases:
                rows = [reader.read(path + f'/{folder}/{name}-{case["id"]}-{i}.json') for i in range(n)]
                allowed = {k for k, v in by_key.items() if v['material'] in case['materials']}
                metrics = [quality(r['keys'], case['relevant_chunks'], case['answerable'], allowed) for r in rows]
                for row, metric in zip(rows, metrics):
                    for key in ('recall_at_10', 'mrr_at_24'):
                        assert_close(metric[key], row['metrics'][key])
                scores.append({k: statistics.mean(m[k] for m in metrics) if case['answerable'] else None
                               for k in ('recall_at_10', 'mrr_at_24')})
                times.append([r['seconds'] for r in rows])
                if any(r['keys'] != rows[0]['keys'] for r in rows):
                    drift.append(case['id'])
            calculated = {**mean_quality(scores), **latency(times), 'ranking_drift': drift}
            expected = report['results'][name]
            target = expected[metric_key]['all'] if metric_key == 'quality' else expected[metric_key]
            for k in ('recall_at_10', 'mrr_at_24'):
                assert_close(calculated[k], target[k])
            for k in ('median_seconds', 'p95_seconds'):
                assert_close(calculated[k], expected[k])
            batch[name] = calculated
        repeats[directory] = batch
    return {'corpus_documents': len(corpus['documents']), 'pages': sum(len(d['pages']) for d in corpus['documents']),
            'chunks': len(by_key), 'questions': len(cases), 'original': original, 'repeated_batches': repeats,
            'scope': 'saved rankings and timings; no new inference; labels are not exhaustive human gold'}


def reflection_audit(reader: Reader) -> dict:
    """重算独立组件与先前整链失败账本，不把原文核对标签当作自动事实证明。"""
    base = 'artifacts/ag7-4-evaluation-20261011'
    component = 'artifacts/ag7-4-reflection-component-20261011'
    original_rows = reader.read(base + '/reflection/physical.json')
    old = []
    for case in ('F1', 'F2', 'F3', 'F4'):
        result = reader.read(base + f'/reflection/{case}.json')
        old.append({'case': case, 'status': result['view']['status'],
                    'patches': len(result['reflection'].get('patches', [])), 'receipts': len(result['receipts'])})
    inputs = reader.read(component + '/inputs.json')
    source = reader.read(component + '/source-review.json')
    physical = reader.read(component + '/physical.json')
    if [c['id'] for c in source['cases']] != [c['id'] for c in inputs['cases']]:
        raise ValueError('SEMANTIC_CASE_ID_MISMATCH')
    receipts, unchanged = [], 0
    for case in inputs['cases']:
        result = reader.read(component + '/' + case['id'] + '.json')
        current = result['current']
        if (result['baseline']['markdown'] != case['markdown']
                or hashlib.sha256(current['markdown'].encode()).hexdigest() != current['content_hash']
                or result['final_review']['content_hash'] != current['content_hash']):
            raise ValueError('DRAFT_BINDING_MISMATCH')
        for claim in case['claims']:
            if claim['label'] == 'supported':
                if claim['text'] not in current['markdown']:
                    raise ValueError('SUPPORTED_CLAIM_CHANGED')
                unchanged += 1
        receipts.extend(r for r in result['receipts'] if r['sent_state'] != 'not_sent')
    if Counter(r['request_id'] for r in receipts) != Counter(r['request_id'] for r in physical):
        raise ValueError('RECEIPT_LEDGER_MISMATCH')
    lookup = {r['request_id']: json.loads(r['usage_json']) for r in receipts}
    if any(lookup[r['request_id']] != r['usage'] for r in physical):
        raise ValueError('RECEIPT_USAGE_MISMATCH')
    transitions = Counter((a, b) for c in source['cases'] for a, b in zip(c['before'], c['after'], strict=True))
    before = Counter(c['label'] for case in inputs['cases'] for c in case['claims'])
    if before != Counter(a for c in source['cases'] for a in c['before']):
        raise ValueError('SEMANTIC_DENOMINATOR_MISMATCH')
    if unchanged != source['metrics']['correct_claims_unchanged']['numerator']:
        raise ValueError('PRESERVATION_COUNT_MISMATCH')
    for pair, key in [(('contradicted', 'supported'), 'contradicted_correctly_repaired'),
                      (('insufficient', 'bounded_disclosure'), 'unsupported_version_safely_handled')]:
        if (transitions[pair] != source['metrics'][key]['numerator']
                or before[pair[0]] != source['metrics'][key]['denominator']):
            raise ValueError('SEMANTIC_PAIRED_COUNT_MISMATCH')
    return {'old_full_loop': {'cases': old, **accounting(original_rows)},
            'component': {**accounting(physical), 'drafts': len(inputs['cases']), 'baseline_labels': dict(before),
                          'paired_labels': {a + ' -> ' + b: n for (a, b), n in transitions.items()},
                          'supported_text_unchanged': unchanged,
                          'semantic_provenance': source['reviewer'], 'label_scope': source['scope']},
            'historical_freeze': reader.freeze_status(component),
            'limits': 'semantic labels reused, not newly human audited; different execution designs are not pooled as a success rate'}


def audit(root: Path) -> dict:
    """读取历史证据并返回报告，缺文件或数值不符时失败而非静默排除。"""
    reader = Reader(root)
    retrieval = retrieval_audit(reader)
    reflection = reflection_audit(reader)
    original_freeze = reader.freeze_status('artifacts/ag7-4-evaluation-20261011')
    return {'retrieval': retrieval, 'reflection': reflection, 'original_freeze': original_freeze,
            'consumed_files': reader.inputs, 'new_model_calls': 0,
            'scope': 'local saved-result arithmetic audit; not production rerun, new semantic judge, or fresh benchmark'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    print(json.dumps(audit(parser.parse_args().root), ensure_ascii=False, indent=2))
