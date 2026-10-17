"""离线重算公开匿名证据包；只读JSON，不访问数据库、网络或模型。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from scripts.report_evidence import assert_close, latency, mean_quality, quality


def verify_usage(tokens: dict) -> None:
    """核对聚合Token分账，不把聚合数冒充逐请求原始收据。

    Args:
        tokens: 公开的prompt、completion和total计数。

    Raises:
        ValueError: 数值非法或合计不相等。
    """
    names = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    if any(type(tokens.get(k)) is not int or tokens[k] < 0 for k in names):
        raise ValueError('INVALID_AGGREGATE_USAGE')
    if tokens['prompt_tokens'] + tokens['completion_tokens'] != tokens['total_tokens']:
        raise ValueError('AGGREGATE_USAGE_MISMATCH')


def retrieval_report(data: dict) -> dict:
    """校验完整题集、候选身份及重复测量，重算所有路线并保留负结果。

    Args:
        data: 匿名片段、固定标签、完整排名与已保存耗时。

    Returns:
        各批次各配置的质量、计时和实际分母。

    Raises:
        ValueError: 缺题、重复行、越权片段、分母或原汇总不一致。
    """
    scope = data['scope']
    chunks = {c['id']: c['material'] for c in data['chunks']}
    cases = {c['id']: c for c in data['cases']}
    if (len(chunks) != len(data['chunks']) or len(chunks) != scope['chunks']
            or len(set(chunks.values())) != scope['documents']
            or len(cases) != len(data['cases']) or len(cases) != scope['questions']
            or sum(c['answerable'] for c in cases.values()) != scope['answerable']
            or len(cases) - scope['answerable'] != scope['missing']
            or scope['recall_cutoff'] != 10 or scope['mrr_cutoff'] != 24):
        raise ValueError('CORPUS_OR_DENOMINATOR_MISMATCH')
    results = {}
    for batch in data['batches']:
        batch_id, n = batch['id'], batch['repeats']
        if batch_id in results or type(n) is not int or n < 1:
            raise ValueError('INVALID_BATCH_OR_REPEATS')
        modes = set(batch['expected'])
        rows = {(r['case_id'], r['mode'], r['repeat']): r for r in batch['runs']}
        expected_keys = {(qid, mode, repeat) for qid in cases for mode in modes for repeat in range(n)}
        if len(rows) != len(batch['runs']) or set(rows) != expected_keys:
            raise ValueError('INCOMPLETE_OR_DUPLICATE_RANKINGS')
        summary = {}
        for mode in sorted(modes):
            scores, times = [], []
            for qid, case in cases.items():
                allowed = {key for key, material in chunks.items() if material in case['materials']}
                measurements = [rows[qid, mode, i] for i in range(n)]
                metrics = [quality(r['ranking'], case['relevant_chunks'], case['answerable'], allowed)
                           for r in measurements]
                scores.append({k: statistics.mean(m[k] for m in metrics) if case['answerable'] else None
                               for k in ('recall_at_10', 'mrr_at_24')})
                times.append([r['seconds'] for r in measurements])
            actual = {**mean_quality(scores), **latency(times)}
            for key in actual:
                assert_close(actual[key], batch['expected'][mode][key])
            summary[mode] = actual
        results[batch_id] = summary
    return {'questions': len(cases), 'answerable': scope['answerable'],
            'excluded_missing': scope['missing'],
            'rankings': sum(len(b['runs']) for b in data['batches']), 'batches': results}


def reflection_report(data: dict) -> dict:
    """复算前后配对标签、正确原句保留检查及聚合分账。

    Args:
        data: 已脱敏的固定槽位标签和历史投入汇总。

    Returns:
        配对计数及两批独立投入；不重新判定语义。
    """
    claims = data['claims']
    identities = {(c['draft_id'], c['slot']) for c in claims}
    if (len(claims) != data['scope']['claim_slots'] or len(identities) != len(claims)
            or len({c['draft_id'] for c in claims}) != data['scope']['drafts']):
        raise ValueError('REFLECTION_DENOMINATOR_MISMATCH')
    transitions = Counter(c['before'] + ' -> ' + c['after'] for c in claims)
    if dict(transitions) != data['expected_transitions']:
        raise ValueError('REFLECTION_TRANSITION_MISMATCH')
    supported = [c for c in claims if c['before'] == 'supported']
    if (len(supported) != data['expected_preserved_supported']
            or any(c['original_supported_text_preserved'] is not True for c in supported)):
        raise ValueError('SUPPORTED_TEXT_PRESERVATION_MISMATCH')
    component = data['component_accounting']
    verify_usage(component['tokens'])
    verify_usage(data['prior_full_loop']['tokens'])
    phases = component['phases']
    if len({r['phase'] for r in phases}) != len(phases):
        raise ValueError('DUPLICATE_REFLECTION_PHASE')
    for phase in phases:
        verify_usage(phase['tokens'])
    if (sum(r['physical_requests'] for r in phases) != component['physical_requests']
            or any(sum(r['tokens'][k] for r in phases) != value for k, value in component['tokens'].items())):
        raise ValueError('REFLECTION_PHASE_TOTAL_MISMATCH')
    return {'drafts': data['scope']['drafts'], 'slots': len(claims),
            'paired_labels': dict(transitions), 'supported_text_unchanged': len(supported),
            'component_accounting': component, 'prior_full_loop': data['prior_full_loop'],
            'semantic_rejudged': False}


def report(bundle: Path) -> dict:
    """读取公开包并核对文件hash；报表只输出统计和非敏感版本信息。

    Args:
        bundle: 包含固定四个JSON文件的公开目录。

    Returns:
        数值、完整性和演示分账检查结果。

    Raises:
        ValueError: 文件清单/hash或演示身份/用量不一致。
    """
    provenance = json.loads((bundle / 'provenance.json').read_text())
    allowed = {'retrieval.json', 'reflection.json', 'demonstrations.json'}
    if set(provenance['files']) != allowed:
        raise ValueError('PUBLIC_FILE_MANIFEST_MISMATCH')
    values = {}
    for name, digest in provenance['files'].items():
        raw = (bundle / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('PUBLIC_FILE_HASH_MISMATCH: ' + name)
        values[name] = json.loads(raw)
    demos = values['demonstrations.json']
    known_batches = {b['id'] for b in provenance['historical_batches']}
    if (demos['new_model_calls'] != 0 or len({d['id'] for d in demos['records']}) != len(demos['records'])
            or any(d['batch_id'] not in known_batches for d in demos['records'])):
        raise ValueError('DEMONSTRATION_IDENTITY_MISMATCH')
    for demo in demos['records']:
        verify_usage(demo['tokens'])
    return {'package_id': provenance['package_id'], 'integrity_verified_files': len(allowed),
            'retrieval': retrieval_report(values['retrieval.json']),
            'reflection': reflection_report(values['reflection.json']),
            'demonstrations': [{k: d[k] for k in ('id', 'batch_id', 'status', 'physical_requests', 'tokens')}
                               for d in demos['records']],
            'new_model_calls': 0,
            'scope': 'stored-result arithmetic; no new inference, timings, semantic judgment or task execution'}


def main() -> None:
    """运行公开包复算，缺省仅写stdout，失败返回非零。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'docs/evidence')
    args = parser.parse_args()
    print(json.dumps(report(args.bundle), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
