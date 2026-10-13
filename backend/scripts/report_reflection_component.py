"""只读重算AG7.4组件对照的账本、版本及固定分母；语义标注由独立原文核对提供。"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json

from scripts.evaluate_reflection_component import OUT, QUESTIONS, PHASE_CAPS
from scripts.evaluation_common import ROOT, sha
from scripts.run_ag6 import summarize_usage


def audit() -> dict:
    """验证冻结文件、实际消息和精确差异，输出可重复计算的统计。

    Returns:
        原始请求分账及结构指标。事实支持性仍来自source-review.json，不能由hash证明。
    """
    frozen = json.loads((OUT/'freeze.json').read_text())
    for path, expected in frozen['files'].items():
        assert sha(ROOT/path) == expected, path
    inputs = json.loads((OUT/'inputs.json').read_text())
    ledger = json.loads((OUT/'physical.json').read_text())
    records = [json.loads((OUT/(c['id']+'.json')).read_text()) for c in inputs['cases']]
    all_receipts = [r for c in records for r in c['receipts']]
    sent = [r for r in all_receipts if r['sent_state'] != 'not_sent']
    assert len({r['request_id'] for r in sent}) == len(sent) == len(ledger)
    lookup = {r['request_id']: r for r in sent}
    for r in ledger:
        assert r['status'] == 'succeeded'
        assert json.loads(lookup[r['request_id']]['usage_json']) == r['usage']
    counts = Counter((r['case'], r['phase']) for r in ledger)
    assert len(ledger) <= frozen['budget']['physical_cap']
    for (case, phase), count in counts.items():
        assert count <= PHASE_CAPS[phase]
    details = []
    for case, result in zip(inputs['cases'], records):
        assert result['baseline']['markdown'] == case['markdown']
        material = next(m for m in inputs['materials'] if m['key'] == case['material'])
        expected_pages = {p['evidence_id']: p['text'] for p in material['pages']}
        current = result['current']
        assert sha256(current['markdown'].encode()).hexdigest() == current['content_hash']
        final = result['final_review']
        assert final['content_hash'] == current['content_hash']
        accepted = json.loads(result['task']['result_json'])
        assert accepted['answer_markdown'] == current['markdown']
        assert accepted['saved_path'] is None
        reviews = result['reflection']['attempts']
        assert len(reviews) <= 2 and len(result['reflection']['patches']) <= 1
        if result['reflection']['patches']:
            assert result['checks']['non_target_text_preserved']
        for receipt in result['receipts']:
            request = json.loads(receipt['request_json'])
            messages = request['messages']
            bundle_message = next((m for m in messages if m.get('content', '').startswith('权威原文证据包')), None)
            if not bundle_message:
                continue
            bundle = json.loads(bundle_message['content'].split('\n', 1)[1])
            assert {p['evidence_id']: p['text'] for p in bundle} == expected_pages
            # 审校任务的第一次请求才是独立入口；后续可含其自身本轮工具观察。
            if receipt['sequence_no'] == 1:
                text = json.dumps(messages, ensure_ascii=False)
                assert json.dumps(QUESTIONS, ensure_ascii=False) in messages[1]['content']
                assert 'current_review' not in text and 'correction' not in messages[1]['content']
                if current['content_hash'] in messages[1]['content'] and len(reviews) == 2:
                    assert result['initial_review']['review_id'] not in text
                    for finding in result['initial_review']['findings']:
                        assert finding['reason'] not in text
        correct = [c for c in case['claims'] if c['label'] == 'supported']
        assert all(c['text'] in current['markdown'] for c in correct)
        failed = [a for values in result['actions'].values() for a in values if a['status'] != 'succeeded']
        details.append({'case': case['id'], 'request_counts': {p: counts[case['id'], p] for p in PHASE_CAPS},
            'requests': sum(counts[case['id'], p] for p in PHASE_CAPS), 'usage': result['usage'],
            'seconds': result['seconds'], 'initial_status': result['initial_review']['status'],
            'final_status': final['status'], 'patches': len(result['reflection']['patches']),
            'correct_sentences_unchanged': len(correct), 'failed_actions': [
                {'tool': a['tool_name'], 'message': a['observation']['message']} for a in failed],
            'baseline_hash': result['baseline']['content_hash'], 'current_hash': current['content_hash']})
    source_audit = json.loads((OUT/'source-review.json').read_text())
    assert [c['id'] for c in source_audit['cases']] == [c['id'] for c in inputs['cases']]
    phase_cost = {}
    for phase in PHASE_CAPS:
        items = [r for r in ledger if r['phase'] == phase]
        phase_cost[phase] = {'requests': len(items), 'total_tokens': sum(r['usage']['total_tokens'] for r in items)}
    return {'frozen_files_verified': len(frozen['files']), 'cases': details,
        'actual_requests': len(ledger), 'usage': summarize_usage(all_receipts), 'phase_cost': phase_cost,
        'case_seconds_total': sum(c['seconds'] for c in records),
        'structural_checks': 'passed; source-review.json is separate semantic judgement',
        'baseline_claims': dict(Counter(c['label'] for case in inputs['cases'] for c in case['claims'])),
        'semantic_counts': source_audit['metrics'], 'money_cost': None,
        'causal_scope': 'fixed-draft component only; not independent generalization or whole-agent ablation'}


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
