"""只读重算AG7.4已保存结果与账本；不接数据库、供应商或重新执行Agent。"""
from __future__ import annotations
from collections import Counter
import json
import statistics
from scripts.evaluation_common import OUT,ROOT,sha,ranked_metrics


def audit_outputs() -> dict:
    """核对冻结来源、排名分母、同稿比较和物理请求/收据账本的一致性。

    Returns:
        可复算指标与工程观测。语义标签单独来自source-review.json，不自动评为正确。

    Raises:
        AssertionError: 冻结文件、原句标注、排名、正文或账本被改动。
    """
    frozen=json.loads((OUT/'freeze.json').read_text())
    archive=json.loads((OUT/'frozen-source-map.json').read_text())
    for name,digest in frozen['files'].items():assert sha(ROOT/archive.get(name,name))==digest,name
    corpus=json.loads((OUT/'corpus.json').read_text())
    cases=json.loads((OUT/'retrieval-cases.json').read_text())
    spec=json.loads((ROOT/archive['backend/tests/acceptance/ag74_cases.json']).read_text())
    modes=frozen['retrieval']['modes']; scores={m:[] for m in modes}
    keys={c['key'] for c in corpus['chunks']}
    by_key={c['key']:c for c in corpus['chunks']}
    for case in cases:
        for anchor in case['source_annotations']:
            doc=next(d for d in corpus['documents'] if d['key']==anchor['material'])
            page=next(p for p in doc['pages'] if p['page_number']==anchor['page'])
            assert page['text'][anchor['char_start']:anchor['char_end']]==anchor['quote']
            for key in anchor['chunk_keys']:
                chunk=by_key[key]
                assert chunk['material']==anchor['material'] and chunk['evidence_id']==anchor['evidence_id']
                assert chunk['char_start']<anchor['char_end'] and chunk['char_end']>anchor['char_start']
        row=json.loads((OUT/f'retrieval/{case["id"]}.json').read_text())
        for mode in modes:
            saved=row['results'][mode]
            recomputed=ranked_metrics([r['key'] for r in saved['ranking']],case['relevant_chunks'],
                                     answerable=case['answerable'],corpus=keys)
            assert recomputed==saved['metrics']
            if case['answerable']:scores[mode].append(recomputed)
    summary={m:{'n':len(rows),'recall_at_10':statistics.mean(r['recall_at_10'] for r in rows),
                'mrr_at_24':statistics.mean(r['reciprocal_rank_at_24'] for r in rows)} for m,rows in scores.items()}
    reflection=[];receipts=[]
    for case in spec['reflection']:
        row=json.loads((OUT/f'reflection/{case["id"]}.json').read_text())
        assert row['baseline']['markdown']==case['markdown']
        assert all(d['markdown']==case['markdown'] for d in row['drafts'])
        receipts+=row['receipts']
        reflection.append({'case':case['id'],'state':row['view']['status'],'patches':len(row['reflection'].get('patches',[])),
            'accepted_reviews':len(row['reviews']),'requests':len(row['receipts']),'seconds':row['seconds']})
    physical=json.loads((OUT/'reflection/physical.json').read_text())
    assert len({r['call_id'] for r in receipts})==len(receipts)==len(physical)==32
    assert all(r['status']=='succeeded' for r in physical)
    assert Counter(r['request_id'] for r in receipts)==Counter(r['request_id'] for r in physical)
    totals={key:sum(json.loads(r['usage_json'])[key] for r in receipts)
            for key in ['prompt_tokens','completion_tokens','total_tokens']}
    assert totals=={key:sum(r['usage'][key] for r in physical) for key in totals}
    return {'frozen_files_verified':len(frozen['files']),'retrieval':summary,
        'missing_questions_not_scored':[c['id'] for c in cases if not c['answerable']],
        'reflection_execution':reflection,'physical_requests':len(physical),'tokens':totals,
        'repairs_observed':0,'semantic_scoring':'see separate source-review.json; no automatic semantic pass'}


if __name__=='__main__':
    print(json.dumps(audit_outputs(),ensure_ascii=False,indent=2))
