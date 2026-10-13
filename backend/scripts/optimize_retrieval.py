"""固定旧材料上的一次有界检索优化；仅本机模型和只读Qdrant，禁止外部连接。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import socket
import statistics
import time

from research_loop.providers.embeddings import LocalEmbedding
from research_loop.providers.reranker import LocalReranker
from research_loop.storage.indexes import terms, rank_terms, reciprocal_rank_fusion, fuse_reranker_ranks
from research_loop.storage.vector_index import VectorIndex

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/retrieval-optimization-20261011'
OLD = ROOT/'artifacts/ag7-4-evaluation-20261011'
MANIFEST = ROOT/'data/ag71-preflight/models/manifest.json'


def read(path: Path):
    """读取已有JSON，不更新时间戳或覆盖历史。"""
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    """计算冻结身份，禁止换材料追分。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, data) -> None:
    """独占写入一次结果；已有文件即停止，不能悄悄重跑。"""
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def verify() -> dict:
    """核对旧输入及结果树，保留原评测而非调用旧runner。"""
    frozen = read(OUT/'freeze.json')
    for rel, expected in {**frozen['inputs'], **frozen['original_artifact']}.items():
        if sha(ROOT/rel) != expected:
            raise ValueError('FROZEN_INPUT_CHANGED: '+rel)
    if (OUT/'execution-freeze.json').exists():
        for rel, expected in read(OUT/'execution-freeze.json')['files'].items():
            if sha(ROOT/rel) != expected:
                raise ValueError('FROZEN_EXECUTION_CHANGED: '+rel)
    return frozen


def freeze_code() -> None:
    """零模型检查后冻结本轮执行代码，测量开始后不改实现。"""
    verify()
    paths = [Path(__file__).resolve(), ROOT/'backend/research_loop/providers/reranker.py',
        ROOT/'backend/research_loop/storage/indexes.py',
        ROOT/'backend/tests/unit/test_retrieval_optimization.py']
    write('execution-freeze.json', {'files': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'scope_sha256': sha(OUT/'freeze.json'), 'created_at': datetime.now(timezone.utc).isoformat()})


def local_network_only() -> None:
    """进程仅允许loopback/Unix连接；错误外连在发包前拒绝。"""
    original = socket.socket.connect

    def connect(sock, address):
        """只读本机Qdrant可联网，供应商和下载连接均禁止。"""
        if isinstance(address, tuple) and address[0] not in {'127.0.0.1', 'localhost', '::1'}:
            raise RuntimeError('EXTERNAL_NETWORK_DISABLED')
        return original(sock, address)
    socket.socket.connect = connect


class Budget:
    """所有本机推理也事前登记，超过冻结上限即停止；没有生成模型入口。"""
    def __init__(self, frozen):
        """跨阶段累加本批计数，不允许删除started标记重跑。"""
        self.caps = frozen['caps']
        self.path = OUT/'local-ledger.json'
        self.value = read(self.path) if self.path.exists() else {
            'started_at': time.time(), 'local_pair_scores': 0, 'local_score_calls': 0,
            'query_embeddings': 0, 'deepseek_requests': 0}

    def book(self, pairs=0, embeddings=0):
        """本机调用前计入上限，即使异常也保留其尝试。"""
        if time.time()-self.value['started_at'] > self.caps['max_wall_seconds']:
            raise RuntimeError('LOCAL_WALL_BUDGET')
        next_value = {**self.value, 'local_pair_scores': self.value['local_pair_scores']+pairs,
            'local_score_calls': self.value['local_score_calls']+bool(pairs),
            'query_embeddings': self.value['query_embeddings']+embeddings}
        if any(next_value[k] > self.caps[k] for k in ('local_pair_scores', 'local_score_calls', 'query_embeddings')):
            raise RuntimeError('LOCAL_CALL_BUDGET')
        self.value = next_value
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(next_value, indent=2))
        temporary.replace(self.path)


def metric(keys: list[str], relevant: list[str]) -> dict:
    """只对冻结可回答题计算排名；未标注题返回空指标，不记满分。"""
    if len(keys) > 24 or len(set(keys)) != len(keys):
        raise ValueError('INVALID_RANKING')
    if not relevant:
        return {'recall_at_10': None, 'recall_at_8': None, 'mrr_at_24': None}
    first = next((i for i, key in enumerate(keys, 1) if key in relevant), None)
    return {'recall_at_10': len(set(keys[:10]) & set(relevant))/len(relevant),
        'recall_at_8': len(set(keys[:8]) & set(relevant))/len(relevant),
        'mrr_at_24': 1/first if first else 0.0}


def aggregate(rows: dict, ids: list[str]) -> dict:
    """按题宏平均；不同重复轮次先在每题内汇总。"""
    selected = [rows[key] for key in ids if rows[key]['mrr_at_24'] is not None]
    return {name: statistics.mean(row[name] for row in selected)
            for name in ('recall_at_10', 'recall_at_8', 'mrr_at_24')}


def no_regression(candidate: dict, baseline: dict) -> bool:
    """Recall两个截点和首条排名均不退化；浮点仅容忍计算尾差。"""
    return all(candidate[k]+1e-12 >= baseline[k] for k in baseline)


def fusion() -> None:
    """一次重放3种预定权重；仅选型集参与选择，确认集单列报告。"""
    frozen = verify()
    write('fusion-started.json', {'deepseek_requests': 0})
    cases = read(OUT/'retrieval-cases.json')
    result = {}
    for weight in [0.0, *frozen['fusion']['weights'], 1.0]:
        per_case = {}
        for case in cases:
            old = read(OLD/'retrieval'/f'{case["id"]}.json')['results']
            rows = old['rrf']['ranking']
            score_map = {r['key']: r['reranker_score'] for r in old['rrf_rerank']['ranking']}
            ranked = fuse_reranker_ranks(rows, [score_map[r['key']] for r in rows], weight=weight)
            keys = [r['key'] for r in ranked]
            per_case[case['id']] = {**metric(keys, case['relevant_chunks']), 'ranking': keys}
        result[str(weight)] = {'per_case': per_case,
            'development': aggregate(per_case, frozen['development']),
            'confirmation': aggregate(per_case, frozen['confirmation']),
            'all_answerable': aggregate(per_case, [c['id'] for c in cases if c['answerable']])}
    baseline = result['1.0']['development']
    eligible = [w for w in frozen['fusion']['weights'] if no_regression(result[str(w)]['development'], baseline)]
    chosen = max(eligible, key=lambda w: (result[str(w)]['development']['mrr_at_24'],
        result[str(w)]['development']['recall_at_10'], -w)) if eligible else 1.0
    write('fusion.json', {'selected_weight': chosen, 'selection_uses_confirmation': False, 'results': result})
    print('fusion selected', chosen, 'development', result[str(chosen)]['development'], flush=True)


def scorer(config: dict) -> LocalReranker:
    """只按冻结配置构造同一份本机权重。"""
    return LocalReranker(MANIFEST, threads=config['threads'], batch_size=config['batch_size'],
                         sort_by_length=config['sort_by_length'])


def timed_score(model, budget, query, texts):
    """记录分词和实际推理耗时；不缓存评分、不截断正文。"""
    budget.book(pairs=len(texts))
    started = time.perf_counter()
    scores = model.score(query, texts)
    return scores, time.perf_counter()-started, dict(model.last_stats)


def efficiency() -> None:
    """五种配置、四道开发题、各五次；只按预定速度门选择一次。"""
    frozen = verify(); budget = Budget(frozen)
    write('efficiency-started.json', {'configs': frozen['efficiency']['configs']})
    corpus = read(OUT/'corpus.json'); cases = {c['id']: c for c in read(OUT/'retrieval-cases.json')}
    texts = {r['key']: r['text'] for r in corpus['chunks']}
    result = {}
    for config in frozen['efficiency']['configs']:
        model = scorer(config)
        cold = time.perf_counter()
        timed_score(model, budget, 'initialization only', ['A fixed sentence without an evaluation answer.'])
        cold = time.perf_counter()-cold
        measures = []
        for repeat in range(frozen['efficiency']['repeats']):
            ids = frozen['efficiency']['cases']
            ids = ids[repeat % len(ids):] + ids[:repeat % len(ids)]
            for cid in ids:
                rows = read(OLD/'retrieval'/f'{cid}.json')['results']['rrf']['ranking']
                scores, elapsed, stats = timed_score(model, budget, cases[cid]['question'], [texts[r['key']] for r in rows])
                record = {'case': cid, 'repeat': repeat, 'seconds': elapsed, 'stats': stats, 'scores': scores}
                write(f'efficiency/{config["id"]}-{cid}-{repeat}.json', record); measures.append(record)
        result[config['id']] = {'config': config, 'median_seconds': statistics.median(r['seconds'] for r in measures),
            'cold_init_seconds': cold, 'padded_tokens': sum(r['stats']['padded_tokens'] for r in measures),
            'input_tokens': sum(r['stats']['input_tokens'] for r in measures), 'identity': model.identity}
        print('efficiency', config['id'], result[config['id']]['median_seconds'], flush=True)
        del model
        __import__('gc').collect()
    fastest = min(result, key=lambda name: result[name]['median_seconds'])
    if result[fastest]['median_seconds'] > .9*result['legacy']['median_seconds']:
        fastest = 'legacy'
    write('efficiency.json', {'selected_config': result[fastest]['config'], 'results': result,
        'quality_note': 'int8组批可能改变分数，下一阶段重算全部质量；本阶段不声称数值等价'})


def common_query(case, corpus, embedding, vectors, budget, bags):
    """真实重测同一候选生成；只读原隔离索引，不更新数据库或向量。"""
    records = corpus['chunks']
    started = time.perf_counter()
    bm25 = rank_terms(case['question'], records, bags, 24)
    lexical_time = time.perf_counter()-started
    budget.book(embeddings=1)
    started = time.perf_counter()
    query = embedding.encode([case['question']], query=True)[0]
    embedding_time = time.perf_counter()-started
    started = time.perf_counter(); hits = []
    for doc in corpus['documents']:
        found = vectors.query(doc['collection'], query, doc['chunk_keys'], 24)
        if any(str(r['id']) not in doc['chunk_keys'] or r['payload']['index_id'] != doc['index_id'] for r in found):
            raise ValueError('QUERY_SCOPE_CHANGED')
        hits.extend({'key': str(r['id']), 'score': r['score']} for r in found)
    vector_time = time.perf_counter()-started
    hits = sorted(hits, key=lambda r: (-r['score'], r['key']))[:24]
    started = time.perf_counter()
    fused = reciprocal_rank_fusion({'bm25': bm25, 'vector': hits})[:24]
    fusion_time = time.perf_counter()-started
    expected = read(OLD/'retrieval'/f'{case["id"]}.json')['results']['rrf']['ranking']
    if [r['key'] for r in fused] != [r['key'] for r in expected]:
        raise ValueError('FROZEN_CANDIDATES_CHANGED: '+case['id'])
    return fused, {'bm25': lexical_time, 'embedding': embedding_time, 'vector_query': vector_time, 'fusion': fusion_time}


def quantity() -> None:
    """候选数量三档与旧路线交错测量，各24题五次；不得追加重复。"""
    frozen = verify(); budget = Budget(frozen)
    write('quantity-started.json', {'repeats': frozen['quantity']['repeats'], 'counts': [12, 16, 24]})
    cases = read(OUT/'retrieval-cases.json'); corpus = read(OUT/'corpus.json')
    by_key = {r['key']: r for r in corpus['chunks']}
    bags = [Counter(terms(r['text'])) for r in corpus['chunks']]
    weight = read(OUT/'fusion.json')['selected_weight']
    config = read(OUT/'efficiency.json')['selected_config']
    embedding = LocalEmbedding(MANIFEST)
    legacy = scorer(frozen['efficiency']['configs'][0]); candidate = scorer(config)
    vectors = VectorIndex('http://127.0.0.1:6339', 'researchloop_test_ag74_20261011')
    for doc in corpus['documents']:
        points = vectors.retrieve(doc['collection'], doc['chunk_keys'])
        if len(points) != len(doc['chunk_keys']):
            raise ValueError('FROZEN_INDEX_MISSING')
        for key in doc['chunk_keys']:
            if points[key]['payload']['chunk_id'] != by_key[key]['chunk_id']:
                raise ValueError('FROZEN_INDEX_CHANGED')
    cold = time.perf_counter()
    budget.book(embeddings=1); embedding.encode(['initialization only'], query=True)
    for model in (legacy, candidate):
        timed_score(model, budget, 'initialization only', ['A fixed sentence without an evaluation answer.'])
    cold = time.perf_counter()-cold
    modes = [('legacy', 24, legacy, 1.0), ('candidate24', 24, candidate, weight),
             ('candidate16', 16, candidate, weight), ('candidate12', 12, candidate, weight)]
    for repeat in range(frozen['quantity']['repeats']):
        ordered = cases[repeat:] + cases[:repeat]
        for index, case in enumerate(ordered):
            rows, common = common_query(case, corpus, embedding, vectors, budget, bags)
            write(f'quantity/common-{case["id"]}-{repeat}.json', {'seconds': common, 'rrf_keys': [r['key'] for r in rows]})
            shift = (repeat+index) % len(modes)
            for name, count, model, alpha in modes[shift:]+modes[:shift]:
                prefix = rows[:count]
                scores, elapsed, stats = timed_score(model, budget, case['question'], [by_key[r['key']]['text'] for r in prefix])
                rank_start = time.perf_counter()
                ranked = fuse_reranker_ranks(rows, scores, weight=alpha)
                rank_seconds = time.perf_counter()-rank_start
                keys = [r['key'] for r in ranked]
                write(f'quantity/{name}-{case["id"]}-{repeat}.json', {'case': case['id'], 'repeat': repeat,
                    'keys': keys, 'scores': scores, 'stats': stats, 'rerank_seconds': elapsed,
                    'seconds': sum(common.values())+elapsed+rank_seconds,
                    'metrics': metric(keys, case['relevant_chunks'])})
            print('quantity', repeat+1, case['id'], 'done', flush=True)
    vectors.close()
    write('quantity-completed.json', {'cold_init_seconds': cold, 'selected_efficiency': config,
        'weight': weight, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'ledger': budget.value})
    report()


def report() -> None:
    """只读重算，先封存开发集选择，再读取其确认集是否通过。"""
    frozen = verify(); read(OUT/'quantity-completed.json')
    cases = read(OUT/'retrieval-cases.json'); results = {}
    for name in ('legacy', 'candidate24', 'candidate16', 'candidate12'):
        rows, per_case, times, drifts = {}, {}, [], []
        for case in cases:
            records = [read(OUT/f'quantity/{name}-{case["id"]}-{r}.json') for r in range(5)]
            if any(r['keys'] != records[0]['keys'] for r in records):
                drifts.append(case['id'])
            rows[case['id']] = {key: statistics.mean(r['metrics'][key] for r in records) if case['answerable'] else None
                for key in ('recall_at_10', 'recall_at_8', 'mrr_at_24')}
            per_case[case['id']] = {**rows[case['id']], 'median_seconds': statistics.median(r['seconds'] for r in records),
                'keys': records[0]['keys']}
            times.append(per_case[case['id']]['median_seconds'])
        results[name] = {'development': aggregate(rows, frozen['development']),
            'confirmation': aggregate(rows, frozen['confirmation']),
            'all_answerable': aggregate(rows, [c['id'] for c in cases if c['answerable']]),
            'median_seconds': statistics.median(times), 'p95_seconds': sorted(times)[math.ceil(.95*len(times))-1],
            'repeated_ranking_drift': drifts, 'per_case': per_case}
    baseline = results['legacy']
    eligible = [name for name in results if name != 'legacy'
        and no_regression(results[name]['development'], baseline['development'])
        and results[name]['median_seconds'] <= .9*baseline['median_seconds']]
    selected = max(eligible, key=lambda name: (results[name]['development']['mrr_at_24'],
        results[name]['development']['recall_at_10'], -results[name]['median_seconds'])) if eligible else None
    # 选择只用开发集质量与无标注时延；确认集失败不回头挑其他候选。
    write('selection.json', {'selected': selected, 'eligible_development': eligible, 'confirmation_used_for_selection': False})
    accepted = bool(selected and no_regression(results[selected]['confirmation'], baseline['confirmation'])
        and results[selected]['all_answerable']['mrr_at_24'] > baseline['all_answerable']['mrr_at_24']+1e-12
        and not results[selected]['repeated_ranking_drift'])
    write('result.json', {'selected': selected, 'deployment_eligible': accepted, 'results': results,
        'generative_requests': 0, 'external_network': 'loopback-only',
        'limitations': ['旧题确认集非独立未见材料', '不完整相关性标注', '组件时间相加非Agent端到端延迟',
            '固定本机单进程热测，不保证并发负载或未来论文表现']})
    print('selected', selected, 'deployment_eligible', accepted, flush=True)
    for name, value in results.items():
        print(name, value['all_answerable'], value['median_seconds'], flush=True)


def main():
    """按冻结阶段顺序执行；异常留证即停止，不自动重试。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze-code', 'fusion', 'efficiency', 'quantity', 'report'])
    args = parser.parse_args()
    local_network_only()
    try:
        {'freeze-code': freeze_code, 'fusion': fusion, 'efficiency': efficiency,
         'quantity': quantity, 'report': report}[args.stage]()
    except Exception as exc:
        write('failure-'+args.stage+'.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
