"""只验证一次固定推理优化：旧排序、24候选、旧材料，禁止业务模型与重复实验。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import time

from research_loop.providers.embeddings import LocalEmbedding
from research_loop.providers.reranker import LocalReranker
from research_loop.storage.indexes import terms
from research_loop.storage.vector_index import VectorIndex
from scripts.optimize_retrieval import common_query, local_network_only, metric, no_regression

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/reranker-efficiency-20261012'
OLD = ROOT/'artifacts/ag7-4-evaluation-20261011'
MANIFEST = ROOT/'data/ag71-preflight/models/manifest.json'


def read(path: Path):
    """读取冻结材料或已落盘结果，不改旧文件。"""
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    """返回字节身份；参数及材料不能在看结果后修改。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value) -> None:
    """独占创建本批结果，重复执行不覆盖原证据。"""
    target = OUT/name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def verify() -> dict:
    """核对旧两批artifact及输入，已有执行冻结时同时检查代码。"""
    frozen = read(OUT/'freeze.json')
    for name, expected in {**frozen['inputs'], **frozen['old_artifacts']}.items():
        if sha(ROOT/name) != expected:
            raise ValueError('FROZEN_INPUT_CHANGED: '+name)
    execution = OUT/'execution-freeze.json'
    if execution.exists():
        entry = read(execution)
        if sha(OUT/'freeze.json') != entry['scope_sha256']:
            raise ValueError('FROZEN_SCOPE_CHANGED')
        for name, expected in entry['files'].items():
            if sha(ROOT/name) != expected:
                raise ValueError('FROZEN_EXECUTION_CHANGED: '+name)
    return frozen


def freeze_code() -> None:
    """零模型检查后记录执行源码；不修改已有范围冻结。"""
    verify()
    files = ['backend/scripts/verify_reranker_efficiency.py','backend/scripts/optimize_retrieval.py',
        'backend/research_loop/providers/reranker.py','backend/research_loop/storage/indexes.py',
        'backend/tests/unit/test_reranker_efficiency.py']
    write('execution-freeze.json', {'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'scope_sha256':sha(OUT/'freeze.json'),'files':{name:sha(ROOT/name) for name in files}})


class Budget:
    """本机推理亦有限额；每次运行只允许一份账本。"""
    def __init__(self, caps):
        """不加载已有账本恢复重复，调用前计入失败尝试。"""
        if (OUT/'local-ledger.json').exists():
            raise ValueError('BATCH_ALREADY_STARTED')
        self.caps = caps
        self.value = {'started_at':time.time(),'pair_scores':0,'score_calls':0,'query_embeddings':0,'deepseek_requests':0}

    def book(self, pairs=0, embeddings=0):
        """复用公共查询入口的计数接口；超限或超时在推理前停止。"""
        if time.time()-self.value['started_at'] > self.caps['wall_seconds']:
            raise RuntimeError('LOCAL_WALL_BUDGET')
        next_value = {**self.value,'pair_scores':self.value['pair_scores']+pairs,
            'score_calls':self.value['score_calls']+bool(pairs),'query_embeddings':self.value['query_embeddings']+embeddings}
        if any(next_value[k] > self.caps[k] for k in ('pair_scores','score_calls','query_embeddings')):
            raise RuntimeError('LOCAL_CALL_BUDGET')
        self.value = next_value
        temporary = OUT/'local-ledger.tmp'
        temporary.write_text(json.dumps(self.value,indent=2))
        temporary.replace(OUT/'local-ledger.json')


def legacy_ranking(rows: list[dict], scores: list[float]) -> list[str]:
    """保持生产原有纯CE排序，平分时沿用RRF名次，不调用新增融合helper。

    Args:
        rows: 同一任务范围内的有序RRF候选。
        scores: 与输入逐项对应的实际模型logit。

    Returns:
        排序后的同一组来源身份，不丢失或新增候选。
    """
    if len(rows) != len(scores) or len({r['key'] for r in rows}) != len(rows):
        raise ValueError('RERANK_IDENTITY_INVALID')
    if not all(math.isfinite(s) for s in scores):
        raise ValueError('RERANK_OUTPUT_INVALID')
    return [rows[i]['key'] for i in sorted(range(len(rows)),key=lambda i:(-scores[i],i))]


def quality_by_group(cases: list[dict], per_case: dict) -> dict:
    """相同标注分母下按题宏平均；缺证题不记满分，中英文单列。"""
    result = {}
    for group in ('all','zh','en'):
        rows = [per_case[c['id']] for c in cases if c['answerable'] and (group=='all' or c['language']==group)]
        result[group] = {k:statistics.mean(r[k] for r in rows) for k in ('recall_at_10','recall_at_8','mrr_at_24')}
    return result


def adoption_gate(baseline: dict, candidate: dict) -> dict:
    """执行冻结的质量/速度门，不根据结果调阈值或选择其他配置。"""
    quality = all(no_regression(candidate['quality'][g], baseline['quality'][g]) for g in ('all','zh','en'))
    speed = candidate['median_seconds'] <= .9*baseline['median_seconds']
    p95 = candidate['p95_seconds'] <= baseline['p95_seconds']
    stable = not baseline['ranking_drift'] and not candidate['ranking_drift']
    return {'quality_nonregression':quality,'median_at_least_10_percent_faster':speed,
        'p95_nonregression':p95,'repeat_rankings_stable':stable,'eligible':quality and speed and p95 and stable}


def report(cases: list[dict], frozen: dict) -> None:
    """只汇总这一次完成的测量，结果、逐题退化和采用判定一起保存。"""
    results = {}
    for name in ('baseline','candidate'):
        per_case, timings, drift = {}, [], []
        for case in cases:
            rows = [read(OUT/f'measurements/{name}-{case["id"]}-{r}.json') for r in range(frozen['repeats'])]
            if any(r['keys'] != rows[0]['keys'] for r in rows):
                drift.append(case['id'])
            metrics = {k:statistics.mean(r['metrics'][k] for r in rows) if case['answerable'] else None
                for k in ('recall_at_10','recall_at_8','mrr_at_24')}
            per_case[case['id']] = {**metrics,'keys':rows[0]['keys'],
                'median_seconds':statistics.median(r['seconds'] for r in rows),
                'rerank_median_seconds':statistics.median(r['rerank_seconds'] for r in rows)}
            timings.append(per_case[case['id']]['median_seconds'])
        results[name] = {'quality':quality_by_group(cases,per_case),'median_seconds':statistics.median(timings),
            'p95_seconds':sorted(timings)[math.ceil(.95*len(timings))-1],'ranking_drift':drift,'per_case':per_case}
    changes=[]
    for case in cases:
        old,new=(results[name]['per_case'][case['id']] for name in ('baseline','candidate'))
        changes.append({'id':case['id'],'question':case['question'],'order_identical':old['keys']==new['keys'],
            **{k:new[k]-old[k] if case['answerable'] else None for k in ('recall_at_10','recall_at_8','mrr_at_24')}})
    gate=adoption_gate(results['baseline'],results['candidate'])
    write('result.json', {'results':results,'gate':gate,'changes':changes,
        'median_reduction':1-results['candidate']['median_seconds']/results['baseline']['median_seconds'],
        'deepseek_requests':0,'limitations':['旧24题，不是未见材料泛化验收','标注非穷举','组件和非Agent端到端','只验证默认24候选，不宣称所有自定义候选数效果']})
    print('gate',gate,flush=True)
    for name,row in results.items():
        print(name,row['quality'],row['median_seconds'],row['p95_seconds'],flush=True)


def run() -> None:
    """只执行冻结的两个配置、五轮24题；出错留证后停止，无重试。"""
    frozen=verify()
    if not (OUT/'execution-freeze.json').exists():
        raise ValueError('EXECUTION_NOT_FROZEN')
    write('started.json', {'at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'deepseek_requests':0})
    budget=Budget(frozen['caps'])
    cases=read(OLD/'retrieval-cases.json'); corpus=read(OLD/'corpus.json')
    by_key={r['key']:r for r in corpus['chunks']}
    bags=[Counter(terms(r['text'])) for r in corpus['chunks']]
    vectors=VectorIndex('http://127.0.0.1:6339','researchloop_test_ag74_20261011')
    for doc in corpus['documents']:
        found=vectors.retrieve(doc['collection'],doc['chunk_keys'])
        if set(found) != set(doc['chunk_keys']) or any(found[k]['payload']['chunk_id'] != by_key[k]['chunk_id'] for k in found):
            raise ValueError('FROZEN_INDEX_CHANGED')
    embed=LocalEmbedding(MANIFEST)
    models={name:LocalReranker(MANIFEST,**frozen[name]) for name in ('baseline','candidate')}
    budget.book(embeddings=1)
    start=time.perf_counter(); embed.encode(['initialization only'],query=True)
    cold={'embedding':time.perf_counter()-start}
    for name,model in models.items():
        budget.book(pairs=1); start=time.perf_counter()
        model.score('initialization only',['A fixed sentence without an evaluation answer.'])
        cold[name]=time.perf_counter()-start
    write('warmup.json',{'seconds':cold,'model_identities':{n:m.identity for n,m in models.items()},'os_cache_cleared':False})
    for repeat in range(frozen['repeats']):
        ordered=cases[repeat:]+cases[:repeat]
        for i,case in enumerate(ordered):
            rows,common=common_query(case,corpus,embed,vectors,budget,bags)
            if len(rows)!=24:
                raise ValueError('CANDIDATE_COUNT_CHANGED')
            write(f'measurements/common-{case["id"]}-{repeat}.json',{'seconds':common,'keys':[r['key'] for r in rows]})
            order=['baseline','candidate'] if (repeat+i)%2==0 else ['candidate','baseline']
            for name in order:
                model=models[name]; budget.book(pairs=24); start=time.perf_counter()
                scores=model.score(case['question'],[by_key[r['key']]['text'] for r in rows])
                rerank_seconds=time.perf_counter()-start
                start=time.perf_counter(); keys=legacy_ranking(rows,scores); ranking_seconds=time.perf_counter()-start
                write(f'measurements/{name}-{case["id"]}-{repeat}.json',{'case':case['id'],'repeat':repeat,
                    'keys':keys,'scores':scores,'stats':dict(model.last_stats),'rerank_seconds':rerank_seconds,
                    'seconds':sum(common.values())+rerank_seconds+ranking_seconds,'metrics':metric(keys,case['relevant_chunks'])})
            print('repeat',repeat+1,case['id'],'done',flush=True)
    vectors.close()
    verify()
    write('completed.json',{'ledger':budget.value,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'wall_seconds':time.time()-budget.value['started_at']})
    report(cases,frozen)


def main():
    """只暴露冻结/一次运行入口，不提供自动恢复或追加批次。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze-code','run']); args=parser.parse_args()
    local_network_only()
    try:
        (freeze_code if args.stage=='freeze-code' else run)()
    except Exception as exc:
        target=OUT/('failure-'+args.stage+'.json')
        if not target.exists():
            write(target.name,{'type':type(exc).__name__,'detail':str(exc)})
        raise


if __name__=='__main__':
    main()
