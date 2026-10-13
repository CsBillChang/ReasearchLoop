"""固定六篇本地PDF的四路检索对照；标注先冻结，真实ONNX/Qdrant不调用生成模型。"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import resource
import re
import shutil
import statistics
import time
from research_loop.providers.embeddings import LocalEmbedding
from research_loop.providers.reranker import LocalReranker
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.services.indexing import IndexingService
from research_loop.storage.files import FileStore
from research_loop.storage.indexes import rank_terms, reciprocal_rank_fusion, terms
from research_loop.storage.mysql import Database
from research_loop.storage.vector_index import VectorIndex, point_id
from scripts.evaluation_common import ROOT, OUT, exclusive, settings, sha, ranked_metrics, verify_freeze


def prepare() -> None:
    """从既有AG6材料建立隔离索引，导出全部页/片段供盲于排名的人工标注。"""
    config = settings(); db = Database(config.database_url); db.ensure_schema()
    exclusive(OUT/'preparation/started.json', {'generative_calls': 0, 'network_discovery': 0})
    materials = json.loads((ROOT/'backend/tests/acceptance/ag6_cases.json').read_text())['materials']
    files = FileStore(config.data_dir, config.output_dir, ROOT)
    pdf = PDFProvider(files)
    papers = PaperService(db, None, pdf)
    task = 'ag74-fixed-retrieval-corpus'
    attachments = [{'input_ref': k, 'local_path': str(ROOT/v['path']), 'paper_id': v['paper_id']} for k,v in materials.items()]
    db.create_task(task, task, task, '读取固定本地原文用于检索对照，不搜索、不保存笔记。', 'deny', attachments)
    embedding = LocalEmbedding(ROOT/'data/ag71-preflight/models/manifest.json')
    vectors = VectorIndex(config.qdrant_url, config.retrieval_namespace)
    indexing = IndexingService(db, embedding, vectors)
    corpus, documents, measurements = [], [], []
    for key, value in materials.items():
        assert sha(ROOT/value['path']) == value['sha256']
        start = time.monotonic()
        # 现有缓存按PDF配置hash命名；缺缓存则仅本地解析，不走Jina或下载。
        import hashlib
        config_hash = hashlib.sha256(json.dumps(pdf.parser_config, sort_keys=True).encode()).hexdigest()
        cache = hashlib.sha256(f'{value["sha256"]}:pymupdf:1:{config_hash}'.encode()).hexdigest()+'.json'
        old = ROOT/'artifacts/ag6-evaluation-20261011/first-pass/data/cache/parsed'/cache
        if old.exists():
            shutil.copy2(old, files.parse_dir/cache)
        read = papers.read(task, key, attachments={a['input_ref']: a for a in attachments})
        refs = [p['evidence_id'] for p in read['pages']]
        built = indexing.build(task, read['document_id'], refs)
        view = indexing.authorized_chunks(task, built['index_id'])
        assert view['vector_status'] == view['bm25_status'] == 'ready'
        rows = [{**i['chunk'], 'key': point_id(built['index_id'], i['chunk']['chunk_id']),
                 'index_id': built['index_id'], 'material': key} for i in view['chunks']]
        corpus += rows
        doc = {'key':key, 'material':value, 'document_id':read['document_id'],
               'index_id':built['index_id'], 'collection':view['manifest']['collection'],
               'pages':read['pages'], 'chunk_keys':[r['key'] for r in rows]}
        documents.append(doc)
        elapsed = time.monotonic()-start
        measurements.append({'material':key,'seconds':elapsed,'chunks':len(rows),
                             'encoded_chunks':built['encoded_chunks']})
        exclusive(OUT/'preparation'/f'{key}-document.json', doc)
        print(key, len(refs), 'pages', len(rows), 'chunks', round(elapsed,2), 'seconds', flush=True)
    exclusive(OUT/'corpus.json', {'task_id':task,'documents':documents,'chunks':corpus,
        'embedding':embedding.identity,'build':measurements,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'rss_platform':'macOS ru_maxrss bytes', 'generative_calls':0})
    pdf.close(); vectors.close()


def annotate() -> None:
    """把事前原句标注绑定到原字符区间；跨块原句保留全部相交片段。

    Raises:
        ValueError: 原句非唯一、片段未覆盖原句或已存在标注产物。
    """
    corpus=json.loads((OUT/'corpus.json').read_text())
    spec=json.loads((ROOT/'backend/tests/acceptance/ag74_cases.json').read_text())
    cases=[]
    for case in spec['retrieval']:
        annotations=[];relevant=[]
        for material,page,quote in case['anchors']:
            doc=next(d for d in corpus['documents'] if d['key']==material)
            source=next(p for p in doc['pages'] if p['page_number']==page)
            matches=list(re.finditer(r'\s+'.join(re.escape(w) for w in quote.split()),source['text']))
            if len(matches)!=1:raise ValueError('NON_UNIQUE_GOLD_QUOTE: '+case['id'])
            start,end=matches[0].span()
            chunks=[c for c in corpus['chunks'] if c['material']==material and c['page_number']==page
                    and c['char_start']<end and c['char_end']>start]
            covered=start
            for chunk in sorted(chunks,key=lambda c:c['char_start']):
                if chunk['char_start']>covered:raise ValueError('UNCOVERED_GOLD_SPAN: '+case['id'])
                covered=max(covered,chunk['char_end'])
            if covered<end:raise ValueError('UNCOVERED_GOLD_SPAN: '+case['id'])
            keys=[c['key'] for c in chunks];relevant+=keys
            annotations.append({'material':material,'page':page,'quote':source['text'][start:end],
                'char_start':start,'char_end':end,'evidence_id':source['evidence_id'],'chunk_keys':keys})
        cases.append({**case,'materials':[d['key'] for d in corpus['documents']],
            'answerable':bool(case['anchors']),'relevant_chunks':list(dict.fromkeys(relevant)),
            'source_annotations':annotations})
    exclusive(OUT/'retrieval-cases.json',cases)
    print('frozen-label preparation:',len(cases),'questions;',sum(c['answerable'] for c in cases),'answerable')


def run() -> None:
    """冻结后单次执行；四模式共享通道中间结果，计时按实际组件求和，不调参重跑。"""
    frozen = verify_freeze(); cases = json.loads((OUT/'retrieval-cases.json').read_text())
    corpus = json.loads((OUT/'corpus.json').read_text()); config = settings()
    exclusive(OUT/'retrieval/started.json', {'cases':len(cases),'modes':frozen['retrieval']['modes']})
    embedding = LocalEmbedding(ROOT/'data/ag71-preflight/models/manifest.json')
    reranker = LocalReranker(ROOT/'data/ag71-preflight/models/manifest.json')
    vectors = VectorIndex(config.qdrant_url, config.retrieval_namespace)
    db = Database(config.database_url); indexing = IndexingService(db, embedding, vectors)
    # 每个版本重新验证权威来源与权限；计时不把这份共享校验重复算作各通道模型成本。
    for document in corpus['documents']:
        current = indexing.authorized_chunks(corpus['task_id'], document['index_id'])
        expected = [r for r in corpus['chunks'] if r['material']==document['key']]
        assert [i['chunk']['text'] for i in current['chunks']] == [r['text'] for r in expected]
        points = vectors.retrieve(document['collection'], document['chunk_keys'])
        for row in expected:
            assert points[row['key']]['payload']['chunk_id'] == row['chunk_id']
            assert points[row['key']]['payload']['index_id'] == row['index_id']
    cold = time.monotonic()
    embedding.encode(['model initialization only'], query=True)
    reranker.score('model initialization only',['A fixed warm-up sentence with no evaluation answer.'])
    cold = time.monotonic()-cold
    output=[]
    for case in cases:
        records = [c for c in corpus['chunks'] if c['material'] in case['materials']]
        by_key = {r['key']:r for r in records}; allkeys=set(by_key)
        bags = [Counter(terms(r['text'])) for r in records]
        start = time.perf_counter()
        bm25 = rank_terms(case['question'], records, bags, 24)
        bm25_time = time.perf_counter()-start
        start = time.perf_counter()
        query = embedding.encode([case['question']], query=True)[0]
        vector = []
        for doc in corpus['documents']:
            if doc['key'] not in case['materials']: continue
            hits = vectors.query(doc['collection'],query,doc['chunk_keys'],24)
            assert len(hits) == min(24,len(doc['chunk_keys']))
            for hit in hits:
                assert str(hit['id']) in allkeys and hit['payload']['index_id'] == doc['index_id']
                vector.append({'key':str(hit['id']),'score':hit['score']})
        vector = sorted(vector,key=lambda r:(-r['score'],r['key']))[:24]
        vector_time = time.perf_counter()-start
        start = time.perf_counter()
        fused = reciprocal_rank_fusion({'bm25':bm25,'vector':vector})[:24]
        fusion_time = time.perf_counter()-start
        start = time.perf_counter(); error = None; reordered = list(fused)
        try:
            scores = reranker.score(case['question'],[by_key[r['key']]['text'] for r in fused])
            if len(scores)!=len(fused):raise ValueError('RERANK_SCORE_COUNT')
            reordered = [{**r,'reranker_score':s,'rrf_rank':i} for i,(r,s) in enumerate(zip(fused,scores),1)]
            reordered.sort(key=lambda r:(-r['reranker_score'],r['rrf_rank']))
        except Exception as exc:
            # 真实超容量等降级保留，不截掉文本后再跑到提升；原始RRF是产品一致的回退。
            error = str(exc) if isinstance(exc,ValueError) else type(exc).__name__
        rerank_time=time.perf_counter()-start
        modes={'bm25':bm25,'vector':vector,'rrf':fused,'rrf_rerank':reordered}
        times={'bm25':bm25_time,'vector':vector_time,'rrf':bm25_time+vector_time+fusion_time,
               'rrf_rerank':bm25_time+vector_time+fusion_time+rerank_time}
        results={mode:{'metrics':ranked_metrics([r['key'] for r in rows],case['relevant_chunks'],answerable=case['answerable'],corpus=allkeys),
                       'seconds':times[mode], 'ranking':[{k:v for k,v in r.items() if k in {'key','score','channels','rrf_score','reranker_score','rrf_rank'}} for r in rows]}
                 for mode,rows in modes.items()}
        row={'id':case['id'],'results':results,'reranker_error':error,'question':case['question'],
             'answerable':case['answerable'],'language':case['language']}
        exclusive(OUT/'retrieval'/f'{case["id"]}.json',row);output.append(row)
        print(case['id'], {m:round(v['metrics']['recall_at_10'],3) if v['metrics']['recall_at_10'] is not None else None for m,v in results.items()},error,flush=True)
    summary={}
    for mode in modes:
        rows=[r['results'][mode] for r in output]; valid=[r for r in rows if r['metrics']['recall_at_10'] is not None]
        times=sorted(r['seconds'] for r in rows)
        summary[mode]={'answerable_count':len(valid),'missing_count':len(rows)-len(valid),
            'macro_recall_at_10':statistics.mean(r['metrics']['recall_at_10'] for r in valid),
            'mrr_at_24':statistics.mean(r['metrics']['reciprocal_rank_at_24'] for r in valid),
            'median_seconds':statistics.median(times),'p95_seconds':times[__import__('math').ceil(.95*len(times))-1]}
        summary[mode]['language']={language:{'n':len(selected),'macro_recall_at_10':statistics.mean(r['results'][mode]['metrics']['recall_at_10'] for r in selected)}
            for language in ['zh','en'] if (selected:=[r for r in output if r['answerable'] and r['language']==language])}
    exclusive(OUT/'retrieval/result.json',{'summary':summary,'question_count':len(output),
        'reranker_degraded_cases':[r['id'] for r in output if r['reranker_error']],
        'cold_model_load_seconds':cold,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'embedding':embedding.identity,'reranker':reranker.identity,'generative_requests':0,
        'timing':'one warm pass; component sums, not independent end-to-end task latency; Qdrant server RSS excluded',
        'qrels':'preselected source spans; incomplete relevance judgments; no tuning on report cases'})
    vectors.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','annotate','run'])
    {'prepare':prepare,'annotate':annotate,'run':run}[parser.parse_args().mode]()
