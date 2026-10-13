"""AG7.1隔离真实验收：已有PDF/缓存、本地编码、MySQL/Qdrant与SIGKILL，不调用LLM。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse
import uuid

from research_loop.domain.retrieval import ChunkConfig, fingerprint
from research_loop.providers.embeddings import LocalEmbedding
from research_loop.providers.pdf import PDFProvider
from research_loop.services.indexing import IndexingService
from research_loop.services.papers import PaperService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.storage.vector_index import VectorIndex


def save(path: Path, value) -> None:
    """新证据用独占创建，禁止覆盖之前失败/通过记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def database() -> Database:
    """只允许显式AG7.1测试库，绝不回退生产URL。"""
    url = os.environ.get('AG71_TEST_DATABASE_URL', '')
    parsed = urlparse(url)
    if parsed.scheme != 'mysql' or not parsed.path.startswith('/researchloop_test_ag71_'):
        raise ValueError('explicit researchloop_test_ag71_ database required')
    return Database(url)


def kill_child(config: dict) -> None:
    """真实upsert成功后立即终止本进程；故障注入不进入生产服务接口。"""
    class KillAfterWrite(VectorIndex):
        """只用于验收的真实写后SIGKILL适配器。"""
        def upsert(self, collection, points):
            """先完成wait=true真实写入，再留下标记并触发不可捕获信号。"""
            super().upsert(collection, points)
            save(Path(config['kill_marker']), {'pid': os.getpid(), 'points': len(points),
                'collection': collection, 'index_id': points[0]['payload']['index_id']})
            os.kill(os.getpid(), signal.SIGKILL)
    vectors = KillAfterWrite(config['qdrant_url'], config['namespace'])
    IndexingService(database(), LocalEmbedding(config['model_manifest']), vectors).build(
        config['task_id'], config['document_id'], config['evidence_ids'])
    raise AssertionError('SIGKILL not reached')


def run(args) -> None:
    """执行固定两篇原文的组件验收，并保存逐项断言与最终读回。"""
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    db = database()
    namespace = 'researchloop_test_ag71_' + uuid.uuid4().hex[:12]
    model_manifest = args.model_manifest.resolve()
    cases = json.loads((root/'backend/tests/acceptance/ag6_cases.json').read_text())['materials']
    selected = {key: cases[key] for key in ['od1', 'mllm1']}
    task_id = 'ag71-' + uuid.uuid4().hex
    source_cache = root/'artifacts/ag6-evaluation-20261011/first-pass/data/cache'
    data = out/'data'
    files = FileStore(data, out/'unused-notes', root)
    provider = PDFProvider(files)
    papers = PaperService(db, None, provider)
    attachments = [{'input_ref': k, 'local_path': str(root/v['path']), 'paper_id': v['paper_id']}
                   for k, v in selected.items()]
    save(out/'freeze.json', {'task_id': task_id, 'database': urlparse(db.url).path.lstrip('/'),
        'namespace': namespace, 'model_manifest_sha256': hashlib.sha256(model_manifest.read_bytes()).hexdigest(),
        'materials': {k: {x: v[x] for x in ('paper_id', 'path', 'sha256', 'pages')} for k,v in selected.items()},
        'generative_model_requests': 0, 'fault': 'SIGKILL after first real Qdrant upsert',
        'source_files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('backend/research_loop', 'backend/scripts') for p in (root/folder).rglob('*.py')}})
    db.create_task(task_id, task_id, fingerprint(attachments), '读取两篇原文用于本机证据索引', 'deny', attachments)
    documents = []
    for key, value in selected.items():
        original = root/value['path']
        assert hashlib.sha256(original.read_bytes()).hexdigest() == value['sha256']
        config_hash = hashlib.sha256(json.dumps(provider.parser_config, sort_keys=True).encode()).hexdigest()
        cache_key = hashlib.sha256(f'{value["sha256"]}:pymupdf:1:{config_hash}'.encode()).hexdigest()
        # 直接复用已验证提取结果；绝不下载论文或重跑精读/模型。
        shutil.copy2(source_cache/'parsed'/(cache_key+'.json'), files.parse_dir/(cache_key+'.json'))
        result = papers.read(task_id, key, attachments={a['input_ref']: a for a in attachments})
        assert len(result['pages']) == value['pages']
        documents.append({'key': key, 'document_id': result['document_id'],
            'evidence_ids': [p['evidence_id'] for p in result['pages']]})
    provider.close()
    save(out/'input-documents.json', documents)
    killed = documents[1]
    config = {**killed, 'task_id': task_id, 'model_manifest': str(model_manifest),
        'namespace': namespace, 'qdrant_url': args.qdrant_url, 'kill_marker': str(out/'kill-marker.json')}
    save(out/'kill-input.json', config)
    with (out/'kill-child.log').open('x') as log:
        child = subprocess.run([sys.executable, __file__, '--kill-input', str(out/'kill-input.json')],
            stdout=log, stderr=subprocess.STDOUT, timeout=180)
    assert child.returncode == -signal.SIGKILL, f'child exit {child.returncode}'
    marker = json.loads((out/'kill-marker.json').read_text())
    vectors = VectorIndex(args.qdrant_url, namespace)
    embedding = LocalEmbedding(model_manifest)
    service = IndexingService(db, embedding, vectors)
    interrupted = service.store.get(marker['index_id'])
    receipt_count = db.fetchone('SELECT COUNT(*) AS n FROM evidence_index_members WHERE index_id=? AND embedding_json IS NOT NULL', (marker['index_id'],))['n']
    assert interrupted['vector_status'] == 'partial'
    assert receipt_count == marker['points'] == vectors.count(marker['collection'], marker['index_id']) == 4
    save(out/'interrupted.json', {'exit_code': child.returncode, 'vector_receipts': receipt_count,
        'qdrant_points': marker['points'], 'build': interrupted})
    started = time.monotonic()
    results = []
    for doc in documents:
        result = service.build(task_id, doc['document_id'], doc['evidence_ids'])
        save(out/(doc['key']+'-build.json'), result)
        assert result['vector_status'] == 'ready', result.get('last_error')
        repeated = service.build(task_id, doc['document_id'], doc['evidence_ids'][::-1])
        assert repeated['index_id'] == result['index_id']
        assert repeated['encoded_chunks'] == repeated['upserted_points'] == 0
        authorized = service.authorized_chunks(task_id, result['index_id'])
        coverage = {}
        for item in authorized['chunks']:
            chunk = item['chunk']
            page = db.get_page(chunk['evidence_id'])
            assert chunk['text'] == page['text'][chunk['char_start']:chunk['char_end']]
            span = set(range(chunk['char_start'], chunk['char_end']))
            occupied = coverage.setdefault(chunk['evidence_id'], set())
            assert not occupied & span
            occupied.update(span)
        for evidence_id in doc['evidence_ids']:
            page = db.get_page(evidence_id)
            assert all(i in coverage.get(evidence_id, set()) for i, c in enumerate(page['text']) if not c.isspace())
        if doc == killed:
            assert result['encoded_chunks'] + receipt_count == result['chunk_count']
            assert result['upserted_points'] + marker['points'] == result['chunk_count']
        save(out/(doc['key']+'-authorized-chunks.json'), authorized)
        results.append({'key': doc['key'], 'pages': len(doc['evidence_ids']), 'index_id': result['index_id'],
            'chunks': result['chunk_count'], 'encoded_this_process': result['encoded_chunks'],
            'repeat_encoded': repeated['encoded_chunks'], 'repeat_upserted': repeated['upserted_points'],
            'non_whitespace_coverage': 'complete', 'offsets_verified': True})
    # 正式API不暴露建库脚本；实际服务跨任务读回必须拒绝，缓存存在不等于授权。
    stranger = task_id+'-stranger'
    db.create_task(stranger, stranger, stranger, '阅读', 'deny', [])
    try:
        service.authorized_chunks(stranger, results[0]['index_id'])
    except PermissionError:
        denied = True
    else:
        raise AssertionError('cross-task evidence leaked')
    index_id = results[0]['index_id']
    with service.store.lock(index_id):
        contender = subprocess.run([sys.executable, '-c',
            'import os; from research_loop.storage.mysql import Database; from research_loop.storage.evidence_index import EvidenceIndexStore; '
            f's=EvidenceIndexStore(Database(os.environ["AG71_TEST_DATABASE_URL"]));\nwith s.lock({index_id!r}): print("unexpected")'],
            capture_output=True, text=True, timeout=30)
        assert contender.returncode != 0 and 'INDEX_BUILD_BUSY' in contender.stderr
    assert db.fetchone('SELECT COUNT(*) AS n FROM model_calls')['n'] == 0
    assert not list((out/'unused-notes').rglob('*.md'))
    report = {'passed': True, 'cases': results, 'sigkill_recovery': True,
        'sigkill_receipts_reused': receipt_count, 'duplicate_points': 0,
        'cross_task_denied': denied, 'mysql_concurrent_lock_rejected': True,
        'local_build_seconds': round(time.monotonic()-started, 3),
        'generative_model_requests': 0, 'business_notes_written': 0,
        'database': urlparse(db.url).path.lstrip('/'), 'namespace': namespace,
        'qdrant_collection': marker['collection'], 'qdrant_total_points': vectors.count(marker['collection'], index_id)
            + vectors.count(marker['collection'], marker['index_id'])}
    save(out/'result.json', report)
    vectors.close()
    print(json.dumps(report, ensure_ascii=False))


def main() -> None:
    """选择正常验收或内部故障子进程；两者均要求显式隔离库。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--qdrant-url', default='http://127.0.0.1:6339')
    parser.add_argument('--kill-input', type=Path)
    args = parser.parse_args()
    if args.kill_input:
        kill_child(json.loads(args.kill_input.read_text()))
    else:
        if not args.model_manifest or not args.output:
            parser.error('--model-manifest and --output required')
        run(args)


if __name__ == '__main__':
    main()
