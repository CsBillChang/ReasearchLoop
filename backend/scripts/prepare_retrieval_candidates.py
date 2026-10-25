"""363候选只读预检与独占准备；无运行、模型装配或网络入口。

360/361的历史代码hash与363当前代码分别封存；允许变更仅限本轮固定文件。
准备结果不能授权真实检索，也不包含新候选质量或速度测量。
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sqlite3

from research_loop.domain.retrieval_strategy import get_retrieval_strategy

SOURCE = 'artifacts/360-retrieval-corpus-preparation-20261023'
RUNTIME = 'artifacts/361-isolated-retrieval-runtime-20261023'
RESULTS = 'artifacts/362-retrieval-fourway-20261023'
CANDIDATES = 'artifacts/363-retrieval-candidates-20261023'
PROFILES = ('baseline', 'fusion_bm25_2_1', 'efficiency_length_sort')
CHANGED_CODE = (
    'backend/research_loop/storage/indexes.py',
    'backend/research_loop/services/retrieval.py',
    'backend/research_loop/providers/reranker.py',
    'backend/scripts/component_eval_adapters.py',
)
CHANGED_DOCS = ('docs/improvement-plan.md', 'docs/status.md', 'docs/file-map.md', 'docs/evaluation.md')
CURRENT_CODE = CHANGED_CODE + (
    'backend/research_loop/domain/retrieval_strategy.py',
    'backend/scripts/prepare_retrieval_candidates.py',
    'backend/tests/unit/test_rrf_candidate_strategy.py',
    'backend/tests/unit/test_reranker_batch_contract.py',
    'backend/tests/unit/test_prepare_retrieval_candidates.py',
    'backend/tests/integration/test_retrieval_candidate_isolation.py',
)
COUNTS = {'documents': 54, 'pages': 973, 'chunks': 3821, 'cases': 114,
          'answerable': 112, 'gap': 2, 'positive_pairs': 144,
          'positive_regression': 58, 'positive_pending': 54, 'independently_certified': 0}


def digest(path: Path) -> str:
    """分块只读计算hash，不载入模型或复制大文件。"""
    value = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path: Path):
    """读取显式JSON，不访问环境配置。"""
    return json.loads(path.read_text(encoding='utf-8'))


def reference(path: Path) -> dict:
    """生成可独立读回的文件身份。"""
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def verify_reference(row: dict, root: Path) -> Path:
    """验证仓库内的原封存文件，拒绝路径重定向与hash漂移。

    Raises:
        ValueError: 文件越界、含符号链接或身份不匹配。
    """
    path = Path(row['path'])
    if not path.is_absolute() or path != path.resolve() or not path.is_relative_to(root):
        raise ValueError('REFERENCE_OUTSIDE_ROOT_OR_REDIRECTED')
    if digest(path) != row['sha256']:
        raise ValueError('REFERENCE_HASH_MISMATCH: ' + str(path))
    if 'bytes' in row and path.stat().st_size != row['bytes']:
        raise ValueError('REFERENCE_SIZE_MISMATCH')
    return path


def validate_materials(root: Path, runtime: dict) -> dict:
    """重放冻结材料、标签与范围，仅计数/校验，不计算检索排名。"""
    folder = root / SOURCE
    questions = read(folder / 'questions-final-v2.json')
    chunks = read(folder / 'chunk-plan.json')
    indexes = read(folder / 'index-plan.json')
    qrels = read(folder / 'qrels-distinct.json')['positive_pairs']
    scope = read(folder / 'comparison-scope.json')
    documents = set(scope['corpus']['selected_document_ids'])
    ids = {row['chunk_id']: row for row in chunks}
    case_ids = {row['case_id'] for row in questions}
    pairs = {(row['case_id'], row['chunk_id']) for row in qrels}
    actual = {'documents': len(documents), 'pages': sum(len(row['manifest']['pages']) for row in indexes),
              'chunks': len(chunks), 'cases': len(questions),
              'answerable': sum(row['answerable'] is True for row in questions),
              'gap': sum(row['answerable'] is False for row in questions), 'positive_pairs': len(pairs),
              'positive_regression': sum(row['answerable'] is True and row['split'] == 'regression' for row in questions),
              'positive_pending': sum(row['answerable'] is True and row['split'] == 'pending' for row in questions),
              'independently_certified': sum(row.get('qualification', {}).get('independent_qualification') == 'certified' for row in questions)}
    if (actual != COUNTS or len(ids) != 3821 or len(indexes) != 54 or len(case_ids) != 114
            or len(qrels) != 144 or {c['document_id'] for c in chunks} != documents):
        raise ValueError('MATERIAL_COUNTS_OR_IDENTITY_CHANGED')
    for row in chunks:
        if sha256(row['text'].encode()).hexdigest() != row['text_sha256']:
            raise ValueError('CHUNK_TEXT_HASH_CHANGED')
    for case in questions:
        allowed = set(case['scope_document_ids'])
        selected = {cid for cid, row in ids.items() if row['document_id'] in allowed}
        gold = {cid for case_id, cid in pairs if case_id == case['case_id']}
        if (not allowed or not allowed <= documents or len(selected) != case['planned_scope_chunk_count']
                or (case['answerable'] and allowed != documents) or bool(gold) != case['answerable']
                or not gold <= selected):
            raise ValueError('CASE_SCOPE_OR_GOLD_CHANGED')
    if any(case_id not in case_ids or cid not in ids for case_id, cid in pairs):
        raise ValueError('QRELS_OUTSIDE_FROZEN_MATERIALS')
    if scope['algorithms']['candidate_limit'] != 24 or scope['algorithms']['rrf_k'] != 60:
        raise ValueError('FROZEN_ALGORITHM_CHANGED')
    if any(runtime['counts'].get(k) != actual[k] for k in ('documents', 'pages', 'chunks', 'cases', 'answerable', 'gap', 'positive_pairs')):
        raise ValueError('RUNTIME_COUNTS_CHANGED')
    return {'counts': actual, 'chunks': ids, 'indexes': indexes}


def validate_sqlite(path: Path, materials: dict) -> dict:
    """只读核对361的全部build、成员顺序、chunk正文与文档/页计数。"""
    expected = {row['index_id']: row['manifest'] for row in materials['indexes']}
    with sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA query_only=ON')
        builds = connection.execute('SELECT * FROM evidence_index_builds').fetchall()
        if len(builds) != 54 or {row['index_id'] for row in builds} != set(expected):
            raise ValueError('SQL_BUILDS_CHANGED')
        for row in builds:
            if (json.loads(row['manifest_json']) != expected[row['index_id']]
                    or row['bm25_status'] != 'ready' or row['vector_status'] != 'ready'):
                raise ValueError('SQL_BUILD_IDENTITY_CHANGED')
        members = connection.execute('SELECT m.*,c.chunk_json FROM evidence_index_members m JOIN evidence_chunks c ON c.chunk_id=m.chunk_id').fetchall()
        if len(members) != 3821 or len({row['chunk_id'] for row in members}) != 3821:
            raise ValueError('SQL_MEMBERS_CHANGED')
        for row in members:
            frozen = materials['chunks'][row['chunk_id']]
            body = json.loads(row['chunk_json'])
            if (row['index_id'] != frozen['index_id']
                    or expected[row['index_id']]['chunk_ids'][row['ordinal_no']] != row['chunk_id']
                    or any(body.get(k) != frozen[k] for k in ('chunk_id', 'document_id', 'evidence_id', 'source_sha256', 'text_sha256', 'page_number', 'char_start', 'char_end', 'text'))):
                raise ValueError('SQL_MEMBER_OR_CHUNK_IDENTITY_CHANGED')
        for table, count in (('eval_documents', 54), ('eval_pages', 973), ('evidence_chunks', 3821)):
            if connection.execute('SELECT count(*) FROM ' + table).fetchone()[0] != count:
                raise ValueError('SQL_COUNTS_CHANGED: ' + table)
    return {'mode': 'read_only_immutable', 'builds': 54, 'members': 3821, 'documents': 54, 'pages': 973}


def build_preparation(root: Path) -> dict:
    """只读预检封存材料、模型文件和当前实现，返回未执行清单。

    Args:
        root: 项目绝对根目录，所有输入均从固定相对位置读取。

    Returns:
        三配置与新旧身份分列的prepared_not_executed清单。

    Raises:
        ValueError: 原件身份、材料、依赖或只读索引检查不通过。
    """
    root = root.resolve(strict=True)
    runtime_path = root / RUNTIME / 'runtime/manifest-final.json'
    result_scope = read(root / RESULTS / 'scope.json')
    verify_reference(result_scope['runtime_manifest'], root)
    if Path(result_scope['runtime_manifest']['path']) != runtime_path:
        raise ValueError('RUNTIME_MANIFEST_PATH_CHANGED')
    runtime = read(runtime_path)
    before_path = root / CANDIDATES / 'audit/before-baseline.json'
    before = read(before_path)
    if before.get('complete') is not True or before.get('errors'):
        raise ValueError('COMPLETE_BEFORE_PROTECTION_REQUIRED')
    for relative in (SOURCE + '/freeze.json', RUNTIME + '/runtime/manifest-final.json',
                     RESULTS + '/scope.json', RESULTS + '/final-readback.json'):
        expected_hash = before['asset_hashes'].get(relative)
        if not expected_hash or digest(root / relative) != expected_hash:
            raise ValueError('BEFORE_SEALED_INPUT_CHANGED: ' + relative)
    historical = read(root / SOURCE / 'freeze.json')
    historical_code, allowed_drift = [], []
    for row in historical['files'] + runtime['source_files']:
        path = Path(row['path'])
        relative = str(path.relative_to(root))
        if relative in CHANGED_CODE + CHANGED_DOCS:
            if relative in CHANGED_CODE:
                backup = root / CANDIDATES / 'baseline-source' / relative
                if digest(backup) != row['sha256'] or before['source_hashes'].get(relative) != row['sha256']:
                    raise ValueError('BASELINE_SOURCE_COPY_MISMATCH: ' + relative)
                historical_code.append({**row, 'preserved_copy': reference(backup)})
            allowed_drift.append({'path': str(path), 'historical_sha256': row['sha256'], 'current_sha256': digest(path)})
        else:
            verify_reference(row, root)
    # 模型manifest与每个大文件hash始终重验；不导入tokenizers或ONNX。
    verify_reference(runtime['model_manifest'], root)
    for row in runtime['model_files']:
        verify_reference(row, root)
    actual_packages = {name: importlib.metadata.version(name) for name in runtime['packages']}
    if actual_packages != runtime['packages']:
        raise ValueError('FROZEN_PACKAGE_VERSION_CHANGED')
    snapshot = runtime['index_snapshot']
    sqlite_path = verify_reference(snapshot['sqlite'], root)
    report_path = verify_reference(snapshot['readback_report'], root)
    report = read(report_path)
    if report.get('complete') is not True or report.get('status') != 'passed' or report.get('counts') != {'documents': 54, 'pages': 973, 'chunks': 3821, 'members': 3821, 'qdrant_points': 3821}:
        raise ValueError('SEALED_FULL_POINT_READBACK_REQUIRED')
    materials = validate_materials(root, runtime)
    sql = validate_sqlite(sqlite_path, materials)
    verify_reference(snapshot['sqlite'], root)
    seal_path = root / RESULTS / 'final-readback.json'
    seal = read(seal_path)
    if seal.get('complete') is not True or seal.get('raw_files_sealed_and_independently_rehashed') != 2287:
        raise ValueError('BASELINE_RESULTS_NOT_SEALED')
    for row in seal['references']:
        verify_reference(row, root)
    raw_path = root / RESULTS / 'execution/raw-hashes.json'
    raw = read(raw_path)
    if len(raw) != 2287 or len({row['path'] for row in raw}) != 2287:
        raise ValueError('BASELINE_RAW_COUNTS_CHANGED')
    for row in raw:
        verify_reference(row, root)
    return {'schema': '363-retrieval-candidates-preparation-v1', 'stage': 'prepared_not_executed',
            'counts': materials['counts'], 'profiles': [get_retrieval_strategy(name).as_dict() for name in PROFILES],
            'historical_runtime_manifest': reference(runtime_path), 'source_freeze': reference(root / SOURCE / 'freeze.json'),
            'historical_code': historical_code, 'explicit_allowed_drift': allowed_drift,
            'current_code': [reference(root / name) for name in CURRENT_CODE],
            'before_protection': reference(before_path), 'model_manifest': runtime['model_manifest'],
            'model_files': runtime['model_files'], 'packages': actual_packages, 'index_snapshot': snapshot,
            'sqlite_readback': sql, 'qdrant': {'requeried': False, 'new_similarity_queries': 0,
                'identity_source': snapshot['readback_report'], 'note': '361 full-point independent readback reused; live Qdrant identity not refreshed'},
            'baseline_results_seal': reference(seal_path), 'baseline_results': seal['references'],
            'baseline_raw_manifest': reference(raw_path), 'baseline_raw_files_rehashed': len(raw),
            'execution': {'real_model_initialized': False, 'embedding_calls': 0, 'score_calls': 0,
                'vector_query_calls': 0, 'network_calls': 0, 'new_rankings_or_measurements': False},
            'limitations': ['Candidate Recall/MRR/latency unmeasured; synthetic scores do not prove semantic rank invariance',
                'Labels remain non-exhaustive; 58 regression / 54 pending / zero independently certified',
                'Future real comparison requires new authorization and same-batch baseline; this script has no run command']}


def prepare(root: Path, output: Path) -> dict:
    """预检后独占写入363/preparation，失败不产生完成标记。

    Raises:
        ValueError: 输出越界、重定向或指向保留原件。
        FileExistsError: 输出已存在，不覆盖任何历史记录。
    """
    root = root.resolve(strict=True)
    allowed = root / CANDIDATES / 'preparation'
    if (not output.is_absolute() or output != output.resolve() or not output.is_relative_to(allowed)
            or output == allowed or output.suffix != '.json'):
        raise ValueError('OUTPUT_OUTSIDE_CANDIDATE_PREPARATION')
    if output.exists():
        raise FileExistsError('OUTPUT_ALREADY_EXISTS')
    manifest = build_preparation(root)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    return manifest


def main(argv=None) -> int:
    """提供只读validate与独占prepare/freeze，不提供任何真实运行命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='只读验证原件、模型hash与SQLite，不加载模型')
    validate.add_argument('--manifest', type=Path, help='同时核对既有363准备清单与当前输入')
    for name in ('prepare', 'freeze'):
        command = commands.add_parser(name, help='独占准备三配置，始终不执行')
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'validate':
        value = build_preparation(args.root)
        if args.manifest is not None and read(args.manifest) != value:
            raise ValueError('PREPARATION_MANIFEST_CHANGED')
    else:
        value = prepare(args.root, args.output)
    print(json.dumps({'status': 'valid', 'stage': value['stage'], 'counts': value['counts'],
                      'execution': value['execution']}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
