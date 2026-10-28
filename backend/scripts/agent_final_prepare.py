"""将已封存的24题设计编译为独立轮次输入，仅复制本机许可材料。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


SUITE_KIND = 'agent_final_v1'
EXPECTED_CATEGORIES = {'QA': 5, 'NUM': 4, 'CROSS': 4, 'REC': 3, 'REV': 3, 'MEM': 3, 'DEEP': 2}


def digest_bytes(value: bytes) -> str:
    """返回原字节身份，不用文件名代替内容。"""
    return hashlib.sha256(value).hexdigest()


def _read(path: Path):
    """读取本地JSON，不触发配置加载。"""
    return json.loads(path.read_text())


def _write_new(path: Path, value) -> None:
    """仅创建新输出，不覆盖前一批结果。"""
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write('\n')


def verify_seal(root: Path) -> dict:
    """验证来源封存清单，拒绝越界、链接、缺失和额外文件。

    Args:
        root: 已封存的选题包根目录。

    Returns:
        已核对的封存回执。

    Raises:
        PermissionError: 来源文件或清单身份不符。
    """
    root = Path(root).resolve(strict=True)
    seal_path = root / 'SEAL.json'
    if seal_path.is_symlink():
        raise PermissionError('FINAL_PREPARE_SEAL_SYMLINK')
    seal = _read(seal_path)
    entries = seal.get('files')
    if not isinstance(entries, dict) or not entries or seal.get('file_count') != len(entries):
        raise PermissionError('FINAL_PREPARE_SEAL_INVALID')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(entries) | {'SEAL.json'}:
        raise PermissionError('FINAL_PREPARE_SEAL_FILE_SET_CHANGED')
    for name, receipt in entries.items():
        relative = Path(name)
        candidate = root / relative
        if (relative.is_absolute() or '..' in relative.parts or candidate.is_symlink()
                or not candidate.resolve(strict=True).is_relative_to(root)):
            raise PermissionError('FINAL_PREPARE_SEAL_PATH_INVALID')
        content = candidate.read_bytes()
        if len(content) != receipt.get('bytes') or digest_bytes(content) != receipt.get('sha256'):
            raise PermissionError('FINAL_PREPARE_SEALED_FILE_CHANGED')
    canonical = json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()
    if digest_bytes(canonical) != seal.get('manifest_sha256'):
        raise PermissionError('FINAL_PREPARE_SEAL_MANIFEST_CHANGED')
    return seal


def _source(path, expected_hash: str) -> tuple[Path, bytes]:
    """读取明确材料，不接受文件链接或漂移版本。"""
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PermissionError('FINAL_PREPARE_SOURCE_PATH_INVALID')
    resolved = path.resolve(strict=True)
    if any(parent.is_symlink() for parent in path.parents if parent.exists()):
        raise PermissionError('FINAL_PREPARE_SOURCE_PARENT_SYMLINK')
    content = path.read_bytes()
    if (not isinstance(expected_hash, str) or len(expected_hash) != 64
            or digest_bytes(content) != expected_hash):
        raise PermissionError('FINAL_PREPARE_SOURCE_HASH_CHANGED')
    return resolved, content


def _material(row: dict, output: Path) -> dict:
    """只将输入身份与副本路径发送给运行器。"""
    return {'material_id': row['material_id'], 'title': row['title'],
            'pdf_path': str(output / 'materials/pdf' / Path(row['pdf_path']).name),
            'pdf_sha256': row['pdf_sha256'], 'physical_pages': row['physical_pages'],
            'authorized_physical_pages': list(range(1, row['physical_pages'] + 1))}


def prepare(design_root, output, *, allowed_output_root, protected_roots=()) -> dict:
    """生成48个轮次任务及96格顺序，运行放行始终关闭。

    Args:
        design_root: 已封存选题包。
        output: 尚不存在的新材料快照目录。
        allowed_output_root: 调用者明确授予的实验输出父根。
        protected_roots: 部署、旧证据等不得写入的额外根目录。

    Returns:
        输出文件身份与计数；不包含执行授权。

    Raises:
        PermissionError: 路径、材料、分母或来源合同不符。
        FileExistsError: 输出已存在，禁止覆盖或继续旧准备。
    """
    raw_design = Path(design_root)
    raw_output = Path(output)
    if raw_design.is_symlink() or raw_output.is_symlink():
        raise PermissionError('FINAL_PREPARE_ROOT_SYMLINK')
    design = raw_design.resolve(strict=True)
    allowed = Path(allowed_output_root).resolve(strict=True)
    target = raw_output.resolve()
    protected = [design, Path(__file__).resolve().parents[2],
                 *(Path(p).resolve() for p in protected_roots)]
    if (not target.is_relative_to(allowed) or target == allowed
            or any(target.is_relative_to(p) or p.is_relative_to(target) for p in protected)):
        raise PermissionError('FINAL_PREPARE_OUTPUT_OUTSIDE_SCOPE')
    # resolve后的目标相同也拒绝链接组件，避免检查后以别名重定向写入。
    if any(parent.is_symlink() for parent in raw_output.parents if parent.exists()):
        raise PermissionError('FINAL_PREPARE_OUTPUT_PARENT_SYMLINK')
    if target.exists():
        raise FileExistsError('FINAL_PREPARE_OUTPUT_MUST_BE_NEW')
    if not target.parent.is_dir():
        raise PermissionError('FINAL_PREPARE_OUTPUT_PARENT_MISSING')
    seal = verify_seal(design)
    source_design = _read(design / 'cases/inputs.json')
    if (source_design.get('schema_version') != 'agent_final_eval_design_v1'
            or source_design.get('execute_authorized') is not False):
        raise PermissionError('FINAL_PREPARE_DESIGN_CONTRACT_INVALID')
    original_cases = source_design['cases']
    corpus = _read(design / 'materials/corpus-54.json')
    catalog = _read(design / 'materials/catalog-54.json')
    order = _read(design / 'ORDER.json')['order']
    if len(corpus) != 54 or len(catalog) != 54 or len(original_cases) != 24 or len(order) != 96:
        raise PermissionError('FINAL_PREPARE_FIXED_COUNTS_REQUIRED')
    by_material = {r['material_id']: r for r in corpus}
    by_case = {c['case_id']: c for c in original_cases}
    if (set(by_material) != {f'P{i:02d}' for i in range(1, 55)} or len({r['pdf_sha256'] for r in corpus}) != 54
            or len(by_case) != 24 or {r['material_id'] for r in catalog} != set(by_material)):
        raise PermissionError('FINAL_PREPARE_DUPLICATE_IDENTITY')
    from collections import Counter
    if Counter(c['case_id'].split('-')[1] for c in original_cases) != EXPECTED_CATEGORIES:
        raise PermissionError('FINAL_PREPARE_CATEGORY_COUNTS_CHANGED')
    if set(by_case) != {f'AFE-{kind}-{i:02d}' for kind, count in EXPECTED_CATEGORIES.items()
                       for i in range(1, count + 1)}:
        raise PermissionError('FINAL_PREPARE_CASE_IDS_CHANGED')
    payloads = []
    targets = set()
    for row in corpus:
        if (not isinstance(row.get('title'), str) or not row['title'].strip()
                or type(row['physical_pages']) is not int or row['physical_pages'] < 1
                or row['authorized_physical_pages'] != list(range(1, row['physical_pages'] + 1))):
            raise PermissionError('FINAL_PREPARE_MATERIAL_PAGE_SCOPE_CHANGED')
        for path_key, hash_key, folder in [('pdf_path', 'pdf_sha256', 'pdf'),
                                           ('parsed_path', 'parsed_cache_sha256', 'parsed')]:
            source, content = _source(row[path_key], row[hash_key])
            if target.is_relative_to(source.parent) or source.is_relative_to(target):
                raise PermissionError('FINAL_PREPARE_OUTPUT_OVERLAPS_ORIGINAL_MATERIAL')
            destination = target / 'materials' / folder / source.name
            if destination in targets:
                raise PermissionError('FINAL_PREPARE_DUPLICATE_COPY_TARGET')
            targets.add(destination)
            payloads.append((source, destination, row[hash_key], content))
    runtime_materials = {m: _material(row, target) for m, row in by_material.items()}
    runtime_catalog = []
    for i, entry in enumerate(catalog):
        row = by_material[entry['material_id']]
        if (entry.get('pdf_sha256') != row['pdf_sha256'] or entry.get('title') != row['title']
                or not isinstance(entry.get('excerpt'), str) or not entry['excerpt'].strip()
                or digest_bytes(entry['excerpt'].encode()) != entry.get('excerpt_sha256')):
            raise PermissionError('FINAL_PREPARE_CATALOG_IDENTITY_CHANGED')
        authors = entry.get('authors') or []
        if not isinstance(authors, list) or any(not isinstance(a, str) for a in authors):
            raise PermissionError('FINAL_PREPARE_CATALOG_AUTHORS_INVALID')
        runtime_catalog.append({'material_index': i, 'title': row['title'],
            'abstract': entry['excerpt'], 'authors': deepcopy(authors),
            'source_url': entry.get('source_url') or '',
            'source': entry.get('source') or 'local',
            'metadata_scope': 'frozen_local_pdf_excerpt; publication_date_unverified'})
    ordered_materials = [runtime_materials[r['material_id']] for r in catalog]
    runtime_cases = []
    lineage = []
    for c in original_cases:
        pre = c['preconditions']
        if set(pre) != {'save_policy', 'history_messages', 'catalog_ref'}:
            raise PermissionError('FINAL_PREPARE_PRECONDITION_FIELDS_CHANGED')
        is_catalog = c['case_id'].startswith('AFE-REC-')
        history = pre['history_messages']
        is_memory = c['case_id'].startswith('AFE-MEM-')
        if (not isinstance(c.get('prompt'), str) or not c['prompt'].strip()
                or pre['save_policy'] not in {'allow', 'deny'}
                or (pre['save_policy'] == 'allow') != c['case_id'].startswith('AFE-DEEP-')
                or not isinstance(history, list) or len(history) > 16
                or bool(history) != is_memory):
            raise PermissionError('FINAL_PREPARE_INPUT_SCOPE_CHANGED')
        for message in history:
            if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                    or message['role'] not in {'user', 'assistant'}
                    or not isinstance(message['content'], str) or not message['content'].strip()
                    or len(message['content']) > 20000):
                raise PermissionError('FINAL_PREPARE_HISTORY_INVALID')
        if bool(pre['catalog_ref']) != is_catalog or (is_catalog and pre['catalog_ref'] != 'materials/catalog-54.json'):
            raise PermissionError('FINAL_PREPARE_CATALOG_SCOPE_CHANGED')
        materials = []
        for m in c['materials']:
            original = by_material.get(m['material_id'])
            if (not original or any(m[k] != original[k] for k in ['title', 'pdf_path', 'pdf_sha256', 'physical_pages', 'authorized_physical_pages'])):
                raise PermissionError('FINAL_PREPARE_CASE_MATERIAL_CHANGED')
            materials.append(runtime_materials[m['material_id']])
        if is_catalog and (materials or pre['history_messages']):
            raise PermissionError('FINAL_PREPARE_CATALOG_MIXED_INPUT')
        if is_memory and materials:
            raise PermissionError('FINAL_PREPARE_MEMORY_MATERIAL_SCOPE_CHANGED')
        for round_no in (1, 2):
            runtime_id = c['case_id'] + '-R' + str(round_no)
            runtime_cases.append({'case_id': runtime_id, 'suite_kind': SUITE_KIND,
                'prompt': c['prompt'], 'materials': deepcopy(ordered_materials if is_catalog else materials),
                'preconditions': {'catalog': deepcopy(runtime_catalog) if is_catalog else [],
                    'history_messages': deepcopy(pre['history_messages']), 'save_policy': pre['save_policy']}})
            lineage.append({'runtime_case_id': runtime_id, 'original_case_id': c['case_id'], 'round': round_no,
                            'category': c['category'], 'title': c['title']})
    execution_order = []
    cells = []
    for i, slot in enumerate(order, 1):
        cid, round_no, variant = slot['case_id'], slot['round'], slot['variant']
        if (cid not in by_case or type(round_no) is not int or round_no not in {1, 2}
                or variant not in {'A', 'B'} or type(slot['ordinal']) is not int or slot['ordinal'] != i
                or slot.get('status') != 'not_created'
                or any(slot.get(k) is not None for k in ['model_calls', 'actual_tokens', 'end_to_end_seconds'])):
            raise PermissionError('FINAL_PREPARE_ORDER_IDENTITY_CHANGED')
        runtime_id = cid + '-R' + str(round_no)
        cell_id = runtime_id + '-' + variant
        if slot['cell_id'] != cell_id:
            raise PermissionError('FINAL_PREPARE_CELL_IDENTITY_CHANGED')
        execution_order.append([runtime_id, variant])
        cells.append({'ordinal': i, 'case_id': runtime_id, 'original_case_id': cid,
                      'round': round_no, 'variant': variant, 'cell_id': cell_id,
                      'status': 'not_created'})
    if (len({tuple(row) for row in execution_order}) != 96
            or Counter(v for _, v in execution_order) != {'A': 48, 'B': 48}
            or {tuple(row) for row in execution_order} != {(c['case_id'], v) for c in runtime_cases for v in ['A', 'B']}):
        raise PermissionError('FINAL_PREPARE_MATRIX_IDENTITY_CHANGED')
    for i in range(0, 96, 2):
        left, right = execution_order[i:i+2]
        if left[0] != right[0] or left[1] == right[1]:
            raise PermissionError('FINAL_PREPARE_PAIR_ADJACENCY_CHANGED')
    # 在任何写入前完成所有来源核对；中途I/O失败只留下新目录，不回写来源。
    target.mkdir(exist_ok=False)
    (target / 'materials/pdf').mkdir(parents=True)
    (target / 'materials/parsed').mkdir()
    copies = []
    for source, destination, expected, content in payloads:
        with destination.open('xb') as handle:
            handle.write(content)
        if (destination.is_symlink() or destination.stat().st_nlink != 1
                or (source.stat().st_dev, source.stat().st_ino) == (destination.stat().st_dev, destination.stat().st_ino)
                or digest_bytes(destination.read_bytes()) != expected):
            raise PermissionError('FINAL_PREPARE_COPY_IDENTITY_CHANGED')
        copies.append({'source': str(source), 'copy': str(destination), 'sha256': expected,
                       'bytes': len(content), 'independent_inode': True})
    case_output = {'schema_version': 'agent_final_runtime_v1', 'suite_kind': SUITE_KIND,
                   'execute_authorized': False, 'cases': runtime_cases}
    plan_output = {'suite_kind': SUITE_KIND, 'original_tasks': 24, 'runtime_cases': 48,
                   'roots': 96, 'execute_authorized': False, 'execution_order': execution_order,
                   'allowed_cells': deepcopy(execution_order), 'lineage': lineage, 'cells': cells,
                   'source_seal_manifest_sha256': seal['manifest_sha256']}
    _write_new(target / 'cases.json', case_output)
    _write_new(target / 'execution-plan.json', plan_output)
    _write_new(target / 'material-snapshot.json', {'copies': copies,
        'materials': [{**deepcopy(row), 'original_pdf_path': row['pdf_path'], 'original_parsed_path': row['parsed_path'],
            'pdf_path': runtime_materials[row['material_id']]['pdf_path'],
            'parsed_path': str(target / 'materials/parsed' / Path(row['parsed_path']).name)} for row in corpus],
        'source_seal_manifest_sha256': seal['manifest_sha256'], 'model_calls': 0})
    result = {'status': 'prepared_not_authorized', 'root': str(target), 'cases_path': str(target / 'cases.json'),
              'execution_plan_path': str(target / 'execution-plan.json'), 'pdf_count': 54,
              'parsed_count': 54, 'runtime_case_count': 48, 'root_count': 96,
              'real_model_calls': 0, 'real_execution_released': False,
              'files': {str(p.relative_to(target)): digest_bytes(p.read_bytes())
                        for p in sorted(target.rglob('*')) if p.is_file()}}
    _write_new(target / 'preparation-receipt.json', result)
    return result


def main() -> None:
    """命令行仅准备新副本，不提供执行模型模式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allowed-output-root', type=Path, required=True)
    parser.add_argument('--protected-root', type=Path, action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(prepare(args.design_root, args.output, allowed_output_root=args.allowed_output_root,
                             protected_roots=args.protected_root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
