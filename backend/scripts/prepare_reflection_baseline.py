"""只读校验与冻结 Reflection 基线包；不构造供应商或业务存储。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORBIDDEN_MODEL_KEYS = {
    'category', 'error_type', 'gold', 'expected', 'expected_text',
    'expected_handling', 'before_status', 'claims', 'labels',
    'processing_summary', 'reviewer_findings',
}


def digest(path: Path) -> str:
    """对保存的原始字节计 hash，不归一化原文。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confined(root: Path, relative: str) -> Path:
    """拒绝通过绝对路径、父目录或软链接读取包外文件。"""
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('REFLECTION_PACKAGE_PATH_ESCAPE')
    path = root / name
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError('REFLECTION_PACKAGE_SYMLINK')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('REFLECTION_PACKAGE_PATH_ESCAPE')
    return path


def check_model_payload(value, location: str = 'model_input') -> None:
    """只检查字段名，避免把论文正文中的普通词误当标签泄漏。"""
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in FORBIDDEN_MODEL_KEYS:
                raise ValueError('REFLECTION_LABEL_IN_MODEL_INPUT:' + location + '.' + key)
            check_model_payload(item, location + '.' + key)
    elif isinstance(value, list):
        for number, item in enumerate(value):
            check_model_payload(item, location + '[' + str(number) + ']')


def validate_package(manifest_path: Path) -> dict:
    """核对来源和稿件身份，保留材料人评未完成的执行阻断。

    Args:
        manifest_path: 包内显式清单，所有引用都相对于其父目录。

    Returns:
        结构校验、各批数量与尚未满足的真实执行条件。
    """
    root = manifest_path.resolve().parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('REFLECTION_MANIFEST_VERSION')
    sources = manifest.get('files', [])
    if not sources:
        raise ValueError('REFLECTION_MANIFEST_EMPTY')
    names = [entry['path'] for entry in sources]
    if len(names) != len(set(names)):
        raise ValueError('REFLECTION_DUPLICATE_FILE')
    for entry in sources:
        if digest(confined(root, entry['path'])) != entry['sha256']:
            raise ValueError('REFLECTION_FILE_DRIFT:' + entry['path'])
    indexed = set(names)
    seen, counts, semantic_pending = set(), {}, 0
    for case in manifest.get('cases', []):
        case_id = case['case_id']
        if case_id in seen:
            raise ValueError('REFLECTION_DUPLICATE_CASE')
        seen.add(case_id)
        if case['input'] not in indexed or case['label'] not in indexed:
            raise ValueError('REFLECTION_CASE_NOT_HASHED')
        data = json.loads(confined(root, case['input']).read_text())
        label = json.loads(confined(root, case['label']).read_text())
        if data.get('case_id') != case_id or label.get('case_id') != case_id:
            raise ValueError('REFLECTION_CASE_IDENTITY')
        model_input = data.get('model_input')
        if not isinstance(model_input, dict):
            raise ValueError('REFLECTION_EXPLICIT_MODEL_INPUT_REQUIRED')
        check_model_payload(model_input)
        for first, second in [('question', 'original_question'), ('draft', 'draft_markdown')]:
            if first in model_input and second in model_input and model_input[first] != model_input[second]:
                raise ValueError('REFLECTION_CONFLICTING_INPUT_ALIASES')
        question = model_input.get('question') or model_input.get('original_question')
        draft = model_input.get('draft') or model_input.get('draft_markdown')
        if not question or not draft:
            raise ValueError('REFLECTION_EMPTY_QUESTION_OR_DRAFT')
        if (label.get('draft_sha256') and label.get('before_draft_sha256') and
                label['draft_sha256'] != label['before_draft_sha256']):
            raise ValueError('REFLECTION_CONFLICTING_LABEL_HASHES')
        label_hash = label.get('draft_sha256') or label.get('before_draft_sha256')
        if label_hash and label_hash != hashlib.sha256(draft.encode()).hexdigest():
            raise ValueError('REFLECTION_LABEL_DRAFT_MISMATCH')
        for key, value in [('draft_sha256', draft), ('question_sha256', question)]:
            expected = data.get('provenance', {}).get(key)
            if expected and expected != hashlib.sha256(value.encode()).hexdigest():
                raise ValueError('REFLECTION_PROVENANCE_MISMATCH:' + key)
        for document in model_input.get('documents') or model_input.get('historical_materials', []):
            for page in document.get('initial_pages', document.get('pages', [])):
                expected = page.get('page_text_sha256') or page.get('text_sha256') or page.get('sha256')
                if not expected or expected != hashlib.sha256(page['text'].encode()).hexdigest():
                    raise ValueError('REFLECTION_PAGE_TEXT_DRIFT')
                physical = page.get('physical_page', page.get('page_number'))
                if (physical is not None and
                        'allowed_pages' in document and
                        physical not in document.get('allowed_pages', [])):
                    raise ValueError('REFLECTION_PAGE_OUTSIDE_SCOPE')
        if label.get('approval_status') != 'user_reviewed':
            semantic_pending += 1
        batch = case['batch']
        counts[batch] = counts.get(batch, 0) + 1
    if not seen:
        raise ValueError('REFLECTION_NO_CASES')
    blockers = ['dedicated_real_runner_not_qualified', 'current_settings_and_rates_not_frozen']
    if semantic_pending:
        blockers.insert(0, 'material_labels_pending_user_review')
    # 计划批准不能自动升级成材料确认或真实发送资格；本工具没有run入口。
    return {'structural_passed': True, 'files_verified': len(sources),
        'cases': len(seen), 'batch_counts': counts,
        'semantic_labels_pending': semantic_pending, 'real_execution_allowed': False,
        'execution_blockers': blockers, 'new_real_model_calls': 0}


def freeze_package(manifest_path: Path, output: Path) -> dict:
    """独占保存准备快照，不覆盖旧冻结或生成调用授权。"""
    result = validate_package(manifest_path)
    result['manifest_sha256'] = digest(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main() -> None:
    """仅提供validate/freeze；普通调用无法加载模型或打开业务数据库。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'freeze'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        if args.output is None:
            parser.error('freeze requires --output')
        result = freeze_package(args.manifest, args.output)
    else:
        result = validate_package(args.manifest)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
