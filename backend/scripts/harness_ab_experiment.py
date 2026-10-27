"""共同实验合同：只接受已枚举的变量，不执行待检查的业务源码。"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re


SCOPE_KIND = 'supervisor_scope_v1'
SUPERVISOR = 'backend/research_loop/agents/supervisor.py'
CONTRACT_VERSION = 'scope-rules-4-7-v1'
OLD_RULE_4 = '问答/比较/综述用investigate围绕问题读取原文。'
NEW_RULE_4 = '问答/比较/综述用investigate，把原问的子问题与解释结论所必需的条件写入objective；未知机制用中性问题表述，具体模块、损失和阈值的核查以用户问题或已读证据为依据。实验数值或额外比较仅在用户要求或解释结论必需时纳入委派。保留用户明确要求的广度、深度和材料范围。'
OLD_RULE_7 = '比较分别说明输入、训练、推理、输出，同一方法可以跨阶段，不能为制造差异删除共同点。'
NEW_RULE_7 = OLD_RULE_7 + '综合答案围绕原问组织结论和必要依据；附加数值、比较或排他性结论须同时与问题直接相关且有对应证据。保留影响结论的条件、例外、原文矛盾和不确定性；详细解读或实验比较按用户要求展开。'
PAIR_REVIEW_POLICY = {'required': True, 'reviewer_kinds': ['human', 'agent_assisted'], 'poll_seconds': 0.5}


def sha256(value):
    """固定以UTF-8字节和SHA-256绑定内容。"""
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def canonical(value):
    """序列化控制字段，不依赖字典插入顺序。"""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def supervisor_parts(text):
    """静态读取唯一系统常量，保留其余AST以检查第二业务变量。"""
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in n.targets)]
    if len(nodes) != 1:
        raise ValueError('one literal SYSTEM_PROMPT is required')
    prompt = ast.literal_eval(nodes[0].value)
    if not isinstance(prompt, str):
        raise ValueError('SYSTEM_PROMPT must be a literal string')
    nodes[0].value = ast.Constant(value='PROMPT_CONSTANT_EXCLUDED')
    return prompt, ast.dump(tree, include_attributes=False)


def scope_candidate(text):
    """只生成已批准的两处替换，禁止附加示例或场景规则。"""
    if text.count(OLD_RULE_4) != 1 or text.count(OLD_RULE_7) != 1:
        raise ValueError('approved baseline clauses are not unique')
    return text.replace(OLD_RULE_4, NEW_RULE_4).replace(OLD_RULE_7, NEW_RULE_7)


def source_inventory(root):
    """绑定运行代码、共用脚本、迁移与配置；运行产物不得放入源码根。"""
    root = Path(root).resolve(strict=True)
    files = {}
    for relative in ('backend/research_loop', 'backend/scripts', 'backend/migrations', 'config'):
        directory = root / relative
        if not directory.exists():
            if relative in ('backend/research_loop', 'backend/scripts'):
                raise ValueError('source directory missing: ' + relative)
            continue
        if directory.is_symlink():
            raise ValueError('source directory symlink is not allowed')
        for path in sorted(directory.rglob('*')):
            if '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.is_symlink():
                raise ValueError('source symlink is not allowed')
            if path.is_file():
                files[str(path.relative_to(root))] = sha256(path.read_bytes())
    if SUPERVISOR not in files:
        raise ValueError('Supervisor source is missing')
    return files


def build_supervisor_contract(source_roots):
    """产生摘要，同时检查两版除了精确两句提示词外没有代码差异。"""
    if not isinstance(source_roots, dict) or set(source_roots) != {'A', 'B'}:
        raise ValueError('two source roots required')
    roots = {v: Path(source_roots[v]).resolve(strict=True) for v in ('A', 'B')}
    if (roots['A'] == roots['B'] or roots['A'].is_relative_to(roots['B'])
            or roots['B'].is_relative_to(roots['A'])):
        raise ValueError('scope variants require independent source directories')
    inventories = {v: source_inventory(roots[v]) for v in roots}
    if set(inventories['A']) != set(inventories['B']):
        raise ValueError('source file sets differ')
    changed = [p for p in inventories['A'] if inventories['A'][p] != inventories['B'][p]]
    if changed != [SUPERVISOR]:
        raise ValueError('only Supervisor source may differ')
    texts = {v: (roots[v] / SUPERVISOR).read_text() for v in roots}
    prompts = {v: supervisor_parts(texts[v]) for v in roots}
    if texts['B'] != scope_candidate(texts['A']) or prompts['A'][1] != prompts['B'][1]:
        raise ValueError('Supervisor diff is not exactly the approved rules 4 and 7')
    return {'version': CONTRACT_VERSION, 'variants': {v: {
        'supervisor_file_sha256': inventories[v][SUPERVISOR],
        'system_prompt_sha256': sha256(prompts[v][0]),
        'source_tree_sha256': sha256(canonical(inventories[v])),
    } for v in ('A', 'B')}}


def validate_supervisor_sources(manifest):
    """格启动与每次模型发送前重新验证源码身份。"""
    try:
        expected = build_supervisor_contract(manifest['source_roots'])
        if manifest.get('supervisor_prompt_contract') != expected:
            raise ValueError('source or prompt digest differs')
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        raise PermissionError('SUPERVISOR_SOURCE_CONTRACT_INVALID') from exc
    return expected


def validate_pair_order(order):
    """每个问题的两版必须相邻，一次矩阵中每格只能出现一次。"""
    if not isinstance(order, list) or not order or len(order) % 2:
        raise PermissionError('SUPERVISOR_PAIR_ORDER_REQUIRED')
    seen = set()
    for i in range(0, len(order), 2):
        left, right = order[i:i + 2]
        if (not isinstance(left, list) or not isinstance(right, list)
                or len(left) != 2 or len(right) != 2 or left[0] != right[0]
                or {left[1], right[1]} != {'A', 'B'}
                or not isinstance(left[0], str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', left[0])
                or left[0] in seen):
            raise PermissionError('SUPERVISOR_PAIR_ORDER_REQUIRED')
        seen.add(left[0])


def validate_scope_manifest(manifest):
    """配对门位于格写权限之外，避免被测工具自行批准下一对。"""
    if manifest.get('variant_settings') != {
            'A': {'context_deduplicate_read_paper': True},
            'B': {'context_deduplicate_read_paper': True}}:
        raise PermissionError('SUPERVISOR_DEDUP_BOTH_ON_REQUIRED')
    validate_supervisor_sources(manifest)
    if manifest.get('pair_review_policy') != PAIR_REVIEW_POLICY:
        raise PermissionError('SUPERVISOR_PAIR_REVIEW_POLICY_REQUIRED')
    runtime = Path(manifest['runtime_root']).resolve()
    control = Path(manifest.get('semantic_gate_root', '')).resolve()
    if control != runtime.parent / 'semantic-gates':
        raise PermissionError('SUPERVISOR_REVIEW_CONTROL_MUST_BE_RUNTIME_SIBLING')
    if control.is_relative_to(runtime) or runtime.is_relative_to(control):
        raise PermissionError('SUPERVISOR_REVIEW_CONTROL_OUTSIDE_RUNTIME_REQUIRED')
    for source in manifest['source_roots'].values():
        source = Path(source).resolve()
        if control.is_relative_to(source) or source.is_relative_to(control):
            raise PermissionError('SUPERVISOR_REVIEW_CONTROL_OUTSIDE_SOURCE_REQUIRED')
        if runtime.is_relative_to(source) or source.is_relative_to(runtime):
            raise PermissionError('SUPERVISOR_RUNTIME_OUTSIDE_SOURCE_REQUIRED')
    if control.is_symlink():
        raise PermissionError('SUPERVISOR_REVIEW_CONTROL_SYMLINK')
    validate_pair_order(manifest.get('execution_order'))
