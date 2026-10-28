"""共同实验合同：只接受已枚举的变量，不执行待检查的业务源码。"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re


SCOPE_KIND = 'supervisor_scope_v1'
GOAL_KIND = 'supervisor_goal_v1'
SUPERVISOR_KINDS = frozenset({SCOPE_KIND, GOAL_KIND})
HANDOFF_KIND = 'research_progress_id_recovery_v1'
PAIRED_KINDS = SUPERVISOR_KINDS | {HANDOFF_KIND}
SUPERVISOR = 'backend/research_loop/agents/supervisor.py'
HANDOFF_SERVICE = 'backend/research_loop/services/research_progress.py'
HANDOFF_RESEARCHER = 'backend/research_loop/agents/researcher.py'
HANDOFF_FILES = frozenset({HANDOFF_SERVICE, HANDOFF_RESEARCHER})
HANDOFF_CONTRACT_VERSION = 'unknown-finding-id-recovery-v1'
CONTRACT_VERSION = 'scope-rules-4-7-v1'
GOAL_CONTRACT_VERSION = 'goal-rule-1-v1'
OLD_RULE_4 = '问答/比较/综述用investigate围绕问题读取原文。'
NEW_RULE_4 = '问答/比较/综述用investigate，把原问的子问题与解释结论所必需的条件写入objective；未知机制用中性问题表述，具体模块、损失和阈值的核查以用户问题或已读证据为依据。实验数值或额外比较仅在用户要求或解释结论必需时纳入委派。保留用户明确要求的广度、深度和材料范围。'
OLD_RULE_7 = '比较分别说明输入、训练、推理、输出，同一方法可以跨阶段，不能为制造差异删除共同点。'
NEW_RULE_7 = OLD_RULE_7 + '综合答案围绕原问组织结论和必要依据；附加数值、比较或排他性结论须同时与问题直接相关且有对应证据。保留影响结论的条件、例外、原文矛盾和不确定性；详细解读或实验比较按用户要求展开。'
OLD_GOAL_RULE = '引句存在不证明理解正确，应按原意区分用户目标与策略。'
NEW_GOAL_RULE = '用户目标及派生objective须保留原问含义、广度与深度；source_quote逐字匹配不代表扩写的要求得到原问支持。只将用户要求的交付列为origin=user。为准确完成交付所必需的补充取证列为origin=strategy，并在description中说明服务于哪个用户目标、弥补什么影响交付的缺口。未知机制或因果关系用中性问题表达；只在用户要求或已读证据支持其相关性时，展开具体机制的逐项核查。不影响用户交付的背景扩展不列为必做目标。'
PAIR_REVIEW_POLICY = {'required': True, 'reviewer_kinds': ['human', 'agent_assisted'], 'poll_seconds': 0.5}
UNKNOWN_ID_RAISE = "raise ValueError('unknown finding_id; omit it for a new finding')"
UNKNOWN_ID_CLASS = '''class UnknownFindingIdError(ValueError):
    """发现ID不属于当前任务，拒绝本次提交并允许有界参数纠正。"""


'''
OLD_PROGRESS_IMPORT = 'from research_loop.services.research_progress import ResearchProgressService'
OLD_PROGRESS_CALL = '    value = ResearchProgressService(db).prepare(ctx.task_id, args)'
NEW_PROGRESS_CALL = '''    try:
        value = ResearchProgressService(db).prepare(ctx.task_id, args)
    except UnknownFindingIdError as exc:
        return Observation(ok=False, kind='validation_error', retryable=True,
            data={'error_code': 'UNKNOWN_FINDING_ID'},
            message=str(exc) + '; new findings omit finding_id; updates use an ID returned for this task')'''


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


def goal_candidate(text):
    """只替换第1条目标解释，并确认基线已包含批准的第4、7条。"""
    prompt, _ = supervisor_parts(text)
    rules = {int(m.group(1)): m.group(0) for m in re.finditer(r'(?m)^(\d+)\. .*$', prompt)}
    if (text.count(OLD_GOAL_RULE) != 1 or OLD_GOAL_RULE not in rules.get(1, '')
            or NEW_RULE_4 not in rules.get(4, '') or NEW_RULE_7 not in rules.get(7, '')):
        raise ValueError('approved rule 1 baseline and retained rules 4 and 7 required')
    return text.replace(OLD_GOAL_RULE, NEW_GOAL_RULE, 1)


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


def build_supervisor_contract(source_roots, *, experiment_kind=SCOPE_KIND):
    """按枚举实验绑定摘要，拒绝批准提示词以外的第二业务变量。"""
    if experiment_kind not in SUPERVISOR_KINDS:
        raise ValueError('unknown Supervisor experiment kind')
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
    transform = goal_candidate if experiment_kind == GOAL_KIND else scope_candidate
    version = GOAL_CONTRACT_VERSION if experiment_kind == GOAL_KIND else CONTRACT_VERSION
    if texts['B'] != transform(texts['A']) or prompts['A'][1] != prompts['B'][1]:
        raise ValueError('Supervisor diff is not exactly the approved prompt treatment')
    return {'version': version, 'variants': {v: {
        'supervisor_file_sha256': inventories[v][SUPERVISOR],
        'system_prompt_sha256': sha256(prompts[v][0]),
        'source_tree_sha256': sha256(canonical(inventories[v])),
    } for v in ('A', 'B')}}


def validate_supervisor_sources(manifest):
    """格启动与每次模型发送前重新验证源码身份。"""
    try:
        expected = build_supervisor_contract(manifest['source_roots'],
            experiment_kind=manifest.get('experiment_kind', SCOPE_KIND))
        if manifest.get('supervisor_prompt_contract') != expected:
            raise ValueError('source or prompt digest differs')
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        raise PermissionError('SUPERVISOR_SOURCE_CONTRACT_INVALID') from exc
    return expected


def handoff_candidate(texts):
    """只生成已批准的窄异常及工具映射，不执行或放宽待检查源码。"""
    if not isinstance(texts, dict) or set(texts) != HANDOFF_FILES:
        raise ValueError('two approved handoff source files required')
    replacements = {
        HANDOFF_SERVICE: [('class ResearchProgressService:', UNKNOWN_ID_CLASS + 'class ResearchProgressService:'),
            (UNKNOWN_ID_RAISE, UNKNOWN_ID_RAISE.replace('ValueError', 'UnknownFindingIdError', 1))],
        HANDOFF_RESEARCHER: [(OLD_PROGRESS_IMPORT, OLD_PROGRESS_IMPORT + ', UnknownFindingIdError'),
            (OLD_PROGRESS_CALL, NEW_PROGRESS_CALL)],
    }
    candidate = {}
    for name, text in texts.items():
        for old, new in replacements[name]:
            if text.count(old) != 1:
                raise ValueError('approved handoff baseline clause is not unique: ' + name)
            text = text.replace(old, new, 1)
        ast.parse(text)
        candidate[name] = text
    return candidate


def build_handoff_contract(source_roots):
    """绑定两文件精确业务差异，Supervisor与共同首错入口必须完全相同。"""
    if not isinstance(source_roots, dict) or set(source_roots) != {'A', 'B'}:
        raise ValueError('two source roots required')
    roots = {v: Path(source_roots[v]).resolve(strict=True) for v in ('A', 'B')}
    if (roots['A'] == roots['B'] or roots['A'].is_relative_to(roots['B'])
            or roots['B'].is_relative_to(roots['A'])):
        raise ValueError('handoff variants require independent source directories')
    inventories = {v: source_inventory(roots[v]) for v in roots}
    if set(inventories['A']) != set(inventories['B']):
        raise ValueError('source file sets differ')
    changed = {p for p in inventories['A'] if inventories['A'][p] != inventories['B'][p]}
    if changed != HANDOFF_FILES:
        raise ValueError('only the two approved handoff files may differ')
    old = {p: (roots['A'] / p).read_text() for p in HANDOFF_FILES}
    expected = handoff_candidate(old)
    if any((roots['B'] / p).read_text() != expected[p] for p in HANDOFF_FILES):
        raise ValueError('handoff diff is not exactly the approved treatment')
    prompt, _ = supervisor_parts((roots['A'] / SUPERVISOR).read_text())
    rules = {int(m.group(1)): m.group(0) for m in re.finditer(r'(?m)^(\d+)\. .*$', prompt)}
    if (OLD_GOAL_RULE not in rules.get(1, '') or NEW_RULE_4 not in rules.get(4, '')
            or NEW_RULE_7 not in rules.get(7, '')):
        raise ValueError('approved baseline Supervisor rules 1, 4 and 7 required')
    budget = roots['A'] / 'backend/scripts/harness_ab_budget.py'
    tree = ast.parse(budget.read_text())
    ledgers = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Ledger']
    constructors = [node for ledger in ledgers for node in ledger.body
        if isinstance(node, ast.FunctionDef) and node.name == '__init__']
    configs = [node.value for constructor in constructors for node in ast.walk(constructor)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name) and target.value.id == 'self'
            and target.attr == 'config' for target in node.targets)]
    entries = [value for key, value in zip(configs[0].keys, configs[0].values)
        if isinstance(key, ast.Constant) and key.value == 'first_error_policy_version'] \
        if len(configs) == 1 and isinstance(configs[0], ast.Dict) else []
    if (len(ledgers) != 1 or len(constructors) != 1 or len(configs) != 1
            or not isinstance(configs[0], ast.Dict) or any(key is None for key in configs[0].keys)
            or len(entries) != 1 or not isinstance(entries[0], ast.Constant)
            or type(entries[0].value) is not int or entries[0].value != 1):
        raise ValueError('shared persistent first-error policy required')
    return {'version': HANDOFF_CONTRACT_VERSION, 'business_files': sorted(HANDOFF_FILES),
        'first_error_policy_version': 1, 'variants': {v: {
            'supervisor_file_sha256': inventories[v][SUPERVISOR],
            'system_prompt_sha256': sha256(prompt),
            'business_file_sha256': {p: inventories[v][p] for p in sorted(HANDOFF_FILES)},
            'source_tree_sha256': sha256(canonical(inventories[v])),
        } for v in ('A', 'B')}}


def source_contract_key(experiment_kind):
    """各实验使用独立合同字段，旧Supervisor类型不吸收新业务变量。"""
    if experiment_kind == HANDOFF_KIND:
        return 'handoff_source_contract'
    if experiment_kind in SUPERVISOR_KINDS:
        return 'supervisor_prompt_contract'
    raise PermissionError('PAIRED_EXPERIMENT_KIND_REQUIRED')


def paired_source_contract(manifest):
    """返回本次枚举类型实际绑定的合同，用于资格和一次性授权身份。"""
    key = source_contract_key(manifest.get('experiment_kind'))
    if manifest.get('experiment_kind') == HANDOFF_KIND and 'supervisor_prompt_contract' in manifest:
        raise PermissionError('HANDOFF_CONTRACT_MUST_NOT_USE_SUPERVISOR_LABEL')
    contract = manifest.get(key)
    if not isinstance(contract, dict):
        raise PermissionError('PAIRED_SOURCE_CONTRACT_REQUIRED')
    return contract


def validate_paired_sources(manifest):
    """启动及每次发送重新验证相应的精确源码合同。"""
    if manifest.get('experiment_kind') in SUPERVISOR_KINDS:
        return validate_supervisor_sources(manifest)
    try:
        if manifest.get('experiment_kind') != HANDOFF_KIND:
            raise ValueError('unknown paired experiment kind')
        expected = build_handoff_contract(manifest['source_roots'])
        if paired_source_contract(manifest) != expected:
            raise ValueError('handoff source digest differs')
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        raise PermissionError('HANDOFF_SOURCE_CONTRACT_INVALID') from exc
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


def validate_paired_manifest(manifest):
    """配对门位于格写权限之外，避免被测工具自行批准下一对。"""
    if manifest.get('experiment_kind') not in PAIRED_KINDS:
        raise PermissionError('PAIRED_EXPERIMENT_KIND_REQUIRED')
    if manifest.get('variant_settings') != {
            'A': {'context_deduplicate_read_paper': True},
            'B': {'context_deduplicate_read_paper': True}}:
        raise PermissionError('SUPERVISOR_DEDUP_BOTH_ON_REQUIRED')
    validate_paired_sources(manifest)
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


def validate_scope_manifest(manifest):
    """保留旧入口的Supervisor类型边界，不允许新业务实验借用旧标签。"""
    if manifest.get('experiment_kind') not in SUPERVISOR_KINDS:
        raise PermissionError('SUPERVISOR_EXPERIMENT_KIND_REQUIRED')
    return validate_paired_manifest(manifest)
