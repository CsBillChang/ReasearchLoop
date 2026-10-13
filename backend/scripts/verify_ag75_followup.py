"""AG7.5最小补验：沿用真实首问和PDF，只追加一次同会话追问。

冻结、执行、只读核对分别调用；旧批次不重跑，不改产品提示或工具协议。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

import httpx
from research_loop.config import Settings
from research_loop.providers.deepseek import RequestNotSent
from research_loop.services.runtime import application_lock
from research_loop.storage.mysql import Database
from research_loop.tools.registry import ToolRegistry
from research_loop.workers.executor import TaskExecutor
from scripts.backup_restore import digest, snapshot
from scripts.evaluation_common import exclusive
from scripts.evaluate_reflection import FrozenLedger
from scripts.manage import ROOT, Manager
from scripts.run_ag6 import EvaluationRunner, summarize_usage
from scripts.run_ag7 import OUT as OLD, SPEC as OLD_SPEC, configuration

OUT = ROOT / 'artifacts/ag7-5-followup-closeout-20261011'
FIRST = json.loads((OLD / 'real/Q1-input.json').read_text())
REF = FIRST['payload']['attachments'][0]['input_ref']
SPEC = {
    'physical_cap': 8, 'token_cap': 400000, 'wall_seconds': 900,
    'cases': [{'id': 'Q2-followup', 'request_cap': 8,
               'prompt': OLD_SPEC['cases'][1]['prompt']}],
    'conversation_id': FIRST['payload']['conversation_id'],
    'first_task_id': FIRST['task_id'], 'input_ref': REF,
    'save_policy': 'deny', 'new_protocol': False, 'retry_batch': False,
    'tool_cap': 20, 'reserve_tail': 2,
    'stop': 'unknown/usage缺失/供应商错误/越界即阻断后续发送；无第二次任务',
}


def row_hashes(db: Database) -> dict:
    """保存每个旧行的规范内容，允许追加但不能静默改旧稿/收据/消息。

    Returns:
        表名到排序JSON字符串列表；仅写入本地私有验收目录，不含配置密钥。
    """
    return {name: sorted(json.dumps(row, sort_keys=True, default=str) for row in rows)
            for name, rows in snapshot(db).items()}


def freeze() -> None:
    """先核对旧成功首问、材料、无活跃任务和零模型检查，再封存独立预算。"""
    config = configuration(); db = Database(config.database_url)
    assert db.get_task(FIRST['task_id'])['status'] == 'completed'
    assert not db.fetchall("SELECT task_id FROM tasks WHERE status IN ('queued','running','needs_input')")
    assert 'OK' in (OUT / 'zero-tests.log').read_text()
    attachment = json.loads(db.get_task(FIRST['task_id'])['attachments_json'])[0]
    assert attachment['input_ref'] == REF
    assert digest(Path(attachment['local_path'])) == REF.removeprefix('upload-')
    current = Settings.from_env(ROOT)
    production = Database(current.database_url)
    assert not production.fetchall("SELECT task_id FROM tasks WHERE status IN ('queued','running')")
    exclusive(OUT / 'production-before.json', row_hashes(production))
    exclusive(OUT / 'isolated-before.json', row_hashes(db))
    # 旧批次全部文件和生产已有PDF/Markdown逐个保护；源码另行冻结。
    old_files = {str(p.relative_to(ROOT)): digest(p) for p in OLD.rglob('*') if p.is_file()}
    asset_list = json.loads((OLD / 'baseline-files.json').read_text())
    exclusive(OUT / 'old-artifacts-before.json', old_files)
    exclusive(OUT / 'asset-baseline-copy.json', asset_list)
    files = list((ROOT / 'backend/research_loop').rglob('*.py')) + [
        Path(__file__), ROOT / 'backend/scripts/run_ag6.py',
        ROOT / 'backend/scripts/evaluate_reflection.py', ROOT / 'backend/scripts/run_ag7.py',
        ROOT / OLD_SPEC['pdf'], OUT / 'zero-tests.log']
    exclusive(OUT / 'freeze.json', {'scope': SPEC, 'model': config.deepseek_model,
        'files': {str(p.relative_to(ROOT)): digest(p) for p in files},
        'checks': ['附件与初始上下文接续', '本轮真实读页及覆盖', '有据回答及页码',
                   '无检索/精读/保存', '原记录与文件保持', '网页真实读回'],
        'prior_Q2': '旧失败仍在同会话历史中，不删除也不改为成功'})


def forbidden(name: str, arguments: dict) -> bool:
    """评测侧只阻止本次明确排除的动作；不隐藏工具或改写模型观察。

    Args:
        name: 模型选择的真实工具名。
        arguments: 原始工具参数；未修改后交给产品校验。
    """
    if name in {'search_papers', 'save_note'}:
        return True
    if name == 'set_research_plan' and arguments.get('deep_reading_requested'):
        return True
    if name == 'delegate_research':
        return any(s.get('mode', 'investigate') == 'deep_read'
                   for s in arguments.get('subtasks', []))
    return False


def run() -> None:
    """沿用同一隔离库与API，提交无附件追问；实际Worker通过冻结账本调用模型。"""
    frozen = json.loads((OUT / 'freeze.json').read_text())
    assert frozen['scope'] == SPEC
    for rel, expected in frozen['files'].items():
        assert digest(ROOT / rel) == expected, rel
    exclusive(OUT / 'started.json', {'scope': SPEC})
    config = configuration(); db = Database(config.database_url)
    ledger = FrozenLedger(OUT / 'physical.json', SPEC); ledger.case = 'Q2-followup'
    runner = object.__new__(EvaluationRunner)
    runner.settings = config; runner.db = db; runner.ledger = ledger; runner.cancel_hook = False
    original_execute = ToolRegistry.execute
    violations = []

    def guarded(registry, name, arguments, context):
        """越界动作执行前终止整批，不通过额外修复调用掩盖越界。"""
        if forbidden(name, arguments):
            violations.append({'tool': name, 'arguments': arguments})
            ledger.manifest = {**SPEC, 'physical_cap': 0}
            raise RequestNotSent('AG75_FROZEN_SCOPE_VIOLATION')
        return original_execute(registry, name, arguments, context)

    manager = Manager(config, 'ag75-validation'); task_id = None
    try:
        manager.start(18375, 4175, worker=False)
        with application_lock(ROOT), httpx.Client(base_url='http://127.0.0.1:18375', timeout=30) as client:
            payload = {'conversation_id': SPEC['conversation_id'], 'prompt': SPEC['cases'][0]['prompt'],
                       'idempotency_key': 'ag75-fixed-followup-20261011', 'save_policy': 'deny'}
            response = client.post('/api/tasks', json=payload); response.raise_for_status()
            task_id = response.json()['task_id']; task = db.get_task(task_id)
            assert json.loads(task['attachments_json'])[0]['input_ref'] == REF
            assert db.task_has_input(task_id, REF)
            assert not db.fetchall('SELECT * FROM task_evidence WHERE task_id=?', (task_id,))
            exclusive(OUT / 'input.json', {'payload': payload, 'task_id': task_id,
                      'effective_attachments': json.loads(task['attachments_json']), 'inherited_page_evidence': 0})
            db.execute('UPDATE task_budgets SET model_limit=8,tool_limit=20,reserve_tail=2 WHERE root_task_id=?', (task_id,))
            with patch.object(ToolRegistry, 'execute', guarded):
                TaskExecutor(config, db, provider_factory=runner.provider).execute(task_id, 'ag75-followup-worker')
            result = runner.collect(client, task_id)
            exclusive(OUT / 'result.json', result)
            replay = client.post('/api/tasks', json=payload); replay.raise_for_status()
            assert replay.json()['task_id'] == task_id
            response = client.get('/api/conversations/' + SPEC['conversation_id']); response.raise_for_status()
            exclusive(OUT / 'conversation.json', response.json())
    finally:
        receipts = db.fetchall('SELECT m.* FROM model_calls m JOIN tasks t ON t.task_id=m.task_id WHERE t.root_task_id=? ORDER BY m.created_at', (task_id,)) if task_id else []
        exclusive(OUT / 'receipts.json', receipts)
        exclusive(OUT / 'accounting.json', {'physical': len(ledger.records), 'cap': 8,
                  'usage': summarize_usage(receipts), 'violations': violations,
                  'notes': list(map(str, config.output_dir.rglob('*.md'))), 'task_id': task_id})
        manager.stop()


def audit() -> None:
    """只读核对旧资产；会话更新时间为追加消息的合法投影，其余旧行必须保留。"""
    changes = {}
    for kind, config in [('production', Settings.from_env(ROOT)), ('isolated', configuration())]:
        before = json.loads((OUT / (kind + '-before.json')).read_text())
        after = row_hashes(Database(config.database_url))
        missing = {name: sum((Counter(rows) - Counter(after[name])).values()) for name, rows in before.items()}
        changes[kind] = {'missing_or_changed_old_rows': missing,
                         'added_rows': {name: len(after[name]) - len(before[name]) for name in before}}
    old_files = json.loads((OUT / 'old-artifacts-before.json').read_text())
    changes['old_files'] = {'count': len(old_files), 'changed': [name for name, h in old_files.items()
                             if not (ROOT / name).is_file() or digest(ROOT / name) != h]}
    assets = json.loads((OUT / 'asset-baseline-copy.json').read_text())
    changes['existing_pdf_markdown'] = {'count': len(assets), 'changed': [name for name, h in assets.items()
                                       if not (ROOT / name).is_file() or digest(ROOT / name) != h]}
    exclusive(OUT / 'asset-audit.json', changes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'run', 'audit'])
    {'freeze': freeze, 'run': run, 'audit': audit}[parser.parse_args().command]()
