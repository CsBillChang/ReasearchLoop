"""复用已下载正文验证精读配置，再独立执行用户的 CVPR 检索输入。

仅用于本轮授权验收，不修改业务逻辑。每个 case 只执行一次；已有标记阻止重发。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from unittest.mock import patch

from fastapi.testclient import TestClient

from research_loop.agents.graph import build_graph, initial_state
from research_loop.config import Settings
from research_loop.domain.tasks import TaskCreate, TaskResult, TaskStatus
from research_loop.main import app
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.mysql import Database
from research_loop.tools.registry import ToolRegistry
from research_loop.workers.executor import TaskExecutor
from scripts.run_dpr_consolidation import snapshot

SEARCH_PROMPT = "找一篇今年CVPR中关于遥感图像检测的论文"


def write_json(path: Path, value: object) -> None:
    """保存审计JSON；凭据配置不得传入此函数。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def sha(path: Path) -> str:
    """计算文件身份，不读取或更改历史任务状态。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readback(db: Database, task_id: str, out: Path) -> dict:
    """读回当前任务及子任务收据；此函数不执行任务或调用供应商。"""
    ids = [task_id] + [child['subtask_id'] for child in db.list_subtasks(task_id)]
    for identity in ids:
        write_json(out / (identity + '.json'), {
            'task': db.get_task(identity), 'actions': db.list_actions(identity),
            'events': db.list_events(identity), 'receipts': db.list_model_calls(identity),
            'checkpoint': db.get_checkpoint(identity),
        })
    view = TaskService(db).view(db.get_task(task_id)).model_dump(mode='json')
    write_json(out / (task_id + '-view.json'), view)
    return view


def summary_checks(settings: Settings, db: Database, out: Path) -> None:
    """各复用一份完整页级正文，只调用当前生产精读节点，最多共4次请求。

    Args:
        settings: 指向已有隔离MySQL的新验收配置。
        db: 隔离库，旧任务只读，新任务使用独立身份。
        out: 新artifact目录；输出不进入项目业务笔记目录。
    """
    old = settings.project_root / 'artifacts/ag12-dpr-consolidation-20261009'
    audit = json.loads((old / 'acceptance-summary.json').read_text())
    prepared = []
    for entry in audit['reads']:
        prior = json.loads((old / (entry['task_id'] + '.json')).read_text())
        checkpoint = prior['checkpoint']
        refs = checkpoint['evidence_ids']
        pdf = old / 'runtime/cache/pdf' / (entry['pdf_sha256'] + '.pdf')
        assert sha(pdf) == entry['pdf_sha256']
        assert len(refs) == entry['total_pages']
        prepared.append((entry, checkpoint, pdf, refs))
    service = TaskService(db)
    records = []
    for entry, prior, pdf, refs in prepared:
        task = service.create(TaskCreate(
            prompt=prior['prompt'], save_policy='deny',
            idempotency_key='dpr-followup-node-' + entry['task_id'],
            attachments=[{'input_ref': 'downloaded-paper', 'local_path': str(pdf),
                          'paper_id': entry['paper_id']}],
        ))
        task_id = task.task_id
        record = {'task_id': task_id, 'paper_id': entry['paper_id'],
                  'previous_task_id': entry['task_id'], 'page_count': len(refs),
                  'pdf_sha256': entry['pdf_sha256']}
        records.append(record)
        write_json(out / 'summary-tasks.json', records)
        db.update_task(task_id, role='researcher')
        db.execute('UPDATE task_budgets SET model_limit=2,tool_limit=0,reserve_tail=0 WHERE root_task_id=? AND model_used=0', (task_id,))
        db.grant_paper(task_id, entry['paper_id'], task_id)
        for ref in refs:
            db.mark_task_evidence(task_id, ref)
        worker_id = 'dpr-node-followup'
        claimed = db.claim_task(worker_id, lease_seconds=600, task_id=task_id)
        assert claimed
        state = initial_state(task_id, task.prompt, 'deny', max_model_requests=2,
                              max_tool_actions=0, reserve_model_requests=0)
        state.update(role='researcher', finished=True, structured_result={'findings': []},
                     evidence_ids=refs, worker_id=worker_id, lease_token=claimed['lease_token'])
        papers = PaperService(db, None, None)
        provider = TaskExecutor._default_provider(settings)
        try:
            graph = build_graph(provider, ToolRegistry([]), {'papers': papers}, db,
                                checkpoint_store=CheckpointStore(db), deep_reading=True)
            output = graph.invoke(state)
            stage = output['deep_reading']
            result = TaskResult(answer_markdown=stage.get('markdown'), evidence_ids=refs,
                                gaps=output.get('gaps') or [], reason=stage.get('error'))
            status = TaskStatus.completed if stage['status'] == 'complete' else TaskStatus.partial
            service.complete(task_id, status, result, owner=(worker_id, claimed['lease_token']))
            record.update(node_status=stage['status'], finish_reason=stage.get('finish_reason'),
                          visible_characters=len(stage.get('markdown') or ''))
            (out / (task_id + '-summary.md')).write_text(stage.get('markdown') or '')
        finally:
            provider.close()
            record['view'] = readback(db, task_id, out)
            write_json(out / 'summary-tasks.json', records)
        print(json.dumps({k: v for k, v in record.items() if k != 'view'}, ensure_ascii=False), flush=True)
        if db.has_unknown_family_call(task_id):
            raise RuntimeError('unknown response; stop this case without retry')


def search_check(settings: Settings, db: Database, out: Path) -> None:
    """原样提交用户检索要求，使用真实API及定向Worker，不人工选候选。"""
    with patch('research_loop.api.dependencies.get_settings', return_value=settings), TestClient(app) as client:
        response = client.post('/api/tasks', json={'prompt': SEARCH_PROMPT,
            'idempotency_key': 'dpr-followup-cvpr-remote-sensing-20261009', 'save_policy': 'deny'})
        response.raise_for_status()
        task_id = response.json()['task_id']
        write_json(out / 'search-task.json', response.json())
        db.execute('UPDATE task_budgets SET model_limit=12,tool_limit=24,reserve_tail=2 WHERE root_task_id=? AND model_used=0', (task_id,))
        print(json.dumps({'task_id': task_id, 'input': SEARCH_PROMPT}, ensure_ascii=False), flush=True)
        try:
            TaskExecutor(settings, db).execute(task_id, 'dpr-followup-cvpr')
        finally:
            view = readback(db, task_id, out)
            write_json(out / 'search-result.json', view)
            result = view.get('result') or {}
            (out / 'search-answer.md').write_text(result.get('answer_markdown') or '')
        print(json.dumps({'task_id': task_id, 'status': view['status'], 'usage': view['usage']}, ensure_ascii=False), flush=True)


def main() -> None:
    """冻结源码后只运行选定case；已有标记时拒绝重发，不扩容或修改历史批次。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['summaries', 'search'], required=True)
    args = parser.parse_args()
    settings = Settings.from_env()
    out = settings.project_root / 'artifacts/ag12-dpr-followup-20261009'
    out.mkdir(parents=True, exist_ok=True)
    marker = out / (args.case + '-started.json')
    with marker.open('x') as handle:
        json.dump({'case': args.case}, handle)
    source_hashes = {str(p.relative_to(settings.project_root)): sha(p)
                     for p in (settings.project_root / 'backend/research_loop').rglob('*.py')}
    write_json(out / (args.case + '-freeze.json'), {'source_hashes': source_hashes,
        'model': settings.deepseek_model, 'case': args.case, 'summary_budget': [4096, 2048],
        'thinking': {'type': 'disabled'}, 'search_input': SEARCH_PROMPT,
        'summary_request_cap': 4, 'search_request_cap': 12})
    production = Database(settings.database_url)
    write_json(out / (args.case + '-production-before.json'), snapshot(production))
    url = urlsplit(settings.database_url)
    isolated = settings.model_copy(update={'database_url': urlunsplit((url.scheme, url.netloc,
        '/researchloop_test_dpr_live_20261009', '', '')), 'data_dir': out / 'runtime',
        'output_dir': out / 'notes', 'max_model_requests': 12, 'max_tool_actions': 24,
        'reserve_model_requests': 2})
    db = Database(isolated.database_url)
    try:
        if args.case == 'summaries':
            summary_checks(isolated, db, out)
        else:
            search_check(isolated, db, out)
    finally:
        write_json(out / (args.case + '-production-after.json'), snapshot(production))


if __name__ == '__main__':
    main()
