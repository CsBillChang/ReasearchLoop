"""视觉补读固定对照入口；默认零模型预检，真实执行必须另行显式授权。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
import statistics
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from research_loop.agents.visual_reading import VisualReadingNode
from research_loop.api.dependencies import get_runtime_settings
from research_loop.config import Settings
from research_loop.domain.visual_reading import VisualConfig, VisualInput, build_request
from research_loop.main import app
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.files import FileStore
from research_loop.storage.memory import digest
from research_loop.storage.mysql import Database
from research_loop.storage.token_budget import actual_usage, estimate_details

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    """读取本机JSON，调用者负责将其作为数据而非指令。"""
    return json.loads(Path(path).read_text())


def file_hash(path):
    """计算冻结文件字节身份。"""
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """独占写入新证据，不覆盖已存在的对照或失败结果。"""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def settings(directory, database=None):
    """沿用原配置，仅隔离目录/数据库；不改请求、Token或上下文额度。"""
    current = Settings.from_env(ROOT)
    if database:
        if not re.fullmatch(r'researchloop_test_visual_[A-Za-z0-9_]+', database):
            raise ValueError('isolated visual database name required')
        url = urlparse(current.database_url)
        if url.hostname not in {'localhost', '127.0.0.1'} or url.scheme != 'mysql':
            raise ValueError('local MySQL required')
        target = urlunparse(url._replace(path='/' + database))
    else:
        target = 'sqlite:///' + str(directory / 'preflight.sqlite')
    return current.model_copy(update={'database_url': target, 'data_dir': directory/'data',
        'output_dir': directory/'notes', 'retrieval_namespace': database or 'visual-preflight'})


def public_settings(config):
    """只冻结与实验相关的非秘密配置，不输出凭据。"""
    keys = ('deepseek_model', 'deepseek_base_url', 'request_timeout_seconds', 'max_model_requests',
            'max_tool_actions', 'reserve_model_requests', 'task_token_limit', 'task_token_tail',
            'context_window_tokens', 'context_output_reserve', 'context_safety_tokens',
            'parser_name', 'parser_version', 'parser_config')
    return {key: getattr(config, key) for key in keys}


@contextmanager
def api(config):
    """正常HTTP路由使用隔离依赖，不启动日常Worker或Scheduler。"""
    previous = app.dependency_overrides.copy()
    app.dependency_overrides[get_runtime_settings] = lambda: config
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear(); app.dependency_overrides.update(previous)


def create_task(client, item, material, files, key):
    """按冻结材料经API创建deny任务，模型不能指定任意路径。"""
    response = client.post('/api/tasks', json={'prompt': item.question + f' 仅使用物理第{item.physical_page}页，不保存。',
        'save_policy': 'deny', 'idempotency_key': key,
        'attachments': [{'input_ref': item.target_ref, 'local_path': str(files.cache_pdf_path(item.pdf_sha256)),
                         'paper_id': material['paper_id']}]})
    response.raise_for_status()
    return response.json()['task_id']


def preflight(directory, plan_path, materials_path, database=None):
    """复制已存在的原材料，逐题经读取服务渲染；没有模型或外部请求。

    Returns:
        配对输入、原文hash、图片清单与预估；不生成模型答案或质量分数。
    """
    plan, materials = read(plan_path), read(materials_path)
    if len(plan['cases']) != 5 or len({(c['material'],c['physical_page']) for c in plan['cases']}) != 4:
        raise ValueError('expected frozen four-page five-question scope')
    if len({c['id'] for c in plan['cases']}) != 5 or len(materials) != 2:
        raise ValueError('duplicate case identity or changed material scope')
    if (directory/'preflight.json').exists():
        raise FileExistsError('preflight already exists; do not overwrite')
    config = settings(directory, database); db = Database(config.database_url); db.ensure_schema()
    files = FileStore(config.data_dir, config.output_dir, ROOT)
    for material in materials:
        for source, expected, target in ((material['pdf_path'], material['pdf_sha256'], files.cache_pdf_path(material['pdf_sha256'])),
                (material['cache_path'], material['cache_sha256'], files.parse_dir/Path(material['cache_path']).name)):
            if file_hash(source) != expected:
                raise ValueError('original material hash mismatch')
            files.write_bytes_atomic(target, Path(source).read_bytes())
    rows = []
    pdf = PDFProvider(files, config.parser_name, config.parser_version, config.parser_config)
    try:
        papers = PaperService(db, None, pdf)
        with api(config) as client:
            for i, case in enumerate(plan['cases']):
                material = next(m for m in materials if m['key'] == case['material'])
                for group in (('text','visual') if i % 2 == 0 else ('visual','text')):
                    item = VisualInput(target_ref='visual-'+case['material'], document_id=case['document_id'],
                        pdf_sha256=case['pdf_sha256'], page_text_sha256=case['page_text_sha256'],
                        physical_page=case['physical_page'], question=case['question'], group=group,
                        bbox=case['candidate_normalized_bbox'])
                    tid = create_task(client, item, material, files, 'visual-preflight-'+uuid4().hex)
                    try:
                        budget = db.assert_budget_policy(tid, config)
                        prepared = papers.prepare_visual(tid, item)
                        request = build_request(prepared, VisualConfig(), files.cache_dir)
                        detail = estimate_details(request)
                        name = case['id'] + '-' + group
                        write(directory/'inputs'/f'{name}.json', prepared)
                        write(directory/'requests'/f'{name}.json', request)
                        rows.append({'id': name, 'case': case['id'], 'group': group,
                            'material': material['key'], 'item': item.model_dump(mode='json'),
                            'request_hash': digest(request), 'estimate': detail, 'task_id': tid,
                            'budget_readback': budget, 'render_seconds': prepared['render_seconds']})
                    finally:
                        TaskService(db, config).cancel(tid)
    finally:
        pdf.close()
    result = {'rows': rows, 'materials': materials, 'settings': public_settings(config),
        'visual_config': VisualConfig().model_dump(), 'plan': str(plan_path.resolve()),
        'materials_path': str(materials_path.resolve()), 'database': database,
        'real_requests': 0, 'semantic_quality': 'not_evaluated'}
    write(directory/'preflight.json', result)
    return result


def freeze(directory, labels_path):
    """冻结已目视核对的裁区、独立标签、材料、代码和请求；不授权真实发送。"""
    pre = read(directory/'preflight.json'); labels = read(labels_path)
    if not labels.get('review_complete') or labels.get('provenance') != 'tool_assisted_source_check':
        raise ValueError('source labels and crop review required before freeze')
    cases = {r['case'] for r in pre['rows']}
    if set(labels['cases']) != cases or any(not c.get('fields') for c in labels['cases'].values()):
        raise ValueError('label case coverage mismatch')
    paths = [directory/'preflight.json', labels_path, Path(pre['plan']), Path(pre['materials_path'])]
    paths += list((directory/'inputs').glob('*.json')) + list((directory/'requests').glob('*.json'))
    paths += list((ROOT/'backend/research_loop').rglob('*.py')) + [Path(__file__), ROOT/'backend/uv.lock', ROOT/'backend/pyproject.toml']
    for material in pre['materials']:
        paths += [Path(material['pdf_path']), Path(material['cache_path'])]
    for row in pre['rows']:
        prepared = read(directory/'inputs'/f'{row["id"]}.json')
        paths += [Path(a['path']) for a in prepared['assets']]
        if digest(read(directory/'requests'/f'{row["id"]}.json')) != row['request_hash']:
            raise ValueError('prepared request hash mismatch')
    output = {'files': {str(p.resolve()): file_hash(p) for p in paths}, 'labels_path': str(labels_path.resolve()),
              'planned_requests': 10, 'real_execution_authorized': False, 'save_policy': 'deny',
              'order': [r['id'] for r in pre['rows']], 'scope': 'component_only_not_daily_agent'}
    write(directory/'freeze.json', output)
    return output


def verify_freeze(directory):
    """拒绝代码/问题/图片/标注漂移，失败时不发请求。"""
    frozen = read(directory/'freeze.json')
    for path, expected in frozen['files'].items():
        if file_hash(path) != expected:
            raise ValueError('frozen file changed: ' + path)
    return frozen


def run(directory, *, execute_real=False, database=None):
    """仅在后续明确授权下执行一次固定批次；重复启动拒绝，不追加重试。"""
    if not execute_real:
        raise PermissionError('real execution requires separate authorization and --execute-real')
    if not database:
        raise ValueError('real comparison requires a new isolated MySQL database')
    frozen = verify_freeze(directory); pre = read(directory/'preflight.json')
    config = settings(directory, database)
    if public_settings(config) != pre['settings']:
        raise ValueError('current config differs from frozen config')
    if config.deepseek_model != 'deepseek-flash' or config.deepseek_base_url.rstrip('/') != 'https://api.deepseek.com':
        raise ValueError('frozen vision provider not configured')
    if not config.deepseek_api_key:
        raise ValueError('model credentials unavailable')
    # 元数据请求不生成内容；先记录实际模态，不能靠模型别名推测账号能力。
    write(directory/'run-started.json', {'order': frozen['order'], 'retry': False})
    try:
        with httpx.Client(timeout=120) as metadata_client:
            response = metadata_client.get(config.deepseek_base_url.rstrip('/')+'/models',
                headers={'Authorization': 'Bearer '+config.deepseek_api_key})
            response.raise_for_status(); metadata = response.json()
    except Exception as exc:
        write(directory/'metadata-failure.json', {'error_type':type(exc).__name__,
            'detail':str(exc).replace(config.deepseek_api_key,'[redacted]'),'generation_requests':0})
        raise
    write(directory/'model-metadata.json', metadata)
    model = next((m for m in metadata.get('data',[]) if m.get('id') == config.deepseek_model), None)
    if not model or 'image' not in model.get('input_modalities',[]):
        raise ValueError('account vision capability not confirmed; no generation sent')
    db = Database(config.database_url); db.ensure_schema()
    files = FileStore(config.data_dir, config.output_dir, ROOT)
    pdf = PDFProvider(files, config.parser_name, config.parser_version, config.parser_config)
    provider = DeepSeekProvider(config.deepseek_api_key, config.deepseek_model, config.deepseek_base_url, timeout=120)
    results = []
    try:
        node = VisualReadingNode(db, PaperService(db,None,pdf), provider)
        with api(config) as client:
            for row in pre['rows']:
                item = VisualInput.model_validate(row['item'])
                material = next(m for m in pre['materials'] if m['key']==row['material'])
                tid = create_task(client,item,material,files,'visual-real-'+row['id']+'-'+uuid4().hex)
                claimed = db.claim_task('visual-evaluation',lease_seconds=300,task_id=tid)
                write(directory/'real'/f'{row["id"]}-task.json', {'task_id':tid,'budget':db.assert_budget_policy(tid,config)})
                record = {'id':row['id'],'task_id':tid}
                try:
                    current = node.papers.prepare_visual(tid,item)
                    if digest(build_request(current,VisualConfig(),files.cache_dir)) != row['request_hash']:
                        raise ValueError('actual input differs from frozen request')
                    result = node.run(tid,item,worker_id='visual-evaluation',lease_token=claimed['lease_token'])
                    record['result'] = result
                    if actual_usage(result['usage']) is None:
                        raise RuntimeError('provider usage unknown; stop remaining cases')
                except Exception as exc:
                    # 异常详情留在本机审计；保留收据，无自动恢复/重发。
                    record['error'] = type(exc).__name__ + ': ' + str(exc)
                finally:
                    record['receipts'] = db.list_model_calls(tid)
                    write(directory/'real'/f'{row["id"]}.json',record); results.append(record)
                    status = 'failed' if 'error' in record else ('completed' if record['result']['status']=='complete' else 'partial')
                    db.update_task(tid,status=status)
                if 'error' in record:
                    break
    finally:
        provider.close(); pdf.close()
    write(directory/'real-results.json', {'results':results,'not_executed': [r['id'] for r in pre['rows'] if r['id'] not in {x['id'] for x in results}]})
    return results


def report(directory, scores_path=None):
    """按固定字段计数与配对变化报告；无标注不以关键词猜测答案正确。"""
    verify_freeze(directory)
    labels = read(read(directory/'freeze.json')['labels_path'])
    records = read(directory/'real-results.json')['results']
    scores = read(scores_path) if scores_path else {}
    expected_ids = {r['id'] for r in records}
    if set(scores) - expected_ids:
        raise ValueError('scores include an unexecuted case')
    output = {'cases':[], 'quality_status':'scored' if scores and set(scores)==expected_ids else 'awaiting_source_audit',
              'sampling':'four pages / five targeted questions; not a generalization benchmark'}
    for record in records:
        key = record['id']; case, group = key.rsplit('-',1); result = record.get('result') or {}
        row = {'id':key,'status':result.get('status','failed'), 'model_seconds':result.get('model_seconds'),
               'render_seconds':result.get('render_seconds'), 'usage':result.get('usage'), 'error':record.get('error')}
        if key in scores:
            score = scores[key]; expected = set(labels['cases'][case]['fields'])
            if set(score['fields']) != expected or any(v not in {'correct','incorrect','missing','uncertain'} for v in score['fields'].values()):
                raise ValueError('score fields do not match frozen labels')
            row['fields'] = score['fields']
            row['counts'] = {s:list(score['fields'].values()).count(s) for s in ('correct','incorrect','missing','uncertain')}
            row['unsupported_claims'] = score.get('unsupported_claims', [])
            row['source_audit_note'] = score.get('note','')
        output['cases'].append(row)
    output['paired_changes'] = []
    for case in labels['cases']:
        a,b = [next((r for r in output['cases'] if r['id']==case+'-'+g),{}) for g in ('text','visual')]
        if 'fields' in a and 'fields' in b:
            output['paired_changes'].append({'case':case,
                'improved':[f for f in a['fields'] if a['fields'][f]!='correct' and b['fields'][f]=='correct'],
                'regressed':[f for f in a['fields'] if a['fields'][f]=='correct' and b['fields'][f]!='correct']})
    output['median_model_seconds'] = {g: statistics.median(values) if (values := [r['model_seconds'] for r in output['cases'] if r['id'].endswith('-'+g) and r['model_seconds'] is not None]) else None for g in ('text','visual')}
    output['counts'] = {'planned':10,'executed':len(records),'not_executed':10-len(records),
        'complete':sum(r['status']=='complete' for r in output['cases']),
        'partial':sum(r['status']=='partial' for r in output['cases']),
        'failed':sum(r['status']=='failed' for r in output['cases'])}
    output['usage_totals'] = {g: sum(actual_usage(r.get('usage') or {}) or 0 for r in output['cases'] if r['id'].endswith('-'+g)) for g in ('text','visual')}
    output['usage_unknown_cases'] = [r['id'] for r in output['cases'] if actual_usage(r.get('usage') or {}) is None]
    output['usage_totals_scope'] = 'known usage only; unknown is excluded, not zero'
    output['cost'] = None
    write(directory/('report-scored.json' if scores_path else 'report-unscored.json'),output)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',nargs='?',default='preflight',choices=['preflight','freeze','run','report'])
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--plan',type=Path)
    parser.add_argument('--materials',type=Path);parser.add_argument('--labels',type=Path)
    parser.add_argument('--scores',type=Path);parser.add_argument('--database')
    parser.add_argument('--execute-real',action='store_true')
    args=parser.parse_args(); directory=args.directory.resolve();directory.mkdir(parents=True,exist_ok=True)
    if args.phase=='preflight':
        if not args.plan or not args.materials:parser.error('preflight requires --plan and --materials')
        preflight(directory,args.plan,args.materials,args.database)
    elif args.phase=='freeze':
        if not args.labels:parser.error('freeze requires --labels')
        freeze(directory,args.labels)
    elif args.phase=='run':run(directory,execute_real=args.execute_real,database=args.database)
    else:report(directory,args.scores)
    print(args.phase,'completed; consult local evidence for validation scope')
