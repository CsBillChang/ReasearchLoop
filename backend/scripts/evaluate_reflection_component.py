"""AG7.4固定四稿的组件对照；复用产品核验/局部patch，不运行外围研究调度。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlparse, urlunparse
import time

from research_loop.agents.graph import build_graph, initial_state
from research_loop.agents.reflection import ReflectionService
from research_loop.config import Settings
from research_loop.domain.evidence import Document, PageEvidence
from research_loop.domain.notes import CreateNoteInput
from research_loop.domain.tasks import TaskCreate, TaskResult, TaskStatus
from research_loop.providers.deepseek import RequestNotSent
from research_loop.services.notes import NoteService
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.tools.notes import specs as note_specs
from research_loop.tools.papers import specs as paper_specs
from research_loop.tools.registry import ToolRegistry
from research_loop.workers.executor import TaskExecutor
from scripts.evaluation_common import ROOT, OUT as PREVIOUS, exclusive, sha
from scripts.evaluate_reflection import FrozenLedger
from scripts.run_ag6 import EvaluationRunner, summarize_usage, write_json

OUT = ROOT / 'artifacts/ag7-4-reflection-component-20261011'
DB_NAME = 'researchloop_test_ag74_component_20261011'
PHASE_CAPS = {'initial_review': 4, 'revision': 3, 'final_review': 4}
QUESTIONS = ['当前简短稿中的两项科学陈述及其必要限定，分别能否由所提供的原文支持？只检查当前实际作出的陈述。']
REVISION_REQUEST = ('这是已有核验结果后的局部修订组件，不是一个新研究任务。'
    '调用方已完成任务分解、原文读取和初次核验。只根据实际核验结果和原文处理未解决陈述，'
    '用revise_note提交一次精确局部patch，保留其余文字；原文缺证应删除无据细节或准确披露缺证，不能猜测。'
    '不重新规划、检索、委派、整篇重写或保存。若无法安全修改则说明原因。'
    '工具成功后调用方会用当前稿和原文重新独立核验，不需要你再调用review_note。')


def config() -> Settings:
    """派生本机隔离配置；不暴露认证信息，不修改生产额度。"""
    current = Settings.from_env(ROOT)
    parsed = urlparse(current.database_url)
    if parsed.scheme != 'mysql' or parsed.hostname not in {'localhost', '127.0.0.1'}:
        raise ValueError('LOCAL_MYSQL_REQUIRED')
    return current.model_copy(update={'database_url': urlunparse(parsed._replace(path='/'+DB_NAME)),
        'data_dir': OUT/'data', 'output_dir': OUT/'notes', 'max_model_requests': 11,
        'max_tool_actions': 30, 'reserve_model_requests': 0, 'max_reflection_rounds': 1,
        'retrieval_enabled': False, 'discovery_backends': {}, 'request_timeout_seconds': 120})


class ComponentLedger(FrozenLedger):
    """给已存在的全批/单例额度增加阶段分账；未用的核验额度不转给修订。"""
    phase = 'initial_review'

    def reserve(self, kind, messages, max_tokens, tools=None):
        """发送前同时验证阶段额度和全批unknown/usage闸门。"""
        with self.lock:
            used = sum(r['case'] == self.case and r['phase'] == self.phase for r in self.records)
            if used >= PHASE_CAPS[self.phase]:
                raise RequestNotSent('COMPONENT_PHASE_CAP')
            n = super().reserve(kind, messages, max_tokens, tools)
            self.records[n]['phase'] = self.phase
            write_json(self.path, self.records)
            return n


class FrozenPDF:
    """仅回放既有解析缓存；正文/身份逐字复用，不重新解析或访问网络。"""
    def __init__(self, documents):
        """注入冻结文档、原始页文本以及合法本地PDF路径。"""
        self.documents = documents
        self.last_fetch = {'route': 'frozen_parsed_source_replay', 'network': False}

    def ingest(self, source, paper_id):
        """只接受本案例绑定的source和paper身份，不允许替换材料。"""
        item = next((d for d in self.documents if d['document']['paper_id'] == paper_id
                     and str(ROOT/d['material']['path']) == str(source)), None)
        if not item:
            raise PermissionError('source is outside frozen component material')
        return Document.model_validate(item['document'])

    ingest_pages = ingest

    def pages(self, document, start_page, count):
        """按真实物理页回放缓存；越界明确拒绝，不制造新页证据。"""
        item = next(d for d in self.documents if d['document']['document_id'] == document.document_id)
        end = start_page + count - 1 if count is not None else document.total_pages
        if start_page < 1 or end > document.total_pages or end < start_page:
            raise ValueError('page range exceeds frozen source')
        return [PageEvidence.model_validate(p) for p in item['pages'] if start_page <= p['page_number'] <= end]


class FrozenPapers(PaperService):
    """沿用真实授权和证据包服务；本实验不新提取图像，也不假装图像已理解。"""
    def inspect_media(self, task_id, target_ref, start_page, page_count, attachments):
        """拒绝实验外图像处理；现有文本和解析边界已完整交付。"""
        raise ValueError('组件评测仅复用冻结文字及既有assets定位，不执行新图像提取或视觉理解')


def seed(db, settings, case, material):
    """经任务服务绑定原稿与所有既有页；gold不传给角色，不创建笔记文件。"""
    attachment = {'input_ref': case['material'], 'paper_id': material['document']['paper_id'],
                  'local_path': str(ROOT/material['material']['path'])}
    # 固定组件合同在创建时冻结；不读取另一份生产配置或事后修改预算行。
    component_settings = settings.model_copy(update={'max_model_requests':11,'max_tool_actions':30,'reserve_model_requests':0})
    task = TaskService(db, component_settings).create(TaskCreate(prompt='独立核验既有短稿并做一次必要局部修订，不保存文件。',
        save_policy='deny', attachments=[attachment], idempotency_key='component-'+case['id']))
    papers = FrozenPapers(db, None, FrozenPDF([material]))
    read = papers.read(task.task_id, case['material'], attachments={case['material']: attachment})
    files = FileStore(settings.data_dir, settings.output_dir, ROOT)
    notes = NoteService(db, files)
    draft = notes.create(task.task_id, CreateNoteInput(title='固定稿 '+case['id'], markdown=case['markdown'],
        evidence_ids=[p['evidence_id'] for p in read['pages']])).model_dump(mode='json')
    db.append_event(task.task_id, 'feedback_draft_bound', {'draft_id': draft['draft_id'],
        'content_hash': draft['content_hash'], 'evidence_ids': draft['evidence_ids'],
        'origin': 'controlled_component_input', 'review_inherited': False})
    return task.task_id, draft, papers, notes, attachment


def revise_component(db, settings, tid, draft, review, papers, reflection, attachment, owner, factory):
    """运行现有Supervisor的局部patch能力；成功工具观察后立即回调用方复核。

    Returns:
        当前稿和图状态。没有成功patch时保留原稿；最多三请求，不另发第二修订批次。
    """
    state = initial_state(tid, REVISION_REQUEST, 'deny', {attachment['input_ref']: attachment},
                          max_model_requests=3, max_tool_actions=10, reserve_model_requests=0)
    state.update({'worker_id': owner[0], 'lease_token': owner[1], 'root_task_id': tid,
                  'draft_id': draft['draft_id'], 'evidence_ids': draft['evidence_ids']})
    state['messages'].append({'role': 'user', 'content': json.dumps({'current_draft': draft,
        'current_review': review, 'source_bundle': papers.evidence_bundle(tid, draft['evidence_ids'])}, ensure_ascii=False)})
    registry = ToolRegistry([s for s in note_specs()+paper_specs() if s.name in {'revise_note', 'read_paper'}])
    provider = factory(settings)
    try:
        graph = build_graph(provider, registry, {'papers': papers, 'reflection': reflection}, db,
                            max_model_requests=3, max_tool_actions=10, reserve_model_requests=0,
                            checkpoint_store=CheckpointStore(db))
        while not state.get('finished'):
            state = graph.invoke(state, config={'recursion_limit': 32}, interrupt_after=['execute_tools'])
            if state.get('draft_id') != draft['draft_id']:
                return db.get_draft(state['draft_id']), state
        return draft, state
    finally:
        provider.close()


def execute_case(db, settings, case, material, factory, phase=lambda value: None):
    """组件入口直接调用现有Worker核验函数；省去选稿/规划/调查的外围成本。"""
    started = time.monotonic()
    tid, baseline, papers, notes, attachment = seed(db, settings, case, material)
    worker = 'ag74-component'
    task = db.claim_task(worker, lease_seconds=2400, task_id=tid)
    owner = (worker, task['lease_token'])
    service = ReflectionService(db, 1, owner)
    executor = TaskExecutor(settings, db, provider_factory=factory)
    # 现有Reviewer仍暴露原工具schema；仅使用已提供全文，检索不可用时明确返回补读入口。
    executor.retrieval = None
    current, error, patch_state = baseline, None, None

    def review(draft):
        """复核只接收当前稿、原文、中性问题及程序差异；不传旧判词。"""
        return service.review(tid, draft['draft_id'], draft['content_hash'], '冻结组件核验',
            draft['evidence_ids'], QUESTIONS, lambda number, refs, questions: executor._review_once(
                tid, draft['draft_id'], draft['content_hash'], *owner, settings, papers, notes,
                number, refs, questions))

    initial = final = None
    try:
        phase('initial_review')
        initial = final = review(current)
        if initial.get('reflection', {}).get('can_continue') and any(
                f['status'] != 'supported' for f in initial.get('findings', [])):
            phase('revision')
            current, patch_state = revise_component(db, settings, tid, current, initial, papers, service, attachment, owner, factory)
            if current['draft_id'] != baseline['draft_id']:
                phase('final_review')
                final = review(current)
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    complete = bool(final and final.get('status') == 'supported' and final.get('content_hash') == current['content_hash'] and not error)
    gaps = list((final or {}).get('gaps', []))
    if not complete:
        gaps += [error or '组件未形成当前稿的supported结论；原始审校与未解决项保留。']
    TaskService(db).complete(tid, TaskStatus.completed if complete else TaskStatus.partial,
        TaskResult(answer_markdown=current['markdown'], draft_id=current['draft_id'], evidence_ids=current['evidence_ids'],
                   gaps=gaps, reason='frozen Reflection component; not full task orchestration'), owner=owner)
    family = [tid]+[s['subtask_id'] for s in db.list_subtasks(tid)]
    receipts = db.fetchall('SELECT c.* FROM model_calls c JOIN tasks t ON t.task_id=c.task_id WHERE t.root_task_id=? ORDER BY c.created_at', (tid,))
    return {'case': case['id'], 'task_id': tid, 'baseline': baseline, 'current': current,
        'initial_review': initial, 'final_review': final, 'reflection': service.snapshot(tid),
        'patch_state': patch_state, 'error': error, 'task': db.get_task(tid),
        'actions': {t: db.list_actions(t) for t in family}, 'receipts': receipts,
        'usage': summarize_usage(receipts), 'seconds': time.monotonic()-started,
        'checks': service.revision_checks(tid, current['draft_id'])}


def prepare():
    """复制四份冻结输入和缓存身份；旧数据库/文件仅读，不重跑论文或检索。"""
    from scripts.evaluation_common import settings as old_config
    from scripts.verify_ag73 import snapshot
    spec = json.loads((ROOT/'backend/tests/acceptance/ag74_cases.json').read_text())
    corpus = json.loads((PREVIOUS/'corpus.json').read_text())
    old_db = Database(old_config().database_url)
    materials = []
    for case in spec['reflection']:
        source = next(d for d in corpus['documents'] if d['key'] == case['material'])
        doc = old_db.fetchone('SELECT * FROM documents WHERE document_id=?', (source['document_id'],))
        doc['gaps'] = json.loads(doc.pop('gaps_json')); doc.pop('created_at', None)
        assert sha(ROOT/source['material']['path']) == source['material']['sha256']
        materials.append({'key': case['material'], 'document': doc, 'material': source['material'], 'pages': source['pages']})
    exclusive(OUT/'inputs.json', {'cases': spec['reflection'], 'materials': materials,
        'questions': QUESTIONS, 'revision_request': REVISION_REQUEST,
        'baseline': 'same frozen draft, no action; not an end-to-end no-reflection agent',
        'budget': {'physical_cap': 44, 'token_cap': 1200000, 'wall_seconds': 1800,
            'cases': [{'id': c['id'], 'request_cap': 11} for c in spec['reflection']], 'phase_caps': PHASE_CAPS},
        'metrics': {'claim_support': '5 supported, 2 contradicted, 1 insufficient; missing not zero',
            'repair': '2 contradicted claims; missing-fact removal/disclosure separately 1',
            'false_alarm': 'all 5 correct claims; uncovered and invalid reviewer output separate',
            'unnecessary_edit': 'exact unchanged correct sentences plus manual semantic audit',
            'validity': 'correct invocation and reliable evidence; positive gain not required'}})
    exclusive(OUT/'production-before.json', snapshot())
    exclusive(OUT/'previous-artifact-hashes.json', {str(p.relative_to(ROOT)): sha(p) for p in PREVIOUS.rglob('*') if p.is_file()})


def ensure_test_database(settings):
    """只新建固定组件测试库；拒绝生产或任意命名，不清空既有数据。"""
    import pymysql
    parsed = urlparse(settings.database_url)
    name = parsed.path.lstrip('/')
    if name not in {DB_NAME, DB_NAME+'_zero'} or parsed.hostname not in {'127.0.0.1', 'localhost'}:
        raise ValueError('COMPONENT_DATABASE_NOT_ISOLATED')
    with pymysql.connect(host=parsed.hostname, port=parsed.port or 3306,
            user=parsed.username, password=parsed.password) as conn:
        with conn.cursor() as cursor:
            cursor.execute('CREATE DATABASE IF NOT EXISTS `'+name+'` CHARACTER SET utf8mb4')


def preflight():
    """四稿原缓存在独立MySQL跑完整替身链；验证工程接线，不宣称论文语义通过。"""
    from tests.unit.test_reflection_component import ScriptedProvider
    settings = config()
    parsed = urlparse(settings.database_url)
    settings = settings.model_copy(update={'database_url': urlunparse(parsed._replace(path='/'+DB_NAME+'_zero')),
        'data_dir': OUT/'zero-data', 'output_dir': OUT/'zero-notes'})
    ensure_test_database(settings)
    db = Database(settings.database_url); db.ensure_schema()
    if db.fetchone('SELECT COUNT(*) AS n FROM tasks')['n']:
        raise ValueError('PREFLIGHT_DATABASE_MUST_BE_EMPTY')
    inputs = json.loads((OUT/'inputs.json').read_text())
    results = []
    for case in inputs['cases']:
        material = next(m for m in inputs['materials'] if m['key'] == case['material'])
        fake = ScriptedProvider(case['kind'], case=case, material=material)
        result = execute_case(db, settings, case, material, lambda settings: fake)
        exclusive(OUT/'zero-model'/(case['id']+'.json'), result)
        assert result['task']['status'] == 'completed', result['error']
        assert result['final_review']['content_hash'] == result['current']['content_hash']
        assert len(fake.messages) == (1 if case['kind'] == 'correct' else 3)
        if case['kind'] == 'correct':
            assert result['baseline']['markdown'] == result['current']['markdown']
        else:
            assert result['checks']['non_target_text_preserved']
            final_input = json.dumps(fake.messages[-1], ensure_ascii=False)
            assert 'synthetic-review-reason-do-not-leak' not in final_input
            assert 'synthetic-local-reason-do-not-leak' not in final_input
        results.append({'case': case['id'], 'fake_calls': len(fake.messages),
                        'pages_replayed': len(material['pages']), 'passed': True})
    assert not list(settings.output_dir.rglob('*.md'))
    assert db.fetchone("SELECT COUNT(*) AS n FROM tasks WHERE status IN ('queued','running')")['n'] == 0
    exclusive(OUT/'zero-model.json', {'passed': True, 'database': DB_NAME+'_zero',
        'results': results, 'real_model_calls': 0, 'semantic_quality': 'not evaluated by fake provider',
        'notes_written': 0, 'active_tasks': 0})


def freeze():
    """零模型验收完成后封存代码、输入、独立预算和配置；不调用业务模型。"""
    preflight = json.loads((OUT/'zero-model.json').read_text())
    if not preflight['passed']:
        raise ValueError('ZERO_MODEL_PREFLIGHT_REQUIRED')
    files = list((ROOT/'backend/research_loop').rglob('*.py')) + list((ROOT/'backend/scripts').glob('*.py'))
    files += [ROOT/'backend/tests/unit/test_reflection_component.py', OUT/'inputs.json', OUT/'zero-model.json']
    inputs = json.loads((OUT/'inputs.json').read_text())
    assert inputs['cases'] == json.loads((ROOT/'backend/tests/acceptance/ag74_cases.json').read_text())['reflection']
    assert inputs['questions'] == QUESTIONS and inputs['revision_request'] == REVISION_REQUEST
    assert inputs['budget']['phase_caps'] == PHASE_CAPS
    files += [OUT/'test.log', OUT/'check.log']
    files += [ROOT/m['material']['path'] for m in inputs['materials']]
    settings = config()
    exclusive(OUT/'freeze.json', {'frozen_at': datetime.now(timezone.utc).isoformat(),
        'files': {str(p.relative_to(ROOT)): sha(p) for p in files}, 'model': settings.deepseek_model,
        'provider_base_url': settings.deepseek_base_url, 'database': DB_NAME, 'budget': inputs['budget'],
        'output_tokens': 'omitted as in product Reviewer; conservative ledger reserve 32768; no context override',
        'no_search_parse_index_deep_read_save': True, 'max_repair_rounds': 1,
        'comparison': 'component-only; initial evidence preparation and outer planning excluded and disclosed'})


def run():
    """四例各一次；任何供应商/unknown/用量缺失停止余下发送，不补跑至成绩改善。"""
    frozen = json.loads((OUT/'freeze.json').read_text())
    for path, expected in frozen['files'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('FROZEN_FILE_CHANGED: '+path)
    settings = config()
    if (settings.deepseek_model, settings.deepseek_base_url) != (frozen['model'], frozen['provider_base_url']):
        raise ValueError('FROZEN_PROVIDER_CHANGED')
    ensure_test_database(settings)
    db = Database(settings.database_url); db.ensure_schema()
    if db.fetchone('SELECT COUNT(*) AS n FROM tasks')['n']:
        raise ValueError('REAL_COMPONENT_DATABASE_MUST_BE_EMPTY')
    inputs = json.loads((OUT/'inputs.json').read_text())
    exclusive(OUT/'started.json', {'at': datetime.now(timezone.utc).isoformat(), 'budget': frozen['budget']})
    ledger = ComponentLedger(OUT/'physical.json', inputs['budget'])
    runner = object.__new__(EvaluationRunner); runner.ledger = ledger; runner.cancel_hook = False
    results = []
    for case in inputs['cases']:
        ledger.case = case['id']
        material = next(m for m in inputs['materials'] if m['key'] == case['material'])
        result = execute_case(db, settings, case, material, runner.provider, lambda p: setattr(ledger, 'phase', p))
        exclusive(OUT/(case['id']+'.json'), result); results.append(result)
        print(case['id'], result['task']['status'], result['usage'], flush=True)
        usage = result['usage']
        if usage['unknown'] or usage['provider_failures'] or usage['missing_usage_call_ids'] or result['error']:
            break
    assert not list(settings.output_dir.rglob('*.md'))
    exclusive(OUT/'result.json', {'executed': [r['case'] for r in results],
        'not_executed': [c['id'] for c in inputs['cases'] if c['id'] not in {r['case'] for r in results}],
        'physical_requests': len(ledger.records), 'usage': summarize_usage([c for r in results for c in r['receipts']]),
        'markdown_files': [], 'quality_audit': 'pending independent source review; task status not semantic gold'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'preflight', 'freeze', 'run'])
    args = parser.parse_args()
    globals()[args.mode]()
