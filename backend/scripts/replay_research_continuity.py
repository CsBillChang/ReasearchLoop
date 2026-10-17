"""只读旧失败轨迹，离线重放来源校验与上下文整理；没有真实模型入口。"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from research_loop.agents.memory import ContextService, PREFIX, estimate
from research_loop.config import Settings
from research_loop.providers.deepseek import validate_messages
from research_loop.storage.mysql import Database
from research_loop.domain.context import project_research_handoffs


def check_handoff_projection(original: dict, projected: dict) -> dict:
    """从目录无损重建每个候选，防止靠丢来源或正文获得容量通过。

    Args:
        original: 当时已保存的完整当前任务，不读取未来发现。
        projected: 同一任务的模型投影。

    Returns:
        被核对的发现与来源数量；任一字段不同直接报错，不降级成通过。
    """
    pairs = [(original.get('research_progress'), projected.get('research_progress'))]
    assert len(original.get('subtasks', [])) == len(projected.get('subtasks', []))
    pairs += [(a.get('research_progress'), b.get('research_progress'))
              for a, b in zip(original.get('subtasks', []), projected.get('subtasks', []))]
    findings, sources = 0, 0
    for old, new in pairs:
        if old is None:
            assert new is None
            continue
        restored = json.loads(json.dumps(new))
        catalog = restored.pop('source_catalog', None)
        if catalog is not None:
            for finding in restored['findings']:
                finding['sources'] = [catalog[k] for k in finding.pop('source_refs')]
        assert restored == old, 'candidate text/source/version/status was changed by projection'
        findings += len(old.get('findings', []))
        sources += sum(len(f['sources']) for f in old.get('findings', []))
    assert original.get('save_policy') == projected.get('save_policy')
    assert original.get('research_plan') == projected.get('research_plan')
    return {'passed': True, 'findings': findings, 'source_references': sources}


def complementary_check(messages: list[dict], projected: list[dict], available: int) -> dict:
    """核对最近两个不同定向读页批次的互补正文是否同时可见。

    Args:
        messages: 当前时刻原始完整往返，不包含未来证据。
        projected: 待送模型的投影。
        available: 已扣除工具后的容量，用于判断整页组合是否确实容得下。

    Returns:
        能容纳的互补组合和逐页完整性；这只是可见性，不是事实支持判断。
    """
    calls = {c['id']:c['function'] for m in messages for c in m.get('tool_calls',[])}
    batches = {}
    for m in messages:
        fn = calls.get(m.get('tool_call_id'),{})
        if m.get('role') != 'tool' or fn.get('name') != 'read_paper': continue
        args = json.loads(fn['arguments'])
        if 'start_page' not in args and args.get('page_count') is None: continue
        data = json.loads(m['content']).get('data',{})
        by_doc = {}
        for p in data.get('pages',[]):
            by_doc.setdefault((p['document_id'],data.get('content_sha256')),[]).append(p)
        for key,pages in by_doc.items(): batches.setdefault(key,[]).append(pages)
    visible = {}
    def keep(source):
        """只收集精确原文段；摘录拼接不能伪装完整页。"""
        eid = source['evidence_id']
        if source.get('context_spans'):
            for span in source['context_spans']:
                start=source.get('char_start',0)+span['start']
                visible.setdefault(eid,[]).append((start,start+len(span['text']),span['text']))
        else:
            start=source.get('char_start',0)
            visible.setdefault(eid,[]).append((start,start+len(source['text']),source['text']))
    for m in projected:
        try: value=json.loads(m.get('content',''))
        except (ValueError,TypeError): continue
        if not isinstance(value,dict): continue
        if m.get('role')=='user' and 'archived_tool_groups' in value:
            for s in value.get('sources',[]):keep(s)
        if m.get('role')=='tool':
            for s in value.get('data',{}).get('pages',[])+value.get('data',{}).get('hits',[]):keep(s)
    checks=[]
    for key,groups in batches.items():
        last=groups[-1]; last_ids={p['evidence_id'] for p in last}
        previous=next((p for p in reversed(groups[:-1]) if {s['evidence_id'] for s in p}!=last_ids),None)
        if not previous:continue
        desired={p['evidence_id']:p for p in previous+last}
        # 只有整组确实容得下才要求完整页；大页仍另报告窗口/原文位置及容量检查。
        if sum(len(p['text'].encode('utf-8'))+500 for p in desired.values())>available//3:continue
        for eid,page in desired.items():
            position=0
            for start,end,text in sorted(visible.get(eid,[])):
                if page['text'][start:end]!=text:continue
                if start<=position:position=max(position,end)
            checks.append({'document_id':key[0],'evidence_id':eid,'page':page['page_number'],
                           'full_page_visible':position>=len(page['text'])})
    return {'checked_pages':len(checks),'pages':checks,'passed':all(p['full_page_visible'] for p in checks)}


def replay(artifact: Path, turn: int = 1) -> dict:
    """用临时SQLite重建原文身份，仅测试保存的请求输入。

    Args:
        artifact: 包含旧 result.json 或 turn-N.json 的本机证据目录。
        turn: 明确选择首问或追问，不默认拿首问代替追问验证。

    Returns:
        每份历史输入的配对、容量和原文定位检查，不声明真实模型任务成功。
    """
    path = artifact / (f'turn-{turn}.json' if (artifact / f'turn-{turn}.json').exists() else 'result.json')
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    trace = json.loads(path.read_text())
    receipts = trace['task']['receipts']
    rows = []
    with tempfile.TemporaryDirectory(prefix='researchloop-replay-') as tmp:
        root = Path(tmp)
        db = Database('sqlite:///' + str(root/'replay.db')); db.ensure_schema()
        settings = Settings(project_root=root, data_dir=root/'data', output_dir=root/'notes')
        documents = {}
        for action in trace['task']['actions']:
            if action['tool_name'] == 'read_paper' and action['status'] == 'succeeded':
                data = action['observation']['data']
                documents[data['document_id']] = {'document_id': data['document_id'], 'paper_id': data['paper_id'],
                    'content_sha256': data['content_sha256'], 'parser_name': 'offline-trace', 'parser_version': '1',
                    'config_sha256': '0'*64, 'total_pages': data['total_pages'], 'cache_path': 'not-opened'}
        for document in documents.values(): db.save_document(document)
        for page in trace['task']['evidence']: db.save_page(page)
        for task_id, state in trace['checkpoints'].items():
            db.create_task(task_id, None, task_id, state['prompt'], state['save_policy'], [],
                           root_task_id=state.get('root_task_id') or task_id, role=state.get('role', 'supervisor'))
        for context in trace['contexts']:
            task_id = context['task_id']; state = dict(trace['checkpoints'][task_id])
            original = json.loads(context['original_json'])['messages']
            old_projected = json.loads(context['projected_json'])
            # 保留当时实际组装的目标/权限头，避免以更少任务状态制造容量通过。
            headers = [m for m in old_projected if m.get('role')=='system' and str(m.get('content','')).startswith(PREFIX)]
            messages = [m for m in original if not (m.get('role')=='system' and str(m.get('content','')).startswith(PREFIX))]
            if headers: messages.insert(1, headers[-1])
            handoff = {'handoff_projected_calls': []}
            integrity = {'passed': True, 'findings': 0, 'source_references': 0}
            full_prepare = None
            if headers:
                # 使用当时实际状态而非未来数据库；失败前未发送的父输入也要重放。
                head = headers[-1]['content']
                split = head.index('\n{') + 1
                current = json.loads(head[split:])
                raw_current = json.loads(context['original_json']).get('current_task', current['current_task'])
                messages, current['current_task'], handoff = project_research_handoffs(messages, raw_current)
                integrity = check_handoff_projection(raw_current, current['current_task'])
                messages[1] = {**headers[-1], 'content': head[:split] + json.dumps(current, ensure_ascii=False, sort_keys=True)}
            # 授权仅来自该次已保存观察；不为模型恢复未来任务的新证据。
            db.execute('DELETE FROM task_evidence WHERE task_id=?', (task_id,))
            def grant(value):
                """递归提取真实观察中的页身份，绝不创建新正文或模型结论。"""
                if isinstance(value, dict):
                    if value.get('evidence_id') and db.get_page(value['evidence_id']):
                        db.mark_task_evidence(task_id, value['evidence_id'])
                    for child in value.values(): grant(child)
                elif isinstance(value, list):
                    for child in value: grant(child)
            for m in original:
                if m.get('role') == 'tool':
                    try: grant(json.loads(m['content']))
                    except (TypeError, ValueError): pass
            matching = [r for r in receipts if r['task_id']==task_id and r['sequence_no']<=context['sequence_no']]
            request = json.loads(max(matching, key=lambda r:r['sequence_no'])['request_json']) if matching else {'tools': []}
            tools = request.get('tools', [])
            available = json.loads(context['metrics_json'])['input_capacity']
            service = ContextService(db, None, settings)
            if headers and json.loads(context['metrics_json'])['after'] > available and state.get('role') == 'supervisor':
                # 对真实未发送的失败输入走完整装配路径。仅注入当时已保存的DB读回，
                # 不拿后来状态补齐缺口；另有隔离API链验证这些记录的生产生成方式。
                from research_loop.services.coverage import CoverageService
                state['model_requests'] = context['sequence_no'] - 1
                history = json.loads(head[split:])
                history.pop('current_task', None)
                with patch.object(service, 'history', return_value=history), \
                     patch.object(CoverageService, 'current_context', return_value=raw_current):
                    prepared, prepared_metrics = service.prepare(state, original, tools)
                validate_messages(prepared)
                payload = next(m['content'] for m in prepared if m.get('role') == 'system'
                               and m.get('content', '').startswith(PREFIX))
                prepared_current = json.loads(payload[payload.index('\n{')+1:])['current_task']
                full_prepare = {'after': prepared_metrics['after'], 'capacity': available,
                                'workspace': prepared_metrics.get('workspace_externalized', []),
                                'integrity': check_handoff_projection(raw_current, prepared_current),
                                'within_capacity': prepared_metrics['after'] <= available}
            if full_prepare:
                projected, metrics = prepared, prepared_metrics
            else:
                projected, metrics = service._working_set(state, messages, tools, available)
            if estimate({'messages': projected, 'tools': tools}) > available:
                projected, _ = service._project_sources(state, projected, tools, available)
            validate_messages(projected)
            exact, source_count = True, 0
            for message in projected:
                if message.get('role')!='user': continue
                try: payload=json.loads(message['content'])
                except (TypeError, ValueError): continue
                if not isinstance(payload,dict) or 'archived_tool_groups' not in payload: continue
                for source in payload['sources']:
                    page=db.get_page(source['evidence_id'])
                    exact &= page['text'][source['char_start']:source['char_end']]==source['text']
                    exact &= source['source_sha256']==documents[source['document_id']]['content_sha256']
                    source_count += 1
            size = estimate({'messages': projected, 'tools': tools})
            receipt = next((r for r in receipts if r['task_id']==task_id and r['sequence_no']==context['sequence_no']), None)
            usage = json.loads(receipt['usage_json']) if receipt and receipt.get('usage_json') else {}
            rows.append({'task_id':task_id,'sequence':context['sequence_no'],'role':state.get('role'),
                'old_projected_bytes':json.loads(context['metrics_json'])['after'],
                'serialized_bytes':size,'estimated_tokens_upper_bound':size,'capacity_tokens':available,
                'actual_original_prompt_tokens':usage.get('prompt_tokens'),
                'within_capacity':size<=available,'exact_working_sources':bool(exact),
                'complementary':complementary_check(messages,projected,max(0,available-estimate({'tools':tools})-64)),
                'old_complementary':complementary_check(messages,old_projected,max(0,available-estimate({'tools':tools})-64)),
                'working_sources':source_count,**metrics})
            rows[-1].update(handoff)
            rows[-1]['handoff_integrity'] = integrity
            rows[-1]['full_prepare'] = full_prepare
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before_hash
    return {'scope':'saved-input replay, not a real task run', 'model_requests':0,
            'input_sha256':before_hash, 'cases':rows,
            'passed':bool(rows) and all(r['within_capacity'] and r['exact_working_sources'] and r['complementary']['passed'] for r in rows),
            'limitations':['No historical findings invented; stage submission uses separate synthetic integration tests.',
                           'UTF-8 byte upper bound is conservative, not the exact supplier tokenizer.']}


def main():
    """显式给出材料与输出路径，脚本不会自动获取PDF或调用模型。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--turn', type=int, default=1, choices=(1, 2))
    parser.add_argument('--report', type=Path, required=True)
    args=parser.parse_args()
    if args.report.resolve().is_relative_to(args.artifact.resolve()):
        parser.error('report must not modify the original artifact directory')
    report=replay(args.artifact, args.turn)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'cases':len(report['cases']),'passed':report['passed'],'model_requests':0}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__': main()
