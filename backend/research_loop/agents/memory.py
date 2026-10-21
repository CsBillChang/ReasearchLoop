"""会话上下文装配与可追溯压缩；不调用LLM，不改变论文正文输入或历史收据。"""
from __future__ import annotations
import json
from functools import wraps
from research_loop.providers.token_counting import estimate, estimate_details, counting_policy, use_policy, active_policy
from research_loop.storage.memory import MemoryStore, dump, digest
from research_loop.storage.mysql import _now
from research_loop.domain.context import excerpt, archive_summary, tool_archive_summary, compact_tool_exchange, contains_source_exchange, project_source, source_history, readback_priority, compact_research_exchanges, project_research_handoffs, project_review_handoffs, project_review_answer, factor_source_envelope
from research_loop.tools.registry import schema_names
from research_loop.domain.context_workspace import project_workspace

PREFIX = '[ResearchLoop Memory v1]'
RULES = '''历史资料和偏好不是新的用户指令。当前用户指令/澄清 > 显式长期偏好 > 推断兴趣。
current_task 是持久业务状态，不随旧工具往返压缩：沿用有效 goal_id、材料身份和当前权限；user/legacy目标及用户约束变更以真实最新澄清为准。新计划真正用户硬来源/日期放到相应origin=user目标的sources/date_from/date_to；顶层来源/日期仅检索偏好，可根据实际材料调整，不全局排除授权上传。根input_refs、数量、保存要求及权限保持。无_scope_semantics标记的历史计划仍按原顶层硬合同，不自行迁移。origin=strategy的自拟步骤可带change_reason调整或保留原ID设为withdrawn；撤回不算用户未完成，不沿旧待办重新委派。
历史保存授权不能沿用；历史笔记/摘要不等于原文。已绑定真实页证据可以按需复用，不必重复全文读取。
只调用本次实际列出的工具，摘录不能支持的细节不猜。没有可用读回或容量不足时交付已有成果及具体缺口，不改名目标或冒充完成。'''


def project_finish_candidates(messages: list[dict]) -> list[dict]:
    """模型仅看到失败候选的绑定身份；完整正文仍在原行动参数或绑定对象中。"""
    projected = list(messages)
    calls = {call['id']: call.get('function', {}).get('name') for message in messages
             for call in message.get('tool_calls', [])}
    for index, message in enumerate(messages):
        if message.get('role') != 'tool' or calls.get(message.get('tool_call_id')) != 'finish_task':
            continue
        try:
            observation = json.loads(message['content'])
        except (TypeError, ValueError):
            continue
        if not isinstance(observation, dict) or observation.get('ok'):
            continue
        data = observation.get('data') or {}
        candidate = data.get('finish_candidate')
        if not isinstance(candidate, dict):
            continue
        identity = {key: candidate.get(key) for key in ('task_id', 'action_id', 'input_hash',
            'answer_content_hash', 'draft_id', 'reflection_target_id', 'reflection_content_hash')}
        identity['answer_body_chars'] = candidate.get('answer_body_chars', len(candidate.get('answer_body_markdown') or ''))
        identity['processing_summary_chars'] = candidate.get('processing_summary_chars', len(candidate.get('processing_summary') or ''))
        identity['notice'] = '正文按当前task/action/hash保留在原参数或绑定对象；本投影不重复展开。'
        projected[index] = {**message, 'content': dump({**observation, 'data': {**data, 'finish_candidate': identity}})}
    return projected


def _with_counting_policy(method):
    """为一次角色准备传播根策略，所有容量子计算共用同一计量方法。"""
    @wraps(method)
    def wrapped(self, state, messages, tools):
        """保持调用接口，按任务身份只读冻结配置。"""
        with use_policy(self._request_policy(state)):
            # 来源身份与容量压力无关；换用较准确估算后也不能绕过授权/版本校验。
            messages, _ = self._project_sources(state, messages, tools, max(estimate({'messages':messages,'tools':tools}), self._available())+1)
            return method(self, state, messages, tools)
    return wrapped


class ContextService:
    """装配单次请求上下文，原始会话/工具消息保持持久化可追溯。"""

    def __init__(self, db, memory, settings):
        """装配角色内上下文；Researcher仅投影来源，不继承父会话或偏好。"""
        self.db,self.memory,self.settings = db,memory,settings
        self.store = MemoryStore(db)

    def _request_policy(self, state):
        """新根使用冻结策略；旧有账本按原计量快照继续，不回填历史。"""
        from research_loop.storage.token_budget import TokenBudgetStore
        task = self.db.get_task(state['task_id']) or {}
        root = state.get('root_task_id') or task.get('root_task_id') or state['task_id']
        store = TokenBudgetStore(self.db)
        policy = store.counting_policy(root)
        if policy:
            if policy['model'] != self.settings.deepseek_model:
                raise ValueError('TOKEN_POLICY_MODEL_CHANGED: 当前模型与根任务冻结模型不同；未发送新请求')
            return policy
        if self.db.fetchone('SELECT root_task_id FROM task_token_budgets WHERE root_task_id=?', (root,)):
            return {'version':1, 'method':'utf8_bytes_upper_bound', 'legacy_context':True}
        return counting_policy(self.settings)

    def _limit(self, key):
        """读取本次冻结容量；无新策略历史任务保持调用方旧配置。"""
        return active_policy().get(key, getattr(self.settings, key))

    def _available(self):
        """模型窗口扣除单请求输出与安全余量，不与累计成本额度混用。"""
        return self._limit('context_window_tokens')-self._limit('context_output_reserve')-self._limit('context_safety_tokens')

    def _validate_source_observation(self, task_id: str, observation: dict) -> None:
        """逐页校验当前原文；仅因用户范围失效的历史页退出事实投影。"""
        from research_loop.services.papers import PaperService
        from research_loop.domain.reading import MaterialScopeDenied
        from research_loop.domain.evidence import validate_source_window
        papers = PaperService(self.db, None, None)
        data = observation.get('data') or {}
        excluded = []
        try:
            for key, kind in (('pages', 'page'), ('hits', 'hit')):
                rows = data.get(key, [])
                if key == 'hits' and data.get('hits_defaults'):
                    rows = [{**data['hits_defaults'], **item} for item in rows]
                valid = []
                for item in rows:
                    try:
                        source = papers.validate_source_observation(task_id, item, data, kind=kind)
                    except MaterialScopeDenied:
                        # 原件/字符窗口损坏仍须停止；范围变化只移除当前事实，不改档案。
                        page = self.db.get_page(item['evidence_id'])
                        document = self.db.fetchone('SELECT * FROM documents WHERE document_id=?', (page['document_id'],))
                        validate_source_window(item, data, page, document, kind=kind)
                        excluded.append({'evidence_id': item['evidence_id'], 'document_id': page['document_id'],
                            'page_number': page['page_number'], 'reason': 'current_user_material_page_scope'})
                        continue
                    item['readback'] = source['readback']
                    valid.append(item)
                if key in data:
                    data[key] = valid
            if excluded:
                data['excluded_historical_sources'] = excluded
                data['historical_scope_notice'] = '旧记录原件保留；这些页已不在当前用户允许范围，不作为本次原文依据。'
                if 'returned_pages' in data:
                    data['returned_pages'] = [item['page_number'] for item in data.get('pages', [])]
                if 'read_coverage' in data:
                    data['read_coverage'] = {**data['read_coverage'], 'returned_pages': data.get('returned_pages', []),
                        'full_document_returned': False, 'all_extracted_text_returned': False,
                        'scope_origin': 'current_user_material_scope_projection'}
        except (PermissionError, ValueError, KeyError, TypeError) as exc:
            raise ValueError('CONTEXT_SOURCE_STALE: 来源未授权、范围或完整版本/窗口失效') from exc

    def _project_sources(self, state, messages, tools, available):
        """超容量先裁历史，优先连续展示最新补读；参数、身份和全局上限不变。

        Returns:
            投影消息及被处理工具ID；若必要状态仍超限，由调用方明确停止。
        """
        calls = {c['id']: c['function'] for m in messages for c in m.get('tool_calls', [])}
        source_rows = []
        projected = list(messages)
        for index, message in enumerate(messages):
            function = calls.get(message.get('tool_call_id'), {})
            if message.get('role') != 'tool' or function.get('name') not in {'read_paper', 'acquire_paper_pdf', 'retrieve_evidence'}:
                continue
            try:
                observation = json.loads(message['content'])
            except (ValueError, TypeError):
                # 历史纯文本工具结果可归档读回，不能当作有身份的原文页。
                continue
            if not isinstance(observation, dict) or not observation.get('ok'):
                continue
            data = observation.get('data', {})
            self._validate_source_observation(state['task_id'], observation)
            if data.get('excluded_historical_sources'):
                message = {**message, 'content': dump(observation)}
                projected[index] = message
            source_rows.append((index, message, observation, function))
        names = schema_names(tools)
        changed = []
        protected = readback_priority(source_rows)
        attempts = [(None, cap, None, None) for cap in (1600, 800, 400, 200, 80)]
        attempts += [(limit, 80, None, None) for limit in (64, 32, 16, 8, 4, 2, 1)]
        # 定向读页是显式补回原文的操作；先缩检索命中，不能让二者同时降成短摘录。
        attempts += [(1, 80, None, cap) for cap in (1600, 800, 400, 80)]
        attempts += [(1, 80, cap, 80) for cap in (8192, 4096, 1600, 400, 80)]
        for limit, cap, page_cap, hit_cap in attempts:
            if estimate({'messages': projected, 'tools': tools}) <= available:
                break
            changed = []
            for index, message, observation, function in source_history(source_rows, state['prompt'], limit, protected):
                args = json.loads(function['arguments'])
                target = args.get('target_ref')
                query = args.get('query') or state['prompt']
                priority_cap = page_cap if function['name'] in {'read_paper', 'acquire_paper_pdf'} else hit_cap
                value = project_source(observation, query, cap, 'read_paper' in names, target, protected, priority_cap)
                # 整批计量前先去掉重复封装，不能先把所有命中裁到极短再计算节省。
                # 每轮从已校验原观察产生新副本，defaults不会写回原件或累积嵌套。
                projected[index] = {**message, 'content': dump(factor_source_envelope(value))}
                changed.append(message['tool_call_id'])
        return projected, changed

    def _working_set(self, state, messages, tools, available):
        """校验原文后整理工作集，只有实际可读回来源时归档来源工具组。"""
        names = schema_names(tools)
        if ('read_paper' not in names and 'read_memory' not in names
                and contains_source_exchange(messages)):
            return messages, {'archived_groups': 0}
        # 非来源的旧长参数也可能占满窗口。整组移入原件档案，不能把真实
        # assistant改造成缺失推理字段的摘要；最新往返仍完整保留。
        # 复用现有严格版本校验，容量给足时不会裁正文；返回值不写回原始消息。
        self._project_sources(state, messages, tools, max(estimate({'messages': messages, 'tools': tools}), available) + 1)
        from research_loop.services.research_progress import ResearchProgressService
        progress = ResearchProgressService(self.db).retained(state['task_id'])
        query = state['prompt'] + '\n' + '\n'.join(progress['gaps']) + '\n' + (progress.get('next_question') or '')
        # 工作集容量扣除工具schema；上层已另行扣除动态Memory头。
        body_capacity = max(0, available - estimate({'tools': tools}) - 64)
        return compact_research_exchanges(messages, query, body_capacity)

    def _capacity_metrics(self, metrics, messages, tools):
        """分别披露Token估算、格式/图像、传输字节及输出预留，usage是真值。"""
        details = estimate_details({'messages': messages, 'tools': tools})
        metrics.update(serialized_input_bytes=details['transport_bytes'],
                       estimated_input_tokens=details['estimated_input_tokens'],
                       token_estimate_is_exact=False, estimate_method=details['method'],
                       counting_accuracy=details['accuracy'],
                       capacity_estimate_notice=details['notice'],
                       image_tokens_upper_bound=details['image_tokens_upper_bound'],
                       format_overhead_tokens=details['format_overhead_tokens'],
                       tokenizer_sha256=details['tokenizer_sha256'],
                       output_reserve_tokens=self._limit('context_output_reserve'),
                       model_max_output_tokens=self._limit('model_max_output_tokens'))
        return metrics

    def prepare_role(self, state, messages, tools):
        """Researcher/Reviewer共用容量准备；Reviewer不继承父会话与研究发现。"""
        return self.prepare_researcher(state, messages, tools)

    @_with_counting_policy
    def prepare_researcher(self, state, messages, tools):
        """子角色仅压缩自身授权原文，保留原始往返到档案；不注入父会话记忆。"""
        available = self._available()
        before = estimate({'messages': messages, 'tools': tools})
        original = list(messages)
        from research_loop.services.research_progress import ResearchProgressService
        progress = ResearchProgressService(self.db).retained(state['task_id'])
        if progress['revision'] and state.get('role') != 'reviewer':
            messages, current, _ = project_research_handoffs(messages, {'research_progress':progress})
            # 与父角色共用同一引用路径；只注入自身成果，不继承父会话或权限。
            messages = messages + [{'role': 'system', 'content': '当前已提交阶段成果（候选，不是原文或批准）：' + dump({'current_task':current})}]
        projected, working = self._working_set(state, messages, tools, available) if estimate({'messages': messages, 'tools': tools}) > available * self._limit('context_compress_ratio') else (messages, {'archived_groups': 0})
        projected, changed = self._project_sources(state, projected, tools, available) if estimate({'messages': projected, 'tools': tools}) > available else (projected, [])
        after = estimate({'messages': projected, 'tools': tools})
        metrics = {'before': before, 'after': after, 'input_capacity': available,
                   'estimate_method': 'utf8_bytes_upper_bound', 'source_projected_calls': changed, **working}
        self._capacity_metrics(metrics, projected, tools)
        archive_id = digest([state['task_id'], messages, tools])
        with self.store.transaction() as tx:
            if not tx.one('SELECT snapshot_id FROM context_archives WHERE snapshot_id=?', (archive_id,)):
                tx.execute('INSERT INTO context_archives(snapshot_id,task_id,sequence_no,original_json,projected_json,metrics_json,created_at) VALUES(?,?,?,?,?,?,?)',
                           (archive_id,state['task_id'],int(state.get('model_requests',0))+1,dump({'messages':original}),dump(projected),dump(metrics),_now()))
        if after > available:
            raise ValueError('CONTEXT_CAPACITY: 子任务必要状态超过容量；未发送模型，原始证据和行动已保留')
        return projected, metrics

    def _restore_sources(self, task_id: str, messages: list[dict], accessible: set[str]) -> tuple[list[dict], int]:
        """按原档案恢复当前任务的压缩往返，不扩大模型的工具权限。

        Args:
            task_id: 当前任务，用于档案来源隔离。
            messages: 可能已有压缩摘要的持久消息。
            accessible: 本次请求实际公开的工具名，含收尾范围收缩。

        Returns:
            恢复后的真实往返及恢复组数；旧assistant摘要不沿用，原件保持不变。

        Raises:
            ValueError: 档案缺失、范围不匹配或不是同一任务的记录。
        """
        output, restored_groups = [], set()
        for message in messages:
            tool_summary = tool_archive_summary(message)
            summary = tool_summary or archive_summary(message)
            if not summary:
                output.append(message)
                continue
            row = self.db.fetchone('SELECT original_json FROM context_archives WHERE snapshot_id=? AND task_id=?',
                                   (summary.get('source_id'), task_id))
            if not row:
                raise ValueError('CONTEXT_ARCHIVE_UNAVAILABLE: 当前任务档案缺失或越权；未据摘录继续生成')
            source_messages = json.loads(row['original_json'])['messages']
            bounds = summary.get('original_message_range', [])
            if (len(bounds) != 2 or not all(isinstance(x, int) for x in bounds)
                    or not 0 <= bounds[0] <= bounds[1] < len(source_messages)):
                raise ValueError('CONTEXT_ARCHIVE_RANGE: 原始往返位置失效')
            group = source_messages[bounds[0]:bounds[1]+1]
            ids = [c['id'] for m in group for c in m.get('tool_calls', [])]
            if ids != [c['id'] for c in summary.get('tools', [])]:
                raise ValueError('CONTEXT_ARCHIVE_RANGE: 原始工具身份不匹配')
            # 原件仍保存，但撤销/丢失的授权页不能借旧档案重新进入事实输入。
            for entry in group:
                if entry.get('role') != 'tool':
                    continue
                try:
                    observation = json.loads(entry.get('content') or '')
                except (ValueError, TypeError):
                    continue
                if not isinstance(observation, dict) or not observation.get('ok'):
                    continue
                # 页与检索命中同样校验；归档不是绕过版本/窗口/授权的入口。
                self._validate_source_observation(task_id, observation)
                if observation.get('data', {}).get('excluded_historical_sources'):
                    entry['content'] = dump(observation)
            if tool_summary:
                # 新预览保留了真实assistant；只按同一组身份恢复这一条tool原件。
                assistant = next((entry for entry in reversed(output) if entry.get('role') == 'assistant'), None)
                call_id = message.get('tool_call_id')
                entry = next((entry for entry in group[1:] if entry.get('tool_call_id') == call_id), None)
                if (assistant != group[0] or entry is None or summary.get('tool_call_id') != call_id):
                    raise ValueError('CONTEXT_ARCHIVE_RANGE: 工具预览与原始assistant往返不匹配')
                output.append(entry)
            else:
                # 历史assistant摘要不能继续冒充真实模型消息。先恢复原组，再按
                # 本次权限和容量压缩tool内容；不补造或猜测供应商推理字段。
                output.extend(group)
            restored_groups.add((summary.get('source_id'), *bounds))
        return output, len(restored_groups)

    @staticmethod
    def _archive_notice(accessible: set[str]) -> str:
        """读回提示与请求schema一致，不替受限任务偷偷开放记忆工具。"""
        if 'read_memory' in accessible:
            return '已执行往返的摘录；按readback中的source_type/source_id/offset分页read_memory；勿重新执行副作用。'
        return '已执行往返的摘录；当前请求无档案读回工具。仅使用保留的真实依据与当前任务状态，缺失细节明确说明。'

    def _redact(self, value):
        """撤销偏好的原句不重新进入模型上下文；历史原件继续留库供审计。"""
        revoked={}
        for row in self.db.fetchall("SELECT content,source_json FROM memory_entries WHERE status='revoked'"):
            source=json.loads(row['source_json'])
            if source.get('task_id'):
                revoked.setdefault(source['task_id'],[]).extend([row['content'],source.get('quote','')])
        def scrub(item):
            """只处理被撤销记录所属历史任务，不删掉其他有效偏好的相同词语。"""
            if isinstance(item,list):
                return [scrub(x) for x in item]
            if isinstance(item,dict):
                phrases=revoked.get(item.get('task_id'),[])
                result={}
                for k,v in item.items():
                    if k=='preferences':
                        result[k]=v
                    elif isinstance(v,str) and k in {'content','prompt','answer_excerpt'}:
                        for phrase in phrases:
                            if phrase:v=v.replace(phrase,'[已撤销偏好]')
                        result[k]=v
                    else:result[k]=scrub(v)
                return result
            return item
        return scrub(value)

    def history(self, task_id: str) -> dict:
        """读取当前消息之前的会话、有效偏好和相关历史，身份来自数据库。"""
        from research_loop.services.conversations import ConversationService
        link = self.db.fetchone('SELECT * FROM conversation_links WHERE task_id=?',(task_id,))
        if not link:
            return {}
        ConversationService(self.db).view(link['conversation_id'],0,1)
        current = self.db.fetchone('SELECT sequence_no FROM conversation_messages WHERE message_id=?',(link['message_id'],))
        rows = self.db.fetchall('SELECT message_id,task_id,role,content,sequence_no FROM conversation_messages WHERE conversation_id=? AND sequence_no<? ORDER BY sequence_no',(link['conversation_id'],current['sequence_no']))
        task = self.db.get_task(task_id)
        recent = self.settings.memory_recent_messages
        history = []
        for index,row in enumerate(rows):
            compact = len(rows)-index > recent
            history.append({**row,'content':excerpt(row['content'],task['prompt']) if compact else row['content'],'excerpted':compact and len(row['content'])>800})
        related = self.memory.search(task['prompt'],'research_history',3)
        # 同会话最近研究的论文身份独立保留，不能靠摘要中的“第二篇”重新猜ID。
        records = self.db.fetchall('SELECT record_json FROM research_memories WHERE conversation_id=? ORDER BY created_at DESC LIMIT 3',(link['conversation_id'],))
        identities = []
        for r in records:
            record = json.loads(r['record_json'])
            identity={k:record.get(k) for k in ('task_id','status','paper_ids','evidence_ids','gaps')}
            identity['papers']=[{k:p.get(k) for k in ('paper_id','title','source','year')} for p in record.get('papers',[])[:8]]
            # 历史阶段发现是候选成果；精确原文仍需当前任务授权工具读回。
            identity['stage_findings'] = [{'task_id': p['task_id'], 'revision': p['revision'],
                'findings': [excerpt(f['text'], task['prompt'], 500) for f in p['findings']],
                'finding_metadata': [{'finding_id': f['finding_id'], 'revision': f.get('revision', 1),
                    'text_index': index, 'source_integrity': f.get('source_integrity', 'historical_unchecked'),
                    'current_source_authorized': False, 'source_task_id': p['task_id'],
                    'readback': {'source_type': 'research_history', 'source_id': record['task_id']}} for index, f in enumerate(p['findings'])],
                'notice': '历史候选，非本次已核验原文或保存授权'} for p in record.get('research_progress', [])]
            identity['total_papers']=len(record.get('paper_ids') or [])
            identity['paper_ids']=(record.get('paper_ids') or [])[:8]
            identity['evidence_ids']=(record.get('evidence_ids') or [])[:8]
            identity['gaps']=[excerpt(str(g),task['prompt'],200) for g in (record.get('gaps') or [])[:3]]
            identity['readback']={'source_type':'research_history','source_id':record['task_id']}
            identities.append(identity)
        for hit in related:
            record=hit['record']
            hit['record']={k:record.get(k) for k in ('task_id','status','prompt','paper_ids','gaps')}
            hit['record']['answer_excerpt']=excerpt(record.get('answer',''),task['prompt'],500)
            hit['record']['prompt']=excerpt(record.get('prompt',''),task['prompt'],400)
        # 近期消息完整保留；旧消息过多时只保留可读回ID清单，不能挤掉当前指令。
        preferences = self.memory.active()
        for preference in preferences:
            if preference['source'].get('type') == 'behavioral_interest':
                # 完整来源保留在画像中；模型只需计数与主题，不按每轮重复注入历史引句。
                preference['source'] = {key: value for key, value in preference['source'].items() if key != 'evidence'}
        context={'conversation_id':link['conversation_id'],'preferences':preferences,'messages':history,'recent_research':identities,'related_history':related}
        cap=self.settings.memory_history_tokens
        for row in history[:-recent]:
            if estimate(context)<=cap:
                break
            row['content']='[旧消息已归档，请用read_memory(source_type=message, source_id='+row['message_id']+')读回]'
            row['excerpted']=True
        omitted = 0
        while estimate(context)>cap and len(history)>recent:
            history.pop(0)
            omitted += 1
        context['older_messages_omitted']=omitted
        context['older_messages_readback']={'source_type':'conversation','source_id':link['conversation_id']}
        context=self._redact(context)
        return context

    def _current_review_answer(self, task_id: str, current: dict) -> dict | None:
        """保留唯一明确来源的当前正文，优先局部新版本，不猜最近一条回答。

        Args:
            task_id: 当前会话任务；旧消息权限由ConversationService校验。
            current: 权威任务状态及当前核验版本。

        Returns:
            当前正文和身份；无明确来源或多个不同对象时不擅自选择。
        """
        from research_loop.services.conversations import ConversationService
        from research_loop.agents.reflection import ReflectionService
        reference = ConversationService(self.db).referenced_answer(task_id)
        candidates = {r['target']['target_id']: r['target'] for r in current.get('research_reviews', [])
                      if r.get('target', {}).get('target_type') == 'answer'
                      and (not reference or r['target'].get('origin', {}).get('answer_source', {}).get('source_id') == reference['source_id'])}
        if len(candidates) == 1:
            target = ReflectionService(self.db).research_target(task_id, next(iter(candidates)))['current']
            return {'text': target['items'][0]['text'], 'target_id': target['target_id'],
                    'content_hash': target['content_hash'],
                    'source_id': target.get('origin', {}).get('answer_source', {}).get('source_id'),
                    'is_original_paper_evidence': False, 'save_permission_inherited': False,
                    'notice': '当前待核验正文；来源版本与判断分离，不表示已批准。'}
        if candidates or not reference:
            return None
        return {'text': reference['content'], 'source_id': reference['source_id'],
                'content_sha256': reference['content_sha256'], 'source_type': 'message',
                'is_original_paper_evidence': False, 'save_permission_inherited': False,
                'notice': '用户明确引用的已有回答；程序可直接交给Reviewer，旧回答不是原文。'}

    @_with_counting_policy
    def prepare(self, state: dict, messages: list[dict], tools: list[dict]) -> tuple[list[dict],dict]:
        """按容量投影请求；整组压缩旧工具往返并保存原文，绝不留下孤立tool回复。

        Raises:
            ValueError: 当前要求、近期消息或工具schema本身超过安全容量。
        """
        from research_loop.services.coverage import CoverageService
        from research_loop.services.research_progress import ResearchProgressService
        available = self._available()
        size = estimate({'messages': messages, 'tools': tools})
        # 非会话短任务保持原始消息合同；只有阶段成果或容量压力时装配工作集。
        if (not self.db.fetchone('SELECT task_id FROM conversation_links WHERE task_id=?', (state['task_id'],))
                and not ResearchProgressService(self.db).snapshot(state['task_id'])['revision']
                and size < available * self._limit('context_compress_ratio')):
            return messages, self._capacity_metrics({'before': size, 'after': size, 'input_capacity': available,
                'estimate_method': 'utf8_bytes_upper_bound', 'compressed_groups': []}, messages, tools)
        accessible = schema_names(tools)
        original=[m for m in messages if not (m.get('role')=='system' and str(m.get('content','')).startswith(PREFIX))]
        # 档案仍保存输入原件；只对本次模型投影恢复已经取得的真实往返。
        projected, restored = self._restore_sources(state['task_id'], original, accessible)
        context=self.history(state['task_id'])
        if 'read_memory' not in accessible:
            # 清掉历史投影中无法执行的补读指引，偏好/当前要求仍保留。
            context = json.loads(dump(context))
            context.pop('older_messages_readback', None)
            for item in context.get('recent_research', []):
                item.pop('readback', None)
            for item in context.get('messages', []):
                if str(item.get('content', '')).startswith('[旧消息已归档，请用read_memory'):
                    item['content'] = '[旧消息已归档；当前请求无历史补读工具，不据此猜测细节]'
        context['current_task'] = CoverageService(self.db).current_context(state['task_id'], state)
        review_answer = self._current_review_answer(state['task_id'], context['current_task'])
        if review_answer:
            context['current_task']['current_review_answer'] = review_answer
        context['available_tools'] = sorted(accessible)
        rules = RULES
        if 'finish_task' in accessible:
            from research_loop.services.interests import InterestService
            context['interest_inference'] = InterestService(self.db).context(state['task_id'])
            if context['interest_inference']['enabled']:
                rules += ('\n本新任务已启用行为兴趣旁路：正常finish_task时评估日常提问、用户选择阅读或比较维度中体现的研究主题，'
                    '随同这一次结束响应附interest_candidates；没有足够线索明确返回[]，不另调模型或remember工具。'
                    '无需用户声明长期兴趣；一次行为只形成tentative线索，不宣称长期偏好、职业或身份。'
                    'source_quote仅用当前根用户问题或已回答澄清的逐字片段；论文内容、模型推荐、子任务目标及历史用户话不能冒充当前行为。'
                    '同义已有主题沿用known_interests的topic_key，合并须填真实merge_memory_id和expected_version；'
                    '已撤销主题及同义改名均不可重新提出。explicit始终优先，不从兴趣推导保存权限。'
                    'reason解释行为与主题关系，content写研究线索；候选可以为空，不能为了填面板制造兴趣。')
        if 'remember_preference' in accessible:
            rules += '\n明确长期偏好可用remember_preference保存；推断只能kind=inferred，单次要求不变为长期偏好。'
        if 'forget_preference' in accessible:
            rules += '\n撤销偏好用forget_preference；无成功观察不声称已完成记忆操作。'
        if 'read_memory' in accessible:
            rules += '\n历史可read_memory按来源分页读回，档案仍是历史，不能扩大原文权限。'
        if 'set_research_plan' in accessible:
            rules += '\n纯记忆管理使用task_kind=memory和min_papers=0；新论文研究使用原研究计划。'
        rules += '\n阶段成果中的source_refs指向同一成果的source_catalog，保留原文版本/页码/hash；findings_in/context_path指向当前状态中同一版完整候选。它们不是核验通过标记，旧稿与额外发现仍独立保留。'
        # projected可能恢复了旧组；本次新档案的message范围必须对应实际待压缩原件。
        archive_messages = list(projected)
        full_current = context['current_task']
        workspace_context = json.loads(dump(context))
        raw_header = {'role': 'system', 'content': PREFIX + '\n' + rules + '\n' + dump(context)}
        before = estimate({'messages': archive_messages[:1] + [raw_header] + archive_messages[1:], 'tools': tools})
        # 完整真值留档，仅对待发副本做去重；不依赖旧组数量，最新单组也可处理。
        projected = project_finish_candidates(projected)
        projected, context['current_task'], handoff = project_research_handoffs(projected, full_current)
        review_projection = bool(full_current.get('research_reviews')) and before > available * self._limit('context_compress_ratio')
        review_metrics = {'review_projected_calls': [], 'review_compacted_entries': []}
        if review_projection:
            projected, context['current_task'], review_metrics = project_review_handoffs(projected, context['current_task'])
        projected, context, answer_metrics = project_review_answer(projected, context)
        projected.insert(1,{'role':'system','content':PREFIX+'\n'+rules+'\n'+dump(context)})
        available=self._available()
        trigger=available*self._limit('context_compress_ratio')
        compressed=[]
        tool_results = {}
        for message in archive_messages:
            if message.get('role') == 'tool':
                try:
                    tool_results[message['tool_call_id']] = json.loads(message['content'])
                except (ValueError, TypeError):
                    continue
        # 先固定全部原件，再计算任何offset；已解码工具值支持精确定位嵌套详情。
        # 这些原件只进档案，不会作为另一份重复正文进入模型请求。
        archive_payload = {'messages': archive_messages, 'current_task': full_current,
                           'workspace_context': workspace_context, 'tool_results': tool_results,
                           'history_message_ids':[m['message_id'] for m in context.get('messages',[])]}
        # 身份绑定实际持久原件；布局变更不能命中旧档案并沿用错误的字符位置。
        archive_id = digest([state['task_id'], archive_payload, tools])
        archive_text = dump(archive_payload)
        if review_projection and 'read_memory' in accessible:
            from research_loop.domain.context_workspace import archive_location
            for index, review in enumerate(context['current_task'].get('research_reviews', [])):
                # 原件先固定，再用实际JSON路径定位；中文和重复文字不靠find猜偏移。
                path = ['current_task', 'research_reviews', index]
                offset, chars = archive_location(archive_payload, path)
                review['original_result_readback'] = {
                    'readback': {'source_type': 'context_archive', 'source_id': archive_id,
                                 'offset': offset, 'limit': min(6000, chars)},
                    'value_chars': chars, 'original_path': path,
                    'notice': '完整原始核验结果可分页读回；finding_in/findings_in引用本次当前状态中的原判断。'}
            projected[1] = {'role': 'system', 'content': PREFIX + '\n' + rules + '\n' + dump(context)}
        if handoff['handoff_projected_calls'] and 'read_memory' in accessible:
            context['current_task']['handoff_readback'] = {
                'source_type': 'context_archive', 'source_id': archive_id,
                'offset': archive_text.index(dump(full_current)), 'limit': 6000,
                'notice': '完整原始交接按offset分页读回；不重新执行委派或继承文件权限。'}
            projected[1] = {'role': 'system', 'content': PREFIX + '\n' + rules + '\n' + dump(context)}
        # 压缩完整的旧assistant/tool组；最后一组观察和所有当前用户/系统消息保留。
        groups=[]
        i=0
        while i<len(projected):
            if projected[i].get('role')=='assistant' and projected[i].get('tool_calls'):
                end=i+1
                while end<len(projected) and projected[end].get('role')=='tool':
                    end+=1
                groups.append((i,end))
                i=end
            else:
                i+=1
        # 历史往返仅缩短tool内容；真实assistant、推理字段和调用结构完整保留。
        shift=0
        for original_start,original_end in groups[:-1]:
            if estimate({'messages':projected,'tools':tools})<=trigger:
                break
            start,end=original_start-shift,original_end-shift
            group=projected[start:end]
            calls=group[0]['tool_calls']
            raw_group = archive_messages[original_start-1:original_end-1]
            if {c['id'] for c in calls}!={m.get('tool_call_id') for m in group[1:]}:
                continue
            # 无档案读回时不把来源替换为摘要；下方可做带位置的原句投影。
            # 连必要状态也无法容纳时才有限退出，不能制造不可读回的来源结论。
            if 'read_memory' not in accessible and contains_source_exchange(group):
                continue
            summary={'archived_tool_exchange':True,'source_type':'context_archive','source_id':archive_id,
                     'original_message_range':[original_start-1,original_end-2],
                     'tools':[{'id':c['id'],'name':c['function']['name']} for c in calls],
                     'observations':[excerpt(str(m.get('content','')),state['prompt'],500) for m in group[1:]],
                     'notice':self._archive_notice(accessible)}
            if 'read_memory' in accessible:
                # offset按实际持久JSON的字符数计算，直接到该往返，不让模型猜页码或从头翻档案。
                offset = archive_text.find(dump(raw_group)[1:-1])
                if offset < 0:
                    raise ValueError('CONTEXT_ARCHIVE_RANGE: 原始往返定位失败')
                summary['readback'] = {'source_type':'context_archive', 'source_id':archive_id, 'offset':offset, 'limit':6000}
            replacement = compact_tool_exchange(group, summary)
            if estimate(replacement)>=estimate(group):
                continue
            projected[start:end]=replacement
            shift+=end-start-len(replacement)
            compressed.append({'start':original_start,'end':original_end,'tool_call_ids':[c['id'] for c in calls]})
        # 最新观察通常完整保留。唯有分页记忆读回导致超限时缩短返回页，并前移续读
        # 游标；原始观察仍在档案中。本步骤不裁剪论文或当前要求，原文投影另行处理。
        memory_pages=[]
        call_names={c['id']:c['function']['name'] for m in projected for c in m.get('tool_calls',[])}
        while 'read_memory' in accessible and estimate({'messages':projected,'tools':tools})>available:
            candidates=[]
            for index,message in enumerate(projected):
                if message.get('role')!='tool' or call_names.get(message.get('tool_call_id'))!='read_memory':
                    continue
                try:
                    observation=json.loads(message.get('content',''))
                except (ValueError,TypeError):
                    continue
                data=observation.get('data') if isinstance(observation,dict) else None
                if not isinstance(data,dict) or not isinstance(data.get('content'),str):
                    continue
                if observation.get('ok') and len(data['content'])>512 and isinstance(data.get('offset'),int) and isinstance(data.get('total_chars'),int):
                    candidates.append((estimate(data['content']),index,observation))
            if not candidates:
                break
            _,index,observation=max(candidates,key=lambda candidate:candidate[0])
            data=observation['data']
            before_chars=len(data['content'])
            data['content']=data['content'][:max(512,before_chars//2)]
            data['next_offset']=data['offset']+len(data['content'])
            data['context_excerpted']=True
            data['notice']='本次上下文仅保留该记忆页前缀；按原source_type/source_id及next_offset继续read_memory，不代表原记录已读完。'
            projected[index]={**projected[index],'content':dump(observation)}
            memory_pages.append({'tool_call_id':projected[index]['tool_call_id'],'before_chars':before_chars,'after_chars':len(data['content']),'next_offset':data['next_offset']})
        working = {'archived_groups': 0}
        source_archived = any(contains_source_exchange(archive_messages[item['start']-1:item['end']-1])
                              for item in compressed)
        if estimate({'messages': projected, 'tools': tools}) > available or source_archived:
            # 从原始往返重组，避免沿用旧摘要中冗长的reasoning/重复元数据。
            # 即使档案摘录已使容量达标，也必须保留同一套互补原文工作集。
            header = {'role': 'system', 'content': PREFIX + '\n' + rules + '\n' + dump(context)}
            compact_messages, _, _ = project_research_handoffs(archive_messages, full_current)
            if review_projection:
                # 回退只重组工作集，不能把最新核验的完整工具副本重新展开。
                compact_messages, _, _ = project_review_handoffs(compact_messages, full_current)
            # 回退只重建封装；已截短的记忆页属于本次请求投影，不能从档案原件恢复。
            # 同时核对来源/版本/偏移，避免重复tool_call_id误替换另一个读取结果。
            shortened_ids = {page['tool_call_id'] for page in memory_pages}
            page_views = {m['tool_call_id']: m for m in projected
                          if m.get('role') == 'tool' and m.get('tool_call_id') in shortened_ids}
            for index, message in enumerate(compact_messages):
                view = page_views.get(message.get('tool_call_id')) if message.get('role') == 'tool' else None
                if view is None:
                    continue
                source = json.loads(message['content']).get('data', {})
                page = json.loads(view['content'])['data']
                identity = ('source_type', 'source_id', 'content_hash', 'offset', 'total_chars')
                if (all(source.get(key) == page.get(key) for key in identity)
                        and isinstance(source.get('content'), str) and source['content'].startswith(page['content'])):
                    compact_messages[index] = {**message, 'content': view['content']}
            compact_view, working = self._working_set(state, compact_messages, tools, max(512, available - estimate(header)))
            compact_view.insert(1, header)
            # 容量足够时保留已有档案往返与精确offset，只补上互补原文窗口；
            # 真正超限才整组卸载，避免破坏既有档案恢复和供应商消息合同。
            source_view = next((m for m in compact_view if m.get('role') == 'user'
                                and str(m.get('content', '')).startswith('{"archived_tool_groups"')), None)
            with_sources = projected[:3] + ([source_view] if source_view else []) + projected[3:]
            projected = (with_sources if source_archived and source_view
                         and estimate({'messages': with_sources, 'tools': tools}) <= available else compact_view)
        def make_header(value):
            """统一包装本请求的规则与状态，工作集容量包含该系统头。"""
            return {'role': 'system', 'content': PREFIX + '\n' + rules + '\n' + dump(value)}
        body = [m for m in projected if not (m.get('role') == 'system' and str(m.get('content', '')).startswith(PREFIX))]
        # 工作集回退可能从原件重新引入回答副本，容量计算前再次无损折叠。
        body, context, answer_metrics = project_review_answer(body, context)
        body, context, workspace = project_workspace(body, context, tools, archive_payload,
            archive_id, state['prompt'], available, make_header)
        projected = body[:1] + [make_header(context)] + body[1:]
        projected, source_calls = self._project_sources(state, projected, tools, available) if estimate({'messages':projected,'tools':tools}) > available else (projected, [])
        after=estimate({'messages':projected,'tools':tools})
        metrics={'estimate_method':'utf8_bytes_upper_bound','before':before,'after':after,'input_capacity':available,'compressed_groups':compressed,'restored_source_groups':restored,'readback_available':'read_memory' in accessible,'active_goal_ids':[g['goal_id'] for g in (context['current_task']['research_plan'] or {}).get('goals',[]) if g.get('status', 'active') == 'active'],'memory_pages_shortened':memory_pages,'history_excerpted':sum(m['excerpted'] for m in context.get('messages',[])),'snapshot_id':archive_id}
        metrics['source_projected_calls'] = source_calls
        metrics.update(handoff)
        metrics.update(review_metrics)
        metrics.update(answer_metrics)
        metrics.update(working)
        metrics.update(workspace)
        self._capacity_metrics(metrics, projected, tools)
        with self.store.transaction() as tx:
            if not tx.one('SELECT snapshot_id FROM context_archives WHERE snapshot_id=?',(archive_id,)):
                tx.execute('INSERT INTO context_archives(snapshot_id,task_id,sequence_no,original_json,projected_json,metrics_json,created_at) VALUES(?,?,?,?,?,?,?)',(archive_id,state['task_id'],int(state.get('model_requests',0))+1,archive_text,dump(projected),dump(metrics),_now()))
        if after>available:
            raise ValueError('CONTEXT_CAPACITY: 当前要求/近期上下文超过可用容量，未发送模型；原始记录已保留，建议缩小当前任务或开新对话。')
        return projected,metrics
