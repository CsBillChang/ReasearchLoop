"""通用行动循环：模型决定下一步，程序校验并执行工具，再把观察交回模型；同时记录恢复状态。"""
from __future__ import annotations

import json
import hashlib
from copy import copy
from typing import Any

from langgraph.graph import END, START, StateGraph

from research_loop.agents.state import AgentState
from research_loop.domain.execution_policy import enforcing, handoff_reservations
from research_loop.agents.supervisor import append_tool_observation, initial_messages
from research_loop.domain.tasks import ActionRecord, Observation, SavePolicy
from research_loop.providers.deepseek import DeepSeekProvider, ModelResponse, InvalidConversation, ProviderResponseError, RequestNotSent, validate_messages
from research_loop.tools.base import ToolContext
from research_loop.tools.registry import ToolRegistry


def reflection_handoffs(actions: list[dict]) -> tuple[set[tuple], int]:
    """识别已结算的独立核验/局部版本交接，不把重放或失败算作新进展。

    Args:
        actions: 当前角色按顺序持久化的工具行动。

    Returns:
        不重复的交接身份及最后一次新交接下标；不代表事实已被支持。
    """
    seen, latest = set(), -1
    for index, action in enumerate(actions):
        observation = action.get('observation') or {}
        if action.get('status') != 'succeeded' or not observation.get('ok'):
            continue
        name, data = action['tool_name'], observation.get('data') or {}
        identity = None
        if name in {'review_research', 'review_note'} and data.get('review_id'):
            identity = ('review', data['review_id'], data.get('content_hash'))
        elif name in {'revise_research', 'revise_note'} and (data.get('revision') or {}).get('key'):
            identity = ('revision', data['revision']['key'], data['revision']['after_hash'])
        elif name == 'close_research_issue' and (data.get('research_disposition') or {}).get('key'):
            identity = ('disposition', data['research_disposition']['key'])
        if identity and identity not in seen:
            seen.add(identity)
            latest = index
    return seen, latest


def failed_review_handoffs(actions: list[dict]) -> dict[tuple, dict]:
    """提取已结算的核验失败身份；与科研进展分开，重放不产生新事件。"""
    failures = {}
    for action in actions:
        observation = action.get('observation') or {}
        data = observation.get('data') or {}
        if (action.get('tool_name') in {'review_research', 'review_note'}
                and action.get('status') == 'failed' and not observation.get('ok')
                and data.get('status') == 'execution_failed' and data.get('review_id')):
            identity = (data['review_id'], data.get('target_id') or data.get('draft_id'), data.get('content_hash'))
            failures.setdefault(identity, {'identity': list(identity), 'message': observation.get('message', '')})
    return failures


def handoff_schemas(schemas: list[dict], role: str) -> list[dict]:
    """保留现有交接工具和Supervisor档案读回，不增加注册表未授权的工具。"""
    allowed = {'record_research_progress', 'finish_research'} if role == 'researcher' else {
        'record_research_progress', 'finish_task', 'report_task_gap', 'create_note', 'save_note',
        'review_research', 'review_note', 'revise_research', 'revise_note', 'close_research_issue', 'read_memory'}
    return [schema for schema in schemas if schema['function']['name'] in allowed]


def handoff_navigation(handoff: dict, actions: list[dict]) -> tuple[dict, bool]:
    """记录交接中实际读回的新字符区间，不把导航当作研究进展。

    Args:
        handoff: 当前交接边界和之前已读的来源版本/区间。
        actions: 本批次实际结算的行动。

    Returns:
        导航状态副本及是否覆盖了此前未读的区间。同类来源绑定首个版本，
        防止每轮生成新档案ID后把相同阅读无限计作进展。
    """
    navigation = json.loads(json.dumps(handoff.get('navigation', {})))
    changed = False
    for action in actions:
        observation = action.get('observation') or {}
        data = observation.get('data') or {}
        kind = data.get('source_type')
        if (action['tool_name'] != 'read_memory' or action.get('status') != 'succeeded'
                or not observation.get('ok') or kind not in {'context_archive', 'message'}):
            continue
        start, text = data.get('offset'), data.get('content')
        if (not isinstance(start, int) or start < 0 or not isinstance(text, str) or not text
                or not isinstance(data.get('total_chars'), int) or start + len(text) > data['total_chars']):
            continue
        identity = [data.get('source_id'), data.get('content_hash')]
        if not all(identity):
            continue
        record = navigation.setdefault(kind, {'identity': identity, 'ranges': []})
        if record['identity'] != identity:
            continue
        end = start + len(text)
        # 区间已合并，因此完全覆盖的新请求只需检查是否包含在任一旧区间内。
        if any(left <= start and end <= right for left, right in record['ranges']):
            continue
        merged = []
        for left, right in sorted(record['ranges'] + [[start, end]]):
            if merged and left <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        record['ranges'] = merged
        changed = True
    return navigation, changed


def handoff_progress_changed(handoff: dict, current: dict) -> bool:
    """按快照实际保存的指纹合同比较，旧格式不能与新格式误作成果变化。

    Args:
        handoff: 已持久化交接状态；旧快照仅有progress_fingerprint。
        current: 当前已提交进展，不从模型内部推理猜测发现。

    Returns:
        当前记录相对同格式基线是否变化。新快照只按实际发现变化继续；
        旧快照保留原比较合同，不伪造其当时没有保存的发现基线。
    """
    from research_loop.services.research_progress import continuation_fingerprint, progress_fingerprint
    if 'continuation_fingerprint' in handoff:
        return continuation_fingerprint(current) != handoff['continuation_fingerprint']
    return progress_fingerprint(current) != handoff.get('progress_fingerprint')


def finish_input_hash(arguments: dict) -> str:
    """按完整原始参数绑定失败交付，不删字段或进行语义归一化。"""
    return hashlib.sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def qualification_handoff_active(handoff: dict, coverage: Any, task_id: str) -> bool:
    """仅由同一目标的资格依赖判断阻断是否解除，旧交接保持原合同。"""
    blocker = handoff.get('coverage_blocker') or {}
    if not blocker or not coverage:
        return False
    return coverage.qualification_fingerprint(task_id, blocker['goal_id']) == blocker.get('state_fingerprint')


def qualification_handoff(actions: list[dict], coverage: Any = None, task_id: str | None = None,
                          preferred: dict | None = None) -> dict | None:
    """从明确资格失败重建交接；正文仅留原参数或合法对象，不复制到交接。"""
    candidates, seen = [], set()
    for action in reversed(actions):
        observation = action.get('observation') or {}
        data = observation.get('data') or {}
        blocker = data.get('coverage_blocker') or {}
        if (action.get('tool_name') == 'finish_task' and action.get('status') == 'failed'
                and not observation.get('ok') and blocker.get('state_fingerprint')
                and blocker.get('goal_id')):
            if blocker['goal_id'] in seen:
                continue
            seen.add(blocker['goal_id'])
            candidate = data.get('finish_candidate') or {}
            identity = {'task_id': action['task_id'], 'action_id': action['action_id'],
                        'input_hash': finish_input_hash(action['arguments'])}
            # 业务服务先验证正文/对象权限；图只接受精确行动绑定，不能从args补造候选。
            bound = all(candidate.get(key) == value for key, value in identity.items())
            handoff = {'phase': 'qualification_blocked', 'trigger_action_id': action['action_id'],
                       'coverage_blocker': blocker, 'finish_candidate_ref': identity if bound else None}
            if coverage is None or qualification_handoff_active(handoff, coverage, task_id):
                candidates.append(handoff)
    # 后来的已解除失败不能掩盖其他仍有效的旧目标阻断。
    preferred_goal = ((preferred or {}).get('coverage_blocker') or {}).get('goal_id')
    return next((item for item in candidates if item['coverage_blocker']['goal_id'] == preferred_goal),
                candidates[0] if candidates else None)


def qualification_material_navigation(handoff: dict, coverage: Any, task_id: str) -> tuple[dict, bool]:
    """合并实际授权材料范围；新范围允许继续，但不解除资格阻断。"""
    navigation = json.loads(json.dumps(handoff.get('material_navigation', {})))
    initialized = 'material_navigation' in handoff
    changed = False
    owners = [task_id, *[child['subtask_id'] for child in coverage.db.list_subtasks(task_id)]]
    for owner in owners:
        for source in coverage.source_coverage(owner):
            identity = json.dumps([source.get(key) for key in
                ('document_id', 'source_sha256', 'page_number', 'text_sha256')], ensure_ascii=False)
            start = source.get('char_start') or 0
            end = source.get('char_end')
            ranges = navigation.setdefault(identity, [])
            if any(left <= start and (right is None or (end is not None and end <= right)) for left, right in ranges):
                continue
            changed = True
            merged = []
            for left, right in sorted(ranges + [[start, end]], key=lambda item: item[0]):
                if merged and (merged[-1][1] is None or left <= merged[-1][1]):
                    merged[-1][1] = None if merged[-1][1] is None or right is None else max(merged[-1][1], right)
                else:
                    merged.append([left, right])
            navigation[identity] = merged
    return navigation, initialized and changed


def matrix_thinking(state: AgentState) -> dict[str, str] | None:
    """仅为已取证的矩阵整理与交付选择 论文 相同的非思考模式。

    Args:
        state: 当前角色、已持久计划和实际工具返回的证据身份。

    Returns:
        显式非思考参数，或 None（沿用原供应商设置）。不从论文正文或
        任意消息关键词推断模式；旧计划缺省关闭，子角色不受影响。
    """
    plan = state.get("research_plan") or {}
    if (state.get("role", "supervisor") == "supervisor"
            and plan.get("evidence_matrix_requested") is True and state.get("evidence_ids")):
        return {"type": "disabled"}
    return None


def build_graph(provider: Any, registry: ToolRegistry, services: dict[str, Any],
                db: Any = None, max_model_requests: int = 48,
                max_tool_actions: int = 80, reserve_model_requests: int = 10,
                checkpoint_store: Any = None, deep_reading: bool = False):
    """构建各角色共用的决策—工具—观察循环。

    Args:
        provider: 模型请求适配器，可注入测试替身。
        registry: 当前角色允许使用的工具注册表。
        services: 工具实际依赖的业务服务。
        db: 可选的业务持久层，用于收据、额度和租约。
        max_model_requests: 当前图的请求上限；状态内冻结值优先。
        max_tool_actions: 工具行动上限；状态内冻结值优先。
        reserve_model_requests: 初始状态配置项，根共享额度由数据库控制。
        checkpoint_store: 可选的持久状态存储。

    Returns:
        已编译的 LangGraph，可输入新状态或恢复状态执行。
    """
    from research_loop.services.research_progress import ResearchProgressService, evidence_progress, readback_handoff, continuation_fingerprint
    progress = ResearchProgressService(db) if db else None

    def record(state: AgentState, update: dict[str, Any]) -> dict[str, Any]:
        """合并状态前验证 Worker 租约；失去所有权时拒绝写入，避免旧进程覆盖新执行结果。"""
        if db and state.get("worker_id") and not db.is_owner(state["task_id"], state["worker_id"], state.get("lease_token")):
            raise RuntimeError("worker lease lost; graph write rejected")
        if progress and state.get("role", "supervisor") in {"researcher", "supervisor"}:
            update["research_progress"] = progress.snapshot(state["task_id"])
        if checkpoint_store:
            checkpoint_store.save(state["task_id"], {**state, **update})
        if state.get("context_metrics") and "context_metrics" not in update:
            update["context_metrics"] = state["context_metrics"]
        return update

    def cancelled(state: AgentState) -> bool:
        if state.get("cancelled"):
            return True
        task_id = state.get("task_id")
        row = db.get_task(task_id) if db and task_id else None
        return bool(row and row.get("cancel_requested"))

    def execution_unavailable(state: AgentState) -> bool:
        """每次实际工具发送前检查取消及角色/根任务租约，防止旧Worker继续付费。"""
        if cancelled(state):
            return True
        if not db or not state.get('worker_id'):
            return False
        root = state.get('root_task_id') or state['task_id']
        return any(not db.is_owner(task, state['worker_id'], state.get('lease_token'))
                   for task in {state['task_id'], root})

    def truncated_response(state: AgentState, response: ModelResponse, request: dict,
                           conversation_messages: list[dict], request_number: int) -> dict[str, Any]:
        """保留截断正文并终止，供新响应与旧待执行checkpoint共用。

        Args:
            state: 当前角色已有成果和待执行状态。
            response: 已取得且落收据的长度截断响应。
            request: 产生该响应的原请求，不能用新配置替换其上限。
            conversation_messages: 去除本批待执行工具消息前的对话。
            request_number: 已发生的请求序号，不增加新的发送。

        Returns:
            明确截断原因和缺口的终止状态；不包含可执行工具或内部推理。
        """
        # 成功收据只说明远端响应与用量已取得。截断不能触发半截工具、
        # 自动补发或正常完成；推理仅留原始收据，不转成公开研究成果。
        from research_loop.domain.context import textual_delivery
        usage = response.usage or {}
        completion = usage.get("completion_tokens")
        cap = request.get("max_tokens")
        output_limited = (type(completion) is int and type(cap) is int
                          and cap > 0 and completion >= cap)
        prefix = "OUTPUT_LIMIT_" if output_limited else "MODEL_LENGTH_"
        cause = ("模型达到单请求输出上限" if output_limited else
                 "模型响应因长度限制截断（未确认是否耗尽单请求输出额度）")
        content = response.content or ""
        candidate = textual_delivery(content)
        internal_call = content.lstrip().startswith((
            "<｜｜DSML｜｜ calls>", "<｜DSML｜function_calls>",
            "<finish_task>", "<finish_research>", "<finish_review>"))
        if response.tool_calls or internal_call or candidate is not None:
            code = prefix + "TOOL_CALL"
            notice = cause + "，工具调用或交付参数可能不完整；该截断响应的待执行工具已阻止，未自动重发。"
            # 合法独立正文与文本化交付候选可以保留；原始参数不公开。
            answer = candidate if candidate is not None else ("" if internal_call else content)
        elif content.strip():
            code = prefix + "PARTIAL_CONTENT"
            notice = cause + "，正文已截断；保留已有正文，未完成部分未自动续写。"
            answer = content
        else:
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            reasoning_only = (type(completion) is int and type(reasoning) is int
                              and completion > 0 and reasoning == completion)
            code = prefix + ("REASONING_ONLY" if reasoning_only else "NO_CONTENT")
            notice = (cause + "，本次输出全部计入内部推理，未生成正文或工具调用；已有成果保留，未自动重发。"
                      if reasoning_only else
                      cause + "，未生成可交付正文；已有成果保留，未自动重发。")
            answer = ""
        return record(state, {
            "model_requests": request_number, "finished": True, "pending_tool_calls": [],
            "messages": conversation_messages + [{"role": "assistant", "content": answer}],
            # 无新正文时保持此前正文；阶段/草稿成果继续由Worker原保留逻辑读回。
            "final_answer": answer or state.get("final_answer") or "",
            "final_reason": code + ": " + notice,
            "gaps": list(state.get("gaps") or []) + [notice],
        })

    def decide(state: AgentState) -> dict[str, Any]:
        """先检查取消、额度和未结算请求，再复用收据或发请求；输出待执行工具或结束状态。"""
        if state.get("finished") or state.get("pending_tool_calls"):
            return record(state, {})
        if cancelled(state):
            return record(state, {"cancelled": True, "finished": True, "final_reason": "cancelled"})
        request_number = int(state.get("model_requests", 0)) + 1
        enforce = enforcing(state)
        local_enforce = enforce and (not state.get("execution_policy") or db is None)
        if local_enforce and request_number > int(state.get("max_model_requests", max_model_requests)):
            return record(state, {"finished": True, "final_reason": "model request budget exhausted"})
        closing = local_enforce and request_number >= int(state.get("max_model_requests", max_model_requests))
        settling = False
        messages = list(state.get("messages") or [])
        if not messages:
            messages = initial_messages(state["prompt"], state.get("save_policy", SavePolicy.deny.value),
                                        state.get("attachments") or {})
        # checkpoint保留完整往返；仅供应商请求使用投影，避免下一轮对摘录再次摘录。
        conversation_messages = list(messages)
        try:
            validate_messages(messages)
        except InvalidConversation as exc:
            return record(state, {"finished": True, "final_reason": "invalid model conversation: " + str(exc),
                                  "gaps": list(state.get("gaps") or []) + ["对话工具回复不完整；未发送模型请求"]})
        role = state.get("role", "supervisor")
        local_tools_left = int(state.get('max_tool_actions', max_tool_actions)) - int(state.get('tool_actions', 0))
        closing = closing or (local_enforce and (local_tools_left <= 1 or (state.get('budget_tail', False) and role != 'supervisor')))
        settling = local_enforce and role == 'supervisor' and (local_tools_left <= 1 or state.get('budget_tail', False))
        if db and enforce:
            root = state.get("root_task_id", state["task_id"])
            budget = db.budget_summary(root)
            if budget:
                active = db.active_role_count(root)
                if role != 'supervisor':
                    active = max(1, active)
                parent_delivery = db.delivery_tool_reserve(root)
                usable_limit = budget["model_limit"] - (budget["reserve_tail"] if role == "researcher" else 0)
                # 保护当前实际运行子角色的交接；队列长度不等于并发资源需求。
                closing = closing or budget["model_used"] >= usable_limit - max(1, active if role != 'supervisor' else 1)
                settling = settling or (role == 'supervisor' and budget['model_limit'] - budget['model_used'] <= budget['reserve_tail'])
                tool_remaining = budget['tool_limit'] - budget['tool_used']
                closing = closing or tool_remaining <= (1 if role == 'supervisor' else
                    handoff_reservations(role, False, active, parent_delivery) + 1)
                settling = settling or (role == 'supervisor' and tool_remaining <= active + parent_delivery + 1)
        if db and db.has_unknown_family_call(state.get("root_task_id", state["task_id"])):
            return record(state, {"finished": True, "final_reason": "model response unknown: task family is blocked",
                                  "gaps": list(state.get("gaps") or []) + ["任务链有未结算请求，未自动继续调用"]})
        schemas = registry.schemas(role)
        # 进展路由不是累计额度：同材料读回一次后仍无阶段成果，转到已有交接工具。
        role_actions = db.list_actions(state['task_id']) if db else state.get('actions', [])
        # 新核验/局部版本已利用上轮材料，不应被更早的重复取证观察永久锁住。
        # 相同review_id重放不移动边界；后续再次重复取证仍触发原交接检查。
        _, last_review = reflection_handoffs(role_actions) if role == 'supervisor' else (set(), -1)
        handoff = readback_handoff(role_actions[last_review + 1:]) if role in {'supervisor','researcher'} else None
        handoff_state = None
        previous_handoff = state.get('research_handoff') or {}
        qualification = qualification_handoff(role_actions, services.get('coverage'), state['task_id'], previous_handoff)
        if (qualification and previous_handoff.get('phase') != 'review_failed_closeout'
                and qualification_handoff_active(qualification, services.get('coverage'), state['task_id'])):
            if (previous_handoff.get('coverage_blocker') or {}).get('state_fingerprint') == qualification['coverage_blocker']['state_fingerprint']:
                qualification = {**previous_handoff, **qualification,
                    'trigger_action_id': previous_handoff['trigger_action_id']}
            handoff_state = qualification
            # 保留修订策略、真正新材料/范围及不同目标的工具；后续依赖检查不把
            # 新增objective、重复读取或无关发现当作旧资格问题已解决。
            messages = messages + [{'role': 'system', 'content':
                '当前交付被资格条件阻断：查看coverage_blocker和goal_achievements，复用已有成果。'
                '修正自拟策略、补实际缺失信息，或finish_task将相关目标标unresolved交付。'
                '改写objective或重复取证不解除阻断；新目标、新版本及未处理范围仍可继续。'
                '资格诊断：' + json.dumps(qualification['coverage_blocker'], ensure_ascii=False)}]
            handoff = None
        elif previous_handoff.get('coverage_blocker') and previous_handoff.get('phase') != 'review_failed_closeout':
            previous_handoff = {}
        failure_closeout = previous_handoff.get('phase') == 'review_failed_closeout'
        if not handoff_state and previous_handoff.get('phase') in {'requested', 'readback', 'review_failed_closeout'}:
            # 失败工具不是新取证，不能抹掉已进入的交接边界、重新开放禁用工具。
            # 合法核验/局部版本或阶段成果更新会在执行节点标记progress_recorded。
            handoff = previous_handoff if failure_closeout else (handoff or previous_handoff)
        if handoff and not closing and not settling:
            schemas = handoff_schemas(schemas, role)
            handoff_state = {**handoff, 'phase':'requested',
                'continuation_fingerprint':continuation_fingerprint(progress.snapshot(state['task_id']) if progress else {})}
            if failure_closeout:
                # 失败观察已持久化；保留档案读回以支持上下文外置，但不再启动核验/取证。
                schemas = [t for t in schemas if t['function']['name'] in {'finish_task', 'read_memory'}]
                handoff_state.update(phase='review_failed_closeout', failed_review=previous_handoff['failed_review'])
                messages = messages + [{'role': 'system', 'content':
                    '独立核验执行失败，未取得语义判断。用finish_task交付当前绑定正文，'
                    '在processing_summary/goals[].reason说明实际失败与未完成项，不称原文缺证或核验通过。'
                    '只可按需读回授权档案/回答，不再核验或取证。失败观察：'
                    + previous_handoff['failed_review']['message']}]
            if previous_handoff.get('trigger_action_id') == handoff.get('trigger_action_id'):
                handoff_state['navigation'] = previous_handoff.get('navigation', {})
                if previous_handoff.get('tool_error'):
                    handoff_state['tool_error'] = previous_handoff['tool_error']
            if not failure_closeout:
                progress_hint = (
                    '依据已有证据形成或修订发现；仍需继续研究时用record_research_progress，已可交接则用本次结束工具。只改缺口/next_question不会重新开放取证；'
                    if any(t['function']['name'] == 'record_research_progress' for t in schemas) else
                    '用本次核验、修订或结束工具交接已有成果，说明具体缺口；')
                messages = messages + [{'role':'system','content':
                    '当前进入研究进展交接：相同材料已再次读回，但没有新的阶段发现。'
                    '依据当前互补原文窗口，用本角色实际提供的结束工具交付；不足则标partial/unresolved并说明具体缺口。'
                    + progress_hint +
                    '不得捏造发现或以重写相同记录绕过交接。本步不再重复取证，不代表事实核验通过。'
                    + ('已有回答/发现确需核验时，用本次提供的review_research或review_note交给Reviewer；'
                       '已有判断则局部修订、披露或交付，不必自行重审。'
                       '可用read_memory时按真实身份/游标读回当前档案或本会话回答；'
                       '保持同一来源版本并向未读区间推进，随后核验或交付。导航不是新研究成果，重复区间应结束。'
                       if role == 'supervisor' else '')}]
        terminal_names = {"finish_task", "finish_research", "finish_review"}
        terminal_tools = [t for t in schemas if t["function"]["name"] in terminal_names]
        if settling and not closing:
            schemas = [t for t in schemas if t['function']['name'] in
                       {'create_note', 'save_note', 'finish_task', 'report_task_gap'}]
            messages = messages + [{'role': 'system', 'content':
                '整条父子任务链已进入收尾预留。用已有原文与子任务成果整理答案，按当前权限创建/保存并结束。'
                '不再检索、补读、委派或修订；如实说明未完成部分。'}]
        terminal_only = closing and bool(terminal_tools)
        force_text = closing and not terminal_tools
        if closing:
            # 收尾仍使用现有结构化交付接口；删掉工具却要求finish会使调用标记落入正文。
            instruction = ("当前是本角色最后一次可用请求。用提供的结束工具交付已有成果；"
                           "未完成目标标unresolved/partial并说明原因；不新增检索、委派、修订或保存。")
            if force_text:
                instruction = "当前是最后一次可用请求。直接用自然语言交付已有成果和未完成原因，不输出或模拟工具调用。"
            messages = messages + [{"role": "system", "content": instruction}]
        request = {"messages": messages, "tools": terminal_tools if terminal_only else ([] if force_text else schemas),
                   "force_text": force_text, "terminal_only": terminal_only}
        if handoff_state:
            request['research_handoff'] = handoff_state
        if settling:
            request['settling'] = True
        from research_loop.storage.token_budget import TokenBudgetStore
        tokens = TokenBudgetStore(db) if db else None
        root = state.get("root_task_id", state["task_id"])
        receipt = db.get_model_call(state["task_id"], request_number) if db else None
        if receipt and receipt.get("request_json"):
            # 已持久请求始终按当时的messages恢复，偏好变化不能改写历史成功收据。
            request = json.loads(receipt["request_json"])
            messages = request["messages"]
            force_text = request.get("force_text", force_text)
            terminal_only = request.get("terminal_only", False)
            handoff_state = request.get('research_handoff')
        elif services.get("context") and role in {"supervisor", "researcher", "reviewer"}:
            try:
                context = services["context"]
                from research_loop.config import Settings
                if (tokens and not tokens.counting_policy(root) and tokens.summary(root)
                        and isinstance(context.settings, Settings)
                        and 'context_output_reserve' not in context.settings.model_fields_set):
                    # 更老的根没有冻结计量策略，不能因新默认改变旧8192合同；
                    # 显式当前值仍优先，只复制上下文实例，避免污染其他新任务。
                    context = copy(context)
                    context.settings = context.settings.model_copy(update={'context_output_reserve': 8192})
                prepare = context.prepare if role == "supervisor" else context.prepare_role
                messages, metrics = prepare(state,messages,request["tools"])
                state = {**state,"context_metrics":metrics}
                validate_messages(messages)
                request = {**request,"messages":messages,"max_tokens":metrics["output_reserve_tokens"]}
            except (ValueError, InvalidConversation) as exc:
                return record(state,{"finished":True,"final_reason":str(exc),
                    "gaps":list(state.get("gaps") or [])+[str(exc)]})
        # 只给新策略任务明确输出上限；旧收据和历史调用参数原样恢复。
        if tokens and tokens.summary(root) and not receipt:
            policy = tokens.counting_policy(root)
            request.setdefault("max_tokens", policy["context_output_reserve"] if policy else 8192)
        # 请求先持久再发送；恢复完全沿用旧参数，不给历史收据补写当前策略。
        if not receipt and (thinking := matrix_thinking(state)) is not None:
            request["thinking"] = thinking
        if tokens and receipt:
            tokens.settle(state["task_id"], request_number)
        if receipt and receipt.get("status") == "succeeded" and receipt.get("response_json"):
            response: ModelResponse = DeepSeekProvider.parse_response(
                json.loads(receipt["response_json"]), receipt.get("request_id"),
                json.loads(receipt.get("usage_json") or "{}"), receipt.get("finish_reason"))
        else:
            if receipt and receipt.get("status") == "not_sent":
                return record(state, {"finished": True, "final_reason": "model request not sent: existing local rejection; no automatic retry"})
            if receipt and receipt.get("status") in {"rejected", "response_error"}:
                return record(state, {"finished": True, "final_reason": "model request rejected: existing receipt; no resend"})
            if receipt and (receipt.get("status") == "unknown" or
                            (receipt.get("status") == "started" and receipt.get("sent_state") != "not_sent")):
                return record(state, {"model_requests": request_number, "finished": True,
                                      "final_reason": "model response unknown: recovery is blocked",
                                      "gaps": list(state.get("gaps") or []) + ["模型请求结果未知，未自动重发"]})
            if db:
                # 已冻结但尚未发送的请求同样不能在切换模型后悄悄改用另一计量合同。
                counting_policy = tokens.counting_policy(root)
                if (counting_policy and services.get("context")
                        and counting_policy.get("model") != services["context"].settings.deepseek_model):
                    reason = "TOKEN_POLICY_MODEL_CHANGED: 当前模型与根任务冻结模型不同；未发送新请求"
                    return record(state, {"finished": True, "final_reason": reason,
                                          "gaps": list(state.get("gaps") or []) + [reason]})
                try:
                    tokens.reserve(root, state["task_id"], request_number, request, role)
                except ValueError as exc:
                    return record(state, {"finished": True, "final_reason": str(exc),
                                          "gaps": list(state.get("gaps") or []) + [str(exc)]})
                if not receipt and not db.reserve_model_slot(root, role):
                    db.save_model_call(state["task_id"], request_number, "not_sent", request,
                                       response={"error": "shared model request budget exhausted"}, sent_state="not_sent")
                    return record(state, {"finished": True, "final_reason": "shared model request budget exhausted",
                                          "gaps": list(state.get("gaps") or []) + ["共享模型额度已用尽，未新增请求"]})
                db.begin_model_call(state["task_id"], request_number, request)
                db.mark_model_call_sent(state["task_id"], request_number)
            try:
                if callable(getattr(provider, "send", None)):
                    kwargs = {key: request[key] for key in ("max_tokens", "thinking") if key in request}
                    if role == "supervisor" and services.get("stream_public", True) and services.get("context") and callable(getattr(provider,"send_stream",None)):
                        def publish_public(payload):
                            # 租约失效不能继续写流；取消后等待本次远端响应结算，不再执行工具。
                            if state.get("worker_id") and not db.is_owner(state["task_id"],state["worker_id"],state.get("lease_token")):
                                raise RuntimeError("stream lease lost")
                            db.append_event(state["task_id"],"answer_stream",{**payload,"request_number":request_number})
                        raw = provider.send_stream(messages,request["tools"],force_text=force_text,on_public=publish_public,**kwargs)
                    else:
                        raw = provider.send(messages, request["tools"], force_text=force_text, **kwargs)
                    # 先保存实际响应及用量，再解析或推进 checkpoint；解析失败也不能重发成功请求。
                    if db:
                        choice = (raw.get("choices") or [{}])[0]
                        db.save_model_call(state["task_id"], request_number, "succeeded", request, response=raw,
                                           usage=raw.get("usage"), request_id=raw.get("id"),
                                           finish_reason=choice.get("finish_reason"), sent_state="sent")
                    response = DeepSeekProvider.parse_response(raw)
                else:
                    response = provider.complete(messages, request["tools"], force_text=force_text)
            except (RequestNotSent, InvalidConversation) as exc:
                # 仅依据适配器的明确发送前异常分类；绝不从普通错误字符串猜远端是否执行。
                if db:
                    db.save_model_call(state["task_id"], request_number, "not_sent", request,
                                       response={"error": str(exc), "error_type": type(exc).__name__},
                                       sent_state="not_sent")
                return record(state, {"model_requests": request_number, "finished": True,
                                      "final_reason": "model request not sent: " + str(exc),
                                      "gaps": list(state.get("gaps") or []) + ["本地发送前检查拒绝，未产生远端请求；已有成果保留"]})
            except ProviderResponseError as exc:
                if db:
                    db.save_model_call(state["task_id"], request_number,
                                       "rejected" if 400 <= exc.status_code < 500 else "response_error", request,
                                       response=exc.as_record(), request_id=exc.request_id, sent_state="sent")
                return record(state, {"model_requests": request_number, "finished": True,
                                      "final_reason": "model request rejected: " + str(exc),
                                      "gaps": list(state.get("gaps") or []) + [f"模型接口返回 HTTP {exc.status_code}；错误已记录，未自动重发"]})
            except Exception as exc:
                prior_response = db.get_model_call(state["task_id"], request_number) if db else None
                if prior_response and prior_response.get("status") == "succeeded":
                    return record(state, {"model_requests": request_number, "finished": True,
                                          "final_reason": "model response parse failed: " + str(exc)})
                if db:
                    db.save_model_call(state["task_id"], request_number, "unknown", request,
                                       response={"error": str(exc)}, sent_state="sent")
                return record(state, {
                    "model_requests": request_number, "finished": True,
                    "final_reason": "model response unknown: " + str(exc),
                    "gaps": list(state.get("gaps") or []) + ["模型请求结果未知，未自动重发"],
                })
            if db:
                db.save_model_call(state["task_id"], request_number, "succeeded", request, response=response.raw,
                                   usage=response.usage, request_id=response.request_id,
                                   finish_reason=response.finish_reason, sent_state="sent")
        if response.finish_reason == "length":
            return truncated_response(state, response, request, conversation_messages, request_number)
        if response.tool_calls and not force_text:
            assistant_calls = [
                {"id": call.call_id, "type": "function",
                 "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
                for call in response.tool_calls
            ]
            assistant_message = {"role": "assistant", "content": response.content or "", "tool_calls": assistant_calls}
            if response.reasoning_content is not None:
                assistant_message["reasoning_content"] = response.reasoning_content
            return record(state, {
                "model_requests": request_number,
                "research_handoff": handoff_state,
                "messages": conversation_messages + [assistant_message],
                "pending_tool_calls": [
                    {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
                    for call in response.tool_calls
                ],
                "pending_tool_scope": sorted(t['function']['name'] for t in request['tools']),
            })
        content = (response.content or "").lstrip()
        from research_loop.domain.context import textual_delivery
        candidate = textual_delivery(content)
        if candidate is not None:
            # JSON文本不是tool_call，不能执行其中的保存/目标满足声明。
            notice = "模型返回了文本化内部交付参数；以下仅为未核验候选，未形成完整交付。"
            return record(state, {"model_requests": request_number, "finished": True,
                "messages": conversation_messages + [{"role":"assistant", "content":response.content or ""}],
                "final_answer": notice + "\n\n" + candidate if candidate else None,
                "final_reason": "model returned textual delivery instead of a valid tool call",
                "gaps": list(state.get("gaps") or []) + [notice]})
        if content.startswith(("<｜｜DSML｜｜ calls>", "<｜DSML｜function_calls>", "<finish_task>", "<finish_research>", "<finish_review>")):
            # 原始响应已落收据；文本化调用既不能执行，也不能伪装成面向用户的研究答案。
            return record(state, {"model_requests": request_number, "finished": True,
                                  "final_reason": "model returned textual tool invocation instead of a valid delivery",
                                  "gaps": list(state.get("gaps") or []) + ["模型把内部工具调用写成了正文，未形成有效交付；原始响应已保留，未执行其中动作"]})
        final_message = {"role": "assistant", "content": response.content or ""}
        if response.reasoning_content is not None:
            final_message["reasoning_content"] = response.reasoning_content
        return record(state, {
            "model_requests": request_number,
            "messages": conversation_messages + [final_message],
            "final_answer": response.content or "", "finished": True,
            "final_reason": "model finished" if response.content else "model returned no content",
        })

    def execute_tools(state: AgentState) -> dict[str, Any]:
        """按调用顺序执行已授权工具，复用已落库观察；汇总证据、子任务、草稿与文件路径。"""
        if cancelled(state):
            return record(state, {"cancelled": True, "finished": True, "final_reason": "cancelled before tool execution"})
        if execution_unavailable(state):
            raise RuntimeError("worker lease lost before tool execution")
        # 旧版checkpoint可能已把length响应路由成待执行工具；恢复也必须
        # 以该批成功收据为准终止，不能执行同批完整动作或自动纠正后再发。
        number = int(state.get('model_requests', 0))
        prior_call = db.get_model_call(state['task_id'], number) if db else None
        if prior_call and prior_call.get('status') == 'succeeded' and prior_call.get('response_json'):
            prior_response = DeepSeekProvider.parse_response(
                json.loads(prior_call['response_json']), usage=json.loads(prior_call.get('usage_json') or '{}'),
                finish_reason=prior_call.get('finish_reason'))
            if prior_response.finish_reason == 'length':
                original = json.loads(prior_call['request_json'])
                history = list(state.get('messages') or [])
                if history and history[-1].get('tool_calls'):
                    history = history[:-1]
                return truncated_response(state, prior_response, original, history, number)
        actions = list(state.get("actions") or [])
        prior_reflection, _ = reflection_handoffs(actions)
        prior_review_failures = failed_review_handoffs(actions)
        observations = list(state.get("observations") or [])
        messages = list(state.get("messages") or [])
        tool_count = int(state.get("tool_actions", 0))
        # 行动序号包含未执行的拒绝回执，工具额度只统计实际准入的调用。
        # 旧checkpoint未分离二者，使用原tool_actions兼容恢复。
        sequence = int(state.get('action_sequence', tool_count))
        budget_tail = state.get('budget_tail', False)
        terminal_denied = False
        draft_id = state.get("draft_id")
        saved_path = state.get("saved_path")
        evidence_ids = list(state.get("evidence_ids") or [])
        gaps = list(state.get("gaps") or [])
        source_warnings = list(state.get("source_warnings") or [])
        subtask_results = list(state.get("subtask_results") or [])
        review_result = state.get("review_result")
        research_plan = state.get("research_plan")
        delivery = state.get("delivery")
        finish_candidate_ref = state.get('finish_candidate_ref')
        needs_input = False
        structured_result = None
        blocked_repeat = False
        repeated_failure = False
        pending = list(state.get("pending_tool_calls") or [])
        # 成功响应恢复时仍受原请求范围约束，不给旧checkpoint补上新增工具。
        receipt = db.get_model_call(state['task_id'], state.get('model_requests', 0)) if db else None
        # 可见性来自产生本批工具调用的冻结请求，不能拿未压缩的持久历史冒充。
        visible_messages = None
        if receipt and receipt.get('request_json'):
            frozen_request = json.loads(receipt['request_json'])
            allowed = {t['function']['name'] for t in frozen_request.get('tools', [])}
            visible_messages = frozen_request.get('messages')
        elif 'pending_tool_scope' in state:
            allowed = set(state['pending_tool_scope'])
        else:
            schemas = registry.schemas(state.get('role', 'supervisor'))
            if state.get('research_handoff') and not state['research_handoff'].get('coverage_blocker'):
                schemas = handoff_schemas(schemas, state.get('role', 'supervisor'))
                if state['research_handoff'].get('phase') == 'review_failed_closeout':
                    schemas = [t for t in schemas if t['function']['name'] in {'finish_task', 'read_memory'}]
            allowed = {t['function']['name'] for t in schemas}
        scope_rejected = False
        recoverable_error = None
        context = ToolContext(
            task_id=state["task_id"], save_policy=SavePolicy(state.get("save_policy", SavePolicy.deny.value)),
            attachments=state.get("attachments") or {}, cancelled=lambda: execution_unavailable(state), services=dict(services),
            role=state.get("role", "supervisor"), root_task_id=state.get("root_task_id", state["task_id"]),
            parent_task_id=state.get("parent_task_id"),
        )
        for call in pending:
            sequence += 1
            terminal = call['name'] in {'finish_task', 'finish_research', 'finish_review'}
            local_tail = 0 if terminal else 1
            active = db.active_role_count(context.root_task_id) if db else 0
            if context.role != 'supervisor':
                active = max(1, active)
            parent_delivery = db.delivery_tool_reserve(context.root_task_id) if db else 1
            shared_tail = handoff_reservations(context.role, terminal, active, parent_delivery)
            prior = db.get_action(state["task_id"], sequence) if db else None
            replayable = prior and prior.get("status") in {"succeeded", "failed", "cancelled"} and prior.get("observation")
            rejection = None
            rejection_code = None
            if not prior:
                if delivery is not None or structured_result is not None:
                    rejection = '本批次已有终结结果，后续调用未执行。'
                    rejection_code = 'AFTER_TERMINAL'
                elif call['name'] not in allowed:
                    rejection = (f"工具{call['name'] or '<empty>'}不在本次请求范围；未执行。"
                                 '请选择实际可用工具：' + ', '.join(sorted(allowed)))
                    rejection_code = 'TOOL_NOT_AVAILABLE'
                elif scope_rejected and call['name'] != 'read_memory':
                    # 无法从模型同批调用推断前后依赖；仅独立只读导航可继续。
                    # 核验/修订/保存/结束等状态操作留给观察错误后的下一轮决策。
                    rejection = '本批前序调用被范围检查拒绝，依赖关系未确认；请观察错误后重新选择本操作。'
                    rejection_code = 'TOOL_DEPENDENCY_UNRESOLVED'
                elif isinstance(call['arguments'], dict) and '_invalid_json' in call['arguments']:
                    # 已知截断参数没有执行机会；在预留工具槽之前拒绝，根/角色
                    # 计数均保持真实，不用“空默认参数”消耗最后的交接槽。
                    rejection = '工具参数不是完整JSON，未执行；请提交合法参数或交付已保留成果。'
                    rejection_code = 'INVALID_TOOL_JSON'
                elif enforcing(state) and (not state.get('execution_policy') or db is None) and tool_count >= int(state.get('max_tool_actions', max_tool_actions)) - local_tail:
                    rejection = '本地工具收尾预留已到达；未执行本调用，请交付已有成果及未完成项。'
                    rejection_code = 'TOOL_TAIL_RESERVED'
                elif db and not db.reserve_tool_slot(context.root_task_id, shared_tail):
                    rejection = '共享工具收尾预留已到达；未执行本调用，请交付已有成果及未完成项。'
                    rejection_code = 'TOOL_TAIL_RESERVED'
            action = ActionRecord(task_id=state["task_id"], sequence=sequence, tool_name=call["name"],
                                  arguments=call["arguments"])
            if replayable:
                action.action_id = prior['action_id']
                result = Observation.model_validate(prior.get("observation") or {})
            elif rejection:
                result = Observation(ok=False, kind='validation_error' if rejection_code in {'TOOL_NOT_AVAILABLE', 'INVALID_TOOL_JSON'} else 'execution_error',
                                     retryable=rejection_code in {'TOOL_NOT_AVAILABLE', 'TOOL_DEPENDENCY_UNRESOLVED', 'INVALID_TOOL_JSON'}, message=rejection,
                                     data={'error_code': rejection_code,
                                           'tool_executed': False, 'budget_counted': False})
                if rejection_code == 'TOOL_NOT_AVAILABLE':
                    result.data['available_tools'] = sorted(allowed)
            else:
                rejected_same = next((a for a in actions[-1:]
                    if a["tool_name"] == call["name"] and a["arguments"] == call["arguments"]
                    and not a.get("observation", {}).get("ok")
                    and not a.get("observation", {}).get("retryable", False)), None)
                if db:
                    db.save_action(action.model_dump(mode="json"))
                if rejected_same:
                    blocked_repeat = True
                    result = Observation(ok=False, kind="execution_error", retryable=False,
                                         message="相同的不可重试操作已失败；未重复执行。请选择其他行动或结束并交付已有结果。",
                                         data={"error_code": "REPEATED_NONRETRYABLE_ACTION"})
                else:
                    context.services['finish_action_identity'] = {
                        'task_id': state['task_id'], 'action_id': action.action_id,
                        'input_hash': finish_input_hash(call['arguments'])}
                    cached = None
                    if progress and call['name'] == 'read_paper' and not context.cancelled():
                        try:
                            spec = registry.get(call['name'])
                            if context.role in spec.roles:
                                spec.args_model.model_validate(call['arguments'])
                                cached = progress.cached_read(context.task_id, call['arguments'], context.attachments,
                                    provider=getattr(services.get('papers'), 'pdf', None))
                        except (KeyError, ValueError):
                            pass
                    if state.get('research_handoff') and call['name'] == 'read_memory':
                        kind = call['arguments'].get('source_type')
                        allowed_read = kind == 'context_archive'
                        if kind == 'message' and db:
                            from research_loop.services.conversations import ConversationService
                            allowed_read = bool(ConversationService(db).answer_message(
                                context.task_id, call['arguments'].get('source_id')))
                        if not allowed_read:
                            result = Observation(ok=False, kind='validation_error', retryable=False,
                                message='交接只允许当前任务档案或当前会话此前assistant回答；不扩展为历史检索。')
                        else:
                            result = registry.execute(call['name'], call['arguments'], context)
                    else:
                        result = Observation.model_validate(cached) if cached else registry.execute(call['name'], call['arguments'], context)
            if result.data.get('error_code') == 'TOOL_NOT_AVAILABLE':
                scope_rejected = True
                blocked_repeat = blocked_repeat or any(
                    a['tool_name'] == call['name'] and a['arguments'] == call['arguments']
                    and (a.get('observation') or {}).get('data', {}).get('error_code') == 'TOOL_NOT_AVAILABLE'
                    for a in actions)
            if (not result.ok and result.retryable and recoverable_error is None
                    and (result.kind == 'validation_error' or
                         result.data.get('error_code') == 'TOOL_DEPENDENCY_UNRESOLVED')):
                recoverable_error = {'tool_name': call['name'], 'message': result.message}
            if result.data.get('budget_counted') is not False:
                tool_count += 1
            if result.data.get('error_code') == 'TOOL_TAIL_RESERVED':
                budget_tail = True
                terminal_denied = terminal_denied or terminal
            if not replayable and call['name'] in {'read_paper', 'acquire_paper_pdf', 'retrieve_evidence'}:
                increment = evidence_progress(actions, call['name'], call['arguments'],
                                              result.model_dump(mode='json'), visible_messages)
                if state.get('model_requests', 0) > 0:
                    increment['request_sequence'] = state['model_requests']
                result.data['research_increment'] = increment
            # 相同参数已收到一次失败反馈后仍失败，没有新的研究进展；监测模式
            # 也必须终止这种无进展循环。更正参数或成功结果不会触发此规则。
            if (not result.ok and result.kind in {'validation_error', 'execution_error'}
                    and result.data.get('error_code') not in {'TOOL_TAIL_RESERVED', 'AFTER_TERMINAL'}):
                # 同一响应批次内的后续调用尚未见过反馈，不能视作模型拒不更正；
                # 资源收尾拒绝负责引导合法交接，也不是参数无进展。
                previous = (next(iter(actions[-1:]), None)
                            if len(actions) == len(state.get('actions') or []) else None)
                if (previous and previous['tool_name'] == call['name']
                        and previous['arguments'] == call['arguments']
                        and not previous.get('observation', {}).get('ok')
                        and previous.get('observation', {}).get('kind') == result.kind):
                    repeated_failure = not blocked_repeat
                    blocked_repeat = True
            action.status = "succeeded" if result.ok else ("cancelled" if result.kind == "cancelled" else "failed")
            action.observation = result.model_dump(mode="json")
            if db and not replayable:
                if call['name'] in {'record_research_progress', 'finish_research'} and result.ok and result.data.get('research_progress'):
                    owner = (state['task_id'], state['worker_id'], state['lease_token']) if state.get('worker_id') else None
                    progress.commit_action(action.model_dump(mode='json'), owner)
                elif call['name'] == 'record_evidence_matrix' and result.ok:
                    from research_loop.services.evidence_matrix import EvidenceMatrixService
                    owner = (state['task_id'], state['worker_id'], state['lease_token']) if state.get('worker_id') else None
                    EvidenceMatrixService(db).commit_action(action.model_dump(mode='json'), owner)
                else:
                    db.save_action(action.model_dump(mode="json"))
                db.append_event(state["task_id"], "observation", {"tool": call["name"], "observation": action.observation})
            observations.append({"tool": call["name"], "observation": action.observation})
            actions.append(action.model_dump(mode="json"))
            append_tool_observation(messages, call["call_id"], action.observation)
            if call["name"] == "set_research_plan" and result.ok:
                research_plan = result.data["research_plan"]
            if call['name'] == 'finish_task' and not result.ok:
                candidate = result.data.get('finish_candidate') or {}
                identity = {'task_id': state['task_id'], 'action_id': action.action_id,
                            'input_hash': finish_input_hash(call['arguments'])}
                if candidate and all(candidate.get(key) == value for key, value in identity.items()):
                    finish_candidate_ref = identity
            if call["name"] == "finish_task" and result.ok:
                finish_candidate_ref = None
                delivery = result.data["delivery"]
                # 继续配齐同批工具回复，但终结之后的副作用调用只记录未执行回执。
            if call["name"] == "request_clarification" and result.ok:
                needs_input = True
            if result.data.get("draft_id"):
                draft_id = result.data["draft_id"]
            if result.data.get("path"):
                saved_path = result.data["path"]
            if result.data.get("pages"):
                evidence_ids.extend(page["evidence_id"] for page in result.data["pages"])
            if result.data.get("evidence_ids"):
                evidence_ids.extend(result.data["evidence_ids"])
            if result.data.get("known_gaps"):
                source_warnings.extend(result.data["known_gaps"])
            if result.data.get("gaps"):
                gaps.extend(result.data["gaps"])
            if result.data.get("subtasks"):
                subtask_results.extend(result.data["subtasks"])
                for child in result.data["subtasks"]:
                    evidence_ids.extend(child.get("evidence_ids") or [])
                    gaps.extend(child.get("gaps") or [])
            if call["name"] in {"review_note", "review_research"} and result.data:
                review_result = result.data
            if call["name"] in {"finish_research", "finish_review"} and result.ok:
                structured_result = result.data
        update = {
            "actions": actions, "observations": observations, "messages": messages,
            "tool_actions": tool_count, "action_sequence": sequence, "budget_tail": budget_tail,
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "gaps": list(dict.fromkeys(gaps)), "source_warnings": list(dict.fromkeys(source_warnings)), "draft_id": draft_id, "saved_path": saved_path,
            "pending_tool_calls": [], "pending_tool_scope": [], "subtask_results": subtask_results, "review_result": review_result,
            "research_plan": research_plan, "delivery": delivery, "finish_candidate_ref": finish_candidate_ref,
        }
        handoff = state.get('research_handoff')
        new_review_failures = {key: value for key, value in failed_review_handoffs(actions).items()
                               if key not in prior_review_failures}
        qualification = qualification_handoff(actions, services.get('coverage'), state['task_id'], handoff)
        if (qualification and not new_review_failures and (handoff or {}).get('phase') != 'review_failed_closeout'
                and structured_result is None and delivery is None and qualification_handoff_active(
                qualification, services.get('coverage'), state['task_id'])):
            same = bool(handoff and (handoff.get('coverage_blocker') or {}).get('state_fingerprint')
                        == qualification['coverage_blocker']['state_fingerprint'])
            update['research_handoff'] = {**(handoff or {}), **qualification}
            if same:
                update['research_handoff']['trigger_action_id'] = handoff['trigger_action_id']
            batch = actions[len(state.get('actions') or []):]
            navigation, advanced = handoff_navigation(handoff or {}, batch)
            update['research_handoff']['navigation'] = navigation
            material_navigation, material_advanced = qualification_material_navigation(
                handoff or {}, services['coverage'], state['task_id'])
            update['research_handoff']['material_navigation'] = material_navigation
            advanced = advanced or material_advanced
            blocked_goal = qualification['coverage_blocker']['goal_id']
            # 其他明确目标仍可研究；保留旧阻断，不能借新目标把旧条件算已满足。
            old_goals = {g['goal_id'] for g in (state.get('research_plan') or {}).get('goals', [])}
            new_goals = {g['goal_id'] for g in (research_plan or {}).get('goals', [])
                         if g.get('status', 'active') == 'active'} - old_goals
            def goal_contributions(records: list[dict]) -> set[tuple[str, str]]:
                """只按具体目标的非空发现正文识别贡献，不按委派/发现ID或改写缺口。"""
                return {(goal_id, finding) for previous in records
                    if previous['tool_name'] == 'delegate_research' and (previous.get('observation') or {}).get('ok')
                    for child in previous['observation'].get('data', {}).get('subtasks', [])
                    if child.get('status') in {'completed', 'partial'}
                    for goal_id in child.get('goal_ids', []) if goal_id != blocked_goal
                    for finding in child.get('findings', []) if isinstance(finding, str) and finding.strip()}
            # 相同材料可形成新分析，不要求每步新增原文。仅新child/objective或
            # 精确收据复用不表示贡献；这里不声称识别语义相同的发现改写。
            outside_goal = bool(new_goals) or bool(goal_contributions(batch) -
                goal_contributions(state.get('actions') or []))
            # 沿用已有交接的有界结束；真正新目标、版本和范围仍可继续。
            # 新档案区间仅允许必要导航，不解除资格条件，也不等于研究进展。
            if same and not advanced and not outside_goal and handoff.get('qualification_feedback_seen'):
                update.update(finished=True, final_reason='qualification handoff unresolved',
                    gaps=list(dict.fromkeys(gaps + ['资格条件仍未解除；保留精确绑定候选，未冒充完成。'])))
                update['research_handoff']['phase'] = 'unresolved'
            elif same and not advanced and not outside_goal:
                update['research_handoff']['qualification_feedback_seen'] = True
        elif (handoff and (not handoff.get('coverage_blocker') or new_review_failures
                           or handoff.get('phase') == 'review_failed_closeout')
              and structured_result is None and delivery is None):
            current = progress.snapshot(state['task_id']) if progress else {}
            artifact_changed = draft_id != state.get('draft_id') or saved_path != state.get('saved_path')
            current_reflection, _ = reflection_handoffs(actions)
            new_failures = {key: value for key, value in failed_review_handoffs(actions).items()
                            if key not in prior_review_failures}
            if new_failures and handoff.get('phase') != 'review_failed_closeout':
                update['research_handoff'] = {**handoff, 'phase': 'review_failed_closeout',
                    'failed_review': next(reversed(new_failures.values()))}
            elif handoff.get('phase') != 'review_failed_closeout' and (handoff_progress_changed(handoff, current) or artifact_changed
                    or bool(current_reflection - prior_reflection)):
                update['research_handoff'] = {**handoff, 'phase':'progress_recorded'}
            else:
                navigation, advanced = handoff_navigation(handoff, actions[len(state.get('actions') or []):])
                if advanced:
                    update['research_handoff'] = {**handoff, 'phase': ('review_failed_closeout' if handoff.get('phase') == 'review_failed_closeout' else 'readback'), 'navigation': navigation}
                elif recoverable_error and not handoff.get('tool_error'):
                    # 错误反馈是一次交接动作，不是科研进展；纠正后仍须产生合法交接。
                    # 不以换参数/换错误反复重开交接，也不改变现有请求或Token配置。
                    update['research_handoff'] = {**handoff, 'navigation': navigation,
                                                   'tool_error': recoverable_error}
                else:
                    # 导航没有覆盖新范围，也没有真实成果更新时，沿用原有有界结束。
                    update.update(finished=True, final_reason=('review failure closeout incomplete'
                        if handoff.get('phase') == 'review_failed_closeout' else 'research readback made no progress'),
                        research_handoff={**handoff,'phase':'unresolved', 'navigation': navigation},
                        gaps=list(dict.fromkeys(gaps + [
                            '核验失败后的收尾未完成；保留当前正文与执行错误，不产生语义判断。'
                            if handoff.get('phase') == 'review_failed_closeout' else
                            '相同材料反复读回后仍未形成有效阶段交接；保留已读原文与已提交发现，未冒充研究完成。'])))
        if terminal_denied:
            update.update({'finished': True, 'final_reason': 'tool budget exhausted before terminal handoff',
                           'gaps': list(dict.fromkeys(gaps + ['工具额度不足以完成交接；已有证据保留，未冒充完成']))})
        if blocked_repeat:
            message = ('相同参数连续失败且没有新进展；已停止并保留已有产物。' if repeated_failure else
                       '相同不可用工具再次请求；未执行，保留已有成果。' if scope_rejected else
                       '同一不可重试操作重复请求；已停止并保留已有产物')
            update.update({"finished": True, "final_reason":
                           ('repeated failed action made no progress' if repeated_failure else
                            'repeated unavailable action stopped: ' + (recoverable_error or {}).get('message', '')
                            if scope_rejected else 'repeated nonretryable action stopped'),
                           "gaps": list(dict.fromkeys(gaps + [message]))})
        if needs_input:
            update.update({"finished": True, "final_reason": "clarification requested"})
        if structured_result is not None:
            update.update({"finished": True, "final_reason": "structured role result",
                           "structured_result": structured_result,
                           "final_answer": "\n\n".join(structured_result.get("findings", [])) if state.get("role") == "researcher" else "independent review completed"})
        if delivery is not None:
            update.update({"finished": True, "final_reason": "task delivered",
                           "final_answer": delivery["answer_markdown"], "draft_id": delivery["draft_id"],
                           "saved_path": delivery["saved_path"], "gaps": delivery["gaps"]})
        return record(state, update)

    def route(state: AgentState) -> str:
        """模型提出工具调用时转执行节点；已结束或没有工具调用时离开图，不另加自动纠错循环。"""
        if state.get("finished") or not state.get("pending_tool_calls"):
            return END
        return "execute_tools"

    def after_tools(state: AgentState) -> str:
        """Researcher 正式结束取证后转到单一精读节点，其余角色保持原行动循环。"""
        if (deep_reading and state.get("role") == "researcher" and state.get("structured_result") is not None
                and state.get("evidence_ids") and not state.get("cancelled")
                and not (state.get("deep_reading") or {}).get("done")):
            return "deep_reading"
        return "decide"

    graph = StateGraph(AgentState)
    graph.add_node("decide", decide)
    graph.add_node("execute_tools", execute_tools)
    if deep_reading:
        from research_loop.agents.deep_reading import deep_reading_node
        graph.add_node("deep_reading", lambda state: deep_reading_node(state, provider, services["papers"], db, record))
        graph.add_edge("deep_reading", END)
        # 节点内响应收据落盘后退出，恢复仍回同一节点，不重新执行 finish_research。
        graph.add_conditional_edges(START, after_tools, {"deep_reading": "deep_reading", "decide": "decide"})
    else:
        graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", route, {"execute_tools": "execute_tools", END: END})
    if deep_reading:
        graph.add_conditional_edges("execute_tools", after_tools, {"deep_reading": "deep_reading", "decide": "decide"})
    else:
        graph.add_edge("execute_tools", "decide")
    return graph.compile()


def initial_state(task_id: str, prompt: str, save_policy: str, attachments: dict[str, dict[str, Any]] | None = None,
                  max_model_requests: int = 48, max_tool_actions: int = 80,
                  reserve_model_requests: int = 10) -> AgentState:
    """创建新任务的图状态；附件以授权 input_ref 注入，计数与产物字段从空状态开始。"""
    return {
        "task_id": task_id, "role": "supervisor", "prompt": prompt, "save_policy": save_policy,
        "attachments": attachments or {}, "messages": initial_messages(prompt, save_policy, attachments or {}),
        "actions": [], "observations": [], "model_requests": 0, "tool_actions": 0,
        "max_model_requests": max_model_requests, "max_tool_actions": max_tool_actions,
        "reserve_model_requests": reserve_model_requests, "evidence_ids": [], "gaps": [],
        "pending_tool_calls": [], "cancelled": False, "finished": False,
        "subtask_results": [], "review_result": None, "root_task_id": task_id,
    }
