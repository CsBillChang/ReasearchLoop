"""通用行动循环：模型决定下一步，程序校验并执行工具，再把观察交回模型；同时记录恢复状态。"""
from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph

from research_loop.agents.state import AgentState
from research_loop.agents.supervisor import append_tool_observation, initial_messages
from research_loop.domain.tasks import ActionRecord, Observation, SavePolicy
from research_loop.providers.deepseek import DeepSeekProvider, ModelResponse, InvalidConversation, ProviderResponseError, RequestNotSent, validate_messages
from research_loop.tools.base import ToolContext
from research_loop.tools.registry import ToolRegistry


def matrix_thinking(state: AgentState) -> dict[str, str] | None:
    """仅为已取证的矩阵整理与交付选择 DPR 相同的非思考模式。

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
                db: Any = None, max_model_requests: int = 12,
                max_tool_actions: int = 20, reserve_model_requests: int = 1,
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
    def record(state: AgentState, update: dict[str, Any]) -> dict[str, Any]:
        """合并状态前验证 Worker 租约；失去所有权时拒绝写入，避免旧进程覆盖新执行结果。"""
        if db and state.get("worker_id") and not db.is_owner(state["task_id"], state["worker_id"], state.get("lease_token")):
            raise RuntimeError("worker lease lost; graph write rejected")
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

    def decide(state: AgentState) -> dict[str, Any]:
        """先检查取消、额度和未结算请求，再复用收据或发请求；输出待执行工具或结束状态。"""
        if state.get("finished") or state.get("pending_tool_calls"):
            return record(state, {})
        if cancelled(state):
            return record(state, {"cancelled": True, "finished": True, "final_reason": "cancelled"})
        request_number = int(state.get("model_requests", 0)) + 1
        if request_number > int(state.get("max_model_requests", max_model_requests)):
            return record(state, {"finished": True, "final_reason": "model request budget exhausted"})
        closing = request_number >= int(state.get("max_model_requests", max_model_requests))
        messages = list(state.get("messages") or [])
        if not messages:
            messages = initial_messages(state["prompt"], state.get("save_policy", SavePolicy.deny.value),
                                        state.get("attachments") or {})
        try:
            validate_messages(messages)
        except InvalidConversation as exc:
            return record(state, {"finished": True, "final_reason": "invalid model conversation: " + str(exc),
                                  "gaps": list(state.get("gaps") or []) + ["对话工具回复不完整；未发送模型请求"]})
        role = state.get("role", "supervisor")
        if db:
            budget = db.budget_summary(state.get("root_task_id", state["task_id"]))
            if budget:
                usable_limit = budget["model_limit"] - (budget["reserve_tail"] if role == "researcher" else 0)
                closing = closing or budget["model_used"] >= usable_limit - 1
        if db and db.has_unknown_family_call(state.get("root_task_id", state["task_id"])):
            return record(state, {"finished": True, "final_reason": "model response unknown: task family is blocked",
                                  "gaps": list(state.get("gaps") or []) + ["任务链有未结算请求，未自动继续调用"]})
        schemas = registry.schemas(role)
        terminal_names = {"finish_task", "finish_research", "finish_review"}
        terminal_tools = [t for t in schemas if t["function"]["name"] in terminal_names]
        terminal_only = closing and bool(terminal_tools)
        force_text = closing and not terminal_tools
        if closing:
            # 收尾仍使用现有结构化交付接口；删掉工具却要求finish会使调用标记落入正文。
            instruction = ("当前是本角色最后一次可用请求。只用提供的结束工具交付已有成果；"
                           "未完成目标明确标为unresolved/partial并说明原因，不新增检索、委派、修订或保存。")
            if force_text:
                instruction = "当前是最后一次可用请求。直接用自然语言交付已有成果和未完成原因，不输出或模拟工具调用。"
            messages = messages + [{"role": "system", "content": instruction}]
        request = {"messages": messages, "tools": terminal_tools if terminal_only else ([] if force_text else schemas),
                   "force_text": force_text, "terminal_only": terminal_only}
        receipt = db.get_model_call(state["task_id"], request_number) if db else None
        if receipt and receipt.get("request_json"):
            # 已持久请求始终按当时的messages恢复，偏好变化不能改写历史成功收据。
            request = json.loads(receipt["request_json"])
            messages = request["messages"]
            force_text = request.get("force_text", force_text)
            terminal_only = request.get("terminal_only", False)
        elif services.get("context") and role == "supervisor":
            try:
                messages, metrics = services["context"].prepare(state,messages,request["tools"])
                state = {**state,"context_metrics":metrics}
                validate_messages(messages)
                request = {**request,"messages":messages,"max_tokens":services["context"].settings.context_output_reserve}
            except (ValueError, InvalidConversation) as exc:
                return record(state,{"finished":True,"final_reason":str(exc),
                    "gaps":list(state.get("gaps") or [])+[str(exc)]})
        from research_loop.storage.token_budget import TokenBudgetStore
        tokens = TokenBudgetStore(db) if db else None
        root = state.get("root_task_id", state["task_id"])
        # 只给新策略任务明确输出上限；旧收据和历史调用参数原样恢复。
        if tokens and tokens.summary(root) and not receipt:
            request.setdefault("max_tokens", 8192)
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
                    if role == "supervisor" and services.get("context") and callable(getattr(provider,"send_stream",None)):
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
        if terminal_only and any(call.name not in {t["function"]["name"] for t in request["tools"]} for call in response.tool_calls):
            # schema收缩之外还校验执行入口，避免模型越过最后一次请求的无副作用边界。
            return record(state, {"model_requests": request_number, "finished": True,
                                  "final_reason": "closing response requested a non-terminal action",
                                  "gaps": list(state.get("gaps") or []) + ["最后一次请求提出了新的执行动作，已拒绝；已有成果保留"]})
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
                "messages": messages + [assistant_message],
                "pending_tool_calls": [
                    {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
                    for call in response.tool_calls
                ],
            })
        content = (response.content or "").lstrip()
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
            "messages": messages + [final_message],
            "final_answer": response.content or "", "finished": True,
            "final_reason": "model finished" if response.content else "model returned no content",
        })

    def execute_tools(state: AgentState) -> dict[str, Any]:
        """按调用顺序执行已授权工具，复用已落库观察；汇总证据、子任务、草稿与文件路径。"""
        if cancelled(state):
            return record(state, {"cancelled": True, "finished": True, "final_reason": "cancelled before tool execution"})
        actions = list(state.get("actions") or [])
        observations = list(state.get("observations") or [])
        messages = list(state.get("messages") or [])
        tool_count = int(state.get("tool_actions", 0))
        draft_id = state.get("draft_id")
        saved_path = state.get("saved_path")
        evidence_ids = list(state.get("evidence_ids") or [])
        gaps = list(state.get("gaps") or [])
        source_warnings = list(state.get("source_warnings") or [])
        subtask_results = list(state.get("subtask_results") or [])
        review_result = state.get("review_result")
        research_plan = state.get("research_plan")
        delivery = state.get("delivery")
        needs_input = False
        structured_result = None
        blocked_repeat = False
        pending = list(state.get("pending_tool_calls") or [])
        context = ToolContext(
            task_id=state["task_id"], save_policy=SavePolicy(state.get("save_policy", SavePolicy.deny.value)),
            attachments=state.get("attachments") or {}, cancelled=lambda: cancelled(state), services=services,
            role=state.get("role", "supervisor"), root_task_id=state.get("root_task_id", state["task_id"]),
            parent_task_id=state.get("parent_task_id"),
        )
        for call in pending:
            if tool_count >= int(state.get("max_tool_actions", max_tool_actions)):
                return record(state, {
                    "actions": actions, "observations": observations, "messages": messages,
                    "tool_actions": tool_count, "finished": True, "final_reason": "tool action budget exhausted",
                })
            tool_count += 1
            prior = db.get_action(state["task_id"], tool_count) if db else None
            replayable = prior and prior.get("status") in {"succeeded", "failed", "cancelled"} and prior.get("observation")
            if db and not prior and not db.reserve_tool_slot(state.get("root_task_id", state["task_id"])):
                return record(state, {"actions": actions, "observations": observations, "messages": messages,
                                      "tool_actions": tool_count - 1, "finished": True,
                                      "final_reason": "shared tool action budget exhausted",
                                      "gaps": gaps + ["共享工具行动额度已用尽"]})
            action = ActionRecord(task_id=state["task_id"], sequence=tool_count, tool_name=call["name"],
                                  arguments=call["arguments"])
            if replayable:
                result = Observation.model_validate(prior.get("observation") or {})
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
                    result = registry.execute(call["name"], call["arguments"], context)
            action.status = "succeeded" if result.ok else ("cancelled" if result.kind == "cancelled" else "failed")
            action.observation = result.model_dump(mode="json")
            if db and not replayable:
                if call['name'] == 'record_evidence_matrix' and result.ok:
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
            if call["name"] == "finish_task" and result.ok:
                delivery = result.data["delivery"]
                # 终结动作之后不再执行同批次的副作用调用；交付收据可供恢复直接复用。
                break
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
            if call["name"] == "review_note" and result.data:
                review_result = result.data
            if call["name"] in {"finish_research", "finish_review"} and result.ok:
                structured_result = result.data
        update = {
            "actions": actions, "observations": observations, "messages": messages,
            "tool_actions": tool_count, "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "gaps": list(dict.fromkeys(gaps)), "source_warnings": list(dict.fromkeys(source_warnings)), "draft_id": draft_id, "saved_path": saved_path,
            "pending_tool_calls": [], "subtask_results": subtask_results, "review_result": review_result,
            "research_plan": research_plan, "delivery": delivery,
        }
        if blocked_repeat:
            update.update({"finished": True, "final_reason": "repeated nonretryable action stopped",
                           "gaps": list(dict.fromkeys(gaps + ["同一不可重试操作重复请求；已停止并保留已有产物"]))})
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
                  max_model_requests: int = 12, max_tool_actions: int = 20,
                  reserve_model_requests: int = 1) -> AgentState:
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
