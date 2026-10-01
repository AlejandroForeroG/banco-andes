'Stateful financial assistant with an observable ReAct cycle.'

from __future__ import annotations

import json
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from agent.session import active_session
from agent.memory.store import load_client_memory
from cli.config import get_settings
from observability.tracing import sanitizar_traza


DEFAULT_MAX_ITERATIONS = 6
SHORT_TERM_DENIAL_MARKERS = (
    "i do not remember our conversat",
    "i cannot access the conversat",
    "you need to sign in",
    "i don't remember",
    "i can't access",
    "i have no memory",
    "please sign in",
    "please log in",
    "you need to log in",
    "/login",
)

SYSTEM_PROMPT = """You are Banco Andes' financial guidance assistant. Always respond in English, clearly and briefly, with at most 100 words in the final answer.
Interpret the user's intent first. Use tools for every product, profile, history, or recorded fact; never invent information. Public queries do not require identification. Private data comes from the authenticated channel session: never request or handle PINs or tokens. If a tool reports no session, ask the customer to use /login.
For questions about my loan, profile, score, borrowing capacity, or active products, call query_financial_profile directly. Do not call verify_customer before a private query or request identification again after login. Never promise loan approval, describe a customer as eligible or preapproved, estimate approval probability, or infer approval from their profile. You may compare published requirements with observed facts, but state that a match does not establish eligibility or approval. Explain tool failures and offer retry or handoff. Do not claim a product is best without verified requirements and the full customer need. Never promise notifications or future actions that tools have not confirmed. Refer advisor requests to /advisor, which asks for consent.

Memory: when an authenticated customer explicitly asks to remember a stable goal or preference, or a tool confirms a milestone such as a handoff, use remember_financial_preference. Recover that context in future conversations and confirm it remains current before acting. Never store PINs, tokens, accounts, income, scores, debts, or other sensitive or changing financial data; query the business API when needed.
The message history and summary belong to the current thread and are available short-term memory without authentication. Answer from that context first. Do not use query_financial_memory to explain what happened in this thread or answer general questions about this conversation. That tool is only for durable preferences across conversations and requires an active session. A missing long-term memory session does not mean the current thread history was lost.

Amounts: normalize colloquial monetary expressions before reasoning or saving. Preserve Colombian peso amounts and any stated payment period. For example, one million five hundred thousand means 1.500.000 COP, never 500 million COP. If an amount has multiple interpretations, ask for the exact amount and wait for confirmation before calling a memory tool. Never save a guess.

Use an explicit, auditable ReAct cycle. Before each action, write one line starting with "Thought:" that briefly states the missing verifiable fact and why the next tool is needed, without revealing extended internal reasoning. Make at most one tool call per iteration and await its observation. If your Thought says a fact available through a tool is missing, include the structured tool call in the same response; never end with only a Thought. Do not announce actions you will not execute. Once enough information is available, stop calling tools and begin the synthesis with "Final answer:". Complex tasks may need several Thought → Action → Observation cycles."""


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    iterations: int
    max_iterations: int
    summary: str
    long_term_memory: dict[str, str]
    compression_before: int
    compression_after: int
    reasoning_trace: list[str]
    pending_draft: str
    review_required: bool
    review: dict


def _is_short_term_denial(message: AnyMessage) -> bool:
    content = str(getattr(message, "content", "")).lower()
    return any(marker in content for marker in SHORT_TERM_DENIAL_MARKERS)


def _summary_source(messages: list[AnyMessage]) -> list[AnyMessage]:
    filtered = [message for message in messages if not _is_short_term_denial(message)]
    return filtered or messages


def _summary_confuses_authentication(summary: str) -> bool:
    normalized = summary.lower()
    return any(marker in normalized for marker in SHORT_TERM_DENIAL_MARKERS)


def _session_status_message() -> SystemMessage:
    """Tell the model only whether the channel is authenticated, never who or how."""
    if active_session() is None:
        content = (
            "Channel status: no authenticated session is active. If the customer requests "
            "private data, ask them to use /login; do not request their customer ID "
            "or PIN inside the conversation."
        )
    else:
        content = (
            "Channel status: an authenticated session is active. For private data, "
            "call the relevant tool directly without asking for identification, "
            "a PIN, or other authentication data."
        )
    return SystemMessage(content=content)


def _long_term_memory_message(state: AgentState) -> SystemMessage:
    'Inject durable customer preferences without revealing their identity.'
    remembered = state.get("long_term_memory", {})
    if not remembered:
        status = (
            "is unavailable because the channel is not authenticated"
            if active_session() is None
            else "contains no saved preferences for the authenticated customer"
        )
        return SystemMessage(
            content=(
                f"Long-term memory: {status}. This does not affect the history or "
                "summary of the current conversation, which remain available."
            )
        )
    return SystemMessage(
        content=(
            "Long-term memory retrieved for the current session: "
            f"{json.dumps(remembered, ensure_ascii=False)}. "
            "Treat it as previously declared context, not current banking data."
        )
    )


def _tool_error_message(error: Exception) -> str:
    return (
        f"The tool failed: {error}. Do not invent a result; "
        "explain the problem and offer a safe alternative."
    )


def _correct_authenticated_thought(content: str, tool_calls: list[dict]) -> str:
    """Remove a local-model authentication contradiction from an audited step."""
    if active_session() is None or not tool_calls:
        return content
    normalized = content.lower()
    contradictions = (
        "no session is active",
        "no session exists",
        "session is not active",
        "cannot store",
        "request authentication",
    )
    if not any(marker in normalized for marker in contradictions):
        return content
    tool_name = tool_calls[0].get("name", "the relevant tool")
    return (
        "Thought: an authenticated session is active and the request requires "
        f"run {tool_name} to obtain a verifiable result."
    )


def _next_explicitly_requested_tool(
    messages: list[AnyMessage], available_tool_names: set[str]
) -> tuple[str, dict] | None:
    """Keep a multi-source request open until every requested source was observed."""
    last_human_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
        ),
        None,
    )
    if last_human_index is None:
        return None
    request = str(messages[last_human_index].content).lower()
    observed = {
        message.name
        for message in messages[last_human_index + 1 :]
        if isinstance(message, ToolMessage) and message.name
    }
    required: list[tuple[str, dict]] = []
    if "profile" in request:
        required.append(("query_financial_profile", {}))
    if "history" in request or "applications" in request:
        required.append(("query_application_history", {}))
    if ("housing" in request or "mortgage" in request) and ("loan" in request or "credit" in request):
        required.append(("query_credit_catalog", {"categoria": "housing"}))
    return next(
        (
            specification
            for specification in required
            if specification[0] in available_tool_names
            and specification[0] not in observed
        ),
        None,
    )


def build_graph(tools: list | None = None, checkpointer=None):
    """Build and compile the agent ↔ tools reasoning cycle."""
    if tools is None:
        from agent.tools import get_local_tools

        tools = get_local_tools()

    settings = get_settings()

    tools = [tool for tool in tools if tool.name != 'register_handoff']
    available_tool_names = {tool.name for tool in tools}
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
        num_ctx=8192,
        num_predict=500,
    )
    llm_with_tools = llm.bind_tools(tools)

    def prepare_turn(_: AgentState) -> dict:
        'Load customer memory and reset the turn budget and trace.'
        session = active_session()
        remembered = load_client_memory(session.client_id) if session else {}
        return {
            "iterations": 0,
            "long_term_memory": remembered,
            "reasoning_trace": [],
            "pending_draft": "",
            "review_required": False,
            "review": {},
        }

    def route_before_agent(state: AgentState) -> str:
        threshold = settings.context_compression_threshold
        if len(state.get("messages", [])) >= threshold:
            return "comprimir_contexto"
        return "reason"

    async def compress_context(state: AgentState) -> dict:
        'Summarize older messages without splitting a recent turn.'
        messages = state.get("messages", [])
        if not messages:
            return {}
        previous_summary = state.get("summary", "")
        cut = max(len(messages) - settings.context_messages_to_keep, 0)
        while cut < len(messages) and not isinstance(messages[cut], HumanMessage):
            cut += 1
        past_messages = messages[:cut]
        recent_messages = messages[cut:]
        if not past_messages:
            return {}
        source_messages = _summary_source(past_messages)
        source_transcript = get_buffer_string(source_messages)
        safe_previous_summary = (
            "" if _summary_confuses_authentication(previous_summary) else previous_summary
        )
        fallback_context = "\n".join(
            part for part in (safe_previous_summary, source_transcript) if part
        )
        summary_request = [
            SystemMessage(
                content=(
                    "Generate only a short list of conversation facts: goals, "
                    "preferences, amounts, terms, decisions, confirmed tool results, "
                    "and pending items. Start with 'Thread facts:'. Do not discuss permissions, "
                    "memory availability, sessions, login, or ability to remember. "
                    "Do not include PINs, tokens, or authentication data."
                )
            ),
            SystemMessage(
                content=f"Previous cumulative summary: {safe_previous_summary or '(none)'}"
            ),
            SystemMessage(content=f"Conversation to compress:\n{source_transcript}"),
        ]
        try:
            response = await llm.ainvoke(summary_request)
            summary = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            if _summary_confuses_authentication(summary):
                summary = f"Thread facts:\n{fallback_context[-6000:]}"
        except Exception:
            summary = f"Thread facts:\n{fallback_context[-6000:]}"
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_messages],
            "summary": summary,
            "compression_before": len(messages),
            "compression_after": len(recent_messages),
        }

    async def reason_node(state: AgentState) -> dict:
        'Choose one ReAct action or produce the final answer.'
        iteration = state.get("iterations", 0)
        max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
        if iteration >= max_iterations:
            stop = (
                "I stopped the cycle because it reached the safety limit. "
                "I can retry or hand off the case to an advisor."
            )
            return {
                "messages": [AIMessage(content=stop)],
                "iterations": iteration,
                "reasoning_trace": [
                    *state.get("reasoning_trace", []),
                    f"[Limit] {stop} (max_iterations={max_iterations})",
                ],
            }

        model_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            _session_status_message(),
            _long_term_memory_message(state),
            SystemMessage(
                content=(
                    "Previous conversation summary: "
                    f"{state.get('summary', '(no summary yet)')}"
                )
            ),
            *state.get("messages", []),
        ]
        response = await llm_with_tools.ainvoke(model_messages)
        initial_content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        ).strip()
        thought_only = (
            initial_content.lower().startswith("thought:")
            and "final answer:" not in initial_content.lower()
            and not getattr(response, "tool_calls", None)
        )
        if thought_only:
            repaired_response = await llm_with_tools.ainvoke(
                [
                    *model_messages,
                    response,
                    SystemMessage(
                        content=(
                            "Format correction: the previous Thought stated that "
                            "information is missing. Continue with ONE structured call "
                            "to the required tool. Do not repeat the thought or "
                            "respond with text alone."
                        )
                    ),
                ]
            )
            repaired_content = (
                repaired_response.content
                if isinstance(repaired_response.content, str)
                else str(repaired_response.content)
            ).strip()
            response = (
                repaired_response.model_copy(update={"content": initial_content})
                if getattr(repaired_response, "tool_calls", None)
                and not repaired_content
                else repaired_response
            )
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        ).strip()
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        if len(tool_calls) > 1:

            tool_calls = tool_calls[:1]
            response = response.model_copy(update={"tool_calls": tool_calls})
        if not tool_calls:
            required_action = _next_explicitly_requested_tool(
                state.get("messages", []), available_tool_names
            )
            if required_action is not None:
                tool_name, tool_args = required_action
                content = (
                    "Thought: a requested source has not been queried yet; "
                    f"I must call {tool_name} before answering."
                )
                tool_calls = [
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"required-{iteration + 1}-{tool_name}",
                        "type": "tool_call",
                    }
                ]
                response = AIMessage(content=content, tool_calls=tool_calls)
        corrected_content = _correct_authenticated_thought(content, tool_calls)
        if corrected_content != content:
            content = corrected_content
            response = response.model_copy(update={"content": corrected_content})
        if tool_calls:
            call = tool_calls[0]
            trace_entry = (
                f"[Step {iteration + 1}] {content or 'Thought: a verifiable fact is needed.'} "
                f"→ Action: {call['name']}({call.get('args', {})})"
            )
        else:
            trace_entry = (
                f"[Step {iteration + 1}] "
                f"{content or 'Final answer: no content was generated.'}"
            )
        from agent.review import needs_review
        requires_review = not tool_calls and needs_review(state.get('messages', []), content)
        if not tool_calls and response.response_metadata.get('done_reason') == 'length':
            from agent.review import SAFE_FAILURE
            return {'messages': [AIMessage(content=SAFE_FAILURE)], 'pending_draft': '',
                    'review_required': False, 'review': {'status': 'failed', 'error': 'DRAFT_TRUNCATED'},
                    'iterations': iteration + 1}
        if requires_review:
            trace_entry = f"[Step {iteration + 1}] Draft withheld for editorial review."
        return {
            "messages": [] if requires_review else [response],
            "pending_draft": content if requires_review else "",
            "review_required": requires_review,
            "iterations": iteration + 1,
            "reasoning_trace": [*state.get("reasoning_trace", []), trace_entry],
        }

    tool_node = ToolNode(
        tools,
        name="act",
        handle_tool_errors=_tool_error_message,
    )

    async def act_node(state: AgentState) -> dict:
        'Execute the selected action and record its observation.'
        result = await tool_node.ainvoke(state)
        tool_messages = list(result.get("messages", []))
        observations = [
            f"Observation: {message.name or 'tool'} → {message.content}"
            for message in tool_messages
        ]
        return {
            "messages": tool_messages,
            "reasoning_trace": [*state.get("reasoning_trace", []), *observations],
        }

    @traceable(
        name="decision_react",
        run_type="chain",
        process_inputs=sanitizar_traza,
        process_outputs=sanitizar_traza,
    )
    def route_after_reason(state: AgentState):
        if state.get('review_required'):
            return 'revisar_orientacion'
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "act"
        return END

    async def review_node(state: AgentState):
        from agent.review import review_draft
        answer, review = await review_draft(state['messages'], state['pending_draft'])
        return {'messages': [AIMessage(content=answer)], 'pending_draft': '',
                'review_required': False, 'review': review,
                'reasoning_trace': [*state.get('reasoning_trace', []),
                                    f"Editorial review: {review['status']}"]}

    graph = StateGraph(AgentState)
    graph.add_node("memoria_largo_plazo", prepare_turn)
    graph.add_node("comprimir_contexto", compress_context)
    graph.add_node("reason", reason_node)
    graph.add_node("act", act_node)
    graph.add_node('revisar_orientacion', review_node)
    graph.add_edge('revisar_orientacion', END)
    graph.add_edge(START, "memoria_largo_plazo")
    graph.add_conditional_edges(
        "memoria_largo_plazo",
        route_before_agent,
        {"comprimir_contexto": "comprimir_contexto", "reason": "reason"},
    )
    graph.add_edge("comprimir_contexto", "reason")
    graph.add_conditional_edges("reason", route_after_reason)
    graph.add_edge("act", "reason")
    return graph.compile(checkpointer=checkpointer)
