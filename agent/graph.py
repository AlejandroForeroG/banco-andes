"""Grafo stateful del asistente financiero con ciclo ReAct observable."""

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
    "no tengo memoria de nuestra conversaci",
    "no tengo acceso a la conversaci",
    "necesitas iniciar sesi",
    "sesión autenticada",
    "/login",
)

SYSTEM_PROMPT = """Eres el asistente de orientación financiera de Banco Andes.
Primero interpreta la intención. Usa herramientas para todo dato de producto,
perfil, historial o registro; nunca inventes información. Las consultas públicas
no requieren identificación. Los datos privados salen de la sesión que el
cliente ya abrió en el canal: nunca pidas ni manejes su clave ni tokens, y si
una herramienta reporta que no hay sesión, pídele que inicie sesión con /login.
Para preguntas como "mi crédito", "mi perfil", "mi score", "mi capacidad" o
"mis productos activos", usa consultar_perfil_financiero. No uses
verificar_cliente como paso previo a una consulta privada y no vuelvas a pedir
la identificación después del login. No prometas la aprobación de un crédito ni
califiques al cliente como "apto", "preaprobado", "con alta probabilidad" o con
un perfil "sólido para aprobar". Puedes comparar requisitos publicados con datos
observados, pero aclara que esa coincidencia no determina elegibilidad ni aprobación. Si una tool
falla, explica la limitación y ofrece reintentar o escalar. No declares que un
producto es el mejor sin conocer el perfil y la necesidad completa. No prometas
notificaciones ni acciones futuras que la tool no haya confirmado. Responde en
caso de solicitar asesor indicando el comando /asesor del CLI, que pide consentimiento.
Responde en
español, de forma clara y breve: la respuesta final tiene como máximo 100 palabras.
Prioriza la necesidad del cliente y las limitaciones verificadas; evita enumerar
todos los campos del catálogo. La disponibilidad no implica aprobación.

Memoria: cuando un cliente autenticado pida explícitamente recordar un objetivo
o preferencia estable, o cuando una tool confirme un hito como un escalamiento,
usa recordar_preferencia_financiera. Recupera ese contexto en conversaciones
futuras, pero confirma si sigue vigente antes de actuar. Nunca guardes claves,
tokens, cuentas, ingresos, score, deudas ni otros datos sensibles o variables;
consúltalos en la API empresarial cuando se necesiten.

El historial de mensajes y el resumen pertenecen a la conversación actual y son
memoria de corto plazo ya disponible; no requieren una sesión autenticada.
Responde primero con ese contexto. No uses consultar_memoria_financiera para
explicar qué ocurrió dentro del mismo thread ni ante preguntas generales como
"¿qué recuerdas de nuestra conversación?". Esa tool se usa únicamente para
preferencias duraderas entre conversaciones y requiere una sesión activa. Si la
tool de memoria larga falla o no hay sesión, no afirmes que perdiste el historial
del thread: ambos mecanismos son independientes.

Montos: normaliza expresiones coloquiales colombianas antes de razonar o guardar.
"Un millón quinientos" o "millón quinientos" significa 1.500.000 COP, nunca
500 millones. Indica también la periodicidad cuando el cliente la expresó, por
ejemplo "1.500.000 COP mensuales". Si una frase admite más de una interpretación,
pregunta el valor exacto y espera confirmación antes de llamar una herramienta de
memoria; no guardes una suposición.

Patrón de razonamiento: trabaja con un ciclo ReAct explícito y auditable. Antes
de cada acción escribe una sola línea que empiece por "Pensamiento:" y resuma,
sin revelar razonamiento interno extenso, qué dato verificable falta y por qué
la siguiente herramienta es necesaria. Haz como máximo una llamada de
herramienta por iteración; espera su observación antes de decidir la siguiente
acción. Si el Pensamiento concluye que falta un dato disponible mediante una
herramienta, DEBES incluir la llamada estructurada a esa herramienta en la misma
respuesta: nunca termines el turno con solo un Pensamiento. No anuncies acciones
que no vayas a ejecutar. Cuando ya tengas toda la información, no llames más
herramientas y empieza la síntesis con "Respuesta final:". Una tarea compleja puede requerir varios ciclos
Pensamiento → Acción → Observación antes de la respuesta final.
"""


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
            "Estado del canal: no hay una sesión autenticada. Si el cliente pide "
            "datos privados, indícale que use /login; no le solicites identificación "
            "ni clave dentro de la conversación."
        )
    else:
        content = (
            "Estado del canal: hay una sesión autenticada activa. Para datos privados, "
            "invoca directamente la herramienta correspondiente sin pedir identificación, "
            "clave ni otro dato de autenticación."
        )
    return SystemMessage(content=content)


def _long_term_memory_message(state: AgentState) -> SystemMessage:
    """Inyecta preferencias duraderas del cliente sin exponer su identidad."""
    remembered = state.get("long_term_memory", {})
    if not remembered:
        status = (
            "no está disponible porque el canal no está autenticado"
            if active_session() is None
            else "no contiene preferencias guardadas para el cliente autenticado"
        )
        return SystemMessage(
            content=(
                f"Memoria de largo plazo: {status}. Esto no afecta el historial ni "
                "el resumen de la conversación actual, que sí están disponibles."
            )
        )
    return SystemMessage(
        content=(
            "Memoria de largo plazo recuperada para la sesión actual: "
            f"{json.dumps(remembered, ensure_ascii=False)}. "
            "Trátala como contexto declarado previamente, no como datos bancarios vigentes."
        )
    )


def _tool_error_message(error: Exception) -> str:
    return (
        f"La herramienta falló: {error}. No inventes un resultado; "
        "explica el problema y ofrece una alternativa segura."
    )


def _correct_authenticated_thought(content: str, tool_calls: list[dict]) -> str:
    """Remove a local-model authentication contradiction from an audited step."""
    if active_session() is None or not tool_calls:
        return content
    normalized = content.lower()
    contradictions = (
        "no hay sesión",
        "no existe una sesión",
        "sesión no está activa",
        "no se puede almacenar",
        "pedir autenticación",
    )
    if not any(marker in normalized for marker in contradictions):
        return content
    tool_name = tool_calls[0].get("name", "la herramienta correspondiente")
    return (
        "Pensamiento: hay una sesión autenticada activa y la solicitud requiere "
        f"ejecutar {tool_name} para obtener un resultado verificable."
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
    if "perfil" in request:
        required.append(("consultar_perfil_financiero", {}))
    if "historial" in request or "solicitudes" in request:
        required.append(("consultar_historial_solicitudes", {}))
    if "vivienda" in request and ("crédito" in request or "credito" in request):
        required.append(("consultar_catalogo_creditos", {"categoria": "vivienda"}))
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

    tools = [tool for tool in tools if tool.name != 'registrar_escalamiento']
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
        """Recupera memoria y reinicia el presupuesto y la traza del turno."""
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
        """Resume el pasado y conserva una cola reciente sin partir un turno."""
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
                    "Genera únicamente una lista breve de hechos del diálogo: objetivos, "
                    "preferencias, montos, plazos, decisiones, resultados confirmados de tools "
                    "y pendientes. Empieza con 'Hechos del hilo:'. No hables de permisos, "
                    "disponibilidad de memoria, sesiones, login ni capacidad para recordar. "
                    "No incluyas claves, tokens ni datos de autenticación."
                )
            ),
            SystemMessage(
                content=f"Resumen acumulado anterior: {safe_previous_summary or '(ninguno)'}"
            ),
            SystemMessage(content=f"Conversación a comprimir:\n{source_transcript}"),
        ]
        try:
            response = await llm.ainvoke(summary_request)
            summary = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            if _summary_confuses_authentication(summary):
                summary = f"Hechos del hilo:\n{fallback_context[-6000:]}"
        except Exception:
            summary = f"Hechos del hilo:\n{fallback_context[-6000:]}"
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_messages],
            "summary": summary,
            "compression_before": len(messages),
            "compression_after": len(recent_messages),
        }

    async def reason_node(state: AgentState) -> dict:
        """Decide una sola acción ReAct o produce la respuesta final."""
        iteration = state.get("iterations", 0)
        max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
        if iteration >= max_iterations:
            stop = (
                "Detuve el ciclo porque alcancé el límite de seguridad. "
                "Puedo intentar de nuevo o escalar el caso a un asesor."
            )
            return {
                "messages": [AIMessage(content=stop)],
                "iterations": iteration,
                "reasoning_trace": [
                    *state.get("reasoning_trace", []),
                    f"[Límite] {stop} (max_iterations={max_iterations})",
                ],
            }

        model_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            _session_status_message(),
            _long_term_memory_message(state),
            SystemMessage(
                content=(
                    "Resumen de la conversación anterior: "
                    f"{state.get('summary', '(todavía no hay resumen)')}"
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
            initial_content.lower().startswith("pensamiento:")
            and "respuesta final:" not in initial_content.lower()
            and not getattr(response, "tool_calls", None)
        )
        if thought_only:
            repaired_response = await llm_with_tools.ainvoke(
                [
                    *model_messages,
                    response,
                    SystemMessage(
                        content=(
                            "Corrección de formato: el Pensamiento anterior declaró que "
                            "falta información. Continúa ahora con UNA llamada estructurada "
                            "a la herramienta necesaria. No repitas el pensamiento ni "
                            "respondas con texto solamente."
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
                    "Pensamiento: todavía falta consultar una fuente solicitada "
                    f"explícitamente; debo ejecutar {tool_name} antes de responder."
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
                f"[Paso {iteration + 1}] {content or 'Pensamiento: se requiere un dato verificable.'} "
                f"→ Acción: {call['name']}({call.get('args', {})})"
            )
        else:
            trace_entry = (
                f"[Paso {iteration + 1}] "
                f"{content or 'Respuesta final: no se generó contenido.'}"
            )
        from agent.review import needs_review
        requires_review = not tool_calls and needs_review(state.get('messages', []), content)
        if not tool_calls and response.response_metadata.get('done_reason') == 'length':
            from agent.review import SAFE_FAILURE
            return {'messages': [AIMessage(content=SAFE_FAILURE)], 'pending_draft': '',
                    'review_required': False, 'review': {'status': 'fallida', 'error': 'BORRADOR_TRUNCADO'},
                    'iterations': iteration + 1}
        if requires_review:
            trace_entry = f"[Paso {iteration + 1}] Borrador reservado para revisión editorial."
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
        """Ejecuta la acción elegida y registra su observación en la traza."""
        result = await tool_node.ainvoke(state)
        tool_messages = list(result.get("messages", []))
        observations = [
            f"Observación: {message.name or 'herramienta'} → {message.content}"
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
                                    f"Revisión editorial: {review['status']}"]}

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
