'Capability and regression tests.'

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool


pytestmark = pytest.mark.reasoning


def test_prompt_declara_el_ciclo_react_y_una_accion_por_iteracion():
    from agent.graph import SYSTEM_PROMPT

    assert "explicit, auditable ReAct" in SYSTEM_PROMPT
    assert "Thought → Action → Observation" in SYSTEM_PROMPT
    assert "at most one tool call" in SYSTEM_PROMPT


def test_diseno_del_cli_conserva_la_identidad_de_banco_andes():
    from cli.ui import design, prompt_symbol

    assert design()["banner"]["title"] == "Banco Andes"
    assert design()["display"]["agent_title"] == "Banco Andes"
    assert prompt_symbol() == "you ❯ "


def test_paso_autenticado_no_afirma_que_falta_la_sesion(monkeypatch):
    import agent.graph as graph_module
    from agent.session import Session

    monkeypatch.setattr(
        graph_module,
        "active_session",
        lambda: Session(client_id="CLI-2001", token="token-de-test"),
    )

    corrected = graph_module._correct_authenticated_thought(
        "Thought: no session is active and authentication must be requested.",
        [{"name": "remember_financial_preference", "args": {}}],
    )

    assert "an authenticated session is active" in corrected
    assert "no session is active" not in corrected


@pytest.mark.asyncio
async def test_traza_encadena_pensamiento_accion_y_observacion(monkeypatch):
    import agent.graph as graph_module

    @tool
    def consultar_dato() -> str:
        'Support the associated capability or regression scenario.'
        return "confirmed fact"

    class FakeChatModel:
        calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="Thought: I need to query a verifiable fact.",
                    tool_calls=[
                        {
                            "name": "consultar_dato",
                            "args": {},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="Final answer: confirmed fact.")

    monkeypatch.setattr(graph_module, "ChatOllama", lambda **_: FakeChatModel())
    monkeypatch.setattr(
        graph_module,
        "get_settings",
        lambda: SimpleNamespace(
            ollama_model="fake",
            ollama_base_url="http://localhost",
            context_compression_threshold=20,
            context_messages_to_keep=4,
        ),
    )

    graph = graph_module.build_graph(tools=[consultar_dato])
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Query the fact")], "max_iterations": 3}
    )

    assert any(isinstance(message, ToolMessage) for message in result["messages"])
    trace = "\n".join(result["reasoning_trace"])
    assert "Thought:" in trace
    assert "Action: consultar_dato" in trace
    assert "Observation: consultar_dato → confirmed fact" in trace
    assert "Final answer:" in trace


@pytest.mark.asyncio
async def test_react_ejecuta_solo_una_tool_por_iteracion(monkeypatch):
    import agent.graph as graph_module

    llamadas = []

    @tool
    def primera() -> str:
        'Support the associated capability or regression scenario.'
        llamadas.append("primera")
        return "first observation"

    @tool
    def segunda() -> str:
        'Support the associated capability or regression scenario.'
        llamadas.append("segunda")
        return "second observation"

    class FakeChatModel:
        calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="Thought: I must observe one action first.",
                    tool_calls=[
                        {"name": "primera", "args": {}, "id": "call-1", "type": "tool_call"},
                        {"name": "segunda", "args": {}, "id": "call-2", "type": "tool_call"},
                    ],
                )
            return AIMessage(content="Final answer: finished after observing.")

    monkeypatch.setattr(graph_module, "ChatOllama", lambda **_: FakeChatModel())
    monkeypatch.setattr(
        graph_module,
        "get_settings",
        lambda: SimpleNamespace(
            ollama_model="fake",
            ollama_base_url="http://localhost",
            context_compression_threshold=20,
            context_messages_to_keep=4,
        ),
    )

    graph = graph_module.build_graph(tools=[primera, segunda])
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Execute the first action")], "max_iterations": 3}
    )

    assert llamadas == ["primera"]
    assert "Action: primera" in "\n".join(result["reasoning_trace"])
    assert "Action: segunda" not in "\n".join(result["reasoning_trace"])


@pytest.mark.asyncio
async def test_solicitud_multifuente_no_finaliza_hasta_consultar_todo(monkeypatch):
    import agent.graph as graph_module

    llamadas = []

    @tool("query_financial_profile")
    def profile() -> str:
        'Support the associated capability or regression scenario.'
        llamadas.append("profile")
        return "profile observado"

    @tool("query_application_history")
    def history() -> str:
        'Support the associated capability or regression scenario.'
        llamadas.append("history")
        return "observed history"

    @tool("query_credit_catalog")
    def catalog(categoria: str = "all") -> str:
        'Support the associated capability or regression scenario.'
        llamadas.append(f"catalog:{categoria}")
        return "observed catalog"

    class FakeChatModel:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(content="Final answer: tried to finish too early.")

    monkeypatch.setattr(graph_module, "ChatOllama", lambda **_: FakeChatModel())
    monkeypatch.setattr(
        graph_module,
        "get_settings",
        lambda: SimpleNamespace(
            ollama_model="fake",
            ollama_base_url="http://localhost",
            context_compression_threshold=20,
            context_messages_to_keep=4,
        ),
    )

    graph = graph_module.build_graph(tools=[profile, history, catalog])
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Review my profile and application history, and query "
                        "housing loans."
                    )
                )
            ],
            "max_iterations": 6,
        }
    )

    assert llamadas == ["profile", "history", "catalog:housing"]
    trace = "\n".join(result["reasoning_trace"])
    assert "Action: query_financial_profile" in trace
    assert "Action: query_application_history" in trace
    assert "Action: query_credit_catalog" in trace
