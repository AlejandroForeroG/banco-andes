"""Pruebas del ReAct real integrado con el agente acumulativo."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool


pytestmark = pytest.mark.reasoning


def test_prompt_declara_el_ciclo_react_y_una_accion_por_iteracion():
    from agent.graph import SYSTEM_PROMPT

    assert "ReAct explícito" in SYSTEM_PROMPT
    assert "Pensamiento → Acción → Observación" in SYSTEM_PROMPT
    assert "como máximo una llamada" in SYSTEM_PROMPT


def test_diseno_del_cli_conserva_la_identidad_de_banco_andes():
    from cli.ui import design, prompt_symbol

    assert design()["banner"]["title"] == "Banco Andes"
    assert design()["display"]["agent_title"] == "Banco Andes"
    assert prompt_symbol() == "cliente ❯ "


def test_paso_autenticado_no_afirma_que_falta_la_sesion(monkeypatch):
    import agent.graph as graph_module
    from agent.session import Session

    monkeypatch.setattr(
        graph_module,
        "active_session",
        lambda: Session(client_id="CLI-2001", token="token-de-prueba"),
    )

    corrected = graph_module._correct_authenticated_thought(
        "Pensamiento: no hay sesión activa y se debe pedir autenticación.",
        [{"name": "recordar_preferencia_financiera", "args": {}}],
    )

    assert "hay una sesión autenticada activa" in corrected
    assert "no hay sesión" not in corrected


@pytest.mark.asyncio
async def test_traza_encadena_pensamiento_accion_y_observacion(monkeypatch):
    import agent.graph as graph_module

    @tool
    def consultar_dato() -> str:
        """Consulta un dato verificable para la prueba."""
        return "dato confirmado"

    class FakeChatModel:
        calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="Pensamiento: necesito consultar un dato verificable.",
                    tool_calls=[
                        {
                            "name": "consultar_dato",
                            "args": {},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="Respuesta final: dato confirmado.")

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
        {"messages": [HumanMessage(content="Consulta el dato")], "max_iterations": 3}
    )

    assert any(isinstance(message, ToolMessage) for message in result["messages"])
    trace = "\n".join(result["reasoning_trace"])
    assert "Pensamiento:" in trace
    assert "Acción: consultar_dato" in trace
    assert "Observación: consultar_dato → dato confirmado" in trace
    assert "Respuesta final:" in trace


@pytest.mark.asyncio
async def test_react_ejecuta_solo_una_herramienta_por_iteracion(monkeypatch):
    import agent.graph as graph_module

    llamadas = []

    @tool
    def primera() -> str:
        """Primera acción observable."""
        llamadas.append("primera")
        return "primera observación"

    @tool
    def segunda() -> str:
        """Segunda acción que no debe ejecutarse en el mismo paso."""
        llamadas.append("segunda")
        return "segunda observación"

    class FakeChatModel:
        calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="Pensamiento: primero debo observar una sola acción.",
                    tool_calls=[
                        {"name": "primera", "args": {}, "id": "call-1", "type": "tool_call"},
                        {"name": "segunda", "args": {}, "id": "call-2", "type": "tool_call"},
                    ],
                )
            return AIMessage(content="Respuesta final: terminé después de observar.")

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
        {"messages": [HumanMessage(content="Ejecuta la primera acción")], "max_iterations": 3}
    )

    assert llamadas == ["primera"]
    assert "Acción: primera" in "\n".join(result["reasoning_trace"])
    assert "Acción: segunda" not in "\n".join(result["reasoning_trace"])


@pytest.mark.asyncio
async def test_solicitud_multifuente_no_finaliza_hasta_consultar_todo(monkeypatch):
    import agent.graph as graph_module

    llamadas = []

    @tool("consultar_perfil_financiero")
    def perfil() -> str:
        """Consulta el perfil."""
        llamadas.append("perfil")
        return "perfil observado"

    @tool("consultar_historial_solicitudes")
    def historial() -> str:
        """Consulta el historial."""
        llamadas.append("historial")
        return "historial observado"

    @tool("consultar_catalogo_creditos")
    def catalogo(categoria: str = "todos") -> str:
        """Consulta el catálogo."""
        llamadas.append(f"catalogo:{categoria}")
        return "catálogo observado"

    class FakeChatModel:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(content="Respuesta final: intenté terminar antes de tiempo.")

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

    graph = graph_module.build_graph(tools=[perfil, historial, catalogo])
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Revisa mi perfil y mi historial de solicitudes, y consulta "
                        "los créditos de vivienda."
                    )
                )
            ],
            "max_iterations": 6,
        }
    )

    assert llamadas == ["perfil", "historial", "catalogo:vivienda"]
    trace = "\n".join(result["reasoning_trace"])
    assert "Acción: consultar_perfil_financiero" in trace
    assert "Acción: consultar_historial_solicitudes" in trace
    assert "Acción: consultar_catalogo_creditos" in trace
