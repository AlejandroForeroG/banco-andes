"""Pruebas adicionales de memoria aplicadas al caso Banco Andes."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.memory import store as memory_store
from tests.utiles import nombres_de_nodos


pytestmark = pytest.mark.memory


def test_grafo_expone_nodos_de_memoria_y_compresion():
    import agent.graph as graph_module

    nodes = nombres_de_nodos(graph_module.build_graph(tools=[]))
    assert {"memoria_largo_plazo", "comprimir_contexto"} <= nodes


def test_prompt_separa_memoria_del_thread_de_memoria_larga():
    import agent.graph as graph_module

    prompt = graph_module.SYSTEM_PROMPT
    assert "memoria de corto plazo ya disponible" in prompt
    assert "no requieren una sesión autenticada" in prompt
    assert "No uses consultar_memoria_financiera" in prompt
    assert "ambos mecanismos son independientes" in prompt


def test_montos_colombianos_se_normalizan_antes_de_recordarlos():
    import agent.graph as graph_module
    from agent.tools import RecordarPreferenciaInput

    prompt = graph_module.SYSTEM_PROMPT
    description = RecordarPreferenciaInput.model_fields["valor"].description or ""

    assert "millón quinientos" in prompt
    assert "1.500.000 COP" in prompt
    assert "espera confirmación" in prompt
    assert "1.500.000 COP" in description
    assert "No invoque la tool hasta aclarar" in description


def test_sin_login_no_se_declara_perdido_el_historial():
    import agent.graph as graph_module

    message = graph_module._long_term_memory_message({}).content
    assert "no está disponible" in message
    assert "Esto no afecta el historial" in message
    assert "sí están disponibles" in message


def test_memoria_larga_persiste_y_aisla_clientes(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    memory_store.clear_store_cache_for_tests()
    memory_store.save_client_memory(
        "CLI-1001", "objetivo_financiero", "comprar vivienda en dos años"
    )
    memory_store.clear_store_cache_for_tests()

    assert memory_store.load_client_memory("CLI-1001") == {
        "objetivo_financiero": "comprar vivienda en dos años"
    }
    assert memory_store.load_client_memory("CLI-2002") == {}
    memory_store.clear_store_cache_for_tests()


def test_memoria_larga_rechaza_campos_sensibles(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    memory_store.clear_store_cache_for_tests()
    with pytest.raises(ValueError, match="Tipo de memoria no permitido"):
        memory_store.save_client_memory("CLI-1001", "token", "no debe guardarse")
    memory_store.clear_store_cache_for_tests()


@pytest.mark.asyncio
async def test_compresion_reduce_mensajes_y_conserva_resumen(monkeypatch):
    import agent.graph as graph_module

    class FakeChatModel:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages):
            if "Genera únicamente" in str(messages[0].content):
                return AIMessage(content="El cliente busca vivienda y prefiere tasa fija.")
            return AIMessage(content="Respuesta de prueba")

    monkeypatch.setattr(graph_module, "ChatOllama", lambda **_: FakeChatModel())
    monkeypatch.setattr(
        graph_module,
        "get_settings",
        lambda: SimpleNamespace(
            ollama_model="fake",
            ollama_base_url="http://localhost",
            context_compression_threshold=4,
            context_messages_to_keep=2,
        ),
    )

    graph = graph_module.build_graph(tools=[], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "compresion-1"}}

    await graph.ainvoke({"messages": [HumanMessage(content="Busco vivienda")]}, config)
    await graph.ainvoke({"messages": [HumanMessage(content="Prefiero tasa fija")]}, config)
    result = await graph.ainvoke({"messages": [HumanMessage(content="¿Qué sigue?")]}, config)

    assert result["compression_before"] == 5
    assert result["compression_after"] == 1
    assert len(result["messages"]) == 2
    assert "vivienda" in result["summary"]


@pytest.mark.asyncio
async def test_compresion_no_convierte_login_en_perdida_del_historial(monkeypatch):
    import agent.graph as graph_module

    class FakeChatModel:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages):
            if "Genera únicamente" in str(messages[0].content):
                return AIMessage(
                    content="No tengo acceso a la conversación; necesitas iniciar sesión con /login."
                )
            return AIMessage(content="Respuesta de prueba")

    monkeypatch.setattr(graph_module, "ChatOllama", lambda **_: FakeChatModel())
    monkeypatch.setattr(
        graph_module,
        "get_settings",
        lambda: SimpleNamespace(
            ollama_model="fake",
            ollama_base_url="http://localhost",
            context_compression_threshold=4,
            context_messages_to_keep=2,
        ),
    )

    graph = graph_module.build_graph(tools=[], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "compresion-login"}}

    await graph.ainvoke(
        {"messages": [HumanMessage(content="Busco vivienda por 180 millones")]}, config
    )
    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="¿Qué recuerdas?"),
                AIMessage(content="No tengo memoria; inicia sesión con /login."),
            ]
        },
        config,
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="¿Cuál era el monto?")]}, config
    )

    assert "180 millones" in result["summary"]
    assert "/login" not in result["summary"]
    assert "no tengo memoria" not in result["summary"].lower()


@pytest.mark.asyncio
async def test_nuevo_thread_recupera_memoria_del_cliente(tmp_path, monkeypatch):
    import agent.graph as graph_module
    import agent.session as session_module

    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    memory_store.clear_store_cache_for_tests()
    memory_store.save_client_memory(
        "CLI-2002", "categoria_producto", "crédito de vivienda"
    )
    monkeypatch.setattr(
        session_module,
        "_session",
        session_module.Session(client_id="CLI-2002", token="token-no-persistido"),
    )

    graph = graph_module.build_graph(tools=[], checkpointer=InMemorySaver())
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Iniciemos una conversación nueva")],
            "max_iterations": 0,
        },
        {"configurable": {"thread_id": "thread-nuevo"}},
    )

    assert result["long_term_memory"] == {
        "categoria_producto": "crédito de vivienda"
    }
    assert "token" not in str(result["long_term_memory"]).lower()
    memory_store.clear_store_cache_for_tests()
