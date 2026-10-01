'Capability and regression tests.'

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
    assert "available short-term memory" in prompt
    assert "without authentication" in prompt
    assert "Do not use query_financial_memory" in prompt
    assert "does not mean the current thread history was lost" in prompt


def test_montos_colombianos_se_normalizan_antes_de_recordarlos():
    import agent.graph as graph_module
    from agent.tools import RecordarPreferenciaInput

    prompt = graph_module.SYSTEM_PROMPT
    description = RecordarPreferenciaInput.model_fields["valor"].description or ""

    assert "one million five hundred thousand" in prompt
    assert "1.500.000 COP" in prompt
    assert "wait for confirmation" in prompt
    assert "1.500.000 COP" in description
    assert "Do not call the tool before clarifying" in description


def test_sin_login_no_se_declara_perdido_el_historial():
    import agent.graph as graph_module

    message = graph_module._long_term_memory_message({}).content
    assert "is unavailable" in message
    assert "This does not affect the history" in message
    assert "remain available" in message


def test_memoria_larga_persiste_y_aisla_clientes(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    memory_store.clear_store_cache_for_tests()
    memory_store.save_client_memory(
        "CLI-1001", "objetivo_financiero", "buy a home in two years"
    )
    memory_store.clear_store_cache_for_tests()

    assert memory_store.load_client_memory("CLI-1001") == {
        "objetivo_financiero": "buy a home in two years"
    }
    assert memory_store.load_client_memory("CLI-2002") == {}
    memory_store.clear_store_cache_for_tests()


def test_memoria_larga_rechaza_campos_sensibles(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    memory_store.clear_store_cache_for_tests()
    with pytest.raises(ValueError, match="Unsupported memory type"):
        memory_store.save_client_memory("CLI-1001", "token", "must not be saved")
    memory_store.clear_store_cache_for_tests()


@pytest.mark.asyncio
async def test_compresion_reduce_mensajes_y_conserva_resumen(monkeypatch):
    import agent.graph as graph_module

    class FakeChatModel:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages):
            if "Generate only" in str(messages[0].content):
                return AIMessage(content="The customer wants housing and prefers a fixed rate.")
            return AIMessage(content="Test answer")

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

    await graph.ainvoke({"messages": [HumanMessage(content="I want housing")]}, config)
    await graph.ainvoke({"messages": [HumanMessage(content="I prefer a fixed rate")]}, config)
    result = await graph.ainvoke({"messages": [HumanMessage(content="What happens next?")]}, config)

    assert result["compression_before"] == 5
    assert result["compression_after"] == 1
    assert len(result["messages"]) == 2
    assert "housing" in result["summary"]


@pytest.mark.asyncio
async def test_compresion_no_convierte_login_en_perdida_del_historial(monkeypatch):
    import agent.graph as graph_module

    class FakeChatModel:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages):
            if "Generate only" in str(messages[0].content):
                return AIMessage(
                    content="I cannot access the conversation; you need to sign in with /login."
                )
            return AIMessage(content="Test answer")

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
        {"messages": [HumanMessage(content="I want housing for 180 million COP")]}, config
    )
    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="What do you remember?"),
                AIMessage(content="I have no memory; sign in with /login."),
            ]
        },
        config,
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What was the amount?")]}, config
    )

    assert "180 million" in result["summary"]
    assert "/login" not in result["summary"]
    assert "I have no memory" not in result["summary"].lower()


@pytest.mark.asyncio
async def test_nuevo_thread_recupera_memoria_del_cliente(tmp_path, monkeypatch):
    import agent.graph as graph_module
    import agent.session as session_module

    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    memory_store.clear_store_cache_for_tests()
    memory_store.save_client_memory(
        "CLI-2002", "categoria_producto", "housing loan"
    )
    monkeypatch.setattr(
        session_module,
        "_session",
        session_module.Session(client_id="CLI-2002", token="token-no-persistido"),
    )

    graph = graph_module.build_graph(tools=[], checkpointer=InMemorySaver())
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Start a new conversation")],
            "max_iterations": 0,
        },
        {"configurable": {"thread_id": "thread-nuevo"}},
    )

    assert result["long_term_memory"] == {
        "categoria_producto": "housing loan"
    }
    assert "token" not in str(result["long_term_memory"]).lower()
    memory_store.clear_store_cache_for_tests()
