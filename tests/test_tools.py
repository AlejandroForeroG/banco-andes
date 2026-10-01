'Capability and regression tests.'


import pytest

import contract
from tests.utiles import nombres_de_nodos

pytestmark = pytest.mark.tools


NOMBRES_DE_EJECUCION = {
    "tools", "tool", "tool_node", "toolnode", "tools", "tool",
    "acciones", "action", "act", "actuar", "ejecutar", "ejecucion", "executor",
}


def test_hay_al_menos_tres_tools():
    tools = contract.get_tools()
    assert len(tools) >= 3, (
        f"Expected at least 3 tools; found {len(tools)}"
    )


def test_cada_tool_tiene_descripcion():
    for tool in contract.get_tools():
        assert getattr(tool, "name", ""), "Every tool needs a name"
        assert getattr(tool, "description", "").strip(), (
            f"The tool {tool.name!r} needs a description that the model reads "
            "to decide when to call it"
        )


def test_el_grafo_compila():
    grafo = contract.build_agent_graph()
    assert grafo is not None, "build_agent_graph() did not return a compiled graph"


def test_el_grafo_tiene_un_nodo_que_ejecuta_tools():
    nodos = nombres_de_nodos(contract.build_agent_graph())
    reales = {n for n in nodos if n not in ("__start__", "__end__")}
    assert len(reales) >= 2, (
        f"Expected at least 2 custom reasoning and action nodes; found {sorted(reales)}"
    )
    assert reales & NOMBRES_DE_EJECUCION, (
        f"Tool execution node not recognized. Found nodes: {sorted(reales)}.\n"
        f"Name the node using one of: {sorted(NOMBRES_DE_EJECUCION)}"
    )


async def test_el_servidor_mcp_expone_tools():
    'Verify this behavior with deterministic test fixtures.'
    tools = await contract.get_mcp_tools()
    assert len(tools) >= 1, "The MCP server must expose at least one tool"
    assert all(getattr(t, "name", "") for t in tools), "Every MCP tool needs a name"
