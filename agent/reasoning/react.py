"""Banco Andes."""


def build_reasoning_graph(tools: list | None = None):
    """Compila el grafo ReAct manual con nodos ``reason`` y ``act``."""
    from agent.graph import build_graph

    return build_graph(tools=tools)
