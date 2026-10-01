'Banco Andes.'


def build_reasoning_graph(tools: list | None = None):
    'Compile the ReAct graph with reason and act nodes.'
    from agent.graph import build_graph

    return build_graph(tools=tools)
