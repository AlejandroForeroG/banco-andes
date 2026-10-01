'Banco Andes.'


def get_cli():
    'Return the root Click command without invoking it.'
    from cli.main import cli

    return cli


def get_settings():
    'Read model, business API, and domain settings from environment variables.'
    from cli.config import get_settings as _impl

    return _impl()


def get_tools():
    """get tools."""


    from agent.tools import get_local_tools

    return get_local_tools()


def build_agent_graph(checkpointer=None):
    """build agent graph."""


    from agent.graph import build_graph

    return build_graph(checkpointer=checkpointer)


async def get_mcp_tools():
    """get mcp tools."""


    from agent.mcp_client import load_mcp_tools

    return await load_mcp_tools()


def build_checkpointer():
    """build checkpointer."""


    from agent.memory.checkpointer import build_checkpointer as _impl

    return _impl()


def build_store():
    'Build the persistent cross-conversation memory store.'
    from agent.memory.store import build_store as _impl

    return _impl()


def build_reasoning_graph():
    'Compile the ReAct graph with reason and act nodes.'


    from agent.reasoning.react import build_reasoning_graph as _impl

    return _impl()
