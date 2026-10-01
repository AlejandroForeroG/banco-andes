"""Banco Andes."""


def get_cli():
    """Objeto CLI del proyecto (el grupo `click`/`typer` raíz, sin invocar)."""
    from cli.main import cli

    return cli


def get_settings():
    """Configuración del proyecto.

    Debe exponer al menos los atributos `ollama_model`, `api_base_url` y
    `caso_negocio`, leídos de variables de entorno (nunca escritos a mano en
    el código).
    """
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
    """Store de memoria de largo plazo (persiste ENTRE conversaciones)."""
    from agent.memory.store import build_store as _impl

    return _impl()


def build_reasoning_graph():
    """build reasoning graph."""


    from agent.reasoning.react import build_reasoning_graph as _impl

    return _impl()
