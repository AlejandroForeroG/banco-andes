'MCP client for discovering Banco Andes actions.'

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.session import required_session


SESSION_FIELDS = ("cliente_id", "token")


def _server_config() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    return {
        "banco-andes": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "mcp_server.server"],
            "cwd": str(project_root),
            "env": {
                "API_BASE_URL": os.getenv("API_BASE_URL", "http://localhost:8080"),

                "LANGCHAIN_TRACING_V2": "false",
                "LANGSMITH_TRACING": "false",
            },
        }
    }


def _schema_without_credentials(schema: dict) -> dict:
    """Copy of the JSON schema with the session fields removed."""
    clean = copy.deepcopy(schema)
    for field in SESSION_FIELDS:
        clean.get("properties", {}).pop(field, None)
    if "required" in clean:
        clean["required"] = [f for f in clean["required"] if f not in SESSION_FIELDS]
    return clean


def with_injected_session(tools: list) -> list:
    """Move the credentials from the model's arguments to the active session."""
    wrapped = []
    for tool in tools:
        schema = getattr(tool, "args_schema", None)
        if not isinstance(schema, dict):
            wrapped.append(tool)
            continue
        present = [f for f in SESSION_FIELDS if f in schema.get("properties", {})]
        if not present or tool.coroutine is None:
            wrapped.append(tool)
            continue

        async def call_with_session(_call=tool.coroutine, _fields=tuple(present), **kwargs):
            session = required_session()
            credentials = {"cliente_id": session.client_id, "token": session.token}
            return await _call(**{f: credentials[f] for f in _fields}, **kwargs)

        wrapped.append(
            tool.model_copy(
                update={
                    "args_schema": _schema_without_credentials(schema),
                    "coroutine": call_with_session,
                }
            )
        )
    return wrapped


async def load_mcp_tools() -> list:
    """Start the stdio server and discover its LangChain-compatible tools."""
    client = MultiServerMCPClient(_server_config(), handle_tool_errors=True)
    tools = await client.get_tools(server_name="banco-andes")
    return with_injected_session(tools)


async def load_mcp_tools_safe() -> tuple[list, str | None]:
    """Return local operation even when the MCP process is unavailable."""
    try:
        return await load_mcp_tools(), None
    except Exception as exc:
        return [], str(exc)
