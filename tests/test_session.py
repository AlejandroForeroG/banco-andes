"""Banco Andes."""


import httpx
import pytest
from langchain_core.tools import StructuredTool

from agent import session as session_mod
from agent import tools as finance_tools
from agent.graph import _session_status_message
from agent.mcp_client import with_injected_session

pytestmark = pytest.mark.tools


MCP_SCHEMA = {
    "type": "object",
    "title": "registrar_quejaArguments",
    "properties": {
        "cliente_id": {"type": "string", "description": "Cliente que presenta la queja"},
        "token": {"type": "string", "description": "Token de la sesión autenticada"},
        "tipo": {"type": "string", "description": "Clasificación breve de la queja"},
    },
    "required": ["cliente_id", "token", "tipo"],
}


@pytest.fixture(autouse=True)
def without_previous_session():
    session_mod.end_session()
    yield
    session_mod.end_session()


def _response(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request("POST", "http://localhost:8080/finanzas/auth/login"),
    )


def test_login_leaves_an_active_session_and_keeps_no_pin(monkeypatch):
    sent = {}

    def post(url, json=None, timeout=None):
        sent["url"] = url
        sent["json"] = json
        return _response(200, {"token": "tok-CLI-2001", "tipo": "bearer", "cliente_id": "CLI-2001"})

    monkeypatch.setattr(session_mod.httpx, "post", post)

    session = session_mod.start_session("1030555001", "4821")

    assert sent["url"].endswith("/finanzas/auth/login")
    assert sent["json"] == {"numero_identificacion": "1030555001", "clave": "4821"}
    assert session == session_mod.Session(client_id="CLI-2001", token="tok-CLI-2001")
    assert session_mod.active_session() == session
    assert "4821" not in repr(session_mod.active_session())


def test_invalid_credentials_open_no_session(monkeypatch):
    monkeypatch.setattr(
        session_mod.httpx, "post",
        lambda *a, **k: _response(401, {"detail": "Credenciales inválidas."}),
    )

    with pytest.raises(RuntimeError, match="Credenciales inválidas"):
        session_mod.start_session("1030555001", "0000")

    assert session_mod.active_session() is None


def test_private_tools_ask_the_model_for_no_credentials():
    for tool in (
        finance_tools.consultar_perfil_financiero,
        finance_tools.consultar_historial_solicitudes,
    ):
        assert "token" not in tool.args
        assert "cliente_id" not in tool.args


def test_model_sees_only_that_the_session_is_active(monkeypatch):
    monkeypatch.setattr(
        session_mod,
        "_session",
        session_mod.Session(client_id="CLI-SECRET", token="tok-super-secret"),
    )

    content = _session_status_message().content

    assert "sesión autenticada activa" in content
    assert "CLI-SECRET" not in content
    assert "tok-super-secret" not in content
    assert "sin pedir identificación" in content


def test_model_is_told_to_use_login_when_session_is_inactive():
    assert session_mod.active_session() is None

    content = _session_status_message().content

    assert "no hay una sesión autenticada" in content
    assert "/login" in content
    assert "no le solicites identificación" in content


def test_private_tool_without_session_asks_to_authenticate():
    with pytest.raises(RuntimeError, match="No hay una sesión autenticada"):
        finance_tools.consultar_perfil_financiero.invoke({})


def test_private_tool_queries_the_client_of_the_session(monkeypatch):
    monkeypatch.setattr(
        session_mod.httpx, "post",
        lambda *a, **k: _response(200, {"token": "tok-CLI-2001", "cliente_id": "CLI-2001"}),
    )
    session_mod.start_session("1030555001", "4821")

    call = {}

    def api_request(method, path, *, token=None, json=None):
        call.update(method=method, path=path, token=token)
        return {"id": "CLI-2001", "score_crediticio": 720}

    monkeypatch.setattr(finance_tools, "_api_request", api_request)

    finance_tools.consultar_perfil_financiero.invoke({})

    assert call["path"] == "/finanzas/clientes/CLI-2001/perfil-financiero"
    assert call["token"] == "tok-CLI-2001"


async def test_the_mcp_client_injects_the_session_outside_the_schema(monkeypatch):
    received = {}

    async def call(**kwargs):
        received.update(kwargs)
        return "queja registrada"

    original = StructuredTool(
        name="registrar_queja",
        description="Registra una queja.",
        args_schema=MCP_SCHEMA,
        coroutine=call,
    )

    wrapped = with_injected_session([original])[0]

    assert set(wrapped.args_schema["properties"]) == {"tipo"}
    assert wrapped.args_schema["required"] == ["tipo"]
    assert MCP_SCHEMA["required"] == ["cliente_id", "token", "tipo"], "no debe mutar el original"

    monkeypatch.setattr(
        session_mod.httpx, "post",
        lambda *a, **k: _response(200, {"token": "tok-CLI-2001", "cliente_id": "CLI-2001"}),
    )
    session_mod.start_session("1030555001", "4821")

    await wrapped.ainvoke({"tipo": "cobro duplicado"})

    assert received == {
        "cliente_id": "CLI-2001",
        "token": "tok-CLI-2001",
        "tipo": "cobro duplicado",
    }
