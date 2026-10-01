"""Banco Andes."""

import httpx
import pytest

from agent import tools as finance_tools
from agent.graph import _tool_error_message
from agent.mcp_client import _server_config


pytestmark = pytest.mark.tools


def test_tools_financieras_tienen_contrato_estable():
    assert {tool.name for tool in finance_tools.get_local_tools()} == {
        "verificar_cliente",
        "consultar_catalogo_creditos",
        "consultar_perfil_financiero",
        "consultar_historial_solicitudes",
    }


def test_timeout_se_convierte_en_error_comprensible(monkeypatch):
    def timeout(*args, **kwargs):
        raise httpx.TimeoutException("demora simulada")

    monkeypatch.setattr(finance_tools.httpx, "request", timeout)

    with pytest.raises(RuntimeError, match="no respondió a tiempo"):
        finance_tools.consultar_catalogo_creditos.invoke({"categoria": "todos"})


def test_error_de_tool_prohibe_inventar_resultados():
    message = _tool_error_message(RuntimeError("servicio no disponible"))
    assert "servicio no disponible" in message
    assert "No inventes" in message


def test_cliente_mcp_arranca_modulo_desde_raiz_del_proyecto():
    config = _server_config()["banco-andes"]
    assert config["transport"] == "stdio"
    assert config["args"] == ["-m", "mcp_server.server"]
    assert config["cwd"]
