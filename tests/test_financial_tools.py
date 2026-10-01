'Capability and regression tests.'

import httpx
import pytest

from agent import tools as finance_tools
from agent.graph import _tool_error_message
from agent.mcp_client import _server_config


pytestmark = pytest.mark.tools


def test_tools_financieras_tienen_contrato_estable():
    assert {tool.name for tool in finance_tools.get_local_tools()} == {
        "verify_customer",
        "query_credit_catalog",
        "query_financial_profile",
        "query_application_history",
    }


def test_timeout_se_convierte_en_error_comprensible(monkeypatch):
    def timeout(*args, **kwargs):
        raise httpx.TimeoutException("demora simulada")

    monkeypatch.setattr(finance_tools.httpx, "request", timeout)

    with pytest.raises(RuntimeError, match="timed out"):
        finance_tools.query_credit_catalog.invoke({"categoria": "all"})


def test_error_de_tool_prohibe_inventar_resultados():
    message = _tool_error_message(RuntimeError("service unavailable"))
    assert "service unavailable" in message
    assert "Do not invent" in message


def test_cliente_mcp_arranca_modulo_desde_raiz_del_proyecto():
    config = _server_config()["banco-andes"]
    assert config["transport"] == "stdio"
    assert config["args"] == ["-m", "mcp_server.server"]
    assert config["cwd"]
