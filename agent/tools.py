"""Herramientas locales de consulta para el caso Banco Andes."""

from __future__ import annotations

from typing import Any, Literal

import httpx
from langchain_core.tools import tool
from langsmith import traceable
from pydantic import BaseModel, Field

from agent.memory.store import load_client_memory, save_client_memory
from agent.session import required_session
from cli.config import get_settings
from observability.tracing import sanitizar_traza


REQUEST_TIMEOUT_SECONDS = 8.0


@traceable(
    name="api_empresarial_request",
    run_type="chain",
    process_inputs=sanitizar_traza,
    process_outputs=sanitizar_traza,
)
def _api_request(
    method: str,
    path: str,
    *,
    token: str | None = None,
    json: dict[str, Any] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Call the business API and normalize failures into useful tool errors."""
    base_url = get_settings().api_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"} if token else None

    try:
        response = httpx.request(
            method,
            f"{base_url}{path}",
            headers=headers,
            json=json,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "La API de Banco Andes no respondió a tiempo. Intente nuevamente."
        ) from exc
    except httpx.HTTPStatusError as exc:
        detail: Any = None
        try:
            detail = exc.response.json().get("detail")
        except (ValueError, AttributeError):
            pass
        message = detail or f"respuesta HTTP {exc.response.status_code}"
        raise RuntimeError(f"La API de Banco Andes rechazó la consulta: {message}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            "No fue posible conectar con la API de Banco Andes."
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            "La API de Banco Andes devolvió una respuesta que no es JSON válido."
        ) from exc

    if not isinstance(payload, (dict, list)):
        raise RuntimeError("La API de Banco Andes devolvió un formato inesperado.")
    return payload


class VerificarClienteInput(BaseModel):
    numero_identificacion: str = Field(
        description="Número de identificación del cliente, sin espacios ni separadores."
    )


@tool("verificar_cliente", args_schema=VerificarClienteInput)
def verificar_cliente(numero_identificacion: str) -> dict[str, bool]:
    """Verifica públicamente una identificación cuando el usuario lo pide explícitamente.

    No se usa para consultar datos privados ni como paso previo si el canal ya tiene sesión.
    """
    payload = _api_request(
        "GET", f"/finanzas/clientes/existe/{numero_identificacion.strip()}"
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("existe"), bool):
        raise RuntimeError("La API devolvió un resultado inesperado al verificar el cliente.")
    return {"existe": payload["existe"]}


class CatalogoCreditosInput(BaseModel):
    categoria: Literal["todos", "vivienda", "vehiculo", "negocio", "libre_inversion"] = Field(
        default="todos",
        description="Categoría ya inferida de la necesidad del cliente; use 'todos' si no es clara.",
    )


def filtrar_catalogo_creditos(
    payload: list[dict[str, Any]],
    categoria: Literal["todos", "vivienda", "vehiculo", "negocio", "libre_inversion"],
) -> list[dict[str, Any]]:
    """Aplica el mismo filtro tanto a la tool local como a su espejo MCP."""
    product_ids = {
        "todos": None,
        "vivienda": {"PRD-HIP", "PRD-LEA", "PRD-CON"},
        "vehiculo": {"PRD-VEH"},
        "negocio": {"PRD-PYM", "PRD-LIB"},
        "libre_inversion": {"PRD-LIB"},
    }[categoria]
    if product_ids is None:
        return payload
    return [product for product in payload if product.get("id") in product_ids]


@tool("consultar_catalogo_creditos", args_schema=CatalogoCreditosInput)
def consultar_catalogo_creditos(
    categoria: Literal["todos", "vivienda", "vehiculo", "negocio", "libre_inversion"] = "todos",
) -> list[dict[str, Any]]:
    """Consulta productos, montos, plazos, tasas y requisitos públicos por categoría."""
    payload = _api_request("GET", "/finanzas/productos")
    if not isinstance(payload, list):
        raise RuntimeError("La API devolvió un catálogo de productos inesperado.")
    return filtrar_catalogo_creditos(payload, categoria)


@tool("consultar_perfil_financiero")
def consultar_perfil_financiero() -> dict[str, Any]:
    """Consulta score, ingresos, endeudamiento y productos activos del cliente autenticado.

    No recibe parámetros: opera sobre la sesión vigente del canal.
    """
    session = required_session()
    payload = _api_request(
        "GET",
        f"/finanzas/clientes/{session.client_id}/perfil-financiero",
        token=session.token,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("La API devolvió un perfil financiero inesperado.")
    return payload


@tool("consultar_historial_solicitudes")
def consultar_historial_solicitudes() -> list[dict[str, Any]]:
    """Consulta las solicitudes de crédito anteriores del cliente autenticado.

    No recibe parámetros: opera sobre la sesión vigente del canal.
    """
    session = required_session()
    payload = _api_request(
        "GET",
        f"/finanzas/clientes/{session.client_id}/historial-solicitudes",
        token=session.token,
    )
    if not isinstance(payload, list):
        raise RuntimeError("La API devolvió un historial de solicitudes inesperado.")
    return payload


class RecordarPreferenciaInput(BaseModel):
    tipo: Literal[
        "objetivo_financiero",
        "categoria_producto",
        "monto_objetivo",
        "cuota_maxima_declarada",
        "plazo_preferido",
        "horizonte",
        "tolerancia_riesgo",
        "preferencia_contacto",
        "hito_gestion",
    ] = Field(
        description=(
            "Tipo de preferencia estable o hito confirmado que debe recordarse. No use "
            "esta tool para ingresos, score, deudas, claves, tokens ni números de cuenta."
        )
    )
    valor: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Preferencia explícita del cliente en forma breve y sin ambigüedad. Para "
            "montos use el valor normalizado en COP y su periodicidad si fue indicada: "
            "'millón quinientos' equivale a '1.500.000 COP', no a '500 millones'. "
            "No invoque la tool hasta aclarar un monto que admita varias interpretaciones."
        ),
    )


@tool("recordar_preferencia_financiera", args_schema=RecordarPreferenciaInput)
def recordar_preferencia_financiera(tipo: str, valor: str) -> dict[str, Any]:
    """Guarda una preferencia financiera estable para conversaciones futuras.

    Úsala cuando el cliente autenticado exprese explícitamente algo que desea
    conservar, o cuando otra tool confirme un hito como un escalamiento. Los
    datos financieros variables se consultan siempre en la API empresarial.
    """
    session = required_session()
    profile = save_client_memory(session.client_id, tipo, valor)
    return {"guardado": True, "tipo": tipo, "valor": profile[tipo]}


@tool("consultar_memoria_financiera")
def consultar_memoria_financiera() -> dict[str, str]:
    """Recupera objetivos y preferencias recordados del cliente autenticado."""
    session = required_session()
    return load_client_memory(session.client_id)


def get_local_tools() -> list:
    """Return the read-only tools available without starting the MCP server."""
    return [
        verificar_cliente,
        consultar_catalogo_creditos,
        consultar_perfil_financiero,
        consultar_historial_solicitudes,
    ]


def get_memory_tools() -> list:
    """get memory tools."""
    return [recordar_preferencia_financiera, consultar_memoria_financiera]
