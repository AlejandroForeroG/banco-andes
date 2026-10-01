"""Servidor MCP de acciones de Banco Andes."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from cli.config import get_settings


mcp = FastMCP(
    "banco-andes-acciones",
    instructions=(
        "Acciones que modifican el estado del negocio financiero. "
        "Solo se debe confirmar una acción después de recibir una respuesta exitosa."
    ),
)


def _get(path: str) -> list[dict[str, Any]]:
    url = f"{get_settings().api_base_url.rstrip('/')}{path}"
    try:
        response = httpx.get(url, timeout=8.0)
        response.raise_for_status()
        result = response.json()
    except httpx.TimeoutException as exc:
        raise RuntimeError("La API de Banco Andes no respondió a tiempo.") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Banco Andes rechazó la consulta: HTTP {exc.response.status_code}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError("No fue posible conectar con la API de Banco Andes.") from exc
    except ValueError as exc:
        raise RuntimeError("Banco Andes devolvió una respuesta inválida.") from exc
    if not isinstance(result, list):
        raise RuntimeError("Banco Andes devolvió un formato inesperado.")
    return result


@mcp.tool()
def consultar_catalogo_creditos_mcp(
    categoria: Literal[
        "todos", "vivienda", "vehiculo", "negocio", "libre_inversion"
    ] = "todos",
) -> list[dict[str, Any]]:
    """Espejo MCP del catálogo local para comparar la latencia de la misma operación."""
    from agent.tools import filtrar_catalogo_creditos

    return filtrar_catalogo_creditos(_get("/finanzas/productos"), categoria)


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{get_settings().api_base_url.rstrip('/')}{path}"
    try:
        response = httpx.post(url, json=payload, timeout=8.0)
        response.raise_for_status()
        result = response.json()
    except httpx.TimeoutException as exc:
        raise RuntimeError("La API de Banco Andes no respondió a tiempo.") from exc
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        raise RuntimeError(
            f"Banco Andes rechazó el registro: {detail or exc.response.status_code}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError("No fue posible conectar con la API de Banco Andes.") from exc
    except ValueError as exc:
        raise RuntimeError("Banco Andes devolvió una respuesta inválida.") from exc

    if not isinstance(result, dict):
        raise RuntimeError("Banco Andes devolvió un formato inesperado.")
    return result


def _require_authenticated_client(cliente_id: str, token: str) -> None:
    """Validate that the bearer token belongs to the affected client."""
    url = (
        f"{get_settings().api_base_url.rstrip('/')}"
        f"/finanzas/clientes/{cliente_id}/perfil-financiero"
    )
    try:
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=8.0,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise RuntimeError("No fue posible validar la sesión a tiempo.") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            "La sesión no autoriza acciones sobre el cliente indicado."
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError("No fue posible validar la sesión del cliente.") from exc


@mcp.tool()
def registrar_escalamiento(
    cliente_id: Annotated[str, Field(description="Cliente autenticado asociado al caso")],
    token: Annotated[str, Field(description="Token de la sesión autenticada del cliente")],
    intencion: Annotated[str, Field(description="Necesidad expresada por el cliente")],
    informacion_recopilada: Annotated[
        dict[str, Any], Field(description="Datos relevantes ya recopilados")
    ],
    razon: Annotated[str, Field(description="Motivo concreto para requerir un asesor")],
    solicitud_id: Annotated[
        str | None, Field(description="Solicitud relacionada, si ya existe")
    ] = None,
) -> dict[str, Any]:
    """Escala un caso no resoluble y devuelve la confirmación y tiempo estimado."""
    _require_authenticated_client(cliente_id, token)
    return _post(
        "/finanzas/escalamientos",
        {
            "solicitud_id": solicitud_id,
            "cliente_id": cliente_id,
            "intencion": intencion,
            "informacion_recopilada": informacion_recopilada,
            "razon": razon,
        },
    )


@mcp.tool()
def registrar_queja(
    cliente_id: Annotated[str, Field(description="Cliente que presenta la queja")],
    token: Annotated[str, Field(description="Token de la sesión autenticada del cliente")],
    tipo: Annotated[str, Field(description="Clasificación breve de la queja")],
    descripcion: Annotated[str, Field(description="Descripción fiel del problema")],
) -> dict[str, Any]:
    """Registra una queja y devuelve el canal de atención al que fue redirigida."""
    _require_authenticated_client(cliente_id, token)
    return _post(
        "/finanzas/quejas",
        {"cliente_id": cliente_id, "tipo": tipo, "descripcion": descripcion},
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
