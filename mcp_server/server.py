'Banco Andes MCP action server.'

from __future__ import annotations

from typing import Annotated, Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from cli.config import get_settings


mcp = FastMCP(
    "banco-andes-actions",
    instructions=(
        "Actions that change financial business state. "
        "Confirm an action only after receiving a successful response."
    ),
)


def _get(path: str) -> list[dict[str, Any]]:
    url = f"{get_settings().api_base_url.rstrip('/')}{path}"
    try:
        response = httpx.get(url, timeout=8.0)
        response.raise_for_status()
        result = response.json()
    except httpx.TimeoutException as exc:
        raise RuntimeError("The Banco Andes API timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Banco Andes rejected the query: HTTP {exc.response.status_code}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError("Could not connect to the Banco Andes API.") from exc
    except ValueError as exc:
        raise RuntimeError("Banco Andes returned an invalid response.") from exc
    if not isinstance(result, list):
        raise RuntimeError("Banco Andes returned an unexpected format.")
    return result


@mcp.tool()
def query_credit_catalog_mcp(
    categoria: Literal[
        "all", "housing", "vehicle", "business", "personal_loan"
    ] = "all",
) -> list[dict[str, Any]]:
    'Expose the local catalog through MCP for equivalent-operation latency comparisons.'
    from agent.tools import filter_credit_catalog

    return filter_credit_catalog(_get("/finanzas/productos"), categoria)


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{get_settings().api_base_url.rstrip('/')}{path}"
    try:
        response = httpx.post(url, json=payload, timeout=8.0)
        response.raise_for_status()
        result = response.json()
    except httpx.TimeoutException as exc:
        raise RuntimeError("The Banco Andes API timed out.") from exc
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        raise RuntimeError(
            f"Banco Andes rejected the submission: {detail or exc.response.status_code}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError("Could not connect to the Banco Andes API.") from exc
    except ValueError as exc:
        raise RuntimeError("Banco Andes returned an invalid response.") from exc

    if not isinstance(result, dict):
        raise RuntimeError("Banco Andes returned an unexpected format.")
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
        raise RuntimeError("Session validation timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            "The session does not authorize actions for this customer."
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError("Could not validate the customer session.") from exc


@mcp.tool()
def register_handoff(
    cliente_id: Annotated[str, Field(description="Authenticated customer associated with the case")],
    token: Annotated[str, Field(description="Customer authenticated session token")],
    intencion: Annotated[str, Field(description="Need expressed by the customer")],
    informacion_recopilada: Annotated[
        dict[str, Any], Field(description="Relevant data already collected")
    ],
    razon: Annotated[str, Field(description="Specific reason for requesting an advisor")],
    solicitud_id: Annotated[
        str | None, Field(description="Related application, if already created")
    ] = None,
) -> dict[str, Any]:
    'Escalate an unresolved case and return its receipt and estimated handling time.'
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
def register_complaint(
    cliente_id: Annotated[str, Field(description="Customer submitting the complaint")],
    token: Annotated[str, Field(description="Customer authenticated session token")],
    tipo: Annotated[str, Field(description="Brief complaint classification")],
    descripcion: Annotated[str, Field(description="Accurate description of the issue")],
) -> dict[str, Any]:
    'Record a complaint and return its destination support channel.'
    _require_authenticated_client(cliente_id, token)
    return _post(
        "/finanzas/quejas",
        {"cliente_id": cliente_id, "tipo": tipo, "descripcion": descripcion},
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
