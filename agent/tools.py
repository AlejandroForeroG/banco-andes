'Local query tools for Banco Andes.'

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
            "The Banco Andes API timed out. Please try again."
        ) from exc
    except httpx.HTTPStatusError as exc:
        detail: Any = None
        try:
            detail = exc.response.json().get("detail")
        except (ValueError, AttributeError):
            pass
        message = detail or f"HTTP response {exc.response.status_code}"
        raise RuntimeError(f"The Banco Andes API rejected the query: {message}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            "Could not connect to the Banco Andes API."
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            "The Banco Andes API returned invalid JSON."
        ) from exc

    if not isinstance(payload, (dict, list)):
        raise RuntimeError("The Banco Andes API returned an unexpected format.")
    return payload


class VerificarClienteInput(BaseModel):
    numero_identificacion: str = Field(
        description="Customer ID without spaces or separators."
    )


@tool("verify_customer", args_schema=VerificarClienteInput)
def verify_customer(numero_identificacion: str) -> dict[str, bool]:
    'Publicly verify a customer ID only when explicitly requested. Do not call before private queries or when the channel already has a session.'
    payload = _api_request(
        "GET", f"/finanzas/clientes/existe/{numero_identificacion.strip()}"
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("existe"), bool):
        raise RuntimeError("The API returned an unexpected customer verification result.")
    return {"existe": payload["existe"]}


class CatalogoCreditosInput(BaseModel):
    categoria: Literal["all", "housing", "vehicle", "business", "personal_loan"] = Field(
        default="all",
        description="Category inferred from the customer need; use 'all' when unclear.",
    )


def filter_credit_catalog(
    payload: list[dict[str, Any]],
    categoria: Literal["all", "housing", "vehicle", "business", "personal_loan"],
) -> list[dict[str, Any]]:
    'Apply the same category filter to local and MCP catalog tools.'
    product_ids = {
        "all": None,
        "housing": {"PRD-HIP", "PRD-LEA", "PRD-CON"},
        "vehicle": {"PRD-VEH"},
        "business": {"PRD-PYM", "PRD-LIB"},
        "personal_loan": {"PRD-LIB"},
    }[categoria]
    if product_ids is None:
        return payload
    return [product for product in payload if product.get("id") in product_ids]


@tool("query_credit_catalog", args_schema=CatalogoCreditosInput)
def query_credit_catalog(
    categoria: Literal["all", "housing", "vehicle", "business", "personal_loan"] = "all",
) -> list[dict[str, Any]]:
    'Query public products, amounts, terms, rates, and requirements by category.'
    payload = _api_request("GET", "/finanzas/productos")
    if not isinstance(payload, list):
        raise RuntimeError("The API returned an unexpected product catalog.")
    return filter_credit_catalog(payload, categoria)


@tool("query_financial_profile")
def query_financial_profile() -> dict[str, Any]:
    'Query the authenticated customer score, income, debt, and active products. Takes no arguments; uses the active channel session.'
    session = required_session()
    payload = _api_request(
        "GET",
        f"/finanzas/clientes/{session.client_id}/perfil-financiero",
        token=session.token,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("The API returned an unexpected financial profile.")
    return payload


@tool("query_application_history")
def query_application_history() -> list[dict[str, Any]]:
    'Query past loan applications for the authenticated customer. Takes no arguments; uses the active channel session.'
    session = required_session()
    payload = _api_request(
        "GET",
        f"/finanzas/clientes/{session.client_id}/historial-solicitudes",
        token=session.token,
    )
    if not isinstance(payload, list):
        raise RuntimeError("The API returned an unexpected application history.")
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
            "Stable preference type or confirmed milestone to remember. Do not use "
            "this tool for income, scores, debts, PINs, tokens, or account numbers."
        )
    )
    valor: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Explicit customer preference stated briefly and unambiguously. Normalize "
            "amounts in COP and include the stated payment period: "
            "one million five hundred thousand means 1.500.000 COP, never 500 million COP. "
            "Do not call the tool before clarifying an ambiguous amount."
        ),
    )


@tool("remember_financial_preference", args_schema=RecordarPreferenciaInput)
def remember_financial_preference(tipo: str, valor: str) -> dict[str, Any]:
    'Save a stable financial preference for future conversations. Use only for explicit customer requests or tool-confirmed milestones. Query changing financial data through the business API.'
    session = required_session()
    profile = save_client_memory(session.client_id, tipo, valor)
    return {"guardado": True, "tipo": tipo, "valor": profile[tipo]}


@tool("query_financial_memory")
def query_financial_memory() -> dict[str, str]:
    'Retrieve remembered goals and preferences for the authenticated customer.'
    session = required_session()
    return load_client_memory(session.client_id)


def get_local_tools() -> list:
    """Return the read-only tools available without starting the MCP server."""
    return [
        verify_customer,
        query_credit_catalog,
        query_financial_profile,
        query_application_history,
    ]


def get_memory_tools() -> list:
    """get memory tools."""
    return [remember_financial_preference, query_financial_memory]
