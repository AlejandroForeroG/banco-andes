"""Sesión autenticada del canal digital, fuera del alcance del agente."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from cli.config import get_settings


LOGIN_TIMEOUT_SECONDS = 8.0

NO_SESSION = (
    "No hay una sesión autenticada y el dato es privado. Pida al cliente que "
    "inicie sesión en el canal con /login antes de continuar."
)


@dataclass(frozen=True)
class Session:
    client_id: str
    token: str


_session: Session | None = None


def start_session(identification_number: str, pin: str) -> Session:
    """Cambia identificación y clave por un token, y deja la sesión activa."""
    base_url = get_settings().api_base_url.rstrip("/")
    try:
        response = httpx.post(
            f"{base_url}/finanzas/auth/login",
            json={
                "numero_identificacion": identification_number.strip(),
                "clave": pin.strip(),
            },
            timeout=LOGIN_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "La API de Banco Andes no respondió a tiempo al autenticar."
        ) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise RuntimeError(
                "Credenciales inválidas: revise la identificación y la clave."
            ) from exc
        raise RuntimeError(
            f"La API de Banco Andes rechazó la autenticación "
            f"(HTTP {exc.response.status_code})."
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            "No fue posible conectar con la API de Banco Andes para autenticar."
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            "La API de Banco Andes devolvió una respuesta de login que no es JSON."
        ) from exc

    if not isinstance(payload, dict) or not payload.get("token") or not payload.get("cliente_id"):
        raise RuntimeError("La API de Banco Andes devolvió una sesión incompleta.")

    global _session
    _session = Session(client_id=str(payload["cliente_id"]), token=str(payload["token"]))
    return _session


def end_session() -> None:
    global _session
    _session = None


def active_session() -> Session | None:
    return _session


def required_session() -> Session:
    """Sesión vigente, o un error accionable si no hay ninguna."""
    if _session is None:
        raise RuntimeError(NO_SESSION)
    return _session
