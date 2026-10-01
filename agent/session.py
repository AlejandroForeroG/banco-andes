'Authenticated channel session kept outside model control.'

from __future__ import annotations

from dataclasses import dataclass

import httpx

from cli.config import get_settings


LOGIN_TIMEOUT_SECONDS = 8.0

NO_SESSION = (
    "No authenticated session is available and the data is private. Ask the customer to "
    "sign in through the channel with /login before continuing."
)


@dataclass(frozen=True)
class Session:
    client_id: str
    token: str


_session: Session | None = None


def start_session(identification_number: str, pin: str) -> Session:
    'Exchange customer ID and PIN for a token and activate the session.'
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
            "The Banco Andes API timed out during authentication."
        ) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise RuntimeError(
                "Invalid credentials: check the customer ID and PIN."
            ) from exc
        raise RuntimeError(
            f"The Banco Andes API rejected authentication "
            f"(HTTP {exc.response.status_code})."
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            "Could not connect to the Banco Andes API for authentication."
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            "The Banco Andes API returned a non-JSON login response."
        ) from exc

    if not isinstance(payload, dict) or not payload.get("token") or not payload.get("cliente_id"):
        raise RuntimeError("The Banco Andes API returned an incomplete session.")

    global _session
    _session = Session(client_id=str(payload["cliente_id"]), token=str(payload["token"]))
    return _session


def end_session() -> None:
    global _session
    _session = None


def active_session() -> Session | None:
    return _session


def required_session() -> Session:
    'Return the active session or raise an actionable error.'
    if _session is None:
        raise RuntimeError(NO_SESSION)
    return _session
