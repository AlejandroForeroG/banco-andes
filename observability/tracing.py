'Shared LangSmith trace configuration and redaction.'

from __future__ import annotations

import os
from typing import Any, Callable


SENSITIVE_FIELDS = {
    "authorization",
    "api_key",
    "clave",
    "langchain_api_key",
    "langsmith_api_key",
    "password",
    "pin",
    "secret",
    "token",
}


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def tracing_activo() -> bool:
    'Support historical and current LangSmith environment variable names.'
    enabled = _enabled(
        os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
    )
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    return enabled and bool(api_key)


def proyecto_langsmith() -> str:
    return os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "")


def redactar_datos(value: Any, field_name: str = "") -> Any:
    'Redact secrets by field name without modifying the original object.'
    normalized = field_name.lower().replace("-", "_")
    if normalized in SENSITIVE_FIELDS or normalized.endswith("_token"):
        return "***"
    if isinstance(value, dict):
        return {key: redactar_datos(item, str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [redactar_datos(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redactar_datos(item) for item in value)
    return value


def sanitizar_traza(payload: Any) -> Any:
    'Provide a process_inputs/process_outputs callback for @traceable.'
    return redactar_datos(payload)


def trazable(name: str | None = None, run_type: str = "chain") -> Callable:
    'Instrument functions with secret-redacting tracing.'

    def decorator(func: Callable) -> Callable:
        from langsmith import traceable

        return traceable(
            name=name or func.__name__,
            run_type=run_type,
            process_inputs=sanitizar_traza,
            process_outputs=sanitizar_traza,
        )(func)

    return decorator
