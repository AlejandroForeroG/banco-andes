from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from threading import Lock

from langgraph.store.sqlite import SqliteStore


PROFILE_KEY = "perfil_financiero_recordado"
ALLOWED_MEMORY_FIELDS = {
    "objetivo_financiero",
    "categoria_producto",
    "monto_objetivo",
    "cuota_maxima_declarada",
    "plazo_preferido",
    "horizonte",
    "tolerancia_riesgo",
    "preferencia_contacto",
    "hito_gestion",
}
_store: SqliteStore | None = None
_store_lock = Lock()


def _store_path() -> Path:
    default_dir = Path(__file__).resolve().parents[2] / ".agent_data"
    data_dir = Path(os.getenv("AGENT_DATA_DIR", str(default_dir))).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "long_term_memory.sqlite3"


def build_store() -> SqliteStore:
    connection = sqlite3.connect(
        _store_path(),
        check_same_thread=False,
        isolation_level=None,
        timeout=30,
    )
    connection.execute("PRAGMA journal_mode=WAL")
    store = SqliteStore(connection)
    store.setup()
    return store


def get_store() -> SqliteStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = build_store()
    return _store


def _namespace(client_id: str) -> tuple[str, ...]:
    return ("banco-andes", "clientes", client_id)


def load_client_memory(client_id: str) -> dict[str, str]:
    item = get_store().get(_namespace(client_id), PROFILE_KEY)
    if item is None:
        return {}
    return {str(key): str(value) for key, value in item.value.items()}


def save_client_memory(client_id: str, field: str, value: str) -> dict[str, str]:
    if field not in ALLOWED_MEMORY_FIELDS:
        raise ValueError(f"Tipo de memoria no permitido: {field}")
    profile = load_client_memory(client_id)
    profile[field] = value.strip()
    get_store().put(_namespace(client_id), PROFILE_KEY, profile)
    return profile


def clear_store_cache_for_tests() -> None:
    global _store
    if _store is not None:
        connection = getattr(_store, "conn", None)
        if connection is not None:
            connection.close()
    _store = None
