from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def _data_dir() -> Path:
    default = Path(__file__).resolve().parents[2] / ".agent_data"
    path = Path(os.getenv("AGENT_DATA_DIR", str(default))).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path() -> Path:
    return _data_dir() / "checkpoints.sqlite3"


def build_checkpointer(backend: str = "sqlite"):
    if backend == "memory":
        return InMemorySaver()
    if backend != "sqlite":
        raise ValueError(f"Unsupported checkpointer backend: {backend}")

    connection = sqlite3.connect(
        checkpoint_path(),
        check_same_thread=False,
        timeout=30,
    )
    connection.execute("PRAGMA journal_mode=WAL")
    return SqliteSaver(connection)


@asynccontextmanager
async def open_async_checkpointer():
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path())) as saver:
        yield saver
