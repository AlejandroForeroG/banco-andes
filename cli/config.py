"""Banco Andes."""


import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from observability.tracing import proyecto_langsmith, tracing_activo

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_model: str
    ollama_base_url: str
    api_base_url: str
    caso_negocio: str
    langsmith_tracing: bool
    langsmith_project: str

    context_compression_threshold: int
    context_messages_to_keep: int


@lru_cache
def get_settings() -> Settings:
    return Settings(
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        api_base_url=os.getenv("API_BASE_URL", "http://localhost:8080"),
        caso_negocio=os.getenv("CASO_NEGOCIO", "finanzas"),
        langsmith_tracing=tracing_activo(),
        langsmith_project=proyecto_langsmith(),
        context_compression_threshold=max(
            4, int(os.getenv("CONTEXT_COMPRESSION_THRESHOLD", "12"))
        ),
        context_messages_to_keep=max(
            1, int(os.getenv("CONTEXT_MESSAGES_TO_KEEP", "4"))
        ),
    )
