"""Banco Andes."""


import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DOMAINS = ("finanzas",)

DB: dict[str, dict] = {}


def load_seed() -> None:
    for domain in DOMAINS:
        DB[domain] = json.loads((DATA_DIR / f"{domain}.json").read_text(encoding="utf-8"))


def reset() -> None:
    load_seed()


def next_id(items: list[dict], prefix: str) -> str:
    """Genera un id consecutivo tipo PREFIX-001 sobre una colección."""
    numbers = []
    for item in items:
        try:
            numbers.append(int(str(item["id"]).rsplit("-", 1)[-1]))
        except (KeyError, ValueError):
            continue
    return f"{prefix}-{(max(numbers) + 1 if numbers else 1):03d}"
