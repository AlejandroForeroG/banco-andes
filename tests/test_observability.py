'Capability and regression tests.'


import os

import pytest

from tests.utiles import nombres_de_decoradores

pytestmark = pytest.mark.observability
requires_langsmith = pytest.mark.skipif(
    os.getenv("RUN_LANGSMITH_EVAL") != "1",
    reason="Optional integration: RUN_LANGSMITH_EVAL=1",
)


_VARIABLES_API_KEY = ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY")
_VARIABLES_PROYECTO = ("LANGCHAIN_PROJECT", "LANGSMITH_PROJECT")


DECORADORES_DE_TRAZA = {"traceable", "trazable"}


def _primera(variables):
    for nombre in variables:
        valor = os.getenv(nombre)
        if valor:
            return valor
    return None


@requires_langsmith
def test_api_key_configurada():
    assert _primera(_VARIABLES_API_KEY), (
        "Missing LangSmith API key.\n"
        "GitHub: Settings → Secrets and variables → Actions → New repository secret,\n"
        "named LANGCHAIN_API_KEY with your own LangSmith API key."
    )


@requires_langsmith
def test_proyecto_configurado():
    proyecto = _primera(_VARIABLES_PROYECTO)
    assert proyecto, (
        "Missing LangSmith project name.\n"
        "GitHub: Settings → Secrets and variables → Actions → Variables →\n"
        "New repository variable, named LANGCHAIN_PROJECT with value\n"
        "banco-andes."
    )
    assert proyecto.strip() == proyecto and " " not in proyecto, (
        f"The project name must not contain spaces: {proyecto!r}"
    )


@requires_langsmith
def test_la_credencial_es_valida():
    'Verify this behavior with deterministic test fixtures.'


    from langsmith import Client

    cliente = Client(api_key=_primera(_VARIABLES_API_KEY))
    try:
        list(cliente.list_projects(limit=1))
    except Exception as exc:
        texto = str(exc)
        if any(codigo in texto for codigo in ("401", "403", "Unauthorized", "Forbidden")):
            pytest.fail(
                "LangSmith rejected the credential. Check that "
                "LANGCHAIN_API_KEY contains the complete, correct key "
                f"(without spaces or quotes).\nDetails: {texto}"
            )
        pytest.skip(f"LangSmith did not respond. Details: {texto}")


def test_hay_instrumentacion_propia():
    'Verify this behavior with deterministic test fixtures.'
    decoradores = nombres_de_decoradores()
    assert decoradores & DECORADORES_DE_TRAZA, (
        "No function decorated with @traceable was found.\n"
        "Instrument at least one project function, such as the one that "
        "prepares context or chooses a tool.\n"
        f"Decorators found in the project: {sorted(decoradores) or 'ninguno'}"
    )
