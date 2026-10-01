'Capability and regression tests.'


from pathlib import Path

import pytest
from click.testing import CliRunner

import contract

pytestmark = pytest.mark.cli

RAIZ = Path(__file__).resolve().parent.parent


def test_el_cli_arranca():
    result = CliRunner().invoke(contract.get_cli(), ["--help"])
    assert result.exit_code == 0, (
        "The CLI must respond to --help without errors.\n"
        f"Output:\n{result.output}\n{result.exception}"
    )
    assert result.output.strip(), "The CLI printed no output for --help"


def test_el_cli_expone_comandos():
    'Verify this behavior with deterministic test fixtures.'
    cli = contract.get_cli()
    comandos = getattr(cli, "commands", None)
    assert comandos, "The CLI must expose at least one subcommand, such as chat"


def test_configuracion_disponible():
    settings = contract.get_settings()
    assert settings.ollama_model, "Missing Ollama model setting"
    assert str(settings.api_base_url).startswith("http"), (
        "api_base_url must be a URL configured through API_BASE_URL"
    )
    caso = str(settings.business_domain or "")
    assert caso.strip(), (
        "Missing business_domain: it identifies the configured domain "
        "(configured through BUSINESS_DOMAIN)"
    )


def test_configuracion_por_defecto_sin_env():
    'Verify this behavior with deterministic test fixtures.'
    settings = contract.get_settings()
    assert settings is not None


def test_secretos_protegidos():
    gitignore = (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [linea.strip() for linea in gitignore], (
        ".env must be listed in .gitignore: never commit credentials"
    )
    assert (RAIZ / ".env.example").exists(), (
        "Missing .env.example configuration template"
    )
