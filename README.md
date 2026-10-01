# Banco Andes

A financial guidance assistant with a local model, persistent memory, and an independent agent that reviews answers before they reach the user. Maintained by [Juan Alejandro Forero Gómez](https://www.linkedin.com/in/alejandroforerog).

## Features

- Authenticated tools for financial profiles, application history, and the product catalog.
- SQLite conversation persistence and customer-specific preferences.
- ReAct reasoning with an explicit iteration limit.
- Independent review against versioned policies: validate, correct, or report insufficient information.
- Advisor handoff through MCP, with consent and a confirmed receipt.
- Optional LangSmith traces with secret redaction.

Customers, products, and policies are synthetic. Review is editorial and does not approve loans. The bundled API is a local demo with simulated authentication and in-memory data; changes are lost on restart. Advisor handling after receipt is outside the prototype's scope.

## Run with Docker

Install Docker Compose and run Ollama on the host. Download the model and start the services:

```bash
ollama pull qwen3:4b-instruct-2507-q8_0
docker compose up --build -d
docker compose exec assistant bank-agent chat
```

Compose builds the demo API, assistant, and reviewer from this repository. It creates a shared signing key and initializes the policy database. Financial guidance does not require an external RAG service or credentials.

In chat, use `/login` with customer ID `DEMO-01` and PIN `2001`. Other synthetic customers are in `demo-api/app/data/finanzas.json`. Ask about the catalog or request guidance, then say `I want to speak to an advisor` to start a handoff. `/help` lists the commands.

```bash
docker compose ps
docker compose exec -T assistant python -m agent.verify_security
docker compose down
```

`down` preserves memory; `down -v` deletes it. Services communicate without publishing host ports. To change the model, copy `.env.example` to `.env` and edit `OLLAMA_MODEL`.

## Local development

With Python 3.11 or later, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
python -m reviewer.init_sources
bank-agent tools --no-mcp
```

Use Compose for the complete flow. A local installation supports module development and tests; it does not automatically start the API or reviewer.

## Design

The CLI opens a session and sends the query to a LangGraph graph. The assistant calls local and MCP tools, loads customer memory, and drafts an answer. The reviewer retrieves criteria from its own SQLite database through read-only queries and returns a signed verdict. The assistant verifies the signature, correlation, coverage, and time budget before accepting the text. The cycle allows two critiques and one refinement; missing evidence or failed review produces a clear limitation.

| Directory | Purpose |
| --- | --- |
| `cli/` | Commands, sessions, and terminal presentation |
| `agent/` | Graph, tools, memory, and review gate |
| `reviewer/` | Reviewer service, protocol, and versioned policies |
| `mcp_server/` | Action and handoff tools |
| `demo-api/` | Local financial domain API |
| `observability/` | Instrumentation and secret redaction |
| `tests/` | Capability tests |

Configure the Banco Andes ASCII banner and colors in `cli_design.toml`. The brand name and canonical business API identifiers retain their original spelling; documentation and user-facing content are in English.

## Tests

```bash
python -m pytest -q
python -m unittest reviewer.test_sources -v
```

The default suite uses test doubles and does not require Ollama or LangSmith. Enable external evaluations explicitly:

```powershell
$env:RUN_OLLAMA_EVAL='1'
python -m pytest tests/test_review_model.py -q
$env:RUN_LANGSMITH_EVAL='1'
python -m pytest tests/test_observability.py -q
```

The second evaluation requires a LangSmith API key and project in the environment. Contract tests alone do not establish the quality of model responses.

## Publish to your account

Create an empty repository named `banco-andes` in your account, then run from this directory:

```bash
git remote add origin https://github.com/YOUR_USERNAME/banco-andes.git
git push -u origin main
```

This copy has its own history and does not modify the original repository. See [credits](NOTICE.md).
