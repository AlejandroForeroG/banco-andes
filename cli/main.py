'Banco Andes.'


import asyncio
import json
import sys
import uuid
from importlib.metadata import PackageNotFoundError, version as pkg_version

import click
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from cli import ui
from cli.config import get_settings


@click.group()
def cli():
    'Banco Andes — financial guidance.'


@cli.command()
def version():
    'Show the installed project version.'
    try:
        click.echo(pkg_version("banco-andes"))
    except PackageNotFoundError:
        click.echo("0.1.0 (not installed; run pip install -e .)")


@cli.command()
def config():
    'Show current assistant settings.'
    settings = get_settings()
    click.echo(f"Ollama model     : {settings.ollama_model}")
    click.echo(f"Ollama URL       : {settings.ollama_base_url}")
    click.echo(f"Business API     : {settings.api_base_url}")
    click.echo(f"Business domain  : {settings.business_domain}")
    click.echo(f"LangSmith enabled: {settings.langsmith_tracing}")


async def _reunir_tools(usar_mcp: bool) -> list:
    from agent.mcp_client import load_mcp_tools_safe
    from agent.tools import get_local_tools, get_memory_tools

    try:
        tools = get_local_tools() + get_memory_tools()
    except NotImplementedError:

        tools = []
    if usar_mcp:
        mcp_tools, error_mcp = await load_mcp_tools_safe()
        if error_mcp:
            ui.aviso(f"MCP server unavailable ({error_mcp}). Continuing with local tools only.")
        tools = tools + mcp_tools
    return tools


@cli.command()
@click.option("--no-mcp", "sin_mcp", is_flag=True, help="Skip the MCP server and use local tools only.")
def tools(sin_mcp: bool):
    'List available assistant tools.'
    tools = asyncio.run(_reunir_tools(usar_mcp=not sin_mcp))
    if not tools:
        ui.aviso("No tools are available. Check the configuration.")
        return
    for tool in tools:
        descripcion = (tool.description or "").split("\n")[0]
        click.echo(f"- {tool.name}: {descripcion}")


def _ask_pin() -> str:
    'Read a hidden PIN from the terminal or stdin in scripted execution.'
    if sys.stdin.isatty():
        return click.prompt("PIN", hide_input=True)
    pin = sys.stdin.readline().strip()
    if not pin:
        raise click.ClickException("No PIN was received on stdin.")
    return pin


@cli.command(name="call-tool")
@click.argument("nombre", metavar="TOOL")
@click.argument("argumentos", default="{}", metavar="ARGUMENTS")
@click.option("--no-mcp", "sin_mcp", is_flag=True, help="Search local tools only.")
@click.option(
    "--customer-id",
    "identification",
    help="Authenticate before calling (prompts for a PIN); required for private tools.",
)
def probar_tool(nombre: str, argumentos: str, sin_mcp: bool, identification: str | None):
    'Call a tool directly without invoking the model.'
    if identification:
        from agent.session import start_session

        pin = _ask_pin()
        try:
            session = start_session(identification, pin)
        except RuntimeError as exc:
            raise click.ClickException(str(exc))
        ui.info(f"Signed in as {session.client_id}.")

    async def _run():
        tools = await _reunir_tools(usar_mcp=not sin_mcp)
        tool = next((t for t in tools if t.name == nombre), None)
        if tool is None:
            available = ", ".join(t.name for t in tools)
            raise click.ClickException(f"Tool '{nombre}' does not exist. Available: {available}")
        try:
            args = json.loads(argumentos)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid arguments (expected JSON): {exc}")
        try:
            result = await tool.ainvoke(args)
        except Exception as exc:
            raise click.ClickException(str(exc))
        click.echo(result if isinstance(result, str) else json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(_run())


@cli.command()
@click.option("--no-mcp", "sin_mcp", is_flag=True, help="Skip the MCP server and use local tools only.")
@click.option(
    "--thread-id",
    help="Conversation ID to resume; creates a new conversation when omitted.",
)
def chat(sin_mcp: bool, thread_id: str | None):
    'Start an interactive conversation with the assistant.'
    asyncio.run(_chat(usar_mcp=not sin_mcp, initial_thread_id=thread_id))


async def _start_session() -> None:
    from agent.session import start_session

    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout

    prompt_session = PromptSession()
    with patch_stdout():
        identification = (await prompt_session.prompt_async("customer ID ❯ ")).strip()
        if not identification:
            ui.aviso("Authentication cancelled.")
            return
        pin = (await prompt_session.prompt_async("PIN ❯ ", is_password=True)).strip()
    try:
        session = start_session(identification, pin)
    except RuntimeError as exc:
        ui.error(str(exc))
        return
    ui.info(f"Signed in as {session.client_id}.")


async def _chat(usar_mcp: bool, initial_thread_id: str | None = None):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout

    from agent.graph import build_graph
    from agent.memory.checkpointer import open_async_checkpointer
    from agent.session import end_session

    settings = get_settings()
    ui.banner(settings)

    tools = await _reunir_tools(usar_mcp)
    from agent.review import listener
    listener.set(lambda event: ui.info('Review: ' + json.dumps(event, ensure_ascii=False)))
    current_thread_id = (initial_thread_id or str(uuid.uuid4())).strip()
    turno = 0
    last_review = {}
    prompt_session = PromptSession()

    async with open_async_checkpointer() as checkpointer:
        graph = build_graph(tools=tools, checkpointer=checkpointer)
        ui.info(f"Conversation: {current_thread_id}")

        while True:
            try:
                with patch_stdout():
                    texto = (await prompt_session.prompt_async(ui.prompt_symbol())).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not texto:
                continue
            if texto in ("/exit", "/exit"):
                break
            if texto == "/new":
                current_thread_id = str(uuid.uuid4())
                turno = 0
                last_review = {}
                ui.aviso(f"New conversation: {current_thread_id}")
                continue
            if texto == "/thread":
                ui.info(f"Current conversation: {current_thread_id}")
                continue
            if texto == "/status":
                state = await graph.aget_state(
                    {"configurable": {"thread_id": current_thread_id}}
                )
                values = state.values
                ui.info(
                    f"State: {len(values.get('messages', []))} messages · "
                    f"summary: {'yes' if values.get('summary') else 'no'} · "
                    f"remembered preferences: {len(values.get('long_term_memory', {}))}"
                )
                continue
            if texto == "/memory":
                from agent.memory.store import load_client_memory
                from agent.session import active_session

                active = active_session()
                if active is None:
                    ui.aviso("Sign in with /login to view your saved preferences.")
                else:
                    remembered = load_client_memory(active.client_id)
                    if remembered:
                        ui.info(json.dumps(remembered, ensure_ascii=False))
                    else:
                        ui.aviso("No preferences are saved for this customer.")
                continue
            if texto == "/login":
                last_review = {}
                await _start_session()
                continue
            wants_advisor = texto == '/advisor'
            if not texto.startswith('/'):
                from cli.intent import requests_advisor
                try:
                    wants_advisor = await requests_advisor(texto)
                except RuntimeError as exc:
                    ui.aviso(str(exc))
                    continue
            if wants_advisor:
                from agent.human import transfer, transfer_summary
                try:
                    if not last_review:
                        last_review = {'status': 'requested', 'error': 'USER_REQUEST'}
                    summary = transfer_summary(last_review)
                    ui.info('Advisor handoff preview: ' + json.dumps(summary, ensure_ascii=False))
                    with patch_stdout():
                        accepted = (await prompt_session.prompt_async('Send this summary to an advisor? [yes/no] ')).strip().lower() in ('y', 'yes')
                    result = await transfer(last_review, accepted, tools)
                    ui.info(json.dumps(result, ensure_ascii=False))
                    if result.get('transferred'):
                        last_review = {}
                except (ValueError, RuntimeError) as exc:
                    ui.aviso(str(exc))
                continue
            if texto == "/logout":
                last_review = {}
                end_session()
                ui.info("Signed out. The assistant can no longer access private data.")
                continue
            if texto == "/help":
                ui.console.print(
                    "[dim]/login signs in · /logout signs out · "
                    "/thread shows the conversation ID · /status inspects context · "
                    "/memory shows preferences · /new starts a conversation · "
                    "Say 'I want to speak to an advisor' for human assistance · "
                    "/advisor also offers a handoff with consent · "
                    "/exit ends the chat[/dim]"
                )
                continue

            turno += 1
            run_id = uuid.uuid4()
            run_config = RunnableConfig(
                run_id=run_id,
                run_name=f"chat:{settings.business_domain}:turno-{turno}",
                tags=[settings.business_domain, "cli"],
                metadata={
                    "caso": settings.business_domain,
                    "turno": turno,
                    "canal": "cli",
                    "mcp_habilitado": usar_mcp,
                },
                configurable={"thread_id": current_thread_id},
            )
            try:
                async for update in graph.astream(
                    {"messages": [HumanMessage(content=texto)]},
                    config=run_config,
                    stream_mode="updates",
                ):
                    for node_name, node_output in update.items():
                        if node_output.get('review_required'):
                            ui.info('I am reviewing the guidance before showing it.')
                        if node_output.get('review'):
                            last_review = node_output['review']
                        if node_name == "comprimir_contexto":
                            ui.info(
                                "Context compressed: "
                                f"{node_output['compression_before']} → "
                                f"{node_output['compression_after']} messages + summary."
                            )
                        for message in node_output.get("messages", []):
                            if isinstance(message, AIMessage) and message.tool_calls:
                                if message.content:
                                    ui.respuesta_agente(str(message.content))
                                for call in message.tool_calls:
                                    ui.tool_llamada(call["name"], call["args"])
                            elif isinstance(message, ToolMessage):
                                ui.tool_resultado(message.name, str(message.content))
                            elif isinstance(message, AIMessage) and message.content:
                                ui.respuesta_agente(message.content)
                if settings.langsmith_tracing:
                    ui.info(f"LangSmith trace: {run_id}")
            except Exception as exc:
                ui.error(f"Assistant invocation failed: {exc}")
                ui.aviso(
                    f"Check that Ollama is running ({settings.ollama_base_url}) and the "
                    f"model is downloaded: ollama pull {settings.ollama_model}"
                )

    ui.console.print("[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    cli()
