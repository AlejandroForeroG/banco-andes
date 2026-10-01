"""Banco Andes."""


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
    """Banco Andes — orientación financiera."""


@cli.command()
def version():
    """Versión instalada del proyecto."""
    try:
        click.echo(pkg_version("banco-andes"))
    except PackageNotFoundError:
        click.echo("0.1.0 (sin instalar; use pip install -e .)")


@cli.command()
def config():
    """Configuración vigente del agente."""
    settings = get_settings()
    click.echo(f"modelo Ollama    : {settings.ollama_model}")
    click.echo(f"Ollama URL       : {settings.ollama_base_url}")
    click.echo(f"API empresarial  : {settings.api_base_url}")
    click.echo(f"caso de negocio  : {settings.caso_negocio}")
    click.echo(f"LangSmith activo : {settings.langsmith_tracing}")


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
            ui.aviso(f"Servidor MCP no disponible ({error_mcp}). Continuando solo con tools locales.")
        tools = tools + mcp_tools
    return tools


@cli.command()
@click.option("--sin-mcp", is_flag=True, help="No conectar el servidor MCP (solo tools locales).")
def herramientas(sin_mcp: bool):
    """Lista las herramientas que el agente tiene disponibles."""
    tools = asyncio.run(_reunir_tools(usar_mcp=not sin_mcp))
    if not tools:
        ui.aviso("No hay herramientas disponibles. Revisa la configuración.")
        return
    for tool in tools:
        descripcion = (tool.description or "").split("\n")[0]
        click.echo(f"- {tool.name}: {descripcion}")


def _ask_pin() -> str:
    """Clave oculta en terminal; por stdin cuando el CLI corre dentro de un script."""
    if sys.stdin.isatty():
        return click.prompt("Clave", hide_input=True)
    pin = sys.stdin.readline().strip()
    if not pin:
        raise click.ClickException("No se recibió la clave por stdin.")
    return pin


@cli.command(name="probar-tool")
@click.argument("nombre")
@click.argument("argumentos", default="{}")
@click.option("--sin-mcp", is_flag=True, help="Buscar solo entre las tools locales.")
@click.option(
    "--identificacion",
    "identification",
    help="Autentica la sesión antes de invocar (pide la clave); requerido por las tools privadas.",
)
def probar_tool(nombre: str, argumentos: str, sin_mcp: bool, identification: str | None):
    """Invoca una tool directamente (sin LLM), útil para probar el acceso a la API.

    Ejemplo: agente probar-tool consultar_contacto '{"contacto_id": "CTO-001"}'
    """
    if identification:
        from agent.session import start_session

        pin = _ask_pin()
        try:
            session = start_session(identification, pin)
        except RuntimeError as exc:
            raise click.ClickException(str(exc))
        ui.info(f"Sesión iniciada para {session.client_id}.")

    async def _run():
        tools = await _reunir_tools(usar_mcp=not sin_mcp)
        tool = next((t for t in tools if t.name == nombre), None)
        if tool is None:
            disponibles = ", ".join(t.name for t in tools)
            raise click.ClickException(f"Tool '{nombre}' no existe. Disponibles: {disponibles}")
        try:
            args = json.loads(argumentos)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Argumentos inválidos (deben ser JSON): {exc}")
        try:
            resultado = await tool.ainvoke(args)
        except Exception as exc:
            raise click.ClickException(str(exc))
        click.echo(resultado if isinstance(resultado, str) else json.dumps(resultado, indent=2, ensure_ascii=False))

    asyncio.run(_run())


@cli.command()
@click.option("--sin-mcp", is_flag=True, help="No conectar el servidor MCP (solo tools locales).")
@click.option(
    "--thread-id",
    help="Identificador de conversación para reanudarla; si se omite se crea uno nuevo.",
)
def chat(sin_mcp: bool, thread_id: str | None):
    """Conversación interactiva con el agente."""
    asyncio.run(_chat(usar_mcp=not sin_mcp, initial_thread_id=thread_id))


async def _start_session() -> None:
    from agent.session import start_session

    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout

    prompt_session = PromptSession()
    with patch_stdout():
        identification = (await prompt_session.prompt_async("identificación ❯ ")).strip()
        if not identification:
            ui.aviso("Autenticación cancelada.")
            return
        pin = (await prompt_session.prompt_async("clave ❯ ", is_password=True)).strip()
    try:
        session = start_session(identification, pin)
    except RuntimeError as exc:
        ui.error(str(exc))
        return
    ui.info(f"Sesión iniciada para {session.client_id}.")


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
    listener.set(lambda event: ui.info('Revisión: ' + json.dumps(event, ensure_ascii=False)))
    current_thread_id = (initial_thread_id or str(uuid.uuid4())).strip()
    turno = 0
    last_review = {}
    prompt_session = PromptSession()

    async with open_async_checkpointer() as checkpointer:
        graph = build_graph(tools=tools, checkpointer=checkpointer)
        ui.info(f"Conversación: {current_thread_id}")

        while True:
            try:
                with patch_stdout():
                    texto = (await prompt_session.prompt_async(ui.prompt_symbol())).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not texto:
                continue
            if texto in ("/salir", "/exit"):
                break
            if texto == "/nueva":
                current_thread_id = str(uuid.uuid4())
                turno = 0
                last_review = {}
                ui.aviso(f"Nueva conversación: {current_thread_id}")
                continue
            if texto == "/thread":
                ui.info(f"Conversación actual: {current_thread_id}")
                continue
            if texto == "/estado":
                state = await graph.aget_state(
                    {"configurable": {"thread_id": current_thread_id}}
                )
                values = state.values
                ui.info(
                    f"Estado: {len(values.get('messages', []))} mensajes · "
                    f"resumen: {'sí' if values.get('summary') else 'no'} · "
                    f"preferencias recordadas: {len(values.get('long_term_memory', {}))}"
                )
                continue
            if texto == "/memoria":
                from agent.memory.store import load_client_memory
                from agent.session import active_session

                active = active_session()
                if active is None:
                    ui.aviso("Inicie sesión con /login para consultar su memoria.")
                else:
                    remembered = load_client_memory(active.client_id)
                    if remembered:
                        ui.info(json.dumps(remembered, ensure_ascii=False))
                    else:
                        ui.aviso("No hay preferencias guardadas para este cliente.")
                continue
            if texto == "/login":
                last_review = {}
                await _start_session()
                continue
            wants_advisor = texto == '/asesor'
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
                        last_review = {'status': 'solicitada', 'error': 'SOLICITUD_USUARIO'}
                    summary = transfer_summary(last_review)
                    ui.info('Se enviaría al asesor: ' + json.dumps(summary, ensure_ascii=False))
                    with patch_stdout():
                        accepted = (await prompt_session.prompt_async('¿Aceptas transferir este resumen? [sí/no] ')).strip().lower() in ('si', 'sí')
                    result = await transfer(last_review, accepted, tools)
                    ui.info(json.dumps(result, ensure_ascii=False))
                    if result.get('transferido'):
                        last_review = {}
                except (ValueError, RuntimeError) as exc:
                    ui.aviso(str(exc))
                continue
            if texto == "/logout":
                last_review = {}
                end_session()
                ui.info("Sesión cerrada. El agente ya no puede consultar datos privados.")
                continue
            if texto == "/ayuda":
                ui.console.print(
                    "[dim]/login abre la sesión · /logout la cierra · "
                    "/thread muestra el identificador · /estado inspecciona el contexto · "
                    "/memoria muestra preferencias · /nueva crea otra conversación · "
                    "Di 'quiero hablar con un asesor' para solicitar atención humana · "
                    "/asesor también ofrece transferencia consentida · "
                    "/salir termina[/dim]"
                )
                continue

            turno += 1
            run_id = uuid.uuid4()
            run_config = RunnableConfig(
                run_id=run_id,
                run_name=f"chat:{settings.caso_negocio}:turno-{turno}",
                tags=[settings.caso_negocio, "cli"],
                metadata={
                    "caso": settings.caso_negocio,
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
                            ui.info('Estoy revisando la orientación antes de mostrarla.')
                        if node_output.get('review'):
                            last_review = node_output['review']
                        if node_name == "comprimir_contexto":
                            ui.info(
                                "Contexto comprimido: "
                                f"{node_output['compression_before']} → "
                                f"{node_output['compression_after']} mensajes + resumen."
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
                    ui.info(f"Traza LangSmith: {run_id}")
            except Exception as exc:
                ui.error(f"Fallo al invocar el agente: {exc}")
                ui.aviso(
                    f"Verifique que Ollama está corriendo ({settings.ollama_base_url}) y que el "
                    f"modelo está descargado: ollama pull {settings.ollama_model}"
                )

    ui.console.print("[dim]Hasta pronto.[/dim]")


if __name__ == "__main__":
    cli()
