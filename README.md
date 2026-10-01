# Banco Andes

Asistente de orientación financiera con un modelo local, memoria persistente y un segundo agente que revisa cada respuesta antes de mostrarla. Mantenido por [Juan Alejandro Forero Gómez](https://www.linkedin.com/in/alejandroforerog).

## Capacidades

- Consulta de perfil, historial y catálogo mediante herramientas autenticadas.
- Conversaciones persistentes en SQLite y preferencias aisladas por cliente.
- Razonamiento ReAct con un límite de iteraciones.
- Revisión independiente contra políticas versionadas: validar, corregir o declarar información insuficiente.
- Transferencia a un asesor mediante MCP, con consentimiento y confirmación del radicado.
- Trazas opcionales en LangSmith con redacción de secretos.

El proyecto utiliza clientes, productos y políticas sintéticos. La revisión es editorial: no aprueba créditos. La API incluida es una demostración local con autenticación simulada y datos en memoria; sus cambios se pierden al reiniciar. La atención del asesor después de recibir el radicado queda fuera del prototipo.

## Ejecutar con Docker

Requiere Docker Compose y Ollama en el anfitrión. Descarga el modelo y arranca los servicios:

```bash
ollama pull qwen3:4b-instruct-2507-q8_0
docker compose up --build -d
docker compose exec agente agente chat
```

Compose construye la API de demostración, el orientador y el revisor desde este repositorio. Genera una clave compartida para firmar los intercambios y monta la base de políticas. No requiere RAG externo ni credenciales para ejecutar la orientación.

En el chat, usa `/login` con la identificación `DEMO-01` y la clave `2001`. Los demás clientes sintéticos están en `demo-api/app/data/finanzas.json`. Consulta el catálogo o pide orientación y escribe `Quiero hablar con un asesor` para iniciar la transferencia. `/ayuda` muestra los comandos.

```bash
docker compose ps
docker compose exec -T agente python -m agent.verify_security
docker compose down
```

`down` conserva la memoria; `down -v` la elimina. Los servicios se comunican sin publicar puertos. Para cambiar el modelo, copia `.env.example` a `.env` y modifica `OLLAMA_MODEL`.

## Desarrollo local

Con Python 3.11 o superior, desde la raíz:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
python -m agente2.montar_fuentes
agente herramientas --sin-mcp
```

Para el flujo completo usa Compose. La instalación local permite trabajar en módulos y ejecutar pruebas; no inicia la API ni el revisor automáticamente.

## Diseño

El CLI abre una sesión y pasa la consulta al grafo de LangGraph. El orientador consulta herramientas locales y MCP, incorpora la memoria del cliente y prepara una respuesta. El revisor recupera criterios de su propia SQLite mediante consultas de solo lectura y devuelve un dictamen firmado. El orientador comprueba firma, correlación, cobertura y presupuesto antes de aceptar el texto. El ciclo admite dos críticas y un refinamiento; si falta evidencia o falla la revisión, comunica el límite.

| Carpeta | Función |
| --- | --- |
| `cli/` | Comandos, sesión y presentación en terminal |
| `agent/` | Grafo, herramientas, memoria y revisión |
| `agente2/` | Servicio revisor, protocolo y políticas versionadas |
| `mcp_server/` | Herramientas de acciones y transferencia |
| `demo-api/` | API local del dominio financiero |
| `observability/` | Instrumentación y redacción de secretos |
| `tests/` | Pruebas por capacidad |

El banner ASCII y los colores de Banco Andes se configuran en `cli_design.toml`.

## Pruebas

```bash
python -m pytest -q
python -m unittest agente2.test_fuentes -v
```

La suite ordinaria usa dobles de prueba y no requiere Ollama ni LangSmith. Las evaluaciones con servicios externos se habilitan explícitamente:

```powershell
$env:RUN_OLLAMA_EVAL='1'
python -m pytest tests/test_review_model.py -q
$env:RUN_LANGSMITH_EVAL='1'
python -m pytest tests/test_observability.py -q
```

La segunda requiere clave y proyecto de LangSmith en el entorno. Las pruebas de contrato no demuestran por sí solas la calidad de las respuestas del modelo.

## Subir a tu cuenta

Crea un repositorio vacío llamado `banco-andes` en tu cuenta y ejecuta desde esta carpeta:

```bash
git remote add origin https://github.com/TU_USUARIO/banco-andes.git
git push -u origin main
```

La copia tiene su propio historial y no modifica el repositorio original. Consulta los [créditos](NOTICE.md).
