'Banco Andes demo credit guidance API. Private financial data requires customer-scoped authentication.'

from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.database import DB, next_id

router = APIRouter()


def _db() -> dict:
    return DB["finanzas"]


def _find(items: list[dict], item_id: str) -> dict | None:
    return next((i for i in items if i["id"] == item_id), None)


def _require_bearer(cliente_id: str, authorization: str | None) -> None:
    'Validate the demo bearer token issued by /auth/login.'
    esperado = f"Bearer tok-{cliente_id}"
    if authorization != esperado:
        raise HTTPException(
            401,
            "Protected financial data: authenticate with POST /finanzas/auth/login "
            "and send the token in the Authorization header.",
        )


class Credenciales(BaseModel):
    numero_identificacion: str
    clave: str


def _find_por_cedula(items: list[dict], numero_identificacion: str) -> dict | None:
    return next((i for i in items if i["numero_identificacion"] == numero_identificacion), None)


@router.get("/clientes/existe/{numero_identificacion}")
async def existe_cliente(numero_identificacion: str):
    'Check registration without exposing private data or requiring a PIN.'
    cliente = _find_por_cedula(_db()["clientes"], numero_identificacion)
    return {"existe": cliente is not None}


@router.post("/auth/login")
async def login(credenciales: Credenciales):
    'Simulate customer ID and PIN authentication and return a customer-scoped token.'
    cliente = _find_por_cedula(_db()["clientes"], credenciales.numero_identificacion)
    if not cliente or cliente["clave"] != credenciales.clave:
        raise HTTPException(401, "Invalid credentials.")
    return {"token": f"tok-{cliente['id']}", "tipo": "bearer", "cliente_id": cliente["id"]}


@router.get("/clientes/{cliente_id}/perfil-financiero")
async def perfil_financiero(cliente_id: str, authorization: str | None = Header(None)):
    'Return credit score, debt, income, and active products.'
    _require_bearer(cliente_id, authorization)
    cliente = _find(_db()["clientes"], cliente_id)
    if not cliente:
        raise HTTPException(404, f"Customer {cliente_id} does not exist.")
    return {k: v for k, v in cliente.items() if k != "clave"}


@router.get("/clientes/{cliente_id}/historial-solicitudes")
async def historial_solicitudes(cliente_id: str, authorization: str | None = Header(None)):
    _require_bearer(cliente_id, authorization)
    return [s for s in _db()["solicitudes_historicas"] if s["cliente_id"] == cliente_id]


@router.get("/productos")
async def listar_productos():
    'List available demo credit products.'


    return _db()["productos"]


class NuevaSolicitud(BaseModel):
    cliente_id: str
    intencion: str = Field(description="Customer intent in natural language")
    monto_estimado: float | None = None
    plazo_meses: int | None = None


class ActualizacionSolicitud(BaseModel):
    respuestas: dict | None = Field(None, description="Dynamically collected answers")
    monto_estimado: float | None = None
    plazo_meses: int | None = None
    estado: str | None = None


class Recomendacion(BaseModel):
    producto_id: str
    justificacion: str
    alternativas: list[str] = []


@router.post("/solicitudes", status_code=201)
async def crear_solicitud(nueva: NuevaSolicitud):
    request_text = nueva.model_dump()
    request_text["id"] = next_id(_db()["solicitudes"], "SOL")
    request_text["estado"] = "en_curso"
    request_text["respuestas"] = {}
    request_text["creada_en"] = datetime.now(timezone.utc).isoformat()
    _db()["solicitudes"].append(request_text)
    return request_text


@router.put("/solicitudes/{solicitud_id}")
async def actualizar_solicitud(solicitud_id: str, cambios: ActualizacionSolicitud):
    request_text = _find(_db()["solicitudes"], solicitud_id)
    if not request_text:
        raise HTTPException(404, f"Application {solicitud_id} does not exist.")
    if cambios.respuestas:
        request_text["respuestas"].update(cambios.respuestas)
    for campo in ("monto_estimado", "plazo_meses", "estado"):
        valor = getattr(cambios, campo)
        if valor is not None:
            request_text[campo] = valor
    return request_text


@router.post("/solicitudes/{solicitud_id}/recomendacion")
async def registrar_recomendacion(solicitud_id: str, recomendacion: Recomendacion):
    request_text = _find(_db()["solicitudes"], solicitud_id)
    if not request_text:
        raise HTTPException(404, f"Application {solicitud_id} does not exist.")
    if not _find(_db()["productos"], recomendacion.producto_id):
        raise HTTPException(422, f"Product {recomendacion.producto_id} does not exist in the catalog.")
    request_text["recomendacion"] = recomendacion.model_dump()
    request_text["estado"] = "recomendada"
    return request_text


class Escalamiento(BaseModel):
    solicitud_id: str | None = None
    cliente_id: str
    intencion: str
    informacion_recopilada: dict
    razon: str


@router.post("/escalamientos", status_code=201)
async def escalar(escalamiento: Escalamiento):
    'Send a structured case summary to a human advisor.'
    registro = escalamiento.model_dump()
    registro["id"] = next_id(_db()["escalamientos"], "ESC")
    registro["creado_en"] = datetime.now(timezone.utc).isoformat()
    registro["tiempo_estimado_horas"] = 24
    _db()["escalamientos"].append(registro)
    return registro


class Queja(BaseModel):
    cliente_id: str
    tipo: str
    descripcion: str


@router.post("/quejas", status_code=201)
async def register_complaint(queja: Queja):
    'Record a complaint and return its destination support channel.'
    registro = queja.model_dump()
    registro["id"] = next_id(_db()["quejas"], "QJA")
    registro["canal_redireccion"] = "servicio_al_cliente"
    registro["creada_en"] = datetime.now(timezone.utc).isoformat()
    _db()["quejas"].append(registro)
    return registro
