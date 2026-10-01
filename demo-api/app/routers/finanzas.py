"""Dominio Banco Andes — orientación de crédito digital.

Regla del caso de negocio: los datos financieros de un cliente solo pueden
consultarse tras autenticación. El login retorna un bearer token acotado al
cliente autenticado.
"""

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
    """El token emitido por /auth/login es 'tok-{cliente_id}'."""
    esperado = f"Bearer tok-{cliente_id}"
    if authorization != esperado:
        raise HTTPException(
            401,
            "Datos financieros protegidos: autentíquese primero con POST /finanzas/auth/login "
            "y envíe el token en el header Authorization.",
        )


class Credenciales(BaseModel):
    numero_identificacion: str
    clave: str


def _find_por_cedula(items: list[dict], numero_identificacion: str) -> dict | None:
    return next((i for i in items if i["numero_identificacion"] == numero_identificacion), None)


@router.get("/clientes/existe/{numero_identificacion}")
async def existe_cliente(numero_identificacion: str):
    """Verifica si una cédula está registrada, sin exponer datos ni requerir clave.
    Úsela antes de pedir la clave al cliente (igual que un cajero reconoce la
    tarjeta antes de pedir el PIN)."""
    cliente = _find_por_cedula(_db()["clientes"], numero_identificacion)
    return {"existe": cliente is not None}


@router.post("/auth/login")
async def login(credenciales: Credenciales):
    """Simula la autenticación del canal digital con cédula + clave (4 dígitos,
    la misma del cajero). Retorna un bearer token asociado al id interno del
    cliente."""
    cliente = _find_por_cedula(_db()["clientes"], credenciales.numero_identificacion)
    if not cliente or cliente["clave"] != credenciales.clave:
        raise HTTPException(401, "Credenciales inválidas.")
    return {"token": f"tok-{cliente['id']}", "tipo": "bearer", "cliente_id": cliente["id"]}


@router.get("/clientes/{cliente_id}/perfil-financiero")
async def perfil_financiero(cliente_id: str, authorization: str | None = Header(None)):
    """Score crediticio (300-850), endeudamiento, ingresos y productos activos."""
    _require_bearer(cliente_id, authorization)
    cliente = _find(_db()["clientes"], cliente_id)
    if not cliente:
        raise HTTPException(404, f"Cliente {cliente_id} no existe.")
    return {k: v for k, v in cliente.items() if k != "clave"}


@router.get("/clientes/{cliente_id}/historial-solicitudes")
async def historial_solicitudes(cliente_id: str, authorization: str | None = Header(None)):
    _require_bearer(cliente_id, authorization)
    return [s for s in _db()["solicitudes_historicas"] if s["cliente_id"] == cliente_id]


@router.get("/productos")
async def listar_productos():
    """listar productos."""


    return _db()["productos"]


class NuevaSolicitud(BaseModel):
    cliente_id: str
    intencion: str = Field(description="Intención expresada por el cliente en lenguaje natural")
    monto_estimado: float | None = None
    plazo_meses: int | None = None


class ActualizacionSolicitud(BaseModel):
    respuestas: dict | None = Field(None, description="Respuestas recopiladas dinámicamente")
    monto_estimado: float | None = None
    plazo_meses: int | None = None
    estado: str | None = None


class Recomendacion(BaseModel):
    producto_id: str
    justificacion: str
    alternativas: list[str] = []


@router.post("/solicitudes", status_code=201)
async def crear_solicitud(nueva: NuevaSolicitud):
    solicitud = nueva.model_dump()
    solicitud["id"] = next_id(_db()["solicitudes"], "SOL")
    solicitud["estado"] = "en_curso"
    solicitud["respuestas"] = {}
    solicitud["creada_en"] = datetime.now(timezone.utc).isoformat()
    _db()["solicitudes"].append(solicitud)
    return solicitud


@router.put("/solicitudes/{solicitud_id}")
async def actualizar_solicitud(solicitud_id: str, cambios: ActualizacionSolicitud):
    solicitud = _find(_db()["solicitudes"], solicitud_id)
    if not solicitud:
        raise HTTPException(404, f"Solicitud {solicitud_id} no existe.")
    if cambios.respuestas:
        solicitud["respuestas"].update(cambios.respuestas)
    for campo in ("monto_estimado", "plazo_meses", "estado"):
        valor = getattr(cambios, campo)
        if valor is not None:
            solicitud[campo] = valor
    return solicitud


@router.post("/solicitudes/{solicitud_id}/recomendacion")
async def registrar_recomendacion(solicitud_id: str, recomendacion: Recomendacion):
    solicitud = _find(_db()["solicitudes"], solicitud_id)
    if not solicitud:
        raise HTTPException(404, f"Solicitud {solicitud_id} no existe.")
    if not _find(_db()["productos"], recomendacion.producto_id):
        raise HTTPException(422, f"Producto {recomendacion.producto_id} no existe en el catálogo.")
    solicitud["recomendacion"] = recomendacion.model_dump()
    solicitud["estado"] = "recomendada"
    return solicitud


class Escalamiento(BaseModel):
    solicitud_id: str | None = None
    cliente_id: str
    intencion: str
    informacion_recopilada: dict
    razon: str


@router.post("/escalamientos", status_code=201)
async def escalar(escalamiento: Escalamiento):
    """Escala a asesor humano con el resumen estructurado de 3 componentes."""
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
async def registrar_queja(queja: Queja):
    """Las quejas se redirigen al área correspondiente, no se resuelven como crédito."""
    registro = queja.model_dump()
    registro["id"] = next_id(_db()["quejas"], "QJA")
    registro["canal_redireccion"] = "servicio_al_cliente"
    registro["creada_en"] = datetime.now(timezone.utc).isoformat()
    _db()["quejas"].append(registro)
    return registro
