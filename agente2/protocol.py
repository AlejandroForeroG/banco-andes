"""Banco Andes."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints, model_validator

MAX_BYTES = 32768
Text = Annotated[str, StringConstraints(min_length=1, max_length=500)]
Identifier = Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9_-]{1,64}$')]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Evidence(StrictModel):
    id: Identifier
    fuente: Literal['perfil', 'historial', 'catalogo']
    afirmaciones: list[Text] = Field(min_length=1, max_length=5)


class ReviewRequest(StrictModel):
    task_id: UUID
    protocol_version: Literal['1.0']
    fecha: date
    ambito: Literal['general', 'vivienda']
    revision: StrictInt = Field(ge=1, le=2)
    deadline_at: datetime
    solicitud: str = Field(min_length=1, max_length=2000)
    borrador: str = Field(min_length=1, max_length=4000)
    evidencias: list[Evidence] = Field(max_length=20)

    @model_validator(mode='after')
    def invariants(self):
        if self.deadline_at.tzinfo is None:
            raise ValueError('PLAZO_SIN_ZONA')
        ids = [item.id for item in self.evidencias]
        if len(set(ids)) != len(ids):
            raise ValueError('EVIDENCIAS_DUPLICADAS')
        reject_private_text(self.solicitud + '\n' + self.borrador + '\n' + json.dumps(
            [e.model_dump() for e in self.evidencias], ensure_ascii=False))
        return self


class Criterion(StrictModel):
    criterio: Identifier
    resultado: Literal['cumple', 'incumple', 'no_evaluable']
    evidencias: list[Identifier] = Field(max_length=20)


class Finding(StrictModel):
    criterio: Identifier
    correccion: Text


class Verdict(StrictModel):
    criterios_evaluados: list[Criterion] = Field(min_length=1, max_length=10)
    hallazgos: list[Finding] = Field(max_length=10)
    faltantes: list[Text] = Field(max_length=10)


class ReviewResponse(StrictModel):
    task_id: UUID
    protocol_version: Literal['1.0']
    revision: StrictInt = Field(ge=1, le=2)
    estado_tarea: Literal['recibida', 'trabajando', 'finalizada', 'fallida']
    status: Literal['validado', 'corregir', 'insuficiente'] | None
    politica_version: str | None
    criterios_evaluados: list[Criterion]
    hallazgos: list[Finding]
    faltantes: list[str]
    error: str | None


def wire(value) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json')
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def shared_key() -> bytes:
    path = os.getenv('REVIEW_KEY_FILE')
    value = Path(path).read_bytes().strip() if path else os.getenv('REVIEW_KEY', '').encode()
    if len(value) < 32:
        raise RuntimeError('CLAVE_SERVICIO_NO_CONFIGURADA')
    return value


def signature(body: bytes, purpose: str, key: bytes | None = None) -> str:
    return hmac.new(key or shared_key(), purpose.encode() + b'\n' + body, hashlib.sha256).hexdigest()


def authenticated(body: bytes, sig: str, purpose: str) -> bool:
    return hmac.compare_digest(signature(body, purpose), sig)


def remaining(request: ReviewRequest) -> float:
    return (request.deadline_at - datetime.now(timezone.utc)).total_seconds()


def reject_private_text(text: str):

    if re.search(r'\b\d{7,}\b|[\w.+-]+@[\w.-]+\.[a-z]{2,}|Bearer\s+\S+|(?:api[_ -]?key|token|clave|password|c[eé]dula|cuenta)\s*[:=]\s*\S+', text, re.I):
        raise ValueError('DATOS_PROHIBIDOS')


def expected_criteria(request: ReviewRequest) -> set[str]:
    if request.fecha < date(2026, 9, 1):
        raise ValueError('POLITICA_DESCONOCIDA')
    return {'C01', 'C02', 'C03', 'C04'} if request.ambito == 'vivienda' else {'C01', 'C02', 'C03'}


def validate_verdict(verdict: Verdict, request: ReviewRequest, expected: set[str]):
    ids = [item.criterio for item in verdict.criterios_evaluados]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('COBERTURA_INVALIDA')
    evidence_ids = {item.id for item in request.evidencias}
    if any(not set(item.evidencias) <= evidence_ids for item in verdict.criterios_evaluados):
        raise ValueError('REFERENCIA_INVALIDA')
    failures = {item.criterio for item in verdict.criterios_evaluados if item.resultado == 'incumple'}
    if {h.criterio for h in verdict.hallazgos} != failures:
        raise ValueError('HALLAZGOS_INCOHERENTES')
    reject_private_text(wire(verdict).decode())
    if re.search(r'ignora|ejecuta|system prompt|instrucci[oó]n del sistema|reglas internas|curl |https?://',
                 ' '.join([h.correccion for h in verdict.hallazgos] + verdict.faltantes), re.I):
        raise ValueError('CORRECCION_FUERA_DE_CONTRATO')


def verdict_status(verdict: Verdict, revision: int) -> str:
    if verdict.faltantes or any(c.resultado == 'no_evaluable' for c in verdict.criterios_evaluados):
        return 'insuficiente'
    if any(c.resultado == 'incumple' for c in verdict.criterios_evaluados):
        return 'corregir' if revision == 1 else 'insuficiente'
    return 'validado'


def validate_response(response: ReviewResponse, request: ReviewRequest):
    if (response.task_id, response.revision, response.protocol_version) != (request.task_id, request.revision, request.protocol_version):
        raise ValueError('CORRELACION_INVALIDA')
    if remaining(request) <= 0:
        raise ValueError('RESPUESTA_TARDIA')
    if response.estado_tarea == 'finalizada':
        if response.politica_version != 'BA-ORIENTACION/2' or response.error is not None:
            raise ValueError('POLITICA_O_ERROR_INVALIDO')
        verdict = Verdict(**response.model_dump(include={'criterios_evaluados', 'hallazgos', 'faltantes'}))
        validate_verdict(verdict, request, expected_criteria(request))
        if response.status != verdict_status(verdict, request.revision):
            raise ValueError('DICTAMEN_INCOHERENTE')
    elif response.status is not None or response.criterios_evaluados or response.hallazgos or response.faltantes:
        raise ValueError('ESTADO_INCOHERENTE')
    elif (response.estado_tarea == 'fallida') != (response.error is not None):
        raise ValueError('ERROR_INCOHERENTE')


def event(request: ReviewRequest, state: str, **values) -> ReviewResponse:
    data = dict(task_id=request.task_id, revision=request.revision, protocol_version='1.0',
                estado_tarea=state, status=None, politica_version=None,
                criterios_evaluados=[], hallazgos=[], faltantes=[], error=None)
    return ReviewResponse(**{**data, **values})
