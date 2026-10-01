'Banco Andes.'
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
    source: Literal['profile', 'history', 'catalog']
    statements: list[Text] = Field(min_length=1, max_length=5)


class ReviewRequest(StrictModel):
    task_id: UUID
    protocol_version: Literal['1.0']
    case_date: date
    scope: Literal['general', 'housing']
    review_number: StrictInt = Field(ge=1, le=2)
    deadline_at: datetime
    request_text: str = Field(min_length=1, max_length=2000)
    draft: str = Field(min_length=1, max_length=4000)
    evidence: list[Evidence] = Field(max_length=20)

    @model_validator(mode='after')
    def invariants(self):
        if self.deadline_at.tzinfo is None:
            raise ValueError('DEADLINE_WITHOUT_TIMEZONE')
        ids = [item.id for item in self.evidence]
        if len(set(ids)) != len(ids):
            raise ValueError('DUPLICATE_EVIDENCE')
        reject_private_text(self.request_text + '\n' + self.draft + '\n' + json.dumps(
            [e.model_dump() for e in self.evidence], ensure_ascii=False))
        return self


class Criterion(StrictModel):
    criterion: Identifier
    result: Literal['pass', 'fail', 'not_evaluable']
    evidence: list[Identifier] = Field(max_length=20)


class Finding(StrictModel):
    criterion: Identifier
    correction: Text


class Verdict(StrictModel):
    criteria_evaluated: list[Criterion] = Field(min_length=1, max_length=10)
    findings: list[Finding] = Field(max_length=10)
    missing_information: list[Text] = Field(max_length=10)


class ReviewResponse(StrictModel):
    task_id: UUID
    protocol_version: Literal['1.0']
    review_number: StrictInt = Field(ge=1, le=2)
    task_state: Literal['received', 'working', 'completed', 'failed']
    status: Literal['validated', 'correct', 'insufficient'] | None
    policy_version: str | None
    criteria_evaluated: list[Criterion]
    findings: list[Finding]
    missing_information: list[str]
    error: str | None


def wire(value) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json')
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def shared_key() -> bytes:
    path = os.getenv('REVIEW_KEY_FILE')
    value = Path(path).read_bytes().strip() if path else os.getenv('REVIEW_KEY', '').encode()
    if len(value) < 32:
        raise RuntimeError('SERVICE_KEY_NOT_CONFIGURED')
    return value


def signature(body: bytes, purpose: str, key: bytes | None = None) -> str:
    return hmac.new(key or shared_key(), purpose.encode() + b'\n' + body, hashlib.sha256).hexdigest()


def authenticated(body: bytes, sig: str, purpose: str) -> bool:
    return hmac.compare_digest(signature(body, purpose), sig)


def remaining(request: ReviewRequest) -> float:
    return (request.deadline_at - datetime.now(timezone.utc)).total_seconds()


def reject_private_text(text: str):

    if re.search(r'\b\d{7,}\b|[\w.+-]+@[\w.-]+\.[a-z]{2,}|Bearer\s+\S+|(?:api[_ -]?key|token|pin|password|customer[_ -]?id|account|clave|c[eé]dula|cuenta)\s*[:=]\s*\S+', text, re.I):
        raise ValueError('PROHIBITED_DATA')


def expected_criteria(request: ReviewRequest) -> set[str]:
    if request.case_date < date(2026, 9, 1):
        raise ValueError('UNKNOWN_POLICY')
    return {'C01', 'C02', 'C03', 'C04'} if request.scope == 'housing' else {'C01', 'C02', 'C03'}


def validate_verdict(verdict: Verdict, request: ReviewRequest, expected: set[str]):
    ids = [item.criterion for item in verdict.criteria_evaluated]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('INVALID_COVERAGE')
    evidence_ids = {item.id for item in request.evidence}
    if any(not set(item.evidence) <= evidence_ids for item in verdict.criteria_evaluated):
        raise ValueError('INVALID_REFERENCE')
    failures = {item.criterion for item in verdict.criteria_evaluated if item.result == 'fail'}
    if {h.criterion for h in verdict.findings} != failures:
        raise ValueError('INCONSISTENT_FINDINGS')
    reject_private_text(wire(verdict).decode())
    if re.search(r'ignore|execute|system prompt|system instruction|internal rules|curl |https?://',
                 ' '.join([h.correction for h in verdict.findings] + verdict.missing_information), re.I):
        raise ValueError('CORRECTION_OUTSIDE_CONTRACT')


def verdict_status(verdict: Verdict, review_number: int) -> str:
    if verdict.missing_information or any(c.result == 'not_evaluable' for c in verdict.criteria_evaluated):
        return 'insufficient'
    if any(c.result == 'fail' for c in verdict.criteria_evaluated):
        return 'correct' if review_number == 1 else 'insufficient'
    return 'validated'


def validate_response(response: ReviewResponse, request: ReviewRequest):
    if (response.task_id, response.review_number, response.protocol_version) != (request.task_id, request.review_number, request.protocol_version):
        raise ValueError('INVALID_CORRELATION')
    if remaining(request) <= 0:
        raise ValueError('LATE_RESPONSE')
    if response.task_state == 'completed':
        if response.policy_version != 'BA-GUIDANCE/2' or response.error is not None:
            raise ValueError('INVALID_POLICY_OR_ERROR')
        verdict = Verdict(**response.model_dump(include={'criteria_evaluated', 'findings', 'missing_information'}))
        validate_verdict(verdict, request, expected_criteria(request))
        if response.status != verdict_status(verdict, request.review_number):
            raise ValueError('INCONSISTENT_VERDICT')
    elif response.status is not None or response.criteria_evaluated or response.findings or response.missing_information:
        raise ValueError('INCONSISTENT_STATE')
    elif (response.task_state == 'failed') != (response.error is not None):
        raise ValueError('INCONSISTENT_ERROR')


def event(request: ReviewRequest, state: str, **values) -> ReviewResponse:
    data = dict(task_id=request.task_id, review_number=request.review_number, protocol_version='1.0',
                task_state=state, status=None, policy_version=None,
                criteria_evaluated=[], findings=[], missing_information=[], error=None)
    return ReviewResponse(**{**data, **values})
