'Authenticated SSE service with deadline-scoped idempotency.'
import asyncio
import hashlib
import json
import logging
import os
import time
import httpx
from dataclasses import dataclass, field

from langsmith import tracing_context
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from reviewer.critic import critique, load_sources
from reviewer.protocol import MAX_BYTES, ReviewRequest, authenticated, event, remaining, signature, wire

log = logging.getLogger('reviewer')


@dataclass
class Job:
    request: ReviewRequest
    digest: str
    events: list = field(default_factory=list)
    task: asyncio.Task | None = None
    sources: dict | None = None
    created: float = field(default_factory=time.monotonic)


jobs: dict[tuple[str, int], Job] = {}


def publish(job, response):
    job.events.append(response)
    log.warning('task_id=%s review_number=%s state=%s status=%s error=%s', response.task_id,
                response.review_number, response.task_state, response.status, response.error)


async def work(job, parent):
    try:
        with tracing_context(parent=parent):
            publish(job, event(job.request, 'working'))
            if job.sources is None:
                job.sources = load_sources(job.request)
            publish(job, await critique(job.request, job.sources))
    except asyncio.CancelledError:
        publish(job, event(job.request, 'failed', error='CANCELLED'))
    except (TimeoutError, httpx.TimeoutException):
        publish(job, event(job.request, 'failed', error='DEADLINE_EXCEEDED'))
    except Exception as exc:
        import sqlite3
        code = 'SOURCE_UNAVAILABLE' if isinstance(exc, (sqlite3.Error, OSError)) else 'INVALID_REVIEW'
        publish(job, event(job.request, 'failed', error=code))


async def receive(request: Request):
    body = b''
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BYTES:
            return JSONResponse({'error': 'MESSAGE_TOO_LARGE'}, status_code=413)
    purpose = 'POST ' + request.url.path
    if not authenticated(body, request.headers.get('x-review-signature', ''), purpose):
        log.warning('rechazo=UNAUTHENTICATED_SENDER')
        return JSONResponse({'error': 'UNAUTHENTICATED_SENDER'}, status_code=401)
    try:
        payload = ReviewRequest.model_validate_json(body)
        if request.url.path != '/cancel' and not 0 < remaining(payload) <= float(os.getenv('REVIEW_TOTAL_SECONDS', '90')) + 2:
            raise ValueError('INVALID_DEADLINE')
        key = (str(payload.task_id), payload.review_number)
        digest = hashlib.sha256(wire(payload)).hexdigest()
    except (ValueError, TypeError):
        return JSONResponse({'error': 'INVALID_CONTRACT'}, status_code=422)
    if request.url.path == '/cancel':
        job = jobs.get(key)
        if job and job.task and not job.task.done():
            job.task.cancel()
        return JSONResponse({'cancelada': True})

    for old in list(jobs):
        if remaining(jobs[old].request) < -60 and (not jobs[old].task or jobs[old].task.done()):
            del jobs[old]
    previous = jobs.get((str(payload.task_id), 1))
    if key in jobs:
        job = jobs[key]
        if job.digest != digest:
            return JSONResponse({'error': 'DUPLICATE_WITH_DIFFERENT_CONTENT'}, status_code=409)
    else:
        if len(jobs) >= 128:
            return JSONResponse({'error': 'CAPACITY_EXCEEDED'}, status_code=503)
        if payload.review_number == 2:
            if (not previous or not previous.events or previous.events[-1].status != 'correct'
                    or payload.deadline_at != previous.request.deadline_at
                    or payload.case_date != previous.request.case_date or payload.scope != previous.request.scope
                    or payload.evidence != previous.request.evidence or payload.request_text != previous.request.request_text):
                return JSONResponse({'error': 'INVALID_TRANSITION'}, status_code=409)
        job = Job(payload, digest, sources=previous.sources if previous else None)
        jobs[key] = job
        publish(job, event(payload, 'received'))
        parent = {k: request.headers[k] for k in ('langsmith-trace',) if k in request.headers}
        job.task = asyncio.create_task(work(job, parent))

    async def stream():
        index, last = 0, time.monotonic()
        while True:
            while index < len(job.events):
                response = job.events[index]
                index += 1
                data = wire(response)
                envelope = {'payload': response.model_dump(mode='json'), 'signature': signature(data, 'EVENT')}
                yield b'data: ' + wire(envelope) + b'\n\n'
                if response.task_state in ('completed', 'failed'):
                    return
            if remaining(payload) <= 0:
                job.task.cancel()
            if time.monotonic() - last >= 10:
                publish(job, event(payload, 'working'))
                last = time.monotonic()
            await asyncio.sleep(0.05)
    return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-store'})


async def health(request):
    from reviewer.query_sources import consultar, DEFAULT_DB
    try:
        from datetime import date
        sources = consultar(os.getenv('POLICIES_DB', str(DEFAULT_DB)), date.today().isoformat())
        from reviewer.protocol import shared_key
        shared_key()
        return JSONResponse({'service': 'reviewer', 'state': sources['source_state'],
                             'criteria': len(sources['criteria']), 'pid': os.getpid()})
    except Exception:
        return JSONResponse({'error': 'SOURCE_OR_KEY_UNAVAILABLE'}, status_code=503)


app = Starlette(routes=[Route('/health', health), Route('/review', receive, methods=['POST']),
                       Route('/cancel', receive, methods=['POST'])])
