"""Prueba real del canal: rechaza un emisor sin firma antes de consultar SQLite."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from uuid import uuid4

import httpx
from langsmith import traceable

from agente2.protocol import ReviewRequest, wire
from cli.config import get_settings


@traceable(name='review.seguridad.emisor_no_autenticado', run_type='chain')
async def verify():
    get_settings()
    now = datetime.now(timezone.utc)
    request = ReviewRequest(task_id=uuid4(), protocol_version='1.0', fecha=now.date(), ambito='vivienda', revision=1,
                            deadline_at=now+timedelta(seconds=40), solicitud='Consulta de demostración',
                            borrador='Producto pendiente de revisión.', evidencias=[])
    async with httpx.AsyncClient(timeout=5) as client:
        result = await client.post(os.environ['REVIEW_URL'].rstrip('/') + '/review', content=wire(request))
    assert result.status_code == 401, f'Se esperaba HTTP 401, llegó {result.status_code}'
    assert result.json() == {'error': 'EMISOR_NO_AUTENTICADO'}
    return {'task_id': str(request.task_id), 'http_status': result.status_code, **result.json()}


if __name__ == '__main__':
    print(json.dumps(asyncio.run(verify()), ensure_ascii=False, indent=2))
