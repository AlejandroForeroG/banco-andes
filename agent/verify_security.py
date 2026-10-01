'Live unsigned-sender rejection check before SQLite access.'
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from uuid import uuid4

import httpx
from langsmith import traceable

from reviewer.protocol import ReviewRequest, wire
from cli.config import get_settings


@traceable(name='review.seguridad.emisor_no_autenticado', run_type='chain')
async def verify():
    get_settings()
    now = datetime.now(timezone.utc)
    request = ReviewRequest(task_id=uuid4(), protocol_version='1.0', case_date=now.date(), scope='housing', review_number=1,
                            deadline_at=now+timedelta(seconds=40), request_text='Demo query',
                            draft='Product awaiting review.', evidence=[])
    async with httpx.AsyncClient(timeout=5) as client:
        result = await client.post(os.environ['REVIEW_URL'].rstrip('/') + '/review', content=wire(request))
    assert result.status_code == 401, f'Expected HTTP 401, received {result.status_code}'
    assert result.json() == {'error': 'UNAUTHENTICATED_SENDER'}
    return {'task_id': str(request.task_id), 'http_status': result.status_code, **result.json()}


if __name__ == '__main__':
    print(json.dumps(asyncio.run(verify()), ensure_ascii=False, indent=2))
