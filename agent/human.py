'Minimal advisor handoff with consent and a verified receipt.'
import json

from agent.session import required_session


def transfer_summary(review):
    if not review:
        raise ValueError('No review is available for handoff.')
    exchanges = review.get('exchanges', [])
    last = exchanges[-1] if exchanges else {}

    validated = review.get('status') == 'validated'
    return {'task_id': review.get('task_id'), 'state': review['status'],
            'reviewed_guidance' if validated else 'blocked_draft': last.get('request', {}).get('draft', ''),
            'evidence': last.get('request', {}).get('evidence', []),
            'findings': last.get('response', {}).get('findings', []),
            'reason': 'USER_REQUEST' if validated else
                     review.get('error') or last.get('response', {}).get('error') or review['status']}


def confirmed_receipt(result, client_id):
    'Validate a handoff receipt; MCP text blocks may contain errors.'
    if isinstance(result, list):
        texts = [b['text'] for b in result if isinstance(b, dict) and b.get('type') == 'text']
        result = texts[0] if len(texts) == 1 else None
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            result = None
    if (not isinstance(result, dict) or not isinstance(result.get('id'), str)
            or not result['id'].startswith('ESC-') or result.get('cliente_id') != client_id
            or not result.get('creado_en')):
        raise RuntimeError('No valid advisor receipt was received; handoff cannot be confirmed.')
    return {key: result[key] for key in ('id', 'creado_en', 'tiempo_estimado_horas') if key in result}


async def transfer(review, consent, tools):
    if not consent:
        return {'transferred': False}
    session = required_session()
    summary = transfer_summary(review)
    tool = next((t for t in tools if t.name == 'register_handoff'), None)
    if tool is None:
        raise RuntimeError('The MCP advisor channel is unavailable.')
    result = await tool.ainvoke({'intencion': 'Continue guidance with an authorized advisor',
                               'informacion_recopilada': summary, 'razon': str(summary['reason'])})
    return {'transferred': True, 'receipt': confirmed_receipt(result, session.client_id)}
