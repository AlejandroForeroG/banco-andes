"""Transferencia mínima al asesor: cierre automático, consentimiento y radicado."""
import json

from agent.session import required_session


def transfer_summary(review):
    if not review:
        raise ValueError('No hay una revisión para transferir.')
    exchanges = review.get('exchanges', [])
    last = exchanges[-1] if exchanges else {}

    validated = review.get('status') == 'validado'
    return {'task_id': review.get('task_id'), 'estado': review['status'],
            'orientacion_revisada' if validated else 'borrador_bloqueado': last.get('peticion', {}).get('borrador', ''),
            'evidencias': last.get('peticion', {}).get('evidencias', []),
            'hallazgos': last.get('respuesta', {}).get('hallazgos', []),
            'causa': 'SOLICITUD_USUARIO' if validated else
                     review.get('error') or last.get('respuesta', {}).get('error') or review['status']}


def confirmed_receipt(result, client_id):
    """MCP puede devolver bloques de texto, incluidos errores: no son un radicado."""
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
        raise RuntimeError('No se recibió confirmación válida del asesor; no se puede afirmar que hubo transferencia.')
    return {key: result[key] for key in ('id', 'creado_en', 'tiempo_estimado_horas') if key in result}


async def transfer(review, consent, tools):
    if not consent:
        return {'transferido': False}
    session = required_session()
    summary = transfer_summary(review)
    tool = next((t for t in tools if t.name == 'registrar_escalamiento'), None)
    if tool is None:
        raise RuntimeError('El canal de asesores MCP no está disponible.')
    result = await tool.ainvoke({'intencion': 'Continuar la orientación con un asesor autorizado',
                               'informacion_recopilada': summary, 'razon': str(summary['causa'])})
    return {'transferido': True, 'radicado': confirmed_receipt(result, session.client_id)}
