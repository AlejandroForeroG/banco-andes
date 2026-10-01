"""Controles del consentimiento y de la confirmación de radicación."""
import json
from types import SimpleNamespace

import pytest

from agent.human import confirmed_receipt, transfer, transfer_summary

pytestmark = pytest.mark.review


def test_continuacion_voluntaria_conserva_caso_validado():
    review = {'task_id': 'caso-revisado', 'status': 'validado',
              'exchanges': [{'peticion': {'borrador': 'Orientación revisada.', 'evidencias': []},
                             'respuesta': {'status': 'validado', 'hallazgos': []}}]}
    summary = transfer_summary(review)
    assert summary['task_id'] == review['task_id']
    assert summary['estado'] == 'validado'
    assert summary['causa'] == 'SOLICITUD_USUARIO'
    assert summary['orientacion_revisada'] == review['exchanges'][0]['peticion']['borrador']
    assert 'borrador_bloqueado' not in summary


@pytest.mark.parametrize('result', ['Error de MCP', {'error': 'API no disponible'},
    [{'type': 'text', 'text': 'No autorizado'}],
    {'id': 'ESC-001', 'cliente_id': 'otro', 'creado_en': '2026-09-24T12:00:00Z'}])
def test_error_o_cliente_ajeno_no_es_radicado(result):
    with pytest.raises(RuntimeError, match='confirmación válida'):
        confirmed_receipt(result, 'CLI-2001')


async def test_radicado_solo_despues_de_respuesta_confirmada(monkeypatch):
    monkeypatch.setattr('agent.human.required_session', lambda: SimpleNamespace(client_id='CLI-2001'))
    receipt = {'id': 'ESC-TEST', 'cliente_id': 'CLI-2001', 'creado_en': '2026-09-24T12:00:00Z',
               'tiempo_estimado_horas': 24}
    calls = []
    async def invoke(args):
        calls.append(args)
        return [{'type': 'text', 'text': json.dumps(receipt)}]
    tool = SimpleNamespace(name='registrar_escalamiento', ainvoke=invoke)
    result = await transfer({'status': 'solicitada'}, True, [tool])
    assert len(calls) == 1
    assert result == {'transferido': True, 'radicado': {
        'id': 'ESC-TEST', 'creado_en': receipt['creado_en'], 'tiempo_estimado_horas': 24}}
