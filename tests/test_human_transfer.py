'Capability and regression tests.'
import json
from types import SimpleNamespace

import pytest

from agent.human import confirmed_receipt, transfer, transfer_summary

pytestmark = pytest.mark.review


def test_continuacion_voluntaria_conserva_caso_validado():
    review = {'task_id': 'reviewed-case', 'status': 'validated',
              'exchanges': [{'request': {'draft': 'Reviewed guidance.', 'evidence': []},
                             'response': {'status': 'validated', 'findings': []}}]}
    summary = transfer_summary(review)
    assert summary['task_id'] == review['task_id']
    assert summary['state'] == 'validated'
    assert summary['reason'] == 'USER_REQUEST'
    assert summary['reviewed_guidance'] == review['exchanges'][0]['request']['draft']
    assert 'blocked_draft' not in summary


@pytest.mark.parametrize('result', ['MCP error', {'error': 'API unavailable'},
    [{'type': 'text', 'text': 'No autorizado'}],
    {'id': 'ESC-001', 'cliente_id': 'otro', 'creado_en': '2026-09-24T12:00:00Z'}])
def test_error_o_cliente_ajeno_no_es_radicado(result):
    with pytest.raises(RuntimeError, match='valid advisor receipt'):
        confirmed_receipt(result, 'CLI-2001')


async def test_radicado_solo_despues_de_respuesta_confirmada(monkeypatch):
    monkeypatch.setattr('agent.human.required_session', lambda: SimpleNamespace(client_id='CLI-2001'))
    receipt = {'id': 'ESC-TEST', 'cliente_id': 'CLI-2001', 'creado_en': '2026-09-24T12:00:00Z',
               'tiempo_estimado_horas': 24}
    calls = []
    async def invoke(args):
        calls.append(args)
        return [{'type': 'text', 'text': json.dumps(receipt)}]
    tool = SimpleNamespace(name='register_handoff', ainvoke=invoke)
    result = await transfer({'status': 'requested'}, True, [tool])
    assert len(calls) == 1
    assert result == {'transferred': True, 'receipt': {
        'id': 'ESC-TEST', 'creado_en': receipt['creado_en'], 'tiempo_estimado_horas': 24}}
