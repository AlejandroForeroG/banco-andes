from fastapi.testclient import TestClient

from app.database import DB
from app.main import app


def test_financial_demo_flow():
    with TestClient(app) as client:
        assert client.get('/health').json() == {'status': 'ok'}
        customer = DB['finanzas']['clientes'][0]
        profile_path = f"/finanzas/clientes/{customer['id']}/perfil-financiero"
        assert client.get(profile_path).status_code == 401
        login = client.post('/finanzas/auth/login', json={
            'numero_identificacion': customer['numero_identificacion'],
            'clave': customer['clave'],
        })
        assert login.status_code == 200
        token = login.json()['token']
        profile = client.get(profile_path, headers={'Authorization': f'Bearer {token}'})
        assert profile.status_code == 200
        assert 'clave' not in profile.json()
        assert client.get('/finanzas/productos').json()
        escalation = client.post('/finanzas/escalamientos', json={
            'cliente_id': customer['id'],
            'intencion': 'Orientación de vivienda',
            'informacion_recopilada': {'consentimiento': True},
            'razon': 'Solicitud de asesor',
        })
        assert escalation.status_code == 201
        assert escalation.json()['id'].startswith('ESC-')
        paths = client.get('/openapi.json').json()['paths']
        assert all(path == '/health' or path.startswith('/finanzas/') for path in paths)
