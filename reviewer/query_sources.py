'Read-only parameterized reviewer source queries.'
import argparse
from datetime import date
import json
from pathlib import Path
import sqlite3

BASE = Path(__file__).resolve().parent
DEFAULT_DB = BASE / 'data' / 'policies.sqlite3'
FILTRO = """
FROM policies p JOIN criteria c
  ON c.policy_id = p.id AND c.policy_version = p.version
WHERE p.effective_from <= :case_date
  AND (p.effective_until IS NULL OR :case_date < p.effective_until)
  AND c.scope IN ('general', :scope)
"""
CONSULTA_CRITERIOS = """SELECT p.id AS policy, p.version, p.provenance,
    c.id AS criterion, c.scope, c.rule, c.action
""" + FILTRO + ' ORDER BY c.id'
CONSULTA_EJEMPLOS = """SELECT e.* FROM examples e WHERE e.criterion_id IN (
    SELECT c.id """ + FILTRO + ') ORDER BY e.id'


def conectar_lectura(db):
    conexion = sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True)
    conexion.execute('PRAGMA query_only = ON')
    conexion.row_factory = sqlite3.Row
    return conexion


def consultar(db=DEFAULT_DB, case_date='2026-09-08', scope='housing'):
    if date.fromisoformat(case_date).isoformat() != case_date:
        raise ValueError('The date must use YYYY-MM-DD format')
    if scope not in ('general', 'housing'):
        raise ValueError('Allowed scope: general or housing')
    parametros = {'case_date': case_date, 'scope': scope}
    conexion = conectar_lectura(db)
    try:
        criteria = [dict(fila) for fila in conexion.execute(CONSULTA_CRITERIOS, parametros)]
        examples = [dict(fila) for fila in conexion.execute(CONSULTA_EJEMPLOS, parametros)]
    finally:
        conexion.close()
    return {'source': 'SYNTHETIC DEMO DATA', 'case_date': case_date, 'scope': scope,
            'source_state': 'available' if criteria else 'no_current_criteria',
            'criteria': criteria, 'examples': examples}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    parser.add_argument('--date', dest='case_date', required=True, help='Case date in ISO YYYY-MM-DD format')
    parser.add_argument('--scope', dest='scope', choices=['general', 'housing'], default='housing')
    args = parser.parse_args()
    print(json.dumps(consultar(args.db, args.case_date, args.scope), ensure_ascii=False, indent=2))
