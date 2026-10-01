"""Consulta parametrizada de las fuentes propias que usa el especialista."""
import argparse
from datetime import date
import json
from pathlib import Path
import sqlite3

BASE = Path(__file__).resolve().parent
DEFAULT_DB = BASE / 'data' / 'politicas.sqlite3'
FILTRO = """
FROM politicas p JOIN criterios c
  ON c.politica_id = p.id AND c.politica_version = p.version
WHERE p.vigente_desde <= :fecha
  AND (p.vigente_hasta IS NULL OR :fecha < p.vigente_hasta)
  AND c.ambito IN ('general', :ambito)
"""
CONSULTA_CRITERIOS = """SELECT p.id AS politica, p.version, p.procedencia,
    c.id AS criterio, c.ambito, c.regla, c.accion
""" + FILTRO + ' ORDER BY c.id'
CONSULTA_EJEMPLOS = """SELECT e.* FROM ejemplos e WHERE e.criterio_id IN (
    SELECT c.id """ + FILTRO + ') ORDER BY e.id'


def conectar_lectura(db):
    conexion = sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True)
    conexion.execute('PRAGMA query_only = ON')
    conexion.row_factory = sqlite3.Row
    return conexion


def consultar(db=DEFAULT_DB, fecha='2026-09-08', ambito='vivienda'):
    if date.fromisoformat(fecha).isoformat() != fecha:
        raise ValueError('La fecha debe tener formato YYYY-MM-DD')
    if ambito not in ('general', 'vivienda'):
        raise ValueError('Ámbito permitido: general o vivienda')
    parametros = {'fecha': fecha, 'ambito': ambito}
    conexion = conectar_lectura(db)
    try:
        criterios = [dict(fila) for fila in conexion.execute(CONSULTA_CRITERIOS, parametros)]
        ejemplos = [dict(fila) for fila in conexion.execute(CONSULTA_EJEMPLOS, parametros)]
    finally:
        conexion.close()
    return {'fuente': 'SINTÉTICA DE DEMOSTRACIÓN', 'fecha': fecha, 'ambito': ambito,
            'estado_fuentes': 'disponibles' if criterios else 'sin_criterios_vigentes',
            'criterios': criterios, 'ejemplos': ejemplos}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    parser.add_argument('--fecha', required=True, help='Fecha ISO YYYY-MM-DD del caso')
    parser.add_argument('--ambito', choices=['general', 'vivienda'], default='vivienda')
    args = parser.parse_args()
    print(json.dumps(consultar(args.db, args.fecha, args.ambito), ensure_ascii=False, indent=2))
