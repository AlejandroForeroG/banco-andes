'Initialize synthetic policy sources without invoking a model.'
import argparse
from pathlib import Path
import sqlite3

BASE = Path(__file__).resolve().parent
DEFAULT_DB = BASE / 'data' / 'policies.sqlite3'


def montar(destino=DEFAULT_DB):
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with destino.open('xb'):
        pass
    try:
        with sqlite3.connect(destino) as conexion:
            conexion.executescript((BASE / 'schema.sql').read_text(encoding='utf-8'))
            conexion.executescript((BASE / 'seed.sql').read_text(encoding='utf-8'))
            if conexion.execute('PRAGMA foreign_key_check').fetchall():
                raise ValueError('Invalid seed references')
            conteos = {tabla: conexion.execute(f'SELECT count(*) FROM {tabla}').fetchone()[0]
                       for tabla in ('policies', 'criteria', 'examples')}
    except Exception:
        if 'conexion' in locals():
            conexion.close()
        destino.unlink(missing_ok=True)
        raise
    finally:
        if 'conexion' in locals():
            conexion.close()
    return conteos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    try:
        print({'source': 'SYNTHETIC DEMO DATA', 'db': str(args.db), 'filas': montar(args.db)})
    except FileExistsError:
        parser.exit(1, 'The database already exists; query it or choose another --db.\n')
