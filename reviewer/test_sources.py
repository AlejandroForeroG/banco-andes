'Source initialization, validity, integrity, and read-only access tests.'
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest

from reviewer.init_sources import montar
from reviewer.query_sources import conectar_lectura, consultar


class FuentesTest(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.db = Path(self.temporal.name) / 'fuentes.sqlite3'
        montar(self.db)

    def test_montaje_e_integridad(self):
        with closing(sqlite3.connect(self.db)) as conexion:
            self.assertEqual(conexion.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(conexion.execute('PRAGMA foreign_key_check').fetchall(), [])
        with self.assertRaises(FileExistsError):
            montar(self.db)
        self.assertEqual(len(consultar(self.db)['criteria']), 4)

    def test_frontera_de_version_y_ausencia_de_fuentes(self):
        historico = consultar(self.db, '2026-08-31')
        vigente = consultar(self.db, '2026-09-01')
        self.assertEqual([c['criterion'] for c in historico['criteria']], ['H01'])
        self.assertEqual({c['version'] for c in vigente['criteria']}, {2})
        self.assertEqual(consultar(self.db, '2026-07-01')['source_state'], 'no_current_criteria')

    def test_ambito_y_ejemplo_consultable(self):
        result = consultar(self.db)
        self.assertEqual({c['criterion'] for c in result['criteria']}, {'C01', 'C02', 'C03', 'C04'})
        self.assertEqual(result['examples'][0]['verdict'], 'correct')
        self.assertEqual(len(consultar(self.db, scope='general')['criteria']), 3)
        with self.assertRaises(ValueError):
            consultar(self.db, scope="housing'); DROP TABLE policies; --")

    def test_fecha_debe_ser_iso_canonica(self):
        for case_date in ('20260908', '2026-02-30', "2026-09-08' OR 1=1 --"):
            with self.subTest(case_date=case_date), self.assertRaises(ValueError):
                consultar(self.db, case_date=case_date)

    def test_lectura_no_permite_modificar_ni_crear(self):
        conexion = conectar_lectura(self.db)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                conexion.execute("DELETE FROM examples")
            with self.assertRaises(sqlite3.OperationalError):
                conexion.execute("CREATE TABLE intrusa (id INTEGER)")
        finally:
            conexion.close()
        inexistente = self.db.parent / 'inexistente.sqlite3'
        with self.assertRaises(sqlite3.OperationalError):
            conectar_lectura(inexistente)
        self.assertFalse(inexistente.exists())


if __name__ == '__main__':
    unittest.main()
