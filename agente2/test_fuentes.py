"""Verificación de montaje, temporalidad, integridad y acceso de solo lectura."""
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest

from agente2.montar_fuentes import montar
from agente2.consultar_fuentes import conectar_lectura, consultar


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
        self.assertEqual(len(consultar(self.db)['criterios']), 4)

    def test_frontera_de_version_y_ausencia_de_fuentes(self):
        historico = consultar(self.db, '2026-08-31')
        vigente = consultar(self.db, '2026-09-01')
        self.assertEqual([c['criterio'] for c in historico['criterios']], ['H01'])
        self.assertEqual({c['version'] for c in vigente['criterios']}, {2})
        self.assertEqual(consultar(self.db, '2026-07-01')['estado_fuentes'], 'sin_criterios_vigentes')

    def test_ambito_y_ejemplo_consultable(self):
        resultado = consultar(self.db)
        self.assertEqual({c['criterio'] for c in resultado['criterios']}, {'C01', 'C02', 'C03', 'C04'})
        self.assertEqual(resultado['ejemplos'][0]['dictamen'], 'corregir')
        self.assertEqual(len(consultar(self.db, ambito='general')['criterios']), 3)
        with self.assertRaises(ValueError):
            consultar(self.db, ambito="vivienda'); DROP TABLE politicas; --")

    def test_fecha_debe_ser_iso_canonica(self):
        for fecha in ('20260908', '2026-02-30', "2026-09-08' OR 1=1 --"):
            with self.subTest(fecha=fecha), self.assertRaises(ValueError):
                consultar(self.db, fecha=fecha)

    def test_lectura_no_permite_modificar_ni_crear(self):
        conexion = conectar_lectura(self.db)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                conexion.execute("DELETE FROM ejemplos")
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
