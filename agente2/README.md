# Revisor editorial

Servicio independiente con HTTP POST y eventos SSE firmados con HMAC-SHA256. Implementa un protocolo de aplicación propio; no implementa el estándar A2A.

`critic.py` consulta políticas sintéticas versionadas en SQLite y evalúa la respuesta del orientador. `server.py` valida la petición y transmite el dictamen; `protocol.py` define el contrato y las firmas. La base contiene dos versiones de política, cinco criterios y tres ejemplos. No representa normativa bancaria real.

La consulta usa dos SELECT parametrizados en modo de solo lectura. El servicio calcula el estado a partir de la cobertura de criterios y los hallazgos. El orientador verifica firma, correlación, plazo y coherencia antes de mostrar el texto validado.

El ciclo permite dos críticas y un refinamiento con 90 segundos totales y 30 segundos por llamada. La segunda revisión reutiliza la instantánea de políticas y el plazo original.

```bash
python -m agente2.montar_fuentes
python -m agente2.consultar_fuentes --fecha 2026-09-23 --ambito vivienda
python -m unittest agente2.test_fuentes -v
```

El montaje no sobrescribe una base existente. Usa `--db RUTA` para una copia nueva. Una fuente rota produce `fallida/FUENTE_NO_DISPONIBLE`.
