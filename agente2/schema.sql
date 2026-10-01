PRAGMA foreign_keys = ON;
CREATE TABLE politicas (
    id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    vigente_desde TEXT NOT NULL,
    vigente_hasta TEXT,
    procedencia TEXT NOT NULL,
    PRIMARY KEY (id, version),
    CHECK (vigente_hasta IS NULL OR vigente_hasta > vigente_desde)
);
CREATE TABLE criterios (
    id TEXT PRIMARY KEY,
    politica_id TEXT NOT NULL,
    politica_version INTEGER NOT NULL,
    ambito TEXT NOT NULL CHECK (ambito IN ('general', 'vivienda')),
    regla TEXT NOT NULL,
    accion TEXT NOT NULL,
    FOREIGN KEY (politica_id, politica_version) REFERENCES politicas(id, version)
);
CREATE TABLE ejemplos (
    id TEXT PRIMARY KEY,
    criterio_id TEXT NOT NULL REFERENCES criterios(id),
    borrador TEXT NOT NULL,
    dictamen TEXT NOT NULL CHECK (dictamen IN ('validado', 'corregir', 'insuficiente')),
    correccion TEXT NOT NULL
);
