PRAGMA foreign_keys = ON;
CREATE TABLE policies (
    id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    effective_from TEXT NOT NULL,
    effective_until TEXT,
    provenance TEXT NOT NULL,
    PRIMARY KEY (id, version),
    CHECK (effective_until IS NULL OR effective_until > effective_from)
);
CREATE TABLE criteria (
    id TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL,
    policy_version INTEGER NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('general', 'housing')),
    rule TEXT NOT NULL,
    action TEXT NOT NULL,
    FOREIGN KEY (policy_id, policy_version) REFERENCES policies(id, version)
);
CREATE TABLE examples (
    id TEXT PRIMARY KEY,
    criterion_id TEXT NOT NULL REFERENCES criteria(id),
    draft TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('validated', 'correct', 'insufficient')),
    correction TEXT NOT NULL
);
