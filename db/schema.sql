-- ── Admission Bot — Database Schema ────────────────────────────────

CREATE TABLE IF NOT EXISTS user_profile (
    id              INTEGER PRIMARY KEY,
    name            TEXT    DEFAULT 'Student',
    level           TEXT,
    field           TEXT,
    nationality     TEXT,
    domicile        TEXT,
    gpa             REAL,
    pending_updates TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT,
    role             TEXT,
    message          TEXT,
    intent           TEXT,
    sources_referenced TEXT,
    timestamp        TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Seed default profile
INSERT OR IGNORE INTO user_profile (id, name) VALUES (1, 'Student');
