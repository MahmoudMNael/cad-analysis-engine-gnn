"""Apply SQL migrations from an empty database."""

from __future__ import annotations

import logging
from pathlib import Path

import psycopg
from psycopg import Connection

from packages.common.configuration import Settings, get_settings

log = logging.getLogger(__name__)
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def apply_migrations(conn: Connection) -> list[str]:
    applied: list[str] = []
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute("SELECT version FROM schema_migrations")
        done = {row[0] for row in cur.fetchall()}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = path.name
            if version in done:
                continue
            sql = path.read_text(encoding="utf-8")
            cur.execute(sql)
            cur.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied.append(version)
            log.info("applied_migration", extra={"stage": "migrate", "block": version})
    conn.commit()
    return applied


def migrate(settings: Settings | None = None) -> list[str]:
    cfg = settings or get_settings()
    with psycopg.connect(cfg.database_url) as conn:
        return apply_migrations(conn)


def main() -> None:
    applied = migrate()
    print("applied:" + (",".join(applied) if applied else " none"))


if __name__ == "__main__":
    main()
