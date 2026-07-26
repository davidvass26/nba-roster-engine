"""DuckDB connection management.

DuckDB rather than Postgres because the whole warehouse is a single file
you can commit-adjacent, back up, and ship inside a container. The
workload is analytical scans over ~10^6-10^7 rows, which is exactly what
it is built for, and there is no server to operate.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator

import duckdb

from nbare.config import DB_PATH

SCHEMA_SQL = Path(__file__).with_name("schema.sql")


def connect(path: Path | str | None = None, read_only: bool = False):
    """Open a connection, applying the schema on first use."""
    target = str(path or DB_PATH)
    con = duckdb.connect(target, read_only=read_only)
    if not read_only:
        apply_schema(con)
    return con


@contextlib.contextmanager
def session(path: Path | str | None = None, read_only: bool = False) -> Iterator:
    con = connect(path, read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def apply_schema(con) -> None:
    """Idempotent: every statement is CREATE ... IF NOT EXISTS."""
    con.execute(SCHEMA_SQL.read_text())


def table_counts(con) -> dict[str, int]:
    """Row counts for every table, for the `make status` sanity check."""
    rows = con.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema IN ('raw', 'stg', 'mart')
        ORDER BY table_schema, table_name
        """
    ).fetchall()
    out: dict[str, int] = {}
    for schema, name in rows:
        n = con.execute(f'SELECT count(*) FROM "{schema}"."{name}"').fetchone()[0]
        out[f"{schema}.{name}"] = n
    return out
