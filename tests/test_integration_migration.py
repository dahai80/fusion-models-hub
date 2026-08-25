"""Migration schema-consistency gate.

NOT part of the default pytest run (excluded via addopts --ignore in
pyproject.toml). Requires a real PostgreSQL up:

    docker compose -f tests/integration/docker-compose-pg-minio.yml up -d

Run explicitly:

    pytest tests/test_integration_migration.py -v -o addopts=""

Purpose: prevent the ORM-vs-Alembic migration divergence that slipped past
the default suite for two audit rounds. The default tests run `init_db`
(create_all from Base.metadata) on SQLite and NEVER run migrations, so a
hand-written migration can silently drift from the ORM until a real PG
deployment tries `alembic upgrade head` and crashes. This gate enforces the
single-source-of-truth contract: the schema produced by `alembic upgrade head`
on a fresh PG MUST equal the schema produced by `Base.metadata.create_all` on
a fresh PG. Any difference (missing table, missing column, type/null/FK/unique
mismatch) fails the gate.

Skips automatically when PG is unreachable or the psycopg driver is missing,
so the default suite stays green. PG is at the compose-mapped port 5433.
"""

import contextlib
import os
import subprocess

import pytest
from sqlalchemy import create_engine, inspect

from fusion_model_hub.db.models import Base

PG_HOST = os.environ.get("FMH_INT_PG_HOST", "127.0.0.1")
PG_PORT = int(os.environ.get("FMH_INT_PG_PORT", "5433"))
PG_USER = os.environ.get("FMH_INT_PG_USER", "fmh")
PG_PASS = os.environ.get("FMH_INT_PG_PASS", "fmh")
SYNC_PG_URL = f"postgresql+psycopg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}"

pytest.importorskip("psycopg")


def _pg_reachable() -> bool:
    try:
        eng = create_engine(SYNC_PG_URL + "/postgres", pool_pre_ping=True)
        with eng.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        eng.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pg_reachable(), reason="PG unreachable for migration gate")


def _psql(sql: str) -> None:
    env = os.environ.copy()
    env["PGPASSWORD"] = PG_PASS
    subprocess.run(
        ["psql", "-h", PG_HOST, "-p", str(PG_PORT), "-U", PG_USER, "-d", "postgres", "-v", "ON_ERROR_STOP=1", "-c", sql],
        check=True,
        capture_output=True,
        env=env,
    )


def _migration_schema_db() -> str:
    return "fmh_mig_gate"


def _orm_schema_db() -> str:
    return "fmh_orm_gate"


@contextlib.contextmanager
def _scratch_db(name: str):
    _psql(f"DROP DATABASE IF EXISTS {name};")
    _psql(f"CREATE DATABASE {name};")
    try:
        yield name
    finally:
        _psql(f"DROP DATABASE IF EXISTS {name};")


def test_migration_matches_orm_schema() -> None:
    """alembic upgrade head schema == Base.metadata.create_all schema on PG."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    with _scratch_db(_migration_schema_db()) as mig_db, _scratch_db(_orm_schema_db()) as orm_db:
        # 1. migration path: alembic upgrade head against a fresh PG db
        env = os.environ.copy()
        env["FMH_ALEMBIC_URL"] = f"{SYNC_PG_URL}/{mig_db}"
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, (
            f"alembic upgrade head failed on fresh PG:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )

        # 2. ORM path: Base.metadata.create_all against a fresh PG db
        orm_engine = create_engine(f"{SYNC_PG_URL}/{orm_db}")
        Base.metadata.create_all(orm_engine)

        # 3. inspect both and diff
        mig_engine = create_engine(f"{SYNC_PG_URL}/{mig_db}")
        orm_ins = inspect(orm_engine)
        mig_ins = inspect(mig_engine)

        orm_tables = set(Base.metadata.tables.keys())
        mig_tables = {t for t in mig_ins.get_table_names() if t != "alembic_version"}

        failures = []

        only_orm = sorted(orm_tables - mig_tables)
        if only_orm:
            failures.append(f"tables only in ORM (missing from migration): {only_orm}")
        only_mig = sorted(mig_tables - orm_tables)
        if only_mig:
            failures.append(f"tables only in migration (not in ORM): {only_mig}")

        for t in sorted(orm_tables & mig_tables):
            orm_cols = {c["name"]: c for c in orm_ins.get_columns(t)}
            mig_cols = {c["name"]: c for c in mig_ins.get_columns(t)}

            for name in sorted(set(orm_cols) - set(mig_cols)):
                failures.append(f"[{t}] column {name!r} in ORM but missing from migration")
            for name in sorted(set(mig_cols) - set(orm_cols)):
                failures.append(f"[{t}] column {name!r} in migration but missing from ORM")

            for name in sorted(set(orm_cols) & set(mig_cols)):
                oc, mc = orm_cols[name], mig_cols[name]
                if str(oc["type"]).upper() != str(mc["type"]).upper():
                    failures.append(
                        f"[{t}.{name}] type mismatch ORM={oc['type']} migration={mc['type']}"
                    )
                if bool(oc.get("nullable")) != bool(mc.get("nullable")):
                    failures.append(
                        f"[{t}.{name}] nullable mismatch ORM={oc.get('nullable')} migration={mc.get('nullable')}"
                    )

            orm_fks = {
                tuple(sorted(f["constrained_columns"])): f["referred_table"]
                for f in orm_ins.get_foreign_keys(t)
            }
            mig_fks = {
                tuple(sorted(f["constrained_columns"])): f["referred_table"]
                for f in mig_ins.get_foreign_keys(t)
            }
            if orm_fks != mig_fks:
                failures.append(f"[{t}] FK mismatch ORM={orm_fks} migration={mig_fks}")

            orm_uq = {tuple(sorted(c["column_names"])) for c in orm_ins.get_unique_constraints(t)}
            mig_uq = {tuple(sorted(c["column_names"])) for c in mig_ins.get_unique_constraints(t)}
            if orm_uq != mig_uq:
                failures.append(f"[{t}] unique-constraint mismatch ORM={orm_uq} migration={mig_uq}")

        orm_engine.dispose()
        mig_engine.dispose()

        assert not failures, (
            "Migration schema diverges from ORM schema (single-source-of-truth broken):\n  - "
            + "\n  - ".join(failures)
        )
