"""End-to-end integration tests against real PostgreSQL + MinIO.

NOT part of the default pytest run. Requires the backing services up:

    docker compose -f tests/integration/docker-compose-pg-minio.yml up -d

Verifies the two non-default deployment code paths that the default
SQLite + local-fs suite never exercises:
  - db_url = postgresql+asyncpg://... (async SQLAlchemy against a real PG)
  - storage_type = minio (MinioStore chunked upload + assemble + get/delete)

Skip automatically when either service is unreachable so the default suite
stays green. Drivers (asyncpg, minio) are in the optional `integration` extra;
if they are not installed, the whole module skips.

Auth OFF (these tests exercise storage + DB backends, not Hub RBAC). MLX is
NOT required — we only validate that model/version records persist to PG and
file artifacts land in MinIO and are retrievable.
"""

import contextlib
import os
import shutil
import subprocess
import time

import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.auth import set_auth_enabled
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import get_store, init_deps

COMPOSE_FILE = os.path.join(os.path.dirname(__file__), "integration", "docker-compose-pg-minio.yml")
PG_HOST = os.environ.get("FMH_INT_PG_HOST", "127.0.0.1")
PG_PORT = int(os.environ.get("FMH_INT_PG_PORT", "5433"))
PG_URL = f"postgresql+asyncpg://fmh:fmh@{PG_HOST}:{PG_PORT}/fmh"
MINIO_ENDPOINT = os.environ.get("FMH_INT_MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_ACCESS_KEY = os.environ.get("FMH_INT_MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("FMH_INT_MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("FMH_INT_MINIO_BUCKET", "fusion-models")

# Compose-managed lifecycle: the test harness can bring the stack up/down so a
# developer running the suite from cold gets a self-contained run. Disabled if
# FMH_INT_NO_COMPOSE=1 (services already up) or docker unavailable.
_MANAGE_COMPOSE = os.environ.get("FMH_INT_NO_COMPOSE", "") != "1"


def _have_drivers() -> bool:
    try:
        import asyncpg  # noqa: F401
        import minio  # noqa: F401

        return True
    except ImportError:
        return False


def _docker_available() -> bool:
    try:
        r = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0 and r.stdout.strip() != ""
    except (OSError, subprocess.SubprocessError):
        return False


def _pg_reachable() -> bool:
    import socket

    with contextlib.suppress(OSError):
        with socket.create_connection((PG_HOST, PG_PORT), timeout=2.0):
            return True
    return False


def _minio_reachable() -> bool:
    try:
        host, port = MINIO_ENDPOINT.split(":")
        import socket

        with socket.create_connection((host, int(port)), timeout=2.0):
            return True
    except (OSError, ValueError):
        return False


_DRIVERS_OK = _have_drivers()
_DOCKER_OK = _docker_available()
_PG_OK = _pg_reachable()
_MINIO_OK = _minio_reachable()


def _compose_up() -> bool:
    r = subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d", "--wait"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if r.returncode != 0:
        print(f"[pg_minio] compose up failed: {r.stderr[-400:]}")
        return False
    return True


def _compose_down() -> None:
    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "down", "-v"],
        capture_output=True,
        text=True,
        timeout=60,
    )


# Compose-managed bring-up: only attempt if docker present, services NOT already
# reachable (avoid double-managing an externally-provided stack), and compose
# file exists. Best-effort; failures fall through to the skip below.
_COMPOSE_MANAGED = False
if _MANAGE_COMPOSE and _DOCKER_OK and not (_PG_OK and _MINIO_OK) and os.path.exists(COMPOSE_FILE):
    _COMPOSE_MANAGED = _compose_up()
    if _COMPOSE_MANAGED:
        # Re-probe after bring-up.
        _PG_OK = _pg_reachable()
        _MINIO_OK = _minio_reachable()


_SKIP_REASON = (
    "PG+MinIO integration tests skipped: "
    f"drivers={_DRIVERS_OK} pg={_PG_OK} minio={_MINIO_OK} "
    "(run: docker compose -f tests/integration/docker-compose-pg-minio.yml up -d "
    "&& pip install -e '.[integration]')"
)
requires_pg_minio = pytest.mark.skipif(
    not (_DRIVERS_OK and _PG_OK and _MINIO_OK),
    reason=_SKIP_REASON,
)


@pytest.fixture
async def pg_minio_client():
    # Auth OFF (exercising backends, not RBAC). PG + MinIO wiring through the
    # real Settings + init_deps so MinioStore is selected and PG engine created.
    # Unique tmp data_dir per run for the cache/local fallback dirs.
    data_dir = f"/tmp/fmh_int_pgminio_{int(time.time())}"
    set_auth_enabled(False)
    s = Settings(
        host="127.0.0.1",
        port=11444,
        data_dir=data_dir,
        db_url=PG_URL,
        log_level="WARNING",
        storage_type="minio",
        minio_endpoint=MINIO_ENDPOINT,
        minio_access_key=MINIO_ACCESS_KEY,
        minio_secret_key=MINIO_SECRET_KEY,
        minio_bucket=MINIO_BUCKET,
        minio_secure=False,
    )
    engine = get_engine(s.db_url)
    await init_db(engine)
    init_deps(s, engine)
    app = create_app(s)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await engine.dispose()
    set_auth_enabled(False)
    shutil.rmtree(data_dir, ignore_errors=True)


class TestPgBackend:
    # The Hub must persist Model/Version rows to PostgreSQL through async
    # SQLAlchemy — the migration path that's only a db_url swap in theory.

    @pytest.mark.asyncio
    @requires_pg_minio
    async def test_model_crud_persists_to_pg(self, pg_minio_client):
        create = await pg_minio_client.post(
            "/api/v1/models",
            json={
                "name": "pg-int-model",
                "model_type": "llm",
                "hf_repo": "test/pg-int",
            },
        )
        assert create.status_code == 201, create.text
        model_id = create.json()["id"]
        try:
            got = await pg_minio_client.get(f"/api/v1/models/{model_id}")
            assert got.status_code == 200
            assert got.json()["name"] == "pg-int-model"
            # Empty version round-trips to PG too (file_path empty -> MinIO
            # untouched for this record).
            ver = await pg_minio_client.post(
                f"/api/v1/models/{model_id}/versions",
                data={"version": "1.0.0", "format": "mlx", "quantization": "4bit"},
                files={"file": ("", b"")},
            )
            assert ver.status_code == 201, ver.text
            assert ver.json()["version"] == "1.0.0"
            # List must read back from PG.
            lst = await pg_minio_client.get("/api/v1/models")
            assert any(m["id"] == model_id for m in lst.json().get("items", lst.json()))
        finally:
            await pg_minio_client.delete(f"/api/v1/models/{model_id}")

    @pytest.mark.asyncio
    @requires_pg_minio
    async def test_pg_survives_separate_session(self, pg_minio_client):
        # A row written in one request must be visible in a fresh session —
        # the async sessionmaker must not leak an uncommitted transaction.
        create = await pg_minio_client.post(
            "/api/v1/models",
            json={
                "name": "pg-persist",
                "model_type": "llm",
            },
        )
        assert create.status_code == 201, create.text
        model_id = create.json()["id"]
        try:
            # New HTTP request = new session from the factory; commit must have
            # landed or this 404s.
            got = await pg_minio_client.get(f"/api/v1/models/{model_id}")
            assert got.status_code == 200, "row not visible across sessions"
        finally:
            await pg_minio_client.delete(f"/api/v1/models/{model_id}")


class TestMinioStorage:
    # MinioStore chunked upload -> assemble -> object exists -> delete_version.
    # Exercises the object-storage path that LocalStore never touches.

    @pytest.mark.asyncio
    @requires_pg_minio
    async def test_chunked_upload_assembles_in_minio(self, pg_minio_client):
        create = await pg_minio_client.post(
            "/api/v1/models",
            json={
                "name": "minio-int-model",
                "model_type": "llm",
            },
        )
        assert create.status_code == 201, create.text
        model_id = create.json()["id"]
        try:
            # 2 chunks of 8 bytes each -> one 16-byte object on MinIO.
            for idx, payload in enumerate([b"AAAABBBB", b"CCCCDDDD"]):
                resp = await pg_minio_client.post(
                    f"/api/v1/models/{model_id}/versions/chunk-upload",
                    data={
                        "version": "2.0.0",
                        "format": "mlx",
                        "quantization": "4bit",
                        "filename": "model.bin",
                        "total_chunks": "2",
                        "chunk_index": str(idx),
                    },
                    files={"chunk": ("model.bin", payload, "application/octet-stream")},
                )
                if idx < 1:
                    assert resp.status_code == 201, resp.text
                    assert resp.json()["status"] == "chunk_received"
                else:
                    assert resp.status_code == 201, resp.text
                    ver = resp.json()
                    assert ver["version"] == "2.0.0"
                    assert ver["file_size"] == 16
                    assert ver["file_hash"], "assembled version has no hash"
                    assert ver["file_path"], "version has no object path"

            # The object must exist on MinIO (stat, not local fs).
            store = get_store()
            from fusion_model_hub.storage.minio_store import MinioStore

            assert isinstance(store, MinioStore), "init_deps did not wire MinioStore"
            obj = store.get_file(ver["file_path"])
            assert obj is not None, "assembled object not found on MinIO"

            # delete_version_files must remove the object.
            removed = store.delete_version_files(model_id, "2.0.0")
            assert removed is True
            assert store.get_file(ver["file_path"]) is None
        finally:
            await pg_minio_client.delete(f"/api/v1/models/{model_id}")


def pytest_sessionfinish(session, exitstatus):
    # Tear down a compose stack this module brought up. Never touch a stack that
    # was already running (FMH_INT_NO_COMPOSE=1 or services pre-reachable).
    if _COMPOSE_MANAGED:
        _compose_down()
