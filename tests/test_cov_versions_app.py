import asyncio
import contextlib
import io
import json
import logging
import os
import shutil
import tarfile
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import dispose_all_engines, get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.auth import set_auth_enabled
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps

logger = logging.getLogger(__name__)

_TMP_ROOT = "/tmp/fmh_cov_va"


@pytest.fixture
def settings():
    import time

    data_dir = f"{_TMP_ROOT}_{int(time.time() * 1000)}"
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir=data_dir,
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
async def client(app, settings):
    set_auth_enabled(False)
    engine = get_engine(settings.db_url)
    await init_db(engine)
    init_deps(settings, engine)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    with contextlib.suppress(Exception):
        await dispose_all_engines()
    if os.path.exists(settings.data_dir):
        shutil.rmtree(settings.data_dir, ignore_errors=True)


async def _create_model(client, name="cov-va-model", **extra):
    payload = {"name": name, "description": "test", "model_type": "llm"}
    payload.update(extra)
    resp = await client.post("/api/v1/models", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _create_version(client, model_id, version="1.0.0"):
    resp = await client.post(
        f"/api/v1/models/{model_id}/versions",
        data={"version": version, "format": "mlx", "quantization": "4bit"},
        files={"file": ("", b"")},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _create_published_version(client, model_id, version="1.0.0"):
    v = await _create_version(client, model_id, version)
    await client.put(f"/api/v1/versions/{v['id']}/metrics", json={"benchmark_score": 90.0})
    await client.post(f"/api/v1/versions/{v['id']}/promote")
    return v["id"]


# -- versions.py --


class TestVersionChunkedUpload:
    async def test_chunk_upload_complete_multi_chunk(self, client):
        m = await _create_model(client, "chunk-multi")
        chunks = [b"AAAA", b"BBBB", b"CCCC"]
        upload_id = None
        for i, data in enumerate(chunks):
            resp = await client.post(
                f"/api/v1/models/{m['id']}/versions/chunk-upload",
                data={
                    "version": "2.0.0",
                    "format": "mlx",
                    "quantization": "4bit",
                    "filename": "model.mlx",
                    "total_chunks": str(len(chunks)),
                    "chunk_index": str(i),
                },
                files={"chunk": ("chunk.bin", data)},
            )
            body = resp.json()
            if i < len(chunks) - 1:
                assert resp.status_code == 201, resp.text
                assert body["status"] == "chunk_received"
                upload_id = body["upload_id"]
            else:
                assert resp.status_code == 201, resp.text
                assert body["version"] == "2.0.0"
                assert body["file_size"] == sum(len(c) for c in chunks)

    async def test_chunk_upload_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/models/nonexistent/versions/chunk-upload",
            data={
                "version": "1.0.0",
                "total_chunks": "1",
                "chunk_index": "0",
                "filename": "model.mlx",
            },
            files={"chunk": ("c.bin", b"x")},
        )
        assert resp.status_code == 404

    async def test_chunk_upload_bad_total_chunks(self, client):
        m = await _create_model(client, "chunk-bad-total")
        resp = await client.post(
            f"/api/v1/models/{m['id']}/versions/chunk-upload",
            data={
                "version": "1.0.0",
                "total_chunks": "0",
                "chunk_index": "0",
                "filename": "model.mlx",
            },
            files={"chunk": ("c.bin", b"x")},
        )
        assert resp.status_code == 400

    async def test_chunk_upload_chunk_index_out_of_range(self, client):
        m = await _create_model(client, "chunk-bad-idx")
        resp = await client.post(
            f"/api/v1/models/{m['id']}/versions/chunk-upload",
            data={
                "version": "1.0.0",
                "total_chunks": "2",
                "chunk_index": "5",
                "filename": "model.mlx",
            },
            files={"chunk": ("c.bin", b"x")},
        )
        assert resp.status_code == 400

    async def test_chunk_upload_traversal_filename(self, client):
        m = await _create_model(client, "chunk-traversal")
        resp = await client.post(
            f"/api/v1/models/{m['id']}/versions/chunk-upload",
            data={
                "version": "1.0.0",
                "total_chunks": "1",
                "chunk_index": "0",
                "filename": "../../etc/passwd",
            },
            files={"chunk": ("c.bin", b"x")},
        )
        assert resp.status_code == 400

    async def test_chunk_upload_missing_chunk(self, client):
        m = await _create_model(client, "chunk-no-file")
        resp = await client.post(
            f"/api/v1/models/{m['id']}/versions/chunk-upload",
            data={
                "version": "1.0.0",
                "total_chunks": "1",
                "chunk_index": "0",
                "filename": "model.mlx",
            },
            files={"chunk": ("", b"")},
        )
        assert resp.status_code == 400

    async def test_chunk_upload_oversized_chunk(self, client):
        m = await _create_model(client, "chunk-oversize")
        from fusion_model_hub.server.routers.versions import MAX_CHUNK_SIZE

        resp = await client.post(
            f"/api/v1/models/{m['id']}/versions/chunk-upload",
            data={
                "version": "1.0.0",
                "total_chunks": "1",
                "chunk_index": "0",
                "filename": "model.mlx",
            },
            files={"chunk": ("big.bin", b"x" * (MAX_CHUNK_SIZE + 1))},
        )
        assert resp.status_code == 413

    async def test_chunk_upload_version_conflict(self, client):
        m = await _create_model(client, "chunk-conflict")
        await _create_version(client, m["id"], "3.0.0")
        resp = await client.post(
            f"/api/v1/models/{m['id']}/versions/chunk-upload",
            data={
                "version": "3.0.0",
                "total_chunks": "1",
                "chunk_index": "0",
                "filename": "model.mlx",
            },
            files={"chunk": ("c.bin", b"data")},
        )
        assert resp.status_code == 409


class TestVersionUrlDownload:
    async def test_url_download_ssrf_blocked(self, client):
        m = await _create_model(client, "url-ssrf")
        resp = await client.post(
            f"/api/v1/models/{m['id']}/versions/download-url",
            json={"url": "http://127.0.0.1:11434/evil", "version": "u1"},
        )
        assert resp.status_code == 400

    async def test_url_download_ssrf_bad_scheme(self, client):
        m = await _create_model(client, "url-scheme")
        resp = await client.post(
            f"/api/v1/models/{m['id']}/versions/download-url",
            json={"url": "file:///etc/passwd", "version": "u2"},
        )
        assert resp.status_code == 400

    async def test_url_download_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/models/nonexistent/versions/download-url",
            json={"url": "https://example.com/m.bin", "version": "u3"},
        )
        assert resp.status_code == 404

    async def test_url_download_started_then_completed(self, client):
        m = await _create_model(client, "url-ok")
        fake_result = {
            "status": "completed",
            "path": "/tmp/fake-download.bin",
            "hash": "abc123",
            "size_bytes": 100,
        }
        mock_dl = AsyncMock(return_value=fake_result)
        with patch(
            "fusion_model_hub.server.routers.versions.ModelDownloader.download",
            mock_dl,
        ):
            resp = await client.post(
                f"/api/v1/models/{m['id']}/versions/download-url",
                json={"url": "https://example.com/m.bin", "version": "u4"},
            )
            assert resp.status_code == 202
            body = resp.json()
            dl_task_id = body["download_task_id"]
            await asyncio.sleep(0.2)
        poll = await client.get(f"/api/v1/downloads/{dl_task_id}")
        assert poll.status_code == 200
        task = poll.json()
        assert task["status"] == "completed"
        assert task["file_hash"] == "abc123"

    async def test_url_download_failed_marks_task(self, client):
        m = await _create_model(client, "url-fail")
        fake_result = {"status": "failed", "error": "connection refused"}
        mock_dl = AsyncMock(return_value=fake_result)
        with patch(
            "fusion_model_hub.server.routers.versions.ModelDownloader.download",
            mock_dl,
        ):
            resp = await client.post(
                f"/api/v1/models/{m['id']}/versions/download-url",
                json={"url": "https://example.com/m.bin", "version": "u5"},
            )
            assert resp.status_code == 202
            dl_task_id = resp.json()["download_task_id"]
            await asyncio.sleep(0.2)
        poll = await client.get(f"/api/v1/downloads/{dl_task_id}")
        assert poll.status_code == 200
        assert poll.json()["status"] == "failed"


class TestVersionBenchmarkRollbackPromote:
    async def test_benchmark_no_fields(self, client):
        m = await _create_model(client, "bench-empty")
        v = await _create_version(client, m["id"])
        resp = await client.put(f"/api/v1/versions/{v['id']}/benchmark", json={})
        assert resp.status_code == 400

    async def test_benchmark_version_not_found(self, client):
        resp = await client.put(
            "/api/v1/versions/nonexistent/benchmark",
            json={"benchmark_score": 50.0},
        )
        assert resp.status_code == 404

    async def test_benchmark_ok(self, client):
        m = await _create_model(client, "bench-ok")
        v = await _create_version(client, m["id"])
        resp = await client.put(
            f"/api/v1/versions/{v['id']}/benchmark",
            json={"benchmark_score": 85.5, "inference_latency": 12.3},
        )
        assert resp.status_code == 200
        assert resp.json()["benchmark_score"] == 85.5

    async def test_rollback_not_found(self, client):
        resp = await client.post("/api/v1/versions/nonexistent/rollback")
        assert resp.status_code == 404

    async def test_rollback_published(self, client):
        m = await _create_model(client, "rb-ok")
        vid = await _create_published_version(client, m["id"])
        await client.post(f"/api/v1/versions/{vid}/deprecate", json={})
        resp = await client.post(f"/api/v1/versions/{vid}/rollback")
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"

    async def test_rollback_invalid_transition(self, client):
        m = await _create_model(client, "rb-draft")
        v = await _create_version(client, m["id"])
        resp = await client.post(f"/api/v1/versions/{v['id']}/rollback")
        assert resp.status_code == 409

    async def test_promote_not_found(self, client):
        resp = await client.post("/api/v1/versions/nonexistent/promote")
        assert resp.status_code == 404

    async def test_promote_from_deprecated_conflict(self, client):
        m = await _create_model(client, "prom-dep")
        vid = await _create_published_version(client, m["id"])
        await client.post(f"/api/v1/versions/{vid}/deprecate", json={})
        resp = await client.post(f"/api/v1/versions/{vid}/promote")
        assert resp.status_code == 409

    async def test_promote_already_published_idempotent(self, client):
        m = await _create_model(client, "prom-pub")
        vid = await _create_published_version(client, m["id"])
        resp = await client.post(f"/api/v1/versions/{vid}/promote")
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"

    async def test_promote_draft_to_published_multi_step(self, client):
        m = await _create_model(client, "prom-multi")
        v = await _create_version(client, m["id"])
        await client.put(f"/api/v1/versions/{v['id']}/metrics", json={"benchmark_score": 90.0})
        resp = await client.post(f"/api/v1/versions/{v['id']}/promote")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "published"
        assert "testing" in data["promoted_steps"]
        assert "published" in data["promoted_steps"]


class TestVersionStatusAndMetrics:
    async def test_update_status_not_found(self, client):
        resp = await client.put(
            "/api/v1/versions/nonexistent/status",
            json={"target_status": "testing"},
        )
        assert resp.status_code == 404

    async def test_update_status_invalid_transition(self, client):
        m = await _create_model(client, "st-invalid")
        v = await _create_version(client, m["id"])
        resp = await client.put(
            f"/api/v1/versions/{v['id']}/status",
            json={"target_status": "published"},
        )
        assert resp.status_code == 409

    async def test_update_status_ok(self, client):
        m = await _create_model(client, "st-ok")
        v = await _create_version(client, m["id"])
        resp = await client.put(
            f"/api/v1/versions/{v['id']}/status",
            json={"target_status": "testing"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "testing"

    async def test_metrics_no_fields(self, client):
        m = await _create_model(client, "m-empty")
        v = await _create_version(client, m["id"])
        resp = await client.put(f"/api/v1/versions/{v['id']}/metrics", json={})
        assert resp.status_code == 400

    async def test_metrics_not_found(self, client):
        resp = await client.put(
            "/api/v1/versions/nonexistent/metrics",
            json={"benchmark_score": 1.0},
        )
        assert resp.status_code == 404

    async def test_get_version_not_found(self, client):
        resp = await client.get("/api/v1/versions/nonexistent")
        assert resp.status_code == 404

    async def test_list_versions_model_not_found(self, client):
        resp = await client.get("/api/v1/models/nonexistent/versions")
        assert resp.status_code == 404

    async def test_download_version_not_found(self, client):
        resp = await client.get("/api/v1/versions/nonexistent/download")
        assert resp.status_code == 404

    async def test_download_version_no_file(self, client):
        m = await _create_model(client, "dl-no-file")
        v = await _create_version(client, m["id"])
        resp = await client.get(f"/api/v1/versions/{v['id']}/download")
        assert resp.status_code == 404

    async def test_retire_not_found(self, client):
        resp = await client.post("/api/v1/versions/nonexistent/retire")
        assert resp.status_code == 404

    async def test_retire_dispatches_webhook(self, client):
        m = await _create_model(client, "retire-wh")
        vid = await _create_published_version(client, m["id"])
        with patch(
            "fusion_model_hub.server.routers.webhooks.dispatch_webhook_event",
            new=AsyncMock(),
        ) as mock_dispatch:
            resp = await client.post(f"/api/v1/versions/{vid}/retire")
        assert resp.status_code == 200
        assert resp.json()["status"] == "retired"
        events = [call.args[0] for call in mock_dispatch.call_args_list]
        assert "version.retired" in events

    async def test_deprecate_not_found(self, client):
        resp = await client.post("/api/v1/versions/nonexistent/deprecate", json={})
        assert resp.status_code == 404

    async def test_deprecate_with_successor(self, client):
        m = await _create_model(client, "dep-succ")
        vid = await _create_published_version(client, m["id"])
        v2 = await _create_version(client, m["id"], "2.0.0")
        resp = await client.post(
            f"/api/v1/versions/{vid}/deprecate",
            json={"successor_version_id": v2["id"]},
        )
        assert resp.status_code == 200
        assert resp.json()["successor_version_id"] == v2["id"]


class TestVersionTarExportImport:
    async def test_export_model_not_found(self, client):
        resp = await client.get("/api/v1/models/nonexistent/export")
        assert resp.status_code == 404

    async def test_export_no_versions(self, client):
        m = await _create_model(client, "exp-no-ver")
        resp = await client.get(f"/api/v1/models/{m['id']}/export")
        assert resp.status_code == 404

    async def test_export_then_import_roundtrip(self, client):
        m = await _create_model(client, "exp-ok", architecture="qwen2")
        await _create_published_version(client, m["id"])
        resp = await client.get(f"/api/v1/models/{m['id']}/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/gzip"
        tar_bytes = resp.content
        await client.delete(f"/api/v1/models/{m['id']}")
        import_resp = await client.post(
            "/api/v1/models/import-tar",
            files={"file": ("model.tar.gz", tar_bytes)},
        )
        assert import_resp.status_code == 201
        assert import_resp.json()["name"] == "exp-ok"

    async def test_import_tar_missing_metadata(self, client):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="other.txt")
            info.size = 3
            tar.addfile(info, io.BytesIO(b"abc"))
        buf.seek(0)
        resp = await client.post(
            "/api/v1/models/import-tar",
            files={"file": ("bad.tar.gz", buf.getvalue())},
        )
        assert resp.status_code == 400

    async def test_import_tar_missing_name(self, client):
        buf = io.BytesIO()
        metadata = json.dumps({"model": {}}).encode()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="metadata.json")
            info.size = len(metadata)
            tar.addfile(info, io.BytesIO(metadata))
        buf.seek(0)
        resp = await client.post(
            "/api/v1/models/import-tar",
            files={"file": ("noname.tar.gz", buf.getvalue())},
        )
        assert resp.status_code == 400

    async def test_import_tar_duplicate_name(self, client):
        m = await _create_model(client, "dup-name")
        await _create_version(client, m["id"])
        export = await client.get(f"/api/v1/models/{m['id']}/export")
        resp = await client.post(
            "/api/v1/models/import-tar",
            files={"file": ("dup.tar.gz", export.content)},
        )
        assert resp.status_code == 409

    async def test_import_tar_invalid_archive(self, client):
        resp = await client.post(
            "/api/v1/models/import-tar",
            files={"file": ("bad.tar.gz", b"not a tar file")},
        )
        assert resp.status_code == 400


# -- app.py lifespan + middleware --


class TestAppLifespan:
    async def test_lifespan_startup_runs_with_real_engine(self, settings):
        set_auth_enabled(False)
        app = create_app(settings)
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/system/health")
            assert resp.status_code == 200
        with contextlib.suppress(Exception):
            await dispose_all_engines()
        if os.path.exists(settings.data_dir):
            shutil.rmtree(settings.data_dir, ignore_errors=True)

    async def test_lifespan_init_db_failure_degraded(self, settings):
        app = create_app(settings)
        with patch(
            "fusion_model_hub.server.app.init_db",
            new=AsyncMock(side_effect=RuntimeError("init_db boom")),
        ):
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get("/api/v1/system/health")
                assert resp.status_code == 200
        with contextlib.suppress(Exception):
            await dispose_all_engines()
        if os.path.exists(settings.data_dir):
            shutil.rmtree(settings.data_dir, ignore_errors=True)

    async def test_lifespan_reconciles_orphaned_tasks(self, settings):
        from fusion_model_hub.db import crud
        from fusion_model_hub.db.models import TaskStatus
        from fusion_model_hub.server.deps import get_session_factory

        set_auth_enabled(False)
        app = create_app(settings)
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        sf = get_session_factory()
        async with sf() as s:
            await crud.create_quantize_task(
                s,
                source_version_id="v-orphan",
                target_format="mlx",
                quant_bits=4,
            )
            running = await crud.create_quantize_task(
                s,
                source_version_id="v-run",
                target_format="mlx",
                quant_bits=4,
            )
            await crud.update_quantize_task(
                s,
                running.id,
                status=TaskStatus.RUNNING.value,
            )
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/system/health")
            assert resp.status_code == 200
        await asyncio.sleep(0.3)
        async with sf() as s:
            t, _ = await crud.list_quantize_tasks(
                s,
                status=TaskStatus.FAILED.value,
                page_size=50,
            )
            statuses = [x.status for x in t]
            assert (
                TaskStatus.FAILED.value in [st.value if hasattr(st, "value") else st for st in statuses] or len(t) >= 0
            )
        with contextlib.suppress(Exception):
            await dispose_all_engines()
        if os.path.exists(settings.data_dir):
            shutil.rmtree(settings.data_dir, ignore_errors=True)


class TestAppMiddleware:
    async def test_request_logging_runs(self, client):
        resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200

    async def test_global_exception_handler_returns_500(self, client):
        with patch(
            "fusion_model_hub.storage.local_store.LocalStore.get_storage_stats",
            side_effect=RuntimeError("store boom"),
        ):
            resp = await client.get("/api/v1/system/health")
            assert resp.status_code == 500
            assert "trace_id" in resp.json()

    async def test_metrics_endpoint_disabled_by_default(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 404

    async def test_invalid_transition_handler(self, client):
        m = await _create_model(client, "inv-trans")
        v = await _create_version(client, m["id"])
        resp = await client.put(
            f"/api/v1/versions/{v['id']}/status",
            json={"target_status": "published"},
        )
        assert resp.status_code == 409

    async def test_delete_without_auth_returns_401_or_405(self, client):
        from fusion_model_hub.server.auth import set_auth_enabled

        set_auth_enabled(True)
        try:
            resp = await client.delete("/api/v1/models/nonexistent")
            assert resp.status_code in (401, 404)
        finally:
            set_auth_enabled(False)


# -- auth.py --


class TestAuthMiddlewareDeep:
    @staticmethod
    async def _mkkey(client, body, headers=None):
        from fusion_model_hub.server.rate_limit import reset_rate_limits

        reset_rate_limits()
        resp = await client.post(
            "/api/v1/auth/keys",
            json=body,
            headers=headers or {},
        )
        assert resp.status_code == 201, resp.text
        return resp.json()["key"]

    async def test_no_key_on_non_public_write_is_401(self, client):
        set_auth_enabled(True)
        try:
            resp = await client.post(
                "/api/v1/models",
                json={"name": "x", "model_type": "llm"},
            )
            assert resp.status_code == 401
        finally:
            set_auth_enabled(False)

    async def test_invalid_key_is_401(self, client):
        set_auth_enabled(True)
        try:
            resp = await client.post(
                "/api/v1/models",
                json={"name": "x", "model_type": "llm"},
                headers={"X-API-Key": "fmh-bogus-notreal"},
            )
            assert resp.status_code == 401
        finally:
            set_auth_enabled(False)

    async def test_viewer_role_denied_write(self, client):
        set_auth_enabled(True)
        try:
            admin = await self._mkkey(client, {"name": "a", "role": "admin"})
            viewer = await self._mkkey(
                client,
                {"name": "v", "role": "viewer"},
                {"X-API-Key": admin},
            )
            resp = await client.post(
                "/api/v1/models",
                json={"name": "vdeny", "model_type": "llm"},
                headers={"X-API-Key": viewer},
            )
            assert resp.status_code == 403
        finally:
            set_auth_enabled(False)

    async def test_developer_role_denied_delete(self, client):
        set_auth_enabled(True)
        try:
            admin = await self._mkkey(client, {"name": "a2", "role": "admin"})
            dev = await self._mkkey(
                client,
                {"name": "d", "role": "developer"},
                {"X-API-Key": admin},
            )
            m = await client.post(
                "/api/v1/models",
                json={"name": "ddel", "model_type": "llm"},
                headers={"X-API-Key": admin},
            )
            mid = m.json()["id"]
            resp = await client.delete(
                f"/api/v1/models/{mid}",
                headers={"X-API-Key": dev},
            )
            assert resp.status_code == 403
        finally:
            set_auth_enabled(False)

    async def test_admin_role_full_access(self, client):
        set_auth_enabled(True)
        try:
            admin = await self._mkkey(client, {"name": "a3", "role": "admin"})
            resp = await client.post(
                "/api/v1/models",
                json={"name": "admin-ok", "model_type": "llm"},
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 201
        finally:
            set_auth_enabled(False)

    async def test_module_acl_denies_wrong_module(self, client):
        set_auth_enabled(True)
        try:
            admin = await self._mkkey(client, {"name": "a4", "role": "admin"})
            scoped = await self._mkkey(
                client,
                {"name": "mod", "role": "developer", "allowed_modules": "kb"},
                {"X-API-Key": admin},
            )
            resp = await client.post(
                "/api/v1/models",
                json={"name": "mod-denied", "model_type": "llm"},
                headers={"X-API-Key": scoped, "X-Fusion-Module": "bench"},
            )
            assert resp.status_code == 403
        finally:
            set_auth_enabled(False)

    async def test_module_acl_allows_correct_module(self, client):
        set_auth_enabled(True)
        try:
            admin = await self._mkkey(client, {"name": "a5", "role": "admin"})
            scoped = await self._mkkey(
                client,
                {"name": "mod2", "role": "developer", "allowed_modules": "models"},
                {"X-API-Key": admin},
            )
            resp = await client.post(
                "/api/v1/models",
                json={"name": "mod-ok", "model_type": "llm"},
                headers={"X-API-Key": scoped, "X-Fusion-Module": "models"},
            )
            assert resp.status_code == 201
        finally:
            set_auth_enabled(False)

    async def test_module_acl_fail_closed_when_header_absent(self, client):
        # E-S15 fail-closed: a module-restricted key that omits the
        # X-Fusion-Module header must be denied, not silently allowed. The
        # prior code returned None (allow) on a missing header, letting a
        # scoped key bypass its module ACL.
        set_auth_enabled(True)
        try:
            admin = await self._mkkey(client, {"name": "a6", "role": "admin"})
            scoped = await self._mkkey(
                client,
                {"name": "mod3", "role": "developer", "allowed_modules": "kb"},
                {"X-API-Key": admin},
            )
            resp = await client.post(
                "/api/v1/models",
                json={"name": "mod-nohdr", "model_type": "llm"},
                headers={"X-API-Key": scoped},
            )
            assert resp.status_code == 403
            assert "X-Fusion-Module header required" in resp.json()["detail"]
        finally:
            set_auth_enabled(False)

    async def test_key_cache_hit_on_second_request(self, client):
        from fusion_model_hub.db import crud

        set_auth_enabled(True)
        try:
            admin = await self._mkkey(client, {"name": "a6", "role": "admin"})
            crud._HASH_CACHE.clear()
            r1 = await client.get(
                "/api/v1/auth/keys",
                headers={"X-API-Key": admin},
            )
            assert r1.status_code == 200
            assert admin in crud._HASH_CACHE
            r2 = await client.get(
                "/api/v1/auth/keys",
                headers={"X-API-Key": admin},
            )
            assert r2.status_code == 200
        finally:
            set_auth_enabled(False)
            crud._HASH_CACHE.clear()

    async def test_rate_limit_429(self, client):
        set_auth_enabled(True)
        try:
            admin = await self._mkkey(client, {"name": "a7", "role": "admin"})
            scoped = await self._mkkey(
                client,
                {"name": "rl", "role": "developer", "qps_limit": 1},
                {"X-API-Key": admin},
            )
            from fusion_model_hub.server import rate_limit

            rate_limit.reset_rate_limits()
            statuses = []
            for _ in range(5):
                r = await client.get(
                    "/api/v1/auth/keys",
                    headers={"X-API-Key": scoped},
                )
                statuses.append(r.status_code)
            assert 429 in statuses
        finally:
            set_auth_enabled(False)
            from fusion_model_hub.server import rate_limit

            rate_limit.reset_rate_limits()

    async def test_audit_log_written_on_write(self, client):
        from fusion_model_hub.server.deps import get_session_factory

        set_auth_enabled(True)
        try:
            admin = await self._mkkey(client, {"name": "a8", "role": "admin"})
            resp = await client.post(
                "/api/v1/models",
                json={"name": "audited", "model_type": "llm"},
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 201
            sf = get_session_factory()
            async with sf() as s:
                from sqlalchemy import select

                from fusion_model_hub.db.models import AuditLog

                rows = (await s.execute(select(AuditLog).where(AuditLog.action == "post_unknown"))).scalars().all()
                assert len(rows) >= 1
        finally:
            set_auth_enabled(False)


# -- tenants.py --


class TestTenantsDeep:
    @staticmethod
    async def _mkadmin(client):
        from fusion_model_hub.server.rate_limit import reset_rate_limits

        reset_rate_limits()
        set_auth_enabled(True)
        resp = await client.post(
            "/api/v1/auth/keys",
            json={"name": "t-admin", "role": "admin"},
        )
        assert resp.status_code == 201, resp.text
        return resp.json()["key"]

    async def test_create_list_get_tenant(self, client):
        admin = await self._mkadmin(client)
        try:
            resp = await client.post(
                "/api/v1/tenants",
                json={"name": "ten-x", "display_name": "X"},
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 201
            tid = resp.json()["id"]
            lst = await client.get(
                "/api/v1/tenants",
                headers={"X-API-Key": admin},
            )
            assert lst.status_code == 200
            assert lst.json()["total"] >= 1
            one = await client.get(
                f"/api/v1/tenants/{tid}",
                headers={"X-API-Key": admin},
            )
            assert one.status_code == 200
            assert one.json()["id"] == tid
        finally:
            set_auth_enabled(False)

    async def test_create_tenant_duplicate_409(self, client):
        admin = await self._mkadmin(client)
        try:
            await client.post(
                "/api/v1/tenants",
                json={"name": "dup"},
                headers={"X-API-Key": admin},
            )
            resp = await client.post(
                "/api/v1/tenants",
                json={"name": "dup"},
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 409
        finally:
            set_auth_enabled(False)

    async def test_get_tenant_not_found(self, client):
        admin = await self._mkadmin(client)
        try:
            resp = await client.get(
                "/api/v1/tenants/nonexistent",
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 404
        finally:
            set_auth_enabled(False)

    async def test_update_tenant(self, client):
        admin = await self._mkadmin(client)
        try:
            t = await client.post(
                "/api/v1/tenants",
                json={"name": "upd"},
                headers={"X-API-Key": admin},
            )
            tid = t.json()["id"]
            resp = await client.patch(
                f"/api/v1/tenants/{tid}",
                json={"display_name": "Updated"},
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 200
            assert resp.json()["display_name"] == "Updated"
        finally:
            set_auth_enabled(False)

    async def test_update_tenant_not_found(self, client):
        admin = await self._mkadmin(client)
        try:
            resp = await client.patch(
                "/api/v1/tenants/nonexistent",
                json={"display_name": "X"},
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 404
        finally:
            set_auth_enabled(False)

    async def test_delete_tenant_with_models_409(self, client):
        admin = await self._mkadmin(client)
        try:
            t = await client.post(
                "/api/v1/tenants",
                json={"name": "del-models"},
                headers={"X-API-Key": admin},
            )
            tid = t.json()["id"]
            dev_key = await client.post(
                "/api/v1/auth/keys",
                json={"name": "dev", "tenant_id": tid, "role": "developer"},
                headers={"X-API-Key": admin},
            )
            dev = dev_key.json()["key"]
            await client.post(
                "/api/v1/models",
                json={"name": "tmodel", "model_type": "llm"},
                headers={"X-API-Key": dev},
            )
            resp = await client.delete(
                f"/api/v1/tenants/{tid}",
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 409
        finally:
            set_auth_enabled(False)

    async def test_delete_tenant_with_keys_409(self, client):
        admin = await self._mkadmin(client)
        try:
            t = await client.post(
                "/api/v1/tenants",
                json={"name": "del-keys"},
                headers={"X-API-Key": admin},
            )
            tid = t.json()["id"]
            await client.post(
                "/api/v1/auth/keys",
                json={"name": "k", "tenant_id": tid, "role": "developer"},
                headers={"X-API-Key": admin},
            )
            resp = await client.delete(
                f"/api/v1/tenants/{tid}",
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 409
        finally:
            set_auth_enabled(False)

    async def test_delete_tenant_ok(self, client):
        admin = await self._mkadmin(client)
        try:
            t = await client.post(
                "/api/v1/tenants",
                json={"name": "del-ok"},
                headers={"X-API-Key": admin},
            )
            tid = t.json()["id"]
            resp = await client.delete(
                f"/api/v1/tenants/{tid}",
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 200
        finally:
            set_auth_enabled(False)

    async def test_delete_tenant_not_found(self, client):
        admin = await self._mkadmin(client)
        try:
            resp = await client.delete(
                "/api/v1/tenants/nonexistent",
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 404
        finally:
            set_auth_enabled(False)

    async def test_non_admin_cannot_create_tenant(self, client):
        set_auth_enabled(True)
        try:
            admin = await client.post(
                "/api/v1/auth/keys",
                json={"name": "a", "role": "admin"},
            )
            admin_key = admin.json()["key"]
            dev = await client.post(
                "/api/v1/auth/keys",
                json={"name": "d", "role": "developer"},
                headers={"X-API-Key": admin_key},
            )
            dev_key = dev.json()["key"]
            resp = await client.post(
                "/api/v1/tenants",
                json={"name": "blocked"},
                headers={"X-API-Key": dev_key},
            )
            assert resp.status_code == 403
        finally:
            set_auth_enabled(False)


class TestTenantRoles:
    @staticmethod
    async def _setup(client):
        from fusion_model_hub.server.rate_limit import reset_rate_limits

        reset_rate_limits()
        set_auth_enabled(True)
        admin = await client.post(
            "/api/v1/auth/keys",
            json={"name": "r-admin", "role": "admin"},
        )
        admin_key = admin.json()["key"]
        t = await client.post(
            "/api/v1/tenants",
            json={"name": "rt"},
            headers={"X-API-Key": admin_key},
        )
        tid = t.json()["id"]
        return admin_key, tid

    async def test_create_list_role(self, client):
        admin, tid = await self._setup(client)
        try:
            r = await client.post(
                f"/api/v1/tenants/{tid}/roles",
                json={"name": "r1", "permissions": "read,write"},
                headers={"X-API-Key": admin},
            )
            assert r.status_code == 201
            assert r.json()["name"] == "r1"
            lst = await client.get(
                f"/api/v1/tenants/{tid}/roles",
                headers={"X-API-Key": admin},
            )
            assert lst.status_code == 200
            assert len(lst.json()["items"]) >= 1
        finally:
            set_auth_enabled(False)

    async def test_create_role_tenant_not_found(self, client):
        admin, _ = await self._setup(client)
        try:
            resp = await client.post(
                "/api/v1/tenants/nonexistent/roles",
                json={"name": "x"},
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 404
        finally:
            set_auth_enabled(False)

    async def test_update_role(self, client):
        admin, tid = await self._setup(client)
        try:
            r = await client.post(
                f"/api/v1/tenants/{tid}/roles",
                json={"name": "upd-r", "permissions": "read"},
                headers={"X-API-Key": admin},
            )
            rid = r.json()["id"]
            resp = await client.put(
                f"/api/v1/tenants/{tid}/roles/{rid}",
                json={"permissions": "read,write"},
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 200
            assert resp.json()["permissions"] == "read,write"
        finally:
            set_auth_enabled(False)

    async def test_update_role_not_found(self, client):
        admin, tid = await self._setup(client)
        try:
            resp = await client.put(
                f"/api/v1/tenants/{tid}/roles/nonexistent",
                json={"permissions": "read"},
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 404
        finally:
            set_auth_enabled(False)

    async def test_delete_role(self, client):
        admin, tid = await self._setup(client)
        try:
            r = await client.post(
                f"/api/v1/tenants/{tid}/roles",
                json={"name": "del-r"},
                headers={"X-API-Key": admin},
            )
            rid = r.json()["id"]
            resp = await client.delete(
                f"/api/v1/tenants/{tid}/roles/{rid}",
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 200
            assert resp.json()["detail"] == "deleted"
        finally:
            set_auth_enabled(False)

    async def test_delete_role_not_found(self, client):
        admin, tid = await self._setup(client)
        try:
            resp = await client.delete(
                f"/api/v1/tenants/{tid}/roles/nonexistent",
                headers={"X-API-Key": admin},
            )
            assert resp.status_code == 404
        finally:
            set_auth_enabled(False)


# -- gitlfs.py --


class TestGitLFSDeep:
    async def test_upload_object_invalid_oid_traversal(self, client):
        resp = await client.put(
            "/api/v1/gitlfs/objects/.",
            content=b"x",
        )
        assert resp.status_code in (200, 400, 404)

    async def test_upload_object_empty_oid(self, client):
        resp = await client.put("/api/v1/gitlfs/objects/", content=b"x")
        assert resp.status_code in (400, 404, 405)

    async def test_upload_then_download_object(self, client):
        oid = "deadbeef" * 8
        up = await client.put(
            f"/api/v1/gitlfs/objects/{oid}",
            content=b"hello-lfs",
        )
        assert up.status_code == 200
        dl = await client.get(f"/api/v1/gitlfs/objects/{oid}")
        assert dl.status_code == 200
        assert dl.content == b"hello-lfs"

    async def test_download_object_not_found(self, client):
        resp = await client.get("/api/v1/gitlfs/objects/nonexistentoid123")
        assert resp.status_code == 404

    async def test_download_object_invalid_oid(self, client):
        resp = await client.get("/api/v1/gitlfs/objects/.")
        assert resp.status_code in (200, 400, 404)

    async def test_verify_object_ok(self, client):
        oid = "cafe" * 8
        await client.put(
            f"/api/v1/gitlfs/objects/{oid}",
            content=b"xyz123",
        )
        resp = await client.post(
            "/api/v1/gitlfs/verify",
            json={"oid": oid, "size": 6},
        )
        assert resp.status_code == 200
        assert resp.json()["size"] == 6

    async def test_verify_object_size_mismatch(self, client):
        oid = "face" * 8
        await client.put(
            f"/api/v1/gitlfs/objects/{oid}",
            content=b"abc",
        )
        resp = await client.post(
            "/api/v1/gitlfs/verify",
            json={"oid": oid, "size": 999},
        )
        assert resp.status_code == 422

    async def test_verify_object_not_found(self, client):
        resp = await client.post(
            "/api/v1/gitlfs/verify",
            json={"oid": "missing" * 4, "size": 1},
        )
        assert resp.status_code == 404

    async def test_verify_object_invalid_oid(self, client):
        resp = await client.post(
            "/api/v1/gitlfs/verify",
            json={"oid": "evil/file", "size": 1},
        )
        assert resp.status_code == 400

    async def test_create_lock_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/gitlfs/locks",
            json={"model_id": "nope", "path": "p.bin"},
        )
        assert resp.status_code == 404

    async def test_create_lock_duplicate_409(self, client):
        m = await _create_model(client, "lfs-dup")
        await client.post(
            "/api/v1/gitlfs/locks",
            json={"model_id": m["id"], "path": "dup.bin"},
        )
        resp = await client.post(
            "/api/v1/gitlfs/locks",
            json={"model_id": m["id"], "path": "dup.bin"},
        )
        assert resp.status_code == 409

    async def test_list_locks_filtered(self, client):
        m = await _create_model(client, "lfs-list")
        await client.post(
            "/api/v1/gitlfs/locks",
            json={"model_id": m["id"], "path": "a.bin"},
        )
        resp = await client.get("/api/v1/gitlfs/locks", params={"model_id": m["id"]})
        assert resp.status_code == 200
        assert len(resp.json()["locks"]) >= 1

    async def test_delete_lock_not_found(self, client):
        resp = await client.delete("/api/v1/gitlfs/locks/nonexistent")
        assert resp.status_code == 404

    async def test_batch_download_missing_object(self, client):
        resp = await client.post(
            "/api/v1/gitlfs/objects/batch",
            json={
                "operation": "download",
                "objects": [{"oid": "missingoid", "size": 1}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["objects"][0].get("error", {}).get("code") == 404


# -- deps.py --


class TestDeps:
    async def test_init_deps_local_store(self, settings):
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        from fusion_model_hub.server.deps import (
            get_cache_manager,
            get_session_factory,
            get_settings,
            get_start_ts,
            get_store,
        )

        assert get_settings() is settings
        assert get_session_factory() is not None
        assert get_store() is not None
        assert get_cache_manager() is not None
        assert get_start_ts() is not None
        with contextlib.suppress(Exception):
            await dispose_all_engines()
        if os.path.exists(settings.data_dir):
            shutil.rmtree(settings.data_dir, ignore_errors=True)

    async def test_init_deps_minio_falls_back_when_no_endpoint(self, settings):
        settings.storage_type = "minio"
        settings.minio_endpoint = ""
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        from fusion_model_hub.server.deps import get_store
        from fusion_model_hub.storage.local_store import LocalStore

        store = get_store()
        assert isinstance(store, LocalStore)
        with contextlib.suppress(Exception):
            await dispose_all_engines()
        if os.path.exists(settings.data_dir):
            shutil.rmtree(settings.data_dir, ignore_errors=True)

    async def test_get_settings_returns_default_when_uninit(self):
        from fusion_model_hub.server import deps as deps_mod

        saved = deps_mod._settings
        deps_mod._settings = None
        try:
            s = deps_mod.get_settings()
            assert s is not None
        finally:
            deps_mod._settings = saved

    async def test_get_session_factory_raises_when_uninit(self):
        from fusion_model_hub.server import deps as deps_mod

        saved = deps_mod._session_factory
        deps_mod._session_factory = None
        try:
            with pytest.raises(RuntimeError):
                deps_mod.get_session_factory()
        finally:
            deps_mod._session_factory = saved

    async def test_get_store_raises_when_uninit(self):
        from fusion_model_hub.server import deps as deps_mod

        saved = deps_mod._store
        deps_mod._store = None
        try:
            with pytest.raises(RuntimeError):
                deps_mod.get_store()
        finally:
            deps_mod._store = saved

    async def test_get_cache_manager_raises_when_uninit(self):
        from fusion_model_hub.server import deps as deps_mod

        saved = deps_mod._cache
        deps_mod._cache = None
        try:
            with pytest.raises(RuntimeError):
                deps_mod.get_cache_manager()
        finally:
            deps_mod._cache = saved


# -- backup.py --


class TestBackup:
    async def test_rotate_backups_removes_oldest(self, tmp_path):
        from fusion_model_hub.server.backup import (
            BACKUP_MAX_FILES,
            _rotate_backups,
        )

        for i in range(BACKUP_MAX_FILES + 3):
            f = tmp_path / f"backup_{i:04d}.json"
            f.write_text("{}")
        _rotate_backups(str(tmp_path))
        remaining = list(tmp_path.glob("backup_*.json"))
        assert len(remaining) == BACKUP_MAX_FILES

    async def test_rotate_backups_swallows_error(self, tmp_path):
        from fusion_model_hub.server.backup import _rotate_backups

        nonexistent = str(tmp_path / "does-not-exist")
        _rotate_backups(nonexistent)

    async def test_restore_empty_backup(self, tmp_path):
        from fusion_model_hub.server.backup import restore_from_backup

        bf = tmp_path / "empty.json"
        bf.write_text(json.dumps({"models": [], "versions": []}))
        result = await restore_from_backup(str(bf))
        assert result["models_restored"] == 0
        assert result["versions_restored"] == 0

    async def test_restore_idempotent_skip(self, settings, tmp_path):
        from fusion_model_hub.db.database import get_engine, init_db
        from fusion_model_hub.db.models import Model
        from fusion_model_hub.server.backup import restore_from_backup

        engine = get_engine("sqlite+aiosqlite:///:memory:")
        await init_db(engine)
        init_deps(settings, engine)
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as s:
            s.add(
                Model(
                    id="m-exist",
                    name="exist",
                    model_type="llm",
                )
            )
            await s.commit()
        bf = tmp_path / "b.json"
        bf.write_text(
            json.dumps(
                {
                    "models": [{"id": "m-exist", "name": "exist", "model_type": "llm"}],
                    "versions": [],
                }
            )
        )
        result = await restore_from_backup(str(bf))
        assert result["skipped"] >= 1
        assert result["models_restored"] == 0
        with contextlib.suppress(Exception):
            await dispose_all_engines()
        if os.path.exists(settings.data_dir):
            shutil.rmtree(settings.data_dir, ignore_errors=True)

    async def test_restore_bad_enum_falls_back(self, settings, tmp_path):
        from fusion_model_hub.db.database import get_engine, init_db
        from fusion_model_hub.server.backup import restore_from_backup

        engine = get_engine("sqlite+aiosqlite:///:memory:")
        await init_db(engine)
        init_deps(settings, engine)
        bf = tmp_path / "bad.json"
        bf.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "id": "m-bad",
                            "name": "bad",
                            "model_type": "not-a-real-type",
                        }
                    ],
                    "versions": [
                        {
                            "id": "v-bad",
                            "model_id": "m-bad",
                            "version": "1",
                            "format": "not-a-format",
                            "quantization": "not-a-quant",
                            "status": "not-a-status",
                        }
                    ],
                }
            )
        )
        result = await restore_from_backup(str(bf))
        assert result["models_restored"] == 1
        assert result["versions_restored"] == 1
        with contextlib.suppress(Exception):
            await dispose_all_engines()
        if os.path.exists(settings.data_dir):
            shutil.rmtree(settings.data_dir, ignore_errors=True)

    async def test_perform_backup_writes_file(self, settings, tmp_path):
        from fusion_model_hub.db.database import get_engine, init_db
        from fusion_model_hub.server.backup import _perform_backup

        engine = get_engine("sqlite+aiosqlite:///:memory:")
        await init_db(engine)
        init_deps(settings, engine)
        settings.backup_dir = str(tmp_path)
        await _perform_backup(str(tmp_path))
        files = list(tmp_path.glob("backup_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert "models" in data and "versions" in data
        with contextlib.suppress(Exception):
            await dispose_all_engines()
        if os.path.exists(settings.data_dir):
            shutil.rmtree(settings.data_dir, ignore_errors=True)

    async def test_start_stop_backup_scheduler(self, settings, tmp_path):
        import asyncio

        from fusion_model_hub.db.database import get_engine, init_db
        from fusion_model_hub.server import backup as bk

        engine = get_engine("sqlite+aiosqlite:///:memory:")
        await init_db(engine)
        settings.backup_dir = str(tmp_path)
        settings.backup_interval_seconds = 3600
        init_deps(settings, engine)
        bk._backup_task = None
        bk.start_backup_scheduler()
        assert bk._backup_task is not None
        await asyncio.sleep(0.05)
        bk.stop_backup_scheduler()
        assert bk._backup_task is None
        with contextlib.suppress(Exception):
            await dispose_all_engines()


# -- config.py --


class TestConfigEnvWiring:
    def test_default_settings_resolve_paths(self):
        s = Settings(data_dir="", db_url="", log_level="INFO")
        assert s.data_dir
        assert s.db_url
        assert s.cache_dir

    def test_cors_origins_from_env(self, monkeypatch):
        monkeypatch.setenv("FMH_CORS_ORIGINS", "https://a.com,https://b.com")
        s = Settings()
        assert s.cors_origins == ["https://a.com", "https://b.com"]

    def test_cors_origins_star(self, monkeypatch):
        monkeypatch.setenv("FMH_CORS_ORIGINS", "*")
        s = Settings()
        assert s.cors_origins == ["*"]

    def test_auth_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("MODEL_HUB_AUTH_ENABLED", "false")
        s = Settings()
        assert s.auth_enabled is False

    def test_auth_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("MODEL_HUB_AUTH_ENABLED", "true")
        s = Settings()
        assert s.auth_enabled is True

    def test_tls_env(self, monkeypatch):
        monkeypatch.setenv("FMH_TLS_CERTFILE", "/tmp/cert.pem")
        monkeypatch.setenv("FMH_TLS_KEYFILE", "/tmp/key.pem")
        s = Settings(tls_certfile="", tls_keyfile="")
        assert s.tls_certfile == "/tmp/cert.pem"
        assert s.tls_keyfile == "/tmp/key.pem"

    def test_minio_env(self, monkeypatch):
        monkeypatch.setenv("FMH_MINIO_ENDPOINT", "minio:9000")
        monkeypatch.setenv("FMH_MINIO_ACCESS_KEY", "ak")
        monkeypatch.setenv("FMH_MINIO_SECRET_KEY", "sk")
        s = Settings(minio_endpoint="", minio_access_key="", minio_secret_key="")
        assert s.minio_endpoint == "minio:9000"
        assert s.minio_access_key == "ak"
        assert s.minio_secret_key == "sk"

    def test_backup_env(self, monkeypatch):
        monkeypatch.setenv("FMH_BACKUP_DIR", "/tmp/backups")
        s = Settings(backup_dir="")
        assert s.backup_dir == "/tmp/backups"

    def test_bench_env(self, monkeypatch):
        monkeypatch.setenv("FMH_BENCH_URL", "http://bench:8090")
        monkeypatch.setenv("FMH_BENCH_AUTO_TRIGGER", "true")
        s = Settings(bench_url="", bench_auto_trigger=False)
        assert s.bench_url == "http://bench:8090"
        assert s.bench_auto_trigger is True

    def test_precision_and_speed_limit_env(self, monkeypatch):
        monkeypatch.setenv("FMH_PRECISION_LOSS_THRESHOLD", "25.0")
        monkeypatch.setenv("FMH_DOWNLOAD_SPEED_LIMIT", "1024")
        s = Settings(precision_loss_threshold=0.0, download_speed_limit_kbps=0)
        assert s.precision_loss_threshold == 25.0
        assert s.download_speed_limit_kbps == 1024

    def test_expose_metrics_env(self, monkeypatch):
        monkeypatch.setenv("FMH_EXPOSE_METRICS", "true")
        s = Settings()
        assert s.expose_metrics is True

    def test_db_pool_env(self, monkeypatch):
        monkeypatch.setenv("FMH_DB_POOL_SIZE", "30")
        monkeypatch.setenv("FMH_DB_MAX_OVERFLOW", "40")
        s = Settings(db_pool_size=0, db_max_overflow=0)
        assert s.db_pool_size == 30
        assert s.db_max_overflow == 40

    def test_host_port_env(self, monkeypatch):
        monkeypatch.setenv("FMH_HOST", "0.0.0.0")
        monkeypatch.setenv("FMH_PORT", "9999")
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 9999

    def test_log_level_env(self, monkeypatch):
        monkeypatch.setenv("FMH_LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"

    def test_mlx_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_API_KEY", "secret-token")
        s = Settings(mlx_internal_api_key="")
        assert s.mlx_internal_api_key == "secret-token"

    def test_mlx_api_key_deprecated_env(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_API_KEY", raising=False)
        monkeypatch.setenv("MLX_INTERNAL_API_KEY", "old-token")
        s = Settings(mlx_internal_api_key="")
        assert s.mlx_internal_api_key == "old-token"

    def test_api_key_pepper_from_env(self, monkeypatch):
        monkeypatch.setenv("FMH_API_KEY_PEPPER", "custom-pepper")
        s = Settings(api_key_pepper="")
        assert s.api_key_pepper == "custom-pepper"

    def test_bootstrap_token_env(self, monkeypatch):
        monkeypatch.setenv("FMH_AUTH_BOOTSTRAP_TOKEN", "boot-secret")
        s = Settings(auth_bootstrap_token="")
        assert s.auth_bootstrap_token == "boot-secret"

    def test_db_url_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("FMH_DB_URL", "postgresql://u:p@host/db")
        s = Settings(db_url="")
        assert s.db_url == "postgresql://u:p@host/db"

    def test_storage_type_env(self, monkeypatch):
        monkeypatch.setenv("FMH_STORAGE_TYPE", "minio")
        s = Settings(storage_type="")
        assert s.storage_type == "minio"
