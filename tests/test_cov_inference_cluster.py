import asyncio
import contextlib
import logging
import os
import shutil
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps

logger = logging.getLogger(__name__)


@pytest.fixture
def settings():
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_cov_inf_cluster",
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
async def client(app, settings):
    from fusion_model_hub.server.auth import set_auth_enabled

    set_auth_enabled(False)
    engine = get_engine(settings.db_url)
    await init_db(engine)
    init_deps(settings, engine)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    shutil.rmtree("/tmp/fmh_cov_inf_cluster", ignore_errors=True)


# ---- helpers -----------------------------------------------------------


async def _create_model(client, name="cov-inf-model", **extra):
    payload = {"name": name, "description": "test", "model_type": "llm"}
    payload.update(extra)
    resp = await client.post("/api/v1/models", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _publish(client, model_id):
    resp = await client.post(f"/api/v1/models/{model_id}/publish")
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _create_published_version(client, model_id, version="1.0.0"):
    resp = await client.post(
        f"/api/v1/models/{model_id}/versions",
        data={"version": version, "format": "mlx", "quantization": "4bit"},
        files={"file": ("", b"")},
    )
    assert resp.status_code == 201, resp.text
    ver = resp.json()
    await client.put(
        f"/api/v1/versions/{ver['id']}/metrics",
        json={"benchmark_score": 90.0},
    )
    await client.post(f"/api/v1/versions/{ver['id']}/promote")
    return ver


def _mock_httpx_ctx(response_status=200, response_json=None, side_effect=None):
    mock_resp = MagicMock()
    mock_resp.status_code = response_status
    mock_resp.json.return_value = response_json or {}
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    if side_effect:
        mock_ctx.post = AsyncMock(side_effect=side_effect)
        mock_ctx.get = AsyncMock(side_effect=side_effect)
    else:
        mock_ctx.post = AsyncMock(return_value=mock_resp)
        mock_ctx.get = AsyncMock(return_value=mock_resp)

    @contextlib.contextmanager
    def _ctx():
        with (
            patch("httpx.AsyncClient", return_value=mock_ctx),
            patch("fusion_model_hub.server.http_client.AsyncClient", return_value=mock_ctx),
        ):
            yield mock_ctx

    return _ctx()


def _mock_httpx_inference(response_status=200, response_json=None, side_effect=None):
    mock_resp = MagicMock()
    mock_resp.status_code = response_status
    mock_resp.json.return_value = response_json or {}
    mock_resp.text = ""
    mock_resp.raise_for_status = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    if side_effect:
        mock_ctx.post = AsyncMock(side_effect=side_effect)
        mock_ctx.get = AsyncMock(side_effect=side_effect)
    else:
        mock_ctx.post = AsyncMock(return_value=mock_resp)
        mock_ctx.get = AsyncMock(return_value=mock_resp)
    patcher = patch(
        "fusion_model_hub.server.routers.inference.httpx.AsyncClient",
        return_value=mock_ctx,
    )
    return patcher, mock_ctx


async def _register_node(client, name, url, capabilities="inference,quantize"):
    resp = await client.post(
        "/api/v1/cluster/nodes",
        json={
            "name": name,
            "url": url,
            "capabilities": capabilities,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# =====================================================================
# inference.py — module ACL, gray route, TTL eviction, pin/unpin
# =====================================================================


class TestInferenceModuleAcl:
    async def test_module_access_denied(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "acl-deny", model_modules="rag")
        await _publish(client, m["id"])
        await _create_published_version(client, m["id"])
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time(),
        }
        patcher, mock_ctx = _mock_httpx_inference(
            response_json={"id": "c1", "choices": [{"message": {"content": "hi"}}], "usage": {"total_tokens": 5}},
        )
        patcher.start()
        try:
            resp = await client.post(
                f"/api/v1/inference/{m['id']}/chat",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Fusion-Module": "code"},
            )
            assert resp.status_code == 403
            assert "not allowed" in resp.json()["detail"].lower()
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()

    async def test_module_access_allowed(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "acl-allow", model_modules="code,rag")
        await _publish(client, m["id"])
        await _create_published_version(client, m["id"])
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time(),
        }
        patcher, mock_ctx = _mock_httpx_inference(
            response_json={"id": "c1", "choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 3}},
        )
        patcher.start()
        try:
            resp = await client.post(
                f"/api/v1/inference/{m['id']}/chat",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Fusion-Module": "code"},
            )
            assert resp.status_code == 200
            assert resp.json()["id"] == "c1"
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()

    async def test_module_unknown_module_skipped(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "acl-unknown", model_modules="rag")
        await _publish(client, m["id"])
        await _create_published_version(client, m["id"])
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time(),
        }
        patcher, _ = _mock_httpx_inference(
            response_json={"id": "c2", "usage": {"total_tokens": 1}},
        )
        patcher.start()
        try:
            resp = await client.post(
                f"/api/v1/inference/{m['id']}/chat",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Fusion-Module": "unknown-mod"},
            )
            assert resp.status_code == 200
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()


class TestInferenceGrayRoute:
    async def test_chat_gray_version_routes_to_gray_model_name(self, client):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "gray-base", hf_repo="base/repo")
        await _publish(client, m["id"])
        gray_ver = await _create_published_version(client, m["id"], "9.0.0")
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": "base/repo",
            "status": "loaded",
            "loaded_at": time.time(),
        }
        sf = get_session_factory()
        async with sf() as session:
            d = await crud.create_deployment(
                session,
                model_id=m["id"],
                name="gray-dep",
                version_id="v1",
            )
            await crud.update_deployment(
                session,
                d.id,
                status="running",
                gray_enabled=True,
                gray_version_id=gray_ver["id"],
                gray_traffic_ratio=100,
            )
        with patch("fusion_model_hub.server.routers.inference.random.randint", return_value=50):
            patcher, mock_ctx = _mock_httpx_inference(
                response_json={"id": "gc", "usage": {"total_tokens": 2}},
            )
            patcher.start()
            try:
                resp = await client.post(
                    f"/api/v1/inference/{m['id']}/chat",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
                assert resp.status_code == 200
                sent_payload = mock_ctx.post.call_args.kwargs.get("json", {})
                assert sent_payload["model"] == "base/repo"
            finally:
                patcher.stop()
                inf_mod._loaded_models.clear()

    async def test_completions_gray_version_lookup(self, client):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "gray-comp", hf_repo="comp/repo")
        await _publish(client, m["id"])
        gray_ver = await _create_published_version(client, m["id"], "2.0.0")
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": "comp/repo",
            "status": "loaded",
            "loaded_at": time.time(),
        }
        sf = get_session_factory()
        async with sf() as session:
            d = await crud.create_deployment(
                session,
                model_id=m["id"],
                name="gray-comp-dep",
                version_id="v1",
            )
            await crud.update_deployment(
                session,
                d.id,
                status="running",
                gray_enabled=True,
                gray_version_id=gray_ver["id"],
                gray_traffic_ratio=100,
            )
        with patch("fusion_model_hub.server.routers.inference.random.randint", return_value=50):
            patcher, mock_ctx = _mock_httpx_inference(
                response_json={"id": "gc2", "usage": {"total_tokens": 4}},
            )
            patcher.start()
            try:
                resp = await client.post(
                    f"/api/v1/inference/{m['id']}/completions",
                    json={"prompt": "hi"},
                )
                assert resp.status_code == 200
                assert resp.json()["id"] == "gc2"
            finally:
                patcher.stop()
                inf_mod._loaded_models.clear()


class TestInferenceTtlEviction:
    async def test_ttl_eviction_unloads_expired_unpinned(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        inf_mod._last_cleanup_ts = 0.0
        m = await _create_model(client, "ttl-evict", idle_timeout_minutes=0)
        await _publish(client, m["id"])
        await _create_published_version(client, m["id"])
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time() - 9999,
        }
        patcher, mock_ctx = _mock_httpx_inference()
        patcher.start()
        try:
            await inf_mod._cleanup_loaded_models()
            assert m["id"] not in inf_mod._loaded_models
            assert mock_ctx.post.called
            called_url = mock_ctx.post.call_args.args[0]
            assert "/unload" in called_url
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()
            inf_mod._last_cleanup_ts = 0.0

    async def test_ttl_skips_pinned_model(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        inf_mod._last_cleanup_ts = 0.0
        m = await _create_model(client, "ttl-pinned")
        await _publish(client, m["id"])
        await _create_published_version(client, m["id"])
        await client.post(f"/api/v1/models/{m['id']}/pin")
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time() - 9999,
        }
        patcher, _ = _mock_httpx_inference()
        patcher.start()
        try:
            await inf_mod._cleanup_loaded_models()
            assert m["id"] in inf_mod._loaded_models
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()
            inf_mod._last_cleanup_ts = 0.0

    async def test_ttl_throttle_skips_recent_sweep(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        inf_mod._last_cleanup_ts = time.time()
        m = await _create_model(client, "ttl-throttle")
        await _publish(client, m["id"])
        await _create_published_version(client, m["id"])
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time() - 9999,
        }
        patcher, mock_ctx = _mock_httpx_inference()
        patcher.start()
        try:
            await inf_mod._cleanup_loaded_models()
            assert not mock_ctx.post.called
            assert m["id"] in inf_mod._loaded_models
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()
            inf_mod._last_cleanup_ts = 0.0


class TestInferencePinUnpin:
    async def test_pin_then_unpin(self, client):
        m = await _create_model(client, "pin-target")
        resp = await client.post(f"/api/v1/models/{m['id']}/pin")
        assert resp.status_code == 200
        assert resp.json()["pinned"] is True
        status = await client.get(f"/api/v1/models/{m['id']}/serve")
        assert status.status_code == 200
        resp = await client.delete(f"/api/v1/models/{m['id']}/pin")
        assert resp.status_code == 200
        assert resp.json()["pinned"] is False

    async def test_pin_model_not_found(self, client):
        resp = await client.post("/api/v1/models/no-such-id/pin")
        assert resp.status_code == 404

    async def test_unpin_model_not_found(self, client):
        resp = await client.delete("/api/v1/models/no-such-id/pin")
        assert resp.status_code == 404


class TestInferenceServeFileIntegrity:
    async def test_serve_missing_file_403(self, client, settings):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "serve-missing-file")
        await _publish(client, m["id"])
        ver = await _create_published_version(client, m["id"])
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            await crud.update_version(session, ver["id"], file_path="/tmp/definitely-missing-fmh-cov")
        patcher, _ = _mock_httpx_inference()
        patcher.start()
        try:
            resp = await client.post(f"/api/v1/models/{m['id']}/serve", json={})
            assert resp.status_code == 403
            assert "integrity" in resp.json()["detail"].lower()
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()

    async def test_serve_file_hash_mismatch_403(self, client, settings, tmp_path):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "serve-hash-mismatch")
        await _publish(client, m["id"])
        ver = await _create_published_version(client, m["id"])
        f = tmp_path / "weights.bin"
        f.write_bytes(b"real-weights-content")
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            await crud.update_version(
                session,
                ver["id"],
                file_path=str(f),
                file_hash="0" * 64,
            )
        patcher, _ = _mock_httpx_inference()
        patcher.start()
        try:
            resp = await client.post(f"/api/v1/models/{m['id']}/serve", json={})
            assert resp.status_code == 403
            assert "integrity" in resp.json()["detail"].lower()
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()

    async def test_serve_computes_missing_hash(self, client, tmp_path):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "serve-compute-hash")
        await _publish(client, m["id"])
        ver = await _create_published_version(client, m["id"])
        f = tmp_path / "weights2.bin"
        f.write_bytes(b"payload-for-hash")
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            await crud.update_version(session, ver["id"], file_path=str(f))
        patcher, _ = _mock_httpx_inference()
        patcher.start()
        try:
            resp = await client.post(f"/api/v1/models/{m['id']}/serve", json={})
            assert resp.status_code == 200
            assert resp.json()["status"] == "loaded"
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()

    async def test_serve_draft_model_forbidden(self, client):
        m = await _create_model(client, "serve-draft")
        resp = await client.post(f"/api/v1/models/{m['id']}/serve", json={})
        assert resp.status_code == 403
        assert "published" in resp.json()["detail"].lower()

    async def test_serve_deprecated_model_forbidden(self, client):
        m = await _create_model(client, "serve-deprecated")
        await _publish(client, m["id"])
        await _create_published_version(client, m["id"])
        await client.post(f"/api/v1/models/{m['id']}/deprecate")
        resp = await client.post(f"/api/v1/models/{m['id']}/serve", json={})
        assert resp.status_code == 403
        assert "deprecated" in resp.json()["detail"].lower()


class TestInferenceEmbeddings:
    async def test_embeddings_success(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "emb-ok", hf_repo="emb/model")
        await _publish(client, m["id"])
        await _create_published_version(client, m["id"])
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": "emb/model",
            "status": "loaded",
            "loaded_at": time.time(),
        }
        patcher, mock_ctx = _mock_httpx_inference(
            response_json={"data": [{"embedding": [0.1, 0.2]}], "usage": {"total_tokens": 2}},
        )
        patcher.start()
        try:
            resp = await client.post(
                f"/api/v1/inference/{m['id']}/embeddings",
                json={"input": "hello"},
            )
            assert resp.status_code == 200
            assert resp.json()["data"][0]["embedding"] == [0.1, 0.2]
            sent = mock_ctx.post.call_args.kwargs.get("json", {})
            assert sent["model"] == "emb/model"
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()

    async def test_embeddings_not_loaded(self, client):
        m = await _create_model(client, "emb-not-loaded")
        await _publish(client, m["id"])
        resp = await client.post(
            f"/api/v1/inference/{m['id']}/embeddings",
            json={"input": "hello"},
        )
        assert resp.status_code == 400
        assert "not loaded" in resp.json()["detail"].lower()

    async def test_embeddings_mlx_connect_error_503(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "emb-conn-err")
        await _publish(client, m["id"])
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time(),
        }
        patcher, _ = _mock_httpx_inference(side_effect=httpx.ConnectError("refused"))
        patcher.start()
        try:
            resp = await client.post(
                f"/api/v1/inference/{m['id']}/embeddings",
                json={"input": "hello"},
            )
            assert resp.status_code == 503
            assert "unavailable" in resp.json()["detail"].lower()
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()


class TestInferenceCompletions:
    async def test_completions_success(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "comp-ok", hf_repo="comp/model")
        await _publish(client, m["id"])
        await _create_published_version(client, m["id"])
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": "comp/model",
            "status": "loaded",
            "loaded_at": time.time(),
        }
        patcher, _ = _mock_httpx_inference(
            response_json={"id": "tc1", "choices": [{"text": "ok"}], "usage": {"total_tokens": 7}},
        )
        patcher.start()
        try:
            resp = await client.post(
                f"/api/v1/inference/{m['id']}/completions",
                json={"prompt": "hi"},
            )
            assert resp.status_code == 200
            assert resp.json()["id"] == "tc1"
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()

    async def test_completions_mlx_status_error(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "comp-err")
        await _publish(client, m["id"])
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time(),
        }
        bad_resp = MagicMock()
        bad_resp.status_code = 500
        bad_resp.text = "boom"
        bad_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err",
            request=MagicMock(),
            response=bad_resp,
        )
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.post = AsyncMock(return_value=bad_resp)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_ctx):
            resp = await client.post(
                f"/api/v1/inference/{m['id']}/completions",
                json={"prompt": "hi"},
            )
        assert resp.status_code == 500
        assert "trace_id" in resp.json()["detail"]
        inf_mod._loaded_models.clear()

    async def test_chat_mlx_status_error(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "chat-err")
        await _publish(client, m["id"])
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time(),
        }
        bad_resp = MagicMock()
        bad_resp.status_code = 429
        bad_resp.text = "rate"
        bad_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err",
            request=MagicMock(),
            response=bad_resp,
        )
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.post = AsyncMock(return_value=bad_resp)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_ctx):
            resp = await client.post(
                f"/api/v1/inference/{m['id']}/chat",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 429
        inf_mod._loaded_models.clear()

    async def test_get_serve_status_loaded(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "status-loaded")
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v9",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time(),
        }
        try:
            resp = await client.get(f"/api/v1/models/{m['id']}/serve")
            assert resp.status_code == 200
            assert resp.json()["status"] == "loaded"
            assert resp.json()["version_id"] == "v9"
        finally:
            inf_mod._loaded_models.clear()

    async def test_unload_model_success(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "unload-ok")
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time(),
        }
        patcher, _ = _mock_httpx_inference()
        patcher.start()
        try:
            resp = await client.delete(f"/api/v1/models/{m['id']}/serve")
            assert resp.status_code == 200
            assert resp.json()["status"] == "unloaded"
            assert m["id"] not in inf_mod._loaded_models
        finally:
            patcher.stop()
            inf_mod._loaded_models.clear()

    async def test_unload_not_loaded_404(self, client):
        m = await _create_model(client, "unload-none")
        resp = await client.delete(f"/api/v1/models/{m['id']}/serve")
        assert resp.status_code == 404


# =====================================================================
# cluster.py — register validation, heartbeat reap, round-robin, failover
# =====================================================================


class TestClusterNodeValidation:
    async def test_register_rejects_non_http_scheme(self, client):
        resp = await client.post(
            "/api/v1/cluster/nodes",
            json={
                "name": "bad",
                "url": "file:///etc/passwd",
            },
        )
        assert resp.status_code == 400
        assert "http" in resp.json()["detail"].lower()

    async def test_register_rejects_missing_host(self, client):
        resp = await client.post(
            "/api/v1/cluster/nodes",
            json={
                "name": "bad",
                "url": "http://",
            },
        )
        assert resp.status_code == 400
        assert "hostname" in resp.json()["detail"].lower()

    async def test_register_rejects_link_local(self, client):
        resp = await client.post(
            "/api/v1/cluster/nodes",
            json={
                "name": "meta",
                "url": "http://169.254.169.254/latest",
            },
        )
        assert resp.status_code == 400
        assert "link-local" in resp.json()["detail"].lower()

    async def test_register_rejects_unspecified(self, client):
        resp = await client.post(
            "/api/v1/cluster/nodes",
            json={
                "name": "zero",
                "url": "http://0.0.0.0:11434",
            },
        )
        assert resp.status_code == 400

    async def test_register_allows_loopback(self, client):
        resp = await client.post(
            "/api/v1/cluster/nodes",
            json={
                "name": "loopback-peer",
                "url": "http://127.0.0.1:11445",
            },
        )
        assert resp.status_code == 201
        assert resp.json()["host"] == "127.0.0.1"
        assert resp.json()["port"] == 11445

    async def test_get_node_and_delete(self, client):
        node = await _register_node(client, "getdel", "http://10.0.0.2:11434")
        resp = await client.get(f"/api/v1/cluster/nodes/{node['id']}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "getdel"
        resp = await client.delete(f"/api/v1/cluster/nodes/{node['id']}")
        assert resp.status_code == 200
        resp = await client.get(f"/api/v1/cluster/nodes/{node['id']}")
        assert resp.status_code == 404

    async def test_get_node_not_found(self, client):
        resp = await client.get("/api/v1/cluster/nodes/nope")
        assert resp.status_code == 404

    async def test_delete_node_not_found(self, client):
        resp = await client.delete("/api/v1/cluster/nodes/nope")
        assert resp.status_code == 404

    async def test_heartbeat_unknown_node(self, client):
        resp = await client.post("/api/v1/cluster/nodes/nope/heartbeat")
        assert resp.status_code == 404


class TestClusterReapAndList:
    async def test_list_nodes_includes_local_and_reaps_stale(self, client):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        node = await _register_node(client, "stale-one", "http://10.0.0.3:11434")
        sf = get_session_factory()
        async with sf() as session:
            n = await crud.get_cluster_node(session, node["id"])
            n.last_heartbeat = datetime.now(UTC) - timedelta(seconds=300)
            await session.commit()
        with _mock_httpx_ctx(response_status=200, response_json={"ok": True}):
            resp = await client.get("/api/v1/cluster/nodes")
        assert resp.status_code == 200
        data = resp.json()
        ids = [x["id"] for x in data["nodes"]]
        assert "local" in ids
        statuses = {x["id"]: x["status"] for x in data["nodes"]}
        assert statuses[node["id"]] == "inactive"

    async def test_heartbeat_marks_active(self, client):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        node = await _register_node(client, "hb-node", "http://10.0.0.4:11434")
        sf = get_session_factory()
        async with sf() as session:
            n = await crud.get_cluster_node(session, node["id"])
            n.last_heartbeat = datetime.now(UTC) - timedelta(seconds=300)
            await session.commit()
        resp = await client.post(f"/api/v1/cluster/nodes/{node['id']}/heartbeat")
        assert resp.status_code == 200
        with _mock_httpx_ctx(response_status=200):
            resp = await client.get("/api/v1/cluster/nodes")
        statuses = {x["id"]: x["status"] for x in resp.json()["nodes"]}
        assert statuses[node["id"]] == "active"

    async def test_topology_builds_edges_for_active(self, client):
        node = await _register_node(client, "topo-node", "http://10.0.0.5:11434")
        await client.post(f"/api/v1/cluster/nodes/{node['id']}/heartbeat")
        with _mock_httpx_ctx(response_status=200, response_json={"ok": True}):
            resp = await client.get("/api/v1/cluster/topology")
        assert resp.status_code == 200
        data = resp.json()
        assert data["localNode"] == "local"
        edge_ids = [e["to"] for e in data["edges"]]
        assert node["id"] in edge_ids


class TestClusterRouteInference:
    async def test_route_inference_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/cluster/route-inference",
            json={
                "model_id": "no-such",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 404

    async def test_route_inference_local_mode_success(self, client):
        m = await _create_model(client, "route-local", hf_repo="route/local")
        await _publish(client, m["id"])
        chat_resp = {
            "id": "r1",
            "model": "route/local",
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"total_tokens": 4, "prompt_tokens": 2, "completion_tokens": 2},
        }
        with _mock_httpx_ctx(response_status=200, response_json=chat_resp):
            resp = await client.post(
                "/api/v1/cluster/route-inference",
                json={
                    "model_id": m["id"],
                    "messages": [{"role": "user", "content": "hi"}],
                    "mode": "local",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["routedTo"] == "local"
        assert data["routeMode"] == "local"
        assert data["content"] == "hello"
        assert data["usage"]["total_tokens"] == 4

    async def test_route_inference_local_mode_connect_error_503(self, client):
        m = await _create_model(client, "route-local-err")
        await _publish(client, m["id"])

        class _LocalErrCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                raise httpx.ConnectError("local mlx refused")

            async def get(self, url, **kw):
                r = MagicMock()
                r.status_code = 200
                return r

        with (
            patch("httpx.AsyncClient", return_value=_LocalErrCtx()),
            patch("fusion_model_hub.server.http_client.AsyncClient", return_value=_LocalErrCtx()),
        ):
            resp = await client.post(
                "/api/v1/cluster/route-inference",
                json={
                    "model_id": m["id"],
                    "messages": [{"role": "user", "content": "hi"}],
                    "mode": "local",
                },
            )
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    async def test_route_inference_cluster_mode_round_robin(self, client):
        from fusion_model_hub.server.routers import cluster as cluster_mod

        cluster_mod._round_robin_counter.__init__()
        m = await _create_model(client, "route-rr", hf_repo="rr/model")
        await _publish(client, m["id"])
        n1 = await _register_node(client, "rr-a", "http://10.0.0.10:11434")
        n2 = await _register_node(client, "rr-b", "http://10.0.0.11:11434")
        n3 = await _register_node(client, "rr-c", "http://10.0.0.12:11434")
        for nid in (n1["id"], n2["id"], n3["id"]):
            await client.post(f"/api/v1/cluster/nodes/{nid}/heartbeat")

        routed = []
        call_counts = {"a": 0, "b": 0, "c": 0}
        host_map = {
            "10.0.0.10": ("a", n1["id"]),
            "10.0.0.11": ("b", n2["id"]),
            "10.0.0.12": ("c", n3["id"]),
        }

        def make_resp(node_id):
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {
                "id": f"r-{node_id}",
                "model": "rr/model",
                "choices": [{"message": {"content": node_id}}],
                "usage": {"total_tokens": 1, "prompt_tokens": 1, "completion_tokens": 0},
            }
            r.raise_for_status = MagicMock()
            return r

        class _RRCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                for host, (key, nid) in host_map.items():
                    if f"//{host}" in url:
                        call_counts[key] += 1
                        return make_resp(nid)
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = {}
                r.raise_for_status = MagicMock()
                return r

            async def get(self, url, **kw):
                r = MagicMock()
                r.status_code = 200
                return r

        with (
            patch("httpx.AsyncClient", return_value=_RRCtx()),
            patch("fusion_model_hub.server.http_client.AsyncClient", return_value=_RRCtx()),
        ):
            for _ in range(6):
                resp = await client.post(
                    "/api/v1/cluster/route-inference",
                    json={
                        "model_id": m["id"],
                        "messages": [{"role": "user", "content": "hi"}],
                        "mode": "cluster",
                    },
                )
                assert resp.status_code == 200
                routed.append(resp.json()["routedTo"])
        assert len(set(routed)) == 3
        assert call_counts["a"] >= 1
        assert call_counts["b"] >= 1
        assert call_counts["c"] >= 1

    async def test_route_inference_failover_to_second_node(self, client):
        m = await _create_model(client, "route-failover", hf_repo="fo/model")
        await _publish(client, m["id"])
        n1 = await _register_node(client, "fo-a", "http://10.0.0.20:11434")
        n2 = await _register_node(client, "fo-b", "http://10.0.0.21:11434")
        for nid in (n1["id"], n2["id"]):
            await client.post(f"/api/v1/cluster/nodes/{nid}/heartbeat")

        good = MagicMock()
        good.status_code = 200
        good.json.return_value = {
            "id": "fo-ok",
            "model": "fo/model",
            "choices": [{"message": {"content": "from-b"}}],
            "usage": {"total_tokens": 2, "prompt_tokens": 1, "completion_tokens": 1},
        }
        good.raise_for_status = MagicMock()

        class _FOCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                if "10.0.0.20" in url:
                    raise httpx.ConnectError("node a down")
                return good

            async def get(self, url, **kw):
                r = MagicMock()
                r.status_code = 200
                return r

        with (
            patch("httpx.AsyncClient", return_value=_FOCtx()),
            patch("fusion_model_hub.server.http_client.AsyncClient", return_value=_FOCtx()),
        ):
            resp = await client.post(
                "/api/v1/cluster/route-inference",
                json={
                    "model_id": m["id"],
                    "messages": [{"role": "user", "content": "hi"}],
                    "mode": "cluster",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["routedTo"] == n2["id"]
        assert resp.json()["content"] == "from-b"

    async def test_route_inference_no_active_nodes_503(self, client):
        m = await _create_model(client, "route-none")
        await _publish(client, m["id"])
        with _mock_httpx_ctx(side_effect=httpx.ConnectError("down")):
            resp = await client.post(
                "/api/v1/cluster/route-inference",
                json={
                    "model_id": m["id"],
                    "messages": [{"role": "user", "content": "hi"}],
                    "mode": "cluster",
                },
            )
        assert resp.status_code == 503
        assert "no available node" in resp.json()["detail"].lower()

    async def test_route_inference_auto_falls_to_cluster_when_local_down(self, client):
        m = await _create_model(client, "route-auto", hf_repo="auto/model")
        await _publish(client, m["id"])
        n1 = await _register_node(client, "auto-a", "http://10.0.0.30:11434")
        await client.post(f"/api/v1/cluster/nodes/{n1['id']}/heartbeat")
        good = MagicMock()
        good.status_code = 200
        good.json.return_value = {
            "id": "auto-ok",
            "model": "auto/model",
            "choices": [{"message": {"content": "remote"}}],
            "usage": {"total_tokens": 1, "prompt_tokens": 1, "completion_tokens": 0},
        }
        good.raise_for_status = MagicMock()

        class _AutoCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                if "127.0.0.1" in url or "localhost" in url:
                    raise httpx.ConnectError("local down")
                return good

            async def get(self, url, **kw):
                r = MagicMock()
                r.status_code = 500
                return r

        with (
            patch("httpx.AsyncClient", return_value=_AutoCtx()),
            patch("fusion_model_hub.server.http_client.AsyncClient", return_value=_AutoCtx()),
        ):
            resp = await client.post(
                "/api/v1/cluster/route-inference",
                json={
                    "model_id": m["id"],
                    "messages": [{"role": "user", "content": "hi"}],
                    "mode": "auto",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["routedTo"] == n1["id"]


class TestClusterSyncModel:
    async def test_sync_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/cluster/sync-model",
            json={
                "model_id": "no-such",
            },
        )
        assert resp.status_code == 404

    async def test_sync_model_local_ok_with_active_node(self, client):
        m = await _create_model(client, "sync-ok", hf_repo="sync/model")
        await _publish(client, m["id"])
        n1 = await _register_node(client, "sync-a", "http://10.0.0.40:11434")
        await client.post(f"/api/v1/cluster/nodes/{n1['id']}/heartbeat")
        load_resp = MagicMock()
        load_resp.status_code = 200

        class _SyncCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                r = MagicMock()
                if "/v1/models/" in url and "/load" in url:
                    r.status_code = 200
                elif "remote-sync" in url:
                    r.status_code = 202
                else:
                    r.status_code = 200
                return r

            async def get(self, url, **kw):
                r = MagicMock()
                r.status_code = 200
                return r

        with (
            patch("httpx.AsyncClient", return_value=_SyncCtx()),
            patch("fusion_model_hub.server.http_client.AsyncClient", return_value=_SyncCtx()),
        ):
            resp = await client.post(
                "/api/v1/cluster/sync-model",
                json={
                    "model_id": m["id"],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "remote_ok=1" in data["message"]

    async def test_sync_model_local_404_falls_back_to_chat_autoload(self, client):
        m = await _create_model(client, "sync-fb", hf_repo="sync/fb")
        await _publish(client, m["id"])

        class _FBCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                r = MagicMock()
                if "/v1/models/" in url and "/load" in url:
                    r.status_code = 404
                elif "/v1/chat/completions" in url:
                    r.status_code = 200
                else:
                    r.status_code = 200
                return r

            async def get(self, url, **kw):
                r = MagicMock()
                r.status_code = 200
                return r

        with (
            patch("httpx.AsyncClient", return_value=_FBCtx()),
            patch("fusion_model_hub.server.http_client.AsyncClient", return_value=_FBCtx()),
        ):
            resp = await client.post(
                "/api/v1/cluster/sync-model",
                json={
                    "model_id": m["id"],
                    "target_nodes": ["local"],
                },
            )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    async def test_sync_model_all_dead_returns_failure(self, client):
        m = await _create_model(client, "sync-dead", hf_repo="sync/dead")
        await _publish(client, m["id"])
        n1 = await _register_node(client, "sync-dead-node", "http://10.0.0.41:11434")
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            n = await crud.get_cluster_node(session, n1["id"])
            n.last_heartbeat = datetime.now(UTC) - timedelta(seconds=300)
            await session.commit()
        with _mock_httpx_ctx(side_effect=httpx.ConnectError("down")):
            resp = await client.post(
                "/api/v1/cluster/sync-model",
                json={
                    "model_id": m["id"],
                },
            )
        assert resp.status_code == 200
        assert resp.json()["success"] is False


@pytest.fixture
async def file_client():
    from fusion_model_hub.server.auth import set_auth_enabled

    set_auth_enabled(False)
    db_path = "/tmp/fmh_cov_inf_cluster_file.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    file_settings = Settings(
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_cov_inf_cluster_file",
        db_url=f"sqlite+aiosqlite:///{db_path}",
        log_level="WARNING",
    )
    file_app = create_app(file_settings)
    engine = get_engine(file_settings.db_url)
    await init_db(engine)
    init_deps(file_settings, engine)
    transport = ASGITransport(app=file_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await engine.dispose()
    for p in (db_path, f"{db_path}-wal", f"{db_path}-shm"):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree("/tmp/fmh_cov_inf_cluster_file", ignore_errors=True)


class TestClusterDistributedTask:
    async def test_submit_task_unknown_target_node_404(self, client):
        m = await _create_model(client, "dist-bad")
        resp = await client.post(
            "/api/v1/cluster/distributed-tasks",
            json={
                "model_id": m["id"],
                "target_nodes": ["no-such-node"],
            },
        )
        assert resp.status_code == 404

    async def test_submit_task_no_targets_fails(self, client):
        m = await _create_model(client, "dist-empty")
        resp = await client.post(
            "/api/v1/cluster/distributed-tasks",
            json={
                "model_id": m["id"],
                "target_nodes": [],
            },
        )
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        await asyncio.sleep(0.1)
        status = await client.get(f"/api/v1/cluster/distributed-tasks/{task_id}")
        assert status.status_code == 200
        assert status.json()["status"] == "failed"

    async def test_submit_task_with_targets_completes(self, file_client):
        c = file_client
        m = await _create_model(c, "dist-ok")
        n1 = await _register_node(c, "dist-a", "http://10.0.0.50:11434")
        await c.post(f"/api/v1/cluster/nodes/{n1['id']}/heartbeat")

        class _DistCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                r = MagicMock()
                r.status_code = 200
                return r

            async def get(self, url, **kw):
                r = MagicMock()
                r.status_code = 200
                return r

        with (
            patch("httpx.AsyncClient", return_value=_DistCtx()),
            patch("fusion_model_hub.server.http_client.AsyncClient", return_value=_DistCtx()),
        ):
            resp = await c.post(
                "/api/v1/cluster/distributed-tasks",
                json={
                    "model_id": m["id"],
                    "target_nodes": [n1["id"]],
                },
            )
            assert resp.status_code == 202
            task_id = resp.json()["task_id"]
            status = None
            for _ in range(20):
                await asyncio.sleep(0.2)
                status = await c.get(f"/api/v1/cluster/distributed-tasks/{task_id}")
                if status.json()["status"] != "running":
                    break
        assert status.status_code == 200
        assert status.json()["status"] == "completed", status.json()

    async def test_submit_task_partial_when_node_sync_fails(self, file_client):
        c = file_client
        m = await _create_model(c, "dist-partial")
        n1 = await _register_node(c, "dist-p1", "http://10.0.0.51:11434")
        n2 = await _register_node(c, "dist-p2", "http://10.0.0.52:11434")
        for nid in (n1["id"], n2["id"]):
            await c.post(f"/api/v1/cluster/nodes/{nid}/heartbeat")

        class _PCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                if "10.0.0.51" in url:
                    r = MagicMock()
                    r.status_code = 200
                    return r
                raise httpx.ConnectError("node b down")

            async def get(self, url, **kw):
                r = MagicMock()
                r.status_code = 200
                return r

        with (
            patch("httpx.AsyncClient", return_value=_PCtx()),
            patch("fusion_model_hub.server.http_client.AsyncClient", return_value=_PCtx()),
        ):
            resp = await c.post(
                "/api/v1/cluster/distributed-tasks",
                json={
                    "model_id": m["id"],
                    "target_nodes": [n1["id"], n2["id"]],
                },
            )
            task_id = resp.json()["task_id"]
            status = None
            for _ in range(20):
                await asyncio.sleep(0.2)
                status = await c.get(f"/api/v1/cluster/distributed-tasks/{task_id}")
                if status.json()["status"] != "running":
                    break
        assert status.status_code == 200
        assert status.json()["status"] == "partial", status.json()

    async def test_get_distributed_task_not_found(self, client):
        resp = await client.get("/api/v1/cluster/distributed-tasks/nope")
        assert resp.status_code == 404


class TestClusterRemoteSync:
    async def test_remote_sync_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/cluster/remote-sync",
            json={
                "model_id": "no-such",
            },
        )
        assert resp.status_code == 404

    async def test_remote_sync_load_accepted(self, client):
        m = await _create_model(client, "rsync-ok", hf_repo="rsync/model")
        await _publish(client, m["id"])
        with _mock_httpx_ctx(response_status=200):
            resp = await client.post(
                "/api/v1/cluster/remote-sync",
                json={
                    "model_id": m["id"],
                },
            )
        assert resp.status_code == 200
        assert resp.json()["accepted"] is True
        assert resp.json()["model_name"] == "rsync/model"

    async def test_remote_sync_load_already_loaded_409(self, client):
        m = await _create_model(client, "rsync-409", hf_repo="rsync/409")
        await _publish(client, m["id"])
        with _mock_httpx_ctx(response_status=409):
            resp = await client.post(
                "/api/v1/cluster/remote-sync",
                json={
                    "model_id": m["id"],
                },
            )
        assert resp.status_code == 200

    async def test_remote_sync_load_rejected_502(self, client):
        m = await _create_model(client, "rsync-rej", hf_repo="rsync/rej")
        await _publish(client, m["id"])
        with _mock_httpx_ctx(response_status=500):
            resp = await client.post(
                "/api/v1/cluster/remote-sync",
                json={
                    "model_id": m["id"],
                },
            )
        assert resp.status_code == 502

    async def test_remote_sync_load_connect_error_503(self, client):
        m = await _create_model(client, "rsync-conn", hf_repo="rsync/conn")
        await _publish(client, m["id"])
        with _mock_httpx_ctx(side_effect=httpx.ConnectError("down")):
            resp = await client.post(
                "/api/v1/cluster/remote-sync",
                json={
                    "model_id": m["id"],
                },
            )
        assert resp.status_code == 503


# =====================================================================
# adapt.py — assess, plan, execute pipeline, status
# =====================================================================


class TestAdaptAssessPlan:
    def _adapt_result(self, model_id="m1", level="L2"):
        from fusion_model_hub.adapt.types import AdaptationLevel, AdaptationResult, MigrationCost

        return AdaptationResult(
            model_id=model_id,
            level=AdaptationLevel(level),
            level_desc="test",
            migration_cost=MigrationCost.medium,
            components_matched=["attention"],
            missing_ops=[],
        )

    def _migration_plan(self, model_id="m1", level="L2"):
        from fusion_model_hub.adapt.types import AdaptationLevel, MigrationPlan

        return MigrationPlan(
            model_id=model_id,
            level=AdaptationLevel(level),
            steps=["convert", "quantize"],
            estimated_vram_gb=8.0,
            estimated_speed_tok_per_sec=50.0,
        )

    async def test_assess_success(self, client):
        result = self._adapt_result("assess-ok", "L2")
        with patch("fusion_model_hub.server.routers.adapt._get_engine") as ge:
            engine = MagicMock()
            engine.assess = AsyncMock(return_value=result)
            ge.return_value = engine
            resp = await client.post(
                "/api/v1/adapt/assess",
                json={
                    "model_id": "assess-ok",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["level"] == "L2"
        assert resp.json()["components_matched"] == ["attention"]

    async def test_assess_engine_failure_503(self, client):
        with patch("fusion_model_hub.server.routers.adapt._get_engine") as ge:
            engine = MagicMock()
            engine.assess = AsyncMock(side_effect=RuntimeError("mlx down"))
            ge.return_value = engine
            resp = await client.post(
                "/api/v1/adapt/assess",
                json={
                    "model_id": "assess-fail",
                },
            )
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    async def test_plan_success(self, client):
        plan = self._migration_plan("plan-ok", "L1")
        with patch("fusion_model_hub.server.routers.adapt._get_engine") as ge:
            engine = MagicMock()
            engine.assess_and_plan = AsyncMock(return_value=plan)
            ge.return_value = engine
            resp = await client.post(
                "/api/v1/adapt/plan",
                json={
                    "model_id": "plan-ok",
                    "params_b": 7.0,
                },
            )
        assert resp.status_code == 200
        assert resp.json()["level"] == "L1"
        assert resp.json()["steps"] == ["convert", "quantize"]

    async def test_plan_engine_failure_503(self, client):
        with patch("fusion_model_hub.server.routers.adapt._get_engine") as ge:
            engine = MagicMock()
            engine.assess_and_plan = AsyncMock(side_effect=RuntimeError("nope"))
            ge.return_value = engine
            resp = await client.post(
                "/api/v1/adapt/plan",
                json={
                    "model_id": "plan-fail",
                    "params_b": 3.0,
                },
            )
        assert resp.status_code == 503


class TestAdaptExecute:
    async def test_execute_accepted_and_status_running(self, client):
        from fusion_model_hub.adapt.types import AdaptationLevel, AdaptationResult, MigrationCost

        result = AdaptationResult(
            model_id="exec-ok",
            level=AdaptationLevel.L1,
            level_desc="ok",
            migration_cost=MigrationCost.low,
        )
        convert_resp = MagicMock()
        convert_resp.status_code = 200
        convert_resp.text = ""
        quant_resp = MagicMock()
        quant_resp.status_code = 202
        quant_resp.text = ""

        class _ExecCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                if "/v1/convert" in url:
                    return convert_resp
                if "/v1/quantize" in url:
                    return quant_resp
                r = MagicMock()
                r.status_code = 200
                return r

        with (
            patch("fusion_model_hub.server.routers.adapt._get_engine") as ge,
            patch("httpx.AsyncClient", return_value=_ExecCtx()),
        ):
            engine = MagicMock()
            engine.assess = AsyncMock(return_value=result)
            ge.return_value = engine
            resp = await client.post(
                "/api/v1/adapt/execute",
                json={
                    "model_id": "exec-ok",
                    "quant_bits": 4,
                    "params_b": 7.0,
                },
            )
            assert resp.status_code == 202
            exec_id = resp.json()["execution_id"]
            assert resp.json()["hub_registered"] is False
            status = None
            for _ in range(20):
                await asyncio.sleep(0.15)
                status = await client.get(f"/api/v1/adapt/execute/{exec_id}")
                if status.json()["status"] != "running":
                    break
        assert status.status_code == 200
        assert status.json()["status"] == "completed"

    async def test_execute_quantize_failure_reports_failed(self, client):
        from fusion_model_hub.adapt.types import AdaptationLevel, AdaptationResult, MigrationCost

        result = AdaptationResult(
            model_id="exec-qfail",
            level=AdaptationLevel.L1,
            level_desc="ok",
            migration_cost=MigrationCost.low,
        )
        convert_resp = MagicMock()
        convert_resp.status_code = 200
        convert_resp.text = ""
        quant_resp = MagicMock()
        quant_resp.status_code = 500
        quant_resp.text = "quantize boom"

        class _QCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                if "/v1/convert" in url:
                    return convert_resp
                if "/v1/quantize" in url:
                    return quant_resp
                r = MagicMock()
                r.status_code = 200
                return r

        with (
            patch("fusion_model_hub.server.routers.adapt._get_engine") as ge,
            patch("httpx.AsyncClient", return_value=_QCtx()),
        ):
            engine = MagicMock()
            engine.assess = AsyncMock(return_value=result)
            ge.return_value = engine
            resp = await client.post(
                "/api/v1/adapt/execute",
                json={
                    "model_id": "exec-qfail",
                    "quant_bits": 4,
                    "params_b": 7.0,
                },
            )
            exec_id = resp.json()["execution_id"]
            status = None
            for _ in range(20):
                await asyncio.sleep(0.15)
                status = await client.get(f"/api/v1/adapt/execute/{exec_id}")
                if status.json()["status"] != "running":
                    break
        assert status.status_code == 200
        assert status.json()["status"] == "failed"
        assert "quantize failed" in status.json()["error"]

    async def test_execute_l4_aborts_without_quantize(self, client):
        from fusion_model_hub.adapt.types import AdaptationLevel, AdaptationResult, MigrationCost

        result = AdaptationResult(
            model_id="exec-l4",
            level=AdaptationLevel.L4,
            level_desc="unsupported",
            migration_cost=MigrationCost.extreme,
        )
        posted = []

        class _L4Ctx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                posted.append(url)
                r = MagicMock()
                r.status_code = 200
                return r

        with (
            patch("fusion_model_hub.server.routers.adapt._get_engine") as ge,
            patch("httpx.AsyncClient", return_value=_L4Ctx()),
        ):
            engine = MagicMock()
            engine.assess = AsyncMock(return_value=result)
            ge.return_value = engine
            resp = await client.post(
                "/api/v1/adapt/execute",
                json={
                    "model_id": "exec-l4",
                    "quant_bits": 4,
                    "params_b": 7.0,
                },
            )
            exec_id = resp.json()["execution_id"]
            status = None
            for _ in range(20):
                await asyncio.sleep(0.15)
                status = await client.get(f"/api/v1/adapt/execute/{exec_id}")
                if status.json()["status"] != "running":
                    break
        assert all("/v1/convert" not in u for u in posted)
        assert status.status_code == 200
        assert status.json()["status"] == "completed"

    async def test_execute_convert_failure_reports_failed(self, client):
        from fusion_model_hub.adapt.types import AdaptationLevel, AdaptationResult, MigrationCost

        result = AdaptationResult(
            model_id="exec-cfail",
            level=AdaptationLevel.L1,
            level_desc="ok",
            migration_cost=MigrationCost.low,
        )
        convert_resp = MagicMock()
        convert_resp.status_code = 500
        convert_resp.text = "convert boom"
        quant_resp = MagicMock()
        quant_resp.status_code = 200
        quant_resp.text = ""

        class _CFailCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                if "/v1/convert" in url:
                    return convert_resp
                if "/v1/quantize" in url:
                    return quant_resp
                r = MagicMock()
                r.status_code = 200
                return r

        with (
            patch("fusion_model_hub.server.routers.adapt._get_engine") as ge,
            patch("httpx.AsyncClient", return_value=_CFailCtx()),
        ):
            engine = MagicMock()
            engine.assess = AsyncMock(return_value=result)
            ge.return_value = engine
            resp = await client.post(
                "/api/v1/adapt/execute",
                json={
                    "model_id": "exec-cfail",
                    "quant_bits": 4,
                    "params_b": 7.0,
                },
            )
            exec_id = resp.json()["execution_id"]
            status = None
            for _ in range(20):
                await asyncio.sleep(0.15)
                status = await client.get(f"/api/v1/adapt/execute/{exec_id}")
                if status.json()["status"] != "running":
                    break
        assert status.status_code == 200
        assert status.json()["status"] == "failed"
        assert "convert failed" in status.json()["error"]

    async def test_get_execution_status_not_found(self, client):
        resp = await client.get("/api/v1/adapt/execute/no-such-exec")
        assert resp.status_code == 404


# =====================================================================
# hardware.py — detect, refresh, no-gpu, error
# =====================================================================


class TestHardwareDetect:
    def _profile(self, with_gpu=True):
        from fusion_model_hub.hardware.types import (
            ChipGeneration,
            CPUProfile,
            GPUProfile,
            HardwareProfile,
        )

        gpu = None
        if with_gpu:
            gpu = GPUProfile(
                name="Apple M2 Max",
                vendor="apple",
                vram_bytes=24_000_000_000,
                vram_gb=24.0,
                memory_bandwidth_gbps=400.0,
                shared_memory=True,
                chip_generation=ChipGeneration.M2_MAX,
            )
        return HardwareProfile(
            gpu=gpu,
            cpu=CPUProfile(name="Apple M2 Max", cores=12),
            ram_bytes=64_000_000_000,
            ram_gb=64.0,
            disk_free_bytes=500_000_000_000,
            disk_free_gb=500.0,
            os_name="macOS",
        )

    async def test_detect_with_gpu(self, client):
        from fusion_model_hub.server.routers import hardware as hw_mod

        hw_mod._detector = None
        profile = self._profile(with_gpu=True)
        with patch("fusion_model_hub.server.routers.hardware._get_detector") as gd:
            detector = MagicMock()
            detector.detect = AsyncMock(return_value=profile)
            gd.return_value = detector
            resp = await client.get("/api/v1/hardware")
        assert resp.status_code == 200
        data = resp.json()
        assert data["chip"] == "Apple M2 Max"
        assert data["cpuCores"] == 12
        assert data["gpuCores"] == 0
        assert data["memoryGB"] == 64.0
        assert data["gpu"]["chip_generation"] == "M2_Max"
        assert data["metalSupport"] is True

    async def test_detect_no_gpu(self, client):
        from fusion_model_hub.server.routers import hardware as hw_mod

        hw_mod._detector = None
        profile = self._profile(with_gpu=False)
        with patch("fusion_model_hub.server.routers.hardware._get_detector") as gd:
            detector = MagicMock()
            detector.detect = AsyncMock(return_value=profile)
            gd.return_value = detector
            resp = await client.get("/api/v1/hardware")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gpu"] is None
        assert data["chip"] == "Unknown"
        assert data["gpuCores"] == 0

    async def test_detect_failure_503(self, client):
        from fusion_model_hub.server.routers import hardware as hw_mod

        hw_mod._detector = None
        with patch("fusion_model_hub.server.routers.hardware._get_detector") as gd:
            detector = MagicMock()
            detector.detect = AsyncMock(side_effect=RuntimeError("mlx down"))
            gd.return_value = detector
            resp = await client.get("/api/v1/hardware")
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    async def test_refresh_success(self, client):
        from fusion_model_hub.server.routers import hardware as hw_mod

        hw_mod._detector = None
        profile = self._profile(with_gpu=True)
        with patch("fusion_model_hub.server.routers.hardware._get_detector") as gd:
            detector = MagicMock()
            detector.detect = AsyncMock(return_value=profile)
            detector.invalidate_cache = MagicMock()
            gd.return_value = detector
            resp = await client.post("/api/v1/hardware/refresh")
        assert resp.status_code == 200
        assert resp.json()["status"] == "refreshed"
        assert resp.json()["chip"] == "M2_Max"
        assert detector.invalidate_cache.called

    async def test_refresh_failure_503(self, client):
        from fusion_model_hub.server.routers import hardware as hw_mod

        hw_mod._detector = None
        with patch("fusion_model_hub.server.routers.hardware._get_detector") as gd:
            detector = MagicMock()
            detector.detect = AsyncMock(side_effect=RuntimeError("nope"))
            detector.invalidate_cache = MagicMock()
            gd.return_value = detector
            resp = await client.post("/api/v1/hardware/refresh")
        assert resp.status_code == 503


# =====================================================================
# recommend.py — recommend, quick, error, parse
# =====================================================================


class TestRecommend:
    def _rec_response(self, total=2):
        from fusion_model_hub.recommend.types import (
            ModelRecommendation,
            RecommendResponse,
        )

        rec = ModelRecommendation(
            model_id="m1",
            name="m1",
            task="llm",
            params_b=7.0,
            quant_type="Q4_K_M",
            can_run=True,
            fit_type="full",
            vram_required_gb=8.0,
            vram_available_gb=24.0,
            estimated_tok_per_sec=50.0,
            rank_score=90.0,
            quality_score=80.0,
            speed_score=70.0,
            hardware_score=85.0,
            popularity_score=60.0,
            reason="fit",
        )
        return RecommendResponse(
            recommendations=[rec],
            hardware_summary={"chip": "M2"},
            total_evaluated=total,
        )

    async def test_recommend_models_success(self, client):
        from fusion_model_hub.server.routers import recommend as rec_mod

        rec_mod._engine = None
        rec_resp = self._rec_response(3)
        with patch("fusion_model_hub.server.routers.recommend._get_engine") as ge:
            engine = MagicMock()
            engine.recommend = AsyncMock(return_value=rec_resp)
            ge.return_value = engine
            resp = await client.post(
                "/api/v1/recommend",
                json={
                    "task": "llm",
                    "preference": "balanced",
                    "max_results": 5,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_evaluated"] == 3
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["model_id"] == "m1"

    async def test_recommend_engine_failure_503(self, client):
        from fusion_model_hub.server.routers import recommend as rec_mod

        rec_mod._engine = None
        with patch("fusion_model_hub.server.routers.recommend._get_engine") as ge:
            engine = MagicMock()
            engine.recommend = AsyncMock(side_effect=RuntimeError("mlx down"))
            ge.return_value = engine
            resp = await client.post(
                "/api/v1/recommend",
                json={
                    "task": "llm",
                    "preference": "balanced",
                },
            )
        assert resp.status_code == 503
        assert "trace_id" in resp.json()["detail"]

    async def test_quick_recommend_success(self, client):
        from fusion_model_hub.server.routers import recommend as rec_mod

        rec_mod._engine = None
        rec_resp = self._rec_response(5)
        with patch("fusion_model_hub.server.routers.recommend._get_engine") as ge:
            engine = MagicMock()
            engine.recommend = AsyncMock(return_value=rec_resp)
            ge.return_value = engine
            resp = await client.get("/api/v1/recommend/quick?task=llm&max_results=3")
        assert resp.status_code == 200
        assert resp.json()["total_evaluated"] == 5

    async def test_quick_recommend_engine_failure_503(self, client):
        from fusion_model_hub.server.routers import recommend as rec_mod

        rec_mod._engine = None
        with patch("fusion_model_hub.server.routers.recommend._get_engine") as ge:
            engine = MagicMock()
            engine.recommend = AsyncMock(side_effect=RuntimeError("fail"))
            ge.return_value = engine
            resp = await client.get("/api/v1/recommend/quick")
        assert resp.status_code == 503

    async def test_recommend_reads_params_size_from_db(self, client):
        from fusion_model_hub.recommend.types import RecommendResponse
        from fusion_model_hub.server.routers import recommend as rec_mod

        rec_mod._engine = None
        await _create_model(client, "rec-7b", params_size="7B", task_types="llm")
        await _create_model(client, "rec-350m", params_size="350M", task_types="embedding")
        captured = {}

        async def fake_recommend(request, models_from_db):
            captured["models"] = models_from_db
            return RecommendResponse(
                recommendations=[],
                hardware_summary={},
                total_evaluated=len(models_from_db),
            )

        with patch("fusion_model_hub.server.routers.recommend._get_engine") as ge:
            engine = MagicMock()
            engine.recommend = AsyncMock(side_effect=fake_recommend)
            ge.return_value = engine
            resp = await client.post(
                "/api/v1/recommend",
                json={
                    "task": "all",
                    "max_results": 10,
                },
            )
        assert resp.status_code == 200
        by_name = {m["name"]: m for m in captured["models"]}
        assert by_name["rec-7b"]["params_b"] == 7.0
        assert by_name["rec-350m"]["params_b"] == 0.35
        assert by_name["rec-350m"]["task"] == "embedding"


# =====================================================================
# #12: per-inference DB roundtrip reduction — audit deferred + session reuse
# =====================================================================


class TestInferenceDbRoundtripReduction:
    async def test_chat_defers_audit_to_background(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "defer-audit")
        await _publish(client, m["id"])
        await _create_published_version(client, m["id"])
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time(),
        }
        patcher, _ = _mock_httpx_inference(
            response_json={"id": "c1", "choices": [{"message": {"content": "hi"}}], "usage": {"total_tokens": 5}},
        )
        patcher.start()
        # Patch the deferred coroutine — if the endpoint awaited it inline, the
        # patch would have run before the response. We assert the response still
        # returns 200 AND the audit coroutine was scheduled (background task).
        with patch("fusion_model_hub.server.routers.inference._write_inference_audit", new=AsyncMock()) as mock_audit:
            try:
                resp = await client.post(
                    f"/api/v1/inference/{m['id']}/chat",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
                assert resp.status_code == 200
                # The audit coroutine was scheduled (fire-and-forget), not
                # awaited inline — give the loop a tick to let it run.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            finally:
                patcher.stop()
                inf_mod._loaded_models.clear()
        mock_audit.assert_awaited()
        assert mock_audit.await_args.args[0] == m["id"]
        assert mock_audit.await_args.args[1] == "chat"

    async def test_chat_does_not_open_extra_session_for_module_acl(self, client):
        # Before #12, a chat call with an X-Fusion-Module header opened a
        # SECOND session inside _check_module_access (it did its own
        # get_model). Now the already-fetched model is reused, so the
        # request session is the only one.
        from fusion_model_hub.server.routers import inference as inf_mod

        inf_mod._loaded_models.clear()
        m = await _create_model(client, "session-reuse", model_modules="code")
        await _publish(client, m["id"])
        await _create_published_version(client, m["id"])
        inf_mod._loaded_models[m["id"]] = {
            "version_id": "v1",
            "model_name": m["name"],
            "status": "loaded",
            "loaded_at": time.time(),
        }
        patcher, _ = _mock_httpx_inference(
            response_json={"id": "c1", "usage": {"total_tokens": 2}},
        )
        # Count how many NEW sessions get_session_factory hands out for the
        # chat call. The request gets its own SessionDep session; the audit
        # runs on a background one. _check_module_access must NOT add another.
        sf_calls = {"n": 0}
        orig_sf = None
        from fusion_model_hub.server import deps as deps_mod

        orig_sf = deps_mod.get_session_factory

        class _CountingFactory:
            def __call__(self):
                return orig_sf()

        real_sf = orig_sf()

        class _CountingSF:
            def __call__(self):
                sf_calls["n"] += 1
                return real_sf

        with (
            patch.object(deps_mod, "get_session_factory", _CountingSF()),
            patch.object(inf_mod, "get_session_factory", _CountingSF()),
        ):
            patcher.start()
            try:
                resp = await client.post(
                    f"/api/v1/inference/{m['id']}/chat",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                    headers={"X-Fusion-Module": "code"},
                )
                assert resp.status_code == 200
                # Drain the deferred audit background task before counting,
                # so its session is counted too (proves audit still runs).
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            finally:
                patcher.stop()
                inf_mod._loaded_models.clear()
        # Request session (1) + deferred-audit session (1) = 2. Before #12,
        # _check_module_access + gray resolution each opened one more = 4+.
        assert sf_calls["n"] <= 2, sf_calls
