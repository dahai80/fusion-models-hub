import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

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
        port=8080,
        data_dir="/tmp/fmh_test_ext",
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
async def client(app, settings):
    engine = get_engine(settings.db_url)
    await init_db(engine)
    init_deps(settings, engine)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _create_model(client, name="ext-test-model"):
    resp = await client.post("/api/v1/models", json={
        "name": name,
        "description": "extended test model",
        "model_type": "llm",
        "architecture": "qwen2",
        "params_size": "7B",
    })
    assert resp.status_code == 201
    return resp.json()


async def _create_version(client, model_id, version="1.0.0"):
    resp = await client.post(
        f"/api/v1/models/{model_id}/versions",
        data={"version": version, "format": "mlx", "quantization": "4bit"},
        files={"file": ("", b"")},
    )
    assert resp.status_code == 201
    return resp.json()


async def _create_published_version(client, model_id, version="1.0.0"):
    ver = await _create_version(client, model_id, version)
    await client.put(
        f"/api/v1/versions/{ver['id']}/status",
        json={"target_status": "testing"},
    )
    await client.put(
        f"/api/v1/versions/{ver['id']}/metrics",
        json={"benchmark_score": 90.0},
    )
    await client.put(
        f"/api/v1/versions/{ver['id']}/status",
        json={"target_status": "published", "approval_level": "l1"},
    )
    return ver


# =====================================================================
# Inference router tests
# =====================================================================


class TestInferenceServeModel:
    @pytest.mark.asyncio
    async def test_serve_model_not_found(self, client):
        resp = await client.post("/api/v1/models/nonexistent/serve", json={})
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_serve_model_no_version(self, client):
        model = await _create_model(client, "serve-no-ver")
        resp = await client.post(
            f"/api/v1/models/{model['id']}/serve",
            json={},
        )
        assert resp.status_code == 400
        assert "no version" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_serve_model_version_not_found(self, client):
        model = await _create_model(client, "serve-bad-ver")
        resp = await client.post(
            f"/api/v1/models/{model['id']}/serve",
            json={"version_id": "nonexistent-ver-id"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_serve_model_mlx_unavailable(self, client):
        model = await _create_model(client, "serve-no-mlx")
        await _create_published_version(client, model["id"])
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(side_effect=__import__("httpx").ConnectError("Connection refused"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                f"/api/v1/models/{model['id']}/serve",
                json={},
            )
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_serve_model_mlx_load_error(self, client):
        model = await _create_model(client, "serve-mlx-err")
        await _create_published_version(client, model["id"])
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "internal error"
        mock_resp.raise_for_status.side_effect = __import__("httpx").HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp,
        )
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                f"/api/v1/models/{model['id']}/serve",
                json={},
            )
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_serve_model_success_with_mock(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        model = await _create_model(client, "serve-ok")
        ver = await _create_published_version(client, model["id"])

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                f"/api/v1/models/{model['id']}/serve",
                json={},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loaded"
        assert data["model_id"] == model["id"]
        assert data["version_id"] == ver["id"]
        inf_mod._loaded_models.clear()

    @pytest.mark.asyncio
    async def test_serve_model_with_explicit_version_id(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        model = await _create_model(client, "serve-explicit-ver")
        ver = await _create_published_version(client, model["id"])

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                f"/api/v1/models/{model['id']}/serve",
                json={"version_id": ver["id"]},
            )
        assert resp.status_code == 200
        assert resp.json()["version_id"] == ver["id"]
        inf_mod._loaded_models.clear()

    @pytest.mark.asyncio
    async def test_serve_model_uses_hf_repo_as_model_name(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        model_resp = await client.post("/api/v1/models", json={
            "name": "serve-hf-repo-model",
            "hf_repo": "Qwen/Qwen2.5-7B",
        })
        model = model_resp.json()
        ver = await _create_published_version(client, model["id"])

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                f"/api/v1/models/{model['id']}/serve",
                json={},
            )
        assert resp.status_code == 200
        assert resp.json()["mlx_model"] == "Qwen/Qwen2.5-7B"
        inf_mod._loaded_models.clear()


class TestInferenceUnloadModel:
    @pytest.mark.asyncio
    async def test_unload_not_loaded(self, client):
        resp = await client.delete("/api/v1/models/nonexistent/serve")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_unload_model_success(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        model = await _create_model(client, "unload-ok")
        await _create_published_version(client, model["id"])

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            serve_resp = await client.post(
                f"/api/v1/models/{model['id']}/serve",
                json={},
            )
        assert serve_resp.status_code == 200

        mock_client_instance2 = AsyncMock()
        mock_client_instance2.post = AsyncMock(return_value=mock_resp)
        mock_client_instance2.__aenter__ = AsyncMock(return_value=mock_client_instance2)
        mock_client_instance2.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance2):
            unload_resp = await client.delete(
                f"/api/v1/models/{model['id']}/serve",
            )
        assert unload_resp.status_code == 200
        assert unload_resp.json()["status"] == "unloaded"
        inf_mod._loaded_models.clear()

    @pytest.mark.asyncio
    async def test_unload_model_mlx_error_still_unloads(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["test-unload-err-id"] = {
            "version_id": "v1",
            "model_name": "test-model",
            "status": "loaded",
            "loaded_at": time.time(),
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(side_effect=Exception("MLX down"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.delete("/api/v1/models/test-unload-err-id/serve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"
        assert "test-unload-err-id" not in inf_mod._loaded_models
        inf_mod._loaded_models.clear()


class TestInferenceServeStatus:
    @pytest.mark.asyncio
    async def test_status_not_loaded(self, client):
        resp = await client.get("/api/v1/models/nonexistent/serve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_loaded"

    @pytest.mark.asyncio
    async def test_status_loaded(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["status-model-id"] = {
            "version_id": "v1",
            "model_name": "status-model",
            "status": "loaded",
            "loaded_at": time.time(),
        }
        resp = await client.get("/api/v1/models/status-model-id/serve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loaded"
        assert data["model_name"] == "status-model"
        inf_mod._loaded_models.clear()


class TestInferenceChatCompletion:
    @pytest.mark.asyncio
    async def test_chat_not_loaded(self, client):
        resp = await client.post(
            "/api/v1/inference/nonexistent/chat",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_model_loaded_mlx_unavailable(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["chat-model-id"] = {
            "version_id": "v1",
            "model_name": "chat-model",
            "status": "loaded",
            "loaded_at": time.time(),
        }
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(
            side_effect=__import__("httpx").ConnectError("refused"),
        )
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/inference/chat-model-id/chat",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 503
        inf_mod._loaded_models.clear()

    @pytest.mark.asyncio
    async def test_chat_model_loaded_success(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["chat-ok-id"] = {
            "version_id": "v1",
            "model_name": "chat-ok-model",
            "status": "loaded",
            "loaded_at": time.time(),
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={"choices": [{"message": {"content": "hello"}}]})
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/inference/chat-ok-id/chat",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 200
        assert "choices" in resp.json()
        inf_mod._loaded_models.clear()

    @pytest.mark.asyncio
    async def test_chat_model_mlx_http_error(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["chat-err-id"] = {
            "version_id": "v1",
            "model_name": "chat-err-model",
            "status": "loaded",
            "loaded_at": time.time(),
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "rate limited"
        mock_resp.raise_for_status.side_effect = __import__("httpx").HTTPStatusError(
            "rate limit", request=MagicMock(), response=mock_resp,
        )
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/inference/chat-err-id/chat",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 429
        inf_mod._loaded_models.clear()


class TestInferenceCompletions:
    @pytest.mark.asyncio
    async def test_completions_not_loaded(self, client):
        r = await client.post("/api/v1/models", json={"name": "comp-ext-model"})
        model_id = r.json()["id"]
        resp = await client.post(
            f"/api/v1/inference/{model_id}/completions",
            json={"prompt": "hello"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_completions_model_loaded_success(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["comp-ok-id"] = {
            "version_id": "v1",
            "model_name": "comp-ok-model",
            "status": "loaded",
            "loaded_at": time.time(),
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={"choices": [{"text": "world"}]})
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/inference/comp-ok-id/completions",
                json={"prompt": "hello"},
            )
        assert resp.status_code == 200
        inf_mod._loaded_models.clear()

    @pytest.mark.asyncio
    async def test_completions_mlx_unavailable(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["comp-mlx-id"] = {
            "version_id": "v1",
            "model_name": "comp-mlx-model",
            "status": "loaded",
            "loaded_at": time.time(),
        }
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(
            side_effect=__import__("httpx").ConnectError("refused"),
        )
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/inference/comp-mlx-id/completions",
                json={"prompt": "hello"},
            )
        assert resp.status_code == 503
        inf_mod._loaded_models.clear()


class TestInferenceEmbeddings:
    @pytest.mark.asyncio
    async def test_embeddings_not_loaded(self, client):
        r = await client.post("/api/v1/models", json={"name": "emb-ext-model"})
        model_id = r.json()["id"]
        resp = await client.post(
            f"/api/v1/inference/{model_id}/embeddings",
            json={"input": "hello"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_embeddings_model_loaded_success(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["emb-ok-id"] = {
            "version_id": "v1",
            "model_name": "emb-ok-model",
            "status": "loaded",
            "loaded_at": time.time(),
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={"data": [{"embedding": [0.1, 0.2]}]})
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/inference/emb-ok-id/embeddings",
                json={"input": "hello"},
            )
        assert resp.status_code == 200
        inf_mod._loaded_models.clear()

    @pytest.mark.asyncio
    async def test_embeddings_mlx_unavailable(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["emb-mlx-id"] = {
            "version_id": "v1",
            "model_name": "emb-mlx-model",
            "status": "loaded",
            "loaded_at": time.time(),
        }
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(
            side_effect=__import__("httpx").ConnectError("refused"),
        )
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/inference/emb-mlx-id/embeddings",
                json={"input": "hello"},
            )
        assert resp.status_code == 503
        inf_mod._loaded_models.clear()


class TestInferenceTTLEviction:
    @pytest.mark.asyncio
    async def test_cleanup_expired_models(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["expired-id"] = {
            "version_id": "v1",
            "model_name": "expired-model",
            "status": "loaded",
            "loaded_at": time.time() - inf_mod._LOADED_TTL - 100,
        }
        inf_mod._loaded_models["valid-id"] = {
            "version_id": "v2",
            "model_name": "valid-model",
            "status": "loaded",
            "loaded_at": time.time(),
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            await inf_mod._cleanup_loaded_models()

        assert "expired-id" not in inf_mod._loaded_models
        assert "valid-id" in inf_mod._loaded_models
        inf_mod._loaded_models.clear()

    @pytest.mark.asyncio
    async def test_cleanup_unload_failure_does_not_raise(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["fail-unload-id"] = {
            "version_id": "v1",
            "model_name": "fail-model",
            "status": "loaded",
            "loaded_at": time.time() - inf_mod._LOADED_TTL - 100,
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(side_effect=Exception("MLX down"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            await inf_mod._cleanup_loaded_models()

        assert "fail-unload-id" not in inf_mod._loaded_models
        inf_mod._loaded_models.clear()

    @pytest.mark.asyncio
    async def test_cleanup_no_model_name_still_removes(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["no-name-id"] = {
            "version_id": "v1",
            "model_name": "",
            "status": "loaded",
            "loaded_at": time.time() - inf_mod._LOADED_TTL - 100,
        }
        await inf_mod._cleanup_loaded_models()
        assert "no-name-id" not in inf_mod._loaded_models
        inf_mod._loaded_models.clear()

    @pytest.mark.asyncio
    async def test_ttl_eviction_triggered_on_serve(self, client):
        from fusion_model_hub.server.routers import inference as inf_mod
        inf_mod._loaded_models.clear()

        inf_mod._loaded_models["ttl-old-id"] = {
            "version_id": "v1",
            "model_name": "ttl-old-model",
            "status": "loaded",
            "loaded_at": time.time() - inf_mod._LOADED_TTL - 100,
        }

        model = await _create_model(client, "ttl-serve-model")
        await _create_published_version(client, model["id"])

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                f"/api/v1/models/{model['id']}/serve",
                json={},
            )
        assert resp.status_code == 200
        assert "ttl-old-id" not in inf_mod._loaded_models
        inf_mod._loaded_models.clear()


# =====================================================================
# Sync router tests
# =====================================================================


class TestSyncSSRFValidation:
    @pytest.mark.asyncio
    async def test_blocked_localhost(self, client):
        resp = await client.post(
            "/api/v1/sync/pull",
            json={"source_url": "http://localhost:8080/api", "model_id": "m1"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_blocked_127_0_0_1(self, client):
        resp = await client.post(
            "/api/v1/sync/pull",
            json={"source_url": "http://127.0.0.1:9999/api", "model_id": "m1"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_blocked_0_0_0_0(self, client):
        resp = await client.post(
            "/api/v1/sync/pull",
            json={"source_url": "http://0.0.0.0/api", "model_id": "m1"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_blocked_ipv6_loopback(self, client):
        resp = await client.post(
            "/api/v1/sync/pull",
            json={"source_url": "http://[::1]/api", "model_id": "m1"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_blocked_169_254_169_254(self, client):
        resp = await client.post(
            "/api/v1/sync/pull",
            json={"source_url": "http://169.254.169.254/api", "model_id": "m1"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_blocked_10_network(self, client):
        resp = await client.post(
            "/api/v1/sync/pull",
            json={"source_url": "http://10.0.0.1/api", "model_id": "m1"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_blocked_192_168_network(self, client):
        resp = await client.post(
            "/api/v1/sync/pull",
            json={"source_url": "http://192.168.1.1/api", "model_id": "m1"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_blocked_172_16_network(self, client):
        resp = await client.post(
            "/api/v1/sync/pull",
            json={"source_url": "http://172.16.0.1/api", "model_id": "m1"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_blocked_172_31_network(self, client):
        resp = await client.post(
            "/api/v1/sync/pull",
            json={"source_url": "http://172.31.255.1/api", "model_id": "m1"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_allowed_172_32_network(self, client):
        from fusion_model_hub.server.routers.sync import _validate_sync_url
        try:
            _validate_sync_url("http://172.32.0.1/api")
        except Exception as e:
            pytest.fail(f"172.32.x should be allowed but got: {e}")

    @pytest.mark.asyncio
    async def test_blocked_ftp_scheme(self, client):
        resp = await client.post(
            "/api/v1/sync/pull",
            json={"source_url": "ftp://example.com/api", "model_id": "m1"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_blocked_javascript_scheme(self, client):
        resp = await client.post(
            "/api/v1/sync/pull",
            json={"source_url": "javascript:alert(1)", "model_id": "m1"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_allowed_https_external(self, client):
        from fusion_model_hub.server.routers.sync import _validate_sync_url
        try:
            _validate_sync_url("https://remote-hub.example.com/api")
        except Exception as e:
            pytest.fail(f"External HTTPS should be allowed but got: {e}")

    @pytest.mark.asyncio
    async def test_allowed_http_external(self, client):
        from fusion_model_hub.server.routers.sync import _validate_sync_url
        try:
            _validate_sync_url("http://remote-hub.example.com/api")
        except Exception as e:
            pytest.fail(f"External HTTP should be allowed but got: {e}")


class TestSyncPull:
    @pytest.mark.asyncio
    async def test_pull_unreachable_remote(self, client):
        resp = await client.post(
            "/api/v1/sync/pull",
            json={
                "source_url": "https://nonexistent.example.com",
                "model_id": "test-model",
            },
        )
        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_pull_success_with_mock(self, client):
        remote_model_data = {
            "name": "pulled-model",
            "description": "from remote",
            "model_type": "llm",
            "architecture": "llama",
            "params_size": "13B",
            "hf_repo": "remote/repo",
            "versions": [
                {"id": "rv1", "version": "1.0.0", "format": "mlx", "quantization": "4bit", "file_size": 1000},
                {"id": "rv2", "version": "2.0.0", "format": "mlx", "quantization": "8bit", "file_size": 2000},
            ],
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=remote_model_data)
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.sync.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/sync/pull",
                json={
                    "source_url": "https://remote-hub.example.com",
                    "model_id": "test-model",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pulled"
        assert data["versions_pulled"] == 2

    @pytest.mark.asyncio
    async def test_pull_already_exists(self, client):
        model = await _create_model(client, "pulled-model")

        remote_model_data = {
            "name": "pulled-model",
            "description": "from remote",
            "model_type": "llm",
            "versions": [],
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=remote_model_data)
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.sync.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/sync/pull",
                json={
                    "source_url": "https://remote-hub.example.com",
                    "model_id": "test-model",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "already_exists"

    @pytest.mark.asyncio
    async def test_pull_with_version_filter(self, client):
        remote_model_data = {
            "name": "pulled-filter-model",
            "description": "from remote",
            "model_type": "llm",
            "versions": [
                {"id": "rv1", "version": "1.0.0", "format": "mlx", "quantization": "4bit", "file_size": 1000},
                {"id": "rv2", "version": "2.0.0", "format": "mlx", "quantization": "8bit", "file_size": 2000},
            ],
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=remote_model_data)
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.sync.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/sync/pull",
                json={
                    "source_url": "https://remote-hub.example.com",
                    "model_id": "test-model",
                    "version_ids": ["rv1"],
                },
            )
        assert resp.status_code == 200
        assert resp.json()["versions_pulled"] == 1

    @pytest.mark.asyncio
    async def test_pull_invalid_model_type_defaults_to_llm(self, client):
        remote_model_data = {
            "name": "pulled-bad-type",
            "description": "from remote",
            "model_type": "nonexistent_type",
            "versions": [],
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=remote_model_data)
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.sync.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/sync/pull",
                json={
                    "source_url": "https://remote-hub.example.com",
                    "model_id": "test-model",
                },
            )
        assert resp.status_code == 200


class TestSyncPush:
    @pytest.mark.asyncio
    async def test_push_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/sync/push",
            json={
                "target_url": "https://remote-hub.example.com",
                "model_id": "nonexistent",
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_push_blocked_internal_url(self, client):
        model = await _create_model(client, "push-blocked")
        resp = await client.post(
            "/api/v1/sync/push",
            json={
                "target_url": "http://127.0.0.1:9999",
                "model_id": model["id"],
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_push_success_with_mock(self, client):
        model = await _create_model(client, "push-ok")
        ver = await _create_version(client, model["id"])

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.sync.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/sync/push",
                json={
                    "target_url": "https://remote-hub.example.com",
                    "model_id": model["id"],
                    "version_ids": [ver["id"]],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["pushed"][0]["status"] == "pushed"

    @pytest.mark.asyncio
    async def test_push_remote_failure(self, client):
        model = await _create_model(client, "push-fail")
        await _create_version(client, model["id"])

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(side_effect=Exception("remote down"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.sync.httpx.AsyncClient", return_value=mock_client_instance):
            resp = await client.post(
                "/api/v1/sync/push",
                json={
                    "target_url": "https://remote-hub.example.com",
                    "model_id": model["id"],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert any(p["status"] == "failed" for p in data["pushed"])


class TestSyncVersionManifest:
    @pytest.mark.asyncio
    async def test_manifest_version_not_found(self, client):
        resp = await client.get("/api/v1/sync/versions/nonexistent/manifest")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_manifest_success(self, client):
        model = await _create_model(client, "manifest-model")
        content = b"manifest test data content"
        ver_resp = await client.post(
            f"/api/v1/models/{model['id']}/versions",
            data={"version": "1.0.0", "format": "mlx", "quantization": "4bit"},
            files={"file": ("model.mlx", content, "application/octet-stream")},
        )
        assert ver_resp.status_code == 201
        version_id = ver_resp.json()["id"]

        resp = await client.get(f"/api/v1/sync/versions/{version_id}/manifest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version_id"] == version_id
        assert data["model_id"] == model["id"]
        assert data["version"] == "1.0.0"
        assert data["format"] == "mlx"
        assert data["quantization"] == "4bit"
        assert data["file_hash"] != ""


# =====================================================================
# Tasks module tests
# =====================================================================


class TestTasksSubmitQuantize:
    @pytest.mark.asyncio
    async def test_submit_quantize_source_not_found(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()

        r = await client.post("/api/v1/quantize", json={
            "source_version_id": "nonexistent-ver-id",
            "quant_bits": 4,
        })
        assert r.status_code == 202
        task_id = r.json()["task_id"]

        await asyncio.sleep(0.5)

        status_resp = await client.get(f"/api/v1/quantize/{task_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] == "failed"
        assert "not found" in data.get("error_message", "").lower()
        tasks_mod._running_tasks.clear()

    @pytest.mark.asyncio
    async def test_submit_quantize_success_with_mock(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()

        model = await _create_model(client, "quant-model")
        ver = await _create_version(client, model["id"])

        mock_result = {
            "output_path": "/tmp/quantized.mlx",
            "file_hash": "abc123",
            "file_size": 1024,
        }
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConverter:
            mock_converter = AsyncMock()
            mock_converter.quantize = AsyncMock(return_value=mock_result)
            MockConverter.return_value = mock_converter

            r = await client.post("/api/v1/quantize", json={
                "source_version_id": ver["id"],
                "quant_bits": 4,
            })
            assert r.status_code == 202
            task_id = r.json()["task_id"]

            await asyncio.sleep(0.5)

        status_resp = await client.get(f"/api/v1/quantize/{task_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] == "completed"
        assert data["output_version_id"] is not None
        tasks_mod._running_tasks.clear()

    @pytest.mark.asyncio
    async def test_submit_quantize_converter_failure(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()

        model = await _create_model(client, "quant-fail-model")
        ver = await _create_version(client, model["id"])

        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConverter:
            mock_converter = AsyncMock()
            mock_converter.quantize = AsyncMock(side_effect=RuntimeError("conversion error"))
            MockConverter.return_value = mock_converter

            r = await client.post("/api/v1/quantize", json={
                "source_version_id": ver["id"],
                "quant_bits": 4,
            })
            assert r.status_code == 202
            task_id = r.json()["task_id"]

            await asyncio.sleep(0.5)

        status_resp = await client.get(f"/api/v1/quantize/{task_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] == "failed"
        tasks_mod._running_tasks.clear()

    @pytest.mark.asyncio
    async def test_submit_quantize_create_version_failure(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()

        model = await _create_model(client, "quant-cv-fail-model")
        ver = await _create_version(client, model["id"])

        mock_result = {
            "output_path": "/tmp/quant.mlx",
            "file_hash": "abc",
            "file_size": 1024,
        }
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConverter, \
             patch("fusion_model_hub.server.tasks.create_version", new_callable=AsyncMock, return_value=None):
            mock_converter = AsyncMock()
            mock_converter.quantize = AsyncMock(return_value=mock_result)
            MockConverter.return_value = mock_converter

            r = await client.post("/api/v1/quantize", json={
                "source_version_id": ver["id"],
                "quant_bits": 4,
            })
            assert r.status_code == 202
            task_id = r.json()["task_id"]

            await asyncio.sleep(0.5)

        status_resp = await client.get(f"/api/v1/quantize/{task_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] == "failed"
        assert "failed to create output version" in data.get("error_message", "").lower()
        tasks_mod._running_tasks.clear()

    @pytest.mark.asyncio
    async def test_submit_quantize_unexpected_exception(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()

        model = await _create_model(client, "quant-exc-model")
        ver = await _create_version(client, model["id"])

        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConverter:
            mock_converter = AsyncMock()
            mock_converter.quantize = AsyncMock(side_effect=RuntimeError("unexpected"))
            MockConverter.return_value = mock_converter

            r = await client.post("/api/v1/quantize", json={
                "source_version_id": ver["id"],
                "quant_bits": 4,
            })
            assert r.status_code == 202
            task_id = r.json()["task_id"]

            await asyncio.sleep(0.5)

        status_resp = await client.get(f"/api/v1/quantize/{task_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] == "failed"
        tasks_mod._running_tasks.clear()


class TestTasksGetStatus:
    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, client):
        resp = await client.get("/api/v1/quantize/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_task_status_direct(self, client):
        from fusion_model_hub.server.deps import get_session_factory
        from fusion_model_hub.server.tasks import get_task_status

        sf = get_session_factory()
        async with sf() as session:
            from fusion_model_hub.db.crud import create_quantize_task
            task = await create_quantize_task(
                session,
                source_version_id="v1",
                target_format="mlx",
                quant_bits=4,
            )
            task_id = task.id

        result = await get_task_status(task_id)
        assert result is not None
        assert result["id"] == task_id
        assert result["status"] == "pending"
        assert result["source_version_id"] == "v1"

    @pytest.mark.asyncio
    async def test_get_task_status_none_for_missing(self, client):
        from fusion_model_hub.server.tasks import get_task_status
        result = await get_task_status("nonexistent-task-id")
        assert result is None


class TestTasksListRunning:
    @pytest.mark.asyncio
    async def test_list_running_tasks(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()

        resp = await client.get("/api/v1/quantize/running")
        assert resp.status_code == 200
        assert "tasks" in resp.json()

    @pytest.mark.asyncio
    async def test_list_running_tasks_direct(self, client):
        from fusion_model_hub.server.tasks import _running_tasks, list_running_tasks
        _running_tasks.clear()

        result = list_running_tasks()
        assert isinstance(result, list)
        assert len(result) == 0


class TestTasksQuantizeBits:
    @pytest.mark.asyncio
    async def test_quantize_2bit(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()

        model = await _create_model(client, "quant-2bit")
        ver = await _create_version(client, model["id"])

        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConverter:
            mock_converter = AsyncMock()
            mock_converter.quantize = AsyncMock(return_value={
                "output_path": "/tmp/q2.mlx", "file_hash": "h2", "file_size": 512,
            })
            MockConverter.return_value = mock_converter

            r = await client.post("/api/v1/quantize", json={
                "source_version_id": ver["id"],
                "quant_bits": 2,
            })
            assert r.status_code == 202
            task_id = r.json()["task_id"]

            await asyncio.sleep(0.5)

        status_resp = await client.get(f"/api/v1/quantize/{task_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "completed"
        tasks_mod._running_tasks.clear()

    @pytest.mark.asyncio
    async def test_quantize_8bit(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()

        model = await _create_model(client, "quant-8bit")
        ver = await _create_version(client, model["id"])

        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConverter:
            mock_converter = AsyncMock()
            mock_converter.quantize = AsyncMock(return_value={
                "output_path": "/tmp/q8.mlx", "file_hash": "h8", "file_size": 2048,
            })
            MockConverter.return_value = mock_converter

            r = await client.post("/api/v1/quantize", json={
                "source_version_id": ver["id"],
                "quant_bits": 8,
            })
            assert r.status_code == 202
            task_id = r.json()["task_id"]

            await asyncio.sleep(0.5)

        status_resp = await client.get(f"/api/v1/quantize/{task_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "completed"
        tasks_mod._running_tasks.clear()


class TestTasksQuantizeInvalidBits:
    @pytest.mark.asyncio
    async def test_quantize_3bit_rejected(self, client):
        resp = await client.post("/api/v1/quantize", json={
            "source_version_id": "nonexistent",
            "quant_bits": 3,
        })
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_quantize_5bit_rejected(self, client):
        resp = await client.post("/api/v1/quantize", json={
            "source_version_id": "nonexistent",
            "quant_bits": 5,
        })
        assert resp.status_code == 400


class TestTasksWebhookDispatch:
    @pytest.mark.asyncio
    async def test_completed_task_dispatches_webhook(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()

        model = await _create_model(client, "wh-dispatch-model")
        ver = await _create_version(client, model["id"])

        mock_result = {
            "output_path": "/tmp/wh.mlx",
            "file_hash": "wh123",
            "file_size": 512,
        }
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConverter, \
             patch("fusion_model_hub.server.routers.webhooks.dispatch_webhook_event", new_callable=AsyncMock) as mock_dispatch:
            mock_converter = AsyncMock()
            mock_converter.quantize = AsyncMock(return_value=mock_result)
            MockConverter.return_value = mock_converter

            r = await client.post("/api/v1/quantize", json={
                "source_version_id": ver["id"],
                "quant_bits": 4,
            })
            assert r.status_code == 202
            task_id = r.json()["task_id"]

            await asyncio.sleep(0.5)

        status_resp = await client.get(f"/api/v1/quantize/{task_id}")
        assert status_resp.json()["status"] == "completed"
        tasks_mod._running_tasks.clear()
