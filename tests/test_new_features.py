import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps
from fusion_model_hub.server.rate_limit import check_rate_limit, reset_rate_limits


@pytest.fixture
def settings():
    return Settings(
        host="127.0.0.1", port=11444,
        data_dir="/tmp/fmh_test_data",
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


def _mock_httpx_client(response_status=200, response_json=None, side_effect=None):
    mock_resp = MagicMock()
    mock_resp.status_code = response_status
    mock_resp.json.return_value = response_json or {}
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
    return patch("httpx.AsyncClient", return_value=mock_ctx)


class TestRateLimiter:
    def setup_method(self):
        reset_rate_limits()

    def test_allow_when_no_limit(self):
        assert check_rate_limit("test", 0) is True

    def test_allow_under_limit(self):
        assert check_rate_limit("test", 5) is True

    def test_block_over_limit(self):
        for _ in range(3):
            check_rate_limit("test", 3)
        assert check_rate_limit("test", 3) is False

    def test_separate_keys_independent(self):
        for _ in range(3):
            check_rate_limit("key_a", 3)
        assert check_rate_limit("key_a", 3) is False
        assert check_rate_limit("key_b", 3) is True

    def test_window_expiry(self):
        for _ in range(3):
            check_rate_limit("test", 3)
        assert check_rate_limit("test", 3) is False
        with patch("time.time", return_value=time.time() + 61):
            assert check_rate_limit("test", 3) is True

    def test_reset(self):
        for _ in range(3):
            check_rate_limit("test", 3)
        reset_rate_limits()
        assert check_rate_limit("test", 3) is True


class TestModelModules:
    @pytest.mark.asyncio
    async def test_create_model_with_modules(self, client):
        resp = await client.post("/api/v1/models", json={
            "name": "mod-model-1",
            "model_modules": "chat,code",
            "idle_timeout_minutes": 30,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["model_modules"] == "chat,code"
        assert data["idle_timeout_minutes"] == 30

    @pytest.mark.asyncio
    async def test_update_model_modules(self, client):
        create = await client.post("/api/v1/models", json={"name": "mod-model-2"})
        mid = create.json()["id"]
        resp = await client.put(
            f"/api/v1/models/{mid}/modules",
            json={"modules": "chat,rag,agent"},
        )
        assert resp.status_code == 200
        assert resp.json()["model_modules"] == "chat,rag,agent"

    @pytest.mark.asyncio
    async def test_update_modules_not_found(self, client):
        resp = await client.put(
            "/api/v1/models/nonexistent/modules",
            json={"modules": "chat"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_model_idle_timeout(self, client):
        create = await client.post("/api/v1/models", json={"name": "mod-model-3"})
        mid = create.json()["id"]
        resp = await client.put(
            f"/api/v1/models/{mid}",
            json={"idle_timeout_minutes": 120},
        )
        assert resp.status_code == 200
        assert resp.json()["idle_timeout_minutes"] == 120


class TestApiKeyQps:
    @pytest.mark.asyncio
    async def test_create_key_with_qps_limit(self, client):
        resp = await client.post("/api/v1/auth/keys", json={
            "name": "qps-key",
            "qps_limit": 10,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["qps_limit"] == 10

    @pytest.mark.asyncio
    async def test_create_key_default_qps(self, client):
        resp = await client.post("/api/v1/auth/keys", json={"name": "no-qps-key"})
        assert resp.status_code == 201
        assert resp.json()["qps_limit"] == 0

    @pytest.mark.asyncio
    async def test_list_keys_includes_qps(self, client):
        await client.post("/api/v1/auth/keys", json={"name": "list-qps", "qps_limit": 5})
        resp = await client.get("/api/v1/auth/keys")
        items = resp.json()["items"]
        qps_key = next(i for i in items if i["name"] == "list-qps")
        assert qps_key["qps_limit"] == 5


class TestMarketSearch:
    @pytest.mark.asyncio
    async def test_market_search_local_only(self, client):
        await client.post("/api/v1/models", json={"name": "local-search-model"})
        resp = await client.get("/api/v1/models/market/search", params={
            "keyword": "local-search",
            "source": "local",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "local" in data
        assert len(data["local"]) >= 1

    @pytest.mark.asyncio
    async def test_market_search_all_sources(self, client):
        await client.post("/api/v1/models", json={"name": "all-search-model"})
        with _mock_httpx_client(response_json=[{"id": "Qwen/Qwen2.5-7B", "pipeline_tag": "text-generation", "downloads": 1000}]):
            with patch("fusion_model_hub.repo.modelscope_search.search_modelscope", new=AsyncMock(return_value={
                "items": [{"name": "qwen-ms", "id": "ms-1"}],
                "total": 1,
                "source": "modelscope",
            })):
                resp = await client.get("/api/v1/models/market/search", params={
                    "keyword": "qwen",
                    "source": "all",
                })
                assert resp.status_code == 200
                data = resp.json()
                assert "local" in data
                assert "huggingface" in data
                assert "modelscope" in data

    @pytest.mark.asyncio
    async def test_market_search_modelscope_mock(self, client):
        with patch("fusion_model_hub.repo.modelscope_search.search_modelscope", new=AsyncMock(return_value={
            "items": [{"name": "qwen-test", "id": "ms-1"}],
            "total": 1,
            "source": "modelscope",
        })) as mock_ms:
            resp = await client.get("/api/v1/models/market/search", params={
                "keyword": "qwen",
                "source": "modelscope",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["modelscope"]) >= 1
            mock_ms.assert_called_once()

    @pytest.mark.asyncio
    async def test_market_search_huggingface_mock(self, client):
        with _mock_httpx_client(response_json=[{"id": "Qwen/Qwen2.5-7B", "pipeline_tag": "text-generation", "downloads": 1000}]):
            resp = await client.get("/api/v1/models/market/search", params={
                "keyword": "qwen",
                "source": "huggingface",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["huggingface"]) >= 1


class TestBenchTrigger:
    @pytest.mark.asyncio
    async def test_bench_trigger_bench_unavailable(self, client):
        with _mock_httpx_client(side_effect=httpx.ConnectError("refused")):
            resp = await client.post("/api/v1/benchmarks/trigger", json={
                "model_id": "any-model",
                "suite": "general",
            })
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_bench_trigger_success_mock(self, client):
        with _mock_httpx_client(response_status=202, response_json={"task_id": "bt-1"}):
            resp = await client.post("/api/v1/benchmarks/trigger", json={
                "model_id": "test-model",
                "suite": "general",
            })
            assert resp.status_code == 200
            assert resp.json()["status"] == "submitted"


class TestClusterSyncModel:
    @pytest.mark.asyncio
    async def test_sync_model_not_found(self, client):
        resp = await client.post("/api/v1/cluster/sync-model", json={
            "model_id": "nonexistent",
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_sync_model_no_nodes(self, client):
        model = await client.post("/api/v1/models", json={"name": "sync-no-node"})
        mid = model.json()["id"]
        resp = await client.post("/api/v1/cluster/sync-model", json={
            "model_id": mid,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == mid
        assert "success" in data
        assert "message" in data

    @pytest.mark.asyncio
    async def test_sync_model_to_nodes_mock(self, client):
        model = await client.post("/api/v1/models", json={"name": "sync-mock"})
        mid = model.json()["id"]
        await client.post("/api/v1/cluster/nodes", json={
            "name": "sync-node", "url": "http://sync-node:11444",
        })
        with _mock_httpx_client(response_status=200, response_json={"ok": True}):
            resp = await client.post("/api/v1/cluster/sync-model", json={
                "model_id": mid,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert "remote_ok=1" in data["message"]


class TestClusterRouteInference:
    @pytest.mark.asyncio
    async def test_route_inference_model_not_found(self, client):
        resp = await client.post("/api/v1/cluster/route-inference", json={
            "model_id": "nonexistent",
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_route_inference_local_available(self, client):
        model = await client.post("/api/v1/models", json={"name": "route-local"})
        mid = model.json()["id"]
        chat_resp = {"id": "chat-1", "model": "route-local", "choices": [
            {"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
        with _mock_httpx_client(response_status=200, response_json=chat_resp):
            resp = await client.post("/api/v1/cluster/route-inference", json={
                "model_id": mid,
                "messages": [{"role": "user", "content": "ping"}],
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["routedTo"] == "local"
            assert data["content"] == "hi"

    @pytest.mark.asyncio
    async def test_route_inference_no_nodes_available(self, client):
        model = await client.post("/api/v1/models", json={"name": "route-no-node"})
        mid = model.json()["id"]
        with _mock_httpx_client(side_effect=Exception("fail")):
            resp = await client.post("/api/v1/cluster/route-inference", json={
                "model_id": mid,
                "messages": [{"role": "user", "content": "ping"}],
            })
            assert resp.status_code == 503


class TestModelsourceEnum:
    def test_modelscope_in_enum(self):
        from fusion_model_hub.db.models import ModelSource
        assert hasattr(ModelSource, "MODELSCOPE")
        assert ModelSource.MODELSCOPE.value == "modelscope"


class TestModelscopeSearch:
    @pytest.mark.asyncio
    async def test_search_modelscope_success(self):
        from fusion_model_hub.repo.modelscope_search import search_modelscope
        with _mock_httpx_client(response_json={
            "Data": {
                "Models": [{
                    "Name": "qwen-test",
                    "Id": "ms-1",
                    "Task": "text-generation",
                    "ModelScopeId": "qwen/qwen-test",
                }],
                "TotalCount": 1,
            },
        }):
            result = await search_modelscope("qwen")
            assert result["source"] == "modelscope"
            assert result["total"] == 1
            assert len(result["items"]) == 1

    @pytest.mark.asyncio
    async def test_search_modelscope_error(self):
        from fusion_model_hub.repo.modelscope_search import search_modelscope
        with _mock_httpx_client(side_effect=Exception("fail")):
            result = await search_modelscope("qwen")
            assert result["source"] == "modelscope"
            assert result["total"] == 0
            assert result["items"] == []


class TestAuthRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limit_unit(self):
        reset_rate_limits()
        assert check_rate_limit("rl-test", 2) is True
        assert check_rate_limit("rl-test", 2) is True
        assert check_rate_limit("rl-test", 2) is False
        reset_rate_limits()
