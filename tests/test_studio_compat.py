import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps


@pytest.fixture
def settings():
    return Settings(
        host="127.0.0.1", port=11444,
        data_dir="/tmp/fmh_studio_compat",
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


async def _make_model(client, name="test-model"):
    resp = await client.post("/api/v1/models", json={
        "name": name, "model_type": "llm", "hf_repo": "test/" + name,
    })
    return resp.json()


# fusion-studio DTOs decode with a plain JSONDecoder + default/explicit CodingKeys.
# These tests pin the exact JSON key names the Hub must emit so studio Codable
# structs (HubHardwareResponse, HubHealthResponse, HubDeployment, list envelopes)
# do not silently decode to all-nil or fail.


class TestHardwareFlatShape:
    @pytest.mark.asyncio
    async def test_hardware_has_studio_flat_keys(self, client):
        resp = await client.get("/api/v1/hardware")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("chip", "cpuCores", "gpuCores", "memoryGB", "diskFree", "metalSupport", "aneSupport"):
            assert key in data, f"missing studio key: {key}"


class TestHealthShape:
    @pytest.mark.asyncio
    async def test_health_has_studio_keys(self, client):
        resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("status", "version", "uptime", "mlxConnected", "storage"):
            assert key in data, f"missing studio key: {key}"
        assert isinstance(data["mlxConnected"], bool)
        disk = data["storage"]
        for key in ("used", "total", "modelsPath", "modelsSize"):
            assert key in disk, f"missing studio disk key: {key}"


class TestListEnvelopes:
    @pytest.mark.asyncio
    async def test_tenants_envelope(self, client):
        resp = await client.get("/api/v1/tenants")
        assert resp.status_code == 200
        data = resp.json()
        assert "tenants" in data and "total" in data

    @pytest.mark.asyncio
    async def test_webhooks_envelope(self, client):
        resp = await client.get("/api/v1/webhooks")
        assert resp.status_code == 200
        data = resp.json()
        assert "webhooks" in data and "total" in data


class TestDeploymentStudioContract:
    @pytest.mark.asyncio
    async def test_create_without_name_uses_scale_alias(self, client):
        model = await _make_model(client, "dep-studio")
        # studio createDeployment sends {model_id, scale, canary_percent} with NO name
        resp = await client.post("/api/v1/deployments", json={
            "model_id": model["id"], "scale": 3, "canary_percent": 20,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["replicas"] == 3
        assert data["name"]
        # studio camelCase mirror keys present
        for key in ("modelId", "modelName", "scale", "canaryPercent", "strategy", "createdAt", "updatedAt"):
            assert key in data, f"missing studio deployment key: {key}"

    @pytest.mark.asyncio
    async def test_scale_endpoint_accepts_scale_alias(self, client):
        model = await _make_model(client, "dep-scale")
        create = await client.post("/api/v1/deployments", json={"model_id": model["id"], "name": "sc"})
        did = create.json()["id"]
        # studio scaleDeployment sends {scale: N} with NO replicas
        resp = await client.post(f"/api/v1/deployments/{did}/scale", json={"scale": 5})
        assert resp.status_code == 200
        assert resp.json()["replicas"] == 5
        assert resp.json()["scale"] == 5

    @pytest.mark.asyncio
    async def test_list_envelope_items_have_camel_keys(self, client):
        model = await _make_model(client, "dep-list")
        await client.post("/api/v1/deployments", json={"model_id": model["id"], "name": "l1"})
        resp = await client.get("/api/v1/deployments")
        assert resp.status_code == 200
        data = resp.json()
        assert "deployments" in data and "total" in data
        assert data["total"] >= 1
        item = data["deployments"][0]
        for key in ("modelId", "modelName", "scale", "canaryPercent"):
            assert key in item, f"missing studio key in list item: {key}"


class TestDeploymentMetricsStudioContract:
    @pytest.mark.asyncio
    async def test_metrics_has_studio_camel_keys(self, client):
        # studio getDeploymentMetrics -> HubDeploymentMetricsResponse
        # {deploymentId, requestsPerSecond, avgLatencyMs, errorRate,
        #  tokensPerSecond, activeConnections} — all optional.
        model = await _make_model(client, "dep-metrics")
        create = await client.post("/api/v1/deployments", json={"model_id": model["id"], "name": "mt"})
        did = create.json()["id"]
        resp = await client.get(f"/api/v1/deployments/{did}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("deploymentId", "requestsPerSecond", "avgLatencyMs",
                    "errorRate", "tokensPerSecond", "activeConnections"):
            assert key in data, f"missing studio metrics key: {key}"
        assert data["deploymentId"] == did

    @pytest.mark.asyncio
    async def test_metrics_fills_live_fields_from_mlx_metrics_json(self, client):
        # RUNNING deployment + MLX /v1/metrics/json 200 (ServerMetrics.to_dict)
        # -> requestsPerSecond/errorRate/activeConnections/tokensPerSecond filled
        # from real counters (total_requests/uptime_seconds, failed/total,
        # active_requests, avg_generation_tps). /v1/models/status gives load
        # state into mlx_metrics. Pins the PR #541 consumer wiring.
        from unittest.mock import AsyncMock, MagicMock, patch
        model = await _make_model(client, "dep-live-metrics")
        # MLX /load POST returns 200 so the deployment becomes RUNNING.
        load_resp = MagicMock()
        load_resp.status_code = 200
        status_resp = MagicMock()
        status_resp.status_code = 200
        status_resp.json.return_value = {"test/dep-live-metrics": {"state": "loaded"}}
        metrics_resp = MagicMock()
        metrics_resp.status_code = 200
        metrics_resp.json.return_value = {
            "total_requests": 100,
            "successful_requests": 95,
            "failed_requests": 5,
            "active_requests": 3,
            "avg_generation_tps": 42.5,
            "uptime_seconds": 50,
        }
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.post = AsyncMock(return_value=load_resp)
        mock_ctx.get = AsyncMock(side_effect=[status_resp, metrics_resp])
        with patch("httpx.AsyncClient", return_value=mock_ctx):
            create = await client.post(
                "/api/v1/deployments", json={"model_id": model["id"], "name": "live"},
            )
            did = create.json()["id"]
            resp = await client.get(f"/api/v1/deployments/{did}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["requestsPerSecond"] == 2.0  # 100 / 50
        assert data["errorRate"] == 0.05  # 5 / 100
        assert data["activeConnections"] == 3
        assert data["tokensPerSecond"] == 42.5
        assert data["mlx_metrics"] == {"state": "loaded"}

    @pytest.mark.asyncio
    async def test_metrics_null_live_fields_when_endpoint_404(self, client):
        # Pre PR #541 merge: MLX /v1/metrics/json 404 -> live 4 fields null,
        # shape stable. /v1/models/status still 200 so mlx_metrics populated.
        from unittest.mock import AsyncMock, MagicMock, patch
        model = await _make_model(client, "dep-404-metrics")
        load_resp = MagicMock()
        load_resp.status_code = 200
        status_resp = MagicMock()
        status_resp.status_code = 200
        status_resp.json.return_value = {"test/dep-404-metrics": {"state": "loaded"}}
        not_found_resp = MagicMock()
        not_found_resp.status_code = 404
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.post = AsyncMock(return_value=load_resp)
        mock_ctx.get = AsyncMock(side_effect=[status_resp, not_found_resp])
        with patch("httpx.AsyncClient", return_value=mock_ctx):
            create = await client.post(
                "/api/v1/deployments", json={"model_id": model["id"], "name": "nf"},
            )
            did = create.json()["id"]
            resp = await client.get(f"/api/v1/deployments/{did}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["requestsPerSecond"] is None
        assert data["errorRate"] is None
        assert data["activeConnections"] is None
        assert data["tokensPerSecond"] is None
        assert data["mlx_metrics"] == {"state": "loaded"}


class TestBenchTriggerStudioContract:
    @pytest.mark.asyncio
    async def test_template_alias_mapped_to_suite(self, client):
        # studio triggerBenchmark posts {model_id, template: "general"}.
        # Hub must forward `suite` (Fusion-Bench's field), mapping template -> suite.
        from unittest.mock import AsyncMock, MagicMock, patch
        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_resp.json.return_value = {"task_id": "bt-template"}
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.post = AsyncMock(return_value=mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_ctx):
            resp = await client.post("/api/v1/benchmarks/trigger", json={
                "model_id": "t-model", "template": "speed",
            })
        assert resp.status_code == 200
        forwarded = mock_ctx.post.call_args.kwargs["json"]
        assert forwarded["suite"] == "speed"
        assert forwarded["model_id"] == "t-model"
