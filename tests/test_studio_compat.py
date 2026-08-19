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
