import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps

GATE = {"gate_mode": "wilson", "candidate_ci": [0.62, 0.79], "claims_row": "CL-030"}


@pytest.fixture
def settings():
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_test_data",
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
        models_dir="/tmp/fmh_test_data/models",
        eval_runner_enabled=False,
    )


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


@pytest.fixture
def app(settings):
    return create_app(settings)


async def _base_model(client):
    r = await client.post("/api/v1/models", json={"name": "Qwen3.8-27B-4bit"})
    assert r.status_code == 201, r.text
    return r.json()


async def _mk_tenant(client, name):
    # TenantCreate has no id field — the router mints one; return it.
    r = await client.post("/api/v1/tenants", json={"name": name})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


async def _publish(client, model_id):
    r = await client.post(f"/api/v1/models/{model_id}/publish")
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_publish_with_adapter_metadata_roundtrip(client):
    tenant_id = await _mk_tenant(client, "tenant-a")
    base = await _base_model(client)
    adapter = {
        "base_model": base["id"],
        "tenant_id": tenant_id,
        "trainer_run_ids": ["ft-1", "eval-9"],
        "eval_gate": GATE,
    }
    r = await client.post("/api/v1/models", json={"name": "office-adapter", "adapter": adapter})
    assert r.status_code == 201, r.text
    m = r.json()
    # base_model accepted as id -> adapter entry forced to LORA type
    assert m["model_type"] == "lora"
    assert m["base_model_id"] == base["id"]
    assert m["adapter"]["base_model"] == base["id"]
    assert m["adapter"]["tenant_id"] == tenant_id
    assert m["adapter"]["trainer_run_ids"] == ["ft-1", "eval-9"]
    assert m["adapter"]["eval_gate"] == GATE
    # read-back round-trips all fields
    r = await client.get(f"/api/v1/models/{m['id']}")
    assert r.json()["adapter"] == adapter


@pytest.mark.asyncio
async def test_adapter_base_model_resolved_by_name(client):
    base = await _base_model(client)
    r = await client.post(
        "/api/v1/models",
        json={"name": "by-name-adapter", "adapter": {"base_model": base["name"]}},
    )
    assert r.status_code == 201, r.text
    assert r.json()["base_model_id"] == base["id"]


@pytest.mark.asyncio
async def test_publish_without_adapter_unchanged(client):
    r = await client.post("/api/v1/models", json={"name": "plain-model"})
    assert r.status_code == 201, r.text
    assert r.json()["adapter"] is None


@pytest.mark.asyncio
async def test_adapter_base_model_unknown_404(client):
    r = await client.post(
        "/api/v1/models",
        json={"name": "orphan-adapter", "adapter": {"base_model": "nope"}},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_adapter_unknown_tenant_404(client):
    r = await client.post(
        "/api/v1/models",
        json={"name": "ghost-tenant-adapter", "adapter": {"tenant_id": "ghost"}},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_adapter_bad_evidence_shapes_422(client):
    r = await client.post(
        "/api/v1/models",
        json={"name": "bad-evidence", "adapter": {"trainer_run_ids": "ft-1"}},
    )
    assert r.status_code == 422
    r = await client.post(
        "/api/v1/models",
        json={"name": "bad-evidence-2", "adapter": {"eval_gate": [1, 2]}},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_update_adapter_metadata(client):
    base = await _base_model(client)
    r = await client.post("/api/v1/models", json={"name": "upd-adapter", "adapter": {"base_model": base["id"]}})
    mid = r.json()["id"]
    r = await client.put(f"/api/v1/models/{mid}", json={"adapter": {"eval_gate": GATE, "trainer_run_ids": ["ft-2"]}})
    assert r.status_code == 200, r.text
    got = r.json()["adapter"]
    assert got["eval_gate"] == GATE
    assert got["trainer_run_ids"] == ["ft-2"]


@pytest.mark.asyncio
async def test_catalog_list_filter_by_tenant(client):
    tenant_id = await _mk_tenant(client, "tenant-a")
    await client.post("/api/v1/models", json={"name": "shared-model"})
    r = await client.post(
        "/api/v1/models",
        json={"name": "tenanted-model", "adapter": {"tenant_id": tenant_id}},
    )
    assert r.status_code == 201, r.text
    # auth disabled: explicit tenant filter passes through
    r = await client.get("/api/v1/models", params={"tenant_id": tenant_id})
    names = [m["name"] for m in r.json()["items"]]
    assert names == ["tenanted-model"]
    r = await client.get("/api/v1/models")
    assert {m["name"] for m in r.json()["items"]} == {"shared-model", "tenanted-model"}
