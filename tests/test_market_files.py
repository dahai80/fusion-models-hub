import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None):
        if "hf-mirror.com" in url:
            if url.endswith("/api/models/org/repo"):
                return _FakeResponse(payload={"siblings": [
                    {"rfilename": "config.json"},
                    {"rfilename": "model.safetensors"},
                ]})
            return _FakeResponse(status_code=404)
        if "modelscope.cn" in url:
            return _FakeResponse(payload={"Data": {"Files": [
                {"Path": "config.json", "Size": 512, "Type": "blob"},
                {"Path": "model.safetensors", "Size": 4096, "Type": "blob"},
                {"Path": "docs", "Size": 0, "Type": "tree"},
            ]}})
        return _FakeResponse(status_code=404)


@pytest.fixture
def settings():
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_test_data",
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
    )


@pytest.fixture
async def client(settings):
    from fusion_model_hub.server.auth import set_auth_enabled

    set_auth_enabled(False)
    app = create_app(settings)
    engine = get_engine(settings.db_url)
    await init_db(engine)
    init_deps(settings, engine)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def fake_http(monkeypatch):
    monkeypatch.setattr(
        "fusion_model_hub.server.routers.models.httpx.AsyncClient", _FakeClient
    )


@pytest.mark.asyncio
async def test_market_files_hf(client, fake_http):
    resp = await client.get(
        "/api/v1/models/market/files?source=hf&repo=org/repo"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo"] == "org/repo"
    paths = [f["path"] for f in data["files"]]
    assert "model.safetensors" in paths
    assert "config.json" in paths


@pytest.mark.asyncio
async def test_market_files_modelscope(client, fake_http):
    resp = await client.get(
        "/api/v1/models/market/files?source=modelscope&repo=org/repo"
    )
    assert resp.status_code == 200
    files = {f["path"]: f["size"] for f in resp.json()["files"]}
    assert files["model.safetensors"] == 4096
    # tree entries are filtered out
    assert "docs" not in files


@pytest.mark.asyncio
async def test_market_files_upstream_error(client, fake_http):
    resp = await client.get(
        "/api/v1/models/market/files?source=hf&repo=missing/repo"
    )
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_market_files_validation(client):
    resp = await client.get("/api/v1/models/market/files?source=hf&repo=")
    assert resp.status_code == 400
    resp = await client.get(
        "/api/v1/models/market/files?source=gitlab&repo=org/repo"
    )
    assert resp.status_code == 400
