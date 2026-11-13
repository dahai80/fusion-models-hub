import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.catalog.schema import validate_dir, validate_entry
from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps


def _entry(**kw):
    base = {
        "schemaVersion": 1,
        "name": "test-model",
        "contextWindowTokens": 32768,
        "maxOutputTokens": 8192,
        "modality": {"input": ["text"], "output": ["text"]},
    }
    base.update(kw)
    return base


class TestSchemaValidation:
    def test_valid_entry(self):
        e = validate_entry(_entry())
        assert e.name == "test-model"
        assert e.pricing.currency == "USD"

    def test_rejects_bad_schema_version(self):
        with pytest.raises(ValueError, match="schemaVersion"):
            validate_entry(_entry(schemaVersion=2))

    def test_rejects_unknown_modality(self):
        with pytest.raises(ValueError, match="unknown values"):
            validate_entry(_entry(modality={"input": ["text"], "output": ["smell"]}))

    def test_rejects_empty_modality(self):
        with pytest.raises(ValueError, match="at least one"):
            validate_entry(_entry(modality={"input": [], "output": ["text"]}))

    def test_rejects_output_exceeding_context(self):
        with pytest.raises(ValueError, match="exceeds contextWindowTokens"):
            validate_entry(_entry(maxOutputTokens=40000))

    def test_rejects_unknown_fields(self):
        # #102: extra fields rejected so schema drift fails loudly.
        with pytest.raises(ValueError):
            validate_entry(_entry(context_window_tokens=1))

    def test_rejects_nonpositive_tokens(self):
        with pytest.raises(ValueError):
            validate_entry(_entry(contextWindowTokens=0))


class TestCatalogDir:
    def test_seeded_catalog_valid(self):
        # The in-repo seed must always pass the gate it ships with.
        entries = validate_dir(Path(__file__).resolve().parents[1] / "catalog" / "models")
        assert len(entries) >= 1
        names = [e.name for e in entries]
        assert len(names) == len(set(names))

    def test_validate_dir_raises_on_bad_file(self, tmp_path):
        (tmp_path / "bad.json").write_text(json.dumps(_entry(schemaVersion=99)))
        with pytest.raises(ValueError, match="schemaVersion"):
            validate_dir(tmp_path)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir=str(tmp_path / "data"),
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
        models_dir=str(tmp_path / "models"),
        eval_runner_enabled=False,
        catalog_dir=str(tmp_path / "catalog"),
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


class TestCatalogApi:
    async def test_list_empty(self, client, settings):
        resp = await client.get("/api/v1/catalog")
        assert resp.status_code == 200
        assert resp.json() == {"models": [], "total": 0, "schemaVersion": 1}

    async def test_list_and_get(self, client, settings):
        Path(settings.catalog_dir).mkdir(parents=True)
        Path(settings.catalog_dir, "test-model.json").write_text(json.dumps(_entry()))
        resp = await client.get("/api/v1/catalog")
        assert resp.json()["total"] == 1
        resp = await client.get("/api/v1/catalog/test-model")
        assert resp.status_code == 200
        assert resp.json()["contextWindowTokens"] == 32768

    async def test_invalid_file_skipped(self, client, settings):
        Path(settings.catalog_dir).mkdir(parents=True)
        Path(settings.catalog_dir, "good.json").write_text(json.dumps(_entry(name="good")))
        Path(settings.catalog_dir, "bad.json").write_text("{not json")
        resp = await client.get("/api/v1/catalog")
        body = resp.json()
        assert body["total"] == 1
        assert body["models"][0]["name"] == "good"

    async def test_get_missing_404(self, client):
        resp = await client.get("/api/v1/catalog/nope")
        assert resp.status_code == 404
