import json

import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server import shell_registry
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.auth import set_auth_enabled
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps


def _settings(tmp_path, **kw):
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir=str(tmp_path),
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
        eval_runner_enabled=False,
        **kw,
    )


def _manifest_dict():
    return {
        "id": "fusion-models-hub",
        "title": "模型仓库",
        "title_en": "Model Hub",
        "category": "platform",
        "sort": 43,
        "icon": "Database",
        "entry": "/shell/fusion-models-hub/",
        "health": "http://127.0.0.1:11444/api/v1/system/health",
        "upstream": "http://127.0.0.1:11444",
        "devices": ["desktop"],
        "pages": [{"route": "/models", "title": "模型", "title_en": "Models"}],
    }


class TestRegistrationPayload:
    def test_strips_api_key_and_resolves_upstream(self):
        manifest = dict(_manifest_dict(), api_key="secret")
        payload = shell_registry.build_registration_payload(manifest)
        assert "api_key" not in payload
        assert payload["upstream"] == "http://127.0.0.1:11444"

    def test_upstream_env_override(self, monkeypatch):
        monkeypatch.setenv("FMH_SHELL_UPSTREAM", "http://127.0.0.1:9999")
        payload = shell_registry.build_registration_payload(_manifest_dict())
        assert payload["upstream"] == "http://127.0.0.1:9999"

    def test_upstream_invalid_rejected(self):
        manifest = dict(_manifest_dict(), upstream="not-a-url")
        with pytest.raises(ValueError):
            shell_registry.build_registration_payload(manifest)

    def test_load_manifest_from_file(self, tmp_path):
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(_manifest_dict()), encoding="utf-8")
        manifest = shell_registry.load_manifest(str(p))
        assert manifest["id"] == "fusion-models-hub"

    def test_load_manifest_missing_id_rejected(self, tmp_path):
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps({"title": "x"}), encoding="utf-8")
        with pytest.raises(ValueError):
            shell_registry.load_manifest(str(p))


class TestModuleKey:
    def test_key_roundtrip(self, tmp_path):
        shell_registry.save_module_key(str(tmp_path), "fm-test-key-1234")
        path = tmp_path / "web_module_key"
        assert path.read_text().strip() == "fm-test-key-1234"
        assert (path.stat().st_mode & 0o777) == 0o600
        assert shell_registry.load_module_key(str(tmp_path)) == "fm-test-key-1234"

    def test_empty_key_not_saved(self, tmp_path):
        assert shell_registry.save_module_key(str(tmp_path), "  ") is False
        assert shell_registry.load_module_key(str(tmp_path)) == ""

    def test_accept_keys_include_issued(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FUSION_CLIENT_KEY", "")
        shell_registry.save_module_key(str(tmp_path), "fm-issued-key")
        keys = shell_registry.shell_accept_keys(str(tmp_path))
        assert "fm-issued-key" in keys

    def test_extract_issued_key_variants(self):
        assert shell_registry._extract_issued_key({"raw_key": "k1"}) == "k1"
        assert shell_registry._extract_issued_key({"data": {"api_key": "k2"}}) == "k2"
        assert shell_registry._extract_issued_key({"other": 1}) == ""
        assert shell_registry._extract_issued_key("nope") == ""


class TestRegistryLoop:
    async def test_register_once_and_persist_key(self, tmp_path, monkeypatch):
        class FakeResp:
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self._payload = payload
                self.text = ""

            def json(self):
                return self._payload

        class FakeClient:
            def __init__(self):
                self.posts = []

            async def post(self, url, json=None, headers=None):
                self.posts.append((url, json, headers))
                if url.endswith("/api/v1/shell/manifests"):
                    return FakeResp(200, {"raw_key": "fm-issued-abc"})
                return FakeResp(200, {})

        monkeypatch.setenv("FUSION_SHELL_GATEWAY_URL", "http://127.0.0.1:11432")
        monkeypatch.setenv("FUSION_ADMIN_KEY", "gw-admin")
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(_manifest_dict()), encoding="utf-8")
        client = FakeClient()
        await shell_registry._registry_loop_body(
            client, json.loads(p.read_text()), str(tmp_path), interval=0, max_ticks=2
        )
        register_calls = [c for c in client.posts if c[0].endswith("/api/v1/shell/manifests")]
        heartbeat_calls = [c for c in client.posts if "/heartbeat" in c[0]]
        assert len(register_calls) == 1
        assert len(heartbeat_calls) == 1
        assert register_calls[0][1]["id"] == "fusion-models-hub"
        assert "api_key" not in register_calls[0][1]
        assert register_calls[0][2]["Authorization"] == "Bearer gw-admin"
        assert heartbeat_calls[0][2]["Authorization"] == "Bearer fm-issued-abc"
        assert shell_registry.load_module_key(str(tmp_path)) == "fm-issued-abc"

    async def test_heartbeat_404_triggers_reregister(self, tmp_path, monkeypatch):
        class FakeResp:
            def __init__(self, status_code, payload=None):
                self.status_code = status_code
                self._payload = payload or {}
                self.text = ""

            def json(self):
                return self._payload

        class FakeClient:
            def __init__(self):
                self.n_register = 0
                self.n_heartbeat = 0

            async def post(self, url, json=None, headers=None):
                if url.endswith("/api/v1/shell/manifests"):
                    self.n_register += 1
                    return FakeResp(200, {"raw_key": "fm-k"})
                self.n_heartbeat += 1
                return FakeResp(404)

        monkeypatch.setenv("FUSION_SHELL_GATEWAY_URL", "http://127.0.0.1:11432")
        monkeypatch.setenv("FUSION_ADMIN_KEY", "gw-admin")
        shell_registry.save_module_key(str(tmp_path), "fm-k")
        client = FakeClient()
        await shell_registry._registry_loop_body(
            client, _manifest_dict(), str(tmp_path), interval=0, max_ticks=3
        )
        assert client.n_register == 2
        assert client.n_heartbeat == 1


class TestShellAuthPath:
    @pytest.fixture
    async def authed_client(self, tmp_path):
        settings = _settings(tmp_path)
        app = create_app(settings)
        set_auth_enabled(True)
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        shell_registry.save_module_key(str(tmp_path), "fm-shell-key-xyz")
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c, tmp_path

    async def test_bearer_issued_key_reads_pass(self, authed_client):
        c, _ = authed_client
        resp = await c.get(
            "/api/v1/models", headers={"Authorization": "Bearer fm-shell-key-xyz"}
        )
        assert resp.status_code == 200

    async def test_bearer_issued_key_write_denied(self, authed_client):
        c, _ = authed_client
        resp = await c.post(
            "/api/v1/models",
            headers={"Authorization": "Bearer fm-shell-key-xyz"},
            json={"name": "m", "model_type": "LLM"},
        )
        assert resp.status_code == 403

    async def test_bearer_wrong_key_rejected(self, authed_client):
        c, _ = authed_client
        resp = await c.get("/api/v1/models", headers={"Authorization": "Bearer fm-wrong"})
        assert resp.status_code == 401


class TestWebStatic:
    async def test_index_served_when_dist_exists(self, tmp_path):
        dist = tmp_path / "webdist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<html>hub</html>", encoding="utf-8")
        (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
        settings = _settings(tmp_path, web_dist_dir=str(dist))
        app = create_app(settings)
        set_auth_enabled(False)
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/")
            assert resp.status_code == 200
            assert "hub" in resp.text
            resp = await c.get("/assets/app.js")
            assert resp.status_code == 200
            # API routes keep winning.
            resp = await c.get("/api/v1/system/health")
            assert resp.status_code == 200

    async def test_no_static_when_dist_missing(self, tmp_path):
        settings = _settings(tmp_path, web_dist_dir=str(tmp_path / "nope"))
        app = create_app(settings)
        assert not any(getattr(r, "path", "") == "/" for r in app.routes)
