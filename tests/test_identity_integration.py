import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps


@pytest.fixture
def settings():
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_test_identity",
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
        eval_runner_enabled=False,
        # #54: build the app with fusion-identity integration ON so the
        # TenantMiddleware is installed (outermost). The verify_jwt callback
        # is monkeypatched per-test — no real fusion-identity HTTP.
        identity_integration_enabled=True,
        identity_service_token="test-svc-token",
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


@pytest.fixture
async def auth_client(app, settings):
    # #55: identity-aware app with local auth ENABLED so a presented X-API-Key
    # is verified and the #55 cross-tenant key check is exercised.
    from fusion_model_hub.server.auth import set_auth_enabled

    set_auth_enabled(True)
    engine = get_engine(settings.db_url)
    await init_db(engine)
    init_deps(settings, engine)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _patch_verify(monkeypatch, *, tid="tenant-A", role="member", scopes=None):
    from fusion_model_hub.server import identity as ident

    claims = {
        "tid": tid,
        "role": role,
        "scope": scopes or [],
        "sub": "user-1",
        "jti": "jti-1",
    }

    def fake_verify(identity_url, service_token, token):
        if token == "bad-token":
            raise ValueError("identity verify failed: 401")
        return claims

    monkeypatch.setattr(ident, "_verify_token_sync", fake_verify)
    return claims


class TestIdentityIntegration:
    @pytest.mark.asyncio
    async def test_missing_tenant_header_rejected(self, client, monkeypatch):
        # #54: a request with a Bearer token but no X-Tenant-Id is rejected
        # 401 — the tenant context fabric requires the header.
        _patch_verify(monkeypatch)
        resp = await client.get(
            "/api/v1/models",
            headers={"Authorization": "Bearer good-token"},
        )
        assert resp.status_code == 401
        assert "X-Tenant-Id" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_missing_token_rejected(self, client, monkeypatch):
        # require_jwt=True: a request with X-Tenant-Id but no Bearer token
        # is rejected 401.
        _patch_verify(monkeypatch)
        resp = await client.get(
            "/api/v1/models",
            headers={"X-Tenant-Id": "tenant-A"},
        )
        assert resp.status_code == 401
        assert "token" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_forged_tenant_header_mismatch_rejected(self, client, monkeypatch):
        # #54 acceptance: a forged X-Tenant-Id that does not match the JWT
        # `tid` is rejected 401 (cross-tenant forge blocked). This replaces
        # the #53 blind-trust of X-Fusion-Tenant.
        _patch_verify(monkeypatch, tid="tenant-A")
        resp = await client.get(
            "/api/v1/models",
            headers={
                "Authorization": "Bearer good-token",
                "X-Tenant-Id": "tenant-B",
            },
        )
        assert resp.status_code == 401
        assert "mismatch" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_revoked_or_invalid_token_rejected(self, client, monkeypatch):
        # #54 acceptance: an invalid/revoked token (fusion-identity /verify
        # rejects it) is rejected 401.
        _patch_verify(monkeypatch)
        resp = await client.get(
            "/api/v1/models",
            headers={
                "Authorization": "Bearer bad-token",
                "X-Tenant-Id": "tenant-A",
            },
        )
        assert resp.status_code == 401
        assert "token" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_valid_token_matching_tenant_allowed(self, client, monkeypatch):
        # #54: a valid token whose JWT tid matches X-Tenant-Id passes the
        # middleware and serves the request.
        _patch_verify(monkeypatch, tid="tenant-A", role="member")
        resp = await client.get(
            "/api/v1/models",
            headers={
                "Authorization": "Bearer good-token",
                "X-Tenant-Id": "tenant-A",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_role_from_verify_response(self, client, monkeypatch):
        # #54 acceptance: the role on request.state comes from the verify
        # response (claims), not a local key row. We surface it indirectly:
        # a viewer role on a write method is rejected by the local RBAC only
        # when auth_enabled; here auth is off so we instead assert the
        # request reaches the handler (role stamped, no local key needed).
        _patch_verify(monkeypatch, tid="tenant-A", role="tenant_admin")
        resp = await client.post(
            "/api/v1/models",
            json={
                "name": "id-model-1",
                "model_type": "llm",
                "architecture": "qwen2",
            },
            headers={
                "Authorization": "Bearer good-token",
                "X-Tenant-Id": "tenant-A",
            },
        )
        assert resp.status_code == 201
        assert resp.json()["tenant_id"] == "tenant-A"

    @pytest.mark.asyncio
    async def test_health_remains_public(self, client, monkeypatch):
        # #54: the hub's health path is exempt — no tenant header or token
        # required, so liveness probes keep working behind the middleware.
        _patch_verify(monkeypatch)
        resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_cross_tenant_model_access_denied(self, client, monkeypatch):
        # #54 acceptance: a caller whose JWT resolves to tenant-A cannot read
        # a model created under tenant-B. The model is created as tenant-B
        # (forging X-Tenant-Id: tenant-B with a tenant-A token is blocked by
        # the middleware), so we seed tenant-B's model via a tenant-B token,
        # then attempt access as tenant-A — the hub's tenant scoping filters
        # it out of the tenant-A list.
        _patch_verify(monkeypatch, tid="tenant-B", role="tenant_admin")
        create = await client.post(
            "/api/v1/models",
            json={
                "name": "tenantB-only-model",
                "model_type": "llm",
                "architecture": "qwen2",
            },
            headers={
                "Authorization": "Bearer good-token",
                "X-Tenant-Id": "tenant-B",
            },
        )
        assert create.status_code == 201
        b_model = create.json()
        assert b_model["tenant_id"] == "tenant-B"

        # Now switch the resolved tenant to tenant-A.
        _patch_verify(monkeypatch, tid="tenant-A", role="member")
        listed = await client.get(
            "/api/v1/models",
            headers={
                "Authorization": "Bearer good-token",
                "X-Tenant-Id": "tenant-A",
            },
        )
        assert listed.status_code == 200
        names = {m["name"] for m in listed.json()["items"]}
        assert "tenantB-only-model" not in names


class TestIdentityApiKeyCombo:
    # #55: identity-aware mode where the studio attaches BOTH
    # Authorization: Bearer <jwt> + X-Tenant-Id AND the existing X-API-Key.
    # Acceptance: valid JWT + matching-tenant key -> tenant-scoped access;
    # valid JWT + mismatched-tenant key -> 401 (cross-tenant key reuse blocked).

    @pytest.mark.asyncio
    async def test_bearer_plus_matching_key_allowed(self, auth_client, monkeypatch):
        _patch_verify(monkeypatch, tid="tenant-A", role="tenant_admin")
        # Create a key under tenant-A (JWT tid stamps request.state.tenant_id,
        # and auth/keys derives the key tenant from the caller tenant).
        create = await auth_client.post(
            "/api/v1/auth/keys",
            json={"name": "combo-key-A", "role": "admin"},
            headers={"Authorization": "Bearer good-token", "X-Tenant-Id": "tenant-A"},
        )
        assert create.status_code == 201, create.text
        raw_key = create.json()["key"]
        assert create.json()["tenant_id"] == "tenant-A"

        # Bearer + X-Tenant-Id + X-API-Key all present, tenants agree -> 200.
        resp = await auth_client.get(
            "/api/v1/models",
            headers={
                "Authorization": "Bearer good-token",
                "X-Tenant-Id": "tenant-A",
                "X-API-Key": raw_key,
            },
        )
        assert resp.status_code == 200, resp.text

    @pytest.mark.asyncio
    async def test_cross_tenant_key_rejected(self, auth_client, monkeypatch):
        # Seed a tenant-B key.
        _patch_verify(monkeypatch, tid="tenant-B", role="tenant_admin")
        create_b = await auth_client.post(
            "/api/v1/auth/keys",
            json={"name": "combo-key-B", "role": "admin"},
            headers={"Authorization": "Bearer good-token", "X-Tenant-Id": "tenant-B"},
        )
        assert create_b.status_code == 201, create_b.text
        b_key = create_b.json()["key"]
        assert create_b.json()["tenant_id"] == "tenant-B"

        # Now authenticate as tenant-A but present tenant-B's key -> 401.
        _patch_verify(monkeypatch, tid="tenant-A", role="tenant_admin")
        resp = await auth_client.get(
            "/api/v1/models",
            headers={
                "Authorization": "Bearer good-token",
                "X-Tenant-Id": "tenant-A",
                "X-API-Key": b_key,
            },
        )
        assert resp.status_code == 401
        assert "tenant" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_apikey_only_preserved_when_identity_off(self, monkeypatch):
        # #55 acceptance: X-API-Key-only (no identity headers) preserves current
        # behavior. Built on a SEPARATE identity-disabled app so the tenant
        # middleware is not installed and no Bearer is required.
        settings = Settings(
            host="127.0.0.1",
            port=11444,
            data_dir="/tmp/fmh_test_identity_off",
            db_url="sqlite+aiosqlite:///:memory:",
            log_level="WARNING",
            eval_runner_enabled=False,
            identity_integration_enabled=False,
        )
        app = create_app(settings)
        from fusion_model_hub.server.auth import set_auth_enabled

        set_auth_enabled(True)
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            # Bootstrap a key on a public path (auth/keys is public w/ zero keys).
            create = await c.post("/api/v1/auth/keys", json={"name": "legacy-key", "role": "admin"})
            assert create.status_code == 201, create.text
            raw_key = create.json()["key"]

            # No Authorization, no X-Tenant-Id — just X-API-Key -> 200.
            resp = await c.get("/api/v1/models", headers={"X-API-Key": raw_key})
            assert resp.status_code == 200, resp.text
