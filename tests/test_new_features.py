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

    @pytest.mark.asyncio
    async def test_route_inference_round_robin_balances_across_nodes(self, client):
        # Issue #31: with 3 active nodes and mode=cluster, N sequential requests
        # must spread evenly (round-robin), not all hit nodes[-1] (the
        # created_at-DESC newest-registered node that the old failover loop hit
        # every call). Mock _check_alive -> False to force the cluster branch,
        # and mock _chat to echo the node URL so the response records which
        # node served each call. The module counter is reset for determinism.
        import fusion_model_hub.server.routers.cluster as cmod
        cmod._round_robin_counter = __import__("itertools").count()

        model = await client.post("/api/v1/models", json={"name": "rr-model"})
        mid = model.json()["id"]

        for i in range(3):
            await client.post("/api/v1/cluster/nodes", json={
                "name": f"node-{i}",
                "url": f"http://10.0.0.{10+i}:11434",
            })

        async def _fake_chat(url, settings, model_name, messages):
            return {"id": f"chat-{url}", "model": model_name,
                    "choices": [{"message": {"role": "assistant", "content": url}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        routed = []
        with patch.object(cmod, "_check_alive", new=AsyncMock(return_value=False)), \
             patch.object(cmod, "_chat", new=_fake_chat):
            for _ in range(6):
                resp = await client.post("/api/v1/cluster/route-inference", json={
                    "model_id": mid,
                    "mode": "cluster",
                    "messages": [{"role": "user", "content": "ping"}],
                })
                assert resp.status_code == 200, resp.text
                routed.append(resp.json()["routedTo"])

        # 3 nodes, 6 calls -> exactly 2 per node (round-robin, not failover).
        from collections import Counter
        counts = Counter(routed)
        assert len(counts) == 3, f"load not spread across all 3 nodes: {counts}"
        assert set(counts.values()) == {2}, f"uneven distribution: {counts}"

    @pytest.mark.asyncio
    async def test_route_inference_round_robin_failover_within_call(self, client):
        # Issue #31: round-robin rotates the START node, but iteration must
        # still fall through to the rest of the active list on failure (failover
        # preserved within a single call). Register 2 nodes; make the
        # round-robin start node's _chat raise so the second node must serve.
        import fusion_model_hub.server.routers.cluster as cmod
        cmod._round_robin_counter = __import__("itertools").count()

        model = await client.post("/api/v1/models", json={"name": "fo-model"})
        mid = model.json()["id"]
        await client.post("/api/v1/cluster/nodes", json={
            "name": "node-0", "url": "http://10.0.0.20:11434"})
        await client.post("/api/v1/cluster/nodes", json={
            "name": "node-1", "url": "http://10.0.0.21:11434"})

        async def _fake_chat(url, settings, model_name, messages):
            if url.endswith(".20:11434"):
                raise RuntimeError("node-0 down")
            return {"id": "chat-ok", "model": model_name,
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        with patch.object(cmod, "_check_alive", new=AsyncMock(return_value=False)), \
             patch.object(cmod, "_chat", new=_fake_chat):
            resp = await client.post("/api/v1/cluster/route-inference", json={
                "model_id": mid,
                "mode": "cluster",
                "messages": [{"role": "user", "content": "ping"}],
            })
            assert resp.status_code == 200, resp.text
            assert resp.json()["content"] == "ok"


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


async def _authed_client(settings_overrides: dict | None = None, auth_on: bool = True):
    # E-E6/E-E7 regression helper: build a client against a fresh in-memory DB
    # so bootstrap/usage tests run in isolation. auth_on toggles auth at build
    # time (the caller may flip it later via set_auth_enabled). Returns
    # (client, settings). Caller is responsible for set_auth_enabled(False)
    # cleanup in a finally block.
    from fusion_model_hub.server.auth import set_auth_enabled
    from fusion_model_hub.server.deps import init_deps
    set_auth_enabled(auth_on)
    overrides = settings_overrides or {}
    s = Settings(
        host="127.0.0.1", port=11444,
        data_dir="/tmp/fmh_test_audit",
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
        **overrides,
    )
    engine = get_engine(s.db_url)
    await init_db(engine)
    init_deps(s, engine)
    app = create_app(s)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    c = AsyncClient(transport=transport, base_url="http://test")
    await c.__aenter__()
    return c, s


class TestBootstrapGuardE6:
    # E-E6: POST /auth/keys is public only while zero active keys exist.
    # Harden bootstrap with an optional shared token (X-Bootstrap-Token) and a
    # per-IP rate limit so the first-to-arrive race cannot grant root to anyone
    # who can reach the Hub.

    @pytest.mark.asyncio
    async def test_bootstrap_token_required_when_set(self):
        reset_rate_limits()
        try:
            c, _ = await _authed_client({"auth_bootstrap_token": "secret-bootstrap-123"})
            # No token -> 403, no key created.
            resp = await c.post("/api/v1/auth/keys", json={"name": "root", "role": "admin"})
            assert resp.status_code == 403
            assert resp.json()["detail"] == "Bootstrap token required to create the first key"
            # Wrong token -> 403.
            resp = await c.post(
                "/api/v1/auth/keys",
                json={"name": "root", "role": "admin"},
                headers={"X-Bootstrap-Token": "wrong"},
            )
            assert resp.status_code == 403
            # Correct token -> 201, first admin key created.
            resp = await c.post(
                "/api/v1/auth/keys",
                json={"name": "root", "role": "admin"},
                headers={"X-Bootstrap-Token": "secret-bootstrap-123"},
            )
            assert resp.status_code == 201
            await c.aclose()
        finally:
            from fusion_model_hub.server.auth import set_auth_enabled
            set_auth_enabled(False)
            reset_rate_limits()

    @pytest.mark.asyncio
    async def test_bootstrap_without_token_still_works_when_unset(self):
        # Backward compat: no FMH_AUTH_BOOTSTRAP_TOKEN -> bootstrap stays open
        # (IP rate-limited only). Local single-user installs rely on this.
        reset_rate_limits()
        try:
            c, _ = await _authed_client({})
            resp = await c.post("/api/v1/auth/keys", json={"name": "root", "role": "admin"})
            assert resp.status_code == 201
            await c.aclose()
        finally:
            from fusion_model_hub.server.auth import set_auth_enabled
            set_auth_enabled(False)
            reset_rate_limits()

    @pytest.mark.asyncio
    async def test_bootstrap_ip_rate_limit(self):
        # 10/min bootstrap budget from one IP. POST /auth/keys is public only
        # while zero active keys exist, so only the FIRST POST travels the
        # bootstrap path; later POSTs hit the admin-auth path (a separate
        # concern). To prove the bootstrap cap fires on the endpoint, pin the
        # client-IP lookup to a known value, drain THAT bucket, then POST — it
        # must 429 and create no key.
        reset_rate_limits()
        try:
            c, _ = await _authed_client({"auth_bootstrap_token": "t"})
            from fusion_model_hub.server.routers import auth as auth_router
            with patch.object(auth_router, "_client_ip", return_value="10.0.0.9"):
                # Drain the 10/min bucket for the pinned IP.
                for _ in range(10):
                    assert auth_router.check_rate_limit("bootstrap:10.0.0.9", 10)
                assert auth_router.check_rate_limit("bootstrap:10.0.0.9", 10) is False
                # The endpoint sees the same denied bucket -> 429, no key created.
                r = await c.post(
                    "/api/v1/auth/keys",
                    json={"name": "x", "role": "admin"},
                    headers={"X-Bootstrap-Token": "t"},
                )
                assert r.status_code == 429
                assert r.json()["detail"] == "Bootstrap rate limit exceeded, retry shortly"
            await c.aclose()
        finally:
            from fusion_model_hub.server.auth import set_auth_enabled
            set_auth_enabled(False)
            reset_rate_limits()

    @pytest.mark.asyncio
    async def test_admin_can_create_second_key_after_bootstrap(self):
        # #58 root-cause regression: POST /auth/keys is in PUBLIC_PATHS, so the
        # auth middleware used to early-return and never stamp request.state.
        # user_role — the route's admin-or-bootstrap guard then saw role="" and
        # 403'd any admin creating a 2nd key once one already existed. Fix:
        # "public" means no key REQUIRED, not "ignore a presented key"; an admin
        # key on POST /auth/keys must authenticate and pass the admin check.
        reset_rate_limits()
        try:
            c, _ = await _authed_client({})
            # Bootstrap: no key, no token -> first admin key (active count 0).
            boot = await c.post("/api/v1/auth/keys", json={"name": "root", "role": "admin"})
            assert boot.status_code == 201, boot.text
            root_key = boot.json()["key"]
            # Post-bootstrap: admin key presented -> must authenticate + pass.
            second = await c.post(
                "/api/v1/auth/keys",
                json={"name": "second", "role": "developer"},
                headers={"X-API-Key": root_key},
            )
            assert second.status_code == 201, second.text
            assert second.json()["name"] == "second"
            assert second.json()["role"] == "developer"
            # A non-admin (developer) key must NOT be able to create more keys.
            dev_key = second.json()["key"]
            third = await c.post(
                "/api/v1/auth/keys",
                json={"name": "third", "role": "viewer"},
                headers={"X-API-Key": dev_key},
            )
            assert third.status_code == 403
            assert third.json()["detail"] == "Only admin can create API keys once bootstrap key exists"
            await c.aclose()
        finally:
            from fusion_model_hub.server.auth import set_auth_enabled
            set_auth_enabled(False)
            reset_rate_limits()


class TestPerKeyUsageE7:
    # E-E7: /auth/keys/{id}/usage must return ONLY this key's inference volume,
    # not the global _model_stats aggregate. Two keys each do inference; usage
    # for key A must not include key B's counts.

    @pytest.mark.asyncio
    async def test_usage_isolated_per_key(self):
        # E-E7 regression: per-key usage isolation. POST /auth/keys is public
        # only for the first key; creating a 2nd/3rd as an admin caller is
        # gated by the PUBLIC_PATHS bypass (middleware sets no user_role on
        # public paths), so create all keys with auth OFF, then turn auth ON
        # for the usage GETs (which are NOT public — sub-path needs a key).
        from fusion_model_hub.server.auth import set_auth_enabled
        from fusion_model_hub.server.routers.inference import (
            _model_stats,
            _update_model_stats,
        )
        reset_rate_limits()
        try:
            c, _ = await _authed_client({"auth_bootstrap_token": "boot"}, auth_on=False)
            # Auth is OFF here — all three keys created directly.
            root = (await c.post("/api/v1/auth/keys", json={"name": "root", "role": "admin"})).json()
            root_key = root["key"]
            ka = (await c.post("/api/v1/auth/keys", json={"name": "keyA", "role": "developer"})).json()
            kb = (await c.post("/api/v1/auth/keys", json={"name": "keyB", "role": "developer"})).json()
            # Seed _model_stats per-key via the internal updater with the two
            # different key_ids — avoids needing a live MLX for inference.
            _model_stats.clear()
            for _ in range(5):
                _update_model_stats("model-1", 10.0, tokens=100, key_id=ka["id"])
            for _ in range(3):
                _update_model_stats("model-2", 20.0, tokens=50, key_id=kb["id"])
            # Now require auth for the usage GETs.
            set_auth_enabled(True)
            # Usage for keyA sees only its 5 requests on model-1.
            ua = await c.get(f"/api/v1/auth/keys/{ka['id']}/usage", headers={"X-API-Key": root_key})
            assert ua.status_code == 200
            ua_data = ua.json()
            assert ua_data["total_requests"] == 5
            assert set(ua_data["by_model"].keys()) == {"model-1"}
            assert ua_data["by_model"]["model-1"]["total_tokens"] == 500
            # Usage for keyB sees only its 3 requests on model-2.
            ub = await c.get(f"/api/v1/auth/keys/{kb['id']}/usage", headers={"X-API-Key": root_key})
            assert ub.status_code == 200
            ub_data = ub.json()
            assert ub_data["total_requests"] == 3
            assert set(ub_data["by_model"].keys()) == {"model-2"}
            assert ub_data["by_model"]["model-2"]["total_tokens"] == 150
            _model_stats.clear()
            await c.aclose()
        finally:
            set_auth_enabled(False)
            reset_rate_limits()


class TestSanitizedErrorDetailE5:
    # E-E5: error responses must not leak internal str(e)/resp.text (DB errors,
    # MLX response bodies, filesystem paths). Sanitized to a fixed message +
    # trace_id. Verify the shape on a quantize endpoint that previously raised
    # detail=str(e) when the async submit fails.

    @pytest.mark.asyncio
    async def test_quantize_submit_failure_sanitized(self):
        # E-E5 regression: when the quantize route's submit call raises, the
        # response must be a fixed message + trace_id, NOT the raw str(e) (which
        # could carry DB errors / MLX response bodies / filesystem paths).
        # Force the failure deterministically by patching submit_quantize to
        # raise an exception whose message contains a sensitive marker; assert
        # the marker never reaches the client.
        from fusion_model_hub.server.auth import set_auth_enabled
        reset_rate_limits()
        try:
            c, _ = await _authed_client({}, auth_on=False)
            sensitive = "SECRET-internal-path=/etc/fmh/db.sqlite"
            with patch(
                "fusion_model_hub.server.routers.quantize.submit_quantize",
                new=AsyncMock(side_effect=RuntimeError(sensitive)),
            ):
                resp = await c.post(
                    "/api/v1/quantize",
                    json={"source_version_id": "v1", "quant_bits": 4},
                )
            assert resp.status_code == 500
            detail = resp.json()["detail"]
            assert "trace_id=" in detail
            assert "Failed to submit quantize task" in detail
            # The raw internal exception text must NOT leak.
            assert sensitive not in detail
            assert "SECRET-internal-path" not in detail
            await c.aclose()
        finally:
            set_auth_enabled(False)
            reset_rate_limits()
