import logging
import shutil
import tempfile
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
    tmp_dir = tempfile.mkdtemp(prefix="fmh_cov_misc_")
    yield Settings(
        host="127.0.0.1",
        port=11444,
        data_dir=tmp_dir,
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
        bench_url="http://bench.example.com:8090",
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)


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


def _mock_httpx_client(get_resp=None, post_resp=None, get_side=None, post_side=None):
    mock_instance = AsyncMock()
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    if get_resp is not None:
        mock_instance.get.return_value = get_resp
    if post_resp is not None:
        mock_instance.post.return_value = post_resp
    if get_side is not None:
        mock_instance.get.side_effect = get_side
    if post_side is not None:
        mock_instance.post.side_effect = post_side
    return mock_instance


def _resp(status_code=200, json_data=None, text=""):
    r = MagicMock()
    r.status_code = status_code
    r.text = text or (str(json_data) if json_data else "")
    r.json.return_value = json_data if json_data is not None else {}
    r.raise_for_status = MagicMock()
    return r


async def _create_model(client, name="cov-model"):
    resp = await client.post(
        "/api/v1/models",
        json={
            "name": name,
            "description": "test",
            "model_type": "llm",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _create_version(client, model_id, version="1.0.0", quantization="4bit"):
    resp = await client.post(
        f"/api/v1/models/{model_id}/versions",
        data={"version": version, "format": "mlx", "quantization": quantization},
        files={"file": ("", b"")},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _create_published_version(client, model_id, version="1.0.0"):
    v = await _create_version(client, model_id, version)
    metrics = await client.put(
        f"/api/v1/versions/{v['id']}/metrics",
        json={"benchmark_score": 90.0},
    )
    assert metrics.status_code == 200, metrics.text
    resp = await client.post(f"/api/v1/versions/{v['id']}/promote")
    assert resp.status_code == 200, resp.text
    return v


class TestMonitorRealtime:
    async def test_realtime_empty(self, client):
        resp = await client.get("/api/v1/monitor/realtime")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["loaded_count"] == 0
        assert data["models"] == []

    async def test_realtime_with_unloaded_model(self, client):
        m = await _create_model(client, "mon-model-1")
        resp = await client.get("/api/v1/monitor/realtime")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1
        entry = data["models"][0]
        assert entry["model_id"] == m["id"]
        assert entry["status"] == "not_loaded"
        assert entry["concurrent_requests"] == 0
        assert entry["tokens_per_second"] == 0.0

    async def test_realtime_with_loaded_model_and_stats(self, client):
        from fusion_model_hub.server.routers.inference import _loaded_models, _model_stats

        m = await _create_model(client, "mon-model-2")
        v = await _create_published_version(client, m["id"])
        import time

        now = time.time()
        _loaded_models[m["id"]] = {
            "version_id": v["id"],
            "model_name": "mon-model-2",
            "status": "loaded",
            "loaded_at": now - 60,
        }
        _model_stats[m["id"]] = {
            "request_count": 5,
            "total_latency": 500.0,
            "total_tokens": 1000,
            "last_request_at": now - 10,
            "first_request_at": now - 100,
            "source_module": "fusion-code",
        }
        try:
            resp = await client.get("/api/v1/monitor/realtime")
            assert resp.status_code == 200
            data = resp.json()
            assert data["summary"]["loaded_count"] == 1
            entry = data["models"][0]
            assert entry["status"] == "loaded"
            assert entry["avg_latency_ms"] == 100.0
            assert entry["source_module"] == "fusion-code"
            assert entry["concurrent_requests"] == 1
            assert entry["tokens_per_second"] > 0
            assert entry["last_request_at"] is not None
            assert entry["loaded_since"] is not None
            assert data["summary"]["total_requests_today"] == 5
        finally:
            _loaded_models.pop(m["id"], None)
            _model_stats.pop(m["id"], None)

    async def test_realtime_stats_stale_no_tps(self, client):
        from fusion_model_hub.server.routers.inference import _model_stats

        m = await _create_model(client, "mon-model-3")
        import time

        now = time.time()
        _model_stats[m["id"]] = {
            "request_count": 2,
            "total_latency": 200.0,
            "total_tokens": 100,
            "last_request_at": now - 3600,
            "first_request_at": now - 3700,
        }
        try:
            resp = await client.get("/api/v1/monitor/realtime")
            assert resp.status_code == 200
            entry = resp.json()["models"][0]
            assert entry["avg_latency_ms"] == 100.0
            assert entry["concurrent_requests"] == 0
            assert entry["tokens_per_second"] == 0.0
            assert entry["last_request_at"] is not None
        finally:
            _model_stats.pop(m["id"], None)


class TestMonitorModelStats:
    async def test_model_stats_empty(self, client):
        resp = await client.get("/api/v1/monitor/model-stats")
        assert resp.status_code == 200
        assert resp.json()["stats"] == []

    async def test_model_stats_with_model(self, client):
        m = await _create_model(client, "stat-model-1")
        resp = await client.get("/api/v1/monitor/model-stats")
        assert resp.status_code == 200
        stats = resp.json()["stats"]
        assert len(stats) == 1
        s = stats[0]
        assert s["model_id"] == m["id"]
        assert s["avg_latency_ms"] == 0.0
        assert s["tokens_per_second"] == 0.0
        assert s["active_sessions"] == 0
        assert s["node"] == "local"

    async def test_model_stats_active_with_loaded_version(self, client):
        from fusion_model_hub.server.routers.inference import _loaded_models, _model_stats

        m = await _create_model(client, "stat-model-2")
        v = await _create_published_version(client, m["id"])
        import time

        now = time.time()
        _loaded_models[m["id"]] = {
            "version_id": v["id"],
            "model_name": "stat-model-2",
            "status": "loaded",
            "loaded_at": now,
        }
        _model_stats[m["id"]] = {
            "request_count": 10,
            "total_latency": 2000.0,
            "total_tokens": 500,
            "last_request_at": now - 5,
            "first_request_at": now - 120,
        }
        try:
            resp = await client.get("/api/v1/monitor/model-stats")
            assert resp.status_code == 200
            s = resp.json()["stats"][0]
            assert s["avg_latency_ms"] == 200.0
            assert s["active_sessions"] == 1
            assert s["tokens_per_second"] > 0
            assert s["requests_per_min"] > 0
        finally:
            _loaded_models.pop(m["id"], None)
            _model_stats.pop(m["id"], None)


class TestAnalyze:
    async def test_analyze_requires_input(self, client):
        resp = await client.post("/api/v1/analyze", json={})
        assert resp.status_code == 400

    async def test_analyze_hf_repo_success(self, client):
        mock_resp = _resp(200, {"architecture": "llama", "params": "7B"})
        mock_inst = _mock_httpx_client(post_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.analyze.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.post("/api/v1/analyze", json={"hf_repo": "meta-llama/Llama-3-8B"})
        assert resp.status_code == 200
        assert resp.json()["architecture"] == "llama"

    async def test_analyze_model_path_success(self, client):
        mock_resp = _resp(200, {"architecture": "mistral"})
        mock_inst = _mock_httpx_client(post_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.analyze.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.post("/api/v1/analyze", json={"model_path": "/models/foo"})
        assert resp.status_code == 200
        assert resp.json()["architecture"] == "mistral"

    async def test_analyze_mlx_error_status(self, client):
        mock_resp = _resp(500, text="mlx internal error")
        mock_inst = _mock_httpx_client(post_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.analyze.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.post("/api/v1/analyze", json={"hf_repo": "foo/bar"})
        assert resp.status_code == 500

    async def test_analyze_mlx_connect_error(self, client):
        import httpx

        mock_inst = _mock_httpx_client()
        mock_inst.post.side_effect = httpx.ConnectError("no connection")
        with patch("fusion_model_hub.server.routers.analyze.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.post("/api/v1/analyze", json={"hf_repo": "foo/bar"})
        assert resp.status_code == 503

    async def test_analyze_mlx_generic_exception(self, client):
        mock_inst = _mock_httpx_client()
        mock_inst.post.side_effect = ValueError("boom")
        with patch("fusion_model_hub.server.routers.analyze.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.post("/api/v1/analyze", json={"hf_repo": "foo/bar"})
        assert resp.status_code == 500


class TestBenchmarksList:
    async def test_list_benchmarks_success(self, client):
        mock_resp = _resp(200, {"items": [{"model_id": "m1", "score": 0.9}]})
        mock_inst = _mock_httpx_client(get_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/benchmarks", params={"chip": "m1", "quant": "4bit"})
        assert resp.status_code == 200
        assert resp.json()["items"][0]["model_id"] == "m1"
        mock_inst.get.assert_awaited()
        call_args = mock_inst.get.call_args
        assert call_args.kwargs["params"]["chip"] == "m1"

    async def test_list_benchmarks_mlx_error_status(self, client):
        mock_resp = _resp(404, text="no benchmarks")
        mock_inst = _mock_httpx_client(get_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/benchmarks")
        assert resp.status_code == 404

    async def test_list_benchmarks_connect_error(self, client):
        import httpx

        mock_inst = _mock_httpx_client()
        mock_inst.get.side_effect = httpx.ConnectError("down")
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/benchmarks")
        assert resp.status_code == 503

    async def test_list_benchmarks_generic_exception(self, client):
        mock_inst = _mock_httpx_client()
        mock_inst.get.side_effect = RuntimeError("weird")
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/benchmarks")
        assert resp.status_code == 500


class TestBenchmarksCompare:
    async def test_compare_requires_model_ids(self, client):
        resp = await client.get("/api/v1/benchmarks/compare")
        assert resp.status_code == 400

    async def test_compare_empty_string(self, client):
        resp = await client.get("/api/v1/benchmarks/compare", params={"model_ids": ", ,"})
        assert resp.status_code == 400

    async def test_compare_success_list_and_dict(self, client):
        list_resp = _resp(200, [{"score": 0.9}])
        dict_resp = _resp(200, {"accuracy": 0.8})
        mock_inst = _mock_httpx_client(get_side=[list_resp, dict_resp])
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get(
                "/api/v1/benchmarks/compare",
                params={"model_ids": "m1,m2", "chip": "m1"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_ids"] == ["m1", "m2"]
        assert len(data["items"]) == 2
        assert data["items"][0]["model_id"] == "m1"
        assert data["items"][1]["model_id"] == "m2"

    async def test_compare_non_200_returns_error_entry(self, client):
        err_resp = _resp(500, text="bad")
        mock_inst = _mock_httpx_client(get_side=[err_resp])
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/benchmarks/compare", params={"model_ids": "m1"})
        assert resp.status_code == 200
        assert resp.json()["items"][0]["error"] == "status 500"

    async def test_compare_connect_error(self, client):
        import httpx

        mock_inst = _mock_httpx_client()
        mock_inst.get.side_effect = httpx.ConnectError("down")
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/benchmarks/compare", params={"model_ids": "m1"})
        assert resp.status_code == 503

    async def test_compare_generic_exception(self, client):
        mock_inst = _mock_httpx_client()
        mock_inst.get.side_effect = RuntimeError("x")
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/benchmarks/compare", params={"model_ids": "m1"})
        assert resp.status_code == 500


class TestBenchmarksGetOne:
    async def test_get_benchmark_success(self, client):
        mock_resp = _resp(200, {"model_id": "m1", "score": 0.9})
        mock_inst = _mock_httpx_client(get_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/benchmarks/m1", params={"chip": "m1", "quant": "4bit"})
        assert resp.status_code == 200
        assert resp.json()["model_id"] == "m1"

    async def test_get_benchmark_404(self, client):
        mock_resp = _resp(404, text="not found")
        mock_inst = _mock_httpx_client(get_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/benchmarks/m1")
        assert resp.status_code == 404

    async def test_get_benchmark_other_error(self, client):
        mock_resp = _resp(503, text="unavailable")
        mock_inst = _mock_httpx_client(get_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/benchmarks/m1")
        assert resp.status_code == 503

    async def test_get_benchmark_connect_error(self, client):
        import httpx

        mock_inst = _mock_httpx_client()
        mock_inst.get.side_effect = httpx.ConnectError("down")
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/benchmarks/m1")
        assert resp.status_code == 503

    async def test_get_benchmark_generic_exception(self, client):
        mock_inst = _mock_httpx_client()
        mock_inst.get.side_effect = RuntimeError("x")
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/benchmarks/m1")
        assert resp.status_code == 500


class TestBenchmarksTrigger:
    async def test_trigger_success_202(self, client):
        mock_resp = _resp(202, {"task_id": "t1"})
        mock_inst = _mock_httpx_client(post_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.post(
                "/api/v1/benchmarks/trigger",
                json={
                    "model_id": "m1",
                    "suite": "full",
                    "callback_url": "http://cb.example.com",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "submitted"
        assert data["detail"]["task_id"] == "t1"

    async def test_trigger_template_alias(self, client):
        mock_resp = _resp(201, {"task_id": "t2"})
        mock_inst = _mock_httpx_client(post_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.post(
                "/api/v1/benchmarks/trigger",
                json={
                    "model_id": "m1",
                    "template": "standard",
                },
            )
        assert resp.status_code == 200
        sent = mock_inst.post.call_args.kwargs["json"]
        assert sent["suite"] == "standard"

    async def test_trigger_bench_error_status(self, client):
        mock_resp = _resp(400, text="bad request")
        mock_inst = _mock_httpx_client(post_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.post("/api/v1/benchmarks/trigger", json={"model_id": "m1"})
        assert resp.status_code == 400

    async def test_trigger_connect_error(self, client):
        import httpx

        mock_inst = _mock_httpx_client()
        mock_inst.post.side_effect = httpx.ConnectError("down")
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.post("/api/v1/benchmarks/trigger", json={"model_id": "m1"})
        assert resp.status_code == 503

    async def test_trigger_generic_exception(self, client):
        mock_inst = _mock_httpx_client()
        mock_inst.post.side_effect = RuntimeError("x")
        with patch("fusion_model_hub.server.routers.benchmarks.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.post("/api/v1/benchmarks/trigger", json={"model_id": "m1"})
        assert resp.status_code == 500


class TestSystemHealth:
    async def test_health_mlx_available(self, client):
        mock_resp = _resp(200, {"version": "0.1", "models": []})
        mock_inst = _mock_httpx_client(get_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.system.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["mlxConnected"] is True
        assert data["mlx"]["status"] == "available"
        assert data["uptime"] != ""
        assert "total" in data["storage"]

    async def test_health_mlx_offline(self, client):
        import httpx

        mock_inst = _mock_httpx_client()
        mock_inst.get.side_effect = httpx.ConnectError("down")
        with patch("fusion_model_hub.server.routers.system.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["mlxConnected"] is False
        assert data["mlx"]["status"] == "offline"

    async def test_health_mlx_error_status(self, client):
        mock_resp = _resp(500, text="err")
        mock_inst = _mock_httpx_client(get_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.system.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200
        assert resp.json()["mlx"]["status"] == "error_500"

    async def test_health_mlx_non_json_response(self, client):
        mock_resp = _resp(200, text="plain text body")
        mock_resp.json.side_effect = ValueError("not json")
        mock_inst = _mock_httpx_client(get_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.system.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mlx"]["status"] == "available"
        assert data["mlx"]["info"]["raw"] == "plain text body"

    async def test_health_mlx_generic_exception(self, client):
        mock_inst = _mock_httpx_client()
        mock_inst.get.side_effect = RuntimeError("boom")
        with patch("fusion_model_hub.server.routers.system.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200
        assert resp.json()["mlx"]["status"] == "error"

    async def test_health_model_count_query(self, client):
        await _create_model(client, "health-model-1")
        mock_resp = _resp(200, {"version": "0.1"})
        mock_inst = _mock_httpx_client(get_resp=mock_resp)
        with patch("fusion_model_hub.server.routers.system.httpx.AsyncClient", return_value=mock_inst):
            resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200
        assert resp.json()["model_count"] == 1


class TestSystemStorageAndAudit:
    async def test_storage_stats(self, client):
        resp = await client.get("/api/v1/system/storage")
        assert resp.status_code == 200
        data = resp.json()
        assert "path" in data
        assert "total_size_gb" in data

    async def test_audit_logs_empty(self, client):
        resp = await client.get("/api/v1/system/audit")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    async def test_audit_logs_with_entries(self, client):
        from fusion_model_hub.db.crud import create_audit_log
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            await create_audit_log(
                session,
                api_key_id="k1",
                action="model.create",
                resource_type="model",
                resource_id="r1",
                detail="created",
            )
            await create_audit_log(
                session,
                api_key_id="k1",
                action="model.update",
                resource_type="model",
                resource_id="r2",
                detail="updated",
            )
        resp = await client.get("/api/v1/system/audit")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2

    async def test_audit_logs_filter_by_action(self, client):
        from fusion_model_hub.db.crud import create_audit_log
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            await create_audit_log(
                session,
                api_key_id="k1",
                action="model.create",
                resource_type="model",
                resource_id="r1",
                detail="c",
            )
            await create_audit_log(
                session,
                api_key_id="k1",
                action="model.delete",
                resource_type="model",
                resource_id="r2",
                detail="d",
            )
        resp = await client.get("/api/v1/system/audit", params={"action": "model.create"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["action"] == "model.create"


class TestSystemExport:
    async def test_export_empty(self, client):
        resp = await client.get("/api/v1/system/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "1.0"
        assert data["models"] == []
        assert data["tenants"] == []
        assert data["webhooks"] == []

    async def test_export_with_model(self, client):
        m = await _create_model(client, "exp-model-1")
        resp = await client.get("/api/v1/system/export", params={"models": m["id"]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "exp-model-1"


class TestSystemImport:
    async def test_import_creates_model(self, client):
        payload = {
            "models": [
                {
                    "name": "imp-model-1",
                    "description": "imported",
                    "model_type": "llm",
                    "architecture": "llama",
                    "tags": [{"key": "k", "value": "v"}],
                },
            ],
        }
        resp = await client.post("/api/v1/system/import", json=payload)
        assert resp.status_code == 200
        assert resp.json()["imported"] == 1
        listing = await client.get("/api/v1/models")
        names = [m["name"] for m in listing.json()["items"]]
        assert "imp-model-1" in names

    async def test_import_creates_tenant(self, client):
        payload = {"tenants": [{"name": "tenant-a", "display_name": "A"}]}
        resp = await client.post("/api/v1/system/import", json=payload)
        assert resp.status_code == 200
        assert resp.json()["imported"] == 1

    async def test_import_creates_webhook(self, client):
        payload = {
            "webhooks": [
                {"name": "wh-1", "url": "https://example.com/hook", "events": "model.published"},
            ],
        }
        resp = await client.post("/api/v1/system/import", json=payload)
        assert resp.status_code == 200
        assert resp.json()["imported"] == 1

    async def test_import_invalid_webhook_url(self, client):
        payload = {
            "webhooks": [
                {"name": "wh-bad", "url": "http://127.0.0.1:11434/internal", "events": "x"},
            ],
        }
        resp = await client.post("/api/v1/system/import", json=payload)
        assert resp.status_code in (400, 422)

    async def test_import_duplicate_model_skipped(self, client):
        await _create_model(client, "dup-model-1")
        payload = {"models": [{"name": "dup-model-1", "model_type": "llm"}]}
        resp = await client.post("/api/v1/system/import", json=payload)
        assert resp.status_code == 200
        assert resp.json()["imported"] == 0

    async def test_import_bad_model_type_falls_back(self, client):
        payload = {"models": [{"name": "imp-fallback", "model_type": "not-a-real-type"}]}
        resp = await client.post("/api/v1/system/import", json=payload)
        assert resp.status_code == 200
        assert resp.json()["imported"] == 1

    async def test_import_forbidden_when_auth_enabled_non_admin(self, client):
        from fusion_model_hub.server.auth import set_auth_enabled

        set_auth_enabled(True)
        try:
            admin_resp = await client.post(
                "/api/v1/auth/keys",
                json={
                    "name": "boot-admin",
                    "role": "admin",
                    "permissions": "read,write",
                },
            )
            assert admin_resp.status_code == 201, admin_resp.text
            admin_key = admin_resp.json()["key"]
            dev_resp = await client.post(
                "/api/v1/auth/keys",
                json={"name": "dev", "role": "developer", "permissions": "read,write"},
                headers={"X-API-Key": admin_key},
            )
            assert dev_resp.status_code == 201, dev_resp.text
            dev_key = dev_resp.json()["key"]
            resp = await client.post(
                "/api/v1/system/import",
                json={"models": []},
                headers={"X-API-Key": dev_key},
            )
            assert resp.status_code == 403
        finally:
            set_auth_enabled(False)


class TestSystemScanDuplicates:
    async def test_scan_no_duplicates(self, client):
        m = await _create_model(client, "dup-scan-1")
        await _create_published_version(client, m["id"])
        resp = await client.post("/api/v1/system/scan-duplicates")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_groups"] == 0
        assert data["duplicate_groups"] == []

    async def test_scan_finds_duplicates_by_hash(self, client):
        from fusion_model_hub.db.crud import update_version
        from fusion_model_hub.server.deps import get_session_factory

        m1 = await _create_model(client, "dup-a")
        v1 = await _create_published_version(client, m1["id"])
        m2 = await _create_model(client, "dup-b")
        v2 = await _create_published_version(client, m2["id"])
        sf = get_session_factory()
        async with sf() as session:
            await update_version(session, v1["id"], file_hash="abc123", file_size=1024)
            await update_version(session, v2["id"], file_hash="abc123", file_size=1024)
        resp = await client.post("/api/v1/system/scan-duplicates")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_groups"] == 1
        assert len(data["duplicate_groups"][0]) == 2


class TestSystemCleanup:
    async def test_cleanup_no_retired(self, client):
        m = await _create_model(client, "clean-model-1")
        await _create_published_version(client, m["id"])
        resp = await client.post("/api/v1/system/cleanup")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["candidates"] == []

    async def test_cleanup_finds_retired_with_file(self, client):
        from fusion_model_hub.db.crud import update_version, update_version_status
        from fusion_model_hub.db.models import VersionStatus
        from fusion_model_hub.server.deps import get_session_factory

        m = await _create_model(client, "clean-model-2")
        v = await _create_published_version(client, m["id"])
        sf = get_session_factory()
        async with sf() as session:
            await update_version(session, v["id"], file_path="/data/model.bin", file_size=2048)
            await update_version_status(session, v["id"], VersionStatus.DEPRECATED)
            await update_version_status(session, v["id"], VersionStatus.RETIRED)
        resp = await client.post("/api/v1/system/cleanup")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["candidates"][0]["file_path"] == "/data/model.bin"
        assert data["candidates"][0]["status"] == "retired"


class TestSystemCleanupRealDelete:
    # #6: dry_run=False actually deletes retired version files + clears the DB row.

    async def _seed_retired_with_real_file(self, client, settings, name="real-clean-1"):
        from pathlib import Path

        from fusion_model_hub.db.crud import update_version, update_version_status
        from fusion_model_hub.db.models import VersionStatus
        from fusion_model_hub.server.deps import get_session_factory

        m = await _create_model(client, name)
        v = await _create_published_version(client, m["id"])
        # Create the on-disk version dir + a fake weight file LocalStore will delete.
        version_dir = Path(settings.data_dir) / "models" / m["id"] / v["version"]
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "weights.bin").write_bytes(b"0" * 4096)
        sf = get_session_factory()
        async with sf() as session:
            await update_version(
                session,
                v["id"],
                file_path=str(version_dir / "weights.bin"),
                file_size=4096,
            )
            await update_version_status(session, v["id"], VersionStatus.RETIRED)
        return m, v, version_dir

    async def test_cleanup_dry_run_default_does_not_delete(self, client, settings):
        m, v, version_dir = await self._seed_retired_with_real_file(client, settings)
        resp = await client.post("/api/v1/system/cleanup")  # dry_run defaults True
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True
        assert data["total"] == 1
        assert "deleted" not in data
        # File + dir still present (dry run = no mutation).
        assert version_dir.exists()
        assert (version_dir / "weights.bin").exists()

    async def test_cleanup_real_delete_clears_files_and_db(self, client, settings):
        m, v, version_dir = await self._seed_retired_with_real_file(client, settings)
        resp = await client.post("/api/v1/system/cleanup?dry_run=false")
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is False
        assert data["deleted"] == 1
        assert data["failed"] == 0
        # On-disk version dir gone.
        assert not version_dir.exists()
        # DB row cleared but still exists (provenance kept).
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            row = await crud.get_version(session, v["id"])
            assert row is not None
            assert row.file_path == ""
            assert row.file_size == 0
            assert row.status.value == "retired"


class TestSystemScanDuplicatesRealDelete:
    # #6: dry_run=False reclaims redundant duplicate weights (keeps oldest).

    async def _seed_dup_pair(self, client, settings, hash_val="dup-hash-9"):
        from pathlib import Path

        from fusion_model_hub.db.crud import update_version
        from fusion_model_hub.server.deps import get_session_factory

        m1 = await _create_model(client, "dedup-a")
        v1 = await _create_published_version(client, m1["id"])
        m2 = await _create_model(client, "dedup-b")
        v2 = await _create_published_version(client, m2["id"], version="2.0.0")
        # Real on-disk weight files for both versions.
        dir1 = Path(settings.data_dir) / "models" / m1["id"] / v1["version"]
        dir2 = Path(settings.data_dir) / "models" / m2["id"] / v2["version"]
        dir1.mkdir(parents=True, exist_ok=True)
        dir2.mkdir(parents=True, exist_ok=True)
        (dir1 / "weights.bin").write_bytes(b"0" * 4096)
        (dir2 / "weights.bin").write_bytes(b"0" * 4096)
        sf = get_session_factory()
        async with sf() as session:
            await update_version(
                session,
                v1["id"],
                file_hash=hash_val,
                file_size=4096,
                file_path=str(dir1 / "weights.bin"),
            )
            await update_version(
                session,
                v2["id"],
                file_hash=hash_val,
                file_size=4096,
                file_path=str(dir2 / "weights.bin"),
            )
        return (m1, v1, dir1), (m2, v2, dir2)

    async def test_scan_dry_run_default_identifies_only(self, client, settings):
        a, b = await self._seed_dup_pair(client, settings)
        resp = await client.post("/api/v1/system/scan-duplicates")
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True
        assert data["total_groups"] == 1
        assert "reclaimed" not in data
        # Both weight dirs still present.
        assert a[2].exists()
        assert b[2].exists()

    async def test_scan_real_delete_reclaims_redundant_keeps_oldest(self, client, settings):
        a, b = await self._seed_dup_pair(client, settings)
        # list_models is created_at-desc; the endpoint keeps the OLDEST version
        # (last in the group) and retires the rest. Created near-simultaneously,
        # so we assert exactly one dir survives + one is reclaimed.
        resp = await client.post("/api/v1/system/scan-duplicates?dry_run=false")
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is False
        assert data["total_groups"] == 1
        assert data["reclaimed"] == 1
        assert data["failed"] == 0
        # Exactly one weight dir reclaimed, one kept.
        remaining = [a[2].exists(), b[2].exists()]
        assert remaining.count(True) == 1
        assert remaining.count(False) == 1
        # Reclaimed version's DB row is RETIRED with cleared file_path.
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            r1 = await crud.get_version(session, a[1]["id"])
            r2 = await crud.get_version(session, b[1]["id"])
            retired = [r for r in (r1, r2) if r.status.value == "retired"]
            kept = [r for r in (r1, r2) if r.status.value != "retired"]
            assert len(retired) == 1
            assert len(kept) == 1
            assert retired[0].file_path == ""
            assert retired[0].file_size == 0


class TestSystemHardware:
    async def test_hardware_info_returns_fields(self, client):
        resp = await client.get("/api/v1/system/hardware")
        assert resp.status_code == 200
        data = resp.json()
        for key in (
            "gpu_name",
            "gpu_memory_total_mb",
            "gpu_memory_used_mb",
            "gpu_utilization",
            "cpu_cores",
            "memory_total_gb",
            "memory_used_gb",
        ):
            assert key in data
        assert isinstance(data["cpu_cores"], int)
        assert data["cpu_cores"] >= 0


class TestQuantizePresets:
    async def test_list_presets(self, client):
        from fusion_model_hub.server.routers.quantize_presets import list_presets

        result = await list_presets()
        names = [p["name"] for p in result["presets"]]
        assert "chat" in names
        assert "code" in names
        assert "embedding" in names

    async def test_apply_preset_unknown(self, client):
        from fastapi import HTTPException

        from fusion_model_hub.server.deps import get_session_factory
        from fusion_model_hub.server.routers.quantize_presets import apply_preset

        sf = get_session_factory()
        async with sf() as session:
            with pytest.raises(HTTPException) as exc:
                await apply_preset("no-such", type("B", (), {"source_version_id": "v1"})(), session)
            assert exc.value.status_code == 404

    async def test_apply_preset_version_not_found(self, client):
        from fastapi import HTTPException

        from fusion_model_hub.server.deps import get_session_factory
        from fusion_model_hub.server.routers.quantize_presets import apply_preset

        sf = get_session_factory()
        async with sf() as session:
            with pytest.raises(HTTPException) as exc:
                await apply_preset("chat", type("B", (), {"source_version_id": "no-such-version"})(), session)
            assert exc.value.status_code == 404

    async def test_apply_preset_chat_success(self, client):
        from fusion_model_hub.server.deps import get_session_factory
        from fusion_model_hub.server.routers.quantize_presets import apply_preset

        m = await _create_model(client, "preset-model-1")
        v = await _create_published_version(client, m["id"])
        sf = get_session_factory()
        async with sf() as session:
            with patch(
                "fusion_model_hub.server.routers.quantize_presets.submit_quantize",
                new=AsyncMock(return_value="task-123"),
            ):
                result = await apply_preset(
                    "chat",
                    type("B", (), {"source_version_id": v["id"]})(),
                    session,
                )
        assert result["task_id"] == "task-123"
        assert result["preset"] == "chat"
        assert result["status"] == "submitted"
        assert result["quant_bits"] == 4

    async def test_apply_preset_embedding_success(self, client):
        from fusion_model_hub.server.deps import get_session_factory
        from fusion_model_hub.server.routers.quantize_presets import apply_preset

        m = await _create_model(client, "preset-model-2")
        v = await _create_published_version(client, m["id"])
        sf = get_session_factory()
        async with sf() as session:
            with patch(
                "fusion_model_hub.server.routers.quantize_presets.submit_quantize",
                new=AsyncMock(return_value="task-456"),
            ):
                result = await apply_preset(
                    "embedding",
                    type("B", (), {"source_version_id": v["id"]})(),
                    session,
                )
        assert result["quant_bits"] == 8

    async def test_apply_preset_submit_failure(self, client):
        from fastapi import HTTPException

        from fusion_model_hub.server.deps import get_session_factory
        from fusion_model_hub.server.routers.quantize_presets import apply_preset

        m = await _create_model(client, "preset-model-3")
        v = await _create_published_version(client, m["id"])
        sf = get_session_factory()
        async with sf() as session:
            with patch(
                "fusion_model_hub.server.routers.quantize_presets.submit_quantize",
                new=AsyncMock(side_effect=RuntimeError("mlx down")),
            ):
                with pytest.raises(HTTPException) as exc:
                    await apply_preset(
                        "chat",
                        type("B", (), {"source_version_id": v["id"]})(),
                        session,
                    )
            assert exc.value.status_code == 500
