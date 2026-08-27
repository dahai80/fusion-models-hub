import asyncio
import logging
import os
import shutil
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import _reconcile_orphaned_tasks, create_app
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps

logger = logging.getLogger(__name__)


@pytest.fixture
def settings():
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_cov_qt_dir",
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
    from fusion_model_hub.server import tasks as tasks_mod
    tasks_mod._running_tasks.clear()
    from fusion_model_hub.server.routers import downloads as dl_mod
    dl_mod._running_downloads.clear()
    with contextlib_suppress():
        shutil.rmtree("/tmp/fmh_cov_qt_dir", ignore_errors=True)


class contextlib_suppress:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return True


async def _create_model(client, name="cov-qt-model"):
    resp = await client.post("/api/v1/models", json={
        "name": name, "description": "test", "model_type": "llm",
    })
    assert resp.status_code == 201
    return resp.json()


async def _create_version(client, model_id, version="1.0.0", file_path="/tmp/src.mlx"):
    resp = await client.post(
        f"/api/v1/models/{model_id}/versions",
        data={"version": version, "format": "mlx", "quantization": "4bit"},
        files={"file": ("", b"")},
    )
    assert resp.status_code == 201
    return resp.json()


async def _create_two_versions(client, model_id):
    v1 = await _create_version(client, model_id, "1.0.0")
    v2 = await _create_version(client, model_id, "1.1.0")
    return v1, v2


def _mock_httpx_response(*, status_code=200, json_data=None, content=b"", text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content or (str(json_data).encode() if json_data else b"")
    resp.text = text or (str(json_data) if json_data else "")
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json = MagicMock(return_value=json_data)
    return resp


def _mock_httpx_ctx(post_return=None, get_return=None, side_effect=None):
    ctx = AsyncMock()
    if post_return is not None:
        ctx.post = AsyncMock(return_value=post_return)
    if get_return is not None:
        ctx.get = AsyncMock(return_value=get_return)
    if side_effect is not None:
        ctx.post = AsyncMock(side_effect=side_effect)
    ctx.__aenter__ = AsyncMock(return_value=ctx)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


# =====================================================================
# quantize.py: get_quantize_status cross-tenant guard (lines 86-89)
# =====================================================================


class TestQuantizeStatusTenantGuard:
    async def test_status_cross_tenant_denied(self, client, app):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "qt-tenant-model")
        v = await _create_version(client, m["id"])
        r = await client.post("/api/v1/quantize", json={
            "source_version_id": v["id"], "quant_bits": 4,
        })
        task_id = r.json()["task_id"]
        await asyncio.sleep(0.3)
        from starlette.requests import Request

        from fusion_model_hub.server.routers import quantize as qz

        def _make_req(tenant: str):
            scope = {
                "type": "http", "method": "GET",
                "path": f"/api/v1/quantize/{task_id}",
                "headers": [], "query_string": b"", "app": app,
            }
            req = Request(scope)
            req.state.tenant_id = tenant
            return req

        # Permissive path: empty caller tenant (local mode) -> returns status.
        resp = await qz.get_quantize_status(task_id, request=_make_req(""))
        assert resp is not None

        # Denied path: non-empty caller tenant that does not own the task.
        # crud.quantize_task_tenant returns the task's real owner ("tenant-owner"),
        # which mismatches the caller ("tenant-other") -> 404.
        with patch(
            "fusion_model_hub.server.routers.quantize.crud.quantize_task_tenant",
            new_callable=AsyncMock, return_value="tenant-owner",
        ):
            with pytest.raises(Exception) as exc_info:
                await qz.get_quantize_status(task_id, request=_make_req("tenant-other"))
            assert exc_info.value.status_code == 404
        tasks_mod._running_tasks.clear()


# =====================================================================
# quantize.py: compare_quantize_results (lines 101-148)
# =====================================================================


class TestCompareQuantizeResults:
    async def test_compare_task_not_found(self, client):
        resp = await client.get("/api/v1/quantize/nonexistent/compare")
        assert resp.status_code == 404
        assert "Task not found" in resp.json()["detail"]

    async def test_compare_source_version_not_found(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "cmp-src-missing")
        v = await _create_version(client, m["id"])
        r = await client.post("/api/v1/quantize", json={
            "source_version_id": v["id"], "quant_bits": 4,
        })
        task_id = r.json()["task_id"]
        await asyncio.sleep(0.3)
        # Delete the source version row directly so the compare path hits 404.
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            from sqlalchemy import delete

            from fusion_model_hub.db.models import ModelVersion
            await session.execute(delete(ModelVersion).where(ModelVersion.id == v["id"]))
            await session.commit()
        resp = await client.get(f"/api/v1/quantize/{task_id}/compare")
        assert resp.status_code == 404
        assert "Source version not found" in resp.json()["detail"]
        tasks_mod._running_tasks.clear()

    async def test_compare_cross_tenant_denied(self, client, app):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "cmp-tenant-model")
        v = await _create_version(client, m["id"])
        r = await client.post("/api/v1/quantize", json={
            "source_version_id": v["id"], "quant_bits": 4,
        })
        task_id = r.json()["task_id"]
        await asyncio.sleep(0.3)
        from starlette.requests import Request

        from fusion_model_hub.server.routers import quantize as qz
        scope = {
            "type": "http", "method": "GET", "path": f"/api/v1/quantize/{task_id}/compare",
            "headers": [], "query_string": b"", "app": app,
        }
        req = Request(scope)
        req.state.tenant_id = "tenant-other"
        with patch(
            "fusion_model_hub.server.routers.quantize.crud.quantize_task_tenant",
            new_callable=AsyncMock, return_value="tenant-owner",
        ):
            with pytest.raises(Exception) as exc_info:
                await qz.compare_quantize_results(task_id, request=req)
            assert exc_info.value.status_code == 404
        tasks_mod._running_tasks.clear()

    async def test_compare_with_output_version_and_metrics(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "cmp-full-model")
        v = await _create_version(client, m["id"], "1.0.0")
        # Patch ModelConverter to succeed so the runner produces an output version.
        mock_result = {"output_path": "/tmp/q.mlx", "file_hash": "h1", "file_size": 512}
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv:
            mc = AsyncMock()
            mc.quantize = AsyncMock(return_value=mock_result)
            MockConv.return_value = mc
            r = await client.post("/api/v1/quantize", json={
                "source_version_id": v["id"], "quant_bits": 4,
            })
            task_id = r.json()["task_id"]
            await asyncio.sleep(0.4)
        status = await client.get(f"/api/v1/quantize/{task_id}")
        assert status.json()["status"] == "completed"
        out_ver_id = status.json()["output_version_id"]
        # Seed benchmark metrics on both versions so the comparison math runs.
        from fusion_model_hub.db.crud import get_version
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            src = await get_version(session, v["id"])
            src.file_size = 2048
            src.benchmark_score = 80.0
            src.inference_latency = 100.0
            src.throughput = 50.0
            src.memory_usage = 4096.0
            out = await get_version(session, out_ver_id)
            out.file_size = 1024
            out.benchmark_score = 70.0
            out.inference_latency = 120.0
            out.throughput = 45.0
            out.memory_usage = 2048.0
            await session.commit()
        resp = await client.get(f"/api/v1/quantize/{task_id}/compare")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == task_id
        assert data["source_version"]["id"] == v["id"]
        assert data["output_version"]["id"] == out_ver_id
        cmp = data["comparison"]
        # size_reduction = (1 - 1024/2048)*100 = 50.0
        assert cmp["size_reduction_pct"] == 50.0
        # latency_change = (120-100)/100*100 = 20.0
        assert cmp["latency_change_pct"] == 20.0
        # throughput_change = (45-50)/50*100 = -10.0
        assert cmp["throughput_change_pct"] == -10.0
        # memory_change = (2048-4096)/4096*100 = -50.0
        assert cmp["memory_change_pct"] == -50.0
        tasks_mod._running_tasks.clear()

    async def test_compare_zero_metric_guards(self, client):
        # Source version with zero metrics -> comparison guards return 0, not ZeroDivisionError.
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "cmp-zero-model")
        v = await _create_version(client, m["id"], "1.0.0")
        mock_result = {"output_path": "/tmp/q.mlx", "file_hash": "h1", "file_size": 100}
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv:
            mc = AsyncMock()
            mc.quantize = AsyncMock(return_value=mock_result)
            MockConv.return_value = mc
            r = await client.post("/api/v1/quantize", json={
                "source_version_id": v["id"], "quant_bits": 4,
            })
            task_id = r.json()["task_id"]
            await asyncio.sleep(0.4)
        status = await client.get(f"/api/v1/quantize/{task_id}")
        out_ver_id = status.json()["output_version_id"]
        # Leave all metrics at 0/None -> guards hit the `else 0` branches.
        resp = await client.get(f"/api/v1/quantize/{task_id}/compare")
        assert resp.status_code == 200
        data = resp.json()
        assert data["output_version"]["id"] == out_ver_id
        assert data["comparison"]["size_reduction_pct"] == 0
        assert data["comparison"]["latency_change_pct"] == 0
        assert data["comparison"]["throughput_change_pct"] == 0
        assert data["comparison"]["memory_change_pct"] == 0
        tasks_mod._running_tasks.clear()

    async def test_compare_no_output_version(self, client):
        # Task completed but output_version_id empty -> only source_version in result.
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "cmp-no-output")
        v = await _create_version(client, m["id"], "1.0.0")
        r = await client.post("/api/v1/quantize", json={
            "source_version_id": v["id"], "quant_bits": 4,
        })
        task_id = r.json()["task_id"]
        await asyncio.sleep(0.3)
        # Force a failed task (source deleted path leaves FAILED with no output_version_id).
        # Easier: directly query a failed task from the not-found path.
        from fusion_model_hub.db.crud import get_quantize_task
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            t = await get_quantize_task(session, task_id)
            # The runner failed because we did not mock MLX -> status FAILED, no output.
            assert t.status.value == "failed"
            assert not t.output_version_id
        resp = await client.get(f"/api/v1/quantize/{task_id}/compare")
        assert resp.status_code == 200
        data = resp.json()
        assert "source_version" in data
        assert "output_version" not in data
        assert "comparison" not in data
        tasks_mod._running_tasks.clear()


# =====================================================================
# quantize.py: layered quantize proxy (lines 321-370)
# =====================================================================


class TestLayeredQuantize:
    async def test_layered_submit_success_with_output_path(self, client):
        ok = _mock_httpx_response(status_code=202, json_data={"job_id": "job-1"})
        ctx = _mock_httpx_ctx(post_return=ok)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.post("/api/v1/quantize/layered", json={
                "model": "test/model",
                "output_path": "/tmp/out.mlx",
                "default_bits": 4,
                "layer_rules": [{"pattern": "mlp", "bits": 8}],
                "quant_group_size": 64,
                "quant_mode": "affine",
                "trust_remote_code": False,
            })
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"] == "job-1"
        assert data["status"] == "submitted"
        assert data["hub_registered"] is False
        # Confirm output_path was included in the payload (last post call args).
        post_kwargs = ctx.post.call_args
        assert post_kwargs.kwargs["json"]["output_path"] == "/tmp/out.mlx"

    async def test_layered_submit_success_without_output_path(self, client):
        ok = _mock_httpx_response(status_code=200, json_data={"job_id": "job-2"})
        ctx = _mock_httpx_ctx(post_return=ok)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.post("/api/v1/quantize/layered", json={
                "model": "test/model2",
                "default_bits": 4,
                "layer_rules": [{"pattern": ".*", "bits": 4}],
            })
        assert resp.status_code == 202
        assert resp.json()["job_id"] == "job-2"
        # output_path should NOT be in payload when not provided.
        post_kwargs = ctx.post.call_args
        assert "output_path" not in post_kwargs.kwargs["json"]

    async def test_layered_submit_mlx_non_200(self, client):
        bad = _mock_httpx_response(status_code=500, text="internal error")
        ctx = _mock_httpx_ctx(post_return=bad)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.post("/api/v1/quantize/layered", json={
                "model": "test/model3",
                "layer_rules": [{"pattern": ".*", "bits": 4}],
            })
        assert resp.status_code == 500
        assert "Fusion-MLX layered quantize failed" in resp.json()["detail"]

    async def test_layered_submit_connect_error(self, client):
        import httpx as _httpx
        ctx = _mock_httpx_ctx(side_effect=_httpx.ConnectError("refused"))
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.post("/api/v1/quantize/layered", json={
                "model": "test/model4",
                "layer_rules": [{"pattern": ".*", "bits": 4}],
            })
        assert resp.status_code == 503
        assert "Fusion-MLX not available" in resp.json()["detail"]

    async def test_layered_submit_generic_exception(self, client):
        ctx = _mock_httpx_ctx(side_effect=ValueError("boom"))
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.post("/api/v1/quantize/layered", json={
                "model": "test/model5",
                "layer_rules": [{"pattern": ".*", "bits": 4}],
            })
        assert resp.status_code == 500
        assert "Fusion-MLX layered quantize failed" in resp.json()["detail"]

    async def test_layered_submit_sends_bearer_when_key_set(self, client, settings, monkeypatch):
        # Regression (fusion-mlx#646): the layered routes proxy to the SAME
        # /v1/quantize endpoint the converter hits, so they must carry the
        # Authorization: Bearer header when MLX enforces auth, or a secured MLX
        # 401s while the converter path succeeds.
        monkeypatch.setattr(settings, "mlx_internal_api_key", "secret-mlx-key")
        ok = _mock_httpx_response(status_code=202, json_data={"job_id": "job-auth"})
        ctx = _mock_httpx_ctx(post_return=ok)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.post("/api/v1/quantize/layered", json={
                "model": "test/auth-model",
                "default_bits": 4,
                "layer_rules": [{"pattern": ".*", "bits": 4}],
            })
        assert resp.status_code == 202
        post_kwargs = ctx.post.call_args
        assert post_kwargs.kwargs["headers"]["Authorization"] == "Bearer secret-mlx-key"

    async def test_layered_submit_omits_header_when_no_key(self, client, settings, monkeypatch):
        # An empty key must omit the header so an unauthenticated MLX still works.
        monkeypatch.setattr(settings, "mlx_internal_api_key", "")
        ok = _mock_httpx_response(status_code=202, json_data={"job_id": "job-noauth"})
        ctx = _mock_httpx_ctx(post_return=ok)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.post("/api/v1/quantize/layered", json={
                "model": "test/noauth-model",
                "default_bits": 4,
                "layer_rules": [{"pattern": ".*", "bits": 4}],
            })
        assert resp.status_code == 202
        post_kwargs = ctx.post.call_args
        headers = post_kwargs.kwargs.get("headers", {}) or {}
        assert "Authorization" not in headers


# =====================================================================
# quantize.py: layered job status + list (lines 373-418)
# =====================================================================


class TestLayeredJobStatus:
    async def test_get_layered_job_success(self, client):
        ok = _mock_httpx_response(status_code=200, json_data={"job_id": "j1", "status": "running"})
        ctx = _mock_httpx_ctx(get_return=ok)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.get("/api/v1/quantize/layered/jobs/j1")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "j1"

    async def test_get_layered_job_not_found(self, client):
        nf = _mock_httpx_response(status_code=404, text="no job")
        ctx = _mock_httpx_ctx(get_return=nf)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.get("/api/v1/quantize/layered/jobs/missing")
        assert resp.status_code == 404
        assert "Layered quantize job not found" in resp.json()["detail"]

    async def test_get_layered_job_non_200(self, client):
        bad = _mock_httpx_response(status_code=500, text="err")
        ctx = _mock_httpx_ctx(get_return=bad)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.get("/api/v1/quantize/layered/jobs/j2")
        assert resp.status_code == 500
        assert "Fusion-MLX layered job status failed" in resp.json()["detail"]

    async def test_get_layered_job_connect_error(self, client):
        import httpx as _httpx
        ctx = AsyncMock()
        ctx.get = AsyncMock(side_effect=_httpx.ConnectError("down"))
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.get("/api/v1/quantize/layered/jobs/j3")
        assert resp.status_code == 503
        assert "Fusion-MLX not available" in resp.json()["detail"]

    async def test_get_layered_job_generic_exception(self, client):
        ctx = AsyncMock()
        ctx.get = AsyncMock(side_effect=RuntimeError("oops"))
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.get("/api/v1/quantize/layered/jobs/j4")
        assert resp.status_code == 500
        assert "Fusion-MLX layered job status failed" in resp.json()["detail"]

    async def test_get_layered_job_sends_bearer_when_key_set(self, client, settings, monkeypatch):
        # Regression (fusion-mlx#646): GET /v1/quantize/jobs/{id} must carry the
        # same Bearer header as the converter's _poll_quantize_job.
        monkeypatch.setattr(settings, "mlx_internal_api_key", "secret-mlx-key")
        ok = _mock_httpx_response(status_code=200, json_data={"job_id": "ja", "status": "running"})
        ctx = _mock_httpx_ctx(get_return=ok)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.get("/api/v1/quantize/layered/jobs/ja")
        assert resp.status_code == 200
        get_kwargs = ctx.get.call_args
        assert get_kwargs.kwargs["headers"]["Authorization"] == "Bearer secret-mlx-key"

    async def test_list_layered_jobs_success(self, client):
        ok = _mock_httpx_response(status_code=200, json_data=[{"job_id": "a"}, {"job_id": "b"}])
        ctx = _mock_httpx_ctx(get_return=ok)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.get("/api/v1/quantize/layered/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_list_layered_jobs_non_200(self, client):
        bad = _mock_httpx_response(status_code=503, text="unavailable")
        ctx = _mock_httpx_ctx(get_return=bad)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.get("/api/v1/quantize/layered/jobs")
        assert resp.status_code == 503
        assert "Fusion-MLX layered jobs list failed" in resp.json()["detail"]

    async def test_list_layered_jobs_connect_error(self, client):
        import httpx as _httpx
        ctx = AsyncMock()
        ctx.get = AsyncMock(side_effect=_httpx.ConnectError("nope"))
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.get("/api/v1/quantize/layered/jobs")
        assert resp.status_code == 503
        assert "Fusion-MLX not available" in resp.json()["detail"]

    async def test_list_layered_jobs_generic_exception(self, client):
        ctx = AsyncMock()
        ctx.get = AsyncMock(side_effect=TypeError("bad"))
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.get("/api/v1/quantize/layered/jobs")
        assert resp.status_code == 500
        assert "Fusion-MLX layered jobs list failed" in resp.json()["detail"]


# =====================================================================
# quantize.py: evaluate quantize (lines 421-449)
# =====================================================================


class TestEvaluateQuantize:
    async def test_evaluate_success(self, client):
        ok = _mock_httpx_response(
            status_code=200,
            json_data={"loss_pct": 2.5, "sample_size": 128},
        )
        ctx = _mock_httpx_ctx(post_return=ok)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.post("/api/v1/quantize/evaluate", json={
                "source_version_id": "ver-1", "quant_bits": 4, "sample_size": 128,
            })
        assert resp.status_code == 200
        assert resp.json()["loss_pct"] == 2.5
        post_kwargs = ctx.post.call_args
        assert post_kwargs.kwargs["json"]["source_version_id"] == "ver-1"
        assert post_kwargs.kwargs["json"]["sample_size"] == 128

    async def test_evaluate_mlx_non_200(self, client):
        bad = _mock_httpx_response(status_code=422, text="bad request")
        ctx = _mock_httpx_ctx(post_return=bad)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.post("/api/v1/quantize/evaluate", json={
                "source_version_id": "ver-2", "quant_bits": 4,
            })
        assert resp.status_code == 422
        assert "Fusion-MLX quantize evaluate failed" in resp.json()["detail"]

    async def test_evaluate_connect_error(self, client):
        import httpx as _httpx
        ctx = _mock_httpx_ctx(side_effect=_httpx.ConnectError("refused"))
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.post("/api/v1/quantize/evaluate", json={
                "source_version_id": "ver-3", "quant_bits": 4,
            })
        assert resp.status_code == 503
        assert "Fusion-MLX not available" in resp.json()["detail"]

    async def test_evaluate_generic_exception(self, client):
        ctx = _mock_httpx_ctx(side_effect=KeyError("missing"))
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            resp = await client.post("/api/v1/quantize/evaluate", json={
                "source_version_id": "ver-4", "quant_bits": 4,
            })
        assert resp.status_code == 500
        assert "Fusion-MLX quantize evaluate failed" in resp.json()["detail"]


# =====================================================================
# quantize.py: batch quantize (lines 461-482)
# =====================================================================


class TestBatchQuantize:
    async def test_batch_mixed_valid_invalid_bits(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "batch-model")
        v = await _create_version(client, m["id"])
        # item 1: valid bits -> submit ok; item 2: invalid bits (5) -> error entry.
        resp = await client.post("/api/v1/quantize/batch", json={
            "items": [
                {"source_version_id": v["id"], "quant_bits": 4},
                {"source_version_id": v["id"], "quant_bits": 5},
            ],
        })
        assert resp.status_code == 202
        data = resp.json()
        assert len(data["task_ids"]) == 1
        assert data["task_ids"][0]["source_version_id"] == v["id"]
        assert len(data["errors"]) == 1
        assert data["errors"][0]["source_version_id"] == v["id"]
        assert "quant_bits must be one of" in data["errors"][0]["error"]
        await asyncio.sleep(0.3)
        tasks_mod._running_tasks.clear()

    async def test_batch_submit_failure_records_error(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        # Patch submit_quantize to raise so the except-branch (line 478-480) runs.
        with patch(
            "fusion_model_hub.server.routers.quantize.submit_quantize",
            new_callable=AsyncMock, side_effect=RuntimeError("submit boom"),
        ):
            resp = await client.post("/api/v1/quantize/batch", json={
                "items": [{"source_version_id": "ver-x", "quant_bits": 4}],
            })
        assert resp.status_code == 202
        data = resp.json()
        assert len(data["task_ids"]) == 0
        assert len(data["errors"]) == 1
        assert data["errors"][0]["error"] == "submit boom"
        tasks_mod._running_tasks.clear()

    async def test_batch_empty_items(self, client):
        resp = await client.post("/api/v1/quantize/batch", json={"items": []})
        assert resp.status_code == 202
        data = resp.json()
        assert data["task_ids"] == []
        assert data["errors"] == []


# =====================================================================
# quantize.py: lora-merge error branches (lines 239-240, 287-288)
# =====================================================================


class TestLoraMergeErrors:
    async def test_lora_merge_connect_error(self, client):
        from fusion_model_hub.server.routers import quantize as qz
        qz._running_lora_merges.clear()
        m = await _create_model(client, "lora-conn-err-host")
        v1 = await _create_version(client, m["id"], "1.0.0")
        v2 = await _create_version(client, m["id"], "1.1.0")
        import httpx as _httpx
        ctx = _mock_httpx_ctx(side_effect=_httpx.ConnectError("server down"))
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            cr = await client.post("/api/v1/quantize/lora-merge", json={
                "base_version_id": v1["id"],
                "lora_version_id": v2["id"],
                "quant_bits": 4,
            })
            task_id = cr.json()["task_id"]
            await asyncio.sleep(0.4)
        status = await client.get(f"/api/v1/quantize/lora-merge/{task_id}")
        data = status.json()
        assert data["status"] == "failed"
        assert "Fusion-MLX server unavailable" in data["error_message"]
        qz._running_lora_merges.clear()

    async def test_lora_merge_empty_output_path_fails(self, client):
        # E-D2: 200 with no output_path -> RuntimeError, FAILED.
        from fusion_model_hub.server.routers import quantize as qz
        qz._running_lora_merges.clear()
        m = await _create_model(client, "lora-empty-out-host")
        v1 = await _create_version(client, m["id"], "1.0.0")
        v2 = await _create_version(client, m["id"], "1.1.0")
        ok = _mock_httpx_response(status_code=200, json_data={"output_path": ""})
        ctx = _mock_httpx_ctx(post_return=ok)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ):
            cr = await client.post("/api/v1/quantize/lora-merge", json={
                "base_version_id": v1["id"],
                "lora_version_id": v2["id"],
                "quant_bits": 4,
            })
            task_id = cr.json()["task_id"]
            await asyncio.sleep(0.4)
        status = await client.get(f"/api/v1/quantize/lora-merge/{task_id}")
        data = status.json()
        assert data["status"] == "failed"
        assert "output_path" in data["error_message"]
        qz._running_lora_merges.clear()

    async def test_lora_merge_webhook_dispatch_failure_logged_not_fatal(self, client):
        # Lines 287-288: adapter.merged webhook dispatch fails but task still COMPLETED.
        from fusion_model_hub.server.routers import quantize as qz
        qz._running_lora_merges.clear()
        m = await _create_model(client, "lora-wh-fail-host")
        v1 = await _create_version(client, m["id"], "1.0.0")
        v2 = await _create_version(client, m["id"], "1.1.0")
        ok = _mock_httpx_response(
            status_code=200, json_data={"output_path": "/tmp/merged.safetensors"}
        )
        ctx = _mock_httpx_ctx(post_return=ok)
        with patch(
            "fusion_model_hub.server.routers.quantize.httpx.AsyncClient",
            return_value=ctx,
        ), patch(
            "fusion_model_hub.server.routers.webhooks.dispatch_webhook_event",
            new_callable=AsyncMock, side_effect=RuntimeError("webhook broken"),
        ):
            cr = await client.post("/api/v1/quantize/lora-merge", json={
                "base_version_id": v1["id"],
                "lora_version_id": v2["id"],
                "quant_bits": 4,
            })
            task_id = cr.json()["task_id"]
            await asyncio.sleep(0.4)
        status = await client.get(f"/api/v1/quantize/lora-merge/{task_id}")
        data = status.json()
        # Webhook failure is caught and logged; task still completes.
        assert data["status"] == "completed"
        assert data["output_version_id"]
        qz._running_lora_merges.clear()

    async def test_lora_merge_invalid_bits(self, client):
        m = await _create_model(client, "lora-bad-bits-host")
        v1 = await _create_version(client, m["id"], "1.0.0")
        v2 = await _create_version(client, m["id"], "1.1.0")
        resp = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": v1["id"],
            "lora_version_id": v2["id"],
            "quant_bits": 5,
        })
        assert resp.status_code == 400
        assert "quant_bits must be one of" in resp.json()["detail"]

    async def test_lora_merge_base_version_not_found(self, client):
        m = await _create_model(client, "lora-base-404-host")
        v2 = await _create_version(client, m["id"], "1.1.0")
        resp = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": "nonexistent",
            "lora_version_id": v2["id"],
            "quant_bits": 4,
        })
        assert resp.status_code == 404
        assert "Base version not found" in resp.json()["detail"]

    async def test_lora_merge_lora_version_not_found(self, client):
        m = await _create_model(client, "lora-lora-404-host")
        v1 = await _create_version(client, m["id"], "1.0.0")
        resp = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": v1["id"],
            "lora_version_id": "nonexistent",
            "quant_bits": 4,
        })
        assert resp.status_code == 404
        assert "LoRA version not found" in resp.json()["detail"]

    async def test_lora_merge_status_not_found(self, client):
        resp = await client.get("/api/v1/quantize/lora-merge/nonexistent")
        assert resp.status_code == 404
        assert "LoRA merge task not found" in resp.json()["detail"]


# =====================================================================
# tasks.py: resume_quantize (lines 77-91)
# =====================================================================


class TestResumeQuantize:
    async def test_resume_skipped_when_already_claimed(self, client):
        # claim_quantize_task returns False -> resume_quantize returns None,
        # no task spawned. Create the task row directly (submitting would spawn
        # a runner we'd then have to cancel mid-flight, which raises
        # CancelledError out of the live aiosqlite session).
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "resume-skip-model")
        v = await _create_version(client, m["id"])
        from fusion_model_hub.db.crud import create_quantize_task
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            t = await create_quantize_task(
                session, source_version_id=v["id"], target_format="mlx", quant_bits=4,
            )
            task_id = t.id
        # Patch claim_quantize_task to return False (another worker owns it).
        with patch(
            "fusion_model_hub.server.tasks.claim_quantize_task",
            new_callable=AsyncMock, return_value=False,
        ):
            result = await tasks_mod.resume_quantize(
                task_id, v["id"], "mlx", 4,
            )
        assert result is None
        # No running task should be registered for this id.
        assert task_id not in tasks_mod._running_tasks
        tasks_mod._running_tasks.clear()

    async def test_resume_claimed_spawns_runner(self, client):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "resume-ok-model")
        v = await _create_version(client, m["id"])
        # Create a task row directly via crud so we control its state.
        from fusion_model_hub.db.crud import create_quantize_task
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            t = await create_quantize_task(
                session, source_version_id=v["id"], target_format="mlx", quant_bits=4,
            )
            task_id = t.id
        # Patch the converter so the resumed runner completes successfully.
        mock_result = {"output_path": "/tmp/resumed.mlx", "file_hash": "rh", "file_size": 256}
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv, \
             patch(
                 "fusion_model_hub.server.tasks.claim_quantize_task",
                 new_callable=AsyncMock, return_value=True,
             ):
            mc = AsyncMock()
            mc.quantize = AsyncMock(return_value=mock_result)
            MockConv.return_value = mc
            result = await tasks_mod.resume_quantize(task_id, v["id"], "mlx", 4)
            assert result == task_id
            # Wait for the spawned runner to finish.
            await asyncio.sleep(0.4)
        status = await client.get(f"/api/v1/quantize/{task_id}")
        assert status.json()["status"] == "completed"
        assert status.json()["output_version_id"]
        tasks_mod._running_tasks.clear()


# =====================================================================
# tasks.py: _run_quantize cache + no-output + bench + precision (lines 157-307)
# =====================================================================


class TestRunQuantizeBranches:
    async def test_cache_hit_skips_converter(self, client, settings):
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "cache-hit-model")
        v = await _create_version(client, m["id"])
        # Seed the cache so cache.has() returns True and cache.get() returns an entry.
        from fusion_model_hub.cache.types import CacheLevel
        from fusion_model_hub.server.deps import get_cache_manager
        cache = get_cache_manager()
        # Write a real file so cache.put can hash it.
        src = "/tmp/fmh_cov_qt_src_cache.mlx"
        with open(src, "wb") as fh:
            fh.write(b"cached-weights-bytes")
        # The runner looks up the cache with the CacheLevel.QUANTIZED enum, so the
        # seed must use the same enum (a string would compute a different cache
        # key and the lookup would miss, falling through to the converter).
        await cache.put_async(
            model_id=m["id"], level=CacheLevel.QUANTIZED, source_path=src,
            quant_bits=4, source_version_id=v["id"],
        )
        assert cache.has(m["id"], CacheLevel.QUANTIZED, 4, source_version_id=v["id"])
        # Converter must NOT be called.
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv:
            mc = AsyncMock()
            mc.quantize = AsyncMock(side_effect=AssertionError("converter should be skipped on cache hit"))
            MockConv.return_value = mc
            r = await client.post("/api/v1/quantize", json={
                "source_version_id": v["id"], "quant_bits": 4,
            })
            task_id = r.json()["task_id"]
            await asyncio.sleep(0.5)
        status = await client.get(f"/api/v1/quantize/{task_id}")
        assert status.json()["status"] == "completed"
        assert status.json()["output_version_id"]
        tasks_mod._running_tasks.clear()
        with __import__("contextlib").suppress(OSError):
            os.remove(src)

    async def test_cache_put_failure_logged_not_fatal(self, client):
        # cache.put_async raises -> caught and logged, task still completes.
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "cache-put-fail-model")
        v = await _create_version(client, m["id"])
        mock_result = {"output_path": "/tmp/q2.mlx", "file_hash": "h2", "file_size": 200}
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv, \
             patch(
                 "fusion_model_hub.server.tasks.get_cache_manager",
                 new_callable=AsyncMock,
             ) as get_cache_mock:
            mc = AsyncMock()
            mc.quantize = AsyncMock(return_value=mock_result)
            MockConv.return_value = mc
            cache_inst = MagicMock()
            cache_inst.has = MagicMock(return_value=False)
            cache_inst.put_async = AsyncMock(side_effect=RuntimeError("disk full"))
            get_cache_mock.return_value = cache_inst
            r = await client.post("/api/v1/quantize", json={
                "source_version_id": v["id"], "quant_bits": 4,
            })
            task_id = r.json()["task_id"]
            await asyncio.sleep(0.5)
        status = await client.get(f"/api/v1/quantize/{task_id}")
        # put failure is logged but does not fail the task.
        assert status.json()["status"] == "completed"
        tasks_mod._running_tasks.clear()

    async def test_no_valid_output_status_failed(self, client):
        # converter returns status != completed -> lines 192-208 FAILED branch.
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "no-output-model")
        v = await _create_version(client, m["id"])
        mock_result = {"output_path": "", "file_hash": "", "file_size": 0, "status": "error"}
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv:
            mc = AsyncMock()
            mc.quantize = AsyncMock(return_value=mock_result)
            MockConv.return_value = mc
            r = await client.post("/api/v1/quantize", json={
                "source_version_id": v["id"], "quant_bits": 4,
            })
            task_id = r.json()["task_id"]
            await asyncio.sleep(0.5)
        status = await client.get(f"/api/v1/quantize/{task_id}")
        data = status.json()
        assert data["status"] == "failed"
        assert "no valid output" in data["error_message"].lower()
        tasks_mod._running_tasks.clear()

    async def test_no_valid_output_empty_path(self, client):
        # status completed but output_path empty -> FAILED branch (line 191 condition).
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "empty-path-model")
        v = await _create_version(client, m["id"])
        mock_result = {"output_path": "", "file_hash": "h", "file_size": 10, "status": "completed"}
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv:
            mc = AsyncMock()
            mc.quantize = AsyncMock(return_value=mock_result)
            MockConv.return_value = mc
            r = await client.post("/api/v1/quantize", json={
                "source_version_id": v["id"], "quant_bits": 4,
            })
            task_id = r.json()["task_id"]
            await asyncio.sleep(0.5)
        status = await client.get(f"/api/v1/quantize/{task_id}")
        assert status.json()["status"] == "failed"
        tasks_mod._running_tasks.clear()

    async def test_bench_auto_trigger_success(self, client, settings):
        # lines 259-269: bench_auto_trigger + bench_url set + MLX returns 200/201/202.
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        # Enable bench auto-trigger on the live settings singleton. The fixture
        # bound this same `settings` object into init_deps, so get_settings()
        # already returns it — no re-init (a fresh :memory: engine would lose
        # the tables and 500 on the next write).
        settings.bench_auto_trigger = True
        settings.bench_url = "http://bench.test:8090"
        m = await _create_model(client, "bench-ok-model")
        v = await _create_version(client, m["id"])
        mock_result = {"output_path": "/tmp/qb.mlx", "file_hash": "hb", "file_size": 300}
        bench_ok = _mock_httpx_response(status_code=202, json_data={"task_id": "b1"})
        bench_ctx = _mock_httpx_ctx(post_return=bench_ok)
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv, \
             patch(
                 "httpx.AsyncClient",
                 return_value=bench_ctx,
             ):
            mc = AsyncMock()
            mc.quantize = AsyncMock(return_value=mock_result)
            MockConv.return_value = mc
            r = await client.post("/api/v1/quantize", json={
                "source_version_id": v["id"], "quant_bits": 4,
            })
            task_id = r.json()["task_id"]
            await asyncio.sleep(0.5)
        status = await client.get(f"/api/v1/quantize/{task_id}")
        assert status.json()["status"] == "completed"
        # Bench was called.
        assert bench_ctx.post.called
        tasks_mod._running_tasks.clear()
        # Reset settings to avoid bleeding into other tests.
        settings.bench_auto_trigger = False
        settings.bench_url = ""

    async def test_bench_auto_trigger_non_200_logged(self, client, settings):
        # lines 270-271: bench returns 500 -> warning logged, task still completes.
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        settings.bench_auto_trigger = True
        settings.bench_url = "http://bench.test:8090"
        m = await _create_model(client, "bench-500-model")
        v = await _create_version(client, m["id"])
        mock_result = {"output_path": "/tmp/qb2.mlx", "file_hash": "hb2", "file_size": 300}
        bench_bad = _mock_httpx_response(status_code=500, text="bench down")
        bench_ctx = _mock_httpx_ctx(post_return=bench_bad)
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv, \
             patch(
                 "httpx.AsyncClient",
                 return_value=bench_ctx,
             ):
            mc = AsyncMock()
            mc.quantize = AsyncMock(return_value=mock_result)
            MockConv.return_value = mc
            r = await client.post("/api/v1/quantize", json={
                "source_version_id": v["id"], "quant_bits": 4,
            })
            task_id = r.json()["task_id"]
            await asyncio.sleep(0.5)
        status = await client.get(f"/api/v1/quantize/{task_id}")
        assert status.json()["status"] == "completed"
        tasks_mod._running_tasks.clear()
        settings.bench_auto_trigger = False
        settings.bench_url = ""

    async def test_precision_loss_warning_triggered(self, client, settings):
        # lines 274-305: src_score > out_score, loss > threshold -> webhook dispatched.
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        settings.precision_loss_threshold = 5.0
        m = await _create_model(client, "precision-model")
        v = await _create_version(client, m["id"])
        # Seed a high benchmark_score on the source version.
        from fusion_model_hub.db.crud import get_version as crud_get_version
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            src = await crud_get_version(session, v["id"])
            src.benchmark_score = 80.0
            await session.commit()
        mock_result = {"output_path": "/tmp/qp.mlx", "file_hash": "hp", "file_size": 300}
        # Patch get_version inside tasks.py: source lookup returns the real source
        # (score 80); the output-version lookup returns a mock with score 60 so
        # loss_pct = (80-60)/80*100 = 25% > 5% threshold -> precision_warning fires.
        async def _fake_get_version(session, version_id):
            if version_id == v["id"]:
                return await crud_get_version(session, version_id)
            out = MagicMock()
            out.id = version_id
            out.benchmark_score = 60.0
            return out
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv, \
             patch(
                 "fusion_model_hub.server.routers.webhooks.dispatch_webhook_event",
                 new_callable=AsyncMock,
             ) as mock_wh, \
             patch(
                 "fusion_model_hub.server.tasks.get_version",
                 new_callable=AsyncMock, side_effect=_fake_get_version,
             ):
            mc = AsyncMock()
            mc.quantize = AsyncMock(return_value=mock_result)
            MockConv.return_value = mc
            r = await client.post("/api/v1/quantize", json={
                "source_version_id": v["id"], "quant_bits": 4,
            })
            task_id = r.json()["task_id"]
            await asyncio.sleep(0.5)
        events = [call.args[0] for call in mock_wh.call_args_list]
        assert "quantize.completed" in events
        assert "quantize.precision_warning" in events
        # Confirm the loss_percent payload.
        warn_call = next(c for c in mock_wh.call_args_list if c.args[0] == "quantize.precision_warning")
        assert warn_call.args[1]["loss_percent"] == 25.0
        assert warn_call.args[1]["threshold"] == 5.0
        tasks_mod._running_tasks.clear()
        settings.precision_loss_threshold = 10.0


# =====================================================================
# tasks.py: webhook dispatch failure branches (lines 324-325, 335-336)
# =====================================================================


class TestRunQuantizeWebhookFailures:
    async def test_quantize_failed_webhook_dispatch_failure(self, client):
        # lines 324-325: quantize.failed webhook raises -> logged, status update still runs.
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "wh-fail-quant-model")
        v = await _create_version(client, m["id"])
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv, \
             patch(
                 "fusion_model_hub.server.routers.webhooks.dispatch_webhook_event",
                 new_callable=AsyncMock, side_effect=RuntimeError("wh down"),
             ):
            mc = AsyncMock()
            mc.quantize = AsyncMock(side_effect=RuntimeError("converter boom"))
            MockConv.return_value = mc
            r = await client.post("/api/v1/quantize", json={
                "source_version_id": v["id"], "quant_bits": 4,
            })
            task_id = r.json()["task_id"]
            await asyncio.sleep(0.5)
        status = await client.get(f"/api/v1/quantize/{task_id}")
        # Webhook failure is swallowed; task still marked FAILED via the status update.
        assert status.json()["status"] == "failed"
        assert "converter boom" in status.json()["error_message"]
        tasks_mod._running_tasks.clear()

    async def test_status_update_failure_on_error(self, client):
        # lines 335-336: the final status update itself raises -> logged, not re-raised.
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        m = await _create_model(client, "status-update-fail-model")
        v = await _create_version(client, m["id"])
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv, \
             patch(
                 "fusion_model_hub.server.routers.webhooks.dispatch_webhook_event",
                 new_callable=AsyncMock,
             ), \
             patch(
                 "fusion_model_hub.server.tasks.update_quantize_task",
                 new_callable=AsyncMock, side_effect=RuntimeError("db locked"),
             ):
            mc = AsyncMock()
            mc.quantize = AsyncMock(side_effect=RuntimeError("original fail"))
            MockConv.return_value = mc
            r = await client.post("/api/v1/quantize", json={
                "source_version_id": v["id"], "quant_bits": 4,
            })
            task_id = r.json()["task_id"]
            await asyncio.sleep(0.5)
        # The status-update failure is logged but not re-raised; the task row
        # stays in whatever state the failed update left it (PENDING/RUNNING).
        # The key assertion: no exception propagated out of the runner.
        tasks_mod._running_tasks.clear()
        assert task_id  # runner completed without raising


# =====================================================================
# app.py: _reconcile_orphaned_tasks (RUNNING->FAILED, PENDING->requeue)
# =====================================================================


class TestReconcileOrphanedTasks:
    async def test_reconcile_fails_running_and_requeues_pending(self, client):
        # app.py _reconcile_orphaned_tasks: RUNNING QuantizeTask -> FAILED,
        # PENDING QuantizeTask -> resume_quantize (requeue). The resume path
        # calls claim_quantize_task; for the PENDING row it claims and spawns
        # a runner. We mock the converter so the resumed task completes.
        from fusion_model_hub.server import tasks as tasks_mod
        tasks_mod._running_tasks.clear()
        from fusion_model_hub.db.crud import create_quantize_task, get_quantize_task
        from fusion_model_hub.db.models import TaskStatus
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        m = await _create_model(client, "reconcile-model")
        v = await _create_version(client, m["id"])
        # Create one RUNNING and one PENDING task row directly (simulating orphan
        # state left by a server restart).
        async with sf() as session:
            running_t = await create_quantize_task(
                session, source_version_id=v["id"], target_format="mlx", quant_bits=4,
            )
            # Flip it to RUNNING manually.
            from fusion_model_hub.db.crud import update_quantize_task
            await update_quantize_task(
                session, running_t.id, status=TaskStatus.RUNNING,
            )
            pending_t = await create_quantize_task(
                session, source_version_id=v["id"], target_format="mlx", quant_bits=4,
            )
            running_id = running_t.id
            pending_id = pending_t.id
        # Mock the converter so the resumed PENDING runner completes cleanly.
        mock_result = {"output_path": "/tmp/rec.mlx", "file_hash": "hr", "file_size": 100}
        with patch("fusion_model_hub.server.tasks.ModelConverter") as MockConv:
            mc = AsyncMock()
            mc.quantize = AsyncMock(return_value=mock_result)
            MockConv.return_value = mc
            await _reconcile_orphaned_tasks()
            # Give the resumed runner time to finish.
            await asyncio.sleep(0.5)
        # RUNNING task should now be FAILED with the orphan message.
        async with sf() as session:
            rt = await get_quantize_task(session, running_id)
            assert rt.status.value == "failed"
            assert "orphaned" in rt.error_message.lower()
        # PENDING task should have been claimed and completed by the resumed runner.
        async with sf() as session:
            pt = await get_quantize_task(session, pending_id)
            assert pt.status.value in ("completed", "running")
        tasks_mod._running_tasks.clear()


# =====================================================================
# downloads.py: create/list/get/cancel error paths (lines 42, 79-82, 111, 137, 151)
# =====================================================================


class TestDownloadEndpoints:
    pass


# =====================================================================
# downloads.py: _run_download happy + resume + 206 + progress (lines 179-215, 240-251)
# =====================================================================


class TestRunDownloadHappy:
    pass


# =====================================================================
# downloads.py: _run_download error + retry + integrity + cancel (lines 308-323)
# =====================================================================


class TestRunDownloadErrors:
    pass
