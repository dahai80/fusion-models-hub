import asyncio
import json
import logging
import os
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
    return Settings(
        host="127.0.0.1",
        port=8080,
        data_dir="/tmp/fmh_cov_test",
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
async def client(app, settings):
    engine = get_engine(settings.db_url)
    await init_db(engine)
    init_deps(settings, engine)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _create_model(client, name="cov-model"):
    resp = await client.post("/api/v1/models", json={
        "name": name, "description": "test", "model_type": "llm",
    })
    assert resp.status_code == 201
    return resp.json()


async def _create_published_version(client, model_id, version="1.0.0"):
    resp = await client.post(
        f"/api/v1/models/{model_id}/versions",
        data={"version": version, "format": "mlx", "quantization": "4bit"},
        files={"file": ("", b"")},
    )
    assert resp.status_code == 201
    vid = resp.json()["id"]
    await client.post(f"/api/v1/versions/{vid}/promote")
    return vid


class TestBranches:
    async def test_create_branch(self, client):
        m = await _create_model(client, "branch-model")
        resp = await client.post(f"/api/v1/models/{m['id']}/branches", json={
            "name": "feature-x", "description": "test branch",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "feature-x"
        assert data["model_id"] == m["id"]

    async def test_create_branch_model_not_found(self, client):
        resp = await client.post("/api/v1/models/nonexistent/branches", json={
            "name": "feature-x",
        })
        assert resp.status_code == 404

    async def test_list_branches(self, client):
        m = await _create_model(client, "branch-list-model")
        await client.post(f"/api/v1/models/{m['id']}/branches", json={"name": "b1"})
        await client.post(f"/api/v1/models/{m['id']}/branches", json={"name": "b2"})
        resp = await client.get(f"/api/v1/models/{m['id']}/branches")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 2

    async def test_list_branches_model_not_found(self, client):
        resp = await client.get("/api/v1/models/nonexistent/branches")
        assert resp.status_code == 404

    async def test_get_branch(self, client):
        m = await _create_model(client, "branch-get-model")
        cr = await client.post(f"/api/v1/models/{m['id']}/branches", json={"name": "b1"})
        bid = cr.json()["id"]
        resp = await client.get(f"/api/v1/models/branches/{bid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == bid

    async def test_get_branch_not_found(self, client):
        resp = await client.get("/api/v1/models/branches/nonexistent")
        assert resp.status_code == 404

    async def test_update_branch(self, client):
        m = await _create_model(client, "branch-upd-model")
        cr = await client.post(f"/api/v1/models/{m['id']}/branches", json={"name": "b1"})
        bid = cr.json()["id"]
        resp = await client.patch(f"/api/v1/models/branches/{bid}", json={
            "description": "updated desc",
        })
        assert resp.status_code == 200
        assert resp.json()["description"] == "updated desc"

    async def test_update_branch_status(self, client):
        m = await _create_model(client, "branch-status-model")
        cr = await client.post(f"/api/v1/models/{m['id']}/branches", json={"name": "b1"})
        bid = cr.json()["id"]
        resp = await client.patch(f"/api/v1/models/branches/{bid}", json={
            "status": "merged",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "merged"

    async def test_update_branch_invalid_status(self, client):
        m = await _create_model(client, "branch-inv-status")
        cr = await client.post(f"/api/v1/models/{m['id']}/branches", json={"name": "b1"})
        bid = cr.json()["id"]
        resp = await client.patch(f"/api/v1/models/branches/{bid}", json={
            "status": "invalid_status",
        })
        assert resp.status_code == 400

    async def test_update_branch_no_fields(self, client):
        m = await _create_model(client, "branch-no-fields")
        cr = await client.post(f"/api/v1/models/{m['id']}/branches", json={"name": "b1"})
        bid = cr.json()["id"]
        resp = await client.patch(f"/api/v1/models/branches/{bid}", json={})
        assert resp.status_code == 400

    async def test_update_branch_not_found(self, client):
        resp = await client.patch("/api/v1/models/branches/nonexistent", json={
            "description": "x",
        })
        assert resp.status_code == 404

    async def test_delete_branch(self, client):
        m = await _create_model(client, "branch-del-model")
        cr = await client.post(f"/api/v1/models/{m['id']}/branches", json={"name": "b1"})
        bid = cr.json()["id"]
        resp = await client.delete(f"/api/v1/models/branches/{bid}")
        assert resp.status_code == 200

    async def test_delete_branch_not_found(self, client):
        resp = await client.delete("/api/v1/models/branches/nonexistent")
        assert resp.status_code == 404

    async def test_merge_branch(self, client):
        m = await _create_model(client, "branch-merge-model")
        cr = await client.post(f"/api/v1/models/{m['id']}/branches", json={"name": "b1"})
        bid = cr.json()["id"]
        resp = await client.post(f"/api/v1/models/branches/{bid}/merge")
        assert resp.status_code == 200
        assert resp.json()["status"] == "merged"

    async def test_merge_branch_not_active(self, client):
        m = await _create_model(client, "branch-merge-inactive")
        cr = await client.post(f"/api/v1/models/{m['id']}/branches", json={"name": "b1"})
        bid = cr.json()["id"]
        await client.patch(f"/api/v1/models/branches/{bid}", json={"status": "merged"})
        resp = await client.post(f"/api/v1/models/branches/{bid}/merge")
        assert resp.status_code == 400

    async def test_merge_branch_not_found(self, client):
        resp = await client.post("/api/v1/models/branches/nonexistent/merge")
        assert resp.status_code == 404

    async def test_list_branches_with_status_filter(self, client):
        m = await _create_model(client, "branch-filter-model")
        await client.post(f"/api/v1/models/{m['id']}/branches", json={"name": "b1"})
        resp = await client.get(f"/api/v1/models/{m['id']}/branches", params={"status": "active"})
        assert resp.status_code == 200


class TestFavorites:
    async def test_add_favorite(self, client):
        m = await _create_model(client, "fav-model")
        resp = await client.post(f"/api/v1/models/{m['id']}/favorites")
        assert resp.status_code == 201
        assert resp.json()["model_id"] == m["id"]

    async def test_add_favorite_model_not_found(self, client):
        resp = await client.post("/api/v1/models/nonexistent/favorites")
        assert resp.status_code == 404

    async def test_add_favorite_duplicate(self, client):
        m = await _create_model(client, "fav-dup-model")
        await client.post(f"/api/v1/models/{m['id']}/favorites")
        resp = await client.post(f"/api/v1/models/{m['id']}/favorites")
        assert resp.status_code == 409

    async def test_list_favorites(self, client):
        m = await _create_model(client, "fav-list-model")
        await client.post(f"/api/v1/models/{m['id']}/favorites")
        resp = await client.get(f"/api/v1/models/{m['id']}/favorites")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) >= 1

    async def test_list_favorites_model_not_found(self, client):
        resp = await client.get("/api/v1/models/nonexistent/favorites")
        assert resp.status_code == 404

    async def test_list_my_favorites(self, client):
        m = await _create_model(client, "fav-me-model")
        await client.post(f"/api/v1/models/{m['id']}/favorites")
        resp = await client.get("/api/v1/models/favorites/me")
        assert resp.status_code == 200

    async def test_remove_favorite(self, client):
        m = await _create_model(client, "fav-rm-model")
        cr = await client.post(f"/api/v1/models/{m['id']}/favorites")
        fid = cr.json()["id"]
        resp = await client.delete(f"/api/v1/models/favorites/{fid}")
        assert resp.status_code == 200

    async def test_remove_favorite_not_found(self, client):
        resp = await client.delete("/api/v1/models/favorites/nonexistent")
        assert resp.status_code == 404


class TestRatings:
    async def test_create_rating(self, client):
        m = await _create_model(client, "rate-model")
        resp = await client.post(f"/api/v1/models/{m['id']}/ratings", json={
            "score": 5, "comment": "excellent",
        })
        assert resp.status_code == 201
        assert resp.json()["score"] == 5

    async def test_create_rating_model_not_found(self, client):
        resp = await client.post("/api/v1/models/nonexistent/ratings", json={
            "score": 3,
        })
        assert resp.status_code == 404

    async def test_create_rating_invalid_score(self, client):
        m = await _create_model(client, "rate-inv-model")
        resp = await client.post(f"/api/v1/models/{m['id']}/ratings", json={
            "score": 0,
        })
        assert resp.status_code == 422

    async def test_list_ratings(self, client):
        m = await _create_model(client, "rate-list-model")
        await client.post(f"/api/v1/models/{m['id']}/ratings", json={"score": 4})
        await client.post(f"/api/v1/models/{m['id']}/ratings", json={"score": 5})
        resp = await client.get(f"/api/v1/models/{m['id']}/ratings")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) >= 2
        assert "average_score" in data

    async def test_list_ratings_model_not_found(self, client):
        resp = await client.get("/api/v1/models/nonexistent/ratings")
        assert resp.status_code == 404

    async def test_get_rating_summary(self, client):
        m = await _create_model(client, "rate-summary-model")
        await client.post(f"/api/v1/models/{m['id']}/ratings", json={"score": 3})
        resp = await client.get(f"/api/v1/models/{m['id']}/ratings/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_ratings"] >= 1
        assert "average_score" in data

    async def test_get_rating_summary_model_not_found(self, client):
        resp = await client.get("/api/v1/models/nonexistent/ratings/summary")
        assert resp.status_code == 404

    async def test_delete_rating(self, client):
        m = await _create_model(client, "rate-del-model")
        cr = await client.post(f"/api/v1/models/{m['id']}/ratings", json={"score": 2})
        rid = cr.json()["id"]
        resp = await client.delete(f"/api/v1/models/ratings/{rid}")
        assert resp.status_code == 200

    async def test_delete_rating_not_found(self, client):
        resp = await client.delete("/api/v1/models/ratings/nonexistent")
        assert resp.status_code == 404


class TestBackup:
    @pytest.mark.asyncio
    async def test_start_and_stop_scheduler(self):
        from fusion_model_hub.server import backup
        backup._backup_task = None
        mock_settings = MagicMock()
        mock_settings.backup_dir = "/tmp/fmh_backup_test"
        mock_settings.backup_interval_seconds = 9999
        with patch("fusion_model_hub.server.backup.get_settings", return_value=mock_settings):
            backup.start_backup_scheduler()
            assert backup._backup_task is not None
            assert not backup._backup_task.done()
        backup.stop_backup_scheduler()
        assert backup._backup_task is None

    def test_stop_scheduler_noop_if_none(self):
        from fusion_model_hub.server import backup
        backup._backup_task = None
        backup.stop_backup_scheduler()
        assert backup._backup_task is None

    @pytest.mark.asyncio
    async def test_perform_backup(self, settings):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = get_engine(settings.db_url)
            await init_db(engine)
            init_deps(settings, engine)

            from fusion_model_hub.db import crud
            from fusion_model_hub.server.deps import get_session_factory
            sf = get_session_factory()
            async with sf() as session:
                await crud.create_model(session, name="backup-model")

            from fusion_model_hub.server.backup import _perform_backup
            await _perform_backup(tmpdir)

            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert files[0].startswith("backup_")
            with open(os.path.join(tmpdir, files[0])) as f:
                data = json.load(f)
            assert "models" in data
            assert "versions" in data
            assert len(data["models"]) >= 1

    @pytest.mark.asyncio
    async def test_run_backup_loop_no_dir(self):
        from fusion_model_hub.server.backup import _run_backup_loop
        mock_settings = MagicMock()
        mock_settings.backup_dir = ""
        mock_settings.backup_interval_seconds = 3600
        with patch("fusion_model_hub.server.backup.get_settings", return_value=mock_settings):
            await _run_backup_loop()

    @pytest.mark.asyncio
    async def test_run_backup_loop_cancelled(self):
        from fusion_model_hub.server.backup import _run_backup_loop
        mock_settings = MagicMock()
        mock_settings.backup_dir = "/tmp/test_backup"
        mock_settings.backup_interval_seconds = 3600
        with patch("fusion_model_hub.server.backup.get_settings", return_value=mock_settings):
            with patch("fusion_model_hub.server.backup._perform_backup", new_callable=AsyncMock):
                with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=asyncio.CancelledError):
                    await _run_backup_loop()


class TestMetrics:
    def test_normalize_path_ids(self):
        from fusion_model_hub.server.metrics import _normalize_path
        assert _normalize_path("/api/v1/models/abc123def456ghi789jkl012mno345") == "/api/v1/models/:id"
        assert _normalize_path("/api/v1/models/12345") == "/api/v1/models/:id"
        assert _normalize_path("/api/v1/models") == "/api/v1/models"
        assert _normalize_path("/") == "/"

    def test_metrics_response(self):
        from fusion_model_hub.server.metrics import metrics_response
        resp = metrics_response()
        assert "text/plain" in resp.media_type
        assert len(resp.body) > 0

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_middleware_skips_docs(self, client):
        resp = await client.get("/docs")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_update_resource_metrics(self, settings):
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        from fusion_model_hub.server.deps import get_session_factory
        from fusion_model_hub.server.metrics import update_resource_metrics
        sf = get_session_factory()
        async with sf() as session:
            await update_resource_metrics(session)


class TestLoraMerge:
    async def test_start_lora_merge(self, client):
        m = await _create_model(client, "lora-merge-model")
        v1 = await _create_published_version(client, m["id"], "1.0.0")
        v2 = await _create_published_version(client, m["id"], "1.1.0")
        resp = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": v1, "lora_version_id": v2,
            "target_format": "mlx", "quant_bits": 4,
        })
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "submitted"

    async def test_start_lora_merge_invalid_bits(self, client):
        m = await _create_model(client, "lora-bits-model")
        v1 = await _create_published_version(client, m["id"], "1.0.0")
        v2 = await _create_published_version(client, m["id"], "1.1.0")
        resp = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": v1, "lora_version_id": v2, "quant_bits": 3,
        })
        assert resp.status_code == 400

    async def test_start_lora_merge_base_not_found(self, client):
        m = await _create_model(client, "lora-nf-model")
        v2 = await _create_published_version(client, m["id"], "1.0.0")
        resp = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": "nonexistent", "lora_version_id": v2, "quant_bits": 4,
        })
        assert resp.status_code == 404

    async def test_start_lora_merge_lora_not_found(self, client):
        m = await _create_model(client, "lora-nf2-model")
        v1 = await _create_published_version(client, m["id"], "1.0.0")
        resp = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": v1, "lora_version_id": "nonexistent", "quant_bits": 4,
        })
        assert resp.status_code == 404

    async def test_get_lora_merge_status(self, client):
        m = await _create_model(client, "lora-status-model")
        v1 = await _create_published_version(client, m["id"], "1.0.0")
        v2 = await _create_published_version(client, m["id"], "1.1.0")
        cr = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": v1, "lora_version_id": v2, "quant_bits": 4,
        })
        task_id = cr.json()["task_id"]
        await asyncio.sleep(0.3)
        resp = await client.get(f"/api/v1/quantize/lora-merge/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == task_id

    async def test_get_lora_merge_status_not_found(self, client):
        resp = await client.get("/api/v1/quantize/lora-merge/nonexistent")
        assert resp.status_code == 404


class TestQuantizeCompare:
    async def test_compare_quantize_not_found(self, client):
        resp = await client.get("/api/v1/quantize/nonexistent/compare")
        assert resp.status_code == 404


class TestAppExceptionHandler:
    @pytest.mark.asyncio
    async def test_global_exception_handler(self, settings):
        app = create_app(settings)
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            with patch("fusion_model_hub.server.routers.models.crud.list_models", side_effect=RuntimeError("boom")):
                resp = await c.get("/api/v1/models")
                assert resp.status_code == 500
