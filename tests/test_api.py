
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


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health(self, client):
        resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "mlx" in data

    @pytest.mark.asyncio
    async def test_storage(self, client):
        resp = await client.get("/api/v1/system/storage")
        assert resp.status_code == 200


class TestModelCRUD:
    @pytest.mark.asyncio
    async def test_create_model(self, client):
        resp = await client.post("/api/v1/models", json={
            "name": "test-model-1",
            "description": "A test model",
            "model_type": "llm",
            "architecture": "qwen2",
            "params_size": "7B",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "test-model-1"
        assert data["model_type"] == "llm"
        assert "id" in data

    @pytest.mark.asyncio
    async def test_create_duplicate_name(self, client):
        await client.post("/api/v1/models", json={"name": "dup-model"})
        resp = await client.post("/api/v1/models", json={"name": "dup-model"})
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_list_models(self, client):
        await client.post("/api/v1/models", json={"name": "list-a"})
        await client.post("/api/v1/models", json={"name": "list-b"})
        resp = await client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2
        assert len(data["items"]) >= 2

    @pytest.mark.asyncio
    async def test_list_models_with_keyword(self, client):
        await client.post("/api/v1/models", json={
            "name": "keyword-unique-xyz", "description": "searchable",
        })
        resp = await client.get("/api/v1/models", params={"keyword": "unique-xyz"})
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    @pytest.mark.asyncio
    async def test_get_model(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "get-model"})
        model_id = create_resp.json()["id"]
        resp = await client.get(f"/api/v1/models/{model_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "get-model"

    @pytest.mark.asyncio
    async def test_get_model_not_found(self, client):
        resp = await client.get("/api/v1/models/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_model(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "update-model"})
        model_id = create_resp.json()["id"]
        resp = await client.put(
            f"/api/v1/models/{model_id}",
            json={"description": "updated desc"},
        )
        assert resp.status_code == 200
        assert resp.json()["description"] == "updated desc"

    @pytest.mark.asyncio
    async def test_delete_model(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "delete-model"})
        model_id = create_resp.json()["id"]
        resp = await client.delete(f"/api/v1/models/{model_id}")
        assert resp.status_code == 200
        resp2 = await client.get(f"/api/v1/models/{model_id}")
        assert resp2.status_code == 404

    @pytest.mark.asyncio
    async def test_model_with_tags(self, client):
        resp = await client.post("/api/v1/models", json={
            "name": "tagged-model",
            "tags": [{"key": "domain", "value": "nlp"}, {"key": "size", "value": "small"}],
        })
        assert resp.status_code == 201
        assert len(resp.json()["tags"]) == 2


class TestVersionCRUD:
    @pytest.mark.asyncio
    async def test_upload_version(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "ver-model"})
        model_id = create_resp.json()["id"]
        resp = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": "1.0.0", "format": "mlx", "quantization": "4bit"},
            files={"file": ("", b"")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["version"] == "1.0.0"
        assert data["model_id"] == model_id

    @pytest.mark.asyncio
    async def test_upload_version_with_file(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "ver-file-model"})
        model_id = create_resp.json()["id"]
        resp = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": "1.0.0", "format": "mlx", "quantization": "4bit"},
            files={"file": ("model.bin", b"fake model data", "application/octet-stream")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["file_size"] > 0
        assert data["file_hash"] != ""

    @pytest.mark.asyncio
    async def test_list_versions(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "ver-list-model"})
        model_id = create_resp.json()["id"]
        await client.post(f"/api/v1/models/{model_id}/versions", data={"version": "1.0.0"}, files={"file": ("", b"")})
        await client.post(f"/api/v1/models/{model_id}/versions", data={"version": "1.1.0"}, files={"file": ("", b"")})
        resp = await client.get(f"/api/v1/models/{model_id}/versions")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 2

    @pytest.mark.asyncio
    async def test_get_version(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "ver-get-model"})
        model_id = create_resp.json()["id"]
        ver_resp = await client.post(
            f"/api/v1/models/{model_id}/versions", data={"version": "2.0.0"}, files={"file": ("", b"")},
        )
        version_id = ver_resp.json()["id"]
        resp = await client.get(f"/api/v1/versions/{version_id}")
        assert resp.status_code == 200
        assert resp.json()["version"] == "2.0.0"

    @pytest.mark.asyncio
    async def test_version_status_change(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "ver-status-model"})
        model_id = create_resp.json()["id"]
        ver_resp = await client.post(
            f"/api/v1/models/{model_id}/versions", data={"version": "1.0.0"}, files={"file": ("", b"")},
        )
        version_id = ver_resp.json()["id"]
        await client.put(
            f"/api/v1/versions/{version_id}/status",
            json={"target_status": "testing"},
        )
        await client.put(
            f"/api/v1/versions/{version_id}/metrics",
            json={"benchmark_score": 90.0},
        )
        resp = await client.put(
            f"/api/v1/versions/{version_id}/status",
            json={"target_status": "published", "approval_level": "l1"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"

    @pytest.mark.asyncio
    async def test_version_rollback(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "ver-rollback-model"})
        model_id = create_resp.json()["id"]
        ver_resp = await client.post(
            f"/api/v1/models/{model_id}/versions", data={"version": "1.0.0"}, files={"file": ("", b"")},
        )
        version_id = ver_resp.json()["id"]
        await client.put(f"/api/v1/versions/{version_id}/status", json={"target_status": "testing"})
        await client.put(f"/api/v1/versions/{version_id}/metrics", json={"benchmark_score": 90.0})
        await client.put(f"/api/v1/versions/{version_id}/status", json={"target_status": "published", "approval_level": "l1"})
        await client.put(f"/api/v1/versions/{version_id}/status", json={"target_status": "deprecated"})
        resp = await client.post(f"/api/v1/versions/{version_id}/rollback")
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"


class TestVersionUploadConcurrency:
    # P1-D: (model_id, version) is unique. A second upload of an existing
    # version -> 409, and a true concurrent pair -> exactly one 201 + one 409
    # (the DB unique constraint is the race winner; create_version catches the
    # IntegrityError and surfaces it instead of a 500 or a duplicate row).
    @pytest.mark.asyncio
    async def test_duplicate_version_returns_409(self, client):
        model_id = (await client.post("/api/v1/models", json={"name": "dup-ver"})).json()["id"]
        first = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": "1.0.0"}, files={"file": ("", b"")},
        )
        assert first.status_code == 201
        second = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": "1.0.0"}, files={"file": ("", b"")},
        )
        assert second.status_code == 409
        # exactly one row for (model_id, "1.0.0") — the constraint prevented the
        # duplicate; create_version caught the IntegrityError and surfaced 409.
        resp = await client.get(f"/api/v1/models/{model_id}/versions")
        vers = [v for v in resp.json().get("items", []) if v["version"] == "1.0.0"]
        assert len(vers) == 1

    @pytest.mark.asyncio
    async def test_concurrent_same_version_one_wins(self, tmp_path):
        # The default :memory: client gives each session its own DB, so a true
        # async-gather race is not representative. Use a file SQLite engine with
        # WAL + busy_timeout (database.get_engine applies both for file URLs) so
        # each session gets its own connection: real concurrent writers, and the
        # uq_model_version constraint makes exactly one win, the other 409s.
        import asyncio

        from fusion_model_hub.db import crud
        from fusion_model_hub.db.database import get_engine, init_db
        from fusion_model_hub.server.config import Settings
        from fusion_model_hub.server.deps import get_session_factory, init_deps

        db_file = tmp_path / "race.db"
        db_url = f"sqlite+aiosqlite:///{db_file}"
        engine = get_engine(db_url)
        await init_db(engine)
        settings = Settings(data_dir=str(tmp_path), db_url=db_url)
        init_deps(settings, engine)
        factory = get_session_factory()

        async with factory() as s:
            m = await crud.create_model(s, name="race-ver", architecture="qwen2", params_size="7B")
            mid = m.id

        async def upload():
            async with factory() as s:
                try:
                    return await crud.create_version(s, model_id=mid, version="2.0.0")
                except crud.VersionConflictError:
                    return None

        v1, v2 = await asyncio.gather(upload(), upload())
        winners = [x for x in (v1, v2) if x is not None]
        assert len(winners) == 1, f"expected exactly one winner, got {[w.id if w else None for w in (v1, v2)]}"
        assert winners[0].version == "2.0.0"
        await engine.dispose()


class TestHFImport:
    @pytest.mark.asyncio
    async def test_import_hf(self, client):
        resp = await client.post("/api/v1/models/import/hf", json={"hf_repo": "Qwen/Qwen2.5-7B"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["hf_repo"] == "Qwen/Qwen2.5-7B"
        assert "qwen" in data["name"]

    @pytest.mark.asyncio
    async def test_import_hf_missing_repo(self, client):
        resp = await client.post("/api/v1/models/import/hf", json={})
        assert resp.status_code == 400


class TestQuantizeAPI:
    @pytest.mark.asyncio
    async def test_submit_quantize_invalid_bits(self, client):
        resp = await client.post("/api/v1/quantize", json={
            "source_version_id": "nonexistent", "quant_bits": 3,
        })
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_get_quantize_task_not_found(self, client):
        resp = await client.get("/api/v1/quantize/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_quantize_tasks(self, client):
        resp = await client.get("/api/v1/quantize")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_running_quantize_tasks(self, client):
        resp = await client.get("/api/v1/quantize/running")
        assert resp.status_code == 200
        assert "tasks" in resp.json()


class TestUrlDownload:
    @pytest.mark.asyncio
    async def test_url_download_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/models/nonexistent/versions/download-url",
            json={"url": "https://example.com/model.bin"},
        )
        assert resp.status_code == 404


class TestHealthMLX:
    @pytest.mark.asyncio
    async def test_health_includes_mlx_info(self, client):
        resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "mlx" in data
        assert "status" in data["mlx"]
        assert "url" in data["mlx"]


class TestLifecycleStateMachine:
    @pytest.mark.asyncio
    async def test_invalid_transition_rejected(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "lifecycle-model"})
        model_id = create_resp.json()["id"]
        ver_resp = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": "1.0.0"}, files={"file": ("", b"")},
        )
        version_id = ver_resp.json()["id"]
        resp = await client.put(
            f"/api/v1/versions/{version_id}/status",
            json={"target_status": "published"},
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_valid_full_lifecycle(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "lifecycle-full-model"})
        model_id = create_resp.json()["id"]
        ver_resp = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": "2.0.0"}, files={"file": ("", b"")},
        )
        version_id = ver_resp.json()["id"]
        r1 = await client.put(f"/api/v1/versions/{version_id}/status", json={"target_status": "testing"})
        assert r1.json()["status"] == "testing"
        await client.put(f"/api/v1/versions/{version_id}/metrics", json={"benchmark_score": 90.0})
        r2 = await client.put(f"/api/v1/versions/{version_id}/status", json={"target_status": "published", "approval_level": "l1"})
        assert r2.json()["status"] == "published"
        r3 = await client.put(f"/api/v1/versions/{version_id}/status", json={"target_status": "deprecated"})
        assert r3.json()["status"] == "deprecated"
        r4 = await client.put(f"/api/v1/versions/{version_id}/status", json={"target_status": "retired"})
        assert r4.json()["status"] == "retired"

    @pytest.mark.asyncio
    async def test_deprecate_with_successor(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "deprecate-succ-model"})
        model_id = create_resp.json()["id"]
        v1 = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": "1.0.0"}, files={"file": ("", b"")},
        )
        v2 = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": "2.0.0"}, files={"file": ("", b"")},
        )
        v1_id = v1.json()["id"]
        v2_id = v2.json()["id"]
        await client.put(f"/api/v1/versions/{v1_id}/status", json={"target_status": "testing"})
        await client.put(f"/api/v1/versions/{v1_id}/metrics", json={"benchmark_score": 90.0})
        await client.put(f"/api/v1/versions/{v1_id}/status", json={"target_status": "published", "approval_level": "l1"})
        resp = await client.post(
            f"/api/v1/versions/{v1_id}/deprecate",
            json={"successor_version_id": v2_id},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "deprecated"
        assert resp.json()["successor_version_id"] == v2_id

    @pytest.mark.asyncio
    async def test_retire_endpoint(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "retire-model"})
        model_id = create_resp.json()["id"]
        ver_resp = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": "1.0.0"}, files={"file": ("", b"")},
        )
        version_id = ver_resp.json()["id"]
        await client.put(f"/api/v1/versions/{version_id}/status", json={"target_status": "testing"})
        await client.put(f"/api/v1/versions/{version_id}/metrics", json={"benchmark_score": 90.0})
        await client.put(f"/api/v1/versions/{version_id}/status", json={"target_status": "published", "approval_level": "l1"})
        await client.put(f"/api/v1/versions/{version_id}/status", json={"target_status": "deprecated"})
        resp = await client.post(f"/api/v1/versions/{version_id}/retire")
        assert resp.status_code == 200
        assert resp.json()["status"] == "retired"


class TestBenchmark:
    @pytest.mark.asyncio
    async def test_update_benchmark(self, client):
        create_resp = await client.post("/api/v1/models", json={"name": "bench-model"})
        model_id = create_resp.json()["id"]
        ver_resp = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": "1.0.0"}, files={"file": ("", b"")},
        )
        version_id = ver_resp.json()["id"]
        resp = await client.put(
            f"/api/v1/versions/{version_id}/benchmark",
            json={"benchmark_score": 85.5, "inference_latency": 12.3, "throughput": 45.6},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["benchmark_score"] == 85.5
        assert data["inference_latency"] == 12.3
        assert data["throughput"] == 45.6

    @pytest.mark.asyncio
    async def test_benchmark_version_not_found(self, client):
        resp = await client.put(
            "/api/v1/versions/nonexistent/benchmark",
            json={"benchmark_score": 10.0},
        )
        assert resp.status_code == 404


class TestApiKeyCRUD:
    @pytest.mark.asyncio
    async def test_create_api_key(self, client):
        resp = await client.post("/api/v1/auth/keys", json={"name": "test-key", "permissions": "read"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "test-key"
        assert "key" in data
        assert data["key"].startswith("fmh-")
        assert data["permissions"] == "read"
        assert data["role"] == "developer"

    @pytest.mark.asyncio
    async def test_create_api_key_with_role(self, client):
        for role in ["admin", "developer", "viewer"]:
            resp = await client.post("/api/v1/auth/keys", json={"name": f"key-{role}", "role": role})
            assert resp.status_code == 201
            assert resp.json()["role"] == role

    @pytest.mark.asyncio
    async def test_list_api_keys(self, client):
        await client.post("/api/v1/auth/keys", json={"name": "list-key"})
        resp = await client.get("/api/v1/auth/keys")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) >= 1

    @pytest.mark.asyncio
    async def test_delete_api_key(self, client):
        create_resp = await client.post("/api/v1/auth/keys", json={"name": "del-key"})
        key_id = create_resp.json()["id"]
        resp = await client.delete(f"/api/v1/auth/keys/{key_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_deactivate_api_key(self, client):
        create_resp = await client.post("/api/v1/auth/keys", json={"name": "deact-key"})
        key_id = create_resp.json()["id"]
        resp = await client.post(f"/api/v1/auth/keys/{key_id}/deactivate")
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False


class TestAuditLog:
    @pytest.mark.asyncio
    async def test_audit_log_query(self, client):
        resp = await client.get("/api/v1/system/audit")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data


class TestInference:
    @pytest.mark.asyncio
    async def test_serve_model_not_found(self, client):
        resp = await client.post("/api/v1/models/nonexistent/serve", json={})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_unload_not_loaded(self, client):
        resp = await client.delete("/api/v1/models/nonexistent/serve")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_serve_status_not_loaded(self, client):
        resp = await client.get("/api/v1/models/nonexistent/serve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_loaded"

    @pytest.mark.asyncio
    async def test_chat_not_loaded(self, client):
        resp = await client.post("/api/v1/inference/nonexistent/chat", json={"messages": []})
        assert resp.status_code == 400


# -- Phase 6: Cluster, Sync, Batch, Compare --

class TestClusterNodes:
    @pytest.mark.asyncio
    async def test_add_node(self, client):
        resp = await client.post(
            "/api/v1/cluster/nodes",
            json={"name": "node-1", "url": "http://node1:11444", "capabilities": "inference"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "node-1"
        assert data["url"] == "http://node1:11444"
        assert data["status"] == "active"

    @pytest.mark.asyncio
    async def test_list_nodes(self, client):
        await client.post("/api/v1/cluster/nodes", json={"name": "node-list", "url": "http://n2:11444"})
        resp = await client.get("/api/v1/cluster/nodes")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data and "total" in data
        assert data["total"] >= 2
        ids = [n["id"] for n in data["nodes"]]
        assert "local" in ids

    @pytest.mark.asyncio
    async def test_get_node(self, client):
        create = await client.post("/api/v1/cluster/nodes", json={"name": "node-get", "url": "http://n3:11444"})
        node_id = create.json()["id"]
        resp = await client.get(f"/api/v1/cluster/nodes/{node_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "node-get"

    @pytest.mark.asyncio
    async def test_delete_node(self, client):
        create = await client.post("/api/v1/cluster/nodes", json={"name": "node-del", "url": "http://n4:11444"})
        node_id = create.json()["id"]
        resp = await client.delete(f"/api/v1/cluster/nodes/{node_id}")
        assert resp.status_code == 200
        resp2 = await client.get(f"/api/v1/cluster/nodes/{node_id}")
        assert resp2.status_code == 404

    @pytest.mark.asyncio
    async def test_heartbeat(self, client):
        create = await client.post("/api/v1/cluster/nodes", json={"name": "node-hb", "url": "http://n5:11444"})
        node_id = create.json()["id"]
        resp = await client.post(f"/api/v1/cluster/nodes/{node_id}/heartbeat")
        assert resp.status_code == 200
        assert resp.json()["detail"] == "ok"


class TestBatchOps:
    @pytest.mark.asyncio
    async def test_batch_delete(self, client):
        ids = []
        for i in range(3):
            r = await client.post("/api/v1/models", json={"name": f"batch-del-{i}"})
            ids.append(r.json()["id"])
        resp = await client.post("/api/v1/models/batch/delete", json={"model_ids": ids})
        assert resp.status_code == 200
        assert resp.json()["count"] == 3

    @pytest.mark.asyncio
    async def test_batch_tag(self, client):
        ids = []
        for i in range(2):
            r = await client.post("/api/v1/models", json={"name": f"batch-tag-{i}"})
            ids.append(r.json()["id"])
        resp = await client.post(
            "/api/v1/models/batch/tag",
            json={"model_ids": ids, "tags": [{"key": "env", "value": "test"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 2


class TestCompare:
    @pytest.mark.asyncio
    async def test_compare_models(self, client):
        r1 = await client.post("/api/v1/models", json={"name": "cmp-a"})
        r2 = await client.post("/api/v1/models", json={"name": "cmp-b"})
        id1, id2 = r1.json()["id"], r2.json()["id"]
        resp = await client.get(f"/api/v1/models/compare?ids={id1},{id2}")
        assert resp.status_code == 200
        assert len(resp.json()["models"]) == 2

    @pytest.mark.asyncio
    async def test_compare_too_few(self, client):
        resp = await client.get("/api/v1/models/compare?ids=abc")
        assert resp.status_code == 400


class TestChunkUpload:
    @pytest.mark.asyncio
    async def test_chunk_upload_full(self, client):
        r = await client.post("/api/v1/models", json={"name": "chunk-model"})
        model_id = r.json()["id"]
        content = b"hello world chunk upload test data"
        chunk_size = len(content) // 2
        chunks = [content[:chunk_size], content[chunk_size:]]
        version_id = None
        for i, chunk in enumerate(chunks):
            resp = await client.post(
                f"/api/v1/models/{model_id}/versions/chunk-upload",
                data={
                    "version": "1.0-chunk",
                    "format": "mlx",
                    "quantization": "4bit",
                    "filename": "model.mlx",
                    "total_chunks": "2",
                    "chunk_index": str(i),
                },
                files={"chunk": ("chunk.bin", chunk, "application/octet-stream")},
            )
            if i < len(chunks) - 1:
                assert resp.status_code == 201
                assert resp.json()["status"] == "chunk_received"
            else:
                assert resp.status_code == 201
                version_id = resp.json()["id"]
        assert version_id is not None


class TestVersionDownload:
    @pytest.mark.asyncio
    async def test_download_version(self, client):
        r = await client.post("/api/v1/models", json={"name": "dl-model"})
        model_id = r.json()["id"]
        content = b"downloadable model content here"
        resp = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": "1.0-dl", "format": "mlx", "quantization": "4bit"},
            files={"file": ("model.mlx", content, "application/octet-stream")},
        )
        assert resp.status_code == 201
        version_id = resp.json()["id"]
        dl = await client.get(f"/api/v1/versions/{version_id}/download")
        assert dl.status_code == 200
        assert dl.content == content

    @pytest.mark.asyncio
    async def test_download_version_no_file(self, client):
        r = await client.post("/api/v1/models", json={"name": "dl-nofile"})
        model_id = r.json()["id"]
        resp = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": "1.0-nofile", "format": "mlx", "quantization": "4bit"},
        )
        assert resp.status_code == 201
        version_id = resp.json()["id"]
        dl = await client.get(f"/api/v1/versions/{version_id}/download")
        assert dl.status_code == 404


class TestModelSync:
    @pytest.mark.asyncio
    async def test_sync_invalid_url_scheme(self, client):
        resp = await client.post(
            "/api/v1/models/sync",
            json={"source_url": "ftp://evil.com", "dry_run": True},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_sync_internal_url_blocked(self, client):
        resp = await client.post(
            "/api/v1/models/sync",
            json={"source_url": "http://127.0.0.1:9999", "dry_run": True},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_sync_dry_run_unreachable(self, client):
        resp = await client.post(
            "/api/v1/models/sync",
            json={"source_url": "https://nonexistent.example.com", "dry_run": True},
        )
        assert resp.status_code == 502


class TestInferenceCompletions:
    @pytest.mark.asyncio
    async def test_completions_not_loaded(self, client):
        r = await client.post("/api/v1/models", json={"name": "comp-model"})
        model_id = r.json()["id"]
        resp = await client.post(
            f"/api/v1/inference/{model_id}/completions",
            json={"prompt": "hello"},
        )
        assert resp.status_code == 400
        assert "not loaded" in resp.json()["detail"].lower()


class TestInferenceEmbeddings:
    @pytest.mark.asyncio
    async def test_embeddings_not_loaded(self, client):
        r = await client.post("/api/v1/models", json={"name": "emb-model"})
        model_id = r.json()["id"]
        resp = await client.post(
            f"/api/v1/inference/{model_id}/embeddings",
            json={"input": "hello"},
        )
        assert resp.status_code == 400
        assert "not loaded" in resp.json()["detail"].lower()


class TestTenantCRUD:
    @pytest.mark.asyncio
    async def test_create_tenant(self, client):
        resp = await client.post("/api/v1/tenants", json={"name": "acme", "display_name": "Acme Corp"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "acme"
        assert data["display_name"] == "Acme Corp"
        assert data["is_active"] is True

    @pytest.mark.asyncio
    async def test_list_tenants(self, client):
        await client.post("/api/v1/tenants", json={"name": "t1"})
        await client.post("/api/v1/tenants", json={"name": "t2"})
        resp = await client.get("/api/v1/tenants")
        assert resp.status_code == 200
        assert len(resp.json()) >= 2

    @pytest.mark.asyncio
    async def test_get_tenant(self, client):
        create = await client.post("/api/v1/tenants", json={"name": "get-test"})
        tid = create.json()["id"]
        resp = await client.get(f"/api/v1/tenants/{tid}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "get-test"

    @pytest.mark.asyncio
    async def test_update_tenant(self, client):
        create = await client.post("/api/v1/tenants", json={"name": "upd-test"})
        tid = create.json()["id"]
        resp = await client.patch(f"/api/v1/tenants/{tid}", json={"display_name": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["display_name"] == "Updated"

    @pytest.mark.asyncio
    async def test_delete_tenant(self, client):
        create = await client.post("/api/v1/tenants", json={"name": "del-test"})
        tid = create.json()["id"]
        resp = await client.delete(f"/api/v1/tenants/{tid}")
        assert resp.status_code == 200
        resp2 = await client.get(f"/api/v1/tenants/{tid}")
        assert resp2.status_code == 404

    @pytest.mark.asyncio
    async def test_duplicate_tenant(self, client):
        await client.post("/api/v1/tenants", json={"name": "dup"})
        resp = await client.post("/api/v1/tenants", json={"name": "dup"})
        assert resp.status_code == 409


class TestWebhookCRUD:
    @pytest.mark.asyncio
    async def test_create_webhook(self, client):
        resp = await client.post("/api/v1/webhooks", json={"name": "wh1", "url": "https://example.com/hook", "events": "model.created"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "wh1"
        assert data["url"] == "https://example.com/hook"
        assert data["is_active"] is True

    @pytest.mark.asyncio
    async def test_list_webhooks(self, client):
        await client.post("/api/v1/webhooks", json={"name": "wh-a", "url": "https://a.com"})
        await client.post("/api/v1/webhooks", json={"name": "wh-b", "url": "https://b.com"})
        resp = await client.get("/api/v1/webhooks")
        assert resp.status_code == 200
        assert len(resp.json()) >= 2

    @pytest.mark.asyncio
    async def test_get_webhook(self, client):
        create = await client.post("/api/v1/webhooks", json={"name": "wh-get", "url": "https://c.com"})
        wid = create.json()["id"]
        resp = await client.get(f"/api/v1/webhooks/{wid}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "wh-get"

    @pytest.mark.asyncio
    async def test_delete_webhook(self, client):
        create = await client.post("/api/v1/webhooks", json={"name": "wh-del", "url": "https://d.com"})
        wid = create.json()["id"]
        resp = await client.delete(f"/api/v1/webhooks/{wid}")
        assert resp.status_code == 200
        resp2 = await client.get(f"/api/v1/webhooks/{wid}")
        assert resp2.status_code == 404


class TestDeploymentCRUD:
    @pytest.mark.asyncio
    async def test_create_deployment(self, client):
        model = await client.post("/api/v1/models", json={"name": "dep-model"})
        mid = model.json()["id"]
        resp = await client.post("/api/v1/deployments", json={"model_id": mid, "name": "dep1", "replicas": 2})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "dep1"
        assert data["replicas"] == 2
        assert data["status"] in ("pending", "running")

    @pytest.mark.asyncio
    async def test_list_deployments(self, client):
        model = await client.post("/api/v1/models", json={"name": "dep-list-m"})
        mid = model.json()["id"]
        await client.post("/api/v1/deployments", json={"model_id": mid, "name": "d1"})
        await client.post("/api/v1/deployments", json={"model_id": mid, "name": "d2"})
        resp = await client.get("/api/v1/deployments")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 2
        assert "deployments" in resp.json()

    @pytest.mark.asyncio
    async def test_update_deployment(self, client):
        model = await client.post("/api/v1/models", json={"name": "dep-upd-m"})
        mid = model.json()["id"]
        create = await client.post("/api/v1/deployments", json={"model_id": mid, "name": "d-upd"})
        did = create.json()["id"]
        resp = await client.patch(f"/api/v1/deployments/{did}", json={"replicas": 3, "status": "running"})
        assert resp.status_code == 200
        assert resp.json()["replicas"] == 3
        assert resp.json()["status"] == "running"

    @pytest.mark.asyncio
    async def test_delete_deployment(self, client):
        model = await client.post("/api/v1/models", json={"name": "dep-del-m"})
        mid = model.json()["id"]
        create = await client.post("/api/v1/deployments", json={"model_id": mid, "name": "d-del"})
        did = create.json()["id"]
        resp = await client.delete(f"/api/v1/deployments/{did}")
        assert resp.status_code == 200
        resp2 = await client.get(f"/api/v1/deployments/{did}")
        assert resp2.status_code == 404


class TestGrayReleaseAndScale:
    @pytest.mark.asyncio
    async def test_enable_gray_release(self, client):
        model = await client.post("/api/v1/models", json={"name": "gray-model"})
        mid = model.json()["id"]
        dep = await client.post("/api/v1/deployments", json={"model_id": mid, "name": "gray-dep"})
        did = dep.json()["id"]
        resp = await client.post(f"/api/v1/deployments/{did}/gray", json={"gray_version_id": "fake-ver-id", "gray_traffic_ratio": 20})
        assert resp.status_code == 200
        data = resp.json()
        assert data["gray_enabled"] is True
        assert data["gray_traffic_ratio"] == 20

    @pytest.mark.asyncio
    async def test_disable_gray_release(self, client):
        model = await client.post("/api/v1/models", json={"name": "gray-off-m"})
        mid = model.json()["id"]
        dep = await client.post("/api/v1/deployments", json={"model_id": mid, "name": "gray-off-dep"})
        did = dep.json()["id"]
        await client.post(f"/api/v1/deployments/{did}/gray", json={"gray_version_id": "some-ver", "gray_traffic_ratio": 10})
        resp = await client.delete(f"/api/v1/deployments/{did}/gray")
        assert resp.status_code == 200
        assert resp.json()["gray_enabled"] is False
        assert resp.json()["gray_traffic_ratio"] == 0

    @pytest.mark.asyncio
    async def test_scale_deployment(self, client):
        model = await client.post("/api/v1/models", json={"name": "scale-model"})
        mid = model.json()["id"]
        dep = await client.post("/api/v1/deployments", json={"model_id": mid, "name": "scale-dep"})
        did = dep.json()["id"]
        resp = await client.post(f"/api/v1/deployments/{did}/scale", json={"replicas": 5})
        assert resp.status_code == 200
        assert resp.json()["replicas"] == 5


class TestExportImport:
    @pytest.mark.asyncio
    async def test_export_data(self, client):
        await client.post("/api/v1/models", json={"name": "export-m"})
        resp = await client.get("/api/v1/system/export")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "tenants" in data
        assert "webhooks" in data
        assert data["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_import_data(self, client):
        payload = {
            "tenants": [{"name": "import-t", "display_name": "Imported"}],
            "models": [{"name": "import-m", "description": "test", "model_type": "llm"}],
            "webhooks": [{"name": "import-w", "url": "https://example.com/hook", "events": "model.created"}],
        }
        resp = await client.post("/api/v1/system/import", json=payload)
        assert resp.status_code == 200
        assert resp.json()["imported"] >= 3
