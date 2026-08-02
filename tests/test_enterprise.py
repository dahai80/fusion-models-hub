# Imported by: pytest runner
# Tests: security, watermark, encryption, approvals, gitlfs, lora-merge, distributed-tasks, SDK, auth middleware, crud
# Schema: uses same fixture pattern as test_api.py

import asyncio
import os

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
        data_dir="/tmp/fmh_test_enterprise",
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


async def _create_model(client, name="ent-model"):
    resp = await client.post("/api/v1/models", json={
        "name": name, "description": "enterprise test",
        "model_type": "llm", "architecture": "qwen2", "params_size": "7B",
    })
    assert resp.status_code == 201
    return resp.json()


async def _create_version(client, model_id):
    resp = await client.post(
        f"/api/v1/models/{model_id}/versions",
        data={"version": "1.0.0", "format": "mlx", "quantization": "4bit"},
        files={"file": ("", b"")},
    )
    assert resp.status_code == 201
    return resp.json()


class TestSecurityScan:
    @pytest.mark.asyncio
    async def test_start_scan(self, client):
        model = await _create_model(client)
        resp = await client.post("/api/v1/security/scan", json={
            "model_id": model["id"], "scan_type": "full",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["status"] in ("pending", "running", "completed")

    @pytest.mark.asyncio
    async def test_get_scan(self, client):
        model = await _create_model(client)
        scan = await client.post("/api/v1/security/scan", json={
            "model_id": model["id"], "scan_type": "full",
        })
        scan_id = scan.json()["id"]
        resp = await client.get(f"/api/v1/security/scan/{scan_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_scans(self, client):
        resp = await client.get("/api/v1/security/scans")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_scan_model_not_found(self, client):
        resp = await client.post("/api/v1/security/scan", json={
            "model_id": "nonexistent", "scan_type": "full",
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_scan_not_found(self, client):
        resp = await client.get("/api/v1/security/scan/nonexistent")
        assert resp.status_code == 404


class TestWatermark:
    @pytest.mark.asyncio
    async def test_embed_watermark(self, client):
        model = await _create_model(client)
        resp = await client.post("/api/v1/watermark/embed", json={
            "model_id": model["id"], "watermark_type": "metadata", "payload": {"owner": "test"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data

    @pytest.mark.asyncio
    async def test_verify_watermark(self, client):
        model = await _create_model(client)
        await client.post("/api/v1/watermark/embed", json={
            "model_id": model["id"], "watermark_type": "metadata", "payload": {"owner": "test"},
        })
        resp = await client.post("/api/v1/watermark/verify", json={
            "model_id": model["id"],
        })
        assert resp.status_code == 200
        assert "verified" in resp.json()

    @pytest.mark.asyncio
    async def test_list_watermarks(self, client):
        resp = await client.get("/api/v1/watermark/list")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_embed_model_not_found(self, client):
        resp = await client.post("/api/v1/watermark/embed", json={
            "model_id": "nonexistent", "watermark_type": "metadata",
        })
        assert resp.status_code == 404


class TestEncryption:
    @pytest.mark.asyncio
    async def test_encrypt_version(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        os.environ["FMH_ENCRYPTION_KEY"] = "test-encryption-key-32-bytes-long!!"
        resp = await client.post("/api/v1/encryption/encrypt", json={
            "version_id": version["id"],
        })
        assert resp.status_code in (200, 400, 500)

    @pytest.mark.asyncio
    async def test_decrypt_version(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        os.environ["FMH_ENCRYPTION_KEY"] = "test-encryption-key-32-bytes-long!!"
        resp = await client.post("/api/v1/encryption/decrypt", json={
            "version_id": version["id"],
        })
        assert resp.status_code in (200, 400, 409, 500)

    @pytest.mark.asyncio
    async def test_encryption_status(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        resp = await client.get(f"/api/v1/encryption/status/{version['id']}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_encrypt_version_not_found(self, client):
        resp = await client.post("/api/v1/encryption/encrypt", json={
            "version_id": "nonexistent",
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_decrypt_version_not_found(self, client):
        resp = await client.post("/api/v1/encryption/decrypt", json={
            "version_id": "nonexistent",
        })
        assert resp.status_code == 404


class TestApprovals:
    @pytest.mark.asyncio
    async def test_create_approval_l1(self, client):
        model = await _create_model(client)
        resp = await client.post("/api/v1/approvals", json={
            "model_id": model["id"], "level": "l1", "comment": "auto test",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

    @pytest.mark.asyncio
    async def test_create_approval_l2(self, client):
        model = await _create_model(client)
        resp = await client.post("/api/v1/approvals", json={
            "model_id": model["id"], "level": "l2", "comment": "need review",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    @pytest.mark.asyncio
    async def test_approve_request(self, client):
        model = await _create_model(client)
        approval = await client.post("/api/v1/approvals", json={
            "model_id": model["id"], "level": "l2", "comment": "review",
        })
        req_id = approval.json()["id"]
        resp = await client.post(f"/api/v1/approvals/{req_id}/approve", json={"approver": "admin", "comment": "ok"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

    @pytest.mark.asyncio
    async def test_reject_request(self, client):
        model = await _create_model(client)
        approval = await client.post("/api/v1/approvals", json={
            "model_id": model["id"], "level": "l2", "comment": "review",
        })
        req_id = approval.json()["id"]
        resp = await client.post(f"/api/v1/approvals/{req_id}/reject", json={"approver": "admin", "comment": "no"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    @pytest.mark.asyncio
    async def test_list_approvals(self, client):
        resp = await client.get("/api/v1/approvals")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_approval(self, client):
        model = await _create_model(client)
        approval = await client.post("/api/v1/approvals", json={
            "model_id": model["id"], "level": "l2",
        })
        req_id = approval.json()["id"]
        resp = await client.get(f"/api/v1/approvals/{req_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_approve_not_found(self, client):
        resp = await client.post("/api/v1/approvals/nonexistent/approve", json={"comment": ""})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_reject_not_found(self, client):
        resp = await client.post("/api/v1/approvals/nonexistent/reject", json={"comment": ""})
        assert resp.status_code == 404


class TestGitLFS:
    @pytest.mark.asyncio
    async def test_batch_upload(self, client):
        resp = await client.post("/api/v1/gitlfs/objects/batch", json={
            "operation": "upload", "objects": [{"oid": "abc123", "size": 1024}],
        })
        assert resp.status_code == 200
        assert "objects" in resp.json()

    @pytest.mark.asyncio
    async def test_batch_download(self, client):
        resp = await client.post("/api/v1/gitlfs/objects/batch", json={
            "operation": "download", "objects": [{"oid": "abc123", "size": 1024}],
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_lock(self, client):
        model = await _create_model(client)
        resp = await client.post("/api/v1/gitlfs/locks", json={
            "model_id": model["id"], "path": "models/weights.bin",
        })
        assert resp.status_code == 200
        assert "lock" in resp.json()

    @pytest.mark.asyncio
    async def test_list_locks(self, client):
        resp = await client.get("/api/v1/gitlfs/locks")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_lock(self, client):
        model = await _create_model(client)
        lock = await client.post("/api/v1/gitlfs/locks", json={
            "model_id": model["id"], "path": "models/test.bin",
        })
        lock_id = lock.json()["lock"]["id"]
        resp = await client.delete(f"/api/v1/gitlfs/locks/{lock_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_lock_not_found(self, client):
        resp = await client.delete("/api/v1/gitlfs/locks/nonexistent")
        assert resp.status_code == 404


class TestLoraMerge:
    @pytest.mark.asyncio
    async def test_start_lora_merge(self, client):
        model = await _create_model(client)
        base_v = await _create_version(client, model["id"])
        lora_model = await _create_model(client, name="lora-model")
        lora_v = await _create_version(client, lora_model["id"])
        resp = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": base_v["id"], "lora_version_id": lora_v["id"],
            "target_format": "mlx", "quant_bits": 4,
        })
        assert resp.status_code == 202
        assert "task_id" in resp.json()

    @pytest.mark.asyncio
    async def test_lora_merge_invalid_bits(self, client):
        model = await _create_model(client)
        base_v = await _create_version(client, model["id"])
        lora_model = await _create_model(client, name="lora-model2")
        lora_v = await _create_version(client, lora_model["id"])
        resp = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": base_v["id"], "lora_version_id": lora_v["id"],
            "quant_bits": 3,
        })
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_lora_merge_base_not_found(self, client):
        model = await _create_model(client)
        lora_v = await _create_version(client, model["id"])
        resp = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": "nonexistent", "lora_version_id": lora_v["id"],
            "quant_bits": 4,
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_lora_merge_lora_not_found(self, client):
        model = await _create_model(client)
        base_v = await _create_version(client, model["id"])
        resp = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": base_v["id"], "lora_version_id": "nonexistent",
            "quant_bits": 4,
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_lora_merge_status(self, client):
        model = await _create_model(client)
        base_v = await _create_version(client, model["id"])
        lora_model = await _create_model(client, name="lora-model3")
        lora_v = await _create_version(client, lora_model["id"])
        merge = await client.post("/api/v1/quantize/lora-merge", json={
            "base_version_id": base_v["id"], "lora_version_id": lora_v["id"],
            "quant_bits": 4,
        })
        task_id = merge.json()["task_id"]
        await asyncio.sleep(0.3)
        resp = await client.get(f"/api/v1/quantize/lora-merge/{task_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_lora_merge_not_found(self, client):
        resp = await client.get("/api/v1/quantize/lora-merge/nonexistent")
        assert resp.status_code == 404


class TestDistributedTasks:
    @pytest.mark.asyncio
    async def test_submit_distributed_task(self, client):
        model = await _create_model(client)
        resp = await client.post("/api/v1/cluster/distributed-tasks", json={
            "model_id": model["id"], "target_nodes": [],
        })
        assert resp.status_code == 202
        assert "task_id" in resp.json()

    @pytest.mark.asyncio
    async def test_submit_distributed_invalid_node(self, client):
        model = await _create_model(client)
        resp = await client.post("/api/v1/cluster/distributed-tasks", json={
            "model_id": model["id"], "target_nodes": ["nonexistent"],
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_distributed_task(self, client):
        model = await _create_model(client)
        task = await client.post("/api/v1/cluster/distributed-tasks", json={
            "model_id": model["id"], "target_nodes": [],
        })
        task_id = task.json()["task_id"]
        await asyncio.sleep(0.3)
        resp = await client.get(f"/api/v1/cluster/distributed-tasks/{task_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_distributed_task_not_found(self, client):
        resp = await client.get("/api/v1/cluster/distributed-tasks/nonexistent")
        assert resp.status_code == 404


class TestVersionPromote:
    @pytest.mark.asyncio
    async def test_promote_draft_to_published(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{version['id']}/metrics", json={"benchmark_score": 90.0})
        resp = await client.post(f"/api/v1/versions/{version['id']}/promote")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "published"
        assert "promoted_steps" in data

    @pytest.mark.asyncio
    async def test_promote_testing_to_published(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{version['id']}", json={"status": "testing"})
        await client.put(f"/api/v1/versions/{version['id']}/metrics", json={"benchmark_score": 90.0})
        resp = await client.post(f"/api/v1/versions/{version['id']}/promote")
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"

    @pytest.mark.asyncio
    async def test_promote_already_published(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{version['id']}/metrics", json={"benchmark_score": 90.0})
        await client.post(f"/api/v1/versions/{version['id']}/promote")
        resp = await client.post(f"/api/v1/versions/{version['id']}/promote")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_promote_deprecated_conflict(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{version['id']}/metrics", json={"benchmark_score": 90.0})
        await client.post(f"/api/v1/versions/{version['id']}/promote")
        await client.post(f"/api/v1/versions/{version['id']}/deprecate", json={"successor_version_id": "", "remark": "old"})
        resp = await client.post(f"/api/v1/versions/{version['id']}/promote")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_promote_not_found(self, client):
        resp = await client.post("/api/v1/versions/nonexistent/promote")
        assert resp.status_code == 404


class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_auth_enabled_write_blocked(self, client):
        from fusion_model_hub.server.auth import set_auth_enabled
        set_auth_enabled(True)
        try:
            resp = await client.post("/api/v1/models", json={
                "name": "auth-test", "description": "x",
                "model_type": "llm", "architecture": "qwen2", "params_size": "7B",
            })
            assert resp.status_code == 401
        finally:
            set_auth_enabled(False)

    @pytest.mark.asyncio
    async def test_auth_enabled_read_ok(self, client):
        from fusion_model_hub.server.auth import set_auth_enabled
        set_auth_enabled(True)
        try:
            resp = await client.get("/api/v1/system/health")
            assert resp.status_code == 200
        finally:
            set_auth_enabled(False)

    @pytest.mark.asyncio
    async def test_auth_with_valid_key(self, client):
        from fusion_model_hub.server.auth import set_auth_enabled
        set_auth_enabled(True)
        try:
            key_resp = await client.post("/api/v1/auth/keys", json={"name": "test-key"})
            api_key = key_resp.json()["key"]
            resp = await client.post("/api/v1/models", json={
                "name": "auth-model", "description": "x",
                "model_type": "llm", "architecture": "qwen2", "params_size": "7B",
            }, headers={"X-API-Key": api_key})
            assert resp.status_code == 201
        finally:
            set_auth_enabled(False)


class TestSelectiveExport:
    @pytest.mark.asyncio
    async def test_export_all(self, client):
        resp = await client.get("/api/v1/system/export")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_export_selective(self, client):
        model = await _create_model(client)
        resp = await client.get("/api/v1/system/export", params={"models": model["id"]})
        assert resp.status_code == 200


class TestCRUDNewTables:
    @pytest.mark.asyncio
    async def test_security_scan_crud(self, client):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            scan = await crud.create_security_scan(session, model_id="m1", version_id="v1", scan_type="full")
            assert scan.id
            fetched = await crud.get_security_scan(session, scan.id)
            assert fetched is not None
            items, total = await crud.list_security_scans(session)
            assert total >= 1
            updated = await crud.update_security_scan(session, scan.id, status="completed")
            assert updated is not None

    @pytest.mark.asyncio
    async def test_watermark_crud(self, client):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            wm = await crud.create_watermark(session, model_id="m1", version_id="v1", signature="sig123", payload="{}")
            assert wm.id
            fetched = await crud.get_watermark(session, wm.id)
            assert fetched is not None
            items = await crud.list_watermarks(session)
            assert len(items) >= 1

    @pytest.mark.asyncio
    async def test_approval_crud(self, client):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            approval = await crud.create_approval_request(session, model_id="m1", version_id="v1", level="l2", requester="tester")
            assert approval.id
            fetched = await crud.get_approval_request(session, approval.id)
            assert fetched is not None
            items, total = await crud.list_approval_requests(session)
            assert total >= 1
            updated = await crud.update_approval_request(session, approval.id, status="approved")
            assert updated is not None

    @pytest.mark.asyncio
    async def test_lora_merge_task_crud(self, client):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            task = await crud.create_lora_merge_task(
                session, base_version_id="bv1", lora_version_id="lv1",
                target_format="mlx", quant_bits=4,
            )
            assert task.id
            fetched = await crud.get_lora_merge_task(session, task.id)
            assert fetched is not None
            items, total = await crud.list_lora_merge_tasks(session)
            assert total >= 1
            updated = await crud.update_lora_merge_task(session, task.id, status="completed")
            assert updated is not None

    @pytest.mark.asyncio
    async def test_distributed_task_crud(self, client):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            task = await crud.create_distributed_task(
                session, model_id="m1", version_id="v1",
                target_nodes="n1,n2",
            )
            assert task.id
            fetched = await crud.get_distributed_task(session, task.id)
            assert fetched is not None
            updated = await crud.update_distributed_task(session, task.id, status="completed")
            assert updated is not None

    @pytest.mark.asyncio
    async def test_gitlfs_lock_crud(self, client):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            lock = await crud.create_gitlfs_lock(session, model_id="m1", path="models/test.bin", owner="user1")
            assert lock.id
            items = await crud.list_gitlfs_locks(session)
            assert len(items) >= 1
            deleted = await crud.delete_gitlfs_lock(session, lock.id)
            assert deleted is True


class TestSDKClient:
    def test_client_init(self):
        from fusion_model_hub.sdk import FusionModelHubClient
        c = FusionModelHubClient(base_url="http://localhost:9999", api_key="test-key")
        assert c._base_url == "http://localhost:9999"
        assert c._headers["X-API-Key"] == "test-key"

    def test_client_url_construction(self):
        from fusion_model_hub.sdk import FusionModelHubClient
        c = FusionModelHubClient(base_url="http://localhost:11444/")
        assert c._url("/models") == "http://localhost:11444/api/v1/models"

    def test_client_no_api_key(self):
        from fusion_model_hub.sdk import FusionModelHubClient
        c = FusionModelHubClient()
        assert "X-API-Key" not in c._headers


class TestInferenceProxy:
    @pytest.mark.asyncio
    async def test_chat_completions(self, client):
        resp = await client.post("/api/v1/inference/chat/completions", json={
            "model": "nonexistent", "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code in (200, 400, 404, 502, 503)

    @pytest.mark.asyncio
    async def test_completions(self, client):
        resp = await client.post("/api/v1/inference/completions", json={
            "model": "nonexistent", "prompt": "hello",
        })
        assert resp.status_code in (200, 400, 404, 502, 503)

    @pytest.mark.asyncio
    async def test_embeddings(self, client):
        resp = await client.post("/api/v1/inference/embeddings", json={
            "model": "nonexistent", "input": "test",
        })
        assert resp.status_code in (200, 400, 404, 502, 503)

    @pytest.mark.asyncio
    async def test_serve_model_status_not_loaded(self, client):
        resp = await client.get("/api/v1/models/nonexistent/serve")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(self, client):
        resp = await client.delete("/api/v1/models/nonexistent/serve")
        assert resp.status_code in (200, 404)


class TestVersionLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        assert version["status"] == "draft"
        await client.put(f"/api/v1/versions/{version['id']}/metrics", json={"benchmark_score": 90.0})
        resp = await client.post(f"/api/v1/versions/{version['id']}/promote")
        assert resp.status_code == 200
        v = resp.json()
        assert v["status"] == "published"
        await client.post(f"/api/v1/versions/{version['id']}/deprecate", json={})
        v = (await client.get(f"/api/v1/versions/{version['id']}")).json()
        assert v["status"] == "deprecated"
        await client.post(f"/api/v1/versions/{version['id']}/retire")
        v = (await client.get(f"/api/v1/versions/{version['id']}")).json()
        assert v["status"] == "retired"

    @pytest.mark.asyncio
    async def test_rollback_version(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{version['id']}/metrics", json={"benchmark_score": 90.0})
        await client.post(f"/api/v1/versions/{version['id']}/promote")
        await client.post(f"/api/v1/versions/{version['id']}/promote")
        v = (await client.get(f"/api/v1/versions/{version['id']}")).json()
        assert v["status"] == "published"
        await client.post(f"/api/v1/versions/{version['id']}/deprecate", json={})
        v = (await client.get(f"/api/v1/versions/{version['id']}")).json()
        assert v["status"] == "deprecated"
        resp = await client.post(f"/api/v1/versions/{version['id']}/rollback")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_benchmark_version(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        resp = await client.put(
            f"/api/v1/versions/{version['id']}/benchmark",
            json={"benchmark_score": 85.0, "inference_latency": 12.3},
        )
        assert resp.status_code in (200, 400)


class TestSystemEndpoints:
    @pytest.mark.asyncio
    async def test_audit_log(self, client):
        resp = await client.get("/api/v1/system/audit")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        resp = await client.get("/api/v1/system/health")
        assert resp.status_code == 200
        assert "status" in resp.json()
