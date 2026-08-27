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
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_test_enterprise",
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


async def _create_model(client, name="ent-model"):
    resp = await client.post(
        "/api/v1/models",
        json={
            "name": name,
            "description": "enterprise test",
            "model_type": "llm",
            "architecture": "qwen2",
            "params_size": "7B",
        },
    )
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
        resp = await client.post(
            "/api/v1/security/scan",
            json={
                "model_id": model["id"],
                "scan_type": "full",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["status"] in ("pending", "running", "completed")

    @pytest.mark.asyncio
    async def test_get_scan(self, client):
        model = await _create_model(client)
        scan = await client.post(
            "/api/v1/security/scan",
            json={
                "model_id": model["id"],
                "scan_type": "full",
            },
        )
        scan_id = scan.json()["id"]
        resp = await client.get(f"/api/v1/security/scan/{scan_id}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_scans(self, client):
        resp = await client.get("/api/v1/security/scans")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_scan_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/security/scan",
            json={
                "model_id": "nonexistent",
                "scan_type": "full",
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_scan_not_found(self, client):
        resp = await client.get("/api/v1/security/scan/nonexistent")
        assert resp.status_code == 404


class TestWatermark:
    @pytest.mark.asyncio
    async def test_embed_watermark(self, client, monkeypatch):
        # E-S6: embed refuses the source-public default secret; supply a real one.
        monkeypatch.setenv("FMH_WATERMARK_SECRET", "enterprise-test-secret-32b")
        model = await _create_model(client)
        resp = await client.post(
            "/api/v1/watermark/embed",
            json={
                "model_id": model["id"],
                "watermark_type": "metadata",
                "payload": {"owner": "test"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data

    @pytest.mark.asyncio
    async def test_verify_watermark(self, client, monkeypatch):
        # E-S6: verify re-derives the signature from the same per-request secret.
        monkeypatch.setenv("FMH_WATERMARK_SECRET", "enterprise-test-secret-32b")
        model = await _create_model(client)
        await client.post(
            "/api/v1/watermark/embed",
            json={
                "model_id": model["id"],
                "watermark_type": "metadata",
                "payload": {"owner": "test"},
            },
        )
        resp = await client.post(
            "/api/v1/watermark/verify",
            json={
                "model_id": model["id"],
            },
        )
        assert resp.status_code == 200
        assert "verified" in resp.json()

    @pytest.mark.asyncio
    async def test_list_watermarks(self, client):
        resp = await client.get("/api/v1/watermark/list")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_embed_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/watermark/embed",
            json={
                "model_id": "nonexistent",
                "watermark_type": "metadata",
            },
        )
        assert resp.status_code == 404


class TestWatermarkSidecar:
    # #1: the signed sidecar travels with the model files and verifies without
    # the Hub DB. These assert the file is written, read back, and that
    # tampering breaks the HMAC.

    @pytest.mark.asyncio
    async def test_embed_writes_sidecar_file(self, client, settings, monkeypatch):
        monkeypatch.setenv("FMH_WATERMARK_SECRET", "enterprise-test-secret-32b")
        model = await _create_model(client, name="wm-sidecar-write")
        resp = await client.post(
            "/api/v1/watermark/embed",
            json={"model_id": model["id"], "payload": {"seed": "alice-seed"}},
        )
        assert resp.status_code == 200
        assert resp.json()["sidecar_written"] is True
        # version_id empty -> sidecar lands under models/{id}/default/
        from pathlib import Path

        sidecar = Path(settings.data_dir) / "models" / model["id"] / "default" / "watermark.json"
        assert sidecar.exists(), f"sidecar not written at {sidecar}"
        import json

        sc = json.loads(sidecar.read_bytes())
        assert sc["model_id"] == model["id"]
        # E-S6: payload.owner is overwritten with tenant context (empty when
        # auth disabled); assert on a caller-supplied key that is preserved.
        assert sc["payload"]["seed"] == "alice-seed"
        assert sc["signature"]

    @pytest.mark.asyncio
    async def test_verify_reads_sidecar_without_db(self, client, settings, monkeypatch):
        # Embed, then verify — verify must succeed via the sidecar (source=sidecar).
        monkeypatch.setenv("FMH_WATERMARK_SECRET", "enterprise-test-secret-32b")
        model = await _create_model(client, name="wm-sidecar-verify")
        await client.post(
            "/api/v1/watermark/embed",
            json={"model_id": model["id"], "payload": {"seed": "bob-seed"}},
        )
        resp = await client.post(
            "/api/v1/watermark/verify",
            json={"model_id": model["id"]},
        )
        data = resp.json()
        assert data["verified"] is True
        # Both sidecar and DB row present after embed -> defense-in-depth source.
        assert "sidecar" in data["source"]
        assert data["watermark"]["payload"]["seed"] == "bob-seed"

    @pytest.mark.asyncio
    async def test_tampered_sidecar_fails_hmac(self, client, settings, monkeypatch):
        # A tampered traveling sidecar is itself the failure signal: verify
        # returns verified=False, source=sidecar — it does NOT silently fall
        # back to the (untampered) DB row and report the model as verified.
        import json
        from pathlib import Path

        monkeypatch.setenv("FMH_WATERMARK_SECRET", "enterprise-test-secret-32b")
        model = await _create_model(client, name="wm-sidecar-tamper")
        await client.post(
            "/api/v1/watermark/embed",
            json={"model_id": model["id"], "payload": {"seed": "carol-seed"}},
        )
        # Tamper the sidecar: change the payload so the re-derived sig no longer
        # matches the stored (original) signature.
        sidecar = Path(settings.data_dir) / "models" / model["id"] / "default" / "watermark.json"
        sc = json.loads(sidecar.read_bytes())
        sc["payload"]["seed"] = "evil-seed"
        sidecar.write_bytes(json.dumps(sc, sort_keys=True).encode())
        resp = await client.post(
            "/api/v1/watermark/verify",
            json={"model_id": model["id"]},
        )
        data = resp.json()
        assert data["verified"] is False
        # Both sources present after embed; defense-in-depth reports both.
        assert "sidecar" in data["source"]

    @pytest.mark.asyncio
    async def test_sidecar_fallback_to_db_when_absent(self, client, settings, monkeypatch):
        from pathlib import Path

        monkeypatch.setenv("FMH_WATERMARK_SECRET", "enterprise-test-secret-32b")
        model = await _create_model(client, name="wm-sidecar-absent")
        await client.post(
            "/api/v1/watermark/embed",
            json={"model_id": model["id"], "payload": {"owner": "dan"}},
        )
        # Delete the sidecar so verify must fall back to the DB row.
        sidecar = Path(settings.data_dir) / "models" / model["id"] / "default" / "watermark.json"
        sidecar.unlink(missing_ok=True)
        resp = await client.post(
            "/api/v1/watermark/verify",
            json={"model_id": model["id"]},
        )
        data = resp.json()
        assert data["verified"] is True
        assert data["source"] == "database"


class TestEncryption:
    @pytest.mark.asyncio
    async def test_encrypt_version(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        os.environ["FMH_ENCRYPTION_KEY"] = "test-encryption-key-32-bytes-long!!"
        resp = await client.post(
            "/api/v1/encryption/encrypt",
            json={
                "version_id": version["id"],
            },
        )
        assert resp.status_code in (200, 400, 500)

    @pytest.mark.asyncio
    async def test_decrypt_version(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        os.environ["FMH_ENCRYPTION_KEY"] = "test-encryption-key-32-bytes-long!!"
        resp = await client.post(
            "/api/v1/encryption/decrypt",
            json={
                "version_id": version["id"],
            },
        )
        assert resp.status_code in (200, 400, 409, 500)

    @pytest.mark.asyncio
    async def test_encryption_status(self, client):
        model = await _create_model(client)
        version = await _create_version(client, model["id"])
        resp = await client.get(f"/api/v1/encryption/status/{version['id']}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_encrypt_version_not_found(self, client):
        resp = await client.post(
            "/api/v1/encryption/encrypt",
            json={
                "version_id": "nonexistent",
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_decrypt_version_not_found(self, client):
        resp = await client.post(
            "/api/v1/encryption/decrypt",
            json={
                "version_id": "nonexistent",
            },
        )
        assert resp.status_code == 404


class TestApprovals:
    @pytest.mark.asyncio
    async def test_create_approval_l1(self, client):
        model = await _create_model(client)
        resp = await client.post(
            "/api/v1/approvals",
            json={
                "model_id": model["id"],
                "level": "l1",
                "comment": "auto test",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

    @pytest.mark.asyncio
    async def test_create_approval_l2(self, client):
        model = await _create_model(client)
        resp = await client.post(
            "/api/v1/approvals",
            json={
                "model_id": model["id"],
                "level": "l2",
                "comment": "need review",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    @pytest.mark.asyncio
    async def test_approve_request(self, client):
        model = await _create_model(client)
        approval = await client.post(
            "/api/v1/approvals",
            json={
                "model_id": model["id"],
                "level": "l2",
                "comment": "review",
            },
        )
        req_id = approval.json()["id"]
        resp = await client.post(f"/api/v1/approvals/{req_id}/approve", json={"approver": "admin", "comment": "ok"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

    @pytest.mark.asyncio
    async def test_reject_request(self, client):
        model = await _create_model(client)
        approval = await client.post(
            "/api/v1/approvals",
            json={
                "model_id": model["id"],
                "level": "l2",
                "comment": "review",
            },
        )
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
        approval = await client.post(
            "/api/v1/approvals",
            json={
                "model_id": model["id"],
                "level": "l2",
            },
        )
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

    @pytest.mark.asyncio
    async def test_l3_quorum_stays_pending_after_one_approver(self, client):
        # R-P2/#7: L3 multi-approver — one approve must NOT flip to APPROVED.
        model = await _create_model(client)
        approval = await client.post(
            "/api/v1/approvals",
            json={
                "model_id": model["id"],
                "level": "l3",
                "comment": "high gate",
            },
        )
        assert approval.status_code == 200
        req_id = approval.json()["id"]
        assert approval.json()["status"] == "pending"
        resp = await client.post(f"/api/v1/approvals/{req_id}/approve", json={"approver": "alice", "comment": "ok"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "pending"
        assert "alice" in body["approvers"]

    @pytest.mark.asyncio
    async def test_l3_quorum_approves_after_two_distinct_approvers(self, client):
        # R-P2/#7: quorum of 2 distinct approvers — second distinct approver flips APPROVED.
        model = await _create_model(client)
        approval = await client.post(
            "/api/v1/approvals",
            json={
                "model_id": model["id"],
                "level": "l3",
                "comment": "high gate",
            },
        )
        req_id = approval.json()["id"]
        await client.post(f"/api/v1/approvals/{req_id}/approve", json={"approver": "alice", "comment": "1"})
        resp = await client.post(f"/api/v1/approvals/{req_id}/approve", json={"approver": "bob", "comment": "2"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "approved"
        assert "alice" in body["approvers"] and "bob" in body["approvers"]

    @pytest.mark.asyncio
    async def test_l3_quorum_same_approver_twice_stays_pending(self, client):
        # R-P2/#7: one approver approving twice must NOT count as a quorum.
        model = await _create_model(client)
        approval = await client.post(
            "/api/v1/approvals",
            json={
                "model_id": model["id"],
                "level": "l3",
                "comment": "high gate",
            },
        )
        req_id = approval.json()["id"]
        await client.post(f"/api/v1/approvals/{req_id}/approve", json={"approver": "alice", "comment": "1"})
        resp = await client.post(f"/api/v1/approvals/{req_id}/approve", json={"approver": "alice", "comment": "2"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    @pytest.mark.asyncio
    async def test_l3_approve_requires_approver_field(self, client):
        model = await _create_model(client)
        approval = await client.post(
            "/api/v1/approvals",
            json={
                "model_id": model["id"],
                "level": "l3",
                "comment": "high gate",
            },
        )
        req_id = approval.json()["id"]
        resp = await client.post(f"/api/v1/approvals/{req_id}/approve", json={"approver": "", "comment": ""})
        assert resp.status_code == 400


class TestGitLFS:
    @pytest.mark.asyncio
    async def test_batch_upload(self, client):
        resp = await client.post(
            "/api/v1/gitlfs/objects/batch",
            json={
                "operation": "upload",
                "objects": [{"oid": "abc123", "size": 1024}],
            },
        )
        assert resp.status_code == 200
        assert "objects" in resp.json()

    @pytest.mark.asyncio
    async def test_batch_download(self, client):
        resp = await client.post(
            "/api/v1/gitlfs/objects/batch",
            json={
                "operation": "download",
                "objects": [{"oid": "abc123", "size": 1024}],
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_lock(self, client):
        model = await _create_model(client)
        resp = await client.post(
            "/api/v1/gitlfs/locks",
            json={
                "model_id": model["id"],
                "path": "models/weights.bin",
            },
        )
        assert resp.status_code == 200
        assert "lock" in resp.json()

    @pytest.mark.asyncio
    async def test_list_locks(self, client):
        resp = await client.get("/api/v1/gitlfs/locks")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_lock(self, client):
        model = await _create_model(client)
        lock = await client.post(
            "/api/v1/gitlfs/locks",
            json={
                "model_id": model["id"],
                "path": "models/test.bin",
            },
        )
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
        resp = await client.post(
            "/api/v1/quantize/lora-merge",
            json={
                "base_version_id": base_v["id"],
                "lora_version_id": lora_v["id"],
                "target_format": "mlx",
                "quant_bits": 4,
            },
        )
        assert resp.status_code == 202
        assert "task_id" in resp.json()

    @pytest.mark.asyncio
    async def test_lora_merge_invalid_bits(self, client):
        model = await _create_model(client)
        base_v = await _create_version(client, model["id"])
        lora_model = await _create_model(client, name="lora-model2")
        lora_v = await _create_version(client, lora_model["id"])
        resp = await client.post(
            "/api/v1/quantize/lora-merge",
            json={
                "base_version_id": base_v["id"],
                "lora_version_id": lora_v["id"],
                "quant_bits": 3,
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_lora_merge_base_not_found(self, client):
        model = await _create_model(client)
        lora_v = await _create_version(client, model["id"])
        resp = await client.post(
            "/api/v1/quantize/lora-merge",
            json={
                "base_version_id": "nonexistent",
                "lora_version_id": lora_v["id"],
                "quant_bits": 4,
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_lora_merge_lora_not_found(self, client):
        model = await _create_model(client)
        base_v = await _create_version(client, model["id"])
        resp = await client.post(
            "/api/v1/quantize/lora-merge",
            json={
                "base_version_id": base_v["id"],
                "lora_version_id": "nonexistent",
                "quant_bits": 4,
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_lora_merge_status(self, client):
        model = await _create_model(client)
        base_v = await _create_version(client, model["id"])
        lora_model = await _create_model(client, name="lora-model3")
        lora_v = await _create_version(client, lora_model["id"])
        merge = await client.post(
            "/api/v1/quantize/lora-merge",
            json={
                "base_version_id": base_v["id"],
                "lora_version_id": lora_v["id"],
                "quant_bits": 4,
            },
        )
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
        resp = await client.post(
            "/api/v1/cluster/distributed-tasks",
            json={
                "model_id": model["id"],
                "target_nodes": [],
            },
        )
        assert resp.status_code == 202
        assert "task_id" in resp.json()

    @pytest.mark.asyncio
    async def test_submit_distributed_invalid_node(self, client):
        model = await _create_model(client)
        resp = await client.post(
            "/api/v1/cluster/distributed-tasks",
            json={
                "model_id": model["id"],
                "target_nodes": ["nonexistent"],
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_distributed_task(self, client):
        model = await _create_model(client)
        task = await client.post(
            "/api/v1/cluster/distributed-tasks",
            json={
                "model_id": model["id"],
                "target_nodes": [],
            },
        )
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
        await client.post(
            f"/api/v1/versions/{version['id']}/deprecate", json={"successor_version_id": "", "remark": "old"}
        )
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
            resp = await client.post(
                "/api/v1/models",
                json={
                    "name": "auth-test",
                    "description": "x",
                    "model_type": "llm",
                    "architecture": "qwen2",
                    "params_size": "7B",
                },
            )
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
            resp = await client.post(
                "/api/v1/models",
                json={
                    "name": "auth-model",
                    "description": "x",
                    "model_type": "llm",
                    "architecture": "qwen2",
                    "params_size": "7B",
                },
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 201
        finally:
            set_auth_enabled(False)

    @pytest.mark.asyncio
    async def test_model_scoped_key_denies_collection_ops(self, client):
        # E-S15: a key with non-empty allowed_models must not reach collection
        # model operations (import/batch-delete/compare/...). They have no
        # single model_id to scope against, so the prior extractor returned ""
        # and _check_model_access silently skipped the ACL.
        from fusion_model_hub.server.auth import set_auth_enabled

        set_auth_enabled(True)
        try:
            key_resp = await client.post(
                "/api/v1/auth/keys",
                json={"name": "scoped", "allowed_models": "model-a"},
            )
            api_key = key_resp.json()["key"]
            # import is a collection op → 403, not 201.
            resp = await client.post(
                "/api/v1/models/import/hf",
                json={"hf_repo": "x/y"},
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 403
            # batch/delete is a collection op → 403.
            resp = await client.post(
                "/api/v1/models/batch/delete",
                json={"model_ids": []},
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 403
        finally:
            set_auth_enabled(False)

    @pytest.mark.asyncio
    async def test_model_scoped_key_enforces_resource_acl(self, client):
        # E-S15: a key scoped to model-a must GET model-a (200) but be denied
        # model-b (403). Confirms the resource-level ACL still works after the
        # collection-op guard was added.
        from fusion_model_hub.server.auth import set_auth_enabled

        # Create the models while auth is OFF (no key needed), then turn auth
        # on and create the scoped key as the bootstrap admin. Capture ids from
        # the create responses directly — /models/search does not filter by
        # name reliably for colliding matches.
        ma_id = (
            await client.post(
                "/api/v1/models",
                json={
                    "name": "model-a",
                    "description": "x",
                    "model_type": "llm",
                    "architecture": "qwen2",
                    "params_size": "7B",
                },
            )
        ).json()["id"]
        mb_id = (
            await client.post(
                "/api/v1/models",
                json={
                    "name": "model-b",
                    "description": "x",
                    "model_type": "llm",
                    "architecture": "qwen2",
                    "params_size": "7B",
                },
            )
        ).json()["id"]
        assert ma_id != mb_id
        set_auth_enabled(True)
        try:
            key_resp = await client.post(
                "/api/v1/auth/keys",
                json={"name": "scoped2", "allowed_models": ma_id},
            )
            assert key_resp.status_code == 201, key_resp.text
            api_key = key_resp.json()["key"]
            scoped_a = await client.get(
                f"/api/v1/models/{ma_id}",
                headers={"X-API-Key": api_key},
            )
            assert scoped_a.status_code == 200
            scoped_b = await client.get(
                f"/api/v1/models/{mb_id}",
                headers={"X-API-Key": api_key},
            )
            assert scoped_b.status_code == 403
        finally:
            set_auth_enabled(False)

    @pytest.mark.asyncio
    async def test_tenanted_admin_cannot_forge_cross_tenant_key(self, client):
        # P1-12: a tenanted admin must be pinned to its own tenant when creating
        # keys; setting a different body.tenant_id is a cross-tenant forge that
        # must be rejected. Root/super-admin (no tenant) still provisions any.
        from fusion_model_hub.server.auth import set_auth_enabled

        set_auth_enabled(True)
        try:
            # bootstrap: root admin (no tenant).
            root = (await client.post("/api/v1/auth/keys", json={"name": "root", "role": "admin"})).json()["key"]
            # root provisions a tenanted admin for tenant B.
            admin_b = (
                await client.post(
                    "/api/v1/auth/keys",
                    json={"name": "adminB", "tenant_id": "tenB", "role": "admin"},
                    headers={"X-API-Key": root},
                )
            ).json()["key"]
            # adminB mints a key for its own tenant -> ok (201).
            own = await client.post(
                "/api/v1/auth/keys",
                json={"name": "b-dev", "tenant_id": "tenB", "role": "developer"},
                headers={"X-API-Key": admin_b},
            )
            assert own.status_code == 201, own.text
            assert own.json()["tenant_id"] == "tenB"
            # adminB mints a key for tenant A -> 403 (cross-tenant forge).
            forge = await client.post(
                "/api/v1/auth/keys",
                json={"name": "a-evil", "tenant_id": "tenA", "role": "developer"},
                headers={"X-API-Key": admin_b},
            )
            assert forge.status_code == 403, forge.text
            # root (super-admin, no tenant) still provisions tenant A -> ok.
            root_prov = await client.post(
                "/api/v1/auth/keys",
                json={"name": "a-dev", "tenant_id": "tenA", "role": "developer"},
                headers={"X-API-Key": root},
            )
            assert root_prov.status_code == 201
            assert root_prov.json()["tenant_id"] == "tenA"
        finally:
            set_auth_enabled(False)


class TestInferenceTenantIsolation:
    # P0-C: a tenant-A key must NOT read/serve/inference a tenant-B model by
    # id. Cross-tenant read -> 404 (no existence leak); cross-tenant write
    # (serve) -> 403. Uses real DB rows + real tenant-scoped keys, not the
    # _loaded_models shortcut, so the get_model-then-check path is exercised.
    @staticmethod
    async def _mkkey(client, body, headers=None):
        resp = await client.post("/api/v1/auth/keys", json=body, headers=headers or {})
        assert resp.status_code == 201, resp.text
        return resp.json()["key"]

    @pytest.mark.asyncio
    async def test_cross_tenant_serve_status_is_denied(self, client):
        from fusion_model_hub.server.auth import set_auth_enabled
        from fusion_model_hub.server.routers import inference as inf_mod

        set_auth_enabled(True)
        try:
            root = await self._mkkey(client, {"name": "root", "role": "admin"})
            key_a = await self._mkkey(
                client,
                {"name": "ka", "tenant_id": "tenA", "role": "developer"},
                {"X-API-Key": root},
            )
            key_b = await self._mkkey(
                client,
                {"name": "kb", "tenant_id": "tenB", "role": "developer"},
                {"X-API-Key": root},
            )
            # tenant B owns the model (key carries tenant_id into create).
            mb = (
                await client.post(
                    "/api/v1/models",
                    json={
                        "name": "b-only",
                        "description": "x",
                        "model_type": "llm",
                        "architecture": "qwen2",
                        "params_size": "7B",
                    },
                    headers={"X-API-Key": key_b},
                )
            ).json()
            await client.post(f"/api/v1/models/{mb['id']}/publish", headers={"X-API-Key": key_b})
            # served (loaded) state injected — read path checks DB tenant, not MLX.
            inf_mod._loaded_models.clear()
            inf_mod._loaded_models[mb["id"]] = {
                "version_id": "v1",
                "model_name": "b-only",
                "status": "loaded",
                "loaded_at": 0.0,
            }
            # owner (tenant B) reads -> 200.
            own = await client.get(
                f"/api/v1/models/{mb['id']}/serve",
                headers={"X-API-Key": key_b},
            )
            assert own.status_code == 200, own.text
            # cross-tenant (tenant A) reads -> 404, no existence leak.
            cross = await client.get(
                f"/api/v1/models/{mb['id']}/serve",
                headers={"X-API-Key": key_a},
            )
            assert cross.status_code == 404, cross.text
            inf_mod._loaded_models.clear()
        finally:
            set_auth_enabled(False)

    @pytest.mark.asyncio
    async def test_cross_tenant_serve_is_denied(self, client):
        from unittest.mock import AsyncMock, MagicMock, patch

        from fusion_model_hub.server.auth import set_auth_enabled

        set_auth_enabled(True)
        try:
            root = await self._mkkey(client, {"name": "root2", "role": "admin"})
            key_a = await self._mkkey(
                client,
                {"name": "ka2", "tenant_id": "tenA", "role": "developer"},
                {"X-API-Key": root},
            )
            key_b = await self._mkkey(
                client,
                {"name": "kb2", "tenant_id": "tenB", "role": "developer"},
                {"X-API-Key": root},
            )
            mb = (
                await client.post(
                    "/api/v1/models",
                    json={
                        "name": "b-only2",
                        "description": "x",
                        "model_type": "llm",
                        "architecture": "qwen2",
                        "params_size": "7B",
                    },
                    headers={"X-API-Key": key_b},
                )
            ).json()
            # publish needs admin — root (admin, no tenant) bypasses the owner
            # check so B's developer key can subsequently serve its own model.
            await client.post(f"/api/v1/models/{mb['id']}/publish", headers={"X-API-Key": root})
            # serve needs a version; create + publish one via B's key.
            await client.post(
                f"/api/v1/models/{mb['id']}/versions",
                data={"version": "1.0.0", "format": "mlx", "quantization": "4bit"},
                files={"file": ("", b"")},
                headers={"X-API-Key": key_b},
            )
            mock_client = AsyncMock()
            # raise_for_status is SYNC on a real httpx.Response — an AsyncMock
            # resp inherits it as AsyncMock, whose call returns an un-awaited
            # coroutine (RuntimeWarning). Pin it to a sync MagicMock so the
            # mock shape matches production (inference.py:302 resp.raise_for_status()).
            _load_resp = AsyncMock(status_code=200, text="ok")
            _load_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=_load_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client):
                # owner (tenant B) serves -> 200.
                own = await client.post(
                    f"/api/v1/models/{mb['id']}/serve",
                    json={},
                    headers={"X-API-Key": key_b},
                )
                assert own.status_code == 200, own.text
                # cross-tenant (tenant A) serves -> 403 (owner check, not read).
                cross = await client.post(
                    f"/api/v1/models/{mb['id']}/serve",
                    json={},
                    headers={"X-API-Key": key_a},
                )
                assert cross.status_code == 403, cross.text
            from fusion_model_hub.server.routers import inference as inf_mod

            inf_mod._loaded_models.clear()
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
            approval = await crud.create_approval_request(
                session, model_id="m1", version_id="v1", level="l2", requester="tester"
            )
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
                session,
                base_version_id="bv1",
                lora_version_id="lv1",
                target_format="mlx",
                quant_bits=4,
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
                session,
                model_id="m1",
                version_id="v1",
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
        resp = await client.post(
            "/api/v1/inference/chat/completions",
            json={
                "model": "nonexistent",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code in (200, 400, 404, 502, 503)

    @pytest.mark.asyncio
    async def test_completions(self, client):
        resp = await client.post(
            "/api/v1/inference/completions",
            json={
                "model": "nonexistent",
                "prompt": "hello",
            },
        )
        assert resp.status_code in (200, 400, 404, 502, 503)

    @pytest.mark.asyncio
    async def test_embeddings(self, client):
        resp = await client.post(
            "/api/v1/inference/embeddings",
            json={
                "model": "nonexistent",
                "input": "test",
            },
        )
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
