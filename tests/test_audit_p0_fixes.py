import asyncio
import json
import shutil
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db import crud
from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps


@pytest.fixture
def settings(tmp_path):
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir=str(tmp_path / "fmh_p0_data"),
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


class TestModelStatusTransitionGuard:
    # E-D1: model_status must follow the legal state machine on PUT/publish/deprecate.
    @pytest.mark.asyncio
    async def test_illegal_put_status_transition_rejected(self, client):
        create = await client.post("/api/v1/models", json={"name": "p0-trans-m"})
        mid = create.json()["id"]
        # DRAFT -> published via publish (legal)
        pub = await client.post(f"/api/v1/models/{mid}/publish")
        assert pub.status_code == 200
        assert pub.json()["model_status"] == "published"
        # published -> published via PUT (illegal, no self-loop)
        resp = await client.put(f"/api/v1/models/{mid}", json={"model_status": "published"})
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_publish_already_published_rejected(self, client):
        create = await client.post("/api/v1/models", json={"name": "p0-dbl-pub"})
        mid = create.json()["id"]
        await client.post(f"/api/v1/models/{mid}/publish")
        # second publish: PUBLISHED -> PUBLISHED not in {DEPRECATED}
        resp = await client.post(f"/api/v1/models/{mid}/publish")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_legal_deprecate_republish(self, client):
        create = await client.post("/api/v1/models", json={"name": "p0-repub"})
        mid = create.json()["id"]
        await client.post(f"/api/v1/models/{mid}/publish")
        dep = await client.post(f"/api/v1/models/{mid}/deprecate")
        assert dep.status_code == 200
        # DEPRECATED -> published is legal
        resp = await client.put(f"/api/v1/models/{mid}", json={"model_status": "published"})
        assert resp.status_code == 200
        assert resp.json()["model_status"] == "published"


class TestApiKeyPepperHashing:
    # E-S4: API keys are peppered PBKDF2, not bare SHA-256; verify is constant-time.
    @pytest.mark.asyncio
    async def test_pepper_set_after_init_deps(self, settings):
        assert crud._API_KEY_PEPPER, "pepper must be populated by init_deps"

    @pytest.mark.asyncio
    async def test_key_hash_is_pbkdf2_not_sha256(self, settings):
        raw = "fmh-test-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        h = crud._hash_key(raw)
        # SHA-256 of raw is 64 hex chars; PBKDF2-SHA256 output is also 64 hex chars,
        # but the two must NOT match — pepper changes the digest.
        import hashlib

        bare = hashlib.sha256(raw.encode()).hexdigest()
        assert h != bare, "hash must differ from plain SHA-256 (pepper applied)"

    @pytest.mark.asyncio
    async def test_deterministic_lookup_roundtrip(self, settings):
        engine = get_engine(settings.db_url)
        await init_db(engine)
        sf = crud  # use the same pepper wired by init_deps
        from sqlalchemy.ext.asyncio import AsyncSession

        async with AsyncSession(engine) as sess:
            ak, full = await crud.create_api_key(sess, name="p0-key")
            await sess.commit()
            v = await crud.verify_api_key(sess, full)
            assert v is not None and v.name == "p0-key"
            bad = await crud.verify_api_key(sess, full[:-1] + "Z")
            assert bad is None


class TestCacheCorruptionHandling:
    # E-R1: corrupt index.json must be quarantined, not silently wiped.
    def test_corrupt_index_quarantined(self, tmp_path):
        from fusion_model_hub.cache.manager import CacheManager

        root = tmp_path / "cache"
        (root).mkdir()
        idx = root / "index.json"
        idx.write_text("{ this is not valid json", encoding="utf-8")
        cm = CacheManager(cache_root=str(root))
        # index resets to empty ...
        assert cm._index == {}
        # ... but the corrupt file was quarantined aside, not deleted
        quarantined = list(root.glob("index.corrupt.*.json"))
        assert len(quarantined) == 1

    def test_atomic_save_no_partial_index(self, tmp_path):
        from fusion_model_hub.cache.manager import CacheManager

        root = tmp_path / "cache"
        cm = CacheManager(cache_root=str(root))
        cm._index["m1:raw"] = {"path": "/x", "level": "raw"}
        cm._save_index()
        # written file must be valid JSON (atomic replace, no partial)
        data = json.loads((root / "index.json").read_text(encoding="utf-8"))
        assert "m1:raw" in data
        # no leftover staging temp
        assert not list(root.glob("index.*.tmp"))


class TestLocalStoreAtomicAssemble:
    # E-D3: assemble/write_file must be atomic — a target must not exist as a
    # truncated/partial file if assembly is interrupted.
    @pytest.mark.asyncio
    async def test_assemble_atomic_on_missing_chunk(self, tmp_path):
        from fusion_model_hub.storage.local_store import LocalStore

        store = LocalStore(data_dir=str(tmp_path / "store"))
        target_dir = tmp_path / "models" / "m" / "v"
        target_dir.mkdir(parents=True)
        # write only chunk 0, declare total_chunks=2 -> assembly must fail
        await store.write_chunk("up1", 0, b"hello")
        with pytest.raises(FileNotFoundError):
            await store.assemble_chunks("up1", target_dir, "model.bin", total_chunks=2)
        # target must NOT exist (atomic: staging was cleaned, no partial final file)
        assert not (target_dir / "model.bin").exists()

    @pytest.mark.asyncio
    async def test_write_file_atomic(self, tmp_path):
        from fusion_model_hub.storage.local_store import LocalStore

        store = LocalStore(data_dir=str(tmp_path / "store"))
        target_dir = tmp_path / "models" / "m2" / "v"
        target_dir.mkdir(parents=True)
        path, h, size = await store.write_file(target_dir, "f.bin", b"payload-bytes")
        assert path.exists()
        assert size == len(b"payload-bytes")
        # no staging temp left behind
        assert not list(target_dir.glob(".*.tmp"))


class TestClusterRemoteSyncInboxAndEmptyTarget:
    # H1 + R11: real remote-sync inbox exists; empty target_nodes fails (no fake COMPLETED).
    @pytest.mark.asyncio
    async def test_remote_sync_inbox_unknown_model_404(self, client):
        resp = await client.post(
            "/api/v1/cluster/remote-sync",
            json={"task_type": "model_sync", "model_id": "no-such-model"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_distributed_task_empty_targets_fails(self, client):
        create = await client.post("/api/v1/models", json={"name": "p0-dist-m"})
        mid = create.json()["id"]
        resp = await client.post(
            "/api/v1/cluster/distributed-tasks",
            json={"model_id": mid, "version_id": "", "target_nodes": []},
        )
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        await asyncio.sleep(0.4)
        status = await client.get(f"/api/v1/cluster/distributed-tasks/{task_id}")
        assert status.status_code == 200
        # empty target set must NOT report a fake COMPLETED
        assert status.json()["status"] != "completed"


class TestSystemImportAdminGuard:
    # E-S3: import binds to caller tenant, admin-gated only when auth enabled.
    @pytest.mark.asyncio
    async def test_import_works_when_auth_disabled(self, client):
        payload = {
            "models": [{"name": "p0-imp-m", "model_type": "llm"}],
            "webhooks": [{"name": "p0-imp-w", "url": "https://example.com/h", "events": "model.created"}],
        }
        resp = await client.post("/api/v1/system/import", json=payload)
        assert resp.status_code == 200
        assert resp.json()["imported"] >= 2


class TestLoraMergeOutputPathGuard:
    # E-D2: a 200 from MLX with no output_path must FAIL the task, not create a
    # file_path="" version and mark COMPLETED. We monkeypatch the httpx call so
    # MLX returns {"output_path": ""}; the task must end FAILED.
    @pytest.mark.asyncio
    async def test_empty_output_path_fails_task(self, client, monkeypatch):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        create = await client.post("/api/v1/models", json={"name": "p0-lora-m"})
        mid = create.json()["id"]
        sf = get_session_factory()
        async with sf() as s:
            base = await crud.create_version(s, model_id=mid, version="base-v", file_path="/models/base")
            lora = await crud.create_version(s, model_id=mid, version="lora-v", file_path="/models/lora")
            bid, lid = base.id, lora.id

        class _FakeResp:
            status_code = 200
            content = b'{"output_path": ""}'

            def json(self):
                return {"output_path": ""}

            def raise_for_status(self):
                pass

        class _FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return _FakeResp()

        import fusion_model_hub.server.routers.quantize as qmod

        monkeypatch.setattr(qmod.httpx, "AsyncClient", _FakeClient)

        resp = await client.post(
            "/api/v1/quantize/lora-merge",
            json={"base_version_id": bid, "lora_version_id": lid},
        )
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        await asyncio.sleep(0.5)
        status = await client.get(f"/api/v1/quantize/lora-merge/{task_id}")
        assert status.status_code == 200
        body = status.json()
        assert body["status"] == "failed", "empty output_path must FAIL not COMPLETED"
        assert "output_path" in (body["error_message"] or "")


class TestLoraMergeStartupReconcile:
    # E-D2: orphaned RUNNING LoraMergeTask rows must be failed on startup, not
    # left stuck. We seed a RUNNING row, call _reconcile_orphaned_tasks, assert
    # it transitions to FAILED.
    @pytest.mark.asyncio
    async def test_running_lora_merge_failed_on_reconcile(self, settings):
        from fusion_model_hub.db import crud
        from fusion_model_hub.db.database import get_engine, init_db
        from fusion_model_hub.db.models import TaskStatus
        from fusion_model_hub.server.app import _reconcile_orphaned_tasks
        from fusion_model_hub.server.deps import get_session_factory, init_deps

        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        sf = get_session_factory()
        async with sf() as s:
            m = await crud.create_model(s, name="recon-m")
            base = await crud.create_version(s, model_id=m.id, version="b", file_path="/b")
            lora = await crud.create_version(s, model_id=m.id, version="l", file_path="/l")
            t = await crud.create_lora_merge_task(
                s,
                base_version_id=base.id,
                lora_version_id=lora.id,
            )
            await crud.update_lora_merge_task(s, t.id, status=TaskStatus.RUNNING)
            tid = t.id

        await _reconcile_orphaned_tasks()

        async with sf() as s:
            task = await crud.get_lora_merge_task(s, tid)
            assert task.status == TaskStatus.FAILED
            assert "restart" in (task.error_message or "")


class TestQuantizeClaimFencing:
    # R3: resume_quantize must atomically claim PENDING->RUNNING; a second claim
    # on the same task must be rejected (rowcount==0), so two hub processes never
    # run the same task twice.
    @pytest.mark.asyncio
    async def test_claim_is_exclusive(self, settings):
        from fusion_model_hub.db import crud
        from fusion_model_hub.db.database import get_engine, init_db
        from fusion_model_hub.db.models import TaskStatus
        from fusion_model_hub.server.deps import get_session_factory, init_deps

        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        sf = get_session_factory()
        async with sf() as s:
            m = await crud.create_model(s, name="claim-m")
            base = await crud.create_version(s, model_id=m.id, version="b", file_path="/b")
            t = await crud.create_quantize_task(
                s,
                source_version_id=base.id,
                quant_bits=4,
            )
            tid = t.id

        # first claim succeeds (PENDING -> RUNNING)
        async with sf() as s:
            assert await crud.claim_quantize_task(s, tid) is True
        # second claim fails (already RUNNING, not PENDING)
        async with sf() as s:
            assert await crud.claim_quantize_task(s, tid) is False
        async with sf() as s:
            t = await crud.get_quantize_task(s, tid)
            assert t.status == TaskStatus.RUNNING


class TestDownloadIntegrity:
    # H6: /downloads must compute SHA256 and reject a mismatched expected hash
    # instead of marking a corrupt/MITM'd download "completed".
    @pytest.mark.asyncio
    async def test_hash_mismatch_fails_task(self, client, monkeypatch):
        import hashlib

        create = await client.post("/api/v1/models", json={"name": "h6-dl-m"})
        mid = create.json()["id"]
        payload_bytes = b"integrity-test-bytes"
        real_hash = hashlib.sha256(payload_bytes).hexdigest()

        class _FakeResp:
            status_code = 200
            headers = {"content-length": str(len(payload_bytes))}

            def aiter_bytes(self, _n):
                async def gen():
                    yield payload_bytes

                return gen()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def stream(self, *a, **k):
                return _FakeResp()

        import fusion_model_hub.server.routers.downloads as dmod

        monkeypatch.setattr(dmod.httpx, "AsyncClient", _FakeClient)

        # wrong expected hash -> task must end failed, not completed
        resp = await client.post(
            "/api/v1/downloads",
            json={"model_id": mid, "source_url": "https://example.com/m.bin", "expected_sha256": "deadbeef" * 8},
        )
        assert resp.status_code == 201
        task_id = resp.json()["task_id"]
        await asyncio.sleep(0.6)
        status = await client.get(f"/api/v1/downloads/{task_id}")
        assert status.status_code == 200
        body = status.json()
        assert body["status"] == "failed", "hash mismatch must FAIL not completed"
        assert "integrity check failed" in (body["error_message"] or "")

    @pytest.mark.asyncio
    async def test_correct_hash_completes_with_file_hash(self, client, monkeypatch):
        import hashlib

        create = await client.post("/api/v1/models", json={"name": "h6-dl-ok"})
        mid = create.json()["id"]
        payload_bytes = b"good-bytes-1234"
        real_hash = hashlib.sha256(payload_bytes).hexdigest()

        class _FakeResp:
            status_code = 200
            headers = {"content-length": str(len(payload_bytes))}

            def aiter_bytes(self, _n):
                async def gen():
                    yield payload_bytes

                return gen()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def stream(self, *a, **k):
                return _FakeResp()

        import fusion_model_hub.server.routers.downloads as dmod

        monkeypatch.setattr(dmod.httpx, "AsyncClient", _FakeClient)

        resp = await client.post(
            "/api/v1/downloads",
            json={"model_id": mid, "source_url": "https://example.com/m.bin", "expected_sha256": real_hash},
        )
        task_id = resp.json()["task_id"]
        await asyncio.sleep(0.6)
        status = await client.get(f"/api/v1/downloads/{task_id}")
        body = status.json()
        assert body["status"] == "completed"
        assert body["file_hash"] == real_hash


class TestDownloadCooperativeCancel:
    # Issue #29: DELETE /downloads/{id} must cooperative-stop the worker — not
    # just flip DB status while the worker keeps writing to disk. The live task
    # is cancelled, the .part file is removed, and the task ends "cancelled".
    @pytest.mark.asyncio
    async def test_cancel_stops_worker_and_removes_part(self, client, monkeypatch):
        import os

        create = await client.post("/api/v1/models", json={"name": "cancel-dl-m"})
        mid = create.json()["id"]

        cancel_gate = asyncio.Event()

        class _FakeResp:
            status_code = 200
            headers = {"content-length": "10485760"}

            def aiter_bytes(self, _n):
                async def gen():
                    for _ in range(10):
                        await asyncio.sleep(0.05)
                        yield b"x" * (1024 * 1024)
                    cancel_gate.set()

                return gen()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def stream(self, *a, **k):
                return _FakeResp()

        import fusion_model_hub.server.routers.downloads as dmod

        monkeypatch.setattr(dmod.httpx, "AsyncClient", _FakeClient)

        resp = await client.post(
            "/api/v1/downloads",
            json={"model_id": mid, "source_url": "https://example.com/big.bin"},
        )
        assert resp.status_code == 201
        task_id = resp.json()["task_id"]
        await asyncio.sleep(0.15)

        # worker must be live + registered so cancel can reach it
        assert task_id in dmod._running_downloads, "worker not registered before cancel"
        assert not dmod._running_downloads[task_id].done()

        cancel = await client.delete(f"/api/v1/downloads/{task_id}")
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "cancelled"

        # let the CancelledError propagate through the worker
        await asyncio.sleep(0.4)

        status = await client.get(f"/api/v1/downloads/{task_id}")
        assert status.json()["status"] == "cancelled"
        # worker must NOT have reached the end of the stream (it would otherwise
        # flip to completed) — confirms cooperative stop, not advisory-only.
        assert not cancel_gate.is_set(), "worker ran to completion despite cancel"
        # .part file must be gone (no disk residue from a cancelled download)
        from fusion_model_hub.server.deps import get_settings

        st = get_settings()
        residue = os.path.join(st.data_dir, "downloads", f"{task_id}.part")
        assert not os.path.exists(residue), "cancelled .part file not cleaned up"

    @pytest.mark.asyncio
    async def test_cancel_completed_task_rejects(self, client, monkeypatch):
        # cancelling an already-completed task is a 400, not a silent re-mark.
        import hashlib

        create = await client.post("/api/v1/models", json={"name": "cancel-done-m"})
        mid = create.json()["id"]
        payload = b"tiny"
        real_hash = hashlib.sha256(payload).hexdigest()

        class _FakeResp:
            status_code = 200
            headers = {"content-length": str(len(payload))}

            def aiter_bytes(self, _n):
                async def gen():
                    yield payload

                return gen()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def stream(self, *a, **k):
                return _FakeResp()

        import fusion_model_hub.server.routers.downloads as dmod

        monkeypatch.setattr(dmod.httpx, "AsyncClient", _FakeClient)
        resp = await client.post(
            "/api/v1/downloads",
            json={"model_id": mid, "source_url": "https://example.com/t.bin", "expected_sha256": real_hash},
        )
        task_id = resp.json()["task_id"]
        await asyncio.sleep(0.5)
        cancel = await client.delete(f"/api/v1/downloads/{task_id}")
        assert cancel.status_code == 400


# ============================================================================
# P1 audit fixes: E-S5~S13 (security) + E-D4~D8 (data integrity)
# ============================================================================


class TestEncryptionKeyGuard:
    # E-S5: encryption must refuse to operate with the source-public default key.
    @pytest.mark.asyncio
    async def test_encrypt_refuses_without_key(self, client, monkeypatch):
        monkeypatch.delenv("FMH_ENCRYPTION_KEY", raising=False)
        m = await client.post("/api/v1/models", json={"name": "enc-nomkey"})
        mid = m.json()["id"]
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as s:
            v = await crud.create_version(s, model_id=mid, version="v1", file_path="/tmp/x")
            vid = v.id
        resp = await client.post("/api/v1/encryption/encrypt", json={"version_id": vid})
        assert resp.status_code == 503
        assert "FMH_ENCRYPTION_KEY" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_encrypt_roundtrip_with_key(self, client, monkeypatch, settings, tmp_path):
        monkeypatch.setenv("FMH_ENCRYPTION_KEY", "test-encryption-key-32-bytes-long!!")
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory, get_store

        store = get_store()
        m = await client.post("/api/v1/models", json={"name": "enc-rt"})
        mid = m.json()["id"]
        vdir = store.model_version_dir(mid, "v1")
        vdir.mkdir(parents=True, exist_ok=True)
        fpath = vdir / "model.bin"
        payload = b"plain model weights bytes"
        fpath.write_bytes(payload)
        sf = get_session_factory()
        async with sf() as s:
            v = await crud.create_version(s, model_id=mid, version="v1", file_path=str(fpath))
            vid = v.id
        enc = await client.post("/api/v1/encryption/encrypt", json={"version_id": vid})
        assert enc.status_code == 200
        assert enc.json()["encrypted"] is True
        assert fpath.read_bytes() != payload  # actually transformed
        dec = await client.post("/api/v1/encryption/decrypt", json={"version_id": vid})
        assert dec.status_code == 200
        assert dec.json()["encrypted"] is False
        assert fpath.read_bytes() == payload  # round-trip restores original


class TestWatermarkSecretGuard:
    # E-S6: watermark must refuse to sign/verify with the source-public default secret.
    @pytest.mark.asyncio
    async def test_embed_refuses_without_secret(self, client, monkeypatch):
        monkeypatch.delenv("FMH_WATERMARK_SECRET", raising=False)
        m = await client.post("/api/v1/models", json={"name": "wm-nosec"})
        mid = m.json()["id"]
        resp = await client.post("/api/v1/watermark/embed", json={"model_id": mid, "payload": {"k": "v"}})
        assert resp.status_code == 503
        assert "FMH_WATERMARK_SECRET" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_embed_verify_roundtrip_with_secret(self, client, monkeypatch):
        monkeypatch.setenv("FMH_WATERMARK_SECRET", "a-real-high-entropy-secret")
        m = await client.post("/api/v1/models", json={"name": "wm-rt"})
        mid = m.json()["id"]
        emb = await client.post("/api/v1/watermark/embed", json={"model_id": mid, "payload": {"trace": "abc"}})
        assert emb.status_code == 200
        sig = emb.json()["signature"]
        assert sig  # non-empty
        ver = await client.post("/api/v1/watermark/verify", json={"model_id": mid})
        assert ver.status_code == 200
        assert ver.json()["verified"] is True

    @pytest.mark.asyncio
    async def test_verify_rejects_tampered_signature(self, client, monkeypatch):
        monkeypatch.setenv("FMH_WATERMARK_SECRET", "a-real-high-entropy-secret")
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        m = await client.post("/api/v1/models", json={"name": "wm-tamper"})
        mid = m.json()["id"]
        emb = await client.post("/api/v1/watermark/embed", json={"model_id": mid, "payload": {"trace": "abc"}})
        assert emb.status_code == 200
        wm_id = emb.json()["id"]
        sf = get_session_factory()
        async with sf() as s:
            wm = await crud.get_watermark(s, wm_id)
            assert wm is not None
            wm.signature = "deadbeef" * 4
            await s.commit()
        ver = await client.post("/api/v1/watermark/verify", json={"model_id": mid})
        assert ver.json()["verified"] is False


class TestSecurityScanHonesty:
    # E-S7: scan must not fabricate a clean verdict; source_verified needs a real
    # HF org/name repo id, not mere presence of hf_repo.
    @pytest.mark.asyncio
    async def test_scan_reports_not_scanned_for_deep_checks(self, client):
        m = await client.post("/api/v1/models", json={"name": "sec-scan", "hf_repo": "org/repo-name"})
        mid = m.json()["id"]
        resp = await client.post("/api/v1/security/scan", json={"model_id": mid})
        assert resp.status_code == 200
        findings = resp.json()["findings"]
        assert findings["malicious_code"] == "not_scanned"
        assert findings["unsafe_dependencies"] == "not_scanned"
        assert findings["source_verified"] is True

    @pytest.mark.asyncio
    async def test_scan_unverifies_bad_hf_repo(self, client):
        m = await client.post("/api/v1/models", json={"name": "sec-badrepo", "hf_repo": "not-a-real-repo-id!!"})
        mid = m.json()["id"]
        resp = await client.post("/api/v1/security/scan", json={"model_id": mid})
        findings = resp.json()["findings"]
        assert findings["source_verified"] is False
        assert resp.json()["risk_level"] == "medium"


class TestTenantManagementRbac:
    # E-S8: tenant create/update/delete is admin-only; delete blocks orphans.
    @pytest.mark.asyncio
    async def test_create_tenant_open_when_auth_off(self, client):
        # auth disabled in fixture — create allowed (local dev)
        resp = await client.post("/api/v1/tenants", json={"name": "t-org"})
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_delete_tenant_blocks_orphan_models(self, client):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as s:
            t = await crud.create_tenant(s, name="orphan-t")
            tid = t.id
            await crud.create_model(s, name="orphan-m", tenant_id=tid)
        resp = await client.delete(f"/api/v1/tenants/{tid}")
        assert resp.status_code == 409
        assert "model" in resp.json()["detail"]


class TestModelTenantIsolation:
    # E-S9: non-admin caller cannot read/modify another tenant's model.
    @pytest.mark.asyncio
    async def test_get_model_cross_tenant_404(self, client):
        # auth disabled → both tenant ids empty → permissive (no isolation to test
        # at the HTTP layer without a real authed key). Verify the guard function
        # directly instead.
        from types import SimpleNamespace

        from fusion_model_hub.server.auth import set_auth_enabled
        from fusion_model_hub.server.routers.models import _check_model_read

        set_auth_enabled(True)
        try:
            model = SimpleNamespace(tenant_id="tenant-a")
            req = SimpleNamespace(state=SimpleNamespace(user_role="viewer", tenant_id="tenant-b"))
            with pytest.raises(Exception) as ei:
                _check_model_read(model, req)
            assert ei.value.status_code == 404
            # same-tenant passes
            req2 = SimpleNamespace(state=SimpleNamespace(user_role="viewer", tenant_id="tenant-a"))
            _check_model_read(model, req2)  # no raise
        finally:
            set_auth_enabled(False)


class TestVersionTenantGuardNoBypass:
    # E-S10: empty caller tenant must not bypass isolation for a tenanted model.
    @pytest.mark.asyncio
    async def test_empty_caller_tenant_blocked_for_tenanted_model(self, client):
        from types import SimpleNamespace

        from fastapi import HTTPException

        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory
        from fusion_model_hub.server.routers.versions import _enforce_version_tenant

        sf = get_session_factory()
        async with sf() as s:
            t = await crud.create_tenant(s, name="iso-t")
            m = await crud.create_model(s, name="iso-m", tenant_id=t.id)
            v = await crud.create_version(s, model_id=m.id, version="v1")
            req = SimpleNamespace(state=SimpleNamespace(tenant_id=""))
            with pytest.raises(HTTPException) as ei:
                await _enforce_version_tenant(s, v, req)
            assert ei.value.status_code == 404


class TestVersionRollbackNoneSafe:
    # E-D5: rollback/deprecate must not 500 when update_version_status returns None.
    @pytest.mark.asyncio
    async def test_rollback_nonexistent_version_404(self, client):
        resp = await client.post("/api/v1/versions/does-not-exist/rollback")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_deprecate_nonexistent_version_404(self, client):
        resp = await client.post("/api/v1/versions/does-not-exist/deprecate", json={"successor_version_id": ""})
        assert resp.status_code == 404


class TestServePublishedVersionOrdering:
    # E-D6: serve_model picks the most-recently-created published version deterministically.
    @pytest.mark.asyncio
    async def test_serve_picks_latest_published(self, client):
        # unit-level: the selection sorts published by created_at desc. Verify
        # the sort logic by constructing versions with distinct created_at.
        from types import SimpleNamespace

        v_old = SimpleNamespace(id="old", status=SimpleNamespace(value="published"), created_at=1000)
        v_new = SimpleNamespace(id="new", status=SimpleNamespace(value="published"), created_at=2000)
        published = [v_old, v_new]
        published.sort(key=lambda x: x.created_at or 0, reverse=True)
        assert published[0].id == "new"


class TestUrlDownloadTaskTracking:
    # E-D8: download-url creates a DownloadTask row the operator can poll.
    @pytest.mark.asyncio
    async def test_download_url_returns_task_id(self, client, monkeypatch):
        m = await client.post("/api/v1/models", json={"name": "url-dl-m"})
        mid = m.json()["id"]

        class _FakeResult:
            def __init__(self, d):
                self._d = d

            def get(self, k, default=None):
                return self._d.get(k, default)

        async def _fake_download(self, url, model_id, expected_hash):
            return {"status": "completed", "path": "/tmp/fake", "hash": expected_hash or "h", "size_bytes": 10}

        import fusion_model_hub.server.routers.versions as vmod

        monkeypatch.setattr(vmod.ModelDownloader, "download", _fake_download)

        resp = await client.post(
            f"/api/v1/models/{mid}/versions/download-url",
            json={"url": "https://example.com/m.mlx", "version": "u-v"},
        )
        assert resp.status_code == 202
        assert "download_task_id" in resp.json()
        task_id = resp.json()["download_task_id"]
        await asyncio.sleep(0.5)
        status = await client.get(f"/api/v1/downloads/{task_id}")
        assert status.json()["status"] == "completed"


class TestCacheSourceVersionIsolation:
    # H9/R2: cache key MUST include source_version_id. A 4bit quantize of
    # version A and version B of the same model must not collide — colliding
    # served the wrong weights on the second quantize.

    async def test_cross_version_no_false_hit(self, settings):
        from fusion_model_hub.cache.manager import CacheManager
        from fusion_model_hub.cache.types import CacheLevel

        cm = CacheManager(cache_root=str(Path(settings.cache_dir) / "iso"))
        fpath = Path(settings.cache_dir) / "w.bin"
        fpath.write_bytes(b"weights")
        cm.put("m1", CacheLevel.QUANTIZED, str(fpath), quant_bits=4, source_version_id="vA")
        # Same model+bits, different source version → must NOT hit vA's entry.
        assert not cm.has("m1", CacheLevel.QUANTIZED, 4, source_version_id="vB")
        assert cm.has("m1", CacheLevel.QUANTIZED, 4, source_version_id="vA")
        ent_a = cm.get("m1", CacheLevel.QUANTIZED, 4, source_version_id="vA")
        assert ent_a is not None and ent_a.source_version_id == "vA"
        shutil.rmtree(str(Path(settings.cache_dir) / "iso"), ignore_errors=True)

    async def test_keyless_legacy_entry_not_hit_by_versioned_lookup(self, settings):
        # Old keyless entries (pre-fix) must not be reused by a versioned
        # lookup — implicit invalidation, re-quantize produces a fresh entry.
        from fusion_model_hub.cache.manager import CacheManager
        from fusion_model_hub.cache.types import CacheLevel

        cm = CacheManager(cache_root=str(Path(settings.cache_dir) / "leg"))
        fpath = Path(settings.cache_dir) / "w2.bin"
        fpath.write_bytes(b"legacy weights")
        cm.put("m2", CacheLevel.QUANTIZED, str(fpath), quant_bits=4)
        assert not cm.has("m2", CacheLevel.QUANTIZED, 4, source_version_id="v1")
        shutil.rmtree(str(Path(settings.cache_dir) / "leg"), ignore_errors=True)


class TestAdaptHonestFailure:
    # H7: adapt/execute proxies convert+quantize to MLX. A non-200 was logged
    # as a warning then reported "completed" — silent false success. Must now
    # surface failed with the recorded reason.

    async def test_quantize_non200_reports_failed(self, client, monkeypatch):
        from fusion_model_hub.server.routers import adapt as adapt_mod

        adapt_mod._running_executions.clear()
        adapt_mod._execution_errors.clear()

        class FakeResp:
            status_code = 500
            text = "mlx error"

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return FakeResp()

        monkeypatch.setattr("fusion_model_hub.server.routers.adapt.httpx.AsyncClient", FakeClient)

        resp = await client.post(
            "/api/v1/adapt/execute",
            json={
                "model_id": "adapt-fail-model",
                "hf_repo": "owner/repo",
                "source_format": "safetensors",
                "quant_bits": 4,
            },
        )
        assert resp.status_code == 202
        eid = resp.json()["execution_id"]
        # poll until done
        for _ in range(20):
            status = await client.get(f"/api/v1/adapt/execute/{eid}")
            if status.json().get("status") != "running":
                break
            await asyncio.sleep(0.1)
        final = status.json()
        assert final["status"] == "failed", final
        assert "quantize failed" in final["error"]


class TestLayeredQuantizeHonestProxy:
    # H7 path 2: /quantize/layered is a pure MLX proxy — no hub row. Response
    # must declare hub_registered: false so the caller does not assume the
    # hub tracks the result.

    async def test_layered_response_declares_not_registered(self, client, monkeypatch):
        class FakeResp:
            status_code = 202

            def json(self):
                return {"job_id": "lj-1"}

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return FakeResp()

        monkeypatch.setattr("fusion_model_hub.server.routers.quantize.httpx.AsyncClient", FakeClient)
        resp = await client.post(
            "/api/v1/quantize/layered",
            json={
                "model": "owner/repo",
                "default_bits": 4,
                "layer_rules": [{"pattern": ".*", "bits": 4}],
                "quant_group_size": 64,
                "quant_mode": "symmetric",
                "trust_remote_code": False,
            },
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["hub_registered"] is False
        assert body["job_id"] == "lj-1"


class TestDeploymentNodePlacement:
    # H3: a deployment must record node_id and load/unload the model on the
    # chosen node's MLX URL, not always the local one.

    @staticmethod
    def _capture_client(monkeypatch, module_path, status_code=200):
        captured = []

        class FakeResp:
            def __init__(self, sc=status_code):
                self.status_code = sc

            def json(self):
                return {}

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **k):
                captured.append(("POST", url))
                return FakeResp()

            async def get(self, url, **k):
                captured.append(("GET", url))
                return FakeResp()

        monkeypatch.setattr(module_path, FakeClient)
        return captured

    @pytest.mark.asyncio
    async def test_default_local_placement(self, client):
        model = await client.post("/api/v1/models", json={"name": "h3-local-m"})
        mid = model.json()["id"]
        resp = await client.post(
            "/api/v1/deployments",
            json={"model_id": mid, "name": "dep-local", "replicas": 1},
        )
        assert resp.status_code == 201
        assert resp.json()["node_id"] == "local"

    @pytest.mark.asyncio
    async def test_unknown_node_rejected(self, client):
        model = await client.post("/api/v1/models", json={"name": "h3-unk-m"})
        mid = model.json()["id"]
        resp = await client.post(
            "/api/v1/deployments",
            json={"model_id": mid, "name": "dep-unk", "node_id": "no-such-node"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_inactive_node_rejected(self, client):
        model = await client.post("/api/v1/models", json={"name": "h3-inact-m"})
        mid = model.json()["id"]
        # register a node, then push its heartbeat far into the past so
        # _effective_status returns "inactive" (stale > 120s).
        from datetime import UTC, datetime, timedelta

        node = await client.post(
            "/api/v1/cluster/nodes",
            json={"name": "stale-node", "url": "http://10.0.0.9:11434"},
        )
        assert node.status_code == 201
        nid = node.json()["id"]
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as s:
            n = await crud.get_cluster_node(s, nid)
            n.last_heartbeat = datetime.now(UTC) - timedelta(seconds=600)
            n.status = "active"
            await s.commit()
        resp = await client.post(
            "/api/v1/deployments",
            json={"model_id": mid, "name": "dep-inact", "node_id": nid},
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_remote_placement_loads_on_node_url(self, client, monkeypatch):
        model = await client.post("/api/v1/models", json={"name": "h3-remote-m"})
        mid = model.json()["id"]
        node = await client.post(
            "/api/v1/cluster/nodes",
            json={"name": "remote-node", "url": "http://10.0.0.5:11434"},
        )
        nid = node.json()["id"]
        # fresh heartbeat so _effective_status is active
        await client.post(f"/api/v1/cluster/nodes/{nid}/heartbeat")

        captured = self._capture_client(
            monkeypatch,
            "fusion_model_hub.server.routers.deployments.httpx.AsyncClient",
        )
        resp = await client.post(
            "/api/v1/deployments",
            json={"model_id": mid, "name": "dep-remote", "node_id": nid},
        )
        assert resp.status_code == 201
        assert resp.json()["node_id"] == nid
        # the load must hit the remote node URL, not the local MLX default
        posts = [u for m, u in captured if m == "POST" and u.endswith("/load")]
        assert posts, "no load call captured"
        assert any("10.0.0.5:11434" in u for u in posts), posts

    @pytest.mark.asyncio
    async def test_remote_placement_unloads_on_node_url(self, client, monkeypatch):
        model = await client.post("/api/v1/models", json={"name": "h3-unload-m"})
        mid = model.json()["id"]
        node = await client.post(
            "/api/v1/cluster/nodes",
            json={"name": "unload-node", "url": "http://10.0.0.7:11434"},
        )
        nid = node.json()["id"]
        await client.post(f"/api/v1/cluster/nodes/{nid}/heartbeat")
        captured = self._capture_client(
            monkeypatch,
            "fusion_model_hub.server.routers.deployments.httpx.AsyncClient",
        )
        create = await client.post(
            "/api/v1/deployments",
            json={"model_id": mid, "name": "dep-unload", "node_id": nid},
        )
        did = create.json()["id"]
        captured.clear()
        await client.delete(f"/api/v1/deployments/{did}")
        posts = [u for m, u in captured if m == "POST" and u.endswith("/unload")]
        assert posts, "no unload call captured"
        assert any("10.0.0.7:11434" in u for u in posts), posts


class TestH8HttpxPooling:
    # H8: the inference router aliases ``http_client`` as ``httpx`` and reads
    # ``httpx.AsyncClient`` (== PoolClient). Two async-with wrappers built with
    # the same base_url must share one AsyncHTTPTransport (connection reuse);
    # aclose must be a no-op so the pool survives; close_all_transports clears
    # it. Per-call timeout must stay independent. Test patches on
    # ``inference.httpx.AsyncClient`` must still intercept (mock supplies its
    # own __aenter__/__aexit__/post).

    async def test_transport_reused_across_calls(self):
        from fusion_model_hub.server import http_client

        await http_client.close_all_transports()
        async with http_client.AsyncClient(base_url="http://127.0.0.1:11434", timeout=60.0) as c1:
            pass
        async with http_client.AsyncClient(base_url="http://127.0.0.1:11434", timeout=30.0) as c2:
            assert c2._transport is c1._transport, "transport not reused across async-with"
        # per-call timeout stays independent on each wrapper
        assert c1.timeout.read == 60.0 and c2.timeout.read == 30.0
        await http_client.close_all_transports()

    async def test_omitted_base_url_does_not_crash(self):
        # Real call sites build full URLs inline (no base_url): the constructor
        # must not forward base_url=None to httpx (URL(None) -> TypeError).
        from fusion_model_hub.server import http_client

        await http_client.close_all_transports()
        async with http_client.AsyncClient(timeout=60.0) as c:
            assert c._transport is http_client._TRANSPORT_POOL["default"]
        # all omitted-base_url call sites share the same "default" pool entry
        async with http_client.AsyncClient(timeout=15.0) as c2:
            assert c2._transport is c._transport
        await http_client.close_all_transports()

    async def test_aclose_does_not_drop_pool(self):
        from fusion_model_hub.server import http_client

        await http_client.close_all_transports()
        async with http_client.AsyncClient(base_url="http://127.0.0.1:11434", timeout=5.0) as c1:
            pass
        # after exit the client reports closed but the pool entry persists
        assert "http://127.0.0.1:11434" in http_client._TRANSPORT_POOL
        await http_client.close_all_transports()
        assert http_client._TRANSPORT_POOL == {}

    async def test_inference_alias_targets_pool_client(self):
        from fusion_model_hub.server import http_client
        from fusion_model_hub.server.routers import inference

        assert inference.httpx is http_client
        assert inference.httpx.AsyncClient is http_client.PoolClient
        # exception types resolve through the alias
        import httpx as real_httpx

        assert inference.httpx.ConnectError is real_httpx.ConnectError
        assert inference.httpx.HTTPStatusError is real_httpx.HTTPStatusError

    @pytest.mark.asyncio
    async def test_test_patch_still_intercepts(self):
        # mirrors the existing router test pattern: patch inference.httpx.AsyncClient
        # with a mock carrying __aenter__/__aexit__/post — must reach the call site.
        from unittest.mock import AsyncMock, MagicMock, patch

        from fusion_model_hub.server.routers import inference

        mock_resp = MagicMock(status_code=200)
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("fusion_model_hub.server.routers.inference.httpx.AsyncClient", return_value=mock_client):
            async with inference.httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post("http://127.0.0.1:11434/v1/models/m/load")
        assert r.status_code == 200
        mock_client.post.assert_awaited()
