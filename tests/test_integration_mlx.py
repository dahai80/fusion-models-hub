"""End-to-end integration tests against a live Fusion-MLX server.

These tests are NOT part of the default pytest run. They require a running
Fusion-MLX server (~/claude-home/fusion-mlx/start.sh start) on the default
MLX URL (http://127.0.0.1:11434) with a small chat-capable model that MLX
can load by name. Per CLAUDE.md, model tests must use a real loaded model.

Run explicitly:
    pytest tests/test_integration_mlx.py -v

Skip automatically when MLX is unreachable (CI has no MLX) so the default
suite stays green. The skip is health-probed at module import via the
_MLX_REACHABLE flag.

Flow under test (full round-trip, no mocks):
    1. Hub /system/health reports MLX reachable
    2. create model (hf_repo = a real MLX model name) + empty version
    3. publish model + version
    4. POST /models/{id}/serve -> MLX loads the model by name
    5. POST /inference/{id}/chat -> real inference, non-empty content
    6. DELETE /models/{id}/serve -> MLX unloads
    7. cleanup: delete model (DB only; MLX model files untouched)

Quantize layer (two tests):
    8a. Direct MLX /v1/quantize (local model path) + job poll -> done
        (probes MLX-side availability; the Hub converter contract gap is
        tracked separately — see module-level note in TestMlxQuantizeCache)
    8b. Hub cache-hit round-trip: pre-seed the 3-level cache, POST
        /api/v1/quantize, poll task -> COMPLETED with output_version_id.
        Verifies _run_quantize's has/get cache short-circuit without a
        real (slow) MLX quantize.
"""

import asyncio
import os
import tempfile

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.cache.types import CacheLevel
from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.auth import set_auth_enabled
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import get_cache_manager, init_deps

MLX_URL = os.environ.get("FMH_MLX_URL", "http://127.0.0.1:11434")
# Small chat-capable model MLX already has cached + can load fast. Override
# via FMH_INTEGRATION_MODEL if a different small model is preferred.
INTEGRATION_MODEL = os.environ.get("FMH_INTEGRATION_MODEL", "Qwen3-0.6B-4bit")
# Absolute path to a small local MLX model dir for direct /v1/quantize probing.
# Must live under ~/.fusion-mlx/models (MLX restricts output_path to allowed dirs).
LOCAL_QUANTIZE_SOURCE = os.environ.get(
    "FMH_INTEGRATION_QUANTIZE_SOURCE",
    os.path.expanduser("~/.fusion-mlx/models/mlx-community/Qwen3-0.6B-8bit"),
)
LOCAL_QUANTIZE_OUT = os.path.expanduser("~/.fusion-mlx/models/fmh-int-test-quantize-out")


def _read_mlx_key() -> str:
    # Mirror config.py's fallback so the Hub->MLX Bearer is valid without
    # depending on env being set in the test environment.
    path = os.path.expanduser("~/.fusion-mlx/settings.json")
    try:
        import json

        with open(path) as f:
            return json.load(f).get("auth", {}).get("api_key", "")
    except (FileNotFoundError, OSError, ValueError):
        return ""


def _mlx_reachable() -> bool:
    try:
        r = httpx.get(f"{MLX_URL}/health", timeout=3.0)
        return r.status_code == 200 and r.json().get("status") == "healthy"
    except Exception:
        return False


_MLX_REACHABLE = _mlx_reachable()
_MLX_KEY = _read_mlx_key()
_SKIP_REASON = (
    f"Fusion-MLX not reachable at {MLX_URL} (start with "
    "~/claude-home/fusion-mlx/start.sh start); skipping live integration tests"
)
requires_mlx = pytest.mark.skipif(
    not _MLX_REACHABLE or not _MLX_KEY,
    reason=_SKIP_REASON,
)


@pytest.fixture
async def mlx_client():
    # Auth OFF (these tests exercise MLX, not Hub RBAC). Real MLX URL + key
    # so Hub->MLX calls authenticate. Fresh in-memory DB per test for isolation.
    set_auth_enabled(False)
    s = Settings(
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_integration_mlx",
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
        mlx_url=MLX_URL,
        mlx_internal_api_key=_MLX_KEY,
    )
    engine = get_engine(s.db_url)
    await init_db(engine)
    init_deps(s, engine)
    app = create_app(s)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    set_auth_enabled(False)


class TestMlxHealth:
    # Hub /system/health must reflect the live MLX server's reachability and
    # loaded-model state — the single integration point Hub relies on.

    @pytest.mark.asyncio
    @requires_mlx
    async def test_health_reports_mlx_healthy(self, mlx_client):
        resp = await mlx_client.get("/api/v1/system/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        mlx = data.get("mlx", {})
        assert mlx.get("status") == "available"
        assert mlx.get("info", {}).get("status") == "healthy"
        assert mlx.get("info", {}).get("ready") is True
        assert len(mlx.get("info", {}).get("loaded_models", [])) > 0


class TestMlxServeChatRoundTrip:
    # Full round-trip: register a model pointing at a real MLX model name,
    # publish, serve (MLX load), chat (real inference), unload. No mocks.

    @pytest.mark.asyncio
    @requires_mlx
    async def test_serve_chat_unload_round_trip(self, mlx_client):
        # 1. Create model with hf_repo = a real MLX-cached model name so
        #    serve resolves model_name = m.hf_repo and MLX can load it.
        create = await mlx_client.post(
            "/api/v1/models",
            json={
                "name": f"int-{INTEGRATION_MODEL}",
                "model_type": "llm",
                "hf_repo": INTEGRATION_MODEL,
            },
        )
        assert create.status_code == 201, create.text
        model_id = create.json()["id"]

        try:
            # 2. Create an empty version (file_path empty -> serve skips the
            #    file-integrity gate; MLX loads by name from its own cache).
            ver = await mlx_client.post(
                f"/api/v1/models/{model_id}/versions",
                data={"version": "1.0.0", "format": "mlx", "quantization": "4bit"},
                files={"file": ("", b"")},
            )
            assert ver.status_code == 201, ver.text
            version_id = ver.json()["id"]

            # 3. Publish model + version (serve requires published state).
            pub_model = await mlx_client.post(f"/api/v1/models/{model_id}/publish")
            assert pub_model.status_code == 200, pub_model.text
            await mlx_client.put(
                f"/api/v1/versions/{version_id}/status",
                json={"target_status": "published", "approval_level": "l1"},
            )

            # 4. Serve -> MLX loads the model by name.
            served = await mlx_client.post(
                f"/api/v1/models/{model_id}/serve",
                json={"version_id": version_id, "gpu": True},
            )
            assert served.status_code == 200, served.text
            assert served.json()["status"] == "loaded"
            assert served.json()["mlx_model"] == INTEGRATION_MODEL

            # 5. Real inference: chat must return non-empty assistant content.
            chat = await mlx_client.post(
                f"/api/v1/inference/{model_id}/chat",
                json={
                    "messages": [{"role": "user", "content": "Say only: OK"}],
                    "max_tokens": 5,
                    "temperature": 0.0,
                },
            )
            assert chat.status_code == 200, chat.text
            choices = chat.json().get("choices", [])
            assert len(choices) > 0
            content = choices[0].get("message", {}).get("content", "")
            assert content.strip() != "", "inference returned empty content"

            # 6. Unload -> MLX releases the model (DELETE /models/{id}/serve).
            unloaded = await mlx_client.delete(f"/api/v1/models/{model_id}/serve")
            assert unloaded.status_code == 200, unloaded.text
            assert unloaded.json()["status"] == "unloaded"
        finally:
            # 7. Cleanup DB record only. MLX model files are shared/cached —
            #    never delete those. Best-effort delete; ignore if already gone.
            await mlx_client.delete(f"/api/v1/models/{model_id}")


class TestMlxQuantizeJob:
    # Direct MLX /v1/quantize probe: submit a quantize job against a small local
    # model dir, poll the job to completion, assert output_path. This validates
    # the MLX-side quantize endpoint the Hub delegates to — without routing
    # through the Hub converter, whose request contract (source_path vs model,
    # sync vs job-poll) is tracked as a separate gap (see module docstring).
    # All test output is written under ~/.fusion-mlx/models (MLX's allowed dir)
    # and removed in finally so the cache is not polluted.

    @pytest.mark.asyncio
    @requires_mlx
    async def test_mlx_quantize_job_completes(self):
        if not os.path.isdir(LOCAL_QUANTIZE_SOURCE):
            pytest.skip(f"local quantize source not found: {LOCAL_QUANTIZE_SOURCE}")
        headers = {"Authorization": f"Bearer {_MLX_KEY}", "X-Fusion-Source": "model-hub"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            submit = await client.post(
                f"{MLX_URL}/v1/quantize",
                json={
                    "model": LOCAL_QUANTIZE_SOURCE,
                    "quant_bits": 4,
                    "output_path": LOCAL_QUANTIZE_OUT,
                },
                headers=headers,
            )
            assert submit.status_code == 200, submit.text
            job_id = submit.json().get("job_id", "")
            assert job_id, "MLX returned no job_id"
            try:
                done = False
                for _ in range(30):
                    await asyncio.sleep(1.0)
                    poll = await client.get(
                        f"{MLX_URL}/v1/quantize/jobs/{job_id}",
                        headers=headers,
                    )
                    assert poll.status_code == 200, poll.text
                    info = poll.json()
                    if info.get("status") in ("done", "completed", "succeeded"):
                        assert info.get("output_path"), "job done but no output_path"
                        done = True
                        break
                    if info.get("status") == "failed":
                        pytest.fail(f"MLX quantize job failed: {info.get('error')}")
                assert done, "MLX quantize job did not complete in 30s"
            finally:
                # Best-effort cleanup of the test output dir.
                import shutil

                with __import__("contextlib").suppress(OSError):
                    shutil.rmtree(LOCAL_QUANTIZE_OUT, ignore_errors=True)


class TestMlxQuantizeCache:
    # Hub cache-hit round-trip: pre-seed the 3-level cache with a fake quantized
    # artifact for a model+version, then POST /api/v1/quantize. _run_quantize
    # must short-circuit on the cache hit (has/get), skip the MLX call, create
    # the output ModelVersion, and mark the task COMPLETED. This verifies the
    # cache wiring in tasks.py without a real (slow) MLX quantize.
    #
    # NOTE: the full MLX quantize path through Hub converter.quantize is NOT
    # exercised here — converter.py sends {source_path} while MLX /v1/quantize
    # expects {model} and returns a job_id (async), so a real round-trip would
    # hang on the 600s timeout. That contract gap is an upstream/Hub-code issue
    # to fix separately; this test pins the cache layer that gates it.

    @pytest.mark.asyncio
    @requires_mlx
    async def test_quantize_cache_hit_completes_task(self, mlx_client):
        # 1. Create model + a published version whose file_path points at a real
        #    small file (so serve/quantize integrity checks have a path, though
        #    the cache hit bypasses the actual quantize call).
        create = await mlx_client.post(
            "/api/v1/models",
            json={
                "name": "int-quant-cache",
                "model_type": "llm",
                "hf_repo": INTEGRATION_MODEL,
            },
        )
        assert create.status_code == 201, create.text
        model_id = create.json()["id"]

        tmpdir = tempfile.mkdtemp(prefix="fmh_int_quant_")
        fake_weights = os.path.join(tmpdir, "weights.mlx")
        with open(fake_weights, "wb") as f:
            f.write(b"\x00" * 1024)
        try:
            ver = await mlx_client.post(
                f"/api/v1/models/{model_id}/versions",
                data={"version": "1.0.0", "format": "mlx", "quantization": "8bit"},
                files={"file": ("", b"")},
            )
            assert ver.status_code == 201, ver.text
            version_id = ver.json()["id"]

            # 2. Publish model + version (quantize needs a tracked source).
            await mlx_client.post(f"/api/v1/models/{model_id}/publish")
            await mlx_client.put(
                f"/api/v1/versions/{version_id}/status",
                json={"target_status": "published", "approval_level": "l1"},
            )

            # 3. Pre-seed the quantized cache: put a QUANTIZED-4bit entry keyed
            #    by (model_id, version_id) pointing at the fake weights file.
            cache = get_cache_manager()
            entry = cache.put(
                model_id=model_id,
                level=CacheLevel.QUANTIZED,
                source_path=fake_weights,
                quant_bits=4,
                source_version_id=version_id,
            )
            assert cache.has(model_id, CacheLevel.QUANTIZED, 4, source_version_id=version_id)

            # 4. Submit quantize -> _run_quantize must hit the cache and complete.
            submit = await mlx_client.post(
                "/api/v1/quantize",
                json={
                    "source_version_id": version_id,
                    "target_format": "mlx",
                    "quant_bits": 4,
                },
            )
            assert submit.status_code == 202, submit.text
            task_id = submit.json()["task_id"]

            # 5. Poll the Hub task until terminal.
            completed = False
            for _ in range(20):
                await asyncio.sleep(0.5)
                status_resp = await mlx_client.get(f"/api/v1/quantize/{task_id}")
                assert status_resp.status_code == 200, status_resp.text
                task = status_resp.json()
                st = task.get("status", "")
                if st == "completed":
                    assert task.get("output_version_id"), "completed task has no output_version_id"
                    completed = True
                    break
                if st == "failed":
                    pytest.fail(f"quantize task failed (cache hit path): {task.get('error_message')}")
            assert completed, "quantize task did not reach COMPLETED via cache hit in 10s"

            # 6. The output version created from the cached path must exist.
            out_ver_id = task["output_version_id"]
            out_ver = await mlx_client.get(f"/api/v1/versions/{out_ver_id}")
            assert out_ver.status_code == 200, out_ver.text

            # 7. Cleanup cache entry + model; temp dir removed in finally.
            cache.remove(model_id, CacheLevel.QUANTIZED, 4, source_version_id=version_id)
            await mlx_client.delete(f"/api/v1/models/{model_id}")
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
