import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.cache.manager import CacheManager
from fusion_model_hub.cache.types import CacheLevel
from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.auth import set_auth_enabled
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps


@pytest.fixture
def cache_dir():
    d = tempfile.mkdtemp(prefix="fmh_cache_test_")
    yield d


@pytest.fixture
def settings(cache_dir):
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_test_cache",
        cache_dir=cache_dir,
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
async def client(app, settings):
    set_auth_enabled(False)
    engine = get_engine(settings.db_url)
    await init_db(engine)
    init_deps(settings, engine)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _write_tmp_file(content: bytes = b"model weights") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mlx") as f:
        f.write(content)
        return f.name


class TestCacheManager:
    def test_put_get_has_remove(self, cache_dir):
        cm = CacheManager(cache_root=cache_dir)
        src = _write_tmp_file(b"weights-4bit")
        entry = cm.put("m1", CacheLevel.QUANTIZED, src, quant_bits=4)
        assert entry.model_id == "m1"
        assert entry.quant_bits == 4
        assert cm.has("m1", CacheLevel.QUANTIZED, 4)
        got = cm.get("m1", CacheLevel.QUANTIZED, 4)
        assert got is not None
        assert got.path.endswith(".mlx")
        # E-R4: ref_count machinery removed (was never persisted/decremented, so
        # the GC "pin" it fed was fictional). get() now refreshes last_accessed,
        # which is the real LRU/age signal gc() evicts on.
        assert got.last_accessed > 0
        assert got.ref_count == 0
        assert cm.remove("m1", CacheLevel.QUANTIZED, 4) is True
        assert not cm.has("m1", CacheLevel.QUANTIZED, 4)

    def test_remove_model_clears_all_levels(self, cache_dir):
        cm = CacheManager(cache_root=cache_dir)
        s1 = _write_tmp_file(b"raw")
        s2 = _write_tmp_file(b"converted")
        cm.put("m2", CacheLevel.RAW, s1)
        cm.put("m2", CacheLevel.CONVERTED, s2)
        removed = cm.remove_model("m2")
        assert removed == 2
        assert not cm.has("m2", CacheLevel.RAW)
        assert not cm.has("m2", CacheLevel.CONVERTED)

    def test_stats_aggregates(self, cache_dir):
        cm = CacheManager(cache_root=cache_dir)
        cm.put("m3", CacheLevel.RAW, _write_tmp_file(b"aaa"))
        cm.put("m3", CacheLevel.QUANTIZED, _write_tmp_file(b"bb"), quant_bits=4)
        stats = cm.stats()
        assert stats.total_entries == 2
        assert stats.raw_count == 1
        assert stats.quantized_count == 1
        assert stats.total_size_bytes > 0

    def test_get_missing_path_self_heals(self, cache_dir):
        cm = CacheManager(cache_root=cache_dir)
        src = _write_tmp_file(b"x")
        cm.put("m4", CacheLevel.CONVERTED, src)
        Path(cm.get("m4", CacheLevel.CONVERTED).path).unlink()
        assert cm.get("m4", CacheLevel.CONVERTED) is None
        assert not cm.has("m4", CacheLevel.CONVERTED)

    def test_validate_detects_missing(self, cache_dir):
        cm = CacheManager(cache_root=cache_dir)
        cm.put("m5", CacheLevel.RAW, _write_tmp_file(b"y"))
        for data in cm._index.values():
            Path(data["path"]).unlink()
        result = cm.validate()
        assert result["missing"] >= 1

    def test_gc_removes_old_entries(self, cache_dir):
        cm = CacheManager(cache_root=cache_dir)
        cm.put("m6", CacheLevel.RAW, _write_tmp_file(b"old"))
        for data in cm._index.values():
            data["last_accessed"] = 0
            data["created_at"] = 0
        cm._save_index()
        removed = cm.gc(max_age_days=1)
        assert removed >= 1

    def test_gc_keeps_fresh_entry_without_ref_count_pin(self, cache_dir):
        # E-R4: a freshly-written entry has ref_count 0 (inference never touches
        # the cache), so the old "ref_count <= 0 and age > max_age_days/2"
        # clause evicted it at half the age threshold under a fictional "in use"
        # pin. GC must now keep a fresh entry and only evict by real age.
        cm = CacheManager(cache_root=cache_dir)
        cm.put("m-fresh", CacheLevel.QUANTIZED, _write_tmp_file(b"q4"), quant_bits=4)
        # Reload from index.json to simulate a restart: ref_count was never
        # persisted, so it reads back 0 — exactly the live-but-evicted scenario.
        cm2 = CacheManager(cache_root=cache_dir)
        removed = cm2.gc(max_age_days=30)
        assert removed == 0
        assert cm2.has("m-fresh", CacheLevel.QUANTIZED, 4)

    def test_gc_evicts_by_size_lru(self, cache_dir):
        # E-R4: with a size cap, gc evicts least-recently-accessed first by
        # last_accessed, not by ref_count.
        cm = CacheManager(cache_root=cache_dir)
        cm.put("old-lru", CacheLevel.RAW, _write_tmp_file(b"old" * 1000))
        cm.put("new-lru", CacheLevel.RAW, _write_tmp_file(b"new" * 1000))
        # Force old-lru to look older.
        for k, data in cm._index.items():
            if "old-lru" in k:
                data["last_accessed"] = 0
        cm._save_index()
        # Cap holds one 3000-byte file: evict the oldest (old-lru) and keep
        # the newer (new-lru).
        removed = cm.gc(max_size_gb=3.0e-6)
        assert removed >= 1
        assert not cm.has("old-lru", CacheLevel.RAW)
        assert cm.has("new-lru", CacheLevel.RAW)

    def test_remove_model_no_prefix_collision(self, cache_dir):
        # E-R5: remove_model used startswith(f"{model_id}:") so deleting "abc"
        # also purged "abcxyz". Keys must match only when their first
        # colon-segment equals the model_id exactly.
        cm = CacheManager(cache_root=cache_dir)
        cm.put("abc", CacheLevel.RAW, _write_tmp_file(b"a"))
        cm.put("abcxyz", CacheLevel.RAW, _write_tmp_file(b"b"))
        removed = cm.remove_model("abc")
        assert removed == 1
        assert not cm.has("abc", CacheLevel.RAW)
        # The unrelated "abcxyz" model must survive.
        assert cm.has("abcxyz", CacheLevel.RAW)

    def test_gc_reconciler_removes_disk_orphans(self, cache_dir):
        # E-R6: a file on disk with no index entry is leaked forever under the
        # old gc (which walked self._index only). gc must now scan the level
        # dirs and delete orphans, while keeping indexed entries intact.
        cm = CacheManager(cache_root=cache_dir)
        cm.put("indexed", CacheLevel.RAW, _write_tmp_file(b"keep me"))
        # Drop an orphan file directly on disk.
        orphan = Path(cm.raw_dir) / "orphan-model.mlx"
        orphan.write_bytes(b"i am a leak")
        assert orphan.exists()
        removed = cm.gc(max_age_days=30)
        assert removed >= 1
        assert not orphan.exists()
        # Indexed entry must survive the reconciler.
        assert cm.has("indexed", CacheLevel.RAW)

    def test_corrupt_index_preserves_disk_then_reconciles(self, cache_dir):
        # E-R6 + E-R1: a corrupt index.json is quarantined and the in-memory
        # index starts empty, but the real cache files on disk must NOT be
        # silently dropped — they become orphans the next gc reclaims, and the
        # quarantine file itself must not be mistaken for a cache entry.
        cm = CacheManager(cache_root=cache_dir)
        cm.put("survivor", CacheLevel.RAW, _write_tmp_file(b"on disk"))
        disk_path = Path(cm.get("survivor", CacheLevel.RAW).path)
        assert disk_path.exists()
        # Corrupt the index on disk, then reload.
        cm._index_file.write_text("{not valid json", encoding="utf-8")
        cm2 = CacheManager(cache_root=cache_dir)
        # Empty index after quarantine — disk file still present.
        assert len(cm2._index) == 0
        assert disk_path.exists()
        # A quarantine sidecar was written alongside the cache dirs' parent.
        quarantined = list(Path(cache_dir).glob("index.corrupt.*.json"))
        assert len(quarantined) == 1
        # gc reconciler reclaims the now-orphaned disk file (and must not try
        # to delete the .corrupt sidecar, which lives in cache_root not a
        # level dir, nor any .tmp staging file).
        removed = cm2.gc(max_age_days=30)
        assert removed >= 1
        assert not disk_path.exists()


class TestCacheRouter:
    async def test_cache_stats_empty(self, client):
        resp = await client.get("/api/v1/cache")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_entries"] == 0
        assert data["raw_count"] == 0

    async def test_cache_entries_after_put(self, client, cache_dir):
        from fusion_model_hub.server.deps import _cache

        _cache.put("router-m1", CacheLevel.QUANTIZED, _write_tmp_file(b"q"), quant_bits=4)
        resp = await client.get("/api/v1/cache/entries")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        assert any(e["model_id"] == "router-m1" for e in data["entries"])

    async def test_cache_entries_level_filter(self, client, cache_dir):
        from fusion_model_hub.server.deps import _cache

        _cache.put("f1", CacheLevel.RAW, _write_tmp_file(b"r"))
        _cache.put("f2", CacheLevel.QUANTIZED, _write_tmp_file(b"q"), quant_bits=4)
        resp = await client.get("/api/v1/cache/entries?level=quantized")
        assert resp.status_code == 200
        data = resp.json()
        assert all(e["level"] == "quantized" for e in data["entries"])
        assert data["count"] == 1

    async def test_cache_remove_model_not_found(self, client):
        resp = await client.delete("/api/v1/cache/no-such-model")
        assert resp.status_code == 404

    async def test_cache_remove_entry_bad_level(self, client):
        resp = await client.delete("/api/v1/cache/m1/badlevel")
        assert resp.status_code == 400

    async def test_cache_remove_entry_not_found(self, client):
        resp = await client.delete("/api/v1/cache/m1/raw")
        assert resp.status_code == 404

    async def test_cache_gc(self, client):
        resp = await client.post("/api/v1/cache/gc?max_age_days=365")
        assert resp.status_code == 200
        assert "removed" in resp.json()

    async def test_cache_validate(self, client):
        resp = await client.post("/api/v1/cache/validate")
        assert resp.status_code == 200
        assert "valid" in resp.json()

    async def test_delete_model_purges_cache(self, client, cache_dir):
        # E-R2: delete_model/batch_delete only rmtree'd the models_dir tree and
        # never called cache.remove_model, so cache files + index.json entries
        # for a deleted model leaked forever. Deleting the model must also drop
        # its cache entries.
        from fusion_model_hub.db.crud import create_model
        from fusion_model_hub.db.models import ModelType
        from fusion_model_hub.server.deps import _cache, get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            model = await create_model(
                session,
                name="del-cache-model",
                model_type=ModelType.LLM,
                architecture="qwen2",
                params_size="7B",
            )
            model_id = model.id
        # Seed a cache entry for the model.
        _cache.put(model_id, CacheLevel.QUANTIZED, _write_tmp_file(b"q4"), quant_bits=4)
        assert _cache.has(model_id, CacheLevel.QUANTIZED, 4)
        # Delete the model via the API.
        resp = await client.delete(f"/api/v1/models/{model_id}")
        assert resp.status_code == 200
        # Cache entry for the deleted model must be gone.
        assert not _cache.has(model_id, CacheLevel.QUANTIZED, 4)

    async def test_batch_delete_purges_cache(self, client, cache_dir):
        # E-R2: batch_delete path must also purge cache entries.
        from fusion_model_hub.db.crud import create_model
        from fusion_model_hub.db.models import ModelType
        from fusion_model_hub.server.deps import _cache, get_session_factory

        sf = get_session_factory()
        model_ids = []
        async with sf() as session:
            for name in ("batch-del-1", "batch-del-2"):
                model = await create_model(
                    session,
                    name=name,
                    model_type=ModelType.LLM,
                    architecture="qwen2",
                    params_size="7B",
                )
                model_ids.append(model.id)
        for mid in model_ids:
            _cache.put(mid, CacheLevel.RAW, _write_tmp_file(b"r"))
            assert _cache.has(mid, CacheLevel.RAW)
        resp = await client.post("/api/v1/models/batch/delete", json={"model_ids": model_ids})
        assert resp.status_code == 200
        for mid in model_ids:
            assert not _cache.has(mid, CacheLevel.RAW)


class TestQuantizeCacheIntegration:
    async def test_quantize_cache_hit_skips_mlx(self, client, cache_dir):
        from fusion_model_hub.db.crud import create_model, create_version, get_quantize_task
        from fusion_model_hub.db.models import ModelFormat, ModelType, Quantization
        from fusion_model_hub.server.deps import get_cache_manager, get_session_factory
        from fusion_model_hub.server.tasks import submit_quantize

        sf = get_session_factory()
        async with sf() as session:
            model = await create_model(
                session,
                name="qc-model",
                model_type=ModelType.LLM,
                architecture="qwen2",
                params_size="7B",
            )
            src_path = _write_tmp_file(b"source weights")
            ver = await create_version(
                session,
                model_id=model.id,
                version="1.0.0",
                format=ModelFormat.MLX,
                quantization=Quantization.NONE,
                file_path=src_path,
            )
            model_id = model.id
            version_id = ver.id

        cm = get_cache_manager()
        cached_file = _write_tmp_file(b"cached 4bit output")
        # H9/R2: cache is keyed by source_version_id so a new version of the
        # same model never reuses a prior version's quantized weights.
        cm.put(model_id, CacheLevel.QUANTIZED, cached_file, quant_bits=4, source_version_id=version_id)

        with patch(
            "fusion_model_hub.server.tasks.ModelConverter.quantize",
            new_callable=AsyncMock,
        ) as mock_q:
            task_id = await submit_quantize(
                source_version_id=version_id,
                quant_bits=4,
            )
            await asyncio.sleep(0.5)
            assert mock_q.call_count == 0, "MLX quantize should be skipped on cache hit"

        async with sf() as session:
            task = await get_quantize_task(session, task_id)
            assert task is not None
            assert task.status.value == "completed"
