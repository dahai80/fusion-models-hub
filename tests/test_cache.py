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
        assert got.ref_count == 1
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


class TestQuantizeCacheIntegration:
    async def test_quantize_cache_hit_skips_mlx(self, client, cache_dir):
        from fusion_model_hub.db.crud import create_model, create_version, get_quantize_task
        from fusion_model_hub.db.models import ModelFormat, ModelType, Quantization
        from fusion_model_hub.server.deps import get_cache_manager, get_session_factory
        from fusion_model_hub.server.tasks import submit_quantize

        sf = get_session_factory()
        async with sf() as session:
            model = await create_model(
                session, name="qc-model", model_type=ModelType.LLM,
                architecture="qwen2", params_size="7B",
            )
            src_path = _write_tmp_file(b"source weights")
            ver = await create_version(
                session, model_id=model.id, version="1.0.0",
                format=ModelFormat.MLX, quantization=Quantization.NONE,
                file_path=src_path,
            )
            model_id = model.id
            version_id = ver.id

        cm = get_cache_manager()
        cached_file = _write_tmp_file(b"cached 4bit output")
        cm.put(model_id, CacheLevel.QUANTIZED, cached_file, quant_bits=4)

        with patch(
            "fusion_model_hub.server.tasks.ModelConverter.quantize",
            new_callable=AsyncMock,
        ) as mock_q:
            task_id = await submit_quantize(
                source_version_id=version_id, quant_bits=4,
            )
            await asyncio.sleep(0.5)
            assert mock_q.call_count == 0, "MLX quantize should be skipped on cache hit"

        async with sf() as session:
            task = await get_quantize_task(session, task_id)
            assert task is not None
            assert task.status.value == "completed"
