import logging
import shutil
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import dispose_all_engines, get_engine, get_session_factory, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps

logger = logging.getLogger(__name__)


@pytest.fixture
def settings():
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_cov_db_core",
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
    shutil.rmtree("/tmp/fmh_cov_db_core", ignore_errors=True)


# =====================================================================
# db/database.py: engine helpers
# =====================================================================


class TestDatabaseHelpers:
    async def test_get_engine_default_url(self):
        engine = get_engine()
        assert engine is not None
        await engine.dispose()

    async def test_get_engine_file_sqlite_pragmas_attached(self, tmp_path):
        db_path = tmp_path / "file.db"
        engine = get_engine(f"sqlite+aiosqlite:///{db_path}")
        async with engine.connect() as conn:
            from sqlalchemy import text
            result = await conn.execute(text("PRAGMA journal_mode"))
            mode = result.scalar()
        assert mode in ("wal", "WAL", "memory", "delete")
        await engine.dispose()

    async def test_dispose_all_engines_handles_errors(self):
        bad = MagicMock()
        bad.dispose = AsyncMock(side_effect=RuntimeError("boom"))
        from fusion_model_hub.db import database as dbmod
        dbmod._engines.append(bad)
        await dispose_all_engines()

    async def test_get_engine_server_db_pool_kwargs(self):
        engine = get_engine(
            "postgresql+asyncpg://u:p@nonhost:5432/db", pool_size=5, max_overflow=3,
        )
        assert engine is not None
        await engine.dispose()


# =====================================================================
# server/ssrf.py: validate_external_url (raises on bad, None on good)
# =====================================================================


class TestSsrfValidation:
    def _v(self, url, **kw):
        from fusion_model_hub.server.ssrf import validate_external_url
        validate_external_url(url, **kw)

    def test_allows_http_https(self):
        with patch("fusion_model_hub.server.ssrf.socket.getaddrinfo", return_value=[]):
            self._v("http://example.com/a")
            self._v("https://example.com/a")

    def test_rejects_non_http_schemes(self):
        from fastapi import HTTPException
        for u in ("file:///etc/passwd", "ftp://example.com", "gopher://x"):
            with pytest.raises(HTTPException):
                self._v(u)

    def test_rejects_localhost(self):
        from fastapi import HTTPException
        for u in ("http://localhost", "http://127.0.0.1", "http://[::1]"):
            with pytest.raises(HTTPException):
                self._v(u)

    def test_rejects_internal_cidr(self):
        from fastapi import HTTPException
        for u in (
            "http://10.0.0.1", "http://172.16.0.1", "http://172.31.0.1",
            "http://192.168.1.1", "http://169.254.169.254", "http://0.0.0.0",
        ):
            with pytest.raises(HTTPException):
                self._v(u)

    def test_rejects_numeric_ip_encoding(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            self._v("http://2130706433")
        with pytest.raises(HTTPException):
            self._v("http://0x7f000001")

    def test_allows_public_ip(self):
        self._v("http://8.8.8.8/path")

    def test_rejects_missing_hostname(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            self._v("http://")

    def test_https_only_flag(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            self._v("http://example.com", allow_https_only=True)
        with patch("fusion_model_hub.server.ssrf.socket.getaddrinfo", return_value=[]):
            self._v("https://example.com", allow_https_only=True)

    def test_dns_rebinding_internal_resolution_rejected(self):
        from fastapi import HTTPException
        fake_info = [("AF_INET", "STREAM", "TCP", "", ("10.0.0.5", 0))]
        with patch("fusion_model_hub.server.ssrf.socket.getaddrinfo", return_value=fake_info):
            with pytest.raises(HTTPException):
                self._v("http://public-looking-host.com")

    def test_unresolvable_hostname_passes(self):
        import socket
        with patch("fusion_model_hub.server.ssrf.socket.getaddrinfo", side_effect=socket.gaierror):
            self._v("http://typo-host-not-real.com")


# =====================================================================
# storage/local_store.py: chunked upload + assemble + guards
# =====================================================================


class TestLocalStore:
    def _store(self, tmp_path):
        from fusion_model_hub.storage.local_store import LocalStore
        return LocalStore(data_dir=str(tmp_path / "store"))

    async def test_write_chunk_and_assemble(self, tmp_path):
        store = self._store(tmp_path)
        upload_id = "up-assemble"
        await store.write_chunk(upload_id, 0, b"AAAA")
        await store.write_chunk(upload_id, 1, b"BBBB")
        target_dir = store.models_dir / "m1" / "v1"
        target_dir.mkdir(parents=True, exist_ok=True)
        path, file_hash, size = await store.assemble_chunks(
            upload_id, target_dir, "model.bin", 2,
        )
        assert size == 8
        assert path.exists()
        import hashlib
        assert file_hash == hashlib.sha256(b"AAAABBBB").hexdigest()

    async def test_assemble_missing_chunk_raises(self, tmp_path):
        store = self._store(tmp_path)
        upload_id = "up-missing"
        await store.write_chunk(upload_id, 0, b"X")
        target_dir = store.models_dir / "m1" / "v1"
        target_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            await store.assemble_chunks(upload_id, target_dir, "ok.bin", 2)

    async def test_assemble_invalid_filename_raises(self, tmp_path):
        store = self._store(tmp_path)
        upload_id = "up-bad"
        await store.write_chunk(upload_id, 0, b"X")
        target_dir = store.models_dir / "m1" / "v1"
        target_dir.mkdir(parents=True, exist_ok=True)
        for bad in ("", ".", ".."):
            with pytest.raises(ValueError):
                await store.assemble_chunks(upload_id, target_dir, bad, 1)

    async def test_assemble_traversal_filename_sanitized(self, tmp_path):
        store = self._store(tmp_path)
        upload_id = "up-trav"
        await store.write_chunk(upload_id, 0, b"X")
        target_dir = store.models_dir / "m1" / "v1"
        target_dir.mkdir(parents=True, exist_ok=True)
        path, _, _ = await store.assemble_chunks(upload_id, target_dir, "../escape.bin", 1)
        assert path.name == "escape.bin"
        assert path.parent == target_dir

    async def test_get_file_existing(self, tmp_path):
        store = self._store(tmp_path)
        await store.write_chunk("up-gf", 0, b"Y")
        target_dir = store.models_dir / "m1" / "v1"
        target_dir.mkdir(parents=True, exist_ok=True)
        path, _, _ = await store.assemble_chunks("up-gf", target_dir, "ok.bin", 1)
        assert store.get_file(str(path)) is not None
        assert store.get_file("nonexistent.bin") is None

    async def test_delete_version_files(self, tmp_path):
        store = self._store(tmp_path)
        vd = store.model_version_dir("m-del", "v1")
        (vd / "f.bin").write_bytes(b"data")
        assert store.delete_version_files("m-del", "v1")
        assert not vd.exists()
        assert not store.delete_version_files("m-del", "v1")

    async def test_delete_model_files(self, tmp_path):
        store = self._store(tmp_path)
        store.model_version_dir("m-dm", "v1")
        assert store.delete_model_files("m-dm")
        assert not store.delete_model_files("m-dm")

    async def test_write_file_atomic(self, tmp_path):
        store = self._store(tmp_path)
        target_dir = store.models_dir / "m1" / "v1"
        target_dir.mkdir(parents=True, exist_ok=True)
        path, file_hash, size = await store.write_file(target_dir, "wf.bin", b"hello")
        assert size == 5
        assert path.exists()
        assert file_hash == __import__("hashlib").sha256(b"hello").hexdigest()

    async def test_lfs_put_get(self, tmp_path):
        store = self._store(tmp_path)
        path = store.put_lfs_object("abc123", b"lfs-data")
        assert path.exists()
        got = store.get_lfs_object("abc123")
        assert got is not None
        assert got.read_bytes() == b"lfs-data"
        assert store.get_lfs_object("nonexistent") is None
        with pytest.raises(ValueError):
            store.put_lfs_object("../escape", b"x")

    async def test_is_path_within_store(self, tmp_path):
        store = self._store(tmp_path)
        inside = store.models_dir / "m1" / "v1" / "f.bin"
        outside = tmp_path / "elsewhere.bin"
        assert store.is_path_within_store(inside)
        assert not store.is_path_within_store(outside)

    async def test_storage_stats(self, tmp_path):
        store = self._store(tmp_path)
        store.model_version_dir("m1", "v1")
        stats = store.get_storage_stats()
        assert stats["model_count"] == 1
        assert "total_size_gb" in stats


# =====================================================================
# hardware/detector.py: detect + fallback + cache
# =====================================================================


def _hw_gpu_data():
    return {
        "gpu": {
            "name": "Apple M2 Max", "vendor": "Apple", "vram_bytes": 32_000_000_000,
            "memory_bandwidth_gbps": 400, "shared_memory": True,
        },
        "cpu": {"name": "Apple M2 Max", "cores": 12},
        "ram": {"total_bytes": 32_000_000_000, "total_gb": 32.0},
        "disk": {"free_bytes": 100_000_000_000, "free_gb": 100.0},
        "os": "macos",
    }


class TestHardwareDetector:
    async def test_detect_returns_profile(self):
        from fusion_model_hub.hardware.detector import HardwareDetector
        det = HardwareDetector(mlx_url="http://mlx.test:11434")
        mock = MagicMock()
        mock.status_code = 200
        mock.json = MagicMock(return_value=_hw_gpu_data())
        mock.raise_for_status = MagicMock()
        ctx = AsyncMock()
        ctx.get = AsyncMock(return_value=mock)
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            profile = await det.detect()
        assert profile is not None
        assert profile.gpu is not None
        from fusion_model_hub.hardware.types import ChipGeneration
        assert profile.gpu.chip_generation == ChipGeneration.M2_MAX

    async def test_detect_mlx_error_returns_fallback(self):
        from fusion_model_hub.hardware.detector import HardwareDetector
        det = HardwareDetector(mlx_url="http://mlx.test:11434")
        import httpx as _httpx
        ctx = AsyncMock()
        ctx.get = AsyncMock(side_effect=_httpx.ConnectError("refused"))
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            profile = await det.detect()
        assert profile is not None
        assert profile.gpu is None
        assert profile.os_name == "unknown"

    async def test_detect_uses_cache_on_second_call(self):
        from fusion_model_hub.hardware.detector import HardwareDetector
        det = HardwareDetector(mlx_url="http://mlx.test:11434")
        mock = MagicMock()
        mock.status_code = 200
        mock.json = MagicMock(return_value=_hw_gpu_data())
        mock.raise_for_status = MagicMock()
        ctx = AsyncMock()
        ctx.get = AsyncMock(return_value=mock)
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            p1 = await det.detect()
            p2 = await det.detect()
        assert ctx.get.await_count == 1
        assert p1 is p2

    def test_invalidate_cache(self):
        from fusion_model_hub.hardware.detector import HardwareDetector
        det = HardwareDetector(mlx_url="http://mlx.test:11434")
        det._cache = "stale"
        det._cache_time = 999
        det.invalidate_cache()
        assert det._cache is None


# =====================================================================
# recommend/engine.py: RecommendEngine.recommend
# =====================================================================


def _fake_hw():
    from fusion_model_hub.hardware.types import CPUProfile, HardwareProfile
    return HardwareProfile(
        gpu=None, cpu=CPUProfile(name="x", cores=8),
        ram_bytes=0, ram_gb=32.0, disk_free_bytes=0, disk_free_gb=100.0, os_name="macos",
    )


class TestRecommendEngine:
    def _mlx_response(self, can_run=True):
        return {
            "results": [
                {
                    "model_id": "m1", "can_run": can_run, "fit_type": "full",
                    "vram_required_gb": 4.0, "vram_available_gb": 32.0,
                    "estimated_tok_per_sec": 120.0,
                },
                {
                    "model_id": "m2", "can_run": False, "fit_type": "none",
                    "vram_required_gb": 80.0, "vram_available_gb": 32.0,
                    "estimated_tok_per_sec": 0.0,
                },
            ]
        }

    def _client_ctx(self, mock_resp, side_effect=None):
        mock = MagicMock()
        mock.status_code = 200
        mock.json = MagicMock(return_value=mock_resp)
        mock.raise_for_status = MagicMock()
        ctx = AsyncMock()
        ctx.post = AsyncMock(side_effect=side_effect) if side_effect else AsyncMock(return_value=mock)
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    async def test_recommend_ranks_runnable_first(self):
        from fusion_model_hub.recommend.engine import RecommendEngine
        from fusion_model_hub.recommend.types import RecommendRequest
        eng = RecommendEngine(mlx_url="http://mlx.test:11434")
        with patch.object(eng._hw_detector, "detect", AsyncMock(return_value=_fake_hw())):
            with patch("httpx.AsyncClient", return_value=self._client_ctx(self._mlx_response())):
                resp = await eng.recommend(
                    RecommendRequest(task="llm", max_results=10),
                    [
                        {"id": "m1", "name": "small", "params_b": 4, "task": "llm"},
                        {"id": "m2", "name": "big", "params_b": 70, "task": "llm"},
                    ],
                )
        ids = [r.model_id for r in resp.recommendations]
        assert ids.index("m1") < ids.index("m2")
        assert resp.total_evaluated == 2

    async def test_recommend_filters_by_task(self):
        from fusion_model_hub.recommend.engine import RecommendEngine
        from fusion_model_hub.recommend.types import RecommendRequest
        eng = RecommendEngine(mlx_url="http://mlx.test:11434")
        with patch.object(eng._hw_detector, "detect", AsyncMock(return_value=_fake_hw())):
            with patch("httpx.AsyncClient", return_value=self._client_ctx({"results": []})):
                resp = await eng.recommend(
                    RecommendRequest(task="embedding"),
                    [
                        {"id": "m1", "name": "x", "params_b": 1, "task": "llm"},
                        {"id": "m2", "name": "y", "params_b": 1, "task": "embedding"},
                    ],
                )
        assert resp.total_evaluated == 1
        assert resp.recommendations[0].model_id == "m2"

    async def test_recommend_mlx_error_falls_back_single(self):
        from fusion_model_hub.recommend.engine import RecommendEngine
        from fusion_model_hub.recommend.types import RecommendRequest
        eng = RecommendEngine(mlx_url="http://mlx.test:11434")
        with patch.object(eng._hw_detector, "detect", AsyncMock(return_value=_fake_hw())):
            ctx = AsyncMock()
            ctx.post = AsyncMock(side_effect=RuntimeError("net err"))
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            with patch("httpx.AsyncClient", return_value=ctx):
                resp = await eng.recommend(
                    RecommendRequest(task="llm"),
                    [{"id": "m1", "name": "x", "params_b": 4, "task": "llm"}],
                )
        assert len(resp.recommendations) == 1
        assert resp.recommendations[0].can_run is False


# =====================================================================
# convert/converter.py: convert / convert_from_hf / quantize
# =====================================================================


class TestModelConverter:
    async def test_convert_missing_source_returns_failed(self):
        from fusion_model_hub.convert.converter import ModelConverter
        conv = ModelConverter(mlx_url="http://mlx.test:11434")
        result = await conv.convert(source_path="/nonexistent/path/model.safetensors")
        assert result["status"] == "failed"

    async def test_convert_success(self, tmp_path):
        from fusion_model_hub.convert.converter import ModelConverter
        conv = ModelConverter(mlx_url="http://mlx.test:11434")
        src = tmp_path / "model.safetensors"
        src.write_bytes(b"weights")
        mock = MagicMock()
        mock.status_code = 200
        mock.json = MagicMock(return_value={
            "output_path": str(tmp_path / "out.mlx"),
            "original_size_gb": 1.0, "converted_size_gb": 0.5, "compression_ratio": 2.0,
        })
        mock.raise_for_status = MagicMock()
        ctx = AsyncMock()
        ctx.post = AsyncMock(return_value=mock)
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            result = await conv.convert(source_path=str(src), quant_bits=4)
        assert result["status"] == "completed"
        assert result["source_format"] == "huggingface"

    async def test_convert_from_hf(self):
        from fusion_model_hub.convert.converter import ModelConverter
        conv = ModelConverter(mlx_url="http://mlx.test:11434")
        mock = MagicMock()
        mock.status_code = 200
        mock.json = MagicMock(return_value={"output_path": "/tmp/out.mlx"})
        mock.raise_for_status = MagicMock()
        ctx = AsyncMock()
        ctx.post = AsyncMock(return_value=mock)
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            result = await conv.convert_from_hf("Qwen/Qwen2.5-7B", quant_bits=4)
        assert result["status"] == "completed"

    async def test_convert_http_error_returns_failed(self, tmp_path):
        from fusion_model_hub.convert.converter import ModelConverter
        conv = ModelConverter(mlx_url="http://mlx.test:11434")
        src = tmp_path / "model.gguf"
        src.write_bytes(b"x")
        import httpx as _httpx
        mock = MagicMock()
        mock.status_code = 500
        mock.text = "err"
        err = _httpx.HTTPStatusError("500", request=MagicMock(), response=mock)
        ctx = AsyncMock()
        ctx.post = AsyncMock(side_effect=err)
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            result = await conv.convert(source_path=str(src))
        assert result["status"] == "failed"

    async def test_quantize_missing_file_returns_failed(self):
        from fusion_model_hub.convert.converter import ModelConverter
        conv = ModelConverter(mlx_url="http://mlx.test:11434")
        result = await conv.quantize("/nonexistent.mlx", bits=4)
        assert result["status"] == "failed"

    async def test_quantize_success(self, tmp_path):
        from fusion_model_hub.convert.converter import ModelConverter
        conv = ModelConverter(mlx_url="http://mlx.test:11434")
        src = tmp_path / "model.mlx"
        src.write_bytes(b"mlx-weights")
        mock = MagicMock()
        mock.status_code = 200
        mock.json = MagicMock(return_value={
            "output_path": str(src), "file_hash": "abc", "file_size": 10,
        })
        mock.raise_for_status = MagicMock()
        ctx = AsyncMock()
        ctx.post = AsyncMock(return_value=mock)
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            result = await conv.quantize(str(src), bits=4)
        assert result["status"] == "completed"
        assert result["file_hash"] == "abc"

    def test_detect_format(self):
        from fusion_model_hub.convert.converter import ModelConverter
        assert ModelConverter._detect_format("m.safetensors") == "huggingface"
        assert ModelConverter._detect_format("m.gguf") == "gguf"
        assert ModelConverter._detect_format("", hf_repo="Qwen/x") == "huggingface"
        assert ModelConverter._detect_format("m.unknownext") == "unknown"


# =====================================================================
# repo/downloader.py: ModelDownloader.download
# =====================================================================


def _stream_cm(stream_obj):
    class _Cm:
        def __init__(self, s):
            self.s = s
        async def __aenter__(self):
            return self.s
        async def __aexit__(self, *a):
            return False
    return _Cm(stream_obj)


class TestModelDownloader:
    async def test_download_writes_file(self, tmp_path):
        from fusion_model_hub.repo.downloader import ModelDownloader
        dl = ModelDownloader(storage_dir=str(tmp_path / "models"))
        stream = AsyncMock()
        stream.status_code = 200
        stream.headers = {"content-length": "12"}

        async def _aiter():
            yield b"chunk1"
            yield b"chunk2"
        stream.aiter_bytes = _aiter
        stream.raise_for_status = MagicMock()

        ctx = MagicMock()
        ctx.stream = MagicMock(return_value=_stream_cm(stream))
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            result = await dl.download("http://example.com/m.mlx", "model-1")
        assert result["status"] == "completed"
        assert result["hash_verified"] is True
        assert (tmp_path / "models" / "model-1.mlx").exists()

    async def test_download_resume_sends_range(self, tmp_path):
        from fusion_model_hub.repo.downloader import ModelDownloader
        dl = ModelDownloader(storage_dir=str(tmp_path / "models"))
        (tmp_path / "models" / "model-r.mlx.part").write_bytes(b"already-here")
        stream = AsyncMock()
        stream.status_code = 206
        stream.headers = {"content-range": "bytes 11-23/24"}

        async def _aiter():
            yield b"-more"
        stream.aiter_bytes = _aiter
        stream.raise_for_status = MagicMock()

        captured_headers = {}

        class _Ctx:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            def stream(self, method, url, **kw):
                captured_headers.update(kw.get("headers", {}))
                return _stream_cm(stream)

        with patch("httpx.AsyncClient", return_value=_Ctx()):
            result = await dl.download("http://example.com/r.mlx", "model-r")
        assert result["status"] == "completed"
        assert result["resumed"] is True
        assert captured_headers.get("Range") == "bytes=12-"

    async def test_download_hash_mismatch_deletes_file(self, tmp_path):
        from fusion_model_hub.repo.downloader import ModelDownloader
        dl = ModelDownloader(storage_dir=str(tmp_path / "models"))
        stream = AsyncMock()
        stream.status_code = 200
        stream.headers = {"content-length": "5"}

        async def _aiter():
            yield b"hello"
        stream.aiter_bytes = _aiter
        stream.raise_for_status = MagicMock()

        ctx = MagicMock()
        ctx.stream = MagicMock(return_value=_stream_cm(stream))
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            result = await dl.download(
                "http://example.com/m.mlx", "model-m", expected_hash="deadbeef" * 8,
            )
        assert result["status"] == "hash_mismatch"
        assert result["hash_verified"] is False
        assert not (tmp_path / "models" / "model-m.mlx").exists()

    async def test_download_network_error_returns_failed(self, tmp_path):
        from fusion_model_hub.repo.downloader import ModelDownloader
        dl = ModelDownloader(storage_dir=str(tmp_path / "models"))
        import httpx as _httpx
        ctx = MagicMock()
        ctx.stream = MagicMock(side_effect=_httpx.ConnectError("refused"))
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            result = await dl.download("http://example.com/m.mlx", "model-f")
        assert result["status"] == "failed"

    def test_get_storage_info(self, tmp_path):
        from fusion_model_hub.repo.downloader import ModelDownloader
        dl = ModelDownloader(storage_dir=str(tmp_path / "models"))
        (tmp_path / "models" / "a.mlx").write_bytes(b"x" * 100)
        info = dl.get_storage_info()
        assert info["file_count"] == 1


# =====================================================================
# repo/modelscope_search.py: search_modelscope
# =====================================================================


class TestModelscopeSearch:
    async def test_search_returns_results(self):
        from fusion_model_hub.repo.modelscope_search import search_modelscope
        mock = MagicMock()
        mock.status_code = 200
        mock.json = MagicMock(return_value={
            "Data": {"Models": [{"Name": "test-model", "Id": "m1", "Task": "llm"}], "TotalCount": 1},
        })
        mock.raise_for_status = MagicMock()
        ctx = AsyncMock()
        ctx.get = AsyncMock(return_value=mock)
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            result = await search_modelscope("test")
        assert result["total"] == 1
        assert result["items"][0]["name"] == "test-model"
        assert result["source"] == "modelscope"

    async def test_search_connect_error_returns_empty(self):
        import httpx as _httpx

        from fusion_model_hub.repo.modelscope_search import search_modelscope
        ctx = AsyncMock()
        ctx.get = AsyncMock(side_effect=_httpx.ConnectError("net err"))
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            result = await search_modelscope("test")
        assert result["items"] == []
        assert result["total"] == 0

    async def test_search_http_status_error_returns_empty(self):
        import httpx as _httpx

        from fusion_model_hub.repo.modelscope_search import search_modelscope
        mock = MagicMock()
        mock.status_code = 500
        mock.text = "err"
        err = _httpx.HTTPStatusError("500", request=MagicMock(), response=mock)
        ctx = AsyncMock()
        ctx.get = AsyncMock(side_effect=err)
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            result = await search_modelscope("test")
        assert result["items"] == []


# =====================================================================
# db/crud.py: less-common CRUD funcs
# =====================================================================


class TestCrudEdgeFuncs:
    async def _session(self):
        engine = get_engine("sqlite+aiosqlite:///:memory:")
        await init_db(engine)
        sf = get_session_factory(engine)
        return sf, engine

    async def test_create_and_get_security_scan(self):
        from fusion_model_hub.db import crud
        sf, engine = await self._session()
        async with sf() as session:
            m = await crud.create_model(session, name="sec-model")
            scan = await crud.create_security_scan(session, model_id=m.id, scan_type="vuln")
            assert scan.id
            got = await crud.get_security_scan(session, scan.id)
            assert got is not None
        await engine.dispose()

    async def test_create_watermark(self):
        from fusion_model_hub.db import crud
        sf, engine = await self._session()
        async with sf() as session:
            m = await crud.create_model(session, name="wm-model")
            wm = await crud.create_watermark(session, model_id=m.id, payload='{"k":1}')
            assert wm.id
        await engine.dispose()

    async def test_create_distributed_task(self):
        from fusion_model_hub.db import crud
        sf, engine = await self._session()
        async with sf() as session:
            m = await crud.create_model(session, name="dist-model")
            v = await crud.create_version(session, model_id=m.id, version="1.0")
            dt = await crud.create_distributed_task(
                session, model_id=m.id, version_id=v.id, target_nodes='["n1"]',
            )
            assert dt.id
        await engine.dispose()

    async def test_list_models_pagination_and_filter(self):
        from fusion_model_hub.db import crud
        sf, engine = await self._session()
        async with sf() as session:
            for i in range(5):
                await crud.create_model(session, name=f"p-model-{i}")
            page1, total = await crud.list_models(session, page=1, page_size=2)
            assert len(page1) == 2
            assert total == 5
            page3, _ = await crud.list_models(session, page=3, page_size=2)
            assert len(page3) == 1
            filtered, ft = await crud.list_models(session, keyword="p-model-0")
            assert ft == 1
        await engine.dispose()

    async def test_update_model_partial_whitelist(self):
        from fusion_model_hub.db import crud
        sf, engine = await self._session()
        async with sf() as session:
            m = await crud.create_model(session, name="upd-model")
            updated = await crud.update_model(session, m.id, description="changed")
            assert updated.description == "changed"
            assert updated.name == "upd-model"
        await engine.dispose()

    async def test_create_quantize_task(self):
        from fusion_model_hub.db import crud
        sf, engine = await self._session()
        async with sf() as session:
            m = await crud.create_model(session, name="qt-model")
            v = await crud.create_version(session, model_id=m.id, version="1.0")
            t = await crud.create_quantize_task(
                session, source_version_id=v.id, target_format="mlx", quant_bits=4,
            )
            assert t.id
            assert t.quant_bits == 4
        await engine.dispose()


# =====================================================================
# server/routers/models.py: batch/sync/compare/publish
# =====================================================================


class TestModelsRoutersDeep:
    async def _seed(self, client, name="deep-model"):
        r = await client.post("/api/v1/models", json={"name": name, "model_type": "llm"})
        assert r.status_code == 201, r.text
        return r.json()

    async def test_batch_delete_models(self, client):
        m1 = await self._seed(client, "bd-1")
        m2 = await self._seed(client, "bd-2")
        r = await client.post("/api/v1/models/batch/delete", json={"model_ids": [m1["id"], m2["id"]]})
        assert r.status_code == 200
        assert r.json()["count"] == 2
        assert len(r.json()["deleted"]) == 2

    async def test_batch_tag_models(self, client):
        m1 = await self._seed(client, "bt-1")
        r = await client.post("/api/v1/models/batch/tag", json={
            "model_ids": [m1["id"]], "tags": [{"key": "k", "value": "v"}],
        })
        assert r.status_code == 200

    async def test_compare_models(self, client):
        m1 = await self._seed(client, "cmp-1")
        m2 = await self._seed(client, "cmp-2")
        r = await client.get(f"/api/v1/models/compare?ids={m1['id']},{m2['id']}")
        assert r.status_code == 200
        assert len(r.json()["models"]) == 2

    async def test_compare_requires_two(self, client):
        m1 = await self._seed(client, "cmp-solo")
        r = await client.get(f"/api/v1/models/compare?ids={m1['id']}")
        assert r.status_code == 400

    async def test_compare_missing_model_404(self, client):
        m1 = await self._seed(client, "cmp-miss")
        r = await client.get(f"/api/v1/models/compare?ids={m1['id']},nonexistent-id")
        assert r.status_code == 404

    async def test_sync_models_dry_run(self, client):
        mock = MagicMock()
        mock.status_code = 200
        mock.json = MagicMock(return_value={"items": [
            {"name": "synced-1", "model_type": "llm", "description": "d"},
        ]})
        mock.raise_for_status = MagicMock()
        ctx = AsyncMock()
        ctx.get = AsyncMock(return_value=mock)
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            r = await client.post("/api/v1/models/sync", json={
                "source_url": "https://example.com", "dry_run": True, "source": "huggingface",
            })
        assert r.status_code == 200
        assert r.json()["dry_run"] is True
        assert r.json()["new_count"] == 1

    async def test_sync_models_ssrf_rejected(self, client):
        r = await client.post("/api/v1/models/sync", json={
            "source_url": "http://127.0.0.1", "dry_run": True, "source": "huggingface",
        })
        assert r.status_code == 400

    async def test_sync_models_fetch_error_502(self, client):
        import httpx as _httpx
        ctx = AsyncMock()
        ctx.get = AsyncMock(side_effect=_httpx.HTTPError("net err"))
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            r = await client.post("/api/v1/models/sync", json={
                "source_url": "https://example.com", "dry_run": False, "source": "huggingface",
            })
        assert r.status_code == 502

    async def test_publish_model(self, client):
        m = await self._seed(client, "pub-1")
        r = await client.post(f"/api/v1/models/{m['id']}/publish")
        assert r.status_code == 200
        assert r.json()["model_status"] == "published"

    async def test_publish_missing_model_404(self, client):
        r = await client.post("/api/v1/models/nonexistent-id/publish")
        assert r.status_code == 404

    async def test_deprecate_model(self, client):
        m = await self._seed(client, "dep-1")
        r = await client.post(f"/api/v1/models/{m['id']}/deprecate")
        assert r.status_code == 200
        assert r.json()["model_status"] == "deprecated"

    async def test_import_hf(self, client):
        mock = MagicMock()
        mock.status_code = 200
        mock.json = MagicMock(return_value={
            "pipeline_tag": "text-generation",
            "description": "test model",
            "config": {"architectures": ["Qwen2ForCausalLM"]},
            "safetensors": {"total": 7000000000},
            "cardData": {"license": "apache-2.0", "language": ["en"]},
            "author": "Qwen",
        })
        mock.raise_for_status = MagicMock()
        ctx = AsyncMock()
        ctx.get = AsyncMock(return_value=mock)
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=ctx):
            r = await client.post("/api/v1/models/import/hf", json={
                "hf_repo": "Qwen/Qwen2.5-7B", "download": False,
            })
        assert r.status_code == 201
        assert r.json()["name"] == "qwen2.5-7b"

    async def test_import_hf_missing_repo_400(self, client):
        r = await client.post("/api/v1/models/import/hf", json={"hf_repo": ""})
        assert r.status_code == 400


# =====================================================================
# server/routers/recommend.py: POST /recommend + GET /quick
# =====================================================================


class TestRecommendRouter:
    def _fake_engine(self, fake_resp):
        from fusion_model_hub.recommend.types import (
            RecommendResponse,
        )
        if fake_resp is None:
            fake_resp = RecommendResponse(
                recommendations=[], hardware_summary={"chip": "M2"}, total_evaluated=0,
            )
        return MagicMock(recommend=AsyncMock(return_value=fake_resp))

    async def test_recommend_endpoint(self, client):
        from fusion_model_hub.recommend.types import (
            ModelRecommendation,
            RecommendResponse,
        )
        fake_resp = RecommendResponse(
            recommendations=[ModelRecommendation(
                model_id="m1", name="x", task="llm", params_b=4, quant_type="Q4",
                can_run=True, fit_type="full", vram_required_gb=4.0,
                vram_available_gb=32.0, estimated_tok_per_sec=120.0,
                rank_score=90.0, quality_score=80.0, speed_score=70.0,
                hardware_score=95.0, popularity_score=50.0, reason="ok",
            )],
            hardware_summary={"chip": "M2", "vram_gb": 32.0, "ram_gb": 32.0},
            total_evaluated=1,
        )
        with patch(
            "fusion_model_hub.server.routers.recommend._get_engine",
            return_value=self._fake_engine(fake_resp),
        ):
            r = await client.post("/api/v1/recommend", json={"task": "llm", "max_results": 5})
        assert r.status_code == 200
        assert len(r.json()["recommendations"]) == 1

    async def test_quick_recommend_endpoint(self, client):
        with patch(
            "fusion_model_hub.server.routers.recommend._get_engine",
            return_value=self._fake_engine(None),
        ):
            r = await client.get("/api/v1/recommend/quick?task=llm")
        assert r.status_code == 200

    async def test_recommend_engine_failure_503(self, client):
        engine = MagicMock()
        engine.recommend = AsyncMock(side_effect=RuntimeError("boom"))
        with patch(
            "fusion_model_hub.server.routers.recommend._get_engine",
            return_value=engine,
        ):
            r = await client.post("/api/v1/recommend", json={"task": "llm"})
        assert r.status_code == 503
