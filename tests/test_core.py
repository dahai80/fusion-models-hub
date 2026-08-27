"""Tests for Fusion-Model-Hub core modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from fusion_model_hub.api.base_binding import FusionMLXBase
from fusion_model_hub.convert.converter import ModelConverter
from fusion_model_hub.manage.manager import LocalModelManager
from fusion_model_hub.repo.downloader import ModelDownloader
from fusion_model_hub.repo.models import (
    ModelFormat,
    ModelInfo,
    ModelType,
    Quantization,
)
from fusion_model_hub.repo.registry import ModelRegistry

# ── ModelInfo ──


class TestModelInfo:
    def test_defaults(self):
        m = ModelInfo(id="test", name="Test Model")
        assert m.model_type == ModelType.CHAT
        assert m.quantization == Quantization.Q4
        assert m.format == ModelFormat.MLX

    def test_to_dict(self):
        m = ModelInfo(id="test", name="Test", model_type=ModelType.CODE, quantization=Quantization.Q8)
        d = m.to_dict()
        assert d["model_type"] == "code"
        assert d["quantization"] == "8bit"


# ── ModelRegistry ──


class TestModelRegistry:
    def setup_method(self):
        ModelRegistry._models.clear()

    def test_register_and_get(self):
        m = ModelInfo(id="test", name="Test")
        ModelRegistry.register(m)
        assert ModelRegistry.get("test") is m

    def test_register_no_id(self):
        with pytest.raises(ValueError):
            ModelRegistry.register(ModelInfo(id="", name=""))

    def test_list_all(self):
        ModelRegistry.register(ModelInfo(id="a", name="A"))
        ModelRegistry.register(ModelInfo(id="b", name="B"))
        assert len(ModelRegistry.list()) == 2

    def test_list_filter_type(self):
        ModelRegistry.register(ModelInfo(id="a", name="A", model_type=ModelType.CHAT))
        ModelRegistry.register(ModelInfo(id="b", name="B", model_type=ModelType.CODE))
        results = ModelRegistry.list(model_type="code")
        assert len(results) == 1
        assert results[0]["id"] == "b"

    def test_list_filter_quant(self):
        ModelRegistry.register(ModelInfo(id="a", name="A", quantization=Quantization.Q4))
        ModelRegistry.register(ModelInfo(id="b", name="B", quantization=Quantization.Q8))
        results = ModelRegistry.list(quant="8bit")
        assert len(results) == 1

    def test_list_search(self):
        ModelRegistry.register(ModelInfo(id="a", name="Qwen Model", description="A chat model"))
        ModelRegistry.register(ModelInfo(id="b", name="DeepSeek", description="Another model"))
        results = ModelRegistry.list(search="qwen")
        assert len(results) == 1

    def test_list_search_all(self):
        ModelRegistry.register(ModelInfo(id="a", name="Test", description="A model for testing"))
        results = ModelRegistry.list(search="testing")
        assert len(results) == 1

    def test_register_defaults(self):
        ModelRegistry.register_defaults()
        assert ModelRegistry.count() >= 5
        assert ModelRegistry.get("qwen2.5-7b-instruct-mlx-4bit") is not None

    def test_load_from_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"id": "m1", "name": "M1"}, {"id": "m2", "name": "M2"}], f)
            path = f.name
        try:
            count = ModelRegistry.load_from_json(path)
            assert count == 2
        finally:
            Path(path).unlink(missing_ok=True)


# ── ModelDownloader ──


class TestModelDownloader:
    @pytest.mark.asyncio
    async def test_download_invalid_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = ModelDownloader(storage_dir=tmpdir)
            result = await d.download("http://localhost:19999/nonexistent.mlx", "test")
            assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_verify_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test.txt"
            f.write_text("hello")
            hash_ok = ModelDownloader._verify_hash(f, "invalid_hash")
            assert hash_ok is False
            import hashlib

            h = hashlib.sha256(b"hello").hexdigest()
            hash_ok = ModelDownloader._verify_hash(f, h)
            assert hash_ok is True

    @pytest.mark.asyncio
    async def test_storage_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = ModelDownloader(storage_dir=tmpdir)
            info = d.get_storage_info()
            assert info["file_count"] == 0
            Path(tmpdir, "test.mlx").write_text("test")
            info = d.get_storage_info()
            assert info["file_count"] == 1


# ── ModelConverter ──


class TestModelConverter:
    @pytest.mark.asyncio
    async def test_convert_nonexistent_source(self):
        c = ModelConverter()
        result = await c.convert("/nonexistent/path")
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_convert_success(self):
        c = ModelConverter()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "output_path": "/tmp/model-q4.mlx",
                "original_size_gb": 5.0,
                "converted_size_gb": 1.5,
                "compression_ratio": 0.3,
                "compatible": True,
            }
            mock_post.return_value = mock_resp
            with tempfile.NamedTemporaryFile(suffix=".bin") as f:
                result = await c.convert(f.name, output_path="/tmp/test.mlx", quant_bits=4)
                assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_convert_api_error(self):
        c = ModelConverter()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=MagicMock(status_code=400)
            )
            with tempfile.NamedTemporaryFile(suffix=".bin") as f:
                result = await c.convert(f.name)
                assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_quantize_nonexistent(self):
        c = ModelConverter()
        result = await c.quantize("/nonexistent.mlx", bits=4)
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_quantize_success(self):
        # Upstream contract (fusion-mlx#646): POST /v1/quantize returns an
        # async job {job_id, status:queued}; poll GET /v1/quantize/jobs/{id}
        # until status=="done".
        c = ModelConverter()
        with (
            patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post,
            patch("httpx.AsyncClient.get", new=AsyncMock()) as mock_get,
        ):
            post_resp = MagicMock()
            post_resp.status_code = 200
            post_resp.json.return_value = {"job_id": "job-1", "status": "queued"}
            mock_post.return_value = post_resp
            get_resp = MagicMock()
            get_resp.status_code = 200
            get_resp.json.return_value = {
                "status": "done",
                "output_path": "/tmp/test-q4.mlx",
                "original_size_gb": 5.0,
                "converted_size_gb": 1.5,
            }
            mock_get.return_value = get_resp
            with tempfile.NamedTemporaryFile(suffix=".mlx") as f:
                result = await c.quantize(f.name, bits=4)
                assert result["status"] == "completed"
                assert result["output_path"] == "/tmp/test-q4.mlx"

    @pytest.mark.asyncio
    async def test_quantize_sync_fallback(self):
        # Older MLX may answer synchronously (no job_id). The converter must
        # still honor a direct completed response.
        c = ModelConverter()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            post_resp = MagicMock()
            post_resp.status_code = 200
            post_resp.json.return_value = {
                "status": "completed",
                "output_path": "/tmp/test-q4.mlx",
                "original_size_gb": 5.0,
                "converted_size_gb": 1.5,
            }
            mock_post.return_value = post_resp
            with tempfile.NamedTemporaryFile(suffix=".mlx") as f:
                result = await c.quantize(f.name, bits=4)
                assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_quantize_job_failed(self):
        c = ModelConverter()
        with (
            patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post,
            patch("httpx.AsyncClient.get", new=AsyncMock()) as mock_get,
        ):
            post_resp = MagicMock()
            post_resp.status_code = 200
            post_resp.json.return_value = {"job_id": "job-2", "status": "queued"}
            mock_post.return_value = post_resp
            get_resp = MagicMock()
            get_resp.status_code = 200
            get_resp.json.return_value = {"status": "failed", "error": "OOM"}
            mock_get.return_value = get_resp
            with tempfile.NamedTemporaryFile(suffix=".mlx") as f:
                result = await c.quantize(f.name, bits=4)
                assert result["status"] == "failed"
                assert "OOM" in result["error"]

    @pytest.mark.asyncio
    async def test_quantize_sync_failed_with_partial_path_not_promoted(self):
        # Regression: a sync response that explicitly reports status="failed"
        # but carries a partial output_path must NOT be coerced to "completed".
        # Before the fix, ANY non-done status with an output_path flipped to
        # completed, creating a corrupt ModelVersion from a failed quantize.
        c = ModelConverter()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            post_resp = MagicMock()
            post_resp.status_code = 200
            post_resp.json.return_value = {
                "status": "failed",
                "error": "quantize aborted",
                "output_path": "/tmp/partial-q4.mlx",
            }
            mock_post.return_value = post_resp
            with tempfile.NamedTemporaryFile(suffix=".mlx") as f:
                result = await c.quantize(f.name, bits=4)
                assert result["status"] == "failed"
                assert "quantize aborted" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_quantize_sync_missing_status_with_path_completes(self):
        # The fix narrows the coercion to a MISSING status only. Legacy MLX
        # returns the result dict with an output_path and no status field —
        # that legitimate case must still resolve to completed.
        c = ModelConverter()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            post_resp = MagicMock()
            post_resp.status_code = 200
            post_resp.json.return_value = {
                "output_path": "/tmp/test-q4.mlx",
                "original_size_gb": 5.0,
                "converted_size_gb": 1.5,
            }
            mock_post.return_value = post_resp
            with tempfile.NamedTemporaryFile(suffix=".mlx") as f:
                result = await c.quantize(f.name, bits=4)
                assert result["status"] == "completed"


# ── LocalModelManager ──


class TestLocalModelManager:
    def test_register_and_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = LocalModelManager(models_dir=tmpdir)
            m.register("test", "Test Model", "/tmp/test.mlx", quant="4bit")
            models = m.list()
            assert len(models) == 1
            assert models[0]["name"] == "Test Model"

    def test_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = LocalModelManager(models_dir=tmpdir)
            m.register("m1", "M1", "/tmp/m1.mlx")
            assert m.get("m1") is not None
            assert m.get("nonexistent") is None

    def test_unregister(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = LocalModelManager(models_dir=tmpdir)
            m.register("m1", "M1", "/tmp/m1.mlx")
            assert m.unregister("m1") is True
            assert m.unregister("nonexistent") is False
            m_reloaded = LocalModelManager(models_dir=tmpdir)
            assert m_reloaded.get("m1") is None

    def test_set_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = LocalModelManager(models_dir=tmpdir)
            m.register("m1", "M1", "/tmp/m1.mlx")
            m.register("m2", "M2", "/tmp/m2.mlx")
            assert m.set_active("m1") is True
            assert m.get("m1")["active"] is True
            assert m.get("m2")["active"] is False
            assert m.set_active("nonexistent") is False

    def test_delete_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = LocalModelManager(models_dir=tmpdir)
            model_path = Path(tmpdir) / "test.mlx"
            model_path.write_text("model data")
            m.register("test", "Test", str(model_path))
            result = m.delete_model("test")
            assert result["status"] == "deleted"
            assert not model_path.exists()

    def test_delete_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = LocalModelManager(models_dir=tmpdir)
            result = m.delete_model("nonexistent")
            assert result["status"] == "not_found"

    def test_get_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = LocalModelManager(models_dir=tmpdir)
            m.register("m1", "M1", str(Path(tmpdir) / "m1.mlx"))
            Path(tmpdir, "m1.mlx").write_text("data")
            stats = m.get_stats()
            assert stats["total_models"] == 1
            assert stats["active_models"] == 0

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = LocalModelManager(models_dir=tmpdir)
            m1.register("m1", "M1", "/tmp/m1.mlx")
            m2 = LocalModelManager(models_dir=tmpdir)
            assert len(m2.list()) == 1


# ── FusionMLXBase ──


class TestFusionMLXBase:
    @pytest.mark.asyncio
    async def test_detect_not_running(self):
        base = FusionMLXBase(mlx_url="http://localhost:19999")
        info = await base.detect()
        # Should detect not running
        assert isinstance(info, dict)

    @pytest.mark.asyncio
    async def test_detect_running(self):
        base = FusionMLXBase(mlx_url="http://localhost:11432")
        with patch("httpx.AsyncClient.get", new=AsyncMock()) as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"data": [{"id": "model1"}], "version": "0.5.0"}
            mock_get.return_value = mock_resp
            info = await base.detect()
            assert info["installed"] is True
            assert info["running"] is True

    @pytest.mark.asyncio
    async def test_check_compatibility_not_installed(self):
        base = FusionMLXBase(mlx_url="http://localhost:19999")
        result = await base.check_compatibility()
        assert result["compatible"] is False

    @pytest.mark.asyncio
    async def test_get_capabilities(self):
        base = FusionMLXBase(mlx_url="http://localhost:19999")
        caps = await base.get_capabilities()
        assert isinstance(caps, dict)

    @pytest.mark.asyncio
    async def test_check_compatibility_meets_requirement(self):
        # H10: a running MLX reporting a satisfying version is compatible.
        base = FusionMLXBase(mlx_url="http://localhost:11432")
        with patch("httpx.AsyncClient.get", new=AsyncMock()) as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"data": [], "version": "0.6.0"}
            mock_get.return_value = mock_resp
            result = await base.check_compatibility(">=0.5.0")
            assert result["compatible"] is True

    @pytest.mark.asyncio
    async def test_check_compatibility_below_requirement(self):
        # H10: an older build must NOT pass — prior code returned compatible
        # True unconditionally once running.
        base = FusionMLXBase(mlx_url="http://localhost:11432")
        with patch("httpx.AsyncClient.get", new=AsyncMock()) as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"data": [], "version": "0.4.0"}
            mock_get.return_value = mock_resp
            result = await base.check_compatibility(">=0.5.0")
            assert result["compatible"] is False


class TestEngineInvalidation:
    # E-E10: adapt/recommend engine singletons must invalidate on mlx_url OR
    # mlx_internal_api_key drift, and their embedded HardwareDetector 5-min
    # cache must clear on rebuild so a hot-reload-swapped MLX URL does not keep
    # serving stale hardware.

    def test_hardware_detector_invalidate_cache_clears(self):
        from fusion_model_hub.hardware.detector import HardwareDetector

        det = HardwareDetector("http://localhost:11434")
        det._cache = MagicMock()
        det._cache_time = 100.0
        det.invalidate_cache()
        assert det._cache is None
        assert det._cache_time == 0

    def test_hardware_detector_carries_api_key(self):
        from fusion_model_hub.hardware.detector import HardwareDetector

        det = HardwareDetector("http://localhost:11434", api_key="secret")
        assert det.api_key == "secret"
        # No key default stays empty so the auth header is omitted (backward compat
        # with an unauthenticated MLX).
        det_nokey = HardwareDetector("http://localhost:11434")
        assert det_nokey.api_key == ""

    def test_adapt_engine_propagates_api_key_to_detector(self):
        from fusion_model_hub.adapt.decision import AdaptDecisionEngine

        eng = AdaptDecisionEngine("http://localhost:11434", api_key="k1")
        assert eng.api_key == "k1"
        assert eng._hw_detector.api_key == "k1"
        eng._hw_detector._cache = MagicMock()
        eng.invalidate_cache()
        assert eng._hw_detector._cache is None

    def test_recommend_engine_propagates_api_key_to_detector(self):
        from fusion_model_hub.recommend.engine import RecommendEngine

        eng = RecommendEngine("http://localhost:11434", api_key="k2")
        assert eng.api_key == "k2"
        assert eng._hw_detector.api_key == "k2"
        eng._hw_detector._cache = MagicMock()
        eng.invalidate_cache()
        assert eng._hw_detector._cache is None

    def test_adapt_engine_headers_include_bearer(self):
        from fusion_model_hub.adapt.decision import AdaptDecisionEngine

        eng = AdaptDecisionEngine("http://localhost:11434", api_key="tok")
        assert eng._headers() == {"Authorization": "Bearer tok"}
        eng_nokey = AdaptDecisionEngine("http://localhost:11434")
        assert eng_nokey._headers() == {}

    def test_recommend_engine_headers_include_bearer(self):
        from fusion_model_hub.recommend.engine import RecommendEngine

        eng = RecommendEngine("http://localhost:11434", api_key="tok")
        assert eng._headers() == {"Authorization": "Bearer tok"}
