"""Coverage tests for Fusion-Model-Hub — targets downloader, converter, base_binding."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_model_hub.repo.downloader import ModelDownloader
from fusion_model_hub.convert.converter import ModelConverter
from fusion_model_hub.api.base_binding import FusionMLXBase


class MockResponse:
    def __init__(self, status_code=200, json_data=None, headers=None, content=b""):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {"content-length": "100"}
        self.content = content
    def raise_for_status(self):
        pass
    def json(self):
        return self._json
    def aiter_bytes(self):
        async def gen():
            yield self.content or b"test data"
        return gen()


class TestDownloaderCoverage:
    @pytest.mark.asyncio
    async def test_download_success_with_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = ModelDownloader(storage_dir=tmpdir)
            url = "http://example.com/model.mlx"
            with patch("httpx.AsyncClient") as mock_client:
                mock_ctx = MagicMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
                mock_ctx.stream = MagicMock()
                mock_ctx.stream.return_value.__aenter__ = AsyncMock(return_value=MockResponse(content=b"test data"))
                mock_client.return_value = mock_ctx
                import hashlib
                h = hashlib.sha256(b"test data").hexdigest()
                result = await d.download(url, "test-model", expected_hash=h)
                assert result["status"] == "completed"
                assert result["hash_verified"] is True

    @pytest.mark.asyncio
    async def test_download_with_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = ModelDownloader(storage_dir=tmpdir)
            progress_calls = []
            def on_progress(downloaded, total):
                progress_calls.append((downloaded, total))
            url = "http://example.com/model.mlx"
            with patch("httpx.AsyncClient") as mock_client:
                mock_ctx = MagicMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
                mock_ctx.stream = MagicMock()
                mock_ctx.stream.return_value.__aenter__ = AsyncMock(return_value=MockResponse(content=b"test data"))
                mock_client.return_value = mock_ctx
                result = await d.download(url, "test-model", on_progress=on_progress)
                assert result["status"] == "completed"
                assert len(progress_calls) >= 1

    @pytest.mark.asyncio
    async def test_download_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = ModelDownloader(storage_dir=tmpdir)
            with patch("httpx.AsyncClient") as mock_client:
                mock_ctx = MagicMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
                mock_ctx.stream = MagicMock()
                mock_ctx.stream.return_value.__aenter__ = AsyncMock(return_value=MockResponse(content=b"test data"))
                mock_client.return_value = mock_ctx
                result = await d.download("http://example.com/m.mlx", "test", expected_hash="badhash")
                assert result["status"] == "hash_mismatch"
                assert result["hash_verified"] is False

    @pytest.mark.asyncio
    async def test_verify_local(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test.mlx"
            f.write_text("test data")
            d = ModelDownloader(storage_dir=tmpdir)
            import hashlib
            h = hashlib.sha256(b"test data").hexdigest()
            assert d.verify_local(f, h) is True
            assert d.verify_local(f, "badhash") is False


class TestConverterCoverage:
    @pytest.mark.asyncio
    async def test_convert_from_hf(self):
        c = ModelConverter()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"output_path": "/tmp/model.mlx", "original_size_gb": 5.0, "converted_size_gb": 1.5}
            mock_post.return_value = mock_resp
            result = await c.convert_from_hf("Qwen/Qwen2.5-7B", quant_bits=4)
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_detect_format(self):
        assert ModelConverter._detect_format("model.safetensors") == "huggingface"
        assert ModelConverter._detect_format("model.bin") == "pytorch"
        assert ModelConverter._detect_format("model.gguf") == "gguf"
        assert ModelConverter._detect_format("model.mlx") == "mlx"
        assert ModelConverter._detect_format("model.pt") == "pytorch"
        assert ModelConverter._detect_format("model.onnx") == "onnx"
        assert ModelConverter._detect_format("model.unknown") == "unknown"
        assert ModelConverter._detect_format("", hf_repo="Qwen/Qwen") == "huggingface"


class TestBaseBindingCoverage:
    @pytest.mark.asyncio
    async def test_detect_running(self):
        base = FusionMLXBase()
        with patch("httpx.AsyncClient.get", new=AsyncMock()) as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"data": [{"id": "m1"}], "version": "0.6.0"}
            mock_get.return_value = mock_resp
            info = await base.detect()
            assert info["installed"] is True

    @pytest.mark.asyncio
    async def test_detect_installed_not_running(self):
        base = FusionMLXBase()
        with patch("httpx.AsyncClient.get", side_effect=Exception("fail")):
            with patch("shutil.which", return_value="/usr/local/bin/fusion-mlx"):
                info = await base.detect()
                assert info["installed"] is True
                assert info["running"] is False

    @pytest.mark.asyncio
    async def test_get_capabilities(self):
        base = FusionMLXBase()
        with patch("httpx.AsyncClient.get", new=AsyncMock()) as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {}
            mock_get.return_value = mock_resp
            caps = await base.get_capabilities()
            assert caps["metal_available"] is True