from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

logger = logging.getLogger(__name__)


class TestScorer:
    def test_score_hardware_fit_cannot_run(self):
        from fusion_model_hub.recommend.scorer import score_hardware_fit

        assert score_hardware_fit(4.0, 8.0, can_run=False) == 0.0

    def test_score_hardware_fit_zero_vram(self):
        from fusion_model_hub.recommend.scorer import score_hardware_fit

        assert score_hardware_fit(4.0, 0.0, can_run=True) == 0.0

    def test_score_hardware_fit_ratio_buckets(self):
        from fusion_model_hub.recommend.scorer import score_hardware_fit

        assert score_hardware_fit(2.0, 8.0, can_run=True) == 100.0
        assert 80.0 <= score_hardware_fit(5.0, 8.0, can_run=True) <= 100.0
        assert 40.0 <= score_hardware_fit(7.0, 8.0, can_run=True) <= 80.0
        assert score_hardware_fit(8.0, 8.0, can_run=True) == 40.0
        assert 0.0 <= score_hardware_fit(10.0, 8.0, can_run=True) < 40.0

    def test_score_quality_known_and_unknown(self):
        from fusion_model_hub.recommend.scorer import score_quality

        assert score_quality("FP16") == 100.0
        assert score_quality("Q4_K_M") == 55.0
        assert score_quality("UNKNOWN") == 50.0

    def test_score_speed_zero_and_scaled(self):
        from fusion_model_hub.recommend.scorer import score_speed

        assert score_speed(0.0) == 0.0
        assert score_speed(25.0) == 50.0
        assert score_speed(100.0) == 100.0

    def test_score_popularity_boundaries(self):
        from fusion_model_hub.recommend.scorer import score_popularity

        assert score_popularity(0) == 20.0
        assert score_popularity(100000) == 100.0
        assert 20.0 < score_popularity(50000) < 100.0

    def test_weighted_total_profiles(self):
        from fusion_model_hub.recommend.scorer import weighted_total

        assert weighted_total(100, 100, 100, 100, "quality") == 100.0
        assert weighted_total(100, 100, 100, 100, "speed") == 100.0
        assert weighted_total(100, 100, 100, 100, "balanced") == 100.0
        assert weighted_total(100, 0, 0, 0, "unknown_profile") == weighted_total(100, 0, 0, 0, "balanced")

    def test_build_reason_cannot_run(self):
        from fusion_model_hub.recommend.scorer import build_reason

        assert "Insufficient VRAM" in build_reason(False, 0, 0, 0, "balanced")

    def test_build_reason_fit_phrases(self):
        from fusion_model_hub.recommend.scorer import build_reason

        assert "excellent hardware fit" in build_reason(True, 90, 50, 50, "balanced")
        assert "good hardware fit" in build_reason(True, 60, 50, 50, "balanced")
        assert "tight VRAM budget" in build_reason(True, 30, 50, 50, "balanced")

    def test_build_reason_quality_and_speed_phrases(self):
        from fusion_model_hub.recommend.scorer import build_reason

        assert "high quantization quality" in build_reason(True, 60, 90, 50, "balanced")
        assert "fast inference" in build_reason(True, 60, 50, 80, "speed")
        assert "slow inference expected" in build_reason(True, 60, 50, 10, "balanced")

    def test_build_reason_all_clauses_combined(self):
        from fusion_model_hub.recommend.scorer import build_reason

        out = build_reason(True, 90, 90, 80, "speed")
        assert "excellent hardware fit" in out
        assert "high quantization quality" in out
        assert "fast inference" in out


class TestCLIRecommend:
    def _patch_engine(self, monkeypatch):
        recs = [
            {"name": "m1", "can_run": True, "rank_score": 88.0, "reason": "excellent hardware fit"},
        ]
        rec_resp = MagicMock()
        rec_resp.total_evaluated = 3
        rec_resp.model_dump = MagicMock(
            return_value={"recommendations": recs, "total_evaluated": 3},
        )
        engine_inst = MagicMock()
        engine_inst.recommend = AsyncMock(return_value=rec_resp)

        def factory(_url):
            return engine_inst

        monkeypatch.setattr("fusion_model_hub.recommend.engine.RecommendEngine", factory)
        return rec_resp

    def test_recommend_models_command(self, monkeypatch):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.recommend import recommend_app

        self._patch_engine(monkeypatch)
        monkeypatch.setattr(
            "fusion_model_hub.cli.recommend._fetch_models_from_api",
            AsyncMock(return_value=[{"name": "m1", "params_b": 7}]),
        )

        result = CliRunner().invoke(recommend_app, ["models", "-n", "5"])
        assert result.exit_code == 0
        assert "recommendations" in result.stdout

    def test_recommend_quick_command(self, monkeypatch):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.recommend import recommend_app

        self._patch_engine(monkeypatch)
        monkeypatch.setattr(
            "fusion_model_hub.cli.recommend._fetch_models_from_api",
            AsyncMock(return_value=[{"name": "m1"}]),
        )

        result = CliRunner().invoke(recommend_app, ["quick"])
        assert result.exit_code == 0
        assert "m1" in result.stdout

    @pytest.mark.asyncio
    async def test_fetch_models_from_api_list(self, monkeypatch):
        from fusion_model_hub.cli.recommend import _fetch_models_from_api

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = [{"name": "m1"}, {"name": "m2"}]
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        import httpx as _real_httpx

        monkeypatch.setattr(_real_httpx, "AsyncClient", lambda **kw: mock_client)
        settings = MagicMock()
        settings.host = "localhost"
        settings.port = 11444
        models = await _fetch_models_from_api(settings)
        assert len(models) == 2

    @pytest.mark.asyncio
    async def test_fetch_models_from_api_envelope(self, monkeypatch):
        from fusion_model_hub.cli.recommend import _fetch_models_from_api

        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"items": [{"name": "m1"}]}
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        import httpx as _real_httpx

        monkeypatch.setattr(_real_httpx, "AsyncClient", lambda **kw: mock_client)
        settings = MagicMock()
        settings.host = "localhost"
        settings.port = 11444
        models = await _fetch_models_from_api(settings)
        assert models == [{"name": "m1"}]

    @pytest.mark.asyncio
    async def test_fetch_models_from_api_error_returns_empty(self, monkeypatch):
        from fusion_model_hub.cli.recommend import _fetch_models_from_api

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("conn refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        import httpx as _real_httpx

        monkeypatch.setattr(_real_httpx, "AsyncClient", lambda **kw: mock_client)
        settings = MagicMock()
        settings.host = "localhost"
        settings.port = 11444
        models = await _fetch_models_from_api(settings)
        assert models == []


class TestCLIMain:
    def test_version_command(self):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.main import app

        result = CliRunner().invoke(app, ["version"])
        assert result.exit_code == 0
        assert "fusion-model-hub" in result.stdout

    def test_help_lists_subcommands(self):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.main import app

        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "recommend" in result.stdout
        assert "list" in result.stdout
        assert "analyze" in result.stdout

    @pytest.mark.asyncio
    async def test_show_hardware_with_gpu(self, monkeypatch):
        from fusion_model_hub.cli.main import _show_hardware

        gpu = MagicMock()
        gpu.name = "Apple M2 Max"
        gpu.chip_generation.value = "m2"
        gpu.vram_gb = 24.0
        gpu.memory_bandwidth_gbps = 400.0
        cpu = MagicMock()
        cpu.name = "Apple M2 Max"
        cpu.cores = 12
        profile = MagicMock()
        profile.gpu = gpu
        profile.cpu = cpu
        profile.ram_gb = 64.0
        profile.disk_free_gb = 500.0
        profile.effective_vram_gb = 24.0

        detector_inst = MagicMock()
        detector_inst.detect = AsyncMock(return_value=profile)
        monkeypatch.setattr("fusion_model_hub.hardware.detector.HardwareDetector", lambda _url: detector_inst)

        out = await _show_hardware()
        assert "M2 Max" in out
        assert "VRAM" in out
        assert "Effective VRAM" in out

    @pytest.mark.asyncio
    async def test_show_hardware_no_gpu(self, monkeypatch):
        from fusion_model_hub.cli.main import _show_hardware

        cpu = MagicMock()
        cpu.name = "Apple M1"
        cpu.cores = 8
        profile = MagicMock()
        profile.gpu = None
        profile.cpu = cpu
        profile.ram_gb = 16.0
        profile.disk_free_gb = 200.0
        profile.effective_vram_gb = 8.0

        detector_inst = MagicMock()
        detector_inst.detect = AsyncMock(return_value=profile)
        monkeypatch.setattr("fusion_model_hub.hardware.detector.HardwareDetector", lambda _url: detector_inst)

        out = await _show_hardware()
        assert "Not detected" in out

    def test_hardware_command_e2e(self, monkeypatch):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.main import app

        cpu = MagicMock()
        cpu.name = "Apple M1"
        cpu.cores = 8
        profile = MagicMock()
        profile.gpu = None
        profile.cpu = cpu
        profile.ram_gb = 16.0
        profile.disk_free_gb = 200.0
        profile.effective_vram_gb = 8.0

        detector_inst = MagicMock()
        detector_inst.detect = AsyncMock(return_value=profile)
        monkeypatch.setattr("fusion_model_hub.hardware.detector.HardwareDetector", lambda _url: detector_inst)

        result = CliRunner().invoke(app, ["hardware"])
        assert result.exit_code == 0
        assert "Hardware Profile" in result.stdout


class TestMinioStoreRelease:
    def _store_with_client(self):
        from fusion_model_hub.storage.minio_store import MinioStore

        store = MinioStore("localhost:9000", "ak", "sk")
        mock_client = MagicMock()
        store._client = mock_client
        return store, mock_client

    def test_write_chunk(self):
        store, mock_client = self._store_with_client()

        path = asyncio.run(store.write_chunk("up1", 0, b"chunk-bytes"))
        assert str(path) == "uploads/up1/000000.part"
        mock_client.put_object.assert_called_once()
        args = mock_client.put_object.call_args.args
        assert args[1] == "uploads/up1/000000.part"

    def test_assemble_chunks_hash_and_cleanup(self):
        import hashlib

        store, mock_client = self._store_with_client()

        chunks = [b"aaaa", b"bbbb", b"cccc"]
        expected_hash = hashlib.sha256(b"".join(chunks)).hexdigest()
        expected_size = len(b"".join(chunks))

        def fake_get(bucket, name):
            idx = int(name.split("/")[-1].split(".")[0])
            resp = MagicMock()
            resp.read.return_value = chunks[idx]
            return resp

        mock_client.get_object.side_effect = fake_get

        path, file_hash, size = asyncio.run(store.assemble_chunks("up2", Path("m1/v1"), "model.bin", total_chunks=3))
        assert size == expected_size
        assert file_hash == expected_hash
        assert str(path) == "m1/v1/model.bin"
        # final object written once; each chunk removed once
        assert mock_client.put_object.call_count == 1
        assert mock_client.remove_object.call_count == 3

    def test_assemble_chunks_best_effort_cleanup_failure(self):
        store, mock_client = self._store_with_client()

        chunks = [b"one"]

        def fake_get(bucket, name):
            resp = MagicMock()
            resp.read.return_value = chunks[0]
            return resp

        mock_client.get_object.side_effect = fake_get
        mock_client.remove_object.side_effect = Exception("already gone")

        path, file_hash, size = asyncio.run(store.assemble_chunks("up3", Path("m1/v1"), "model.bin", total_chunks=1))
        assert size == 3
        assert len(file_hash) == 64

    def test_delete_model_files_deletes_some(self):
        store, mock_client = self._store_with_client()

        obj1 = MagicMock()
        obj1.object_name = "m1/v1/a.bin"
        obj2 = MagicMock()
        obj2.object_name = "m1/v2/b.bin"
        mock_client.list_objects.return_value = [obj1, obj2]

        assert store.delete_model_files("m1") is True
        assert mock_client.remove_object.call_count == 2

    def test_models_dir_not_implemented(self):
        store, _ = self._store_with_client()

        with pytest.raises(NotImplementedError):
            _ = store.models_dir
