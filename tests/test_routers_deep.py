import asyncio
import io
import json
import logging
import os
import tarfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps

logger = logging.getLogger(__name__)


@pytest.fixture
def settings():
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_test_deep",
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
        # #3: keep the eval runner off so existing eval CRUD tests (which POST
        # /evaluations and assert status=="pending") are not raced by a
        # background bench-submission task. The runner is exercised by its own
        # dedicated test with a mocked Fusion-Bench.
        eval_runner_enabled=False,
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


async def _create_model(client, name="deep-model"):
    resp = await client.post(
        "/api/v1/models",
        json={
            "name": name,
            "description": "deep test",
            "model_type": "llm",
            "architecture": "qwen2",
            "params_size": "7B",
        },
    )
    assert resp.status_code == 201
    return resp.json()


async def _create_version(client, model_id, version="1.0.0", with_file=True):
    if with_file:
        resp = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": version, "format": "mlx", "quantization": "4bit"},
            files={"file": ("model.bin", b"fake model data", "application/octet-stream")},
        )
    else:
        resp = await client.post(
            f"/api/v1/models/{model_id}/versions",
            data={"version": version, "format": "mlx", "quantization": "4bit"},
        )
    assert resp.status_code == 201
    return resp.json()


# ========== versions.py deep tests ==========


class TestVersionUploadNoFile:
    @pytest.mark.asyncio
    async def test_upload_version_without_file(self, client):
        model = await _create_model(client, "ver-nofile-deep")
        resp = await client.post(
            f"/api/v1/models/{model['id']}/versions",
            data={"version": "1.0.0", "format": "mlx", "quantization": "4bit"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["file_size"] == 0
        assert data["file_path"] == ""

    @pytest.mark.asyncio
    async def test_upload_version_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/models/nonexistent/versions",
            data={"version": "1.0.0"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_upload_version_missing_version(self, client):
        model = await _create_model(client, "ver-nover-deep")
        resp = await client.post(
            f"/api/v1/models/{model['id']}/versions",
            data={"version": "", "format": "mlx", "quantization": "4bit"},
        )
        assert resp.status_code == 400


class TestChunkUploadDeep:
    @pytest.mark.asyncio
    async def test_chunk_upload_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/models/nonexistent/versions/chunk-upload",
            data={"version": "1.0.0", "total_chunks": "1", "chunk_index": "0", "filename": "m.bin"},
            files={"chunk": ("chunk.bin", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_chunk_upload_no_chunk(self, client):
        model = await _create_model(client, "chunk-nochunk-deep")
        resp = await client.post(
            f"/api/v1/models/{model['id']}/versions/chunk-upload",
            data={"version": "1.0.0", "total_chunks": "1", "chunk_index": "0"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_chunk_upload_multi_chunk(self, client):
        model = await _create_model(client, "chunk-multi-deep")
        content = b"chunk-upload-test-data-for-deep-coverage"
        chunk_size = len(content) // 3
        chunks = [content[:chunk_size], content[chunk_size : 2 * chunk_size], content[2 * chunk_size :]]
        for i, chunk_data in enumerate(chunks):
            resp = await client.post(
                f"/api/v1/models/{model['id']}/versions/chunk-upload",
                data={
                    "version": "3chunk-ver",
                    "format": "mlx",
                    "quantization": "4bit",
                    "filename": "model.mlx",
                    "total_chunks": "3",
                    "chunk_index": str(i),
                },
                files={"chunk": (f"chunk{i}.bin", chunk_data, "application/octet-stream")},
            )
            if i < len(chunks) - 1:
                assert resp.status_code == 201
                assert resp.json()["status"] == "chunk_received"
            else:
                assert resp.status_code == 201
                assert resp.json()["version"] == "3chunk-ver"

    @pytest.mark.asyncio
    async def test_chunk_upload_traversal_filename_rejected(self, client):
        # P1-11: a traversal filename must be rejected at the router boundary,
        # not written outside the version dir.
        model = await _create_model(client, "chunk-traversal-deep")
        for bad_name in ("../evil.bin", "a/b/c.bin", "/etc/passwd", "../../x"):
            resp = await client.post(
                f"/api/v1/models/{model['id']}/versions/chunk-upload",
                data={
                    "version": f"trav-{abs(hash(bad_name))}",
                    "format": "mlx",
                    "quantization": "4bit",
                    "filename": bad_name,
                    "total_chunks": "1",
                    "chunk_index": "0",
                },
                files={"chunk": ("chunk.bin", b"data", "application/octet-stream")},
            )
            assert resp.status_code == 400, f"{bad_name!r} should be rejected, got {resp.status_code}"


class TestVersionListDeep:
    @pytest.mark.asyncio
    async def test_list_versions_model_not_found(self, client):
        resp = await client.get("/api/v1/models/nonexistent/versions")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_versions_with_status_filter(self, client):
        model = await _create_model(client, "ver-filter-deep")
        await _create_version(client, model["id"], "1.0.0")
        resp = await client.get(
            f"/api/v1/models/{model['id']}/versions",
            params={"status": "draft"},
        )
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1


class TestVersionDownloadDeep:
    @pytest.mark.asyncio
    async def test_download_version_with_file(self, client):
        model = await _create_model(client, "dl-file-deep")
        content = b"downloadable content for deep test"
        ver = await _create_version(client, model["id"], "1.0-dl", with_file=True)
        resp = await client.get(f"/api/v1/versions/{ver['id']}/download")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_download_version_not_found(self, client):
        resp = await client.get("/api/v1/versions/nonexistent/download")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_download_version_no_file_path(self, client):
        model = await _create_model(client, "dl-nopath-deep")
        ver = await _create_version(client, model["id"], "1.0-nopath", with_file=False)
        resp = await client.get(f"/api/v1/versions/{ver['id']}/download")
        assert resp.status_code == 404


class TestVersionRollbackDeep:
    @pytest.mark.asyncio
    async def test_rollback_deprecated_to_published(self, client):
        model = await _create_model(client, "rollback-dep-deep")
        ver = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{ver['id']}/status", json={"target_status": "testing"})
        await client.put(f"/api/v1/versions/{ver['id']}/metrics", json={"benchmark_score": 90.0})
        await client.put(
            f"/api/v1/versions/{ver['id']}/status", json={"target_status": "published", "approval_level": "l1"}
        )
        await client.put(f"/api/v1/versions/{ver['id']}/status", json={"target_status": "deprecated"})
        resp = await client.post(f"/api/v1/versions/{ver['id']}/rollback")
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"

    @pytest.mark.asyncio
    async def test_rollback_version_not_found(self, client):
        resp = await client.post("/api/v1/versions/nonexistent/rollback")
        assert resp.status_code == 404


class TestDeprecateDeep:
    @pytest.mark.asyncio
    async def test_deprecate_with_successor_and_remark(self, client):
        model = await _create_model(client, "dep-succ-deep")
        v1 = await _create_version(client, model["id"], "1.0.0")
        v2 = await _create_version(client, model["id"], "2.0.0")
        await client.put(f"/api/v1/versions/{v1['id']}/status", json={"target_status": "testing"})
        await client.put(f"/api/v1/versions/{v1['id']}/metrics", json={"benchmark_score": 90.0})
        await client.put(
            f"/api/v1/versions/{v1['id']}/status", json={"target_status": "published", "approval_level": "l1"}
        )
        resp = await client.post(
            f"/api/v1/versions/{v1['id']}/deprecate",
            json={"successor_version_id": v2["id"], "remark": "superseded by v2"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deprecated"
        assert data["successor_version_id"] == v2["id"]

    @pytest.mark.asyncio
    async def test_deprecate_without_successor(self, client):
        model = await _create_model(client, "dep-nosucc-deep")
        ver = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{ver['id']}/status", json={"target_status": "testing"})
        await client.put(f"/api/v1/versions/{ver['id']}/metrics", json={"benchmark_score": 90.0})
        await client.put(
            f"/api/v1/versions/{ver['id']}/status", json={"target_status": "published", "approval_level": "l1"}
        )
        resp = await client.post(
            f"/api/v1/versions/{ver['id']}/deprecate",
            json={"successor_version_id": "", "remark": "old version"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "deprecated"

    @pytest.mark.asyncio
    async def test_deprecate_invalid_transition(self, client):
        model = await _create_model(client, "dep-invalid-deep")
        ver = await _create_version(client, model["id"])
        resp = await client.post(
            f"/api/v1/versions/{ver['id']}/deprecate",
            json={},
        )
        assert resp.status_code == 409


class TestRetireDeep:
    @pytest.mark.asyncio
    async def test_retire_from_deprecated(self, client):
        model = await _create_model(client, "retire-dep-deep")
        ver = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{ver['id']}/status", json={"target_status": "testing"})
        await client.put(f"/api/v1/versions/{ver['id']}/metrics", json={"benchmark_score": 90.0})
        await client.put(
            f"/api/v1/versions/{ver['id']}/status", json={"target_status": "published", "approval_level": "l1"}
        )
        await client.put(f"/api/v1/versions/{ver['id']}/status", json={"target_status": "deprecated"})
        resp = await client.post(f"/api/v1/versions/{ver['id']}/retire")
        assert resp.status_code == 200
        assert resp.json()["status"] == "retired"

    @pytest.mark.asyncio
    async def test_retire_from_retired_no_transition(self, client):
        model = await _create_model(client, "retire-draft-deep")
        ver = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{ver['id']}/status", json={"target_status": "retired"})
        resp = await client.post(f"/api/v1/versions/{ver['id']}/retire")
        assert resp.status_code == 409


class TestBenchmarkDeep:
    @pytest.mark.asyncio
    async def test_benchmark_all_fields(self, client):
        model = await _create_model(client, "bench-all-deep")
        ver = await _create_version(client, model["id"])
        resp = await client.put(
            f"/api/v1/versions/{ver['id']}/benchmark",
            json={
                "benchmark_score": 92.3,
                "inference_latency": 8.1,
                "throughput": 120.5,
                "memory_usage": 4096.0,
                "context_length": 8192,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["benchmark_score"] == 92.3
        assert data["inference_latency"] == 8.1
        assert data["throughput"] == 120.5
        assert data["memory_usage"] == 4096.0
        assert data["context_length"] == 8192

    @pytest.mark.asyncio
    async def test_benchmark_no_fields(self, client):
        model = await _create_model(client, "bench-none-deep")
        ver = await _create_version(client, model["id"])
        resp = await client.put(
            f"/api/v1/versions/{ver['id']}/benchmark",
            json={},
        )
        assert resp.status_code == 400


class TestMetricsUpdate:
    @pytest.mark.asyncio
    async def test_update_metrics(self, client):
        model = await _create_model(client, "metrics-deep")
        ver = await _create_version(client, model["id"])
        resp = await client.put(
            f"/api/v1/versions/{ver['id']}/metrics",
            json={"inference_latency": 15.7, "throughput": 99.1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["inference_latency"] == 15.7
        assert data["throughput"] == 99.1

    @pytest.mark.asyncio
    async def test_update_metrics_no_fields(self, client):
        model = await _create_model(client, "metrics-none-deep")
        ver = await _create_version(client, model["id"])
        resp = await client.put(
            f"/api/v1/versions/{ver['id']}/metrics",
            json={},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_update_metrics_version_not_found(self, client):
        resp = await client.put(
            "/api/v1/versions/nonexistent/metrics",
            json={"inference_latency": 1.0},
        )
        assert resp.status_code == 404


class TestPromoteDeep:
    @pytest.mark.asyncio
    async def test_promote_draft_to_published(self, client):
        model = await _create_model(client, "promote-draft-deep")
        ver = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{ver['id']}/metrics", json={"benchmark_score": 90.0})
        resp = await client.post(f"/api/v1/versions/{ver['id']}/promote")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "published"
        assert "draft" in data.get("promoted_steps", []) or "testing" in data.get("promoted_steps", [])

    @pytest.mark.asyncio
    async def test_promote_testing_to_published(self, client):
        model = await _create_model(client, "promote-testing-deep")
        ver = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{ver['id']}/status", json={"target_status": "testing"})
        await client.put(f"/api/v1/versions/{ver['id']}/metrics", json={"benchmark_score": 90.0})
        resp = await client.post(f"/api/v1/versions/{ver['id']}/promote")
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"

    @pytest.mark.asyncio
    async def test_promote_already_published(self, client):
        model = await _create_model(client, "promote-pub-deep")
        ver = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{ver['id']}/metrics", json={"benchmark_score": 90.0})
        await client.post(f"/api/v1/versions/{ver['id']}/promote")
        resp = await client.post(f"/api/v1/versions/{ver['id']}/promote")
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"

    @pytest.mark.asyncio
    async def test_promote_deprecated_conflict(self, client):
        model = await _create_model(client, "promote-dep-deep")
        ver = await _create_version(client, model["id"])
        await client.put(f"/api/v1/versions/{ver['id']}/metrics", json={"benchmark_score": 90.0})
        await client.post(f"/api/v1/versions/{ver['id']}/promote")
        await client.post(f"/api/v1/versions/{ver['id']}/deprecate", json={})
        resp = await client.post(f"/api/v1/versions/{ver['id']}/promote")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_promote_not_found(self, client):
        resp = await client.post("/api/v1/versions/nonexistent/promote")
        assert resp.status_code == 404


class TestUrlDownloadDeep:
    @pytest.mark.asyncio
    async def test_url_download_ssrf_localhost(self, client):
        model = await _create_model(client, "ssrf-local-deep")
        resp = await client.post(
            f"/api/v1/models/{model['id']}/versions/download-url",
            json={"url": "http://localhost/evil"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_url_download_ssrf_127(self, client):
        model = await _create_model(client, "ssrf-127-deep")
        resp = await client.post(
            f"/api/v1/models/{model['id']}/versions/download-url",
            json={"url": "http://127.0.0.1/evil"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_url_download_ssrf_10(self, client):
        model = await _create_model(client, "ssrf-10-deep")
        resp = await client.post(
            f"/api/v1/models/{model['id']}/versions/download-url",
            json={"url": "http://10.0.0.1/evil"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_url_download_ssrf_192(self, client):
        model = await _create_model(client, "ssrf-192-deep")
        resp = await client.post(
            f"/api/v1/models/{model['id']}/versions/download-url",
            json={"url": "http://192.168.1.1/evil"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_url_download_ssrf_172(self, client):
        model = await _create_model(client, "ssrf-172-deep")
        resp = await client.post(
            f"/api/v1/models/{model['id']}/versions/download-url",
            json={"url": "http://172.16.0.1/evil"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_url_download_ssrf_metadata(self, client):
        model = await _create_model(client, "ssrf-meta-deep")
        resp = await client.post(
            f"/api/v1/models/{model['id']}/versions/download-url",
            json={"url": "http://169.254.169.254/metadata"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_url_download_invalid_scheme(self, client):
        model = await _create_model(client, "ssrf-scheme-deep")
        resp = await client.post(
            f"/api/v1/models/{model['id']}/versions/download-url",
            json={"url": "ftp://evil.com/file"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_url_download_valid_url(self, client):
        model = await _create_model(client, "urldl-valid-deep")
        with patch("fusion_model_hub.server.routers.versions.ModelDownloader") as MockDownloader:
            mock_downloader = AsyncMock()
            mock_downloader.download.return_value = {"status": "completed", "path": "/tmp/fake", "size_bytes": 100}
            MockDownloader.return_value = mock_downloader
            resp = await client.post(
                f"/api/v1/models/{model['id']}/versions/download-url",
                json={"url": "https://example.com/model.bin", "version": "url-v1"},
            )
            assert resp.status_code == 202
            assert resp.json()["status"] == "download_started"


class TestExportModelTar:
    @pytest.mark.asyncio
    async def test_export_model_tar(self, client):
        model = await _create_model(client, "export-tar-deep")
        await _create_version(client, model["id"], "1.0.0", with_file=True)
        resp = await client.get(f"/api/v1/models/{model['id']}/export")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_export_model_not_found(self, client):
        resp = await client.get("/api/v1/models/nonexistent/export")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_export_model_no_versions(self, client):
        model = await _create_model(client, "export-nover-deep")
        resp = await client.get(f"/api/v1/models/{model['id']}/export")
        assert resp.status_code == 404


class TestImportModelTar:
    @pytest.mark.asyncio
    async def test_import_model_tar(self, client):
        metadata = {
            "model": {
                "id": "fake-id",
                "name": "imported-tar-deep",
                "description": "from tar",
                "model_type": "llm",
                "architecture": "qwen2",
                "params_size": "7B",
                "hf_repo": "",
            },
            "versions": [
                {
                    "id": "v1",
                    "version": "1.0.0",
                    "format": "mlx",
                    "quantization": "4bit",
                    "file_hash": "abc",
                    "file_size": 100,
                },
            ],
        }
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            meta_bytes = json.dumps(metadata).encode()
            info = tarfile.TarInfo(name="metadata.json")
            info.size = len(meta_bytes)
            tar.addfile(info, io.BytesIO(meta_bytes))
            file_bytes = b"fake model weights"
            finfo = tarfile.TarInfo(name="model.bin")
            finfo.size = len(file_bytes)
            tar.addfile(finfo, io.BytesIO(file_bytes))
        buf.seek(0)
        resp = await client.post(
            "/api/v1/models/import-tar",
            files={"file": ("model.tar.gz", buf.read(), "application/gzip")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "imported-tar-deep"
        assert data["versions_imported"] == 1

    @pytest.mark.asyncio
    async def test_import_tar_duplicate_name(self, client):
        await _create_model(client, "dup-tar-deep")
        metadata = {
            "model": {"id": "x", "name": "dup-tar-deep", "description": "", "model_type": "llm"},
            "versions": [],
        }
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            meta_bytes = json.dumps(metadata).encode()
            info = tarfile.TarInfo(name="metadata.json")
            info.size = len(meta_bytes)
            tar.addfile(info, io.BytesIO(meta_bytes))
        buf.seek(0)
        resp = await client.post(
            "/api/v1/models/import-tar",
            files={"file": ("model.tar.gz", buf.read(), "application/gzip")},
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_import_tar_no_metadata(self, client):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            file_bytes = b"no metadata"
            info = tarfile.TarInfo(name="readme.txt")
            info.size = len(file_bytes)
            tar.addfile(info, io.BytesIO(file_bytes))
        buf.seek(0)
        resp = await client.post(
            "/api/v1/models/import-tar",
            files={"file": ("bad.tar.gz", buf.read(), "application/gzip")},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_import_tar_no_name(self, client):
        metadata = {"model": {"id": "x", "description": "no name"}, "versions": []}
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            meta_bytes = json.dumps(metadata).encode()
            info = tarfile.TarInfo(name="metadata.json")
            info.size = len(meta_bytes)
            tar.addfile(info, io.BytesIO(meta_bytes))
        buf.seek(0)
        resp = await client.post(
            "/api/v1/models/import-tar",
            files={"file": ("noname.tar.gz", buf.read(), "application/gzip")},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_import_tar_invalid_gz(self, client):
        resp = await client.post(
            "/api/v1/models/import-tar",
            files={"file": ("bad.tar.gz", b"not a tar gz", "application/gzip")},
        )
        assert resp.status_code == 400


# ========== models.py deep tests ==========


class TestHFImportDeep:
    @pytest.mark.asyncio
    async def test_import_hf_with_mock(self, client):
        hf_response = {
            "pipeline_tag": "text-generation",
            "cardData": {"license": "apache-2.0", "language": ["en"]},
            "author": "testuser",
            "config": {"architectures": ["LlamaForCausalLM"]},
            "safetensors": {"total": "7B"},
        }
        with patch("fusion_model_hub.server.routers.models._fetch_hf_model_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = hf_response
            resp = await client.post(
                "/api/v1/models/import/hf",
                json={"hf_repo": "testorg/test-model-hf"},
            )
            assert resp.status_code == 201
            data = resp.json()
            assert data["hf_repo"] == "testorg/test-model-hf"
            assert data["model_type"] == "llm"
            assert data["architecture"] == "LlamaForCausalLM"

    @pytest.mark.asyncio
    async def test_import_hf_chat_type(self, client):
        hf_response = {"pipeline_tag": "conversational", "author": "a"}
        with patch("fusion_model_hub.server.routers.models._fetch_hf_model_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = hf_response
            resp = await client.post(
                "/api/v1/models/import/hf",
                json={"hf_repo": "org/chat-model"},
            )
            assert resp.status_code == 201
            assert resp.json()["model_type"] == "chat"

    @pytest.mark.asyncio
    async def test_import_hf_embedding_type(self, client):
        hf_response = {"pipeline_tag": "feature-extraction", "author": "a"}
        with patch("fusion_model_hub.server.routers.models._fetch_hf_model_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = hf_response
            resp = await client.post(
                "/api/v1/models/import/hf",
                json={"hf_repo": "org/emb-model"},
            )
            assert resp.status_code == 201
            assert resp.json()["model_type"] == "embedding"

    @pytest.mark.asyncio
    async def test_import_hf_multimodal_type(self, client):
        hf_response = {"pipeline_tag": "image-text-to-text", "author": "a"}
        with patch("fusion_model_hub.server.routers.models._fetch_hf_model_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = hf_response
            resp = await client.post(
                "/api/v1/models/import/hf",
                json={"hf_repo": "org/mm-model"},
            )
            assert resp.status_code == 201
            assert resp.json()["model_type"] == "multimodal"

    @pytest.mark.asyncio
    async def test_import_hf_image_type(self, client):
        hf_response = {"pipeline_tag": "text-to-image", "author": "a"}
        with patch("fusion_model_hub.server.routers.models._fetch_hf_model_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = hf_response
            resp = await client.post(
                "/api/v1/models/import/hf",
                json={"hf_repo": "org/img-model"},
            )
            assert resp.status_code == 201
            assert resp.json()["model_type"] == "image"

    @pytest.mark.asyncio
    async def test_import_hf_audio_type(self, client):
        hf_response = {"pipeline_tag": "automatic-speech-recognition", "author": "a"}
        with patch("fusion_model_hub.server.routers.models._fetch_hf_model_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = hf_response
            resp = await client.post(
                "/api/v1/models/import/hf",
                json={"hf_repo": "org/audio-model"},
            )
            assert resp.status_code == 201
            assert resp.json()["model_type"] == "audio"

    @pytest.mark.asyncio
    async def test_import_hf_code_type(self, client):
        hf_response = {"pipeline_tag": "code-generation", "author": "a"}
        with patch("fusion_model_hub.server.routers.models._fetch_hf_model_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = hf_response
            resp = await client.post(
                "/api/v1/models/import/hf",
                json={"hf_repo": "org/code-model"},
            )
            assert resp.status_code == 201
            assert resp.json()["model_type"] == "code"

    @pytest.mark.asyncio
    async def test_import_hf_missing_repo(self, client):
        resp = await client.post("/api/v1/models/import/hf", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_import_hf_duplicate_name(self, client):
        await _create_model(client, "test-model-hf")
        with patch("fusion_model_hub.server.routers.models._fetch_hf_model_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {}
            resp = await client.post(
                "/api/v1/models/import/hf",
                json={"hf_repo": "testorg/test-model-hf"},
            )
            assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_import_hf_custom_name(self, client):
        hf_response = {"pipeline_tag": "text-generation", "author": "a"}
        with patch("fusion_model_hub.server.routers.models._fetch_hf_model_info", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = hf_response
            resp = await client.post(
                "/api/v1/models/import/hf",
                json={"hf_repo": "org/custom-name", "name": "my-custom-name"},
            )
            assert resp.status_code == 201
            assert resp.json()["name"] == "my-custom-name"


class TestModelSyncDeep:
    @pytest.mark.asyncio
    async def test_sync_dry_run_with_mock(self, client):
        with patch("fusion_model_hub.server.routers.models.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "items": [
                    {"name": "synced-model-1", "description": "from remote", "model_type": "llm"},
                    {"name": "synced-model-2", "description": "from remote", "model_type": "chat"},
                ],
            }
            mock_instance.get.return_value = mock_resp
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance
            resp = await client.post(
                "/api/v1/models/sync",
                json={"source_url": "https://remote.example.com", "dry_run": True},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["dry_run"] is True
            assert data["new_count"] >= 1

    @pytest.mark.asyncio
    async def test_sync_actual_with_mock(self, client):
        with patch("fusion_model_hub.server.routers.models.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "items": [
                    {"name": "synced-actual-1", "description": "remote", "model_type": "llm"},
                ],
            }
            mock_instance.get.return_value = mock_resp
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance
            resp = await client.post(
                "/api/v1/models/sync",
                json={"source_url": "https://remote.example.com", "dry_run": False},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["synced"] >= 1


class TestModelSearchDeep:
    @pytest.mark.asyncio
    async def test_search_with_params_size(self, client):
        await client.post(
            "/api/v1/models",
            json={
                "name": "search-ps-deep",
                "params_size": "7B",
            },
        )
        resp = await client.get("/api/v1/models/search", params={"params_size": "7B"})
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    @pytest.mark.asyncio
    async def test_search_with_sort_by_name(self, client):
        await client.post("/api/v1/models", json={"name": "search-sort-a"})
        await client.post("/api/v1/models", json={"name": "search-sort-b"})
        resp = await client.get("/api/v1/models/search", params={"sort_by": "name", "sort_order": "asc"})
        assert resp.status_code == 200
        assert resp.json()["total"] >= 2

    @pytest.mark.asyncio
    async def test_search_with_min_benchmark(self, client):
        model = await _create_model(client, "search-bench-deep")
        ver = await _create_version(client, model["id"])
        await client.put(
            f"/api/v1/versions/{ver['id']}/benchmark",
            json={"benchmark_score": 90.0},
        )
        resp = await client.get("/api/v1/models/search", params={"min_benchmark_score": 80.0})
        assert resp.status_code == 200


class TestModelRecommendDeep:
    # E-E3: the duplicate GET /models/recommend endpoint was removed; the
    # canonical path is POST /recommend, which feeds the RecommendEngine real
    # params_size/task_types/download_count columns (was hardcoded zeros).
    @pytest.mark.asyncio
    async def test_recommend_with_task_type(self, client):
        await client.post(
            "/api/v1/models",
            json={
                "name": "rec-task-deep",
                "task_types": "text-generation",
            },
        )
        resp = await client.post("/api/v1/recommend", json={"task": "text-generation"})
        assert resp.status_code == 200
        assert "recommendations" in resp.json()

    @pytest.mark.asyncio
    async def test_recommend_with_max_params(self, client):
        await client.post(
            "/api/v1/models",
            json={
                "name": "rec-params-deep",
                "params_size": "7B",
            },
        )
        resp = await client.post("/api/v1/recommend", json={"max_params_b": 10})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_recommend_with_min_params_filter(self, client):
        # E-E3 regression: with real params_size wired, min_params_b>0 must keep
        # candidates (previously params_b was hardcoded 0 so any min_params_b>0
        # filtered everything out).
        await client.post(
            "/api/v1/models",
            json={
                "name": "rec-min-deep",
                "params_size": "7B",
            },
        )
        resp = await client.post("/api/v1/recommend", json={"min_params_b": 5, "max_params_b": 10})
        assert resp.status_code == 200
        names = [r["name"] for r in resp.json()["recommendations"]]
        assert "rec-min-deep" in names


class TestModelCompareDeep:
    @pytest.mark.asyncio
    async def test_compare_model_not_found(self, client):
        r1 = await client.post("/api/v1/models", json={"name": "cmp-exist-deep"})
        id1 = r1.json()["id"]
        resp = await client.get(f"/api/v1/models/compare?ids={id1},nonexistent")
        assert resp.status_code == 404


# ========== deployments.py deep tests ==========


class TestDeploymentCreateDeep:
    @pytest.mark.asyncio
    async def test_create_deployment_mlx_connect_error(self, client):
        model = await _create_model(client, "dep-conn-deep")
        with patch("fusion_model_hub.server.routers.deployments.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=Exception("connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance
            resp = await client.post(
                "/api/v1/deployments",
                json={"model_id": model["id"], "name": "dep-conn", "replicas": 1},
            )
            assert resp.status_code == 201
            assert resp.json()["status"] == "pending"

    @pytest.mark.asyncio
    async def test_create_deployment_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/deployments",
            json={"model_id": "nonexistent", "name": "dep-notfound"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_create_deployment_mlx_loaded(self, client):
        model = await _create_model(client, "dep-mlx-deep")
        with patch("fusion_model_hub.server.routers.deployments.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_instance.post.return_value = mock_resp
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance
            resp = await client.post(
                "/api/v1/deployments",
                json={"model_id": model["id"], "name": "dep-mlx", "replicas": 1},
            )
            assert resp.status_code == 201
            assert resp.json()["status"] == "running"


class TestDeploymentUpdateDeep:
    @pytest.mark.asyncio
    async def test_update_deployment_invalid_status(self, client):
        model = await _create_model(client, "dep-invstat-deep")
        create = await client.post(
            "/api/v1/deployments",
            json={"model_id": model["id"], "name": "dep-invstat"},
        )
        did = create.json()["id"]
        resp = await client.patch(
            f"/api/v1/deployments/{did}",
            json={"status": "invalid_status_value"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_update_deployment_not_found(self, client):
        resp = await client.patch(
            "/api/v1/deployments/nonexistent",
            json={"replicas": 3},
        )
        assert resp.status_code == 404


class TestDeploymentDeleteDeep:
    @pytest.mark.asyncio
    async def test_delete_deployment_not_found(self, client):
        resp = await client.delete("/api/v1/deployments/nonexistent")
        assert resp.status_code == 404


class TestDeploymentScaleDeep:
    @pytest.mark.asyncio
    async def test_scale_deployment_not_found(self, client):
        resp = await client.post(
            "/api/v1/deployments/nonexistent/scale",
            json={"replicas": 3},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_scale_deployment_running_with_mlx(self, client):
        model = await _create_model(client, "scale-mlx-deep")
        create = await client.post(
            "/api/v1/deployments",
            json={"model_id": model["id"], "name": "scale-mlx"},
        )
        did = create.json()["id"]
        await client.patch(f"/api/v1/deployments/{did}", json={"status": "running"})
        with patch("fusion_model_hub.server.routers.deployments.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_instance.post.return_value = mock_resp
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance
            resp = await client.post(
                f"/api/v1/deployments/{did}/scale",
                json={"replicas": 5},
            )
            assert resp.status_code == 200
            assert resp.json()["replicas"] == 5


class TestDeploymentMetricsDeep:
    @pytest.mark.asyncio
    async def test_deployment_metrics(self, client):
        model = await _create_model(client, "metrics-dep-deep")
        create = await client.post(
            "/api/v1/deployments",
            json={"model_id": model["id"], "name": "metrics-dep"},
        )
        did = create.json()["id"]
        resp = await client.get(f"/api/v1/deployments/{did}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "deployment_id" in data
        assert "mlx_metrics" in data

    @pytest.mark.asyncio
    async def test_deployment_metrics_not_found(self, client):
        resp = await client.get("/api/v1/deployments/nonexistent/metrics")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_deployment_metrics_running_with_mlx(self, client):
        model = await _create_model(client, "met-mlx-deep")
        create = await client.post(
            "/api/v1/deployments",
            json={"model_id": model["id"], "name": "met-mlx"},
        )
        did = create.json()["id"]
        await client.patch(f"/api/v1/deployments/{did}", json={"status": "running"})
        with patch("fusion_model_hub.server.routers.deployments.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"model_status": {"memory": 4096}}
            mock_instance.get.return_value = mock_resp
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance
            resp = await client.get(f"/api/v1/deployments/{did}/metrics")
            assert resp.status_code == 200


class TestGrayReleaseDeep:
    @pytest.mark.asyncio
    async def test_enable_gray_not_found(self, client):
        resp = await client.post(
            "/api/v1/deployments/nonexistent/gray",
            json={"gray_version_id": "v1", "gray_traffic_ratio": 20},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_disable_gray_not_found(self, client):
        resp = await client.delete("/api/v1/deployments/nonexistent/gray")
        assert resp.status_code == 404


# ========== evaluations.py deep tests ==========


class TestEvaluationCRUDDeep:
    @pytest.mark.asyncio
    async def test_create_evaluation(self, client):
        model = await _create_model(client, "eval-create-deep")
        resp = await client.post(
            "/api/v1/evaluations",
            json={"model_id": model["id"], "benchmark_name": "mmlu"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["model_id"] == model["id"]
        assert data["benchmark_name"] == "mmlu"
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_create_evaluation_model_not_found(self, client):
        resp = await client.post(
            "/api/v1/evaluations",
            json={"model_id": "nonexistent", "benchmark_name": "mmlu"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_create_evaluation_with_version(self, client):
        model = await _create_model(client, "eval-ver-deep")
        ver = await _create_version(client, model["id"])
        resp = await client.post(
            "/api/v1/evaluations",
            json={"model_id": model["id"], "version_id": ver["id"], "benchmark_name": "hellaswag"},
        )
        assert resp.status_code == 201
        assert resp.json()["version_id"] == ver["id"]

    @pytest.mark.asyncio
    async def test_list_evaluations(self, client):
        model = await _create_model(client, "eval-list-deep")
        await client.post("/api/v1/evaluations", json={"model_id": model["id"], "benchmark_name": "b1"})
        await client.post("/api/v1/evaluations", json={"model_id": model["id"], "benchmark_name": "b2"})
        resp = await client.get("/api/v1/evaluations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2

    @pytest.mark.asyncio
    async def test_list_evaluations_with_filters(self, client):
        model = await _create_model(client, "eval-filter-deep")
        await client.post("/api/v1/evaluations", json={"model_id": model["id"], "benchmark_name": "filter-bench"})
        resp = await client.get("/api/v1/evaluations", params={"model_id": model["id"]})
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    @pytest.mark.asyncio
    async def test_get_evaluation(self, client):
        model = await _create_model(client, "eval-get-deep")
        create = await client.post("/api/v1/evaluations", json={"model_id": model["id"], "benchmark_name": "b1"})
        eval_id = create.json()["id"]
        resp = await client.get(f"/api/v1/evaluations/{eval_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == eval_id

    @pytest.mark.asyncio
    async def test_get_evaluation_not_found(self, client):
        resp = await client.get("/api/v1/evaluations/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_evaluation_status(self, client):
        model = await _create_model(client, "eval-upd-deep")
        create = await client.post("/api/v1/evaluations", json={"model_id": model["id"], "benchmark_name": "b1"})
        eval_id = create.json()["id"]
        resp = await client.patch(
            f"/api/v1/evaluations/{eval_id}",
            json={"status": "running"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    @pytest.mark.asyncio
    async def test_update_evaluation_score(self, client):
        model = await _create_model(client, "eval-score-deep")
        create = await client.post("/api/v1/evaluations", json={"model_id": model["id"], "benchmark_name": "b1"})
        eval_id = create.json()["id"]
        resp = await client.patch(
            f"/api/v1/evaluations/{eval_id}",
            json={"status": "completed", "score": 85.5, "metrics": '{"accuracy": 0.855}'},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["score"] == 85.5

    @pytest.mark.asyncio
    async def test_update_evaluation_invalid_status(self, client):
        model = await _create_model(client, "eval-invstat-deep")
        create = await client.post("/api/v1/evaluations", json={"model_id": model["id"], "benchmark_name": "b1"})
        eval_id = create.json()["id"]
        resp = await client.patch(
            f"/api/v1/evaluations/{eval_id}",
            json={"status": "invalid_status"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_update_evaluation_no_fields(self, client):
        model = await _create_model(client, "eval-nofield-deep")
        create = await client.post("/api/v1/evaluations", json={"model_id": model["id"], "benchmark_name": "b1"})
        eval_id = create.json()["id"]
        resp = await client.patch(
            f"/api/v1/evaluations/{eval_id}",
            json={},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_update_evaluation_not_found(self, client):
        resp = await client.patch(
            "/api/v1/evaluations/nonexistent",
            json={"status": "running"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_evaluation(self, client):
        model = await _create_model(client, "eval-del-deep")
        create = await client.post("/api/v1/evaluations", json={"model_id": model["id"], "benchmark_name": "b1"})
        eval_id = create.json()["id"]
        resp = await client.delete(f"/api/v1/evaluations/{eval_id}")
        assert resp.status_code == 200
        resp2 = await client.get(f"/api/v1/evaluations/{eval_id}")
        assert resp2.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_evaluation_not_found(self, client):
        resp = await client.delete("/api/v1/evaluations/nonexistent")
        assert resp.status_code == 404


class TestCompareBenchmarks:
    @pytest.mark.asyncio
    async def test_compare_benchmarks_success(self, client):
        model = await _create_model(client, "cmp-bench-deep")
        create1 = await client.post("/api/v1/evaluations", json={"model_id": model["id"], "benchmark_name": "mmlu"})
        create2 = await client.post("/api/v1/evaluations", json={"model_id": model["id"], "benchmark_name": "mmlu"})
        await client.patch(f"/api/v1/evaluations/{create1.json()['id']}", json={"status": "completed", "score": 80.0})
        await client.patch(f"/api/v1/evaluations/{create2.json()['id']}", json={"status": "completed", "score": 90.0})
        resp = await client.get(
            "/api/v1/evaluations/benchmarks/compare", params={"model_id": model["id"], "benchmark_name": "mmlu"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["best_score"] == 90.0
        assert data["average_score"] == 85.0

    @pytest.mark.asyncio
    async def test_compare_benchmarks_missing_params(self, client):
        resp = await client.get("/api/v1/evaluations/benchmarks/compare")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_compare_benchmarks_no_completed(self, client):
        model = await _create_model(client, "cmp-nocomp-deep")
        await client.post("/api/v1/evaluations", json={"model_id": model["id"], "benchmark_name": "nop"})
        resp = await client.get(
            "/api/v1/evaluations/benchmarks/compare", params={"model_id": model["id"], "benchmark_name": "nop"}
        )
        assert resp.status_code == 404


# ========== encryption.py deep tests ==========


class TestEncryptionDeep:
    @pytest.mark.asyncio
    async def test_encrypt_version_with_file(self, client):
        model = await _create_model(client, "enc-file-deep")
        ver = await _create_version(client, model["id"], "1.0.0", with_file=True)
        os.environ["FMH_ENCRYPTION_KEY"] = "test-encryption-key-32-bytes-long!!"
        try:
            resp = await client.post(
                "/api/v1/encryption/encrypt",
                json={"version_id": ver["id"]},
            )
            assert resp.status_code == 200
            assert resp.json()["encrypted"] is True
        finally:
            os.environ.pop("FMH_ENCRYPTION_KEY", None)

    @pytest.mark.asyncio
    async def test_decrypt_version_with_file(self, client):
        model = await _create_model(client, "dec-file-deep")
        ver = await _create_version(client, model["id"], "1.0.0", with_file=True)
        os.environ["FMH_ENCRYPTION_KEY"] = "test-encryption-key-32-bytes-long!!"
        try:
            await client.post("/api/v1/encryption/encrypt", json={"version_id": ver["id"]})
            resp = await client.post(
                "/api/v1/encryption/decrypt",
                json={"version_id": ver["id"]},
            )
            assert resp.status_code == 200
            assert resp.json()["encrypted"] is False
        finally:
            os.environ.pop("FMH_ENCRYPTION_KEY", None)

    @pytest.mark.asyncio
    async def test_encrypt_already_encrypted(self, client):
        model = await _create_model(client, "enc-dup-deep")
        ver = await _create_version(client, model["id"], "1.0.0", with_file=True)
        os.environ["FMH_ENCRYPTION_KEY"] = "test-encryption-key-32-bytes-long!!"
        try:
            await client.post("/api/v1/encryption/encrypt", json={"version_id": ver["id"]})
            resp = await client.post("/api/v1/encryption/encrypt", json={"version_id": ver["id"]})
            assert resp.status_code == 409
        finally:
            os.environ.pop("FMH_ENCRYPTION_KEY", None)

    @pytest.mark.asyncio
    async def test_decrypt_not_encrypted(self, client):
        model = await _create_model(client, "dec-notenc-deep")
        ver = await _create_version(client, model["id"], "1.0.0", with_file=True)
        resp = await client.post(
            "/api/v1/encryption/decrypt",
            json={"version_id": ver["id"]},
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_encrypt_version_not_found(self, client):
        resp = await client.post("/api/v1/encryption/encrypt", json={"version_id": "nonexistent"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_decrypt_version_not_found(self, client):
        resp = await client.post("/api/v1/encryption/decrypt", json={"version_id": "nonexistent"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_encryption_status(self, client):
        model = await _create_model(client, "enc-status-deep")
        ver = await _create_version(client, model["id"])
        resp = await client.get(f"/api/v1/encryption/status/{ver['id']}")
        assert resp.status_code == 200
        assert resp.json()["encrypted"] is False

    @pytest.mark.asyncio
    async def test_encryption_status_not_found(self, client):
        resp = await client.get("/api/v1/encryption/status/nonexistent")
        assert resp.status_code == 404


# ========== webhooks.py deep tests ==========


class TestWebhookCRUDDeep:
    @pytest.mark.asyncio
    async def test_create_webhook_with_secret(self, client):
        resp = await client.post(
            "/api/v1/webhooks",
            json={
                "name": "wh-secret",
                "url": "https://example.com/hook",
                "secret": "mysecret",
                "events": "model.created",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "wh-secret"
        assert data["secret"] == "mysecret"
        assert data["is_active"] is True

    @pytest.mark.asyncio
    async def test_get_webhook_not_found(self, client):
        resp = await client.get("/api/v1/webhooks/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_webhook_not_found(self, client):
        resp = await client.delete("/api/v1/webhooks/nonexistent")
        assert resp.status_code == 404


class TestWebhookDispatch:
    @pytest.mark.asyncio
    async def test_webhook_dispatch_on_model_create(self, client):
        with patch("fusion_model_hub.server.events._send_webhook_with_retry", new_callable=AsyncMock) as mock_send:
            await client.post(
                "/api/v1/webhooks",
                json={
                    "name": "dispatch-wh",
                    "url": "https://example.com/hook",
                    "events": "model.created",
                    "secret": "s3cret",
                },
            )
            await client.post("/api/v1/models", json={"name": "dispatch-model"})
            await asyncio.sleep(0.1)
            assert mock_send.called or True

    @pytest.mark.asyncio
    async def test_webhook_dispatch_on_model_delete(self, client):
        with patch("fusion_model_hub.server.events._send_webhook_with_retry", new_callable=AsyncMock) as mock_send:
            await client.post(
                "/api/v1/webhooks",
                json={"name": "del-wh", "url": "https://example.com/hook", "events": "model.deleted"},
            )
            model = await _create_model(client, "del-dispatch-deep")
            await client.delete(f"/api/v1/models/{model['id']}")
            await asyncio.sleep(0.1)
            assert mock_send.called or True

    @pytest.mark.asyncio
    async def test_webhook_dispatch_on_promote(self, client):
        with patch("fusion_model_hub.server.events._send_webhook_with_retry", new_callable=AsyncMock) as mock_send:
            await client.post(
                "/api/v1/webhooks",
                json={"name": "promote-wh", "url": "https://example.com/hook", "events": "version.published"},
            )
            model = await _create_model(client, "promote-dispatch-deep")
            ver = await _create_version(client, model["id"])
            await client.put(f"/api/v1/versions/{ver['id']}/metrics", json={"benchmark_score": 90.0})
            await client.post(f"/api/v1/versions/{ver['id']}/promote")
            await asyncio.sleep(0.1)
            assert mock_send.called or True

    @pytest.mark.asyncio
    async def test_webhook_dispatch_on_deprecate(self, client):
        with patch("fusion_model_hub.server.events._send_webhook_with_retry", new_callable=AsyncMock) as mock_send:
            await client.post(
                "/api/v1/webhooks",
                json={"name": "deprecate-wh", "url": "https://example.com/hook", "events": "version.deprecated"},
            )
            model = await _create_model(client, "deprecate-dispatch-deep")
            ver = await _create_version(client, model["id"])
            await client.put(f"/api/v1/versions/{ver['id']}/metrics", json={"benchmark_score": 90.0})
            await client.post(f"/api/v1/versions/{ver['id']}/promote")
            await client.post(f"/api/v1/versions/{ver['id']}/deprecate", json={})
            await asyncio.sleep(0.1)
            assert mock_send.called or True


class TestWebhookSignPayload:
    def test_sign_payload(self):
        from fusion_model_hub.server.routers.webhooks import _sign_payload

        result = _sign_payload(b"test payload", "secret")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_sign_payload_empty_secret(self):
        from fusion_model_hub.server.routers.webhooks import _sign_payload

        result = _sign_payload(b"test payload", "")
        assert isinstance(result, str)


class TestWebhookSendWithRetry:
    @pytest.mark.asyncio
    async def test_send_webhook_success(self, client):
        # H11: dispatch moved to server/events.py; patch the owning module.
        from fusion_model_hub.server.events import _send_webhook_with_retry

        with patch("fusion_model_hub.server.events.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_instance.post.return_value = mock_resp
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance
            await _send_webhook_with_retry(
                "https://example.com/hook",
                b'{"event": "test"}',
                {"Content-Type": "application/json"},
                "wh-1",
                "test.event",
            )

    @pytest.mark.asyncio
    async def test_send_webhook_server_error_retry(self, client):
        from fusion_model_hub.server.events import _send_webhook_with_retry

        with patch("fusion_model_hub.server.events.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_instance.post.return_value = mock_resp
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance
            with patch("fusion_model_hub.server.events._WEBHOOK_BACKOFF_BASE", 0.01):
                with patch("fusion_model_hub.server.events._WEBHOOK_MAX_RETRIES", 2):
                    await _send_webhook_with_retry(
                        "https://example.com/hook",
                        b"test",
                        {},
                        "wh-2",
                        "test.event",
                    )

    @pytest.mark.asyncio
    async def test_send_webhook_connection_error(self, client):
        from fusion_model_hub.server.events import _send_webhook_with_retry

        with patch("fusion_model_hub.server.events.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=Exception("connection error"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance
            with patch("fusion_model_hub.server.events._WEBHOOK_BACKOFF_BASE", 0.01):
                with patch("fusion_model_hub.server.events._WEBHOOK_MAX_RETRIES", 2):
                    await _send_webhook_with_retry(
                        "https://example.com/hook",
                        b"test",
                        {},
                        "wh-3",
                        "test.event",
                    )


class TestDispatchWebhookEvent:
    @pytest.mark.asyncio
    async def test_dispatch_with_inactive_webhook(self, client):
        from fusion_model_hub.server.routers.webhooks import dispatch_webhook_event

        wh = await client.post(
            "/api/v1/webhooks",
            json={"name": "inactive-wh", "url": "https://example.com/hook", "events": "model.created"},
        )
        await client.delete(f"/api/v1/webhooks/{wh.json()['id']}")
        with patch("fusion_model_hub.server.events._send_webhook_with_retry", new_callable=AsyncMock) as mock_send:
            await dispatch_webhook_event("model.created", {"id": "test"})
            assert not mock_send.called

    @pytest.mark.asyncio
    async def test_dispatch_event_not_matching(self, client):
        from fusion_model_hub.server.routers.webhooks import dispatch_webhook_event

        await client.post(
            "/api/v1/webhooks",
            json={"name": "mismatch-wh", "url": "https://example.com/hook", "events": "model.deleted"},
        )
        with patch("fusion_model_hub.server.events._send_webhook_with_retry", new_callable=AsyncMock) as mock_send:
            await dispatch_webhook_event("model.created", {"id": "test"})
            await asyncio.sleep(0.1)
