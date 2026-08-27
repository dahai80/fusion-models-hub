import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps

logger = logging.getLogger(__name__)


class TestCLIMain:
    def test_export_to_stdout(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_export

        class Args:
            data_dir = str(tmp_path / "export_data")
            db_url = "sqlite+aiosqlite:///:memory:"
            output = "-"
            models = ""

        _run_export(Args())

    def test_export_to_file(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_export

        out_file = str(tmp_path / "export.json")
        dd = str(tmp_path / "export_data2")

        class Args2:
            data_dir = dd
            db_url = "sqlite+aiosqlite:///:memory:"
            output = out_file
            models = ""

        _run_export(Args2())
        assert os.path.exists(out_file)
        with open(out_file) as f:
            data = json.load(f)
        assert "models" in data
        assert "tenants" in data

    def test_export_with_model_filter(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_export

        out_file = str(tmp_path / "export_filtered.json")

        class Args:
            data_dir = str(tmp_path / "export_data3")
            db_url = "sqlite+aiosqlite:///:memory:"
            output = out_file
            models = "nonexistent-id"

        _run_export(Args())

    def test_import_from_file(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_import

        input_file = tmp_path / "import_data.json"
        input_data = {
            "models": [{"name": "imported-model", "model_type": "llm", "architecture": "qwen2"}],
            "tenants": [{"name": "test-tenant", "display_name": "Test"}],
            "webhooks": [{"name": "test-hook", "url": "http://example.com/hook", "events": "model.created"}],
        }
        input_file.write_text(json.dumps(input_data))

        class Args:
            data_dir = str(tmp_path / "import_data_dir")
            db_url = "sqlite+aiosqlite:///:memory:"
            input = str(input_file)

        _run_import(Args())

    def test_import_duplicate_skips(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_import

        input_file = tmp_path / "import_dup.json"
        input_data = {
            "models": [{"name": "dup-model", "model_type": "llm"}],
            "tenants": [{"name": "dup-tenant"}],
            "webhooks": [],
        }
        input_file.write_text(json.dumps(input_data))

        class Args:
            data_dir = str(tmp_path / "import_dup_dir")
            db_url = "sqlite+aiosqlite:///:memory:"
            input = str(input_file)

        _run_import(Args())
        _run_import(Args())

    def test_migrate_no_alembic(self):
        from fusion_model_hub.server.__main__ import _run_migrate

        class Args:
            db_url = ""
            revision = ""

        with patch.dict(sys.modules, {"alembic": None, "alembic.config": None, "alembic.command": None}):
            with pytest.raises(SystemExit):
                _run_migrate(Args())

    def test_main_serve(self):
        from fusion_model_hub.server.__main__ import main

        with patch("sys.argv", ["fusion-model-hub", "serve", "--host", "0.0.0.0", "--port", "9999"]):
            with patch("uvicorn.run") as mock_run:
                main()
                mock_run.assert_called_once()

    def test_main_no_command_defaults_to_serve(self):
        from fusion_model_hub.server.__main__ import main

        with patch("sys.argv", ["fusion-model-hub", "serve"]):
            with patch("uvicorn.run") as mock_run:
                main()
                mock_run.assert_called_once()


class TestMinioStore:
    def test_init(self):
        from fusion_model_hub.storage.minio_store import MinioStore

        store = MinioStore("localhost:9000", "access", "secret", bucket="test-bucket", secure=False)
        assert store.endpoint == "localhost:9000"
        assert store.bucket == "test-bucket"

    def test_model_version_dir(self):
        from fusion_model_hub.storage.minio_store import MinioStore

        store = MinioStore("localhost:9000", "ak", "sk")
        p = store.model_version_dir("m1", "v1")
        assert str(p) == "m1/v1"

    def test_write_file(self):
        from fusion_model_hub.storage.minio_store import MinioStore

        store = MinioStore("localhost:9000", "ak", "sk")
        mock_client = MagicMock()
        store._client = mock_client

        path, file_hash, size = asyncio.run(store.write_file(Path("m1/v1"), "model.bin", b"hello"))
        assert size == 5
        mock_client.put_object.assert_called_once()

    def test_get_file_exists(self):
        from fusion_model_hub.storage.minio_store import MinioStore

        store = MinioStore("localhost:9000", "ak", "sk")
        mock_client = MagicMock()
        store._client = mock_client

        result = store.get_file("m1/v1/model.bin")
        assert result is not None

    def test_get_file_not_exists(self):
        from fusion_model_hub.storage.minio_store import MinioStore

        store = MinioStore("localhost:9000", "ak", "sk")
        mock_client = MagicMock()
        mock_client.stat_object.side_effect = Exception("not found")
        store._client = mock_client

        result = store.get_file("m1/v1/missing.bin")
        assert result is None

    def test_delete_version_files(self):
        from fusion_model_hub.storage.minio_store import MinioStore

        store = MinioStore("localhost:9000", "ak", "sk")
        mock_client = MagicMock()
        mock_obj = MagicMock()
        mock_obj.object_name = "m1/v1/model.bin"
        mock_client.list_objects.return_value = [mock_obj]
        store._client = mock_client

        result = store.delete_version_files("m1", "v1")
        assert result is True
        mock_client.remove_object.assert_called()

    def test_delete_model_files(self):
        from fusion_model_hub.storage.minio_store import MinioStore

        store = MinioStore("localhost:9000", "ak", "sk")
        mock_client = MagicMock()
        mock_client.list_objects.return_value = []
        store._client = mock_client

        result = store.delete_model_files("m1")
        assert result is False

    def test_get_storage_stats(self):
        from fusion_model_hub.storage.minio_store import MinioStore

        store = MinioStore("localhost:9000", "ak", "sk")
        mock_client = MagicMock()
        mock_obj = MagicMock()
        mock_obj.object_name = "m1/v1/model.bin"
        mock_obj.size = 1024
        mock_client.list_objects.return_value = [mock_obj]
        store._client = mock_client

        stats = store.get_storage_stats()
        assert stats["file_count"] == 1
        assert stats["model_count"] == 1

    def test_get_client_creates_bucket(self):
        from fusion_model_hub.storage.minio_store import MinioStore

        store = MinioStore("localhost:9000", "ak", "sk")
        mock_minio_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.bucket_exists.return_value = False
        mock_minio_cls.return_value = mock_instance

        with patch.dict(sys.modules, {"minio": MagicMock(Minio=mock_minio_cls)}):
            store._client = None
            client = store._get_client()
            mock_instance.make_bucket.assert_called_once()

    def test_get_client_import_error(self):
        from fusion_model_hub.storage.minio_store import MinioStore

        store = MinioStore("localhost:9000", "ak", "sk")
        store._client = None

        with patch.dict(sys.modules, {"minio": None}):
            with pytest.raises(RuntimeError, match="minio package not installed"):
                store._get_client()


class TestAppCreation:
    @pytest.mark.asyncio
    async def test_create_app_default_settings(self):
        from fusion_model_hub.server.app import create_app
        from fusion_model_hub.server.deps import Settings

        app = create_app(
            Settings(
                db_url="sqlite+aiosqlite:///:memory:",
                data_dir="/tmp/fmh_app_test",
            )
        )
        assert app.title == "Fusion Model Hub"

    @pytest.mark.asyncio
    async def test_create_app_with_cors_wildcard(self):
        from fusion_model_hub.server.app import create_app
        from fusion_model_hub.server.deps import Settings

        app = create_app(
            Settings(
                db_url="sqlite+aiosqlite:///:memory:",
                data_dir="/tmp/fmh_cors_test",
                cors_origins=["*"],
            )
        )
        assert app.title == "Fusion Model Hub"


class TestLocalStoreExtended:
    def test_init_default_data_dir(self):
        from fusion_model_hub.storage.local_store import LocalStore

        store = LocalStore()
        assert "data" in str(store.data_dir)

    def test_write_chunk_and_assemble(self):
        from fusion_model_hub.storage.local_store import LocalStore

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(data_dir=tmp)
            upload_id = "test-upload-1"

            chunk1 = b"hello " * 100
            chunk2 = b"world " * 100
            asyncio.run(store.write_chunk(upload_id, 0, chunk1))
            asyncio.run(store.write_chunk(upload_id, 1, chunk2))

            target_dir = store.model_version_dir("m1", "v1")
            path, file_hash, size = asyncio.run(store.assemble_chunks(upload_id, target_dir, "model.bin", 2))
            assert path.exists()
            assert size == len(chunk1) + len(chunk2)
            assert len(file_hash) == 64

    def test_assemble_missing_chunk(self):
        from fusion_model_hub.storage.local_store import LocalStore

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(data_dir=tmp)
            upload_id = "test-upload-missing"

            with pytest.raises(FileNotFoundError):
                asyncio.run(store.assemble_chunks(upload_id, Path(tmp), "model.bin", 1))

    def test_is_path_within_store(self):
        from fusion_model_hub.storage.local_store import LocalStore

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(data_dir=tmp)
            assert store.is_path_within_store(store.models_dir / "m1" / "v1" / "file.bin")
            assert not store.is_path_within_store(Path("/etc/passwd"))

    def test_delete_version_files_no_dir(self):
        from fusion_model_hub.storage.local_store import LocalStore

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(data_dir=tmp)
            assert store.delete_version_files("nonexistent", "v1") is False

    def test_delete_model_files_no_dir(self):
        from fusion_model_hub.storage.local_store import LocalStore

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(data_dir=tmp)
            assert store.delete_model_files("nonexistent") is False

    def test_verify_hash(self):
        from fusion_model_hub.storage.local_store import LocalStore

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(data_dir=tmp)
            test_file = Path(tmp) / "hash_test.bin"
            test_file.write_bytes(b"test content")
            import hashlib

            expected = hashlib.sha256(b"test content").hexdigest()
            assert LocalStore.verify_hash(test_file, expected)
            assert not LocalStore.verify_hash(test_file, "wronghash")

    def test_get_storage_stats(self):
        from fusion_model_hub.storage.local_store import LocalStore

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(data_dir=tmp)
            model_dir = store.model_version_dir("m1", "v1")
            (model_dir / "weights.bin").write_bytes(b"x" * 100)
            stats = store.get_storage_stats()
            assert stats["model_count"] == 1
            assert stats["file_count"] == 1

    def test_write_file(self):
        from fusion_model_hub.storage.local_store import LocalStore

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(data_dir=tmp)
            target = store.model_version_dir("m2", "v1")
            path, file_hash, size = asyncio.run(store.write_file(target, "test.bin", b"data"))
            assert path.exists()
            assert size == 4


class TestCRUDExtended:
    @pytest.fixture(autouse=True)
    async def _init_deps(self):
        settings = Settings(
            host="127.0.0.1",
            port=11444,
            data_dir="/tmp/fmh_test_cov_ext",
            db_url="sqlite+aiosqlite:///:memory:",
            log_level="WARNING",
        )
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        yield
        import shutil

        shutil.rmtree("/tmp/fmh_test_cov_ext", ignore_errors=True)

    @pytest.mark.asyncio
    async def test_list_models_with_filters(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.db.models import ModelType
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            m = await crud.create_model(
                session,
                name="filter-test-model",
                model_type=ModelType.LLM,
                architecture="qwen2",
                tenant_id="t1",
            )
            models, total = await crud.list_models(
                session,
                keyword="filter-test",
                model_type="llm",
                architecture="qwen2",
                tenant_id="t1",
            )
            assert total >= 1

    @pytest.mark.asyncio
    async def test_increment_download(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.db.models import ModelType
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            m = await crud.create_model(session, name="dl-test-model", model_type=ModelType.LLM)
            await crud.increment_download(session, m.id)
            fetched = await crud.get_model(session, m.id)
            assert fetched.download_count == 1

    @pytest.mark.asyncio
    async def test_create_quantize_task(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            t = await crud.create_quantize_task(
                session,
                source_version_id="v1",
                target_format="mlx",
                quant_bits=4,
            )
            assert t.id
            fetched = await crud.get_quantize_task(session, t.id)
            assert fetched is not None

    @pytest.mark.asyncio
    async def test_update_quantize_task(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            t = await crud.create_quantize_task(session, source_version_id="v1")
            updated = await crud.update_quantize_task(session, t.id, status="running")
            assert updated is not None
            not_found = await crud.update_quantize_task(session, "nonexistent", status="running")
            assert not_found is None

    @pytest.mark.asyncio
    async def test_list_quantize_tasks_with_status(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            tasks, total = await crud.list_quantize_tasks(session, status="running")
            assert isinstance(total, int)

    @pytest.mark.asyncio
    async def test_api_key_crud(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            full_key, key_hash, prefix = crud._generate_api_key()
            assert full_key.startswith("fmh-")
            assert len(key_hash) == 64

            ak, raw_key = await crud.create_api_key(session, name="test-key")
            assert ak.id
            assert raw_key.startswith("fmh-")

            fetched = await crud.verify_api_key(session, raw_key)
            assert fetched is not None

            invalid = await crud.verify_api_key(session, "invalid-key")
            assert invalid is None

            keys = await crud.list_api_keys(session)
            assert len(keys) >= 1

            deactivated = await crud.deactivate_api_key(session, ak.id)
            assert deactivated

            deleted = await crud.delete_api_key(session, ak.id)
            assert deleted

    @pytest.mark.asyncio
    async def test_audit_log_crud(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            await crud.create_audit_log(
                session,
                action="test_action",
                resource_type="model",
                resource_id="m1",
                api_key_id="ak1",
                detail="test",
            )
            logs, total = await crud.list_audit_logs(
                session,
                resource_type="model",
                action="test_action",
            )
            assert total >= 1

    @pytest.mark.asyncio
    async def test_tenant_crud(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            t = await crud.create_tenant(session, name="test-tenant-ext", display_name="Test Tenant")
            assert t.id

            fetched = await crud.get_tenant(session, t.id)
            assert fetched is not None

            by_name = await crud.get_tenant_by_name(session, "test-tenant-ext")
            assert by_name is not None

            tenants = await crud.list_tenants(session)
            assert len(tenants) >= 1

            deleted = await crud.delete_tenant(session, t.id)
            assert deleted

    @pytest.mark.asyncio
    async def test_webhook_crud(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            w = await crud.create_webhook(
                session,
                name="test-hook",
                url="http://example.com/hook",
                events="model.created",
                tenant_id="t1",
            )
            assert w.id

            fetched = await crud.get_webhook(session, w.id)
            assert fetched is not None

            hooks = await crud.list_webhooks(session, tenant_id="t1")
            assert len(hooks) >= 1

            deleted = await crud.delete_webhook(session, w.id)
            assert deleted

    @pytest.mark.asyncio
    async def test_deployment_crud(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            d = await crud.create_deployment(
                session,
                model_id="m1",
                name="deploy-1",
                version_id="v1",
                replicas=1,
                tenant_id="t1",
            )
            assert d.id

            fetched = await crud.get_deployment(session, d.id)
            assert fetched is not None

            deps = await crud.list_deployments(session, model_id="m1")
            assert len(deps) >= 1

            updated = await crud.update_deployment(session, d.id, replicas=3)
            assert updated.replicas == 3

            deleted = await crud.delete_deployment(session, d.id)
            assert deleted

    @pytest.mark.asyncio
    async def test_evaluation_crud_full(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            e = await crud.create_evaluation(
                session,
                model_id="m1",
                benchmark_name="mmlu",
                tenant_id="t1",
                version_id="v1",
            )
            assert e.id

            fetched = await crud.get_evaluation(session, e.id)
            assert fetched is not None

            evals, total = await crud.list_evaluations(
                session,
                model_id="m1",
                version_id="v1",
                benchmark_name="mmlu",
            )
            assert total >= 1

            updated = await crud.update_evaluation(
                session,
                e.id,
                score=85.5,
                status="completed",
            )
            assert updated is not None

            not_found = await crud.update_evaluation(session, "nonexistent", score=1.0)
            assert not_found is None

            deleted = await crud.delete_evaluation(session, e.id)
            assert deleted

            not_found_del = await crud.delete_evaluation(session, "nonexistent")
            assert not not_found_del

    @pytest.mark.asyncio
    async def test_cluster_node_crud(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            n = await crud.create_cluster_node(
                session,
                name="node-ext-1",
                url="http://node1:11444",
                capabilities="inference",
            )
            assert n.id

            fetched = await crud.get_cluster_node(session, n.id)
            assert fetched is not None

            nodes = await crud.list_cluster_nodes(session)
            assert len(nodes) >= 1

            deleted = await crud.delete_cluster_node(session, n.id)
            assert deleted

    @pytest.mark.asyncio
    async def test_lora_merge_task_crud(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            t = await crud.create_lora_merge_task(
                session,
                base_version_id="v1",
                lora_version_id="v2",
                target_format="mlx",
                quant_bits=4,
            )
            assert t.id

            fetched = await crud.get_lora_merge_task(session, t.id)
            assert fetched is not None

            tasks, total = await crud.list_lora_merge_tasks(session, status="pending")
            assert isinstance(total, int)

            updated = await crud.update_lora_merge_task(session, t.id, status="completed")
            assert updated is not None

    @pytest.mark.asyncio
    async def test_version_status_transitions(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.db.models import VersionStatus
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            m = await crud.create_model(session, name="transition-model")
            v = await crud.create_version(
                session,
                model_id=m.id,
                version="1.0.0",
                format="mlx",
                quantization="4bit",
            )
            assert v.status == VersionStatus.DRAFT

            v = await crud.update_version_status(session, v.id, VersionStatus.TESTING)
            assert v.status == VersionStatus.TESTING

            v = await crud.update_version(session, v.id, benchmark_score=90.0)
            v = await crud.update_version_status(session, v.id, VersionStatus.PUBLISHED, approval_level="l1")
            assert v.status == VersionStatus.PUBLISHED

    @pytest.mark.asyncio
    async def test_version_status_invalid_transition(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.db.models import VersionStatus
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            m = await crud.create_model(session, name="invalid-trans-model")
            v = await crud.create_version(
                session,
                model_id=m.id,
                version="1.0.0",
                format="mlx",
                quantization="4bit",
            )
            v = await crud.update_version_status(session, v.id, VersionStatus.RETIRED)
            with pytest.raises(crud.InvalidTransition):
                await crud.update_version_status(session, v.id, VersionStatus.PUBLISHED)

    @pytest.mark.asyncio
    async def test_list_audit_logs_no_filter(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            logs, total = await crud.list_audit_logs(session)
            assert isinstance(total, int)

    @pytest.mark.asyncio
    async def test_deployment_crud_not_found(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            fetched = await crud.get_deployment(session, "nonexistent")
            assert fetched is None

            deleted = await crud.delete_deployment(session, "nonexistent")
            assert not deleted

            updated = await crud.update_deployment(session, "nonexistent", replicas=1)
            assert updated is None

    @pytest.mark.asyncio
    async def test_list_deployments_with_filters(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            deps = await crud.list_deployments(
                session,
                tenant_id="t1",
                model_id="m1",
                status="running",
            )
            assert isinstance(deps, list)

    @pytest.mark.asyncio
    async def test_update_version_whitelist(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            m = await crud.create_model(session, name="whitelist-model")
            v = await crud.create_version(
                session,
                model_id=m.id,
                version="1.0.0",
                format="mlx",
                quantization="4bit",
            )
            updated = await crud.update_version(
                session,
                v.id,
                file_path="/new/path",
                file_hash="abc123",
                file_size=1024,
                release_notes="updated",
                benchmark_score=85.0,
                inference_latency=12.3,
                throughput=45.6,
                memory_usage=2048,
                context_length=4096,
                encrypted=True,
            )
            assert updated is not None
            assert updated.encrypted is True

    @pytest.mark.asyncio
    async def test_list_webhooks_no_tenant(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            hooks = await crud.list_webhooks(session)
            assert isinstance(hooks, list)

    @pytest.mark.asyncio
    async def test_distributed_task_update(self):
        from fusion_model_hub.db import crud
        from fusion_model_hub.server.deps import get_session_factory

        sf = get_session_factory()
        async with sf() as session:
            t = await crud.create_distributed_task(
                session,
                model_id="m1",
                version_id="v1",
                target_nodes='["node1"]',
            )
            updated = await crud.update_distributed_task(session, t.id, status="completed", progress=100)
            assert updated is not None

            not_found = await crud.update_distributed_task(session, "nonexistent", status="running")
            assert not_found is None


class TestAuthExtended:
    @pytest.mark.asyncio
    async def test_auth_middleware_public_paths(self):
        from fusion_model_hub.server.auth import PUBLIC_PATHS, WRITE_METHODS

        assert "/api/v1/system/health" in PUBLIC_PATHS
        assert "POST" in WRITE_METHODS
        assert "GET" not in WRITE_METHODS

    def test_extract_resource_type(self):
        from fusion_model_hub.server.auth import _extract_resource_type

        assert _extract_resource_type("/api/v1/models/m1/versions") == "m1"
        assert _extract_resource_type("/api/v1/system/health") == "health"

    def test_extract_resource_id(self):
        from fusion_model_hub.server.auth import _extract_resource_id

        result = _extract_resource_id("/api/v1/models/m1/versions")
        assert result == "m1"
        assert _extract_resource_id("/api/v1/versions") == ""


class TestConfig:
    def test_settings_defaults(self):
        from fusion_model_hub.server.config import Settings

        s = Settings()
        assert s.host == "127.0.0.1"
        assert s.port == 11444

    def test_settings_env_data_dir(self):
        from fusion_model_hub.server.config import Settings

        with patch.dict(os.environ, {"FMH_DATA_DIR": "/tmp/env_test"}):
            s = Settings()
            assert s.data_dir == "/tmp/env_test"

    def test_settings_auth_enabled(self):
        from fusion_model_hub.server.config import Settings

        s = Settings(auth_enabled=True)
        assert s.auth_enabled is True

    def test_settings_minio(self):
        from fusion_model_hub.server.config import Settings

        s = Settings(storage_type="minio", minio_endpoint="minio:9000")
        assert s.storage_type == "minio"
        assert s.minio_endpoint == "minio:9000"


class TestDeps:
    def test_init_deps_with_engine(self):
        from fusion_model_hub.db.database import get_engine
        from fusion_model_hub.server.deps import (
            Settings,
            get_session_factory,
            init_deps,
        )

        settings = Settings(
            db_url="sqlite+aiosqlite:///:memory:",
            data_dir="/tmp/fmh_deps_test",
        )
        engine = get_engine(settings.db_url)
        init_deps(settings, engine)
        sf = get_session_factory()
        assert sf is not None

    def test_get_store(self):
        from fusion_model_hub.db.database import get_engine
        from fusion_model_hub.server.deps import Settings, get_store, init_deps

        settings = Settings(
            db_url="sqlite+aiosqlite:///:memory:",
            data_dir="/tmp/fmh_store_test",
        )
        engine = get_engine(settings.db_url)
        init_deps(settings, engine)
        store = get_store()
        assert store is not None

    def test_get_settings(self):
        from fusion_model_hub.db.database import get_engine
        from fusion_model_hub.server.deps import Settings, get_settings, init_deps

        settings = Settings(
            db_url="sqlite+aiosqlite:///:memory:",
            data_dir="/tmp/fmh_settings_test",
        )
        engine = get_engine(settings.db_url)
        init_deps(settings, engine)
        s = get_settings()
        assert s.data_dir == "/tmp/fmh_settings_test"
