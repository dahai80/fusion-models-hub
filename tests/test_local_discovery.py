import os
from pathlib import Path

import pytest

from fusion_model_hub.manage.discovery import discover_local_models, guess_model_type


@pytest.fixture()
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


def _hf_cache(root: Path, repo: str, with_snapshot: bool = True) -> Path:
    d = root / f"models--{repo.replace('/', '--')}"
    snap = d / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    if with_snapshot:
        (snap / "config.json").write_text("{}")
    return d


class TestDiscoverLocalModels:
    def test_discovers_hf_cache_layout(self, models_dir: Path):
        _hf_cache(models_dir, "mlx-community/Qwen3-0.6B-8bit")
        found = discover_local_models(models_dir)
        assert len(found) == 1
        item = found[0]
        assert item["name"] == "mlx-community/Qwen3-0.6B-8bit"
        assert item["hf_repo"] == "mlx-community/Qwen3-0.6B-8bit"
        assert item["kind"] == "hf"

    def test_discovers_plain_model_dir_with_config(self, models_dir: Path):
        d = models_dir / "Qwen3.8-27B-DFlash2"
        d.mkdir()
        (d / "config.json").write_text("{}")
        found = discover_local_models(models_dir)
        assert len(found) == 1
        assert found[0]["name"] == "Qwen3.8-27B-DFlash2"
        assert found[0]["hf_repo"] == ""
        assert found[0]["kind"] == "local"

    def test_discovers_plain_dir_with_safetensors(self, models_dir: Path):
        d = models_dir / "my-weights"
        d.mkdir()
        (d / "model.safetensors").write_text("x")
        assert len(discover_local_models(models_dir)) == 1

    def test_skips_junk_and_non_model_dirs(self, models_dir: Path):
        for name in (".locks", "blobs", "xet"):
            (models_dir / name).mkdir()
        (models_dir / "empty-dir").mkdir()
        (models_dir / "notes.txt").write_text("x")
        assert discover_local_models(models_dir) == []

    def test_skips_hf_cache_without_snapshot(self, models_dir: Path):
        _hf_cache(models_dir, "org/repo", with_snapshot=False)
        assert discover_local_models(models_dir) == []

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        assert discover_local_models(tmp_path / "nope") == []

    def test_symlinked_dir_skipped(self, models_dir: Path, tmp_path: Path):
        real = tmp_path / "real-model"
        real.mkdir()
        (real / "config.json").write_text("{}")
        link = models_dir / "linked"
        link.symlink_to(real)
        assert discover_local_models(models_dir) == []


class TestGuessModelType:
    def test_embedding(self):
        assert guess_model_type("BAAI/bge-m3", Path("/x")) == "embedding"
        assert guess_model_type("openai/clip-vit-large", Path("/x")) == "embedding"

    def test_audio(self):
        assert guess_model_type("mlx-community/Kokoro-82M", Path("/x")) == "audio"
        assert guess_model_type("Spark-TTS-0.5B", Path("/x")) == "audio"

    def test_image(self):
        assert guess_model_type("black-forest-labs/FLUX.1-dev", Path("/x")) == "image"
        assert guess_model_type("Qwen/Qwen-Image", Path("/x")) == "image"

    def test_default_llm(self):
        assert guess_model_type("Qwen/Qwen3-0.6B", Path("/x")) == "llm"


class TestSyncLocalModels:
    @pytest.mark.asyncio()
    async def test_registers_missing_and_is_idempotent(self, tmp_path: Path, monkeypatch):
        os.environ["FMH_MODELS_DIR"] = str(tmp_path)
        try:
            d = tmp_path / "sync-test-model"
            d.mkdir()
            (d / "config.json").write_text("{}")

            from fusion_model_hub.db.database import get_engine, init_db
            from fusion_model_hub.server.app import _sync_local_models
            from fusion_model_hub.server.config import Settings
            from fusion_model_hub.server.deps import init_deps

            settings = Settings(data_dir=str(tmp_path / "data"))
            engine = get_engine(settings.db_url)
            await init_db(engine)
            init_deps(settings, engine)

            created = await _sync_local_models(settings)
            assert created == 1
            again = await _sync_local_models(settings)
            assert again == 0
        finally:
            os.environ.pop("FMH_MODELS_DIR", None)

    @pytest.mark.asyncio()
    async def test_matches_existing_hf_repo_by_name(self, tmp_path: Path):
        d = tmp_path / "models--org--repoA"
        snap = d / "snapshots" / "c1"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")

        from sqlalchemy import select

        from fusion_model_hub.db.crud import create_model
        from fusion_model_hub.db.database import get_engine, init_db
        from fusion_model_hub.db.models import Model
        from fusion_model_hub.server.app import _sync_local_models
        from fusion_model_hub.server.config import Settings
        from fusion_model_hub.server.deps import init_deps

        settings = Settings(data_dir=str(tmp_path / "data"), models_dir=str(tmp_path))
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        from fusion_model_hub.server.deps import get_session_factory

        async with get_session_factory()() as session:
            await create_model(session, name="pre-existing", hf_repo="org/repoA")
        created = await _sync_local_models(settings)
        assert created == 0
        async with get_session_factory()() as session:
            rows = (await session.execute(select(Model))).scalars().all()
            assert len(rows) == 1
