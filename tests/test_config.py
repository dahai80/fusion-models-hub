import json

from fusion_model_hub.server.config import Settings


class TestMlxApiKeyResolution:
    # mlx_internal_api_key resolution order (config.py __post_init__):
    #   FUSION_MLX_API_KEY env > MLX_INTERNAL_API_KEY env >
    #   ~/.fusion-mlx/settings.json auth.api_key > unset (warn).
    # MLX gates on its settings.json api_key; a hub without a matching Bearer
    # gets 401 on every hub->MLX call. Pin each resolution branch so a refactor
    # cannot silently drop the local-install fallback.

    def _clear_key_env(self, monkeypatch):
        for var in ("FUSION_MLX_API_KEY", "MLX_INTERNAL_API_KEY"):
            monkeypatch.delenv(var, raising=False)

    def test_env_fusion_mlx_api_key_wins(self, monkeypatch):
        self._clear_key_env(monkeypatch)
        monkeypatch.setenv("FUSION_MLX_API_KEY", "env-key-123")
        s = Settings()
        assert s.mlx_internal_api_key == "env-key-123"

    def test_deprecated_mlx_internal_api_key_used_when_primary_unset(self, monkeypatch):
        self._clear_key_env(monkeypatch)
        monkeypatch.setenv("MLX_INTERNAL_API_KEY", "legacy-key-456")
        s = Settings()
        assert s.mlx_internal_api_key == "legacy-key-456"

    def test_falls_back_to_mlx_settings_json(self, monkeypatch, tmp_path):
        self._clear_key_env(monkeypatch)
        mlx_dir = tmp_path / ".fusion-mlx"
        mlx_dir.mkdir()
        (mlx_dir / "settings.json").write_text(json.dumps({"auth": {"api_key": "dahai168"}}))
        monkeypatch.setenv("HOME", str(tmp_path))
        s = Settings()
        assert s.mlx_internal_api_key == "dahai168"

    def test_env_overrides_mlx_settings_json(self, monkeypatch, tmp_path):
        self._clear_key_env(monkeypatch)
        mlx_dir = tmp_path / ".fusion-mlx"
        mlx_dir.mkdir()
        (mlx_dir / "settings.json").write_text(json.dumps({"auth": {"api_key": "settings-key"}}))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("FUSION_MLX_API_KEY", "env-wins")
        s = Settings()
        assert s.mlx_internal_api_key == "env-wins"

    def test_no_key_when_neither_env_nor_settings(self, monkeypatch, tmp_path):
        self._clear_key_env(monkeypatch)
        monkeypatch.setenv("HOME", str(tmp_path))
        s = Settings()
        assert s.mlx_internal_api_key == ""

    def test_missing_settings_json_is_silent(self, monkeypatch, tmp_path):
        self._clear_key_env(monkeypatch)
        monkeypatch.setenv("HOME", str(tmp_path))
        # no ~/.fusion-mlx/settings.json — must not raise
        s = Settings()
        assert s.mlx_internal_api_key == ""


class TestOpsEnvWiring:
    # P1-21: FMH_HOST/FMH_PORT/FMH_LOG_LEVEL/FMH_DB_URL must be honored. Before,
    # the serve CLI passed non-empty argparse defaults (127.0.0.1/11444/INFO)
    # into Settings(), so __post_init__'s `if not self.x` hooks never fired and
    # an operator's container env was silently ignored.

    def test_fmh_host_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("FMH_HOST", "0.0.0.0")
        s = Settings(host="127.0.0.1")
        assert s.host == "0.0.0.0"

    def test_fmh_port_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("FMH_PORT", "22000")
        s = Settings(port=11444)
        assert s.port == 22000

    def test_fmh_port_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("FMH_PORT", "not-an-int")
        s = Settings(port=11444)
        assert s.port == 11444

    def test_fmh_log_level_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("FMH_LOG_LEVEL", "DEBUG")
        s = Settings(log_level="INFO")
        assert s.log_level == "DEBUG"

    def test_fmh_db_url_env_overrides_derived_sqlite(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FMH_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("FMH_DB_URL", "postgresql+asyncpg://user:pass@db:5432/hub")
        s = Settings()
        assert s.db_url == "postgresql+asyncpg://user:pass@db:5432/hub"

    def test_db_url_derives_sqlite_when_env_unset(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FMH_DATA_DIR", str(tmp_path))
        monkeypatch.delenv("FMH_DB_URL", raising=False)
        s = Settings()
        assert s.db_url.startswith("sqlite+aiosqlite:///")
        assert s.db_url.endswith("hub.db")

    def test_explicit_constructor_arg_beats_env(self, monkeypatch):
        # an explicit non-default constructor value must not be clobbered by env
        monkeypatch.setenv("FMH_HOST", "0.0.0.0")
        s = Settings(host="10.0.0.5")
        assert s.host == "10.0.0.5"
