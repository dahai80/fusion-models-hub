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
